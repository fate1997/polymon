import argparse
import os
import pickle
from typing import List, Tuple

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from tabpfn import TabPFNRegressor
from tabpfn.model_loading import (load_fitted_tabpfn_model,
                                  save_fitted_tabpfn_model)
from tqdm import tqdm
from xgboost import XGBRegressor

from polymon.data.dataset import PolymerDataset
from polymon.exp.score import normalize_property_weight, scaling_error
from polymon.exp.utils import loader2numpy, predict_batch, seed_everything
from polymon.hparams import get_hparams
from polymon.setting import REPO_DIR, TARGETS

MODELS = {
    'rf': RandomForestRegressor,
    'xgb': XGBRegressor,
    'lgbm': LGBMRegressor,
    'catboost': CatBoostRegressor,
    'tabpfn': TabPFNRegressor,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-csv-path', type=str, default='database/internal/train.csv')
    parser.add_argument('--sources', type=str, nargs='+', default=['official_external'])
    parser.add_argument('--tag', type=str, default='debug')
    parser.add_argument('--labels', choices=TARGETS, nargs='+', default=None)
    parser.add_argument('--feature-names', type=str, nargs='+', default=['rdkit2d'])
    parser.add_argument('--model', choices=MODELS.keys(), default='rf')
    parser.add_argument('--optimize-hparams', action='store_true')
    parser.add_argument('--n-trials', type=int, default=10)
    parser.add_argument('--out-dir', type=str, default='./results')
    parser.add_argument('--hparams-from', type=str, default=None)
    return parser.parse_args()

PREDICT_BATCH_SIZE = 128

def train(
    out_dir: str,
    model: str,
    label: str,
    feature_names: List[str],
    optimize_hparams: bool,
    hparams_from: str,
    raw_csv_path: str,
    sources: List[str],
    n_trials: int,
    tag: str,
) -> Tuple[float, float]:
    seed_everything(42)
    out_dir = os.path.join(out_dir, model)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, tag), exist_ok=True)
    out_dir = os.path.join(out_dir, tag)
    name = f'{model}-{label}-{"-".join(feature_names)}-{tag}'
    logger.add(os.path.join(out_dir, f'{name}.log'))
    model_type = model

    # 1. Load data
    logger.info(f'Training {label}...')
    logger.info(f'Feature names: {feature_names}')
    dataset = PolymerDataset(
        raw_csv_path=raw_csv_path,
        feature_names=feature_names,
        sources=sources,
        label_column=label,
        force_reload=True,
        add_hydrogens=False,
    )
    train_loader, val_loader, test_loader = dataset.get_loaders(
        batch_size=128,
        n_train=0.8,
        n_val=0.1,
    )
    x_train, y_train = loader2numpy(train_loader)
    x_val, y_val = loader2numpy(val_loader)
    x_test, y_test = loader2numpy(test_loader)
    
    # remove inf
    mask_train = ~np.isinf(x_train).any(axis=1)
    mask_val = ~np.isinf(x_val).any(axis=1)
    mask_test = ~np.isinf(x_test).any(axis=1)
    logger.info(f'Train inf: {len(x_train) - mask_train.sum()}, Val inf: {len(x_val) - mask_val.sum()}, Test inf: {len(x_test) - mask_test.sum()}')
    x_train = x_train[mask_train]
    y_train = y_train[mask_train]
    x_val = x_val[mask_val]
    y_val = y_val[mask_val]
    x_test = x_test[mask_test]
    y_test = y_test[mask_test]
    logger.info(f'Train size: {x_train.shape[0]}, Val size: {x_val.shape[0]}, Test size: {x_test.shape[0]}')
    
    # 2. Train model
    if not optimize_hparams:
        logger.info(f'Training {model}...')
        if hparams_from is not None:
            with open(hparams_from, 'rb') as f:
                hparams = pickle.load(f)
            model = MODELS[model](**hparams)
        else:
            model = MODELS[model]()
        if model_type == 'tabpfn':
            model = TabPFNRegressor(
                n_estimators=32,
                ignore_pretraining_limits=True, 
                inference_config={
                    "SUBSAMPLE_SAMPLES": 10000,
                },
            )
        model.fit(x_train, y_train)
        # Batchify the test data
        y_pred = predict_batch(model, x_test, batch_size=PREDICT_BATCH_SIZE)
        logger.info(f'Scaled MAE: {scaling_error(y_test, y_pred, label): .4f}')
        logger.info(f'MAE: {mean_absolute_error(y_test, y_pred): .4f}')
        logger.info(f'R2: {r2_score(y_test, y_pred): .4f}')
    elif optimize_hparams:
        logger.info(f'Optimizing hyper-parameters for {model}...')
        
        def objective(trial: optuna.Trial, model: str = model) -> float:
            hparams = get_hparams(trial, model)
            if model_type == 'tabpfn':
                hparams.update({
                    'ignore_pretraining_limits': True,
                    'inference_config': {
                        "SUBSAMPLE_SAMPLES": 10000,
                    },
                })
            model = MODELS[model](**hparams)
            model.fit(x_train, y_train)
            y_pred = predict_batch(model, x_val, batch_size=PREDICT_BATCH_SIZE)
            return mean_absolute_error(y_val, y_pred)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        logger.info(f'--------------------------------')
        logger.info(f'{name}')
        logger.info(f'Best hyper-parameters: {study.best_params}')
        hparams = get_hparams(study.best_trial, model)
        hparams.update(study.best_params)
        model = MODELS[model](**hparams)
        model.fit(x_train, y_train)
        y_pred = predict_batch(model, x_test, batch_size=PREDICT_BATCH_SIZE)
        
        logger.info(f'Scaled MAE: {scaling_error(y_test, y_pred, label): .4f}')
        logger.info(f'MAE: {mean_absolute_error(y_test, y_pred): .4f}')
        logger.info(f'R2: {r2_score(y_test, y_pred): .4f}')
    
    # 3. Train production model
    logger.info(f'Training production model...')
    model = MODELS[model_type](**hparams)
    X_total = np.concatenate([x_train, x_val, x_test], axis=0)
    y_total = np.concatenate([y_train, y_val, y_test], axis=0)
    model.fit(X_total, y_total)
    
    # 4. Save model and results
    if model_type == 'tabpfn':
        model.model_path = '/kaggle/input/tabpfn-models/tabpfn-v2-regressor.ckpt'
        save_fitted_tabpfn_model(model, os.path.join(out_dir, f'{name}.tabpfn_fit'))
    else:
        model_path = os.path.join(out_dir, f'{name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    results_path = os.path.join(out_dir, f'{name}.csv')
    pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
    }).to_csv(results_path, index=False)
    
    return scaling_error(y_test, y_pred, label), x_test.shape[0]


def main():
    seed_everything(42)
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.labels is None:
        args.labels = TARGETS
    performance = {}
    n_tests = []
    for label in args.labels:
        scaling_error, n_test = train(
            out_dir=args.out_dir,
            model=args.model,
            label=label,
            feature_names=args.feature_names,
            optimize_hparams=args.optimize_hparams,
            hparams_from=args.hparams_from,
            raw_csv_path=args.raw_csv_path,
            sources=args.sources,
            n_trials=args.n_trials,
            tag=args.tag,
        )    
        performance[label] = scaling_error
        n_tests.append(n_test)

    # 2. Save results
    results_path = os.path.join(REPO_DIR, 'performance.csv')
    df = pd.read_csv(results_path)
    property_weight = normalize_property_weight(n_tests)
    performance['score'] = np.average(list(performance.values()), weights=property_weight)
    performance['model'] = args.model
    performance['extra_info'] = '-'.join(args.feature_names) + f'-{args.tag}'
    new_df = pd.DataFrame(performance, index=[0]).round(4)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(results_path, index=False, float_format="%.4f")

if __name__ == '__main__':
    main()
