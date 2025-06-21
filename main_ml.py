import argparse
import logging
import os
from typing import List, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from tabpfn import TabPFNRegressor
from xgboost import XGBRegressor

from polymon.data.dataset import PolymerDataset
from polymon.exp.score import normalize_property_weight, scaling_error
from polymon.exp.utils import get_logger, loader2numpy, seed_everything
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
    parser.add_argument('--labels', choices=TARGETS, nargs='+', default=None)
    parser.add_argument('--feature-names', type=str, nargs='+', default=['rdkit2d'])
    parser.add_argument('--model', choices=MODELS.keys(), default='rf')
    parser.add_argument('--optimize-hparams', action='store_true')
    parser.add_argument('--n-trials', type=int, default=25)
    parser.add_argument('--out-dir', type=str, default='./results')
    return parser.parse_args()


def train(
    out_dir: str,
    model: str,
    label: str,
    feature_names: List[str],
    optimize_hparams: bool,
    raw_csv_path: str,
    n_trials: int,
) -> Tuple[float, float]:
    seed_everything(42)
    out_dir = os.path.join(out_dir, model)
    os.makedirs(out_dir, exist_ok=True)
    name = f'{model}-{"-".join(feature_names)}'
    logger = get_logger(out_dir, name)

    # 1. Load data
    logger.info(f'Training {label}...')
    dataset = PolymerDataset(
        raw_csv_path=raw_csv_path,
        feature_names=feature_names,
        label_column=label,
        force_reload=True,
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
        model = MODELS[model]()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        logger.info(f'Scaled MAE: {scaling_error(y_test, y_pred, label): .4f}')
        logger.info(f'MAE: {mean_absolute_error(y_test, y_pred): .4f}')
        logger.info(f'R2: {r2_score(y_test, y_pred): .4f}')
    else:
        logger.info(f'Optimizing hyper-parameters for {model}...')
        
        def objective(trial: optuna.Trial, model: str = model) -> float:
            hparams = get_hparams(trial, model)
            model = MODELS[model](**hparams)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_val)
            return mean_absolute_error(y_val, y_pred)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        logger.info(f'--------------------------------')
        logger.info(f'{name}')
        logger.info(f'Best hyper-parameters: {study.best_params}')
        model = MODELS[model](**study.best_params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        logger.info(f'Scaled MAE: {scaling_error(y_test, y_pred, label): .4f}')
        logger.info(f'MAE: {mean_absolute_error(y_test, y_pred): .4f}')
        logger.info(f'R2: {r2_score(y_test, y_pred): .4f}')
    
    # 3. Save model and results
    model_path = os.path.join(out_dir, f'{name}.pth')
    torch.save(model, model_path)
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
            raw_csv_path=args.raw_csv_path,
            n_trials=args.n_trials,
        )    
        performance[label] = scaling_error
        n_tests.append(n_test)

    # 2. Save results
    results_path = os.path.join(REPO_DIR, 'performance.csv')
    df = pd.read_csv(results_path)
    property_weight = normalize_property_weight(n_tests)
    performance['score'] = np.average(list(performance.values()), weights=property_weight)
    performance['model'] = args.model
    performance['extra_info'] = '-'.join(args.feature_names)
    new_df = pd.DataFrame(performance, index=[0]).round(4)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(results_path, index=False)

if __name__ == '__main__':
    main()