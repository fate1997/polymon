import argparse
import os
from typing import Tuple

import numpy as np
import optuna
import pandas as pd
import torch
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from tabpfn import TabPFNRegressor
from torch.utils.data import DataLoader
from xgboost import XGBRegressor

from polymon.data.dataset import PolymerDataset
from polymon.exp.utils import get_logger, loader2numpy, seed_everything
from polymon.hparams import get_hparams
from polymon.setting import TARGETS
from polymon.exp.score import scaling_error

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
    parser.add_argument('--label', choices=TARGETS, default='Tg')
    parser.add_argument('--feature-names', type=str, nargs='+', default=['rdkit2d'])
    parser.add_argument('--model', choices=MODELS.keys(), default='rf')
    parser.add_argument('--optimize-hparams', action='store_true')
    parser.add_argument('--out-dir', type=str, default='./results')
    return parser.parse_args()


def main():
    seed_everything(42)
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    name = f'{args.model}-{args.label}-{"-".join(args.feature_names)}'
    logger = get_logger(args.out_dir, name)
    
    # 1. Load data
    logger.info(f'Loading dataset from {args.raw_csv_path}...')
    dataset = PolymerDataset(
        raw_csv_path=args.raw_csv_path,
        feature_names=args.feature_names,
        label_column=args.label,
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
    if args.label == 'FFV' and 'rdkit2d' in args.feature_names:
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
    if not args.optimize_hparams:
        logger.info(f'Training {args.model}...')
        model = MODELS[args.model]()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        logger.info(f'Scaled MAE: {scaling_error(y_test, y_pred, args.label): .4f}')
        logger.info(f'MAE: {mean_absolute_error(y_test, y_pred): .4f}')
        logger.info(f'R2: {r2_score(y_test, y_pred): .4f}')
    else:
        logger.info(f'Optimizing hyper-parameters for {args.model}...')
        
        def objective(trial: optuna.Trial) -> float:
            hparams = get_hparams(trial, args.model)
            model = MODELS[args.model](**hparams)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_val)
            return mean_absolute_error(y_val, y_pred)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=25)
        logger.info(f'Best hyper-parameters: {study.best_params}')
        model = MODELS[args.model](**study.best_params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        logger.info(f'Scaled MAE: {scaling_error(y_test, y_pred, args.label): .4f}')
        logger.info(f'MAE: {mean_absolute_error(y_test, y_pred): .4f}')
        logger.info(f'R2: {r2_score(y_test, y_pred): .4f}')
    
    # 3. Save model and results
    model_path = os.path.join(args.out_dir, f'{name}.pth')
    torch.save(model, model_path)
    results_path = os.path.join(args.out_dir, f'{name}.csv')
    pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
    }).to_csv(results_path, index=False)


if __name__ == '__main__':
    main()