import argparse
import os
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader
from xgboost import XGBRegressor

from polymon.data.dataset import PolymerDataset
from polymon.exp.utils import get_logger
from polymon.setting import TARGETS

#!TODO: Add hyper-parameter tuning

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-csv-path', type=str, default='database/internal/train.csv')
    parser.add_argument('--label', choices=TARGETS, default='Tg')
    parser.add_argument('--model', choices=['rf', 'xgb'], default='rf')
    parser.add_argument('--out-dir', type=str, default='./results')
    return parser.parse_args()


def loader2numpy(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    x = []
    y = []
    for batch in loader:
        x.append(batch.descriptors.numpy())
        y.append(batch.y.numpy())
    return np.concatenate(x, 0), np.concatenate(y, 0)

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    logger = get_logger(args.out_dir, 'ml')
    
    logger.info(f'Loading dataset from {args.raw_csv_path}...')
    dataset = PolymerDataset(
        raw_csv_path=args.raw_csv_path,
        feature_names=['desc'],
        label_column=args.label,
    )
    train_loader, val_loader, test_loader = dataset.get_loaders(
        batch_size=128,
        n_train=400,
        n_val=100,
    )
    
    logger.info('Loading data...')
    x_train, y_train = loader2numpy(train_loader)
    x_val, y_val = loader2numpy(val_loader)
    x_test, y_test = loader2numpy(test_loader)
    
    logger.info(f'Training {args.model}...')
    if args.model == 'rf':
        model = RandomForestRegressor()
    elif args.model == 'xgb':
        model = XGBRegressor()
    else:
        raise ValueError(f'Invalid model: {args.model}')
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    logger.info(f'MAE: {mean_absolute_error(y_val, y_pred): .4f}')
    logger.info(f'R2: {r2_score(y_val, y_pred): .4f}')


if __name__ == '__main__':
    main()