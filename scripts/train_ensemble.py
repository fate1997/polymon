import argparse
import os
import pickle

from typing import List, Tuple

import numpy as np
import torch
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, r2_score

from polymon.data.dataset import PolymerDataset
from polymon.exp.score import normalize_property_weight, scaling_error
from polymon.exp.utils import seed_everything
from polymon.setting import REPO_DIR, TARGETS
from polymon.model.base import ModelWrapper
from polymon.model.ensemble import EnsembleRegressor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-csv-path', type=str, default='database/internal/train.csv')
    parser.add_argument('--sources', type=str, nargs='+', default=['official_external'])
    parser.add_argument('--tag', type=str, default='debug')
    parser.add_argument('--labels', choices=TARGETS, nargs='+', default=None)
    parser.add_argument('--model-paths', type=str, nargs='+', default=[])
    parser.add_argument('--out-dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()

PREDICT_BATCH_SIZE = 128


def train_ensemble(
    out_dir: str,
    model_paths: List[str],
    label: str,
    raw_csv_path: str,
    sources: List[str],
    tag: str,
    device: str = 'cpu',
) -> Tuple[float, float]:
    '''
    Train an ensemble of models.
    Args:
        out_dir: The directory to save the results.
        models: The list of trained models.
        label: The label to train.
        feature_names: The feature names to use.
        tag: The tag to save the results.
        sources: The sources to use.
    '''

    out_dir = os.path.join(out_dir, 'ensemble')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, tag), exist_ok=True)
    out_dir = os.path.join(out_dir, tag)
    name = f'{tag}'
    logger.add(os.path.join(out_dir, f'{name}.log'))

    logger.info(f'Training ensemble of {len(model_paths)} models...')
    logger.info(f'Loading dataset...')
    dataset = PolymerDataset(
        raw_csv_path=raw_csv_path,
        feature_names=['x'],
        sources=sources,
        label_column=label,
        force_reload=True,
        add_hydrogens=False,
    )

    smiles_list = [data.smiles for data in dataset]
    y = np.array([data.y for data in dataset]).squeeze(1)

    base_builders = {}
    for i, model_path in enumerate(model_paths):
        if model_path.endswith('pkl'):
            logger.info(f'Loading ML model {model_path}...')
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            base_builders[f'estimator_{i}_ML'] = model

        if model_path.endswith('pt'):
            logger.info(f'Loading DL model {model_path}...')
            model_cls = ModelWrapper
            model = model_cls.from_file(model_path, weights_only=False, map_location=device)
            base_builders[f'estimator_{i}_DL'] = model

    ensemble = EnsembleRegressor(
        base_builders=base_builders,
        meta_weigths=None,
        meta_bias=None,
        random_state=42,
    )
    ensemble.fit(smiles_list, y)
    y_pred = ensemble.predict(smiles_list, batch_size=PREDICT_BATCH_SIZE, device=device)
    y_pred = y_pred.reshape(-1, 1)
    logger.info(f'Scaled MAE: {scaling_error(y, y_pred, label): .4f}')
    logger.info(f'MAE: {mean_absolute_error(y, y_pred): .4f}')
    logger.info(f'R2: {r2_score(y, y_pred): .4f}')

    # 4. Save model and results
    model_path = os.path.join(out_dir, f'{tag}.pt')
    torch.save(ensemble, model_path)
    results_path = os.path.join(out_dir, f'{tag}.csv')
    pd.DataFrame(y_pred).to_csv(results_path, index=False)
    # pd.DataFrame({
    #     'y_true': y,
    #     'y_pred': y_pred,
    # }).to_csv(results_path, index=False)

    return scaling_error(y, y_pred, label), len(y)


def main():
    seed_everything(42)
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.labels is None:
        args.labels = TARGETS
    performance = {}
    n_tests = []
    for label in args.labels:
        scaling_error, n_test = train_ensemble(
            out_dir=args.out_dir,
            model_paths=args.model_paths,
            label=label,
            raw_csv_path=args.raw_csv_path,
            sources=args.sources,
            tag=args.tag,
            device=args.device,
        )
            
        performance[label] = scaling_error
        n_tests.append(n_test)

    # 2. Save results
    results_path = os.path.join(REPO_DIR, 'performance.csv')
    df = pd.read_csv(results_path)
    property_weight = normalize_property_weight(n_tests)
    performance['score'] = np.average(list(performance.values()), weights=property_weight)
    performance['model'] = 'ensemble'
    performance['extra_info'] = f'-{args.tag}'
    new_df = pd.DataFrame(performance, index=[0]).round(4)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(results_path, index=False, float_format="%.4f")

if __name__ == '__main__':
    main()
