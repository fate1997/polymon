import argparse
import os
from typing import Any, Dict, Optional, List

import numpy as np
import optuna
import pandas as pd
import yaml
import loguru
from torch import nn
from torch_geometric.loader import DataLoader

from polymon.data.dataset import PolymerDataset
from polymon.exp.score import normalize_property_weight
from polymon.exp.train import Trainer
from polymon.model.base import ModelWrapper
from polymon.exp.pipeline import Pipeline
from polymon.hparams import get_hparams
from polymon.model.gnn import AttentiveFPWrapper, DimeNetPP, GATv2
from polymon.setting import REPO_DIR, TARGETS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='test')
    parser.add_argument('--out-dir', type=str, default='./results')
    
    # Dataset
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--raw-csv-path', type=str, default='database/internal/train.csv')
    parser.add_argument('--labels', choices=TARGETS, nargs='+', default=None)

    # Model
    parser.add_argument(
        '--model', 
        choices=['gatv2', 'attentivefp', 'dimenetpp'], 
        default='gatv2'
    )
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--descriptors', type=str, default=None, nargs='+')

    # Training
    parser.add_argument('--num-epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--early-stopping-patience', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n-trials', type=int, default=25)
    parser.add_argument('--optimize-hparams', action='store_true')
    parser.add_argument('--run-production', action='store_true')
    parser.add_argument('--finetune-csv-path', type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)
    performance = {}
    n_tests = []
    if args.labels is None:
        args.labels = TARGETS
    for label in args.labels:
        out_dir = os.path.join(args.out_dir, args.model, label, args.tag)
        pipeline = Pipeline(
            tag=args.tag,
            out_dir=out_dir,
            batch_size=args.batch_size,
            raw_csv_path=args.raw_csv_path,
            label=label,
            model_type=args.model,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            descriptors=args.descriptors,
            num_epochs=args.num_epochs,
            lr=args.lr,
            early_stopping_patience=args.early_stopping_patience,
            device=args.device,
            n_trials=args.n_trials,
        )
        if args.optimize_hparams:
            test_err, hparams = pipeline.optimize_hparams()
            model_path = os.path.join(out_dir, 'hparams_opt', f'{pipeline.model_name}.pt')
        else:
            test_err = pipeline.train()
            hparams = {'hidden_dim': args.hidden_dim, 'num_layers': args.num_layers}
            model_path = os.path.join(out_dir, 'train', f'{pipeline.model_name}.pt')
        if args.finetune_csv_path is not None:
            test_err = pipeline.finetune(
                0.001, 
                args.finetune_csv_path, 
                model_path, 
                args.run_production
            )
        elif args.run_production:
            pipeline.production_run(hparams)
        
        performance[label] = test_err
        n_tests.append(len(pipeline.test_loader.dataset))

    results_path = os.path.join(REPO_DIR, 'performance.csv')
    df = pd.read_csv(results_path)
    property_weight = normalize_property_weight(n_tests)
    performance['score'] = np.average(list(performance.values()), weights=property_weight)
    performance['model'] = args.model
    performance['extra_info'] = f'{args.tag}-{args.hidden_dim}-{args.num_layers}-{args.batch_size}-{args.lr}-{args.num_epochs}'
    new_df = pd.DataFrame(performance, index=[0]).round(4)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(results_path, index=False, float_format="%.4f")


if __name__ == '__main__':
    main()