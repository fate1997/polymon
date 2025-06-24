import argparse
import os
import logging

import torch
import yaml
import pandas as pd
import numpy as np
import torch_geometric.transforms as T

from polymon.data.dataset import PolymerDataset
from polymon.exp.utils import get_logger, Normalizer
from polymon.model.gnn import GATv2, AttentiveFPWrapper, DimeNetPP, GPS
from polymon.exp.train import Trainer
from polymon.exp.utils import seed_everything
from polymon.setting import TARGETS, REPO_DIR
from polymon.exp.score import normalize_property_weight


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, default='debug')
    parser.add_argument('--out-dir', type=str, default='./results')
    
    # Dataset
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--raw-csv-path', type=str, default='database/internal/train.csv')
    parser.add_argument('--labels', choices=TARGETS, nargs='+', default=None)

    # Model
    parser.add_argument(
        '--model', 
        choices=['gatv2', 'attentivefp', 'dimenetpp', 'gps'], 
        default='gatv2'
    )
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=2)

    # Training
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ema-decay', type=float, default=0.0)
    parser.add_argument('--early-stopping-patience', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


def train(config: dict, out_dir: str, label: str):
    seed_everything(42)
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger(out_dir, 'training')
    
    # 1. Load dataset
    logger.info('Loading dataset...')
    feature_names = ['x', 'bond', 'z', 'degree', 'is_aromatic']
    if config['model'].lower() in ['dimenetpp']:
        feature_names.append('pos')
    
    if config['model'].lower() == 'gps':
        pre_transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    else:
        pre_transform = None
    dataset = PolymerDataset(
        raw_csv_path=config['raw_csv_path'],
        feature_names=feature_names,
        label_column=label,
        force_reload=True,
        pre_transform=pre_transform,
    )
    train_loader, val_loader, test_loader = dataset.get_loaders(
        batch_size=config['batch_size'],
        n_train=0.8,
        n_val=0.1,
    )
    normalizer = Normalizer.from_loader(train_loader)
    logger.info(
        f'Train: {len(train_loader.dataset)}, '
        f'Val: {len(val_loader.dataset)}, '
        f'Test: {len(test_loader.dataset)}'
    )
    with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
        save_config = config.copy()
        yaml.dump(save_config, f)
    
    # 3. Create model
    logger.info('Creating model...')
    if config['model'].lower() == 'gatv2':
        model = GATv2(
            num_atom_features=dataset.num_node_features,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            edge_dim=dataset.num_edge_features,
        )
    elif config['model'].lower() == 'attentivefp':
        model = AttentiveFPWrapper(
            in_channels=dataset.num_node_features,
            hidden_channels=config['hidden_dim'],
            out_channels=1,
            edge_dim=dataset.num_edge_features,
            num_layers=config['num_layers'],
        )
    elif config['model'].lower() == 'dimenetpp':
        model = DimeNetPP(
            hidden_channels=config['hidden_dim'],
            out_channels=1,
            num_blocks=config['num_layers'],
        )
    elif config['model'].lower() == 'gps':
        model = GPS(
            num_atom_features=dataset.num_node_features,
            channels=config['hidden_dim'],
            pe_dim=8,
            num_layers=config['num_layers'],
            edge_dim=dataset.num_edge_features,
            attn_type='performer',
            attn_kwargs={'dropout': 0.5},
        )
    else:
        raise NotImplementedError(f"Model type {config['model']} not implemented")
    
    params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model parameters: {params / 1e6:.2f}M')
    
    # 4. Train model
    logger.info('Training model...')
    trainer = Trainer(
        out_dir=out_dir,
        model=model,
        lr=config['lr'],
        num_epochs=config['num_epochs'],
        normalizer=normalizer,
        logger=logger,
        ema_decay=config['ema_decay'],
        device=config['device'],
        early_stopping_patience=config['early_stopping_patience'],
    )
    scaling_error = trainer.train(train_loader, val_loader, test_loader, label)
    return scaling_error, len(test_loader)


def main():
    args = parse_args()
    out_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)
    performance = {}
    n_tests = []
    if args.labels is None:
        args.labels = TARGETS
    for label in args.labels:
        out_dir = os.path.join(args.out_dir, args.model, label)
        scaling_error, n_test = train(
            config=args.__dict__,
            label=label,
            out_dir=out_dir,
        )
        performance[label] = scaling_error
        n_tests.append(n_test)
    results_path = os.path.join(REPO_DIR, 'performance.csv')
    df = pd.read_csv(results_path)
    property_weight = normalize_property_weight(n_tests)
    performance['score'] = np.average(list(performance.values()), weights=property_weight)
    performance['model'] = args.model
    performance['extra_info'] = f'{args.hidden_dim}-{args.num_layers}-{args.batch_size}-{args.lr}-{args.num_epochs}'
    new_df = pd.DataFrame(performance, index=[0]).round(4)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(results_path, index=False)


if __name__ == '__main__':
    main()