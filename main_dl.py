import argparse
import logging
import os
from typing import Any, Dict

import numpy as np
import optuna
import pandas as pd
import torch
import yaml
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader

from polymon.data.dataset import PolymerDataset
from polymon.exp.score import normalize_property_weight
from polymon.exp.train import Trainer
from polymon.exp.utils import Normalizer, get_logger, seed_everything
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
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--descriptors', type=str, default=None, nargs='+')

    # Training
    parser.add_argument('--num-epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ema-decay', type=float, default=0.0)
    parser.add_argument('--early-stopping-patience', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n-trials', type=int, default=10)
    parser.add_argument('--optimize-hparams', action='store_true')

    return parser.parse_args()


def train(config: dict, out_dir: str, label: str):
    seed_everything(42)
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger(out_dir, 'training')
    
    # 1. Load dataset
    logger.info('Loading dataset...')
    feature_names = ['x', 'bond', 'z']
    if config['model'].lower() in ['dimenetpp']:
        feature_names.append('pos')
    if config['descriptors'] is not None:
        feature_names.extend(config['descriptors'])
    dataset = PolymerDataset(
        raw_csv_path=config['raw_csv_path'],
        feature_names=feature_names,
        label_column=label,
        force_reload=True,
    )
    
    logger.info(f'Number of atom features: {dataset[0].x.shape[1]}')

    with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
        save_config = config.copy()
        yaml.dump(save_config, f)
    
    # 3. Create model
    logger.info('Creating model...')
    num_descriptors = dataset[0].descriptors.shape[1] if config['descriptors'] is not None else 0
    logger.info(f'Number of descriptors used: {num_descriptors}')

    def build_model(hparams: Dict[str, Any]):
        if config['model'].lower() == 'gatv2':
            return GATv2(
                num_atom_features=dataset.num_node_features,
                hidden_dim=hparams['hidden_dim'],
                num_layers=hparams['num_layers'],
                edge_dim=dataset.num_edge_features,
                num_descriptors=num_descriptors,
            )   
        elif config['model'].lower() == 'attentivefp':
            return AttentiveFPWrapper(
                in_channels=dataset.num_node_features,
                hidden_channels=hparams['hidden_channels'],
                out_channels=1,
                edge_dim=dataset.num_edge_features,
                num_layers=hparams['num_layers'],   
            )
        elif config['model'].lower() == 'dimenetpp':
            return DimeNetPP(
                hidden_channels=hparams['hidden_channels'],
                out_channels=1,
                num_blocks=hparams['num_blocks'],
            )
        else:
            raise NotImplementedError(f"Model type {config['model']} not implemented")
        
    if config.get('optimize_hparams', False):
        logger.info('Optimizing hyper-parameters... for {}'.format(config['model']))

        def objective(trial: optuna.Trial, model: str = config['model']) -> float:
            model_hparams = get_hparams(trial, model)
            train_hparams = {
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            }
            train_loader, val_loader, test_loader = dataset.get_loaders(
                batch_size=train_hparams['batch_size'],
                n_train=0.8,
                n_val=0.1,
            )
            norm = Normalizer.from_loader(train_loader)
            model = build_model(model_hparams)
            trainer = Trainer(
                out_dir=out_dir,
                model=model,
                lr=train_hparams['lr'],
                num_epochs=config['num_epochs'],
                normalizer=norm,
                logger=logger,
                ema_decay=config['ema_decay'],
                device=config['device'],
                early_stopping_patience=config['early_stopping_patience'],
            )
            val_err = trainer.train(train_loader, val_loader, test_loader, label)
            return val_err
    
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials = config['n_trials'])
        logger.info(f'--------------------------------')
        logger.info(f'{config["model"]}')
        logger.info(f'Best hyper-parameters: {study.best_params}')
        hparams = get_hparams(study.best_trial, config['model'])
        hparams.update(study.best_params)
        config.update(hparams)
    
    
    logger.info('Training production model...')
    train_loader, val_loader, test_loader = dataset.get_loaders(
        batch_size=config['batch_size'],
        n_train=0.8,
        n_val=0.1,
    )
    normalizer = Normalizer.from_loader(train_loader)
    final_model = build_model(hparams if config.get('optimize_hparams') else config)
    params = sum(p.numel() for p in final_model.parameters())
    logger.info(f'Model parameters: {params / 1e6:.2f}M')
    

    trainer = Trainer(
        out_dir=out_dir,
        model=final_model,
        lr=config['lr'],
        num_epochs=config['num_epochs'],
        normalizer=normalizer,
        logger=logger,
        ema_decay=config['ema_decay'],
        device=config['device'],
        early_stopping_patience=config['early_stopping_patience'],
    )
    scaling_error = trainer.train(train_loader, val_loader, test_loader, label)
    
    # 5. Production Run Training Without Test Set
    split = int(len(dataset) - 0.05 * len(dataset))
    out_dir = os.path.join(out_dir, 'production')
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f'Production run training without test set in {out_dir}')
    dataset = dataset.shuffle()
    train_loader = DataLoader(
        dataset[:split], batch_size=config['batch_size'], shuffle=True
    )
    val_loader = DataLoader(
        dataset[split:], batch_size=config['batch_size'], shuffle=True
    )
    trainer = Trainer(
        out_dir=out_dir,
        model=model,
        lr=config['lr'],
        num_epochs=100,
        normalizer=normalizer,
        logger=logger,
        ema_decay=config['ema_decay'],
        device=config['device'],
        early_stopping_patience=config['early_stopping_patience'],
    )
    trainer.train(train_loader, val_loader, label=label)
    
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
        out_dir = os.path.join(args.out_dir, args.model, label, args.tag)
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
    performance['extra_info'] = f'{args.tag}-{args.hidden_dim}-{args.num_layers}-{args.batch_size}-{args.lr}-{args.num_epochs}'
    new_df = pd.DataFrame(performance, index=[0]).round(4)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(results_path, index=False, float_format="%.4f")


if __name__ == '__main__':
    main()