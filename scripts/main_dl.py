import argparse
import yaml
import os
import json

import numpy as np
import pandas as pd
import torch

from polymon.exp.pipeline import Pipeline
from polymon.exp.score import normalize_property_weight
from polymon.setting import REPO_DIR, TARGETS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='test')
    parser.add_argument('--out-dir', type=str, default='./results')
    
    # Dataset
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--raw-csv', type=str, default=str(REPO_DIR / 'database' / 'database.csv'))
    ## Internal data is `must_keep`
    parser.add_argument('--sources', type=str, nargs='+', default=['official_external'])
    parser.add_argument('--labels', choices=TARGETS, nargs='+', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split-mode', type=str, default='random')

    # Model
    parser.add_argument(
        '--model', 
        type=str, 
        default='gatv2'
    )
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--descriptors', type=str, default=None, nargs='+')
    parser.add_argument('--hparams-from', type=str, default=None)

    # Training
    parser.add_argument('--num-epochs', type=int, default=2500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--early-stopping-patience', type=int, default=250)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n-trials', type=int, default=25)
    parser.add_argument('--optimize-hparams', action='store_true')
    parser.add_argument('--run-production', action='store_true')
    parser.add_argument('--finetune-csv-path', type=str, default=None)
    parser.add_argument('--n-fold', type=int, default=1)

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
            raw_csv=args.raw_csv,
            sources=args.sources,
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
            seed=args.seed,
            split_mode=args.split_mode,
        )
        with open(os.path.join(out_dir, 'args.yaml'), 'w') as f:
            yaml.dump(args.__dict__, f)
        
        # CHOICE 1: Optimize hyperparameters OR train model on default parameters
        if args.optimize_hparams:
            test_err, hparams = pipeline.optimize_hparams(n_fold=args.n_fold)
            model_path = os.path.join(out_dir, 'hparams_opt', f'{pipeline.model_name}.pt')
        else:
            hparams = {
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers}
            if args.hparams_from is not None:
                if args.hparams_from.endswith('.pt'):
                    hparams_loaded = torch.load(args.hparams_from)['model_init_params']
                else:
                    with open(args.hparams_from, 'r') as f:
                        hparams_loaded = json.load(f)
                pipeline.logger.info(f'Loading hparams from {args.hparams_from}')
                hparams.update(hparams_loaded)
            
            # CHOICE 2: Train model on pre-defined split OR run K-Fold cross-validation
            if args.n_fold == 1:
                test_err = pipeline.train(model_hparams=hparams)
            else:
                test_err = pipeline.cross_validation(
                    n_fold=args.n_fold,
                    model_hparams=hparams,
                )
                test_err = np.mean(test_err)
            model_path = os.path.join(out_dir, 'train', f'{pipeline.model_name}.pt')
        
        # CHOICE 3: Finetune model OR run production
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
    performance['extra_info'] = f'{args.tag}'
    if args.n_fold > 1:
        performance['extra_info'] += f' (K-Fold)'
    if args.optimize_hparams:
        performance['extra_info'] += f' (Optuna)'
    if args.split_mode != 'random':
        performance['extra_info'] += f' ({args.split_mode})'
    new_df = pd.DataFrame(performance, index=[0]).round(4)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(results_path, index=False, float_format="%.4f")


if __name__ == '__main__':
    main()