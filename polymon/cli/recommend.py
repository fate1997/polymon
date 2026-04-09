import argparse
from typing import Literal

import pandas as pd
import torch

from polymon.exp.acquisition import Acquisition
from polymon.setting import REPO_DIR


def recommend(
    pool_csv: str,
    trained_model: str,
    model_type: Literal['kfold', 'ensemble'] = 'kfold',
    acquisition: Literal['uncertainty', 'random', 'epig'] = 'uncertainty',
    sample_size: int = 100,
    save_path: str = None,
):
    
    pool_smiles = pd.read_csv(pool_csv)['SMILES'].tolist()[:100]
    scorer = Acquisition(
        acquisition_function= acquisition,
        model_file = trained_model,
        model_type=model_type,
        device='cpu',
    )
    scores = scorer.score(pool_smiles)
    top_n_smiles = [pool_smiles[i] for i in torch.topk(scores, sample_size).indices.tolist()]
    if save_path is not None:
        save_path = REPO_DIR / 'database' / save_path
        pd.DataFrame({'SMILES': top_n_smiles}).to_csv(save_path, index=False)
    return top_n_smiles

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool-csv', type=str, required=True)
    parser.add_argument('--trained-model', type=str, required=True)
    parser.add_argument('--model-type', type=str, required=True, choices=['kfold', 'ensemble'])
    parser.add_argument('--acquisition', type=str, required=True, choices=['epig', 'uncertainty', 'random'])
    parser.add_argument('--sample-size', type=int, default=20)
    parser.add_argument('--save-path', type=str, default=None)
    return parser.parse_args()

def main(args: argparse.Namespace):
    recommend(
        pool_csv=args.pool_csv,
        trained_model=args.trained_model,
        model_type=args.model_type,
        acquisition=args.acquisition,
        sample_size=args.sample_size,
        save_path=args.save_path,
    )

if __name__ == '__main__':
    args = arg_parser()
    main(args)