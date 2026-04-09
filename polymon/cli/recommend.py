"""Active learning recommendation module for PolyMon.

This module provides functionality for recommending molecules to label next
in an active learning loop. Supports multiple acquisition functions including
uncertainty sampling, expected improvement (EPIG), and random sampling.
"""

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
    """Recommend molecules from a pool for experimental labeling.

    This function scores molecules in a pool using an acquisition function
    and returns the top-N molecules with highest acquisition scores.

    Args:
        pool_csv: Path to CSV file with candidate molecules. Must contain a
            'SMILES' column with polymer SMILES strings.
        trained_model: Path to trained model file (.pt or .pkl).
        model_type: Type of model - 'kfold' for K-fold models or 'ensemble'
            for ensemble models.
        acquisition: Acquisition function to use:
            - 'uncertainty': Select molecules with highest prediction uncertainty
            - 'epig': Expected improvement in generalization
            - 'random': Random sampling (baseline)
        sample_size: Number of molecules to recommend.
        save_path: Optional path to save recommended molecules as CSV.
            If None, saves to database/ directory.

    Returns:
        List of SMILES strings for the recommended molecules.

    Example:
        >>> recommended = recommend(
        ...     pool_csv='database/pool.csv',
        ...     trained_model='results/gatv2/Rg/model.pt',
        ...     acquisition='uncertainty',
        ...     sample_size=20,
        ...     save_path='recommended.csv'
        ... )
        >>> print(f"Recommended {len(recommended)} molecules")
    """
    # Load pool SMILES (limited to first 100 for memory)
    pool_smiles = pd.read_csv(pool_csv)['SMILES'].tolist()[:100]

    # Initialize acquisition scorer
    scorer = Acquisition(
        acquisition_function=acquisition,
        model_file=trained_model,
        model_type=model_type,
        device='cpu',
    )

    # Score all molecules in pool
    scores = scorer.score(pool_smiles)

    # Get top-N molecules by acquisition score
    top_indices = torch.topk(scores, sample_size).indices.tolist()
    top_n_smiles = [pool_smiles[i] for i in top_indices]

    # Save results if path provided
    if save_path is not None:
        save_path = REPO_DIR / 'database' / save_path
        pd.DataFrame({'SMILES': top_n_smiles}).to_csv(save_path, index=False)
        print(f"Recommended molecules saved to: {save_path}")

    return top_n_smiles


def arg_parser():
    """Parse command-line arguments for the recommend command.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Recommend molecules for active learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--pool-csv',
        type=str,
        required=True,
        help='Path to CSV file with candidate molecules (must have SMILES column)'
    )
    parser.add_argument(
        '--trained-model',
        type=str,
        required=True,
        help='Path to trained model (.pt or .pkl file)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        required=True,
        choices=['kfold', 'ensemble'],
        help='Type of model: kfold or ensemble'
    )
    parser.add_argument(
        '--acquisition',
        type=str,
        required=True,
        choices=['epig', 'uncertainty', 'random'],
        help='Acquisition function: epig (expected improvement), uncertainty, or random'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=20,
        help='Number of molecules to recommend'
    )
    parser.add_argument(
        '--save-path',
        type=str,
        default=None,
        help='Path to save recommended molecules (CSV format)'
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    """Main entry point for the recommend command.

    Args:
        args: Parsed command-line arguments.
    """
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
