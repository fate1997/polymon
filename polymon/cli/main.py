"""PolyMon Command-Line Interface.

This module provides the main CLI entry point for the PolyMon framework.
It supports three main modes:
    - train: Train ML/DL models for polymer property prediction
    - rec: Active learning - recommend molecules for labeling
    - predict: Make predictions on new polymer data

Example:
    Train a model:
        $ polymon train --labels Tg --model rf --n-fold 5

    Make predictions:
        $ polymon predict --model-path model.pt --csv-path data.csv --smiles-column SMILES

    Recommend molecules:
        $ polymon rec --pool-csv pool.csv --trained-model model.pt --acquisition uncertainty
"""

import argparse

from polymon.cli.train_dl import main as main_dl
from polymon.cli.train_ml import MODELS
from polymon.cli.train_ml import main as main_ml
from polymon.cli.predict import main as main_predict
from polymon.cli.recommend import main as main_recommend


def parse_args():
    """Parse command-line arguments for the PolyMon CLI.

    Returns:
        argparse.Namespace: Parsed arguments with the following attributes:
            - mode (str): Subcommand to run ('train', 'rec', or 'predict')
            - ... (additional arguments depend on the subcommand)
    """
    parser = argparse.ArgumentParser(
        description='PolyMon - Polymer Property Prediction Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # ============================================================================
    # Train subcommand
    # ============================================================================
    train_parser = subparsers.add_parser(
        'train',
        help='Train a ML/DL model for polymer property prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument(
        '--raw-csv',
        type=str,
        default='database/database.csv',
        help='Path to the raw CSV file containing polymer data'
    )
    train_parser.add_argument(
        '--sources',
        type=str,
        nargs='+',
        default=['Kaggle'],
        help='Data sources to filter from the dataset (e.g., Kaggle, PI1070, PolyMetriX)'
    )
    train_parser.add_argument(
        '--tag',
        type=str,
        default='debug',
        help='Tag for organizing this training run'
    )
    train_parser.add_argument(
        '--labels',
        nargs='+',
        required=True,
        choices=['Tg', 'FFV', 'Density', 'Rg', 'Tc'],
        help='Target property/properties to predict'
    )
    train_parser.add_argument(
        '--feature-names',
        type=str,
        nargs='+',
        default=['rdkit2d'],
        help='Feature names for tabular models (rdkit2d, ecfp4, mordred, maccs, xenonpy_desc)'
    )
    train_parser.add_argument(
        '--n-trials',
        type=int,
        default=None,
        help='Number of trials for hyperparameter optimization (enables optimization if set)'
    )
    train_parser.add_argument(
        '--out-dir',
        type=str,
        default='./results',
        help='Directory to save training results'
    )
    train_parser.add_argument(
        '--hparams-from',
        type=str,
        default=None,
        help='Path to hyperparameters file (.json, .pt, or .pkl) to reuse from previous run'
    )
    train_parser.add_argument(
        '--n-fold',
        type=int,
        default=1,
        help='Number of folds for cross-validation (use 1 for single split)'
    )
    train_parser.add_argument(
        '--split-mode',
        type=str,
        default='random',
        choices=['random', 'source', 'scaffold'],
        help='Data splitting strategy: random, by source, or by molecular scaffold'
    )
    train_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    train_parser.add_argument(
        '--remove-hydrogens',
        action='store_true',
        help='Remove hydrogens from molecular graphs (reduces computation)'
    )
    train_parser.add_argument(
        '--descriptors',
        type=str,
        nargs='+',
        default=None,
        help='Additional descriptors to concatenate with graph features for DL models'
    )
    train_parser.add_argument(
        '--model',
        type=str,
        default='rf',
        help='Model type: tabular (rf, xgb, lgbm, catboost, tabpfn) or GNN (gatv2, gin, pna, etc.)'
    )
    train_parser.add_argument(
        '--hidden-dim',
        type=int,
        default=32,
        help='Hidden dimension for neural network models'
    )
    train_parser.add_argument(
        '--num-layers',
        type=int,
        default=3,
        help='Number of layers for neural network models'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for training neural networks'
    )
    train_parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate for neural network training'
    )
    train_parser.add_argument(
        '--num-epochs',
        type=int,
        default=2500,
        help='Maximum number of training epochs'
    )
    train_parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=250,
        help='Patience for early stopping (epochs without improvement)'
    )
    train_parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for training (cuda or cpu)'
    )
    train_parser.add_argument(
        '--run-production',
        action='store_true',
        help='Run in production mode (95:5 train:val split, no test set)'
    )
    train_parser.add_argument(
        '--finetune',
        action='store_true',
        help='Fine-tune a pretrained model on new data'
    )
    train_parser.add_argument(
        '--finetune-csv-path',
        type=str,
        default=None,
        help='Path to CSV file for fine-tuning data'
    )
    train_parser.add_argument(
        '--pretrained-model',
        type=str,
        default=None,
        help='Path to pretrained model for fine-tuning'
    )
    train_parser.add_argument(
        '--n-estimator',
        type=int,
        default=1,
        help='Number of estimators for ensemble learning (>1 enables ensemble mode)'
    )
    train_parser.add_argument(
        '--additional-features',
        type=str,
        nargs='+',
        default=None,
        help='Additional graph features (monomer, periodic_bond, etc.)'
    )
    train_parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Skip training step (use with --n-estimator to only build ensemble)'
    )
    train_parser.add_argument(
        '--low-fidelity-model',
        type=str,
        default=None,
        help='Path to low-fidelity model for residual learning'
    )
    train_parser.add_argument(
        '--estimator-name',
        type=str,
        default=None,
        help='Name of empirical estimator for delta-learning (e.g., Density-IBM, Rg-monomer)'
    )
    train_parser.add_argument(
        '--emb-model',
        type=str,
        default=None,
        help='Path to embedding model for knowledge transfer'
    )
    train_parser.add_argument(
        '--ensemble-type',
        type=str,
        default='voting',
        choices=['voting', 'bagging', 'gradient_boosting', 'snapshot', 'soft_gradient_boosting'],
        help='Type of ensemble to use'
    )
    train_parser.add_argument(
        '--train-residual',
        action='store_true',
        help='Train on residuals from base estimator or low-fidelity model'
    )
    train_parser.add_argument(
        '--normalizer-type',
        type=str,
        default='normalizer',
        choices=['normalizer', 'log_normalizer', 'none'],
        help='Type of label normalization'
    )
    train_parser.add_argument(
        '--augmentation',
        action='store_true',
        help='Use data augmentation (oligomer building)'
    )

    # ============================================================================
    # Recommend subcommand
    # ============================================================================
    recommend_parser = subparsers.add_parser(
        'rec',
        help='Recommend molecules for active learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    recommend_parser.add_argument(
        '--pool-csv',
        type=str,
        required=True,
        help='Path to CSV file with candidate molecules (must have SMILES column)'
    )
    recommend_parser.add_argument(
        '--trained-model',
        type=str,
        required=True,
        help='Path to trained model (.pt or .pkl file)'
    )
    recommend_parser.add_argument(
        '--model-type',
        type=str,
        default='kfold',
        choices=['kfold', 'ensemble'],
        help='Type of model: kfold or ensemble'
    )
    recommend_parser.add_argument(
        '--acquisition',
        type=str,
        default='uncertainty',
        choices=['epig', 'uncertainty', 'random'],
        help='Acquisition function: epig (expected improvement), uncertainty, or random'
    )
    recommend_parser.add_argument(
        '--sample-size',
        type=int,
        default=100,
        help='Number of molecules to recommend'
    )
    recommend_parser.add_argument(
        '--save-path',
        type=str,
        default=None,
        help='Path to save recommended molecules (CSV format)'
    )

    # ============================================================================
    # Predict subcommand
    # ============================================================================
    predict_parser = subparsers.add_parser(
        'predict',
        help='Make predictions on new polymer data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    predict_parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model (.pt or .pkl file)'
    )
    predict_parser.add_argument(
        '--csv-path',
        type=str,
        required=True,
        help='Path to CSV file with molecules to predict'
    )
    predict_parser.add_argument(
        '--smiles-column',
        type=str,
        required=True,
        help='Name of column containing SMILES strings'
    )

    return parser.parse_args()


def main():
    """Main entry point for the PolyMon CLI.

    Parses command-line arguments and routes to the appropriate subcommand handler:
    - train: Routes to ML or DL training based on model type
    - rec: Active learning recommendations
    - predict: Inference on new data
    """
    args = parse_args()
    if args.mode == 'train':
        # Route to appropriate training function
        if args.model in MODELS.keys():
            main_ml(args)
        else:
            main_dl(args)
    elif args.mode == 'rec':
        main_recommend(args)
    elif args.mode == 'predict':
        main_predict(args)


if __name__ == '__main__':
    main()
