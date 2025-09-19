import argparse

from polymon.cli.merge import main as main_merge
from polymon.cli.train_dl import main as main_dl
from polymon.cli.train_ml import MODELS
from polymon.cli.train_ml import main as main_ml
from polymon.cli.predict import main as main_predict



def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', required=True)
    
    # Train
    train_parser = subparsers.add_parser('train', help='Train a ML/DL model')
    train_parser.add_argument(
        '--raw-csv', 
        type=str, 
        default='database/database.csv',
        help='Path to the raw csv file'
    )
    train_parser.add_argument(
        '--sources', 
        type=str, 
        nargs='+', 
        default=['Kaggle'],
        help='Sources to use for training'
    )
    train_parser.add_argument(
        '--tag', 
        type=str, 
        default='debug',
        help='Tag to use for training'
    )
    train_parser.add_argument(
        '--labels', 
        nargs='+', 
        required=True,
        help='Labels to use for training'
    )
    train_parser.add_argument(
        '--feature-names', 
        type=str, 
        nargs='+', 
        default=['rdkit2d'],
        help='Feature names to use for training'
    )
    train_parser.add_argument(
        '--n-trials', 
        type=int, 
        default=None,
        help='Number of trials to run for hyperparameter optimization.'
    )
    train_parser.add_argument(
        '--out-dir', 
        type=str, 
        default='./results',
        help='Path to the output directory'
    )
    train_parser.add_argument(
        '--hparams-from', 
        type=str, 
        default=None,
        help='Path to the hparams file. Allowed formats: .json, .pt, .pkl'
    )
    train_parser.add_argument(
        '--n-fold', 
        type=int, 
        default=1,
        help='Number of folds to use for cross-validation'
    )
    train_parser.add_argument(
        '--split-mode', 
        type=str, 
        default='random',
        help='Mode to split the data into training, validation, and test sets'
    )
    train_parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Seed to use for training'
    )
    train_parser.add_argument(
        '--remove-hydrogens', 
        action='store_true',
        help='Whether to remove hydrogens from the molecules'
    )
    train_parser.add_argument(
        '--descriptors', 
        type=str, 
        nargs='+', 
        default=None,
        help='Descriptors to use for training. For ML models, this must be specified.'
    )
    train_parser.add_argument(
        '--model', 
        type=str, 
        default='rf',
        help='Model to use for training'
    )
    
    # If not model in `MODELS.keys()`, add more arguments for DL models
    train_parser.add_argument(
        '--hidden-dim', 
        type=int, 
        default=32,
        help='Hidden dimension of the model'
    )
    train_parser.add_argument(
        '--num-layers', 
        type=int, 
        default=3,
        help='Number of layers of the model'
    )
    
    # DL Training arguments
    train_parser.add_argument(
        '--batch-size', 
        type=int, 
        default=128,
        help='Batch size to use for training'
    )
    train_parser.add_argument(
        '--lr', 
        type=float, 
        default=1e-3,
        help='Learning rate to use for training'
    )
    train_parser.add_argument(
        '--num-epochs', 
        type=int, 
        default=2500,
        help='Number of epochs to use for training'
    )
    train_parser.add_argument(
        '--early-stopping-patience', 
        type=int, 
        default=250,
        help='Number of epochs to wait before early stopping'
    )
    train_parser.add_argument(
        '--device', 
        type=str, 
        default='cuda',
        help='Device to use for training'
    )
    train_parser.add_argument(
        '--run-production', 
        action='store_true',
        help=(
            'Whether to run the training in production mode, which means '
            'train:val:test splits will be forced to 0.95:0.05:0.0'
        )
    )
    train_parser.add_argument(
        '--finetune', 
        action='store_true',
        help='Whether to finetune the model'
    )
    train_parser.add_argument(
        '--finetune-csv-path', 
        type=str, 
        default=None,
        help='Path to the csv file to finetune the model on'
    )
    train_parser.add_argument(
        '--pretrained-model', 
        type=str, 
        default=None,
        help='Path to the pretrained model'
    )
    train_parser.add_argument(
        '--n-estimator', 
        type=int, 
        default=1,
        help='Number of estimators to use for training'
    )
    train_parser.add_argument(
        '--additional-features', 
        type=str, 
        nargs='+', 
        default=None,
        help='Additional features to use for training'
    )
    train_parser.add_argument(
        '--skip-train', 
        action='store_true',
        help='Whether to skip the training step'
    )
    train_parser.add_argument(
        '--low-fidelity-model', 
        type=str, 
        default=None,
        help='Path to the low fidelity model'
    )
    train_parser.add_argument(
        '--estimator-name', 
        type=str, 
        default=None,
        help='Name of the estimator to give base predictions'
    )
    train_parser.add_argument(
        '--emb-model', 
        type=str, 
        default=None,
        help='Name of the embedding model for base graph embeddings'
    )
    train_parser.add_argument(
        '--ensemble-type', 
        type=str, 
        default='voting',
        help='Type of ensemble to use for training'
    )
    train_parser.add_argument(
        '--train-residual', 
        action='store_true',
        help='Whether to train the residual of the model'
    )
    train_parser.add_argument(
        '--normalizer-type', 
        type=str, 
        default='normalizer', 
        choices=['normalizer', 'log_normalizer', 'none'],
        help='Type of normalizer to use for training'
    )
    train_parser.add_argument(
        '--augmentation', 
        action='store_true',
        help='Whether to use data augmentation'
    )
    
    # Merge
    merge_parser = subparsers.add_parser('merge', help='Merge two datasets')
    merge_parser.add_argument(
        '--sources', 
        type=str, 
        required=True, 
        nargs='+',
        help='Sources to merge'
    )
    merge_parser.add_argument(
        '--label', 
        type=str, 
        required=True,
        help='Label to merge'
    )
    merge_parser.add_argument(
        '--hparams-from', 
        type=str, 
        required=True,
        help='Path to the hparams file'
    )
    merge_parser.add_argument(
        '--acquisition', 
        type=str, 
        required=True,
        choices=['epig', 'uncertainty', 'difference'],
        help='Acquisition function to use for merging'
    )
    merge_parser.add_argument(
        '--sample-size', 
        type=int, 
        default=20,
        help='Sample size to use for merging'
    )
    merge_parser.add_argument(
        '--uncertainty-threshold', 
        type=float, 
        default=0.1,
        help='Uncertainty threshold to use for merging'
    )
    merge_parser.add_argument(
        '--difference-threshold', 
        type=float, 
        default=0.1,
        help='Difference threshold to use for merging'
    )
    merge_parser.add_argument(
        '--target-size', 
        type=int, 
        default=1000,
        help='Target size to use for merging'
    )
    merge_parser.add_argument(
        '--base-csv', 
        type=str, 
        default=None,
        help='Path to the base csv file'
    )
    
    # Predict
    predict_parser = subparsers.add_parser('predict', help='Predict labels')
    predict_parser.add_argument(
        '--model-path', 
        type=str, 
        required=True,
        help='Path to the model'
    )
    predict_parser.add_argument(
        '--csv-path', 
        type=str, 
        required=True,
        help='Path to the csv file'
    )
    predict_parser.add_argument(
        '--smiles-column', 
        type=str, 
        required=True,
        help='Name of the smiles column'
    )    
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == 'train':
        if args.model in MODELS.keys():
            main_ml(args)
        else:
            main_dl(args)
    elif args.mode == 'merge':
        main_merge(args)
    elif args.mode == 'predict':
        main_predict(args)


if __name__ == '__main__':
    main()