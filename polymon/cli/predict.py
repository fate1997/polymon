"""Prediction module for PolyMon.

This module provides functionality for making predictions on new polymer data
using trained models. Supports both single models and ensemble models.
"""

import argparse

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from polymon.data.featurizer import ComposeFeaturizer


def main(args: argparse.Namespace):
    """Make predictions on a CSV file using a trained model.

    This function loads a trained model, featurizes molecules from the input CSV,
    and adds predictions as a new column. The output is saved as a new CSV file
    with '_predicted' appended to the original filename.

    Args:
        args: Command-line arguments with the following attributes:
            - model_path (str): Path to the trained model file (.pt or .pkl)
            - csv_path (str): Path to input CSV file with molecules
            - smiles_column (str): Name of column containing SMILES strings

    Example:
        $ polymon predict \
            --model-path ./results/gatv2/Tg/model.pt \
            --csv-path ./new_data.csv \
            --smiles-column SMILES

    Note:
        - For .pt files, the model is loaded as a PyTorch model
        - For .pkl files, the model is loaded using pickle
        - If the model has feature_names stored, those are used; otherwise rdkit2d is default
    """
    # Load the trained model
    if args.model_path.endswith('.pkl'):
        import pickle
        with open(args.model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = torch.load(args.model_path)
        model.eval()

    # Read input data
    df = pd.read_csv(args.csv_path)
    smiles_list = df[args.smiles_column].tolist()

    # Get feature names from model or use default
    if hasattr(model, 'feature_names'):
        featurizer = ComposeFeaturizer(model.feature_names)
    else:
        featurizer = ComposeFeaturizer(['rdkit2d'])

    # Featurize molecules
    features = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            feat_dict = featurizer(mol)
            features.append(feat_dict['descriptors'])
        else:
            # Handle invalid SMILES by using NaN
            features.append(np.full((1, len(featurizer.featurizers[0].feature_names)), np.nan))

    features = np.array(features)

    # Make predictions
    if isinstance(model, torch.nn.Module):
        features = torch.from_numpy(features).float()
        with torch.no_grad():
            predictions = model(features).numpy()
    else:
        predictions = model.predict(features)

    # Save results
    df['predictions'] = predictions
    output_path = args.csv_path.replace('.csv', '_predicted.csv')
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
