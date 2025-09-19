import argparse
import pickle

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from polymon.data.featurizer import ComposeFeaturizer


def main(args: argparse.Namespace):
    if args.model_path.endswith('.pkl'):
        with open(args.model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = torch.load(args.model_path)
        model.eval()

    df = pd.read_csv(args.csv_path)
    smiles_list = df[args.smiles_column].tolist()

    if hasattr(model, 'feature_names'):
        featurizer = ComposeFeaturizer(model.feature_names)
    else:
        featurizer = ComposeFeaturizer(['rdkit2d'])
        
    features = [featurizer(Chem.MolFromSmiles(smiles))['descriptors'] for smiles in smiles_list]
    features = np.array(features)

    if isinstance(model, torch.nn.Module):
        features = torch.from_numpy(features).float()
        with torch.no_grad():
            predictions = model(features).numpy()
    else:
        predictions = model.predict(features)

    df['predictions'] = predictions
    df.to_csv(args.csv_path.replace('.csv', '_predicted.csv'), index=False)
