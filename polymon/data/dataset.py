import os
import os.path as osp
from typing import List, Tuple, Union

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Batch, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from polymon.data.featurizer import ComposeFeaturizer
from polymon.data.polymer import Polymer
from polymon.data.pretrained import (get_polybert_embeddings,
                                     get_polycl_embeddings,
                                     assign_pretrained_embeddings,
                                     get_gaff2_features)
from polymon.setting import TARGETS, UNIQUE_ATOM_NUMS, PRETRAINED_MODELS


class PolymerDataset(Dataset):
    
    def __init__(
        self,
        raw_csv_path: str,
        feature_names: List[str],
        label_column: str,
        smiles_column: str = 'SMILES',
        identifier_column: str = 'id',
        save_processed: bool = True,
        force_reload: bool = False,
        add_hydrogens: bool = False,
    ):
        super().__init__()
        
        self.raw_csv_path = raw_csv_path
        self.label_column = label_column
        self.feature_names = feature_names
        assert self.label_column in TARGETS
        
        processed_path = raw_csv_path.replace('.csv', f'_{label_column}.pt')
        
        if osp.exists(processed_path) and not force_reload:
            data = torch.load(processed_path)
            self.data_list = data['data_list']
            self.label_column = data['label_column']
        else:
            df = pd.read_csv(raw_csv_path)
            df_nonan = df.dropna(subset=[label_column])
            feature_names = list(set(self.feature_names) - set(PRETRAINED_MODELS))
            
            if 'pos' in feature_names:
                add_hydrogens = True
            if 'x' in feature_names:
                config = {'x': {'unique_atom_nums': UNIQUE_ATOM_NUMS}}
            else:
                config = {}
            featurizer = ComposeFeaturizer(feature_names, config, add_hydrogens)
            self.featurizer = featurizer
        
            data_list = []
            for i in tqdm(range(len(df_nonan)), desc='Featurizing'):
                row = df_nonan.iloc[i]
                rdmol = Chem.MolFromSmiles(row[smiles_column])
                label = row[self.label_column]
                mol_dict = self.featurizer(rdmol)
                mol_dict['y'] = torch.tensor(label).unsqueeze(0).unsqueeze(0).float()
                if identifier_column is not None and identifier_column in df_nonan.columns:
                    mol_dict['identifier'] = torch.tensor(row[identifier_column])
                mol_dict['smiles'] = Chem.MolToSmiles(rdmol)
                data_list.append(Polymer(**mol_dict))

            # Add pretrained embeddings
            if len(self.feature_names) != len(feature_names):
                print(f'Building pretrained embeddings...')
            if 'polycl' in self.feature_names:
                pretrained_embeddings = get_polycl_embeddings(
                    df_nonan[smiles_column].tolist(),
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                )
                assign_pretrained_embeddings(data_list, pretrained_embeddings)
            if 'polybert' in self.feature_names:
                pretrained_embeddings = get_polybert_embeddings(
                    df_nonan[smiles_column].tolist(),
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                )
                assign_pretrained_embeddings(data_list, pretrained_embeddings)

            if 'gaff2_mod' in self.feature_names:
                gaff2_mod_desc = get_gaff2_features(
                    df_nonan[smiles_column].tolist(),
                )
                assign_pretrained_embeddings(data_list, gaff2_mod_desc)

            self.data_list = data_list
            
            if save_processed:
                os.makedirs(osp.dirname(processed_path), exist_ok=True)
                torch.save({
                    'data_list': self.data_list,
                    'label_column': self.label_column
                }, processed_path)
    
    def get(self, idx: int) -> Polymer:
        return self.data_list[idx]
    
    def len(self) -> int:
        return len(self.data_list)

    def sample_batch(
        self,
        batch_size: int = 64,
    ) -> Batch:
        loader = DataLoader(self, batch_size, shuffle=True)
        return next(iter(loader))
    
    def get_loaders(
        self,
        batch_size: int,
        n_train: Union[int, float],
        n_val: Union[int, float],
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        
        if isinstance(n_train, float):
            train_ratio = n_train
            n_train = int(train_ratio * len(self))
        if isinstance(n_val, float):
            val_ratio = n_val
            n_val = int(val_ratio * len(self))
            if abs(train_ratio + val_ratio - 1.0) < 1e-4:
                n_train = len(self) - n_val
        
        dataset = self.shuffle(self)
        assert n_train + n_val <= len(dataset)
        train_loader = DataLoader(
            dataset[:n_train], batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            dataset[n_train:n_train+n_val], batch_size, num_workers=num_workers
        )
        if n_train + n_val < len(dataset):
            test_loader = DataLoader(
                dataset[n_train+n_val:], batch_size, num_workers=num_workers
            )
        else:
            test_loader = None
        return train_loader, val_loader, test_loader