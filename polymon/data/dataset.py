import os.path as osp
import os
from typing import List, Tuple, Union

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Batch, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from polymon.data.featurizer import ComposeFeaturizer
from polymon.data.polymer import Polymer
from polymon.setting import TARGETS, UNIQUE_ATOM_NUMS


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
        remove_hydrogens: bool = True,
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
        
            data_list = []
            for i in tqdm(range(len(df_nonan)), desc='Featurizing'):
                row = df_nonan.iloc[i]
                rdmol = Chem.MolFromSmiles(row[smiles_column])
                label = row[self.label_column]
                if remove_hydrogens:
                    rdmol = Chem.RemoveHs(rdmol, sanitize=False)
                if 'pos' in self.feature_names:
                    rdmol = Chem.AddHs(rdmol)
                if 'x' in self.feature_names:
                    config = {'x': {'unique_atom_nums': UNIQUE_ATOM_NUMS}}
                else:
                    config = {}
                mol_dict = ComposeFeaturizer(self.feature_names, config)(rdmol)
                mol_dict['y'] = torch.tensor(label).unsqueeze(0).unsqueeze(0).float()
                mol_dict['identifier'] = torch.tensor(row[identifier_column])
                mol_dict['smiles'] = Chem.MolToSmiles(rdmol)
                data_list.append(Polymer(**mol_dict))
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
            n_train = int(n_train * len(self))
        if isinstance(n_val, float):
            n_val = int(n_val * len(self))
        
        dataset = self.shuffle(self)
        assert n_train + n_val < len(dataset)
        train_loader = DataLoader(
            dataset[:n_train], batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            dataset[n_train:n_train+n_val], batch_size, num_workers=num_workers
        )
        test_loader = DataLoader(
            dataset[n_train+n_val:], batch_size, num_workers=num_workers
        )
        return train_loader, val_loader, test_loader