import os
import os.path as osp
from typing import List, Tuple, Union, Optional, Callable
import random

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Batch, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from loguru import logger

from polymon.data.featurizer import ComposeFeaturizer
from polymon.data.polymer import Polymer
from polymon.data.pretrained import (get_polybert_embeddings,
                                     get_polycl_embeddings,
                                     assign_pretrained_embeddings,
                                     get_gaff2_features)
from polymon.setting import TARGETS, UNIQUE_ATOM_NUMS, PRETRAINED_MODELS
from polymon.data.dedup import Dedup
from polymon.setting import REPO_DIR


class PolymerDataset(Dataset):
    
    def __init__(
        self,
        raw_csv_path: str,
        feature_names: List[str],
        label_column: str,
        sources: List[str],
        smiles_column: str = 'SMILES',
        identifier_column: str = 'id',
        save_processed: bool = True,
        force_reload: bool = False,
        add_hydrogens: bool = False,
        fitting_source: List[str] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__()
        
        self.raw_csv_path = raw_csv_path
        self.label_column = label_column
        self.feature_names = feature_names
        self.sources = sources
        assert self.label_column in TARGETS
        self.pre_transform = pre_transform
        
        processed_name = f'{label_column}_{"_".join(sources)}.pt'
        os.makedirs(str(REPO_DIR / 'database' / 'processed'), exist_ok=True)
        processed_path = str(REPO_DIR / 'database' / 'processed' / processed_name)
        
        if osp.exists(processed_path) and not force_reload:
            data = torch.load(processed_path)
            self.data_list = data['data_list']
            self.label_column = data['label_column']
        else:
            df_nonan = pd.read_csv(raw_csv_path).dropna(subset=[label_column])
            if 'Source' in df_nonan.columns:
                dedup = Dedup(df_nonan, label_column)
                if fitting_source is not None:
                    for source in fitting_source:
                        dedup.compare('internal', source, fitting=True)
                        sources.remove(source)
                        sources.append(f'{source}_fitted')
                df_nonan = dedup.run(sources=sources)
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
                if row[smiles_column].count('*') != 2:
                    logger.warning(f'Skipping {row[smiles_column]} because of not 2 attachments')
                    continue
                rdmol = Chem.MolFromSmiles(row[smiles_column])
                label = row[self.label_column]
                mol_dict = self.featurizer(rdmol)
                mol_dict['y'] = torch.tensor(label).unsqueeze(0).unsqueeze(0).float()
                if identifier_column is not None and identifier_column in df_nonan.columns:
                    mol_dict['identifier'] = torch.tensor(row[identifier_column])
                mol_dict['smiles'] = Chem.MolToSmiles(rdmol)
                if 'Source' in df_nonan.columns:
                    mol_dict['source'] = row['Source']
                if None in mol_dict.values():
                    logger.warning(f'Skipping {row[smiles_column]} because of None in featurization')
                    continue
                data = Polymer(**mol_dict)
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

            # Add pretrained embeddings
            if len(self.feature_names) != len(feature_names):
                logger.info(f'Building pretrained embeddings...')
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
        data = self.data_list[idx]
        if getattr(data, 'bridge_index', None) is not None:
            import random
            direction = random.randint(0, 1)
            if direction == 0:
                data.bridge_index = data.bridge_index[[1, 0], :]
        return data
    
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
        production_run: bool = False,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        
        if isinstance(n_train, float):
            train_ratio = n_train
            n_train = int(train_ratio * len(self))
        if isinstance(n_val, float):
            val_ratio = n_val
            n_val = int(val_ratio * len(self))
            if abs(train_ratio + val_ratio - 1.0) < 1e-4:
                n_train = len(self) - n_val
        n_test = len(self) - n_train - n_val
        assert n_test >= 0
        logger.info(f'Split: {n_train} train, {n_val} val, {n_test} test')
        
        if getattr(self.data_list[0], 'source', None) is not None:
            logger.info('Splitting dataset by source...')
            internal = [data for data in self.data_list if data.source == 'internal']
            external = [data for data in self.data_list if data.source != 'internal']
            random.shuffle(internal)
            random.shuffle(external)
            if production_run:
                test_set = external[:n_test]
                val_set = internal[:n_val]
                train_set = internal[n_val:] + external[n_test:]
            else:
                test_set = internal[:n_test]
                val_set = internal[n_test:n_test+n_val]
                train_set = internal[n_test+n_val:] + external[n_val:]
                if len(val_set) == 0:
                    val_set = external[:n_val]
                    train_set = internal[n_test:] + external[n_val:]
                    logger.warning('No val set, using external as val')
        if getattr(self.data_list[0], 'source', None) is None or len(external) == 0:
            logger.info('Splitting dataset randomly...')
            dataset = self.shuffle()
            train_set = dataset[:n_train]
            val_set = dataset[n_train:n_train+n_val]
            test_set = dataset[n_train+n_val:]
        
        train_loader = DataLoader(
            train_set, batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_set, batch_size, num_workers=num_workers
        )
        if len(test_set) > 0:
            test_loader = DataLoader(
                test_set, batch_size, num_workers=num_workers
            )
        else:
            test_loader = None
        return train_loader, val_loader, test_loader