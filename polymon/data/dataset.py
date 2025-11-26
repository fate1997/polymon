import os
import os.path as osp
import random
from typing import Callable, List, Literal, Optional, Tuple, Union

import pandas as pd
import torch
from loguru import logger
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from torch_geometric.data import Batch, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from polymon.data.dedup import Dedup
from polymon.data.featurizer import ComposeFeaturizer
from polymon.data.polymer import OligomerBuilder, Polymer
from polymon.setting import PRETRAINED_MODELS, UNIQUE_ATOM_NUMS


class PolymerDataset(Dataset):
    r"""A dataset that contains the polymers. During the initialization, the 
    SMILES strings will be featurized to the corresponding features and converted
    to the :class:`Polymer` object. If :obj:`pre_transform` is provided, it will
    be used to update the :class:`Polymer` object. If :obj:`estimator` is provided,
    it will be used to estimate the labels of the :class:`Polymer` object, and
    make the prediction task be the residual between the ground truth and the
    estimated labels.
    
    Args:
        raw_csv_path (str): The path to the raw csv file. Typically, it should 
            have at least three columns: :obj:`SMILES`, :obj:`label` (e.g., 
            :obj:`Rg`), and :obj:`Source` (e.g., :obj:`PI1070`).
        feature_names (List[str]): The names of the features to use. The 
            available features are in :obj:`AVAIL_FEATURES` from 
            :mod:`polymon.data.featurizer`.
        label_column (str): Label column name.
        sources (List[str]): The names of the sources.
        smiles_column (str): SMILES column name.
        identifier_column (str): Identifier column name.
        save_processed (bool): Whether to save the processed dataset.
        force_reload (bool): Whether to force reload the processed dataset.
        add_hydrogens (bool): Whether to add hydrogens to the molecules.
        pre_transform (Callable): The pre-transform to apply to the data. It 
            should be a function that takes a :class:`Polymer` object and returns
            a :class:`Polymer` object.
        estimator (Callable): The estimator to apply to the data. It is used to 
            provide the estimated labels for the :class:`Polymer` object. It 
            should be the object of :class:`polymon.estimator.BaseEstimator`.
    """
    
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
        pre_transform: Optional[Callable] = None,
        estimator: Optional[Callable] = None,
        must_keep: List[str] = None,
    ):
        super().__init__()
        
        self.raw_csv_path = raw_csv_path
        self.label_column = label_column
        self.feature_names = feature_names
        self.sources = sources
        self.pre_transform = pre_transform
        self.estimator = estimator
        self.must_keep = must_keep
        
        # Create processed path in the current directory
        processed_name = f'{label_column}_{"_".join(sources)}.pt'
        os.makedirs(os.path.join('.', 'database', 'processed'), exist_ok=True)
        processed_path = os.path.join('.', 'database', 'processed', processed_name)
        
        if osp.exists(processed_path) and not force_reload:
            data = torch.load(processed_path)
            self.data_list = data['data_list']
            self.label_column = data['label_column']
        else:
            if self.pre_transform is not None:
                logger.info(f'Applying pre-transform: {self.pre_transform}')
            
            df_nonan = pd.read_csv(raw_csv_path).dropna(subset=[label_column])
            dedup = Dedup(df_nonan, label_column, must_keep=self.must_keep)
            df_nonan = dedup.run(sources=sources)
            feature_names = list(set(self.feature_names) - set(PRETRAINED_MODELS))
            
            # Initialize the featurizer
            if 'pos' in feature_names:
                add_hydrogens = True
            config = {}
            if 'x' in feature_names:
                config['x'] = {'unique_atom_nums': UNIQUE_ATOM_NUMS}
            featurizer = ComposeFeaturizer(feature_names, config, add_hydrogens)
            self.featurizer = featurizer
        
            data_list = []
            for i in tqdm(range(len(df_nonan)), desc='Featurizing'):
                row = df_nonan.iloc[i]
                smiles = row[smiles_column]
                rdmol = Chem.MolFromSmiles(smiles)
                if 'source' in feature_names:
                    rdmol.SetProp('Source', row['Source'])
                label = row[self.label_column]
                mol_dict = self.featurizer(rdmol)
                mol_dict['y'] = torch.tensor(label).unsqueeze(0).unsqueeze(0).float()
                if identifier_column is not None and identifier_column in df_nonan.columns:
                    mol_dict['identifier'] = torch.tensor(row[identifier_column])
                mol_dict['smiles'] = Chem.MolToSmiles(rdmol)
                if 'Source' in df_nonan.columns:
                    mol_dict['source'] = int(row['Source'] == 'HT-exp')
                if None in mol_dict.values():
                    logger.warning(
                        f'Skipping {smiles} because of None in featurization'
                    )
                    continue
                data = Polymer(**mol_dict)
                
                # Apply pre-transform
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                
                # Apply estimator
                if self.estimator is not None:
                    data = self.estimator(data)
                data_list.append(data)

            # Add pretrained embeddings
            if len(self.feature_names) != len(feature_names):
                logger.info(f'Building pretrained embeddings...')
                from polymon.data._pretrained import (
                    assign_pretrained_embeddings, get_gaff2_features,
                    get_polybert_embeddings, get_polycl_embeddings)
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
                logger.info(f'Saving processed dataset to {processed_path}')
                torch.save({
                    'data_list': self.data_list,
                    'label_column': self.label_column
                }, processed_path)
    
    def get(self, idx: int) -> Polymer:
        """Get the :class:`Polymer` object at the given index.
        """
        data = self.data_list[idx]
        return data
    
    def len(self) -> int:
        """Get the number of :class:`Polymer` objects in the dataset.
        """
        return len(self.data_list)

    def sample_batch(
        self,
        batch_size: int = 64,
    ) -> Batch:
        """Sample a batch of :class:`Polymer` objects.
        """
        loader = DataLoader(self, batch_size, shuffle=True)
        return next(iter(loader))
    
    def get_loaders(
        self,
        batch_size: int,
        n_train: Union[int, float],
        n_val: Union[int, float],
        mode: Literal['random', 'scaffold'] = 'random',
        num_workers: int = 0,
        augmentation: bool = False,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get the data loaders for the training, validation, and test sets.

        Args:
            batch_size (int): The batch size.
            n_train (Union[int, float]): The number of training samples. If it
                is a float, it will be converted to an integer by multiplying
                the length of the dataset.
            n_val (Union[int, float]): The number of validation samples. If it
                is a float, it will be converted to an integer by multiplying
                the length of the dataset.
            mode (Literal['random', 'scaffold']): The split mode. :obj:`random`: 
                Split the dataset randomly. :obj:`scaffold`: Split the dataset 
                by scaffold.
            num_workers (int): The number of workers for the data loaders.
            augmentation (bool): Whether to augment the training set. Currently,
                it only supports the augmentation of oligomers.
        """
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
        
        if mode == 'random':
            train_set, val_set, test_set = self._get_random_splits(n_train, n_val)
        elif mode == 'scaffold':
            train_set, val_set, test_set = self._get_scaffold_splits(n_train, n_val)
        else:
            raise ValueError(f'Invalid split mode: {mode}')

        if augmentation:
            train_set_aug = []
            for data in train_set:
                train_set_aug.append(data)
                for i in range(1):
                    oligomer = OligomerBuilder.get_oligomer(data.smiles, i+2)
                    mol_dict = self.featurizer(oligomer)
                    aug_data = data.clone()
                    for key, value in mol_dict.items():
                        setattr(aug_data, key, value)
                    train_set_aug.append(aug_data)
            logger.info(f'Train set augmented from {len(train_set)} to {len(train_set_aug)}')
        else:
            train_set_aug = train_set
            
        train_loader = DataLoader(
            train_set_aug, batch_size, shuffle=True, num_workers=num_workers
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

    def _get_random_splits(
        self,
        n_train: int,
        n_val: int,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        logger.info('Splitting dataset randomly...')
        dataset = self.shuffle()
        train_set = dataset[:n_train]
        val_set = dataset[n_train:n_train+n_val]
        test_set = dataset[n_train+n_val:]
        return train_set, val_set, test_set
    
    def _get_scaffold_splits(
        self,
        n_train: int,
        n_val: int,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        logger.info('Splitting dataset by scaffold...')
        
        split_1 = n_train
        split_2 = n_train + n_val
        
        train_inds = []
        valid_inds = []
        test_inds = []

        scaffolds = {}

        for idx, data in enumerate(self.data_list):
            smiles = data.smiles
            mol = Chem.MolFromSmiles(smiles)
            scaffold = MurckoScaffoldSmiles(mol=mol)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [idx]
            else:
                scaffolds[scaffold].append(idx)
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)]

        for scaffold_set in scaffold_sets:
            if len(train_inds) + len(scaffold_set) > split_1:
                if len(train_inds) + len(valid_inds) + len(scaffold_set) > split_2:
                    test_inds += scaffold_set
                else:
                    valid_inds += scaffold_set
            else:
                train_inds += scaffold_set
        
        train_set = self[train_inds]
        val_set = self[valid_inds]
        test_set = self[test_inds]
        return train_set, val_set, test_set