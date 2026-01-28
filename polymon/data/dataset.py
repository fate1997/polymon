import os
import os.path as osp
import random
from typing import Callable, List, Literal, Optional, Tuple, Union, Dict, Any

import numpy as np
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
from polymon.setting import PRETRAINED_MODELS, UNIQUE_ATOM_NUMS, MORDRED_VOCAB, MORDRED_DIMER_VOCAB, MORDRED_3D_VOCAB

# from polymon.model.esa.esa_utils import get_max_node_edge_global

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
        label_column: Union[str, List[str]],
        sources: List[str],
        smiles_column: str = 'SMILES',
        identifier_column: str = 'id',
        save_processed: bool = True,
        force_reload: bool = False,
        add_hydrogens: bool = False,
        pre_transform: Optional[Callable] = None,
        estimator: Optional[Callable] = None,
        must_keep: List[str] = None,
        mt_train: bool = False,
    ):
        super().__init__()
        
        self.raw_csv_path = raw_csv_path
        self.label_column = label_column
        self.feature_names = feature_names
        self.sources = sources
        self.pre_transform = pre_transform
        self.estimator = estimator
        self.must_keep = must_keep
        self.mt_train = mt_train
        # Create processed path in the current directory
        if isinstance(label_column, str):
            processed_name = f'{label_column}_{"_".join(sources)}.pt'
        #else:
        if self.mt_train:
            processed_name = f'mt_{"_".join(sources)}.pt'
        os.makedirs(os.path.join('.', 'database', 'processed'), exist_ok=True)
        processed_path = os.path.join('.', 'database', 'processed', processed_name)
        
        if osp.exists(processed_path) and not force_reload:
            data = torch.load(processed_path)
            self.data_list = data['data_list']
            self.label_column = data['label_column']
        else:
            if self.pre_transform is not None:
                logger.info(f'Applying pre-transform: {self.pre_transform}')
            
            # Allow label_column to be either a string or a list of strings
            label_cols = [label_column] if isinstance(label_column, str) else list(label_column)
            if not self.mt_train:
                df_nonan = pd.read_csv(raw_csv_path).dropna(subset=label_cols)
            else:
                df_nonan = pd.read_csv(raw_csv_path)
            dedup = Dedup(df_nonan, label_column, must_keep=self.must_keep, mt_train=self.mt_train)
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

            cache_path = None
            if 'mordred' in feature_names:
                cache_path = MORDRED_VOCAB 
                cache: Dict[str, Any]
                if cache_path.exists():
                    logger.info(f'Loading mordred cache from {cache_path}')
                    cache = torch.load(
                        cache_path, 
                        map_location='cpu', 
                    )
                else:
                    cache = {}
            elif 'oligomer_mordred' in feature_names:
                cache_path = MORDRED_DIMER_VOCAB
                cache: Dict[str, Any]
                if cache_path.exists():
                    logger.info(f'Loading mordred cache from {cache_path}')
                    cache = torch.load(
                        cache_path, 
                        map_location='cpu', 
                    )
                else:
                    cache = {}
            elif 'mordred3d' in feature_names:
                cache_path = MORDRED_3D_VOCAB
                cache: Dict[str, Any]
                if cache_path.exists():
                    logger.info(f'Loading mordred cache from {cache_path}')
                    cache = torch.load(
                        cache_path, 
                        map_location='cpu', 
                    )
                else:
                    cache = {}
            
            def _save_cache(cache: Dict[str, Any], path: str):
                import os.path as osp
                import tempfile
                d = osp.dirname(path)
                os.makedirs(d, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir = d, delete=False) as tmp:
                    tmp_name = tmp.name
                    torch.save(cache, tmp_name)
                os.replace(tmp_name, path)

            
            data_list = []
            for i in tqdm(range(len(df_nonan)), desc='Featurizing'):
                row = df_nonan.iloc[i]
                smiles = row[smiles_column]
                rdmol = Chem.MolFromSmiles(smiles)
                if 'source' in feature_names:
                    rdmol.SetProp('Source', row['Source'])
                
                # Support for label_column as str or list of str
                if isinstance(self.label_column, (list, tuple)):
                    label = row[self.label_column].values.astype(float)
                else:
                    label = row[self.label_column]
                
                if 'mordred' in feature_names or 'oligomer_mordred' in feature_names or 'mordred3d' in feature_names:
                    if smiles in cache:
                        mol_dict = {'descriptors': cache[smiles]}
                    else:
                        mol_dict = self.featurizer(rdmol)
                        cache[smiles] = mol_dict['descriptors']
                        _save_cache(cache, cache_path)
                else:
                    mol_dict = self.featurizer(rdmol)
                
                # Set 'y' as tensor with correct dimensions for single- or multi-label task
                y_tensor = torch.tensor(label, dtype=torch.float)
                if y_tensor.dim() == 0:
                    # scalar, make shape (1,1)
                    y_tensor = y_tensor.unsqueeze(0).unsqueeze(0)
                elif y_tensor.dim() == 1:
                    # vector, make shape (1, num_labels)
                    y_tensor = y_tensor.unsqueeze(0)
                mol_dict['y'] = y_tensor
                if identifier_column is not None and identifier_column in df_nonan.columns:
                    mol_dict['identifier'] = torch.tensor(row[identifier_column])
                mol_dict['smiles'] = Chem.MolToSmiles(rdmol)
                if 'Source' in df_nonan.columns:
                    mol_dict['source'] = int(row['Source'] == 'internal')
                    mol_dict['source_name'] = row['Source']
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
            # for ESA data attribute
            # self.max_edge_global, self.max_node_global = get_max_node_edge_global(self.data_list)
            # for g in self.data_list:
            #     g.max_node_global = self.max_node_global
            #     g.max_edge_global = self.max_edge_global
            if 'mordred' in feature_names or 'oligomer_mordred' in feature_names or 'mordred3d' in feature_names:
                _save_cache(cache, cache_path)
                
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
        mode: Literal['random', 'scaffold', 'similarity'] = 'random',
        num_workers: int = 0,
        augmentation: bool = False,
        must_keep: str = "initial",
    ) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, np.ndarray]]:
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
        elif mode == 'similarity':
            train_set, val_set, test_set, indices = self._get_similarity_splits(n_train, n_val, must_keep)
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
        if mode == 'similarity':
            return train_loader, val_loader, test_loader, indices
        return train_loader, val_loader, test_loader

    def _get_random_splits(
        self,
        n_train: int,
        n_val: int,
        must_keep: str = "initial",
    ) -> Tuple[Dataset, Dataset, Dataset]:
    
        import random
        logger.info('Splitting dataset randomly...')
        dataset = self.shuffle()
        train_set = dataset[:n_train]
        val_set = dataset[n_train:n_train+n_val]
        n_test = len(dataset) - n_train - n_val
        rng = random.Random(42)
        all_indices = list(range(len(dataset)))
        rng.shuffle(all_indices)
        must_idxs = [i for i in all_indices if getattr(dataset[i], "source_name", None) == must_keep]
        test_idxs = set(rng.sample(must_idxs, n_test))
        remaining = [i for i in all_indices if i not in test_idxs]
        train_idxs = remaining[:n_train]
        val_idxs = remaining[n_train:n_train+n_val]
        
        train_set = [dataset[i] for i in train_idxs]
        val_set = [dataset[i] for i in val_idxs]
        test_set = [dataset[i] for i in test_idxs]
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
    
    def _get_similarity_splits(
        self,
        n_train: int,
        n_val: int,
        must_keep: str = "initial",
    ) -> Tuple[Dataset, Dataset, Dataset]:
        logger.info('Splitting dataset by similarity...')
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.DataStructs import BulkTanimotoSimilarity, ConvertToNumpyArray
        fps, valid_indices = [], []
        for idx, data in enumerate(self.data_list):
            smiles = data.smiles
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mfgen = rdFingerprintGenerator.GetMorganGenerator(4, fpSize=2048)
                fps.append(mfgen.GetFingerprint(mol))
                valid_indices.append(idx)

        n = len(valid_indices)
        if n_train + n_val > n:
            raise ValueError(f'Sum of n_train and n_val exceeds dataset size: {n_train + n_val} > {n}')
        
        scores = np.array([sum(BulkTanimotoSimilarity(fp, fps)) for fp in tqdm(fps, desc='Calculating similarity scores')])
        order = np.argsort(scores)  # ascending => most dissimilar first

        test_count = n - n_train - n_val
        def is_initial(local_idx: int) -> bool:
            # local_idx is an index into fps/valid_indices
            orig_idx = valid_indices[local_idx]
            return getattr(self.data_list[orig_idx], "source_name", None) == must_keep

        order_initial = [i for i in order if is_initial(i)]

        if len(order_initial) < test_count:
            have = len(order_initial)
            raise ValueError(
                f'Not enough samples from source_name="initial" to form test set of size {test_count}: only {have} available.'
            )

        # Take the most dissimilar 'initial' samples for the test set
        test_local = np.array(order_initial[:test_count], dtype=int)

        # Remaining local indices (for train/val) are all others
        remaining = np.array([i for i in order if i not in set(test_local)], dtype=int)

        # Random split remaining into train/val
        perm = np.random.permutation(remaining)
        if len(perm) < (n_train + n_val):
            # This should not happen if counts sum correctly, but guard anyway
            raise RuntimeError(
                f"Internal error: remaining({len(perm)}) < n_train + n_val ({n_train + n_val})."
            )
        train_local = perm[:n_train]
        val_local = perm[n_train:n_train + n_val]

        # Map back to original indices
        test_inds  = [valid_indices[i] for i in test_local.tolist()]
        train_inds = [valid_indices[i] for i in train_local.tolist()]
        val_inds   = [valid_indices[i] for i in val_local.tolist()]

        # Slice datasets
        train_set = self[train_inds]
        val_set   = self[val_inds]
        test_set  = self[test_inds]

        return train_set, val_set, test_set, {
            'train': np.asarray(train_inds, dtype=int),
            'val': np.asarray(val_inds, dtype=int),
            'test': np.asarray(test_inds, dtype=int),
        }

        
        