import os
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from rdkit import Chem
from sklearn.linear_model import LinearRegression
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torchensemble._base import BaseRegressor
from tqdm import tqdm

from polymon.data.featurizer import ComposeFeaturizer
from polymon.data.polymer import Polymer
from polymon.data.utils import LogNormalizer, Normalizer
from polymon.exp.score import scaling_error
from polymon.model.base import ModelWrapper

if TYPE_CHECKING:
    from polymon.estimator.base import BaseEstimator

PREDICT_BATCH_SIZE = 128

class LinearEnsembleRegressor(nn.Module):
    """Linear Ensemble Regressor. The output is the weighted sum of the predictions 
    of the base models.

    Args:
        base_builders (Dict[str, Callable[[], object]]): The base builders.
        meta_weigths (torch.Tensor): The meta weights.
        meta_bias (torch.Tensor): The meta bias.
        random_state (int): The random state.
    """
    def __init__(
            self,
            base_builders: Dict[str, Callable[[], object]],
            meta_weigths: torch.Tensor,
            meta_bias: torch.Tensor,
            random_state: int = 42,
            strategy: str = 'stacking',
    ):

        super().__init__()
        assert strategy in ['stacking', 'averaging']
        self.base_builders = base_builders
        self.meta_weigths = meta_weigths
        self.meta_bias = meta_bias
        self.random_state = random_state
        self.strategy = strategy

        self._feature_cache: Dict[Tuple[str, ...], Dict[str, np.ndarray]] = {}

    def fit(
            self, 
            smiles_list: List[str], 
            y: np.ndarray, 
            device: str = 'cpu',
    ):
        """Fit the model from a list of SMILES strings and target values.

        Args:
            smiles_list (List[str]): The list of SMILES strings.
            y (np.ndarray): The target values.
            device (str): The device to use.
        """
        base_preds = self.base_predict(
            smiles_list, 
            batch_size=PREDICT_BATCH_SIZE, 
            device=device,
        )

        n_models = len(self.base_builders)
        if self.strategy == 'averaging': 
            weights = np.full(n_models, 1.0/n_models, dtype=np.float32)
            bias = np.array(0.0, dtype=np.float32)
            self.meta_weigths = torch.from_numpy(weights)
            self.meta_bias = torch.from_numpy(bias)
            return self.meta_weigths, self.meta_bias

        meta_builder = LinearRegression(fit_intercept=True)
        meta_builder.fit(base_preds, y)
        corf = meta_builder.coef_.astype(np.float32).reshape(-1)
        intercept = meta_builder.intercept_.astype(np.float32).reshape(())
        self.meta_weigths = torch.from_numpy(corf)
        self.meta_bias = torch.from_numpy(intercept)
        return self.meta_weigths, self.meta_bias
    
    def predict(
        self,
        smiles_list: List[str],
        batch_size: int = 128,
        device: str = 'cpu',
    ) -> torch.Tensor:
        """Predict the output of the model for a list of polymer SMILES strings.

        Args:
            smiles_list (List[str]): The list of SMILES strings.
            batch_size (int): The batch size.
            device (str): The device to use.
        """
        base_preds = self._base_predict(smiles_list, batch_size, device)
        base_preds = torch.from_numpy(base_preds).to(device)
        meta_preds = self.meta_bias + base_preds @ self.meta_weigths
        meta_preds = meta_preds.detach().numpy()
        return meta_preds, base_preds.detach().numpy()
    
    def _base_predict(
        self,
        smiles_list: List[str],
        batch_size: int = 128,
        device: str = 'cpu',
    ) -> np.ndarray:
        base_preds = []
        for name, model in self.base_builders.items():
            if name.endswith('DL'):
                y_preds = model.predict(
                    smiles_list, 
                    batch_size=batch_size, 
                    device=device,
                )
            elif name.endswith('ML'):
                y_preds = self._ml_predict(model, smiles_list)
            else:
                raise ValueError(f'Unknown builder: {name}')
            base_preds.append(y_preds)
        base_preds = np.column_stack(base_preds)
        
        return base_preds
            
    def _ml_predict(self, model: Any, smiles_list: List[str]) -> torch.Tensor:
        feature_names = model.feature_names
        feature_key = tuple(feature_names)
        featurizer = ComposeFeaturizer(feature_names)
        feats_cache = self._feature_cache.setdefault(feature_key, {})
        feats_permol = []
        for smiles in tqdm(smiles_list, desc = f'Featurizing for {feature_names}'):
            X = feats_cache.get(smiles)
            if X is None:
                mol = Chem.MolFromSmiles(smiles)
                X = featurizer(mol)['descriptors']
                X = np.asarray(X).reshape(-1)
                feats_cache[smiles] = X
            feats_permol.append(X)
        
        # X = np.array(feats_permol).squeeze(1)
        # X = np.array([
        #     featurizer(Chem.MolFromSmiles(smiles)) for smiles in tqdm(smiles_list)
        # ])
        # X = np.array([X[i]['descriptors'] for i in range(len(X))]).squeeze(1)
        X = np.vstack(feats_permol)
        feature_name_str = '_'.join(feature_names)
        y_pred = model.predict(X)
        return torch.from_numpy(y_pred)
    
    def write(self, path: str) -> str:
        """Write the model to a file.

        Args:
            path (str): The path to the file.

        Returns:
            str: The path to the file.
        """
        torch.save(self.info, path)
        return os.path.abspath(path)
    
    @classmethod
    def from_file(
        cls,
        path: str,
        map_location: str = 'cpu',
        weights_only: bool = False,
    ) -> 'LinearEnsembleRegressor':
        """Build a LinearEnsembleRegressor from a file.

        Args:
            path (str): The path to the file.
            map_location (str): The map location.
            weights_only (bool): Whether to load only the weights.
        """
        info = torch.load(path, map_location=map_location, weights_only=weights_only)
        base_builders = {}
        for name, builder in info['base_builders'].items():
            if name.endswith('DL'):
                base_builders[name] = ModelWrapper.from_dict(builder)
            elif name.endswith('ML'):
                base_builders[name] = builder
            else:
                raise ValueError(f'Unknown builder: {name}')
        meta_weigths = info['meta_weigths']
        meta_bias = info['meta_bias']
        return cls(
            base_builders=base_builders,
            meta_weigths=meta_weigths,
            meta_bias=meta_bias,
            random_state=info['random_state'],
        )
    
    @property
    def info(self) -> Dict[str, Any]:
        """Get the information of the model.

        Returns:
            Dict[str, Any]: The information of the model.
        """
        output = {
            'meta_weigths': self.meta_weigths,
            'meta_bias': self.meta_bias,
            'random_state': self.random_state,
        }
        output['base_builders'] = {}
        for name, builder in self.base_builders.items():
            if name.endswith('DL'):
                output['base_builders'][name] = builder.info
            elif name.endswith('ML'):
                output['base_builders'][name] = builder
            else:
                raise ValueError(f'Unknown builder: {name}')

        return output


class EnsembleModelWrapper(nn.Module):
    """Ensemble Model Wrapper. This wrapper is used to wrap the ensemble model, 
    normalizer, featurizer, transform, and estimator, and make it easier to do 
    inference.

    Args:
        model (BaseRegressor): The ensemble model. It is from :mod:`torchensemble`.
        normalizer (Normalizer): The normalizer.
        featurizer (ComposeFeaturizer): The featurizer.
        transform_cls (str): The class of the transform.
        transform_kwargs (Dict[str, Any]): The initial parameters of the transform.
        estimator (BaseEstimator): The estimator.
    """
    def __init__(
        self, 
        model: BaseRegressor,
        normalizer: 'Normalizer',
        featurizer: ComposeFeaturizer,
        transform_cls: str = None,
        transform_kwargs: Dict[str, Any] = None,
        estimator: 'BaseEstimator' = None,
    ):
        super().__init__()
        self.model = model
        self.normalizer = normalizer
        self.featurizer = featurizer
        self.transform_cls = transform_cls
        self.transform_kwargs = transform_kwargs
        if transform_cls is not None:
            transform_cls = getattr(import_module('polymon.data.utils'), transform_cls)
            self.transform = transform_cls(**transform_kwargs)
        else:
            self.transform = None
        self.estimator = estimator

    def fit(
        self, 
        epochs: int,
        train_loader: DataLoader,
        save_dir: str,
        save_model: bool,
        log_interval: int,
        label: str,
        val_loader: DataLoader = None,
        test_loader: DataLoader = None,
    ) -> torch.Tensor:
        """Fit the ensemble model.

        Args:
            epochs (int): The number of epochs.
            train_loader (DataLoader): The training loader.
            save_dir (str): The directory to save the model.
            save_model (bool): Whether to save the model.
            log_interval (int): The interval to log the training progress.
            label (str): The label of the model.
            val_loader (DataLoader): The validation loader.
            test_loader (DataLoader): The test loader.
        """
        if self.model.__class__.__name__ != 'BaggingRegressor':
            train_loader = self._loader_wrapper(train_loader)
            test_loader = self._loader_wrapper(val_loader) if val_loader is not None else None
        else:
            # train_loader_ = []
            # for data in train_loader.dataset:
            #     data_copy = data.clone()
            #     data_copy.y = self.normalizer(data_copy.y)
            #     train_loader_.append(data_copy)
            #     test_loader = None
            # train_loader = DataLoader(train_loader_, batch_size=128, shuffle=True)
            wrapped_dataset = TorchEnsembleDataset(
                train_loader.dataset,
                normalizer=self.normalizer
            )

            train_loader = DataLoader(
                wrapped_dataset,
                batch_size=128,
                shuffle=True
            )
            test_loader = None
        
        self.model.fit(
            train_loader=train_loader,
            epochs=epochs,
            test_loader=test_loader,
            save_dir=save_dir,
            save_model=save_model,
            log_interval=log_interval,
        )
        if test_loader is not None:
            test_err = self.evaluate(test_loader, label, device='cuda')
            return test_err
        return None

    def evaluate(
        self, 
        loader: DataLoader,
        label: str,
        device: str = 'cuda',
    ) -> float:
        """Evaluate the model.

        Args:
            loader (DataLoader): The loader.
            label (str): The label of the model.
            device (str): The device to use.
        """
        self.model.eval()
        self.model.to(device)
        y_trues = []
        y_preds = []
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            y_pred = self.model(batch)
            y_pred = self.normalizer.inverse(y_pred)
            y_pred = y_pred + getattr(batch, 'estimated_y', 0)
            y_true = batch.y.detach() + getattr(batch, 'estimated_y', 0)
            y_trues.extend(y_true.cpu().numpy())
            y_preds.extend(y_pred.detach().cpu().numpy())
        y_trues = np.array(y_trues)
        y_preds = np.array(y_preds)
        return scaling_error(y_trues, y_preds, label)

    @torch.no_grad()
    def predict(
        self, 
        smiles_list: List[str],
        batch_size: int = 128,
        device: str = 'cpu',
    ) -> torch.Tensor:
        """Predict the output of the model for a list of polymer SMILES strings.

        Args:
            smiles_list (List[str]): The list of SMILES strings.
            batch_size (int): The batch size.
            device (str): The device to use.
        """
        self.model.eval()
        self.model.to(device)
        
        polymers = []
        for i, smiles in enumerate(smiles_list):
            rdmol = Chem.MolFromSmiles(smiles)
            mol_dict = self.featurizer(rdmol)
            mol_dict['smiles'] = smiles
            polymer = Polymer(**mol_dict)
            if self.transform is not None:
                polymer = self.transform(polymer)
            if self.estimator is not None:
                polymer = self.estimator.forward(polymer, device=device)
            polymers.append(polymer)
        loader = DataLoader(polymers, batch_size=batch_size)
        
        y_pred_list = []
        for i, batch in enumerate(loader):
            batch.to(device)
            output = self.model(batch)
            output = self.normalizer.inverse(output)
            output = output.squeeze(0).squeeze(0)
            if hasattr(batch, 'estimated_y'):
                output = output + batch.estimated_y.squeeze(0).squeeze(0)
            y_pred_list.append(output)
        if len(y_pred_list) == 1:
            return y_pred_list[0].detach().cpu()
        if batch_size == 1:
            return torch.stack(y_pred_list, dim=0).detach().cpu().unsqueeze(-1)
        return torch.cat(y_pred_list, dim=0).detach().cpu()
    
    @property
    def info(self) -> Dict[str, Any]:
        """Get the information of the model.

        Returns:
            Dict[str, Any]: The information of the model.
        """
        output = {
            'model_cls': self.model.__class__.__name__,
            'model': self.model.state_dict(),
            'normalizer': self.normalizer.init_params,
            'model_init_params': self.model.estimator_args,
            'estimator_cls': self.model.base_estimator_.__name__,
            'n_estimators': self.model.n_estimators,
            'featurizer_names': self.featurizer.names,
            'featurizer_config': self.featurizer.config,
            'featurizer_add_hydrogens': self.featurizer.add_hydrogens,
            'transform_cls': self.transform_cls,
            'transform_kwargs': self.transform_kwargs,
        }
        if self.estimator is not None:
            output['base_estimator_cls'] = self.estimator.__class__.__name__
            output['base_estimator_init_params'] = self.estimator.init_params
        return output
    
    def write(self, path: str) -> str:
        """Write the model to a file.

        Args:
            path (str): The path to the file.

        Returns:
            str: The path to the file.
        """
        torch.save(self.info, path)
        return os.path.abspath(path)
    
    @classmethod
    def from_dict(
        cls, 
        model_info: Dict[str, Any], 
        device: str = 'cpu',
    ) -> 'EnsembleModelWrapper':
        """Build an EnsembleModelWrapper from a dictionary.

        Args:
            model_info (Dict[str, Any]): The information of the model.
            device (str): The device to use.

        Returns:
            'EnsembleModelWrapper': The EnsembleModelWrapper.
        """
        model_cls = model_info['model_cls']
        model_cls = getattr(import_module('torchensemble'), model_cls)
        model = model_cls(
            estimator=getattr(import_module('polymon.model'), model_info['estimator_cls']),
            estimator_args=model_info['model_init_params'],
            n_estimators=model_info['n_estimators'],
        )
        model.device = device
        for _ in range(model_info['n_estimators']):
            model.estimators_.append(model._make_estimator())
        
        criterion_info = model_info.get('criterion', None)
        if criterion_info is not None:
            from polymon.exp.train import HeteroGaussianNLLCriterion
            criterion = HeteroGaussianNLLCriterion()
            model.set_criterion(criterion)

        model.load_state_dict(model_info['model'])
        if 'mean' in model_info['normalizer']:
            normalizer = Normalizer(
                mean=model_info['normalizer']['mean'],
                std=model_info['normalizer']['std'],
            )
        else:
            normalizer = LogNormalizer(eps=model_info['normalizer']['eps'])
        featurizer = ComposeFeaturizer(
            names=model_info['featurizer_names'],
            config=model_info['featurizer_config'],
            add_hydrogens=model_info['featurizer_add_hydrogens'],
        )
        transform_cls = model_info.get('transform_cls', None)
        transform_kwargs = model_info.get('transform_kwargs', None)
        
        base_estimator_cls = model_info.get('base_estimator_cls', None)
        base_estimator_kwargs = model_info.get('base_estimator_init_params', None)
        if base_estimator_cls is not None:
            base_estimator_cls = getattr(import_module('polymon.estimator'), base_estimator_cls)
            base_estimator = base_estimator_cls(**base_estimator_kwargs)
        else:
            base_estimator = None
        return cls(model, normalizer, featurizer, transform_cls, transform_kwargs, base_estimator)
    
    @classmethod
    def from_file(
        cls, 
        path: str,
        map_location: str = 'cpu',
        weights_only: bool = False,
    ) -> 'ModelWrapper':
        """Build an EnsembleModelWrapper from a file.

        Args:
            path (str): The path to the file.
            map_location (str): The map location.
            weights_only (bool): Whether to load only the weights.
        """
        output = torch.load(path, map_location=map_location, weights_only=weights_only)
        return cls.from_dict(output, device=map_location)
    
    def _loader_wrapper(self, loader: DataLoader) -> List[Tuple[Polymer, torch.Tensor]]:
        data = []
        for batch in loader:
            data.append((batch, self.normalizer(batch.y)))
        return data

class TorchEnsembleDataset(torch.utils.data.Dataset):
    def __init__(self, pyg_dataset, normalizer=None):
        self.dataset = pyg_dataset
        self.normalizer = normalizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx].clone()

        y = data.y
        if self.normalizer is not None:
            y = self.normalizer(y)

        data.y = None  # important!
        return data, y