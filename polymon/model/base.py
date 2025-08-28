import os
from abc import ABC, abstractmethod
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Tuple
from functools import partial
from torchensemble._base import BaseRegressor
import numpy as np
import torch
from rdkit import Chem
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from sklearn.linear_model import LinearRegression

from polymon.data.featurizer import ComposeFeaturizer
from polymon.exp.score import scaling_error
from polymon.data.polymer import Polymer
from polymon.data.utils import Normalizer
from polymon.estimator.base import BaseEstimator


class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch: Polymer) -> torch.Tensor:
        pass
    
    @property
    def init_params(self) -> Dict[str, Any]:
        return getattr(self, '_init_params')


class ModelWrapper(nn.Module):
    def __init__(
        self, 
        model: BaseModel,
        normalizer: 'Normalizer',
        featurizer: ComposeFeaturizer,
        transform_cls: str = None,
        transform_kwargs: Dict[str, Any] = None,
        estimator: BaseEstimator = None,
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

    def forward(
        self, 
        batch: Batch,
        loss_fn: nn.Module,
        device: str = 'cuda',
    ) -> torch.Tensor:
        self.model.train()
        self.model.to(device)
        batch = batch.to(device)
        y_pred = self.model(batch)
        y_true_transformed = self.normalizer(batch.y)
        loss = loss_fn(y_pred, y_true_transformed)
        return loss

    @torch.no_grad()
    def predict(
        self, 
        smiles_list: List[str],
        batch_size: int = 128,
        device: str = 'cpu',
        backup_model: 'ModelWrapper' = None,
    ) -> torch.Tensor:
        self.model.eval()
        self.model.to(device)
        if backup_model is not None:
            backup_model.model.eval()
            backup_model.model.to(device)
            batch_size = 1
        
        polymers = []
        backup_ids = []
        for i, smiles in enumerate(smiles_list):
            rdmol = Chem.MolFromSmiles(smiles)
            mol_dict = self.featurizer(rdmol)
            mol_dict['smiles'] = smiles
            if None in mol_dict.values():
                mol_dict = backup_model.featurizer(rdmol)
                backup_ids.append(i)
            polymer = Polymer(**mol_dict)
            if self.transform is not None:
                polymer = self.transform(polymer)
            if self.estimator is not None:
                polymer = self.estimator(polymer)
            polymers.append(polymer)
        loader = DataLoader(polymers, batch_size=batch_size)
        
        y_pred_list = []
        for i, batch in enumerate(loader):
            batch.to(device)
            if i not in backup_ids:
                output = self.model(batch)
                output = self.normalizer.inverse(output)
                output = output.squeeze(0).squeeze(0)
                if hasattr(batch, 'estimated_y'):
                    output = output + batch.estimated_y.squeeze(0).squeeze(0)
                y_pred_list.append(output)
            else:
                y_pred_list.append(backup_model.predict([smiles_list[i]]))
        if len(y_pred_list) == 1:
            return y_pred_list[0].detach().cpu()
        if batch_size == 1:
            return torch.stack(y_pred_list, dim=0).detach().cpu().unsqueeze(-1)
        return torch.cat(y_pred_list, dim=0).detach().cpu()
    
    @property
    def info(self) -> Dict[str, Any]:
        output = {
            'model_cls': self.model.__class__.__name__,
            'model': self.model.state_dict(),
            'normalizer': {
                'mean': self.normalizer.mean,
                'std': self.normalizer.std,
            },
            'model_init_params': self.model.init_params,
            'featurizer_names': self.featurizer.names,
            'featurizer_config': self.featurizer.config,
            'featurizer_add_hydrogens': self.featurizer.add_hydrogens,
            'transform_cls': self.transform_cls,
            'transform_kwargs': self.transform_kwargs,
        }
        if self.estimator is not None:
            output['estimator_cls'] = self.estimator.__class__.__name__
            output['estimator_init_params'] = self.estimator.init_params
        return output
    
    def write(self, path: str) -> str:
        torch.save(self.info, path)
        return os.path.abspath(path)
    
    @classmethod
    def from_dict(cls, model_info: Dict[str, Any]) -> 'ModelWrapper':
        model_cls = model_info['model_cls']
        model_cls = getattr(import_module('polymon.model'), model_cls)
        model = model_cls(**model_info['model_init_params'])
        model.load_state_dict(model_info['model'])
        normalizer = Normalizer(
            mean=model_info['normalizer']['mean'],
            std=model_info['normalizer']['std'],
        )
        featurizer = ComposeFeaturizer(
            names=model_info['featurizer_names'],
            config=model_info['featurizer_config'],
            add_hydrogens=model_info['featurizer_add_hydrogens'],
        )
        transform_cls = model_info.get('transform_cls', None)
        transform_kwargs = model_info.get('transform_kwargs', None)
        
        estimator_cls = model_info.get('estimator_cls', None)
        estimator_kwargs = model_info.get('estimator_init_params', None)
        if estimator_cls is not None:
            estimator_cls = getattr(import_module('polymon.estimator'), estimator_cls)
            estimator = estimator_cls(**estimator_kwargs)
        else:
            estimator = None
        return cls(model, normalizer, featurizer, transform_cls, transform_kwargs, estimator)
    
    @classmethod
    def from_file(
        cls, 
        path: str,
        map_location: str = 'cpu',
        weights_only: bool = False,
    ) -> 'ModelWrapper':
        output = torch.load(path, map_location=map_location, weights_only=weights_only)
        return cls.from_dict(output)


class EnsembleModelWrapper(nn.Module):
    def __init__(
        self, 
        model: BaseRegressor,
        normalizer: 'Normalizer',
        featurizer: ComposeFeaturizer,
        transform_cls: str = None,
        transform_kwargs: Dict[str, Any] = None,
        estimator: BaseEstimator = None,
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
        self.model.fit(
            train_loader=self.loader_wrapper(train_loader),
            epochs=epochs,
            test_loader=self.loader_wrapper(val_loader) if val_loader is not None else None,
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
                polymer = self.estimator(polymer)
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
        output = {
            'model_cls': self.model.__class__.__name__,
            'model': self.model.state_dict(),
            'normalizer': {
                'mean': self.normalizer.mean,
                'std': self.normalizer.std,
            },
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
        torch.save(self.info, path)
        return os.path.abspath(path)
    
    @classmethod
    def from_dict(cls, model_info: Dict[str, Any], device: str = 'cpu') -> 'EnsembleModelWrapper':
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
        model.load_state_dict(model_info['model'])
        normalizer = Normalizer(
            mean=model_info['normalizer']['mean'],
            std=model_info['normalizer']['std'],
        )
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
        output = torch.load(path, map_location=map_location, weights_only=weights_only)
        return cls.from_dict(output, device=map_location)
    
    def loader_wrapper(self, loader: DataLoader) -> List[Tuple[Polymer, torch.Tensor]]:
        data = []
        for batch in loader:
            data.append((batch, self.normalizer(batch.y)))
        return data