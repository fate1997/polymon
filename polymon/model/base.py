import os
from abc import ABC, abstractmethod
from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict, List
import math

import numpy as np
import torch
from rdkit import Chem
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from polymon.data.featurizer import ComposeFeaturizer
from polymon.data.polymer import Polymer
from polymon.data.utils import LogNormalizer, Normalizer

if TYPE_CHECKING:
    from polymon.estimator.base import BaseEstimator


class BaseModel(nn.Module, ABC):
    """Base Model.

    Args:
        nn.Module (ABC): The base class of the model.
    """
    @abstractmethod
    def forward(self, batch: Polymer) -> torch.Tensor:
        pass
    
    @property
    def init_params(self) -> Dict[str, Any]:
        return getattr(self, '_init_params')


class KFoldModel(BaseModel):
    """K-Fold Model. The output is the average of the predictions of the models
    trained on the different folds.

    Args:
        model_cls (str): The class of the model.
        model_init_params (Dict[str, Any]): The initial parameters of the model.
        n_fold (int): The number of folds.
    """
    def __init__(
        self,
        model_cls: str,
        model_init_params: Dict[str, Any],
        n_fold: int = 5,
    ):
        super().__init__()
        self.model_cls = model_cls
        self.model_init_params = model_init_params
        self.n_fold = n_fold
        self.models = nn.ModuleList()
        for _ in range(n_fold):
            model_cls = getattr(import_module('polymon.model'), self.model_cls)
            model = model_cls(**self.model_init_params)
            self.models.append(model)
    
    @classmethod
    def from_models(cls, models: List['ModelWrapper']) -> 'KFoldModel':
        """Build a K-Fold Model from a list of models.

        Args:
            models (List['ModelWrapper']): The models.

        Returns:
            'KFoldModel': The K-Fold Model.
        """
        model_cls = models[0].info['model_cls']
        model_init_params = models[0].info['model_init_params']
        n_fold = len(models)
        kfold_model = cls(model_cls, model_init_params, n_fold)
        for i, model in enumerate(models):
            kfold_model.models[i].load_state_dict(model.info['model'])
        return kfold_model
    
    def forward(self, batch: Polymer) -> torch.Tensor:
        """Forward pass. The output is the average of the predictions of the 
        models trained on the different folds.

        Args:
            batch (Polymer): The batch of polymers.

        Returns:
            torch.Tensor: The output of the model.
        """
        output = []
        for model in self.models:
            output.append(model(batch))
        output = torch.stack(output, dim=0).mean(0)
        return output

    @property
    def init_params(self) -> Dict[str, Any]:
        """Get the initial parameters of the model.

        Returns:
            Dict[str, Any]: The initial parameters of the model.
        """
        return {
            'model_cls': self.model_cls,
            'model_init_params': self.model_init_params,
            'n_fold': self.n_fold,
        }


class ModelWrapper(nn.Module):
    """Model Wrapper. This wrapper is used to wrap the model, normalizer, 
    featurizer, transform, and estimator, and make it easier to do inference.

    Args:
        model (BaseModel): The model.
        normalizer (Normalizer): The normalizer.
        featurizer (ComposeFeaturizer): The featurizer.
        transform_cls (str): The class of the transform.
        transform_kwargs (Dict[str, Any]): The initial parameters of the transform.
        estimator (BaseEstimator): The estimator.
    """
    def __init__(
        self, 
        model: BaseModel,
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

    def forward(
        self, 
        batch: Batch,
        loss_fn: nn.Module,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            batch (Batch): The batch of polymers.
            loss_fn (nn.Module): The loss function.
            device (str): The device to use.

        Returns:
            torch.Tensor: The loss.
        """
        self.model.train()
        self.model.to(device)
        batch = batch.to(device)
        y_pred = self.model(batch)
        y_true_transformed = self.normalizer(batch.y)
        loss = loss_fn(y_pred, y_true_transformed)
        return loss
    
    @torch.no_grad()
    def predict_batch(
        self, 
        batch: Batch,
        device: str = 'cuda',
        return_var: bool = False,
    ) -> torch.Tensor:
        """Predict the output of the model for a batch of polymers.

        Args:
            batch (Batch): The batch of polymers.
            device (str): The device to use.

        Returns:
            torch.Tensor: The output of the model.
        """
        self.model.eval()
        self.model.to(device)
        batch = batch.to(device)
        y_pred = self.model(batch)
        num_task = self.normalizer.init_params['mean'].shape[0]
        predict_logvar = getattr(self.model, 'predict_logvar', False)
        if predict_logvar:
            y_pred, raw_var = torch.split(y_pred, [num_task, num_task], dim=1)
            raw_var = torch.where(
                torch.isfinite(raw_var), raw_var, torch.zeros_like(raw_var)
            )
            y_var_norm = F.softplus(raw_var) + 1e-3
            y_var_norm = torch.clamp(y_var_norm, max=1e3)
            y_logvar_norm = torch.log(y_var_norm)
            if hasattr(self, 'task_log_sigma') and self.task_log_sigma is not None:
                s = self.task_log_sigma.view(1, -1).to(device)
            else:
                s = torch.zeros(1, num_task, device=device)
            total_log_var = (y_logvar_norm + s).clamp_(math.log(1e-3), math.log(1e3))
            var_norm = torch.exp(total_log_var)
        else:
            y_pred = y_pred
            var_norm = None
        y_pred = self.normalizer.inverse(y_pred)
        y_pred = y_pred.squeeze(0).squeeze(0)
        if var_norm is not None:
            std = self.normalizer.init_params['std'].to(device).view(1, -1)
            y_var = var_norm * std.pow(2)
        else:
            y_var = None

        if hasattr(batch, 'estimated_y'):
            y_pred = y_pred + batch.estimated_y.squeeze(0).squeeze(0)
        if y_var is not None:
            y_var = y_var.squeeze(0).squeeze(0)
        if return_var:
            return y_pred, y_var
        return y_pred

    @torch.no_grad()
    def predict(
        self, 
        smiles_list: List[str],
        batch_size: int = 128,
        device: str = 'cpu',
        backup_model: 'ModelWrapper' = None,
        return_var: bool = False,
    ) -> torch.Tensor:
        """Predict the output of the model for a list of polymer SMILES strings.

        Args:
            smiles_list (List[str]): The list of SMILES strings.
            batch_size (int): The batch size.
            device (str): The device to use.
            backup_model (ModelWrapper): The backup model. If the model fails to 
                predict the output for a polymer, the backup model will be used 
                to predict the output.

        Returns:
            torch.Tensor: The output of the model.
        """
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
                try:
                    polymer = self.estimator.forward(polymer, device=device)
                except Exception as e:
                    backup_ids.append(i)
            polymers.append(polymer)
        loader = DataLoader(polymers, batch_size=batch_size)
        
        y_pred_list = []
        y_var_list = [] if return_var else None
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            if i not in backup_ids:
                try:
                    if return_var:
                        y_pred, y_var = self.predict_batch(batch, device, return_var=True)
                    else:
                        y_pred = self.predict_batch(batch, device, return_var=False)
                        y_var = None
                except Exception as e:
                    y_pred = backup_model.predict([smiles_list[i]], device=device)
            else:
                y_pred = backup_model.predict([smiles_list[i]], device=device)
            y_pred_list.append(y_pred)
            if return_var:
                if y_var is None:
                    y_var = torch.full_like(y_pred, float('nan'))
                y_var_list.append(y_var)
        if len(y_pred_list) == 1:
            y_pred_out = y_pred_list[0].detach().cpu()
            if return_var:
                y_var_out = y_var_list[0].detach().cpu()
                return y_pred_out, y_var_out
            else:
                return y_pred_out
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
        """Write the model to a file.

        Args:
            path (str): The path to the file.

        Returns:
            str: The path to the file.
        """
        torch.save(self.info, path)
        return os.path.abspath(path)
    
    @classmethod
    def from_dict(cls, model_info: Dict[str, Any]) -> 'ModelWrapper':
        """Build a ModelWrapper from a dictionary.

        Args:
            model_info (Dict[str, Any]): The information of the model.

        Returns:
            'ModelWrapper': The ModelWrapper.
        """
        model_cls = model_info['model_cls']
        if model_cls == 'KFoldModel':
            model_cls = KFoldModel
        else:
            model_cls = getattr(import_module('polymon.model'), model_cls)
        model = model_cls(**model_info['model_init_params'])
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
        """Build a ModelWrapper from a file.

        Args:
            path (str): The path to the file.
            map_location (str): The map location.
            weights_only (bool): Whether to load only the weights.

        Returns:
            'ModelWrapper': The ModelWrapper.
        """
        output = torch.load(
            path, 
            map_location=map_location, 
            weights_only=weights_only
        )
        return cls.from_dict(output)


