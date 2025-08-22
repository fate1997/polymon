import os
from abc import ABC, abstractmethod
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import torch
from rdkit import Chem
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from polymon.data.featurizer import ComposeFeaturizer
from polymon.data.polymer import Polymer
from polymon.data.utils import Normalizer


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
    ):
        super().__init__()
        self.model = model
        self.normalizer = normalizer
        self.featurizer = featurizer
        self.transform_cls = transform_cls
        self.transform_kwargs = transform_kwargs
        if transform_cls is not None:
            transform_cls = getattr(import_module('torch_geometric.transforms'), transform_cls)
            self.transform = transform_cls(**transform_kwargs)
        else:
            self.transform = None

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
            if None in mol_dict.values():
                mol_dict = backup_model.featurizer(rdmol)
                backup_ids.append(i)
            polymer = Polymer(**mol_dict)
            if self.transform is not None:
                polymer = self.transform(polymer)
            polymers.append(polymer)
        loader = DataLoader(polymers, batch_size=batch_size)
        
        y_pred_list = []
        for i, batch in enumerate(loader):
            batch.to(device)
            if i not in backup_ids:
                output = self.model(batch)
                output = self.normalizer.inverse(output)
                output = output.squeeze(0).squeeze(0)
                y_pred_list.append(output)
            else:
                y_pred_list.append(backup_model.predict([smiles_list[i]]))
        if len(y_pred_list) == 1:
            return y_pred_list[0].detach().cpu()
        if batch_size == 1:
            return torch.stack(y_pred_list, dim=0).detach().cpu().unsqueeze(-1)
        return torch.cat(y_pred_list, dim=0).detach().cpu()
    
    def write(self, path: str, other_info: Dict[str, Any] = None) -> str:
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
        return os.path.abspath(path)
    
    @classmethod
    def from_file(
        cls, 
        path: str,
        map_location: str = 'cpu',
        weights_only: bool = False,
    ) -> 'ModelWrapper':
        output = torch.load(path, map_location=map_location, weights_only=weights_only)
        model_cls = output['model_cls']
        model_cls = getattr(import_module('polymon.model'), model_cls)
        model = model_cls(**output['model_init_params'])
        model.load_state_dict(output['model'])
        normalizer = Normalizer(
            mean=output['normalizer']['mean'],
            std=output['normalizer']['std'],
        )
        featurizer = ComposeFeaturizer(
            names=output['featurizer_names'],
            config=output['featurizer_config'],
            add_hydrogens=output['featurizer_add_hydrogens'],
        )
        transform_cls = output.get('transform_cls', None)
        transform_kwargs = output.get('transform_kwargs', None)
        return cls(model, normalizer, featurizer, transform_cls, transform_kwargs)