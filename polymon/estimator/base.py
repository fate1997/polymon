from abc import ABC, abstractmethod
from typing import Dict, Any

import torch
from torch_geometric.transforms import BaseTransform

from polymon.data.polymer import Polymer


class BaseEstimator(BaseTransform, ABC):

    def forward(self, data: Polymer) -> Polymer:
        estimated_y = self.estimated_y(data.smiles)
        data.estimated_y = torch.tensor([[estimated_y]]).to(data.x.device)
        if getattr(data, 'y', None) is not None:
            data.y = data.y - estimated_y
        return data
    
    @property
    def init_params(self) -> Dict[str, Any]:
        return getattr(self, '_init_params')
    
    @abstractmethod
    def estimated_y(self, smiles: str) -> float:
        pass