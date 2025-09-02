from typing import Any, Dict

import torch

from polymon.data.polymer import Polymer
from polymon.estimator.base import BaseEstimator
from polymon.model.base import ModelWrapper
from polymon.model.register import register_init_params


@register_init_params
class LowFidelityEstimator(BaseEstimator):
    def __init__(
        self,
        model_info: Dict[str, Any],
    ):
        self.model = ModelWrapper.from_dict(model_info)

    def forward(self, data: Polymer, **kwargs) -> Polymer:
        estimated_y = self.estimated_y(data.smiles)
        data.estimated_y = torch.tensor([[estimated_y]]).to(data.x.device)
        if getattr(data, 'y', None) is not None:
            data.y = data.y - estimated_y
        return data
    
    def estimated_y(self, smiles: str) -> float:
        estimated_y = self.model.predict([smiles]).item()
        return estimated_y