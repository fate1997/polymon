import torch
import numpy as np
import os
from typing import List, Literal, Optional

from polymon.model.base import ModelWrapper, KFoldModel
from polymon.model.ensemble import EnsembleModelWrapper

class Acquisition:
    def __init__(
        self,
        acquisition_function: str,
        model_file: str,
        model_type: Literal['default', 'KFold', 'ensemble'] = 'default',
        n_fold: Optional[int] = None,
        device: str = 'cpu',
    ):
        self.acquisition_function = acquisition_function
        self.device = device
        if model_type == 'default':
            self.model = ModelWrapper.from_file(model_file)
        elif model_type == 'KFold':
            model_names = [os.path.join(model_file.split('/')[-3], f'fold_{i}', f'{model_file.split('/')[-1]}') for i in range(1, n_fold + 1)]
            models = [ModelWrapper.from_file(model_name) for model_name in model_names]
            self.model = models
        elif model_type == 'ensemble':
            self.model = EnsembleModelWrapper.from_file(model_file)


    def __call__(self, pool_smiles: List[str]):
        return self.acquire(pool_smiles)
    
    def acquire(self, pool_smiles: List[str]):
        if self.acquisition_function == "uncertainty":
            return self.uncertainty(pool_smiles)
        elif self.acquisition_function == "entropy":
            return self.entropy(pool_smiles)
        elif self.acquisition_function == "margin":
            return self.margin(pool_smiles)
        elif self.acquisition_function == "entropy_margin":
            return self.entropy_margin(pool_smiles)
        elif self.acquisition_function == "epig":
            return self.epig(pool_smiles)
        elif self.acquisition_function == "similarity":
            return self.similarity(pool_smiles)
        elif self.acquisition_function == "random":
            return self.random(pool_smiles)
    
    def uncertainty(self, pool_smiles: List[str], n_sample: int = 50) -> List[str]:
        assert isinstance(self.model, EnsembleModelWrapper) or (isinstance(self.model, list) and len(self.model) > 1), \
            "self.model must be an EnsembleModelWrapper or a list of models with length > 1"
        if isinstance(self.model, list):
            for model in self.model:
                model.eval()
                model.to(self.device)
                preds = model.predict(pool_smiles, batch_size=128)
                preds = torch.stack(preds, dim=0)
            uncertainty = preds.std(0)
            # top n_sample most uncertain
            top_n_sample = uncertainty.argsort()[-n_sample:]
            top_smiles = [pool_smiles[i] for i in top_n_sample]
        
        return top_smiles
    
    def entropy(self, pool_smiles: List[str]):
        pass
    
    def margin(self, pool_smiles: List[str]):
        pass
    
    def entropy_margin(self, pool_smiles: List[str]):
        pass
    
    def epig(self, pool_smiles: List[str]):
        pass
    
    def similarity(self, pool_smiles: List[str]):
        pass
    
    def random(self, pool_smiles: List[str]):
        pass