import os

from typing import List, Callable, Any, Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from rdkit import Chem
from tqdm import tqdm
from polymon.data.featurizer import ComposeFeaturizer
from polymon.model.base import ModelWrapper


PREDICT_BATCH_SIZE = 128

class EnsembleRegressor(nn.Module):
    def __init__(
        self,
        base_builders: Dict[str, Callable[[], object]],
        meta_weigths: torch.Tensor,
        meta_bias: torch.Tensor,
        random_state: int = 42,
    ):

        super().__init__()

        self.base_builders = base_builders
        self.meta_weigths = meta_weigths
        self.meta_bias = meta_bias
        self.random_state = random_state
        
        self.feature_cache = defaultdict(dict)


    def fit(
        self, 
        smiles_list: List[str], 
        y: np.ndarray, 
        device: str = 'cpu',
    ):

        base_preds = self.base_predict(
            smiles_list, 
            batch_size=PREDICT_BATCH_SIZE, 
            device=device,
        )

        meta_builder = LinearRegression(fit_intercept=True)
        meta_builder.fit(base_preds, y)
        corf = meta_builder.coef_.astype(np.float32).reshape(-1)
        intercept = meta_builder.intercept_.astype(np.float32).reshape(())
        self.meta_weigths = torch.from_numpy(corf)
        self.meta_bias = torch.from_numpy(intercept)
        return self
    
    def predict(
        self,
        smiles_list: List[str],
        batch_size: int = 128,
        device: str = 'cpu',
    ) -> torch.Tensor:
        base_preds = self.base_predict(smiles_list, batch_size, device)
        base_preds = torch.from_numpy(base_preds).to(device)
        meta_preds = self.meta_bias + base_preds @ self.meta_weigths
        meta_preds = meta_preds.cpu().detach().unsqueeze(-1)
        return meta_preds
    
    def base_predict(
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
                y_preds = self.ml_predict(model, smiles_list)
            else:
                raise ValueError(f'Unknown builder: {name}')
            base_preds.append(y_preds)
        base_preds = np.column_stack(base_preds)
        
        return base_preds
            
    def ml_predict(self, model: Any, smiles_list: List[str]) -> torch.Tensor:
        feature_names = model.feature_names
        feature_name_str = '_'.join(feature_names)
        
        X = []
        for smiles in tqdm(smiles_list):
            flag = False
            if feature_name_str in self.feature_cache:
                if smiles in self.feature_cache[feature_name_str]:
                    X.append(self.feature_cache[feature_name_str][smiles])
                    flag = True
            if not flag:
                featurizer = ComposeFeaturizer(feature_names)
                X.append(featurizer(Chem.MolFromSmiles(smiles))['descriptors'])
                self.feature_cache[feature_name_str][smiles] = X[-1]
        X = np.stack(X, axis=0).squeeze(1)

        y_pred = model.predict(X)
        return torch.from_numpy(y_pred)
    
    def write(self, path: str) -> str:
        torch.save(self.info, path)
        return os.path.abspath(path)
    
    @classmethod
    def from_file(
        cls,
        path: str,
        map_location: str = 'cpu',
        weights_only: bool = False,
    ) -> 'EnsembleRegressor':
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
        model = cls(
            base_builders=base_builders,
            meta_weigths=meta_weigths,
            meta_bias=meta_bias,
            random_state=info['random_state'],
        )
        model.to(map_location)
        model.meta_bias = model.meta_bias.to(map_location)
        model.meta_weigths = model.meta_weigths.to(map_location)
        return model
    
    @property
    def info(self) -> Dict[str, Any]:

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