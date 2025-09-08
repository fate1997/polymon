import os

from typing import List, Callable, Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
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
        base_preds = self.base_predict(smiles_list, batch_size, device)
        base_preds = torch.from_numpy(base_preds)
        self.meta_weigths = self.meta_weigths.to(base_preds.device, dtype=base_preds.dtype)
        self.meta_bias = self.meta_bias.to(base_preds.device, dtype=base_preds.dtype)
        meta_preds = self.meta_bias + base_preds @ self.meta_weigths
        meta_preds = meta_preds.detach().numpy()
        return meta_preds, base_preds.detach().numpy()
    
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
        y_pred = model.predict(X)
        return torch.from_numpy(y_pred)
    
    def write(self, path: str) -> str:
        torch.save(self.info, path)
        return os.path.abspath(path)
    
    @classmethod
    def from_file(cls, path: str) -> 'EnsembleRegressor':
        info = torch.load(path)
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