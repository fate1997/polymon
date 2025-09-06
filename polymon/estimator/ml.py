from typing import List
import pickle

from rdkit import Chem

from polymon.data.featurizer import DescFeaturizer
from polymon.estimator.base import BaseEstimator
from polymon.model.register import register_init_params


@register_init_params
class MLEstimator(BaseEstimator):
    def __init__(
        self, 
        model,
        feature_names: List[str] = ['rdkit2d'],
    ):
        self.model = model
        self.feature_names = feature_names
        self.featurizer = DescFeaturizer(feature_names)
    
    @classmethod
    def from_pickle(
        cls,
        path: str,
        feature_names: List[str] = None,
    ):        
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        if feature_names is None:
            feature_names = getattr(model, 'feature_names', None)
        if feature_names is None:
            raise ValueError('Feature names are not set')
        return cls(model, feature_names)
    
    def write_pickle(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def estimated_y(self, smiles: str) -> float:
        X = self.featurizer(Chem.MolFromSmiles(smiles))['descriptors']
        y = self.model.predict(X)
        return y.item()