from typing import List
import pickle

from rdkit import Chem

from polymon.data.featurizer import DescFeaturizer
from polymon.estimator.base import BaseEstimator
from polymon.model.register import register_init_params


@register_init_params
class MLEstimator(BaseEstimator):
    """Machine learning estimator. This is a simple estimator that uses a model
    to estimate the label of a polymer. The model is a trained model on a
    dataset. The model should be an object of sklearn model.
    """
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
        """Load the machine learning estimator from a pickle file.

        Args:
            path (str): The path to the pickle file.
            feature_names (List[str]): The feature names.

        Returns:
            MLEstimator: The loaded machine learning estimator.
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        if feature_names is None:
            feature_names = getattr(model, 'feature_names', None)
        if feature_names is None:
            raise ValueError('Feature names are not set')
        return cls(model, feature_names)
    
    def write_pickle(self, path: str):
        """Write the machine learning estimator to a pickle file.

        Args:
            path (str): The path to the pickle file.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def estimated_y(self, smiles: str) -> float:
        """Estimate the label of a polymer based on the machine learning model.

        Args:
            smiles (str): The SMILES of the polymer.

        Returns:
            float: The estimated label of the polymer.
        """
        X = self.featurizer(Chem.MolFromSmiles(smiles))['descriptors']
        y = self.model.predict(X)
        return y.item()