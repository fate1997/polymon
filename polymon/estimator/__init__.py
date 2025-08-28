from polymon.estimator.base import BaseEstimator
from polymon.estimator.density import DensityEstimator
from polymon.estimator.rg import RgEstimator


def get_estimator(label: str, **kwargs) -> BaseEstimator:
    if label == 'Rg':
        return RgEstimator(**kwargs)
    elif label == 'Density':
        return DensityEstimator(**kwargs)
    else:
        raise ValueError(f'Invalid estimator label: {label}')