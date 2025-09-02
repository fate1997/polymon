from polymon.estimator.base import BaseEstimator
from polymon.estimator.density import DensityEstimator
from polymon.estimator.rg import RgEstimator
from polymon.estimator.atom_contrib import AtomContribEstimator
from polymon.estimator.low_fidelity import LowFidelityEstimator
from polymon.setting import REPO_DIR


def get_estimator(label: str, **kwargs) -> BaseEstimator:
    if label == 'Rg':
        return RgEstimator(**kwargs)
    elif label == 'Density':
        return DensityEstimator(**kwargs)
    elif label == 'FFV':
        return AtomContribEstimator.from_npy(
            REPO_DIR / 'polymon' / 'estimator' / 'FFV_atom_contrib.npy'
        )
    elif label == 'Tc':
        return AtomContribEstimator.from_npy(
            REPO_DIR / 'polymon' / 'estimator' / 'Tc_atom_contrib.npy'
        )
    elif label == 'LowFidelity':
        return LowFidelityEstimator(**kwargs)
    else:
        raise ValueError(f'Invalid estimator label: {label}')