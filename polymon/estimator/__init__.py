from polymon.estimator.atom_contrib import AtomContribEstimator
from polymon.estimator.base import BaseEstimator
from polymon.estimator.density import DensityEstimator
from polymon.estimator.density_Fedors import DensityFedorsEstimator
from polymon.estimator.density_ibm import DensityIBMEstimator
from polymon.estimator.low_fidelity import LowFidelityEstimator
from polymon.estimator.ml import MLEstimator
from polymon.estimator.nx_rg import NxRgEstimator
from polymon.estimator.rg import RgEstimator
from polymon.estimator.tg import TgEstimator
from polymon.setting import REPO_DIR


def get_estimator(label: str, **kwargs) -> BaseEstimator:
    if label == 'Rg':
        return RgEstimator(**kwargs)
    elif label == 'NxRg':
        return NxRgEstimator(**kwargs)
    elif label == 'Density':
        return DensityEstimator(**kwargs)
    elif label == 'FFV':
        return AtomContribEstimator.from_npy(
            REPO_DIR / 'polymon' / 'estimator' / 'FFV_atom_contrib.npy'
        )
    elif label == 'Density-IBM':
        return DensityIBMEstimator(**kwargs)
    elif label == 'Density-Fedors':
        return DensityFedorsEstimator(**kwargs)
    elif label == 'Tc':
        return AtomContribEstimator.from_npy(
            REPO_DIR / 'polymon' / 'estimator' / 'Tc_atom_contrib.npy'
        )
    elif label == 'Tg':
        return AtomContribEstimator.from_npy(
            REPO_DIR / 'polymon' / 'estimator' / 'Tg_atom_contrib.npy'
        )
    elif label == 'LowFidelity':
        return LowFidelityEstimator(**kwargs)
    else:
        raise ValueError(f'Invalid estimator label: {label}')