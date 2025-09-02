from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.linear_model import LinearRegression

from polymon.estimator.base import BaseEstimator
from polymon.model.register import register_init_params

MAX_NUM_ELEMENTS = 100

@register_init_params
class AtomContribEstimator(BaseEstimator):
    def __init__(
        self,
        atom_contrib: np.ndarray,
    ):
        self.atom_contrib = atom_contrib

    @classmethod
    def from_fitting(
        cls,
        df: pd.DataFrame,
        smiles_col: str = 'SMILES',
        label_col: str = 'FFV'
    ) -> 'AtomContribEstimator':
        atom_nums = []
        for smiles in df[smiles_col]:
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            atom_num = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
            atom_nums.append(atom_num)

        X = np.zeros((len(atom_nums), MAX_NUM_ELEMENTS))
        y = np.zeros([len(atom_nums)])
        for i, (atom_num, energy) in enumerate(zip(atom_nums, df[label_col])):
            composition_fea = np.bincount(atom_num, minlength=MAX_NUM_ELEMENTS)
            X[i, :] = composition_fea
            y[i] = energy
        
        # 2. train a linear model
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        return cls(model.coef_)
    
    @classmethod
    def from_npy(cls, path: str) -> 'AtomContribEstimator':
        atom_contrib = np.load(path)
        return cls(atom_contrib)
    
    def write(self, path: str) -> None:
        np.save(path, self.atom_contrib)
    
    def estimated_y(self, smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        atom_num = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        composition_fea = np.bincount(atom_num, minlength=MAX_NUM_ELEMENTS)
        return np.dot(composition_fea, self.atom_contrib)