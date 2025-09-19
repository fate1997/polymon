from typing import Any, Dict

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.linear_model import LinearRegression

from polymon.estimator.base import BaseEstimator
from polymon.model.register import register_init_params

MAX_NUM_ELEMENTS = 100


@register_init_params
class AtomContribEstimator(BaseEstimator):
    """Atom contribution estimator. It is used to estimate the label of a 
    polymer based on the atom composition.

    Args:
        atom_contrib (np.ndarray): The atom contribution coefficients.
            It is a 1D array of shape (MAX_NUM_ELEMENTS,).
            The i-th element is the contribution of the i-th element to the label.
            The i-th element is the number of atoms of the i-th element in the polymer.
            The MAX_NUM_ELEMENTS is 100.
    """
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
        """Fit the atom contribution estimator from a dataframe.

        Args:
            df (pd.DataFrame): The dataframe containing the SMILES and label.
            smiles_col (str): The column name of the SMILES.
            label_col (str): The column name of the label.

        Returns:
            AtomContribEstimator: The fitted atom contribution estimator.
        """
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
        """Load the atom contribution estimator from a npy file.

        Args:
            path (str): The path to the npy file.

        Returns:
            AtomContribEstimator: The loaded atom contribution estimator.
        """
        atom_contrib = np.load(path)
        return cls(atom_contrib)
    
    def write(self, path: str) -> None:
        """Write the atom contribution estimator to a npy file.

        Args:
            path (str): The path to the npy file.
        """
        np.save(path, self.atom_contrib)
    
    def estimated_y(self, smiles: str) -> float:
        """Estimate the label of a polymer based on the atom composition.

        Args:
            smiles (str): The SMILES of the polymer.

        Returns:
            float: The estimated label of the polymer.
        """
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        atom_num = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        composition_fea = np.bincount(atom_num, minlength=MAX_NUM_ELEMENTS)
        return np.dot(composition_fea, self.atom_contrib)