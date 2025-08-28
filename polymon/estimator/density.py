import math

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from polymon.estimator.base import BaseEstimator
from polymon.model.register import register_init_params

NA = 6.022e23
A3_TO_CM3 = 1e-24
DEFAULT_PACKING_COEFF = 0.80

@register_init_params
class DensityEstimator(BaseEstimator):
    def __init__(self, packing_coeff: float = DEFAULT_PACKING_COEFF):
        self.packing_coeff = packing_coeff

    def estimated_y(self, smiles: str) -> float:
        return self.density_vdw(smiles)
    
    def density_vdw(self, smiles: str) -> float: 
        packing_coeff = self.packing_coeff
        
        mol = Chem.MolFromSmiles(smiles)
        mw = Descriptors.MolWt(mol)
        pt = AllChem.GetPeriodicTable()

        Vvdw = 0.0  # in Å^3
        for atom in mol.GetAtoms():
            r = pt.GetRvdw(atom.GetAtomicNum())  # Å
            Vvdw += (4.0/3.0) * math.pi * (r**3)

        # molar van der Waals volume (cm3/mol)
        Vvdw_molar = Vvdw * A3_TO_CM3 * NA
        Vm = Vvdw_molar / packing_coeff
        rho = mw / Vm
        return rho