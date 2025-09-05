from .density_Fedors import get_fedors_density
import math

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

def get_ffv(mol) -> float:
    Vvdw = 0.0  # in Å^3
    pt = AllChem.GetPeriodicTable()
    for atom in mol.GetAtoms():
        r = pt.GetRvdw(atom.GetAtomicNum())  # Å
        Vvdw += (4.0/3.0) * math.pi * (r**3)
    
    density = get_fedors_density(mol)
    return 1 - 1.3*density*(Vvdw / Descriptors.ExactMolWt(Chem.AddHs(mol)))

