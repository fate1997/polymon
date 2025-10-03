from rdkit import Chem

from polymon.estimator.base import BaseEstimator
from polymon.model.register import register_init_params

BOND_LENGTHS = {
    (6, 6, Chem.BondType.SINGLE): 0.154,   # C-C single
    (6, 6, Chem.BondType.DOUBLE): 0.134,   # C=C double
    (6, 6, Chem.BondType.AROMATIC): 0.140, # Aromatic C-C
    (6, 8, Chem.BondType.SINGLE): 0.143,   # C-O
    (6, 7, Chem.BondType.SINGLE): 0.147,   # C-N
    (7, 7, Chem.BondType.SINGLE): 0.146,   # N-N
    (8, 8, Chem.BondType.SINGLE): 0.143,   # O-O
}

DEFAULT_N = 1000
DEFAULT_C_INF = 6.7
DEFAULT_SOLVENT = "theta"

@register_init_params
class RgEstimator(BaseEstimator):
    """Radius of gyration estimator using DFS. This is a simple estimator
    that uses DFS to estimate the radius of gyration of a polymer.
    """
    def __init__(
        self, 
        N: int = DEFAULT_N, 
        C_inf: float = DEFAULT_C_INF, 
        solvent: str = DEFAULT_SOLVENT,
    ):
        self.N = N
        self.C_inf = C_inf
        self.solvent = solvent

    def estimated_y(self, smiles: str) -> float:
        """Estimate the radius of gyration of a polymer based on the DFS method.

        Args:
            smiles (str): The SMILES of the polymer.

        Returns:
            float: The estimated radius of gyration of the polymer.
        """
        return self.radius_of_gyration(smiles)
    
    def longest_backbone_path_length(self, mol):
        """Find the length of the longest simple path (approximate backbone)."""
        n_atoms = mol.GetNumAtoms()
        best_len = 0.0

        # Depth-first search over paths
        def dfs(atom_idx, visited, acc_len):
            nonlocal best_len
            best_len = max(best_len, acc_len)
            for bond in mol.GetAtomWithIdx(atom_idx).GetBonds():
                nbr = bond.GetOtherAtomIdx(atom_idx)
                if nbr in visited:
                    continue
                a1 = bond.GetBeginAtom().GetAtomicNum()
                a2 = bond.GetEndAtom().GetAtomicNum()
                key = (min(a1, a2), max(a1, a2), bond.GetBondType())
                length = BOND_LENGTHS.get(key, 0.154)  # fallback to C-C
                dfs(nbr, visited | {nbr}, acc_len + length)

        for start in range(n_atoms):
            dfs(start, {start}, 0.0)

        return best_len

    def monomer_contour_length(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        return self.longest_backbone_path_length(mol)

    def kuhn_length(self, char_ratio=6.7, bond_length=0.154):
        return char_ratio * bond_length

    def radius_of_gyration(self, smiles):
        """
        Estimate Rg from a monomer SMILES without using conformers.
        - N: degree of polymerization
        - C_inf: characteristic ratio (defaults ~6.7 for flexible chains)
        - solvent: 'theta' or 'good'
        """
        N = self.N
        C_inf = self.C_inf
        solvent = self.solvent
        
        l_m = self.monomer_contour_length(smiles)
        b = self.kuhn_length(C_inf)
        L_c = N * l_m
        N_K = L_c / b

        if solvent == "theta":
            Rg = (b / (6 ** 0.5)) * (N_K ** 0.5)
        elif solvent == "good":
            nu = 0.588
            Rg = (b / (6 ** 0.5)) * (N_K ** nu)
        else:
            raise ValueError("solvent must be 'theta' or 'good'")

        return Rg