from rdkit import Chem

from polymon.estimator.base import BaseEstimator
from polymon.model.register import register_init_params
import networkx as nx

BOND_LENGTHS = {
    (6, 6, Chem.BondType.SINGLE): 0.154,   # C-C single
    (6, 6, Chem.BondType.DOUBLE): 0.134,   # C=C double
    (6, 6, Chem.BondType.AROMATIC): 0.140, # Aromatic C-C
    (6, 8, Chem.BondType.SINGLE): 0.143,   # C-O
    (6, 7, Chem.BondType.SINGLE): 0.147,   # C-N
    (7, 7, Chem.BondType.SINGLE): 0.146,   # N-N
    (8, 8, Chem.BondType.SINGLE): 0.143,   # O-O
    (6, 6, Chem.BondType.TRIPLE): 0.120,   # C≡C triple
    (7, 8, Chem.BondType.SINGLE): 0.136,   # N-O
    (6, 9, Chem.BondType.SINGLE): 0.141,   # C-F
    (6, 16, Chem.BondType.SINGLE): 0.181,   # C-S
    (7, 9, Chem.BondType.SINGLE): 0.134,   # N-F
    (6, 7, Chem.BondType.DOUBLE): 0.127,   # C=N double
    (6, 7, Chem.BondType.TRIPLE): 0.115,   # C≡N triple
    (6, 8, Chem.BondType.DOUBLE): 0.121,   # C=O double
    (8, 16, Chem.BondType.SINGLE): 0.265,   # O-S single
    (8, 8, Chem.BondType.DOUBLE): 0.133,   # O=O double
}
DEFAULT_N = 680
DEFAULT_C_INF = 6.7
DEFAULT_SOLVENT = "theta"


@register_init_params
class NxRgEstimator(BaseEstimator):
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
        return self.radius_of_gyration(smiles)
    
    def mol_to_nx(self, mol):
        """Convert RDKit Mol to NetworkX weighted graph."""
        G = nx.Graph()
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            key = (min(a1.GetAtomicNum(), a2.GetAtomicNum()),
                max(a1.GetAtomicNum(), a2.GetAtomicNum()),
                bond.GetBondType())
            length = BOND_LENGTHS.get(key, 0.154)  # fallback to C-C
            G.add_edge(a1.GetIdx(), a2.GetIdx(), weight=length)
        return G

    def longest_backbone_path_length(self, mol):
        """
        Approximate the longest backbone path length using NetworkX.
        Finds the longest shortest path (diameter in weighted sense).
        """
        G = self.mol_to_nx(mol)
        if len(G) == 0:
            return 0.0
        
        # All-pairs shortest paths weighted by bond lengths
        lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
        
        # Find maximum among all shortest path lengths (graph diameter)
        best_len = 0.0
        for d in lengths.values():
            max_len = max(d.values(), default=0.0)
            if max_len > best_len:
                best_len = max_len
        
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