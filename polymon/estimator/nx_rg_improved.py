from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional

from polymon.estimator.base import BaseEstimator
from polymon.model.register import register_init_params

# Extended bond lengths database with more accurate values
BOND_LENGTHS = {
    # Carbon-Carbon bonds
    (6, 6, Chem.BondType.SINGLE): 0.154,   # C-C single
    (6, 6, Chem.BondType.DOUBLE): 0.134,   # C=C double
    (6, 6, Chem.BondType.TRIPLE): 0.120,   # C≡C triple
    (6, 6, Chem.BondType.AROMATIC): 0.140, # Aromatic C-C
    
    # Carbon-Heteroatom bonds
    (6, 1, Chem.BondType.SINGLE): 0.109,   # C-H
    (6, 7, Chem.BondType.SINGLE): 0.147,   # C-N single
    (6, 7, Chem.BondType.DOUBLE): 0.127,   # C=N double
    (6, 7, Chem.BondType.TRIPLE): 0.115,   # C≡N triple
    (6, 8, Chem.BondType.SINGLE): 0.143,   # C-O single
    (6, 8, Chem.BondType.DOUBLE): 0.121,   # C=O double
    (6, 9, Chem.BondType.SINGLE): 0.141,   # C-F
    (6, 15, Chem.BondType.SINGLE): 0.184,  # C-P
    (6, 16, Chem.BondType.SINGLE): 0.181,  # C-S
    (6, 17, Chem.BondType.SINGLE): 0.177,  # C-Cl
    (6, 35, Chem.BondType.SINGLE): 0.194,  # C-Br
    (6, 53, Chem.BondType.SINGLE): 0.214,  # C-I
    
    # Nitrogen bonds
    (7, 1, Chem.BondType.SINGLE): 0.101,   # N-H
    (7, 7, Chem.BondType.SINGLE): 0.146,   # N-N single
    (7, 7, Chem.BondType.DOUBLE): 0.125,   # N=N double
    (7, 8, Chem.BondType.SINGLE): 0.136,   # N-O single
    (7, 8, Chem.BondType.DOUBLE): 0.120,   # N=O double
    (7, 9, Chem.BondType.SINGLE): 0.134,   # N-F
    
    # Oxygen bonds
    (8, 1, Chem.BondType.SINGLE): 0.096,   # O-H
    (8, 8, Chem.BondType.SINGLE): 0.143,   # O-O single
    (8, 8, Chem.BondType.DOUBLE): 0.133,   # O=O double
    (8, 16, Chem.BondType.SINGLE): 0.265,  # O-S single
    (8, 16, Chem.BondType.DOUBLE): 0.149,  # O=S double
    
    # Sulfur bonds
    (16, 1, Chem.BondType.SINGLE): 0.134,  # S-H
    (16, 16, Chem.BondType.SINGLE): 0.204, # S-S single
}

# Persistence length estimates for different polymer types (in Angstroms)
PERSISTENCE_LENGTHS = {
    'flexible': 2.0,      # Typical flexible polymer
    'semi-flexible': 10.0, # Semi-flexible polymer
    'stiff': 50.0,        # Stiff polymer
    'rigid': 200.0,       # Very rigid polymer
}

# Characteristic ratio estimates based on polymer structure
CHARACTERISTIC_RATIOS = {
    'flexible': 6.7,      # Typical flexible chain
    'semi-flexible': 8.5, # Semi-flexible chain
    'stiff': 12.0,        # Stiff chain
    'rigid': 20.0,        # Very rigid chain
}

DEFAULT_N = 375
DEFAULT_SOLVENT = "theta"


@register_init_params
class NxRgEstimatorImproved(BaseEstimator):
    def __init__(
        self, 
        N: int = DEFAULT_N, 
        solvent: str = DEFAULT_SOLVENT,
        use_dynamic_c_inf: bool = True,
        use_persistence_length: bool = True,
    ):
        self.N = N
        self.solvent = solvent
        self.use_dynamic_c_inf = use_dynamic_c_inf
        self.use_persistence_length = use_persistence_length

    def estimated_y(self, smiles: str) -> float:
        return self.radius_of_gyration(smiles)
    
    def mol_to_nx(self, mol):
        """Convert RDKit Mol to NetworkX weighted graph with enhanced bond analysis."""
        G = nx.Graph()
        
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            
            # Create bond key
            key = (min(a1.GetAtomicNum(), a2.GetAtomicNum()),
                   max(a1.GetAtomicNum(), a2.GetAtomicNum()),
                   bond.GetBondType())
            
            # Get bond length with better fallback
            length = BOND_LENGTHS.get(key, self._estimate_bond_length(a1, a2, bond))
            
            # Add edge with weight as bond length
            G.add_edge(a1.GetIdx(), a2.GetIdx(), 
                      weight=length,
                      bond_type=bond.GetBondType(),
                      atoms=(a1.GetAtomicNum(), a2.GetAtomicNum()))
        
        return G

    def _estimate_bond_length(self, atom1, atom2, bond):
        """Estimate bond length based on atomic radii and bond type."""
        # Covalent radii (in Angstroms)
        covalent_radii = {
            1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
            15: 1.07, 16: 1.05, 17: 0.99, 35: 1.20, 53: 1.39
        }
        
        r1 = covalent_radii.get(atom1.GetAtomicNum(), 0.8)
        r2 = covalent_radii.get(atom2.GetAtomicNum(), 0.8)
        
        # Bond order factor
        bond_order_factor = {
            Chem.BondType.SINGLE: 1.0,
            Chem.BondType.DOUBLE: 0.9,
            Chem.BondType.TRIPLE: 0.8,
            Chem.BondType.AROMATIC: 0.95
        }
        
        factor = bond_order_factor.get(bond.GetBondType(), 1.0)
        return (r1 + r2) * factor

    def analyze_polymer_structure(self, mol):
        """Analyze polymer structure to determine flexibility characteristics."""
        # Count different bond types
        single_bonds = 0
        double_bonds = 0
        triple_bonds = 0
        aromatic_bonds = 0
        ring_atoms = 0
        
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                single_bonds += 1
            elif bond.GetBondType() == Chem.BondType.DOUBLE:
                double_bonds += 1
            elif bond.GetBondType() == Chem.BondType.TRIPLE:
                triple_bonds += 1
            elif bond.GetBondType() == Chem.BondType.AROMATIC:
                aromatic_bonds += 1
        
        # Count ring atoms
        ring_info = mol.GetRingInfo()
        ring_atoms = len(ring_info.AtomRings())
        
        # Calculate molecular weight
        mw = Descriptors.MolWt(mol)
        
        # Determine flexibility based on structure
        total_bonds = single_bonds + double_bonds + triple_bonds + aromatic_bonds
        
        if aromatic_bonds > total_bonds * 0.3 or ring_atoms > 0:
            # Aromatic/cyclic structures are stiffer
            if triple_bonds > 0:
                return 'rigid'
            elif double_bonds > total_bonds * 0.2:
                return 'stiff'
            else:
                return 'semi-flexible'
        elif double_bonds > total_bonds * 0.1 or triple_bonds > 0:
            return 'semi-flexible'
        else:
            return 'flexible'

    def calculate_contour_length(self, mol):
        """Calculate more accurate contour length using multiple methods."""
        G = self.mol_to_nx(mol)
        if len(G) == 0:
            return 0.0
        
        # Method 1: Longest path (current method)
        longest_path = self._longest_path_length(G)
        
        # Method 2: Average path length weighted by frequency
        avg_path = self._average_path_length(G)
        
        # Method 3: Root mean square path length
        rms_path = self._rms_path_length(G)
        
        # Combine methods with weights
        contour_length = 0.5 * longest_path + 0.3 * avg_path + 0.2 * rms_path
        
        return contour_length

    def _longest_path_length(self, G):
        """Find the longest shortest path length (diameter)."""
        if len(G) == 0:
            return 0.0
        
        lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
        best_len = 0.0
        for d in lengths.values():
            max_len = max(d.values(), default=0.0)
            if max_len > best_len:
                best_len = max_len
        return best_len

    def _average_path_length(self, G):
        """Calculate average path length weighted by frequency."""
        if len(G) == 0:
            return 0.0
        
        lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
        all_lengths = []
        for d in lengths.values():
            all_lengths.extend(d.values())
        
        return np.mean(all_lengths) if all_lengths else 0.0

    def _rms_path_length(self, G):
        """Calculate root mean square path length."""
        if len(G) == 0:
            return 0.0
        
        lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
        all_lengths = []
        for d in lengths.values():
            all_lengths.extend(d.values())
        
        return np.sqrt(np.mean([l**2 for l in all_lengths])) if all_lengths else 0.0

    def estimate_characteristic_ratio(self, mol, contour_length):
        """Estimate characteristic ratio based on polymer structure."""
        if not self.use_dynamic_c_inf:
            return 6.7
        
        # Analyze polymer structure
        structure_type = self.analyze_polymer_structure(mol)
        base_c_inf = CHARACTERISTIC_RATIOS[structure_type]
        
        # Adjust based on molecular weight and structure
        mw = Descriptors.MolWt(mol)
        
        # Higher MW typically leads to higher C_inf
        mw_factor = min(1.2, 1.0 + 0.1 * np.log10(mw / 100))
        
        # Adjust for ring content
        ring_info = mol.GetRingInfo()
        ring_atoms = len(ring_info.AtomRings())
        total_atoms = mol.GetNumAtoms()
        ring_factor = 1.0 + 0.2 * (ring_atoms / max(total_atoms, 1))
        
        # Adjust for branching (simplified)
        branching_factor = 1.0  # Could be enhanced with more sophisticated analysis
        
        c_inf = base_c_inf * mw_factor * ring_factor * branching_factor
        
        return c_inf

    def estimate_persistence_length(self, mol):
        """Estimate persistence length based on polymer structure."""
        if not self.use_persistence_length:
            return 2.0  # Default flexible polymer
        
        structure_type = self.analyze_polymer_structure(mol)
        base_lp = PERSISTENCE_LENGTHS[structure_type]
        
        # Adjust based on molecular weight
        mw = Descriptors.MolWt(mol)
        mw_factor = min(2.0, 1.0 + 0.5 * np.log10(mw / 100))
        
        return base_lp * mw_factor

    def monomer_contour_length(self, smiles):
        """Calculate monomer contour length with improved accuracy."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        
        return self.calculate_contour_length(mol)

    def kuhn_length(self, char_ratio, bond_length=0.154):
        """Calculate Kuhn length from characteristic ratio."""
        return char_ratio * bond_length

    def radius_of_gyration(self, smiles):
        """
        Estimate Rg using improved polymer physics models.
        """
        N = self.N
        solvent = self.solvent
        
        # Calculate monomer contour length
        l_m = self.monomer_contour_length(smiles)
        
        # Estimate characteristic ratio dynamically
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        
        C_inf = self.estimate_characteristic_ratio(mol, l_m)
        
        # Calculate Kuhn length
        b = self.kuhn_length(C_inf)
        
        # Total contour length
        L_c = N * l_m
        
        # Number of Kuhn segments
        N_K = L_c / b
        
        # Apply solvent conditions
        if solvent == "theta":
            # Theta solvent: ideal chain behavior
            Rg = (b / (6 ** 0.5)) * (N_K ** 0.5)
        elif solvent == "good":
            # Good solvent: excluded volume effects
            nu = 0.588  # Flory exponent for 3D
            Rg = (b / (6 ** 0.5)) * (N_K ** nu)
        elif solvent == "melt":
            # Melt conditions: screening effects
            nu = 0.5  # Ideal behavior due to screening
            Rg = (b / (6 ** 0.5)) * (N_K ** nu)
        else:
            raise ValueError("solvent must be 'theta', 'good', or 'melt'")
        
        # Apply persistence length correction if enabled
        if self.use_persistence_length:
            lp = self.estimate_persistence_length(mol)
            # Correction factor for semi-flexible chains
            if lp > 0:
                correction_factor = min(1.2, 1.0 + 0.1 * np.log10(lp / 2.0))
                Rg *= correction_factor
        
        return Rg