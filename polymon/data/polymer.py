from typing import Optional

import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdChemReactions

class Polymer(Data):
    """Data object for multi-modal representation of polymers.
    
    Graph (2D/3D) Attributes:
        - x: Node features (num_nodes, num_node_features)
        - edge_index: Edge indices (2, num_edges)
        - edge_attr: Edge features (num_edges, num_edge_features)
        - attachments: Attachments (num_nodes, ) with 1 for attachment nodes
        - z: Atomic numbers (num_nodes, )
        - pos: Positions (num_nodes, 3)
    
    Sequence Attributes:
        - seq: Sequence (1, max_seq_len)
        - seq_len: Sequence length (1, )
    
    Descriptor Attributes:
        - descriptors: Descriptors (1, num_descriptors)
    
    Other Attributes:
        - y: Target (num_targets, )
        - smiles: a SMILES string
        - identifier: a unique identifier for the polymer
    """

    def __init__(
        self,
        x: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        attachments: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        seq: Optional[torch.Tensor] = None,
        seq_len: Optional[torch.Tensor] = None,
        descriptors: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        smiles: Optional[str] = None,
        identifier: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.attachments = attachments
        self.z = z
        self.pos = pos
        self.seq = seq
        self.seq_len = seq_len
        self.descriptors = descriptors
        self.y = y
        
        self.smiles = smiles
        self.identifier = identifier
        
    def __repr__(self) -> str:
        if self.identifier is None:
            return f"Polymer({self.smiles})"
        return f"Polymer({self.identifier})"
    
    @property
    def num_atoms(self) -> int:
        return self.x.shape[0]
    
    @property
    def num_bonds(self) -> int:
        return self.edge_index.shape[1]
    
    @property
    def num_descriptors(self) -> int:
        return self.descriptors.shape[0]
    
    
class OligomerBuilder:
    def __init__(self):
        self._rxn_smarts = [
            '[*:1][Au].[*:2][Cu]>>[*:1][*:2]',
            '[*:1]=[Au].[*:2]=[Cu]>>[*:1]=[*:2]',
        ]
        self._reactions = [
            rdChemReactions.ReactionFromSmarts(s) for s in self._rxn_smarts
        ]
    
    def _label(self, smiles: str) -> Chem.Mol:
        labeled = smiles.replace('*', '[Au]', 1).replace('*', '[Cu]', 1)
        mol = Chem.MolFromSmiles(labeled)
        if mol is None:
            raise ValueError(f'Invalid smiles: {smiles}')
        return mol
    
    def _attach(self, chain: Chem.Mol, mono: Chem.Mol) -> str:
        for rxn in self._reactions:
            prods = rxn.RunReactants((chain, mono))
            if prods:
                smi = Chem.MolToSmiles(prods[0][0])
                return smi.replace('[Au]', '*').replace('[Cu]', '*')
        raise ValueError(f'No product found for {Chem.MolToSmiles(mono)}')
    
    def _build(self, smiles: str, n: int) -> Chem.Mol:
        if n < 1:
            raise ValueError(f'n must be greater than 0, got {n}')
        current = smiles
        if current.count('*') != 2:
            return Chem.MolFromSmiles(smiles)

        for _ in range(n - 1):
            chain_mol = self._label(current)
            mono_mol = self._label(smiles)
            current = self._attach(chain_mol, mono_mol)
        
        mol = Chem.MolFromSmiles(current)
        return mol
    
    @staticmethod
    def get_oligomer(smiles: str, n_oligomer: int) -> Chem.Mol:
        return OligomerBuilder()._build(smiles, n_oligomer)
                