from typing import Optional

import torch
from torch_geometric.data import Data


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