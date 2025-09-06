import numpy as np
import torch
import networkx as nx
from rdkit import Chem
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform, AddRandomWalkPE
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict, OrderedDict
from itertools import chain, combinations
from typing import Dict

from polymon.data.polymer import Polymer
from polymon.setting import DEFAULT_TOPK_DESCRIPTORS


class Normalizer:
    def __init__(self, mean: float, std: float, eps: float = 1e-6):
        self.mean = mean
        self.std = std + eps
    
    @property
    def init_params(self) -> Dict[str, float]:
        return {
            'mean': self.mean,
            'std': self.std,
        }
    
    @classmethod
    def from_loader(cls, loader: DataLoader) -> 'Normalizer':
        x = []
        for batch in loader:
            x.append(batch.y.cpu().numpy().ravel())
        x = np.concatenate(x, 0)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        return cls(mean, std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class LogNormalizer:
    def __init__(
        self,
        eps: float = 1e-10,
    ):
        self.eps = eps
    
    @property
    def init_params(self) -> Dict[str, float]:
        return {
            'eps': self.eps,
        }
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + self.eps)
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x) - self.eps


class DescriptorSelector(BaseTransform):
    def __init__(
        self, 
        ids: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ):
        self.ids = ids
        self.mean = mean
        self.std = std
    
    @classmethod
    def from_rf(
        cls, 
        train_loader: DataLoader, 
        top_k: int = DEFAULT_TOPK_DESCRIPTORS,
    ) -> 'DescriptorSelector':
        X = []
        y = []
        for batch in train_loader:
            X.append(batch.descriptors.detach().cpu().numpy())
            y.append(batch.y.detach().cpu().numpy())
        X = np.concatenate(X, 0)
        y = np.concatenate(y, 0).ravel()
        rf = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            criterion='absolute_error',
            n_jobs=-1,
        )
        rf.fit(X, y)
        ids = np.argsort(rf.feature_importances_)[::-1]
        selected_ids = torch.from_numpy(ids[:top_k].copy())
        mean = torch.from_numpy(X[:, selected_ids].mean(axis=0))
        std = torch.from_numpy(X[:, selected_ids].std(axis=0))
        return cls(selected_ids, mean, std)
    
    def forward(self, data: Polymer) -> Polymer:
        if self.ids is None:
            self.ids = torch.arange(data.descriptors.shape[1])
        ids = self.ids.to(data.descriptors.device)
        data.descriptors = (data.descriptors[:, ids] - self.mean) / self.std
        return data


class LineEvoTransform(BaseTransform):
    def __init__(self, depth: int = 2):
        self.depth = depth
    
    def forward(self, data: Polymer) -> Polymer:
        edges = torch.LongTensor(
            np.array(nx.from_edgelist(data.edge_index.T.tolist()).edges)
        )
            
        num_nodes = data.x.shape[0]
        isolated_nodes = set(range(num_nodes)).difference(set(edges.flatten().tolist()))
        edges = torch.cat(
            [edges, torch.LongTensor([[i, i] for i in isolated_nodes])], dim=0
        ).to(torch.long)
        
        setattr(data, f'edges_{0}', edges)
        
        for i in range(self.depth):
            
            num_nodes = edges.shape[0]
            edges = self.evolve_edges_generater(edges)

            # create edges for isolated nodes
            isolated_nodes = set(range(num_nodes)).difference(set(edges.flatten().tolist()))
            edges = torch.cat(
                [edges, torch.LongTensor([[i, i] for i in isolated_nodes]).to(edges.device)], 
                dim=0
            )
            
            setattr(data, f'edges_{i+1}', edges)
        return data
    
    @staticmethod
    def evolve_edges_generater(edges):
        l = edges[:, 0].tolist()+ edges[:, 1].tolist()
        tally = defaultdict(list)
        for i, item in enumerate(l):
            tally[item].append(i if i < len(l)//2 else i - len(l)//2)
        
        output = []
        for _, locs in tally.items():
            if len(locs) > 1:
                output.append(list(combinations(locs, 2)))
        
        return torch.LongTensor(list(chain(*output))).to(edges.device)


class RgEstimator(BaseTransform):
    BOND_LENGTHS = {
        (6, 6, Chem.BondType.SINGLE): 0.154,   # C-C single
        (6, 6, Chem.BondType.DOUBLE): 0.134,   # C=C double
        (6, 6, Chem.BondType.AROMATIC): 0.140, # Aromatic C-C
        (6, 8, Chem.BondType.SINGLE): 0.143,   # C-O
        (6, 7, Chem.BondType.SINGLE): 0.147,   # C-N
        (7, 7, Chem.BondType.SINGLE): 0.146,   # N-N
        (8, 8, Chem.BondType.SINGLE): 0.143,   # O-O
    }
    
    def __init__(
        self, 
        N: int = 500, 
        C_inf: float = 6.7, 
        solvent: str = "theta",
    ):
        self.N = N
        self.C_inf = C_inf
        self.solvent = solvent
    
    def forward(self, data: Polymer) -> Polymer:
        estimated_rg = self.radius_of_gyration(
            data.smiles, self.N, self.C_inf, self.solvent
        )
        data.estimated_y = torch.tensor([[estimated_rg]]).to(data.x.device)
        if getattr(data, 'y', None) is not None:
            data.y = data.y - estimated_rg
        return data

    def estimated_y(self, smiles: str) -> float:
        return self.radius_of_gyration(smiles, self.N, self.C_inf, self.solvent)
    
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
                length = self.BOND_LENGTHS.get(key, 0.154)  # fallback to C-C
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

    def radius_of_gyration(self, smiles, N=1000, C_inf=6.7, solvent="theta"):
        """
        Estimate Rg from a monomer SMILES without using conformers.
        - N: degree of polymerization
        - C_inf: characteristic ratio (defaults ~6.7 for flexible chains)
        - solvent: 'theta' or 'good'
        """
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