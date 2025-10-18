from collections import OrderedDict, defaultdict
from itertools import chain, combinations
from typing import Dict

import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import AddRandomWalkPE, BaseTransform

from polymon.data._dmpnn_transform import DMPNNTransform
from polymon.data.polymer import Polymer
from polymon.setting import DEFAULT_TOPK_DESCRIPTORS


class Normalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
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
            x.append(batch.y.cpu().numpy())
        x = np.concatenate(x, 0)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        mean = torch.from_numpy(mean)
        std = torch.from_numpy(std)
        return cls(mean, std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)


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