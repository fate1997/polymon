import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform, AddRandomWalkPE
from sklearn.ensemble import RandomForestRegressor

from polymon.data.polymer import Polymer
from polymon.setting import DEFAULT_TOPK_DESCRIPTORS


class Normalizer:
    def __init__(self, mean: float, std: float, eps: float = 1e-6):
        self.mean = mean
        self.std = std + eps
    
    @classmethod
    def from_loader(cls, loader: DataLoader) -> 'Normalizer':
        x = []
        for batch in loader:
            x.append(batch.y.numpy().ravel())
        x = np.concatenate(x, 0)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        return cls(mean, std)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std
    
    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


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