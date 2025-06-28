import numpy as np
from torch_geometric.loader import DataLoader


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