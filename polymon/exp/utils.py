import os
import random
from glob import glob
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from polymon.model.base import ModelWrapper


class EarlyStopping:
    """Early stopping for training.
    
    Args:
        patience (int): The number of epochs to wait before stopping.
        save_dir (str): The directory to save the model.
        max_ckpt (int): The maximum number of checkpoints to save.
    """
    def __init__(
        self,
        patience: int,
        save_dir: str,
        max_ckpt: int = 5,
    ):
        os.makedirs(save_dir, exist_ok=True)
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        self.patience = patience
        self.save_dir = save_dir
        self.max_ckpt = max_ckpt

    def __call__(self, val_score: float, model: ModelWrapper, epoch: int):
        """Early stopping for training.
        
        Args:
            val_score (float): The current validation score.
            model (nn.Module): The current model.
            epoch (int): The current epoch.
        """
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model, epoch) 
        elif val_score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.save_checkpoint(model, epoch)
            self.counter = 0

    def save_checkpoint(self, model: ModelWrapper, epoch: int):
        """Save the model checkpoint.
        
        Args:
            model (ModelWrapper): The model.
            epoch (int): The current epoch.
        """
        save_path = os.path.join(self.save_dir, f'epoch_{epoch}.pt')
        model.write(save_path)
        if len(os.listdir(self.save_dir)) > self.max_ckpt:
            ckpts = glob(os.path.join(self.save_dir, '*.pt'))
            ckpts.sort(key=os.path.getmtime)
            os.remove(ckpts[0])


def seed_everything(seed: int):
    """Seed everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loader2numpy(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    x = []
    y = []
    for batch in loader:
        x.append(batch.descriptors.numpy())
        y.append(batch.y.numpy().ravel())
    return np.concatenate(x, 0), np.concatenate(y, 0)


def predict_batch(
    sklearn_model, 
    X: np.ndarray, 
    batch_size: int=1024,
    show_progress: bool=True,
) -> np.ndarray:
    y_pred = []
    if show_progress:
        iterator = tqdm(range(0, len(X), batch_size))
    else:
        iterator = range(0, len(X), batch_size)
    for i in iterator:
        X_batch = X[i:i+batch_size]
        y_pred.append(sklearn_model.predict(X_batch))
    return np.concatenate(y_pred, 0)