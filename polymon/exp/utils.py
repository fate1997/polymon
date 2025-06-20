import logging
import os
import random
import sys
from glob import glob
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def get_logger(out_dir: str, name: str) -> logging.Logger:
    """Get a logger for the training process.
    
    Args:
        out_dir (str): The directory to save the log file.
        name (str): The name of the logger.
    
    Returns: 
        `logging.Logger`: The logger.
    """
    path = os.path.join(out_dir, f'{name}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(path), 
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(name)  


class EMA():
    """Exponential Moving Average (EMA) for model weights.
    
    Args:
        beta (float): The decay rate for the EMA.
    """
    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta

    def update_model_average(
        self, 
        ma_model: nn.Module, 
        current_model: nn.Module
    ):
        """Update the model average.
        
        Args:
            ma_model (nn.Module): The model to update.
            current_model (nn.Module): The current model.
        """
        for current_params, ma_params in zip(current_model.parameters(), 
                                             ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(
        self, 
        old: torch.Tensor, 
        new: torch.Tensor
    ) -> torch.Tensor:
        """Update the average.
        
        Args:
            old (torch.Tensor): The old weight.
            new (torch.Tensor): The new weight.

        Returns:
            `torch.Tensor`: The updated weight.
        """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


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

    def __call__(self, val_score: float, model: nn.Module, epoch: int):
        """Early stopping for training.
        
        Args:
            val_score (float): The current validation score.
            model (nn.Module): The current model.
            epoch (int): The current epoch.
        """
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model, epoch) 
        elif val_score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.save_checkpoint(model, epoch)
            self.counter = 0

    def save_checkpoint(self, model: nn.Module, epoch: int):
        """Save the model checkpoint.
        
        Args:
            model (nn.Module): The model.
            epoch (int): The current epoch.
        """
        save_path = os.path.join(self.save_dir, f'epoch_{epoch}.pt')
        torch.save(model.state_dict(), save_path)
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