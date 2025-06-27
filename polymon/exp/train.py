import logging
import os
from copy import deepcopy
from glob import glob
from time import perf_counter
from typing import Dict, List, Literal

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score
from torch import nn
from torch.utils.data import DataLoader

from polymon.exp.score import scaling_error
from polymon.exp.utils import EMA, EarlyStopping, Normalizer, get_logger


class Trainer:
    """Trainer for the model.
    
    Args:
        out_dir (str): The directory to save the model and results.
        model (nn.Module): The model to train.
        lr (float): The learning rate.
        num_epochs (int): The number of epochs to train.
        logger (logging.Logger): The logger. If not provided, a logger will be
            created in the `out_dir`.
        ema_decay (float): The decay rate for the EMA. If 0, EMA will not be
            used. Default is 0.
        device (torch.device): The device to train on. Default is `cuda`.
        early_stopping_patience (int): The number of epochs to wait before 
            stopping the training. Default is 10.
    """
    def __init__(
        self,
        out_dir: str,
        model: nn.Module,
        lr: float,
        num_epochs: int,
        normalizer: Normalizer,
        logger: logging.Logger = None,
        ema_decay: float = 0.0,
        device: torch.device = 'cuda',
        early_stopping_patience: int = 10,
    ):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device
        self.model = model
        self.normalizer = normalizer
        self.logger = logger if logger is not None else \
            get_logger(out_dir, 'training')
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            save_dir=os.path.join(self.out_dir, 'ckpt'),
        )
        
        # EMA
        self.ema = EMA(ema_decay) if ema_decay > 0 else None
        self.model_ema = deepcopy(self.model) if self.ema is not None else None      

    def build_optimizer(self) -> torch.optim.Optimizer:
        """Build the optimizer.
        
        Returns:
            `torch.optim.Optimizer`: The optimizer.
        """
        return torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr,
            weight_decay=1e-12
        )

    def train_step(
        self,
        ith_epoch: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        label: str,
    ) -> float:
        """Train the model for one epoch.
        
        Args:
            ith_epoch (int): The current epoch.
            train_loader (DataLoader): The training data loader.
            val_loader (DataLoader): The validation data loader.
            optimizer (torch.optim.Optimizer): The optimizer.

        Returns:
            `float`: The F1 score on the validation set.
        """
        epoch_digits = len(str(self.num_epochs))
        
        for i, batch in enumerate(train_loader):
            # Set model to training mode
            self.model.train()
            self.model.to(self.device)
            if self.ema is not None:
                self.model_ema.train()
                self.model_ema.to(self.device)

            # Forward and backward pass
            optimizer.zero_grad()
            batch = batch.to(self.device)
            y_pred = self.model(batch)
            y_true_transformed = self.normalizer(batch.y)

            loss = F.huber_loss(y_pred, y_true_transformed)
            loss.backward()
            optimizer.step()

            # Update EMA
            if self.ema is not None:
                self.ema.update_model_average(self.model_ema, self.model)

        # Report progress
        val_metrics = self.eval(val_loader, label)
        train_metrics = self.eval(train_loader, label)
        self.logger.info(
            f'[{str(ith_epoch).zfill(epoch_digits)}/{self.num_epochs}]'
            f'[Loss: {loss.item():.2f}]'
            f'[Train MAE: {train_metrics["mae"]:.3f}]'
            f'[Train R2: {train_metrics["r2"]:.3f}]'
            f'[Val MAE: {val_metrics["mae"]:.3f}]'
            f'[Val R2: {val_metrics["r2"]:.3f}]'
        )
        return val_metrics['mae']

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader = None,
        label: str = 'Rg',
    ):
        """Train the model.
        
        Args:
            train_loader (DataLoader): The training data loader.
            val_loader (DataLoader): The validation data loader.
            test_loader (DataLoader): The test data loader.
        """
        start_time = perf_counter()
        optimizer = self.build_optimizer()
        for ith_epoch in range(self.num_epochs):
            val_mae = self.train_step(
                ith_epoch, 
                train_loader, 
                val_loader, 
                optimizer,
                label,
            )

            # Early stopping
            model = self.model_ema if self.ema is not None else self.model
            self.early_stopping(-val_mae, model, ith_epoch)
            if self.early_stopping.early_stop:
                self.logger.info(f'Early stopping at epoch {ith_epoch}')
                break
        
        # Load the best model
        ckpts = glob(os.path.join(self.out_dir, 'ckpt', '*.pt'))
        ckpts.sort(key=os.path.getmtime)
        save_path = ckpts[-1]
        if self.ema is not None:
            self.model_ema.load_state_dict(torch.load(save_path))
        else:
            self.model.load_state_dict(torch.load(save_path))
        self.logger.info(f'Load best model from {save_path}')

        # Evaluate the best model on the test set
        if test_loader is not None:
            test_metrics = self.eval(test_loader, label)
            for metric_name, metric_value in test_metrics.items():
                self.logger.info(f'{metric_name}: {metric_value:.3f}')
            self.logger.info(f'Test scaling error: {test_metrics["scaling_error"]:.4f}')

        end_time = perf_counter()
        self.logger.info(f'Time taken: {end_time - start_time:.2f} seconds')
        self.logger.info(f'--------------------------------')
        
        # Save the model and normalizer
        torch.save(
            {
                'model': self.model.state_dict(),
                'normalizer': self.normalizer,
            },
            os.path.join(self.out_dir, 'final_model.pt'),
        )
        
        return test_metrics['scaling_error'] if test_loader is not None else None

    @torch.no_grad()
    def eval(
        self,
        loader: DataLoader,
        label: str,
    ) -> Dict[str, float]:
        """Evaluate the model on the given data loader.
        
        Args:
            loader (DataLoader): The data loader.
            metrics (List[Literal['mae', 'r2']]): The metrics to evaluate. If
                `None`, all metrics will be evaluated.
        
        Returns:
            `Dict[str, float]`: The metrics and their values.
        """
        model = self.model_ema if self.ema is not None else self.model
        model.eval()
        model.to(self.device)
        
        # Evaluate the model
        y_trues = []
        y_preds = []
        for i, batch in enumerate(loader):
            batch = batch.to(self.device)
            y_pred = model(batch)
            y_pred = self.normalizer.inverse(y_pred)
            y_trues.extend(batch.y.detach().cpu().numpy())
            y_preds.extend(y_pred.detach().cpu().numpy())
        y_trues = np.array(y_trues)
        y_preds = np.array(y_preds)
        metrics = {}
        metrics['mae'] = mean_absolute_error(y_trues, y_preds)
        metrics['r2'] = r2_score(y_trues, y_preds)
        metrics['scaling_error'] = scaling_error(y_trues, y_preds, label)
        return metrics