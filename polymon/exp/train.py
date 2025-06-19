import logging
import os
from copy import deepcopy
from glob import glob
from time import perf_counter
from typing import Dict, List, Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.regression import MeanAbsoluteError, R2Score

from polymon.exp.utils import EMA, EarlyStopping, get_logger


class Trainer:
    """Trainer for the model.
    
    Args:
        out_dir (str): The directory to save the model and results.
        model (nn.Module): The model to train.
        lr (float): The learning rate.
        num_epochs (int): The number of epochs to train.
        num_classes (int): The number of classes (families) in the dataset.
        logger (logging.Logger): The logger. If not provided, a logger will be
            created in the `out_dir`.
        report_every (int): The number of steps to report the training progress.
            Default is 500.
        ema_decay (float): The decay rate for the EMA. If 0, EMA will not be
            used. Default is 0.
        device (torch.device): The device to train on. Default is `cuda`.
        early_stopping_patience (int): The number of epochs to wait before 
            stopping the training. Default is 10.
        class_weights (torch.Tensor): The class weights. If not provided, 
            class weights will not be used.
    """
    def __init__(
        self,
        out_dir: str,
        model: nn.Module,
        lr: float,
        num_epochs: int,
        logger: logging.Logger = None,
        report_every: int = 500,
        ema_decay: float = 0.0,
        device: torch.device = 'cuda',
        early_stopping_patience: int = 10,
    ):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.report_every = report_every
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device
        self.model = model
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
        step_digits = len(str(len(train_loader)))
        
        mae_score = MeanAbsoluteError().to(self.device)
        r2_score = R2Score().to(self.device)
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

            loss = F.mse_loss(y_pred, batch.y)
            mae_score.update(y_pred, batch.y)
            r2_score.update(y_pred, batch.y)
            loss.backward()
            optimizer.step()

            # Update EMA
            if self.ema is not None:
                self.ema.update_model_average(self.model_ema, self.model)

            # Report progress
            if (i + 1) % self.report_every == 0:
                val_metrics = self.eval(val_loader, ['mae', 'r2'])
                self.logger.info(
                    f'[{str(ith_epoch).zfill(epoch_digits)}/{self.num_epochs}]'
                    f'[{str(i + 1).zfill(step_digits)}/{len(train_loader)}] '
                    f'[Loss: {loss.item():.2f}]'
                    f'[Train MAE: {mae_score.compute():.3f}]'
                    f'[Train R2: {r2_score.compute():.3f}]'
                    f'[Dev MAE: {val_metrics["mae"]:.3f}]'
                    f'[Dev R2: {val_metrics["r2"]:.3f}]'
                )
        val_metrics = self.eval(val_loader, ['mae', 'r2'])
        return val_metrics['mae']

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
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
                optimizer
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
        test_metrics = self.eval(test_loader, None)
        for metric_name, metric_value in test_metrics.items():
            self.logger.info(f'{metric_name}: {metric_value:.3f}')

        end_time = perf_counter()
        self.logger.info(f'Time taken: {end_time - start_time:.2f} seconds')

    @torch.no_grad()
    def eval(
        self,
        loader: DataLoader,
        metrics: List[Literal['mae', 'r2']] = None
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
        
        metric_dict = {
            'mae': MeanAbsoluteError(),
            'r2': R2Score(),
        }
        if metrics is None:
            metrics = list(metric_dict.keys())
        metric_dict = {
            metric: metric_dict[metric].to(self.device) for metric in metrics
        }
        
        # Evaluate the model
        for i, batch in enumerate(loader):
            batch = batch.to(self.device)
            y_pred = model(batch)
            for metric_fn in metric_dict.values():
                metric_fn.update(y_pred, batch.y)
        return {
            name: metric_fn.compute() for name, metric_fn in metric_dict.items()
        }