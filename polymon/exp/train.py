import logging
import os
import shutil
from glob import glob
from time import perf_counter
from typing import Dict, Optional, Tuple, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader

from polymon.exp.score import scaling_error
from polymon.exp.utils import EarlyStopping
from polymon.model.base import ModelWrapper


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
        model: ModelWrapper,
        lr: float,
        num_epochs: int,
        logger: logging.Logger,
        minmax_dict: Dict[str, Tuple[float, float]],
        device: torch.device = 'cuda',
        early_stopping_patience: int = 10,
        if_save: bool = False,
    ):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device
        self.model = model
        self.logger = logger
        self.if_save = if_save
    
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            save_dir=os.path.join(self.out_dir, 'ckpt'),
        )  

        self.minmax_dict = minmax_dict
        self.log_sigma = torch.nn.Parameter(
            torch.zeros(self.model.normalizer.init_params['mean'].shape[0]
            ))

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
        minmax_dict: Dict[str, Tuple[float, float]] = None,
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
            optimizer.zero_grad()
            loss = self.model(batch, F.huber_loss, self.device)
            loss.backward()
            optimizer.step()

        # Report progress
        val_metrics = self.eval(val_loader, label, self.minmax_dict)
        train_metrics = self.eval(train_loader, label, self.minmax_dict)
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
        test_loader: Optional[DataLoader] = None,
        label: str = 'Rg',
        minmax_dict: Dict[str, Tuple[float, float]] = None,
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
                self.minmax_dict,
            )

            # Early stopping
            self.early_stopping(-val_mae, self.model, ith_epoch)
            if self.early_stopping.early_stop:
                self.logger.info(f'Early stopping at epoch {ith_epoch}')
                break
        
        # Load the best model
        ckpts = glob(os.path.join(self.out_dir, 'ckpt', '*.pt'))
        ckpts.sort(key=os.path.getmtime)
        save_path = ckpts[-1]
        self.model = ModelWrapper.from_file(save_path)
        self.logger.info(f'Load best model from {save_path}')

        # Evaluate the best model on the test set
        if test_loader is None:
            test_loader = val_loader
            self.logger.info('No test set provided, using validation set as test set')
            
        test_metrics = self.eval(test_loader, label, self.minmax_dict)
        for metric_name, metric_value in test_metrics.items():
            self.logger.info(f'{metric_name}: {metric_value:.4f}')

        end_time = perf_counter()
        self.logger.info(f'Time taken: {end_time - start_time:.2f} seconds')
        self.logger.info(f'--------------------------------')
        
        test_err = test_metrics['scaling_error']
        test_mae_error = test_metrics['mae']
        return test_err, test_mae_error

    @torch.no_grad()
    def eval(
        self,
        loader: DataLoader,
        label: str,
        minmax_dict: Dict[str, Tuple[float, float]] = None,
    ) -> Dict[str, float]:
        """Evaluate the model on the given data loader.
        
        Args:
            loader (DataLoader): The data loader.
            metrics (List[Literal['mae', 'r2']]): The metrics to evaluate. If
                `None`, all metrics will be evaluated.
        
        Returns:
            `Dict[str, float]`: The metrics and their values.
        """
        self.model.eval()
        self.model.to(self.device)
        
        # Evaluate the model
        y_trues = []
        y_preds = []
        # sources = []
        for i, batch in enumerate(loader):
            batch = batch.to(self.device)
            y_pred = self.model.model(batch)
            y_pred = self.model.normalizer.inverse(y_pred)
            y_pred = y_pred + getattr(batch, 'estimated_y', 0)
            y_true = batch.y.detach() + getattr(batch, 'estimated_y', 0)
            y_trues.extend(y_true.cpu().numpy())
            y_preds.extend(y_pred.detach().cpu().numpy())
            # if getattr(batch, 'source', None) is not None:
            #     sources.extend(batch.source.cpu().numpy())
        
        # if len(sources) > 0:
        #     internal_mask = np.array(sources) == 1
        # else:
        #     internal_mask = np.ones(len(y_trues), dtype=bool)

        # if internal_mask.sum() == 0:
        #     return {'mae': np.nan, 'r2': np.nan, 'scaling_error': np.nan}

        y_trues = np.array(y_trues) # [internal_mask]
        y_preds = np.array(y_preds) # [internal_mask]
        if np.isnan(y_trues).any() or np.isnan(y_preds).any():
            return {'mae': np.nan, 'r2': np.nan, 'scaling_error': np.nan}
        metrics = {}
        metrics['mae'] = mean_absolute_error(y_trues, y_preds)
        metrics['r2'] = r2_score(y_trues, y_preds)
        metrics['scaling_error'] = scaling_error(y_trues, y_preds, label, self.minmax_dict)
        return metrics
    
    @staticmethod
    def masked_huber_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        delta: float = 1.0,
        mask: Optional[torch.Tensor] = None,
        task_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Masked Huber loss.
        
        Args:
            pred (torch.Tensor): The predicted values.
            target (torch.Tensor): The target values.
            mask (torch.Tensor | None): The mask.
            delta (float | torch.Tensor): The delta.
            task_weights (torch.Tensor | None): The task weights.
        """
        if mask is None:
            mask = ~torch.isnan(target)
            
        target_filled = torch.where(mask, target, pred.detach())
        
        # if isinstance(delta, torch.Tensor) and delta.ndim == 1:
        #     delta = delta.view(-1, 1).to(pred.device)
            
        loss_per_task = F.huber_loss(
            pred, 
            target_filled, 
            delta = delta,
            reduction='none'
        )
        loss_per_task = loss_per_task * mask.float()
        
        if task_weights is not None:
            loss_per_task = loss_per_task * task_weights.view(1, -1).to(pred.device)
            
        denom = mask.float()
        if task_weights is not None:
            denom = denom * task_weights.view(1, -1).to(pred.device)
            
        denom = denom.sum(dim=0).clamp_min(1)
        losses = loss_per_task.sum(dim=0) / denom
        # losses[2] /= 10
        return losses
    
    def train_step_mt(
        self,
        ith_epoch: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        labels: List[str],
        minmax_dict: Dict[str, Tuple[float, float]] = None,
    ) -> float:
        """Train the model for one epoch.
        """
        epoch_digits = len(str(self.num_epochs))
        
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(self.device)
            losses = self.model(
                batch, 
                loss_fn = lambda pred, target: self.masked_huber_loss(
                    pred, 
                    target, 
                    delta = 1.0, 
                    mask = ~torch.isnan(target),
                    task_weights = None,
                ),
                device=self.device,
            )
            
            inv_var = torch.exp(-2 * self.log_sigma).to(self.device)
            loss = 0.5 * (losses * inv_var).sum() + self.log_sigma.sum()
            # loss = losses.mean()
            loss.backward()
            optimizer.step()
            
        val_metrics = self.eval_mt(val_loader, labels, self.minmax_dict)
        train_metrics = self.eval_mt(train_loader, labels, self.minmax_dict)
        tasks = ', '.join(
            f'{labels[i]} '
            f'MAE: {train_metrics[f"mae/{labels[i]}"]:.3f}/{val_metrics[f"mae/{labels[i]}"]:.3f} '
            f'R2: {train_metrics.get(f"r2/{labels[i]}", float("nan")):.3f}/{val_metrics.get(f"r2/{labels[i]}", float("nan")):.3f}'
            for i in range(len(labels))
        )
        self.logger.info(
            f'[{str(ith_epoch).zfill(epoch_digits)}/{self.num_epochs}]'
            f'[Loss: {loss.item():.2f}] '
            f'[Loss1: {train_metrics["losses"][0]:.3f}] '
            f'[Loss2: {train_metrics["losses"][1]:.3f}] '
            f'[Loss3: {train_metrics["losses"][2]:.3f}] '
            f'[Train MAE: {train_metrics["mae_mt"]:.3f}] '
            f'[Val MAE: {val_metrics["mae_mt"]:.3f}] '
            f'| [{tasks}]'
        )
        return val_metrics['mae_mt']
    
    def train_mt(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        labels: List[str] = ['Rg', 'Density', 'Bulk_modulus'],
        minmax_dict: Dict[str, Tuple[float, float]] = None,
    ):
        """Train the model.
        
        Args:
            train_loader (DataLoader): The training data loader.
            val_loader (DataLoader): The validation data loader.
            test_loader (DataLoader): The test data loader.
        """
        start_time = perf_counter()
        optimizer = self.build_optimizer()
        optimizer.add_param_group({'params': [self.log_sigma]})
        for ith_epoch in range(self.num_epochs):
            val_mae = self.train_step_mt(
                ith_epoch,
                train_loader,
                val_loader,
                optimizer,
                labels,
                self.minmax_dict,
            )
            
            self.early_stopping(-val_mae, self.model, ith_epoch)
            if self.early_stopping.early_stop:
                self.logger.info(f'Early stopping at epoch {ith_epoch}')
                break

        ckpts = glob(os.path.join(self.out_dir, 'ckpt', '*.pt'))
        ckpts.sort(key=os.path.getmtime)
        save_path = ckpts[-1]
        self.model = ModelWrapper.from_file(save_path)
        self.logger.info(f'Load best model from {save_path}')
        if not self.if_save:
            shutil.rmtree(os.path.join(self.out_dir, 'ckpt'))
        
        if test_loader is None:
            test_loader = val_loader
            self.logger.info('No test set provided, using validation set as test set')
            
        test_metrics = self.eval_mt(test_loader, labels, self.minmax_dict)
        self.logger.info(f'Test MAE MT: {test_metrics["mae_mt"]:.4f}')
        #self.logger.info(f'Test R2 MT: {test_metrics["r2_mt"]:.4f}')
        self.logger.info(f'Test Scaling Error MT: {test_metrics["scaling_error_mt"]:.4f}')
        self.logger.info(f'--------------------------------')
        for name in labels:
            self.logger.info(
                f'{name} MAE: {test_metrics[f"mae/{name}"]:.4f} '
                f'R2: {test_metrics[f"r2/{name}"]:.4f} '
                f'Scaling Error: {test_metrics[f"scaling_error/{name}"]:.4f} '
            )
            
        return test_metrics['scaling_error_mt'], test_metrics['mae_mt']
    
    @torch.no_grad()
    def eval_mt(
        self,
        loader: DataLoader,
        labels: List[str],
        minmax_dict: Dict[str, Tuple[float, float]] = None,
    ) -> Dict[str, float]:
        """Evaluate the model on the given data loader.
        """
        self.model.eval()
        self.model.to(self.device)
        y_trues = []
        y_preds = []
        y_preds_denorm = []
        y_trues_norm = []
        for i, batch in enumerate(loader):
            batch = batch.to(self.device)
            y_pred = self.model.model(batch)
            y_pred_denorm = self.model.normalizer.inverse(y_pred)
            y_pred_denorm = y_pred_denorm + getattr(batch, 'estimated_y', 0)
            y_true = batch.y.detach() + getattr(batch, 'estimated_y', 0)
            y_true_norm = self.model.normalizer(y_true)
            y_trues.extend(y_true.cpu().numpy())
            y_preds.extend(y_pred.detach().cpu().numpy())
            y_preds_denorm.extend(y_pred_denorm.detach().cpu().numpy())
            y_trues_norm.extend(y_true_norm.detach().cpu().numpy())
    
        y_trues = np.array(y_trues) # denormalized
        y_preds = np.array(y_preds) # normalized
        y_preds_denorm = np.array(y_preds_denorm) # denormalized
        y_trues_norm = np.array(y_trues_norm) # normalized
        mask = ~np.isnan(y_trues)
        if mask.sum() == 0:
            return {'mae_mt': np.nan, 'r2_mt': np.nan, 'scaling_error_mt': np.nan}
        
        maes, r2s, scaling_errors = [], [], []
        maes_denorm, r2s_denorm, scaling_errors_denorm = [], [], []
        for i, name in enumerate(labels):
            m = mask[:, i]
            if m.sum() == 0:
                maes.append(np.nan), r2s.append(np.nan), scaling_errors.append(np.nan); continue

            # normalized metrics    
            y_true_i = y_trues_norm[m, i]
            y_pred_i = y_preds[m, i]
            # denormalized metrics
            y_true_i_denorm = y_trues[m, i]
            y_pred_i_denorm = y_preds_denorm[m, i]
            maes.append(mean_absolute_error(y_true_i, y_pred_i))
            maes_denorm.append(mean_absolute_error(y_true_i_denorm, y_pred_i_denorm))
            try:
                #r2s.append(r2_score(y_true_i, y_pred_i))
                r2s_denorm.append(r2_score(y_true_i_denorm, y_pred_i_denorm))
            except ValueError:
                r2s_denorm.append(np.nan)
            if self.minmax_dict is not None and name in self.minmax_dict:
                scaling_errors.append(scaling_error(y_true_i, y_pred_i, name, self.minmax_dict))
                scaling_errors_denorm.append(scaling_error(y_true_i_denorm, y_pred_i_denorm, name, self.minmax_dict))
            else:
                scaling_errors.append(np.nan)
                scaling_errors_denorm.append(np.nan)
                
        mae_mt = np.nanmean(maes) if len(maes) else np.nan
        #r2_mt = np.nanmean(r2s) if len(r2s) else np.nan
        scaling_error_mt = np.nanmean(scaling_errors) if len(scaling_errors) else np.nan
        metrics = {
            'mae_mt': mae_mt,
            #'r2_mt': r2_mt,
            'scaling_error_mt': scaling_error_mt,
        }
        for i, name in enumerate(labels):
            metrics[f'mae/{name}'] = maes_denorm[i]
            metrics[f'r2/{name}'] = r2s_denorm[i]
            metrics[f'scaling_error/{name}'] = scaling_errors_denorm[i]
        losses = self.masked_huber_loss(torch.from_numpy(y_preds), torch.from_numpy(y_trues_norm), mask=torch.from_numpy(mask))
        metrics['losses'] = losses
        return metrics
        