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
from torch_geometric.data import Batch

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
        
        # for different weighting strategies
        self.mt_strategy = 'none'
        self.dwa_T = 2.0
        self.gradnorm_alpha = 0.5
        self.gradnorm_lr = 0.025
        self._K = None
        self._task_weights = None
        self._loss_hist = []

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
    
    # @staticmethod
    # def masked_huber_loss(
    #     pred: torch.Tensor,
    #     target: torch.Tensor,
    #     delta: float = 1.0,
    #     mask: Optional[torch.Tensor] = None,
    #     task_weights: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     """Masked Huber loss.
        
    #     Args:
    #         pred (torch.Tensor): The predicted values.
    #         target (torch.Tensor): The target values.
    #         mask (torch.Tensor | None): The mask.
    #         delta (float | torch.Tensor): The delta.
    #         task_weights (torch.Tensor | None): The task weights.
    #     """
    #     if mask is None:
    #         mask = ~torch.isnan(target)
            
    #     target_filled = torch.where(mask, target, pred.detach())
        
    #     # if isinstance(delta, torch.Tensor) and delta.ndim == 1:
    #     #     delta = delta.view(-1, 1).to(pred.device)
            
    #     loss_per_task = F.huber_loss(
    #         pred, 
    #         target_filled, 
    #         delta = delta,
    #         reduction='none'
    #     )
    #     loss_per_task = loss_per_task * mask.float()
        
    #     if task_weights is not None:
    #         loss_per_task = loss_per_task * task_weights.view(1, -1).to(pred.device)
            
    #     denom = mask.float()
    #     if task_weights is not None:
    #         denom = denom * task_weights.view(1, -1).to(pred.device)
            
    #     denom = denom.sum(dim=0).clamp_min(1)
    #     losses = loss_per_task.sum(dim=0) / denom
    #     # losses[2] /= 10
    #     return losses

    @staticmethod
    def heteroscedastic_gaussian_nll_masked(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        task_weights: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
        task_log_sigma: Optional[torch.Tensor] = None,
        include_const: bool = True,
    ) -> torch.Tensor:
        """
        Masked heteroscedastic Gaussian NLL.
        pred: (B, 2T) packed as [mean | log_var]
        target: (B, T)
        Returns: per-task losses (T,)
        """
        import math
        pred = pred.float()
        target = target.float()
        device = pred.device

        B, twoT = pred.shape
        T = target.shape[-1]
        assert twoT == 2 * T, f"Predictions should have shape (B, 2*T) but got {pred.shape}, T = {T}"

        mean = pred[:, :T]
        log_var = pred[:, T:]

        if mask is None:
            mask = ~torch.isnan(target)
        mask = mask.to(device=device, dtype=torch.bool)

        mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))
        raw_log_var = torch.where(torch.isfinite(log_var), log_var, torch.zeros_like(log_var))
        # log_var = raw_log_var.clamp_(math.log(1e-3), math.log(1e3))
        var = F.softplus(raw_log_var)+1e-3
        var = torch.clamp(var, max=1e3)
        log_var = torch.log(var)
        if task_log_sigma is not None:
            s = task_log_sigma.view(1, -1).to(device)  # unconstrained parameter
        else:
            s = torch.zeros(1, T, device=device)
        total_log_var = (log_var + s).clamp_(math.log(1e-3), math.log(1e3))

        inv_total_var = torch.exp(-total_log_var)
        target_filled = torch.where(mask, target, mean.detach())
        resid2 = (target_filled - mean) ** 2

        nll = 0.5 * (resid2 * inv_total_var + total_log_var)
        if include_const:
            nll = nll + 0.5 * math.log(2 * math.pi)
        nll = nll * mask.float()

        denom = mask.float().sum(dim=0).clamp_min(1.0)
        per_task = nll.sum(dim=0) / denom
        per_task = torch.where(torch.isfinite(per_task), per_task, torch.zeros_like(per_task))
        per_task = per_task + 1e-3 * torch.exp(total_log_var).mean(0)
        return per_task
        
        #return torch.where(torch.isfinite(per_task), per_task, torch.zeros_like(per_task))
        # s = task_log_sigma.view(1, -1).to(device)
        # target_filled = torch.where(mask, target, mean.detach())
        
        # var = torch.exp(log_var).clamp_min(eps)
        # inv_var = 1.0 / var

        # loss_per_elem = 0.5 * ((target_filled - mean) ** 2 * inv_var + log_var)
        # loss_per_elem = loss_per_elem * mask.float()

        # if task_weights is not None:
        #     tw = task_weights.view(1, -1).to(device).float()
        #     loss_per_elem = loss_per_elem * tw

        # denom = mask.float()
        # if task_weights is not None:
        #     denom = denom * tw
        # denom = denom.sum(dim=0).clamp_min(1.0)

        # losses = loss_per_elem.sum(dim=0) / denom
        # losses = torch.where(torch.isfinite(losses), losses, torch.zeros_like(losses))
        # return losses
    
    @staticmethod
    def masked_huber_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        delta: float = 1.0,
        mask: Optional[torch.Tensor] = None,
        task_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Masked Huber loss that is robust to NaNs in both target and pred.
        Shape: (B, T)
        Returns: per-task losses (T,)
        """
        # dtypes/devices
        pred = pred.float()
        target = target.float()
        device = pred.device
        
        # B, twoT = pred.shape
        # T = target.shape[-1]
        # assert twoT == 2*T, f"Predictions should have shape (B, 2*T) but got {pred.shape}, T = {T}"

        # mean = pred[:, :T]
        # log_var = pred[:, T:]
        # mask: True where label is valid
        if mask is None:
            mask = ~torch.isnan(target)
        mask = mask.to(device=device, dtype=torch.bool)

        # mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))
        # log_var = torch.where(torch.isfinite(log_var), log_var, torch.zeros_like(log_var))
        # If model ever spits NaNs/Infs, zero them out (don’t let them poison the loss)
        pred = torch.where(torch.isfinite(pred), pred, torch.zeros_like(pred))

        # Fill missing targets with the (detached) prediction so Huber doesn't see NaNs
        target_filled = torch.where(mask, target, pred.detach())
        #target_filled = torch.where(mask, target, mean.detach())

        # var = torch.exp(log_var).clamp_min(1e-6)
        # inv_var = 1.0 / var

        # Compute elementwise huber; NOTE: for older PyTorch versions this kwarg is "beta" not "delta"
        loss_per_elem = F.huber_loss(
            pred,
            target_filled,
            delta=delta,
            reduction='none',
        )
        #loss_per_elem = 0.5 * (inv_var * (target_filled - mean)**2 + log_var)

        # Zero out invalid elements
        loss_per_elem = loss_per_elem * mask.float()

        if task_weights is not None:
            tw = task_weights.view(1, -1).to(device).float()
            loss_per_elem = loss_per_elem * tw

        # Denominator = count of valid (and weighted) labels per task
        denom = mask.float()
        if task_weights is not None:
            denom = denom * tw
        denom = denom.sum(dim=0).clamp_min(1.0)  # avoid /0

        # Sum over batch, normalize per task
        losses = loss_per_elem.sum(dim=0) / denom

        # Final guard: replace any residual non-finite with 0 so training can proceed
        losses = torch.where(torch.isfinite(losses), losses, torch.zeros_like(losses))
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
        
        first_batch = True
        running_epoch_loss = None
        init_losses = None
        if self.mt_strategy == 'gradnorm':
            self._ensure_task_state(len(labels), self.device)
        
        self.model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(self.device)
            losses = self.model(
                batch, 
                loss_fn = lambda pred, target: self.heteroscedastic_gaussian_nll_masked(
                    pred, 
                    target, 
                    task_log_sigma = self.log_sigma,
                    include_const = True,
                    #delta = 1.0, 
                    mask = ~torch.isnan(target),
                    #task_weights = None,
                ),
                device=self.device,
            )
            
            if first_batch:
                K = losses.numel()
                self._ensure_task_state(K, self.device)
                if self.mt_strategy == 'gradnorm':
                    init_losses = losses.detach().clone()
                first_batch = False
            
            running_epoch_loss = losses.detach() if running_epoch_loss is None else running_epoch_loss + losses.detach()
            
            if self.mt_strategy == 'uncertainty':
                inv_var = torch.exp(-2 * self.log_sigma).to(self.device)
                inv_var = torch.where(torch.isfinite(inv_var), inv_var, torch.ones_like(inv_var))
                loss = 0.5 * (losses * inv_var).sum() + self.log_sigma.sum()
                if not torch.isfinite(loss):
                    # Optional: log and continue instead of poisoning the optimizer step
                    self.logger.warning("Non-finite loss detected; skipping optimizer step this batch.")
                    continue
                loss.backward()
                optimizer.step()
                
            elif self.mt_strategy == 'pcgrad':
                per_task_scalars = [losses[k] for k in range(len(losses))]
                self._pcgrad_step(per_task_scalars, optimizer)
                loss = torch.sum(losses)
                
            elif self.mt_strategy == 'gradnorm':
                loss = self._gradnorm_update_and_step(losses, init_losses, optimizer)
                
            elif self.mt_strategy == 'dwa':
                alpha = self._dwa_weights().to(self.device)
                total = (alpha * losses).sum()
                total.backward()
                optimizer.step()
                self._task_weights = alpha.detach()
                loss = total
            
            else:
                loss = losses.mean()
                loss.backward()
                optimizer.step()
                
        if self.mt_strategy == 'dwa':
            epoch_avg = (running_epoch_loss / max(1, len(train_loader))).detach()
            self._loss_hist.append(epoch_avg)
            
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
            # f'[Loss1: {train_metrics["losses"][0]:.3f}] '
            # f'[Loss2: {train_metrics["losses"][1]:.3f}] '
            # f'[Loss3: {train_metrics["losses"][2]:.3f}] '
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
        if self.mt_strategy == 'uncertainty':
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
            y_raw = self.model.model(batch)  # <<< changed name
            y_raw = torch.where(torch.isfinite(y_raw), y_raw, torch.zeros_like(y_raw))
            # figure out number of tasks from targets
            num_tasks = batch.y.shape[-1]
            # If heteroscedastic head: take only mean part
            if y_raw.shape[-1] == 2 * num_tasks:      # <<< handle [mean | log_var]
                y_pred = y_raw[:, :num_tasks]         # (B, T) means
                # Optional: if you want to inspect variances/logvars:
                # y_log_var = y_raw[:, num_tasks:]
            else:
                y_pred = y_raw                        # (B, T) classic case
            # y_pred = self.model.model(batch)
            # y_pred = torch.where(torch.isfinite(y_pred), y_pred, torch.zeros_like(y_pred))
            y_pred_denorm = self.model.normalizer.inverse(y_pred)
            y_pred_denorm = y_pred_denorm + getattr(batch, 'estimated_y', 0)
            y_true = batch.y.detach() + getattr(batch, 'estimated_y', 0)
            y_true_norm = self.model.normalizer(y_true)
            y_true_norm = torch.where(torch.isfinite(y_true_norm), y_true_norm, torch.zeros_like(y_true_norm))
            
            y_trues.extend(y_true.cpu().numpy())
            y_preds.extend(y_pred.detach().cpu().numpy())
            y_preds_denorm.extend(y_pred_denorm.detach().cpu().numpy())
            y_trues_norm.extend(y_true_norm.detach().cpu().numpy())
    
        y_trues = np.array(y_trues) # denormalized
        y_preds = np.array(y_preds) # normalized
        y_preds_denorm = np.array(y_preds_denorm) # denormalized
        y_trues_norm = np.array(y_trues_norm) # normalized
        
        label_mask = ~np.isnan(y_trues)
        pred_finite_mask = np.isfinite(y_preds_denorm) & np.isfinite(y_trues)
        mask = label_mask & pred_finite_mask
        
        maes, r2s, scaling_errors = [], [], []
        maes_denorm, r2s_denorm, scaling_errors_denorm = [], [], []
        if mask.sum() == 0:
            return {'mae_mt': np.nan, 'r2_mt': np.nan, 'scaling_error_mt': np.nan}

        for i, name in enumerate(labels):
            m = mask[:, i]
            if m.sum() == 0:
                maes.append(np.nan); maes_denorm.append(np.nan); r2s_denorm.append(np.nan)
                scaling_errors.append(np.nan); scaling_errors_denorm.append(np.nan)
                continue

            yt_norm_i = y_trues_norm[m, i]
            yp_norm_i = y_preds[m, i]
            yt_i = y_trues[m, i]
            yp_i = y_preds_denorm[m, i]

            # Extra guards
            fmask_norm = np.isfinite(yt_norm_i) & np.isfinite(yp_norm_i)
            fmask_denorm = np.isfinite(yt_i) & np.isfinite(yp_i)

            yt_norm_i, yp_norm_i = yt_norm_i[fmask_norm], yp_norm_i[fmask_norm]
            yt_i, yp_i = yt_i[fmask_denorm], yp_i[fmask_denorm]

            if yt_norm_i.size == 0 or yt_i.size == 0:
                maes.append(np.nan); maes_denorm.append(np.nan); r2s_denorm.append(np.nan)
                scaling_errors.append(np.nan); scaling_errors_denorm.append(np.nan)
                continue

            maes.append(mean_absolute_error(yt_norm_i, yp_norm_i))
            maes_denorm.append(mean_absolute_error(yt_i, yp_i))
            try:
                r2s_denorm.append(r2_score(yt_i, yp_i))
            except ValueError:
                r2s_denorm.append(np.nan)

            if self.minmax_dict is not None and name in self.minmax_dict:
                scaling_errors.append(scaling_error(yt_norm_i, yp_norm_i, name, self.minmax_dict))
                scaling_errors_denorm.append(scaling_error(yt_i, yp_i, name, self.minmax_dict))
            else:
                scaling_errors.append(np.nan)
                scaling_errors_denorm.append(np.nan)
            # if m_norm.sum() == 0 or m_denorm.sum() == 0:
            #     maes.append(np.nan), r2s.append(np.nan), scaling_errors.append(np.nan); continue

            # # normalized metrics    
            # y_true_i = y_trues_norm[m, i]
            # y_pred_i = y_preds[m, i]
            # # denormalized metrics
            # y_true_i_denorm = y_trues[m, i]
            # y_pred_i_denorm = y_preds_denorm[m, i]
            # maes.append(mean_absolute_error(y_true_i, y_pred_i))
            # maes_denorm.append(mean_absolute_error(y_true_i_denorm, y_pred_i_denorm))
            # try:
            #     #r2s.append(r2_score(y_true_i, y_pred_i))
            #     r2s_denorm.append(r2_score(y_true_i_denorm, y_pred_i_denorm))
            # except ValueError:
            #     r2s_denorm.append(np.nan)
            # if self.minmax_dict is not None and name in self.minmax_dict:
            #     scaling_errors.append(scaling_error(y_true_i, y_pred_i, name, self.minmax_dict))
            #     scaling_errors_denorm.append(scaling_error(y_true_i_denorm, y_pred_i_denorm, name, self.minmax_dict))
            # else:
            #     scaling_errors.append(np.nan)
            #     scaling_errors_denorm.append(np.nan)
                
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
        with torch.no_grad():
            t_pred = torch.from_numpy(np.where(np.isfinite(y_preds), y_preds, 0.0)).float()
            t_true_norm = torch.from_numpy(np.where(np.isfinite(y_trues_norm), y_trues_norm, 0.0)).float()
            t_mask = torch.from_numpy(mask).bool()
            # Put on CPU; returns per-task tensor
            losses = self.masked_huber_loss(t_pred, t_true_norm, mask=t_mask)
            # Convert to plain floats for logging
            metrics['losses'] = losses.detach().cpu().numpy()
        return metrics
        
    # helper functions for different weighting strategies
    def _ensure_task_state(self, K: int, device: torch.device):
        """Ensure the task state is initialized.
        """
        if self._K is None:
            self._K = K
            self._task_weights = torch.ones(K, device=device) / K
            
    def _compute_per_task_losses(self, batch: Batch, labels: List[str]):
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
            device = self.device,
        )
        return losses
    
    def _pcgrad_step(self, per_task_losses, optimizer):
        shared_params = [p for p in self.model.parameters() if p.requires_grad]
        grads = []
        for Lk in per_task_losses:
            optimizer.zero_grad(set_to_none=True)
            Lk.backward(retain_graph=True)
            gk = []
            for p in shared_params:
                gk.append(torch.zeros_like(p) if p.grad is None else p.grad.detach().clone())
            grads.append(gk)
            
        K = len(grads)
        for i in range(K):
            gi = grads[i]
            for j in range(K):
                if i == j:
                    continue
                gj = grads[j]
                dot = torch.tensor(0.0, device = gi[0].device)
                norm_gj_sq = torch.tensor(0.0, device = gi[0].device)
                for p_i, p_j in zip(gi, gj):
                    dot += (p_i * p_j).sum()
                    norm_gj_sq += (p_j * p_j).sum()
                if dot < 0 and norm_gj_sq > 0:
                    coeff = dot / norm_gj_sq
                    for idx in range(len(gi)):
                        gi[idx] = gi[idx] - coeff * gj[idx]
            grads[i] = gi
            
        optimizer.zero_grad(set_to_none=True)
        for params_idx, p in enumerate(shared_params):
            agg = torch.zeros_like(p)
            for k in range(K):
                agg = agg + grads[k][params_idx]
            p.grad = agg
        optimizer.step()
        
    def _gradnorm_update_and_step(self, per_task_losses, init_losses, optimizer):
        """
        per_task_losses: Tensor (K,) with differentiable per-task losses L_k
        init_losses:     Tensor (K,) snapshot at the start (no grad needed)
        Updates self._task_weights (w_k) and does one optimizer step on model params.
        """
        device = per_task_losses.device
        K = per_task_losses.numel()

        # ---- 1) define a single w that requires grad and use it everywhere below ----
        # normalize to sum K (GradNorm convention) for stability
        w = self._task_weights.to(device)
        w = K * w / (w.sum() + 1e-12)
        w = w.clone().detach().requires_grad_(True)   # this w is the ONLY one used below

        # ---- 2) compute gradient norms G_k = ||∂(w_k L_k)/∂θ_shared|| with graph ----
        shared = [p for p in self.model.parameters() if p.requires_grad]
        Gk_list = []
        for k in range(K):
            # grads wrt shared parameters, keep graph so we can backprop into w
            gk = torch.autograd.grad(
                w[k] * per_task_losses[k],
                shared,
                create_graph=True,   # <- critical: enables grad flow back to w
                retain_graph=True
            )
            # L2 norm over all shared params
            gnorm_sq = torch.zeros([], device=device)
            for g in gk:
                if g is not None:
                    gnorm_sq = gnorm_sq + (g.reshape(-1) @ g.reshape(-1))
            Gk_list.append(torch.sqrt(gnorm_sq + 1e-12))
        Gk = torch.stack(Gk_list)                 # (K,)
        G_avg = Gk.mean().detach()                # detach target stats

        # ---- 3) relative inverse training rate r_k (detached target) ----
        r = (per_task_losses.detach() / (init_losses + 1e-12))
        r = (r / r.mean()).detach()
        target = G_avg * (r ** self.gradnorm_alpha)   # (K,)

        # ---- 4) GradNorm objective and gradient wrt w ----
        gradnorm_obj = torch.sum(torch.abs(Gk - target))   # scalar
        grad_w = torch.autograd.grad(gradnorm_obj, w, retain_graph=True)[0]  # (K,)

        # ---- 5) update w (manual small step) and renormalize ----
        with torch.no_grad():
            new_w = w - self.gradnorm_lr * grad_w
            new_w = torch.clamp(new_w, min=1e-6)
            new_w = K * new_w / (new_w.sum() + 1e-12)
            self._task_weights = new_w.detach()

        # ---- 6) finally update model params using the NEW weights ----
        optimizer.zero_grad(set_to_none=True)
        final_loss = torch.sum(self._task_weights.to(device) * per_task_losses)
        final_loss.backward()
        optimizer.step()

        return final_loss

    
    def _dwa_weights(self):
        if len(self._loss_hist) < 2:
            return torch.ones(self._K, device=self.device) / self._K
        
        L_t1 = self._loss_hist[-1]
        L_t2 = self._loss_hist[-2]
        ratio = L_t1 / (L_t2 + 1e-12)
        
        logits = ratio / self.dwa_T
        w = torch.exp(logits)
        w = self._K * w / (w.sum() + 1e-12)
        return w
    
    
class HeteroGaussianNLLCriterion(torch.nn.Module):
    def __init__(self, task_log_sigma: Optional[torch.Tensor] = None, include_const: bool = True):
        super().__init__()
        if task_log_sigma is not None:
            self.register_buffer('task_log_sigma', task_log_sigma.clone().detach())
        else:
            self.task_log_sigma = None
        self.include_const = include_const

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_task = Trainer.heteroscedastic_gaussian_nll_masked(
            pred,
            target,
            task_log_sigma = self.task_log_sigma,
            include_const = self.include_const,
        )
        return per_task.mean()
        