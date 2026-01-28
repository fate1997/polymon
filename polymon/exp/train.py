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
        self.mt_strategy = 'uncertainty'

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
    def masked_nig_nll(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        coeff: float = 2.0,
        include_const: bool = True,
    ):

        # B, fourT = pred.shape
        # T = target.shape[-1]

        #mu, loglam, logalpha, logbeta = torch.split(pred, T, dim=-1)
        mu, lam, alpha, beta = torch.unbind(pred, dim=-1)
        # lam = torch.exp(loglam)
        # alpha = torch.exp(logalpha) + 1.0
        # beta = torch.exp(logbeta)
        # lam = F.softplus(loglam) + 1e-6
        # alpha = F.softplus(logalpha) + 1.0 + 1e-6
        # beta = F.softplus(logbeta) + 1e-6
        
        if mask is None:
            mask = ~torch.isnan(target)
        mask = mask.bool()

        target_filled = torch.where(mask, target, mu.detach())
        twoBlambda = 2 * beta * (1 + lam)
        resid = (target_filled - mu)
        nll = (
            0.5 * (torch.pi / lam).log()
            - alpha * twoBlambda.log()
            + (alpha + 0.5) * torch.log(lam * resid**2 + twoBlambda)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        # if include_const is False:
        #     nll = nll - 0.5 * (torch.pi).log()
 
        evidence = 2 * lam + alpha
        reg = resid.abs() * evidence

        loss = nll + (reg - 1e-8) * coeff

        # ---- Apply mask ----
        loss = loss * mask.float()
        denom = mask.float().sum(dim=0).clamp_min(1.0)

        per_task_loss = loss.sum(dim=0) / denom
        return per_task_loss

    @staticmethod
    def masked_niw_nll(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        coeff: float = 1.0,
        eps: float = 1e-6,
    ):
        B, p = target.shape

        # ---- unpack ----
        idx = 0
        mu = pred[:, idx:idx+p]; idx += p
        log_kappa = pred[:, idx:idx+1]; idx += 1
        num_tri = p*(p+1)//2
        L_elements = pred[:, idx:idx+num_tri]; idx += num_tri
        log_nu = pred[:, idx:idx+1]

        # ---- constraints ----
        kappa = F.softplus(log_kappa) + 1e-3
        nu = F.softplus(log_nu) + p + 2.0   # IMPORTANT: > p+1

        # ---- build Cholesky of Psi safely ----
        L = torch.zeros(B, p, p, device=pred.device)
        tril = torch.tril_indices(p, p)

        L[:, tril[0], tril[1]] = L_elements

        # enforce positive diagonal
        diag_idx = torch.arange(p)
        L[:, diag_idx, diag_idx] = F.softplus(L[:, diag_idx, diag_idx]) + 1e-3

        Psi = L @ L.transpose(-1, -2)
        Psi = Psi + eps * torch.eye(p, device=pred.device)

        # ---- masking ----
        if mask is None:
            mask = ~torch.isnan(target)
        target_filled = torch.where(mask, target, mu.detach())
        diff = target_filled - mu

        # ---- predictive covariance ----
        scale = (1.0 + 1.0/kappa) / (nu - p + 1.0)
        pred_cov = Psi * scale.unsqueeze(-1)

        # ---- Cholesky for stability ----
        chol = torch.linalg.cholesky(pred_cov)

        # ---- Mahalanobis distance (CORRECT) ----
        # solve (L Lᵀ)⁻¹ diff
        solve = torch.cholesky_solve(diff.unsqueeze(-1), chol)
        mahal = torch.sum(diff.unsqueeze(-1) * solve, dim=(-2, -1))

        # ---- log determinant ----
        logdet = 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)), dim=-1)

        # ---- Student-t NLL ----
        nll = 0.5 * (
            logdet
            + (nu.squeeze(-1) + p) * torch.log1p(mahal)
        )

        # ---- evidence regularizer ----
        reg = coeff * kappa.squeeze(-1) * (diff**2).sum(-1)

        #loss = nll + reg
        #print(nll, reg)
        loss = nll
        return loss.mean()



    @staticmethod
    def heteroscedastic_gaussian_nll_masked(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        include_const: bool = True,
    ):
        import math

        B, twoT = pred.shape
        T = target.shape[-1]

        mean = pred[:, :T]
        log_var = pred[:, T:]

        if mask is None:
            mask = ~torch.isnan(target)
        mask = mask.bool()

        # 🔑 fill NaNs BEFORE residuals
        target_filled = torch.where(mask, target, mean.detach())

        log_var = torch.clamp(log_var, min=math.log(1e-3), max=math.log(1e3))
        inv_var = torch.exp(-log_var)

        resid2 = (target_filled - mean) ** 2
        nll = 0.5 * (resid2 * inv_var + log_var)

        if include_const:
            nll = nll + 0.5 * math.log(2 * math.pi)

        nll = nll * mask.float()
        denom = mask.float().sum(dim=0).clamp_min(1.0)

        per_task = nll.sum(dim=0) / denom
        return per_task
    
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
        target = target.view_as(pred).float()
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
        mask = mask.view_as(pred)

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
        
        self.model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(self.device)
            if self.model.info['model_init_params']['train_loss'] == 'evidential':
                losses = self.model(
                    batch, 
                    loss_fn = lambda pred, target: self.masked_nig_nll(
                        pred, 
                        target, 
                        mask = ~torch.isnan(target),
                        coeff = 1.0,
                        include_const = True,
                    ),
                )
            else:
                losses = self.model(
                    batch, 
                    # loss_fn = lambda pred, target: self.heteroscedastic_gaussian_nll_masked(
                    #     pred, 
                    #     target, 
                    #     # task_log_sigma = self.log_sigma,
                    #     # include_const = True,
                    #     #delta = 1.0, 
                    #     mask = ~torch.isnan(target),
                    #     #task_weights = None,
                    # ),
                    # loss_fn = lambda pred, target: self.masked_nig_nll(
                    #     pred, 
                    #     target, 
                    #     mask = ~torch.isnan(target),
                    #     coeff = 0.01,
                    #     include_const = True,
                    # ),
                    loss_fn = lambda pred, target: self.masked_huber_loss(
                        pred, 
                        target, 
                        mask = ~torch.isnan(target),
                        delta = 1.0,
                    ),
                    device=self.device,
                )
            
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

            else:
                loss = losses.mean()
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
            #num_tasks = batch.y.shape[-1]
            # If heteroscedastic head: take only mean part
            # if y_raw.shape[-1] == 2 * num_tasks:      # <<< handle [mean | log_var]
            #     y_pred = y_raw[:, :num_tasks]         # (B, T) means
            #     # Optional: if you want to inspect variances/logvars:
            #     # y_log_var = y_raw[:, num_tasks:]        # (B, T) means
            # elif y_raw.ndim == 3 and y_raw.shape[-1] == 4:
            #     y_pred = y_raw[:, :, 0]   # mean μ, shape (B, T)
            #     # Optional: extract uncertainties if needed
            #     # lam   = y_raw[:, :, 1]
            #     # alpha = y_raw[:, :, 2]
            #     # beta  = y_raw[:, :, 3]
            if self.model.info['model_init_params']['train_loss'] == 'evidential':
                y_pred = y_raw[:, :, 0]   # mean μ, shape (B, T)
                # Optional: extract uncertainties if needed
                # lam   = y_raw[:, :, 1]
                # alpha = y_raw[:, :, 2]
                # beta  = y_raw[:, :, 3]
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
        # print(y_trues.shape, y_preds_denorm.shape)
        # print(label_mask.shape, pred_finite_mask.shape)
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
            #losses = self.masked_huber_loss(t_pred, t_true_norm, mask=t_mask)
            # losses = self.masked_nig_nll(t_pred, t_true_norm, mask=t_mask)
            # Convert to plain floats for logging
            # metrics['losses'] = losses.detach().cpu().numpy()
        return metrics


class HeteroGaussianNLLCriterion(torch.nn.Module):
    def forward(self, pred, target):
        return Trainer.heteroscedastic_gaussian_nll_masked(pred, target).mean()

class NIGCriterion(torch.nn.Module):
    def __init__(self, coeff: float = 1.0, include_const: bool = True):
        super().__init__()
        self.coeff = coeff
        self.include_const = include_const

    def forward(self, pred, target, mask: Optional[torch.Tensor] = None):
        return Trainer.masked_nig_nll(
            pred, 
            target, 
            mask = ~torch.isnan(target),
            coeff = self.coeff,
            include_const = self.include_const,
        ).mean()

class NIWCriterion(torch.nn.Module):
    def __init__(self, coeff: float = 1.0):
        super().__init__()
        self.coeff = coeff

    def forward(self, pred, target, mask: Optional[torch.Tensor] = None):
        return Trainer.masked_niw_nll(
            pred, 
            target, 
            mask = ~torch.isnan(target),
            coeff = self.coeff,
        ).mean()

class HuberLoss(torch.nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target, mask: Optional[torch.Tensor] = None):
        return Trainer.masked_huber_loss(
            pred, 
            target, 
            mask = ~torch.isnan(target),
            delta = self.delta,
        ).mean()
        