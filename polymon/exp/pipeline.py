import os
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple

import loguru
import logging
import numpy as np
import optuna
import torch_geometric.transforms as T
from pytorch_lightning import seed_everything
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from torchensemble import VotingRegressor
from torch import nn
from polymon.exp.score import scaling_error
from torchensemble.utils import io
from polymon.data.dataset import PolymerDataset
from polymon.data.utils import Normalizer, DescriptorSelector, RgEstimator
from polymon.exp.train import Trainer
from polymon.hparams import get_hparams
from polymon.model import PNA, build_model
from polymon.model.base import ModelWrapper, EnsembleModelWrapper
from polymon.setting import REPO_DIR, DEFAULT_ATOM_FEATURES


class Pipeline:
    def __init__(
        self,
        tag: str,
        out_dir: str,
        batch_size: int,
        raw_csv: str,
        sources: List[str],
        label: str,
        model_type: str,
        hidden_dim: int,
        num_layers: int,
        descriptors: Optional[List[str]],
        num_epochs: int,
        lr: float,
        early_stopping_patience: int,
        device: str,
        n_trials: int,
        seed: int = 42,
        split_mode: Literal['source', 'random', 'scaffold'] = 'random',
        train_residual: bool = False,
        additional_features: Optional[List[str]] = None,
    ):
        seed_everything(seed)
        
        self.tag = tag
        self.out_dir = out_dir
        self.batch_size = batch_size
        self.raw_csv = raw_csv
        self.sources = sources
        self.label = label
        self.model_type = model_type.lower()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.descriptors = descriptors
        self.num_epochs = num_epochs
        self.lr = lr
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.n_trials = n_trials
        self.model_name = f'{self.model_type}_{self.label}_{self.tag}'
        self.split_mode = split_mode
        self.train_residual = train_residual
        self.additional_features = additional_features
        
        logger = loguru.logger
        log_path = os.path.join(out_dir, 'pipeline.log')
        logger.add(log_path)
        handler = logging.FileHandler(log_path)
        optuna.logging.get_logger('optuna').addHandler(handler)
        self.logger = logger    

        self.logger.info(f'Building dataset for {label}...')
        self.dataset = self._build_dataset(sources)
        if self.descriptors is not None:
            self.num_descriptors = self.dataset[0].descriptors.shape[1]
        else:
            self.num_descriptors = 0
        loaders = self.dataset.get_loaders(self.batch_size, 0.8, 0.1, self.split_mode)
        self.train_loader, self.val_loader, self.test_loader = loaders
        self.normalizer = Normalizer.from_loader(self.train_loader)
        self.logger.info(f'Number of descriptors used: {self.num_descriptors}')
        self.transform_cls, self.transform_kwargs = self._init_transform()
    
    def train(
        self,
        lr: Optional[float] = None,
        model_hparams: Optional[Dict[str, Any]] = None,
        out_dir: Optional[str] = None,
        loaders: Optional[Tuple[DataLoader, DataLoader, DataLoader]] = None,
        model: Optional[ModelWrapper] = None,
    ):
        self.logger.info(f'Training {self.model_type} model for {self.label}...')
        if lr is None:
            lr = self.lr
        if model is None:
            if model_hparams is None:
                model_hparams = {
                    'num_layers': self.num_layers,
                    'hidden_dim': self.hidden_dim,
                }
            model = self._build_model(model_hparams)
        
        if out_dir is None:
            out_dir = os.path.join(self.out_dir, 'train')
        os.makedirs(out_dir, exist_ok=True)
        trainer = Trainer(
            out_dir=out_dir,
            model=model,
            lr=lr,
            num_epochs=self.num_epochs,
            logger=self.logger,
            device=self.device,
            early_stopping_patience=self.early_stopping_patience,
        )
        if loaders is None:
            loaders = (self.train_loader, self.val_loader, self.test_loader)
        test_err = trainer.train(loaders[0], loaders[1], loaders[2], self.label)
        trainer.model.write(os.path.join(out_dir, f'{self.model_name}.pt'))
        return test_err
    
    def cross_validation(
        self, 
        n_fold: int = 5,
        model_hparams: Optional[Dict[str, Any]] = None,
        model: Optional[ModelWrapper] = None,
    ) -> List[float]:
        kfold = KFold(n_splits=n_fold, shuffle=True)
        val_errors = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.dataset)):
            self.logger.info(f'Running fold {fold+1}/{n_fold}...')
            train_loader = DataLoader(
                self.dataset[train_idx], batch_size=self.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                self.dataset[val_idx], batch_size=self.batch_size, shuffle=False
            )
            out_dir = os.path.join(self.out_dir, f'fold_{fold+1}')
            os.makedirs(out_dir, exist_ok=True)
            val_error = self.train(
                model_hparams=model_hparams,
                out_dir=out_dir,
                loaders=(train_loader, val_loader, None),
                model=model,
            )
            self.logger.info(f'{fold+1}/{n_fold} val error: {val_error}')
            val_errors.append(val_error)
        mean, std = np.mean(val_errors), np.std(val_errors)
        self.logger.info(f'K-Fold validation error: {mean:.4f} ± {std:.4f}')
        return val_errors
    
    def optimize_hparams(self, n_fold: int = 1) -> Tuple[float, Dict[str, Any]]:
        self.logger.info(f'Optimizing hyperparameters for {self.model_type}...')
        
        out_dir = os.path.join(self.out_dir, 'hparams_opt')
        os.makedirs(out_dir, exist_ok=True)
        def objective(trial: optuna.Trial) -> float:
            model_hparams = get_hparams(trial, self.model_type)
            self.logger.info(f'Number of trials: {trial.number+1}/{self.n_trials}')
            self.logger.info(f'Hyper-parameters: {model_hparams}')
            if n_fold > 1:
                val_errors = self.cross_validation(
                    n_fold=n_fold,
                    model_hparams=model_hparams, 
                )
                return np.mean(val_errors)
            else:
                return self.train(self.lr, model_hparams, out_dir)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials = self.n_trials)
        self.logger.info(f'--------------------------------')
        self.logger.info(f'{self.model_type}')
        self.logger.info(f'Best hyper-parameters: {study.best_params}')
        self.logger.info(f'Best test error: {study.best_value}')
        
        return study.best_value, study.best_params
    
    def finetune(
        self,
        lr: float,
        pretrained_model_path: str,
        csv_path: str = None,
        production_run: bool = False,
        freeze_repr_layers: bool = True,
        n_fold: int = 1,
    ):
        out_dir = os.path.join(self.out_dir, 'finetune')
        os.makedirs(out_dir, exist_ok=True)
        self.logger.info(f'Finetuning {self.model_type} model for {self.label}...')
        if csv_path is not None:
            self.raw_csv = csv_path
            self.dataset = self._build_dataset(self.sources)
        
        pretrained_model = ModelWrapper.from_file(pretrained_model_path, self.device)
        pretrained_dict = {k: v for k, v in pretrained_model.info['model'].items() if not k.startswith('predict')}
        model = self._build_model(pretrained_model.info['model_init_params'])
        model_dict = model.info['model']
        model_dict.update(pretrained_dict)
        model.model.load_state_dict(model_dict)
        if freeze_repr_layers:
            # Freeze the layers that are not starting with 'predict'
            for name, param in model.model.named_parameters():
                if not name.startswith('predict'):
                    param.requires_grad = False
        if n_fold == 1:
            test_err = self.train(
                lr=lr,
                model_hparams=pretrained_model.info['model_init_params'],
                out_dir=out_dir,
                model=model,
            )
        else:
            test_err = self.cross_validation(
                n_fold=n_fold,
                model_hparams=pretrained_model.info['model_init_params'],
                model=model,
            )
            test_err = np.mean(test_err)
        self.logger.info(f'test error: {test_err}')
        if production_run:
            self.production_run(
                model_hparams=pretrained_model.info['model_init_params'],
                model=model,
            )
        return test_err
    
    def production_run(
        self, 
        model_hparams: Optional[Dict[str, Any]] = None,
        model: Optional[ModelWrapper] = None,
    ):
        self.logger.info(f'Running production run for {self.model_type}...')
        out_dir = os.path.join(self.out_dir, 'production')
        os.makedirs(out_dir, exist_ok=True)

        loaders = self.dataset.get_loaders(self.batch_size, 0.95, 0.05, production_run=True)
        self.train(None, model_hparams, out_dir, loaders, model)
        self.logger.info(f'Production run complete.')
    
    def run_ensemble(
        self,
        n_estimator: int,
        model_hparams: Optional[Dict[str, Any]] = None,
        run_production: bool = False,
    ):
        self.logger.info(f'Running ensemble for {self.model_type}...')
        out_dir = os.path.join(self.out_dir, 'ensemble', 'train')
        os.makedirs(out_dir, exist_ok=True)
        if model_hparams is None:
            model_hparams = {
                'num_layers': self.num_layers,
                'hidden_dim': self.hidden_dim,
            }
        model_wrapper = self._build_model(model_hparams)
        
        def build_ensemble(model_wrapper: ModelWrapper, n_estimator: int) -> VotingRegressor:
            model = VotingRegressor(
                estimator=model_wrapper.model.__class__,
                estimator_args=model_wrapper.info['model_init_params'],
                n_estimators=n_estimator,
            )
            model.logger = self.logger
            model.set_criterion(nn.L1Loss())
            model.set_optimizer('AdamW', lr=self.lr, weight_decay=1e-12)
            return model

        ensemble_model = build_ensemble(model_wrapper, n_estimator)
        ensemble_wrapper = EnsembleModelWrapper(
            model=ensemble_model,
            normalizer=self.normalizer,
            featurizer=self.dataset.featurizer,
            transform_cls=self.transform_cls,
            transform_kwargs=self.transform_kwargs,
        )
        test_err = ensemble_wrapper.fit(
            epochs=self.num_epochs,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            save_dir=out_dir,
            save_model=True,
            log_interval=1000000000,
            label=self.label,
        )
        self.logger.info(f'Ensemble test error: {test_err}')
        
        if run_production:
            self.logger.info(f'Running ensemble production run for {self.model_type}...')
            train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
            prod_ensemble_model = build_ensemble(model_wrapper, n_estimator)
            prod_ensemble_wrapper = EnsembleModelWrapper(
                model=prod_ensemble_model,
                normalizer=self.normalizer,
                featurizer=self.dataset.featurizer,
                transform_cls=self.transform_cls,
                transform_kwargs=self.transform_kwargs,
            )
            save_dir = os.path.join(self.out_dir, 'ensemble', 'production')
            prod_ensemble_wrapper.fit(
                train_loader=train_loader,
                epochs=self.num_epochs,
                save_dir=save_dir,
                save_model=True,
                log_interval=1000000000,
                label=self.label,
            )
            prod_ensemble_model = build_ensemble(model_wrapper, n_estimator)
            io.load(prod_ensemble_model, save_dir)
            prod_ensemble_wrapper.model = prod_ensemble_model
            prod_ensemble_wrapper.write(
                os.path.join(self.out_dir, 'ensemble', 'production', f'{self.model_name}.pt')
            )
        
        return test_err
    
    def _build_dataset(self, sources: List[str]) -> PolymerDataset:
        feature_names = ['x', 'bond', 'z']
        if self.additional_features is not None:
            feature_names.extend(self.additional_features + DEFAULT_ATOM_FEATURES)
        if self.model_type.lower() in ['dimenetpp', 'gvp']:
            feature_names.append('pos')
            feature_names.remove('bond')
        if self.model_type.lower() in ['gatv2vn']:
            feature_names.append('virtual_bond')
            feature_names.remove('bond')
        if self.model_type.lower() in ['gatchain']:
            feature_names.append('bridge')
        if self.model_type.lower() in ['fastkan', 'efficientkan', 'kan', 'fourierkan']:
            assert self.descriptors is not None, 'Descriptors are required for KAN'
        if self.descriptors is not None:
            feature_names.extend(self.descriptors)
        self.logger.info(f'Feature names: {feature_names}')
        pre_transform = None
        if self.model_type.lower() in ['gps', 'kan_gps']:
            pre_transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        if self.train_residual:
            pre_transform = RgEstimator(
                N=600,
                C_inf=6.7,
                solvent='theta',
            )
        dataset = PolymerDataset(
            raw_csv_path=self.raw_csv,
            sources=sources,
            feature_names=feature_names,
            label_column=self.label,
            force_reload=True,
            add_hydrogens=True,
            pre_transform=pre_transform
        )
        if self.descriptors is not None and self.model_type.lower():
            self.logger.info(f'Creating descriptor selector for {self.descriptors}...')
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            if self.model_type.lower() in ['gatv2']:
                selector = DescriptorSelector.from_rf(loader)
            else:
                selector = DescriptorSelector.from_rf(loader, top_k=-1)
            if pre_transform is not None:
                self.logger.error(f'Not support multiple pre-transforms')
                raise NotImplementedError
            dataset = PolymerDataset(
                raw_csv_path=self.raw_csv,
                sources=sources,
                feature_names=feature_names,
                label_column=self.label,
                force_reload=True,
                add_hydrogens=True,
                pre_transform=selector
            )
        
        self.logger.info(f'Atom features: {dataset.num_node_features}')
        self.logger.info(f'Bond features: {dataset.num_edge_features}')
        return dataset
    
    def _build_model(self, model_hparams: Dict[str, Any]) -> ModelWrapper:
        if self.model_type.lower() in ['pna']:
            model_hparams['deg'] = PNA.compute_deg(self.train_loader)
        
        model = build_model(
            model_type=self.model_type,
            num_node_features=self.dataset.num_node_features,
            num_edge_features=self.dataset.num_edge_features,
            num_descriptors=self.num_descriptors,
            hparams=model_hparams,
        )
        num_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f'Model Parameters: {num_params / 1e6:.4f}M')
        
        model = ModelWrapper(
            model,
            self.normalizer,
            self.dataset.featurizer,
            self.transform_cls,
            self.transform_kwargs,
        )
        return model
    
    def _init_transform(self):
        if self.model_type.lower() in ['gps', 'kan_gps']:
            transform_cls = 'AddRandomWalkPE'
            transform_kwargs = {
                'walk_length': 20,
                'attr_name': 'pe',
            }
        else:
            transform_cls = None
            transform_kwargs = None
        if self.descriptors is not None:
            transform_cls = 'DescriptorSelector'
            transform_kwargs = {
                'ids': self.dataset.pre_transform.ids,
                'mean': self.dataset.pre_transform.mean,
                'std': self.dataset.pre_transform.std,
            }
        if self.train_residual:
            assert self.label == 'Rg', 'Train residual is only supported for Rg'
            transform_cls = 'RgEstimator'
            transform_kwargs = {
                'N': 600,
                'C_inf': 6.7,
                'solvent': 'theta',
            }
        return transform_cls, transform_kwargs