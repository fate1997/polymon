import os
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple

import loguru
import logging
import numpy as np
import optuna
import torch_geometric.transforms as T
from pytorch_lightning import seed_everything
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader

from polymon.data.dataset import PolymerDataset
from polymon.data.utils import Normalizer
from polymon.exp.train import Trainer
from polymon.hparams import get_hparams
from polymon.model import PNA, build_model
from polymon.model.base import ModelWrapper
from polymon.setting import REPO_DIR


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
    
    def train(
        self,
        lr: Optional[float] = None,
        model_hparams: Optional[Dict[str, Any]] = None,
        out_dir: Optional[str] = None,
        loaders: Optional[Tuple[DataLoader, DataLoader, DataLoader]] = None,
    ):
        self.logger.info(f'Training {self.model_type} model for {self.label}...')
        if lr is None:
            lr = self.lr
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
        out_dir: Optional[str] = None,
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
        csv_path: str,
        pretrained_model_path: str,
        production_run: bool = False,
        freeze_repr_layers: bool = True,
    ):
        out_dir = os.path.join(self.out_dir, 'finetune')
        os.makedirs(out_dir, exist_ok=True)
        self.logger.info(f'Finetuning {self.model_type} model for {self.label}...')
        dataset = self._build_dataset(csv_path)
        loaders = dataset.get_loaders(self.batch_size, 0.8, 0.1)
        
        model = ModelWrapper.from_file(pretrained_model_path, self.device)
        if freeze_repr_layers:
            # Freeze the layers that are not starting with 'predict'
            for name, param in model.model.named_parameters():
                if not name.startswith('predict'):
                    param.requires_grad = False
        
        trainer = Trainer(
            out_dir=out_dir,
            model=deepcopy(model),
            lr=lr,
            num_epochs=self.num_epochs,
            logger=self.logger,
            device=self.device,
            early_stopping_patience=self.early_stopping_patience,
        )
        test_err = trainer.train(loaders[0], loaders[1], loaders[2], self.label)
        self.logger.info(f'test error: {test_err}')
        if production_run:
            self.logger.info(f'Running production run for {self.model_type}...')
            loaders = dataset.get_loaders(self.batch_size, 0.95, 0.05)
            trainer = Trainer(
                out_dir=out_dir,
                model=deepcopy(model),
                lr=lr,
                num_epochs=self.num_epochs,
                logger=self.logger,
                device=self.device,
                early_stopping_patience=self.early_stopping_patience,
            )
            trainer.train(loaders[0], loaders[1], loaders[2], self.label)
        trainer.model.write(os.path.join(out_dir, f'{self.model_name}.pt'))
        return test_err
    
    def production_run(self, model_hparams: Optional[Dict[str, Any]] = None):
        self.logger.info(f'Running production run for {self.model_type}...')
        out_dir = os.path.join(self.out_dir, 'production')
        os.makedirs(out_dir, exist_ok=True)

        loaders = self.dataset.get_loaders(self.batch_size, 0.95, 0.05, production_run=True)
        self.train(None, model_hparams, out_dir, loaders)
        self.logger.info(f'Production run complete.')
    
    def _build_dataset(self, sources: List[str]) -> PolymerDataset:
        feature_names = ['x', 'bond', 'z']
        # feature_names += [
        #     'degree', 
        #     'is_aromatic', 
        #     'chiral_tag', 
        #     'num_hydrogens', 
        #     'hybridization', 
        #     'mass', 
        #     'formal_charge', 
        #     'is_attachment',
        #     'cgcnn'
        # ]
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
        if self.model_type.lower() in ['gps', 'kan_gps']:
            pre_transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        else:
            pre_transform = None
        dataset = PolymerDataset(
            raw_csv_path=self.raw_csv,
            sources=sources,
            feature_names=feature_names,
            label_column=self.label,
            force_reload=True,
            add_hydrogens=True,
            pre_transform=pre_transform
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
        if self.model_type.lower() in ['gps', 'kan_gps']:
            transform_cls = 'AddRandomWalkPE'
            transform_kwargs = {
                'walk_length': 20,
                'attr_name': 'pe',
            }
        else:
            transform_cls = None
            transform_kwargs = None
        
        model = ModelWrapper(
            model,
            self.normalizer,
            self.dataset.featurizer,
            transform_cls,
            transform_kwargs,
        )
        return model