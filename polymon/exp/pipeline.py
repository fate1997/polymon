import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import loguru
import optuna
from torch_geometric.loader import DataLoader

from polymon.data.dataset import PolymerDataset
from polymon.data.utils import Normalizer
from polymon.exp.train import Trainer
from polymon.exp.utils import seed_everything
from polymon.hparams import get_hparams
from polymon.model import (AttentiveFPWrapper, DimeNetPP, GATPort, GATv2,
                           GATv2VirtualNode, GIN, PNA, GVPModel, GATChain,
                           GATv2ChainReadout)
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
    ):
        seed_everything(42)
        
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
        
        logger = loguru.logger
        logger.add(os.path.join(out_dir, 'pipeline.log'))
        self.logger = logger    

        self.logger.info(f'Building dataset for {label}...')
        self.dataset = self._build_dataset(sources)
        if self.descriptors is not None:
            self.num_descriptors = self.dataset[0].descriptors.shape[1]
        else:
            self.num_descriptors = 0
        loaders = self.dataset.get_loaders(self.batch_size, 0.8, 0.1)
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
        self.logger.info(f'test error: {test_err}')
        trainer.model.write(os.path.join(out_dir, f'{self.model_name}.pt'))
        return test_err
    
    def optimize_hparams(self) -> Tuple[float, Dict[str, Any]]:
        self.logger.info(f'Optimizing hyperparameters for {self.model_type}...')
        
        out_dir = os.path.join(self.out_dir, 'hparams_opt')
        os.makedirs(out_dir, exist_ok=True)
        def objective(trial: optuna.Trial) -> float:
            model_hparams = get_hparams(trial, self.model_type)
            self.logger.info(f'Number of trials: {trial.number}')
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
        if self.model_type.lower() in ['dimenetpp', 'gvp']:
            feature_names.append('pos')
            feature_names.remove('bond')
        if self.model_type.lower() in ['gatv2vn']:
            feature_names.append('virtual_bond')
            feature_names.remove('bond')
        if self.model_type.lower() in ['gatchain']:
            feature_names.append('bridge')
        if self.descriptors is not None:
            feature_names.extend(self.descriptors)
        self.logger.info(f'Feature names: {feature_names}')
        dataset = PolymerDataset(
            raw_csv_path=self.raw_csv,
            sources=sources,
            feature_names=feature_names,
            label_column=self.label,
            force_reload=True,
            add_hydrogens=True,
        )
        self.logger.info(f'Atom features: {dataset.num_node_features}')
        self.logger.info(f'Bond features: {dataset.num_edge_features}')
        return dataset
    
    def _build_model(self, hparams: Dict[str, Any]) -> ModelWrapper:
        if self.model_type == 'gatv2':
            input_args = {
                'num_atom_features': self.dataset.num_node_features,
                'edge_dim': self.dataset.num_edge_features,
                'num_descriptors': self.num_descriptors,
            }
            input_args.update(hparams)
            model = GATv2(**input_args)
        elif self.model_type == 'attentivefp':
            input_args = {
                'in_channels': self.dataset.num_node_features,
                'edge_dim': self.dataset.num_edge_features,
                'out_channels': 1,
            }
            input_args.update(hparams)
            model = AttentiveFPWrapper(**input_args)
        elif self.model_type == 'dimenetpp':
            input_args = {
                'out_channels': 1,
            }
            input_args.update(hparams)
            model = DimeNetPP(**input_args)
        elif self.model_type == 'gatport':
            input_args = {
                'num_atom_features': self.dataset.num_node_features,
                'edge_dim': self.dataset.num_edge_features,
                'num_descriptors': self.num_descriptors,
            }
            input_args.update(hparams)
            model = GATPort(**input_args)
        elif self.model_type == 'gatv2vn':
            input_args = {
                'num_atom_features': self.dataset.num_node_features,
                'edge_dim': self.dataset.num_edge_features,
                'num_descriptors': self.num_descriptors,
            }
            input_args.update(hparams)
            model = GATv2VirtualNode(**input_args)
        elif self.model_type == 'gin':
            input_args = {
                'num_atom_features': self.dataset.num_node_features,
            }
            input_args.update(hparams)
            model = GIN(**input_args)
        elif self.model_type == 'pna':
            input_args = {
                'in_channels': self.dataset.num_node_features,
                'edge_dim': self.dataset.num_edge_features,
                'deg': PNA.compute_deg(self.train_loader),
            }
            input_args.update(hparams)
            model = PNA(**input_args)
        elif self.model_type == 'gvp':
            input_args = {
                'in_node_nf': self.dataset.num_node_features,
            }
            input_args.update(hparams)
            model = GVPModel(**input_args)
        elif self.model_type == 'gatchain':
            input_args = {
                'num_atom_features': self.dataset.num_node_features,
            }
            input_args.update(hparams)
            model = GATChain(**input_args)
        elif self.model_type == 'gatv2chainreadout':
            input_args = {
                'num_atom_features': self.dataset.num_node_features,
                'edge_dim': self.dataset.num_edge_features,
                'num_descriptors': self.num_descriptors,
            }
            input_args.update(hparams)
            model = GATv2ChainReadout(**input_args)
        else:
            raise ValueError(f"Model type {self.model_type} not implemented")
        
        num_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f'Model Parameters: {num_params / 1e6:.4f}M')
        model = ModelWrapper(model, self.normalizer, self.dataset.featurizer)
        return model