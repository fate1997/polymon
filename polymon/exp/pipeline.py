import json
import logging
import os
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple

import loguru
import numpy as np
import optuna
import torch
import torch_geometric.transforms as T
from pytorch_lightning import seed_everything
from sklearn.model_selection import KFold
from torch import nn
from torch_geometric.loader import DataLoader
from torchensemble import VotingRegressor
from torchensemble.utils import io

from polymon.data.dataset import PolymerDataset
from polymon.data.utils import DescriptorSelector, Normalizer, RgEstimator, LineEvoTransform, LogNormalizer
from polymon.estimator import get_estimator
from polymon.exp.score import scaling_error
from polymon.exp.train import Trainer
from polymon.hparams import get_hparams
from polymon.model import PNA, build_model
from polymon.model.base import EnsembleModelWrapper, KFoldModel, ModelWrapper
from polymon.setting import DEFAULT_ATOM_FEATURES, REPO_DIR
from polymon.estimator.ml import MLEstimator


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
        low_fidelity_model: Optional[str] = None,
        normalizer_type: Literal['normalizer', 'log_normalizer', 'none'] = 'normalizer',
        estimator_name: Optional[str] = None,
        remove_hydrogens: bool = False,
        augmentation: bool = False,
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
        self.low_fidelity_model = low_fidelity_model
        self.normalizer_type = normalizer_type
        self.estimator_name = estimator_name if estimator_name is not None else self.label
        self.remove_hydrogens = remove_hydrogens
        self.augmentation = augmentation
        
        logger = loguru.logger
        log_path = os.path.join(out_dir, 'pipeline.log')
        logger.add(log_path)
        handler = logging.FileHandler(log_path)
        optuna.logging.get_logger('optuna').addHandler(handler)
        self.logger = logger    
        
        self.estimator = None
        if self.train_residual:
            self.logger.info(f'Train residual: {self.label}')
            if self.low_fidelity_model is not None:
                self.logger.info(f'Using low fidelity model: {self.low_fidelity_model}')
                if self.low_fidelity_model.endswith('.pt'):
                    model = ModelWrapper.from_file(self.low_fidelity_model, self.device)
                    self.estimator = get_estimator(
                        'LowFidelity', model_info=model.info
                    )
                elif self.low_fidelity_model.endswith('.pkl'):
                    self.estimator = MLEstimator.from_pickle(self.low_fidelity_model)
            else:
                self.estimator = get_estimator(self.estimator_name)

        self.logger.info(f'Building dataset for {label}...')
        self.dataset = self._build_dataset(sources)
        if self.descriptors is not None:
            self.num_descriptors = self.dataset[0].descriptors.shape[1]
        else:
            self.num_descriptors = 0
        loaders = self.dataset.get_loaders(self.batch_size, 0.8, 0.1, self.split_mode, augmentation=self.augmentation)
        self.train_loader, self.val_loader, self.test_loader = loaders
        
        self.logger.info(f'Using {self.normalizer_type} normalizer')
        if self.normalizer_type in ['normalizer', 'none']:
            self.normalizer = Normalizer.from_loader(self.train_loader)
            if self.normalizer_type == 'none':
                self.normalizer.mean = 0.0
                self.normalizer.std = 1.0
        elif self.normalizer_type == 'log_normalizer':
            eps = 273.15 if self.label == 'Tg' else 1e-10
            self.normalizer = LogNormalizer(eps=eps)
        else:
            raise ValueError(f'Invalid normalizer type: {self.normalizer_type}')
        
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
        return test_err, model
    
    def cross_validation(
        self, 
        n_fold: int = 5,
        model_hparams: Optional[Dict[str, Any]] = None,
        model: Optional[ModelWrapper] = None,
        current_trial: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> List[float]:
        kfold = KFold(n_splits=n_fold, shuffle=True)
        val_errors = []
        models = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.dataset)):
            print_str = f'Running fold {fold+1}/{n_fold}'
            if current_trial is not None:
                print_str += f' (trial {current_trial}/{self.n_trials})'
            self.logger.info(print_str)
            train_set = self.dataset[train_idx]
            if self.augmentation:
                from polymon.data.polymer import OligomerBuilder
                train_set_aug = []
                for data in train_set:
                    train_set_aug.append(data)
                    for i in range(1):
                        oligomer = OligomerBuilder.get_oligomer(data.smiles, i+2)
                        mol_dict = self.dataset.featurizer(oligomer)
                        aug_data = data.clone()
                        for key, value in mol_dict.items():
                            setattr(aug_data, key, value)
                        train_set_aug.append(aug_data)
                    # from rdkit import Chem
                    # from rdkit.Chem.MolStandardize import rdMolStandardize
                    # ps = rdMolStandardize.CleanupParameters()
                    # ps.maxTransforms = 5000
                    # te = rdMolStandardize.TautomerEnumerator(ps)
                    # rdmol = Chem.MolFromSmiles(data.smiles)
                    # tautomers = te.Enumerate(rdmol)
                    # for tautomer in tautomers:
                    #     if Chem.MolToSmiles(tautomer) == Chem.MolToSmiles(rdmol):
                    #         continue
                    #     mol_dict = self.dataset.featurizer(tautomer)
                    #     aug_data = data.clone()
                    #     for key, value in mol_dict.items():
                    #         setattr(aug_data, key, value)
                    #     train_set_aug.append(aug_data)
                self.logger.info(f'Train set augmented from {len(train_set)} to {len(train_set_aug)}')
            else:
                train_set_aug = train_set
            
            train_loader = DataLoader(
                train_set_aug, batch_size=self.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                self.dataset[val_idx], batch_size=self.batch_size, shuffle=False
            )
            out_dir = os.path.join(self.out_dir, f'fold_{fold+1}')
            os.makedirs(out_dir, exist_ok=True)
            val_error, trained_model = self.train(
                lr=lr,
                model_hparams=model_hparams,
                out_dir=out_dir,
                loaders=(train_loader, val_loader, None),
                model=deepcopy(model) if model is not None else None,
            )
            self.logger.info(f'{fold+1}/{n_fold} val error: {val_error}')
            models.append(trained_model)
            val_errors.append(val_error)
        mean, std = np.mean(val_errors), np.std(val_errors)
        self.logger.info(f'K-Fold validation error: {mean:.4f} ± {std:.4f}')
        kfold_model = KFoldModel.from_models(models)
        kfold_model_wrapper = models[0]
        kfold_model_wrapper.model = kfold_model
        kfold_model_wrapper.write(os.path.join(self.out_dir, f'{self.model_name}-KFold.pt'))
        return val_errors
    
    def optimize_hparams(self, n_fold: int = 1) -> Tuple[float, Dict[str, Any]]:
        self.logger.info(f'Optimizing hyperparameters for {self.model_type}...')
        
        out_dir = os.path.join(self.out_dir, 'hparams_opt')
        os.makedirs(out_dir, exist_ok=True)
        def objective(trial: optuna.Trial) -> float:
            model_hparams = get_hparams(trial, self.model_type)
            hparams = {
                'lr': trial.suggest_float("lr", 1e-4, 2e-3, log=True),
                **model_hparams,
            }
            self.logger.info(f'Number of trials: {trial.number+1}/{self.n_trials}')
            self.logger.info(f'Hyper-parameters: {hparams}')
            if n_fold > 1:
                val_errors = self.cross_validation(
                    n_fold=n_fold,
                    model_hparams=model_hparams, 
                    current_trial=trial.number+1,
                    lr=hparams['lr'],
                )
                return np.mean(val_errors)
            else:
                return self.train(self.lr, model_hparams, out_dir)[0]

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials = self.n_trials)
        self.logger.info(f'--------------------------------')
        self.logger.info(f'{self.model_type}')
        self.logger.info(f'Best hyper-parameters: {study.best_params}')
        self.logger.info(f'Best test error: {study.best_value}')
        
        hparams_path = os.path.join(out_dir, 'hparams.json')
        with open(hparams_path, 'w') as f:
            json.dump(study.best_params, f)
        
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
            )[0]
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

        loaders = self.dataset.get_loaders(self.batch_size, 0.95, 0.05, production_run=True, augmentation=self.augmentation)
        self.train(None, model_hparams, out_dir, loaders, model)
        self.logger.info(f'Production run complete.')
    
    def run_ensemble(
        self,
        n_estimator: int,
        model_hparams: Optional[Dict[str, Any]] = None,
        run_production: bool = False,
        skip_train: bool = False,
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

        if skip_train:
            test_err = np.nan
            self.logger.info(f'Skipping ensemble training')
        else:
            ensemble_model = build_ensemble(model_wrapper, n_estimator)
            ensemble_wrapper = EnsembleModelWrapper(
                model=ensemble_model,
                normalizer=self.normalizer,
                featurizer=self.dataset.featurizer,
                transform_cls=self.transform_cls,
                transform_kwargs=self.transform_kwargs,
                estimator=self.estimator,
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
                estimator=self.estimator,
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
        # Set feature names
        feature_names = ['x', 'bond', 'z']
        if self.additional_features is not None:
            feature_names.extend(self.additional_features + DEFAULT_ATOM_FEATURES)
        if 'periodic_bond' in feature_names:
            feature_names.remove('bond')
        if self.model_type.lower() in ['dimenetpp', 'gvp']:
            feature_names.append('pos')
            feature_names.remove('bond')
        if self.model_type.lower() in ['gatv2vn']:
            feature_names.append('virtual_bond')
            feature_names.remove('bond')
        if self.model_type.lower() in ['gatchain']:
            feature_names.append('bridge')
        if self.model_type.lower() in ['gatv2_pe']:
            feature_names.append('relative_position')
        if self.model_type.lower() in ['fastkan', 'efficientkan', 'kan', 'fourierkan']:
            assert self.descriptors is not None, 'Descriptors are required for KAN'
        if self.descriptors is not None:
            feature_names.extend(self.descriptors)
        self.logger.info(f'Feature names: {feature_names}')
        
        # Set pre-transforms
        pre_transform = None
        if self.model_type.lower() in ['gps', 'kan_gps']:
            pre_transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        if self.model_type.lower() in ['gatv2_lineevo']:
            pre_transform = LineEvoTransform(depth=2)

        dataset = PolymerDataset(
            raw_csv_path=self.raw_csv,
            sources=sources,
            feature_names=feature_names,
            label_column=self.label,
            force_reload=True,
            add_hydrogens=not self.remove_hydrogens,
            pre_transform=pre_transform,
            estimator=self.estimator,
        )
        
        # Post-transforms after creating dataset
        if self.descriptors is not None and self.model_type.lower() in []: #'gatv2'
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
                pre_transform=selector,
                estimator=self.estimator,
            )
        
        self.logger.info(f'Atom features: {dataset.num_node_features}')
        self.logger.info(f'Bond features: {dataset.num_edge_features}')
        return dataset
    
    def _build_model(self, model_hparams: Dict[str, Any]) -> ModelWrapper:
        if self.model_type.lower() in ['pna']:
            model_hparams['deg'] = PNA.compute_deg(self.train_loader)
        if self.model_type.lower() in ['gatv2_source']:
            model_hparams['source_names'] = self.sources + ['internal']

        if self.model_type.lower() in ['gatv2_embed_residual'] and self.low_fidelity_model is not None:
            model_hparams['pretrained_model'] = ModelWrapper.from_file(self.low_fidelity_model, self.device).model

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
            self.estimator,
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
            # transform_cls = 'DescriptorSelector'
            # transform_kwargs = {
            #     'ids': self.dataset.pre_transform.ids,
            #     'mean': self.dataset.pre_transform.mean,
            #     'std': self.dataset.pre_transform.std,
            # }
            pass
        if self.model_type.lower() in ['gatv2_lineevo']:
            transform_cls = 'LineEvoTransform'
            transform_kwargs = {
                'depth': 2,
            }
        return transform_cls, transform_kwargs