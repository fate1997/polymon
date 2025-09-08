import argparse
import logging
import math
import pickle
from typing import List, Optional, Dict
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import json

from polymon.data.dedup import Dedup
from polymon.data.featurizer import ComposeFeaturizer
from polymon.setting import REPO_DIR

import argparse
import os
import pickle

from typing import List, Tuple

import numpy as np
import torch
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from polymon.data.dataset import PolymerDataset
from polymon.data.featurizer import ComposeFeaturizer
from polymon.exp.score import scaling_error
from polymon.exp.utils import loader2numpy, predict_batch, seed_everything
from polymon.hparams import get_hparams
from polymon.setting import REPO_DIR
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

MODELS = {
    'rf': RandomForestRegressor,
    'xgb': XGBRegressor,
    'lgbm': LGBMRegressor,
    'catboost': CatBoostRegressor,
}

PREDICT_BATCH_SIZE = 128

def conditional_epig_from_probs(
    probs_pool: torch.Tensor, probs_targ: torch.Tensor, batch_size: int = 100
) -> torch.Tensor:
    """
    See conditional_epig_from_logprobs.

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]
        batch_size: int, size of the batch to process at a time

    Returns:
        Tensor[float], [N_p, N_t]
    """
    # Get sizes
    N_p, K, Cl = probs_pool.size()
    N_t = probs_targ.size(0)

    # Prepare tensors
    scores = torch.zeros(N_p, N_t)

    # Process in batches to save memory
    for i in range(0, N_p, batch_size):
        for j in range(0, N_t, batch_size):
            # Get the batch
            probs_pool_batch = probs_pool[i : i + batch_size]
            probs_targ_batch = probs_targ[j : j + batch_size]

            # Estimate the joint predictive distribution.
            probs_pool_batch = probs_pool_batch.permute(1, 0, 2)  # [K, batch_size, Cl]
            probs_targ_batch = probs_targ_batch.permute(1, 0, 2)  # [K, batch_size, Cl]
            probs_pool_batch = probs_pool_batch[
                :, :, None, :, None
            ]  # [K, batch_size, 1, Cl, 1]
            probs_targ_batch = probs_targ_batch[
                :, None, :, None, :
            ]  # [K, 1, batch_size, 1, Cl]
            probs_pool_targ_joint = probs_pool_batch * probs_targ_batch
            probs_pool_targ_joint = torch.mean(probs_pool_targ_joint, dim=0)

            # Estimate the marginal predictive distributions.
            probs_pool_batch = torch.mean(probs_pool_batch, dim=0)
            probs_targ_batch = torch.mean(probs_targ_batch, dim=0)

            # Estimate the product of the marginal predictive distributions.
            probs_pool_targ_indep = probs_pool_batch * probs_targ_batch

            # Estimate the conditional expected predictive information gain for each pair of examples.
            # This is the KL divergence between probs_pool_targ_joint and probs_pool_targ_joint_indep.
            nonzero_joint = probs_pool_targ_joint > 0
            log_term = torch.clone(probs_pool_targ_joint)
            log_term[nonzero_joint] = torch.log(probs_pool_targ_joint[nonzero_joint])
            log_term[nonzero_joint] -= torch.log(probs_pool_targ_indep[nonzero_joint])
            score_batch = torch.sum(probs_pool_targ_joint * log_term, dim=(-2, -1))

            # Store the results
            scores[i : i + batch_size, j : j + batch_size] = score_batch

    return scores  # [N_p, N_t]


def conditional_epig_from_values(
    values_pool: torch.Tensor,
    values_targ: torch.Tensor,
    batch_size: int = 1000,
) -> torch.Tensor:
    """
    Calculate conditional EPIG (Expected Predictive Information Gain)
    from continuous regression values.

    Arguments:
        values_pool: Tensor[float], [N_p, K]
            Continuous regression values for the pool set.
        values_targ: Tensor[float], [N_t, K]
            Continuous regression values for the target set.

    Returns:
        Tensor[float], [N_p, N_t]
            Conditional EPIG scores.
    """
    targ_mean = torch.mean(values_targ, dim=1)
    targ_mean = targ_mean.reshape(1, -1)

    num_samples_pool = values_pool.shape[0]

    scores_list = []

    for i in range(0, num_samples_pool, batch_size):

        values_pool_batch = values_pool[i : i + batch_size]

        # Estimate the joint predictive distribution.
        joint_mean_batch = torch.matmul(values_pool_batch, values_targ.unsqueeze(2))

        # Estimate the marginal predictive distributions.
        pool_mean_batch = torch.mean(values_pool_batch, dim=1)

        pool_mean_batch = pool_mean_batch.reshape(-1, 1)

        # Estimate the product of the marginal predictive distributions.
        indep_mean = pool_mean_batch * targ_mean

        # Estimate the conditional expected predictive information gain for each pair of examples.
        # This is the KL divergence between the joint predictive distribution and the product of the marginal predictive distributions.
        scores_list.append(
            torch.sum(
                joint_mean_batch
                * (torch.log(joint_mean_batch) - torch.log(indep_mean)),
                dim=1,
            )
        )

    scores = torch.cat(scores_list, dim=0)

    return scores


def conditional_epig_from_continuous(
    pred_pool: torch.Tensor, pred_targ: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the mean squared error (MSE) between the predicted values for pairs of examples.
    Suitable for regression models.

    Arguments:
        predictions_pool: Tensor[float], [N_p]
        predictions_targ: Tensor[float], [N_t]

    Returns:
        Tensor[float], [N_p, N_t]
    """
    # Reshape pred_pool and pred_targ to have 2D shape for broadcasting
    pred_pool = pred_pool.unsqueeze(1)  # [N_p, 1]
    pred_targ = pred_targ.unsqueeze(0)  # [1, N_t]

    # Calculate the joint predictive distribution for all pairs of examples
    joint_pred_dist = pred_pool - pred_targ  # [N_p, N_t]

    # Calculate the conditional expected predictive information gain
    scores = joint_pred_dist**2

    print(len(pred_pool), len(pred_targ))
    print(joint_pred_dist.shape)
    print(scores.shape)
    return scores  # [N_p, N_t]


def check(
    scores: torch.Tensor,
    max_value: float = math.inf,
    epsilon: float = 1e-6,
    score_type: str = "",
) -> torch.Tensor:
    """
    Warn if any element of scores is negative, a nan or exceeds max_value.

    We set epilson = 1e-6 based on the fact that torch.finfo(torch.float).eps ~= 1e-7.
    """
    if not torch.all((scores + epsilon >= 0) & (scores - epsilon <= max_value)):
        min_score = torch.min(scores).item()
        max_score = torch.max(scores).item()

        logging.warning(
            f"Invalid {score_type} score (min = {min_score}, max = {max_score})"
        )

    return scores


def epig_from_conditional_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    Arguments:
        scores: Tensor[float], [N_p, N_t]

    Returns:
        Tensor[float], [N_p,]
    """
    scores = torch.mean(scores, dim=-1)  # [N_p,]
    scores = check(scores, score_type="EPIG")  # [N_p,]
    return scores  # [N_p,]


def epig_from_probs(
    probs_pool: torch.Tensor, probs_targ: torch.Tensor, classification: str = True
) -> torch.Tensor:
    """
    See epig_from_logprobs.

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    if classification:
        scores = conditional_epig_from_probs(probs_pool, probs_targ)  # [N_p, N_t]
    else:
        scores = conditional_epig_from_values(probs_pool, probs_targ)  # [N_p, N_t]
    return epig_from_conditional_scores(scores)  # [N_p,]


class RFUncertainty:
    def __init__(self, fitted_model):
        super().__init__()
        self.fitted_model = fitted_model
    
    def _estimators(self):
        return self.fitted_model.estimators_
    
    def _predict(self, x):
        return self.fitted_model.predict(x)
    
    def _retrain(self, x, y, sample_weight = None, save_to_path = None):
        init_params = self.fitted_model.get_params()
        init_params['n_jobs'] = -1
        rf = RandomForestRegressor(**init_params)
        rf.fit(x, y, sample_weight = sample_weight)
        if save_to_path is not None:
            pickle.dump(rf, open(save_to_path, 'wb'))
    
    def _uncertainty(self, x):
        individual_trees = self.fitted_model.estimators_
        subEstimates = np.array(
            [tree.predict(x) for tree in individual_trees]
        )
        return np.std(subEstimates, axis=0)
    
    def _get_prob_distribution(self, x):
        prob_dist = [estimator.predict(x) for estimator in self._estimators()]
        prob_dist = np.stack(prob_dist, axis = 1)
        prob_dist = torch.tensor(prob_dist)
        return prob_dist
    
    def _estimate_epig(self, probs_pool, probs_target):
        return epig_from_probs(probs_pool, probs_target, classification = False)
    
def merge(
    sources: List[str],
    label: str,
    hparams_from: str,
    acquisition: str,
    sample_size: int = 20,
    uncertainty_threshold: float = 0.1,
    difference_threshold: float = 0.1,
    target_size: int = 1000,
    internal_path: str = str(REPO_DIR / 'database' / 'database.csv'),
    base_csv: str = None,
):
    
    df = pd.read_csv(internal_path)
    dedup = Dedup(df, label)
    df = dedup.run(sources)

    df_external = df[df['Source'].isin(sources)]
    smiles_list = df_external['SMILES'].tolist()
    props_dict = dict(zip(smiles_list, df_external[label].tolist()))
    def featurize_mols(smiles_list):
        features = []
        for smiles in tqdm(smiles_list, desc='Featurizing...'):
            rdmol = Chem.MolFromSmiles(smiles)
            mol_dict = ComposeFeaturizer(['rdkit2d'])(rdmol)
            features.append(mol_dict['descriptors'])
        features = torch.from_numpy(np.array(features)).squeeze(1)
        return features
    features_pool = featurize_mols(smiles_list)
    features_target = featurize_mols(smiles_list[:target_size])
    
    with open(hparams_from, 'rb') as f:
        rf_model = pickle.load(f)
    model = RFUncertainty(rf_model)

    if acquisition == 'epig':
        probs_pool = model._get_prob_distribution(features_pool)
        probs_target = model._get_prob_distribution(features_target)
        scores = model._estimate_epig(probs_pool, probs_target)
        query_idx = np.argsort(scores.numpy())[::-1][:sample_size] # select top 10 highest EPIG scores
        query_smiles = [smiles_list[i] for i in query_idx]
    elif acquisition == 'uncertainty':
        uncertainty = model._uncertainty(features_pool)
        eligible_idx = np.where(uncertainty < uncertainty_threshold)[0]
        if len(eligible_idx) <= sample_size:
            query_idx = eligible_idx
        else:
            sorted_idx = eligible_idx[np.argsort(uncertainty[eligible_idx])]
            query_idx = sorted_idx[:sample_size]
        query_smiles = [smiles_list[i] for i in query_idx]
    elif acquisition == 'difference':
        predictions = model._predict(features_pool)
        props_list = [props_dict[smiles] for smiles in smiles_list]
        relative_difference = np.abs(predictions - props_list) / props_list
        eligible_idx = np.where(relative_difference < difference_threshold)[0]
        if len(eligible_idx) <= sample_size:
            query_idx = eligible_idx
        else:
            sorted_idx = eligible_idx[np.argsort(relative_difference[eligible_idx])]
            query_idx = sorted_idx[:sample_size]
        query_smiles = [smiles_list[i] for i in query_idx]
    else:
        raise ValueError(f'Invalid acquisition method: {acquisition}')
    
    
    # df_internal = df[df['Source'] == 'internal']
    # df_internal = df_internal[df_internal[label].notna()]
    # df_internal = df_internal[['SMILES', label]]
    # query_smiles = [smiles for smiles in query_smiles if smiles not in df_internal['SMILES'].tolist()]
    ref_props = [props_dict[smiles] for smiles in query_smiles]
    df_add = pd.DataFrame(zip(query_smiles, ref_props), columns=['SMILES', label])
    
    if base_csv is not None:
        df_base = pd.read_csv(base_csv)[['SMILES', label]]
    else:
        df_base = df_internal
        
    df_merged = pd.concat([df_base, df_add], ignore_index=True)
    merge_path = str(REPO_DIR / 'database' / 'merged' / f'{label}_{"_".join(sources)}_{acquisition}.csv')
    df_merged.to_csv(merge_path, index=False)
    logger.info(f'Merged {len(df_add)} samples from {sources} to internal {len(df_base)}: {len(df_merged)}')
    return merge_path, len(df_merged),len(df_add)

def train(
    out_dir: str,
    model: str,
    label: str,
    feature_names: List[str],
    optimize_hparams: bool,
    hparams_from: str,
    raw_csv_path: str,
    sources: List[str],
    n_trials: int,
    tag: str,
    n_fold: int,
    model_path: str
) -> Tuple[float, float]:
    seed_everything(42)
    out_dir = os.path.join(out_dir, model)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, tag), exist_ok=True)
    out_dir = os.path.join(out_dir, tag)
    name = f'{model}-{label}-{"-".join(feature_names)}-{tag}'
    logger.add(os.path.join(out_dir, f'{name}.log'))
    model_type = model

    # 1. Load data
    logger.info(f'Training {label}...')
    logger.info(f'Feature names: {feature_names}')
    dataset = PolymerDataset(
        raw_csv_path=raw_csv_path,
        feature_names=feature_names,
        sources=sources,
        label_column=label,
        force_reload=True,
        add_hydrogens=False,
    )
    train_loader, val_loader, test_loader = dataset.get_loaders(
        batch_size=128,
        n_train=0.8,
        n_val=0.1,
    )
    x_train, y_train = loader2numpy(train_loader)
    x_val, y_val = loader2numpy(val_loader)
    x_test, y_test = loader2numpy(test_loader)
    
    # remove inf
    mask_train = ~np.isinf(x_train).any(axis=1)
    mask_val = ~np.isinf(x_val).any(axis=1)
    mask_test = ~np.isinf(x_test).any(axis=1)
    logger.info(f'Train inf: {len(x_train) - mask_train.sum()}, Val inf: {len(x_val) - mask_val.sum()}, Test inf: {len(x_test) - mask_test.sum()}')
    x_train = x_train[mask_train]
    y_train = y_train[mask_train]
    x_val = x_val[mask_val]
    y_val = y_val[mask_val]
    x_test = x_test[mask_test]
    y_test = y_test[mask_test]
    logger.info(f'Train size: {x_train.shape[0]}, Val size: {x_val.shape[0]}, Test size: {x_test.shape[0]}')
    
    # 2. Train model
    if not optimize_hparams:
        logger.info(f'Training {model}...')
        if hparams_from is not None:
            with open(hparams_from, 'rb') as f:
                hparams = pickle.load(f)
            model = MODELS[model](**hparams)
        else:
            model = MODELS[model]()
        if n_fold is not None:
            kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
            x_train = np.concatenate([x_train, x_val], axis=0)
            y_train = np.concatenate([y_train, y_val], axis=0)
            maes = []
            scaled_maes = []
            r2s = []
            y_pred_test = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(x_train)):
                logger.info(f'Training fold {fold+1}/{n_fold}...')
                x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
                y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx] 
                model.fit(x_train_fold, y_train_fold)
                y_hat = predict_batch(model, x_val_fold, batch_size=PREDICT_BATCH_SIZE)
                maes.append(mean_absolute_error(y_val_fold, y_hat))
                scaled_maes.append(scaling_error(y_val_fold, y_hat, label))
                r2s.append(r2_score(y_val_fold, y_hat))
                y_pred_test.append(predict_batch(model, x_test, batch_size=PREDICT_BATCH_SIZE))
            y_pred = np.mean(np.stack(y_pred_test, axis=0), axis=0)
            logger.info(f'Scaled MAE: {np.mean(scaled_maes): .4f}')
            logger.info(f'MAE: {np.mean(maes): .4f}')
            logger.info(f'R2: {np.mean(r2s): .4f}')
        else:
            model.fit(x_train, y_train)
            y_pred = predict_batch(model, x_test, batch_size=PREDICT_BATCH_SIZE)
            logger.info(f'Scaled MAE: {scaling_error(y_test, y_pred, label): .4f}')
            logger.info(f'MAE: {mean_absolute_error(y_test, y_pred): .4f}')
            logger.info(f'R2: {r2_score(y_test, y_pred): .4f}')
    elif optimize_hparams:
        logger.info(f'Optimizing hyper-parameters for {model}...')
        x_train = np.concatenate([x_train, x_val], axis=0)
        y_train = np.concatenate([y_train, y_val], axis=0)
        logger.info(f'Concatenating train and val data..., train size: {x_train.shape[0]}')
        def objective(trial: optuna.Trial, model: str = model) -> float:
            hparams = get_hparams(trial, model)
            model = MODELS[model](**hparams)
            if n_fold is not None:
                kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
                maes = []
                for fold, (train_idx, val_idx) in enumerate(kf.split(x_train)):
                    logger.info(f'Training fold {fold+1}/{n_fold}...')
                    logger.info(f'trial {trial.number+1}/{n_trials}')
                    x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
                    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
                    model.fit(x_train_fold, y_train_fold)
                    y_hat = predict_batch(model, x_val_fold, batch_size=PREDICT_BATCH_SIZE)
                    maes.append(mean_absolute_error(y_val_fold, y_hat))
                return float(np.mean(maes))
            else:
                model.fit(x_train, y_train)
                y_pred = predict_batch(model, x_val, batch_size=PREDICT_BATCH_SIZE)
                return mean_absolute_error(y_val, y_pred)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        logger.info(f'--------------------------------')
        logger.info(f'{name}')
        logger.info(f'Best hyper-parameters: {study.best_params}')
        hparams = get_hparams(study.best_trial, model)
        hparams.update(study.best_params)
        model = MODELS[model](**hparams)
        model.fit(x_train, y_train)
        y_pred = predict_batch(model, x_test, batch_size=PREDICT_BATCH_SIZE)
        
        logger.info(f'Scaled MAE: {scaling_error(y_test, y_pred, label): .4f}')
        logger.info(f'MAE: {mean_absolute_error(y_test, y_pred): .4f}')
        logger.info(f'R2: {r2_score(y_test, y_pred): .4f}')
    
    # 3. Train production model
    logger.info(f'Training production model...')
    model = MODELS[model_type](**hparams)
    X_total = np.concatenate([x_train, x_val, x_test], axis=0)
    y_total = np.concatenate([y_train, y_val, y_test], axis=0)
    model.fit(X_total, y_total)
    model.feature_names = feature_names
    
    # 4. Save model and results
    #model_path = os.path.join(out_dir, f'{name}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    hparams_path = model_path.replace('.pkl', '_hparams.pkl')
    with open(hparams_path, 'wb') as f:
        pickle.dump(hparams, f)
    if n_fold is None:  
        results_path = os.path.join(out_dir, f'{name}.csv')
        pd.DataFrame({
            'y_true': y_test,
            'y_pred': y_pred,
        }).to_csv(results_path, index=False)
    
    return scaling_error(y_test, y_pred, label), x_test.shape[0], hparams_path, model_path


class ActiveLearning:
    def __init__(
            self, 
            model: str, 
            label: str, 
            feature_names: List[str], 
            sources: List[str], 
            n_trials: int, 
            tag: str, 
            n_fold: int, 
            out_dir: str = REPO_DIR / 'scripts' / 'results' / 'active',
            model_base_path: str = REPO_DIR / 'scripts' / 'results' / 'active'
        ):
        self.model = model
        self.label = label
        self.feature_names = feature_names
        self.sources = sources
        self.n_trials = n_trials
        self.tag = tag
        self.n_fold = n_fold

        self.model_base_path = model_base_path
        self.out_dir = out_dir

        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.model_base_path, exist_ok=True)

        self.hparams_iter1: Optional[str] = None

    def _model_path(self, iter_idx: int) -> str:
        filename = f'{self.model}-{self.label}-{"-".join(self.feature_names)}-iter{iter_idx}.pkl'
        return os.path.join(self.model_base_path, filename)

    def _tag(self, iter_idx: int) -> str:
        return f'active-iter{iter_idx}'
    
    def _dataset_size(self, csv_path: str) -> int:
        df = pd.read_csv(csv_path)
        if self.label in df.columns:
            return int(df[self.label].notna().sum())
        return int(len(df))

    def run(
            self, 
            base_csv: str,
            n_iter: int = 10, 
            patience: int = 5, 
            acquisition: str = 'uncertainty',
            sample_size: int = 20,
            uncertainty_threshold: float = 0.1,
            difference_threshold: float = 0.1,
            target_size: int = 1000,
            internal_path: str = REPO_DIR / 'database' / 'database.csv',
            summary_csv: str = REPO_DIR / 'scripts' / 'results' / 'active' / 'summary.csv'
        ):
    
        logger.info(f'Start active learning loop...')
        summary = []
        dataset_iter: Dict[int, str] = {}
        dataset_iter[0] = base_csv

        best_mae = float('inf')
        patience_counter = 0

        for i in range(n_iter):
            logger.info(f'Iteration {i+1}/{n_iter}...')
            logger.info(f'First round, training model with hparams optimization') if i == 0 else logger.info(f'Training model at iteration {i+1}...')

            al_tag = self._tag(i+1)
            model_path = self._model_path(i+1)

            optimize_hparams = (i == 0)
            hparams_from = None

            if not optimize_hparams:
                if self.hparams_iter1 is None or not os.path.exists(self.hparams_iter1):
                    raise ValueError(f'Hparams file for iteration 1 not found')
                hparams_from = self.hparams_iter1
                print('hparams_from', hparams_from)

            train_csv = dataset_iter[i]
            scaling_error, n_test, prev_hparams_path, prev_model_path = train(
                out_dir=self.out_dir,
                model=self.model,
                label=self.label,
                feature_names=self.feature_names,
                optimize_hparams=optimize_hparams,
                hparams_from=hparams_from,
                raw_csv_path=train_csv,
                sources=self.sources,
                n_trials=self.n_trials,
                tag=al_tag,
                n_fold=self.n_fold,
                model_path=model_path
            )

            if optimize_hparams:
                self.hparams_iter1 = prev_hparams_path

            n_samples = self._dataset_size(train_csv)
            improvement = (best_mae - scaling_error) > 0.0

            if improvement:
                best_mae = scaling_error
                patience_counter = 0
            else:
                patience_counter += 1
            
            summary.append({
                'iter': i+1,
                'n_samples': n_samples,
                'scaling_error': scaling_error,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            if patience_counter >= patience:
                logger.info(f'Patience reached, stopping active learning...')
                break
            
            logger.info(f'Selecting samples...')
            merge_path, merge_size, added_count = merge(
                sources=self.sources,
                label=self.label,
                hparams_from=prev_model_path,
                acquisition=acquisition,
                sample_size=sample_size,
                uncertainty_threshold=uncertainty_threshold,
                difference_threshold=difference_threshold,
                target_size=target_size,
                internal_path=internal_path,
                base_csv=train_csv
            )

            summary[-1]['selected_count'] = int(added_count)
            if added_count == 0:
                logger.info(f'No samples added, stopping active learning...')
                break

            dataset_iter[i+1] = merge_path
        
        summary_df = pd.DataFrame(summary)
        if summary_csv is None:
            summary_csv = os.path.join(self.out_dir, 'al_summary.csv')
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f'Active learning summary saved to {summary_csv}')
        return summary_df
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['rf', 'xgb', 'lgbm', 'catboost'])
    parser.add_argument('--label', type=str, required=True, choices=['Tc', 'Tg', 'FFV', 'Density', 'Rg'])
    parser.add_argument('--feature-names', type=str, nargs='+', default=['rdkit2d'])
    parser.add_argument('--n-trials', type=int, default=10)
    parser.add_argument('--n-fold', type=int, default=5)
    parser.add_argument('--tag', type=str, default='debug')
    parser.add_argument('--out-dir', type=str, default=REPO_DIR / 'scripts' / 'results' / 'active')
    parser.add_argument('--sources', type=str, nargs='+', default=['MD'])
    parser.add_argument('--base-csv', type=str, default=REPO_DIR / 'database' / 'database.csv')
    parser.add_argument('--n-iter', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--acquisition', type=str, default='uncertainty')
    parser.add_argument('--sample-size', type=int, default=20)
    parser.add_argument('--uncertainty-threshold', type=float, default=0.1)
    parser.add_argument('--difference-threshold', type=float, default=0.1)
    parser.add_argument('--target-size', type=int, default=1000)
    parser.add_argument('--internal-path', type=str, default=REPO_DIR / 'database' / 'database.csv')
    parser.add_argument('--summary-csv', type=str, default=REPO_DIR / 'scripts' / 'results' / 'active' / 'summary.csv')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    al = ActiveLearning(
        model=args.model,
        label=args.label,
        feature_names=args.feature_names,
        sources=args.sources,
        n_trials=args.n_trials,
        n_fold=args.n_fold,
        out_dir=args.out_dir,
        tag=args.tag,
    )
    summary_df = al.run(
        base_csv=args.base_csv, 
        n_iter=args.n_iter, 
        patience=args.patience, 
        acquisition=args.acquisition,
        sample_size=args.sample_size,
        uncertainty_threshold=args.uncertainty_threshold,
        difference_threshold=args.difference_threshold,
        target_size=args.target_size,
        internal_path=args.internal_path,
        summary_csv=args.summary_csv
    )

if __name__ == '__main__':
    main()


    