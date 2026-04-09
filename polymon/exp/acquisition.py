import logging
import math
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from polymon.cli.merge import epig_from_probs
from polymon.data.polymer import Polymer
from polymon.model.base import BaseModel, ModelWrapper
from polymon.model.ensemble import EnsembleModelWrapper


class Acquisition:
    def __init__(
        self,
        acquisition_function: str,
        model_file: str,
        model_type: Literal['kfold', 'ensemble'] = 'kfold',
        device: str = 'cpu',
        prev_hits: Optional[str] = None,
        prev_hits_tag: Optional[Union[List[str], str]] = None
        
    ):
        self.acquisition_function = acquisition_function
        self.device = device
        self.prev_hits = prev_hits
        self.prev_hits_tag = prev_hits_tag
        
        if model_type == 'kfold':
            self.model = ModelWrapper.from_file(model_file)
        elif model_type == 'ensemble':
            info = torch.load(model_file)
            self.model = EnsembleModelWrapper.from_dict(info)
            self.mean = self.model.normalizer.init_params['mean']
            self.std = self.model.normalizer.init_params['std']

    def score(self, pool_smiles: List[str]):
        n_sample = len(pool_smiles)
        if self.acquisition_function == "uncertainty":
            return self.uncertainty(pool_smiles, n_sample)
        elif self.acquisition_function == "margin":
            return self.margin(pool_smiles, n_sample)
        elif self.acquisition_function == "random":
            return self.random(pool_smiles, n_sample)
        else:
            raise ValueError(f'Invalid acquisition function: {self.acquisition_function}')
    
    def uncertainty(self, pool_smiles: List[str], n_sample: int = 50) -> List[str]:
        """
        Compute uncertainty as the average (over tasks) of estimator std for each molecule.
        preds shape: (num_estimators, num_pool, num_tasks)
        """
        preds = self.get_preds(pool_smiles)
        uncertainty = preds.std(-1) 
        return uncertainty

    
    def epig(self, pool_smiles: List[str], sample_size: int = 50, target_size: int = 1000):
        preds_pool = self.get_preds(pool_smiles)  
        target_smiles = np.random.choice(pool_smiles, size=target_size, replace=False)
        preds_target = self.get_preds(target_smiles) 
        num_tasks = preds_pool.shape[-1]
        scores_all_tasks = []
        for i in range(num_tasks):
            prob_pool_task = preds_pool[..., i].T.detach().cpu()  
            prob_target_task = preds_target[..., i].T.detach().cpu()  
        
            score_task = epig_from_probs(prob_pool_task, prob_target_task, classification=False)  # (num_pool,)
            weight_per_task = 1 / score_task.std(dim=0)
            score_task = score_task * weight_per_task
            scores_all_tasks.append(score_task)
        scores_all_tasks = torch.stack(scores_all_tasks, dim=0)
        avg_scores = scores_all_tasks.mean(dim=0)
        return avg_scores
    
    
    def fps(self, pool_smiles: List[str], query_smiles: List[str], sample_size: int = 50):
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.DataStructs import BulkTanimotoSimilarity
        mfgen = rdFingerprintGenerator.GetMorganGenerator(4, fpSize=2048)
        fps_pool = [mfgen.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in pool_smiles]
        fps_query = [mfgen.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in query_smiles]
        scores = [
            max(BulkTanimotoSimilarity(fp_pool, fps_query))
            for fp_pool in fps_pool
        ]
        return scores
    
    def random(self, pool_smiles: List[str], n_sample: int = 50):
        query_idx = np.random.choice(len(pool_smiles), size=n_sample, replace=False)
        return [pool_smiles[i] for i in query_idx]
    
    @torch.no_grad()
    def get_preds(self, pool_smiles: List[str]):
        if isinstance(self.model, EnsembleModelWrapper):
            model = self.model.model
            all_preds = []
            loader = self._get_loader(pool_smiles)
            with torch.no_grad():
                for i, estimator in enumerate(model.estimators_):
                    estimator.eval()
                    estimator.to(self.device)
                    preds = []
                    for batch in tqdm(loader, desc = f'Estimator {i+1} inference'):
                        batch = batch.to(self.device)
                        pred = estimator(batch)
                        pred = self.model.normalizer.inverse(pred)
                        preds.append(pred)
                    preds = torch.cat(preds, dim=0)
                    # preds = self.model.normalizer.inverse(preds)
                    all_preds.append(preds)
            all_preds = torch.stack(all_preds, dim=0).squeeze(-1)
            return all_preds
        elif isinstance(self.model, ModelWrapper):
            model = self.model
            model.eval()
            model.to(self.device)
            preds = model.predict(pool_smiles, batch_size=128)
            preds = self.model.normalizer.inverse(preds)
            return preds

    def _get_loader(self, pool_smiles: List[str]):
        featurizer = self.model.featurizer
        polymers = []
        for smiles in tqdm(pool_smiles, desc='Featurizing'):
            rdmol = Chem.MolFromSmiles(smiles)
            mol_dict = featurizer(rdmol)
            mol_dict['smiles'] = smiles
            polymer = Polymer(**mol_dict)
            polymers.append(polymer)
        return DataLoader(polymers, batch_size=128)


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
