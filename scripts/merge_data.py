import argparse
import logging
import math
import pickle
from typing import List

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from polymon.data.dedup import Dedup
from polymon.data.featurizer import ComposeFeaturizer
from polymon.setting import REPO_DIR


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
    internal_path: str = str(REPO_DIR / 'database' / 'database.csv')
):
    
    df = pd.read_csv(internal_path)
    dedup = Dedup(df, label)
    df = dedup.run(sources)

    df_external = df[df['Source'].isin(sources)]
    smiles_list = df_external['SMILES'].tolist()
    props_dict = dict(zip(smiles_list, df_external[label].tolist()))
    def featurize_mols(smiles_list):
        features = []
        for smiles in tqdm(smiles_list):
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
    
    
    df_internal = df[df['Source'] == 'internal']
    df_internal = df_internal[df_internal[label].notna()]
    df_internal = df_internal[['SMILES', label]]
    query_smiles = [smiles for smiles in query_smiles if smiles not in df_internal['SMILES'].tolist()]
    ref_props = [props_dict[smiles] for smiles in query_smiles]
    df_add = pd.DataFrame(zip(query_smiles, ref_props), columns=['SMILES', label])
    df_merged = pd.concat([df_internal, df_add], ignore_index=True)
    merge_path = str(REPO_DIR / 'database' / 'merged' / f'{label}_{"_".join(sources)}_{acquisition}.csv')
    df_merged.to_csv(merge_path, index=False)
    print(f'Merged {sample_size} samples from {sources} to internal {len(df_internal)}: {len(df_merged)}')

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=str, required=True, nargs='+')
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--hparams-from', type=str, required=True)
    parser.add_argument('--acquisition', type=str, required=True, choices=['epig', 'uncertainty', 'difference'])
    parser.add_argument('--sample-size', type=int, default=20)
    parser.add_argument('--uncertainty-threshold', type=float, default=0.1)
    parser.add_argument('--difference-threshold', type=float, default=0.1)
    parser.add_argument('--target-size', type=int, default=1000)
    return parser.parse_args()

def main():
    args = arg_parser()
    merge(
        sources=args.sources,
        label=args.label,
        hparams_from=args.hparams_from,
        acquisition=args.acquisition,
        sample_size=args.sample_size,
        uncertainty_threshold=args.uncertainty_threshold,
        difference_threshold=args.difference_threshold,
        target_size=args.target_size,
    )

if __name__ == '__main__':
    main()