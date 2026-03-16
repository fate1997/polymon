import os
from typing import List, Literal, Optional, Union
import math

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from polymon.cli.merge import epig_from_probs
from polymon.data.featurizer import ComposeFeaturizer
from polymon.data.polymer import Polymer
from polymon.model.base import BaseModel, ModelWrapper
from polymon.model.ensemble import EnsembleModelWrapper
from polymon.setting import UNIQUE_ATOM_NUMS


def get_evidential_stats(pred: torch.Tensor, eps: float = 0.0):
    """
    pred: (B, 4T) = [μ | logλ | logα | logβ]
    """
    # B, fourT = pred.shape
    # T = fourT // 4

    #mu, loglam, logalpha, logbeta = torch.split(pred, T, dim=-1)
    mu, lam, alpha, beta = torch.unbind(pred, dim=-1)
    #print(mu.min(), lam.min(), alpha.min(), beta.min())
    # lam = F.softplus(loglam) + 1e-6
    # alpha = F.softplus(logalpha) + 1.0 + 1e-6
    # beta = F.softplus(logbeta) + 1e-6

    # aleatoric_var = beta / (alpha - 1 + eps)
    # epistemic_var = beta / (lam * (alpha - 1 + eps))
    # total_var = aleatoric_var + epistemic_var
    aleatoric_var = beta / (alpha - 1 + eps)
    epistemic_var = beta / (lam * (alpha - 1 + eps))
    total_uncertainty = aleatoric_var + epistemic_var
    return mu, epistemic_var

class Acquisition:
    def __init__(
        self,
        acquisition_function: str,
        model_file: str,
        model_type: Literal['default', 'KFold', 'ensemble'] = 'default',
        n_fold: Optional[int] = None,
        device: str = 'cpu',
        ordered_tasks: Optional[Union[List[str], str]] = None,
        prev_hits: Optional[str] = None,
        prev_hits_tag: Optional[Union[List[str], str]] = None
        
    ):
        self.acquisition_function = acquisition_function
        self.device = device
        self.ordered_tasks = ordered_tasks
        self.prev_hits = prev_hits
        self.prev_hits_tag = prev_hits_tag
        
        if model_type == 'default':
            self.model = ModelWrapper.from_file(model_file)
        elif model_type == 'KFold':
            model_names = [os.path.join(f'./results/gatv2/{ordered_tasks}', model_file.split('/')[-3], f'fold_{i}', model_file.split('/')[-1]) for i in range(1, n_fold + 1)]
            models = [ModelWrapper.from_file(model_name) for model_name in model_names]
            self.model = models
            self.mean = models[0].normalizer.init_params['mean']
            self.std = models[0].normalizer.init_params['std']
        elif model_type == 'ensemble':
            info = torch.load(model_file)
            if self.ordered_tasks is not None:
                if isinstance(self.ordered_tasks, str):
                    self.ordered_tasks = [self.ordered_tasks]
                elif isinstance(self.ordered_tasks, list):
                    if len(self.ordered_tasks) == 1:
                        pass
                    else:
                        info['model_init_params']['num_tasks'] = len(self.ordered_tasks)
            #info['model_init_params']['num_tasks'] = 1 if not self.ordered_tasks else len(self.ordered_tasks)
            self.model = EnsembleModelWrapper.from_dict(info)
            self.mean = self.model.normalizer.init_params['mean']
            self.std = self.model.normalizer.init_params['std']

    # def __call__(self, pool_smiles: List[str]):
    #     return self.acquire(pool_smiles)

    def score(self, pool_smiles: List[str], train_smiles: Optional[List[str]] = None, prev_smiles: Optional[List[str]] = None):
        # remove train_smiles from pool_smiles
        # if self.prev_hits is not None:
        #     df_hits = pd.read_csv(self.prev_hits)
        #     if isinstance(self.prev_hits_tag, str):
        #         hits_smiles = df_hits[df_hits['Source'] == self.prev_hits_tag]['SMILES'].tolist()
        #     elif isinstance(self.prev_hits_tag, list):
        #         hits_smiles = df_hits[df_hits['Source'].isin(self.prev_hits_tag)]['SMILES'].tolist()
        # else:
        #     hits_smiles = []
        if train_smiles is None:
            train_smiles = []
        if prev_smiles is None:
            prev_smiles = []
        pool_smiles = [smiles for smiles in pool_smiles if smiles not in prev_smiles and smiles not in train_smiles]
        #print('Number of pool smiles after removing previous hits and train smiles:', len(pool_smiles))
        n_sample = len(pool_smiles)
        if self.acquisition_function == "uncertainty":
            return self.uncertainty(pool_smiles, n_sample)[2]
        elif self.acquisition_function == "margin":
            return self.margin(pool_smiles, n_sample)[1]
        elif self.acquisition_function == "bald":
            return self.bald(pool_smiles, n_sample)[1]
        elif self.acquisition_function == "similarity":
            return self.similarity(pool_smiles, n_sample)[1]
        elif self.acquisition_function == "epig":
            return self.epig(pool_smiles, n_sample, target_size = 1000)[1]
        elif self.acquisition_function == "ei":
            return self.expected_improvement(pool_smiles, n_sample)[1]
        elif self.acquisition_function == "fps":
            return self.fps(pool_smiles, query_smiles = None, sample_size = n_sample)[1]
        elif self.acquisition_function == "random":
            return self.random(pool_smiles, n_sample)[1]
    
    def acquire(
        self,
        pool_smiles: List[str],
        n_sample: int = 50,
        target_size: int = 1000,
        query_smiles: List[str] = None,
        train_smiles: List[str] = None
    ):
        if self.prev_hits is not None:
            df_hits = pd.read_csv(self.prev_hits)
            if isinstance(self.prev_hits_tag, str):
                hits_smiles = df_hits[df_hits['Source'] == self.prev_hits_tag]['SMILES'].tolist()
            elif isinstance(self.prev_hits_tag, list):
                hits_smiles = df_hits[df_hits['Source'].isin(self.prev_hits_tag)]['SMILES'].tolist()
        
        print('Length of pool smiles:', len(pool_smiles))
        if train_smiles is not None and self.prev_hits is not None:
            hits_smiles += train_smiles
            pool_smiles = [
                smiles for smiles in pool_smiles if smiles not in hits_smiles
            ]
            print('Length of pool smiles after removing previous hits:', len(pool_smiles))
        
        if train_smiles is not None:
            pool_smiles = [smiles for smiles in pool_smiles if smiles not in train_smiles]

        if self.acquisition_function == "uncertainty":
            return self.uncertainty(pool_smiles, n_sample)[0]
        elif self.acquisition_function == "margin":
            return self.margin(pool_smiles, n_sample)[0]
        elif self.acquisition_function == "epig":
            return self.epig(pool_smiles, n_sample, target_size)[0]
        elif self.acquisition_function == "bald":
            return self.bald(pool_smiles, n_sample)[0]
        elif self.acquisition_function == "fps":
            return self.fps(pool_smiles, query_smiles, sample_size = n_sample)[0]
        elif self.acquisition_function == "similarity":
            return self.similarity(pool_smiles, n_sample)[0]
        elif self.acquisition_function == "random":
            return self.random(pool_smiles, n_sample)[0]
    
    def uncertainty(self, pool_smiles: List[str], n_sample: int = 50) -> List[str]:
        """
        Compute uncertainty as the average (over tasks) of estimator std for each molecule.
        preds shape: (num_estimators, num_pool, num_tasks)
        """
          # shape: (num_estimators, num_pool, num_tasks)
        if isinstance(self.model, EnsembleModelWrapper):
            all_preds = self.get_preds(pool_smiles)
            uncertainty_per_task = all_preds.var(dim=0, unbiased=False)
            uncertainty_per_task_norm = uncertainty_per_task / self.std.to(uncertainty_per_task.device)
            if uncertainty_per_task_norm.ndim == 2:
                uncertainty = uncertainty_per_task_norm.mean(dim=-1)
            else:
                uncertainty = uncertainty_per_task_norm
            # uncertainty = uncertainty_per_task_norm.mean(dim=-1)
            # #uncertainty = uncertainty_per_task_norm
            top_idx = torch.topk(uncertainty, n_sample).indices.tolist()
            top_smiles = [pool_smiles[i] for i in top_idx]
            return top_smiles, uncertainty_per_task_norm, uncertainty
            # mu = preds[..., :self.model.num_tasks]     
            # epistemic_vars = []
            # ens_epi = mu.var(dim=0)   
            # for m in range(preds.shape[0]):
            #     _, _, epi, _ = get_evidential_stats(preds[m])
            #     epistemic_vars.append(epi)

            # epistemic_vars = torch.stack(epistemic_vars, dim=0)  # (num_models, B, T)

            # # Average over models, then tasks
            # #uncertainty = epistemic_vars.mean(dim=0).mean(dim=-1)  # (B,)
            # epi_evid = epistemic_vars.mean(dim=0)
            # uncertainty = (ens_epi + epi_evid).mean(dim=-1)
            # top_idx = torch.topk(uncertainty, n_sample).indices.tolist()
            # top_smiles = [pool_smiles[i] for i in top_idx]
            # return top_smiles, uncertainty

        elif isinstance(self.model, ModelWrapper):
            preds, uncertainty = self.get_preds(pool_smiles)
            #_, _, epi, _ = get_evidential_stats(preds)  # epi: (B, T)

            # Average over tasks
            #uncertainty = epi.mean(dim=-1)  # (B,)
            n_sample = min(n_sample, len(pool_smiles))
            top_idx = torch.topk(uncertainty.mean(dim=-1), n_sample).indices.tolist()
            top_smiles = [pool_smiles[i] for i in top_idx]

            return top_smiles, uncertainty

    def bald(self, pool_smiles: List[str], n_sample: int = 50, eps: float = 1e-8):
        """
        BALD-style acquisition for evidential ensemble regression.
        Uses ensemble disagreement (epistemic) + evidential aleatoric variance.
        """

        preds = self.get_raw_preds(pool_smiles)  # (E, B, 4T)
        E, B, fourT = preds.shape
        T = fourT // 4

        mus = []
        ales = []

        for m in range(E):
            mu, ale, _, _ = get_evidential_stats(preds[m])
            mus.append(mu)    # (B, T)
            ales.append(ale)  # (B, T)

        mus = torch.stack(mus, dim=0)    # (E, B, T)
        ales = torch.stack(ales, dim=0)  # (E, B, T)

        # Ensemble epistemic: disagreement of means
        epi_ens = mus.var(dim=0)         # (B, T)

        # Expected aleatoric variance
        ale_mean = ales.mean(dim=0)      # (B, T)

        # BALD score per task
        bald = 0.5 * torch.log(
            (epi_ens + ale_mean) / (ale_mean + eps)
        )                                # (B, T)

        # Aggregate across tasks
        bald_score = bald.mean(dim=-1)   # (B,)

        top_idx = torch.topk(bald_score, n_sample).indices.tolist()
        return [pool_smiles[i] for i in top_idx], bald_score

    
    def epig(self, pool_smiles: List[str], sample_size: int = 50, target_size: int = 1000):
        preds_pool = self.get_preds(pool_smiles)  # (num_models, B, 4T)

        if target_size > len(pool_smiles):
            target_smiles = pool_smiles
        else:
            target_smiles = np.random.choice(pool_smiles, size=target_size, replace=False)

        preds_target = self.get_preds(target_smiles)
        if len(self.ordered_tasks) == 1:
            preds_target = preds_target.unsqueeze(-1)
            preds_pool = preds_pool.unsqueeze(-1)
        epig_scores = []
        for i in range(preds_pool.shape[-1]):
            prob_pool_task = preds_pool[..., i].T.detach().cpu()  # (num_estimators, num_pool)
            prob_target_task = preds_target[..., i].T.detach().cpu()  # (num_estimators, target_size)
            # epig_from_probs expects input shape: (num_estimators, num_pool/target)
            score_task = epig_from_probs(prob_pool_task, prob_target_task, classification=False)  # (num_pool,)
            weight_per_task = 1 / score_task.std(dim=0)
            score_task = score_task * weight_per_task
            #print(score_task[:10])
            epig_scores.append(score_task)
        # for m in range(preds_pool.shape[0]):
        #     _, _, epi_pool, _ = get_evidential_stats(preds_pool[m])
        #     _, ale_t, epi_t, _ = get_evidential_stats(preds_target[m])

        #     denom = (ale_t + epi_t).mean(dim=0, keepdim=True)
        #     score = torch.log(epi_pool.mean(dim=-1) / (denom.mean(dim=-1) + 1e-8))
        #     epig_scores.append(score)
        epig_score = torch.stack(epig_scores, dim=0).mean(dim=0)

        top_idx = torch.topk(epig_score, sample_size).indices.tolist()
        return [pool_smiles[i] for i in top_idx], epig_score

    
    def margin(self, pool_smiles: List[str]):
        pass
    
    def similarity(self, pool_smiles: List[str]):
        pass

    def expected_improvement(self, pool_smiles: List[str], n_sample: int = 50) -> List[str]:
        """
        Expected Improvement acquisition for regression.
        Selects molecules with high expected improvement over current best property.
        """
        preds = self.get_preds(pool_smiles)  # (num_estimators, num_pool, num_tasks)
        mu = preds.mean(0)  # (num_pool, num_tasks)
        sigma = preds.std(0) + 1e-9  # avoid div by zero
        best = self.y_train.max(0).values  # current best for each task

        z = (mu - best) / sigma
        ei = (mu - best) * torch.distributions.Normal(0, 1).cdf(z) + sigma * torch.distributions.Normal(0, 1).log_prob(z).exp()
        ei = ei.mean(-1)  # average across tasks

        top_indices = torch.topk(ei, n_sample).indices.tolist()
        top_smiles = [pool_smiles[i] for i in top_indices]
        return top_smiles, ei
    
    def fps(self, pool_smiles: List[str], query_smiles: List[str] = None, sample_size: int = 50):
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.DataStructs import BulkTanimotoSimilarity
        mfgen = rdFingerprintGenerator.GetMorganGenerator(4, fpSize=2048)
        fps_pool = [mfgen.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in pool_smiles]
        if query_smiles is not None:
            fps_query = [mfgen.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in query_smiles]
        else:
            query_df = pd.read_csv('/home/rengp/projects/YY/polyactive/database/sampled/database_sample_1it.csv')
            query_smiles = query_df.loc[query_df['Source'].isin(['FFV-Active', 'PI1070', 'initial'])]['SMILES'].tolist()
            fps_query = [mfgen.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in query_smiles]
        sims = [
            max(BulkTanimotoSimilarity(fp_pool, fps_query))
            for fp_pool in fps_pool
        ]
        sims = torch.tensor(sims)
        scores = 1.0 - sims
        #print('Top scores:', scores.topk(10).values.tolist())
        top_n_sample = torch.topk(scores, sample_size).indices.tolist()
        top_smiles = [pool_smiles[i] for i in top_n_sample]
        return top_smiles, scores
    
    def random(self, pool_smiles: List[str], n_sample: int = 50):
        query_idx = np.random.choice(len(pool_smiles), size=n_sample, replace=False)
        scores = torch.ones(n_sample).to(self.device)
        return [pool_smiles[i] for i in query_idx], scores
    
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
                    for batch in loader:
                        batch = batch.to(self.device)
                        pred = estimator(batch)
                        pred = self.model.normalizer.inverse(pred)
                        preds.append(pred)
                    preds = torch.cat(preds, dim=0)
                    all_preds.append(preds)
            all_preds = torch.stack(all_preds, dim=0).squeeze(-1)
            return all_preds
        elif isinstance(self.model, ModelWrapper):
            model = self.model
            model.eval()
            model.to(self.device)
            preds= model.predict(pool_smiles, batch_size=128, return_uncertainty=True)
            #mean = self.model.normalizer.inverse(preds[0])
            mean = preds[0]
            uncertainty = preds[1]
            #uncertainty = self.model.normalizer.inverse(preds[1])
            return mean, uncertainty
        
        elif isinstance(self.model, list):
            all_preds = []
            for model in self.model:
                model.eval()
                model.to(self.device)
                preds = model.predict(pool_smiles, batch_size=128)
                preds = model.normalizer.inverse(preds)
                all_preds.append(preds)
            all_preds = torch.stack(all_preds, dim=0)  # shape: (num_models, num_samples, num_tasks)
            return all_preds

    def _get_loader(self, pool_smiles: List[str]):
        # config = {}
        # config['x'] = {'unique_atom_nums': UNIQUE_ATOM_NUMS}
        # featurizer = ComposeFeaturizer(['x', 'bond', 'z'], config)
        featurizer = self.model.featurizer
        polymers = []
        for smiles in tqdm(pool_smiles, desc='Featurizing'):
            rdmol = Chem.MolFromSmiles(smiles)
            mol_dict = featurizer(rdmol)
            mol_dict['smiles'] = smiles
            polymer = Polymer(**mol_dict)
            polymers.append(polymer)
        return DataLoader(polymers, batch_size=128)

    def _enable_dropout(self, model: BaseModel):
        for module in model._modules['predict'].layers.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()