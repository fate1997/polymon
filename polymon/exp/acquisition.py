import os
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from polymon.cli.merge import epig_from_probs
from polymon.data.featurizer import ComposeFeaturizer
from polymon.data.polymer import Polymer
from polymon.model.base import BaseModel, ModelWrapper
from polymon.model.ensemble import EnsembleModelWrapper
from polymon.setting import UNIQUE_ATOM_NUMS


class Acquisition:
    def __init__(
        self,
        acquisition_function: str,
        model_file: str,
        model_type: Literal['default', 'KFold', 'ensemble'] = 'default',
        n_fold: Optional[int] = None,
        device: str = 'cpu',
        ordered_tasks: Optional[List[str]] = None,
        prev_hits: Optional[str] = None,
        prev_hits_tag: Optional[List[str]] = None,
    ):
        self.acquisition_function = acquisition_function
        self.device = device
        self.ordered_tasks = ordered_tasks
        self.prev_hits = prev_hits
        self.prev_hits_tag = prev_hits_tag

        if model_type == 'default':
            self.model = ModelWrapper.from_file(model_file)
        elif model_type == 'KFold':
            model_names = [os.path.join(model_file.split('/')[-3], f'fold_{i}', model_file.split('/')[-1]) for i in range(1, n_fold + 1)]
            models = [ModelWrapper.from_file(model_name) for model_name in model_names]
            self.model = models
            self.mean = models[0].normalizer.init_params['mean']
            self.std = models[0].normalizer.init_params['std']
        elif model_type == 'ensemble':
            info = torch.load(model_file)
            info['model_init_params']['num_tasks'] = 1 if not self.ordered_tasks else len(self.ordered_tasks)
            self.model = EnsembleModelWrapper.from_dict(info)
            self.mean = self.model.normalizer.init_params['mean']
            self.std = self.model.normalizer.init_params['std']

    def __call__(self, pool_smiles: List[str]):
        return self.acquire(pool_smiles)

    def score(self, pool_smiles: List[str]):
        n_sample = len(pool_smiles)
        if self.acquisition_function == "uncertainty":
            return self.uncertainty(pool_smiles, n_sample)[1]
        elif self.acquisition_function == "margin":
            return self.margin(pool_smiles, n_sample)[1]
        elif self.acquisition_function == "epig":
            return self.epig(pool_smiles, n_sample, target_size = 1000)[1]
        elif self.acquisition_function == "bald":
            return self.bald(pool_smiles, n_sample)[1]
        elif self.acquisition_function == "similarity":
            return self.similarity(pool_smiles, n_sample)[1]
    
    def acquire(
        self,
        pool_smiles: List[str],
        n_sample: int = 50,
        target_size: int = 1000,
        query_smiles: List[str] = None,
        train_smiles: List[str] = None,
    ):
        if self.prev_hits is not None:
            df_prev_hits = pd.read_csv(self.prev_hits)[['SMILES', 'Source']]
            if isinstance(self.prev_hits_tag, str):
                hits_smiles = df_prev_hits[df_prev_hits['Source'] == self.prev_hits_tag]['SMILES'].tolist()
            elif isinstance(self.prev_hits_tag, list):
                hits_smiles = df_prev_hits[df_prev_hits['Source'].isin(self.prev_hits_tag)]['SMILES'].tolist()
            else:
                raise ValueError(f'Invalid prev_hits_tag: {self.prev_hits_tag}')
        
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
        elif self.acquisition_function == "ei":
            return self.expected_improvement(pool_smiles, n_sample)[0]
        elif self.acquisition_function == "fps":
            return self.fps(pool_smiles, query_smiles, n_sample)[0]
        elif self.acquisition_function == "similarity":
            return self.similarity(pool_smiles, n_sample)[0]
        elif self.acquisition_function == "random":
            return self.random(pool_smiles, n_sample)[0]
    
    def uncertainty(self, pool_smiles: List[str], n_sample: int = 50) -> List[str]:
        """
        Compute uncertainty as the average (over tasks) of estimator std for each molecule.
        preds shape: (num_estimators, num_pool, num_tasks)
        """
        preds = self.get_preds(pool_smiles)
        uncertainty_per_task = preds.std(0)
        uncertainty_per_task_norm = uncertainty_per_task / self.std.to(uncertainty_per_task.device)
        uncertainty = uncertainty_per_task_norm.mean(-1)
        top_n_sample = torch.topk(uncertainty, n_sample).indices.tolist()
        top_smiles = [pool_smiles[i] for i in top_n_sample]
        return top_smiles, uncertainty

    def bald(self, pool_smiles: List[str], n_sample: int = 50, eps: float = 1e-12) -> List[str]:
        """
        Compute BALD acquisition scores averaged over all tasks.

        mc_preds shape: (num_passes, num_estimators, num_pool, num_tasks)
        """
        mc_preds = self.get_mc_preds(n_passes=20, pool_smiles=pool_smiles, if_drop=True).detach().cpu()
        mc_preds = (mc_preds - self.mean.to(mc_preds.device)) / self.std.to(mc_preds.device)
        # Calculate total variance: shape (num_pool, num_tasks)
        var_total = mc_preds.var(dim=(0, 1), unbiased=False)  # (num_pool, num_tasks)
        # Calculate estimator variance: shape (num_passes, num_pool, num_tasks)
        var_est = mc_preds.var(dim=1, unbiased=False)  # (num_passes, num_pool, num_tasks)
        # Aleatoric variance: mean across passes -> shape (num_pool, num_tasks)
        var_ale = var_est.mean(dim=0)  # (num_pool, num_tasks)
        # Epistemic variance: shape (num_pool, num_tasks)
        var_epi = torch.clamp(var_total - var_ale, min=0.0)
        # BALD score for each pool and task
        bald = 0.5 * torch.log1p(var_epi / (var_ale + eps))  # (num_pool, num_tasks)
        # Average BALD score over all tasks to get a unified score
        bald_score = bald.mean(dim=-1)  # (num_pool,)
        # Get top n_sample molecules with largest BALD score
        top_n_sample = torch.topk(bald_score, n_sample).indices.tolist()
        top_smiles = [pool_smiles[i] for i in top_n_sample]
        return top_smiles, bald_score
    
    def epig(self, pool_smiles: List[str], sample_size: int = 50, target_size: int = 1000):
        # get_preds now returns (num_estimators, num_pool, num_tasks)
        # We want a unified EPIG score by averaging over all task scores for each molecule
        preds_pool = self.get_preds(pool_smiles)  # (num_estimators, num_pool, num_tasks)
        #preds_pool = (preds_pool - self.mean.to(preds_pool.device)) / self.std.to(preds_pool.device)
        target_smiles = np.random.choice(pool_smiles, size=target_size, replace=False)
        preds_target = self.get_preds(target_smiles)  # (num_estimators, target_size, num_tasks)
        #preds_target = (preds_target - self.mean.to(preds_target.device)) / self.std.to(preds_target.device)
        # For epig_from_probs, we need shape: (num_estimators, num_pool/target, num_tasks)
        # Compute scores for each task, then average
        num_tasks = preds_pool.shape[-1]
        scores_all_tasks = []
        for i in range(num_tasks):
            prob_pool_task = preds_pool[..., i].T.detach().cpu()  # (num_estimators, num_pool)
            prob_target_task = preds_target[..., i].T.detach().cpu()  # (num_estimators, target_size)
            
            # epig_from_probs expects input shape: (num_estimators, num_pool/target)
            score_task = epig_from_probs(prob_pool_task, prob_target_task, classification=False)  # (num_pool,)
            weight_per_task = 1 / score_task.std(dim=0)
            score_task = score_task * weight_per_task
            #print(score_task[:10])
            scores_all_tasks.append(score_task)
        # Stack per-task scores to (num_tasks, num_pool), then average -> (num_pool,)
        scores_all_tasks = torch.stack(scores_all_tasks, dim=0)
        avg_scores = scores_all_tasks.mean(dim=0)
        # print(avg_scores)
        # Select top molecules by unified average score
        top_n_sample = torch.topk(avg_scores, sample_size).indices.tolist()
        top_smiles = [pool_smiles[i] for i in top_n_sample]
        return top_smiles, avg_scores
    
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
        top_n_sample = np.argsort(scores)[:sample_size]
        top_smiles = [pool_smiles[i] for i in top_n_sample]
        return top_smiles, scores
    
    def random(self, pool_smiles: List[str], n_sample: int = 50):
        query_idx = np.random.choice(len(pool_smiles), size=n_sample, replace=False, seed=42)
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
                    for batch in loader:
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
        
        elif isinstance(self.model, list):
            all_preds = []
            for model in self.model:
                model.eval()
                model.to(self.device)
                preds = model.predict(pool_smiles, batch_size=128)
                preds = self.model.normalizer.inverse(preds)
                all_preds.append(preds)
            return all_preds

    @torch.no_grad()
    def get_mc_preds(
        self,
        n_passes: int = 20,
        pool_smiles: List[str] = None,
        if_drop: bool = False,
    ) -> torch.Tensor:
        assert isinstance(self.model, EnsembleModelWrapper) or (isinstance(self.model, ModelWrapper))
        if isinstance(self.model, EnsembleModelWrapper):
            model = self.model.model
        
        loader = self._get_loader(pool_smiles)

        all_preds = []
        for i in tqdm(range(n_passes), desc='MC Dropout'):
            torch.manual_seed(i)
            model.eval()
            model.to(self.device)
            pass_preds = []
            for j, estimator in enumerate(model.estimators_):
                estimator.eval()
                estimator.to(self.device)
                if if_drop:
                    self._enable_dropout(estimator)
                est_preds = []
                for batch in loader:
                    with torch.no_grad():
                        batch = batch.to(self.device)
                        pred = estimator(batch)
                        pred = self.model.normalizer.inverse(pred)
                        est_preds.append(pred)
                est_preds = torch.cat(est_preds, dim=0).squeeze(-1)  # shape: (num_data,)
                pass_preds.append(est_preds)
            pass_preds = torch.stack(pass_preds, dim=0)  # shape: (num_estimators, num_data)
            all_preds.append(pass_preds)
        all_preds = torch.stack(all_preds, dim=0)  # shape: (num_passes, num_estimators, num_data)
        return all_preds

    def _get_loader(self, pool_smiles: List[str]):
        config = {}
        config['x'] = {'unique_atom_nums': UNIQUE_ATOM_NUMS}
        featurizer = ComposeFeaturizer(['x', 'bond', 'z'], config)
        polymers = []
        for smiles in pool_smiles:
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
