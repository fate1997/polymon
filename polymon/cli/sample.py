import argparse
import os
import pathlib
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity, ConvertToNumpyArray
from sklearn.cluster import DBSCAN, KMeans
from tqdm import tqdm
from collections import defaultdict
from polymon.exp.acquisition import Acquisition


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles-cluster-file', type=str, default=None)
    parser.add_argument('--train-file', type=str, default=None)
    parser.add_argument('--model-file', type=str, default=None)
    parser.add_argument('--model-type', type=str, default='ensemble')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ordered-tasks', type=str, default=['Rg', 'Density', 'Bulk_modulus', 'FFV', 'PLD', 'CLD'])
    parser.add_argument('--acquisition-function', type=str, default='uncertainty')
    parser.add_argument('--output-file', type=str, default='smiles_scores.csv')
    parser.add_argument('--all-clusters-file', type=str, default='all_clusters.npy')
    parser.add_argument('--n-sample', type=int, default=50)
    parser.add_argument('--sample-tag', type=str, default='AL1-Uncertainty')
    return parser.parse_args()

INITIAL_SOURCES = ['initial', 'PI1070', 'FFV-Active']

# def sample_representative_smiles(
#     smiles_list: List[str],
#     n_clusters: int,
#     radius: int = 4,
#     n_bits: int = 2048,
#     random_state: Optional[int] = None,
#     cluster_method: str = 'kmeans',
# ):
#     # generate fingerprints for valid molecules
#     fps, valid_smiles = [], []
#     for smi in tqdm(smiles_list, desc='Generating fingerprints'):
#         mol = Chem.MolFromSmiles(smi)
#         if mol:
#             mfgen = rdFingerprintGenerator.GetMorganGenerator(radius, fpSize=n_bits)
#             fps.append(mfgen.GetFingerprint(mol))
#             valid_smiles.append(smi)
#     if len(fps) < n_clusters:
#         raise ValueError(f'Not enough molecules: {len(fps)} < n_clusters ({n_clusters})')
    
#     # convert fingerprints to numpy array
#     arr = np.zeros((len(fps), n_bits), dtype=int)
#     for i, fp in enumerate(fps):
#         ConvertToNumpyArray(fp, arr[i])
    
#     # perform k-means clustering
#     if cluster_method == 'kmeans':
#         print('Performing k-means clustering...')
#         kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
#         labels = kmeans.fit_predict(arr)
#         cluster_members = {cid: np.where(labels == cid)[0] for cid in range(n_clusters)}
#         rep_smiles = []
#         rep_indices = []
#         for cid in tqdm(range(n_clusters), desc='Selecting representatives'):
#             idxs = np.where(labels == cid)[0]
#             cluster_fps = [fps[i] for i in idxs]
#             # compute sum of pairwise Tanimoto similarities for each member
#             scores = [sum(BulkTanimotoSimilarity(fp, cluster_fps)) for fp in cluster_fps]
#             best_idx = idxs[int(np.argmax(scores))]
#             rep_smiles.append(valid_smiles[best_idx])
#             rep_indices.append(best_idx)
#     elif cluster_method == 'dbscan':
#         print('Performing DBSCAN clustering...')
#         dbscan = DBSCAN(eps=0.5, min_samples=100, metric='jaccard')
#         labels = dbscan.fit_predict(arr)
#         unique_labels = set(labels)
#         if -1 in unique_labels:
#             unique_labels.remove(-1)
#             found_clusters = sorted(list(unique_labels))
#             cluster_members = {cid: np.where(labels == cid)[0] for cid in found_clusters}
#         rep_smiles = []
#         rep_indices = []
#         for cid in tqdm(found_clusters, desc='Selecting representatives'):
#             idxs = np.where(labels == cid)[0]
#             cluster_fps = [fps[i] for i in idxs]
#             # compute sum of pairwise Tanimoto similarities for each member
#             scores = [sum(BulkTanimotoSimilarity(fp, cluster_fps)) for fp in cluster_fps]
#             best_idx = idxs[int(np.argmax(scores))]
#             rep_smiles.append(valid_smiles[best_idx])
#             rep_indices.append(best_idx)
    
#     # select the most representative SMILES per cluster (medoid approach)
    
#     return rep_smiles, rep_indices, labels, valid_smiles, cluster_members

# def hierarchical_screen(
#     smiles_list: List[str],
#     n_representatives: int = 1000,
#     scorer: Optional[Callable[[List[str]], np.ndarray]] = None,
#     top_k_reps: int = 50,
#     random_state: Optional[int] = None,
#     second_stage_exclude_rep: bool = True,
#     acquisition_function: str = 'uncertainty',
#     train_smiles: List[str] = None,
#     cluster_method: str = 'kmeans',
# ):
#     """
#     Pipeline:
#       (1) Cluster & select n_representatives representative molecules.
#       (2) Score just those reps. Pick top_k_reps.
#       (3) Expand to their clusters and score only those cluster members.

#     Returns dict with:
#       - reps_df: pd.DataFrame with columns ['smiles','index','cluster','score','cluster_size']
#       - selected_clusters: list of cluster ids chosen at Stage (2)
#       - second_stage_indices: indices (into valid_smiles) evaluated in Stage (3)
#       - second_stage_scores: np.array of scores (aligned with second_stage_indices)
#       - valid_smiles: the filtered SMILES used throughout
#       - cluster_members: mapping cid -> np.array of member indices
#     """
#     import pandas as pd
#     if scorer is None:
#         #raise ValueError("Please provide a scorer callable: scorer(list_of_smiles) -> scores")
#         scorer = Acquisition(
#             acquisition_function= acquisition_function,
#             model_file = '/home/rengp/projects/YY/polyactive/results/gatv2/mt_train/debug-logvar-prod-ensemble/ensemble/production/gatv2_Rg_Density_Bulk_modulus_FFV_PLD_CLD_debug-logvar-prod-ensemble.pt',
#             #model_file = '/home/rengp/projects/YY/polyactive/results/gatv2/mt_train/debug-similarity-mt-poro-prod-ensemble/ensemble/production/gatv2_Rg_Density_Bulk_modulus_FFV_PLD_CLD_debug-similarity-mt-poro-prod-ensemble.pt',
#             #model_file='/home/rengp/projects/YY/polyactive/results/gatv2/mt_train/debug-similarity-mt-prod/ensemble/production/gatv2_Rg_Density_Bulk_modulus_debug-similarity-mt-prod-ensemble.pt',
#             #model_file = '/home/rengp/projects/YY/polymon/results/gatv2/Rg/AL-5-Uncertainty-NN/fold_1/gatv2_Rg_AL-5-Uncertainty-NN.pt',
#             model_type='ensemble',
#             device='cuda',
#             ordered_tasks=['Rg', 'Density', 'Bulk_modulus', 'FFV', 'PLD', 'CLD'],
#         )
    
#     if acquisition_function == 'random':
#         rep_smiles = scorer.acquire(smiles_list)
#         print(len(rep_smiles))
#         return {
#             "reps_df": pd.DataFrame({"smiles": rep_smiles, "index": range(len(rep_smiles)), "cluster": [0] * len(rep_smiles), "score": [0] * len(rep_smiles), "cluster_size": [0] * len(rep_smiles)}),
#             "selected_clusters": [0],
#             "second_stage_indices": [],
#             "second_stage_scores": [],
#             "valid_smiles": smiles_list,
#             "cluster_members": {0: np.arange(len(smiles_list))},
#         }
#     # Stage (1): representatives
#     rep_smiles, rep_indices, labels, valid_smiles, cluster_members = sample_representative_smiles(
#         smiles_list=smiles_list,
#         n_clusters=n_representatives,
#         random_state=random_state,
#         cluster_method=cluster_method,
#     )

#     # Map each rep to its cluster id quickly: label of its index
#     rep_cluster_ids = [labels[i] for i in rep_indices]

#     # Stage (2): score only the representatives
#     scores = scorer.score(rep_smiles, train_smiles)
#     if isinstance(scores, torch.Tensor):
#         rep_scores = np.asarray(scores.detach().cpu(), dtype=float)
#     else:
#         rep_scores = np.asarray(scores, dtype=float)
#     if rep_scores.shape[0] != len(rep_smiles):
#         raise ValueError("scorer returned wrong number of scores for representatives.")

#     # Rank reps and choose top_k
#     order = np.argsort(-rep_scores)
#     top_order = order[:top_k_reps]
#     top_rep_indices = [rep_indices[i] for i in top_order]
#     selected_clusters = [rep_cluster_ids[i] for i in top_order]

#     # Prepare Stage (3) candidate list: union of selected clusters
#     #second_stage_indices = []
#     second_stage_indices = {}
#     for cid in selected_clusters:
#         members = cluster_members[cid]
#         if second_stage_exclude_rep:
#             # remove the representative from the cluster expansion to avoid re-scoring it
#             rep_idx = rep_indices[rep_cluster_ids.index(cid)]
#             members = members[members != rep_idx]
#         #second_stage_indices.append(members)
#         second_stage_indices[cid] = members
#     #second_stage_indices = np.concatenate(second_stage_indices) if len(second_stage_indices) else np.array([], dtype=int)

#     # Stage (3): score the expanded set
#     cluster_keys = list(second_stage_indices.keys())
#     #second_stage_smiles = [valid_smiles[i] for i in second_stage_indices]
#     second_stage_smiles = {cid: [valid_smiles[i] for i in second_stage_indices[cid]] for cid in cluster_keys}
#     print('Scoring molecules within the top-1 cluster:', len(second_stage_smiles[cluster_keys[0]]))
#     second_stage_scores = np.asarray(scorer.score(second_stage_smiles[cluster_keys[0]], train_smiles).detach().cpu(), dtype=float) if len(second_stage_smiles) else np.array([])

#     # Build tidy outputs
#     cluster_sizes = [len(cluster_members[c]) for c in rep_cluster_ids]
#     reps_df = pd.DataFrame({
#         "smiles": rep_smiles,
#         "index": rep_indices,
#         "cluster": rep_cluster_ids,
#         "score": rep_scores,
#         "cluster_size": cluster_sizes
#     }).sort_values("score", ascending=False).reset_index(drop=True)

#     return {
#         "reps_df": reps_df,
#         "selected_clusters": selected_clusters,
#         "second_stage_indices": second_stage_indices,
#         "second_stage_scores": second_stage_scores,
#         "second_stage_smiles": second_stage_smiles,
#         # "valid_smiles": valid_smiles,
#         "cluster_members": cluster_members,
#     }

def sample_smiles(
    smiles_list: List[str],
    n_clusters: int,
    radius: int = 4,
    n_bits: int = 2048,
    random_state: Optional[int] = None,
    n_sample: int = 1000,
    return_all: bool = False,
):
    # generate fingerprints for valid molecules
    fps, valid_smiles = [], []
    for smi in tqdm(smiles_list, desc='Generating fingerprints'):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mfgen = rdFingerprintGenerator.GetMorganGenerator(radius, fpSize=n_bits)
            fps.append(mfgen.GetFingerprint(mol))
            valid_smiles.append(smi)
    if len(fps) < n_clusters:
        raise ValueError(f'Not enough molecules: {len(fps)} < n_clusters ({n_clusters})')
    
    # convert fingerprints to numpy array
    arr = np.zeros((len(fps), n_bits), dtype=int)
    for i, fp in enumerate(fps):
        ConvertToNumpyArray(fp, arr[i])
    
    # perform k-means clustering and sample from each cluster
    print('Performing k-means clustering...')
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(arr)
    
    # Sample n_sample molecules from each cluster

    rng = np.random.default_rng(random_state)
    cluster_indices = defaultdict(list)
    for idx, lbl in enumerate(labels):
        cluster_indices[lbl].append(idx)
    
    n_sample = min(n_sample, len(fps))  # Default to 1, can be set as function argument as needed
    if return_all:
        all_smiles = defaultdict(list)
    sampled_smiles = defaultdict(list)
    for cluster_id, idxs in cluster_indices.items():
        if len(idxs) <= n_sample:
            chosen = idxs  # take all if less than n_sample in cluster
        else:
            chosen = rng.choice(idxs, n_sample, replace=False)
        for i in chosen:
            sampled_smiles[cluster_id].append(valid_smiles[i])
        if return_all:
            for i in idxs:
                all_smiles[cluster_id].append(valid_smiles[i])
    if return_all:
        np.save('all_smiles.npy', all_smiles)
    return sampled_smiles
    
def score(
    train_file: str, 
    model_file: str, 
    smiles_cluster_file: str = None, 
    model_type: str = 'ensemble', 
    device: str = 'cuda', 
    ordered_tasks: List[str] = ['Rg', 'Density', 'Bulk_modulus', 'FFV', 'PLD', 'CLD'],
    acquisition_function: str = 'uncertainty',
    all_clusters_file: str = 'all_clusters.npy',
    n_sample: int = 50,
):
    if smiles_cluster_file is not None:
        smiles_cluster_df = pd.read_csv(smiles_cluster_file)
        cluster_smiles = smiles_cluster_df.groupby('cluster_id')['SMILES'].apply(list).to_dict()
    else:
        all_smiles = np.load(all_clusters_file, allow_pickle=True).item()
        
    train_df = pd.read_csv(train_file)
    train_df = train_df[train_df['Source'].isin(INITIAL_SOURCES)]
    train_smiles = train_df['SMILES'].tolist()
    scorer = Acquisition(
        acquisition_function= acquisition_function,
        model_file = model_file,
        model_type=model_type,
        device=device,
        ordered_tasks=ordered_tasks,
    )
    sampled_scores = defaultdict(list)
    for cluster_id, smiles in tqdm(cluster_smiles.items(), desc='Scoring'):
        score = scorer.score(
            pool_smiles = smiles,
            train_smiles = train_smiles,
        )
        sampled_scores[cluster_id] = score
    
    for cluster_id, scores in sampled_scores.items():
        if isinstance(scores, list):
            scores_np = []
            for s in scores:
                if hasattr(s, 'detach'):  # torch.Tensor
                    s = s.detach().cpu().numpy()
                scores_np.append(s)
            sampled_scores[cluster_id] = np.array(scores_np)
        elif hasattr(scores, 'detach'):
            sampled_scores[cluster_id] = scores.detach().cpu().numpy()

    all_scores = [score for scores in sampled_scores.values() for score in scores]
    threshold = np.quantile(all_scores, 0.95)
    best_cluster = max(
        sampled_scores,
        key=lambda c: sum(s > threshold for s in sampled_scores[c]) / len(sampled_scores[c])
    )
    all_clusters = np.load(all_clusters_file, allow_pickle=True)
    cluster_smiles = all_clusters.item()[best_cluster]
    cluster_scores = scorer.score(cluster_smiles, train_smiles)
    cluster_scores = cluster_scores.detach().cpu()
    topk = min(n_sample, len(cluster_scores))
    print(f"Top {topk} scores:", cluster_scores.topk(topk).values.tolist())
    
    top_n_sample = torch.topk(cluster_scores, n_sample).indices.tolist()
    top_n_smiles = [cluster_smiles[i] for i in top_n_sample]
    return dict(zip(top_n_smiles, cluster_scores[top_n_sample]))
    # import random
    # random.seed(42)
    # subsample_smiles = random.sample(smiles_list, min(subsample_size, len(smiles_list)))
    # scores = np.asarray(scorer.score(subsample_smiles, train_smiles).detach().cpu(), dtype=float)
    # smiles_scores = {smiles: score for smiles, score in zip(subsample_smiles, scores)}
    # return smiles_scores

def main(args: argparse.Namespace):
    if args.acquisition_function != 'random':
        smiles_scores = score(
            args.train_file, 
            args.model_file, 
            args.smiles_cluster_file, 
            args.model_type, 
            args.device, 
            args.ordered_tasks, 
            args.acquisition_function, 
            args.all_clusters_file,
            args.n_sample,
        )
        smiles_scores = {smiles: score.item() for smiles, score in smiles_scores.items()}
        sampled_smiles = list(smiles_scores.keys())
    else:
        all_smiles = [
            s for v in np.load(args.all_clusters_file, allow_pickle=True).item().values() for s in v
            ]
        sampled_smiles = np.random.choice(all_smiles, args.n_sample, replace=False)
    CWD = pathlib.Path.cwd()
    output_dir = os.path.join(CWD, 'database', 'sampled')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, args.output_file)
    
    df_sampled = pd.DataFrame({'SMILES': sampled_smiles, 'Source': args.sample_tag})
    df_train = pd.read_csv(args.train_file)
    df = pd.concat([df_train, df_sampled], ignore_index=True)
    #print(smiles_scores)
    df.to_csv(output_file, index=False)
    # out_df = pd.DataFrame.from_dict({'SMILES': list(smiles_scores.keys()), 'score': list(smiles_scores.values())})
    # # sort by score descending
    # out_df = out_df.sort_values('score', ascending=False)
    # out_df.to_csv(output_file, index=False)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)