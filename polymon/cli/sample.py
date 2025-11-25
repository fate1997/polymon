from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity, ConvertToNumpyArray
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from typing import List, Callable, Optional

from polymon.exp.acquisition import Acquisition

def sample_representative_smiles(
    smiles_list: List[str],
    n_clusters: int,
    radius: int = 4,
    n_bits: int = 2048,
    random_state: Optional[int] = None,
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
    
    # perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(arr)

    cluster_members = {cid: np.where(labels == cid)[0] for cid in range(n_clusters)}
    
    # select the most representative SMILES per cluster (medoid approach)
    rep_smiles = []
    rep_indices = []
    for cid in tqdm(range(n_clusters), desc='Selecting representatives'):
        idxs = np.where(labels == cid)[0]
        cluster_fps = [fps[i] for i in idxs]
        # compute sum of pairwise Tanimoto similarities for each member
        scores = [sum(BulkTanimotoSimilarity(fp, cluster_fps)) for fp in cluster_fps]
        best_idx = idxs[int(np.argmax(scores))]
        rep_smiles.append(valid_smiles[best_idx])
        rep_indices.append(best_idx)
    
    return rep_smiles, rep_indices, labels, valid_smiles, cluster_members

def hierarchical_screen(
    smiles_list: List[str],
    n_representatives: int = 1000,
    scorer: Optional[Callable[[List[str]], np.ndarray]] = None,
    top_k_reps: int = 50,
    random_state: Optional[int] = None,
    second_stage_exclude_rep: bool = True,
):
    """
    Pipeline:
      (1) Cluster & select n_representatives representative molecules.
      (2) Score just those reps. Pick top_k_reps.
      (3) Expand to their clusters and score only those cluster members.

    Returns dict with:
      - reps_df: pd.DataFrame with columns ['smiles','index','cluster','score','cluster_size']
      - selected_clusters: list of cluster ids chosen at Stage (2)
      - second_stage_indices: indices (into valid_smiles) evaluated in Stage (3)
      - second_stage_scores: np.array of scores (aligned with second_stage_indices)
      - valid_smiles: the filtered SMILES used throughout
      - cluster_members: mapping cid -> np.array of member indices
    """
    import pandas as pd
    if scorer is None:
        #raise ValueError("Please provide a scorer callable: scorer(list_of_smiles) -> scores")
        scorer = Acquisition(
            acquisition_function='uncertainty',
            model_file='/home/rengp/projects/YY/polyactive/results/gatv2/mt_train/debug-similarity-mt-prod/ensemble/production/gatv2_Rg_Density_Bulk_modulus_debug-similarity-mt-prod-ensemble.pt',
            #model_file = '/home/rengp/projects/YY/polymon/results/gatv2/Rg/AL-5-Uncertainty-NN/fold_1/gatv2_Rg_AL-5-Uncertainty-NN.pt',
            model_type='ensemble',
            device='cuda',
            ordered_tasks=['Rg', 'Density', 'Bulk_modulus'],
        )
    # Stage (1): representatives
    rep_smiles, rep_indices, labels, valid_smiles, cluster_members = sample_representative_smiles(
        smiles_list=smiles_list,
        n_clusters=n_representatives,
        random_state=random_state,
    )

    # Map each rep to its cluster id quickly: label of its index
    rep_cluster_ids = [labels[i] for i in rep_indices]

    # Stage (2): score only the representatives
    rep_scores = np.asarray(scorer.score(rep_smiles).detach().cpu(), dtype=float)
    if rep_scores.shape[0] != len(rep_smiles):
        raise ValueError("scorer returned wrong number of scores for representatives.")

    # Rank reps and choose top_k
    order = np.argsort(-rep_scores)
    top_order = order[:top_k_reps]
    top_rep_indices = [rep_indices[i] for i in top_order]
    selected_clusters = [rep_cluster_ids[i] for i in top_order]

    # Prepare Stage (3) candidate list: union of selected clusters
    second_stage_indices = []
    for cid in selected_clusters:
        members = cluster_members[cid]
        if second_stage_exclude_rep:
            # remove the representative from the cluster expansion to avoid re-scoring it
            rep_idx = rep_indices[rep_cluster_ids.index(cid)]
            members = members[members != rep_idx]
        second_stage_indices.append(members)
    second_stage_indices = np.concatenate(second_stage_indices) if len(second_stage_indices) else np.array([], dtype=int)

    # Stage (3): score the expanded set
    second_stage_smiles = [valid_smiles[i] for i in second_stage_indices]
    second_stage_scores = np.asarray(scorer.score(second_stage_smiles).detach().cpu(), dtype=float) if len(second_stage_smiles) else np.array([])

    # Build tidy outputs
    cluster_sizes = [len(cluster_members[c]) for c in rep_cluster_ids]
    reps_df = pd.DataFrame({
        "smiles": rep_smiles,
        "index": rep_indices,
        "cluster": rep_cluster_ids,
        "score": rep_scores,
        "cluster_size": cluster_sizes
    }).sort_values("score", ascending=False).reset_index(drop=True)

    return {
        "reps_df": reps_df,
        "selected_clusters": selected_clusters,
        "second_stage_indices": second_stage_indices,
        "second_stage_scores": second_stage_scores,
        "second_stage_smiles": second_stage_smiles,
        # "valid_smiles": valid_smiles,
        "cluster_members": cluster_members,
    }
    