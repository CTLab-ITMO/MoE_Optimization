import torch
from datasets import load_dataset
from typing import Dict, List, Generator, Optional
import numpy as np
import copy
import pandas as pd

from transformers import BitsAndBytesConfig
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.sparse.csgraph import laplacian
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

SELECTED_TASKS = ["anatomy",
            # "conceptual_physics",
            # "human_sexuality",
            "machine_learning",
            "management",
            # "marketing",
            # "nutrition",
             "philosophy",
            # "us_foreign_policy",
              "world_religions"]


def get_qwen(model=None, quantization_config=None, cache_dir=None):

    MODEL = "Qwen/Qwen1.5-MoE-A2.7B"

    if quantization_config is None:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,

            bnb_4bit_compute_dtype=torch.float16,  # Compute in float16 for speed
            bnb_4bit_use_double_quant=True,  # Double quantization for extra memory savings
            # Normalized float 4-bit (optimal for LLMs)
            bnb_4bit_quant_type="nf4"
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True,
                                                device_map='cuda',
                                                )
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
                                                device_map='cuda',
                                                dtype=torch.float16,
                                                quantization_config=quantization_config,
                                                cache_dir=cache_dir
                                                )
    return model, tokenizer




def spectral_bisect(mat, nodes, min_cluster_size=2):
    '''make graph bisection using Laplase operator and Fielder vector.
    
       More weight on edges - more connectivity. Bisectrion tries to delete weak edges to maximize weight on parts.
    '''
    nodes = np.asarray(nodes)
    if len(nodes) < 2 * min_cluster_size:
        return nodes.tolist(), []

    sub = mat[np.ix_(nodes, nodes)].numpy()
    if np.allclose(sub, 0.0):
        order = np.arange(len(nodes))
    else:
        lap = laplacian(sub, normed=True)
        evals, evecs = np.linalg.eigh(lap)
        fiedler = evecs[:, 1] if evecs.shape[1] > 1 else evecs[:, 0]
        order = np.argsort(fiedler)

    split = len(nodes) // 2
    split = max(min_cluster_size, min(
        split, len(nodes) - min_cluster_size))
    left = nodes[order[:split]].tolist()
    right = nodes[order[split:]].tolist()

    if not left or not right:
        strengths = sub.sum(axis=1)
        order = np.argsort(strengths)
        left = nodes[order[:split]].tolist()
        right = nodes[order[split:]].tolist()

    return left, right

def compute_metric_on_dense_cluster(mat, nodes, reduce="mean"):
    '''
    Args:
        nodes: indexes of nodes in cluster
    '''
    #if mean - have sence of density
    if mat is None:
        return np.nan
    nodes = np.asarray(nodes)
    if len(nodes) < 2:
        return 0.0

    sub = mat[np.ix_(nodes, nodes)]
    upper = np.triu(sub, 1)
    pair_count = len(nodes) * (len(nodes) - 1) // 2
    total = float(upper.sum())

    if reduce == "sum":
       return total
    if reduce == "mean":
        return total / pair_count if pair_count > 0 else 0.0

    raise ValueError("reduce must be 'mean' or 'sum'")
    
def cluster_pair_count(nodes):
    n = len(nodes)
    return n * (n - 1) // 2    
    
def weighted_average_cluster_metric(mat, clusters):
    if mat is None:
        return np.nan

    total_sum = 0.0
    total_pairs = 0

    for cluster in clusters:
        nodes = np.asarray(cluster)
        if len(nodes) < 2:
            continue
        sub = mat[np.ix_(nodes, nodes)]
        upper = np.triu(sub, 1)
        total_sum += float(upper.sum())
        total_pairs += cluster_pair_count(nodes)

    return total_sum / total_pairs if total_pairs > 0 else 0.0
    
def make_iterative_clustering(adjacency_matrix_of_experts: np.ndarray, 
                              min_cluster_size=2,
                              maximize_metric=True, 
                              other_matrices_for_stats: Optional[dict[str, np.ndarray]] = None,
                              devide_worst_candidate_first = True,
                              log_history=False) -> tuple[list, list]:
    '''
    Makes iterative deviding clustering using `adjacency_matrix_of_experts`.
    Additionally compute stats of main metric on every step. If `other_matrices_for_stats` provided then
    ads their metrics (using name of key) during clustering in history.
        
    Returns: 
            clusters: as indexes
            history: as dicts
    '''

    if adjacency_matrix_of_experts.ndim != 2 or adjacency_matrix_of_experts.shape[0] != adjacency_matrix_of_experts.shape[1]:
        raise ValueError("adjacency_matrix_of_experts must be a square matrix")

    clusters = [list(range(adjacency_matrix_of_experts.shape[0]))]
    history = []
    step = 0

    while True:
        objective_mat = adjacency_matrix_of_experts
        if (maximize_metric == False):
            objective_mat = objective_mat.max() - objective_mat

        candidates = [
            (compute_metric_on_dense_cluster(objective_mat, cluster), len(cluster), idx)
            for idx, cluster in enumerate(clusters)
            if len(cluster) >= 2 * min_cluster_size
        ]
        if not candidates:
            break

        if (devide_worst_candidate_first):
            _, _, split_idx = max(candidates, key=lambda item: (-item[0], item[1]))
        else:
            _, _, split_idx = max(candidates, key=lambda item: (item[0], item[1]))
            
        cluster_to_divide = clusters.pop(split_idx)
        left, right = spectral_bisect(objective_mat, cluster_to_divide, min_cluster_size=min_cluster_size)
            
        if len(left) < min_cluster_size or len(right) < min_cluster_size:
            clusters.insert(split_idx, cluster_to_divide)
            break

        clusters.extend([left, right])
        step += 1
        
        if (log_history):
            step_dict = {
                    "step": step,
                    "divided_cluster": cluster_to_divide,
                    "left": left,
                    "right": right,
                    "all_clusters": copy.copy(clusters),
                    "metric_of_split_cluster": compute_metric_on_dense_cluster(adjacency_matrix_of_experts, cluster_to_divide),
                    "total_metric_after_split": weighted_average_cluster_metric(adjacency_matrix_of_experts, clusters),
                    }
            
            if (other_matrices_for_stats is not None):
                for key in other_matrices_for_stats.keys():
                    step_dict[f'{key}_of_split_cluster'] = compute_metric_on_dense_cluster(other_matrices_for_stats[key], cluster_to_divide)
                    step_dict[f'total_{key}_after_split'] = weighted_average_cluster_metric(other_matrices_for_stats[key], clusters)
                
            history.append(step_dict)

    return clusters, history 

def make_greedy_agglomerative_pairs(
    adjacency_matrix_of_experts,
    maximize_metric=False,
    log_history=False,
    other_matrices_for_stats: Optional[dict[str, np.ndarray]] = None,
):
    '''
    Greedy matching over all edges.

    Build a queue of all expert pairs, sort it by PI, and then keep selecting
    edges whose endpoints are still unused. The result is a set of disjoint
    pairs plus singleton leftovers.

    Signature intentionally matches find_by_dividive_clustering.
    '''
    if adjacency_matrix_of_experts.ndim != 2 or adjacency_matrix_of_experts.shape[0] != adjacency_matrix_of_experts.shape[1]:
        raise ValueError('adjacency_matrix_of_experts must be a square matrix')

    score_mat = np.asarray(adjacency_matrix_of_experts, dtype=np.float64)
    n_experts = score_mat.shape[0]

    if n_experts == 0:
        return [], []
    if n_experts == 1:
        return [[0]], []

    edge_queue = [
        (float(score_mat[i, j]), i, j)
        for i in range(n_experts)
        for j in range(i + 1, n_experts)
    ]

    if maximize_metric:
        edge_queue.sort(key=lambda item: (-item[0], item[1], item[2]))
    else:
        edge_queue.sort(key=lambda item: (item[0], item[1], item[2]))

    used_vertices = set()
    clusters = []
    history = []

    for score, i, j in edge_queue:
        if i in used_vertices or j in used_vertices:
            continue

        used_vertices.add(i)
        used_vertices.add(j)
        clusters.append([i, j])

    if log_history:
        step_dict = {
            'step': len(history) + 1,
            'all_clusters': clusters,
            "total_metric_after_split": weighted_average_cluster_metric(adjacency_matrix_of_experts, clusters),

        }
        if other_matrices_for_stats is not None:
            for key in other_matrices_for_stats.keys():
                step_dict[f'total_{key}_after_split'] = weighted_average_cluster_metric(other_matrices_for_stats[key], clusters)
        history.append(step_dict)

    return clusters, history


def detect_pi_front_points(
    values,
    skip_start=5, #tested in check_activations_of_clusters.ipynb (optimized for 60 experts Qwen! ~  30 steps)
    radius=1, #tested in check_activations_of_clusters.ipynb
    neighborhood=3,
    z_threshold=0.5, #tested in check_activations_of_clusters.ipynb
    min_curvature=0.0,
    require_local_peak=True,
):
    """Detect local front points on a curve.
    
    Args:
        values: numpy array or pandas Series of values
        skip_start: skip first N points
        radius: size of left/right window
        neighborhood: neighborhood size for local z-score
        z_threshold: z-score threshold for detection
        min_curvature: minimum front_score threshold
        require_local_peak: only accept local peaks
    
    Returns:
        DataFrame with detected front points (indices and scores)
    """
    values = np.asarray(values, dtype=np.float64)

    front_score = np.full(len(values), np.nan, dtype=np.float64)
    left_delta = np.full(len(values), np.nan, dtype=np.float64)
    right_delta = np.full(len(values), np.nan, dtype=np.float64)

    start_idx = skip_start + radius
    stop_idx = len(values) - radius
    for idx in range(start_idx, stop_idx):
        left_window = values[idx - radius:idx]
        right_window = values[idx + 1:idx + 1 + radius]
        if left_window.size == 0 or right_window.size == 0:
            continue

        left_delta[idx] = values[idx] - left_window.mean()
        right_delta[idx] = values[idx] - right_window.mean()
        front_score[idx] = left_delta[idx] + right_delta[idx]

    score_series = pd.Series(front_score)
    local_mean = score_series.rolling(window=2 * neighborhood + 1, center=True, min_periods=1).mean()
    local_std = score_series.rolling(window=2 * neighborhood + 1, center=True, min_periods=1).std(ddof=0)
    score_z = (score_series - local_mean) / (local_std + 1e-12)

    if require_local_peak:
        local_max = score_series.rolling(window=2 * neighborhood + 1, center=True, min_periods=1).max()
        local_peak_mask = score_series.eq(local_max)
    else:
        local_peak_mask = pd.Series(True, index=score_series.index)

    candidate_mask = (
        np.arange(len(values)) >= start_idx
    ) & np.isfinite(front_score) & (front_score > min_curvature) & (score_z.to_numpy() > z_threshold) & local_peak_mask.fillna(False).to_numpy()

    indices = np.where(candidate_mask)[0]
    result = pd.DataFrame({
        'idx': indices,
        'value': values[indices],
        'left_delta': left_delta[indices],
        'right_delta': right_delta[indices],
        'front_score': front_score[indices],
        'front_zscore': score_z.to_numpy()[indices],
    })
    return result

def labels_from_clusters(clusters, row_experts):
    '''transforms labels to merged labels'''
    expert_to_cluster = {}
    for cluster_id, cluster in enumerate(clusters):
        for expert in cluster:
            expert_to_cluster[expert] = cluster_id
    return np.asarray([expert_to_cluster.get(expert, -1) for expert in row_experts], dtype=np.int64)

def build_calinski_score_df(history: dict, X: np.ndarray | pd.DataFrame, row_experts, some_popularity_metric_column='total_PI_after_split'):
    '''Eats history of expert division to clusters and embeddings (X) of activations. Also eats row_experts - indexes of experts belonging to X
    Returns:
        another table with history and with calculated calinski_score metric for every cluster division
    '''
    rows = []

    for step, entry in enumerate(history):
        clusters = entry.get("all_clusters", entry.get("clusters"))
        if clusters is None or len(clusters) < 2:
            continue

        labels = labels_from_clusters(clusters, row_experts) #insted of series of expert indices makes indices of clusters
        valid_mask = labels >= 0
        X_step = X[valid_mask]
        labels_step = labels[valid_mask]

        if X_step.shape[0] < 2 or np.unique(labels_step).size < 2:
            continue

        row = {
            "step": step,
            "n_clusters": len(clusters),
            some_popularity_metric_column: entry[some_popularity_metric_column]
        }

        try:
            row["calinski_harabasz"] = calinski_harabasz_score(X_step, labels_step)
        except Exception:
            row["calinski_harabasz"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)