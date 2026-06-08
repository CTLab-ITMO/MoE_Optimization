import os
from typing import Optional
import torch
from tqdm import tqdm
from .mlp import build_matches


def random_pairs(n_experts: int) -> list[tuple[int, int]]:
    perm = torch.randperm(n_experts).tolist()
    return [(perm[2 * i], perm[2 * i + 1]) for i in range(n_experts // 2)]


def eval_match(sim_matrix: torch.Tensor, matches, top_k: int = 8) -> float:
    win = 0.0
    for i, j in matches:
        win += float(sim_matrix[i, j])
    return win / (float(sim_matrix.sum()) / top_k)


def evolutionary_match(
    sim_matrix: torch.Tensor,
    evolution_steps: int,
    progress: bool = False,
) -> list[tuple[int, int]]:
    assert sim_matrix.dim() == 2 and sim_matrix.shape[0] == sim_matrix.shape[1]
    n = sim_matrix.shape[0]

    best_win = -float("inf")
    best_pairs: list[tuple[int, int]] = []

    iterator = range(evolution_steps)
    if progress:
        iterator = tqdm(iterator)

    with torch.no_grad():
        for _ in iterator:
            perm = torch.randperm(n).tolist()
            pairs = set((perm[2 * i], perm[2 * i + 1]) for i in range(n // 2))
            friend = [-1] * n
            for i, j in pairs:
                friend[i] = j
                friend[j] = i

            def swap(i, j):
                if (i, friend[i]) in pairs:
                    pairs.remove((i, friend[i]))
                else:
                    pairs.remove((friend[i], i))
                if (j, friend[j]) in pairs:
                    pairs.remove((j, friend[j]))
                else:
                    pairs.remove((friend[j], j))
                pairs.add((i, j))
                pairs.add((friend[i], friend[j]))
                fi, fj = friend[i], friend[j]
                friend[fi], friend[fj] = fj, fi
                friend[i], friend[j] = j, i

            def win_from_swap(i, j):
                return (
                    sim_matrix[i, j]
                    + sim_matrix[friend[i], friend[j]]
                    - sim_matrix[i, friend[i]]
                    - sim_matrix[j, friend[j]]
                ).item()

            while True:
                delta = 0.0
                bi, bj = -1, -1
                for i in range(n):
                    for j in range(i + 1, n):
                        d = win_from_swap(i, j)
                        if d > delta:
                            delta = d
                            bi, bj = i, j
                if delta < 1e-6:
                    break
                swap(bi, bj)

            win = eval_match(sim_matrix, pairs)
            if win > best_win:
                best_win = win
                best_pairs = list(pairs)

    return best_pairs


def compute_activation_correlation_matrix(layer: int, data_dir: str) -> torch.Tensor:
    path = f"{data_dir}/activations/train_matrix_{layer}.pt"
    matrix = torch.load(path).cpu().float()
    matrix = matrix / matrix.sum()
    return matrix


def compute_output_similarity_matrix(
    layer: int,
    data_dir: str,
    n_experts: int,
    device: torch.device,
    num_samples: int = 4096,
    batch_size: int = 512,
) -> torch.Tensor:
    experts_dir = f"{data_dir}/experts/{layer}"
    inputs_path = f"{data_dir}/mlp_inputs/train_{layer}.pt"
    if not os.path.isdir(experts_dir):
        raise FileNotFoundError("PAIRING_METHOD='output_similarity' requires step 3 ")
    if not os.path.isfile(inputs_path):
        raise FileNotFoundError("PAIRING_METHOD='output_similarity' requires step 4 ")

    raw_inputs = torch.load(inputs_path)
    if isinstance(raw_inputs, list):
        inputs = torch.stack(raw_inputs, dim=0)
    else:
        inputs = raw_inputs
    if inputs.dim() == 3:
        inputs = inputs.reshape(-1, inputs.shape[-1])
    if inputs.shape[0] > num_samples:
        idx = torch.randperm(inputs.shape[0])[:num_samples]
        inputs = inputs[idx]
    inputs = inputs.to(device)

    norm_outputs: list[torch.Tensor] = []
    for i in range(n_experts):
        expert = torch.load(f"{experts_dir}/expert_{i}.pt", map_location=device)
        expert.eval()
        outs = []
        with torch.no_grad():
            for start in range(0, inputs.shape[0], batch_size):
                chunk = inputs[start : start + batch_size]
                out = expert(chunk).float()
                out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                outs.append(out)
        norm_outputs.append(torch.cat(outs, dim=0).cpu())
        del expert
        if device.type == "cuda":
            torch.cuda.empty_cache()

    stacked = torch.stack(norm_outputs, dim=0)  # [E, N, H]
    n = stacked.shape[1]
    sim = torch.einsum("inh,jnh->ij", stacked, stacked) / n
    return sim


def get_pairs(
    method: str,
    layer: int,
    n_experts: int,
    data_dir: str,
    device: Optional[torch.device] = None,
    evolution_steps: int = 0,
    progress: bool = False,
) -> list[tuple[int, int]]:
    if method == "random":
        return random_pairs(n_experts)

    if method == "activation_correlation":
        sim = compute_activation_correlation_matrix(layer, data_dir)
    elif method == "output_similarity":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sim = compute_output_similarity_matrix(layer, data_dir, n_experts, device)
    else:
        raise ValueError(
            f"Unknown PAIRING_METHOD: {method!r}. "
            f"Expected: 'activation_correlation', 'random', 'output_similarity'."
        )

    if evolution_steps > 0:
        return evolutionary_match(sim, evolution_steps, progress=progress)
    return build_matches(sim)
