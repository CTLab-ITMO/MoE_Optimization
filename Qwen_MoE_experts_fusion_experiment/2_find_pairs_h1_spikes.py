#!/usr/bin/env python
# coding: utf-8

"""
  - Weight-unfolding X=[E x D_params] is infeasible for large experts.
  - Instead build a *functional sketch* of each expert on cached `mlp_inputs`.
  - Then apply a cheap RMT-inspired spike filter in expert space and build disjoint pairs with maxi similarity.

Outputs:
  - data/pairs_<layer>.pt: list[(i,j)] disjoint pairs.
  - data/h1_spikes_<layer>.pt: diagnostics.

This script is compatible with the rest of the pipeline:
  4_write_mlp_inputs.py -> 2_find_pairs_h1_spikes.py -> 5_activations.py -> 6_fusion.py
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import torch

from common.projection_tools import build_functional_sketch
from common.rmt_tools import spike_denoised_embedding


def greedy_pairs_from_similarity(sim: torch.Tensor) -> List[Tuple[int, int]]:
    """Greedy maximum weight matching on a dense similarity matrix."""
    n = sim.shape[0]
    sim = sim.clone()
    sim.fill_diagonal_(-float("inf"))
    used = torch.zeros(n, dtype=torch.bool)
    pairs: List[Tuple[int, int]] = []

    for _ in range(n // 2):
        # mask used
        sim[used, :] = -float("inf")
        sim[:, used] = -float("inf")
        flat_idx = torch.argmax(sim.view(-1))
        i = int(flat_idx // n)
        j = int(flat_idx % n)
        if not torch.isfinite(sim[i, j]):
            break
        used[i] = True
        used[j] = True
        pairs.append((i, j))
    return pairs



class FindingPairsPipeline:

    def __init__(self, args):
        self.layer = args.layer
        self.args = args
        if args.mlp_inputs is None:
            args.mlp_inputs = f"data/mlp_inputs/train_{self.layer}.pt"

        self.experts_path = os.path.join(args.experts_dir, str(self.layer))
        if not os.path.exists(self.experts_path):
            raise FileNotFoundError(f"Experts not found: {self.experts_path}. Run 3_save_experts.py first.")

    def load_experts(self):
        # Load experts
        experts = []
        idx = 0
        while True:
            path = os.path.join(self.experts_path, f"{idx}.pt")
            if not os.path.exists(path):
                break
            experts.append(torch.load(path, weights_only=False))
            idx += 1
        if len(experts) == 0:
            raise RuntimeError(f"No experts loaded from {self.experts_path}")
        return experts

    def load_inputs(self):
        # Load inputs and subsample
        mlp_inputs_list = torch.load(args.mlp_inputs)
        if isinstance(mlp_inputs_list, list):
            # saved by 4_write_mlp_inputs.py: list of [d] tensors
            mlp_inputs = torch.stack(mlp_inputs_list, dim=0)
        else:
            mlp_inputs = mlp_inputs_list
        if mlp_inputs.ndim != 2:
            raise ValueError(f"Expected mlp_inputs as [N,d], got {tuple(mlp_inputs.shape)}")

        N = min(args.n_inputs, mlp_inputs.shape[0])
        mlp_inputs = mlp_inputs[:N].contiguous()
        return mlp_inputs

    def get_sketch(self, experts, mlp_inputs):
        X = build_functional_sketch(
            experts,
            mlp_inputs,
            proj_dim=args.proj_dim,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )
        Y, info = spike_denoised_embedding(X, mp_factor=args.mp_factor, min_rank=1)

        # Similarity in denoised space
        Yn = Y / (torch.linalg.norm(Y, dim=1, keepdim=True).clamp_min(1e-12))
        sim = Yn @ Yn.t()
        pairs = greedy_pairs_from_similarity(sim)
        return pairs, info

    def get_result(self, pairs, experts, info):
        os.makedirs("data", exist_ok=True)
        torch.save(pairs, f"data/pairs_{self.layer}.pt")
        torch.save({"pairs": pairs, "info": info}, f"data/h1_spikes_{self.layer}.pt")

        print(f"[H1] layer={self.layer} experts={len(experts)} proj_dim={self.args.proj_dim}  pairs={len(pairs)}")
        print(f"[H1] spike_rank={info['rank']} tau={info['tau']:.6f} lambda_plus={info['lambda_plus']:.6f}")


    def run(self):
        experts = self.load_experts()
        inputs = self.load_inputs()
        pair, info = self.get_sketch(experts,inputs)
        self.get_result(pair,experts, info)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--experts_dir", type=str, default="data/experts")
    p.add_argument("--mlp_inputs", type=str, default=None)
    p.add_argument("--n_inputs", type=int, default=4096)
    p.add_argument("--proj_dim", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--mp_factor", type=float, default=1.10)
    args = p.parse_args()
    experiment_handler = FindingPairsPipeline(args)
    experiment_handler.run()