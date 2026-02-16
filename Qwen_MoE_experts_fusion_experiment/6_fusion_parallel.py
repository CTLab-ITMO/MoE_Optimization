"""Parallel version of 6_fusion.py for 4xA100 (or any N GPUs).

Usage examples:
  # 4 GPUs on one node
  torchrun --standalone --nproc_per_node=4 6_fusion_parallel.py --layer 0
Each process handles a disjoint subset of expert pairs.Outputs are written directly to:
  data/fusion_experts/<layer>/<pair_idx>.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple
import torch
import torch.distributed as dist
from tqdm import tqdm
from config import EXPERTS

sys.path.append("..")
from common.mlp import AdvancedMLP


def turn_in_mlp(mlp: torch.nn.Module) -> AdvancedMLP:
    new_mlp = AdvancedMLP(
        mlp.gate_proj.in_features,
        mlp.gate_proj.out_features,
        mlp.down_proj.out_features,
    )
    new_mlp.gate = mlp.gate_proj
    new_mlp.up = mlp.up_proj
    new_mlp.down = mlp.down_proj
    return new_mlp


@torch.no_grad()
def zipit_fuse(mlpA: AdvancedMLP, mlpB: AdvancedMLP, train_x: torch.Tensor) -> AdvancedMLP:
    mlpA = mlpA
    mlpB = mlpB
    fused = mlpA
    fused.zip_with(mlpB)
    fused.train_with(train_x)
    fused.zip()
    return fused


def get_rank_world() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    # torchrun sets env vars even before init
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world


class FusionExpertParallel:
    def __init__(self, args):
        self.args = args
        self._init_layer(args)
        self._init_distrib(args)
        self.out_dir = os.path.join("data", "fusion_experts", str(self.layer))
        os.makedirs(self.out_dir, exist_ok=True)

    def _init_distrib(self, args):
        # Init distributed if launched with torchrun.
        if "RANK" in os.environ and not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        self.rank, self.world = get_rank_world()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = args.device or (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

    def _init_layer(self, args):
        layer = args.layer
        if args.pairs is None:
            args.pairs = f"data/pairs_{layer}.pt"
        if args.train_inputs is None:
            args.train_inputs = f"data/mlp_inputs/train_{layer}.pt"
        if args.test_inputs is None:
            args.test_inputs = f"data/mlp_inputs/test_{layer}.pt"
        if args.pairs_train_acts is None:
            args.pairs_train_acts = f"data/activations/pairs_train_{layer}.pt"
        if args.pairs_test_acts is None:
            args.pairs_test_acts = f"data/activations/pairs_test_{layer}.pt"
        self.layer = layer

    def get_experts(self):
        experts = [
            torch.load(os.path.join(args.experts_dir, str(self.layer), f"{i}.pt"), weights_only=False)
            for i in range(EXPERTS)
        ]
        pairs = torch.load(args.pairs)
        idx = [i for i in range(len(pairs)) if i % self.world == self.rank]
        return experts, pairs, idx

    def fuse(self, experts, pairs, idx):
        pbar = tqdm(idx, desc=f"rank {self.rank}/{self.world}")

        train_data = torch.load(args.train_inputs)
        test_data = torch.load(args.test_inputs)
        train_pairs_acts = torch.load(args.pairs_train_acts)
        test_pairs_acts = torch.load(args.pairs_test_acts)
        for pair_idx in pbar:
            out_path = os.path.join(self.out_dir, f"{pair_idx}.pt")
            if os.path.exists(out_path):
                continue
            e1, e2 = pairs[pair_idx]
            # gather inputs where both experts in the pair are co-activated
            tr_idx = train_pairs_acts[pair_idx]
            if len(tr_idx) == 0:
                continue
            train_x = torch.stack([train_data[i] for i in tr_idx], dim=0).to(self.device)

            mlp1 = turn_in_mlp(experts[e1].to(self.device))
            mlp2 = turn_in_mlp(experts[e2].to(self.device))
            fused = zipit_fuse(mlp1, mlp2, train_x)
            torch.save(fused.cpu(), out_path)

            # free
            del train_x, mlp1, mlp2, fused
            torch.cuda.empty_cache()

        if dist.is_initialized():
            dist.barrier()
            if self.rank == 0:
                print("[fusion_parallel] done")

    def run(self):
        self.fuse(*self.get_experts())




if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--experts_dir", type=str, default="data/experts")
    p.add_argument("--pairs", type=str, default=None)
    p.add_argument("--train_inputs", type=str, default=None)
    p.add_argument("--test_inputs", type=str, default=None)
    p.add_argument("--pairs_train_acts", type=str, default=None)
    p.add_argument("--pairs_test_acts", type=str, default=None)
    args = p.parse_args()
    experiment_handler = FusionExpertParallel(args)
    experiment_handler.run()
