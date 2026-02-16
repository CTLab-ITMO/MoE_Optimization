"""1 гипотеза - expert-unfolding -> spike filtering -> greedy pairing.

Why functional?
  Parameter unfolding X[e,:]=vec(W_e) is too large for modern MoE MLPs.
  Instead we build a compact sketch of each expert's *function* on cached MLP inputs.

Input:
  --mlp_inputs: torch file produced by Qwen_MoE_experts_fusion_experiment/4_write_mlp_inputs.py
               It is typically a Python list of tensors [hidden_dim].

Output:
  A torch file with:
    pairs: List[Tuple[int,int]]  (greedy max matching by cosine similarity)
    scores: List[float]
    meta: diagnostics incl. spike rank and bulk edge.

Distributed:
  If launched under torchrun and torch.distributed is initialized, each rank processes
  a shard of experts and then gathers sketches to rank0 for pairing.

This script is architecture-aware via common.moe_utils.discover_moe_layers().
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist

from transformers import AutoModelForCausalLM

from common.moe_utils import discover_moe_layers, get_gate_weight
from common.projection_tools import expert_functional_sketch, _random_omega
from common.rmt_tools import spike_filter_embeddings


def _init_dist() -> Tuple[int, int]:
    """Initialize torch.distributed if launched under torchrun.

    torchrun sets RANK/WORLD_SIZE/LOCAL_RANK env vars, but does not call init_process_group for you.
    We auto-init with NCCL when CUDA is available.
    """
    if dist.is_available():
        if dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        # auto-init if env hints are present
        if os.environ.get("RANK") is not None and os.environ.get("WORLD_SIZE") is not None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
            return dist.get_rank(), dist.get_world_size()
    return 0, 1


@torch.no_grad()
def load_inputs(path: Path, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, list):
        data = data[:n]
        X = torch.stack([t for t in data], dim=0)
    elif isinstance(data, torch.Tensor):
        X = data[:n]
    else:
        raise ValueError(f"Unsupported mlp_inputs format: {type(data)}")
    return X.to(device=device, dtype=dtype)


def _cosine_sim_matrix(E: torch.Tensor) -> torch.Tensor:
    # E: [num_experts, D]
    En = torch.nn.functional.normalize(E, dim=1)
    return En @ En.t()


def _greedy_pairing(sim: torch.Tensor) -> Tuple[List[Tuple[int, int]], List[float]]:
    """Greedy maximum matching from similarity matrix (symmetric)."""
    n = sim.shape[0]
    used = torch.zeros(n, dtype=torch.bool, device=sim.device)
    pairs: List[Tuple[int, int]] = []
    scores: List[float] = []
    # mask diagonal
    sim = sim.clone()
    sim.fill_diagonal_(-1e9)

    for _ in range(n // 2):
        # find best unused pair
        sim_masked = sim.clone()
        sim_masked[used, :] = -1e9
        sim_masked[:, used] = -1e9
        flat_idx = torch.argmax(sim_masked).item()
        i = flat_idx // n
        j = flat_idx % n
        if sim_masked[i, j].item() < -1e8:
            break
        used[i] = True
        used[j] = True
        pairs.append((int(i), int(j)))
        scores.append(float(sim[i, j].item()))
    return pairs, scores


class UnfoldingPipeline:

    def __init__(self, args):
        self.args = args
        self.rank, self.world = _init_dist()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
        self.out_path = Path(args.out) if args.out else Path("data") / f"pairs_h1_layer{args.layer}.pt"
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
    def _load_model(self):
        if self.rank == 0:
            print(f"[H1] Loading model: {self.args.model}")
        self.model = AutoModelForCausalLM.from_pretrained(self.args.model, torch_dtype=self.dtype, device_map=None)
        self.model.to(self.device)
        self.model.eval()

    def _check_moe_model(self):
        handles = discover_moe_layers(self.model)
        self.layer_handle = None
        for h in handles:
            if h.layer_idx == self.args.layer:
                layer_handle = h
                break
        if self.layer_handle is None:
            raise RuntimeError(
                f"Could not find MoE layer {self.args.layer}. Found layers: {[h.layer_idx for h in handles]}")
        experts = self.layer_handle.experts
        if isinstance(experts, nn.ModuleList):
            experts_list = list(experts)
        else:
            # some models store experts in a python list-like
            experts_list = list(experts)
        return experts_list

    def expert_projection(self, experts_list):
        X = load_inputs(Path(self.args.mlp_inputs), n=self.args.n_inputs, device=self.device, dtype=self.dtype)
        omega = _random_omega(X.shape[0], self.args.proj_dim, device=self.device, seed=1234)

        # shard experts across ranks
        idxs = list(range(len(experts_list)))
        shard = idxs[self.rank::self.world]

        if self.rank == 0:
            print(f" Experts={len(experts_list)}, "
                  f"shard per rank ~{len(shard)}; "
                  f"inputs={X.shape[0]} proj_dim={self.args.proj_dim}")

        sketches = [(e, expert_functional_sketch(experts_list[e], X, omega, batch=self.args.batch_size).cpu())
                    for e in shard]
        return sketches

    def rank_gahering(self, sketches):
        # gather to rank0
        if self.world > 1:
            gathered = [None for _ in range(self.world)]
            dist.all_gather_object(gathered, sketches)
            if self.rank == 0:
                all_items = []
                for g in gathered:
                    all_items.extend(g)
            else:
                all_items = None
        else:
            all_items = sketches

        if self.rank != 0:
            return

        all_items.sort(key=lambda t: t[0])
        Xe = torch.stack([t[1] for t in all_items], dim=0).to(torch.float32)  # [E, D]
        Xe = Xe - Xe.mean(dim=0, keepdim=True)
        spike_filter = spike_filter_embeddings(Xe, mp_factor=self.args.mp_factor)
        Xe_dn = spike_filter["Xe_dn"]
        sim = _cosine_sim_matrix(Xe_dn)
        pairs, scores = _greedy_pairing(sim)
        out_path = Path(self.args.out) if self.args.out else Path("data") / f"pairs_h1_layer{args.layer}.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gate_w = get_gate_weight(self.layer_handle.gate).cpu()
        return spike_filter, pairs, scores, gate_w

    def get_result(self,spike_filter, pairs, scores, gate_w, list_of_exp):
        meta = {
            "model": args.model,
            "layer": args.layer,
            "E": len(list_of_exp),
            "proj_dim": self.args.proj_dim,
            "n_inputs": args.n_inputs,
            "spike_rank": spike_filter["k"],
            "bulk_edge": spike_filter["edge"],
            "eigs": spike_filter["eigs"].cpu(),
            "gate_weight_shape": tuple(gate_w.shape),
            "moe_kind": self.layer_handle.kind,
        }

        torch.save({"pairs": pairs, "scores": scores, "meta": meta}, self.out_path)
        print(f"[H1] Saved pairs to {self.out_path} (pairs={len(pairs)}) spike_rank={spike_filter['k']}")

    def run(self):
        self._load_model()
        num_of_exp = self._check_moe_model()
        sketches = self.expert_projection(num_of_exp)
        spike_filter, pairs, scores, gate_w = self.rank_gahering(sketches)
        self.get_result(spike_filter, pairs, scores, gate_w, num_of_exp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--layer", required=True, type=int)
    ap.add_argument("--mlp_inputs", required=True, type=str)
    ap.add_argument("--n_inputs", default=4096, type=int)
    ap.add_argument("--proj_dim", default=8, type=int)
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--mp_factor", default=1.5, type=float)
    ap.add_argument("--out", default=None, type=str)
    args = ap.parse_args()
    exp_handler = UnfoldingPipeline(args)
    exp_handler.run()
