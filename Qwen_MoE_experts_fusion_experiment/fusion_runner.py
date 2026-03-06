"""Shared fusion runner for single-process and torchrun modes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

from tqdm import tqdm

try:
    import torch
    import torch.distributed as dist
    _TORCH_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    torch = None
    dist = None
    _TORCH_IMPORT_ERROR = exc


def _ensure_torch() -> None:
    if torch is None or dist is None:
        raise ModuleNotFoundError("torch is required for fusion runner") from _TORCH_IMPORT_ERROR


def _no_grad(fn):
    if torch is None:
        return fn
    return torch.no_grad()(fn)


def _discover_expert_indices(experts_dir: str, layer: int) -> list[int]:
    base = Path(experts_dir) / str(layer)
    ids = sorted(int(p.stem) for p in base.glob("*.pt") if p.stem.isdigit())
    if not ids:
        raise FileNotFoundError(f"No expert files found under {base}")
    return ids


def _candidate(primary: Path, fallback: Path) -> str:
    return str(primary if primary.exists() else fallback)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--experts-dir", dest="experts_dir", type=str, default=None)
    parser.add_argument("--pairs", type=str, default=None)
    parser.add_argument("--train-inputs", dest="train_inputs", type=str, default=None)
    parser.add_argument("--test-inputs", dest="test_inputs", type=str, default=None)
    parser.add_argument("--pairs-train-acts", dest="pairs_train_acts", type=str, default=None)
    parser.add_argument("--pairs-test-acts", dest="pairs_test_acts", type=str, default=None)
    return parser


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args

    from moe_opt_core.config import load_experiment_config

    cfg = load_experiment_config(args.config)
    if args.layer is None:
        if not cfg.layers:
            raise ValueError("Config has empty layers list")
        args.layer = int(cfg.layers[0])

    layer = int(args.layer)
    if args.experts_dir is None:
        args.experts_dir = str(cfg.h1.experts_dir)
    if args.pairs is None:
        args.pairs = _candidate(Path(cfg.h1.output_pairs_path), Path("data") / f"pairs_{layer}.pt")

    inputs_dir = Path(cfg.h1.mlp_inputs_path).parent
    if args.train_inputs is None:
        args.train_inputs = _candidate(inputs_dir / f"train_{layer}.pt", Path("data") / "mlp_inputs" / f"train_{layer}.pt")
    if args.test_inputs is None:
        args.test_inputs = _candidate(inputs_dir / f"test_{layer}.pt", Path("data") / "mlp_inputs" / f"test_{layer}.pt")

    acts_dir = Path(cfg.artifacts.root_dir) / "legacy_activations"
    if args.pairs_train_acts is None:
        args.pairs_train_acts = _candidate(acts_dir / f"pairs_train_{layer}.pt", Path("data") / "activations" / f"pairs_train_{layer}.pt")
    if args.pairs_test_acts is None:
        args.pairs_test_acts = _candidate(acts_dir / f"pairs_test_{layer}.pt", Path("data") / "activations" / f"pairs_test_{layer}.pt")

    return args


def _turn_in_mlp(mlp):
    from common.mlp import AdvancedMLP

    new_mlp = AdvancedMLP(
        mlp.gate_proj.in_features,
        mlp.gate_proj.out_features,
        mlp.down_proj.out_features,
    )
    new_mlp.gate = mlp.gate_proj
    new_mlp.up = mlp.up_proj
    new_mlp.down = mlp.down_proj
    return new_mlp


@_no_grad
def _zipit_fuse(mlp_a, mlp_b, train_x):
    fused = mlp_a
    fused.zip_with(mlp_b)
    fused.train_with(train_x)
    fused.zip()
    return fused


def _get_rank_world() -> Tuple[int, int]:
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world


class FusionRunner:
    def __init__(self, args: argparse.Namespace):
        _ensure_torch()
        self.args = args
        self._init_distrib(args)
        self._init_layer(args)
        self.out_dir = args.out_dir or os.path.join("data", "fusion_experts", str(self.layer))
        os.makedirs(self.out_dir, exist_ok=True)

    def _init_distrib(self, args: argparse.Namespace) -> None:
        if "RANK" in os.environ and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)

        self.rank, self.world = _get_rank_world()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = args.device or (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

    def _init_layer(self, args: argparse.Namespace) -> None:
        if args.layer is None:
            raise ValueError("--layer is required when --config is not provided")
        layer = int(args.layer)

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
        if args.experts_dir is None:
            args.experts_dir = "data/experts"

        self.layer = layer

    def _get_experts(self):
        expert_ids = _discover_expert_indices(self.args.experts_dir, self.layer)
        experts = [
            torch.load(os.path.join(self.args.experts_dir, str(self.layer), f"{i}.pt"), weights_only=False)
            for i in expert_ids
        ]
        pairs = torch.load(self.args.pairs)
        pair_indices = [i for i in range(len(pairs)) if i % self.world == self.rank]
        return experts, pairs, pair_indices

    def _fuse(self, experts, pairs, pair_indices) -> None:
        pbar = tqdm(pair_indices, desc=f"rank {self.rank}/{self.world}")
        train_data = torch.load(self.args.train_inputs)
        train_pairs_acts = torch.load(self.args.pairs_train_acts)

        for pair_idx in pbar:
            out_path = os.path.join(self.out_dir, f"{pair_idx}.pt")
            if os.path.exists(out_path):
                continue

            e1, e2 = pairs[pair_idx]
            tr_idx = train_pairs_acts[pair_idx]
            if len(tr_idx) == 0:
                continue

            train_x = torch.stack([train_data[i] for i in tr_idx], dim=0).to(self.device)
            mlp1 = _turn_in_mlp(experts[e1].to(self.device))
            mlp2 = _turn_in_mlp(experts[e2].to(self.device))
            fused = _zipit_fuse(mlp1, mlp2, train_x)
            torch.save(fused.cpu(), out_path)

            del train_x, mlp1, mlp2, fused

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        if dist is not None and dist.is_initialized():
            dist.barrier()
            if self.rank == 0:
                print("[fusion_runner] done")

    def run(self) -> None:
        self._fuse(*self._get_experts())


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args = apply_config_defaults(args)

    _ensure_torch()
    FusionRunner(args).run()


if __name__ == "__main__":
    main()


