from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict

import torch

from common.rmt_tools import denoise_importance_gate_spikes, normalize_direction
from common.spectrum_tools import whitened_mean_bias


@dataclass
class H2PipelineResult:
    bias: torch.Tensor
    diagnostics: Dict[str, float | int]


@torch.no_grad()
def denoise_bias_covariance(
    importance_matrix: torch.Tensor,
    alpha: float,
    lam: float,
    normalize: str,
) -> H2PipelineResult:
    res = whitened_mean_bias(importance_matrix, alpha=alpha, lam=lam)
    bias = normalize_direction(res["bias"], mode=normalize)
    diag = {
        "method": "covariance",
        "mean_norm": float(torch.linalg.norm(res["mean"]).item()),
        "bias_norm": float(torch.linalg.norm(bias).item()),
    }
    return H2PipelineResult(bias=bias, diagnostics=diag)


@torch.no_grad()
def denoise_bias_gate_spikes(
    importance_vector: torch.Tensor,
    gate_weight: torch.Tensor,
    mp_factor: float,
    normalize: str,
) -> H2PipelineResult:
    bias_hat, info = denoise_importance_gate_spikes(
        importance=importance_vector,
        gate_weight=gate_weight,
        mp_factor=mp_factor,
        min_rank=1,
        wiener=True,
    )
    bias = normalize_direction(bias_hat, mode=normalize)
    diag = {
        "method": "gate_spike",
        "rank": int(info.rank),
        "lambda_plus": float(info.lambda_plus),
        "bias_norm": float(torch.linalg.norm(bias).item()),
    }
    return H2PipelineResult(bias=bias, diagnostics=diag)


@torch.no_grad()
def run_h2_pipeline(
    mode: str,
    output_bias_path: str,
    output_diagnostics_path: str,
    importance_matrix_path: str = "",
    importance_vector_path: str = "",
    gate_weight_path: str = "",
    alpha: float = 0.2,
    lam: float = 1e-3,
    mp_factor: float = 1.1,
    normalize: str = "maxabs",
    scale: float = 0.1,
) -> H2PipelineResult:
    if mode == "covariance":
        if not importance_matrix_path:
            raise ValueError("importance_matrix_path is required for covariance mode")
        x = torch.load(importance_matrix_path, map_location="cpu")
        if x.ndim != 2:
            raise ValueError(f"importance matrix must be 2D [N,E], got {tuple(x.shape)}")
        result = denoise_bias_covariance(x, alpha=alpha, lam=lam, normalize=normalize)
    elif mode == "gate_spike":
        if not importance_vector_path or not gate_weight_path:
            raise ValueError("importance_vector_path and gate_weight_path are required for gate_spike mode")
        v = torch.load(importance_vector_path, map_location="cpu").flatten()
        w = torch.load(gate_weight_path, map_location="cpu")
        result = denoise_bias_gate_spikes(v, w, mp_factor=mp_factor, normalize=normalize)
    else:
        raise ValueError(f"Unsupported H2 mode: {mode}")

    scaled_bias = result.bias * float(scale)
    out_bias = Path(output_bias_path)
    out_diag = Path(output_diagnostics_path)
    out_bias.parent.mkdir(parents=True, exist_ok=True)
    out_diag.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scaled_bias, out_bias)

    diag = dict(result.diagnostics)
    diag["scale"] = float(scale)
    with out_diag.open("w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    return H2PipelineResult(bias=scaled_bias, diagnostics=diag)
