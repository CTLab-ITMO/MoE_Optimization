from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch
import torch.nn.functional as functional

from moe_opt_core.experiment import EvaluationMetricsContract
from moe_opt_runner.transforms import model_parameter_mb


def _input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _loss_from_output(output, labels: torch.Tensor) -> torch.Tensor:
    loss = getattr(output, "loss", None)
    if loss is not None:
        return loss
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, Mapping):
        logits = output.get("logits")
    if logits is None:
        raise ValueError("Model output provides neither loss nor logits")
    return functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def evaluate_language_model(
    model,
    batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    warmup_steps: int,
    measure_steps: int,
) -> EvaluationMetricsContract:
    if not batches:
        raise ValueError("Evaluation batches are empty")
    device = _input_device(model)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    warmup = min(max(0, warmup_steps), len(batches))
    for batch in batches[:warmup]:
        model(**{key: value.to(device) for key, value in batch.items()})
    _synchronize(device)

    measured = batches[warmup : warmup + measure_steps]
    if not measured:
        measured = batches[: min(measure_steps, len(batches))]
    total_loss = 0.0
    total_tokens = 0
    start = time.perf_counter()
    for batch in measured:
        moved = {key: value.to(device) for key, value in batch.items()}
        output = model(**moved)
        labels = moved["labels"]
        tokens = int((labels[:, 1:] != -100).sum().item())
        loss = _loss_from_output(output, labels)
        total_loss += float(loss.item()) * tokens
        total_tokens += tokens
    _synchronize(device)
    elapsed = time.perf_counter() - start
    nll = total_loss / max(total_tokens, 1)
    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    return EvaluationMetricsContract(
        nll=nll,
        perplexity=float(math.exp(min(nll, 50.0))),
        tokens_per_second=float(total_tokens / max(elapsed, 1e-12)),
        peak_memory_mb=peak_memory_mb,
        model_parameter_mb=model_parameter_mb(model),
        evaluated_tokens=total_tokens,
        elapsed_seconds=float(elapsed),
    )


def append_incremental_record(path: str | Path, record: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
