from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch
from tqdm import tqdm

from moe_opt_core.config import ExperimentConfig
from moe_opt_core.experiment import CalibrationTraceContract, TraceSource
from moe_opt_core.experiment.data_planning import resolve_text_column, split_text_rows
from moe_opt_core.experiment.errors import ExpectedExperimentFailure, ExperimentFailureCode
from moe_opt_core.experiment.trace_validation import validate_trace_payload
from moe_opt_runner.runtime import RuntimeContext


def _model_input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dataset_texts(config: ExperimentConfig) -> Mapping[str, tuple[str, ...]]:
    try:
        import datasets
    except ModuleNotFoundError as exc:
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.DATASET_UNAVAILABLE,
            "The datasets package is required for calibration.source='hf_dataset'",
        ) from exc

    calibration = config.calibration
    args = [calibration.dataset_name]
    if calibration.dataset_config:
        args.append(calibration.dataset_config)
    try:
        dataset = datasets.load_dataset(*args, split=calibration.dataset_split)
    except Exception as exc:
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.DATASET_UNAVAILABLE,
            f"Failed to load calibration dataset '{calibration.dataset_name}'",
            context={"dataset_config": calibration.dataset_config, "split": calibration.dataset_split},
        ) from exc

    text_column = resolve_text_column(dataset.column_names, calibration.text_column)
    max_samples = min(len(dataset), calibration.max_samples)
    rows = [str(value).strip() for value in dataset.select(range(max_samples))[text_column]]
    rows = [row for row in rows if row]
    return split_text_rows(
        rows,
        fractions=(
            calibration.calibration_fraction,
            calibration.validation_fraction,
            calibration.test_fraction,
        ),
        seed=calibration.seed,
    )


def _tokenize_splits(
    config: ExperimentConfig,
    tokenizer,
    text_splits: Mapping[str, Sequence[str]],
) -> Dict[str, list[dict[str, torch.Tensor]]]:
    output: Dict[str, list[dict[str, torch.Tensor]]] = {}
    batch_size = config.calibration.batch_size
    for split, texts in text_splits.items():
        batches: list[dict[str, torch.Tensor]] = []
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                list(texts[start : start + batch_size]),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.calibration.max_length,
            )
            input_ids = encoded["input_ids"].cpu()
            attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).cpu()
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            batches.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )
        output[split] = batches
    return output


def _capture_gate_rows(
    gate,
    attention_mask_ref: dict[str, torch.Tensor | None],
    hidden_rows: list[torch.Tensor],
    logit_rows: list[torch.Tensor],
):
    def hook(_module, inputs, output) -> None:
        hidden = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
        logits = output[0] if isinstance(output, tuple) else output
        logits = logits.detach().reshape(-1, logits.shape[-1])
        mask = attention_mask_ref.get("value")
        if mask is not None and mask.numel() == hidden.shape[0]:
            valid = mask.reshape(-1).to(dtype=torch.bool, device=hidden.device)
            hidden = hidden[valid]
            logits = logits[valid.to(logits.device)]
        hidden_rows.append(hidden.float().cpu())
        logit_rows.append(logits.float().cpu())

    return gate.register_forward_hook(hook)


@torch.no_grad()
def collect_traces_from_batches(
    config: ExperimentConfig,
    ctx: RuntimeContext,
    split_batches: Mapping[str, Sequence[Mapping[str, torch.Tensor]]],
) -> tuple[CalibrationTraceContract, ...]:
    output_root = Path(config.calibration.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    device = _model_input_device(ctx.model)
    handles = []
    for layer in config.layers:
        handle = ctx.handles_by_layer.get(layer)
        if handle is None:
            raise ExpectedExperimentFailure(
                ExperimentFailureCode.LAYER_UNAVAILABLE,
                f"Layer {layer} is not available for trace collection",
            )
        handles.append(handle)

    contracts: list[CalibrationTraceContract] = []
    ctx.model.eval()
    for split, batches in split_batches.items():
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        token_batches_path = split_dir / "token_batches.pt"
        cpu_batches = [
            {key: value.detach().cpu() for key, value in batch.items()}
            for batch in batches
        ]
        torch.save(cpu_batches, token_batches_path)

        attention_mask_ref: dict[str, torch.Tensor | None] = {"value": None}
        captured: dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]] = {
            int(handle.layer_idx): ([], []) for handle in handles
        }
        hooks = [
            _capture_gate_rows(
                handle.gate,
                attention_mask_ref,
                captured[int(handle.layer_idx)][0],
                captured[int(handle.layer_idx)][1],
            )
            for handle in handles
        ]
        try:
            for batch in tqdm(batches, desc=f"Collect traces [{split}]", leave=False):
                attention_mask_ref["value"] = batch.get("attention_mask")
                model_batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                    if key != "labels"
                }
                ctx.model(**model_batch)
        finally:
            for hook in hooks:
                hook.remove()

        for handle in handles:
            layer = int(handle.layer_idx)
            runtime_contract = ctx.runtime_contracts_by_layer.get(layer)
            if runtime_contract is None:
                raise ExpectedExperimentFailure(
                    ExperimentFailureCode.RUNTIME_ARCHITECTURE_MISMATCH,
                    f"Layer {layer} has no validated runtime contract",
                )
            hidden_parts, logit_parts = captured[layer]
            if not hidden_parts or not logit_parts:
                raise ExpectedExperimentFailure(
                    ExperimentFailureCode.REAL_TRACES_REQUIRED,
                    f"No gate traces were captured for layer {layer}",
                )
            hidden = torch.cat(hidden_parts, dim=0)
            logits = torch.cat(logit_parts, dim=0)
            if logits.shape[1] != runtime_contract.routed_experts:
                raise ExpectedExperimentFailure(
                    ExperimentFailureCode.RUNTIME_ARCHITECTURE_MISMATCH,
                    f"Router output width does not match routed experts at layer {layer}",
                    context={
                        "router_output_width": int(logits.shape[1]),
                        "routed_experts": runtime_contract.routed_experts,
                    },
                )
            probs = torch.softmax(logits, dim=-1)
            top_k = runtime_contract.effective_routed_top_k
            selected = torch.topk(probs, k=top_k, dim=-1).indices
            trace_path = split_dir / f"layer_{layer}.pt"
            trace_payload = {
                "metadata": {
                    "source": config.calibration.source,
                    "layer": layer,
                    "split": split,
                    "routed_experts": runtime_contract.routed_experts,
                    "shared_experts": runtime_contract.shared_experts,
                    "routed_top_k": top_k,
                },
                "hidden_states": hidden,
                "router_logits": logits,
                "routing_probs": probs,
                "selected_experts": selected,
            }
            validate_trace_payload(
                trace_payload,
                runtime_contract,
                require_real_traces=config.pilot.require_real_traces,
            )
            torch.save(trace_payload, trace_path)
            contracts.append(
                CalibrationTraceContract(
                    source=TraceSource(config.calibration.source),
                    layer=layer,
                    split=split,
                    hidden_states_path=str(trace_path),
                    routing_probs_path=str(trace_path),
                    token_batches_path=str(token_batches_path),
                    rows=int(hidden.shape[0]),
                    tokens=int(hidden.shape[0]),
                    sha256=_sha256_file(trace_path),
                    routed_experts=runtime_contract.routed_experts,
                    shared_experts=runtime_contract.shared_experts,
                    routed_top_k=top_k,
                )
            )

            if split == "calibration" and layer == config.h1.layer:
                h1_path = Path(config.h1.mlp_inputs_path)
                h1_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(hidden, h1_path)
                h3_path = Path(config.h3.routing_probs_path)
                h3_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(probs, h3_path)

    manifest_path = output_root / "trace_manifest.json"
    existing_payload = []
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            existing_payload = loaded
    rows_by_key = {
        (int(row["layer"]), str(row["split"])): row
        for row in existing_payload
        if isinstance(row, dict) and "layer" in row and "split" in row
    }
    for contract in contracts:
        row = asdict(contract)
        row["source"] = contract.source.value
        rows_by_key[(contract.layer, contract.split)] = row
    payload = [rows_by_key[key] for key in sorted(rows_by_key, key=lambda item: (item[1], item[0]))]
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return tuple(contracts)


def collect_real_calibration_traces(
    config: ExperimentConfig,
    ctx: RuntimeContext,
    splits: Sequence[str] | None = None,
) -> tuple[CalibrationTraceContract, ...]:
    if config.calibration.source != TraceSource.HF_DATASET.value:
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.REAL_TRACES_REQUIRED,
            "Real trace collection requires calibration.source='hf_dataset'",
        )
    if ctx.tokenizer is None:
        raise ExpectedExperimentFailure(
            ExperimentFailureCode.MODEL_UNAVAILABLE,
            "Runtime tokenizer is required for real trace collection",
        )
    texts = _load_dataset_texts(config)
    requested = tuple(splits) if splits is not None else tuple(texts)
    unknown = sorted(set(requested) - set(texts))
    if unknown:
        raise ValueError(f"Unknown trace splits: {unknown}")
    selected_texts = {split: texts[split] for split in requested}
    batches = _tokenize_splits(config, ctx.tokenizer, selected_texts)
    return collect_traces_from_batches(config, ctx, batches)
