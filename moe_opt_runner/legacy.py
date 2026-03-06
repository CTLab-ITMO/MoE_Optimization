from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import torch

from moe_opt_core.artifacts import write_artifact_manifest as write_artifact_manifest_file, write_run_summary
from moe_opt_core.config import config_hash, load_experiment_config
from moe_opt_runner.runtime import build_runtime_context
from moe_opt_runner.stages import run_h1_stage


def _inputs_dir_from_config(config) -> Path:
    return Path(config.h1.mlp_inputs_path).parent


def _activations_dir_from_config(config) -> Path:
    return Path(config.artifacts.root_dir) / "legacy_activations"


def _legacy_inputs_dir() -> Path:
    return Path("data") / "mlp_inputs"


def _legacy_activations_dir() -> Path:
    return Path("data") / "activations"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _collect_present_paths(paths: dict[str, str | Path | None]) -> dict[str, str]:
    present: dict[str, str] = {}
    for key, value in paths.items():
        if value is None:
            continue
        rendered = str(value).strip()
        if not rendered:
            continue
        candidate = Path(rendered)
        if candidate.exists():
            present[str(key)] = rendered
    return present


def _write_legacy_artifact_contract(
    config,
    *,
    step: str,
    source_config_path: str,
    extra_paths: dict[str, str | Path | None] | None = None,
    summary_payload: dict[str, object] | None = None,
) -> None:
    manifest_path = Path(config.artifacts.root_dir) / f"legacy_manifest_{step}.json"
    paths: dict[str, str | Path | None] = {
        "source_config": Path(source_config_path).resolve(),
        "stage_manifest": config.artifacts.manifest_path,
        "cfg_inputs_dir": _inputs_dir_from_config(config),
        "legacy_inputs_dir": _legacy_inputs_dir(),
        "cfg_activations_dir": _activations_dir_from_config(config),
        "legacy_activations_dir": _legacy_activations_dir(),
        "h1_pairs": config.h1.output_pairs_path,
        "h1_diag": config.h1.output_diagnostics_path,
        "h2_bias": config.h2.output_bias_path,
        "h3_diag": config.h3.output_diagnostics_path,
    }
    if extra_paths:
        paths.update(extra_paths)

    write_artifact_manifest_file(
        schema_version=config.artifacts.schema_version,
        config_hash=config_hash(config),
        root_dir=config.artifacts.root_dir,
        manifest_path=manifest_path,
        paths=_collect_present_paths(paths),
    )

    if summary_payload is not None:
        write_run_summary(
            root_dir=config.artifacts.root_dir,
            summary_name=f"legacy_summary_{step}.json",
            payload=summary_payload,
        )


def _pairs_path_candidates(config, layer: int) -> list[Path]:
    return [
        Path(config.h1.output_pairs_path),
        Path("data") / f"pairs_{layer}.pt",
    ]


def _load_pairs_for_layer(config, layer: int):
    for p in _pairs_path_candidates(config, layer):
        if p.exists():
            return torch.load(p, map_location="cpu")
    return None


def _pair_activation_indices(activations: torch.Tensor, pairs, num_experts: int) -> list[list[int]]:
    pair_idx = [-1] * int(num_experts)
    for i, (a, b) in enumerate(pairs):
        pair_idx[int(a)] = i
        pair_idx[int(b)] = i

    out: list[list[int]] = [[] for _ in range(len(pairs))]
    for row_idx in range(int(activations.shape[0])):
        row = activations[row_idx]
        if row.ndim == 0:
            continue
        vals = [int(v) for v in row.tolist()]
        m = len(vals)
        for i in range(m):
            for j in range(i + 1, m):
                pi = pair_idx[vals[i]] if 0 <= vals[i] < len(pair_idx) else -1
                pj = pair_idx[vals[j]] if 0 <= vals[j] < len(pair_idx) else -1
                if pi >= 0 and pi == pj:
                    out[pi].append(row_idx)
    return out

def _set_legacy_config_env(config_path: str) -> None:
    os.environ["MOE_EXPERIMENT_CONFIG"] = str(Path(config_path).resolve())


def _select_layer(config, layer: int | None) -> int:
    if layer is not None:
        return int(layer)
    if not config.layers:
        raise ValueError("Config has empty layers list")
    return int(config.layers[0])


def _load_train_inputs_for_layer(config, layer: int) -> torch.Tensor:
    p_cfg = _inputs_dir_from_config(config) / f"train_{layer}.pt"
    p_legacy = _legacy_inputs_dir() / f"train_{layer}.pt"

    path = p_cfg if p_cfg.exists() else p_legacy
    if not path.exists():
        raise FileNotFoundError(f"Train inputs not found for layer {layer}: {p_cfg} or {p_legacy}")

    x = torch.load(path, map_location="cpu")
    if isinstance(x, list):
        if not x:
            raise ValueError(f"Empty train inputs list: {path}")
        x = torch.stack(x)
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Unsupported train input type: {type(x)}")
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        raise ValueError(f"Train inputs must be 2D [N,d], got {tuple(x.shape)}")
    return x.float()


def _to_batch(x: torch.Tensor, start: int, batch_size: int, device: torch.device) -> torch.Tensor:
    chunk = x[start : start + batch_size]
    if chunk.ndim == 1:
        chunk = chunk.unsqueeze(0)
    return chunk.to(device)


def _extract_text_column(dataset_obj) -> list[str]:
    if hasattr(dataset_obj, "column_names"):
        for key in ["question", "text", "prompt", "input"]:
            if key in dataset_obj.column_names:
                vals = dataset_obj[key]
                return [str(v) for v in vals]
        first = dataset_obj.column_names[0]
        vals = dataset_obj[first]
        return [str(v) for v in vals]
    raise ValueError("Unsupported dataset object format")


def _load_test_prompts(config, examples: int) -> list[str]:
    import datasets

    ds_spec = config.constants.get("training", {}).get("test_dataset_name", ["cais/mmlu", "abstract_algebra"])
    if not isinstance(ds_spec, (list, tuple)) or not ds_spec:
        raise ValueError("training.test_dataset_name must be a non-empty list")

    if len(ds_spec) == 1:
        ds = datasets.load_dataset(ds_spec[0], split="test")
    else:
        ds = datasets.load_dataset(ds_spec[0], ds_spec[1], split="test")

    texts = _extract_text_column(ds)
    if not texts:
        raise ValueError("No test prompts loaded from dataset")
    return texts[: max(1, examples)]


def _generate_texts(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    outs: list[str] = []
    for prompt in prompts:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        gen = model.generate(**inputs, max_new_tokens=max_new_tokens)
        outs.append(tokenizer.decode(gen[0], skip_special_tokens=True))
    return outs


@torch.no_grad()
def export_experts(config, ctx) -> None:
    for layer in config.layers:
        handle = ctx.handles_by_layer.get(layer)
        if handle is None:
            continue

        base = Path(config.h1.experts_dir) / str(layer)
        _ensure_dir(base)

        experts = list(handle.experts)
        for i, expert in enumerate(experts):
            torch.save(expert, base / f"{i}.pt")
        torch.save(handle.gate, base / "gate.pt")

        shared_expert = getattr(handle.moe, "shared_expert", None)
        shared_expert_gate = getattr(handle.moe, "shared_expert_gate", None)
        if shared_expert is not None:
            torch.save(shared_expert, base / "shared_expert.pt")
        if shared_expert_gate is not None:
            torch.save(shared_expert_gate, base / "shared_expert_gate.pt")


@torch.no_grad()
def collect_mlp_inputs(config, ctx) -> None:
    seed = int(config.constants["training"]["seed"])
    n = int(config.h1.n_inputs)
    test_n = max(256, n // 4)

    out_dir_cfg = _inputs_dir_from_config(config)
    out_dir_legacy = _legacy_inputs_dir()
    _ensure_dir(out_dir_cfg)
    _ensure_dir(out_dir_legacy)

    for layer in config.layers:
        handle = ctx.handles_by_layer.get(layer)
        if handle is None:
            continue

        gate_w = ctx.adapter.get_gate_weight(handle.gate).detach().cpu()
        d = int(gate_w.shape[1])

        g1 = torch.Generator(device="cpu").manual_seed(seed + layer)
        g2 = torch.Generator(device="cpu").manual_seed(seed + 10_000 + layer)
        x_train = torch.randn((n, d), generator=g1)
        x_test = torch.randn((test_n, d), generator=g2)

        cfg_train = out_dir_cfg / f"train_{layer}.pt"
        cfg_test = out_dir_cfg / f"test_{layer}.pt"
        legacy_train = out_dir_legacy / f"train_{layer}.pt"
        legacy_test = out_dir_legacy / f"test_{layer}.pt"

        torch.save(x_train, cfg_train)
        torch.save(x_test, cfg_test)
        if legacy_train != cfg_train:
            torch.save(x_train, legacy_train)
        if legacy_test != cfg_test:
            torch.save(x_test, legacy_test)

        if config.h1.layer == layer:
            p = Path(config.h1.mlp_inputs_path)
            _ensure_dir(p.parent)
            if p != cfg_train:
                shutil.copy2(cfg_train, p)


@torch.no_grad()
def build_activations(config, ctx) -> None:
    inputs_cfg = _inputs_dir_from_config(config)
    inputs_legacy = _legacy_inputs_dir()
    out_cfg = _activations_dir_from_config(config)
    out_legacy = _legacy_activations_dir()
    _ensure_dir(out_cfg)
    _ensure_dir(out_legacy)

    for layer in config.layers:
        handle = ctx.handles_by_layer.get(layer)
        if handle is None:
            continue

        x_train_path = inputs_cfg / f"train_{layer}.pt"
        x_test_path = inputs_cfg / f"test_{layer}.pt"
        if not x_train_path.exists() or not x_test_path.exists():
            x_train_path = inputs_legacy / f"train_{layer}.pt"
            x_test_path = inputs_legacy / f"test_{layer}.pt"

        if not x_train_path.exists() or not x_test_path.exists():
            raise FileNotFoundError("MLP inputs are missing. Run collect_mlp_inputs first.")

        x_train = torch.load(x_train_path, map_location="cpu").float()
        x_test = torch.load(x_test_path, map_location="cpu").float()

        gate_w = ctx.adapter.get_gate_weight(handle.gate).detach().cpu().float()
        top_k = int(config.runtime_top_k)

        train_probs = torch.softmax(x_train @ gate_w.t(), dim=-1)
        test_probs = torch.softmax(x_test @ gate_w.t(), dim=-1)
        _, train_sel = torch.topk(train_probs, k=min(top_k, train_probs.shape[1]), dim=-1)
        _, test_sel = torch.topk(test_probs, k=min(top_k, test_probs.shape[1]), dim=-1)

        torch.save([row for row in train_sel], out_cfg / f"train_{layer}.pt")
        torch.save([row for row in test_sel], out_cfg / f"test_{layer}.pt")
        torch.save([row for row in train_sel], out_legacy / f"train_{layer}.pt")
        torch.save([row for row in test_sel], out_legacy / f"test_{layer}.pt")

        pairs = _load_pairs_for_layer(config, layer)
        if pairs is not None and len(pairs) > 0:
            num_experts = int(gate_w.shape[0])
            train_pair_idx = _pair_activation_indices(train_sel, pairs, num_experts=num_experts)
            test_pair_idx = _pair_activation_indices(test_sel, pairs, num_experts=num_experts)
            torch.save(train_pair_idx, out_cfg / f"pairs_train_{layer}.pt")
            torch.save(test_pair_idx, out_cfg / f"pairs_test_{layer}.pt")
            torch.save(train_pair_idx, out_legacy / f"pairs_train_{layer}.pt")
            torch.save(test_pair_idx, out_legacy / f"pairs_test_{layer}.pt")

        if config.h1.layer == layer:
            p = Path(config.h3.routing_probs_path)
            _ensure_dir(p.parent)
            torch.save(train_probs, p)


@torch.no_grad()
def find_pairs(config, ctx) -> dict[str, object]:
    out = run_h1_stage(config, ctx=ctx)
    src = Path(config.h1.output_pairs_path)
    if src.exists():
        legacy_pairs = Path("data") / f"pairs_{config.h1.layer}.pt"
        _ensure_dir(legacy_pairs.parent)
        if legacy_pairs != src:
            shutil.copy2(src, legacy_pairs)
    return out


@torch.no_grad()
def run_final_benchmark_from_config(
    config_path: str,
    layer: int | None = None,
    batch_size: int = 2048,
    total_batches: int = 10,
) -> dict[str, float | int | str]:
    config = load_experiment_config(config_path)
    target_layer = _select_layer(config, layer)
    _set_legacy_config_env(config_path)

    from Qwen_MoE_experts_fusion_experiment.fusion_layer import get_fusion_layer_by_index

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = _load_train_inputs_for_layer(config, target_layer)
    model = get_fusion_layer_by_index(target_layer, device=device).to(device)

    n_total = min(x.shape[0], int(batch_size) * int(total_batches))
    if n_total <= 0:
        raise ValueError("No data available for benchmark")

    err_sum = 0.0
    norm_sum = 0.0
    steps = 0
    t_fused = 0.0
    t_default = 0.0

    for start in range(0, n_total, int(batch_size)):
        batch = _to_batch(x, start, int(batch_size), device)

        t0 = time.perf_counter()
        y_fused = model.forward(batch)[0]
        t1 = time.perf_counter()
        y_default = model.forward_default(batch)[0]
        t2 = time.perf_counter()

        err_sum += float((y_fused - y_default).square().mean().item())
        norm_sum += float(y_default.square().mean().item())
        t_fused += float(t1 - t0)
        t_default += float(t2 - t1)
        steps += 1

    rel_mse = err_sum / max(norm_sum, 1e-12)
    out = {
        "layer": int(target_layer),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "samples": int(n_total),
        "err_mean": float(err_sum / max(steps, 1)),
        "norm_mean": float(norm_sum / max(steps, 1)),
        "rel_mse": float(rel_mse),
        "time_fused_s": float(t_fused),
        "time_default_s": float(t_default),
        "speedup_default_over_fused": float(t_default / max(t_fused, 1e-12)),
    }

    out_path = Path(config.artifacts.root_dir) / f"legacy_forward_benchmark_layer{target_layer}.json"
    _ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["report_path"] = str(out_path)
    _write_legacy_artifact_contract(
        config,
        step=f"benchmark_layer{target_layer}",
        source_config_path=config_path,
        extra_paths={"report_path": out_path},
        summary_payload=out,
    )
    return out


@torch.no_grad()
def run_generation_from_config(
    config_path: str,
    examples: int = 10,
    max_new_tokens: int = 40,
) -> dict[str, object]:
    config = load_experiment_config(config_path)
    _set_legacy_config_env(config_path)

    import transformers
    from Qwen_MoE_experts_fusion_experiment.fusion_layer import get_fusion_layer_by_index

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = transformers.AutoModelForCausalLM.from_pretrained(config.model.name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model.name)
    model = model.to(device)

    prompts = _load_test_prompts(config, examples=int(examples))
    base_outputs = _generate_texts(model, tokenizer, prompts, max_new_tokens=int(max_new_tokens))

    for lyr in config.layers:
        model.model.layers[int(lyr)].mlp = get_fusion_layer_by_index(int(lyr), device=device).to(device)

    fused_outputs = _generate_texts(model, tokenizer, prompts, max_new_tokens=int(max_new_tokens))

    rows = []
    for prompt, base, fused in zip(prompts, base_outputs, fused_outputs):
        rows.append({"prompt": prompt, "base": base, "fused": fused})

    out_path = Path(config.artifacts.root_dir) / "legacy_generation_compare.json"
    _ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    out = {
        "examples": int(len(rows)),
        "max_new_tokens": int(max_new_tokens),
        "report_path": str(out_path),
    }
    _write_legacy_artifact_contract(
        config,
        step="generation",
        source_config_path=config_path,
        extra_paths={"report_path": out_path},
        summary_payload=out,
    )
    return out


@torch.no_grad()
def export_experts_from_config(config_path: str) -> None:
    config = load_experiment_config(config_path)
    ctx = build_runtime_context(config)
    export_experts(config, ctx)

    extra_paths = {f"experts_layer_{int(layer)}": Path(config.h1.experts_dir) / str(int(layer)) for layer in config.layers}
    _write_legacy_artifact_contract(
        config,
        step="export_experts",
        source_config_path=config_path,
        extra_paths=extra_paths,
    )


@torch.no_grad()
def collect_mlp_inputs_from_config(config_path: str) -> None:
    config = load_experiment_config(config_path)
    ctx = build_runtime_context(config)
    collect_mlp_inputs(config, ctx)

    inputs_cfg = _inputs_dir_from_config(config)
    inputs_legacy = _legacy_inputs_dir()
    extra_paths: dict[str, str | Path] = {
        "h1_train_input": config.h1.mlp_inputs_path,
        "cfg_inputs_dir": inputs_cfg,
        "legacy_inputs_dir": inputs_legacy,
    }
    for layer in config.layers:
        lyr = int(layer)
        extra_paths[f"cfg_train_layer_{lyr}"] = inputs_cfg / f"train_{lyr}.pt"
        extra_paths[f"cfg_test_layer_{lyr}"] = inputs_cfg / f"test_{lyr}.pt"
        extra_paths[f"legacy_train_layer_{lyr}"] = inputs_legacy / f"train_{lyr}.pt"
        extra_paths[f"legacy_test_layer_{lyr}"] = inputs_legacy / f"test_{lyr}.pt"

    _write_legacy_artifact_contract(
        config,
        step="collect_mlp_inputs",
        source_config_path=config_path,
        extra_paths=extra_paths,
    )


@torch.no_grad()
def build_activations_from_config(config_path: str) -> None:
    config = load_experiment_config(config_path)
    ctx = build_runtime_context(config)
    build_activations(config, ctx)

    acts_cfg = _activations_dir_from_config(config)
    acts_legacy = _legacy_activations_dir()
    extra_paths: dict[str, str | Path] = {
        "h3_routing_probs": config.h3.routing_probs_path,
        "cfg_activations_dir": acts_cfg,
        "legacy_activations_dir": acts_legacy,
    }
    for layer in config.layers:
        lyr = int(layer)
        extra_paths[f"cfg_act_train_layer_{lyr}"] = acts_cfg / f"train_{lyr}.pt"
        extra_paths[f"cfg_act_test_layer_{lyr}"] = acts_cfg / f"test_{lyr}.pt"
        extra_paths[f"legacy_act_train_layer_{lyr}"] = acts_legacy / f"train_{lyr}.pt"
        extra_paths[f"legacy_act_test_layer_{lyr}"] = acts_legacy / f"test_{lyr}.pt"
        extra_paths[f"cfg_pairs_train_layer_{lyr}"] = acts_cfg / f"pairs_train_{lyr}.pt"
        extra_paths[f"cfg_pairs_test_layer_{lyr}"] = acts_cfg / f"pairs_test_{lyr}.pt"
        extra_paths[f"legacy_pairs_train_layer_{lyr}"] = acts_legacy / f"pairs_train_{lyr}.pt"
        extra_paths[f"legacy_pairs_test_layer_{lyr}"] = acts_legacy / f"pairs_test_{lyr}.pt"

    _write_legacy_artifact_contract(
        config,
        step="build_activations",
        source_config_path=config_path,
        extra_paths=extra_paths,
    )


@torch.no_grad()
def find_pairs_from_config(config_path: str) -> dict[str, object]:
    config = load_experiment_config(config_path)
    ctx = build_runtime_context(config) if config.h1.experts_source == "hf_model" else None
    out = find_pairs(config, ctx)
    _write_legacy_artifact_contract(
        config,
        step="find_pairs",
        source_config_path=config_path,
        extra_paths={"legacy_pairs": Path("data") / f"pairs_{config.h1.layer}.pt"},
        summary_payload=out,
    )
    return out


@torch.no_grad()
def prepare_legacy_data_from_config(config_path: str, include_experts: bool = True) -> None:
    config = load_experiment_config(config_path)
    ctx = build_runtime_context(config)

    if include_experts:
        export_experts(config, ctx)
    collect_mlp_inputs(config, ctx)
    build_activations(config, ctx)

    _write_legacy_artifact_contract(
        config,
        step="prepare_legacy_data",
        source_config_path=config_path,
        extra_paths={
            "experts_root": config.h1.experts_dir,
            "cfg_inputs_dir": _inputs_dir_from_config(config),
            "cfg_activations_dir": _activations_dir_from_config(config),
            "legacy_inputs_dir": _legacy_inputs_dir(),
            "legacy_activations_dir": _legacy_activations_dir(),
        },
        summary_payload={
            "include_experts": bool(include_experts),
            "layers": [int(layer) for layer in config.layers],
        },
    )








