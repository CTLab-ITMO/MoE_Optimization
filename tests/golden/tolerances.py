"""Two-family tolerance policy and comparator for the golden snapshot.

* **tight** -- deterministic, RNG-free quantities (exact ``Partition`` and
  orientation, registry snapshot, reconstruction/accuracy of deterministic
  fusion methods). Narrow rtol/atol; effectively exact on a single CPU thread.
* **loose** -- order-invariant aggregates for stochastic methods (Fisher's
  ``randn_like`` and, from S4 on, random pairing / output_similarity subsample).
  Wide rtol; ``index_add_``/bf16 noise means these are never compared to the
  byte.

In S0 the legacy and new modules are verbatim copies running with an identical
seed and traversal order, so even the stochastic methods reproduce exactly; the
loose family is applied anyway to exercise the machinery the later (S4) golden
re-basis relies on.
"""
import math

TIGHT_RTOL = 1e-5
TIGHT_ATOL = 1e-7
LOOSE_RTOL = 0.25
LOOSE_ATOL = 1e-3

# Methods whose fused weights depend on RNG -> loose family.
STOCHASTIC_FUSIONS = {"Fisher", "LS_Fisher", "Distillation", "LS_Distillation"}


def _tol_for(method: str):
    if method in STOCHASTIC_FUSIONS:
        return LOOSE_RTOL, LOOSE_ATOL
    return TIGHT_RTOL, TIGHT_ATOL


def _close(a: float, b: float, rtol: float, atol: float) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def diff_partition(actual: dict, expected: dict) -> list:
    """Exact equality of pairs, orientation and ``i.pt`` order."""
    problems = []
    if set(actual) != set(expected):
        problems.append(f"partition layers {sorted(actual)} != {sorted(expected)}")
        return problems
    for layer in expected:
        if actual[layer] != expected[layer]:
            problems.append(
                f"partition layer {layer}: {actual[layer]} != {expected[layer]}"
            )
    return problems


def diff_registry(actual: dict, expected: dict) -> list:
    problems = []
    for key in ("fusion_methods", "metrics", "pairing_methods"):
        if actual.get(key) != expected.get(key):
            problems.append(f"registry.{key}: {actual.get(key)} != {expected.get(key)}")
    if actual.get("best_metrics") != expected.get("best_metrics"):
        problems.append("registry.best_metrics differs")
    return problems


def diff_reconstruction(actual: dict, expected: dict) -> list:
    problems = []
    if set(actual) != set(expected):
        problems.append(f"reconstruction methods {sorted(actual)} != {sorted(expected)}")
        return problems
    for method in expected:
        rtol, atol = _tol_for(method)
        for layer in expected[method]:
            for metric, exp_value in expected[method][layer].items():
                act_value = actual[method][layer][metric]
                if not _close(act_value, exp_value, rtol, atol):
                    problems.append(
                        f"reconstruction[{method}][{layer}][{metric}]: "
                        f"{act_value} != {exp_value} (rtol={rtol})"
                    )
    return problems


def diff_accuracy(actual: dict, expected: dict) -> list:
    problems = []
    if set(actual) != set(expected):
        problems.append(f"accuracy methods {sorted(actual)} != {sorted(expected)}")
        return problems
    for method in expected:
        rtol, atol = _tol_for(method)
        exp_list = expected[method]
        act_list = actual[method]
        if len(exp_list) != len(act_list):
            problems.append(
                f"accuracy[{method}] length {len(act_list)} != {len(exp_list)}"
            )
            continue
        for i, (act_value, exp_value) in enumerate(zip(act_list, exp_list)):
            if not _close(act_value, exp_value, rtol, atol):
                problems.append(
                    f"accuracy[{method}][{i}]: {act_value} != {exp_value} (rtol={rtol})"
                )
    return problems


def diff_aggregates(actual: dict, expected: dict) -> list:
    """All non-registry aggregate diffs (partition + reconstruction + accuracy)."""
    problems = []
    problems += diff_partition(actual["partition"], expected["partition"])
    problems += diff_reconstruction(actual["reconstruction"], expected["reconstruction"])
    problems += diff_accuracy(actual["accuracy"], expected["accuracy"])
    return problems
