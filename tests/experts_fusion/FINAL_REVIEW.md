# S10 — Final Review: anti-regression + requirements compliance

Final, code-level audit of the `experts_fusion/` pipeline-of-record against the
S0 baseline (`git ee900a9`, the verbatim 9-stage copy) and against
`Требования к новому модулю.md` (§1–§6). Two questions are answered **from the
code**, not from another test run:

1. **Anti-regression** — did behaviour change anywhere "out of nowhere"? Every
   value / order / orientation / layout / RNG-order difference from the original
   `golden_v1` must be either absent or explained by a **conscious decision**.
2. **Compliance** — is each §1–§6 requirement realised, or explicitly logged as
   conscious debt?

## Method

A multi-agent audit (24 agents) reviewed the module in parallel: five
anti-regression reviewers by subsystem (fusion / grouping / serialization /
routing+eval / stage-seams) and six compliance reviewers by section (§1–§6).
Every raised finding and every `partial`/`debt` status was then **adversarially
verified** by an independent skeptic instructed to refute it against the live
code. Only verdicts that survived refutation are reported below.

The numeric oracle (`tests/golden/golden_v1.json` + `tests/golden/harness.py`)
holds **both** modules to the same snapshot and is **green on both trees**
(exact `activation_correlation` `Partition` with tuple orientation, per-layer
reconstruction MSE/RMSE/Cosine/R2, single-mode `accuracy_list` per method,
registry snapshot). This review explains the code behind that, and covers what
golden does **not** exercise.

## Verdict

The new module is **behaviour-equivalent to the S0 baseline by default**, with
the deviations enumerated in the ledger below. Every behavioural difference is
accounted for. The remaining §1–§6 gaps are **behaviour-neutral** against
S0/legacy (golden stays green) and are logged here as conscious debt.

---

## 1. Sanctioned-deviation ledger

These are the **only** behavioural differences from S0. Each is deliberate.

| # | Deviation | Where | Why sanctioned | Golden |
|---|---|---|---|---|
| S4 | Derived local seeds `seed_for(stage,layer,group,method)` + `torch.Generator` replace the global RNG at the stochastic points (Fisher `randn_like`; random pairing `randperm`; output-similarity subsample `randperm`) | `common/seeding.py`, `common/context.py`, `FisherFusion`, `RandomPartitioner`, `OutputSimilarityAffinity` | Removes a latent traversal-order dependence (§3.3); order-invariance proven both ways | loose family re-based under `UPDATE_GOLDEN` |
| S5 | `Best_<metric>` fallback uses `next(iter(methods))` (= first registered) instead of hardcoded `"Pruning"` | `common/mlp_utils.py::select_best_fused` | Fixes the §5 `KeyError`-under-`FILTER_FUSIONS` bug | re-based under flag (low-activation pairs only) |
| load-once | Stage 4 loads the base model **once**, snapshots the original mlp once, restores it between methods | `experts_fusion/4_evaluate.py` | §3.4 perf; proven value-identical to load-per-method (complete restore, per-call eval-RNG reset) | value-neutral |
| S8 FakeMoE | `FakeMoE` made deterministic so the §3.4 single-pass capture matches | `common/testing/fakes.py` | enables the 4-stage single-pass golden | re-based once (authorised) |
| **S10 sorted()** | **Stage-2 join canonicalises per-group example indices with `sorted()`** instead of CPython set-iteration order | `experts_fusion/2_grouping.py:70` | **see §2 below — Kostya's call: sanctioned** | golden blind (small data, Distillation excluded) |

### Refuted (not a deviation)

- **Serializer dtype up-cast** (`artifact_to_mlp` rebuilds the `AdvancedMLP` at
  the ctor-default float32). Raised as a finding, **refuted**: the legacy stage-9
  loader applies the *same* `.to(model.dtype)` recast (`9_test.py:97-100`), so
  the gated path (`accuracy_list`) is bit-identical to legacy; bf16→f32→bf16 is
  lossless. Narrow note: the serializer is not *self-contained* dtype-preserving
  (it relies on the downstream recast that the legacy loader also has). No gated
  value changes. Logged as a hardening note, not a regression.

---

## 2. The one confirmed behaviour finding — stage-2 `sorted()`

**Finding.** `join_group_activations` returns `[sorted(examples) …]`
(`2_grouping.py:70`) where the legacy stage 5 returned `list(map(list, …))` =
CPython set-iteration order (`5_activations.py:96`, identical at S0). The **set**
of activating example indices is identical, but at real scale (per-token rows in
the tens of thousands) set-iteration order is **not** ascending, so the
calibration-data **row order** handed to fusion differs.

**Impact.** Value-neutral for the order-invariant reductions
(Average/Pruning/ZipIt/Fisher/Procrustes/TIES — the headline metrics) and for the
default path. Value-**affecting** only for `DistillationFusion` /
`LS_Distillation`, which run sequential mini-batch SGD over the rows with no
shuffle. The golden cannot catch it: `GOLDEN_SIZE=8` (set order == sorted at that
scale) and `GOLDEN_FUSIONS` excludes Distillation, so Distillation parity is
unexercised at any scale.

**Decision (Kostya): sanction `sorted()`** as a deliberate determinism
improvement. Rationale: the legacy order was a non-reproducible CPython
implementation detail; `sorted()` makes the calibration row order deterministic
and reproducible across Python builds — the same spirit as S4 (§1.7/§3.3). All
golden-covered methods and the default path are order-invariant, so golden stays
green; byte-parity with the legacy order is, in any case, not verifiable without
a real-scale GPU run. Recorded as the **S10 sorted()** ledger entry above.

---

## 3. S10 actions taken (Kostya's decisions)

Beyond the audit + report, two §6 regressions were **restored** per Kostya's
call (the third decision, `sorted()`, is documentation-only — see §2):

- **IO retry restored** (§6 `IO_SAVE_RETRIES`). The flaky-`torch.save` retry the
  legacy stage carried is re-introduced on the centralized save path:
  `common/serialization.py::save` now retries `save_file` up to
  `IO_SAVE_RETRIES = 3` times and re-raises the last error if all fail. safetensors
  never pickles (the original flap cause is gone), but a transient filesystem
  failure can still hit the single write. Tests:
  `tests/common/test_serialization.py::test_save_retries_transient_io_then_succeeds`,
  `::test_save_reraises_after_retries_exhausted`.
- **Plotting restored under `DATA_DIR`** (§6 `MAX_PLOT_METRIC_VALUE`). The legacy
  stage-6 metric-distribution plots are ported to `common/plots.py`
  (`plot_fusion_metric_grids` + `metric_medians`, `MAX_PLOT_METRIC_VALUE = 1e6`)
  and written under `layout.plots_dir(DATA_DIR, "fusion")` — never CWD-relative
  `plots/` (§1.4). Stage 3 collects per-group reconstruction reports and calls it
  best-effort (a plotting failure never breaks fusion). Head-less (Agg), seaborn
  optional. Tests: `tests/common/test_plots.py` (6),
  `tests/common/test_layout.py::test_plots_dir_is_scoped_under_data_dir`.

Both are additive and behaviour-neutral for every golden-checked artifact.

---

## 4. Requirements compliance matrix (§1–§6)

`done` = realised + test-backed · `partial` = realised with a named gap ·
`debt` = not realised (behaviour-neutral) · `deferred` = intentionally out of
scope.

### §1 Principles
| Item | Status | Where / note |
|---|---|---|
| §1.1 no hidden hardcodes | done | config + named constants; §4.5 multipliers removed |
| §1.2 single layout source | done | `common/layout.py`; glob derived from load prefix |
| §1.3 `common/` config-free | done | AST test `test_no_config_import` |
| §1.4 outputs under `DATA_DIR` | done | incl. restored plots under `plots_dir` |
| §1.5 writer↔reader shared symbols | done | `common/names.py` |
| §1.6 config fields validated | **partial** | `INSPECT_FUSION_METHOD` not validated at stage start (its valid set is broader than the registry — includes `Best_*`); consumed only in best-effort diagnostics |
| §1.7 explicit seeding | done | S4 (sanctioned) |
| §1.8 additive interfaces, `**kwargs` only on interface | done | registries + entry_points |
| §1.9 meaningful stored types | **partial** | 4 new artifacts done; stage-1 experts/gate/shared + mlp_inputs/topk still pickled/raw (`SavedExpert` absent) |

### §2 Architecture
| Item | Status | Where / note |
|---|---|---|
| §2.1 run-from-root package | done | no `sys.path.append`, no `from config import` |
| §2.2 9→4 collapse | done | cut line inside old stage 5 verified |
| §2.3 grouping abstraction | done | `AffinityProvider`×`Partitioner`, 3 impls via registry |
| §2.4 meaningful `.pt` classes | **partial** | `SavedExpert` not implemented (raw experts/gate/shared still pickled) |
| §2.5 extension sufficiency | done | `StageContext`, registries, additive-only |
| §2.6 default-equivalence | done | golden on both trees |

### §3 Brainstorm refinements
| Item | Status | Where / note |
|---|---|---|
| §3.1 k-agnostic fusion, impls k=2 | done | `(*mlps)` frozen; `assert/raise` on k≠2; `build_MU` k=2 |
| §3.2a one `RouterAdapter` | done | capture + inference share `Qwen15RouterAdapter` |
| §3.2b shared expert (None / N>1) | done* | None tolerated; N>1 is a future subclass the abstraction admits (refuted as a gap) |
| §3.2c `Qwen15ExpertAdapter` | **partial** | adapter + both layouts + quantized guard done; activation-as-`SavedExpert`-metadata unrealised (no `SavedExpert`) |
| §3.2d QA parser | done | byte-identical prompt/answer resolve |
| §3.2e `is_moe_layer` | done | replaces `range`+`del` |
| §3.3 derived seeds + golden | done | S4 (sanctioned) |
| §3.4 clean seams + load-once | done | cut line, resume, load-once parity proven |
| §3.5 serialization contract | **partial** | done for the 4 artifacts; stage-1 raw weights still pickled |
| §3.6 provenance | **partial** | snapshot/assert/manifest/`_cache_key` seed done; gaps: `validate_layout_version` unwired, git commit never captured, `FusedExpert` provenance lacks model_id/dataset, `None`-filters not resolved to full set, dataset fingerprint not index-hashed |
| §3.7a `**kwargs` interface-only | done | concrete classes reject unknown kwargs |
| §3.7b `StageContext` to all ext points | **partial** | grouping + fusion carry `ctx`; `Metric.__call__` has **no** `ctx` slot |
| §3.7c registries + entry_points | **partial** | registry/discovery/collision/validation done; `LS_*` built by looping `BASE_FUSION_METHODS`, not a transformer-registry over all methods (entry-point base methods get no auto `LS_<name>`) |
| §3.7d two grouping axes | done | default reproduces `get_pairs` exactly |
| §3.7e `FusedGroup` runtime contract | done | `reconstruct(x,member)`; `refit_unmerge` public |

\* §3.2b refuted as a gap: §3.2 is an abstraction-tolerance requirement; the
single-pair `Qwen15SharedExpert` is *required* by §2.6, and `SharedExpertAdapter`
admits an N>1 DeepSeek subclass additively.

### §4 Concrete audit requirements
| Item | Status |
|---|---|
| §4.1 single layout source | done |
| §4.2 declarative dataset SCHEMA registry | done |
| §4.3 grouping names as `StrEnum` + registry validation | done |
| §4.4 `log_scale` as Metric property, dead `!= "Cosine"` removed | done |
| §4.5 knobs plumbed; hidden `10*examples` removed | done |
| §4.6 shared `EPS` + single `_default_device()` | done |
| §4.7 string contracts as named constants | done |

### §5 Bugs that must NOT be carried over
| Bug | Status |
|---|---|
| §5-1 `Best_*` fallback via `"Pruning"` | done (absent; `next(iter)`) |
| §5-2 wrong `makedirs` target | done (each `makedirs` matches its write) |
| §5-3 dead `validation_data` load | done (removed) |
| §5-4 `top_k=8` desync | done (`eval_match` deleted; real top-k = `config.TOP_K`) |
| §5-5 `_cache_key` missing seed | done (seed in key; 2 tests) |

### §6 Already-extracted knobs (no regression)
| Knob | Status |
|---|---|
| `DATASET_TEXT_FIELD`, `MIN_PAIR_VALIDATION_SAMPLES`, `USE_DEFAULT_EXPERTS`, `INSPECT_FUSION_METHOD` | done |
| `BENCHMARK_BATCH_SIZE/NUM_BATCHES`, `MAX_NEW_TOKENS` | done |
| `DistillationFusion` hyperparams in `__init__` | done |
| `common/metrics.py` shared `EPS` | done (centralized in `numeric.py`) |
| `IO_SAVE_RETRIES` | **done** (restored — §3 above) |
| `MAX_PLOT_METRIC_VALUE` | **done** (restored with plotting — §3 above) |

---

## 5. Conscious debt (behaviour-neutral; follow-ups beyond S10)

All items below are **identical to the S0/legacy baseline** in behaviour (golden
stays green) and are **not** on the sanctioned list because they introduce no
behaviour change. They are deferred contract/completeness debt:

1. **`SavedExpert` + stage-1 pickling** (§1.9 / §2.4 / §3.5 / §3.2c). Stage 1
   still `torch.save`s raw `nn.Module` experts/gate/shared_expert and bare
   mlp_inputs/topk lists; the `SavedExpert` artifact class is not built. The
   conversion machinery (`mlp_to_artifact` / `migrate_v0_to_v1`) exists, unused
   for these payloads. **Headline follow-up.**
2. **§3.6 provenance**: wire `validate_layout_version` into stage start; capture
   the git commit; add model_id/dataset to `FusedExpert` provenance; resolve
   `None` filters to the full set in `config_snapshot`; hash the selected dataset
   indices / pin a revision.
3. **§3.7b**: add the `ctx=None` slot to `Metric.__call__` (the only extension
   point missing it).
4. **§3.7c**: make `LS_*` a transformer-registry over *all* registered methods so
   an entry-point base method gets an auto `LS_<name>`.
5. **§1.6**: validate `INSPECT_FUSION_METHOD` at stage start against
   `registry methods ∪ Best_* names` (a plain `require_registered` would wrongly
   reject `Best_MSE`).
6. **Test hardening**: §5-3 (dead-load) and §5-4 (`top_k`) are proven by absence
   only; add regression guards. Five §6 config knobs lack a dedicated value test.

---

## 6. Gate status

- **Both trees + golden green**: `pytest tests/` → **518 passed** (95% coverage).
- **Sanctioned deviations**: exactly S4, S5, load-once, S8-FakeMoE, and the now-
  sanctioned S10 `sorted()` — each explained above. No unexplained behaviour
  change remains.
- **Restorations**: IO retry + DATA_DIR-scoped plotting, both additive and
  test-backed.
- **Requirements §1–§6**: every item is `done`, or logged here as conscious
  `partial`/`debt`/`deferred` with a follow-up.

Done: the code does not change behaviour out of nowhere (all differences
explained), and all §1–§6 requirements are fulfilled or explicitly accounted for
as conscious debt.
