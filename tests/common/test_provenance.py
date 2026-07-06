"""Tests for common.provenance (config snapshot, manifest, fingerprints, idempotency)."""
import torch

from common import provenance, serialization
from common.fused_group import FusedExpert
from common.mlp import AdvancedMLP


def test_git_commit_returns_hash_or_none():
    commit = provenance.git_commit()
    assert commit is None or (len(commit) == 40 and all(c in "0123456789abcdef" for c in commit))


def test_git_commit_outside_a_repo_is_none(tmp_path):
    # A directory that is not a git repo (or an unreachable cwd) yields None,
    # never an exception -- provenance capture is best-effort.
    assert provenance.git_commit(repo_dir=str(tmp_path)) is None


def _cfg():
    return {
        "MODEL_NAME": "q",
        "SEED": 42,
        "MODEL_DTYPE": torch.bfloat16,
        "device": torch.device("cpu"),
        "LAYERS": [0, 1, 2],
    }


def test_config_hash_is_process_stable_and_drops_device():
    cfg = _cfg()
    assert provenance.config_hash(cfg) == provenance.config_hash(dict(cfg))
    assert "device" not in provenance.canonical_config(cfg)
    # dtype is stabilized to its string name (not an unstable repr/address).
    assert provenance.canonical_config(cfg)["MODEL_DTYPE"] == "torch.bfloat16"


def test_config_hash_changes_on_a_real_key():
    cfg = _cfg()
    drifted = {**cfg, "SEED": 7}
    assert provenance.config_hash(cfg) != provenance.config_hash(drifted)


def test_config_hash_ignores_only_device(tmp_path):
    cfg = _cfg()
    moved = {**cfg, "device": torch.device("cuda")}
    assert provenance.config_hash(cfg) == provenance.config_hash(moved)


def _cfg_with_eval_flags():
    # A config carrying both fusion-relevant keys and eval-only flags (stage-4 only).
    return {
        "MODEL_NAME": "q",
        "SEED": 42,
        "LAYERS": [0, 1],
        "USE_DEFAULT_EXPERTS": False,
        "REPLACE_LAYERS_SEQUENTIAL": False,
        "REPLACE_BEST_LAYERS_SEQUENTIAL": True,
        "REPLACE_LAYERS_SMART_SEQUENTIAL": False,
        "INSPECT_FUSION_METHOD": "ZipIt",
    }


def test_config_hash_exclude_drops_the_given_keys():
    cfg = _cfg_with_eval_flags()
    # Excluding a key yields the same hash as a config that never had it.
    without = {k: v for k, v in cfg.items() if k != "SEED"}
    assert provenance.config_hash(cfg, exclude=frozenset({"SEED"})) == provenance.config_hash(without)
    assert "SEED" not in provenance.canonical_config(cfg, exclude=frozenset({"SEED"}))
    # exclude is additive to EPHEMERAL_CONFIG_KEYS, never replaces it.
    moved = {**cfg, "device": torch.device("cuda")}
    assert provenance.config_hash(moved, exclude=frozenset({"SEED"})) == provenance.config_hash(
        cfg, exclude=frozenset({"SEED"})
    )


def test_eval_only_config_keys_are_exactly_the_stage4_flags():
    assert provenance.EVAL_ONLY_CONFIG_KEYS == frozenset({
        "REPLACE_LAYERS_SEQUENTIAL",
        "REPLACE_BEST_LAYERS_SEQUENTIAL",
        "REPLACE_LAYERS_SMART_SEQUENTIAL",
        "INSPECT_FUSION_METHOD",
        "USE_DEFAULT_EXPERTS",
    })


def test_fusion_config_hash_ignores_eval_only_flags():
    cfg = _cfg_with_eval_flags()
    base = provenance.fusion_config_hash(cfg)
    # Flipping any single eval-only flag leaves the fusion hash unchanged...
    for key, flipped in {
        "USE_DEFAULT_EXPERTS": True,
        "REPLACE_LAYERS_SEQUENTIAL": True,
        "REPLACE_BEST_LAYERS_SEQUENTIAL": False,
        "REPLACE_LAYERS_SMART_SEQUENTIAL": True,
        "INSPECT_FUSION_METHOD": "Average",
    }.items():
        drifted = {**cfg, key: flipped}
        assert provenance.fusion_config_hash(drifted) == base, key
        # ...but the full config hash DOES change (the flag is still real config).
        assert provenance.config_hash(drifted) != provenance.config_hash(cfg), key


def test_fusion_config_hash_changes_on_a_fusion_relevant_key():
    cfg = _cfg_with_eval_flags()
    for key, drifted_value in {"SEED": 7, "LAYERS": [0, 1, 2], "MODEL_NAME": "other"}.items():
        drifted = {**cfg, key: drifted_value}
        assert provenance.fusion_config_hash(drifted) != provenance.fusion_config_hash(cfg), key


def test_canonical_config_keeps_eval_only_keys_by_default():
    # Snapshot invariant: eval-only keys are real config and must stay in the
    # snapshot (only fusion's idempotency hash drops them, via ``exclude``).
    canonical = provenance.canonical_config(_cfg_with_eval_flags())
    for key in provenance.EVAL_ONLY_CONFIG_KEYS:
        assert key in canonical, key


def test_flipping_an_eval_flag_does_not_invalidate_a_fused_artifact(tmp_path):
    # End-to-end: a fused expert stored under config A stays fresh when only an
    # eval-only flag flips (config B), but goes stale when a fusion key changes.
    cfg_a = _cfg_with_eval_flags()
    cfg_b_eval = {**cfg_a, "USE_DEFAULT_EXPERTS": True}   # eval-only drift
    cfg_c_real = {**cfg_a, "SEED": 7}                     # fusion-relevant drift

    def upstream(cfg):
        return provenance.fused_upstream_hash(
            config_hash=provenance.fusion_config_hash(cfg),
            layer=0,
            method="Average",
            group=0,
            extra="part",
        )

    path = str(tmp_path / "f.pt")
    FusedExpert(AdvancedMLP(4, 6, 4, device=torch.device("cpu"))).save(
        path, provenance=serialization.make_provenance(upstream_hash=upstream(cfg_a))
    )
    assert provenance.is_fresh(path, upstream(cfg_a))
    assert provenance.is_fresh(path, upstream(cfg_b_eval))   # eval flag flip -> still fresh
    assert not provenance.is_fresh(path, upstream(cfg_c_real))  # real change -> recompute


def test_snapshot_roundtrip_and_assert(tmp_path):
    data_dir = str(tmp_path)
    cfg = _cfg()
    provenance.write_config_snapshot(data_dir, cfg)
    assert provenance.read_config_snapshot(data_dir)["config_hash"] == provenance.config_hash(cfg)
    provenance.assert_config_matches_snapshot(data_dir, cfg)  # no raise


def test_assert_raises_on_drift_with_key_diff(tmp_path):
    data_dir = str(tmp_path)
    provenance.write_config_snapshot(data_dir, _cfg())
    try:
        provenance.assert_config_matches_snapshot(data_dir, {**_cfg(), "SEED": 99})
        raise AssertionError("expected ConfigMismatchError")
    except provenance.ConfigMismatchError as exc:
        assert "SEED" in str(exc)


def test_assert_is_noop_without_snapshot(tmp_path):
    # An absent snapshot (no stage 1 yet) is allowed -- nothing to compare.
    provenance.assert_config_matches_snapshot(str(tmp_path), _cfg())


def test_run_manifest_roundtrip(tmp_path):
    data_dir = str(tmp_path)
    manifest = provenance.RunManifest(seed=42, model_id="q", config_hash="abc")
    manifest.save(data_dir)
    loaded = provenance.RunManifest.load(data_dir)
    assert loaded.seed == 42 and loaded.model_id == "q" and loaded.config_hash == "abc"


def test_dataset_fingerprint_varies_with_n_and_seed():
    ds = list(range(8))
    a = provenance.dataset_fingerprint(ds, n=4, seed=1)
    assert a == provenance.dataset_fingerprint(ds, n=4, seed=1)
    assert a != provenance.dataset_fingerprint(ds, n=4, seed=2)
    assert a != provenance.dataset_fingerprint(ds, n=8, seed=1)


def test_fused_upstream_hash_depends_on_coordinates():
    base = dict(config_hash="c", layer=0, method="Average", group=0)
    h = provenance.fused_upstream_hash(**base)
    assert h == provenance.fused_upstream_hash(**base)
    assert h != provenance.fused_upstream_hash(**{**base, "group": 1})
    assert h != provenance.fused_upstream_hash(**{**base, "config_hash": "d"})
    assert h != provenance.fused_upstream_hash(**{**base, "extra": "p"})


def test_is_fresh_matches_stored_upstream_hash(tmp_path):
    path = str(tmp_path / "f.pt")
    upstream = provenance.fused_upstream_hash(config_hash="c", layer=0, method="Average", group=0)
    FusedExpert(AdvancedMLP(4, 6, 4, device=torch.device("cpu"))).save(
        path, provenance=serialization.make_provenance(upstream_hash=upstream)
    )
    assert provenance.is_fresh(path, upstream)
    assert not provenance.is_fresh(path, "other")
    assert not provenance.is_fresh(str(tmp_path / "missing.pt"), upstream)
