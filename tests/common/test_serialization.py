"""Tests for the versioned safetensors artifact (de)serializer.

Covers the S2 gate: round-trip equality (incl. bf16/int dtypes), the version
guard (``version > CURRENT`` is an explicit error), the legacy-pickle migration
path, the forward-identity of a reconstructed ``AdvancedMLP``, the "no pickled
``nn.Module``" guarantee, and the additive upgrader registry.
"""

import json

import pytest
import torch
from torch import nn

from common import serialization as S
from common.mlp import AdvancedMLP
from common.fusion import build_fusion_methods

CPU = torch.device("cpu")


@pytest.fixture
def path(tmp_path):
    return tmp_path / "artifact.safetensors"


@pytest.fixture
def restore_upgraders():
    """Snapshot/restore the global upgrader registry around a test."""
    saved = dict(S._UPGRADERS)
    yield
    S._UPGRADERS.clear()
    S._UPGRADERS.update(saved)


# --------------------------------------------------------------------------- #
# Core round-trip.
# --------------------------------------------------------------------------- #
def test_roundtrip_preserves_tensors_and_header(path):
    tensors = {
        "w": torch.randn(3, 4, dtype=torch.bfloat16),
        "idx": torch.arange(5),
        "f64": torch.randn(2, 2, dtype=torch.float64),
    }
    S.save(path, tensors, header={"foo": "bar"}, payload_type="generic")
    art = S.load(path)

    assert set(art.tensors) == set(tensors)
    for key in tensors:
        assert art.tensors[key].dtype == tensors[key].dtype
        assert torch.equal(art.tensors[key], tensors[key])
    assert art.header["foo"] == "bar"
    assert art.header[S.PAYLOAD_TYPE_FIELD] == "generic"
    assert art.header[S.VERSION_FIELD] == S.FORMAT_VERSION


def test_save_stamps_current_format_version(path):
    S.save(path, {"w": torch.zeros(2)})
    assert S.load_header(path)[S.VERSION_FIELD] == S.FORMAT_VERSION


def test_provenance_slot_is_stored(path):
    prov = S.make_provenance(seed=42, commit="abc123", model_id="m")
    S.save(path, {"w": torch.zeros(2)}, provenance=prov)
    assert S.load_header(path)[S.PROVENANCE_FIELD] == prov


def test_make_provenance_drops_unset_fields():
    assert S.make_provenance(seed=1) == {"seed": 1}


def test_metadata_only_artifact_roundtrips(path):
    # A pure-metadata artifact (no tensors) -- e.g. a pairs list.
    S.save(path, {}, header={"pairs": [[0, 1], [2, 3]]}, payload_type=S.PAYLOAD_PAIRS)
    art = S.load(path)
    assert art.tensors == {}
    assert art.header["pairs"] == [[0, 1], [2, 3]]


# --------------------------------------------------------------------------- #
# Lazy header / peek.
# --------------------------------------------------------------------------- #
def test_peek_reports_shapes_without_loading_weights(path):
    S.save(path, {"w": torch.randn(3, 4, dtype=torch.bfloat16)}, header={"k": 1})
    info = S.peek(path)
    assert info.header["k"] == 1
    assert info.tensors["w"]["shape"] == [3, 4]
    assert info.tensors["w"]["dtype"] == "BF16"


def test_load_header_returns_header_only(path):
    S.save(path, {"w": torch.zeros(2)}, header={"a": 1})
    header = S.load_header(path)
    assert header["a"] == 1
    assert S.VERSION_FIELD in header


# --------------------------------------------------------------------------- #
# Version guard.
# --------------------------------------------------------------------------- #
def _write_with_version(path, version):
    from safetensors.torch import save_file

    save_file(
        {"w": torch.zeros(2)},
        str(path),
        metadata={S._HEADER_KEY: json.dumps({S.VERSION_FIELD: version})},
    )


def test_load_rejects_newer_format(path):
    _write_with_version(path, S.FORMAT_VERSION + 1)
    with pytest.raises(S.FormatVersionError):
        S.load(path)


def test_peek_and_load_header_reject_newer_format(path):
    _write_with_version(path, S.FORMAT_VERSION + 5)
    with pytest.raises(S.FormatVersionError):
        S.load_header(path)
    with pytest.raises(S.FormatVersionError):
        S.peek(path)


def test_missing_version_field_warns_and_assumes_current(path):
    from safetensors.torch import save_file

    save_file({"w": torch.zeros(2)}, str(path), metadata={S._HEADER_KEY: json.dumps({})})
    with pytest.warns(RuntimeWarning):
        header = S.load_header(path)
    assert header.get(S.VERSION_FIELD) is None  # header had none; not invented


# --------------------------------------------------------------------------- #
# No pickled nn.Module.
# --------------------------------------------------------------------------- #
def test_save_rejects_non_tensor_values(path):
    with pytest.raises(S.SerializationError):
        S.save(path, {"m": nn.Linear(2, 2)})
    with pytest.raises(S.SerializationError):
        S.save(path, {"x": [1, 2, 3]})


def test_save_rejects_non_dict_tensors(path):
    with pytest.raises(S.SerializationError):
        S.save(path, [torch.zeros(2)])


def test_saved_file_is_safetensors_not_pickle(path):
    # The artifact must be openable as safetensors (proves it is not a torch pickle).
    from safetensors import safe_open

    S.save(path, {"w": torch.zeros(2)})
    with safe_open(str(path), framework="pt") as handle:
        assert "w" in handle.keys()


# --------------------------------------------------------------------------- #
# AdvancedMLP <-> artifact forward identity.
# --------------------------------------------------------------------------- #
def _pair(seed_a=1, seed_b=2):
    torch.manual_seed(seed_a)
    a = AdvancedMLP(4, 6, 4, device=CPU)
    torch.manual_seed(seed_b)
    b = AdvancedMLP(4, 6, 4, device=CPU)
    return a, b


@pytest.mark.parametrize("method", ["Pruning", "Average", "ZipIt", "LS_ZipIt"])
def test_advanced_mlp_roundtrips_forward_identical(path, method):
    a, b = _pair()
    fused = build_fusion_methods()[method](a, b, torch.randn(40, 4))

    art = S.mlp_to_artifact(fused)
    S.save(path, art.tensors, art.header, payload_type=S.PAYLOAD_ADVANCED_MLP)
    recon = S.artifact_to_mlp(S.load(path), device=CPU)

    x = torch.randn(7, 4)
    with torch.no_grad():
        assert torch.allclose(fused.forward_unzip(x), recon.forward_unzip(x), atol=1e-6)
    assert (recon.down is None) == (fused.down is None)
    assert (recon.U is not None) == (fused.U is not None)


def test_artifact_to_mlp_uses_default_device(path):
    a, _ = _pair()
    art = S.mlp_to_artifact(a)
    S.save(path, art.tensors, art.header)
    recon = S.artifact_to_mlp(S.load(path))  # no device -> _default_device()
    x = torch.randn(7, 4)
    with torch.no_grad():
        assert torch.allclose(a.forward_unzip(x), recon.forward_unzip(x), atol=1e-6)


def test_artifact_to_mlp_round_trips_a_registered_activation():
    mlp = AdvancedMLP(4, 6, 4, device=CPU, activation="GELU")
    art = S.mlp_to_artifact(mlp)
    assert art.header["activation"] == "GELU"
    rebuilt = S.artifact_to_mlp(art, device=CPU)
    assert type(rebuilt.act).__name__ == "GELU"


def test_artifact_to_mlp_rejects_unregistered_activation(path):
    a, _ = _pair()
    art = S.mlp_to_artifact(a)
    bad = S.Artifact(tensors=art.tensors, header={**art.header, "activation": "Mystery"})
    with pytest.raises(ValueError):
        S.artifact_to_mlp(bad, device=CPU)


def test_advanced_mlp_artifact_header_records_shape_and_flags():
    a, b = _pair()
    fused = build_fusion_methods()["LS_ZipIt"](a, b, torch.randn(40, 4))
    header = S.mlp_to_artifact(fused).header
    assert header == {
        "in_features": 4,
        "hid_features": 6,
        "out_features": 4,
        "activation": "SiLU",
        "dtype": "torch.float32",
        "has_down": False,
        "has_U": True,
    }


def test_artifact_round_trip_preserves_bf16_dtype(path):
    mlp = AdvancedMLP(4, 6, 4, device=CPU).to(dtype=torch.bfloat16)
    art = S.mlp_to_artifact(mlp)
    S.save(path, art.tensors, header=art.header)
    assert S.peek(path).header["dtype"] == "torch.bfloat16"
    rebuilt = S.artifact_to_mlp(S.load(path), device=CPU)
    assert rebuilt.gate.weight.dtype == torch.bfloat16
    assert rebuilt.up.weight.dtype == torch.bfloat16
    assert rebuilt.down.weight.dtype == torch.bfloat16


# --------------------------------------------------------------------------- #
# Legacy (v0 pickle) -> v1 migration.
# --------------------------------------------------------------------------- #
def test_migrate_legacy_pickled_mlp_is_forward_identical(tmp_path):
    a, _ = _pair()
    legacy = tmp_path / "legacy_expert.pt"
    torch.save(a, legacy)  # the legacy pipeline pickled the whole module

    art = S.load_legacy(legacy)
    assert art.header[S.PAYLOAD_TYPE_FIELD] == S.PAYLOAD_ADVANCED_MLP
    recon = S.artifact_to_mlp(art, device=CPU)

    x = torch.randn(7, 4)
    with torch.no_grad():
        assert torch.allclose(a.forward_unzip(x), recon.forward_unzip(x), atol=1e-6)


def test_migrate_legacy_pairs_list_roundtrips(tmp_path):
    legacy = tmp_path / "legacy_pairs.pt"
    pairs = [(0, 1), (2, 3), (4, 5)]
    torch.save(pairs, legacy)

    art = S.load_legacy(legacy)
    assert art.header[S.PAYLOAD_TYPE_FIELD] == S.PAYLOAD_PAIRS
    assert S.artifact_to_pairs(art) == pairs


def test_migrate_empty_pairs_list(tmp_path):
    legacy = tmp_path / "empty.pt"
    torch.save([], legacy)
    art = S.load_legacy(legacy)
    assert art.header[S.PAYLOAD_TYPE_FIELD] == S.PAYLOAD_PAIRS
    assert S.artifact_to_pairs(art) == []


def test_migrate_unknown_type_raises():
    with pytest.raises(S.SerializationError):
        S.migrate_v0_to_v1({"not": "a known artifact"})


# --------------------------------------------------------------------------- #
# Upgrader registry (additive within-format header upgrades).
# --------------------------------------------------------------------------- #
def test_upgrader_chain_applied_on_load(path, restore_upgraders, monkeypatch):
    # Save a v1 artifact, then pretend CURRENT is 2 with a registered 1->2 upgrader.
    S.save(path, {"w": torch.zeros(2)}, header={"old": True})
    monkeypatch.setattr(S, "FORMAT_VERSION", 2)

    @S.register_upgrader(1)
    def _v1_to_v2(art):
        new_header = {**art.header, S.VERSION_FIELD: 2, "upgraded": True}
        return S.Artifact(tensors=art.tensors, header=new_header)

    art = S.load(path)
    assert art.header["upgraded"] is True
    assert art.header[S.VERSION_FIELD] == 2


def test_missing_upgrader_raises(path, restore_upgraders, monkeypatch):
    S.save(path, {"w": torch.zeros(2)})
    monkeypatch.setattr(S, "FORMAT_VERSION", 3)  # no 1->2 upgrader registered
    with pytest.raises(S.SerializationError):
        S.load(path)


def test_register_upgrader_collision_raises(restore_upgraders):
    @S.register_upgrader(7)
    def _first(art):
        return art

    with pytest.raises(S.SerializationError):
        @S.register_upgrader(7)
        def _second(art):
            return art


# --------------------------------------------------------------------------- #
# Transient-IO retry on save (§6 IO_SAVE_RETRIES, kept on the central save path).
# --------------------------------------------------------------------------- #
def test_save_retries_transient_io_then_succeeds(path, monkeypatch):
    real_save_file = S.save_file
    calls = {"n": 0}

    def flaky(prepared, target, metadata):
        calls["n"] += 1
        if calls["n"] < S.IO_SAVE_RETRIES:  # fail every attempt but the last
            raise OSError("transient write failure")
        return real_save_file(prepared, target, metadata=metadata)

    monkeypatch.setattr(S, "save_file", flaky)
    S.save(path, {"w": torch.ones(3)})

    assert calls["n"] == S.IO_SAVE_RETRIES
    assert torch.equal(S.load(path).tensors["w"], torch.ones(3))


def test_save_reraises_after_retries_exhausted(path, monkeypatch):
    calls = {"n": 0}

    def always_fail(prepared, target, metadata):
        calls["n"] += 1
        raise OSError("disk gone")

    monkeypatch.setattr(S, "save_file", always_fail)
    with pytest.raises(OSError, match="disk gone"):
        S.save(path, {"w": torch.ones(3)})

    assert calls["n"] == S.IO_SAVE_RETRIES
