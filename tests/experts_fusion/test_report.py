"""Unit tests for the E4 summary-table report (``experts_fusion/report.py``).

Every test builds fake stage-1/stage-4 artifacts on disk -- a ``config_snapshot``
(via the real provenance writer) plus torch-saved accuracy lists laid out exactly
as ``4_evaluate.py`` writes them -- then drives the report end to end. No model,
GPU or pipeline run is involved; the point is that one command turns those
artifacts into the paper table.
"""
import csv
import io

import pytest
import torch

from common import provenance
from common.names import TEST_RESULT_PREFIX, ReplaceMode
from experts_fusion import report


def _dataset(path, split):
    return {"path": path, "split": split}


def seed_data_dir(
    root,
    *,
    model="Qwen/Qwen1.5-MoE-A2.7B-Chat",
    grouping="activation_correlation",
    train=("cais/mmlu", "auxiliary_train"),
    test=("cais/mmlu", "test"),
    accuracy_lists=None,
    write_snapshot=True,
):
    """Create a fake DATA_DIR with a config snapshot and stage-4 accuracy files.

    ``accuracy_lists`` maps ``(mode, fusion)`` -> list of accuracy floats and is
    written under ``test_result_<train>_<test>/accuracy/`` exactly as stage 4 does.
    Returns the DATA_DIR path (str).
    """
    data_dir = root
    data_dir.mkdir(parents=True, exist_ok=True)
    if write_snapshot:
        provenance.write_config_snapshot(
            str(data_dir),
            {
                "MODEL_NAME": model,
                "PAIRING_METHOD": grouping,
                "TRAIN_DATASET_NAME": _dataset(*train),
                "TEST_DATASET_NAME": _dataset(*test),
            },
        )
    if accuracy_lists:
        train_tag = f"{train[0].split('/')[-1]}_{train[1]}"
        test_tag = f"{test[0].split('/')[-1]}_{test[1]}"
        acc_dir = data_dir / f"{TEST_RESULT_PREFIX}{train_tag}_{test_tag}" / report.ACCURACY_KIND
        acc_dir.mkdir(parents=True, exist_ok=True)
        for (mode, fusion), values in accuracy_lists.items():
            torch.save(values, str(acc_dir / f"{report._ACCURACY_LIST_PREFIX}{mode}_{fusion}.pt"))
    return str(data_dir)


# --------------------------------------------------------------------------- #
# Row extraction + numeric fields.
# --------------------------------------------------------------------------- #
def test_reads_dimensions_and_accuracy_fields(tmp_path):
    data_dir = seed_data_dir(
        tmp_path / "ef",
        accuracy_lists={(ReplaceMode.SEQ, "ZipIt"): [0.6, 0.55, 0.45]},
    )
    (row,) = report.collect_rows([data_dir])
    assert row.model == "Qwen/Qwen1.5-MoE-A2.7B-Chat"
    assert row.dataset == "mmlu"
    assert row.grouping == "activation_correlation"
    assert row.fusion == "ZipIt"
    assert row.mode == "seq"
    assert row.base_accuracy == pytest.approx(0.6)
    assert row.final_accuracy == pytest.approx(0.45)
    assert row.accuracy_drop == pytest.approx(0.15)
    assert row.relative_error == pytest.approx(0.25)
    assert row.num_points == 3
    assert row.data_dir == data_dir


def test_relative_error_nan_when_base_zero(tmp_path):
    data_dir = seed_data_dir(
        tmp_path / "ef", accuracy_lists={(ReplaceMode.SEQ, "ZipIt"): [0.0, 0.0]}
    )
    (row,) = report.collect_rows([data_dir])
    assert row.base_accuracy == 0.0
    assert row.relative_error != row.relative_error  # nan


def test_single_element_list_has_zero_drop(tmp_path):
    data_dir = seed_data_dir(
        tmp_path / "ef", accuracy_lists={(ReplaceMode.SINGLE, "Average"): [0.5]}
    )
    (row,) = report.collect_rows([data_dir])
    assert row.base_accuracy == pytest.approx(0.5)
    assert row.final_accuracy == pytest.approx(0.5)
    assert row.accuracy_drop == pytest.approx(0.0)
    assert row.relative_error == pytest.approx(0.0)
    assert row.num_points == 1


# --------------------------------------------------------------------------- #
# Filename parsing (underscored modes and fusion names).
# --------------------------------------------------------------------------- #
def test_parses_underscored_modes_and_fusion_names(tmp_path):
    data_dir = seed_data_dir(
        tmp_path / "ef",
        accuracy_lists={
            (ReplaceMode.SMART_SEQ, "LS_ZipIt"): [0.6, 0.5],
            (ReplaceMode.SEQ, "Best_MSE"): [0.6, 0.5],
            (ReplaceMode.SINGLE, "zero"): [0.6, 0.1],
        },
    )
    by_fusion = {(r.mode, r.fusion) for r in report.collect_rows([data_dir])}
    assert by_fusion == {("smart_seq", "LS_ZipIt"), ("seq", "Best_MSE"), ("single", "zero")}


def test_parses_baseline_fusion_names(tmp_path):
    """REAP/HC-SMoE baseline fusion names (``Pruning`` / ``FrequencyWeightedAverage``,
    step-2 design note) parse across all three replace modes.

    Neither name starts with a mode token (``seq_``/``single_``/``smart_seq_``), so
    the report splits ``<mode>_<fusion>`` correctly -- the structural guarantee the
    baseline configs (step 7) rely on when stage 4 writes their accuracy lists.
    """
    data_dir = seed_data_dir(
        tmp_path / "ef",
        accuracy_lists={
            (ReplaceMode.SEQ, "Pruning"): [0.6, 0.5],
            (ReplaceMode.SINGLE, "Pruning"): [0.6, 0.55],
            (ReplaceMode.SMART_SEQ, "Pruning"): [0.6, 0.45],
            (ReplaceMode.SEQ, "FrequencyWeightedAverage"): [0.6, 0.5],
            (ReplaceMode.SINGLE, "FrequencyWeightedAverage"): [0.6, 0.55],
            (ReplaceMode.SMART_SEQ, "FrequencyWeightedAverage"): [0.6, 0.45],
        },
    )
    by_fusion = {(r.mode, r.fusion) for r in report.collect_rows([data_dir])}
    assert by_fusion == {
        ("seq", "Pruning"),
        ("single", "Pruning"),
        ("smart_seq", "Pruning"),
        ("seq", "FrequencyWeightedAverage"),
        ("single", "FrequencyWeightedAverage"),
        ("smart_seq", "FrequencyWeightedAverage"),
    }


def test_unparseable_filename_warns_and_skips(tmp_path):
    data_dir = seed_data_dir(
        tmp_path / "ef", accuracy_lists={(ReplaceMode.SEQ, "ZipIt"): [0.6, 0.5]}
    )
    # Add a file whose stem has no known mode prefix.
    acc_dir = tmp_path / "ef"
    stray = next((acc_dir).glob(f"{TEST_RESULT_PREFIX}*/{report.ACCURACY_KIND}"))
    torch.save([0.1], str(stray / f"{report._ACCURACY_LIST_PREFIX}bogusmode_ZipIt.pt"))
    with pytest.warns(RuntimeWarning, match="cannot parse"):
        rows = report.collect_rows([data_dir])
    assert {r.fusion for r in rows} == {"ZipIt"}  # the valid file still parsed


# --------------------------------------------------------------------------- #
# Dataset label + multi-DATA_DIR aggregation.
# --------------------------------------------------------------------------- #
def test_dataset_label_is_path_basename(tmp_path):
    data_dir = seed_data_dir(
        tmp_path / "arc",
        test=("allenai/ai2_arc", "test"),
        train=("allenai/ai2_arc", "train"),
        accuracy_lists={(ReplaceMode.SEQ, "ZipIt"): [0.4, 0.3]},
    )
    (row,) = report.collect_rows([data_dir])
    assert row.dataset == "ai2_arc"


def test_aggregates_and_sorts_across_data_dirs(tmp_path):
    a = seed_data_dir(
        tmp_path / "a",
        grouping="random",
        accuracy_lists={(ReplaceMode.SEQ, "ZipIt"): [0.6, 0.5]},
    )
    b = seed_data_dir(
        tmp_path / "b",
        grouping="activation_correlation",
        accuracy_lists={(ReplaceMode.SEQ, "Average"): [0.6, 0.4]},
    )
    rows = report.collect_rows([a, b])
    # Sorted by (model, dataset, grouping, mode, fusion): activation_correlation < random.
    assert [(r.grouping, r.fusion) for r in rows] == [
        ("activation_correlation", "Average"),
        ("random", "ZipIt"),
    ]


def test_mode_filter(tmp_path):
    data_dir = seed_data_dir(
        tmp_path / "ef",
        accuracy_lists={
            (ReplaceMode.SEQ, "ZipIt"): [0.6, 0.5],
            (ReplaceMode.SINGLE, "ZipIt"): [0.6, 0.55],
        },
    )
    rows = report.collect_rows([data_dir], modes=[ReplaceMode.SEQ])
    assert [r.mode for r in rows] == ["seq"]


# --------------------------------------------------------------------------- #
# Robustness: malformed / incomplete DATA_DIRs warn and are skipped.
# --------------------------------------------------------------------------- #
def test_missing_snapshot_warns_and_skips(tmp_path):
    data_dir = seed_data_dir(
        tmp_path / "ef",
        write_snapshot=False,
        accuracy_lists={(ReplaceMode.SEQ, "ZipIt"): [0.6, 0.5]},
    )
    with pytest.warns(RuntimeWarning, match="no config_snapshot"):
        assert report.collect_rows([data_dir]) == []


def test_no_results_dir_warns_and_skips(tmp_path):
    data_dir = seed_data_dir(tmp_path / "ef", accuracy_lists=None)
    with pytest.warns(RuntimeWarning, match="no test_result"):
        assert report.collect_rows([data_dir]) == []


def test_empty_accuracy_list_warns_and_skips(tmp_path):
    data_dir = seed_data_dir(
        tmp_path / "ef", accuracy_lists={(ReplaceMode.SEQ, "ZipIt"): []}
    )
    with pytest.warns(RuntimeWarning, match="empty accuracy list"):
        assert report.collect_rows([data_dir]) == []


def test_corrupt_accuracy_file_warns_and_skips(tmp_path):
    data_dir = seed_data_dir(
        tmp_path / "ef", accuracy_lists={(ReplaceMode.SEQ, "ZipIt"): [0.6, 0.5]}
    )
    acc_dir = next((tmp_path / "ef").glob(f"{TEST_RESULT_PREFIX}*/{report.ACCURACY_KIND}"))
    (acc_dir / f"{report._ACCURACY_LIST_PREFIX}seq_Average.pt").write_bytes(b"not a torch file")
    with pytest.warns(RuntimeWarning, match="failed to load"):
        rows = report.collect_rows([data_dir])
    assert {r.fusion for r in rows} == {"ZipIt"}  # the valid file still parsed


def test_result_dir_without_accuracy_subdir_yields_no_rows(tmp_path):
    data_dir = seed_data_dir(
        tmp_path / "ef", accuracy_lists={(ReplaceMode.SEQ, "ZipIt"): [0.6, 0.5]}
    )
    # A second results dir carrying no accuracy/ subfolder must be skipped silently.
    (tmp_path / "ef" / f"{TEST_RESULT_PREFIX}other").mkdir()
    rows = report.collect_rows([data_dir])
    assert {r.fusion for r in rows} == {"ZipIt"}


def test_unknown_dataset_and_model_when_config_keys_absent(tmp_path):
    data_dir = tmp_path / "ef"
    data_dir.mkdir()
    provenance.write_config_snapshot(str(data_dir), {})  # snapshot present but empty
    acc_dir = data_dir / f"{TEST_RESULT_PREFIX}x_y" / report.ACCURACY_KIND
    acc_dir.mkdir(parents=True)
    torch.save([0.6, 0.5], str(acc_dir / f"{report._ACCURACY_LIST_PREFIX}seq_ZipIt.pt"))
    (row,) = report.collect_rows([str(data_dir)])
    assert row.model == report.UNKNOWN
    assert row.grouping == report.UNKNOWN
    assert row.dataset == report.UNKNOWN


# --------------------------------------------------------------------------- #
# CSV rendering.
# --------------------------------------------------------------------------- #
def test_to_csv_roundtrips_all_columns(tmp_path):
    data_dir = seed_data_dir(
        tmp_path / "ef", accuracy_lists={(ReplaceMode.SEQ, "ZipIt"): [0.6, 0.45]}
    )
    rows = report.collect_rows([data_dir])
    text = report.to_csv(rows)
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert list(parsed[0].keys()) == report.CSV_COLUMNS
    assert len(parsed) == 1
    assert parsed[0]["fusion"] == "ZipIt"
    assert float(parsed[0]["final_accuracy"]) == pytest.approx(0.45)
    assert float(parsed[0]["relative_error"]) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# LaTeX rendering.
# --------------------------------------------------------------------------- #
def test_to_latex_structure_and_escaping(tmp_path):
    data_dir = seed_data_dir(
        tmp_path / "ef",
        test=("tau/commonsense_qa", "validation"),
        train=("tau/commonsense_qa", "train"),
        accuracy_lists={(ReplaceMode.SEQ, "LS_ZipIt"): [0.6, 0.45]},
    )
    rows = report.collect_rows([data_dir])
    latex = report.to_latex(rows)
    assert r"\begin{tabular}{llllrr}" in latex
    assert r"\toprule" in latex and r"\midrule" in latex and r"\bottomrule" in latex
    # Underscores in dataset ("commonsense_qa") and fusion ("LS_ZipIt") are escaped.
    assert r"commonsense\_qa" in latex
    assert r"LS\_ZipIt" in latex
    assert "commonsense_qa" not in latex  # no raw underscore leaked through
    # One data row between the rules with the curated numeric columns.
    body = latex.split(r"\midrule")[1].split(r"\bottomrule")[0]
    data_rows = [ln for ln in body.splitlines() if ln.strip().endswith(r"\\")]
    assert len(data_rows) == 1
    assert "0.4500" in data_rows[0] and "0.2500" in data_rows[0]


def test_to_latex_custom_columns_include_num_points(tmp_path):
    data_dir = seed_data_dir(
        tmp_path / "ef", accuracy_lists={(ReplaceMode.SEQ, "ZipIt"): [0.6, 0.55, 0.45]}
    )
    rows = report.collect_rows([data_dir])
    latex = report.to_latex(rows, columns=[("fusion", "Fusion"), ("num_points", "N")])
    # num_points is rendered as a bare integer (not float-formatted, not escaped).
    body = latex.split(r"\midrule")[1].split(r"\bottomrule")[0]
    assert "ZipIt & 3 \\\\" in body


def test_to_latex_formats_nan_relative_error(tmp_path):
    data_dir = seed_data_dir(
        tmp_path / "ef", accuracy_lists={(ReplaceMode.SEQ, "ZipIt"): [0.0, 0.0]}
    )
    rows = report.collect_rows([data_dir])
    latex = report.to_latex(rows)
    assert "nan" in latex


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def test_cli_writes_latex_file(tmp_path):
    data_dir = seed_data_dir(
        tmp_path / "ef", accuracy_lists={(ReplaceMode.SEQ, "ZipIt"): [0.6, 0.45]}
    )
    out = tmp_path / "table.tex"
    rows = report.main([data_dir, "--format", "latex", "--out", str(out)])
    assert len(rows) == 1
    written = out.read_text(encoding="utf-8")
    assert written.startswith(r"\begin{tabular}")
    assert "ZipIt" in written


def test_cli_mode_filter_and_csv_stdout(tmp_path, capsys):
    data_dir = seed_data_dir(
        tmp_path / "ef",
        accuracy_lists={
            (ReplaceMode.SEQ, "ZipIt"): [0.6, 0.5],
            (ReplaceMode.SINGLE, "ZipIt"): [0.6, 0.55],
        },
    )
    rows = report.main([data_dir, "--mode", "seq"])
    assert [r.mode for r in rows] == ["seq"]
    captured = capsys.readouterr().out
    assert captured.startswith(",".join(report.CSV_COLUMNS))
    # Exactly one data line beyond the header.
    assert len([ln for ln in captured.splitlines() if ln]) == 2
