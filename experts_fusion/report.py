"""Summary-table report over evaluation results (stage-4 artifacts).

Assembles the paper's headline table from the accuracy artifacts stage 4 writes
under one or more ``DATA_DIR`` trees:

    model x dataset x grouping x fusion -> accuracy + relative-error

For each ``DATA_DIR`` the model, grouping strategy and evaluation dataset are read
from ``config_snapshot.json`` (written by stage 1); the per-fusion accuracy comes
from ``test_result_*/accuracy/accuracy_list_<mode>_<fusion>.pt`` (written by stage
4). Every accuracy list is ``[base, after_1_layer, ..., after_all_layers]``, so the
report reports the fully-compressed accuracy (last element) and its relative error
against the uncompressed base (first element).

One command assembles the whole table across every listed ``DATA_DIR`` and emits
CSV (machine-readable, all columns) or LaTeX (curated paper columns):

    python -m experts_fusion.report ./data/ef_mmlu_activation_correlation ./data/ef_... \
        --format latex --out table.tex

Imports no experiment ``config`` module: everything comes from the on-disk
artifacts, so a single report can span ``DATA_DIR`` trees produced by different
configs, models and datasets. Malformed inputs (missing snapshot, absent or empty
accuracy list, unparseable filename) warn and are skipped so one bad tree never
sinks the whole table.
"""
import argparse
import csv
import glob
import io
import math
import os
import re
import warnings
from dataclasses import asdict, dataclass, fields

import torch

from common import provenance
from common.names import TEST_RESULT_PREFIX, ReplaceMode

# Subfolder + filename kind stage 4 writes accuracy lists under. Mirrors
# ``4_evaluate.py:_save_list`` (``<kind>/<kind>_list_<mode>_<name>.pt``); kept in
# sync with that writer, the single producer of these files.
ACCURACY_KIND = "accuracy"
_ACCURACY_LIST_PREFIX = f"{ACCURACY_KIND}_list_"
_PT_SUFFIX = ".pt"

# Known eval modes, longest first so "smart_seq" is matched before "seq" when the
# "<mode>_<fusion>" stem is split (fusion names never start with a mode token).
_MODES_LONGEST_FIRST = sorted((str(mode) for mode in ReplaceMode), key=len, reverse=True)

# Placeholder for a field the snapshot did not carry (defensive; stage 1 always
# writes model/grouping/dataset, so this only surfaces for hand-made artifacts).
UNKNOWN = "<unknown>"


@dataclass(frozen=True)
class ResultRow:
    """One (model, dataset, grouping, fusion, mode) evaluation summary.

    ``base_accuracy``/``final_accuracy`` are the first/last entries of the saved
    accuracy list (uncompressed vs. fully-compressed). ``accuracy_drop`` is their
    difference (percentage points when accuracy is a fraction) and feeds the E5
    "<2 p.p." gate; ``relative_error`` is that drop relative to the base.
    """

    model: str
    dataset: str
    grouping: str
    fusion: str
    mode: str
    base_accuracy: float
    final_accuracy: float
    accuracy_drop: float
    relative_error: float
    num_points: int
    data_dir: str


def _dataset_label(dataset) -> str:
    """Human label for a dataset spec: the HF path basename (mirrors stage 4).

    ``{"path": "cais/mmlu", ...}`` -> ``"mmlu"``. Returns ``UNKNOWN`` when the spec
    is not a mapping carrying a ``path``.
    """
    if isinstance(dataset, dict) and dataset.get("path"):
        return str(dataset["path"]).split("/")[-1]
    return UNKNOWN


def _split_mode_and_fusion(stem: str):
    """Split ``"<mode>_<fusion>"`` into ``(mode, fusion)``, or ``None`` if no known
    mode prefixes the stem."""
    for mode in _MODES_LONGEST_FIRST:
        marker = f"{mode}_"
        if stem.startswith(marker):
            return mode, stem[len(marker):]
    return None


def _relative_error(base: float, final: float) -> float:
    """Relative accuracy error ``(base - final) / base``; ``nan`` when base is 0."""
    if base == 0:
        return float("nan")
    return (base - final) / base


def _load_accuracy_list(path: str) -> list:
    """Load a saved accuracy list as a list of Python floats."""
    values = torch.load(path, map_location="cpu")
    return [float(value) for value in values]


def read_data_dir(data_dir: str) -> list:
    """Return every :class:`ResultRow` for one ``DATA_DIR`` (``[]`` if unusable).

    Model / grouping / dataset come from ``config_snapshot.json``; accuracy lists
    come from every ``test_result_*/accuracy/accuracy_list_<mode>_<fusion>.pt``.
    A missing snapshot, missing/empty accuracy file, or unparseable filename warns
    and is skipped rather than raising.
    """
    snapshot = provenance.read_config_snapshot(data_dir)
    if snapshot is None:
        warnings.warn(
            f"{data_dir}: no {provenance.CONFIG_SNAPSHOT_FILENAME}; skipping",
            RuntimeWarning,
            stacklevel=2,
        )
        return []
    config = snapshot.get("config", {})
    model = str(config.get("MODEL_NAME", UNKNOWN))
    grouping = str(config.get("PAIRING_METHOD", UNKNOWN))
    dataset = _dataset_label(config.get("TEST_DATASET_NAME"))

    result_dirs = sorted(glob.glob(os.path.join(data_dir, f"{TEST_RESULT_PREFIX}*")))
    if not result_dirs:
        warnings.warn(
            f"{data_dir}: no {TEST_RESULT_PREFIX}* results; skipping",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    rows = []
    for result_dir in result_dirs:
        acc_dir = os.path.join(result_dir, ACCURACY_KIND)
        if not os.path.isdir(acc_dir):
            continue
        pattern = os.path.join(acc_dir, f"{_ACCURACY_LIST_PREFIX}*{_PT_SUFFIX}")
        for path in sorted(glob.glob(pattern)):
            stem = os.path.basename(path)[len(_ACCURACY_LIST_PREFIX):-len(_PT_SUFFIX)]
            split = _split_mode_and_fusion(stem)
            if split is None:
                warnings.warn(
                    f"{path}: cannot parse <mode>_<fusion> from {stem!r}; skipping",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            mode, fusion = split
            try:
                values = _load_accuracy_list(path)
            except Exception as exc:  # a corrupt file must not sink the report
                warnings.warn(
                    f"{path}: failed to load ({exc!r}); skipping",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if not values:
                warnings.warn(
                    f"{path}: empty accuracy list; skipping", RuntimeWarning, stacklevel=2
                )
                continue
            base, final = values[0], values[-1]
            rows.append(
                ResultRow(
                    model=model,
                    dataset=dataset,
                    grouping=grouping,
                    fusion=fusion,
                    mode=mode,
                    base_accuracy=base,
                    final_accuracy=final,
                    accuracy_drop=base - final,
                    relative_error=_relative_error(base, final),
                    num_points=len(values),
                    data_dir=data_dir,
                )
            )
    return rows


def _sort_key(row: ResultRow):
    return (row.model, row.dataset, row.grouping, row.mode, row.fusion)


def collect_rows(data_dirs, *, modes=None) -> list:
    """Read every listed ``DATA_DIR`` into a sorted list of :class:`ResultRow`.

    ``modes`` (an iterable of ``ReplaceMode`` values) restricts the rows to those
    eval modes; ``None`` keeps every mode. Rows are sorted by
    ``(model, dataset, grouping, mode, fusion)`` for stable output.
    """
    rows = []
    for data_dir in data_dirs:
        rows.extend(read_data_dir(data_dir))
    if modes is not None:
        wanted = {str(mode) for mode in modes}
        rows = [row for row in rows if row.mode in wanted]
    return sorted(rows, key=_sort_key)


# --------------------------------------------------------------------------- #
# Rendering.
# --------------------------------------------------------------------------- #
CSV_COLUMNS = [field.name for field in fields(ResultRow)]

# Curated (attribute, header) columns for the LaTeX paper table.
LATEX_COLUMNS = [
    ("model", "Model"),
    ("dataset", "Dataset"),
    ("grouping", "Grouping"),
    ("fusion", "Fusion"),
    ("final_accuracy", "Accuracy"),
    ("relative_error", "Rel. error"),
]

# Float-valued attributes formatted with fixed precision in the LaTeX table.
_NUMERIC_ATTRS = frozenset(
    {"base_accuracy", "final_accuracy", "accuracy_drop", "relative_error"}
)

_LATEX_SPECIAL = re.compile(r"([&%$#_{}~^\\])")
_LATEX_REPLACEMENTS = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def _latex_escape(text) -> str:
    """Escape LaTeX special characters (underscores in ``LS_ZipIt`` etc.)."""
    return _LATEX_SPECIAL.sub(lambda match: _LATEX_REPLACEMENTS[match.group(1)], str(text))


def _fmt(value, digits: int) -> str:
    """Fixed-precision float, or ``"nan"`` for a non-finite value."""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def to_csv(rows) -> str:
    """Render rows as CSV text with every :class:`ResultRow` field as a column."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(asdict(row))
    return buffer.getvalue()


def to_latex(rows, *, columns=LATEX_COLUMNS, digits: int = 4) -> str:
    """Render rows as a booktabs-style LaTeX ``tabular``.

    ``columns`` is a list of ``(attribute, header)`` pairs. Numeric attributes are
    right-aligned and fixed to ``digits`` decimals; text cells are escaped.
    """
    col_spec = "".join("r" if attr in _NUMERIC_ATTRS else "l" for attr, _ in columns)
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        " & ".join(_latex_escape(header) for _, header in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        cells = []
        for attr, _ in columns:
            value = getattr(row, attr)
            if attr in _NUMERIC_ATTRS:
                cells.append(_fmt(value, digits))
            elif attr == "num_points":
                cells.append(str(value))
            else:
                cells.append(_latex_escape(value))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experts_fusion.report",
        description="Assemble the model x dataset x grouping x fusion accuracy table "
        "from stage-4 results under one or more DATA_DIR trees.",
    )
    parser.add_argument(
        "data_dirs",
        nargs="+",
        help="one or more DATA_DIR trees to read stage-4 results from",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=[str(mode) for mode in ReplaceMode],
        help="only include this eval mode (repeatable; default: all modes)",
    )
    parser.add_argument(
        "--format", choices=["csv", "latex"], default="csv", help="output format"
    )
    parser.add_argument("--out", help="write the table to this file (default: stdout)")
    parser.add_argument(
        "--digits", type=int, default=4, help="float precision for the LaTeX table"
    )
    return parser


def main(argv=None) -> list:
    """CLI entry point: collect rows and emit the table. Returns the rows."""
    args = _build_parser().parse_args(argv)
    rows = collect_rows(args.data_dirs, modes=args.mode)
    text = to_latex(rows, digits=args.digits) if args.format == "latex" else to_csv(rows)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text, end="")
    return rows


if __name__ == "__main__":
    main()
