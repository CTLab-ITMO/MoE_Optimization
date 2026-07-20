"""Named string constants shared across pipeline stages.

Centralising these strings prevents silent mismatches when a literal drifts
on either the writing or reading side.

Exports: PairingMethod (grouping method names), ReplaceMode (eval filename
tags), TIME_METRIC, BEST_PREFIX, ZERO_FUSION, TEST_RESULT_PREFIX.
No experiment config is imported here.
"""

from enum import StrEnum


class PairingMethod(StrEnum):
    """Expert-grouping strategy names.

    StrEnum: each member compares equal to its string value, so
    config.PAIRING_METHOD strings and members are interchangeable.
    """

    ACTIVATION_CORRELATION = "activation_correlation"
    RANDOM = "random"
    OUTPUT_SIMILARITY = "output_similarity"
    REAP_SALIENCY = "reap_saliency"
    HCSMOE_CLUSTERING = "hcsmoe_clustering"


class ReplaceMode(StrEnum):
    """Layer-replacement mode tags at evaluation (used in saved filenames)."""

    SEQ = "seq"
    SINGLE = "single"
    SMART_SEQ = "smart_seq"


# Wall-clock key produced by mlp_utils and consumed by the fusion report.
TIME_METRIC = "Time"

# Prefix for the per-metric best-fusion pseudo-method (e.g. "Best_MSE").
BEST_PREFIX = "Best_"

# Name of the zeroed-MLP baseline replacement used at evaluation.
ZERO_FUSION = "zero"

# Prefix of the evaluation results directory (e.g. ``test_result_<train>_<test>``).
TEST_RESULT_PREFIX = "test_result_"
