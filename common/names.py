"""Named string constants shared across pipeline stages.

Centralising these strings prevents silent mismatches when a literal drifts
on either the writing or reading side.

Exports: PairingMethod (grouping method names), TIME_METRIC.
No experiment config is imported here.
"""

from enum import StrEnum


class PairingMethod(StrEnum):
    """Expert-grouping strategy names.

    StrEnum: each member compares equal to its string value, so
    config.PAIRING_METHOD strings and members are interchangeable.
    """

    COACTIVATION = "coactivation"
    RANDOM = "random"
    OUTPUT_SIMILARITY = "output_similarity"


# Wall-clock key produced by mlp_utils and consumed by the fusion report.
TIME_METRIC = "Time"
