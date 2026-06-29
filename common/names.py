"""Named string contracts shared by pipeline producers and consumers.

The pipeline used to couple stages through bare string literals duplicated on
both sides (the writer and the reader). When a literal drifts on one side the
break is silent. This module makes each such contract a single importable
symbol so producer and consumer reference the same source of truth.

Covered contracts:

* :class:`PairingMethod` -- grouping/pairing method names (dispatch in
  :func:`common.pairing.get_pairs`, the ``config.PAIRING_METHOD`` value and the
  grouping config guard);
* :class:`ReplaceMode` -- the layer-replacement mode tags written into
  the ``accuracy_list_<mode>_<method>.pt`` filenames at evaluation;
* :data:`TIME_METRIC` -- the wall-clock key produced by
  :func:`common.mlp_utils.test_model_on_data` and consumed by the fusion report;
* :data:`BEST_PREFIX` -- the ``Best_<metric>`` pseudo-method prefix (written
  during fusion, read at evaluation);
* :data:`ZERO_FUSION` -- the zeroed-MLP baseline name used at evaluation;
* :data:`TEST_RESULT_PREFIX` -- the evaluation results-directory prefix.

Library code only -- this module never imports an experiment ``config``.
"""

from enum import StrEnum


class PairingMethod(StrEnum):
    """Expert-pairing method names.

    Order matches the frozen registry snapshot (``pairing_methods``). As a
    ``StrEnum`` each member compares equal to its plain-string value, so a
    ``config.PAIRING_METHOD`` string and these members are interchangeable.
    """

    ACTIVATION_CORRELATION = "activation_correlation"
    RANDOM = "random"
    OUTPUT_SIMILARITY = "output_similarity"


class ReplaceMode(StrEnum):
    """Layer-replacement mode tags at evaluation (used in saved filenames)."""

    SEQ = "seq"
    SINGLE = "single"
    SMART_SEQ = "smart_seq"


# Wall-clock metric key in the fusion report (producer: mlp_utils,
# consumer: the report's plotting/aggregation).
TIME_METRIC = "Time"

# Prefix of the per-metric "best fusion" pseudo-method (written during fusion,
# read at evaluation).
BEST_PREFIX = "Best_"

# Name of the zeroed-MLP baseline replacement used at evaluation.
ZERO_FUSION = "zero"

# Prefix of the evaluation results directory (e.g. ``test_result_<train>_<test>``).
TEST_RESULT_PREFIX = "test_result_"
