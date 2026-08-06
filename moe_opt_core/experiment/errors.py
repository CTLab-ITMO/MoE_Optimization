from __future__ import annotations

from enum import Enum
from typing import Mapping


class ExperimentFailureCode(str, Enum):
    INVALID_CONFIG = "invalid_config"
    TRACE_ARTIFACT_MISSING = "trace_artifact_missing"
    REAL_TRACES_REQUIRED = "real_traces_required"
    DATASET_UNAVAILABLE = "dataset_unavailable"
    MODEL_UNAVAILABLE = "model_unavailable"
    LAYER_UNAVAILABLE = "layer_unavailable"
    RUNTIME_ARCHITECTURE_MISMATCH = "runtime_architecture_mismatch"
    PILOT_STATE_MISMATCH = "pilot_state_mismatch"
    INVALID_CANDIDATE = "invalid_candidate"
    EVALUATION_FAILED = "evaluation_failed"


class ExpectedExperimentFailure(RuntimeError):
    def __init__(
        self,
        code: ExperimentFailureCode,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = dict(context or {})

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "message": str(self),
            "context": self.context,
        }
