from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import time
from typing import Dict, Iterator


@dataclass
class MetricsRecord:
    stage: str
    quality: float = 0.0
    tokens_per_sec: float = 0.0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    memory_mb: float = 0.0
    router_ms: float = 0.0
    dispatch_ms: float = 0.0
    expert_ms: float = 0.0
    routing_error_budget: float = 0.0
    load_balance_cv: float = 0.0

    def to_dict(self) -> Dict[str, float | str]:
        return asdict(self)


@dataclass
class StageTimer:
    elapsed_s: Dict[str, float]

    def __init__(self) -> None:
        self.elapsed_s = {}

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.elapsed_s[name] = self.elapsed_s.get(name, 0.0) + (time.perf_counter() - start)


@contextmanager
def timed_block() -> Iterator[float]:
    start = time.perf_counter()
    holder = {"elapsed": 0.0}
    try:
        yield holder
    finally:
        holder["elapsed"] = time.perf_counter() - start
