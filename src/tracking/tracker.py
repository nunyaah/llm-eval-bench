import time
from dataclasses import dataclass, field


@dataclass
class RequestMetrics:
    """Metrics captured for a single LLM request."""
    model: str
    latency_ms: float
    tokens_used: int
    cost: float


@dataclass
class CostLatencyTracker:
    """Tracks cost and latency across evaluation requests."""

    records: list[RequestMetrics] = field(default_factory=list)

    def record(self, model: str, latency_ms: float, tokens_used: int, cost: float) -> None:
        self.records.append(
            RequestMetrics(
                model=model,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                cost=cost,
            )
        )

    def summary(self, model: str | None = None) -> dict:
        """Get summary statistics for tracked requests."""
        filtered = self.records
        if model:
            filtered = [r for r in self.records if r.model == model]

        if not filtered:
            return {
                "total_requests": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
                "avg_latency_ms": 0.0,
                "avg_cost_per_request": 0.0,
            }

        latencies = [r.latency_ms for r in filtered]
        latencies.sort()
        n = len(latencies)

        return {
            "total_requests": n,
            "total_cost": sum(r.cost for r in filtered),
            "total_tokens": sum(r.tokens_used for r in filtered),
            "avg_latency_ms": sum(latencies) / n,
            "p50_latency_ms": latencies[n // 2],
            "p95_latency_ms": latencies[int(n * 0.95)] if n >= 20 else latencies[-1],
            "p99_latency_ms": latencies[int(n * 0.99)] if n >= 100 else latencies[-1],
            "avg_cost_per_request": sum(r.cost for r in filtered) / n,
        }
