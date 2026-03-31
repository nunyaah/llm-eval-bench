from pydantic import BaseModel, Field


class EvalRequest(BaseModel):
    models: list[str] = Field(..., min_length=1, max_length=3, description="Model identifiers")
    dataset: str = Field(..., description="Path to JSON dataset file")
    evaluators: list[str] | None = Field(
        default=None, description="Evaluator names. Defaults to ['exact_match']"
    )
    run_name: str | None = Field(default=None, description="Optional name for this run")
    primary_metric: str | None = Field(default=None, description="Primary metric for comparison")


class ScoreStats(BaseModel):
    mean: float
    lower: float
    upper: float
    std: float
    confidence_level: float
    n_samples: int | None = None
    ci_method: str | None = None
    n_bootstrap: int | None = None
    warning: str | None = None


class TrackingSummary(BaseModel):
    total_requests: int
    total_cost: float
    total_tokens: int
    avg_latency_ms: float
    avg_cost_per_request: float
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None


class ComparisonResult(BaseModel):
    model_a: str
    model_b: str
    evaluator: str
    mean_a: float
    mean_b: float
    mean_diff: float
    p_value: float
    is_significant: bool
    significance_level: float
    interpretation: str
    n_samples: int | None = None
    comparison_method: str | None = None
    ci_method: str | None = None
    n_bootstrap: int | None = None
    warning: str | None = None
    winner: str | None = None


class ModelSummary(BaseModel):
    model_name: str
    exact_match: float | None = None
    semantic_similarity: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    avg_latency_ms: float | None = None
    total_cost: float | None = None


class EvalRunResponse(BaseModel):
    run_id: int
    models: list[str]
    evaluators: list[str]
    primary_metric: str | None = None
    dataset: str
    num_samples: int
    model_stats: dict
    comparison: ComparisonResult | None = None


class RunInfo(BaseModel):
    id: int
    name: str | None
    dataset_path: str
    models: list[str]
    evaluators: list[str]
    primary_metric: str | None = None
    sample_count: int | None = None
    status: str
    created_at: str
    completed_at: str | None


class ResultItem(BaseModel):
    id: int
    run_id: int
    model: str
    input: str
    expected_output: str
    actual_output: str | None
    normalized_actual: str | None = None
    scores: dict | None
    latency_ms: float | None
    tokens_used: int | None
    cost: float | None
    created_at: str
