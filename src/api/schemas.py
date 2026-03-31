from pydantic import BaseModel, Field


class EvalRequest(BaseModel):
    models: list[str] = Field(..., min_length=1, max_length=3, description="Model identifiers")
    dataset: str = Field(..., description="Path to JSON dataset file")
    evaluators: list[str] | None = Field(
        default=None, description="Evaluator names. Defaults to ['exact_match']"
    )
    run_name: str | None = Field(default=None, description="Optional name for this run")


class ScoreStats(BaseModel):
    mean: float
    lower: float
    upper: float
    std: float
    confidence_level: float


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


class EvalRunResponse(BaseModel):
    run_id: int
    models: list[str]
    evaluators: list[str]
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
    scores: dict | None
    latency_ms: float | None
    tokens_used: int | None
    cost: float | None
    created_at: str
