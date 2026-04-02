import time

import litellm

from src.datasets.loader import load_dataset
from src.evaluators.base import BaseEvaluator
from src.evaluators.exact_match import ExactMatchEvaluator
from src.evaluators.semantic_similarity import SemanticSimilarityEvaluator
from src.statistics.bootstrap import bootstrap_confidence_interval
from src.statistics.comparison import paired_bootstrap_test
from src.tracking.database import Database
from src.tracking.tracker import CostLatencyTracker

EVALUATOR_REGISTRY: dict[str, type[BaseEvaluator]] = {
    "exact_match": ExactMatchEvaluator,
    "semantic_similarity": SemanticSimilarityEvaluator,
}


def _get_normalized_output(ev_instances: list[BaseEvaluator], actual: str) -> str | None:
    """Return the normalized form of *actual* using the first ExactMatchEvaluator found."""
    for ev in ev_instances:
        if isinstance(ev, ExactMatchEvaluator):
            return ev.normalized_output(actual)
    return None


def _call_model(model: str, prompt: str, system_prompt: str | None = None) -> dict:
    """Call an LLM via LiteLLM and return response with metrics."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    start = time.perf_counter()
    response = litellm.completion(
        model=model,
        messages=messages,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    output = response.choices[0].message.content or ""
    usage = response.usage
    tokens_used = (usage.prompt_tokens or 0) + (usage.completion_tokens or 0) if usage else 0

    # litellm tracks cost via response_cost when available
    cost = 0.0
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        pass

    return {
        "output": output,
        "latency_ms": latency_ms,
        "tokens_used": tokens_used,
        "cost": cost,
    }


def evaluate(
    models: list[str],
    dataset: str,
    evaluators: list[str] | None = None,
    run_name: str | None = None,
    db_path: str | None = None,
    system_prompt: str | None = None,
) -> dict:
    """Run an evaluation comparing models on a dataset.

    Args:
        models: List of model identifiers (e.g. ["gpt-4", "claude-3-haiku-20240307"])
        dataset: Path to JSON dataset file
        evaluators: List of evaluator names. Defaults to ["exact_match"]
        run_name: Optional name for this evaluation run
        db_path: Optional database path override
        system_prompt: Optional system prompt prepended to every model call

    Returns:
        dict with run_id, per-model results, statistics, and tracking summaries
    """
    evaluator_names = evaluators or ["exact_match"]
    evaluator_instances = []
    for name in evaluator_names:
        if name not in EVALUATOR_REGISTRY:
            raise ValueError(f"Unknown evaluator: {name}. Available: {list(EVALUATOR_REGISTRY.keys())}")
        evaluator_instances.append(EVALUATOR_REGISTRY[name]())

    data = load_dataset(dataset)
    db = Database(db_path)
    tracker = CostLatencyTracker()
    primary_metric = evaluator_names[0]

    run_id = db.create_run(
        dataset_path=dataset,
        models=models,
        evaluators=evaluator_names,
        name=run_name,
        primary_metric=primary_metric,
        sample_count=len(data),
    )

    # Collect scores per model per evaluator for statistical comparison
    model_scores: dict[str, dict[str, list[float]]] = {
        model: {ev.name: [] for ev in evaluator_instances} for model in models
    }

    try:
        for item in data:
            input_text = item["input"]
            expected = item["expected_output"]

            for model in models:
                try:
                    result = _call_model(model, input_text, system_prompt=system_prompt)
                    actual = result["output"]
                    latency_ms = result["latency_ms"]
                    tokens_used = result["tokens_used"]
                    cost = result["cost"]
                except Exception as e:
                    actual = None
                    latency_ms = 0.0
                    tokens_used = 0
                    cost = 0.0

                # Score with each evaluator
                scores = {}
                normalized_actual = _get_normalized_output(evaluator_instances, actual) if actual is not None else None
                if actual is not None:
                    for ev in evaluator_instances:
                        score_val = ev.score(expected, actual)
                        scores[ev.name] = score_val
                        model_scores[model][ev.name].append(score_val)
                else:
                    for ev in evaluator_instances:
                        scores[ev.name] = 0.0
                        model_scores[model][ev.name].append(0.0)

                tracker.record(model, latency_ms, tokens_used, cost)
                db.insert_result(
                    run_id=run_id,
                    model=model,
                    input_text=input_text,
                    expected_output=expected,
                    actual_output=actual,
                    normalized_actual=normalized_actual,
                    scores=scores,
                    latency_ms=latency_ms,
                    tokens_used=tokens_used,
                    cost=cost,
                )

        db.complete_run(run_id)
    except Exception:
        db.fail_run(run_id)
        raise

    # Build per-model statistics
    model_stats = {}
    for model in models:
        stats = {}
        for ev in evaluator_instances:
            scores_list = model_scores[model][ev.name]
            if scores_list:
                stats[ev.name] = bootstrap_confidence_interval(scores_list)
        tracking = tracker.summary(model)
        stats["tracking"] = tracking
        model_stats[model] = stats

        # Persist model summary to DB
        em_stats = stats.get("exact_match", {})
        sim_stats = stats.get("semantic_similarity", {})
        db.upsert_model_summary(
            run_id=run_id,
            model_name=model,
            exact_match=em_stats.get("mean"),
            semantic_similarity=sim_stats.get("mean"),
            ci_lower=em_stats.get("lower"),
            ci_upper=em_stats.get("upper"),
            avg_latency_ms=tracking.get("avg_latency_ms"),
            total_cost=tracking.get("total_cost"),
        )

    # Paired comparison (first evaluator, first two models)
    comparison = None
    if len(models) >= 2:
        scores_a = model_scores[models[0]][primary_metric]
        scores_b = model_scores[models[1]][primary_metric]
        if scores_a and scores_b:
            comparison = paired_bootstrap_test(scores_a, scores_b)
            comparison["model_a"] = models[0]
            comparison["model_b"] = models[1]
            comparison["evaluator"] = primary_metric

    return {
        "run_id": run_id,
        "models": models,
        "evaluators": evaluator_names,
        "primary_metric": primary_metric,
        "dataset": dataset,
        "num_samples": len(data),
        "model_stats": model_stats,
        "comparison": comparison,
    }
