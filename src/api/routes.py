from fastapi import APIRouter, HTTPException

from src.api.schemas import EvalRequest, EvalRunResponse, ResultItem, RunInfo
from src.runner import evaluate
from src.statistics.bootstrap import bootstrap_confidence_interval
from src.statistics.comparison import paired_bootstrap_test
from src.tracking.database import Database

router = APIRouter()


@router.post("/run-eval", response_model=EvalRunResponse)
def run_eval(request: EvalRequest):
    """Run an evaluation comparing models on a dataset."""
    try:
        result = evaluate(
            models=request.models,
            dataset=request.dataset,
            evaluators=request.evaluators,
            run_name=request.run_name,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/results/{run_id}")
def get_results(run_id: int):
    """Get results for a specific evaluation run."""
    db = Database()
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    results = db.get_results(run_id)
    return {
        "run": run,
        "results": results,
    }


@router.get("/compare/{run_id}")
def compare_models(run_id: int):
    """Compare models from an evaluation run with statistical analysis."""
    db = Database()
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    models = run["models"]
    evaluators = run["evaluators"]

    if len(models) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 models to compare")

    # Gather scores per model per evaluator
    model_stats = {}
    model_scores: dict[str, dict[str, list[float]]] = {}

    for model in models:
        results = db.get_results(run_id, model=model)
        scores_by_evaluator: dict[str, list[float]] = {ev: [] for ev in evaluators}
        total_cost = 0.0
        total_latency = 0.0
        total_tokens = 0

        for r in results:
            if r["scores"]:
                for ev_name, score_val in r["scores"].items():
                    if ev_name in scores_by_evaluator:
                        scores_by_evaluator[ev_name].append(score_val)
            total_cost += r.get("cost", 0) or 0
            total_latency += r.get("latency_ms", 0) or 0
            total_tokens += r.get("tokens_used", 0) or 0

        stats = {}
        for ev_name, scores in scores_by_evaluator.items():
            if scores:
                stats[ev_name] = bootstrap_confidence_interval(scores)

        n = len(results)
        stats["tracking"] = {
            "total_requests": n,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "avg_latency_ms": total_latency / n if n else 0,
            "avg_cost_per_request": total_cost / n if n else 0,
        }

        model_stats[model] = stats
        model_scores[model] = scores_by_evaluator

    # Paired comparison on the first evaluator
    primary_ev = evaluators[0]
    scores_a = model_scores[models[0]].get(primary_ev, [])
    scores_b = model_scores[models[1]].get(primary_ev, [])

    comparison = None
    if scores_a and scores_b and len(scores_a) == len(scores_b):
        comparison = paired_bootstrap_test(scores_a, scores_b)
        comparison["model_a"] = models[0]
        comparison["model_b"] = models[1]
        comparison["evaluator"] = primary_ev

    return {
        "run_id": run_id,
        "models": models,
        "evaluators": evaluators,
        "model_stats": model_stats,
        "comparison": comparison,
    }


@router.get("/runs")
def list_runs():
    """List all evaluation runs."""
    db = Database()
    return db.list_runs()
