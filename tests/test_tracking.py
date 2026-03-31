import json

import pytest

from src.tracking.database import Database
from src.tracking.tracker import CostLatencyTracker


class TestDatabase:
    @pytest.fixture
    def db(self, tmp_path):
        return Database(str(tmp_path / "test.db"))

    def test_create_and_get_run(self, db):
        run_id = db.create_run(
            dataset_path="data.json",
            models=["model-a", "model-b"],
            evaluators=["exact_match"],
            name="test_run",
        )
        run = db.get_run(run_id)
        assert run is not None
        assert run["name"] == "test_run"
        assert run["models"] == ["model-a", "model-b"]
        assert run["status"] == "running"

    def test_complete_run(self, db):
        run_id = db.create_run("data.json", ["a"], ["exact_match"])
        db.complete_run(run_id)
        run = db.get_run(run_id)
        assert run["status"] == "completed"
        assert run["completed_at"] is not None

    def test_fail_run(self, db):
        run_id = db.create_run("data.json", ["a"], ["exact_match"])
        db.fail_run(run_id)
        run = db.get_run(run_id)
        assert run["status"] == "failed"

    def test_insert_and_get_results(self, db):
        run_id = db.create_run("data.json", ["a"], ["exact_match"])
        db.insert_result(
            run_id=run_id,
            model="a",
            input_text="What is 2+2?",
            expected_output="4",
            actual_output="4",
            scores={"exact_match": 1.0},
            latency_ms=150.0,
            tokens_used=20,
            cost=0.001,
        )
        results = db.get_results(run_id)
        assert len(results) == 1
        assert results[0]["model"] == "a"
        assert results[0]["scores"]["exact_match"] == 1.0

    def test_get_results_by_model(self, db):
        run_id = db.create_run("data.json", ["a", "b"], ["exact_match"])
        db.insert_result(run_id, "a", "q", "ans", "ans", {"exact_match": 1.0}, 100, 10, 0.001)
        db.insert_result(run_id, "b", "q", "ans", "wrong", {"exact_match": 0.0}, 200, 15, 0.002)
        results_a = db.get_results(run_id, model="a")
        results_b = db.get_results(run_id, model="b")
        assert len(results_a) == 1
        assert len(results_b) == 1
        assert results_a[0]["scores"]["exact_match"] == 1.0
        assert results_b[0]["scores"]["exact_match"] == 0.0

    def test_list_runs(self, db):
        db.create_run("data1.json", ["a"], ["exact_match"], name="run1")
        db.create_run("data2.json", ["b"], ["exact_match"], name="run2")
        runs = db.list_runs()
        assert len(runs) == 2

    def test_get_nonexistent_run(self, db):
        assert db.get_run(999) is None


class TestCostLatencyTracker:
    def test_record_and_summary(self):
        tracker = CostLatencyTracker()
        tracker.record("model-a", 100.0, 50, 0.01)
        tracker.record("model-a", 200.0, 60, 0.02)
        summary = tracker.summary("model-a")
        assert summary["total_requests"] == 2
        assert summary["total_cost"] == 0.03
        assert summary["total_tokens"] == 110
        assert summary["avg_latency_ms"] == 150.0

    def test_empty_summary(self):
        tracker = CostLatencyTracker()
        summary = tracker.summary("nonexistent")
        assert summary["total_requests"] == 0

    def test_filter_by_model(self):
        tracker = CostLatencyTracker()
        tracker.record("a", 100, 10, 0.01)
        tracker.record("b", 200, 20, 0.02)
        summary_a = tracker.summary("a")
        summary_b = tracker.summary("b")
        assert summary_a["total_requests"] == 1
        assert summary_b["total_requests"] == 1
