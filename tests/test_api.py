import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_root_redirects_to_dashboard(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 307
    assert "/dashboard" in resp.headers["location"]


def test_dashboard_returns_html(client):
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "llm-eval-bench" in resp.text


def test_list_runs_empty(client):
    resp = client.get("/api/runs")
    assert resp.status_code == 200


def test_get_nonexistent_run(client):
    resp = client.get("/api/results/99999")
    assert resp.status_code == 404


def test_compare_nonexistent_run(client):
    resp = client.get("/api/compare/99999")
    assert resp.status_code == 404


def test_run_eval_missing_dataset(client):
    resp = client.post(
        "/api/run-eval",
        json={"models": ["gpt-4"], "dataset": "nonexistent.json"},
    )
    assert resp.status_code == 404


def test_run_eval_invalid_evaluator(client):
    # Create a temp dataset first
    import tempfile, os

    data = [{"input": "test", "expected_output": "test"}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        tmp_path = f.name
    try:
        resp = client.post(
            "/api/run-eval",
            json={
                "models": ["gpt-4"],
                "dataset": tmp_path,
                "evaluators": ["nonexistent_evaluator"],
            },
        )
        assert resp.status_code == 400
    finally:
        os.unlink(tmp_path)
