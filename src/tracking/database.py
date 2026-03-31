import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.config import DATABASE_PATH


class Database:
    """SQLite database manager for storing evaluation results."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or DATABASE_PATH
        self._ensure_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        conn = self._get_connection()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS eval_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    dataset_path TEXT NOT NULL,
                    models TEXT NOT NULL,
                    evaluators TEXT NOT NULL,
                    primary_metric TEXT,
                    sample_count INTEGER,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS model_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    exact_match REAL,
                    semantic_similarity REAL,
                    ci_lower REAL,
                    ci_upper REAL,
                    avg_latency_ms REAL,
                    total_cost REAL,
                    FOREIGN KEY (run_id) REFERENCES eval_runs(id)
                );

                CREATE TABLE IF NOT EXISTS eval_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    input TEXT NOT NULL,
                    expected_output TEXT NOT NULL,
                    actual_output TEXT,
                    normalized_actual TEXT,
                    scores TEXT,
                    latency_ms REAL,
                    tokens_used INTEGER,
                    cost REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES eval_runs(id)
                );
            """)
            conn.commit()
            # Migrate existing databases that predate the new columns
            self._migrate(conn)
        finally:
            conn.close()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add columns introduced after the initial schema without data loss."""
        migrations = [
            ("eval_runs", "primary_metric", "TEXT"),
            ("eval_runs", "sample_count", "INTEGER"),
            ("eval_results", "normalized_actual", "TEXT"),
        ]
        for table, column, col_type in migrations:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                conn.commit()
            except sqlite3.OperationalError:
                # Column already exists — ignore
                pass

    def create_run(
        self,
        dataset_path: str,
        models: list[str],
        evaluators: list[str],
        name: str | None = None,
        primary_metric: str | None = None,
        sample_count: int | None = None,
    ) -> int:
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """INSERT INTO eval_runs
                   (name, dataset_path, models, evaluators, primary_metric, sample_count,
                    status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'running', ?)""",
                (
                    name or f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                    dataset_path,
                    json.dumps(models),
                    json.dumps(evaluators),
                    primary_metric,
                    sample_count,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def complete_run(self, run_id: int) -> None:
        conn = self._get_connection()
        try:
            conn.execute(
                "UPDATE eval_runs SET status = 'completed', completed_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), run_id),
            )
            conn.commit()
        finally:
            conn.close()

    def fail_run(self, run_id: int) -> None:
        conn = self._get_connection()
        try:
            conn.execute(
                "UPDATE eval_runs SET status = 'failed', completed_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), run_id),
            )
            conn.commit()
        finally:
            conn.close()

    def insert_result(
        self,
        run_id: int,
        model: str,
        input_text: str,
        expected_output: str,
        actual_output: str | None,
        scores: dict | None,
        latency_ms: float | None,
        tokens_used: int | None,
        cost: float | None,
        normalized_actual: str | None = None,
    ) -> int:
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """INSERT INTO eval_results
                   (run_id, model, input, expected_output, actual_output, normalized_actual,
                    scores, latency_ms, tokens_used, cost, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    model,
                    input_text,
                    expected_output,
                    actual_output,
                    normalized_actual,
                    json.dumps(scores) if scores else None,
                    latency_ms,
                    tokens_used,
                    cost,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def upsert_model_summary(
        self,
        run_id: int,
        model_name: str,
        exact_match: float | None = None,
        semantic_similarity: float | None = None,
        ci_lower: float | None = None,
        ci_upper: float | None = None,
        avg_latency_ms: float | None = None,
        total_cost: float | None = None,
    ) -> None:
        conn = self._get_connection()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO model_summaries
                   (run_id, model_name, exact_match, semantic_similarity,
                    ci_lower, ci_upper, avg_latency_ms, total_cost)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    model_name,
                    exact_match,
                    semantic_similarity,
                    ci_lower,
                    ci_upper,
                    avg_latency_ms,
                    total_cost,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_model_summaries(self, run_id: int) -> list[dict]:
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM model_summaries WHERE run_id = ?", (run_id,)
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_run(self, run_id: int) -> dict | None:
        conn = self._get_connection()
        try:
            row = conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["models"] = json.loads(result["models"])
            result["evaluators"] = json.loads(result["evaluators"])
            return result
        finally:
            conn.close()

    def get_results(self, run_id: int, model: str | None = None) -> list[dict]:
        conn = self._get_connection()
        try:
            if model:
                rows = conn.execute(
                    "SELECT * FROM eval_results WHERE run_id = ? AND model = ?",
                    (run_id, model),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM eval_results WHERE run_id = ?", (run_id,)
                ).fetchall()

            results = []
            for row in rows:
                r = dict(row)
                if r["scores"]:
                    r["scores"] = json.loads(r["scores"])
                results.append(r)
            return results
        finally:
            conn.close()

    def list_runs(self) -> list[dict]:
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM eval_runs ORDER BY created_at DESC"
            ).fetchall()
            results = []
            for row in rows:
                r = dict(row)
                r["models"] = json.loads(r["models"])
                r["evaluators"] = json.loads(r["evaluators"])
                results.append(r)
            return results
        finally:
            conn.close()
