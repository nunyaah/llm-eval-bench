import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATABASE_PATH = os.getenv("DATABASE_PATH", "eval_results.db")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Model & evaluator defaults  change these once, used everywhere
# ---------------------------------------------------------------------------
# LiteLLM model identifiers for the two models under comparison.
# For Ollama models use the "ollama/<name>" prefix.
# For cloud models use e.g. "gpt-4o-mini", "claude-3-haiku-20240307".
MODEL_A = os.getenv("MODEL_A", "ollama/llama3.2:1b")
MODEL_B = os.getenv("MODEL_B", "ollama/llama3.2:3b")
MODELS: list[str] = [MODEL_A, MODEL_B]

# Judge model used by LLMJudgeEvaluator and FaithfulnessEvaluator
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "ollama/llama3.2:3b")

# Evaluator names passed to the runner / API
EVALUATORS: list[str] = ["exact_match", "semantic_similarity", "llm_judge", "faithfulness"]

# Primary metric used for statistical comparison and dashboard failure view
# Must be one of the names in EVALUATORS
PRIMARY_METRIC = os.getenv("PRIMARY_METRIC", "llm_judge")

# Default dataset path
DATASET = os.getenv("DATASET", "data/complex_qa.json")

# System prompt shared by scripts that call models directly
SYSTEM_PROMPT = (
    "You are a concise factual assistant. "
    "Answer each question with ONLY the answer itself — no explanation, "
    "no punctuation, no extra words. "
    "For example: if asked 'What is the capital of France?' reply only 'Paris'."
)
