import json
from pathlib import Path


def load_dataset(path: str) -> list[dict]:
    """Load a JSON evaluation dataset.

    Expected format:
        [{"input": "...", "expected_output": "..."}, ...]
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array")

    for i, item in enumerate(data):
        if "input" not in item:
            raise ValueError(f"Item {i} missing 'input' field")
        if "expected_output" not in item:
            raise ValueError(f"Item {i} missing 'expected_output' field")

    return data
