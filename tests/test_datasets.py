import json
import tempfile

import pytest

from src.datasets.loader import load_dataset


@pytest.fixture
def sample_dataset(tmp_path):
    data = [
        {"input": "What is 2+2?", "expected_output": "4"},
        {"input": "Capital of France?", "expected_output": "Paris"},
    ]
    path = tmp_path / "test_data.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_load_dataset_success(sample_dataset):
    data = load_dataset(sample_dataset)
    assert len(data) == 2
    assert data[0]["input"] == "What is 2+2?"
    assert data[1]["expected_output"] == "Paris"


def test_load_dataset_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_dataset("nonexistent.json")


def test_load_dataset_invalid_format(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"not": "a list"}')
    with pytest.raises(ValueError, match="must be a JSON array"):
        load_dataset(str(path))


def test_load_dataset_missing_fields(tmp_path):
    path = tmp_path / "missing.json"
    path.write_text(json.dumps([{"input": "hello"}]))
    with pytest.raises(ValueError, match="missing 'expected_output'"):
        load_dataset(str(path))
