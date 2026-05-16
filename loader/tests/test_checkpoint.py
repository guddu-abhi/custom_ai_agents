import json

import pytest
from loader.utils.checkpoint import CheckpointManager


@pytest.fixture
def cp(tmp_path):
    return CheckpointManager(tmp_path / "progress.json")


def test_load_returns_zero_when_file_missing(cp):
    assert cp.load() == 0


def test_save_and_load_roundtrip(cp, tmp_path):
    cp.save(42)
    assert cp.load() == 42


def test_load_returns_zero_on_corrupt_file(cp, tmp_path):
    path = tmp_path / "progress.json"
    path.write_text("not json{{{")
    assert cp.load() == 0


def test_load_returns_zero_on_missing_key(cp, tmp_path):
    path = tmp_path / "progress.json"
    path.write_text(json.dumps({"other_key": 99}))
    assert cp.load() == 0


def test_clear_removes_file(cp):
    cp.save(10)
    cp.clear()
    assert not (cp._path.exists())


def test_clear_is_safe_when_no_file(cp):
    cp.clear()  # should not raise


def test_save_overwrites_previous(cp):
    cp.save(10)
    cp.save(99)
    assert cp.load() == 99
