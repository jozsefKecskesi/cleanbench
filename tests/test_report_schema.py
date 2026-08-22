from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

REPO = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO / "src" / "cleanbench" / "schema" / "report.schema.json"
REPORTS = REPO / "reports"


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _validator() -> Draft202012Validator:
    return Draft202012Validator(_schema())


def _error_report() -> dict:
    return {
        "schema_version": "0.1.0",
        "dataset_id": "not-a-real/hub-id",
        "revision": None,
        "scored_at": "2026-08-22T00:00:00+00:00",
        "format": {"detected": "unknown", "valid_lerobot": False, "robot_type": None},
        "sample": {"n_episodes": 0, "n_frames": 0, "max_episodes": 0},
        "scores": {
            "overall": None,
            "physical": None,
            "annotation": None,
            "integrity": None,
        },
        "band": "unknown",
        "checks": [],
        "flagged_episodes": [],
        "warnings": [],
        "error": "DatasetNotFound: not-a-real/hub-id",
    }


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    return _validator()


def test_schema_file_exists() -> None:
    assert SCHEMA_PATH.is_file()


@pytest.mark.parametrize("path", sorted(REPORTS.glob("*.json")))
def test_committed_reports_match_schema(
    path: Path, validator: Draft202012Validator
) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    validator.validate(report)
    assert report["error"] is None
    assert report["schema_version"] == "0.1.0"


def test_invalid_hub_id_still_matches_schema(
    validator: Draft202012Validator,
) -> None:
    report = _error_report()
    validator.validate(report)
    assert report["error"]
    assert report["band"] == "unknown"
    assert report["checks"] == []


def test_missing_required_key_is_rejected(
    validator: Draft202012Validator,
) -> None:
    report = _error_report()
    del report["error"]
    errors = list(validator.iter_errors(report))
    assert errors