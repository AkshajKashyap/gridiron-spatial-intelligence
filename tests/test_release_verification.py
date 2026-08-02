from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

import gridiron_spatial.portfolio_demo as portfolio
import gridiron_spatial.release_verification as verification


@pytest.fixture(scope="module", autouse=True)
def _restore_packaging_test_import_state():
    yield
    for name in list(sys.modules):
        if name == "gridiron_spatial" or name.startswith("gridiron_spatial."):
            sys.modules.pop(name, None)


def _write_json(path: Path, value: object) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2) + "\n").encode("utf-8")
    path.write_bytes(data)
    return data


def _workflow() -> str:
    return """name: CI

on:
  push:
  pull_request:
  workflow_dispatch:

permissions:
  contents: read

jobs:
  gate:
    strategy:
      matrix:
        python-version: ["3.11", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: python -m pip install -e ".[test]"
      - run: python -m pytest -q
      - run: python -m compileall -q src scripts tests
      - run: python -m pip check
      - run: python -m build
      - run: python scripts/build_release_evidence_manifest.py
      - run: python scripts/run_portfolio_demo.py --check
      - run: python scripts/verify_release.py
      - run: git diff --exit-code
      - run: git diff --check
"""


def _release_fixture(root: Path) -> tuple[list[str], dict[str, bytes]]:
    (root / "pyproject.toml").write_text(
        """[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "gridiron-spatial-intelligence"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numpy", "pandas>=2.0", "pyarrow", "scikit-learn"]

[project.optional-dependencies]
test = ["pytest"]

[tool.setuptools.packages.find]
where = ["src"]
"""
    )
    all_results = [
        {"horizon": h, "pair_count": count, "mean_separation_change": change}
        for h, count, change in (
            (5, 31937, -0.23),
            (10, 20680, -0.17),
            (15, 8539, 0.42),
        )
    ]
    nearest = [
        {"horizon": h, "pair_count": count, "mean_separation_change": change}
        for h, count, change in (
            (5, 12596, -0.04),
            (10, 7293, 0.12),
            (15, 2718, 0.56),
        )
    ]
    evaluations = []
    for task, metric in (("regression", "mae"), ("classification", "log_loss")):
        for horizon in (5, 10, 15):
            evaluations.append(
                {
                    "population": "all_pairs",
                    "horizon": horizon,
                    "task": task,
                    "selected_metrics": {metric: horizon / 10},
                    "comparator_metrics": {metric: horizon / 10 + 0.1},
                    "metric_differences": {metric: -0.1},
                }
            )
    documents: dict[str, object] = {
        verification.EVIDENCE_PATHS[0]: {
            "cohort_summary": {
                "aggregate_table_counts": {"source_plays": {"rows": 14108}}
            },
            "reporting_summary": {"observed_game_count": 272},
        },
        verification.EVIDENCE_PATHS[1]: {"artifact_format_version": "1.0"},
        verification.EVIDENCE_PATHS[2]: {
            "aggregate": {"combined_rows": 5443515}
        },
        verification.EVIDENCE_PATHS[3]: {
            "aggregate_counts": {
                "origin_pair_count": 94293,
                "horizon_pair_count": 61156,
                "horizon_pair_counts": {"5": 31937, "10": 20680, "15": 8539},
            },
            "all_pair_results": all_results,
            "nearest_observed_defender_results": nearest,
        },
        verification.EVIDENCE_PATHS[4]: {"result_format_version": "selection"},
        verification.EVIDENCE_PATHS[5]: {},
        verification.EVIDENCE_PATHS[6]: {"frozen_test_weeks_accessed": 0},
        verification.EVIDENCE_PATHS[7]: {"frozen_test_weeks_accessed": 0},
        verification.EVIDENCE_PATHS[8]: {"frozen_test_weeks_accessed": 0},
    }
    originals: dict[str, bytes] = {}
    for path, document in documents.items():
        if path != verification.EVIDENCE_PATHS[5]:
            originals[path] = _write_json(root / path, document)
    selection_checksum = hashlib.sha256(
        originals[verification.EVIDENCE_PATHS[4]]
    ).hexdigest()
    frozen = {
        "selection_result_sha256": selection_checksum,
        "evaluations": evaluations,
        "decision_evidence": {
            "confirmed_direction_count": 12,
            "reversed_direction_count": 0,
        },
        "leakage_diagnostics": {
            "status": "PASS",
            "validation_rows_used_for_fitting": 0,
            "cross_split_game_play_count": 0,
        },
        "reconciliation_diagnostics": {"status": "PASS", "mismatch_count": 0},
        "bootstrap_configuration": {"unit": "play"},
    }
    originals[verification.EVIDENCE_PATHS[5]] = _write_json(
        root / verification.EVIDENCE_PATHS[5], frozen
    )
    entries = [
        {
            "relative_path": path,
            "semantic_role": f"role {index}",
            "sha256": hashlib.sha256(originals[path]).hexdigest(),
            "byte_size": len(originals[path]),
        }
        for index, path in enumerate(verification.EVIDENCE_PATHS, 1)
    ]
    manifest = {
        "manifest_format_version": "release_evidence_manifest_v1",
        "release_version": "0.1.0",
        "distribution_name": "gridiron-spatial-intelligence",
        "evidence_file_count": 9,
        "evidence_files": entries,
        "frozen_evaluation_policy": {
            "frozen_evaluation_execution_count": 1,
            "frozen_selections_changed_count": 0,
            "frozen_comparators_changed_count": 0,
            "leakage_validation_status": "PASS",
            "reconciliation_status": "PASS",
            "reconciliation_mismatch_count": 0,
            "final_for_release": "0.1.0",
        },
        "aggregate_byte_count": sum(map(len, originals.values())),
        "overall_validation_status": "PASS",
    }
    _write_json(root / verification.MANIFEST_PATH, manifest)

    for relative in verification.REQUIRED_MARKDOWN:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Release\n")
    (root / "README.md").write_text(
        "# Release\n\n[Architecture](docs/ARCHITECTURE.md)\n"
    )
    (root / "docs/ARCHITECTURE.md").write_text(
        "# Architecture\n\n[External](https://example.com)\n\n"
        "```markdown\n[Ignored](missing.md)\n```\n"
    )
    workflow_path = root / verification.WORKFLOW_PATH
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text(_workflow())
    for relative in (
        "tests/test_packaging.py",
        "tests/test_release_evidence.py",
        "tests/test_portfolio_demo.py",
        "tests/test_release_verification.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic\n")

    summary = portfolio.build_portfolio_summary(root)
    report = root / verification.REPORT_PATH
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(portfolio.render_portfolio_markdown(summary))
    tracked = sorted(
        list(verification.EVIDENCE_PATHS)
        + [
            verification.MANIFEST_PATH.as_posix(),
            verification.REPORT_PATH.as_posix(),
            verification.WORKFLOW_PATH.as_posix(),
            *verification.REQUIRED_MARKDOWN,
            "tests/test_packaging.py",
            "tests/test_release_evidence.py",
            "tests/test_portfolio_demo.py",
            "tests/test_release_verification.py",
        ]
    )
    return tracked, originals


def _verify(root: Path, monkeypatch, tracked: list[str]) -> dict[str, object]:
    monkeypatch.setattr(verification, "_tracked_files", lambda unused: list(tracked))
    return verification.verify_release(root)


def test_passing_release_is_deterministic_and_does_not_mutate_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tracked, originals = _release_fixture(tmp_path)
    before = {
        path: (tmp_path / path).read_bytes()
        for path in originals
    }
    first = _verify(tmp_path, monkeypatch, tracked)
    second = _verify(tmp_path, monkeypatch, tracked)

    assert first == second
    assert json.loads(json.dumps(first, sort_keys=True)) == first
    assert first["gates"]["tracked_artifact_policy"]["diagnostics"] == []
    assert first["overall_status"] == "PASS", first
    assert all(
        gate["status"] == "PASS" for gate in first["gates"].values()
    )
    assert before == {
        path: (tmp_path / path).read_bytes()
        for path in originals
    }


@pytest.mark.parametrize(
    ("change", "diagnostic"),
    [
        ("version", "version mismatch"),
        ("dependency", "missing runtime dependencies"),
    ],
)
def test_package_metadata_failures(
    tmp_path: Path,
    monkeypatch,
    change: str,
    diagnostic: str,
) -> None:
    tracked, _ = _release_fixture(tmp_path)
    pyproject = (tmp_path / "pyproject.toml").read_text()
    if change == "version":
        pyproject = pyproject.replace('version = "0.1.0"', 'version = "9.9.9"')
    else:
        pyproject = pyproject.replace(', "pyarrow"', "")
    (tmp_path / "pyproject.toml").write_text(pyproject)
    result = _verify(tmp_path, monkeypatch, tracked)
    assert result["overall_status"] == "FAIL"
    assert any(
        diagnostic in value
        for value in result["gates"]["package_metadata"]["diagnostics"]
    )


def test_missing_and_checksum_mismatched_evidence_fail(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tracked, _ = _release_fixture(tmp_path)
    target = tmp_path / verification.EVIDENCE_PATHS[0]
    target.unlink()
    missing = _verify(tmp_path, monkeypatch, tracked)
    assert missing["gates"]["evidence_manifest"]["status"] == "FAIL"

    _release_fixture(tmp_path)
    target.write_text('{"changed": true}\n')
    mismatch = _verify(tmp_path, monkeypatch, tracked)
    assert mismatch["gates"]["evidence_manifest"]["checksum_status"] == "FAIL"


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("frozen_evaluation_execution_count", 2, "execution_count"),
        ("frozen_selections_changed_count", 1, "selections_changed"),
        ("reconciliation_status", "FAIL", "reconciliation_status"),
    ],
)
def test_frozen_safeguard_failures(
    tmp_path: Path,
    monkeypatch,
    key: str,
    value: object,
    expected: str,
) -> None:
    tracked, _ = _release_fixture(tmp_path)
    path = tmp_path / verification.MANIFEST_PATH
    manifest = json.loads(path.read_text())
    manifest["frozen_evaluation_policy"][key] = value
    _write_json(path, manifest)
    result = _verify(tmp_path, monkeypatch, tracked)
    assert result["gates"]["frozen_evaluation_safeguards"]["status"] == "FAIL"
    assert any(
        expected in item
        for item in result["gates"]["frozen_evaluation_safeguards"]["diagnostics"]
    )


@pytest.mark.parametrize(
    "extra",
    [
        "artifacts/secret/player_predictions.json",
        "artifacts/milestone_3/pairs.parquet",
    ],
)
def test_unapproved_tracked_artifacts_fail(
    tmp_path: Path,
    monkeypatch,
    extra: str,
) -> None:
    tracked, _ = _release_fixture(tmp_path)
    result = _verify(tmp_path, monkeypatch, tracked + [extra])
    assert result["gates"]["tracked_artifact_policy"]["status"] == "FAIL"


def test_markdown_links_detect_breakage_but_ignore_external_and_fenced(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tracked, _ = _release_fixture(tmp_path)
    passing = _verify(tmp_path, monkeypatch, tracked)
    assert passing["gates"]["documentation_links"]["status"] == "PASS"
    (tmp_path / "README.md").write_text("[Broken](docs/missing.md)\n")
    failing = _verify(tmp_path, monkeypatch, tracked)
    links = failing["gates"]["documentation_links"]
    assert links["status"] == "FAIL"
    assert links["broken_links"] == ["README.md:1: docs/missing.md"]


def test_missing_required_file_and_portfolio_drift_fail(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tracked, _ = _release_fixture(tmp_path)
    (tmp_path / "docs/MODEL_CARD.md").unlink()
    missing = _verify(tmp_path, monkeypatch, tracked)
    assert missing["gates"]["required_release_files"]["status"] == "FAIL"

    _release_fixture(tmp_path)
    (tmp_path / verification.REPORT_PATH).write_text("drift\n")
    drift = _verify(tmp_path, monkeypatch, tracked)
    assert drift["gates"]["portfolio_report"]["drift_status"] == "FAIL"


def test_diagnostic_ordering_is_stable(tmp_path: Path, monkeypatch) -> None:
    tracked, _ = _release_fixture(tmp_path)
    result = _verify(
        tmp_path,
        monkeypatch,
        tracked
        + [
            "artifacts/z.parquet",
            "artifacts/a.parquet",
        ],
    )
    diagnostics = result["gates"]["tracked_artifact_policy"]["diagnostics"]
    assert diagnostics == sorted(diagnostics)
