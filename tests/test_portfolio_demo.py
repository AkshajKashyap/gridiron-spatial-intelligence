from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "gridiron_spatial"
    / "portfolio_demo.py"
)
SPEC = importlib.util.spec_from_file_location("portfolio_demo_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


def _write_json(path: Path, value: object) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(data)
    return data


def _fixture(root: Path) -> tuple[Path, dict[str, bytes]]:
    (root / "pyproject.toml").write_text(
        '[project]\nname = "gridiron-spatial-intelligence"\nversion = "0.1.0"\n'
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
    documents = {
        demo.EVIDENCE_PATHS["cohort"]: {
            "cohort_summary": {
                "aggregate_table_counts": {"source_plays": {"rows": 14108}}
            },
            "reporting_summary": {"observed_game_count": 272},
        },
        demo.EVIDENCE_PATHS["cohort_manifest"]: {"artifact_format_version": "1.0"},
        demo.EVIDENCE_PATHS["normalized"]: {
            "aggregate": {"combined_rows": 5443515}
        },
        demo.EVIDENCE_PATHS["descriptive"]: {
            "aggregate_counts": {
                "origin_pair_count": 94293,
                "horizon_pair_count": 61156,
                "horizon_pair_counts": {"5": 31937, "10": 20680, "15": 8539},
            },
            "all_pair_results": all_results,
            "nearest_observed_defender_results": nearest,
        },
        demo.EVIDENCE_PATHS["selection"]: {"result_format_version": "selection"},
        demo.EVIDENCE_PATHS["frozen"]: {
            "evaluations": evaluations,
            "decision_evidence": {
                "confirmed_direction_count": 12,
                "reversed_direction_count": 0,
            },
            "leakage_diagnostics": {
                "validation_rows_used_for_fitting": 0,
                "cross_split_game_play_count": 0,
            },
            "bootstrap_configuration": {"unit": "play"},
        },
        demo.EVIDENCE_PATHS["interpretation"]: {"frozen_test_weeks_accessed": 0},
        demo.EVIDENCE_PATHS["calibration"]: {"frozen_test_weeks_accessed": 0},
        demo.EVIDENCE_PATHS["errors"]: {"frozen_test_weeks_accessed": 0},
    }
    originals = {
        path: _write_json(root / path, document)
        for path, document in documents.items()
    }
    entries = [
        {
            "relative_path": path,
            "semantic_role": f"role {index}",
            "sha256": hashlib.sha256(originals[path]).hexdigest(),
            "byte_size": len(originals[path]),
        }
        for index, path in enumerate(demo.EVIDENCE_PATHS.values(), 1)
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
        },
        "aggregate_byte_count": sum(map(len, originals.values())),
        "overall_validation_status": "PASS",
    }
    manifest_path = root / demo.DEFAULT_MANIFEST
    _write_json(manifest_path, manifest)
    return manifest_path, originals


def test_summary_and_markdown_are_deterministic_and_do_not_mutate_evidence(
    tmp_path: Path,
) -> None:
    _, originals = _fixture(tmp_path)
    first = demo.build_portfolio_summary(tmp_path)
    second = demo.build_portfolio_summary(tmp_path)
    first_markdown = demo.render_portfolio_markdown(first)
    second_markdown = demo.render_portfolio_markdown(second)

    assert first == second
    assert first_markdown == second_markdown
    assert first_markdown.encode() == demo.render_portfolio_markdown(first).encode()
    assert first["scale"]["normalized_entity_frame_rows"] == 5443515
    assert first["frozen_results"]["direction_agreement"] == 12
    assert {path: (tmp_path / path).read_bytes() for path in originals} == originals


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest: manifest.update(overall_validation_status="FAIL"), "not PASS"),
        (lambda manifest: manifest.update(evidence_file_count=8), "count"),
        (lambda manifest: manifest.update(release_version="9.9.9"), "version"),
        (
            lambda manifest: manifest["frozen_evaluation_policy"].update(
                frozen_evaluation_execution_count=2
            ),
            "execution",
        ),
        (
            lambda manifest: manifest["frozen_evaluation_policy"].update(
                frozen_selections_changed_count=1
            ),
            "selections",
        ),
        (
            lambda manifest: manifest["frozen_evaluation_policy"].update(
                frozen_comparators_changed_count=1
            ),
            "comparators",
        ),
    ],
)
def test_manifest_policy_failures_are_rejected(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    manifest_path, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    mutation(manifest)
    _write_json(manifest_path, manifest)
    with pytest.raises(demo.PortfolioDemoError, match=message):
        demo.build_portfolio_summary(tmp_path)


def test_missing_checksum_size_and_package_mismatches_are_rejected(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    target = tmp_path / manifest["evidence_files"][0]["relative_path"]
    target.unlink()
    with pytest.raises(demo.PortfolioDemoError, match="Missing"):
        demo.build_portfolio_summary(tmp_path)

    _, _ = _fixture(tmp_path)
    target.write_text('{"changed": true}\n')
    with pytest.raises(demo.PortfolioDemoError, match="Checksum"):
        demo.build_portfolio_summary(tmp_path)

    _, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["evidence_files"][0]["byte_size"] += 1
    _write_json(manifest_path, manifest)
    with pytest.raises(demo.PortfolioDemoError, match="Byte-size"):
        demo.build_portfolio_summary(tmp_path)

    _, _ = _fixture(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "gridiron-spatial-intelligence"\nversion = "9.9.9"\n'
    )
    with pytest.raises(demo.PortfolioDemoError, match="Package version"):
        demo.build_portfolio_summary(tmp_path)


def test_absolute_path_is_rejected_from_rendered_report(tmp_path: Path) -> None:
    manifest_path, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["evidence_files"][0]["semantic_role"] = "/tmp/private"
    _write_json(manifest_path, manifest)
    summary = demo.build_portfolio_summary(tmp_path)
    with pytest.raises(demo.PortfolioDemoError, match="absolute"):
        demo.render_portfolio_markdown(summary)


def test_atomic_write_and_check_mode_detect_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _fixture(tmp_path)
    output = Path("reports/portfolio/release_0.1.0.md")
    assert demo.run_cli(["--repository-root", str(tmp_path), "--output", str(output)]) == 0
    report = tmp_path / output
    stable = report.read_bytes()
    assert demo.run_cli(
        ["--repository-root", str(tmp_path), "--output", str(output), "--check"]
    ) == 0
    report.write_text("drift\n")
    assert demo.run_cli(
        ["--repository-root", str(tmp_path), "--output", str(output), "--check"]
    ) == 1

    report.write_bytes(stable)

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("synthetic atomic failure")

    monkeypatch.setattr(demo.os, "replace", fail_replace)
    assert demo.run_cli(["--repository-root", str(tmp_path), "--output", str(output)]) == 1
    assert report.read_bytes() == stable
    assert not list(report.parent.glob(f".{report.name}.*"))
