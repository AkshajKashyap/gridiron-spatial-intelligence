from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.build_release_evidence_manifest as release


FULL_WEEKS = [f"2023_w{week:02d}" for week in range(1, 19)]


def _write_json(path: Path, value: object) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(data)
    return data


def _synthetic_repository(root: Path) -> dict[str, bytes]:
    (root / "pyproject.toml").write_text(
        '[project]\nname = "gridiron-spatial-intelligence"\nversion = "0.1.0"\n'
    )
    selection_path = release.EVIDENCE_ALLOWLIST[4][0]
    source: dict[str, object] = {}
    original: dict[str, bytes] = {}
    for relative, _, _ in release.EVIDENCE_ALLOWLIST:
        if relative in {item[0] for item in release.EVIDENCE_ALLOWLIST[:4]}:
            source[relative] = {
                "result_format_version": "synthetic_v1",
                "processed_weeks": FULL_WEEKS,
                "status": "PASS",
            }
        elif relative == selection_path:
            source[relative] = {
                "result_format_version": "selection_v1",
                "processed_weeks": FULL_WEEKS[:15],
            }
        elif relative == release.EVIDENCE_ALLOWLIST[5][0]:
            source[relative] = {}
        else:
            source[relative] = {
                "result_format_version": "interpretation_v1",
                "processed_weeks": FULL_WEEKS[:15],
            }
    for relative, value in source.items():
        if relative != release.EVIDENCE_ALLOWLIST[5][0]:
            original[relative] = _write_json(root / relative, value)
    selection_sha = hashlib.sha256(original[selection_path]).hexdigest()
    for relative, _, _ in release.EVIDENCE_ALLOWLIST[6:]:
        value = source[relative]
        assert isinstance(value, dict)
        value["binding_milestone_4_selection_checksum"] = {
            "algorithm": "sha256",
            "value": selection_sha,
        }
        original[relative] = _write_json(root / relative, value)
    frozen = {
        "result_format_version": "frozen_v1",
        "selection_result_sha256": selection_sha,
        "development_weeks": FULL_WEEKS[:12],
        "frozen_weeks": FULL_WEEKS[15:],
        "frozen_evaluation_execution_count": 1,
        "frozen_selections_changed_count": 0,
        "frozen_comparators_changed_count": 0,
        "leakage_diagnostics": {"status": "PASS"},
        "reconciliation_diagnostics": {"status": "PASS", "mismatch_count": 0},
    }
    frozen_path = release.EVIDENCE_ALLOWLIST[5][0]
    original[frozen_path] = _write_json(root / frozen_path, frozen)
    return original


def test_release_manifest_allowlist_hashes_determinism_and_cross_checks(
    tmp_path: Path,
) -> None:
    original = _synthetic_repository(tmp_path)
    expected_paths = (
        "artifacts/milestone_2/cohorts/cohort_summary.json",
        "artifacts/milestone_2/cohorts/manifest.json",
        "artifacts/milestone_2/normalized_tracking/manifest.json",
        "artifacts/milestone_3/full_season_separation_summary.json",
        "artifacts/milestone_4/baseline_selection.json",
        "artifacts/milestone_4/frozen_test_result.json",
        "artifacts/milestone_5/model_interpretation_summary.json",
        "artifacts/milestone_5/classifier_calibration_summary.json",
        "artifacts/milestone_5/validation_error_summary.json",
    )
    assert tuple(item[0] for item in release.EVIDENCE_ALLOWLIST) == expected_paths

    first, hashes = release.build_manifest(tmp_path)
    second, _ = release.build_manifest(tmp_path)

    assert first["overall_validation_status"] == "PASS"
    assert [item["relative_path"] for item in first["evidence_files"]] == list(
        expected_paths
    )
    for entry in first["evidence_files"]:
        data = original[entry["relative_path"]]
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
        assert entry["byte_size"] == len(data)
        assert hashes[entry["relative_path"]] == entry["sha256"]
    assert release.manifest_bytes(first) == release.manifest_bytes(second)
    assert {
        path: (tmp_path / path).read_bytes() for path in expected_paths
    } == original


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (b"{broken", "Invalid UTF-8 JSON"),
        (b"[]", "JSON root must be an object"),
        (b'{"path": "/tmp/private"}', "absolute path"),
        (b'{"path": "../private"}', "parent traversal"),
    ],
)
def test_release_manifest_rejects_invalid_evidence(
    tmp_path: Path,
    replacement: bytes,
    message: str,
) -> None:
    _synthetic_repository(tmp_path)
    target = tmp_path / release.EVIDENCE_ALLOWLIST[0][0]
    target.write_bytes(replacement)
    with pytest.raises(release.EvidenceValidationError, match=message):
        release.build_manifest(tmp_path)


def test_release_manifest_rejects_missing_and_oversized_evidence(
    tmp_path: Path,
) -> None:
    _synthetic_repository(tmp_path)
    target = tmp_path / release.EVIDENCE_ALLOWLIST[0][0]
    target.unlink()
    with pytest.raises(release.EvidenceValidationError, match="Missing"):
        release.build_manifest(tmp_path)
    target.write_bytes(b" " * (release.MAX_EVIDENCE_BYTES + 1))
    with pytest.raises(release.EvidenceValidationError, match="1 MiB"):
        release.build_manifest(tmp_path)


def test_release_manifest_reports_version_checksum_and_frozen_policy_failures(
    tmp_path: Path,
) -> None:
    _synthetic_repository(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "gridiron-spatial-intelligence"\nversion = "9.9.9"\n'
    )
    frozen_path = tmp_path / release.EVIDENCE_ALLOWLIST[5][0]
    frozen = json.loads(frozen_path.read_text())
    frozen.update(
        {
            "selection_result_sha256": "wrong",
            "frozen_evaluation_execution_count": 2,
            "frozen_selections_changed_count": 1,
            "frozen_comparators_changed_count": 1,
        }
    )
    _write_json(frozen_path, frozen)

    manifest, _ = release.build_manifest(tmp_path)
    statuses = {
        check["check"]: check["status"]
        for check in manifest["cross_artifact_validation"]["checks"]
    }
    assert manifest["overall_validation_status"] == "FAIL"
    assert statuses["package_version"] == "FAIL"
    assert statuses["selection_checksum"] == "FAIL"
    assert statuses["frozen_execution_count"] == "FAIL"
    assert statuses["frozen_selections_changed"] == "FAIL"
    assert statuses["frozen_comparators_changed"] == "FAIL"


def test_atomic_write_preserves_destination_on_replace_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "manifest.json"
    destination.write_bytes(b"existing")

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(release.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic"):
        release.write_manifest_atomic(destination, b"replacement")

    assert destination.read_bytes() == b"existing"
    assert sorted(tmp_path.iterdir()) == [destination]
