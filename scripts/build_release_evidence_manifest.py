#!/usr/bin/env python3
"""Build the deterministic compact-evidence manifest for release 0.1.0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import tomllib
from typing import Any, Iterable, Mapping, Sequence


RELEASE_VERSION = "0.1.0"
MANIFEST_FORMAT_VERSION = "release_evidence_manifest_v1"
MAX_EVIDENCE_BYTES = 1024 * 1024
DEFAULT_OUTPUT = Path("artifacts/release/v0.1.0/evidence_manifest.json")
EVIDENCE_ALLOWLIST = (
    (
        "artifacts/milestone_2/cohorts/cohort_summary.json",
        "cohort validation summary",
        "provenance",
    ),
    (
        "artifacts/milestone_2/cohorts/manifest.json",
        "cohort artifact manifest",
        "provenance",
    ),
    (
        "artifacts/milestone_2/normalized_tracking/manifest.json",
        "normalized tracking artifact manifest",
        "provenance",
    ),
    (
        "artifacts/milestone_3/full_season_separation_summary.json",
        "full-season separation summary",
        "descriptive result",
    ),
    (
        "artifacts/milestone_4/baseline_selection.json",
        "audited baseline selection",
        "frozen selection",
    ),
    (
        "artifacts/milestone_4/frozen_test_result.json",
        "one-time frozen-test result",
        "frozen evaluation",
    ),
    (
        "artifacts/milestone_5/model_interpretation_summary.json",
        "model interpretation summary",
        "interpretation result",
    ),
    (
        "artifacts/milestone_5/classifier_calibration_summary.json",
        "classifier calibration summary",
        "interpretation result",
    ),
    (
        "artifacts/milestone_5/validation_error_summary.json",
        "validation error summary",
        "interpretation result",
    ),
)

EVIDENCE_POLICY = {
    "versioned": "Compact aggregate evidence is versioned.",
    "raw_data": "Raw NFL data is not versioned.",
    "parquet": "Normalized or cohort Parquet files are not versioned.",
    "row_level": "Pair-level rows and individual predictions are not versioned.",
    "immutability": "Evidence JSON files are immutable once included in the release.",
    "regeneration": (
        "Analytical evidence may be regenerated only through the documented pipeline."
    ),
    "frozen_evaluation": (
        "The one-time frozen evaluation must not be rerun for model selection."
    ),
    "checksum_review": (
        "Checksum changes require explicit review and a new release decision."
    ),
}


class EvidenceValidationError(ValueError):
    """Raised when evidence cannot safely enter a release manifest."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _walk_values(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, Mapping):
        for nested in value.values():
            yield from _walk_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_values(nested)


def _unsafe_reference(value: Any) -> str | None:
    for nested in _walk_values(value):
        if not isinstance(nested, str):
            continue
        if nested.startswith(("/", "\\\\")) or re.match(r"^[A-Za-z]:[\\/]", nested):
            return f"absolute path: {nested}"
        if ".." in PurePosixPath(nested.replace("\\", "/")).parts:
            return f"parent traversal: {nested}"
    return None


def _find_first(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    for value in mapping.values():
        if isinstance(value, Mapping):
            found = _find_first(value, keys)
            if found is not None:
                return found
    return None


def _format_version(document: Mapping[str, Any]) -> Any:
    return _find_first(
        document,
        (
            "result_format_version",
            "analysis_format_version",
            "artifact_format_version",
        ),
    )


def _declared_status(document: Mapping[str, Any]) -> Any:
    return _find_first(
        document,
        (
            "overall_validation_status",
            "validation_status",
            "status",
            "aggregate_reconciliation_status",
        ),
    )


def _processed_week_summary(document: Mapping[str, Any]) -> dict[str, Any] | None:
    weeks = _find_first(document, ("processed_weeks",))
    if not isinstance(weeks, list):
        return None
    return {
        "count": len(weeks),
        "first": weeks[0] if weeks else None,
        "last": weeks[-1] if weeks else None,
    }


def _validation(check: str, passed: bool, actual: Any) -> dict[str, Any]:
    return {"check": check, "status": "PASS" if passed else "FAIL", "actual": actual}


def _execution_count(frozen: Mapping[str, Any]) -> tuple[int | None, str]:
    for key in ("frozen_evaluation_execution_count", "evaluation_execution_count"):
        if key in frozen:
            return frozen[key], key
    authorization = frozen.get("audit_authorization")
    if (
        isinstance(authorization, str)
        and "ONE-TIME FROZEN EVALUATION" in authorization
        and frozen.get("evaluation_timestamp_utc")
    ):
        return 1, "audit_authorization plus recorded evaluation result"
    return None, "missing"


def _change_counts(frozen: Mapping[str, Any]) -> tuple[Any, Any, str]:
    selection = frozen.get("frozen_selections_changed_count")
    comparator = frozen.get("frozen_comparators_changed_count")
    if selection is not None and comparator is not None:
        return selection, comparator, "explicit fields"
    combined = frozen.get("selection_or_comparator_change_count")
    if combined == 0:
        return 0, 0, "selection_or_comparator_change_count"
    return selection, comparator, "missing separate counts"


def _contains_row_level_results(frozen: Mapping[str, Any]) -> bool:
    prohibited = {
        "predictions",
        "individual_predictions",
        "pair_level_records",
        "pair_records",
        "prediction_records",
    }
    return any(
        isinstance(value, Mapping) and bool(prohibited.intersection(value))
        for value in _walk_values(frozen)
    )


def _week_boundaries(documents: Mapping[str, Mapping[str, Any]]) -> bool:
    full = [f"2023_w{week:02d}" for week in range(1, 19)]
    development_validation = full[:15]
    full_paths = (
        EVIDENCE_ALLOWLIST[0][0],
        EVIDENCE_ALLOWLIST[1][0],
        EVIDENCE_ALLOWLIST[2][0],
        EVIDENCE_ALLOWLIST[3][0],
    )
    dev_val_paths = (
        EVIDENCE_ALLOWLIST[4][0],
        EVIDENCE_ALLOWLIST[6][0],
        EVIDENCE_ALLOWLIST[7][0],
        EVIDENCE_ALLOWLIST[8][0],
    )
    if any(
        _find_first(documents[path], ("processed_weeks",)) != full
        for path in full_paths
    ):
        return False
    if any(
        _find_first(documents[path], ("processed_weeks",))
        != development_validation
        for path in dev_val_paths
    ):
        return False
    frozen = documents[EVIDENCE_ALLOWLIST[5][0]]
    return (
        frozen.get("development_weeks") == full[:12]
        and frozen.get("frozen_weeks") == full[15:]
    )


def build_manifest(
    repository_root: Path,
    *,
    allowlist: Sequence[tuple[str, str, str]] = EVIDENCE_ALLOWLIST,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate source evidence and return its deterministic manifest."""
    repository_root = repository_root.resolve()
    project = tomllib.loads((repository_root / "pyproject.toml").read_text("utf-8"))
    entries: list[dict[str, Any]] = []
    documents: dict[str, Mapping[str, Any]] = {}
    initial_hashes: dict[str, str] = {}

    for relative, role, category in allowlist:
        path = repository_root / relative
        if not path.is_file():
            raise EvidenceValidationError(f"Missing evidence file: {relative}")
        data = path.read_bytes()
        if len(data) > MAX_EVIDENCE_BYTES:
            raise EvidenceValidationError(f"Evidence exceeds 1 MiB: {relative}")
        try:
            document = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvidenceValidationError(f"Invalid UTF-8 JSON: {relative}") from exc
        if not isinstance(document, dict):
            raise EvidenceValidationError(f"JSON root must be an object: {relative}")
        unsafe = _unsafe_reference(document)
        if unsafe:
            raise EvidenceValidationError(f"Unsafe reference in {relative}: {unsafe}")
        digest = _sha256(data)
        initial_hashes[relative] = digest
        documents[relative] = document
        entries.append(
            {
                "relative_path": relative,
                "semantic_role": role,
                "evidence_category": category,
                "sha256": digest,
                "byte_size": len(data),
                "top_level_json_keys": sorted(document),
                "declared_format_version": _format_version(document),
                "declared_validation_status": _declared_status(document),
                "processed_week_summary": _processed_week_summary(document),
            }
        )

    if tuple(allowlist) == EVIDENCE_ALLOWLIST and len(entries) != 9:
        raise EvidenceValidationError("Production evidence allowlist must contain nine files")

    frozen = documents[EVIDENCE_ALLOWLIST[5][0]]
    selection_digest = initial_hashes[EVIDENCE_ALLOWLIST[4][0]]
    declared_selection_digest = frozen.get("selection_result_sha256")
    execution_count, execution_source = _execution_count(frozen)
    selections_changed, comparators_changed, change_source = _change_counts(frozen)
    leakage_status = _find_first(frozen, ("leakage_diagnostics",))
    leakage_status = leakage_status.get("status") if isinstance(leakage_status, dict) else None
    reconciliation = _find_first(frozen, ("reconciliation_diagnostics",))
    reconciliation_status = (
        reconciliation.get("status") if isinstance(reconciliation, dict) else None
    )
    reconciliation_mismatches = (
        reconciliation.get("mismatch_count")
        if isinstance(reconciliation, dict)
        else None
    )
    milestone_5_checksums = {
        path: _find_first(
            documents[path], ("binding_milestone_4_selection_checksum",)
        )
        for path, _, _ in EVIDENCE_ALLOWLIST[6:]
    }
    milestone_5_values = {
        path: value.get("value") if isinstance(value, dict) else None
        for path, value in milestone_5_checksums.items()
    }

    checks = [
        _validation(
            "package_version",
            project["project"].get("version") == RELEASE_VERSION,
            project["project"].get("version"),
        ),
        _validation("evidence_file_count", len(entries) == 9, len(entries)),
        _validation("all_json_object_rooted", True, len(entries)),
        _validation("absolute_path_scan", True, "no unsafe paths"),
        _validation(
            "maximum_file_size",
            all(entry["byte_size"] <= MAX_EVIDENCE_BYTES for entry in entries),
            max(entry["byte_size"] for entry in entries),
        ),
        _validation(
            "selection_checksum",
            declared_selection_digest == selection_digest,
            {
                "declared": declared_selection_digest,
                "computed": selection_digest,
            },
        ),
        _validation(
            "milestone_5_selection_checksums",
            all(value == selection_digest for value in milestone_5_values.values()),
            milestone_5_values,
        ),
        _validation("frozen_execution_count", execution_count == 1, execution_count),
        _validation(
            "frozen_selections_changed", selections_changed == 0, selections_changed
        ),
        _validation(
            "frozen_comparators_changed", comparators_changed == 0, comparators_changed
        ),
        _validation("frozen_leakage", leakage_status == "PASS", leakage_status),
        _validation(
            "frozen_reconciliation",
            reconciliation_status == "PASS" and reconciliation_mismatches == 0,
            {
                "status": reconciliation_status,
                "mismatch_count": reconciliation_mismatches,
            },
        ),
        _validation(
            "no_row_level_frozen_results",
            not _contains_row_level_results(frozen),
            not _contains_row_level_results(frozen),
        ),
        _validation(
            "processed_week_boundaries",
            _week_boundaries(documents),
            _week_boundaries(documents),
        ),
    ]
    unchanged = all(
        _sha256((repository_root / relative).read_bytes()) == digest
        for relative, digest in initial_hashes.items()
    )
    checks.append(_validation("source_file_mutation_check", unchanged, unchanged))
    cross_status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"

    manifest = {
        "manifest_format_version": MANIFEST_FORMAT_VERSION,
        "release_version": RELEASE_VERSION,
        "distribution_name": project["project"]["name"],
        "evidence_policy": EVIDENCE_POLICY,
        "frozen_evaluation_policy": {
            "selection_artifact": EVIDENCE_ALLOWLIST[4][0],
            "frozen_result": EVIDENCE_ALLOWLIST[5][0],
            "frozen_evaluation_execution_count": execution_count,
            "execution_count_source": execution_source,
            "frozen_selections_changed_count": selections_changed,
            "frozen_comparators_changed_count": comparators_changed,
            "change_count_source": change_source,
            "leakage_validation_status": leakage_status,
            "reconciliation_status": reconciliation_status,
            "reconciliation_mismatch_count": reconciliation_mismatches,
            "final_for_release": RELEASE_VERSION,
            "model_selection_rerun_permitted": False,
        },
        "evidence_file_count": len(entries),
        "evidence_files": entries,
        "cross_artifact_validation": {
            "status": cross_status,
            "checks": checks,
        },
        "aggregate_byte_count": sum(entry["byte_size"] for entry in entries),
        "overall_validation_status": cross_status,
    }
    return manifest, initial_hashes


def manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def write_manifest_atomic(output: Path, data: bytes) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output.parent, prefix=f".{output.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _print_summary(manifest: Mapping[str, Any], output: Path) -> None:
    policy = manifest["frozen_evaluation_policy"]
    checks = {
        item["check"]: item for item in manifest["cross_artifact_validation"]["checks"]
    }
    print(f"Release version: {manifest['release_version']}")
    print(f"Evidence-file count: {manifest['evidence_file_count']}")
    for entry in manifest["evidence_files"]:
        print(
            f"{entry['relative_path']} sha256={entry['sha256']} "
            f"bytes={entry['byte_size']}"
        )
    print(f"Aggregate bytes: {manifest['aggregate_byte_count']}")
    print(f"Package-version validation: {checks['package_version']['status']}")
    print(f"Selection-checksum validation: {checks['selection_checksum']['status']}")
    print(f"Frozen execution count: {policy['frozen_evaluation_execution_count']}")
    print(f"Frozen selections changed: {policy['frozen_selections_changed_count']}")
    print(f"Frozen comparators changed: {policy['frozen_comparators_changed_count']}")
    print(f"Leakage validation: {policy['leakage_validation_status']}")
    print(f"Reconciliation validation: {checks['frozen_reconciliation']['status']}")
    print(f"Absolute-path scan: {checks['absolute_path_scan']['status']}")
    print(f"Source-file mutation check: {checks['source_file_mutation_check']['status']}")
    print(f"Manifest path: {output.as_posix()}")
    print(f"Overall validation status: {manifest['overall_validation_status']}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path.cwd()
    if not (root / "pyproject.toml").is_file():
        raise EvidenceValidationError("Run from the repository root")
    manifest, _ = build_manifest(root)
    output = args.output
    if output.is_absolute():
        raise EvidenceValidationError("Manifest output must be repository-relative")
    write_manifest_atomic(output, manifest_bytes(manifest))
    _print_summary(manifest, output)
    return 0 if manifest["overall_validation_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
