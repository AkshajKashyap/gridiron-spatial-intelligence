"""Deterministic, data-free verification for the portfolio release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tomllib
from typing import Any, Iterable, Mapping, Sequence

from .portfolio_demo import build_portfolio_summary, render_portfolio_markdown


DEFAULT_VERSION = "0.1.0"
MANIFEST_PATH = Path("artifacts/release/v0.1.0/evidence_manifest.json")
REPORT_PATH = Path("reports/portfolio/release_0.1.0.md")
WORKFLOW_PATH = Path(".github/workflows/ci.yml")
MAX_EVIDENCE_BYTES = 1024 * 1024
EVIDENCE_PATHS = (
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
REQUIRED_MARKDOWN = (
    "README.md",
    "docs/ARCHITECTURE.md",
    "docs/MODEL_CARD.md",
    "docs/EVALUATION_METHODOLOGY.md",
    "docs/REPRODUCIBILITY.md",
    "docs/RELEASE_CHECKLIST.md",
    "docs/INTERVIEW_NOTES.md",
)
REQUIRED_FILES = REQUIRED_MARKDOWN + (
    MANIFEST_PATH.as_posix(),
    REPORT_PATH.as_posix(),
    WORKFLOW_PATH.as_posix(),
    "tests/test_packaging.py",
    "tests/test_release_evidence.py",
    "tests/test_portfolio_demo.py",
    "tests/test_release_verification.py",
)
PROHIBITED_WORKFLOW_SCRIPTS = (
    "run_data_audit.py",
    "run_milestone_1_validation.py",
    "build_cohort_artifacts.py",
    "build_normalized_tracking.py",
    "analyze_full_season_separation.py",
    "select_baseline_models.py",
    "evaluate_frozen_baselines.py",
    "analyze_model_interpretation.py",
    "analyze_classifier_calibration.py",
    "analyze_validation_errors.py",
)


class ReleaseVerificationError(ValueError):
    """Raised for an invalid release input."""


def _normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _requirement_name(value: str) -> str:
    return _normalized_name(re.split(r"\s|[<>=!~;\[]", value, maxsplit=1)[0])


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for nested in value.values():
            yield from _walk(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk(nested)


def _contains_absolute_path(value: Any) -> bool:
    return any(
        isinstance(item, str)
        and (
            item.startswith(("/", "\\\\"))
            or bool(re.match(r"^[A-Za-z]:[\\/]", item))
        )
        for item in _walk(value)
    )


def _read_object(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise ReleaseVerificationError(f"missing file: {path.name}") from exc
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseVerificationError(f"invalid JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise ReleaseVerificationError(f"JSON root is not an object: {path.name}")
    return value, data


def _gate(diagnostics: Iterable[str], **values: Any) -> dict[str, Any]:
    ordered = sorted(set(diagnostics))
    return {"status": "PASS" if not ordered else "FAIL", **values, "diagnostics": ordered}


def _package_gate(root: Path, version: str) -> tuple[dict[str, Any], dict[str, Any]]:
    diagnostics: list[str] = []
    try:
        project_file = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))
        project = project_file["project"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        return _gate([f"pyproject metadata unavailable: {exc}"]), {}
    dependencies = {_requirement_name(item) for item in project.get("dependencies", [])}
    required = {"numpy", "pandas", "pyarrow", "scikit-learn"}
    missing = sorted(required - dependencies)
    if missing:
        diagnostics.append(f"missing runtime dependencies: {', '.join(missing)}")
    test_dependencies = {
        _requirement_name(item)
        for item in project.get("optional-dependencies", {}).get("test", [])
    }
    if "pytest" not in test_dependencies:
        diagnostics.append("test optional dependency does not include pytest")
    if project.get("version") != version:
        diagnostics.append(
            f"package version mismatch: {project.get('version')!r} != {version!r}"
        )
    if not project.get("name"):
        diagnostics.append("distribution name is missing")
    if not project.get("requires-python"):
        diagnostics.append("Python requirement is missing")
    discovery = (
        project_file.get("tool", {})
        .get("setuptools", {})
        .get("packages", {})
        .get("find", {})
        .get("where")
    )
    if discovery != ["src"]:
        diagnostics.append("src-layout package discovery is not configured")
    return (
        _gate(
            diagnostics,
            distribution_name=project.get("name"),
            version=project.get("version"),
            python_requirement=project.get("requires-python"),
            runtime_dependencies=sorted(dependencies),
            test_dependencies=sorted(test_dependencies),
            src_layout=discovery == ["src"],
        ),
        project,
    )


def _evidence_gate(
    root: Path,
    version: str,
    distribution_name: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    diagnostics: list[str] = []
    documents: dict[str, dict[str, Any]] = {}
    try:
        manifest, _ = _read_object(root / MANIFEST_PATH)
    except ReleaseVerificationError as exc:
        return _gate([str(exc)], evidence_file_count=0, checksum_status="FAIL"), {}, {}
    if manifest.get("manifest_format_version") != "release_evidence_manifest_v1":
        diagnostics.append("manifest format mismatch")
    if manifest.get("release_version") != version:
        diagnostics.append("manifest release version mismatch")
    if manifest.get("distribution_name") != distribution_name:
        diagnostics.append("manifest distribution name mismatch")
    if manifest.get("evidence_file_count") != 9:
        diagnostics.append("manifest evidence count is not 9")
    if manifest.get("overall_validation_status") != "PASS":
        diagnostics.append("manifest validation status is not PASS")
    entries = manifest.get("evidence_files")
    if not isinstance(entries, list) or len(entries) != 9:
        diagnostics.append("manifest does not list exactly 9 evidence entries")
        entries = entries if isinstance(entries, list) else []
    aggregate_bytes = 0
    observed_paths: list[str] = []
    checksum_failures = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            diagnostics.append(f"evidence entry {index} is not an object")
            continue
        relative = entry.get("relative_path")
        if not isinstance(relative, str):
            diagnostics.append(f"evidence entry {index} has no relative path")
            continue
        path = PurePosixPath(relative.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            diagnostics.append(f"unsafe evidence path: {relative}")
            continue
        observed_paths.append(relative)
        try:
            document, data = _read_object(root / relative)
        except ReleaseVerificationError as exc:
            diagnostics.append(f"{relative}: {exc}")
            continue
        documents[relative] = document
        aggregate_bytes += len(data)
        if len(data) > MAX_EVIDENCE_BYTES:
            diagnostics.append(f"evidence exceeds 1 MiB: {relative}")
        if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
            diagnostics.append(f"checksum mismatch: {relative}")
            checksum_failures += 1
        if len(data) != entry.get("byte_size"):
            diagnostics.append(f"byte-size mismatch: {relative}")
        if _contains_absolute_path(document):
            diagnostics.append(f"absolute local path in evidence: {relative}")
    if tuple(observed_paths) != EVIDENCE_PATHS:
        diagnostics.append("evidence paths or ordering differ from the allowlist")
    if aggregate_bytes != manifest.get("aggregate_byte_count"):
        diagnostics.append("aggregate evidence byte count mismatch")
    return (
        _gate(
            diagnostics,
            evidence_file_count=len(entries),
            checksum_status="PASS" if checksum_failures == 0 else "FAIL",
            aggregate_byte_count=aggregate_bytes,
        ),
        manifest,
        documents,
    )


def _frozen_gate(
    root: Path,
    version: str,
    manifest: Mapping[str, Any],
    documents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    diagnostics: list[str] = []
    policy = manifest.get("frozen_evaluation_policy")
    if not isinstance(policy, dict):
        return _gate(["frozen-evaluation policy is missing"])
    expected = {
        "frozen_evaluation_execution_count": 1,
        "frozen_selections_changed_count": 0,
        "frozen_comparators_changed_count": 0,
        "leakage_validation_status": "PASS",
        "reconciliation_status": "PASS",
        "reconciliation_mismatch_count": 0,
        "final_for_release": version,
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            diagnostics.append(f"{key} mismatch: {policy.get(key)!r}")
    selection_path = EVIDENCE_PATHS[4]
    frozen_path = EVIDENCE_PATHS[5]
    selection_file = root / selection_path
    frozen = documents.get(frozen_path, {})
    try:
        selection_checksum = hashlib.sha256(selection_file.read_bytes()).hexdigest()
    except FileNotFoundError:
        selection_checksum = None
    if frozen.get("selection_result_sha256") != selection_checksum:
        diagnostics.append("selection checksum cross-reference mismatch")
    leakage = frozen.get("leakage_diagnostics")
    if not isinstance(leakage, dict) or leakage.get("status") != "PASS":
        diagnostics.append("frozen evidence leakage status is not PASS")
    reconciliation = frozen.get("reconciliation_diagnostics")
    if (
        not isinstance(reconciliation, dict)
        or reconciliation.get("status") != "PASS"
        or reconciliation.get("mismatch_count") != 0
    ):
        diagnostics.append("frozen evidence reconciliation failed")
    return _gate(
        diagnostics,
        execution_count=policy.get("frozen_evaluation_execution_count"),
        selections_changed=policy.get("frozen_selections_changed_count"),
        comparators_changed=policy.get("frozen_comparators_changed_count"),
        leakage_status=policy.get("leakage_validation_status"),
        reconciliation_status=policy.get("reconciliation_status"),
        mismatch_count=policy.get("reconciliation_mismatch_count"),
        final_for_release=policy.get("final_for_release"),
        selection_checksum_status=(
            "PASS"
            if frozen.get("selection_result_sha256") == selection_checksum
            else "FAIL"
        ),
    )


def _tracked_files(root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ReleaseVerificationError("git tracked-file inventory failed")
    return sorted(
        item.decode("utf-8")
        for item in completed.stdout.split(b"\0")
        if item
    )


def _artifact_policy_gate(root: Path) -> dict[str, Any]:
    diagnostics: list[str] = []
    try:
        tracked = _tracked_files(root)
    except ReleaseVerificationError as exc:
        return _gate([str(exc)], tracked_artifact_count=0)
    tracked_artifacts = {path for path in tracked if path.startswith("artifacts/")}
    approved = set(EVIDENCE_PATHS) | {MANIFEST_PATH.as_posix(), "artifacts/.gitkeep"}
    unapproved = sorted(tracked_artifacts - approved)
    missing = sorted(
        (set(EVIDENCE_PATHS) | {MANIFEST_PATH.as_posix()}) - tracked_artifacts
    )
    if unapproved:
        diagnostics.append(f"unapproved tracked artifacts: {', '.join(unapproved)}")
    if missing:
        diagnostics.append(f"approved evidence is not tracked: {', '.join(missing)}")
    parquet = sorted(path for path in tracked if path.lower().endswith(".parquet"))
    if parquet:
        diagnostics.append(f"tracked Parquet files: {', '.join(parquet)}")
    raw = sorted(
        path
        for path in tracked
        if path.startswith("data/raw/") and not path.endswith("/.gitkeep")
    )
    if raw:
        diagnostics.append(f"tracked raw data: {', '.join(raw)}")
    forbidden_artifacts = sorted(
        path
        for path in tracked_artifacts
        if re.search(
            r"(prediction|pair[_-]?(level|rows|dataset)|staging|temporary|\\.tmp)",
            path,
            re.IGNORECASE,
        )
    )
    if forbidden_artifacts:
        diagnostics.append(
            f"tracked row-level or temporary artifacts: {', '.join(forbidden_artifacts)}"
        )
    return _gate(
        diagnostics,
        tracked_artifact_count=len(tracked_artifacts),
        approved_evidence_count=len(set(EVIDENCE_PATHS) & tracked_artifacts),
        parquet_count=len(parquet),
        raw_data_count=len(raw),
    )


def _strip_fenced_code(lines: Sequence[str]) -> list[tuple[int, str]]:
    visible: list[tuple[int, str]] = []
    fence: str | None = None
    for number, line in enumerate(lines, 1):
        stripped = line.lstrip()
        marker = "```" if stripped.startswith("```") else "~~~" if stripped.startswith("~~~") else None
        if marker:
            fence = None if fence == marker else marker if fence is None else fence
            continue
        if fence is None:
            visible.append((number, line))
    return visible


def _documentation_gate(root: Path) -> dict[str, Any]:
    broken: list[str] = []
    pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    for relative in REQUIRED_MARKDOWN:
        path = root / relative
        if not path.is_file():
            broken.append(f"{relative}: missing document")
            continue
        for line_number, line in _strip_fenced_code(path.read_text("utf-8").splitlines()):
            for raw_target in pattern.findall(line):
                target = raw_target.strip().strip("<>")
                if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                    continue
                target = target.split("#", 1)[0].split("?", 1)[0]
                if not (path.parent / target).resolve().is_file():
                    broken.append(f"{relative}:{line_number}: {raw_target}")
    return _gate(broken, broken_links=broken, checked_document_count=len(REQUIRED_MARKDOWN))


def _portfolio_gate(root: Path, version: str) -> dict[str, Any]:
    diagnostics: list[str] = []
    try:
        summary = build_portfolio_summary(root, release_version=version)
        rendered = render_portfolio_markdown(summary).encode("utf-8")
        tracked = (root / REPORT_PATH).read_bytes()
        if rendered != tracked:
            diagnostics.append("portfolio report drift")
        if summary["release_identity"]["evidence_file_count"] != 9:
            diagnostics.append("portfolio evidence count mismatch")
        if summary["release_identity"]["version"] != version:
            diagnostics.append("portfolio release version mismatch")
        if _contains_absolute_path(rendered.decode("utf-8")):
            diagnostics.append("absolute local path in portfolio report")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        diagnostics.append(f"portfolio verification failed: {exc}")
    return _gate(diagnostics, drift_status="PASS" if not diagnostics else "FAIL")


def _required_files_gate(root: Path) -> dict[str, Any]:
    missing = sorted(path for path in REQUIRED_FILES if not (root / path).is_file())
    return _gate(
        [f"missing required release file: {path}" for path in missing],
        missing_files=missing,
    )


def _workflow_gate(root: Path) -> dict[str, Any]:
    path = root / WORKFLOW_PATH
    if not path.is_file():
        return _gate(["CI workflow is missing"])
    text = path.read_text("utf-8")
    diagnostics: list[str] = []
    top_level = {
        match.group(1)
        for match in re.finditer(r"(?m)^([A-Za-z][A-Za-z_-]*):(?:\s|$)", text)
    }
    if not {"name", "on", "permissions", "jobs"} <= top_level:
        diagnostics.append("workflow top-level structure is incomplete")
    triggers: set[str] = set()
    workflow_lines = text.splitlines()
    if "on:" in workflow_lines:
        start = workflow_lines.index("on:") + 1
        for line in workflow_lines[start:]:
            if line and not line[0].isspace():
                break
            match = re.match(r"^  ([A-Za-z_]+):", line)
            if match:
                triggers.add(match.group(1))
    if not {"push", "pull_request", "workflow_dispatch"} <= triggers:
        diagnostics.append("workflow triggers are incomplete")
    if not re.search(r"(?ms)^permissions:\s*\n  contents:\s*read\s*$", text):
        diagnostics.append("least-privilege contents: read permission is missing")
    matrix = re.search(r"(?m)^\s+python-version:\s*\[([^\]]+)\]", text)
    versions = (
        {item.strip().strip("'\"") for item in matrix.group(1).split(",")}
        if matrix
        else set()
    )
    if versions != {"3.11", "3.13"}:
        diagnostics.append("Python matrix must contain exactly 3.11 and 3.13")
    required_commands = (
        'python -m pip install -e ".[test]"',
        "python -m pytest -q",
        "python -m compileall -q src scripts tests",
        "python -m pip check",
        "python -m build",
        "python scripts/build_release_evidence_manifest.py",
        "python scripts/run_portfolio_demo.py --check",
        "python scripts/verify_release.py",
        "git diff --exit-code",
        "git diff --check",
    )
    for command in required_commands:
        if command not in text:
            diagnostics.append(f"workflow command missing: {command}")
    uses = re.findall(r"(?m)^\s+-?\s*uses:\s*(\S+)", text)
    if "actions/checkout@v4" not in uses or "actions/setup-python@v5" not in uses:
        diagnostics.append("official checkout/setup-python actions are missing")
    if any(not item.startswith(("actions/checkout@", "actions/setup-python@")) for item in uses):
        diagnostics.append("workflow contains an unapproved action")
    prohibited = sorted(script for script in PROHIBITED_WORKFLOW_SCRIPTS if script in text)
    if prohibited:
        diagnostics.append(f"workflow invokes analytical scripts: {', '.join(prohibited)}")
    return _gate(
        diagnostics,
        triggers=sorted(triggers),
        python_versions=sorted(versions),
        prohibited_script_count=len(prohibited),
    )


def _configuration_gate(root: Path, workflow: Mapping[str, Any]) -> dict[str, Any]:
    required_tests = (
        "tests/test_packaging.py",
        "tests/test_release_evidence.py",
        "tests/test_portfolio_demo.py",
        "tests/test_release_verification.py",
    )
    missing = sorted(path for path in required_tests if not (root / path).is_file())
    diagnostics = [f"missing release test: {path}" for path in missing]
    if workflow.get("status") != "PASS":
        diagnostics.append("workflow policy is not PASS")
    return _gate(
        diagnostics,
        required_tests=list(required_tests),
        workflow_status=workflow.get("status"),
    )


def verify_release(
    repository_root: str | Path,
    *,
    release_version: str = DEFAULT_VERSION,
) -> dict[str, Any]:
    """Return a deterministic JSON-compatible release-gate result."""
    root = Path(repository_root).resolve()
    package, project = _package_gate(root, release_version)
    evidence, manifest, documents = _evidence_gate(
        root, release_version, project.get("name")
    )
    frozen = _frozen_gate(root, release_version, manifest, documents)
    artifact_policy = _artifact_policy_gate(root)
    documentation = _documentation_gate(root)
    portfolio = _portfolio_gate(root, release_version)
    required_files = _required_files_gate(root)
    workflow = _workflow_gate(root)
    configuration = _configuration_gate(root, workflow)
    gates = {
        "package_metadata": package,
        "evidence_manifest": evidence,
        "frozen_evaluation_safeguards": frozen,
        "tracked_artifact_policy": artifact_policy,
        "documentation_links": documentation,
        "portfolio_report": portfolio,
        "required_release_files": required_files,
        "workflow_policy": workflow,
        "test_and_workflow_configuration": configuration,
    }
    overall = "PASS" if all(gate["status"] == "PASS" for gate in gates.values()) else "FAIL"
    return {
        "overall_status": overall,
        "distribution_name": package.get("distribution_name"),
        "release_version": release_version,
        "gates": gates,
    }


def _print_human(result: Mapping[str, Any]) -> None:
    gates = result["gates"]
    evidence = gates["evidence_manifest"]
    frozen = gates["frozen_evaluation_safeguards"]
    print(f"Distribution: {result['distribution_name']}")
    print(f"Version: {result['release_version']}")
    print(f"Package metadata: {gates['package_metadata']['status']}")
    print(f"Evidence-file count: {evidence['evidence_file_count']}")
    print(f"Evidence checksums: {evidence['checksum_status']}")
    print(f"Frozen execution count: {frozen.get('execution_count')}")
    print(f"Frozen selections changed: {frozen.get('selections_changed')}")
    print(f"Frozen comparators changed: {frozen.get('comparators_changed')}")
    print(f"Leakage: {frozen.get('leakage_status')}")
    print(f"Reconciliation: {frozen.get('reconciliation_status')}")
    print(f"Tracked artifact policy: {gates['tracked_artifact_policy']['status']}")
    print(f"Documentation links: {gates['documentation_links']['status']}")
    print(f"Portfolio report drift: {gates['portfolio_report']['drift_status']}")
    print(f"CI workflow policy: {gates['workflow_policy']['status']}")
    print(f"Overall: {result['overall_status']}")
    for name, gate in gates.items():
        for diagnostic in gate["diagnostics"]:
            print(f"FAIL [{name}] {diagnostic}")


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--release-version", default=DEFAULT_VERSION)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = verify_release(
        args.repository_root,
        release_version=args.release_version,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_human(result)
    return 0 if result["overall_status"] == "PASS" else 1
