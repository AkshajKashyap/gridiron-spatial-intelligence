"""Aggregate-only, data-free portfolio summary and Markdown rendering."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import tomllib
from typing import Any, Mapping, Sequence


DEFAULT_RELEASE_VERSION = "0.1.0"
DEFAULT_MANIFEST = Path("artifacts/release/v0.1.0/evidence_manifest.json")
DEFAULT_OUTPUT = Path("reports/portfolio/release_0.1.0.md")
RESEARCH_QUESTION = (
    "How origin geometry for a competition-designated target and observed "
    "defensive entities relates to future target–defender separation dynamics."
)
EVIDENCE_PATHS = {
    "cohort": "artifacts/milestone_2/cohorts/cohort_summary.json",
    "cohort_manifest": "artifacts/milestone_2/cohorts/manifest.json",
    "normalized": "artifacts/milestone_2/normalized_tracking/manifest.json",
    "descriptive": "artifacts/milestone_3/full_season_separation_summary.json",
    "selection": "artifacts/milestone_4/baseline_selection.json",
    "frozen": "artifacts/milestone_4/frozen_test_result.json",
    "interpretation": "artifacts/milestone_5/model_interpretation_summary.json",
    "calibration": "artifacts/milestone_5/classifier_calibration_summary.json",
    "errors": "artifacts/milestone_5/validation_error_summary.json",
}
PIPELINE = [
    "Source and schema audit",
    "Deterministic analytic cohort and exclusion ledger",
    "Reversible coordinate normalization",
    "Target–defender pair construction",
    "Full-season descriptive separation analysis",
    "Chronological development/validation model selection",
    "One-time frozen evaluation",
    "Interpretation, calibration, and validation-error diagnostics",
    "Checksum-backed compact release evidence",
]
CLAIM_BOUNDARY = [
    "official coverage responsibility",
    "quarterback decision quality or target selection",
    "completion probability",
    "causal receiver or defender effects",
    "complete passing-window openness",
    "calibration on another season",
    "betting value",
    "production readiness",
]


class PortfolioDemoError(ValueError):
    """Raised when release evidence cannot support the portfolio demo."""


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise PortfolioDemoError(f"Missing {context}.{key}")
    return mapping[key]


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PortfolioDemoError(f"{context} must be a JSON object")
    return value


def _safe_relative_path(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise PortfolioDemoError(f"{context} must be a nonempty relative path")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or re.match(r"^[A-Za-z]:/", normalized):
        raise PortfolioDemoError(f"Unsafe absolute or escaping path in {context}")
    return normalized


def _read_json_object(path: Path, context: str) -> tuple[Mapping[str, Any], bytes]:
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise PortfolioDemoError(f"Missing {context}: {path.name}") from exc
    try:
        parsed = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PortfolioDemoError(f"Invalid UTF-8 JSON in {context}") from exc
    return _object(parsed, context), data


def _all_pair_evaluation(
    frozen: Mapping[str, Any],
    *,
    horizon: int,
    task: str,
) -> dict[str, Any]:
    evaluations = _require(frozen, "evaluations", "frozen result")
    if not isinstance(evaluations, list):
        raise PortfolioDemoError("frozen result.evaluations must be a list")
    matches = [
        item
        for item in evaluations
        if isinstance(item, dict)
        and item.get("population") == "all_pairs"
        and item.get("horizon") == horizon
        and item.get("task") == task
    ]
    if len(matches) != 1:
        raise PortfolioDemoError(
            f"Expected one all-pair {task} evaluation at H{horizon}"
        )
    item = matches[0]
    metric = "mae" if task == "regression" else "log_loss"
    selected = _object(_require(item, "selected_metrics", "evaluation"), "selected")
    comparator = _object(
        _require(item, "comparator_metrics", "evaluation"), "comparator"
    )
    differences = _object(
        _require(item, "metric_differences", "evaluation"), "differences"
    )
    return {
        "horizon": horizon,
        "metric": metric,
        "selected": _require(selected, metric, "selected metrics"),
        "comparator": _require(comparator, metric, "comparator metrics"),
        "difference": _require(differences, metric, "metric differences"),
    }


def _horizon_results(document: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    values = _require(document, key, "descriptive result")
    if not isinstance(values, list):
        raise PortfolioDemoError(f"descriptive result.{key} must be a list")
    indexed = {
        item.get("horizon"): item
        for item in values
        if isinstance(item, dict) and item.get("horizon") in {5, 10, 15}
    }
    if set(indexed) != {5, 10, 15}:
        raise PortfolioDemoError(f"Missing H5/H10/H15 values in {key}")
    return [
        {
            "horizon": horizon,
            "pair_count": _require(indexed[horizon], "pair_count", key),
            "mean_separation_change": _require(
                indexed[horizon], "mean_separation_change", key
            ),
        }
        for horizon in (5, 10, 15)
    ]


def build_portfolio_summary(
    repository_root: str | Path,
    *,
    release_version: str = DEFAULT_RELEASE_VERSION,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify compact evidence and return a deterministic JSON-compatible summary."""
    root = Path(repository_root).resolve()
    manifest_file = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST
    if not manifest_file.is_absolute():
        manifest_file = root / manifest_file
    manifest, _ = _read_json_object(manifest_file, "release manifest")

    if _require(manifest, "overall_validation_status", "manifest") != "PASS":
        raise PortfolioDemoError("Release manifest validation status is not PASS")
    if _require(manifest, "release_version", "manifest") != release_version:
        raise PortfolioDemoError("Release version mismatch")
    if _require(manifest, "evidence_file_count", "manifest") != 9:
        raise PortfolioDemoError("Evidence file count must equal 9")

    project = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))["project"]
    if project.get("version") != release_version:
        raise PortfolioDemoError("Package version mismatch")
    if manifest.get("distribution_name") != project.get("name"):
        raise PortfolioDemoError("Distribution name mismatch")

    entries = _require(manifest, "evidence_files", "manifest")
    if not isinstance(entries, list) or len(entries) != 9:
        raise PortfolioDemoError("Manifest must list exactly 9 evidence files")
    documents: dict[str, Mapping[str, Any]] = {}
    evidence_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_entry in enumerate(entries):
        entry = _object(raw_entry, f"evidence entry {index}")
        relative = _safe_relative_path(
            _require(entry, "relative_path", "evidence entry"), "evidence path"
        )
        if relative in seen:
            raise PortfolioDemoError(f"Duplicate evidence path: {relative}")
        seen.add(relative)
        document, data = _read_json_object(root / relative, relative)
        digest = hashlib.sha256(data).hexdigest()
        if digest != _require(entry, "sha256", relative):
            raise PortfolioDemoError(f"Checksum mismatch: {relative}")
        if len(data) != _require(entry, "byte_size", relative):
            raise PortfolioDemoError(f"Byte-size mismatch: {relative}")
        role = _require(entry, "semantic_role", relative)
        if not isinstance(role, str):
            raise PortfolioDemoError(f"Invalid semantic role: {relative}")
        documents[relative] = document
        evidence_rows.append(
            {
                "relative_path": relative,
                "semantic_role": role,
                "sha256": digest,
                "byte_size": len(data),
            }
        )

    if seen != set(EVIDENCE_PATHS.values()):
        raise PortfolioDemoError("Evidence paths do not match the release allowlist")

    cohort = documents[EVIDENCE_PATHS["cohort"]]
    cohort_core = _object(_require(cohort, "cohort_summary", "cohort"), "cohort")
    reporting = _object(_require(cohort, "reporting_summary", "cohort"), "reporting")
    source_plays = _object(
        _require(
            _object(
                _require(cohort_core, "aggregate_table_counts", "cohort"),
                "aggregate table counts",
            ),
            "source_plays",
            "aggregate table counts",
        ),
        "source plays",
    )
    normalized = _object(
        _require(
            documents[EVIDENCE_PATHS["normalized"]], "aggregate", "normalized manifest"
        ),
        "normalized aggregate",
    )
    descriptive = documents[EVIDENCE_PATHS["descriptive"]]
    aggregate = _object(
        _require(descriptive, "aggregate_counts", "descriptive result"),
        "descriptive aggregate",
    )
    frozen = documents[EVIDENCE_PATHS["frozen"]]
    decision = _object(
        _require(frozen, "decision_evidence", "frozen result"), "decision evidence"
    )
    leakage = _object(
        _require(frozen, "leakage_diagnostics", "frozen result"),
        "frozen leakage diagnostics",
    )
    frozen_policy = _object(
        _require(manifest, "frozen_evaluation_policy", "manifest"), "frozen policy"
    )
    if frozen_policy.get("frozen_evaluation_execution_count") != 1:
        raise PortfolioDemoError("Frozen evaluation execution count must equal 1")
    if frozen_policy.get("frozen_selections_changed_count") != 0:
        raise PortfolioDemoError("Frozen selections changed")
    if frozen_policy.get("frozen_comparators_changed_count") != 0:
        raise PortfolioDemoError("Frozen comparators changed")
    if frozen_policy.get("leakage_validation_status") != "PASS":
        raise PortfolioDemoError("Frozen leakage validation is not PASS")
    if (
        frozen_policy.get("reconciliation_status") != "PASS"
        or frozen_policy.get("reconciliation_mismatch_count") != 0
    ):
        raise PortfolioDemoError("Frozen reconciliation is not PASS")
    later_access = {
        name: _require(documents[EVIDENCE_PATHS[name]], "frozen_test_weeks_accessed", name)
        for name in ("interpretation", "calibration", "errors")
    }
    if any(value != 0 for value in later_access.values()):
        raise PortfolioDemoError("Later robustness evidence accessed frozen weeks")

    horizon_counts = _object(
        _require(aggregate, "horizon_pair_counts", "descriptive aggregate"),
        "horizon pair counts",
    )
    summary = {
        "release_identity": {
            "project_name": "Gridiron Spatial Intelligence",
            "distribution_name": project["name"],
            "version": release_version,
            "manifest_format_version": _require(
                manifest, "manifest_format_version", "manifest"
            ),
            "evidence_file_count": 9,
            "evidence_validation_status": "PASS",
        },
        "research_question": RESEARCH_QUESTION,
        "pipeline": list(PIPELINE),
        "scale": {
            "games": _require(reporting, "observed_game_count", "reporting"),
            "source_plays": _require(source_plays, "rows", "source plays"),
            "normalized_entity_frame_rows": _require(
                normalized, "combined_rows", "normalized aggregate"
            ),
            "origin_pairs": _require(aggregate, "origin_pair_count", "aggregate"),
            "horizon_pairs": _require(aggregate, "horizon_pair_count", "aggregate"),
            "horizon_pair_counts": {
                str(horizon): _require(
                    horizon_counts, str(horizon), "horizon pair counts"
                )
                for horizon in (5, 10, 15)
            },
        },
        "descriptive_results": {
            "all_pairs": _horizon_results(descriptive, "all_pair_results"),
            "nearest_observed_defender": _horizon_results(
                descriptive, "nearest_observed_defender_results"
            ),
        },
        "frozen_results": {
            "regression": [
                _all_pair_evaluation(frozen, horizon=horizon, task="regression")
                for horizon in (5, 10, 15)
            ],
            "classification": [
                _all_pair_evaluation(frozen, horizon=horizon, task="classification")
                for horizon in (5, 10, 15)
            ],
            "direction_agreement": _require(
                decision, "confirmed_direction_count", "decision evidence"
            ),
            "direction_reversals": _require(
                decision, "reversed_direction_count", "decision evidence"
            ),
            "execution_count": 1,
            "selections_changed": 0,
            "comparators_changed": 0,
        },
        "robustness": {
            "milestone_4_decision": (
                "GO — ORIGIN GEOMETRY HAS REPRODUCIBLE OUT-OF-SAMPLE "
                "PREDICTIVE SIGNAL"
            ),
            "milestone_5_decision": "LIMITED ROBUSTNESS — PROCEED WITH CAUTION",
        },
        "leakage_controls": {
            "development_weeks": "01–12",
            "validation_weeks": "13–15",
            "frozen_weeks": "16–18",
            "validation_rows_used_for_final_fitting": _require(
                leakage, "validation_rows_used_for_fitting", "leakage"
            ),
            "cross_split_play_overlap": _require(
                leakage, "cross_split_game_play_count", "leakage"
            ),
            "bootstrap_unit": _require(
                _object(
                    _require(frozen, "bootstrap_configuration", "frozen result"),
                    "bootstrap configuration",
                ),
                "unit",
                "bootstrap configuration",
            ),
            "later_frozen_weeks_accessed": later_access,
        },
        "release_evidence": {
            "files": evidence_rows,
            "aggregate_byte_count": _require(
                manifest, "aggregate_byte_count", "manifest"
            ),
            "validation_status": "PASS",
        },
        "claim_boundary": list(CLAIM_BOUNDARY),
    }
    return summary


def render_portfolio_markdown(summary: Mapping[str, Any]) -> str:
    """Render deterministic portfolio Markdown from a verified summary."""
    identity = summary["release_identity"]
    scale = summary["scale"]
    descriptive = summary["descriptive_results"]
    frozen = summary["frozen_results"]
    robustness = summary["robustness"]
    leakage = summary["leakage_controls"]
    evidence = summary["release_evidence"]
    lines = [
        "# Gridiron Spatial Intelligence — Data-Free Release Demo",
        "",
        "## 1. Release identity",
        "",
        f"- Project: **{identity['project_name']}**",
        f"- Distribution: `{identity['distribution_name']}`",
        f"- Version: `{identity['version']}`",
        f"- Manifest format: `{identity['manifest_format_version']}`",
        f"- Evidence files: `{identity['evidence_file_count']}`",
        f"- Evidence validation: **{identity['evidence_validation_status']}**",
        "",
        "## 2. Research question",
        "",
        f"> {summary['research_question']}",
        "",
        "The original passing-window concept was narrowed because the available "
        "data lack complete 22-player tracking, all receiving options, and "
        "direct ball-path information.",
        "",
        "## 3. Pipeline",
        "",
    ]
    lines.extend(
        f"{index}. {stage}" for index, stage in enumerate(summary["pipeline"], 1)
    )
    lines.extend(
        [
            "",
            "## 4. Scale",
            "",
            "| Quantity | Count |",
            "|---|---:|",
            f"| Games | {scale['games']:,} |",
            f"| Source plays | {scale['source_plays']:,} |",
            (
                "| Normalized entity-frame rows | "
                f"{scale['normalized_entity_frame_rows']:,} |"
            ),
            f"| Origin target–defender pairs | {scale['origin_pairs']:,} |",
            f"| Horizon-evaluable pairs | {scale['horizon_pairs']:,} |",
            f"| H5 pairs | {scale['horizon_pair_counts']['5']:,} |",
            f"| H10 pairs | {scale['horizon_pair_counts']['10']:,} |",
            f"| H15 pairs | {scale['horizon_pair_counts']['15']:,} |",
            "",
            "## 5. Descriptive separation",
            "",
            "| Horizon | All-pair mean change | Nearest-origin-defender mean change |",
            "|---:|---:|---:|",
        ]
    )
    nearest = {
        row["horizon"]: row
        for row in descriptive["nearest_observed_defender"]
    }
    for row in descriptive["all_pairs"]:
        near = nearest[row["horizon"]]
        lines.append(
            f"| H{row['horizon']} | {row['mean_separation_change']:.3f} | "
            f"{near['mean_separation_change']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Negative change means contraction; positive change means expansion. "
            "All-pair separation contracted most at H5, contracted less at H10, "
            "and expanded in the smaller H15 cohort. The nearest-origin-defender "
            "pattern was weaker and less consistent. These are aggregate patterns, "
            "not universal trajectories, and nearest does not mean official coverage.",
            "",
            "## 6. Frozen predictive result",
            "",
            "Differences are selected minus comparator; negative values favor the "
            "selected model.",
            "",
            "### Regression",
            "",
            "| Horizon | Selected MAE | Comparator MAE | Difference |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in frozen["regression"]:
        lines.append(
            f"| H{row['horizon']} | {row['selected']:.6f} | "
            f"{row['comparator']:.6f} | {row['difference']:.6f} |"
        )
    lines.extend(
        [
            "",
            "### Classification",
            "",
            "| Horizon | Selected log loss | Comparator log loss | Difference |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in frozen["classification"]:
        lines.append(
            f"| H{row['horizon']} | {row['selected']:.6f} | "
            f"{row['comparator']:.6f} | {row['difference']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- Validation-to-frozen direction agreement: "
            f"`{frozen['direction_agreement']}/12`",
            f"- Frozen reversals: `{frozen['direction_reversals']}`",
            f"- Frozen evaluator executions: `{frozen['execution_count']}`",
            f"- Selections changed: `{frozen['selections_changed']}`",
            f"- Comparators changed: `{frozen['comparators_changed']}`",
            "",
            "## 7. Robustness",
            "",
            f"- Milestone 4: **{robustness['milestone_4_decision']}**",
            f"- Milestone 5: **{robustness['milestone_5_decision']}**",
            "",
            "Separation-only was the weakest feature ablation; defender context "
            "mattered more at H10/H15, while absolute field location contributed "
            "little. Validation probabilities were somewhat too extreme, with H10 "
            "showing the best calibration. Performance was strongest below 15 yards. "
            "Long-separation buckets showed comparator reversals with limited support, "
            "and every H15 error bucket had limited support.",
            "",
            "## 8. Leakage controls",
            "",
            f"- Weeks {leakage['development_weeks']}: development fitting.",
            f"- Weeks {leakage['validation_weeks']}: validation selection and diagnostics.",
            f"- Weeks {leakage['frozen_weeks']}: evaluated exactly once.",
            (
                "- Validation rows used for final fitting: "
                f"`{leakage['validation_rows_used_for_final_fitting']}`."
            ),
            f"- Cross-split play overlap: `{leakage['cross_split_play_overlap']}`.",
            f"- Bootstrap unit: `{leakage['bootstrap_unit']}`.",
            "- The frozen result is preserved through exact-byte checksums.",
            "- Interpretation, calibration, and error diagnostics each accessed "
            "`0` frozen-test weeks.",
            "",
            "## 9. Release evidence",
            "",
            "| Relative path | Role | SHA-256 | Bytes |",
            "|---|---|---|---:|",
        ]
    )
    for entry in evidence["files"]:
        lines.append(
            f"| `{entry['relative_path']}` | {entry['semantic_role']} | "
            f"`{entry['sha256']}` | {entry['byte_size']:,} |"
        )
    lines.extend(
        [
            "",
            f"- Aggregate evidence bytes: `{evidence['aggregate_byte_count']:,}`",
            f"- Manifest validation: **{evidence['validation_status']}**",
            "",
            "## 10. Claim boundary",
            "",
            "This project does not establish:",
            "",
        ]
    )
    lines.extend(f"- {item};" for item in summary["claim_boundary"][:-1])
    lines.append(f"- {summary['claim_boundary'][-1]}.")
    lines.extend(
        [
            "",
            "## 11. How to verify",
            "",
            "```bash",
            "python -m pytest -q \\",
            "  tests/test_packaging.py \\",
            "  tests/test_release_evidence.py \\",
            "  tests/test_portfolio_demo.py",
            "",
            "python scripts/build_release_evidence_manifest.py \\",
            "  --output artifacts/release/v0.1.0/evidence_manifest.json",
            "",
            "python scripts/run_portfolio_demo.py --check",
            "```",
            "",
            "These commands require no NFL data.",
            "",
        ]
    )
    rendered = "\n".join(lines)
    if re.search(r"(?m)(?:^|[\s`(])(?:/[A-Za-z0-9_.-]|[A-Za-z]:[\\/])", rendered):
        raise PortfolioDemoError("Generated report contains an absolute local path")
    return rendered


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _resolve_from_root(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = args.repository_root.resolve()
    output = _resolve_from_root(root, args.output)
    try:
        summary = build_portfolio_summary(
            root,
            manifest_path=_resolve_from_root(root, args.manifest),
        )
        rendered = render_portfolio_markdown(summary).encode("utf-8")
        if args.check:
            try:
                existing = output.read_bytes()
            except FileNotFoundError as exc:
                raise PortfolioDemoError("Tracked portfolio report is missing") from exc
            if existing != rendered:
                raise PortfolioDemoError("Portfolio report drift detected")
        else:
            _write_atomic(output, rendered)
    except (OSError, PortfolioDemoError, KeyError, TypeError) as exc:
        print(f"Portfolio demo failed: {exc}", file=sys.stderr)
        return 1
    scale = summary["scale"]
    print(
        f"Portfolio demo PASS: v{summary['release_identity']['version']}, "
        f"{summary['release_identity']['evidence_file_count']} evidence files, "
        f"{scale['games']} games, {scale['horizon_pairs']} horizon pairs"
    )
    print(f"Report: {args.output.as_posix()}")
    print("Mode: check" if args.check else "Mode: atomic write")
    return 0
