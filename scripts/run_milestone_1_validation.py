#!/usr/bin/env python3
"""Run dataset-specific Milestone 1 validation for the extracted BDB release."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridiron_spatial.data_audit import AuditError  # noqa: E402
from gridiron_spatial.milestone_1_validation import run_milestone_1_validation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Milestone 1 validation on explicit weekly files.")
    parser.add_argument("--week", action="append", required=True, help="Week label exactly matching a filename, e.g. 2023_w01. May be repeated.")
    args = parser.parse_args()
    dataset_root = ROOT / "data" / "raw" / "bdb_2026" / "114239_nfl_competition_files_published_analytics_final"
    selected_weeks = tuple(args.week)
    run_label = "_".join(selected_weeks)
    artifact_root = ROOT / "artifacts" / f"milestone_1_benchmark_{run_label}"
    try:
        result = run_milestone_1_validation(
            dataset_root,
            artifact_root,
            None,
            None,
            weeks=selected_weeks,
            progress=print,
        )
    except AuditError as error:
        print(f"Milestone 1 validation failed: {error}", file=sys.stderr)
        return 1
    print(f"Milestone 1 benchmark decision: {result['decision']['status']}; artifacts: {artifact_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
