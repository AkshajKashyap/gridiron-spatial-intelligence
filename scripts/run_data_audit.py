#!/usr/bin/env python3
"""Run the read-only Milestone 1 source inventory from the repository root."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridiron_spatial.data_audit import AuditError, run_data_audit  # noqa: E402


def main() -> int:
    try:
        manifest, validation = run_data_audit(
            raw_directory=ROOT / "data" / "raw",
            artifacts_directory=ROOT / "artifacts",
            report_path=ROOT / "docs" / "data_audit.md",
        )
    except AuditError as error:
        print(f"data audit failed: {error}", file=sys.stderr)
        return 1
    print(f"Inventoried {manifest['file_count']} raw file(s). Overall status: {validation['overall_status']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
