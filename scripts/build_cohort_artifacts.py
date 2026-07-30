"""Build and atomically persist cohorts for an explicit release-week list."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gridiron_spatial.cohort import (  # noqa: E402
    COHORT_TABLE_NAMES,
    EXCLUSION_LEDGER_SCHEMA,
    INPUT_USECOLS,
    OUTPUT_USECOLS,
    TABLE_KEY_COLUMNS,
    _table_reconciliation,
    _validate_game_splits,
    build_exclusion_ledger,
    build_week_cohorts,
    summarize_cohort_reporting,
)
from gridiron_spatial.cohort_artifacts import (  # noqa: E402
    write_cohort_artifacts,
)
from scripts.smoke_two_week_cohort import (  # noqa: E402
    _counts,
    _read_tracking,
    _validate_tables,
    parse_week_args,
)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse paths and the existing strict explicit-week contract."""

    parser = argparse.ArgumentParser(
        description="Build validated target-centric cohort artifacts."
    )
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weeks", nargs="+", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parsed = parser.parse_args(args)
    parsed.weeks = parse_week_args(parsed.weeks)
    return parsed


def _load_supplementary(release_root: Path) -> pd.DataFrame:
    path = release_root / "supplementary_data.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing supplementary metadata: {path}")
    return pd.read_csv(
        path,
        usecols=["game_id", "play_id", "week"],
        dtype={"game_id": "string", "play_id": "string"},
        low_memory=False,
    )


def _load_week(
    release_root: Path,
    week: str,
    supplementary_source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    input_path = release_root / "train" / f"input_{week}.csv"
    output_path = release_root / "train" / f"output_{week}.csv"
    if not input_path.is_file() or not output_path.is_file():
        raise FileNotFoundError(
            f"Missing explicit weekly pair for {week}: "
            f"{input_path.name}, {output_path.name}"
        )
    inputs = _read_tracking(input_path, INPUT_USECOLS)
    outputs = _read_tracking(output_path, OUTPUT_USECOLS)
    play_keys = inputs[["game_id", "play_id"]].drop_duplicates()
    supplementary = supplementary_source.merge(
        play_keys,
        on=["game_id", "play_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(supplementary) != len(play_keys):
        raise RuntimeError(
            f"{week} supplementary coverage mismatch: "
            f"{len(supplementary)} != {len(play_keys)}"
        )
    return inputs, outputs, supplementary


def _require_reconciliation_pass(
    reconciliation: dict[str, Any],
    scope: str,
) -> None:
    status = reconciliation["overall"]["reconciliation_status"]
    if status != "PASS":
        raise RuntimeError(f"{scope} reconciliation failed: {status}")


def build_and_write(
    release_root: Path,
    output_dir: Path,
    weeks: tuple[str, ...],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run existing cohort components and call the writer exactly once."""

    supplementary_source = _load_supplementary(release_root)
    weekly: dict[str, dict[str, Any]] = {}
    source_totals = {"input": 0, "output": 0}
    for week in weeks:
        started = time.perf_counter()
        inputs, outputs, supplementary = _load_week(
            release_root, week, supplementary_source
        )
        source_totals["input"] += int(len(inputs))
        source_totals["output"] += int(len(outputs))
        cohorts = build_week_cohorts(inputs, outputs, supplementary, week)
        tables = {
            name: getattr(cohorts, name) for name in COHORT_TABLE_NAMES
        }
        _validate_tables(tables)
        ledger = build_exclusion_ledger(
            *(tables[name] for name in COHORT_TABLE_NAMES)
        )
        reconciliation = _table_reconciliation(tables, ledger)
        _require_reconciliation_pass(reconciliation, week)
        weekly[week] = {
            "tables": tables,
            "ledger": ledger,
            "runtime_seconds": round(time.perf_counter() - started, 3),
        }

    if tuple(weekly) != weeks:
        raise RuntimeError(
            f"Processed weeks differ from requested weeks: {tuple(weekly)}"
        )
    aggregate_tables = {
        name: pd.concat(
            [weekly[week]["tables"][name] for week in weeks],
            ignore_index=True,
        )
        for name in COHORT_TABLE_NAMES
    }
    duplicate_counts = _validate_tables(aggregate_tables)
    if any(duplicate_counts.values()):
        raise RuntimeError(f"Duplicate aggregate table keys: {duplicate_counts}")
    aggregate_ledger = pd.concat(
        [weekly[week]["ledger"] for week in weeks],
        ignore_index=True,
    )
    if list(aggregate_ledger.columns) != EXCLUSION_LEDGER_SCHEMA:
        raise RuntimeError("Aggregate exclusion-ledger schema mismatch")
    duplicate_ledger_ids = int(
        aggregate_ledger["ledger_id"].duplicated().sum()
    )
    if duplicate_ledger_ids:
        raise RuntimeError(
            f"Duplicate aggregate ledger IDs: {duplicate_ledger_ids}"
        )

    reconciliation = _table_reconciliation(
        aggregate_tables, aggregate_ledger
    )
    _require_reconciliation_pass(reconciliation, "aggregate")
    split_validation = _validate_game_splits(aggregate_tables)
    if split_validation["status"] != "PASS":
        raise RuntimeError(
            f"Aggregate split validation failed: "
            f"{split_validation['status']}"
        )
    reporting_summary = summarize_cohort_reporting(
        aggregate_tables, aggregate_ledger
    )
    cohort_summary = {
        "processed_weeks": list(weeks),
        "source_row_totals": source_totals,
        "weekly_runtimes": {
            week: weekly[week]["runtime_seconds"] for week in weeks
        },
        "aggregate_table_counts": {
            name: _counts(aggregate_tables[name])
            for name in COHORT_TABLE_NAMES
        },
        "aggregate_reconciliation_status": reconciliation["overall"][
            "reconciliation_status"
        ],
        "aggregate_split_validation_status": split_validation["status"],
    }
    return write_cohort_artifacts(
        output_dir,
        aggregate_tables,
        aggregate_ledger,
        cohort_summary,
        reporting_summary,
        overwrite=overwrite,
    )


def main(args: Sequence[str] | None = None) -> int:
    parsed = parse_args(args)
    try:
        build_and_write(
            parsed.release_root,
            parsed.output_dir,
            parsed.weeks,
            overwrite=parsed.overwrite,
        )
    except Exception as error:
        print(f"cohort artifact build failed: {error}", file=sys.stderr)
        return 1
    print(f"output_directory={parsed.output_dir}")
    print(f"manifest_path={parsed.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
