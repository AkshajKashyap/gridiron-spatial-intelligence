"""Atomic persistence for already-built analytic cohort artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cohort import (
    COHORT_TABLE_NAMES,
    EXCLUSION_LEDGER_SCHEMA,
    TABLE_KEY_COLUMNS,
    TABLE_SCHEMAS,
    WeekCohorts,
)


ARTIFACT_FORMAT_VERSION = "1.0"
CLAIM_BOUNDARY = (
    "These outputs are target-centric and do not represent complete passing "
    "windows or full-field defensive control."
)
_SPLITS = ("development_train", "validation", "frozen_test")
_PARQUET_TABLES = (*COHORT_TABLE_NAMES, "exclusion_ledger")
_EXPECTED_FILES = {
    *(f"{name}.parquet" for name in _PARQUET_TABLES),
    "cohort_summary.json",
    "manifest.json",
}


class CohortArtifactError(ValueError):
    """Raised when supplied or staged cohort artifacts are invalid."""


def _cohort_tables(
    cohorts: Mapping[str, pd.DataFrame] | WeekCohorts,
) -> dict[str, pd.DataFrame]:
    if isinstance(cohorts, Mapping):
        names = set(cohorts)
        expected = set(COHORT_TABLE_NAMES)
        if names != expected:
            raise CohortArtifactError(
                "Expected exactly six cohort tables; "
                f"missing={sorted(expected - names)}, "
                f"unexpected={sorted(names - expected)}"
            )
        return {name: cohorts[name] for name in COHORT_TABLE_NAMES}
    if isinstance(cohorts, WeekCohorts):
        return {
            name: getattr(cohorts, name) for name in COHORT_TABLE_NAMES
        }
    raise CohortArtifactError("cohorts must be a mapping or WeekCohorts")


def _validate_frame(
    name: str,
    frame: pd.DataFrame,
    columns: list[str],
    keys: list[str],
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise CohortArtifactError(f"{name} is not a DataFrame")
    if list(frame.columns) != columns:
        raise CohortArtifactError(f"{name} does not match its frozen schema")
    if frame.duplicated(keys).any():
        raise CohortArtifactError(f"{name} contains duplicate frozen keys")
    if frame[keys].isna().any(axis=None):
        raise CohortArtifactError(f"{name} contains null frozen keys")


def _summary_value(
    summary: Mapping[str, Any],
    *names: str,
) -> Any:
    for name in names:
        if name in summary:
            return summary[name]
    raise CohortArtifactError(f"Missing summary field: {' or '.join(names)}")


def _source_row_totals(summary: Mapping[str, Any]) -> dict[str, int]:
    if "source_row_totals" in summary:
        totals = summary["source_row_totals"]
        return {
            "input": int(totals["input"]),
            "output": int(totals["output"]),
        }
    weekly = _summary_value(summary, "weekly")
    return {
        phase: sum(
            int(details["raw_rows"][phase]) for details in weekly.values()
        )
        for phase in ("input", "output")
    }


def _split_counts(frame: pd.DataFrame) -> dict[str, int]:
    if "split" not in frame:
        return {}
    values = frame["split"].astype("string")
    unknown = sorted(set(values.dropna()) - set(_SPLITS))
    if values.isna().any() or unknown:
        raise CohortArtifactError(
            f"Invalid split values: null={bool(values.isna().any())}, "
            f"unknown={unknown}"
        )
    return {split: int(values.eq(split).sum()) for split in _SPLITS}


def _validate_summaries(
    tables: Mapping[str, pd.DataFrame],
    cohort_summary: Mapping[str, Any],
    reporting_summary: Mapping[str, Any],
) -> dict[str, Any]:
    reconciliation = _summary_value(
        cohort_summary,
        "aggregate_reconciliation_status",
        "reconciliation_status",
    )
    split_status = _summary_value(
        cohort_summary,
        "aggregate_split_validation_status",
        "split_validation_status",
    )
    if reconciliation != "PASS" or split_status != "PASS":
        raise CohortArtifactError(
            "Reconciliation and split-validation statuses must both be PASS"
        )

    supplied_counts = _summary_value(
        cohort_summary, "aggregate_table_counts", "table_counts"
    )
    for name in COHORT_TABLE_NAMES:
        frame = tables[name]
        eligible = int(frame["eligible"].sum())
        actual = {
            "rows": int(len(frame)),
            "eligible": eligible,
            "excluded": int(len(frame) - eligible),
        }
        if supplied_counts.get(name) != actual:
            raise CohortArtifactError(
                f"Summary counts do not match {name}: "
                f"supplied={supplied_counts.get(name)}, actual={actual}"
            )

    observed_games = int(
        tables["source_plays"]["game_id"].nunique(dropna=True)
    )
    if int(_summary_value(reporting_summary, "observed_game_count")) != observed_games:
        raise CohortArtifactError("Observed-game count does not match source_plays")

    supplied_splits = _summary_value(reporting_summary, "counts_by_split")
    for split in _SPLITS:
        for name in COHORT_TABLE_NAMES:
            frame = tables[name]
            selected = frame.loc[frame["split"].astype("string").eq(split)]
            eligible = int(selected["eligible"].sum())
            actual = {
                "rows": int(len(selected)),
                "eligible": eligible,
                "excluded": int(len(selected) - eligible),
            }
            if supplied_splits.get(split, {}).get(name) != actual:
                raise CohortArtifactError(
                    f"Reporting split counts do not match {split}/{name}"
                )

    weeks = list(
        _summary_value(cohort_summary, "processed_weeks", "weeks")
    )
    return {
        "aggregate_reconciliation_status": reconciliation,
        "split_validation_status": split_status,
        "observed_game_count": observed_games,
        "processed_weeks": weeks,
        "source_row_totals": _source_row_totals(cohort_summary),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _reject_absolute_paths(value: Any) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_absolute_paths(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_absolute_paths(item)
    elif isinstance(value, (str, Path)) and Path(str(value)).is_absolute():
        raise CohortArtifactError("Summary JSON must not contain absolute paths")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _json_ready(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_parquet(
    path: Path,
    source: pd.DataFrame,
) -> None:
    restored = pd.read_parquet(path)
    if len(restored) != len(source):
        raise CohortArtifactError(f"Parquet row-count mismatch: {path.name}")
    if list(restored.columns) != list(source.columns):
        raise CohortArtifactError(f"Parquet column mismatch: {path.name}")
    for column in source.columns:
        if source[column].dtype == object:
            continue
        if str(restored[column].dtype) != str(source[column].dtype):
            raise CohortArtifactError(
                f"Parquet dtype mismatch in {path.name}/{column}: "
                f"{source[column].dtype} != {restored[column].dtype}"
            )


def write_cohort_artifacts(
    output_dir: str | Path,
    cohorts: Mapping[str, pd.DataFrame] | WeekCohorts,
    exclusion_ledger: pd.DataFrame,
    cohort_summary: Mapping[str, Any],
    reporting_summary: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate and atomically write the nine approved cohort artifacts."""

    destination = Path(output_dir)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {destination}; "
            "pass overwrite=True to replace it"
        )
    tables = _cohort_tables(cohorts)
    for name in COHORT_TABLE_NAMES:
        _validate_frame(
            name,
            tables[name],
            TABLE_SCHEMAS[name],
            TABLE_KEY_COLUMNS[name],
        )
    _validate_frame(
        "exclusion_ledger",
        exclusion_ledger,
        EXCLUSION_LEDGER_SCHEMA,
        ["ledger_id"],
    )
    summary_fields = _validate_summaries(
        tables, cohort_summary, reporting_summary
    )
    summary_payload = {
        "cohort_summary": _json_ready(cohort_summary),
        "reporting_summary": _json_ready(reporting_summary),
    }
    _reject_absolute_paths(summary_payload)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    backup: Path | None = None
    try:
        written_frames = {
            **tables,
            "exclusion_ledger": exclusion_ledger,
        }
        for name in _PARQUET_TABLES:
            written_frames[name].to_parquet(
                temporary / f"{name}.parquet", index=False
            )

        summary_path = temporary / "cohort_summary.json"
        _write_json(summary_path, summary_payload)
        json.loads(summary_path.read_text(encoding="utf-8"))

        table_manifest = []
        for name in _PARQUET_TABLES:
            path = temporary / f"{name}.parquet"
            source = written_frames[name]
            _validate_parquet(path, source)
            eligible = (
                int(source["eligible"].sum())
                if "eligible" in source
                else None
            )
            entry = {
                "table_filename": path.name,
                "table_name": name,
                "row_count": int(len(source)),
                "column_count": int(len(source.columns)),
                "ordered_columns": list(source.columns),
                "ordered_key_columns": (
                    TABLE_KEY_COLUMNS[name]
                    if name in TABLE_KEY_COLUMNS
                    else ["ledger_id"]
                ),
                "file_size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "eligible_count": eligible,
                "excluded_count": (
                    int(len(source) - eligible)
                    if eligible is not None
                    else None
                ),
                "split_counts": _split_counts(source),
                "relative_path": path.name,
            }
            table_manifest.append(entry)

        manifest = {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "generation_timestamp_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "tables": table_manifest,
            "cohort_summary_file": {
                "filename": summary_path.name,
                "relative_path": summary_path.name,
                "file_size_bytes": summary_path.stat().st_size,
                "sha256": _sha256(summary_path),
            },
            **summary_fields,
            "project_claim_boundary": CLAIM_BOUNDARY,
        }
        manifest_path = temporary / "manifest.json"
        _write_json(manifest_path, manifest)
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise CohortArtifactError("Manifest JSON validation failed")
        if {path.name for path in temporary.iterdir()} != _EXPECTED_FILES:
            raise CohortArtifactError("Staged artifact file set is incorrect")
        for entry in table_manifest:
            path = temporary / entry["relative_path"]
            if _sha256(path) != entry["sha256"]:
                raise CohortArtifactError(
                    f"Checksum validation failed: {path.name}"
                )

        if destination.exists():
            if not destination.is_dir():
                raise CohortArtifactError(
                    f"Existing destination is not a directory: {destination}"
                )
            backup = destination.parent / (
                f".{destination.name}.backup-{uuid.uuid4().hex}"
            )
            os.replace(destination, backup)
        try:
            os.replace(temporary, destination)
        except Exception:
            if backup is not None and backup.exists():
                os.replace(backup, destination)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup is not None and backup.exists():
            shutil.rmtree(backup)
