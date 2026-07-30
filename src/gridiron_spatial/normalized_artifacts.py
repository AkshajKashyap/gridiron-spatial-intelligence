"""Atomic, partitioned persistence for normalized tracking releases."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .coordinate_frame import COORDINATE_TRANSFORM_VERSION
from .normalized_tracking import (
    NORMALIZED_ENTITY_FRAME_KEY,
    NORMALIZED_ENTITY_FRAME_SCHEMA,
    freeze_normalized_entity_frames,
)


NORMALIZED_ARTIFACT_FORMAT_VERSION = "1.0"
PROJECT_CLAIM_BOUNDARY = (
    "Target-centric receiver/defender tracking only; these partitions do not "
    "represent complete passing windows, quarterback target selection, "
    "official coverage assignments, or full-field defensive control."
)
_CLASSES = ("nominal", "extended_tolerance", "invalid")
_TRANSFORMS = ("identity_right", "rotate_180_left")
_WEEK_01_ROWS = {"input": 285_714, "output": 32_088, "combined": 317_802}


class NormalizedArtifactError(ValueError):
    """Raised when a normalized release cannot be validated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts(series: pd.Series, values: Sequence[str]) -> dict[str, int]:
    strings = series.astype("string")
    return {value: int(strings.eq(value).sum()) for value in values}


def _validate_reconciliation(reconciliation: Mapping[str, Any]) -> None:
    unsupported_classes = reconciliation[
        "unsupported_coordinate_class_values"
    ]
    failures = [
        reconciliation["status"] != "PASS",
        bool(reconciliation["missing_normalized_keys"]),
        bool(reconciliation["unexpected_normalized_keys"]),
        reconciliation["duplicate_raw_keys"] != 0,
        reconciliation["duplicate_normalized_keys"] != 0,
        any(reconciliation["raw_field_mismatch_counts"].values()),
        bool(reconciliation["unsupported_phase_values"]),
        any(unsupported_classes.values()),
        bool(reconciliation["unsupported_transform_versions"]),
        bool(reconciliation["unsupported_transform_applied_values"]),
        reconciliation["transform_direction_mismatches"] != 0,
        reconciliation["invalid_raw_to_nominal_normalized"] != 0,
        reconciliation["raw_rows"] != reconciliation["normalized_rows"],
    ]
    if any(failures):
        raise NormalizedArtifactError(
            "Normalized entity-frame reconciliation did not pass every gate"
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_normalized_tracking_release(
    output_dir: str | Path,
    requested_weeks: Sequence[str],
    build_week: Callable[[str], Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build, stage, validate, and atomically release weekly partitions."""

    destination = Path(output_dir)
    weeks = tuple(requested_weeks)
    if not weeks:
        raise NormalizedArtifactError("At least one explicit week is required")
    if len(set(weeks)) != len(weeks):
        raise NormalizedArtifactError("Requested weeks must be unique")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Destination exists: {destination}; pass overwrite=True"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    backup: Path | None = None
    partitions: list[dict[str, Any]] = []
    aggregate_rows = {"input": 0, "output": 0, "combined": 0}
    aggregate_classes = {
        "raw": {value: 0 for value in _CLASSES},
        "normalized": {value: 0 for value in _CLASSES},
    }
    aggregate_transforms = {value: 0 for value in _TRANSFORMS}
    game_keys: set[str] = set()
    play_keys: set[tuple[str, str]] = set()
    runtimes: dict[str, float] = {}
    processed: list[str] = []
    try:
        for week in weeks:
            built = build_week(week)
            frame = built["normalized_frame"]
            reconciliation = built["reconciliation"]
            if list(frame.columns) != NORMALIZED_ENTITY_FRAME_SCHEMA:
                raise NormalizedArtifactError(
                    f"{week} normalized schema is not frozen"
                )
            if frame.duplicated(NORMALIZED_ENTITY_FRAME_KEY).any():
                raise NormalizedArtifactError(
                    f"{week} contains duplicate normalized frozen keys"
                )
            _validate_reconciliation(reconciliation)
            row_counts = {
                "input": int(built["input_rows"]),
                "output": int(built["output_rows"]),
                "combined": int(len(frame)),
            }
            if row_counts["combined"] != (
                row_counts["input"] + row_counts["output"]
            ):
                raise NormalizedArtifactError(
                    f"{week} input/output rows do not equal combined rows"
                )
            if reconciliation["normalized_rows"] != row_counts["combined"]:
                raise NormalizedArtifactError(
                    f"{week} reconciliation row count does not match partition"
                )
            if week == "2023_w01" and row_counts != _WEEK_01_ROWS:
                raise NormalizedArtifactError(
                    f"Week 01 row mismatch: {row_counts} != {_WEEK_01_ROWS}"
                )

            raw_classes = _counts(
                frame["raw_coordinate_class"], _CLASSES
            )
            normalized_classes = _counts(
                frame["normalized_coordinate_class"], _CLASSES
            )
            transforms = _counts(
                frame["coordinate_transform_applied"], _TRANSFORMS
            )
            filename = f"normalized_{week}.parquet"
            path = staging / filename
            frame.to_parquet(path, index=False)
            restored_raw = pd.read_parquet(path)
            if list(restored_raw.columns) != NORMALIZED_ENTITY_FRAME_SCHEMA:
                raise NormalizedArtifactError(
                    f"{week} read-back schema or column order mismatch"
                )
            restored = freeze_normalized_entity_frames(restored_raw)
            if len(restored) != len(frame):
                raise NormalizedArtifactError(
                    f"{week} read-back row count mismatch"
                )
            if restored.duplicated(NORMALIZED_ENTITY_FRAME_KEY).any():
                raise NormalizedArtifactError(
                    f"{week} read-back contains duplicate frozen keys"
                )
            checksum = _sha256(path)
            partitions.append(
                {
                    "week": week,
                    "relative_filename": filename,
                    "input_rows": row_counts["input"],
                    "output_rows": row_counts["output"],
                    "combined_rows": row_counts["combined"],
                    "unique_games": int(frame["game_id"].nunique()),
                    "unique_plays": int(
                        frame[["game_id", "play_id"]]
                        .drop_duplicates()
                        .shape[0]
                    ),
                    "file_size_bytes": path.stat().st_size,
                    "sha256": checksum,
                    "reconciliation_status": reconciliation["status"],
                    "raw_coordinate_class_counts": raw_classes,
                    "normalized_coordinate_class_counts": normalized_classes,
                    "transform_applied_counts": transforms,
                    "extended_tolerance_count": raw_classes[
                        "extended_tolerance"
                    ],
                    "invalid_coordinate_count": raw_classes["invalid"],
                }
            )
            processed.append(week)
            for name in aggregate_rows:
                aggregate_rows[name] += row_counts[name]
            for coordinate_class in _CLASSES:
                aggregate_classes["raw"][coordinate_class] += raw_classes[
                    coordinate_class
                ]
                aggregate_classes["normalized"][
                    coordinate_class
                ] += normalized_classes[coordinate_class]
            for transform in _TRANSFORMS:
                aggregate_transforms[transform] += transforms[transform]
            game_keys.update(frame["game_id"].astype("string").astype(str))
            play_keys.update(
                zip(
                    frame["game_id"].astype("string").astype(str),
                    frame["play_id"].astype("string").astype(str),
                )
            )
            runtimes[week] = float(built["runtime_seconds"])
            del restored, restored_raw, frame, built

        if tuple(processed) != weeks:
            raise NormalizedArtifactError(
                "Processed weeks do not match requested weeks"
            )
        manifest = {
            "artifact_format_version": NORMALIZED_ARTIFACT_FORMAT_VERSION,
            "generation_timestamp_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "coordinate_transform_version": COORDINATE_TRANSFORM_VERSION,
            "frozen_schema": NORMALIZED_ENTITY_FRAME_SCHEMA,
            "frozen_key": NORMALIZED_ENTITY_FRAME_KEY,
            "requested_weeks": list(weeks),
            "processed_weeks": processed,
            "partitions": partitions,
            "aggregate": {
                **{f"{name}_rows": count for name, count in aggregate_rows.items()},
                "unique_games": len(game_keys),
                "unique_plays": len(play_keys),
                "coordinate_class_counts": aggregate_classes,
                "transform_applied_counts": aggregate_transforms,
                "runtime_seconds_by_week": runtimes,
            },
            "validation_status": "PASS",
            "project_claim_boundary": PROJECT_CLAIM_BOUNDARY,
        }
        manifest_path = staging / "manifest.json"
        _write_json(manifest_path, manifest)
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise NormalizedArtifactError("Manifest JSON validation failed")
        expected_files = {
            *(f"normalized_{week}.parquet" for week in weeks),
            "manifest.json",
        }
        if {path.name for path in staging.iterdir()} != expected_files:
            raise NormalizedArtifactError("Staged release file set is incorrect")

        if destination.exists():
            if not destination.is_dir():
                raise NormalizedArtifactError(
                    f"Existing destination is not a directory: {destination}"
                )
            backup = destination.parent / (
                f".{destination.name}.backup-{uuid.uuid4().hex}"
            )
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except Exception:
            if backup is not None and backup.exists():
                os.replace(backup, destination)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists():
            shutil.rmtree(backup)
