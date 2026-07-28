"""Deterministic analytic-cohort construction for Milestone 2, Task 1.

This module establishes eligibility only.  It deliberately does not normalize
coordinates, calculate geometry, create prediction features, or fit models.
Raw input and output frame identifiers remain in separate phase namespaces.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from .data_audit import AuditError
from .milestone_1_validation import (
    INPUT_COLUMNS,
    OUTPUT_COLUMNS,
    SUPPLEMENTARY_REQUIRED_COLUMNS,
)


EXPECTED_WEEKS = tuple(f"2023_w{week:02d}" for week in range(1, 19))
HORIZONS = (5, 10, 15)
REASON_CODES = tuple(f"C{number:02d}" for number in range(1, 15))

PLAY_KEYS = ["game_id", "play_id"]
ENTITY_KEYS = [*PLAY_KEYS, "nfl_id"]
FRAME_KEYS = [*PLAY_KEYS, "frame_id"]
ENTITY_FRAME_KEYS = [*PLAY_KEYS, "phase", "frame_id", "nfl_id"]

INPUT_USECOLS = [
    "game_id",
    "play_id",
    "player_to_predict",
    "nfl_id",
    "frame_id",
    "play_direction",
    "player_side",
    "player_role",
    "x",
    "y",
    "num_frames_output",
]
OUTPUT_USECOLS = list(OUTPUT_COLUMNS)

FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.3
BOUNDARY_TOLERANCE = 1.0

TARGET_ROLE = "Targeted Receiver"
DEFENDER_ROLE = "Defensive Coverage"
DEFENSE_SIDE = "Defense"

COHORT_TABLE_NAMES = (
    "source_plays",
    "descriptive_target_frames",
    "primary_origins",
    "trajectory_eligibility",
    "future_separation_eligibility",
    "pair_exclusions",
)
_STATUS_COLUMNS = [
    *REASON_CODES,
    "primary_exclusion_reason",
    "secondary_exclusion_reasons",
    "eligible",
]
TABLE_SCHEMAS: dict[str, list[str]] = {
    "source_plays": [
        *PLAY_KEYS,
        "phase",
        "week",
        "week_number",
        "split",
        "input_rows",
        "valid_input_frame_count",
        "tracked_entity_count",
        "origin_frame",
        "target_nfl_id",
        "target_nfl_id_count",
        "target_rows",
        "play_direction",
        "direction_value_count",
        "invalid_direction_rows",
        "invalid_entity_frame_rows",
        "ambiguous_entity_assignments",
        "metadata_week_matches_file",
        *_STATUS_COLUMNS,
    ],
    "descriptive_target_frames": [
        *PLAY_KEYS,
        "phase",
        "frame_id",
        "target_nfl_id",
        "week",
        "week_number",
        "split",
        "frame_raw_rows",
        "target_row_count",
        "target_entity_count",
        "valid_target_coordinate_rows",
        "registered_defender_rows",
        "observed_defender_count",
        "valid_observed_defender_count",
        "target_duplicate_rows",
        "defender_duplicate_rows",
        "target_coordinate_missing_rows",
        "defender_coordinate_missing_rows",
        "target_outside_extended_rows",
        "defender_outside_extended_rows",
        "target_extended_tolerance_rows",
        "defender_extended_tolerance_rows",
        "target_invalid_key_rows",
        "defender_invalid_key_rows",
        *_STATUS_COLUMNS,
    ],
    "primary_origins": [
        *PLAY_KEYS,
        "target_nfl_id",
        "origin_kind",
        "phase",
        "origin_frame",
        "relative_frame",
        "play_direction",
        "week",
        "week_number",
        "split",
        "raw_history_2_frame_count",
        "valid_history_2_frame_count",
        "raw_history_5_frame_count",
        "valid_history_5_frame_count",
        "history_2_eligible",
        "history_5_eligible",
        "five_frame_history_eligible",
        "registered_at_origin",
        "valid_origin_rows",
        "registered_origin_defender_count",
        "valid_observed_origin_defender_count",
        "observed_origin_defender_ids",
        "duplicate_at_origin",
        "origin_coordinate_missing",
        "origin_outside_extended",
        "origin_extended_tolerance",
        *_STATUS_COLUMNS,
    ],
    "trajectory_eligibility": [
        *ENTITY_KEYS,
        "target_nfl_id",
        "origin_frame",
        "horizon",
        "phase",
        "label_phase",
        "max_input_relative_frame_used",
        "label_start_frame",
        "label_end_frame",
        "week",
        "week_number",
        "split",
        "player_role",
        "player_side",
        "is_target_role",
        "is_defender",
        "player_to_predict",
        "declared_output_frames",
        "matched_output_group",
        "full_output_group_exact",
        "raw_history_2_frame_count",
        "valid_history_2_frame_count",
        "history_2_eligible",
        "history_5_eligible",
        "observed_at_origin",
        "horizon_output_row_count",
        "horizon_output_frame_count",
        "horizon_frames_contiguous",
        "horizon_coordinate_missing_rows",
        "horizon_outside_extended_rows",
        "horizon_extended_tolerance_rows",
        *_STATUS_COLUMNS,
    ],
    "future_separation_eligibility": [
        *PLAY_KEYS,
        "target_nfl_id",
        "origin_frame",
        "horizon",
        "phase",
        "label_phase",
        "max_input_relative_frame_used",
        "label_start_frame",
        "label_end_frame",
        "week",
        "week_number",
        "split",
        "history_2_eligible",
        "five_frame_history_eligible",
        "observed_origin_defender_ids",
        "observed_origin_defender_count",
        "prediction_candidate_origin_defender_count",
        "two_frame_history_origin_defender_count",
        "output_candidate_defender_count",
        "evaluable_defender_count",
        "evaluable_defender_ids",
        "removed_defender_count",
        "target_trajectory_eligible",
        "defender_set_definition",
        *_STATUS_COLUMNS,
    ],
    "pair_exclusions": [
        *PLAY_KEYS,
        "phase",
        "frame_id",
        "target_nfl_id",
        "defender_nfl_id",
        "week",
        "week_number",
        "split",
        "coordinate_category",
        "coordinate_missing",
        "outside_nominal",
        "outside_extended_tolerance",
        *_STATUS_COLUMNS,
    ],
}
TABLE_KEY_COLUMNS: dict[str, list[str]] = {
    "source_plays": [*PLAY_KEYS],
    "descriptive_target_frames": [*PLAY_KEYS, "phase", "frame_id", "target_nfl_id"],
    "primary_origins": [*PLAY_KEYS, "target_nfl_id", "origin_frame"],
    "trajectory_eligibility": [
        *PLAY_KEYS,
        "nfl_id",
        "origin_frame",
        "horizon",
    ],
    "future_separation_eligibility": [
        *PLAY_KEYS,
        "target_nfl_id",
        "origin_frame",
        "horizon",
    ],
    "pair_exclusions": [
        *PLAY_KEYS,
        "phase",
        "frame_id",
        "target_nfl_id",
        "defender_nfl_id",
    ],
}
EXCLUSION_LEDGER_SCHEMA = [
    "ledger_id",
    "source_table",
    "unit_type",
    "exclusion_level",
    "game_id",
    "play_id",
    "phase",
    "frame_id",
    "nfl_id",
    "target_nfl_id",
    "defender_nfl_id",
    "origin_frame",
    "horizon",
    "week",
    "week_number",
    "split",
    *REASON_CODES,
    "primary_exclusion_reason",
    "secondary_exclusion_reasons",
]


@dataclass(frozen=True)
class ReleaseFiles:
    """Preflighted paths for the extracted release."""

    dataset_root: Path
    supplementary: Path
    inputs: Mapping[str, Path]
    outputs: Mapping[str, Path]
    archive_paths_excluded: tuple[Path, ...]


@dataclass
class WeekCohorts:
    """Compact cohort tables and audit statistics for one week."""

    source_plays: pd.DataFrame
    descriptive_target_frames: pd.DataFrame
    primary_origins: pd.DataFrame
    trajectory_eligibility: pd.DataFrame
    future_separation_eligibility: pd.DataFrame
    pair_exclusions: pd.DataFrame
    statistics: dict[str, Any]
    stage_seconds: dict[str, float]


def _read_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration as error:
            raise AuditError(f"Empty CSV: {path}") from error


def split_for_week(week_label: str) -> str:
    """Return the binding chronological split for an exact weekly label."""

    if week_label not in EXPECTED_WEEKS:
        raise ValueError(f"Unexpected week label: {week_label!r}")
    week_number = int(week_label[-2:])
    if week_number <= 12:
        return "development_train"
    if week_number <= 15:
        return "validation"
    return "frozen_test"


def discover_release(dataset_root: Path) -> ReleaseFiles:
    """Discover and schema-check exactly 18 weekly CSV pairs.

    Discovery is restricted to extracted CSV tables.  Archive files are
    recorded as excluded packaging artifacts and are never offered to pandas.
    """

    dataset_root = dataset_root.resolve()
    train_directory = dataset_root / "train"
    supplementary = dataset_root / "supplementary_data.csv"
    if not train_directory.is_dir() or not supplementary.is_file():
        raise AuditError(
            f"Expected train/ and supplementary_data.csv under {dataset_root}."
        )

    input_paths = sorted(train_directory.glob("input_*.csv"))
    output_paths = sorted(train_directory.glob("output_*.csv"))

    def index_paths(paths: Iterable[Path], prefix: str) -> dict[str, Path]:
        indexed: dict[str, Path] = {}
        for path in paths:
            expected_prefix = f"{prefix}_"
            if not path.stem.startswith(expected_prefix):
                raise AuditError(f"Unexpected weekly filename: {path.name}")
            label = path.stem.removeprefix(expected_prefix)
            if label not in EXPECTED_WEEKS:
                raise AuditError(f"Unexpected weekly label in {path.name}: {label}")
            if label in indexed:
                raise AuditError(
                    f"Duplicate {prefix} file for {label}: "
                    f"{indexed[label].name}, {path.name}"
                )
            indexed[label] = path
        return indexed

    inputs = index_paths(input_paths, "input")
    outputs = index_paths(output_paths, "output")
    expected = set(EXPECTED_WEEKS)
    missing_inputs = sorted(expected - set(inputs))
    missing_outputs = sorted(expected - set(outputs))
    if missing_inputs or missing_outputs:
        raise AuditError(
            "Incomplete weekly release: "
            f"missing inputs={missing_inputs}, missing outputs={missing_outputs}"
        )
    if set(inputs) != set(outputs):
        raise AuditError("Input and output weekly file sets do not match.")

    for label in EXPECTED_WEEKS:
        input_header = _read_header(inputs[label])
        output_header = _read_header(outputs[label])
        if input_header != INPUT_COLUMNS:
            raise AuditError(
                f"Input schema mismatch for {inputs[label].name}: {input_header}"
            )
        if output_header != OUTPUT_COLUMNS:
            raise AuditError(
                f"Output schema mismatch for {outputs[label].name}: {output_header}"
            )
    supplementary_header = _read_header(supplementary)
    missing_metadata = sorted(
        set(SUPPLEMENTARY_REQUIRED_COLUMNS) - set(supplementary_header)
    )
    if missing_metadata:
        raise AuditError(
            f"Supplementary table is missing required columns: {missing_metadata}"
        )

    archive_paths = tuple(sorted(dataset_root.parent.glob("*.zip")))
    return ReleaseFiles(
        dataset_root=dataset_root,
        supplementary=supplementary,
        inputs=inputs,
        outputs=outputs,
        archive_paths_excluded=archive_paths,
    )


def _canonical_positive_integer(
    values: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return canonical string, numeric value, and validity for an ID/index."""

    numeric = pd.to_numeric(values, errors="coerce")
    numeric_float = numeric.astype("float64")
    finite = pd.Series(
        np.isfinite(numeric_float.to_numpy()),
        index=values.index,
        dtype=bool,
    )
    valid = (
        numeric.notna()
        & finite
        & numeric_float.gt(0)
        & numeric_float.eq(np.floor(numeric_float))
    )
    canonical = pd.Series(pd.NA, index=values.index, dtype="string")
    canonical.loc[valid] = (
        numeric_float.loc[valid].astype("int64").astype("string").to_numpy()
    )
    integral = numeric_float.where(valid).astype("Int64")
    return canonical, integral, valid.astype(bool)


def classify_coordinates(
    x_values: pd.Series,
    y_values: pd.Series,
) -> pd.DataFrame:
    """Classify raw coordinate pairs without clipping or transformation."""

    x = pd.to_numeric(x_values, errors="coerce").astype("float64")
    y = pd.to_numeric(y_values, errors="coerce").astype("float64")
    finite = pd.Series(
        np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy()),
        index=x.index,
        dtype=bool,
    )
    nominal = (
        finite
        & x.between(0.0, FIELD_LENGTH, inclusive="both")
        & y.between(0.0, FIELD_WIDTH, inclusive="both")
    )
    within_tolerance = (
        finite
        & x.between(
            -BOUNDARY_TOLERANCE,
            FIELD_LENGTH + BOUNDARY_TOLERANCE,
            inclusive="both",
        )
        & y.between(
            -BOUNDARY_TOLERANCE,
            FIELD_WIDTH + BOUNDARY_TOLERANCE,
            inclusive="both",
        )
    )
    category = pd.Series("invalid", index=x.index, dtype="string")
    category.loc[within_tolerance] = "extended-tolerance"
    category.loc[nominal] = "nominal"
    return pd.DataFrame(
        {
            "x_raw": x,
            "y_raw": y,
            "coordinate_category": category,
            "coordinate_valid": within_tolerance.astype(bool),
            "coordinate_missing": (~finite).astype(bool),
            "outside_nominal": (finite & ~nominal).astype(bool),
            "outside_extended_tolerance": (
                finite & ~within_tolerance
            ).astype(bool),
        },
        index=x.index,
    )


def phase_qualified_entity_frame_keys(
    frame: pd.DataFrame,
    phase: str,
) -> pd.DataFrame:
    """Create phase-aware entity-frame keys for tests and audit assertions."""

    if phase not in {"input", "output"}:
        raise ValueError("phase must be 'input' or 'output'")
    missing = sorted(set([*PLAY_KEYS, "frame_id", "nfl_id"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing key columns: {missing}")
    keys = frame[[*PLAY_KEYS, "frame_id", "nfl_id"]].copy()
    keys.insert(2, "phase", phase)
    return keys[ENTITY_FRAME_KEYS]


def _ensure_reason_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for code in REASON_CODES:
        if code not in frame:
            frame[code] = False
        frame[code] = frame[code].fillna(False).astype(bool)
    return frame


def assign_exclusion_reasons(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign deterministic primary/secondary reasons in C01..C14 order."""

    result = _ensure_reason_columns(frame.copy())
    flags = result[list(REASON_CODES)].to_numpy(dtype=bool, copy=False)
    any_reason = flags.any(axis=1)
    first_positions = flags.argmax(axis=1)
    codes = np.asarray(REASON_CODES, dtype=object)
    primary = np.full(len(result), "", dtype=object)
    primary[any_reason] = codes[first_positions[any_reason]]
    secondary = np.full(len(result), "", dtype=object)
    for position, code in enumerate(REASON_CODES):
        mask = flags[:, position] & (primary != code)
        if not mask.any():
            continue
        empty = secondary[mask] == ""
        secondary[mask] = np.where(
            empty,
            code,
            np.char.add(np.char.add(secondary[mask].astype(str), "|"), code),
        )
    result["primary_exclusion_reason"] = pd.Series(
        primary, index=result.index, dtype="string"
    )
    result["secondary_exclusion_reasons"] = pd.Series(
        secondary, index=result.index, dtype="string"
    )
    result["eligible"] = ~any_reason
    return result


_SCHEMA_STRING_COLUMNS = {
    "ledger_id",
    "source_table",
    "unit_type",
    "exclusion_level",
    "game_id",
    "play_id",
    "nfl_id",
    "target_nfl_id",
    "defender_nfl_id",
    "phase",
    "label_phase",
    "week",
    "split",
    "play_direction",
    "origin_kind",
    "player_role",
    "player_side",
    "observed_origin_defender_ids",
    "evaluable_defender_ids",
    "defender_set_definition",
    "coordinate_category",
    "primary_exclusion_reason",
    "secondary_exclusion_reasons",
}
_SCHEMA_BOOLEAN_COLUMNS = {
    "eligible",
    "metadata_week_matches_file",
    "history_2_eligible",
    "history_5_eligible",
    "five_frame_history_eligible",
    "duplicate_at_origin",
    "origin_coordinate_missing",
    "origin_outside_extended",
    "origin_extended_tolerance",
    "is_target_role",
    "is_defender",
    "player_to_predict",
    "matched_output_group",
    "full_output_group_exact",
    "observed_at_origin",
    "horizon_frames_contiguous",
    "target_trajectory_eligible",
    "coordinate_missing",
    "outside_nominal",
    "outside_extended_tolerance",
    *REASON_CODES,
}


def _freeze_table_schema(frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Select and type the stable public columns for one returned table."""

    schema = TABLE_SCHEMAS[table_name]
    missing = [column for column in schema if column not in frame.columns]
    if missing and not frame.empty:
        raise AuditError(f"{table_name} is missing frozen columns: {missing}")
    result = frame.copy()
    for column in missing:
        if column in _SCHEMA_STRING_COLUMNS:
            result[column] = pd.Series(pd.NA, index=result.index, dtype="string")
        elif column in _SCHEMA_BOOLEAN_COLUMNS:
            result[column] = pd.Series(False, index=result.index, dtype=bool)
        else:
            result[column] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result = result[schema].copy()
    for column in schema:
        if column in _SCHEMA_STRING_COLUMNS:
            result[column] = result[column].astype("string")
        elif column in _SCHEMA_BOOLEAN_COLUMNS:
            result[column] = result[column].fillna(False).astype(bool)
        else:
            result[column] = pd.to_numeric(
                result[column], errors="coerce"
            ).astype("Int64")
    keys = TABLE_KEY_COLUMNS[table_name]
    if not result.empty and result.duplicated(keys).any():
        raise AuditError(f"{table_name} has duplicate frozen keys: {keys}")
    return result.reset_index(drop=True)


def _week_columns(frame: pd.DataFrame, week_label: str) -> pd.DataFrame:
    result = frame.copy()
    result["week"] = week_label
    result["week_number"] = int(week_label[-2:])
    result["split"] = split_for_week(week_label)
    return result


def _prepare_input(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(INPUT_USECOLS) - set(frame.columns))
    if missing:
        raise AuditError(f"Input frame is missing required columns: {missing}")
    result = frame[INPUT_USECOLS].copy()
    for column in ("game_id", "play_id", "nfl_id"):
        canonical, _, valid = _canonical_positive_integer(result[column])
        result[column] = canonical
        result[f"_valid_{column}"] = valid
    _, integral_frame, valid_frame = _canonical_positive_integer(
        result["frame_id"]
    )
    result["frame_id"] = integral_frame
    result["_valid_frame_id"] = valid_frame
    coordinates = classify_coordinates(result["x"], result["y"])
    for column in coordinates:
        result[column] = coordinates[column]
    result["play_direction"] = (
        result["play_direction"].astype("string").str.strip().str.lower()
    )
    result["player_side"] = result["player_side"].astype("string").str.strip()
    result["player_role"] = result["player_role"].astype("string").str.strip()
    result["_role_value"] = result["player_role"].fillna("<missing>")
    result["_side_value"] = result["player_side"].fillna("<missing>")
    predict_text = (
        result["player_to_predict"].astype("string").str.strip().str.lower()
    )
    result["_predict_valid"] = predict_text.isin(["true", "false"])
    result["player_to_predict"] = predict_text.eq("true").fillna(False)
    declaration = pd.to_numeric(
        result["num_frames_output"], errors="coerce"
    ).astype("float64")
    declaration_finite = pd.Series(
        np.isfinite(declaration.to_numpy()), index=result.index, dtype=bool
    )
    declaration_valid = (
        declaration.notna()
        & declaration_finite
        & declaration.gt(0)
        & declaration.eq(np.floor(declaration))
    )
    result["_declared_output_frames"] = declaration.where(
        declaration_valid
    ).astype("Int64")
    result["_declaration_nonnull"] = declaration.notna()
    result["_declaration_valid"] = declaration_valid
    result["_valid_play_key"] = (
        result["_valid_game_id"] & result["_valid_play_id"]
    )
    result["_valid_entity_key"] = (
        result["_valid_play_key"] & result["_valid_nfl_id"]
    )
    result["_valid_entity_frame_key"] = (
        result["_valid_entity_key"] & result["_valid_frame_id"]
    )
    result["_is_target"] = result["player_role"].eq(TARGET_ROLE).fillna(False)
    result["_is_defender"] = (
        result["player_role"].eq(DEFENDER_ROLE)
        & result["player_side"].eq(DEFENSE_SIDE)
    ).fillna(False)
    result["phase"] = "input"
    result["_duplicate_key"] = False
    valid = result["_valid_entity_frame_key"]
    result.loc[valid, "_duplicate_key"] = result.loc[valid].duplicated(
        [*PLAY_KEYS, "frame_id", "nfl_id"], keep=False
    )
    return result


def _prepare_output(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(OUTPUT_USECOLS) - set(frame.columns))
    if missing:
        raise AuditError(f"Output frame is missing required columns: {missing}")
    result = frame[OUTPUT_USECOLS].copy()
    for column in ("game_id", "play_id", "nfl_id"):
        canonical, _, valid = _canonical_positive_integer(result[column])
        result[column] = canonical
        result[f"_valid_{column}"] = valid
    _, integral_frame, valid_frame = _canonical_positive_integer(
        result["frame_id"]
    )
    result["frame_id"] = integral_frame
    result["_valid_frame_id"] = valid_frame
    coordinates = classify_coordinates(result["x"], result["y"])
    for column in coordinates:
        result[column] = coordinates[column]
    result["_valid_play_key"] = (
        result["_valid_game_id"] & result["_valid_play_id"]
    )
    result["_valid_entity_key"] = (
        result["_valid_play_key"] & result["_valid_nfl_id"]
    )
    result["_valid_entity_frame_key"] = (
        result["_valid_entity_key"] & result["_valid_frame_id"]
    )
    result["phase"] = "output"
    result["_duplicate_key"] = False
    valid = result["_valid_entity_frame_key"]
    result.loc[valid, "_duplicate_key"] = result.loc[valid].duplicated(
        [*PLAY_KEYS, "frame_id", "nfl_id"], keep=False
    )
    return result


def _metadata_frame(
    supplementary: pd.DataFrame | set[tuple[str, str]],
) -> pd.DataFrame:
    if isinstance(supplementary, set):
        metadata = pd.DataFrame(
            sorted(supplementary), columns=[*PLAY_KEYS]
        )
        metadata["metadata_week"] = pd.NA
        return metadata
    missing = sorted(set(PLAY_KEYS) - set(supplementary.columns))
    if missing:
        raise AuditError(f"Supplementary keys are missing columns: {missing}")
    metadata = supplementary.copy()
    for column in PLAY_KEYS:
        canonical, _, valid = _canonical_positive_integer(metadata[column])
        if not bool(valid.all()):
            raise AuditError(
                f"Supplementary table contains invalid {column} values."
            )
        metadata[column] = canonical
    if metadata.duplicated(PLAY_KEYS).any():
        raise AuditError("Supplementary table has duplicate game/play keys.")
    if "metadata_week" not in metadata:
        if "week" in metadata:
            metadata["metadata_week"] = pd.to_numeric(
                metadata["week"], errors="coerce"
            ).astype("Int64")
        else:
            metadata["metadata_week"] = pd.NA
    return metadata[[*PLAY_KEYS, "metadata_week"]]


def _safe_integer(value: Any) -> int:
    if value is None or value is pd.NA:
        return 0
    try:
        if pd.isna(value):
            return 0
    except (TypeError, ValueError):
        pass
    return int(value)


def _build_source_plays(
    inputs: pd.DataFrame,
    metadata: pd.DataFrame,
    week_label: str,
) -> pd.DataFrame:
    valid_play_rows = inputs.loc[inputs["_valid_play_key"]].copy()
    if valid_play_rows.empty:
        columns = [
            *PLAY_KEYS,
            "input_rows",
            "origin_frame",
            "target_nfl_id",
            "target_nfl_id_count",
            "play_direction",
            "direction_value_count",
        ]
        return assign_exclusion_reasons(
            _week_columns(pd.DataFrame(columns=columns), week_label)
        )

    play_base = (
        valid_play_rows.groupby(PLAY_KEYS, sort=False, observed=True)
        .agg(
            input_rows=("phase", "size"),
            origin_frame=("frame_id", "max"),
            valid_input_frame_count=("frame_id", "nunique"),
            invalid_entity_frame_rows=(
                "_valid_entity_frame_key",
                lambda values: int((~values).sum()),
            ),
        )
        .reset_index()
    )

    target_rows = valid_play_rows.loc[
        valid_play_rows["_is_target"] & valid_play_rows["_valid_nfl_id"]
    ]
    target_summary = (
        target_rows.groupby(PLAY_KEYS, sort=False, observed=True)
        .agg(
            target_nfl_id=("nfl_id", "min"),
            target_nfl_id_count=("nfl_id", "nunique"),
            target_rows=("nfl_id", "size"),
        )
        .reset_index()
    )
    play_base = play_base.merge(target_summary, on=PLAY_KEYS, how="left")
    play_base["target_nfl_id_count"] = (
        play_base["target_nfl_id_count"].fillna(0).astype("int64")
    )
    play_base["target_rows"] = play_base["target_rows"].fillna(0).astype("int64")

    direction_rows = valid_play_rows.assign(
        _direction_valid=valid_play_rows["play_direction"].isin(["left", "right"])
    )
    direction_summary = (
        direction_rows.groupby(PLAY_KEYS, sort=False, observed=True)
        .agg(
            play_direction=("play_direction", "min"),
            direction_value_count=("play_direction", "nunique"),
            invalid_direction_rows=(
                "_direction_valid",
                lambda values: int((~values).sum()),
            ),
        )
        .reset_index()
    )
    play_base = play_base.merge(direction_summary, on=PLAY_KEYS, how="left")

    valid_entities = valid_play_rows.loc[
        valid_play_rows["_valid_entity_key"]
    ].copy()
    entity_consistency = (
        valid_entities.groupby(ENTITY_KEYS, sort=False, observed=True)
        .agg(
            role_value_count=("_role_value", "nunique"),
            side_value_count=("_side_value", "nunique"),
            predict_value_count=("player_to_predict", "nunique"),
            invalid_predict_rows=(
                "_predict_valid",
                lambda values: int((~values).sum()),
            ),
        )
        .reset_index()
    )
    entity_consistency["_ambiguous"] = (
        entity_consistency["role_value_count"].ne(1)
        | entity_consistency["side_value_count"].ne(1)
        | entity_consistency["predict_value_count"].ne(1)
        | entity_consistency["invalid_predict_rows"].gt(0)
    )
    ambiguity = (
        entity_consistency.groupby(PLAY_KEYS, sort=False, observed=True)
        .agg(
            ambiguous_entity_assignments=("_ambiguous", "sum"),
            tracked_entity_count=("nfl_id", "nunique"),
        )
        .reset_index()
    )
    play_base = play_base.merge(ambiguity, on=PLAY_KEYS, how="left")
    play_base["ambiguous_entity_assignments"] = (
        play_base["ambiguous_entity_assignments"].fillna(0).astype("int64")
    )
    play_base["tracked_entity_count"] = (
        play_base["tracked_entity_count"].fillna(0).astype("int64")
    )

    play_base = play_base.merge(
        metadata.assign(_metadata_match=True),
        on=PLAY_KEYS,
        how="left",
    )
    play_base["_metadata_match"] = play_base["_metadata_match"].fillna(False)
    metadata_week = pd.to_numeric(
        play_base["metadata_week"], errors="coerce"
    ).astype("Int64")
    expected_week = int(week_label[-2:])
    play_base["metadata_week_matches_file"] = (
        metadata_week.isna() | metadata_week.eq(expected_week)
    )
    play_base["phase"] = "input"
    play_base = _week_columns(play_base, week_label)

    play_base["C01"] = ~play_base["_metadata_match"]
    play_base["C02"] = play_base["origin_frame"].isna()
    play_base["C03"] = False
    play_base["C04"] = play_base["target_nfl_id_count"].ne(1)
    play_base["C05"] = (
        play_base["direction_value_count"].ne(1)
        | play_base["invalid_direction_rows"].gt(0)
        | ~play_base["play_direction"].isin(["left", "right"])
    )
    play_base["C14"] = play_base["ambiguous_entity_assignments"].gt(0)
    result = assign_exclusion_reasons(play_base)
    result = result.sort_values(PLAY_KEYS, kind="stable").reset_index(drop=True)
    return result


def _build_entity_dimension(
    inputs: pd.DataFrame,
    source_plays: pd.DataFrame,
) -> pd.DataFrame:
    entity_rows = inputs.loc[inputs["_valid_entity_key"]].copy()
    entity = (
        entity_rows.groupby(ENTITY_KEYS, sort=False, observed=True)
        .agg(
            player_role=("_role_value", "min"),
            player_side=("_side_value", "min"),
            role_value_count=("_role_value", "nunique"),
            side_value_count=("_side_value", "nunique"),
            player_to_predict=("player_to_predict", "max"),
            predict_value_count=("player_to_predict", "nunique"),
            invalid_predict_rows=(
                "_predict_valid",
                lambda values: int((~values).sum()),
            ),
            declared_output_frames=("_declared_output_frames", "min"),
            declared_output_frame_value_count=(
                "_declared_output_frames",
                "nunique",
            ),
            nonnull_declaration_rows=("_declaration_nonnull", "sum"),
            invalid_declaration_rows=(
                "_declaration_valid",
                lambda values: int((~values).sum()),
            ),
        )
        .reset_index()
    )
    entity["role_side_unambiguous"] = (
        entity["role_value_count"].eq(1)
        & entity["side_value_count"].eq(1)
        & entity["predict_value_count"].eq(1)
        & entity["invalid_predict_rows"].eq(0)
    )

    origins = source_plays[[*PLAY_KEYS, "origin_frame"]]
    relative = entity_rows.merge(origins, on=PLAY_KEYS, how="left")
    relative["relative_frame"] = (
        relative["frame_id"].astype("Float64")
        - relative["origin_frame"].astype("Float64")
    )
    relative["_history_raw_usable"] = (
        relative["_valid_entity_frame_key"] & ~relative["_duplicate_key"]
    )
    relative["_history_valid_usable"] = (
        relative["_history_raw_usable"] & relative["coordinate_valid"]
    )

    def history_counts(mask: pd.Series, name: str) -> pd.Series:
        return (
            relative.loc[mask]
            .groupby(ENTITY_KEYS, sort=False, observed=True)["relative_frame"]
            .nunique()
            .rename(name)
        )

    history = pd.concat(
        [
            history_counts(
                relative["_history_raw_usable"]
                & relative["relative_frame"].between(-1, 0, inclusive="both"),
                "raw_history_2_frame_count",
            ),
            history_counts(
                relative["_history_valid_usable"]
                & relative["relative_frame"].between(-1, 0, inclusive="both"),
                "valid_history_2_frame_count",
            ),
            history_counts(
                relative["_history_raw_usable"]
                & relative["relative_frame"].between(-4, 0, inclusive="both"),
                "raw_history_5_frame_count",
            ),
            history_counts(
                relative["_history_valid_usable"]
                & relative["relative_frame"].between(-4, 0, inclusive="both"),
                "valid_history_5_frame_count",
            ),
        ],
        axis=1,
    ).fillna(0)
    history = history.astype("int64").reset_index()
    entity = entity.merge(history, on=ENTITY_KEYS, how="left")
    for column in (
        "raw_history_2_frame_count",
        "valid_history_2_frame_count",
        "raw_history_5_frame_count",
        "valid_history_5_frame_count",
    ):
        entity[column] = entity[column].fillna(0).astype("int64")

    origin_rows = relative.loc[relative["relative_frame"].eq(0)].copy()
    origin_status = (
        origin_rows.groupby(ENTITY_KEYS, sort=False, observed=True)
        .agg(
            registered_at_origin=("phase", "size"),
            valid_origin_rows=("_history_valid_usable", "sum"),
            duplicate_at_origin=("_duplicate_key", "max"),
            origin_coordinate_missing=("coordinate_missing", "max"),
            origin_outside_extended=(
                "outside_extended_tolerance",
                "max",
            ),
            origin_extended_tolerance=(
                "coordinate_category",
                lambda values: bool(values.eq("extended-tolerance").any()),
            ),
        )
        .reset_index()
    )
    entity = entity.merge(origin_status, on=ENTITY_KEYS, how="left")
    entity["registered_at_origin"] = (
        entity["registered_at_origin"].fillna(0).astype("int64")
    )
    entity["valid_origin_rows"] = (
        entity["valid_origin_rows"].fillna(0).astype("int64")
    )
    for column in (
        "duplicate_at_origin",
        "origin_coordinate_missing",
        "origin_outside_extended",
        "origin_extended_tolerance",
    ):
        entity[column] = entity[column].fillna(False).astype(bool)

    history_2_rows = relative.loc[
        relative["relative_frame"].between(-1, 0, inclusive="both")
    ]
    history_quality = (
        history_2_rows.groupby(ENTITY_KEYS, sort=False, observed=True)
        .agg(
            duplicate_in_history_2=("_duplicate_key", "max"),
            missing_coordinate_in_history_2=("coordinate_missing", "max"),
            outside_extended_in_history_2=(
                "outside_extended_tolerance",
                "max",
            ),
            extended_tolerance_in_history_2=(
                "coordinate_category",
                lambda values: bool(values.eq("extended-tolerance").any()),
            ),
        )
        .reset_index()
    )
    entity = entity.merge(history_quality, on=ENTITY_KEYS, how="left")
    for column in (
        "duplicate_in_history_2",
        "missing_coordinate_in_history_2",
        "outside_extended_in_history_2",
        "extended_tolerance_in_history_2",
    ):
        entity[column] = entity[column].fillna(False).astype(bool)
    entity["history_2_eligible"] = (
        entity["valid_history_2_frame_count"].eq(2)
    )
    entity["history_5_eligible"] = (
        entity["valid_history_5_frame_count"].eq(5)
    )
    entity["observed_at_origin"] = (
        entity["registered_at_origin"].gt(0)
        & entity["valid_origin_rows"].eq(1)
        & ~entity["duplicate_at_origin"]
    )
    entity["is_defender"] = (
        entity["player_role"].eq(DEFENDER_ROLE)
        & entity["player_side"].eq(DEFENSE_SIDE)
    )
    entity["is_target_role"] = entity["player_role"].eq(TARGET_ROLE)
    return entity


def _build_descriptive_target_frames(
    inputs: pd.DataFrame,
    source_plays: pd.DataFrame,
    week_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame_rows = inputs.loc[
        inputs["_valid_play_key"] & inputs["_valid_frame_id"]
    ].copy()
    frames = (
        frame_rows.groupby(FRAME_KEYS, sort=False, observed=True)
        .agg(frame_raw_rows=("phase", "size"))
        .reset_index()
    )
    frames["phase"] = "input"

    target_rows = frame_rows.loc[frame_rows["_is_target"]]
    target_summary = (
        target_rows.groupby(FRAME_KEYS, sort=False, observed=True)
        .agg(
            target_row_count=("phase", "size"),
            target_entity_count=("nfl_id", "nunique"),
            target_nfl_id=("nfl_id", "min"),
            valid_target_coordinate_rows=("coordinate_valid", "sum"),
            target_duplicate_rows=("_duplicate_key", "sum"),
            target_coordinate_missing_rows=("coordinate_missing", "sum"),
            target_outside_extended_rows=(
                "outside_extended_tolerance",
                "sum",
            ),
            target_extended_tolerance_rows=(
                "coordinate_category",
                lambda values: int(values.eq("extended-tolerance").sum()),
            ),
            target_invalid_key_rows=(
                "_valid_entity_frame_key",
                lambda values: int((~values).sum()),
            ),
        )
        .reset_index()
    )
    frames = frames.merge(target_summary, on=FRAME_KEYS, how="left")

    defender_rows = frame_rows.loc[frame_rows["_is_defender"]].copy()
    defender_rows["_defender_usable"] = (
        defender_rows["_valid_entity_frame_key"]
        & defender_rows["coordinate_valid"]
        & ~defender_rows["_duplicate_key"]
    )
    defender_summary = (
        defender_rows.groupby(FRAME_KEYS, sort=False, observed=True)
        .agg(
            registered_defender_rows=("phase", "size"),
            observed_defender_count=("nfl_id", "nunique"),
            valid_observed_defender_count=(
                "_defender_usable",
                "sum",
            ),
            defender_duplicate_rows=("_duplicate_key", "sum"),
            defender_coordinate_missing_rows=("coordinate_missing", "sum"),
            defender_outside_extended_rows=(
                "outside_extended_tolerance",
                "sum",
            ),
            defender_extended_tolerance_rows=(
                "coordinate_category",
                lambda values: int(values.eq("extended-tolerance").sum()),
            ),
            defender_invalid_key_rows=(
                "_valid_entity_frame_key",
                lambda values: int((~values).sum()),
            ),
        )
        .reset_index()
    )
    valid_defender_counts = (
        defender_rows.loc[defender_rows["_defender_usable"]]
        .groupby(FRAME_KEYS, sort=False, observed=True)["nfl_id"]
        .nunique()
        .rename("_valid_defender_nunique")
        .reset_index()
    )
    defender_summary = defender_summary.drop(
        columns=["valid_observed_defender_count"]
    ).merge(valid_defender_counts, on=FRAME_KEYS, how="left")
    defender_summary["valid_observed_defender_count"] = (
        defender_summary["_valid_defender_nunique"].fillna(0).astype("int64")
    )
    defender_summary = defender_summary.drop(
        columns=["_valid_defender_nunique"]
    )
    frames = frames.merge(defender_summary, on=FRAME_KEYS, how="left")

    integer_columns = [
        "target_row_count",
        "target_entity_count",
        "valid_target_coordinate_rows",
        "target_duplicate_rows",
        "target_coordinate_missing_rows",
        "target_outside_extended_rows",
        "target_extended_tolerance_rows",
        "target_invalid_key_rows",
        "registered_defender_rows",
        "observed_defender_count",
        "valid_observed_defender_count",
        "defender_duplicate_rows",
        "defender_coordinate_missing_rows",
        "defender_outside_extended_rows",
        "defender_extended_tolerance_rows",
        "defender_invalid_key_rows",
    ]
    for column in integer_columns:
        frames[column] = frames[column].fillna(0).astype("int64")

    parent_columns = [
        *PLAY_KEYS,
        "target_nfl_id",
        "C01",
        "C02",
        "C04",
        "C05",
        "C14",
    ]
    frames = frames.drop(columns=["target_nfl_id"]).merge(
        source_plays[parent_columns],
        on=PLAY_KEYS,
        how="left",
    )
    frames = _week_columns(frames, week_label)
    frames["C01"] = frames["C01"].fillna(True)
    frames["C02"] = (
        frames["C02"].fillna(False)
        | frames["target_invalid_key_rows"].gt(0)
        | frames["defender_invalid_key_rows"].gt(0)
    )
    frames["C03"] = (
        frames["target_duplicate_rows"].gt(0)
        | frames["defender_duplicate_rows"].gt(0)
    )
    frames["C04"] = (
        frames["C04"].fillna(True)
        | frames["target_row_count"].ne(1)
        | frames["target_entity_count"].ne(1)
    )
    frames["C05"] = frames["C05"].fillna(True)
    frames["C06"] = (
        frames["valid_target_coordinate_rows"].ne(1)
        | frames["defender_coordinate_missing_rows"].gt(0)
    )
    frames["C07"] = frames["valid_observed_defender_count"].lt(1)
    frames["C08"] = (
        frames["target_outside_extended_rows"].gt(0)
        | frames["defender_outside_extended_rows"].gt(0)
    )
    frames["C14"] = frames["C14"].fillna(True)
    frames = assign_exclusion_reasons(frames)
    frames = frames.sort_values(FRAME_KEYS, kind="stable").reset_index(drop=True)

    pair_candidates = defender_rows.loc[
        defender_rows["_valid_play_key"]
        & (
            ~defender_rows["_valid_nfl_id"]
            | ~defender_rows["coordinate_valid"]
            | defender_rows["_duplicate_key"]
        )
    ].copy()
    if pair_candidates.empty:
        pair_exclusions = pd.DataFrame(
            columns=[
                *PLAY_KEYS,
                "frame_id",
                "target_nfl_id",
                "defender_nfl_id",
                "phase",
                "week",
                "week_number",
                "split",
                *REASON_CODES,
                "primary_exclusion_reason",
                "secondary_exclusion_reasons",
                "eligible",
            ]
        )
    else:
        pair_candidates = pair_candidates.rename(
            columns={"nfl_id": "defender_nfl_id"}
        )
        pair_candidates = pair_candidates.merge(
            source_plays[
                [*PLAY_KEYS, "target_nfl_id", "C01", "C04", "C05", "C14"]
            ],
            on=PLAY_KEYS,
            how="left",
        )
        pair_candidates = _week_columns(pair_candidates, week_label)
        pair_candidates["C01"] = pair_candidates["C01"].fillna(True)
        pair_candidates["C02"] = ~pair_candidates["_valid_nfl_id"]
        pair_candidates["C03"] = pair_candidates["_duplicate_key"]
        pair_candidates["C04"] = pair_candidates["C04"].fillna(True)
        pair_candidates["C05"] = pair_candidates["C05"].fillna(True)
        pair_candidates["C06"] = pair_candidates["coordinate_missing"]
        pair_candidates["C08"] = pair_candidates[
            "outside_extended_tolerance"
        ]
        pair_candidates["C14"] = pair_candidates["C14"].fillna(True)
        pair_exclusions = assign_exclusion_reasons(pair_candidates)
        pair_exclusions = pair_exclusions.loc[~pair_exclusions["eligible"]]
    return frames, pair_exclusions


def _joined_id_list(
    frame: pd.DataFrame,
    group_columns: list[str],
    id_column: str,
    output_column: str,
) -> pd.DataFrame:
    """Build a deterministic compact pipe-delimited identifier set."""

    if frame.empty:
        return pd.DataFrame(columns=[*group_columns, output_column])
    ordered = frame.sort_values(
        [*group_columns, id_column], kind="stable"
    ).drop_duplicates([*group_columns, id_column])
    joined = (
        ordered.groupby(group_columns, sort=False, observed=True)[id_column]
        .agg("|".join)
        .rename(output_column)
        .reset_index()
    )
    return joined


def _build_primary_origins(
    source_plays: pd.DataFrame,
    entity: pd.DataFrame,
    week_label: str,
) -> pd.DataFrame:
    parent_columns = [
        *PLAY_KEYS,
        "target_nfl_id",
        "origin_frame",
        "play_direction",
        "C01",
        "C02",
        "C04",
        "C05",
        "C14",
    ]
    origins = source_plays[parent_columns].copy()
    target_history = entity.rename(columns={"nfl_id": "target_nfl_id"})
    target_columns = [
        *PLAY_KEYS,
        "target_nfl_id",
        "raw_history_2_frame_count",
        "valid_history_2_frame_count",
        "raw_history_5_frame_count",
        "valid_history_5_frame_count",
        "history_2_eligible",
        "history_5_eligible",
        "registered_at_origin",
        "valid_origin_rows",
        "duplicate_at_origin",
        "origin_coordinate_missing",
        "origin_outside_extended",
        "origin_extended_tolerance",
        "duplicate_in_history_2",
        "missing_coordinate_in_history_2",
        "outside_extended_in_history_2",
        "extended_tolerance_in_history_2",
    ]
    origins = origins.merge(
        target_history[target_columns],
        on=[*PLAY_KEYS, "target_nfl_id"],
        how="left",
    )

    count_columns = [
        "raw_history_2_frame_count",
        "valid_history_2_frame_count",
        "raw_history_5_frame_count",
        "valid_history_5_frame_count",
        "registered_at_origin",
        "valid_origin_rows",
    ]
    for column in count_columns:
        origins[column] = origins[column].fillna(0).astype("int64")
    boolean_columns = [
        "history_2_eligible",
        "history_5_eligible",
        "duplicate_at_origin",
        "origin_coordinate_missing",
        "origin_outside_extended",
        "origin_extended_tolerance",
        "duplicate_in_history_2",
        "missing_coordinate_in_history_2",
        "outside_extended_in_history_2",
        "extended_tolerance_in_history_2",
    ]
    for column in boolean_columns:
        origins[column] = origins[column].fillna(False).astype(bool)

    origin_defenders = entity.loc[
        entity["is_defender"] & entity["registered_at_origin"].gt(0)
    ]
    defender_counts = (
        origin_defenders.groupby(PLAY_KEYS, sort=False, observed=True)
        .agg(
            registered_origin_defender_count=("nfl_id", "nunique"),
            valid_observed_origin_defender_count=(
                "observed_at_origin",
                "sum",
            ),
        )
        .reset_index()
    )
    valid_origin_defenders = origin_defenders.loc[
        origin_defenders["observed_at_origin"]
    ]
    defender_ids = _joined_id_list(
        valid_origin_defenders,
        PLAY_KEYS,
        "nfl_id",
        "observed_origin_defender_ids",
    )
    origins = origins.merge(defender_counts, on=PLAY_KEYS, how="left").merge(
        defender_ids, on=PLAY_KEYS, how="left"
    )
    origins["registered_origin_defender_count"] = (
        origins["registered_origin_defender_count"].fillna(0).astype("int64")
    )
    origins["valid_observed_origin_defender_count"] = (
        origins["valid_observed_origin_defender_count"]
        .fillna(0)
        .astype("int64")
    )
    origins["observed_origin_defender_ids"] = (
        origins["observed_origin_defender_ids"].fillna("").astype("string")
    )

    origins["origin_kind"] = "last_input_frame"
    origins["phase"] = "input"
    origins["relative_frame"] = 0
    origins = _week_columns(origins, week_label)
    origins["C03"] = (
        origins["duplicate_at_origin"]
        | origins["duplicate_in_history_2"]
    )
    origins["C06"] = (
        origins["origin_coordinate_missing"]
        | origins["missing_coordinate_in_history_2"]
        | origins["valid_origin_rows"].ne(1)
    )
    origins["C08"] = (
        origins["origin_outside_extended"]
        | origins["outside_extended_in_history_2"]
    )
    origins["C09"] = origins["raw_history_2_frame_count"].ne(2)
    origins["C12"] = ~origins["history_2_eligible"]
    origins = assign_exclusion_reasons(origins)
    origins["five_frame_history_eligible"] = origins["history_5_eligible"]
    origins = origins.sort_values(PLAY_KEYS, kind="stable").reset_index(drop=True)
    return origins


def _output_group_summaries(
    outputs: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    valid_group_rows = outputs.loc[outputs["_valid_entity_key"]].copy()
    valid_group_rows["_invalid_frame"] = ~valid_group_rows["_valid_frame_id"]
    valid_group_rows["_invalid_coordinate"] = ~valid_group_rows[
        "coordinate_valid"
    ]
    valid_group_rows["_extended_tolerance"] = valid_group_rows[
        "coordinate_category"
    ].eq("extended-tolerance")
    full = (
        valid_group_rows.groupby(ENTITY_KEYS, sort=False, observed=True)
        .agg(
            output_row_count=("phase", "size"),
            output_frame_count=("frame_id", "nunique"),
            output_min_frame=("frame_id", "min"),
            output_max_frame=("frame_id", "max"),
            output_invalid_frame_rows=("_invalid_frame", "sum"),
            output_duplicate_rows=("_duplicate_key", "sum"),
            output_invalid_coordinate_rows=("_invalid_coordinate", "sum"),
            output_coordinate_missing_rows=("coordinate_missing", "sum"),
            output_outside_extended_rows=(
                "outside_extended_tolerance",
                "sum",
            ),
            output_extended_tolerance_rows=(
                "_extended_tolerance",
                "sum",
            ),
            output_outside_nominal_rows=("outside_nominal", "sum"),
        )
        .reset_index()
    )
    horizon_summaries: dict[int, pd.DataFrame] = {}
    for horizon in HORIZONS:
        consumed = valid_group_rows.loc[
            valid_group_rows["frame_id"].between(
                1, horizon, inclusive="both"
            )
        ]
        summary = (
            consumed.groupby(ENTITY_KEYS, sort=False, observed=True)
            .agg(
                horizon_output_row_count=("phase", "size"),
                horizon_output_frame_count=("frame_id", "nunique"),
                horizon_output_min_frame=("frame_id", "min"),
                horizon_output_max_frame=("frame_id", "max"),
                horizon_output_duplicate_rows=("_duplicate_key", "sum"),
                horizon_invalid_coordinate_rows=(
                    "_invalid_coordinate",
                    "sum",
                ),
                horizon_coordinate_missing_rows=("coordinate_missing", "sum"),
                horizon_outside_extended_rows=(
                    "outside_extended_tolerance",
                    "sum",
                ),
                horizon_extended_tolerance_rows=(
                    "_extended_tolerance",
                    "sum",
                ),
                horizon_outside_nominal_rows=("outside_nominal", "sum"),
            )
            .reset_index()
        )
        horizon_summaries[horizon] = summary
    return full, horizon_summaries


def _build_trajectory_eligibility(
    entity: pd.DataFrame,
    source_plays: pd.DataFrame,
    outputs: pd.DataFrame,
    week_label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    full_output, by_horizon = _output_group_summaries(outputs)
    candidates = entity.loc[entity["player_to_predict"]].copy()
    candidates = candidates.merge(
        source_plays[
            [
                *PLAY_KEYS,
                "target_nfl_id",
                "origin_frame",
                "C01",
                "C02",
                "C04",
                "C05",
                "C14",
            ]
        ],
        on=PLAY_KEYS,
        how="left",
        suffixes=("", "_play"),
    )
    candidates = candidates.merge(
        full_output.assign(matched_output_group=True),
        on=ENTITY_KEYS,
        how="left",
    )

    numeric_output_columns = [
        "output_row_count",
        "output_frame_count",
        "output_invalid_frame_rows",
        "output_duplicate_rows",
        "output_invalid_coordinate_rows",
        "output_coordinate_missing_rows",
        "output_outside_extended_rows",
        "output_extended_tolerance_rows",
        "output_outside_nominal_rows",
    ]
    for column in numeric_output_columns:
        candidates[column] = candidates[column].fillna(0).astype("int64")
    candidates["matched_output_group"] = (
        candidates["matched_output_group"].fillna(False).astype(bool)
    )

    declared = candidates["declared_output_frames"].astype("Int64")
    full_group_exact = (
        candidates["matched_output_group"]
        & candidates["declared_output_frame_value_count"].eq(1)
        & declared.notna()
        & candidates["output_invalid_frame_rows"].eq(0)
        & candidates["output_duplicate_rows"].eq(0)
        & candidates["output_row_count"].eq(declared)
        & candidates["output_frame_count"].eq(declared)
        & candidates["output_min_frame"].eq(1)
        & candidates["output_max_frame"].eq(declared)
    )
    candidates["full_output_group_exact"] = full_group_exact.fillna(False)

    horizon_tables: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        current = candidates.merge(
            by_horizon[horizon],
            on=ENTITY_KEYS,
            how="left",
        )
        horizon_count_columns = [
            "horizon_output_row_count",
            "horizon_output_frame_count",
            "horizon_output_duplicate_rows",
            "horizon_invalid_coordinate_rows",
            "horizon_coordinate_missing_rows",
            "horizon_outside_extended_rows",
            "horizon_extended_tolerance_rows",
            "horizon_outside_nominal_rows",
        ]
        for column in horizon_count_columns:
            current[column] = current[column].fillna(0).astype("int64")
        current["horizon"] = horizon
        current["phase"] = "input"
        current["label_phase"] = "output"
        current["max_input_relative_frame_used"] = 0
        current["label_start_frame"] = 1
        current["label_end_frame"] = horizon
        current["horizon_frames_contiguous"] = (
            current["horizon_output_row_count"].eq(horizon)
            & current["horizon_output_frame_count"].eq(horizon)
            & current["horizon_output_min_frame"].eq(1)
            & current["horizon_output_max_frame"].eq(horizon)
            & current["horizon_output_duplicate_rows"].eq(0)
        )
        current["C02"] = current["C02"].fillna(False)
        current["C03"] = (
            current["duplicate_in_history_2"]
            | current["horizon_output_duplicate_rows"].gt(0)
        )
        current["C06"] = (
            current["missing_coordinate_in_history_2"]
            | current["horizon_coordinate_missing_rows"].gt(0)
        )
        current["C08"] = (
            current["outside_extended_in_history_2"]
            | current["horizon_outside_extended_rows"].gt(0)
        )
        current["C09"] = (
            current["raw_history_2_frame_count"].ne(2)
            | (
                declared.ge(horizon).fillna(False)
                & ~current["horizon_frames_contiguous"]
            )
        )
        current["C10"] = ~current["full_output_group_exact"]
        current["C11"] = False
        current["C12"] = ~current["history_2_eligible"]
        current["C13"] = (
            declared.lt(horizon).fillna(True)
            | ~current["horizon_frames_contiguous"]
        )
        current["C14"] = (
            current["C14"].fillna(True)
            | ~current["role_side_unambiguous"]
        )
        current = _week_columns(current, week_label)
        current = assign_exclusion_reasons(current)
        horizon_tables.append(current)

    trajectories = pd.concat(horizon_tables, ignore_index=True)
    trajectories = trajectories.sort_values(
        [*ENTITY_KEYS, "horizon"], kind="stable"
    ).reset_index(drop=True)

    observed_groups = full_output[ENTITY_KEYS].drop_duplicates()
    expected_groups = candidates[ENTITY_KEYS].drop_duplicates()
    observed_index = pd.MultiIndex.from_frame(observed_groups)
    expected_index = pd.MultiIndex.from_frame(expected_groups)
    statistics = {
        "expected_output_groups": int(len(expected_groups)),
        "observed_output_groups": int(len(observed_groups)),
        "missing_expected_output_groups": int(
            (~expected_index.isin(observed_index)).sum()
        ),
        "unmatched_observed_output_groups": int(
            (~observed_index.isin(expected_index)).sum()
        ),
        "declaration_conflict_groups": int(
            candidates["declared_output_frame_value_count"].ne(1).sum()
        ),
        "declaration_or_full_group_mismatch_groups": int(
            (~candidates["full_output_group_exact"]).sum()
        ),
    }
    return trajectories, statistics


def _build_future_separation_eligibility(
    origins: pd.DataFrame,
    entity: pd.DataFrame,
    trajectories: pd.DataFrame,
    week_label: str,
) -> pd.DataFrame:
    base = pd.concat(
        [origins.assign(horizon=horizon) for horizon in HORIZONS],
        ignore_index=True,
    )
    target_trajectory = trajectories.loc[
        trajectories["nfl_id"].eq(trajectories["target_nfl_id"])
    ].rename(
        columns={
            "eligible": "target_trajectory_eligible",
            "primary_exclusion_reason": "target_primary_exclusion_reason",
            "secondary_exclusion_reasons": "target_secondary_exclusion_reasons",
        }
    )
    target_reason_columns = {
        code: f"target_{code}" for code in REASON_CODES
    }
    target_trajectory = target_trajectory.rename(columns=target_reason_columns)
    target_columns = [
        *PLAY_KEYS,
        "target_nfl_id",
        "horizon",
        "target_trajectory_eligible",
        "target_primary_exclusion_reason",
        "target_secondary_exclusion_reasons",
        *target_reason_columns.values(),
    ]
    base = base.merge(
        target_trajectory[target_columns],
        on=[*PLAY_KEYS, "target_nfl_id", "horizon"],
        how="left",
    )
    base["target_trajectory_eligible"] = (
        base["target_trajectory_eligible"].fillna(False).astype(bool)
    )

    origin_defenders = entity.loc[
        entity["is_defender"] & entity["observed_at_origin"]
    ].copy()
    origin_counts = (
        origin_defenders.groupby(PLAY_KEYS, sort=False, observed=True)
        .agg(
            observed_origin_defender_count=("nfl_id", "nunique"),
            prediction_candidate_origin_defender_count=(
                "player_to_predict",
                "sum",
            ),
        )
        .reset_index()
    )
    history_counts = (
        origin_defenders.loc[origin_defenders["player_to_predict"]]
        .groupby(PLAY_KEYS, sort=False, observed=True)
        .agg(
            two_frame_history_origin_defender_count=(
                "history_2_eligible",
                "sum",
            )
        )
        .reset_index()
    )
    base = base.drop(
        columns=[
            "valid_observed_origin_defender_count",
        ],
        errors="ignore",
    ).merge(origin_counts, on=PLAY_KEYS, how="left").merge(
        history_counts, on=PLAY_KEYS, how="left"
    )
    for column in (
        "observed_origin_defender_count",
        "prediction_candidate_origin_defender_count",
        "two_frame_history_origin_defender_count",
    ):
        base[column] = base[column].fillna(0).astype("int64")

    defensive_trajectories = trajectories.loc[
        trajectories["is_defender"] & trajectories["observed_at_origin"]
    ].copy()
    eligible_defenders = defensive_trajectories.loc[
        defensive_trajectories["eligible"]
    ]
    eligible_counts = (
        eligible_defenders.groupby(
            [*PLAY_KEYS, "horizon"], sort=False, observed=True
        )
        .agg(evaluable_defender_count=("nfl_id", "nunique"))
        .reset_index()
    )
    eligible_ids = _joined_id_list(
        eligible_defenders,
        [*PLAY_KEYS, "horizon"],
        "nfl_id",
        "evaluable_defender_ids",
    )
    base = base.merge(
        eligible_counts, on=[*PLAY_KEYS, "horizon"], how="left"
    ).merge(eligible_ids, on=[*PLAY_KEYS, "horizon"], how="left")
    base["evaluable_defender_count"] = (
        base["evaluable_defender_count"].fillna(0).astype("int64")
    )
    base["evaluable_defender_ids"] = (
        base["evaluable_defender_ids"].fillna("").astype("string")
    )

    candidate_defenders = defensive_trajectories.copy()
    removal_aggregations: dict[str, tuple[str, str]] = {
        "output_candidate_defender_count": ("nfl_id", "nunique"),
        "removed_defender_count": ("eligible", lambda values: int((~values).sum())),
    }
    for code in REASON_CODES:
        candidate_defenders[f"_removed_{code}"] = (
            ~candidate_defenders["eligible"] & candidate_defenders[code]
        )
        removal_aggregations[f"removed_defender_{code}_count"] = (
            f"_removed_{code}",
            "sum",
        )
    removal_counts = (
        candidate_defenders.groupby(
            [*PLAY_KEYS, "horizon"], sort=False, observed=True
        )
        .agg(**removal_aggregations)
        .reset_index()
    )
    base = base.merge(
        removal_counts, on=[*PLAY_KEYS, "horizon"], how="left"
    )
    removal_columns = [
        "output_candidate_defender_count",
        "removed_defender_count",
        *(f"removed_defender_{code}_count" for code in REASON_CODES),
    ]
    for column in removal_columns:
        base[column] = base[column].fillna(0).astype("int64")

    for code in REASON_CODES:
        target_flag = base[f"target_{code}"].fillna(False).astype(bool)
        no_evaluable = base["evaluable_defender_count"].eq(0)
        defender_failure = base[f"removed_defender_{code}_count"].gt(0)
        base[code] = target_flag | (no_evaluable & defender_failure)
    base["C07"] = (
        base["C07"]
        | base["observed_origin_defender_count"].eq(0)
    )
    base["C11"] = False
    base["C13"] = (
        base["C13"]
        | base["evaluable_defender_count"].eq(0)
    )
    base["phase"] = "input"
    base["label_phase"] = "output"
    base["max_input_relative_frame_used"] = 0
    base["label_start_frame"] = 1
    base["label_end_frame"] = base["horizon"]
    base["defender_set_definition"] = (
        "origin-observed, prediction-designated defenders with valid "
        "two-frame histories and supplied contiguous output through horizon"
    )
    base = _week_columns(
        base.drop(columns=["week", "week_number", "split"], errors="ignore"),
        week_label,
    )
    base = assign_exclusion_reasons(base)
    base = base.sort_values(
        [*PLAY_KEYS, "horizon"], kind="stable"
    ).reset_index(drop=True)
    return base


def build_week_cohorts(
    input_frame: pd.DataFrame,
    output_frame: pd.DataFrame,
    supplementary: pd.DataFrame | set[tuple[str, str]],
    week_label: str,
) -> WeekCohorts:
    """Build every named cohort for one deterministic week.

    The function is intentionally pure with respect to the filesystem so
    synthetic tests can exercise all eligibility and exclusion behavior.
    """

    if week_label not in EXPECTED_WEEKS:
        raise ValueError(f"Unexpected week label: {week_label}")
    stages: dict[str, float] = {}

    started = time.perf_counter()
    inputs = _prepare_input(input_frame)
    outputs = _prepare_output(output_frame)
    metadata = _metadata_frame(supplementary)
    stages["prepare"] = time.perf_counter() - started

    started = time.perf_counter()
    source = _build_source_plays(inputs, metadata, week_label)
    entity = _build_entity_dimension(inputs, source)
    stages["source_and_entity"] = time.perf_counter() - started

    started = time.perf_counter()
    descriptive, pair_exclusions = _build_descriptive_target_frames(
        inputs, source, week_label
    )
    stages["descriptive_frames"] = time.perf_counter() - started

    started = time.perf_counter()
    origins = _build_primary_origins(source, entity, week_label)
    stages["primary_origins"] = time.perf_counter() - started

    started = time.perf_counter()
    trajectories, output_statistics = _build_trajectory_eligibility(
        entity, source, outputs, week_label
    )
    stages["trajectory_cohorts"] = time.perf_counter() - started

    started = time.perf_counter()
    future = _build_future_separation_eligibility(
        origins, entity, trajectories, week_label
    )
    stages["future_separation_cohorts"] = time.perf_counter() - started

    input_valid_keys = inputs["_valid_entity_frame_key"]
    output_valid_keys = outputs["_valid_entity_frame_key"]
    input_target_rows = inputs.loc[inputs["_is_target"] & input_valid_keys]
    input_target_frame_count = int(
        len(input_target_rows.drop_duplicates(FRAME_KEYS))
    )
    input_coordinate_missing = int(inputs["coordinate_missing"].sum())
    output_coordinate_missing = int(outputs["coordinate_missing"].sum())
    target_frame_mask = descriptive["target_row_count"].gt(0)
    zero_defender_target_frames = int(
        (
            target_frame_mask
            & descriptive["valid_observed_defender_count"].eq(0)
        ).sum()
    )

    def boundary_counts(frame: pd.DataFrame) -> dict[str, int]:
        x = frame["x_raw"]
        y = frame["y_raw"]
        return {
            "extended_tolerance_rows": int(
                frame["coordinate_category"].eq("extended-tolerance").sum()
            ),
            "outside_extended_tolerance_rows": int(
                frame["outside_extended_tolerance"].sum()
            ),
            "outside_nominal_x_cells": int(
                (x.notna() & ~x.between(0.0, FIELD_LENGTH)).sum()
            ),
            "outside_nominal_y_cells": int(
                (y.notna() & ~y.between(0.0, FIELD_WIDTH)).sum()
            ),
        }

    statistics: dict[str, Any] = {
        "week": week_label,
        "split": split_for_week(week_label),
        "input_rows": int(len(inputs)),
        "output_rows": int(len(outputs)),
        "games": int(source["game_id"].nunique()),
        "source_plays": int(len(source)),
        "source_plays_eligible": int(source["eligible"].sum()),
        "target_labelled_plays": int(
            source["target_nfl_id_count"].eq(1).sum()
        ),
        "target_frames_before_exclusion": input_target_frame_count,
        "descriptive_target_frames_eligible": int(
            descriptive["eligible"].sum()
        ),
        "zero_defender_target_frames": zero_defender_target_frames,
        "descriptive_defender_pair_rows": int(inputs["_is_defender"].sum()),
        "input_invalid_entity_frame_key_rows": int((~input_valid_keys).sum()),
        "output_invalid_entity_frame_key_rows": int((~output_valid_keys).sum()),
        "input_duplicate_entity_frame_rows": int(
            inputs["_duplicate_key"].sum()
        ),
        "output_duplicate_entity_frame_rows": int(
            outputs["_duplicate_key"].sum()
        ),
        "input_coordinate_missing_rows": input_coordinate_missing,
        "output_coordinate_missing_rows": output_coordinate_missing,
        "input_boundary": boundary_counts(inputs),
        "output_boundary": boundary_counts(outputs),
        "primary_origins_eligible": int(origins["eligible"].sum()),
        "five_frame_target_history_eligible": int(
            origins["five_frame_history_eligible"].sum()
        ),
        "role_counts": {
            str(key): int(value)
            for key, value in inputs["player_role"]
            .value_counts(dropna=False)
            .sort_index()
            .items()
        },
        **output_statistics,
    }
    statistics["trajectory_by_horizon"] = {
        str(horizon): {
            "denominator": int((trajectories["horizon"] == horizon).sum()),
            "eligible": int(
                (
                    trajectories["horizon"].eq(horizon)
                    & trajectories["eligible"]
                ).sum()
            ),
        }
        for horizon in HORIZONS
    }
    statistics["future_separation_by_horizon"] = {
        str(horizon): {
            "denominator": int((future["horizon"] == horizon).sum()),
            "eligible": int(
                (future["horizon"].eq(horizon) & future["eligible"]).sum()
            ),
        }
        for horizon in HORIZONS
    }
    source = _freeze_table_schema(source, "source_plays")
    descriptive = _freeze_table_schema(
        descriptive, "descriptive_target_frames"
    )
    origins = _freeze_table_schema(origins, "primary_origins")
    trajectories = _freeze_table_schema(
        trajectories, "trajectory_eligibility"
    )
    future = _freeze_table_schema(
        future, "future_separation_eligibility"
    )
    pair_exclusions = _freeze_table_schema(
        pair_exclusions, "pair_exclusions"
    )
    return WeekCohorts(
        source_plays=source,
        descriptive_target_frames=descriptive,
        primary_origins=origins,
        trajectory_eligibility=trajectories,
        future_separation_eligibility=future,
        pair_exclusions=pair_exclusions,
        statistics=statistics,
        stage_seconds={key: round(value, 6) for key, value in stages.items()},
    )


def _ledger_table(
    frame: pd.DataFrame,
    *,
    source_table: str,
    unit_type: str,
    level: str,
    id_columns: list[str],
) -> pd.DataFrame:
    normalized = assign_exclusion_reasons(frame)
    failed = normalized[list(REASON_CODES)].any(axis=1)
    excluded = normalized.loc[failed].copy()
    if excluded.empty:
        return pd.DataFrame()
    excluded["source_table"] = source_table
    excluded["unit_type"] = unit_type
    excluded["exclusion_level"] = level
    optional_identifiers = (
        "phase",
        "frame_id",
        "nfl_id",
        "target_nfl_id",
        "defender_nfl_id",
        "origin_frame",
        "horizon",
    )
    for column in optional_identifiers:
        if column not in excluded:
            excluded[column] = pd.NA
        elif column not in id_columns:
            excluded[column] = pd.NA
    ledger_id = pd.Series(
        f"{source_table}|{unit_type}",
        index=excluded.index,
        dtype="string",
    )
    for column in id_columns:
        present = excluded[column].notna()
        values = excluded.loc[present, column].astype("string")
        ledger_id.loc[present] = (
            ledger_id.loc[present] + f"|{column}=" + values
        )
    excluded["ledger_id"] = ledger_id
    return excluded[EXCLUSION_LEDGER_SCHEMA]


def _freeze_exclusion_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reindex(columns=EXCLUSION_LEDGER_SCHEMA).copy()
    for column in EXCLUSION_LEDGER_SCHEMA:
        if column in _SCHEMA_STRING_COLUMNS:
            result[column] = result[column].astype("string")
        elif column in REASON_CODES:
            result[column] = result[column].fillna(False).astype(bool)
        else:
            result[column] = pd.to_numeric(
                result[column], errors="coerce"
            ).astype("Int64")
    if result["ledger_id"].duplicated().any():
        duplicates = result.loc[
            result["ledger_id"].duplicated(keep=False), "ledger_id"
        ].tolist()
        raise AuditError(f"Duplicate exclusion ledger IDs: {duplicates[:5]}")
    return result.reset_index(drop=True)


def build_exclusion_ledger(
    source: pd.DataFrame,
    descriptive: pd.DataFrame,
    origins: pd.DataFrame,
    trajectories: pd.DataFrame,
    future: pd.DataFrame,
    pair_exclusions: pd.DataFrame,
) -> pd.DataFrame:
    """Combine excluded units while retaining their distinct unit levels."""

    tables = [
        _ledger_table(
            source,
            source_table="source_plays",
            unit_type="source_play",
            level="play",
            id_columns=PLAY_KEYS,
        ),
        _ledger_table(
            descriptive,
            source_table="descriptive_target_frames",
            unit_type="target_frame",
            level="frame",
            id_columns=[
                *PLAY_KEYS,
                "phase",
                "frame_id",
                "target_nfl_id",
            ],
        ),
        _ledger_table(
            origins,
            source_table="primary_origins",
            unit_type="primary_origin",
            level="entity-origin",
            id_columns=[*PLAY_KEYS, "target_nfl_id", "origin_frame"],
        ),
        _ledger_table(
            trajectories,
            source_table="trajectory_eligibility",
            unit_type="trajectory_entity_horizon",
            level="entity-origin",
            id_columns=[
                *PLAY_KEYS,
                "nfl_id",
                "target_nfl_id",
                "origin_frame",
                "horizon",
            ],
        ),
        _ledger_table(
            future,
            source_table="future_separation_eligibility",
            unit_type="target_horizon",
            level="horizon sample",
            id_columns=[
                *PLAY_KEYS,
                "target_nfl_id",
                "origin_frame",
                "horizon",
            ],
        ),
    ]
    if not pair_exclusions.empty:
        tables.append(
            _ledger_table(
                pair_exclusions,
                source_table="pair_exclusions",
                unit_type="target_defender_pair",
                level="pair",
                id_columns=[
                    *PLAY_KEYS,
                    "phase",
                    "frame_id",
                    "target_nfl_id",
                    "defender_nfl_id",
                ],
            )
        )
    nonempty = [table for table in tables if not table.empty]
    if not nonempty:
        return _freeze_exclusion_ledger(pd.DataFrame())
    ledger = pd.concat(nonempty, ignore_index=True)
    ledger = ledger.sort_values(
        [
            "week_number",
            "source_table",
            "game_id",
            "play_id",
            "horizon",
            "frame_id",
            "nfl_id",
        ],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    return _freeze_exclusion_ledger(ledger)


def _table_reconciliation(
    tables: Mapping[str, pd.DataFrame],
    ledger: pd.DataFrame,
) -> dict[str, Any]:
    """Reconcile six frozen cohort tables to one exclusion ledger."""

    expected_names = set(COHORT_TABLE_NAMES)
    supplied_names = set(tables)
    if supplied_names != expected_names:
        raise AuditError(
            "Reconciliation requires exactly the six frozen tables: "
            f"missing={sorted(expected_names - supplied_names)}, "
            f"unexpected={sorted(supplied_names - expected_names)}"
        )
    missing_ledger_columns = sorted(
        set(EXCLUSION_LEDGER_SCHEMA) - set(ledger.columns)
    )
    if missing_ledger_columns:
        raise AuditError(
            f"Exclusion ledger is missing columns: {missing_ledger_columns}"
        )

    expected_ledger = build_exclusion_ledger(
        *(tables[name] for name in COHORT_TABLE_NAMES)
    )
    ledger_sources = ledger["source_table"].astype("string")
    unknown_source_mask = ~ledger_sources.isin(COHORT_TABLE_NAMES)
    unknown_source_values = sorted(
        {
            None if pd.isna(value) else str(value)
            for value in ledger_sources.loc[unknown_source_mask]
        },
        key=lambda value: "" if value is None else value,
    )

    records: list[dict[str, Any]] = []
    overall_primary = {code: 0 for code in REASON_CODES}
    for name in COHORT_TABLE_NAMES:
        normalized = assign_exclusion_reasons(tables[name])
        failed = normalized[list(REASON_CODES)].any(axis=1)
        total_rows = int(len(normalized))
        excluded_rows = int(failed.sum())
        eligible_rows = total_rows - excluded_rows
        primary_values = normalized.loc[
            failed, "primary_exclusion_reason"
        ].value_counts()
        primary_counts = {
            code: int(primary_values.get(code, 0)) for code in REASON_CODES
        }
        for code in REASON_CODES:
            overall_primary[code] += primary_counts[code]

        expected_ids = expected_ledger.loc[
            expected_ledger["source_table"].eq(name), "ledger_id"
        ].astype("string")
        actual_ids = ledger.loc[
            ledger_sources.eq(name), "ledger_id"
        ].astype("string")
        expected_counts = expected_ids.value_counts(dropna=False)
        actual_counts = actual_ids.value_counts(dropna=False)
        missing_rows = int(
            sum(
                max(
                    int(expected_count)
                    - int(actual_counts.get(identifier, 0)),
                    0,
                )
                for identifier, expected_count in expected_counts.items()
            )
        )
        duplicate_rows = int(
            sum(max(int(count) - 1, 0) for count in actual_counts.values)
        )
        expected_id_set = set(expected_ids.dropna())
        unexpected_rows = int(
            (
                actual_ids.isna()
                | ~actual_ids.isin(expected_id_set)
            ).sum()
        )
        ledger_rows = int(len(actual_ids))
        difference = excluded_rows - ledger_rows
        primary_sum = sum(primary_counts.values())
        status = (
            "PASS"
            if (
                total_rows == eligible_rows + excluded_rows
                and primary_sum == excluded_rows
                and difference == 0
                and missing_rows == 0
                and duplicate_rows == 0
                and unexpected_rows == 0
            )
            else "FAIL"
        )
        records.append(
            {
                "source_table": name,
                "total_rows": total_rows,
                "eligible_rows": eligible_rows,
                "excluded_rows": excluded_rows,
                "exclusion_ledger_rows": ledger_rows,
                "excluded_minus_ledger_rows": difference,
                "primary_exclusion_counts": primary_counts,
                "missing_ledger_rows": missing_rows,
                "duplicate_ledger_rows": duplicate_rows,
                "unexpected_ledger_rows": unexpected_rows,
                "reconciliation_status": status,
            }
        )

    overall = {
        "total_rows": sum(record["total_rows"] for record in records),
        "eligible_rows": sum(record["eligible_rows"] for record in records),
        "excluded_rows": sum(record["excluded_rows"] for record in records),
        "exclusion_ledger_rows": sum(
            record["exclusion_ledger_rows"] for record in records
        ),
        "excluded_minus_ledger_rows": sum(
            record["excluded_minus_ledger_rows"] for record in records
        ),
        "primary_exclusion_counts": overall_primary,
        "missing_ledger_rows": sum(
            record["missing_ledger_rows"] for record in records
        ),
        "duplicate_ledger_rows": sum(
            record["duplicate_ledger_rows"] for record in records
        ),
        "unexpected_ledger_rows": sum(
            record["unexpected_ledger_rows"] for record in records
        ),
        "unknown_source_table_rows": int(unknown_source_mask.sum()),
        "unknown_source_tables": unknown_source_values,
    }
    overall["reconciliation_status"] = (
        "PASS"
        if (
            all(
                record["reconciliation_status"] == "PASS"
                for record in records
            )
            and overall["unknown_source_table_rows"] == 0
            and not ledger["ledger_id"].isna().any()
        )
        else "FAIL"
    )
    return {
        "table_order": list(COHORT_TABLE_NAMES),
        "tables": records,
        "overall": overall,
    }


def _distribution(values: pd.Series) -> dict[str, float | int] | None:
    if values.empty:
        return None
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    quantiles = numeric.quantile([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    return {
        "min": float(quantiles.loc[0.0]),
        "p01": float(quantiles.loc[0.01]),
        "p05": float(quantiles.loc[0.05]),
        "p50": float(quantiles.loc[0.5]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(quantiles.loc[1.0]),
    }


def _validate_game_splits(source: pd.DataFrame) -> None:
    if source.duplicated(PLAY_KEYS).any():
        duplicates = int(source.duplicated(PLAY_KEYS, keep=False).sum())
        raise AuditError(
            f"Play keys cross weekly partitions or are duplicated: {duplicates}"
        )
    game_splits = source.groupby("game_id", observed=True)["split"].nunique()
    if game_splits.gt(1).any():
        games = sorted(game_splits.loc[game_splits.gt(1)].index.astype(str))
        raise AuditError(f"Games cross chronological splits: {games[:10]}")
    play_splits = source.groupby(PLAY_KEYS, observed=True)["split"].nunique()
    if play_splits.gt(1).any():
        raise AuditError("At least one play crosses chronological splits.")


FULL_RELEASE_REFERENCE = {
    "input_rows": 4_880_579,
    "output_rows": 562_936,
    "games": 272,
    "source_plays": 14_108,
    "target_labelled_plays": 14_108,
    "target_frames_before_exclusion": 396_914,
    "zero_defender_target_frames": 35,
    "descriptive_target_frames_eligible": 396_879,
    "descriptive_defender_pair_rows": 2_662_657,
    "expected_output_groups": 46_045,
    "observed_output_groups": 46_045,
    "missing_expected_output_groups": 0,
    "unmatched_observed_output_groups": 0,
    "input_duplicate_entity_frame_rows": 0,
    "output_duplicate_entity_frame_rows": 0,
    "input_coordinate_missing_rows": 0,
    "output_coordinate_missing_rows": 0,
}


def _reference_reconciliation(
    aggregate: dict[str, Any],
    selected_weeks: tuple[str, ...],
    reference_path: Path | None,
) -> dict[str, Any]:
    if selected_weeks == EXPECTED_WEEKS:
        expected = FULL_RELEASE_REFERENCE
        source = "Milestone 1 full-release evidence"
    elif (
        selected_weeks == ("2023_w01",)
        and reference_path is not None
        and reference_path.is_file()
    ):
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        expected = {
            "input_rows": int(reference["entities"]["input_rows"]),
            "output_rows": int(reference["benchmark"]["rows_processed"]["output"]),
            "source_plays": int(reference["entities"]["input_game_play_pairs"]),
            "target_labelled_plays": int(
                reference["entities"]["plays_with_exactly_one_target"]
            ),
            "target_frames_before_exclusion": int(
                reference["entities"]["player_role_counts"][TARGET_ROLE]
            ),
            "expected_output_groups": int(
                reference["relationships"]["expected_output_groups_from_input"]
            ),
            "observed_output_groups": int(
                reference["relationships"][
                    "output_unique_game_play_player_groups"
                ]
            ),
            "missing_expected_output_groups": int(
                reference["relationships"]["missing_output_groups"]
            ),
            "input_duplicate_entity_frame_rows": int(
                reference["relationships"][
                    "input_duplicate_game_play_player_frame_rows"
                ]
            ),
            "output_duplicate_entity_frame_rows": int(
                reference["relationships"][
                    "output_duplicate_game_play_player_frame_rows"
                ]
            ),
            "input_coordinate_missing_rows": int(
                reference["coordinates"]["missing_input_coordinate_rows"]
            ),
            "output_coordinate_missing_rows": int(
                reference["relationships"]["output_rows_missing_coordinates"]
            ),
        }
        source = str(reference_path)
    else:
        return {
            "source": None,
            "status": "NOT_APPLICABLE",
            "checks": {},
        }
    checks: dict[str, Any] = {}
    for name, expected_value in expected.items():
        observed_value = aggregate.get(name)
        checks[name] = {
            "expected": expected_value,
            "observed": observed_value,
            "matches": observed_value == expected_value,
        }
    failures = [
        name for name, check in checks.items() if not check["matches"]
    ]
    if failures:
        details = ", ".join(
            f"{name}: expected {checks[name]['expected']}, "
            f"observed {checks[name]['observed']}"
            for name in failures
        )
        raise AuditError(f"Milestone 1 reconciliation failed: {details}")
    return {"source": source, "status": "PASS", "checks": checks}
