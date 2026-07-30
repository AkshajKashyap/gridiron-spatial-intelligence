"""Frozen schema and reconciliation for normalized entity-frame rows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .coordinate_frame import COORDINATE_TRANSFORM_VERSION


NORMALIZED_ENTITY_FRAME_KEY = [
    "game_id",
    "play_id",
    "phase",
    "frame_id",
    "nfl_id",
]
NORMALIZED_ENTITY_FRAME_SCHEMA = [
    "game_id",
    "play_id",
    "phase",
    "frame_id",
    "nfl_id",
    "week",
    "week_number",
    "split",
    "player_side",
    "player_role",
    "player_position",
    "player_to_predict",
    "play_direction",
    "x",
    "y",
    "dir",
    "o",
    "x_norm",
    "y_norm",
    "dir_norm",
    "o_norm",
    "raw_coordinate_class",
    "normalized_coordinate_class",
    "coordinate_transform_version",
    "coordinate_transform_applied",
]
_STRING_COLUMNS = [
    "game_id",
    "play_id",
    "phase",
    "nfl_id",
    "week",
    "split",
    "player_side",
    "player_role",
    "player_position",
    "play_direction",
    "raw_coordinate_class",
    "normalized_coordinate_class",
    "coordinate_transform_version",
    "coordinate_transform_applied",
]
_INTEGER_COLUMNS = ["frame_id", "week_number"]
_NUMERIC_COLUMNS = [
    "x",
    "y",
    "dir",
    "o",
    "x_norm",
    "y_norm",
    "dir_norm",
    "o_norm",
]
_COORDINATE_CLASSES = ("nominal", "extended_tolerance", "invalid")
_TRANSFORMS = ("identity_right", "rotate_180_left")
_PHASES = ("input", "output")


def _unsupported(series: pd.Series, allowed: tuple[str, ...]) -> list[str]:
    values = series.astype("string")
    return sorted(
        {"<missing>" if pd.isna(value) else str(value) for value in values}
        - set(allowed)
    )


def freeze_normalized_entity_frames(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the exact normalized entity-frame schema and dtypes."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = [
        column
        for column in NORMALIZED_ENTITY_FRAME_SCHEMA
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing frozen normalized columns: {missing}")

    result = frame.loc[:, NORMALIZED_ENTITY_FRAME_SCHEMA].copy(deep=True)
    try:
        for column in _STRING_COLUMNS:
            result[column] = result[column].astype("string")
        for column in _INTEGER_COLUMNS:
            result[column] = pd.to_numeric(
                result[column], errors="raise"
            ).astype("Int64")
        for column in _NUMERIC_COLUMNS:
            result[column] = pd.to_numeric(
                result[column], errors="raise"
            ).astype("Float64")
        result["player_to_predict"] = result["player_to_predict"].astype(
            "boolean"
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Normalized entity-frame dtype failure: {error}") from error

    if result[NORMALIZED_ENTITY_FRAME_KEY].isna().any(axis=None):
        raise ValueError("Frozen normalized key fields may not be null")
    if result.duplicated(NORMALIZED_ENTITY_FRAME_KEY).any():
        raise ValueError("Duplicate phase-qualified entity-frame keys")

    validations = (
        ("phase", _PHASES),
        ("raw_coordinate_class", _COORDINATE_CLASSES),
        ("normalized_coordinate_class", _COORDINATE_CLASSES),
        ("coordinate_transform_version", (COORDINATE_TRANSFORM_VERSION,)),
        ("coordinate_transform_applied", _TRANSFORMS),
    )
    for column, allowed in validations:
        invalid = _unsupported(result[column], allowed)
        if invalid:
            raise ValueError(f"Unsupported {column} values: {invalid}")
    return result


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing reconciliation columns: {missing}")


def _canonical_keys(frame: pd.DataFrame) -> list[tuple[str, str, str, int, str]]:
    if frame[NORMALIZED_ENTITY_FRAME_KEY].isna().any(axis=None):
        raise ValueError("Reconciliation key fields may not be null")
    frame_ids = pd.to_numeric(frame["frame_id"], errors="raise")
    if not frame_ids.eq(frame_ids.astype("int64")).all():
        raise ValueError("frame_id must contain integral values")
    return list(
        zip(
            frame["game_id"].astype("string").astype(str),
            frame["play_id"].astype("string").astype(str),
            frame["phase"].astype("string").astype(str),
            frame_ids.astype("int64").tolist(),
            frame["nfl_id"].astype("string").astype(str),
        )
    )


def _key_records(
    keys: set[tuple[str, str, str, int, str]],
) -> list[dict[str, Any]]:
    return [
        dict(zip(NORMALIZED_ENTITY_FRAME_KEY, key))
        for key in sorted(keys)
    ]


def _raw_field_mismatch_counts(
    raw_frame: pd.DataFrame,
    normalized_frame: pd.DataFrame,
    raw_keys: list[tuple[str, str, str, int, str]],
    normalized_keys: list[tuple[str, str, str, int, str]],
    raw_fields: list[str],
) -> dict[str, int]:
    """Compare immutable fields after a deduplicated one-to-one key alignment."""

    raw_compare = raw_frame.loc[:, raw_fields].copy()
    raw_compare.insert(
        0, "_alignment_key", pd.Series(raw_keys, index=raw_frame.index)
    )
    normalized_compare = normalized_frame.loc[:, raw_fields].copy()
    normalized_compare.insert(
        0,
        "_alignment_key",
        pd.Series(normalized_keys, index=normalized_frame.index),
    )
    raw_compare = raw_compare.drop_duplicates(
        "_alignment_key", keep="first"
    )
    normalized_compare = normalized_compare.drop_duplicates(
        "_alignment_key", keep="first"
    )
    aligned = raw_compare.merge(
        normalized_compare,
        on="_alignment_key",
        how="inner",
        suffixes=("_raw", "_normalized"),
        sort=False,
        validate="one_to_one",
    )

    mismatch_counts: dict[str, int] = {}
    for field in raw_fields:
        raw_values = aligned[f"{field}_raw"]
        normalized_values = aligned[f"{field}_normalized"]
        matches = raw_values.eq(normalized_values) | (
            raw_values.isna() & normalized_values.isna()
        )
        mismatch_counts[field] = int((~matches.fillna(False)).sum())
    return mismatch_counts


def _ordered_counts(
    series: pd.Series,
    preferred: tuple[str, ...],
) -> dict[str, int]:
    values = series.astype("string").fillna("<missing>")
    observed = set(values.astype(str))
    order = [*preferred, *sorted(observed - set(preferred))]
    return {
        value: int(values.eq(value).sum())
        for value in order
        if value in observed or value in preferred
    }


def reconcile_normalized_entity_frames(
    raw_frame: pd.DataFrame,
    normalized_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Return deterministic diagnostics without repairing either input."""

    raw_fields = ["x", "y", "dir", "o", "play_direction", "phase"]
    _require_columns(
        raw_frame,
        [*NORMALIZED_ENTITY_FRAME_KEY, *raw_fields],
        "raw_frame",
    )
    _require_columns(
        normalized_frame,
        NORMALIZED_ENTITY_FRAME_SCHEMA,
        "normalized_frame",
    )
    raw_keys = _canonical_keys(raw_frame)
    normalized_keys = _canonical_keys(normalized_frame)
    raw_key_set = set(raw_keys)
    normalized_key_set = set(normalized_keys)
    duplicate_raw = int(
        pd.Series(raw_keys, dtype=object).duplicated(keep=False).sum()
    )
    duplicate_normalized = int(
        pd.Series(normalized_keys, dtype=object).duplicated(keep=False).sum()
    )

    mismatch_counts = _raw_field_mismatch_counts(
        raw_frame,
        normalized_frame,
        raw_keys,
        normalized_keys,
        raw_fields,
    )

    coordinate_counts = {
        column: _ordered_counts(
            normalized_frame[column], _COORDINATE_CLASSES
        )
        for column in (
            "raw_coordinate_class",
            "normalized_coordinate_class",
        )
    }
    transform_counts = _ordered_counts(
        normalized_frame["coordinate_transform_applied"], _TRANSFORMS
    )
    unsupported_phases = _unsupported(normalized_frame["phase"], _PHASES)
    unsupported_raw_classes = _unsupported(
        normalized_frame["raw_coordinate_class"], _COORDINATE_CLASSES
    )
    unsupported_normalized_classes = _unsupported(
        normalized_frame["normalized_coordinate_class"], _COORDINATE_CLASSES
    )
    unsupported_versions = _unsupported(
        normalized_frame["coordinate_transform_version"],
        (COORDINATE_TRANSFORM_VERSION,),
    )
    unsupported_transforms = _unsupported(
        normalized_frame["coordinate_transform_applied"], _TRANSFORMS
    )
    expected_transform = normalized_frame["play_direction"].astype(
        "string"
    ).map({"right": "identity_right", "left": "rotate_180_left"})
    transform_direction_mismatches = int(
        (
            expected_transform.isna()
            | normalized_frame["coordinate_transform_applied"]
            .astype("string")
            .ne(expected_transform)
        ).sum()
    )
    invalid_to_nominal = int(
        (
            normalized_frame["raw_coordinate_class"]
            .astype("string")
            .eq("invalid")
            & normalized_frame["normalized_coordinate_class"]
            .astype("string")
            .eq("nominal")
        ).sum()
    )
    index_preserved = raw_frame.index.equals(normalized_frame.index)
    row_order_preserved = raw_keys == normalized_keys
    missing = raw_key_set - normalized_key_set
    unexpected = normalized_key_set - raw_key_set

    failures = [
        len(raw_frame) != len(normalized_frame),
        bool(missing),
        bool(unexpected),
        duplicate_raw > 0,
        duplicate_normalized > 0,
        any(mismatch_counts.values()),
        not index_preserved,
        not row_order_preserved,
        bool(unsupported_phases),
        bool(unsupported_raw_classes),
        bool(unsupported_normalized_classes),
        bool(unsupported_versions),
        bool(unsupported_transforms),
        transform_direction_mismatches > 0,
        invalid_to_nominal > 0,
    ]
    return {
        "status": "FAIL" if any(failures) else "PASS",
        "raw_rows": int(len(raw_frame)),
        "normalized_rows": int(len(normalized_frame)),
        "raw_unique_keys": int(len(raw_key_set)),
        "normalized_unique_keys": int(len(normalized_key_set)),
        "missing_normalized_keys": _key_records(missing),
        "unexpected_normalized_keys": _key_records(unexpected),
        "duplicate_raw_keys": duplicate_raw,
        "duplicate_normalized_keys": duplicate_normalized,
        "raw_field_mismatch_counts": mismatch_counts,
        "index_preserved": index_preserved,
        "row_order_preserved": row_order_preserved,
        "coordinate_class_counts": coordinate_counts,
        "transform_applied_counts": transform_counts,
        "unsupported_phase_values": unsupported_phases,
        "unsupported_coordinate_class_values": {
            "raw_coordinate_class": unsupported_raw_classes,
            "normalized_coordinate_class": unsupported_normalized_classes,
        },
        "unsupported_transform_versions": unsupported_versions,
        "unsupported_transform_applied_values": unsupported_transforms,
        "transform_direction_mismatches": transform_direction_mismatches,
        "invalid_raw_to_nominal_normalized": invalid_to_nominal,
    }
