"""Copy-on-write DataFrame adapter for validated coordinate transforms."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .coordinates import (
    classify_coordinates,
    normalize_angles,
    normalize_positions,
)


COORDINATE_TRANSFORM_VERSION = "nfl_common_direction_v1"
_FIXED_OUTPUT_COLUMNS = (
    "x_norm",
    "y_norm",
    "raw_coordinate_class",
    "normalized_coordinate_class",
    "coordinate_transform_version",
    "coordinate_transform_applied",
)


def _numeric_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    try:
        numeric = pd.to_numeric(frame[column], errors="raise")
        return numeric.to_numpy(dtype=float, na_value=np.nan, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Column {column!r} must contain only numeric or missing values"
        ) from error


def add_normalized_coordinates(
    frame: pd.DataFrame,
    *,
    x_column: str = "x",
    y_column: str = "y",
    play_direction_column: str = "play_direction",
    direction_column: str | None = "dir",
    orientation_column: str | None = "o",
) -> pd.DataFrame:
    """Return a new frame with normalized spatial fields and provenance."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.columns.duplicated().any():
        raise ValueError("Input DataFrame contains duplicate column names")

    required = [x_column, y_column, play_direction_column]
    required.extend(
        column
        for column in (direction_column, orientation_column)
        if column is not None
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    angle_outputs = []
    if direction_column is not None:
        angle_outputs.append("dir_norm")
    if orientation_column is not None:
        angle_outputs.append("o_norm")
    output_columns = [
        "x_norm",
        "y_norm",
        *angle_outputs,
        "raw_coordinate_class",
        "normalized_coordinate_class",
        "coordinate_transform_version",
        "coordinate_transform_applied",
    ]
    collisions = [
        column for column in output_columns if column in frame.columns
    ]
    if collisions:
        raise ValueError(
            f"Normalized output columns already exist: {collisions}"
        )

    x_values = _numeric_values(frame, x_column)
    y_values = _numeric_values(frame, y_column)
    directions = frame[play_direction_column].to_numpy(copy=True)
    x_normalized, y_normalized = normalize_positions(
        x_values, y_values, directions
    )

    result = frame.copy(deep=True)
    result["x_norm"] = pd.Series(x_normalized, index=frame.index, dtype=float)
    result["y_norm"] = pd.Series(y_normalized, index=frame.index, dtype=float)
    if direction_column is not None:
        result["dir_norm"] = pd.Series(
            normalize_angles(
                _numeric_values(frame, direction_column),
                directions,
            ),
            index=frame.index,
            dtype=float,
        )
    if orientation_column is not None:
        result["o_norm"] = pd.Series(
            normalize_angles(
                _numeric_values(frame, orientation_column),
                directions,
            ),
            index=frame.index,
            dtype=float,
        )
    result["raw_coordinate_class"] = pd.Series(
        classify_coordinates(x_values, y_values),
        index=frame.index,
        dtype="string",
    )
    result["normalized_coordinate_class"] = pd.Series(
        classify_coordinates(x_normalized, y_normalized),
        index=frame.index,
        dtype="string",
    )
    result["coordinate_transform_version"] = pd.Series(
        COORDINATE_TRANSFORM_VERSION,
        index=frame.index,
        dtype="string",
    )
    result["coordinate_transform_applied"] = pd.Series(
        np.where(
            directions == "right",
            "identity_right",
            "rotate_180_left",
        ),
        index=frame.index,
        dtype="string",
    )
    return result
