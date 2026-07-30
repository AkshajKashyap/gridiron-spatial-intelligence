"""Pure, reversible coordinate transforms for target-centric analysis."""

from __future__ import annotations

from typing import Any

import numpy as np


FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.3
BOUNDARY_TOLERANCE = 1.0


def _numeric_array(value: Any, name: str) -> np.ndarray:
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only numeric values") from error


def _paired_arrays(
    first: Any,
    second: Any,
    first_name: str,
    second_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    first_array = _numeric_array(first, first_name)
    second_array = _numeric_array(second, second_name)
    if first_array.shape != second_array.shape:
        raise ValueError(
            f"{first_name} and {second_name} must have identical shapes; "
            f"got {first_array.shape} and {second_array.shape}"
        )
    return first_array, second_array


def _left_mask(play_direction: Any, shape: tuple[int, ...]) -> bool | np.ndarray:
    directions = np.asarray(play_direction, dtype=object)
    if directions.ndim == 0:
        direction = directions.item()
        if direction not in {"left", "right"}:
            raise ValueError(
                "play_direction must be exactly 'left' or 'right'; "
                f"got {direction!r}"
            )
        return direction == "left"
    if directions.shape != shape:
        raise ValueError(
            "play_direction must be scalar or match the coordinate shape; "
            f"got {directions.shape} and {shape}"
        )
    valid = np.isin(directions, ("left", "right"))
    if not bool(np.all(valid)):
        invalid = sorted({repr(value) for value in directions[~valid].flat})
        raise ValueError(
            "play_direction must contain only 'left' or 'right'; "
            f"invalid values: {invalid}"
        )
    return directions == "left"


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    return float(value) if value.ndim == 0 else value


def normalize_positions(
    x: Any,
    y: Any,
    play_direction: Any,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Rotate leftward positions so normalized offense attacks increasing x."""

    x_values, y_values = _paired_arrays(x, y, "x", "y")
    left = _left_mask(play_direction, x_values.shape)
    x_normalized = np.where(left, FIELD_LENGTH - x_values, x_values)
    y_normalized = np.where(left, FIELD_WIDTH - y_values, y_values)
    return _scalar_or_array(x_normalized), _scalar_or_array(y_normalized)


def reverse_positions(
    x_normalized: Any,
    y_normalized: Any,
    play_direction: Any,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Reverse normalized positions to their raw coordinate convention."""

    return normalize_positions(x_normalized, y_normalized, play_direction)


def normalize_angles(
    angle: Any,
    play_direction: Any,
) -> float | np.ndarray:
    """Canonicalize angles to [0, 360), rotating leftward values by 180."""

    values = _numeric_array(angle, "angle")
    if bool(np.isinf(values).any()):
        raise ValueError("angle must be finite or missing")
    left = _left_mask(play_direction, values.shape)
    normalized = np.mod(values + np.where(left, 180.0, 0.0), 360.0)
    return _scalar_or_array(normalized)


def reverse_angles(
    angle_normalized: Any,
    play_direction: Any,
) -> float | np.ndarray:
    """Recover the canonical raw angle from a normalized angle."""

    return normalize_angles(angle_normalized, play_direction)


def normalize_vectors(
    x_component: Any,
    y_component: Any,
    play_direction: Any,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Rotate explicit vector components without changing vector magnitude."""

    x_values, y_values = _paired_arrays(
        x_component,
        y_component,
        "x_component",
        "y_component",
    )
    left = _left_mask(play_direction, x_values.shape)
    x_normalized = np.where(left, -x_values, x_values)
    y_normalized = np.where(left, -y_values, y_values)
    return _scalar_or_array(x_normalized), _scalar_or_array(y_normalized)


def classify_coordinates(
    x: Any,
    y: Any,
) -> str | np.ndarray:
    """Classify raw coordinates without clipping or repairing any value."""

    x_values, y_values = _paired_arrays(x, y, "x", "y")
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    nominal = (
        finite
        & (x_values >= 0.0)
        & (x_values <= FIELD_LENGTH)
        & (y_values >= 0.0)
        & (y_values <= FIELD_WIDTH)
    )
    extended = (
        finite
        & (x_values >= -BOUNDARY_TOLERANCE)
        & (x_values <= FIELD_LENGTH + BOUNDARY_TOLERANCE)
        & (y_values >= -BOUNDARY_TOLERANCE)
        & (y_values <= FIELD_WIDTH + BOUNDARY_TOLERANCE)
    )
    classifications = np.full(x_values.shape, "invalid", dtype=object)
    classifications[extended] = "extended_tolerance"
    classifications[nominal] = "nominal"
    if classifications.ndim == 0:
        return str(classifications.item())
    return classifications
