import numpy as np
import pytest

from gridiron_spatial.coordinates import (
    BOUNDARY_TOLERANCE,
    FIELD_LENGTH,
    FIELD_WIDTH,
    classify_coordinates,
    normalize_angles,
    normalize_positions,
    normalize_vectors,
    reverse_angles,
    reverse_positions,
)


def test_rightward_identity_and_leftward_scalar_position_transform():
    assert normalize_positions(31.5, 12.25, "right") == (31.5, 12.25)
    assert normalize_positions(31.5, 12.25, "left") == (
        FIELD_LENGTH - 31.5,
        FIELD_WIDTH - 12.25,
    )
    assert reverse_positions(
        FIELD_LENGTH - 31.5,
        FIELD_WIDTH - 12.25,
        "left",
    ) == pytest.approx((31.5, 12.25))


def test_position_round_trip_and_pairwise_distance_preservation():
    x = np.array([12.0, 36.5, 120.26])
    y = np.array([8.0, 40.5, 52.12])
    x_before = x.copy()
    y_before = y.copy()

    x_normalized, y_normalized = normalize_positions(x, y, "left")
    x_raw, y_raw = reverse_positions(x_normalized, y_normalized, "left")

    np.testing.assert_allclose(x_raw, x, atol=1e-5)
    np.testing.assert_allclose(y_raw, y, atol=1e-5)
    raw_distance = np.hypot(x[1] - x[0], y[1] - y[0])
    normalized_distance = np.hypot(
        x_normalized[1] - x_normalized[0],
        y_normalized[1] - y_normalized[0],
    )
    assert normalized_distance == pytest.approx(raw_distance, abs=1e-5)
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(y, y_before)


def test_vector_norm_and_mixed_direction_array_behavior():
    vx = np.array([3.0, 3.0])
    vy = np.array([4.0, 4.0])
    vx_before = vx.copy()
    vy_before = vy.copy()

    x_normalized, y_normalized = normalize_vectors(
        vx, vy, np.array(["right", "left"])
    )

    np.testing.assert_array_equal(x_normalized, [3.0, -3.0])
    np.testing.assert_array_equal(y_normalized, [4.0, -4.0])
    np.testing.assert_allclose(
        np.hypot(x_normalized, y_normalized),
        np.hypot(vx, vy),
        atol=1e-5,
    )
    np.testing.assert_array_equal(vx, vx_before)
    np.testing.assert_array_equal(vy, vy_before)


def test_angle_wrapping_round_trip_arrays_and_nulls():
    raw = np.array([360.0, -10.0, 725.0, np.nan])
    canonical = normalize_angles(raw, "right")
    np.testing.assert_allclose(
        canonical[:3], [0.0, 350.0, 5.0], atol=1e-5
    )
    assert np.isnan(canonical[3])

    left_normalized = normalize_angles(raw, "left")
    restored = reverse_angles(left_normalized, "left")
    np.testing.assert_allclose(restored[:3], canonical[:3], atol=1e-5)
    assert np.isnan(restored[3])
    assert normalize_angles(-10.0, "right") == 350.0
    assert reverse_angles(normalize_angles(725.0, "left"), "left") == 5.0


def test_coordinate_classification_bounds_and_known_tolerated_examples():
    assert BOUNDARY_TOLERANCE == 1.0
    assert classify_coordinates(0.0, 0.0) == "nominal"
    assert classify_coordinates(120.0, 53.3) == "nominal"
    assert classify_coordinates(-0.5, 20.0) == "extended_tolerance"
    assert classify_coordinates(121.0, 54.3) == "extended_tolerance"

    known = classify_coordinates(
        np.array([120.26, 120.57, 120.83]),
        np.array([52.12, 52.96, 53.72]),
    )
    np.testing.assert_array_equal(
        known,
        np.array(["extended_tolerance"] * 3, dtype=object),
    )

    invalid = classify_coordinates(
        np.array([-1.01, 121.01, 10.0, np.nan, np.inf]),
        np.array([20.0, 20.0, 54.31, 10.0, 10.0]),
    )
    np.testing.assert_array_equal(
        invalid, np.array(["invalid"] * 5, dtype=object)
    )


def test_invalid_direction_shapes_and_nonnumeric_values_fail_clearly():
    with pytest.raises(ValueError, match="play_direction"):
        normalize_positions(1.0, 2.0, "up")
    with pytest.raises(ValueError, match="coordinate shape"):
        normalize_positions([1.0, 2.0], [3.0, 4.0], ["right"])
    with pytest.raises(ValueError, match="identical shapes"):
        normalize_vectors([1.0], [2.0, 3.0], "right")
    with pytest.raises(ValueError, match="numeric"):
        classify_coordinates(["bad"], [2.0])
    with pytest.raises(ValueError, match="finite"):
        normalize_angles(np.inf, "right")
