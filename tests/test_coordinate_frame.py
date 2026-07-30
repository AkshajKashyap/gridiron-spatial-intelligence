import numpy as np
import pandas as pd
import pytest

from gridiron_spatial.coordinate_frame import (
    COORDINATE_TRANSFORM_VERSION,
    add_normalized_coordinates,
)


def _synthetic_frame():
    return pd.DataFrame(
        {
            "play_id": ["a", "b", "c"],
            "x": [10.0, 120.26, 122.0],
            "y": [20.0, 52.12, 10.0],
            "play_direction": ["right", "left", "right"],
            "dir": [360.0, -10.0, np.nan],
            "o": [-10.0, 725.0, 90.0],
            "label": ["nominal", "tolerated", "invalid"],
        },
        index=[7, 3, 9],
    )


def test_adapter_preserves_raw_data_index_and_appends_deterministic_columns():
    frame = _synthetic_frame()
    before = frame.copy(deep=True)

    result = add_normalized_coordinates(frame)

    assert list(result.columns) == [
        *frame.columns,
        "x_norm",
        "y_norm",
        "dir_norm",
        "o_norm",
        "raw_coordinate_class",
        "normalized_coordinate_class",
        "coordinate_transform_version",
        "coordinate_transform_applied",
    ]
    assert result.index.tolist() == [7, 3, 9]
    pd.testing.assert_frame_equal(
        result.loc[:, frame.columns],
        before,
    )
    pd.testing.assert_frame_equal(frame, before)

    np.testing.assert_allclose(result["x_norm"], [10.0, -0.26, 122.0])
    np.testing.assert_allclose(result["y_norm"], [20.0, 1.18, 10.0])
    np.testing.assert_allclose(
        result["dir_norm"].iloc[:2], [0.0, 170.0]
    )
    assert np.isnan(result["dir_norm"].iloc[2])
    np.testing.assert_allclose(result["o_norm"], [350.0, 185.0, 90.0])
    assert result["raw_coordinate_class"].tolist() == [
        "nominal",
        "extended_tolerance",
        "invalid",
    ]
    assert result["normalized_coordinate_class"].tolist() == [
        "nominal",
        "extended_tolerance",
        "invalid",
    ]
    assert result["coordinate_transform_version"].tolist() == [
        COORDINATE_TRANSFORM_VERSION
    ] * 3
    assert result["coordinate_transform_applied"].tolist() == [
        "identity_right",
        "rotate_180_left",
        "identity_right",
    ]


def test_repeated_calls_on_raw_input_are_deterministic_and_do_not_clip():
    frame = _synthetic_frame()

    first = add_normalized_coordinates(frame)
    second = add_normalized_coordinates(frame)

    pd.testing.assert_frame_equal(first, second)
    assert first.loc[9, "x_norm"] == 122.0
    assert first.loc[9, "normalized_coordinate_class"] == "invalid"
    with pytest.raises(ValueError, match="already exist"):
        add_normalized_coordinates(first)


@pytest.mark.parametrize("bad_direction", ["up", None])
def test_unknown_or_missing_play_direction_fails(bad_direction):
    frame = _synthetic_frame()
    frame.loc[3, "play_direction"] = bad_direction

    with pytest.raises(ValueError, match="play_direction"):
        add_normalized_coordinates(frame)


def test_missing_columns_and_malformed_coordinates_fail():
    with pytest.raises(ValueError, match="Missing required"):
        add_normalized_coordinates(_synthetic_frame().drop(columns="x"))
    with pytest.raises(ValueError, match="Missing required"):
        add_normalized_coordinates(_synthetic_frame().drop(columns="dir"))

    malformed = _synthetic_frame()
    malformed["x"] = malformed["x"].astype(object)
    malformed.loc[7, "x"] = "not-numeric"
    with pytest.raises(ValueError, match="numeric"):
        add_normalized_coordinates(malformed)


def test_optional_angle_columns_are_omitted_only_when_explicitly_disabled():
    frame = _synthetic_frame().drop(columns=["dir", "o"])

    result = add_normalized_coordinates(
        frame,
        direction_column=None,
        orientation_column=None,
    )

    assert "dir_norm" not in result
    assert "o_norm" not in result
    assert list(result.columns[-6:]) == [
        "x_norm",
        "y_norm",
        "raw_coordinate_class",
        "normalized_coordinate_class",
        "coordinate_transform_version",
        "coordinate_transform_applied",
    ]
