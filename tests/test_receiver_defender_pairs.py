import pandas as pd
import pytest

from gridiron_spatial.receiver_defender_pairs import (
    PAIR_COLUMNS,
    build_receiver_defender_pairs,
)


def _fixture():
    rows = []

    def add(phase, frame, nfl_id, x, y, side, role):
        rows.append(
            {
                "game_id": "1",
                "play_id": "10",
                "phase": phase,
                "frame_id": frame,
                "nfl_id": nfl_id,
                "week": "2023_w01",
                "split": "development_train",
                "player_side": side,
                "player_role": role,
                "x_norm": x,
                "y_norm": y,
                "normalized_coordinate_class": "nominal",
            }
        )

    add("input", 5, "T", 0.0, 0.0, "Offense", "Targeted Receiver")
    add("input", 5, "D1", 3.0, 4.0, "Defense", "Defensive Coverage")
    add("input", 5, "D2", 0.0, 10.0, "Defense", "Defensive Coverage")
    future_positions = {
        5: {"T": (0, 0), "D1": (6, 8), "D2": (0, 5)},
        10: {"T": (0, 0), "D1": (3, 4), "D2": (0, 12)},
        15: {"T": (0, 0), "D1": (0, 2), "D2": (0, 15)},
    }
    for horizon, positions in future_positions.items():
        for nfl_id, (x, y) in positions.items():
            is_target = nfl_id == "T"
            add(
                "output",
                horizon,
                nfl_id,
                x,
                y,
                "Offense" if is_target else "Defense",
                "Targeted Receiver" if is_target else "Defensive Coverage",
            )
    tracking = pd.DataFrame(rows)
    origins = pd.DataFrame(
        [
            {
                "game_id": "1",
                "play_id": "10",
                "target_nfl_id": "T",
                "origin_frame": 5,
                "week": "2023_w01",
                "split": "development_train",
                "observed_origin_defender_ids": "D2|D1",
                "eligible": True,
            }
        ]
    )
    trajectories = []
    for horizon in (5, 10, 15):
        for nfl_id in ("T", "D1", "D2"):
            trajectories.append(
                {
                    "game_id": "1",
                    "play_id": "10",
                    "nfl_id": nfl_id,
                    "target_nfl_id": "T",
                    "origin_frame": 5,
                    "horizon": horizon,
                    "label_phase": "output",
                    "label_end_frame": horizon,
                    "is_target_role": nfl_id == "T",
                    "is_defender": nfl_id != "T",
                    "eligible": True,
                }
            )
    future = pd.DataFrame(
        [
            {
                "game_id": "1",
                "play_id": "10",
                "target_nfl_id": "T",
                "origin_frame": 5,
                "horizon": horizon,
                "evaluable_defender_ids": "D2|D1",
                "evaluable_defender_count": 2,
                "eligible": True,
            }
            for horizon in (5, 10, 15)
        ]
    )
    return tracking, origins, pd.DataFrame(trajectories), future


def test_all_observed_defenders_geometry_horizons_order_and_nonmutation():
    tracking, origins, trajectories, future = _fixture()
    copies = [
        frame.copy(deep=True)
        for frame in (tracking, origins, trajectories, future)
    ]

    result = build_receiver_defender_pairs(
        tracking, origins, trajectories, future
    )

    assert list(result.pairs.columns) == PAIR_COLUMNS
    assert len(result.origin_pairs) == 2
    assert len(result.pairs) == 6
    assert result.pairs[["defender_nfl_id", "horizon"]].values.tolist() == [
        ["D1", 5],
        ["D1", 10],
        ["D1", 15],
        ["D2", 5],
        ["D2", 10],
        ["D2", 15],
    ]
    d1_origin = result.origin_pairs.loc[
        result.origin_pairs["defender_nfl_id"].eq("D1")
    ].iloc[0]
    assert d1_origin["dx"] == 3.0
    assert d1_origin["dy"] == 4.0
    assert d1_origin["separation_origin"] == 5.0
    d1_h5 = result.pairs.loc[
        result.pairs["defender_nfl_id"].eq("D1")
        & result.pairs["horizon"].eq(5)
    ].iloc[0]
    d2_h5 = result.pairs.loc[
        result.pairs["defender_nfl_id"].eq("D2")
        & result.pairs["horizon"].eq(5)
    ].iloc[0]
    assert d1_h5["separation_future"] == 10.0
    assert d1_h5["separation_change"] == 5.0
    assert d2_h5["separation_future"] == 5.0
    assert d2_h5["separation_change"] == -5.0
    assert set(result.pairs["horizon"]) == {5, 10, 15}
    assert result.diagnostics["eligibility_mismatch_count"] == 0
    assert result.diagnostics["duplicate_pair_keys"] == 0
    for frame, before in zip(
        (tracking, origins, trajectories, future), copies, strict=True
    ):
        pd.testing.assert_frame_equal(frame, before)


def test_missing_future_target_or_defender_is_not_calculated():
    tracking, origins, trajectories, future = _fixture()
    missing_target = tracking.loc[
        ~(
            tracking["phase"].eq("output")
            & tracking["frame_id"].eq(10)
            & tracking["nfl_id"].eq("T")
        )
    ]
    result = build_receiver_defender_pairs(
        missing_target, origins, trajectories, future
    )
    assert not result.pairs["horizon"].eq(10).any()
    assert result.diagnostics["unavailable_future_counts"][
        "target_tracking"
    ] == 2

    missing_defender = tracking.loc[
        ~(
            tracking["phase"].eq("output")
            & tracking["frame_id"].eq(15)
            & tracking["nfl_id"].eq("D1")
        )
    ]
    result = build_receiver_defender_pairs(
        missing_defender, origins, trajectories, future
    )
    assert not (
        result.pairs["horizon"].eq(15)
        & result.pairs["defender_nfl_id"].eq("D1")
    ).any()
    assert result.diagnostics["unavailable_future_counts"][
        "defender_tracking"
    ] == 1


@pytest.mark.parametrize("entity", ["target", "defender"])
def test_duplicate_target_and_defender_rows_are_rejected(entity):
    tracking, origins, trajectories, future = _fixture()
    nfl_id = "T" if entity == "target" else "D1"
    duplicate = tracking.loc[
        tracking["phase"].eq("input") & tracking["nfl_id"].eq(nfl_id)
    ]
    tracking = pd.concat([tracking, duplicate], ignore_index=True)
    with pytest.raises(ValueError, match=f"Duplicate {entity}"):
        build_receiver_defender_pairs(
            tracking, origins, trajectories, future
        )


def test_target_self_pair_and_unsupported_defender_role_are_rejected():
    tracking, origins, trajectories, future = _fixture()
    self_pair = origins.copy()
    self_pair["observed_origin_defender_ids"] = "T"
    with pytest.raises(ValueError, match="self-pairs"):
        build_receiver_defender_pairs(
            tracking, self_pair, trajectories, future
        )

    unsupported = tracking.copy()
    unsupported.loc[
        unsupported["nfl_id"].eq("D1"), "player_role"
    ] = "Unknown Defender"
    with pytest.raises(ValueError, match="Unsupported defender"):
        build_receiver_defender_pairs(
            unsupported, origins, trajectories, future
        )
