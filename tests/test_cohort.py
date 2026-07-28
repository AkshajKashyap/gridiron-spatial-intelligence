import pandas as pd

from gridiron_spatial.cohort import (
    COHORT_TABLE_NAMES,
    EXCLUSION_LEDGER_SCHEMA,
    REASON_CODES,
    TABLE_KEY_COLUMNS,
    TABLE_SCHEMAS,
    build_exclusion_ledger,
    build_week_cohorts,
    phase_qualified_entity_frame_keys,
)


STATUS_COLUMNS = [
    *REASON_CODES,
    "primary_exclusion_reason",
    "secondary_exclusion_reasons",
    "eligible",
]
EXPECTED_SCHEMAS = {
    "source_plays": [
        "game_id", "play_id", "phase", "week", "week_number", "split",
        "input_rows", "valid_input_frame_count", "tracked_entity_count",
        "origin_frame", "target_nfl_id", "target_nfl_id_count", "target_rows",
        "play_direction", "direction_value_count", "invalid_direction_rows",
        "invalid_entity_frame_rows", "ambiguous_entity_assignments",
        "metadata_week_matches_file", *STATUS_COLUMNS,
    ],
    "descriptive_target_frames": [
        "game_id", "play_id", "phase", "frame_id", "target_nfl_id", "week",
        "week_number", "split", "frame_raw_rows", "target_row_count",
        "target_entity_count", "valid_target_coordinate_rows",
        "registered_defender_rows", "observed_defender_count",
        "valid_observed_defender_count", "target_duplicate_rows",
        "defender_duplicate_rows", "target_coordinate_missing_rows",
        "defender_coordinate_missing_rows", "target_outside_extended_rows",
        "defender_outside_extended_rows", "target_extended_tolerance_rows",
        "defender_extended_tolerance_rows", "target_invalid_key_rows",
        "defender_invalid_key_rows", *STATUS_COLUMNS,
    ],
    "primary_origins": [
        "game_id", "play_id", "target_nfl_id", "origin_kind", "phase",
        "origin_frame", "relative_frame", "play_direction", "week",
        "week_number", "split", "raw_history_2_frame_count",
        "valid_history_2_frame_count", "raw_history_5_frame_count",
        "valid_history_5_frame_count", "history_2_eligible",
        "history_5_eligible", "five_frame_history_eligible",
        "registered_at_origin", "valid_origin_rows",
        "registered_origin_defender_count",
        "valid_observed_origin_defender_count",
        "observed_origin_defender_ids", "duplicate_at_origin",
        "origin_coordinate_missing", "origin_outside_extended",
        "origin_extended_tolerance", *STATUS_COLUMNS,
    ],
    "trajectory_eligibility": [
        "game_id", "play_id", "nfl_id", "target_nfl_id", "origin_frame",
        "horizon", "phase", "label_phase", "max_input_relative_frame_used",
        "label_start_frame", "label_end_frame", "week", "week_number", "split",
        "player_role", "player_side", "is_target_role", "is_defender",
        "player_to_predict", "declared_output_frames",
        "matched_output_group", "full_output_group_exact",
        "raw_history_2_frame_count", "valid_history_2_frame_count",
        "history_2_eligible", "history_5_eligible", "observed_at_origin",
        "horizon_output_row_count", "horizon_output_frame_count",
        "horizon_frames_contiguous", "horizon_coordinate_missing_rows",
        "horizon_outside_extended_rows", "horizon_extended_tolerance_rows",
        *STATUS_COLUMNS,
    ],
    "future_separation_eligibility": [
        "game_id", "play_id", "target_nfl_id", "origin_frame", "horizon",
        "phase", "label_phase", "max_input_relative_frame_used",
        "label_start_frame", "label_end_frame", "week", "week_number", "split",
        "history_2_eligible", "five_frame_history_eligible",
        "observed_origin_defender_ids", "observed_origin_defender_count",
        "prediction_candidate_origin_defender_count",
        "two_frame_history_origin_defender_count",
        "output_candidate_defender_count", "evaluable_defender_count",
        "evaluable_defender_ids", "removed_defender_count",
        "target_trajectory_eligible", "defender_set_definition",
        *STATUS_COLUMNS,
    ],
    "pair_exclusions": [
        "game_id", "play_id", "phase", "frame_id", "target_nfl_id",
        "defender_nfl_id", "week", "week_number", "split",
        "coordinate_category", "coordinate_missing", "outside_nominal",
        "outside_extended_tolerance", *STATUS_COLUMNS,
    ],
}


def _input_row(
    game_id,
    play_id,
    nfl_id,
    frame_id,
    role,
    side,
    *,
    predict=False,
    declared=None,
    direction="right",
    x=30.0,
    y=20.0,
):
    return {
        "game_id": game_id,
        "play_id": play_id,
        "player_to_predict": predict,
        "nfl_id": nfl_id,
        "frame_id": frame_id,
        "play_direction": direction,
        "player_side": side,
        "player_role": role,
        "x": x,
        "y": y,
        "num_frames_output": declared,
    }


def _output_rows(game_id, play_id, nfl_id, count=5):
    return [
        {
            "game_id": game_id,
            "play_id": play_id,
            "nfl_id": nfl_id,
            "frame_id": frame,
            "x": 35.0 + frame,
            "y": 20.0,
        }
        for frame in range(1, count + 1)
    ]


def _tiny_fixture():
    inputs = []
    for frame in (1, 2):
        inputs.extend(
            [
                _input_row(
                    "1", "10", "101", frame, "Targeted Receiver", "Offense",
                    predict=True, declared=5, x=30.0 + frame,
                ),
                _input_row(
                    "1", "10", "201", frame, "Defensive Coverage", "Defense",
                    predict=True, declared=5, x=32.0 + frame,
                ),
                _input_row(
                    "1", "10", "202", frame, "Defensive Coverage", "Defense",
                    x=34.0 + frame,
                ),
            ]
        )
    inputs.append(
        _input_row(
            "2", "20", "301", 1, "Targeted Receiver", "Offense",
            predict=True, declared=5, direction="left", x=70.0,
        )
    )
    outputs = [
        *_output_rows("1", "10", "101"),
        *_output_rows("1", "10", "201"),
    ]
    supplementary = {("1", "10"), ("2", "20")}
    return pd.DataFrame(inputs), pd.DataFrame(outputs), supplementary


def test_build_week_cohorts_freezes_six_table_interfaces():
    inputs, outputs, supplementary = _tiny_fixture()
    result = build_week_cohorts(inputs, outputs, supplementary, "2023_w01")

    assert COHORT_TABLE_NAMES == tuple(EXPECTED_SCHEMAS)
    assert TABLE_SCHEMAS == EXPECTED_SCHEMAS
    expected_rows = {
        "source_plays": 2,
        "descriptive_target_frames": 3,
        "primary_origins": 2,
        "trajectory_eligibility": 9,
        "future_separation_eligibility": 6,
        "pair_exclusions": 0,
    }
    for name in COHORT_TABLE_NAMES:
        table = getattr(result, name)
        assert isinstance(table, pd.DataFrame)
        assert list(table.columns) == EXPECTED_SCHEMAS[name]
        assert len(table) == expected_rows[name]
        assert not table.duplicated(TABLE_KEY_COLUMNS[name]).any()
        assert str(table["game_id"].dtype) == "string"
        assert str(table["play_id"].dtype) == "string"
        assert str(table["week_number"].dtype) == "Int64"
        assert table["eligible"].dtype == bool
        assert all(table[code].dtype == bool for code in REASON_CODES)


def test_build_week_cohorts_membership_history_and_defender_sets():
    inputs, outputs, supplementary = _tiny_fixture()
    result = build_week_cohorts(inputs, outputs, supplementary, "2023_w01")

    source = result.source_plays.set_index(["game_id", "play_id"])
    assert set(source.index) == {("1", "10"), ("2", "20")}

    descriptive = result.descriptive_target_frames
    valid_frames = descriptive.loc[descriptive["eligible"]]
    assert set(valid_frames["frame_id"].astype(int)) == {1, 2}
    zero_defender = descriptive.loc[
        descriptive["game_id"].eq("2") & descriptive["play_id"].eq("20")
    ].iloc[0]
    assert zero_defender["valid_observed_defender_count"] == 0
    assert bool(zero_defender["C07"])
    assert zero_defender["primary_exclusion_reason"] == "C07"

    origins = result.primary_origins.set_index(["game_id", "play_id"])
    valid_origin = origins.loc[("1", "10")]
    assert valid_origin["origin_frame"] == 2
    assert bool(valid_origin["history_2_eligible"])
    assert not bool(valid_origin["five_frame_history_eligible"])
    assert valid_origin["observed_origin_defender_ids"] == "201|202"

    short_origin = origins.loc[("2", "20")]
    assert short_origin["origin_frame"] == 1
    assert not bool(short_origin["history_2_eligible"])
    assert bool(short_origin["C09"]) and bool(short_origin["C12"])
    assert short_origin["primary_exclusion_reason"] == "C09"
    assert short_origin["secondary_exclusion_reasons"] == "C12"

    trajectories = result.trajectory_eligibility
    target_h5 = trajectories.loc[
        trajectories["game_id"].eq("1")
        & trajectories["play_id"].eq("10")
        & trajectories["nfl_id"].eq("101")
        & trajectories["horizon"].eq(5)
    ].iloc[0]
    assert bool(target_h5["eligible"])

    future_h5 = result.future_separation_eligibility.loc[
        result.future_separation_eligibility["game_id"].eq("1")
        & result.future_separation_eligibility["play_id"].eq("10")
        & result.future_separation_eligibility["horizon"].eq(5)
    ].iloc[0]
    assert future_h5["observed_origin_defender_ids"] == "201|202"
    assert future_h5["evaluable_defender_ids"] == "201"
    assert future_h5["observed_origin_defender_count"] == 2
    assert future_h5["evaluable_defender_count"] == 1
    assert bool(future_h5["eligible"])


def test_input_and_output_numeric_frames_remain_phase_qualified():
    inputs, outputs, _ = _tiny_fixture()
    input_key = phase_qualified_entity_frame_keys(
        inputs.loc[
            inputs["game_id"].eq("1")
            & inputs["play_id"].eq("10")
            & inputs["nfl_id"].eq("101")
            & inputs["frame_id"].eq(1)
        ],
        "input",
    )
    output_key = phase_qualified_entity_frame_keys(
        outputs.loc[
            outputs["game_id"].eq("1")
            & outputs["play_id"].eq("10")
            & outputs["nfl_id"].eq("101")
            & outputs["frame_id"].eq(1)
        ],
        "output",
    )
    combined = pd.concat([input_key, output_key], ignore_index=True)
    assert len(combined) == 2
    assert not combined.duplicated(
        ["game_id", "play_id", "phase", "frame_id", "nfl_id"]
    ).any()
    assert set(combined["phase"]) == {"input", "output"}


def _ledger_fixture_tables():
    inputs, outputs, supplementary = _tiny_fixture()
    result = build_week_cohorts(inputs, outputs, supplementary, "2023_w01")

    source = result.source_plays.copy()
    source.loc[source["game_id"].eq("1"), ["C03", "C07"]] = True

    descriptive = result.descriptive_target_frames.loc[
        (
            result.descriptive_target_frames["game_id"].eq("1")
            & result.descriptive_target_frames["frame_id"].eq(1)
        )
        | result.descriptive_target_frames["game_id"].eq("2")
    ].copy()
    origins = result.primary_origins.loc[
        result.primary_origins["game_id"].eq("2")
    ].copy()
    trajectories = result.trajectory_eligibility.loc[
        result.trajectory_eligibility["game_id"].eq("1")
        & result.trajectory_eligibility["nfl_id"].eq("101")
        & result.trajectory_eligibility["horizon"].eq(10)
    ].copy()
    future = result.future_separation_eligibility.loc[
        result.future_separation_eligibility["game_id"].eq("1")
        & result.future_separation_eligibility["horizon"].eq(10)
    ].copy()

    pair_values = {
        "game_id": "1",
        "play_id": "10",
        "phase": "input",
        "frame_id": 2,
        "target_nfl_id": "101",
        "defender_nfl_id": "299",
        "week": "2023_w01",
        "week_number": 1,
        "split": "development_train",
        "coordinate_category": "invalid",
        "coordinate_missing": True,
        "outside_nominal": False,
        "outside_extended_tolerance": False,
        **{code: False for code in REASON_CODES},
        "primary_exclusion_reason": "",
        "secondary_exclusion_reasons": "",
        "eligible": True,
    }
    pair_values["C06"] = True
    pairs = pd.DataFrame([pair_values]).reindex(
        columns=TABLE_SCHEMAS["pair_exclusions"]
    )
    return {
        "source_plays": source,
        "descriptive_target_frames": descriptive,
        "primary_origins": origins,
        "trajectory_eligibility": trajectories,
        "future_separation_eligibility": future,
        "pair_exclusions": pairs,
    }


def test_build_exclusion_ledger_covers_each_unit_once_and_reconciles():
    tables = _ledger_fixture_tables()
    ledger = build_exclusion_ledger(
        tables["source_plays"],
        tables["descriptive_target_frames"],
        tables["primary_origins"],
        tables["trajectory_eligibility"],
        tables["future_separation_eligibility"],
        tables["pair_exclusions"],
    )

    assert list(ledger.columns) == EXCLUSION_LEDGER_SCHEMA
    assert len(ledger) == 6
    assert ledger["ledger_id"].is_unique
    assert not ledger["ledger_id"].str.contains("<NA>", regex=False).any()
    assert set(zip(ledger["source_table"], ledger["unit_type"])) == {
        ("source_plays", "source_play"),
        ("descriptive_target_frames", "target_frame"),
        ("primary_origins", "primary_origin"),
        ("trajectory_eligibility", "trajectory_entity_horizon"),
        ("future_separation_eligibility", "target_horizon"),
        ("pair_exclusions", "target_defender_pair"),
    }

    expected_counts = {
        name: int(table[list(REASON_CODES)].any(axis=1).sum())
        for name, table in tables.items()
    }
    assert ledger["source_table"].value_counts().to_dict() == expected_counts

    source_row = ledger.loc[ledger["source_table"].eq("source_plays")].iloc[0]
    assert source_row["primary_exclusion_reason"] == "C03"
    assert source_row["secondary_exclusion_reasons"] == "C07"
    assert pd.isna(source_row["phase"])
    assert pd.isna(source_row["frame_id"])
    assert pd.isna(source_row["nfl_id"])
    assert pd.isna(source_row["target_nfl_id"])
    assert pd.isna(source_row["origin_frame"])
    assert pd.isna(source_row["horizon"])

    assert not (
        ledger["source_table"].eq("source_plays")
        & ledger["game_id"].eq("2")
    ).any()
    assert not (
        ledger["source_table"].eq("descriptive_target_frames")
        & ledger["game_id"].eq("1")
    ).any()


def test_build_exclusion_ledger_empty_schema_and_dtypes_are_stable():
    tables = _ledger_fixture_tables()
    empty = {
        name: table.iloc[0:0].copy() for name, table in tables.items()
    }
    ledger = build_exclusion_ledger(
        empty["source_plays"],
        empty["descriptive_target_frames"],
        empty["primary_origins"],
        empty["trajectory_eligibility"],
        empty["future_separation_eligibility"],
        empty["pair_exclusions"],
    )

    assert ledger.empty
    assert list(ledger.columns) == EXCLUSION_LEDGER_SCHEMA
    assert str(ledger["ledger_id"].dtype) == "string"
    assert str(ledger["game_id"].dtype) == "string"
    assert str(ledger["frame_id"].dtype) == "Int64"
    assert str(ledger["horizon"].dtype) == "Int64"
    assert ledger["C01"].dtype == bool
