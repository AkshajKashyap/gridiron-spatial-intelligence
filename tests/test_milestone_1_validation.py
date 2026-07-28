import csv
from pathlib import Path

from gridiron_spatial.milestone_1_validation import (
    INPUT_COLUMNS,
    OUTPUT_COLUMNS,
    _stream_inputs_fast,
    _stream_outputs_fast,
    run_milestone_1_validation,
)


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _input_row(
    game_id: str,
    play_id: str,
    nfl_id: str,
    frame_id: int,
    *,
    role: str,
    side: str,
    predict: str = "False",
    expected_frames: str = "",
) -> dict[str, str]:
    return {
        "game_id": game_id,
        "play_id": play_id,
        "player_to_predict": predict,
        "nfl_id": nfl_id,
        "frame_id": str(frame_id),
        "play_direction": "right",
        "absolute_yardline_number": "50",
        "player_name": nfl_id,
        "player_height": "72",
        "player_weight": "200",
        "player_birth_date": "1995-01-01",
        "player_position": "WR" if side == "Offense" else "CB",
        "player_side": side,
        "player_role": role,
        "x": str(30 + frame_id),
        "y": "20",
        "s": "1.0",
        "a": "0.0",
        "dir": "90.0",
        "o": "90.0",
        "num_frames_output": expected_frames,
        "ball_land_x": "40",
        "ball_land_y": "25",
    }


def _output_row(game_id: str, play_id: str, nfl_id: str, frame_id: int, *, x: str = "40", y: str = "25") -> dict[str, str]:
    return {
        "game_id": game_id,
        "play_id": play_id,
        "nfl_id": nfl_id,
        "frame_id": str(frame_id),
        "x": x,
        "y": y,
    }


def _supplementary_row(game_id: str, play_id: str, week: str) -> dict[str, str]:
    return {
        "game_id": game_id,
        "season": "2023",
        "week": week,
        "play_id": play_id,
        "possession_team": "AAA",
        "defensive_team": "BBB",
        "pass_result": "C",
        "pass_location_type": "middle",
    }


def _minimal_release(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    train = root / "train"
    supplementary_columns = [
        "game_id",
        "season",
        "week",
        "play_id",
        "possession_team",
        "defensive_team",
        "pass_result",
        "pass_location_type",
    ]
    _write_csv(
        root / "supplementary_data.csv",
        supplementary_columns,
        [_supplementary_row("1", "10", "1"), _supplementary_row("2", "20", "2")],
    )
    for label, game_id, play_id in (("2023_w01", "1", "10"), ("2023_w02", "2", "20")):
        _write_csv(
            train / f"input_{label}.csv",
            INPUT_COLUMNS,
            [
                _input_row(game_id, play_id, "receiver", 1, role="Targeted Receiver", side="Offense", predict="True", expected_frames="2"),
                _input_row(game_id, play_id, "receiver", 2, role="Targeted Receiver", side="Offense", predict="True", expected_frames="2"),
                _input_row(game_id, play_id, "defender", 1, role="Defensive Coverage", side="Defense"),
                _input_row(game_id, play_id, "defender", 2, role="Defensive Coverage", side="Defense"),
            ],
        )
        _write_csv(
            train / f"output_{label}.csv",
            OUTPUT_COLUMNS,
            [_output_row(game_id, play_id, "receiver", 1), _output_row(game_id, play_id, "receiver", 2)],
        )
    return root


def test_validation_uses_only_explicitly_requested_week_and_accepts_zero_counter(tmp_path):
    root = _minimal_release(tmp_path)

    result = run_milestone_1_validation(
        root,
        tmp_path / "artifacts",
        None,
        None,
        weeks=("2023_w01",),
    )

    assert result["benchmark"]["selected_weeks"] == ["2023_w01"]
    assert result["dataset"]["input_file_count"] == 1
    assert result["dataset"]["output_file_count"] == 1
    assert result["entities"]["input_rows"] == 4
    assert result["benchmark"]["rows_processed"]["output"] == 2
    assert result["checks"]["input_coordinates"]["status"] == "PASS"
    assert sum(result["coordinates"]["reference_out_of_range_rows"].values()) == 0


def test_output_missing_coordinates_are_counted_by_row_not_cell(tmp_path):
    output = tmp_path / "output_2023_w01.csv"
    _write_csv(
        output,
        OUTPUT_COLUMNS,
        [
            _output_row("1", "10", "receiver", 1, x="", y=""),
            _output_row("1", "10", "receiver", 2),
        ],
    )

    stats = _stream_outputs_fast(
        [output],
        {("1", "10", "receiver"): 2},
    )

    assert stats["coordinate_missing"] == 1


def test_conflicting_declared_output_lengths_are_reported(tmp_path):
    input_path = tmp_path / "input_2023_w01.csv"
    _write_csv(
        input_path,
        INPUT_COLUMNS,
        [
            _input_row("1", "10", "receiver", 1, role="Targeted Receiver", side="Offense", predict="True", expected_frames="2"),
            _input_row("1", "10", "receiver", 2, role="Targeted Receiver", side="Offense", predict="True", expected_frames="3"),
        ],
    )

    _, _, stats = _stream_inputs_fast(
        [input_path],
        {("1", "10"): _supplementary_row("1", "10", "1")},
    )

    assert stats["expected_output_frame_conflicts"] == 1
