import pandas as pd
import pytest

import scripts.build_normalized_tracking as entry
from gridiron_spatial.normalized_tracking import (
    NORMALIZED_ENTITY_FRAME_SCHEMA,
)


def test_cli_passes_explicit_weeks_and_overwrite_in_order(monkeypatch, tmp_path):
    built_weeks = []
    writer_calls = []

    def fake_build(release_root, week):
        built_weeks.append((release_root, week))
        return {"week": week}

    def fake_writer(output_dir, weeks, builder, *, overwrite):
        writer_calls.append((output_dir, tuple(weeks), overwrite))
        for week in weeks:
            assert builder(week) == {"week": week}
        return {"validation_status": "PASS"}

    monkeypatch.setattr(entry, "_build_week", fake_build)
    monkeypatch.setattr(
        entry, "write_normalized_tracking_release", fake_writer
    )
    release_root = tmp_path / "release"
    output_dir = tmp_path / "output"
    status = entry.main(
        [
            "--release-root",
            str(release_root),
            "--output-dir",
            str(output_dir),
            "--weeks",
            "2023_w02",
            "2023_w03",
            "2023_w18",
            "--overwrite",
        ]
    )

    assert status == 0
    assert writer_calls == [
        (output_dir, ("2023_w02", "2023_w03", "2023_w18"), True)
    ]
    assert built_weeks == [
        (release_root, "2023_w02"),
        (release_root, "2023_w03"),
        (release_root, "2023_w18"),
    ]


@pytest.mark.parametrize(
    "weeks",
    [
        [],
        ["2023_w01", "2023_w01"],
        ["2023_w1"],
        ["2023_w02", "2023_w01"],
    ],
)
def test_invalid_cli_weeks_fail_before_loading(monkeypatch, weeks):
    called = False

    def forbidden_writer(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid CLI reached writer")

    monkeypatch.setattr(
        entry, "write_normalized_tracking_release", forbidden_writer
    )
    arguments = [
        "--release-root",
        "/synthetic/release",
        "--output-dir",
        "/synthetic/output",
        "--weeks",
        *weeks,
    ]
    with pytest.raises(SystemExit) as exc_info:
        entry.main(arguments)
    assert exc_info.value.code != 0
    assert not called


def test_week_builder_uses_synthetic_frames_without_mutation(
    monkeypatch,
    tmp_path,
):
    inputs = pd.DataFrame(
        {
            "game_id": ["1", "1"],
            "play_id": ["10", "10"],
            "nfl_id": ["101", "101"],
            "frame_id": [1, 2],
            "player_side": ["Offense", "Offense"],
            "player_role": ["Targeted Receiver", "Targeted Receiver"],
            "player_position": ["WR", "WR"],
            "player_to_predict": [True, True],
            "play_direction": ["right", "right"],
            "x": [10.0, 11.0],
            "y": [20.0, 20.0],
            "dir": [0.0, 0.0],
            "o": [0.0, 0.0],
        }
    )
    outputs = pd.DataFrame(
        {
            "game_id": ["1"],
            "play_id": ["10"],
            "nfl_id": ["101"],
            "frame_id": [1],
            "x": [12.0],
            "y": [20.0],
        }
    )
    inputs_before = inputs.copy(deep=True)
    outputs_before = outputs.copy(deep=True)

    def fake_read(path, usecols):
        source = outputs if path.name.startswith("output_") else inputs
        return source.loc[:, usecols].copy(deep=True)

    monkeypatch.setattr(entry, "_read_csv", fake_read)
    result = entry._build_week(tmp_path, "2023_w02")

    assert list(result["normalized_frame"].columns) == (
        NORMALIZED_ENTITY_FRAME_SCHEMA
    )
    assert result["reconciliation"]["status"] == "PASS"
    assert result["input_rows"] == 2
    assert result["output_rows"] == 1
    pd.testing.assert_frame_equal(inputs, inputs_before)
    pd.testing.assert_frame_equal(outputs, outputs_before)


def test_writer_failure_returns_nonzero(monkeypatch, tmp_path):
    monkeypatch.setattr(
        entry,
        "write_normalized_tracking_release",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic writer failure")
        ),
    )
    assert (
        entry.main(
            [
                "--release-root",
                str(tmp_path / "release"),
                "--output-dir",
                str(tmp_path / "output"),
                "--weeks",
                "2023_w02",
            ]
        )
        != 0
    )
