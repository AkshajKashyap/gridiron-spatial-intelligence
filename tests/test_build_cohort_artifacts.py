from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.build_cohort_artifacts as entry
from gridiron_spatial.cohort import (
    COHORT_TABLE_NAMES,
    EXCLUSION_LEDGER_SCHEMA,
    REASON_CODES,
    TABLE_SCHEMAS,
)


def _empty_frame(columns):
    return pd.DataFrame(
        {
            column: pd.Series(
                dtype=bool
                if column in {*REASON_CODES, "eligible"}
                else object
            )
            for column in columns
        }
    )


def _tables(*, source_key=None):
    result = {
        name: _empty_frame(TABLE_SCHEMAS[name])
        for name in COHORT_TABLE_NAMES
    }
    if source_key is not None:
        row = {
            column: (
                False if column in REASON_CODES
                else True if column == "eligible"
                else None
            )
            for column in TABLE_SCHEMAS["source_plays"]
        }
        row.update(
            {
                "game_id": source_key[0],
                "play_id": source_key[1],
                "phase": "input",
                "week": "2023_w01",
                "split": "development_train",
            }
        )
        result["source_plays"] = pd.DataFrame(
            [row], columns=TABLE_SCHEMAS["source_plays"]
        )
    return result


def _ledger(ledger_id):
    if ledger_id is None:
        return _empty_frame(EXCLUSION_LEDGER_SCHEMA)
    row = {
        column: False if column in REASON_CODES else None
        for column in EXCLUSION_LEDGER_SCHEMA
    }
    row["ledger_id"] = ledger_id
    return pd.DataFrame([row], columns=EXCLUSION_LEDGER_SCHEMA)


def _pass_reconciliation():
    return {"overall": {"reconciliation_status": "PASS"}}


def _install_pipeline(
    monkeypatch,
    *,
    same_source_key=False,
    duplicate_ledger=False,
):
    state = {
        "loaded_weeks": [],
        "built_weeks": [],
        "writer_calls": [],
        "input_frames": [],
        "input_snapshots": [],
        "table_frames": [],
        "table_snapshots": [],
        "ledger_calls": 0,
    }
    monkeypatch.setattr(
        entry,
        "_load_supplementary",
        lambda release_root: pd.DataFrame(),
    )

    def load_week(release_root, week, supplementary):
        state["loaded_weeks"].append(week)
        inputs = pd.DataFrame(
            {"game_id": [week, week], "play_id": ["1", "2"]}
        )
        outputs = pd.DataFrame({"frame_id": [1]})
        metadata = pd.DataFrame({"week": [week]})
        state["input_frames"].extend([inputs, outputs, metadata])
        state["input_snapshots"].extend(
            [inputs.copy(deep=True), outputs.copy(deep=True), metadata.copy(deep=True)]
        )
        return inputs, outputs, metadata

    def build_week(inputs, outputs, supplementary, week):
        state["built_weeks"].append(week)
        tables = _tables(
            source_key=("same", "play") if same_source_key else None
        )
        state["table_frames"].extend(tables.values())
        state["table_snapshots"].extend(
            table.copy(deep=True) for table in tables.values()
        )
        return SimpleNamespace(**tables)

    def build_ledger(*tables):
        state["ledger_calls"] += 1
        ledger_id = (
            "same-ledger"
            if duplicate_ledger
            else f"ledger-{state['ledger_calls']}"
        )
        return _ledger(ledger_id)

    monkeypatch.setattr(entry, "_load_week", load_week)
    monkeypatch.setattr(entry, "build_week_cohorts", build_week)
    monkeypatch.setattr(entry, "build_exclusion_ledger", build_ledger)
    monkeypatch.setattr(
        entry, "_table_reconciliation", lambda tables, ledger: _pass_reconciliation()
    )
    monkeypatch.setattr(
        entry, "_validate_game_splits", lambda tables: {"status": "PASS"}
    )
    monkeypatch.setattr(
        entry,
        "summarize_cohort_reporting",
        lambda tables, ledger: {
            "observed_game_count": 0,
            "counts_by_split": {},
        },
    )

    def writer(*args, **kwargs):
        state["writer_calls"].append((args, kwargs))
        return {"artifact_format_version": "test"}

    monkeypatch.setattr(entry, "write_cohort_artifacts", writer)
    return state


def test_explicit_weeks_aggregate_once_and_reach_writer_without_mutation(
    monkeypatch,
    tmp_path,
):
    state = _install_pipeline(monkeypatch)
    weeks = ("2023_w01", "2023_w02", "2023_w18")

    result = entry.build_and_write(
        tmp_path / "release",
        tmp_path / "output",
        weeks,
        overwrite=True,
    )

    assert result == {"artifact_format_version": "test"}
    assert state["loaded_weeks"] == list(weeks)
    assert state["built_weeks"] == list(weeks)
    assert len(state["writer_calls"]) == 1
    args, kwargs = state["writer_calls"][0]
    output_dir, tables, ledger, cohort_summary, reporting_summary = args
    assert output_dir == tmp_path / "output"
    assert tuple(tables) == COHORT_TABLE_NAMES
    assert list(ledger["ledger_id"]) == [
        "ledger-1",
        "ledger-2",
        "ledger-3",
    ]
    assert cohort_summary["processed_weeks"] == list(weeks)
    assert cohort_summary["source_row_totals"] == {"input": 6, "output": 3}
    assert tuple(cohort_summary["weekly_runtimes"]) == weeks
    assert reporting_summary["observed_game_count"] == 0
    assert kwargs == {"overwrite": True}
    for frame, snapshot in zip(
        state["input_frames"], state["input_snapshots"], strict=True
    ):
        pd.testing.assert_frame_equal(frame, snapshot)
    for frame, snapshot in zip(
        state["table_frames"], state["table_snapshots"], strict=True
    ):
        pd.testing.assert_frame_equal(frame, snapshot)


@pytest.mark.parametrize(
    "failure",
    ["weekly_reconciliation", "aggregate_reconciliation", "split_validation"],
)
def test_validation_failures_prevent_writer(
    monkeypatch,
    tmp_path,
    failure,
):
    state = _install_pipeline(monkeypatch)
    calls = 0

    def reconcile(tables, ledger):
        nonlocal calls
        calls += 1
        status = (
            "FAIL"
            if failure == "weekly_reconciliation" and calls == 1
            else "FAIL"
            if failure == "aggregate_reconciliation" and calls == 3
            else "PASS"
        )
        return {"overall": {"reconciliation_status": status}}

    monkeypatch.setattr(entry, "_table_reconciliation", reconcile)
    if failure == "split_validation":
        monkeypatch.setattr(
            entry, "_validate_game_splits", lambda tables: {"status": "FAIL"}
        )
    with pytest.raises(RuntimeError):
        entry.build_and_write(
            tmp_path / "release",
            tmp_path / "output",
            ("2023_w01", "2023_w02"),
        )
    assert state["writer_calls"] == []


@pytest.mark.parametrize("duplicate_kind", ["table", "ledger"])
def test_duplicate_aggregate_keys_prevent_writer(
    monkeypatch,
    tmp_path,
    duplicate_kind,
):
    state = _install_pipeline(
        monkeypatch,
        same_source_key=duplicate_kind == "table",
        duplicate_ledger=duplicate_kind == "ledger",
    )
    with pytest.raises(RuntimeError):
        entry.build_and_write(
            tmp_path / "release",
            tmp_path / "output",
            ("2023_w01", "2023_w02"),
        )
    assert state["writer_calls"] == []


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
    loaded = False

    def unexpected_loader(release_root):
        nonlocal loaded
        loaded = True
        raise AssertionError("invalid CLI reached the loader")

    monkeypatch.setattr(entry, "_load_supplementary", unexpected_loader)
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
    assert not loaded


def test_writer_failure_returns_nonzero(monkeypatch, tmp_path):
    _install_pipeline(monkeypatch)

    def fail_writer(*args, **kwargs):
        raise RuntimeError("synthetic writer failure")

    monkeypatch.setattr(entry, "write_cohort_artifacts", fail_writer)
    status = entry.main(
        [
            "--release-root",
            str(tmp_path / "release"),
            "--output-dir",
            str(tmp_path / "output"),
            "--weeks",
            "2023_w01",
        ]
    )
    assert status != 0
