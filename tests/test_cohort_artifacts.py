import copy
import hashlib
import json

import pandas as pd
import pytest

import gridiron_spatial.cohort_artifacts as artifacts
from gridiron_spatial.cohort import (
    COHORT_TABLE_NAMES,
    EXCLUSION_LEDGER_SCHEMA,
    REASON_CODES,
    TABLE_SCHEMAS,
)
from gridiron_spatial.cohort_artifacts import (
    CohortArtifactError,
    write_cohort_artifacts,
)


EXPECTED_FILES = {
    *(f"{name}.parquet" for name in COHORT_TABLE_NAMES),
    "exclusion_ledger.parquet",
    "cohort_summary.json",
    "manifest.json",
}


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


def _synthetic_inputs():
    tables = {
        name: _empty_frame(TABLE_SCHEMAS[name])
        for name in COHORT_TABLE_NAMES
    }
    source = {
        column: (
            False if column in REASON_CODES
            else True if column == "eligible"
            else None
        )
        for column in TABLE_SCHEMAS["source_plays"]
    }
    source.update(
        {
            "game_id": "1",
            "play_id": "10",
            "phase": "input",
            "week": "2023_w01",
            "split": "development_train",
        }
    )
    tables["source_plays"] = pd.DataFrame(
        {
            column: pd.Series(
                [source[column]],
                dtype=bool
                if column in {*REASON_CODES, "eligible"}
                else object,
            )
            for column in TABLE_SCHEMAS["source_plays"]
        }
    )
    ledger = _empty_frame(EXCLUSION_LEDGER_SCHEMA)
    counts = {}
    split_counts = {
        split: {} for split in (
            "development_train",
            "validation",
            "frozen_test",
        )
    }
    for name, table in tables.items():
        eligible = int(table["eligible"].sum())
        counts[name] = {
            "rows": len(table),
            "eligible": eligible,
            "excluded": len(table) - eligible,
        }
        for split in split_counts:
            selected = table.loc[table["split"].astype("string").eq(split)]
            split_eligible = int(selected["eligible"].sum())
            split_counts[split][name] = {
                "rows": len(selected),
                "eligible": split_eligible,
                "excluded": len(selected) - split_eligible,
            }
    cohort_summary = {
        "aggregate_table_counts": counts,
        "aggregate_reconciliation_status": "PASS",
        "aggregate_split_validation_status": "PASS",
        "processed_weeks": ["2023_w01"],
        "source_row_totals": {"input": 7, "output": 2},
    }
    reporting_summary = {
        "observed_game_count": 1,
        "counts_by_split": split_counts,
    }
    return tables, ledger, cohort_summary, reporting_summary


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_writer_round_trip_manifest_checksums_empty_table_and_no_mutation(
    tmp_path,
):
    tables, ledger, cohort_summary, reporting_summary = _synthetic_inputs()
    table_copies = {
        name: table.copy(deep=True) for name, table in tables.items()
    }
    ledger_copy = ledger.copy(deep=True)
    cohort_copy = copy.deepcopy(cohort_summary)
    reporting_copy = copy.deepcopy(reporting_summary)
    destination = tmp_path / "cohorts"

    manifest = write_cohort_artifacts(
        destination,
        tables,
        ledger,
        cohort_summary,
        reporting_summary,
    )

    assert {path.name for path in destination.iterdir()} == EXPECTED_FILES
    assert [
        entry["table_name"] for entry in manifest["tables"]
    ] == [*COHORT_TABLE_NAMES, "exclusion_ledger"]
    for entry in manifest["tables"]:
        path = destination / entry["relative_path"]
        restored = pd.read_parquet(path)
        source = (
            ledger
            if entry["table_name"] == "exclusion_ledger"
            else tables[entry["table_name"]]
        )
        assert len(restored) == len(source) == entry["row_count"]
        assert list(restored.columns) == list(source.columns)
        assert _sha256(path) == entry["sha256"]
    empty_pairs = pd.read_parquet(destination / "pair_exclusions.parquet")
    assert empty_pairs.empty
    assert list(empty_pairs.columns) == TABLE_SCHEMAS["pair_exclusions"]
    assert json.loads((destination / "manifest.json").read_text())["tables"]
    assert json.loads((destination / "cohort_summary.json").read_text())
    assert str(tmp_path) not in json.dumps(manifest)
    assert str(tmp_path) not in (destination / "cohort_summary.json").read_text()
    for name in COHORT_TABLE_NAMES:
        pd.testing.assert_frame_equal(tables[name], table_copies[name])
    pd.testing.assert_frame_equal(ledger, ledger_copy)
    assert cohort_summary == cohort_copy
    assert reporting_summary == reporting_copy


def test_writer_rejects_duplicate_keys_ledger_ids_and_schema_mismatch(tmp_path):
    tables, ledger, cohort_summary, reporting_summary = _synthetic_inputs()
    duplicate_tables = {
        name: table.copy(deep=True) for name, table in tables.items()
    }
    duplicate_tables["source_plays"] = pd.concat(
        [tables["source_plays"], tables["source_plays"]],
        ignore_index=True,
    )
    with pytest.raises(CohortArtifactError, match="duplicate frozen keys"):
        write_cohort_artifacts(
            tmp_path / "duplicate-table",
            duplicate_tables,
            ledger,
            cohort_summary,
            reporting_summary,
        )

    ledger_row = {
        column: (
            False if column in REASON_CODES else None
        )
        for column in EXCLUSION_LEDGER_SCHEMA
    }
    ledger_row["ledger_id"] = "ledger-1"
    duplicate_ledger = pd.DataFrame(
        [ledger_row, ledger_row], columns=EXCLUSION_LEDGER_SCHEMA
    )
    with pytest.raises(CohortArtifactError, match="duplicate frozen keys"):
        write_cohort_artifacts(
            tmp_path / "duplicate-ledger",
            tables,
            duplicate_ledger,
            cohort_summary,
            reporting_summary,
        )

    bad_schema = {
        name: table.copy(deep=True) for name, table in tables.items()
    }
    bad_schema["source_plays"] = bad_schema["source_plays"].drop(
        columns="phase"
    )
    with pytest.raises(CohortArtifactError, match="frozen schema"):
        write_cohort_artifacts(
            tmp_path / "bad-schema",
            bad_schema,
            ledger,
            cohort_summary,
            reporting_summary,
        )


def test_writer_rejects_failed_status_existing_output_and_absolute_path(
    tmp_path,
):
    tables, ledger, cohort_summary, reporting_summary = _synthetic_inputs()
    for status_field in (
        "aggregate_reconciliation_status",
        "aggregate_split_validation_status",
    ):
        failed = copy.deepcopy(cohort_summary)
        failed[status_field] = "FAIL"
        with pytest.raises(CohortArtifactError, match="must both be PASS"):
            write_cohort_artifacts(
                tmp_path / f"failed-{status_field}",
                tables,
                ledger,
                failed,
                reporting_summary,
            )

    destination = tmp_path / "existing"
    write_cohort_artifacts(
        destination,
        tables,
        ledger,
        cohort_summary,
        reporting_summary,
    )
    with pytest.raises(FileExistsError):
        write_cohort_artifacts(
            destination,
            tables,
            ledger,
            cohort_summary,
            reporting_summary,
        )

    absolute = copy.deepcopy(cohort_summary)
    absolute["raw_data_path"] = str(tmp_path / "raw.csv")
    with pytest.raises(CohortArtifactError, match="absolute paths"):
        write_cohort_artifacts(
            tmp_path / "absolute",
            tables,
            ledger,
            absolute,
            reporting_summary,
        )


def test_atomic_validation_failure_leaves_destination_untouched(
    tmp_path,
    monkeypatch,
):
    tables, ledger, cohort_summary, reporting_summary = _synthetic_inputs()
    destination = tmp_path / "atomic"
    destination.mkdir()
    marker = destination / "existing.txt"
    marker.write_text("unchanged")

    def fail_validation(path, source):
        raise CohortArtifactError("synthetic staged validation failure")

    monkeypatch.setattr(artifacts, "_validate_parquet", fail_validation)
    with pytest.raises(CohortArtifactError, match="synthetic"):
        write_cohort_artifacts(
            destination,
            tables,
            ledger,
            cohort_summary,
            reporting_summary,
            overwrite=True,
        )
    assert marker.read_text() == "unchanged"
    assert not list(tmp_path.glob(".atomic.tmp-*"))
