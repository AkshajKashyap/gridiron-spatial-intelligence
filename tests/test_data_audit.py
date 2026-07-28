import json

import pytest

from gridiron_spatial.data_audit import AuditError, run_data_audit


def run_audit(tmp_path, contents: dict[str, str]):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name, content in contents.items():
        (raw / name).write_text(content, encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    report = tmp_path / "data_audit.md"
    return run_data_audit(raw, artifacts, report), artifacts, report


def test_profiles_valid_csv_with_missing_values_and_candidate_fields(tmp_path):
    (result, artifacts, _) = run_audit(
        tmp_path,
        {
            "tracking.csv": (
                "gameId,playId,frameId,nflId,x,y,event\n"
                "1,10,1,100,20.5,30.0,ball_snap\n"
                "1,10,2,100,,31.0,\n"
            )
        },
    )
    manifest, validation = result
    profile = manifest["files"][0]["table_profile"]
    x_column = next(column for column in profile["columns"] if column["name"] == "x")
    assert profile["row_count"] == 2
    assert x_column["null_count"] == 1
    assert profile["candidate_fields"]["game_id"]["status"] == "plausible_candidate_requiring_validation"
    assert validation["checks"]["schema_profile"]["status"] == "PASS"
    assert json.loads((artifacts / "data_manifest.json").read_text())["file_count"] == 1


def test_reports_duplicate_rows_and_duplicate_candidate_keys(tmp_path):
    (manifest, _), _, _ = run_audit(
        tmp_path,
        {
            "tracking.csv": (
                "gameId,playId,frameId,nflId,x,y\n"
                "1,10,1,100,20,30\n"
                "1,10,1,100,20,30\n"
            )
        },
    )
    profile = manifest["files"][0]["table_profile"]
    assert profile["duplicated_row_check"]["status"] == "FAIL"
    assert profile["key_checks"]["candidate_composite_key"]["status"] == "FAIL"


def test_empty_data_directory_generates_blocker_report(tmp_path):
    (manifest, validation), artifacts, report = run_audit(tmp_path, {})
    assert manifest["file_count"] == 0
    assert validation["checks"]["raw_data_present"]["status"] == "FAIL"
    assert "Data acquisition blocker" in report.read_text()
    assert (artifacts / "milestone_1_validation.json").exists()


def test_malformed_or_unreadable_delimited_input_fails_clearly(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "bad.csv").write_bytes(b"gameId,playId\n1,2,3\n")
    with pytest.raises(AuditError, match="Malformed row"):
        run_data_audit(raw, tmp_path / "artifacts", tmp_path / "data_audit.md")

    (raw / "bad.csv").unlink()
    (raw / "unreadable.csv").write_bytes(b"gameId,playId\n\xff\xfe")
    with pytest.raises(AuditError, match="Malformed or empty"):
        run_data_audit(raw, tmp_path / "artifacts", tmp_path / "data_audit.md")
