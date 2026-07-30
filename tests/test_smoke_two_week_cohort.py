import pytest

import scripts.smoke_two_week_cohort as smoke


def test_week_cli_preserves_complete_lists_and_rejects_invalid_input(
    monkeypatch,
):
    def unexpected_csv_read(*args, **kwargs):
        raise AssertionError("argument parsing must not read NFL data")

    monkeypatch.setattr(smoke.pd, "read_csv", unexpected_csv_read)

    two_weeks = ["2023_w01", "2023_w18"]
    assert smoke.parse_week_args(two_weeks) == tuple(two_weeks)

    all_weeks = [f"2023_w{week:02d}" for week in range(1, 19)]
    assert smoke.parse_week_args(all_weeks) == tuple(all_weeks)
    assert list(smoke.parse_week_args(all_weeks)) == all_weeks

    for invalid_args in (
        [],
        ["2023_w01", "2023_w01"],
        ["2023_w1"],
        ["2024_w01"],
        ["2023_w19"],
        ["2023_w02", "2023_w01"],
    ):
        with pytest.raises(SystemExit) as exc_info:
            smoke.parse_week_args(invalid_args)
        assert exc_info.value.code != 0

    received = []

    def capture_execution_weeks(weeks):
        received.extend(weeks)
        return 0

    monkeypatch.setattr(smoke, "_run_weeks", capture_execution_weeks)
    assert smoke.main(all_weeks) == 0
    assert received == all_weeks
