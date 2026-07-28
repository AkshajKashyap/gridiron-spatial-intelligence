"""Read-only inventory and structural profiling for raw tracking data.

This module deliberately makes no football-semantic claims.  Column-name
matches are reported as candidates that require validation against a source
data dictionary.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SUPPORTED_DELIMITERS = {".csv": ",", ".tsv": "\t", ".txt": ","}
MAX_EXACT_DUPLICATE_ROWS = 1_000_000
MAX_UNIQUE_VALUES = 10_000
MAX_NUMERIC_SAMPLES = 100_000
MAX_EXAMPLES = 5

FIELD_ALIASES = {
    "game_id": {"gameid", "game_id"},
    "play_id": {"playid", "play_id"},
    "player_id": {"nflid", "nfl_id", "playerid", "player_id"},
    "frame_id": {"frameid", "frame_id"},
    "timestamp": {"timestamp", "time", "frametime", "frame_time"},
    "x_coordinate": {"x", "xcoord", "x_coord", "xcoordinate"},
    "y_coordinate": {"y", "ycoord", "y_coord", "ycoordinate"},
    "speed": {"s", "speed"},
    "acceleration": {"a", "acceleration", "accel"},
    "orientation": {"o", "orientation"},
    "direction": {"dir", "direction"},
    "team_or_club": {"team", "club", "teamabbr", "team_abbr"},
    "player_position": {"position", "pos", "officialposition", "official_position"},
    "event": {"event", "eventname", "event_name"},
    "possession": {"possessionteam", "possession_team", "offenseteam", "offense_team"},
    "defense": {"defensiveteam", "defensive_team", "defenseteam", "defense_team"},
}


class AuditError(RuntimeError):
    """Raised when a relevant raw file cannot be safely profiled."""


def _normalise_name(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum() or character == "_")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_safe) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _missing(value: str | None) -> bool:
    return value is None or value.strip() == ""


def _value_kind(value: str) -> str:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return "boolean"
    try:
        parsed = float(value)
    except ValueError:
        return "string"
    if not math.isfinite(parsed):
        return "float"
    return "integer" if parsed.is_integer() and all(token not in value.lower() for token in (".", "e")) else "float"


def _combine_kinds(kinds: set[str]) -> str:
    if not kinds:
        return "unknown"
    if kinds == {"integer"}:
        return "integer"
    if kinds <= {"integer", "float"}:
        return "float"
    if kinds == {"boolean"}:
        return "boolean"
    return "string"


def _number(value: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _timestamp(value: str) -> float | None:
    numeric = _number(value)
    if numeric is not None:
        return numeric
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def percentile(probability: float) -> float:
        index = (len(ordered) - 1) * probability
        lower, upper = math.floor(index), math.ceil(index)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {"p01": percentile(0.01), "p05": percentile(0.05), "p50": percentile(0.50), "p95": percentile(0.95), "p99": percentile(0.99)}


def _candidate_fields(columns: list[str]) -> dict[str, dict[str, Any]]:
    normalised = {_normalise_name(column): column for column in columns}
    findings: dict[str, dict[str, Any]] = {}
    for field, aliases in FIELD_ALIASES.items():
        matches = [normalised[alias] for alias in sorted(aliases) if alias in normalised]
        findings[field] = (
            {
                "status": "plausible_candidate_requiring_validation",
                "matched_columns": matches,
                "note": "Column-name match only; semantics are not confirmed.",
            }
            if matches
            else {"status": "absent", "matched_columns": [], "note": "No matching column name observed."}
        )
    return findings


def _first_candidate(candidates: dict[str, dict[str, Any]], name: str) -> str | None:
    matches = candidates[name]["matched_columns"]
    return matches[0] if matches else None


def _status(status: str, detail: str, **values: Any) -> dict[str, Any]:
    return {"status": status, "detail": detail, **values}


def _sample_add(values: list[float], value: float) -> None:
    if len(values) < MAX_NUMERIC_SAMPLES:
        values.append(value)


def _profile_delimited_file(path: Path, root: Path) -> dict[str, Any]:
    delimiter = SUPPORTED_DELIMITERS[path.suffix.lower()]
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise AuditError(f"Cannot read {path}: {error}") from error

    with handle:
        try:
            reader = csv.reader(handle, delimiter=delimiter)
            header = next(reader)
        except (csv.Error, StopIteration, UnicodeDecodeError) as error:
            raise AuditError(f"Malformed or empty delimited file {path}: {error}") from error

        if not header or any(not column.strip() for column in header):
            raise AuditError(f"Malformed header in {path}: column names must be non-empty.")
        if len(set(header)) != len(header):
            raise AuditError(f"Malformed header in {path}: duplicate column names are not supported.")

        candidates = _candidate_fields(header)
        canonical_columns = {name: _first_candidate(candidates, name) for name in FIELD_ALIASES}
        column_state = {
            column: {"null_count": 0, "kinds": set(), "examples": [], "unique_values": set(), "unique_capped": False}
            for column in header
        }
        numeric_values = {column: [] for column in (canonical_columns["x_coordinate"], canonical_columns["y_coordinate"]) if column}
        reference_out_of_range = {column: 0 for column in numeric_values}
        row_hashes: set[bytes] = set()
        row_duplicate_count = 0
        row_duplicates_capped = False
        composite_key_hashes: set[tuple[str, ...]] = set()
        composite_duplicates = 0
        composite_key_capped = False
        repeated_frame_groups: Counter[tuple[str, str, str]] = Counter()
        repeated_player_groups: Counter[tuple[str, str, str]] = Counter()
        temporal_last: dict[tuple[str, ...], float] = {}
        temporal_regressions = 0
        temporal_duplicates = 0
        temporal_deltas: list[float] = []
        temporal_unparseable = 0
        temporal_unkeyable = 0
        frame_values: list[float] = []
        timestamp_values: list[float] = []
        row_count = 0

        game_column = canonical_columns["game_id"]
        play_column = canonical_columns["play_id"]
        player_column = canonical_columns["player_id"]
        frame_column = canonical_columns["frame_id"]
        timestamp_column = canonical_columns["timestamp"]

        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise AuditError(
                    f"Malformed row in {path} at line {line_number}: expected {len(header)} values, found {len(row)}."
                )
            row_count += 1
            record = dict(zip(header, row, strict=True))
            if row_count <= MAX_EXACT_DUPLICATE_ROWS:
                digest = hashlib.sha256("\x1f".join(row).encode("utf-8")).digest()
                if digest in row_hashes:
                    row_duplicate_count += 1
                else:
                    row_hashes.add(digest)
            else:
                row_duplicates_capped = True

            for column, value in record.items():
                state = column_state[column]
                if _missing(value):
                    state["null_count"] += 1
                    continue
                state["kinds"].add(_value_kind(value))
                if len(state["examples"]) < MAX_EXAMPLES and value not in state["examples"]:
                    state["examples"].append(value)
                if not state["unique_capped"]:
                    state["unique_values"].add(value)
                    if len(state["unique_values"]) > MAX_UNIQUE_VALUES:
                        state["unique_values"].clear()
                        state["unique_capped"] = True

            for coordinate, bounds in ((canonical_columns["x_coordinate"], (0.0, 120.0)), (canonical_columns["y_coordinate"], (0.0, 53.3))):
                if coordinate and not _missing(record[coordinate]):
                    value = _number(record[coordinate])
                    if value is not None:
                        _sample_add(numeric_values[coordinate], value)
                        if value < bounds[0] or value > bounds[1]:
                            reference_out_of_range[coordinate] += 1

            game = record[game_column] if game_column else None
            play = record[play_column] if play_column else None
            player = record[player_column] if player_column else None
            frame = record[frame_column] if frame_column else None
            if all(value is not None and not _missing(value) for value in (game, play, frame)):
                repeated_frame_groups[(game, play, frame)] += 1
            if all(value is not None and not _missing(value) for value in (game, play, player)):
                repeated_player_groups[(game, play, player)] += 1
            if all(value is not None and not _missing(value) for value in (game, play, frame, player)):
                key = (game, play, frame, player)
                if row_count <= MAX_EXACT_DUPLICATE_ROWS:
                    if key in composite_key_hashes:
                        composite_duplicates += 1
                    else:
                        composite_key_hashes.add(key)
                else:
                    composite_key_capped = True

            time_column = timestamp_column or frame_column
            if time_column and game_column and play_column:
                raw_time = record[time_column]
                parsed_time = _timestamp(raw_time)
                if _missing(raw_time) or parsed_time is None:
                    temporal_unparseable += 1
                else:
                    if time_column == frame_column:
                        _sample_add(frame_values, parsed_time)
                    else:
                        _sample_add(timestamp_values, parsed_time)
                    temporal_key = (record[game_column], record[play_column])
                    if player_column:
                        if _missing(record[player_column]):
                            temporal_unkeyable += 1
                            continue
                        temporal_key += (record[player_column],)
                    previous = temporal_last.get(temporal_key)
                    if previous is not None:
                        delta = parsed_time - previous
                        if delta < 0:
                            temporal_regressions += 1
                        elif delta == 0 and player_column:
                            temporal_duplicates += 1
                        else:
                            _sample_add(temporal_deltas, delta)
                    temporal_last[temporal_key] = parsed_time

    columns = []
    for column in header:
        state = column_state[column]
        unique_count = None if state["unique_capped"] else len(state["unique_values"])
        columns.append(
            {
                "name": column,
                "inferred_dtype": _combine_kinds(state["kinds"]),
                "null_count": state["null_count"],
                "null_percentage": round((100 * state["null_count"] / row_count), 4) if row_count else 0.0,
                "example_values": state["examples"],
                "unique_count": unique_count,
                "unique_count_status": "capped" if state["unique_capped"] else "actual",
            }
        )

    coordinate_summary: dict[str, Any] = {}
    for label in ("x_coordinate", "y_coordinate"):
        column = canonical_columns[label]
        if not column:
            coordinate_summary[label] = _status("NOT_TESTED", "No candidate coordinate column was observed.")
            continue
        values = numeric_values[column]
        coordinate_summary[label] = _status(
            "UNKNOWN",
            "Reference range is a plausibility check only; coordinate semantics require the data dictionary.",
            column=column,
            min=min(values) if values else None,
            max=max(values) if values else None,
            quantiles=_quantiles(values),
            numeric_sample_count=len(values),
            reference_out_of_range_count=reference_out_of_range[column],
            missing_count=column_state[column]["null_count"],
        )

    has_composite_key = all((game_column, play_column, frame_column, player_column))
    key_checks = {
        "candidate_identifiers": _status(
            "UNKNOWN",
            "Observed column names are candidates only; their meanings require validation.",
            columns={name: value for name, value in canonical_columns.items() if value},
        ),
        "candidate_composite_key": (
            _status(
                "NOT_TESTED" if composite_key_capped else ("FAIL" if composite_duplicates else "PASS"),
                "Candidate key uniqueness is structural only and does not confirm entity semantics.",
                columns=[game_column, play_column, frame_column, player_column],
                duplicate_count=composite_duplicates if not composite_key_capped else None,
            )
            if has_composite_key
            else _status("NOT_TESTED", "Not all candidate game, play, frame, and player identifier columns were observed.")
        ),
        "repeated_frame_ids_within_candidate_plays": (
            _status(
                "PASS",
                "Count is reported; repeated frame IDs may be expected when several entities share a frame.",
                repeated_groups=sum(count > 1 for count in repeated_frame_groups.values()),
                total_candidate_frame_groups=len(repeated_frame_groups),
            )
            if game_column and play_column and frame_column
            else _status("NOT_TESTED", "Candidate game, play, and frame columns were not all observed.")
        ),
        "repeated_player_ids_across_candidate_frames": (
            _status(
                "PASS",
                "Count is reported; recurrence may be expected in tracking data.",
                repeated_player_groups=sum(count > 1 for count in repeated_player_groups.values()),
                total_candidate_player_groups=len(repeated_player_groups),
            )
            if game_column and play_column and player_column
            else _status("NOT_TESTED", "Candidate game, play, and player columns were not all observed.")
        ),
    }

    temporal_column = timestamp_column or frame_column
    if temporal_column and game_column and play_column:
        delta_summary = _quantiles(temporal_deltas)
        median_delta = delta_summary["p50"] if delta_summary else None
        frequency = (1 / median_delta) if timestamp_column and median_delta and median_delta > 0 else None
        temporal_checks = _status(
            "FAIL" if temporal_regressions or temporal_duplicates else "PASS",
            "Monotonicity is measured in source row order within candidate play/entity groups; semantics remain unconfirmed.",
            time_column=temporal_column,
            observed_range={"min": min(timestamp_values or frame_values, default=None), "max": max(timestamp_values or frame_values, default=None)},
            regressions=temporal_regressions,
            duplicate_time_values=temporal_duplicates,
            unparseable_or_missing_time_values=temporal_unparseable,
            unkeyable_time_rows=temporal_unkeyable,
            positive_spacing_quantiles=delta_summary,
            apparent_frequency_hz=frequency,
        )
    else:
        temporal_checks = _status("NOT_TESTED", "Candidate time/frame plus game/play columns were not all observed.")

    return {
        "relative_path": str(path.relative_to(root)),
        "filename": path.name,
        "extension": path.suffix.lower(),
        "file_size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "table_profile": {
            "row_count": row_count,
            "column_count": len(header),
            "columns": columns,
            "duplicated_row_check": _status(
                "NOT_TESTED" if row_duplicates_capped else ("FAIL" if row_duplicate_count else "PASS"),
                "Exact duplicate rows are checked only while the configured row limit is not exceeded.",
                duplicate_row_count=row_duplicate_count if not row_duplicates_capped else None,
            ),
            "candidate_fields": candidates,
            "key_checks": key_checks,
            "temporal_checks": temporal_checks,
            "coordinate_checks": coordinate_summary,
        },
    }


def _inventory_file(path: Path, root: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    item = {
        "relative_path": str(path.relative_to(root)),
        "filename": path.name,
        "extension": suffix or "[no extension]",
        "file_size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if suffix in SUPPORTED_DELIMITERS:
        return _profile_delimited_file(path, root)
    item["table_profile"] = _status(
        "NOT_TESTED",
        "Unsupported file type; inventoried without parsing. Add a parser only after confirming it is a relevant source format.",
    )
    return item


def _empty_validation() -> dict[str, Any]:
    return {
        "overall_status": "UNKNOWN",
        "checks": {
            "raw_data_present": _status("FAIL", "No raw source files were found under data/raw; data acquisition is blocked."),
            "schema_profile": _status("NOT_TESTED", "No parseable source table is available."),
            "candidate_key_checks": _status("NOT_TESTED", "No parseable source table is available."),
            "temporal_checks": _status("NOT_TESTED", "No parseable source table is available."),
            "coordinate_checks": _status("NOT_TESTED", "No parseable source table is available."),
        },
    }


def _validation_from_inventory(files: list[dict[str, Any]]) -> dict[str, Any]:
    parseable = [item for item in files if isinstance(item.get("table_profile"), dict) and "row_count" in item["table_profile"]]
    if not files:
        return _empty_validation()
    if not parseable:
        return {
            "overall_status": "UNKNOWN",
            "checks": {
                "raw_data_present": _status("PASS", "Raw files were inventoried, but none are currently parseable table formats."),
                "schema_profile": _status("NOT_TESTED", "No supported delimited table was found."),
                "candidate_key_checks": _status("NOT_TESTED", "No supported delimited table was found."),
                "temporal_checks": _status("NOT_TESTED", "No supported delimited table was found."),
                "coordinate_checks": _status("NOT_TESTED", "No supported delimited table was found."),
            },
        }
    profiles = [item["table_profile"] for item in parseable]
    failures = []
    for item in profiles:
        for check_name in ("duplicated_row_check", "temporal_checks"):
            if item[check_name]["status"] == "FAIL":
                failures.append(check_name)
        candidate_key = item["key_checks"]["candidate_composite_key"]
        if candidate_key["status"] == "FAIL":
            failures.append("candidate_composite_key")
    return {
        "overall_status": "FAIL" if failures else "UNKNOWN",
        "checks": {
            "raw_data_present": _status("PASS", f"Inventoried {len(files)} raw file(s)."),
            "schema_profile": _status("PASS", f"Profiled {len(parseable)} supported delimited table(s)."),
            "candidate_key_checks": _status("FAIL" if "candidate_composite_key" in failures else "UNKNOWN", "See per-table candidate key checks; column-name matches do not prove semantics."),
            "temporal_checks": _status("FAIL" if "temporal_checks" in failures else "UNKNOWN", "See per-table temporal checks; source semantics require validation."),
            "coordinate_checks": _status("UNKNOWN", "See per-table coordinate plausibility summaries; units and bounds require validation."),
        },
    }


def _render_audit_markdown(manifest: dict[str, Any], validation: dict[str, Any]) -> str:
    files = manifest["files"]
    lines = [
        "# Data Audit — Initial Source Inventory",
        "",
        "This report is generated by the read-only source inventory and schema profiler. "
        "It reports only local observations; column-name matches are not football semantics.",
        "",
        "## Confirmed",
        "",
    ]
    if not files:
        lines.extend([
            "- No raw NFL tracking files are present under `data/raw/`; only the repository placeholder was found.",
            "- No expected Big Data Bowl file names, tables, columns, cadence, events, coordinates, or identifiers can be confirmed locally.",
        ])
    else:
        lines.append(f"- Inventoried {len(files)} raw file(s) under `data/raw/`.")
        for item in files:
            profile = item.get("table_profile", {})
            if "row_count" in profile:
                lines.append(f"- `{item['relative_path']}`: {profile['row_count']} rows, {profile['column_count']} columns, `{item['extension']}`.")
            else:
                lines.append(f"- `{item['relative_path']}`: inventoried but not parsed as a supported delimited table.")

    lines.extend(["", "## Unknown", ""])
    lines.extend([
        "- Whether the authorized 2026 Big Data Bowl files have been downloaded and whether their schema matches the planning assumptions.",
        "- The meaning, units, timing, and availability-by-split of any future candidate identifier, coordinate, event, target, or player fields.",
        "- Whether the source contains all relevant offensive/defensive entities, a ball row, a reliable release boundary, and locally evaluable labels.",
    ])

    lines.extend(["", "## Potential issues", ""])
    if not files:
        lines.append("- **Data acquisition blocker:** no usable raw dataset is available, so schema, key, temporal, and coordinate validation are NOT_TESTED.")
    else:
        lines.append(f"- Overall structural validation status: **{validation['overall_status']}**. Consult `artifacts/milestone_1_validation.json` for per-table details.")
        for item in files:
            profile = item.get("table_profile", {})
            if "row_count" not in profile:
                lines.append(f"- `{item['relative_path']}` was not parsed; confirm whether its format is relevant before adding support.")

    lines.extend([
        "",
        "## Next validation step",
        "",
        "Acquire the authorized competition archive into `data/raw/bdb_2026/`, retain it unchanged, and rerun the profiler. "
        "Then review the generated manifest before implementing any pass-play selection, coordinate handling, or spatial analysis.",
        "",
        "## Generated artifacts",
        "",
        "- `artifacts/data_manifest.json`\n- `artifacts/milestone_1_validation.json`",
        "",
    ])
    return "\n".join(lines)


def run_data_audit(raw_directory: Path, artifacts_directory: Path, report_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Inventory and profile raw files without modifying them.

    Raises :class:`AuditError` for a malformed or unreadable supported table so
    a relevant source issue cannot be silently skipped.
    """

    if not raw_directory.exists() or not raw_directory.is_dir():
        raise AuditError(f"Raw data directory does not exist or is not a directory: {raw_directory}")
    raw_files = sorted(path for path in raw_directory.rglob("*") if path.is_file() and path.name != ".gitkeep")
    inventory = [_inventory_file(path, raw_directory) for path in raw_files]
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "raw_directory": str(raw_directory),
        "file_count": len(inventory),
        "files": inventory,
    }
    validation = _validation_from_inventory(inventory)
    _json_dump(artifacts_directory / "data_manifest.json", manifest)
    _json_dump(artifacts_directory / "milestone_1_validation.json", validation)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_audit_markdown(manifest, validation), encoding="utf-8")
    return manifest, validation
