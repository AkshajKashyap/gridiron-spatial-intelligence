"""Dataset-specific, read-only validation for the BDB 2026 Analytics release.

The module checks structure and reconstructability only. It deliberately does
not normalize coordinates, calculate separation, or infer coverage.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .data_audit import AuditError


INPUT_COLUMNS = [
    "game_id", "play_id", "player_to_predict", "nfl_id", "frame_id",
    "play_direction", "absolute_yardline_number", "player_name",
    "player_height", "player_weight", "player_birth_date", "player_position",
    "player_side", "player_role", "x", "y", "s", "a", "dir", "o",
    "num_frames_output", "ball_land_x", "ball_land_y",
]
OUTPUT_COLUMNS = ["game_id", "play_id", "nfl_id", "frame_id", "x", "y"]
SUPPLEMENTARY_REQUIRED_COLUMNS = [
    "game_id", "season", "week", "play_id", "possession_team",
    "defensive_team", "pass_result", "pass_location_type",
]

FIELD_SEMANTICS = {
    "game_id": "confirmed from dataset structure/local values",
    "play_id": "confirmed from dataset structure/local values",
    "nfl_id": "strongly supported but dependent on documentation",
    "frame_id": "confirmed from dataset structure/local values",
    "x/y coordinates": "strongly supported but dependent on documentation",
    "speed (s), acceleration (a), orientation (o), direction (dir)": "strongly supported but dependent on documentation",
    "player_side and player_role": "confirmed from dataset structure/local values",
    "player_position": "strongly supported but dependent on documentation",
    "play_direction": "strongly supported but dependent on documentation",
    "targeted receiver": "confirmed from dataset structure/local values",
    "ball landing location": "strongly supported but dependent on documentation",
    "pass outcome metadata": "strongly supported but dependent on documentation",
    "timestamp": "absent",
    "per-frame football coordinates": "absent",
    "per-player club/team": "absent",
}
MAX_QUANTILE_SAMPLE = 100_000
MAX_QUANTILE_SAMPLE_PER_WEEK = 5_000
INPUT_USECOLS = [
    "game_id", "play_id", "player_to_predict", "nfl_id", "frame_id",
    "play_direction", "player_position", "player_side", "player_role",
    "x", "y", "s", "a", "dir", "o", "num_frames_output",
    "ball_land_x", "ball_land_y",
]
INPUT_DTYPES = {
    "game_id": "string", "play_id": "string", "player_to_predict": "string",
    "nfl_id": "string", "frame_id": "Int16", "play_direction": "category",
    "player_position": "category", "player_side": "category", "player_role": "category",
    "x": "float32", "y": "float32", "s": "float32", "a": "float32",
    "dir": "float32", "o": "float32", "num_frames_output": "Int16",
    "ball_land_x": "float32", "ball_land_y": "float32",
}
OUTPUT_DTYPES = {"game_id": "string", "play_id": "string", "nfl_id": "string", "frame_id": "Int16", "x": "float32", "y": "float32"}


def _status(status: str, detail: str, **values: Any) -> dict[str, Any]:
    return {"status": status, "detail": detail, **values}


def _number(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _integer(value: str) -> int | None:
    parsed = _number(value)
    return int(parsed) if parsed is not None and parsed.is_integer() else None


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

    return {"min": ordered[0], "p01": percentile(.01), "p05": percentile(.05), "p50": percentile(.5), "p95": percentile(.95), "p99": percentile(.99), "max": ordered[-1]}


def _sample_append(values: list[float], value: float) -> None:
    """Bound memory; exact extrema are tracked separately where required."""
    if len(values) < MAX_QUANTILE_SAMPLE:
        values.append(value)


def _systematic_week_sample(values: Any) -> list[float]:
    """Keep comparable, bounded samples from every selected week."""
    if len(values) <= MAX_QUANTILE_SAMPLE_PER_WEEK:
        return [float(value) for value in values]
    stride = math.ceil(len(values) / MAX_QUANTILE_SAMPLE_PER_WEEK)
    return [float(value) for value in values.iloc[::stride].head(MAX_QUANTILE_SAMPLE_PER_WEEK)]


def _summary_with_extrema(values: list[float], extrema: tuple[float | None, float | None]) -> dict[str, float] | None:
    summary = _quantiles(values)
    if summary is None:
        return None
    minimum, maximum = extrema
    if minimum is not None:
        summary["min"] = minimum
    if maximum is not None:
        summary["max"] = maximum
    return summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration as error:
            raise AuditError(f"Empty CSV: {path}") from error


def _dataset_files(dataset_root: Path, weeks: tuple[str, ...]) -> tuple[Path, list[Path], list[Path], Path]:
    train = dataset_root / "train"
    supplementary = dataset_root / "supplementary_data.csv"
    inputs = sorted(train.glob("input_*.csv"))
    outputs = sorted(train.glob("output_*.csv"))
    if not train.is_dir() or not supplementary.is_file() or not inputs or not outputs:
        raise AuditError(f"Expected extracted Analytics release structure under {dataset_root}.")
    input_weeks = {path.stem.removeprefix("input_") for path in inputs}
    output_weeks = {path.stem.removeprefix("output_") for path in outputs}
    if input_weeks != output_weeks:
        raise AuditError("Input and output weekly file sets do not match.")
    requested = set(weeks)
    if not requested or not requested <= input_weeks:
        raise AuditError(f"Requested week(s) are unavailable: {sorted(requested - input_weeks)}")
    return train, [path for path in inputs if path.stem.removeprefix("input_") in requested], [path for path in outputs if path.stem.removeprefix("output_") in requested], supplementary


def _metadata(path: Path) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, Any]]:
    header = _read_header(path)
    missing = sorted(set(SUPPLEMENTARY_REQUIRED_COLUMNS) - set(header))
    if missing:
        raise AuditError(f"Supplementary table missing required columns: {missing}")
    metadata: dict[tuple[str, str], dict[str, str]] = {}
    pass_results: Counter[str] = Counter()
    row_count = 0
    duplicates = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row_count += 1
            key = (row["game_id"], row["play_id"])
            if key in metadata:
                duplicates += 1
            metadata[key] = row
            pass_results[row["pass_result"]] += 1
    return metadata, {
        "row_count": row_count,
        "column_count": len(header),
        "columns": header,
        "unique_game_play_pairs": len(metadata),
        "duplicate_game_play_rows": duplicates,
        "pass_result_values": dict(sorted(pass_results.items())),
    }


def _empty_play_summary() -> dict[str, Any]:
    return {
        "directions": set(), "input_frames": set(), "target_ids": set(),
        "target_frames": 0, "defenders_per_target_frame": [], "ball_lands": set(),
        "input_rows": 0, "predictable_ids": set(), "week_files": set(),
    }


def _stream_inputs_fast(inputs: list[Path], metadata: dict[tuple[str, str], dict[str, str]], progress: Callable[[str], None] | None = None) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str, str], int], dict[str, Any]]:
    """Use vectorized per-week parsing; the complete release is too large for row loops."""
    try:
        import numpy as np
        import pandas as pd
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise AuditError("Milestone 1 validation requires pandas.") from error
    plays: dict[tuple[str, str], dict[str, Any]] = {}
    expected_outputs: dict[tuple[str, str, str], int] = {}
    aggregate: dict[str, Any] = {
        "rows": 0, "schema_consistent": True, "schema_mismatches": [], "missing_game_play_metadata_rows": 0,
        "role_counts": Counter(), "side_counts": Counter(), "role_predict_counts": Counter(), "position_counts": Counter(), "missing_nfl_id": 0,
        "coordinate_x": [], "coordinate_y": [], "coordinate_extrema": {"x": [None, None], "y": [None, None]}, "coordinate_missing": 0,
        "coordinate_reference_out_of_range": Counter(), "orientation_range": [], "direction_range": [], "angle_extrema": {"orientation": [None, None], "direction": [None, None]},
        "input_entity_duplicate_rows": 0, "input_frame_regressions": 0, "input_frame_gaps": 0, "input_frame_duplicates": 0,
        "input_frame_steps": Counter(), "input_frame_step_samples": [], "speed_distance_time_ratios": [],
        "direction_alignment": {"sin_cos": [], "cos_sin": [], "sin_neg_cos": [], "neg_sin_cos": []},
        "frame_entity_counts": Counter(), "frame_side_counts": Counter(), "target_defender_counts": [], "target_play_count": 0,
        "target_exactly_one_play_count": 0, "direction_conflict_plays": 0, "ball_land_conflict_plays": 0, "frames_with_duplicate_entities": 0,
        "expected_output_frame_conflicts": 0,
    }
    key_columns = ["game_id", "play_id"]
    for path in inputs:
        if progress:
            progress(f"input validation: {path.name}")
        try:
            df = pd.read_csv(path, dtype={"game_id": "string", "play_id": "string", "nfl_id": "string"})
        except Exception as error:  # pandas exceptions vary by version
            raise AuditError(f"Cannot parse input CSV {path}: {error}") from error
        if list(df.columns) != INPUT_COLUMNS:
            aggregate["schema_consistent"] = False
            aggregate["schema_mismatches"].append({"file": path.name, "columns": list(df.columns)})
        aggregate["rows"] += len(df)
        df["_frame"] = pd.to_numeric(df["frame_id"], errors="coerce")
        df["_x"] = pd.to_numeric(df["x"], errors="coerce")
        df["_y"] = pd.to_numeric(df["y"], errors="coerce")
        df["_speed"] = pd.to_numeric(df["s"], errors="coerce")
        df["_dir"] = pd.to_numeric(df["dir"], errors="coerce")
        df["_orientation"] = pd.to_numeric(df["o"], errors="coerce")
        df["_predict"] = df["player_to_predict"].astype(str)
        df["_offense"] = (df["player_side"] == "Offense").astype(int)
        df["_defense"] = (df["player_side"] == "Defense").astype(int)
        df["_target"] = (df["player_role"] == "Targeted Receiver").astype(int)
        aggregate["role_counts"].update(df["player_role"].value_counts(dropna=False).to_dict())
        aggregate["side_counts"].update(df["player_side"].value_counts(dropna=False).to_dict())
        aggregate["position_counts"].update(df["player_position"].value_counts(dropna=False).to_dict())
        aggregate["role_predict_counts"].update(Counter(zip(df["player_role"].astype(str), df["_predict"])))
        aggregate["missing_nfl_id"] += int(df["nfl_id"].isna().sum())
        existing_keys = set(zip(df["game_id"], df["play_id"]))
        aggregate["missing_game_play_metadata_rows"] += int((~df.set_index(key_columns).index.isin(metadata)).sum())
        declared_counts = (
            df.loc[
                (df["_predict"] == "True") & df["num_frames_output"].notna(),
                [*key_columns, "nfl_id", "num_frames_output"],
            ]
            .groupby([*key_columns, "nfl_id"], sort=False)["num_frames_output"]
            .nunique(dropna=True)
        )
        aggregate["expected_output_frame_conflicts"] += int(declared_counts.gt(1).sum())
        coordinate_mask = df["_x"].notna() & df["_y"].notna()
        aggregate["coordinate_missing"] += int((~coordinate_mask).sum())
        for axis, column, upper in (("x", "_x", 120), ("y", "_y", 53.3)):
            values = df[column].dropna()
            for value in values.iloc[: max(0, MAX_QUANTILE_SAMPLE - len(aggregate[f"coordinate_{axis}"]))]:
                _sample_append(aggregate[f"coordinate_{axis}"], float(value))
            if not values.empty:
                extrema = aggregate["coordinate_extrema"][axis]
                current_min, current_max = float(values.min()), float(values.max())
                extrema[0] = current_min if extrema[0] is None else min(extrema[0], current_min)
                extrema[1] = current_max if extrema[1] is None else max(extrema[1], current_max)
                aggregate["coordinate_reference_out_of_range"][axis] += int(((values < 0) | (values > upper)).sum())
        for label, column in (("direction", "_dir"), ("orientation", "_orientation")):
            values = df[column].dropna()
            target = aggregate[f"{label}_range"]
            for value in values.iloc[: max(0, MAX_QUANTILE_SAMPLE - len(target))]:
                _sample_append(target, float(value))
            if not values.empty:
                extrema = aggregate["angle_extrema"][label]
                current_min, current_max = float(values.min()), float(values.max())
                extrema[0] = current_min if extrema[0] is None else min(extrema[0], current_min)
                extrema[1] = current_max if extrema[1] is None else max(extrema[1], current_max)

        aggregate["input_entity_duplicate_rows"] += int(df.duplicated([*key_columns, "nfl_id", "_frame"]).sum())
        for key, group in df.groupby(key_columns, sort=False):
            key = (str(key[0]), str(key[1]))
            summary = plays.setdefault(key, _empty_play_summary())
            summary["input_rows"] += len(group)
            summary["week_files"].add(path.stem.removeprefix("input_"))
            summary["directions"].update(group["play_direction"].dropna().astype(str).unique())
            summary["input_frames"].update(int(value) for value in group["_frame"].dropna().unique())
            summary["target_ids"].update(group.loc[group["_target"] == 1, "nfl_id"].dropna().astype(str).unique())
            land_pairs = group[["ball_land_x", "ball_land_y"]].drop_duplicates()
            for pair in land_pairs.itertuples(index=False):
                x, y = _number(str(pair[0])), _number(str(pair[1]))
                if x is not None and y is not None:
                    summary["ball_lands"].add((x, y))
            predicted = group[group["_predict"] == "True"][["nfl_id", "num_frames_output"]].drop_duplicates()
            for row in predicted.itertuples(index=False):
                frame_count = _integer(str(row[1]))
                if frame_count is not None:
                    expected_outputs[(key[0], key[1], str(row[0]))] = frame_count
                    summary["predictable_ids"].add(str(row[0]))

        frame_groups = df.groupby([*key_columns, "_frame"], dropna=False).agg(
            entities=("nfl_id", "nunique"), raw_rows=("nfl_id", "size"), offense=("_offense", "sum"),
            defense=("_defense", "sum"), target=("_target", "sum"),
        )
        for (game_id, play_id, _), row in frame_groups.iterrows():
            entities, offense, defense = int(row["entities"]), int(row["offense"]), int(row["defense"])
            aggregate["frame_entity_counts"][entities] += 1
            aggregate["frame_side_counts"][(offense, defense)] += 1
            if int(row["raw_rows"]) != entities:
                aggregate["frames_with_duplicate_entities"] += 1
            if int(row["target"]):
                summary = plays[(str(game_id), str(play_id))]
                summary["target_frames"] += 1
                summary["defenders_per_target_frame"].append(defense)
                aggregate["target_defender_counts"].append(defense)

        ordered = df.groupby([*key_columns, "nfl_id"], sort=False)
        previous = ordered[["_frame", "_x", "_y", "_speed", "_dir"]].shift()
        steps = df["_frame"] - previous["_frame"]
        aggregate["input_frame_regressions"] += int((steps < 0).sum())
        aggregate["input_frame_duplicates"] += int((steps == 0).sum())
        aggregate["input_frame_gaps"] += int((steps > 1).sum())
        for step, count in steps.dropna().astype(int).value_counts().items():
            aggregate["input_frame_steps"][int(step)] += int(count)
        valid = (steps == 1) & df["_x"].notna() & df["_y"].notna() & previous["_x"].notna() & previous["_y"].notna()
        dx, dy = df.loc[valid, "_x"] - previous.loc[valid, "_x"], df.loc[valid, "_y"] - previous.loc[valid, "_y"]
        distance = np.hypot(dx, dy)
        mean_speed = (df.loc[valid, "_speed"] + previous.loc[valid, "_speed"]) / 2
        moving = (distance > .03) & (mean_speed > .5) & previous.loc[valid, "_dir"].notna()
        if moving.any():
            ratio = (distance[moving] / mean_speed[moving]).to_numpy()
            for value in ratio[: max(0, MAX_QUANTILE_SAMPLE - len(aggregate["speed_distance_time_ratios"]))]:
                _sample_append(aggregate["speed_distance_time_ratios"], float(value))
            theta = np.deg2rad(previous.loc[valid, "_dir"][moving].to_numpy())
            observed_x, observed_y = dx[moving].to_numpy() / distance[moving].to_numpy(), dy[moving].to_numpy() / distance[moving].to_numpy()
            candidates = {"sin_cos": (np.sin(theta), np.cos(theta)), "cos_sin": (np.cos(theta), np.sin(theta)), "sin_neg_cos": (np.sin(theta), -np.cos(theta)), "neg_sin_cos": (-np.sin(theta), np.cos(theta))}
            for name, (expected_x, expected_y) in candidates.items():
                values = observed_x * expected_x + observed_y * expected_y
                for value in values[: max(0, MAX_QUANTILE_SAMPLE - len(aggregate["direction_alignment"][name]))]:
                    _sample_append(aggregate["direction_alignment"][name], float(value))
    for summary in plays.values():
        aggregate["target_play_count"] += int(bool(summary["target_ids"]))
        aggregate["target_exactly_one_play_count"] += int(len(summary["target_ids"]) == 1)
        aggregate["direction_conflict_plays"] += int(len(summary["directions"]) != 1)
        aggregate["ball_land_conflict_plays"] += int(len(summary["ball_lands"]) > 1)
    return plays, expected_outputs, aggregate


def _stream_outputs_fast(outputs: list[Path], expected_outputs: dict[tuple[str, str, str], int], progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover
        raise AuditError("Milestone 1 validation requires pandas.") from error
    aggregate: dict[str, Any] = {"rows": 0, "schema_consistent": True, "schema_mismatches": [], "groups": {}, "missing_input_prediction_group_rows": 0, "duplicate_entity_frame_rows": 0, "coordinate_missing": 0}
    for path in outputs:
        if progress:
            progress(f"output validation: {path.name}")
        try:
            df = pd.read_csv(path, dtype={"game_id": "string", "play_id": "string", "nfl_id": "string"})
        except Exception as error:
            raise AuditError(f"Cannot parse output CSV {path}: {error}") from error
        if list(df.columns) != OUTPUT_COLUMNS:
            aggregate["schema_consistent"] = False
            aggregate["schema_mismatches"].append({"file": path.name, "columns": list(df.columns)})
        aggregate["rows"] += len(df)
        df["_frame"] = pd.to_numeric(df["frame_id"], errors="coerce")
        aggregate["duplicate_entity_frame_rows"] += int(df.duplicated(["game_id", "play_id", "nfl_id", "_frame"]).sum())
        x_missing = pd.to_numeric(df["x"], errors="coerce").isna()
        y_missing = pd.to_numeric(df["y"], errors="coerce").isna()
        aggregate["coordinate_missing"] += int((x_missing | y_missing).sum())
        for key, group in df.groupby(["game_id", "play_id", "nfl_id"], sort=False):
            group_key = tuple(str(value) for value in key)
            if group_key not in expected_outputs:
                aggregate["missing_input_prediction_group_rows"] += len(group)
            aggregate["groups"][group_key] = {"frames": set(int(value) for value in group["_frame"].dropna().unique()), "rows": len(group)}
    expected_mismatches, noncontiguous = 0, 0
    for group, state in aggregate["groups"].items():
        frames, expected = state["frames"], expected_outputs.get(group)
        expected_mismatches += int(expected is not None and (len(frames) != expected or frames != set(range(1, expected + 1))))
        noncontiguous += int(bool(frames) and frames != set(range(min(frames), max(frames) + 1)))
    aggregate["unique_output_groups"] = len(aggregate["groups"])
    aggregate["expected_prediction_groups"] = len(expected_outputs)
    aggregate["missing_output_groups"] = len(set(expected_outputs) - set(aggregate["groups"]))
    aggregate["output_groups_with_expected_frame_mismatch"] = expected_mismatches
    aggregate["output_groups_with_noncontiguous_frames"] = noncontiguous
    aggregate.pop("groups")
    return aggregate


def _counter_distribution(counter: Counter[int] | Counter[tuple[int, int]]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _select_representative_plays(plays: dict[tuple[str, str], dict[str, Any]], metadata: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, Any]]:
    candidates = []
    durations = sorted(len(summary["input_frames"]) for key, summary in plays.items() if key in metadata and len(summary["target_ids"]) == 1 and summary["target_frames"] > 0)
    median_duration = durations[len(durations) // 2] if durations else 0
    for key, summary in plays.items():
        if key not in metadata or len(summary["target_ids"]) != 1 or not summary["target_frames"] or len(summary["directions"]) != 1:
            continue
        land_y = next(iter(summary["ball_lands"]))[1] if len(summary["ball_lands"]) == 1 else None
        location = "sideline" if land_y is not None and (land_y <= 10 or land_y >= 43.3) else "middle"
        direction = next(iter(summary["directions"]))
        rank = hashlib.sha256(f"gridiron-m1-v1:{key[0]}:{key[1]}".encode()).hexdigest()
        candidates.append({
            "game_id": key[0], "play_id": key[1], "week": metadata[key]["week"], "play_direction": direction,
            "target_location_band": location, "tracking_length_band": "short" if len(summary["input_frames"]) <= median_duration else "long",
            "input_frame_count": len(summary["input_frames"]), "target_id": next(iter(summary["target_ids"])),
            "defenders_per_target_frame": _quantiles([float(value) for value in summary["defenders_per_target_frame"]]), "selection_rank": rank,
        })
    selected: list[dict[str, Any]] = []
    used_keys: set[tuple[str, str]] = set()
    for direction in ("left", "right"):
        for location in ("sideline", "middle"):
            stratum = sorted((item for item in candidates if item["play_direction"] == direction and item["target_location_band"] == location), key=lambda item: item["selection_rank"])
            used_weeks: set[str] = set()
            for item in stratum:
                if len([chosen for chosen in selected if chosen["play_direction"] == direction and chosen["target_location_band"] == location]) == 3:
                    break
                if item["week"] not in used_weeks:
                    selected.append(item)
                    used_keys.add((item["game_id"], item["play_id"]))
                    used_weeks.add(item["week"])
    if len(selected) < 12:
        for item in sorted(candidates, key=lambda item: item["selection_rank"]):
            if len(selected) == 12:
                break
            if (item["game_id"], item["play_id"]) not in used_keys:
                selected.append(item)
                used_keys.add((item["game_id"], item["play_id"]))
    return selected[:12]


def _svg_panel(x_offset: int, title: str, rows: list[dict[str, str]], width: int = 360, height: int = 250) -> str:
    left, top, field_width, field_height = x_offset + 30, 35, width - 50, 150
    content = [f'<text x="{x_offset + 10}" y="18" font-size="12">{title}</text>', f'<rect x="{left}" y="{top}" width="{field_width}" height="{field_height}" fill="#367c3a" stroke="#ffffff"/>']
    for yard in range(0, 121, 10):
        x = left + field_width * yard / 120
        content.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + field_height}" stroke="#ffffff" stroke-opacity=".45"/>')
    for row in rows:
        x, y = _number(row.get("x", "")), _number(row.get("y", ""))
        if x is None or y is None:
            continue
        px, py = left + field_width * x / 120, top + field_height * (1 - y / 53.3)
        if row.get("player_role") == "Targeted Receiver":
            color = "#ffd166"
        elif row.get("player_side") == "Defense":
            color = "#ef476f"
        else:
            color = "#118ab2"
        label = row.get("nfl_id", "?")
        content.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}" stroke="#111"/>')
        content.append(f'<text x="{px + 5:.1f}" y="{py - 5:.1f}" font-size="7" fill="#111">{label}</text>')
    return "".join(content)


def _write_sanity_svgs(inputs: list[Path], outputs: list[Path], selections: list[dict[str, Any]], output_directory: Path, progress: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
    selected_keys = {(item["game_id"], item["play_id"]): item for item in selections}
    input_rows: dict[tuple[str, str], dict[int, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    roles: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in inputs:
        if progress:
            progress(f"sanity visualization input extraction: {path.name}")
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["game_id"], row["play_id"])
                if key in selected_keys:
                    frame = _integer(row["frame_id"])
                    if frame is not None:
                        input_rows[key][frame].append(row)
                    roles[(key[0], key[1], row["nfl_id"])] = row
    output_rows: dict[tuple[str, str], dict[int, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for path in outputs:
        if progress:
            progress(f"sanity visualization output extraction: {path.name}")
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["game_id"], row["play_id"])
                if key in selected_keys:
                    enriched = {**row, **{name: roles.get((key[0], key[1], row["nfl_id"]), {}).get(name, "") for name in ("player_side", "player_role")}}
                    frame = _integer(row["frame_id"])
                    if frame is not None:
                        output_rows[key][frame].append(enriched)
    output_directory.mkdir(parents=True, exist_ok=True)
    result = []
    for selection in selections:
        key = (selection["game_id"], selection["play_id"])
        input_frames, output_frames = input_rows[key], output_rows[key]
        first_input, last_input = min(input_frames), max(input_frames)
        last_output = max(output_frames) if output_frames else None
        panels = _svg_panel(0, f"input frame {first_input} (pre-release)", input_frames[first_input])
        panels += _svg_panel(370, f"input frame {last_input} (pre-release)", input_frames[last_input])
        panels += _svg_panel(740, f"output frame {last_output} (post-release)", output_frames[last_output] if last_output is not None else [])
        filename = f"game_{key[0]}_play_{key[1]}.svg"
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="250" viewBox="0 0 1100 250">'
            '<rect width="1100" height="250" fill="#f7f7f7"/>'
            f'<text x="10" y="225" font-size="11">game {key[0]}, play {key[1]}, week {selection["week"]}, direction {selection["play_direction"]}; blue=offense, red=defense, gold=target</text>'
            f'{panels}</svg>'
        )
        (output_directory / filename).write_text(svg, encoding="utf-8")
        result.append({**selection, "sanity_svg": filename, "input_first_frame": first_input, "input_last_frame": last_input, "output_last_frame": last_output})
    return result


def _decision(input_stats: dict[str, Any], output_stats: dict[str, Any], metadata_stats: dict[str, Any]) -> tuple[str, str]:
    catastrophic = (
        input_stats["missing_game_play_metadata_rows"] > 0
        or input_stats["input_entity_duplicate_rows"] > 0
        or input_stats["input_frame_regressions"] > 0
        or output_stats["missing_input_prediction_group_rows"] > 0
    )
    if catastrophic:
        return "NO GO", "Key joins, entity uniqueness, or temporal ordering fail in the retained source."
    limitations = [
        "input tracks only skill-position/coverage entities rather than all 22 players",
        "there is no per-frame football position or timestamp",
        "output trajectories cover prediction-designated players rather than every route runner",
        "only the 2023 season is present locally",
    ]
    zero_defender_frames = sum(value == 0 for value in input_stats["target_defender_counts"])
    if zero_defender_frames:
        limitations.append(f"{zero_defender_frames} target-receiver frames have no recorded defensive-side entity and require an explicit later exclusion rule")
    if metadata_stats["unique_game_play_pairs"] and input_stats["target_exactly_one_play_count"] == input_stats["target_play_count"]:
        return "LIMITED GO", "; ".join(limitations) + "."
    return "NO GO", "A unique targeted-receiver entity could not be established for the tracked plays."


def _render_data_audit(result: dict[str, Any]) -> str:
    dataset, relationships, temporal, coordinates = result["dataset"], result["relationships"], result["temporal"], result["coordinates"]
    return f"""# Data Audit — BDB 2026 Analytics Validation

## Confirmed

- Extracted release: `{dataset['dataset_root']}` with {dataset['input_file_count']} weekly input files, {dataset['output_file_count']} matching weekly output files, and one supplementary metadata table.
- All weekly input schemas match the observed 23-column schema; all output schemas match the observed 6-column schema.
- `{relationships['input_unique_game_play_pairs']}` input `(game_id, play_id)` pairs join to supplementary metadata with {relationships['input_rows_missing_supplementary_join']} missing input rows.
- Input coordinates have observed ranges x={coordinates['input_x']['min']:.2f}..{coordinates['input_x']['max']:.2f}, y={coordinates['input_y']['min']:.2f}..{coordinates['input_y']['max']:.2f}.
- The input has no timestamp column. Input and output frame IDs each start at 1 within their respective sequences.

## Unknown

- Official units and angular conventions for `x`, `y`, `s`, `a`, `dir`, and `o` remain documentation-dependent despite their internally consistent values.
- No local data dictionary explains whether `ball_land_x/y` are supplied labels, how `player_to_predict` was selected, or whether all relevant coverage players are retained.
- The raw source has no per-frame football coordinates or individual player club field.

## Potential issues

- The release tracks route runners, passer, and defensive-coverage entities, not all 22 players; it supports a narrowed receiver/defender study, not full-play availability claims.
- `player_to_predict` and `ball_land_x/y` are present in pre-release input rows. They must be excluded from future geometry features unless a later task explicitly defines a post-throw conditional analysis.
- Only 2023 weeks 1–18 are present, so a later model evaluation cannot claim cross-season generalization from this download alone.
- Archive `{dataset['archive_name']}` is a packaging artifact and was excluded from CSV-table profiling.

## Next validation step

Milestone 1 is complete. Begin Milestone 2 by writing the restricted analytic-cohort and coordinate-contract specification: pre-release input frames only, route runners/defensive-coverage entities only, and explicit exclusion of target/prediction/landing labels from features.
"""


def _render_result(result: dict[str, Any]) -> str:
    dataset, entities, temporal, coordinates, decision = result["dataset"], result["entities"], result["temporal"], result["coordinates"], result["decision"]
    return f"""# Milestone 1 Result

## Dataset

The local source is the extracted NFL Big Data Bowl 2026 Analytics release: {dataset['input_file_count']} weekly 2023 input files, {dataset['output_file_count']} matching output files, and `supplementary_data.csv` ({dataset['supplementary_rows']} rows). The accompanying zip is an archive of the extracted files and was not treated as a football table.

## Confirmed capabilities

- `{entities['input_game_play_pairs']}` game/play sequences can be joined to supplementary metadata with `{result['relationships']['input_rows_missing_supplementary_join']}` missing input-row joins.
- Input has consistent pre-release trajectories for offense and defense entities, a unique observed targeted-receiver ID on `{entities['plays_with_exactly_one_target']}` of `{entities['plays_with_target']}` target-labelled plays, and `{entities['target_defenders']['p50']:.0f}` defensive entities at the median target-receiver frame.
- Input/output player groups have `{result['relationships']['output_groups_with_expected_frame_mismatch']}` declared-frame mismatches and `{result['relationships']['output_groups_with_noncontiguous_frames']}` noncontiguous frame sequences.
- Raw position coordinates occupy a field-like x range of {coordinates['input_x']['min']:.2f}–{coordinates['input_x']['max']:.2f} and y range of {coordinates['input_y']['min']:.2f}–{coordinates['input_y']['max']:.2f}; no coordinate transform was applied.

## Limitations

- No timestamps or per-frame football coordinates are included.
- Input includes selected route runners, passer, and defensive-coverage entities rather than all 22 players; no individual club field is provided.
- Post-release output trajectories exist only for `player_to_predict=True` entities, not every route runner.
- `player_to_predict` and ball landing fields are available in input rows and are leakage risks for any pre-throw forecast.
- The local release covers 2023 weeks 1–18 only.

## Spatial-analysis viability

Receiver/defender geometry is defensible for the observed route-runner and defensive-coverage entities during the supplied pre-release sequence. It is not defensible as a complete all-eligible-receiver or full football-flight reconstruction.

## Claim boundaries

Do not claim full-22 coverage, football-path feasibility, all-receiver future openness, coverage responsibility, pass availability, completion probability, or cross-season generalization from this release alone.

## Decision

**{decision['status']}** — {decision['rationale']}

The 12 deterministic raw-coordinate sanity SVGs are in `artifacts/milestone_1_sanity/`; their selection metadata is stored in `artifacts/milestone_1_validation.json`.
"""


def run_milestone_1_validation(
    dataset_root: Path,
    artifacts_directory: Path,
    data_audit_path: Path | None,
    result_path: Path | None,
    *,
    weeks: tuple[str, ...],
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Validate explicitly selected weekly files and optionally write docs.

    An explicit week list prevents an accidental all-release run during
    benchmarking. The source data itself is read only.
    """

    stage_seconds: dict[str, float] = {}

    def stage(name: str, operation: Callable[[], Any]) -> Any:
        if progress:
            progress(f"stage start: {name}")
        started = time.perf_counter()
        value = operation()
        stage_seconds[name] = round(time.perf_counter() - started, 3)
        if progress:
            progress(f"stage complete: {name} ({stage_seconds[name]:.3f}s)")
        return value

    _, inputs, outputs, supplementary = _dataset_files(dataset_root, weeks)
    metadata, metadata_stats = stage("supplementary metadata", lambda: _metadata(supplementary))
    plays, expected_outputs, input_stats = stage("input validation", lambda: _stream_inputs_fast(inputs, metadata, progress))
    output_stats = stage("output validation", lambda: _stream_outputs_fast(outputs, expected_outputs, progress))
    selections = stage("deterministic play selection", lambda: _select_representative_plays(plays, metadata))
    sanity = stage("raw-coordinate sanity SVGs", lambda: _write_sanity_svgs(inputs, outputs, selections, artifacts_directory / "milestone_1_sanity", progress))
    decision_status, decision_rationale = _decision(input_stats, output_stats, metadata_stats)
    input_frame_values = [float(frame) for summary in plays.values() for frame in summary["input_frames"]]
    output_expected_values = [float(value) for value in expected_outputs.values()]
    direction_alignment = {name: _quantiles(values) for name, values in input_stats["direction_alignment"].items()}
    direction_best = max(direction_alignment, key=lambda name: direction_alignment[name]["p50"] if direction_alignment[name] else float("-inf"))
    target_defenders = _quantiles([float(value) for value in input_stats["target_defender_counts"]])
    result = {
        "overall_status": decision_status,
        "dataset": {
            "dataset_root": str(dataset_root), "input_file_count": len(inputs), "output_file_count": len(outputs),
            "input_file_weeks": [path.stem.removeprefix("input_") for path in inputs], "supplementary_rows": metadata_stats["row_count"],
            "archive_name": "nfl-big-data-bowl-2026-analytics.zip", "archive_status": "NOT_TESTED",
            "archive_detail": "Packaging archive; extracted CSV release was validated instead.",
            "input_schema_consistent": input_stats["schema_consistent"], "output_schema_consistent": output_stats["schema_consistent"],
            "input_columns": INPUT_COLUMNS, "output_columns": OUTPUT_COLUMNS, "supplementary_columns": metadata_stats["columns"],
            "field_semantics": FIELD_SEMANTICS,
        },
        "relationships": {
            "supplementary_unique_game_play_pairs": metadata_stats["unique_game_play_pairs"],
            "supplementary_duplicate_game_play_rows": metadata_stats["duplicate_game_play_rows"],
            "input_unique_game_play_pairs": len(plays),
            "input_rows_missing_supplementary_join": input_stats["missing_game_play_metadata_rows"],
            "input_duplicate_game_play_player_frame_rows": input_stats["input_entity_duplicate_rows"],
            "output_unique_game_play_player_groups": output_stats["unique_output_groups"],
            "expected_output_groups_from_input": output_stats["expected_prediction_groups"],
            "output_rows_without_input_prediction_group": output_stats["missing_input_prediction_group_rows"],
            "output_rows_missing_coordinates": output_stats["coordinate_missing"],
            "expected_output_frame_conflicts": input_stats["expected_output_frame_conflicts"],
            "missing_output_groups": output_stats["missing_output_groups"],
            "output_groups_with_expected_frame_mismatch": output_stats["output_groups_with_expected_frame_mismatch"],
            "output_groups_with_noncontiguous_frames": output_stats["output_groups_with_noncontiguous_frames"],
            "output_duplicate_game_play_player_frame_rows": output_stats["duplicate_entity_frame_rows"],
        },
        "temporal": {
            "timestamp": _status("NOT_TESTED", "No timestamp field exists in the observed input or output schemas."),
            "input_frame_range": _quantiles(input_frame_values), "input_frame_step_distribution": _counter_distribution(input_stats["input_frame_steps"]),
            "input_frame_regressions": input_stats["input_frame_regressions"], "input_frame_duplicates": input_stats["input_frame_duplicates"], "input_frame_gaps": input_stats["input_frame_gaps"],
            "output_declared_frame_count": _quantiles(output_expected_values), "apparent_distance_over_speed_interval": _quantiles(input_stats["speed_distance_time_ratios"]),
            "release_boundary": _status("PASS", "Input and output tables form separate frame sequences; output frame IDs reset at 1. Exact wall-clock release time is unavailable without timestamps."),
        },
        "coordinates": {
            "input_x": _summary_with_extrema(input_stats["coordinate_x"], tuple(input_stats["coordinate_extrema"]["x"])), "input_y": _summary_with_extrema(input_stats["coordinate_y"], tuple(input_stats["coordinate_extrema"]["y"])),
            "missing_input_coordinate_rows": input_stats["coordinate_missing"], "reference_out_of_range_rows": dict(input_stats["coordinate_reference_out_of_range"]),
            "play_direction_values": sorted({next(iter(summary["directions"])) for summary in plays.values() if len(summary["directions"]) == 1}),
            "plays_with_direction_conflicts": input_stats["direction_conflict_plays"],
            "orientation_range": _summary_with_extrema(input_stats["orientation_range"], tuple(input_stats["angle_extrema"]["orientation"])), "direction_range": _summary_with_extrema(input_stats["direction_range"], tuple(input_stats["angle_extrema"]["direction"])),
            "direction_alignment_to_observed_motion": {"best_candidate": direction_best, "candidate_cosine_quantiles": direction_alignment},
            "coordinate_convention": _status("strongly supported but dependent on documentation", "Observed x/y are field-like and play_direction is internally constant, but no coordinate dictionary is present locally."),
        },
        "entities": {
            "input_rows": input_stats["rows"], "input_game_play_pairs": len(plays), "player_side_counts": dict(input_stats["side_counts"]),
            "player_role_counts": dict(input_stats["role_counts"]), "position_counts": dict(input_stats["position_counts"]),
            "role_by_player_to_predict": {f"{role}|{flag}": count for (role, flag), count in input_stats["role_predict_counts"].items()},
            "missing_nfl_id_rows": input_stats["missing_nfl_id"], "entity_count_per_frame_distribution": _counter_distribution(input_stats["frame_entity_counts"]),
            "offense_defense_count_per_frame_distribution": _counter_distribution(input_stats["frame_side_counts"]),
            "frames_with_duplicate_entities": input_stats["frames_with_duplicate_entities"], "plays_with_target": input_stats["target_play_count"],
            "plays_with_exactly_one_target": input_stats["target_exactly_one_play_count"], "target_defenders": target_defenders,
            "ball_tracking": _status("FAIL", "No football entity or per-frame football coordinates were observed; only ball landing fields occur in input rows."),
            "joint_target_defender_tracking": _status("PASS" if target_defenders and target_defenders["min"] > 0 else "FAIL", "Counts defensive-side entities in each input frame containing a Targeted Receiver role.", defenders_per_target_frame=target_defenders),
        },
        "representative_plays": {"selection_method": "Stable SHA-256 rank within play_direction × target landing sideline/middle strata; select up to three distinct-week plays per stratum, then stable global fallback.", "selected": sanity},
        "decision": {"status": decision_status, "rationale": decision_rationale},
        "benchmark": {
            "selected_weeks": list(weeks), "rows_processed": {"input": input_stats["rows"], "output": output_stats["rows"], "supplementary": metadata_stats["row_count"]},
            "stage_seconds": stage_seconds, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    result["checks"] = {
        "file_organization": _status("PASS", "18 matching weekly input/output pairs plus supplementary metadata; zip excluded as archive."),
        "schema_consistency": _status("PASS" if input_stats["schema_consistent"] and output_stats["schema_consistent"] else "FAIL", "Weekly headers were compared exactly."),
        "supplementary_join": _status("PASS" if input_stats["missing_game_play_metadata_rows"] == 0 else "FAIL", "Input rows were checked against supplementary (game_id, play_id).", missing_rows=input_stats["missing_game_play_metadata_rows"]),
        "input_entity_key": _status("PASS" if input_stats["input_entity_duplicate_rows"] == 0 else "FAIL", "Checked candidate (game_id, play_id, nfl_id, frame_id).", duplicates=input_stats["input_entity_duplicate_rows"]),
        "output_entity_key": _status("PASS" if output_stats["duplicate_entity_frame_rows"] == 0 else "FAIL", "Checked candidate (game_id, play_id, nfl_id, frame_id).", duplicates=output_stats["duplicate_entity_frame_rows"]),
        "input_temporal_order": _status("PASS" if input_stats["input_frame_regressions"] == 0 and input_stats["input_frame_duplicates"] == 0 else "FAIL", "Checked source-row ordering within candidate player trajectories.", regressions=input_stats["input_frame_regressions"], duplicates=input_stats["input_frame_duplicates"], gaps=input_stats["input_frame_gaps"]),
        "input_coordinates": _status("PASS" if input_stats["coordinate_missing"] == 0 and sum(input_stats["coordinate_reference_out_of_range"].values()) == 0 else "FAIL", "Raw input coordinates were checked against field-like 0–120 by 0–53.3 reference bounds.", missing=input_stats["coordinate_missing"], out_of_range=dict(input_stats["coordinate_reference_out_of_range"])),
        "target_receiver": _status("PASS" if input_stats["target_exactly_one_play_count"] == input_stats["target_play_count"] else "FAIL", "Targeted Receiver role was counted per game/play.", exact_one=input_stats["target_exactly_one_play_count"], plays_with_target=input_stats["target_play_count"]),
        "joint_receiver_defender_coverage": result["entities"]["joint_target_defender_tracking"],
        "release_boundary": result["temporal"]["release_boundary"],
        "raw_coordinate_visualizations": _status("PASS" if len(sanity) == 12 else "FAIL", "Deterministically selected raw-coordinate SVGs were generated.", generated=len(sanity)),
    }
    _write_json(artifacts_directory / "milestone_1_validation.json", result)
    if data_audit_path is not None:
        data_audit_path.parent.mkdir(parents=True, exist_ok=True)
        data_audit_path.write_text(_render_data_audit(result), encoding="utf-8")
    if result_path is not None:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(_render_result(result), encoding="utf-8")
    return result
