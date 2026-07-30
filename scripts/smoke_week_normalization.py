"""Read-only Week 01 coordinate-normalization smoke test."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gridiron_spatial.cohort import split_for_week  # noqa: E402
from gridiron_spatial.coordinate_frame import (  # noqa: E402
    add_normalized_coordinates,
)
from gridiron_spatial.normalized_tracking import (  # noqa: E402
    NORMALIZED_ENTITY_FRAME_KEY,
    freeze_normalized_entity_frames,
    reconcile_normalized_entity_frames,
)


DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bdb_2026"
    / "114239_nfl_competition_files_published_analytics_final"
)
WEEK = "2023_w01"
EXPECTED_ROWS = {"input": 285_714, "output": 32_088, "combined": 317_802}
IDENTITY_COLUMNS = ["game_id", "play_id", "nfl_id"]
CONTEXT_COLUMNS = [
    "player_side",
    "player_role",
    "player_position",
    "player_to_predict",
    "play_direction",
]
RAW_COLUMNS = [
    "game_id",
    "play_id",
    "phase",
    "frame_id",
    "nfl_id",
    "week",
    "week_number",
    "split",
    *CONTEXT_COLUMNS,
    "x",
    "y",
    "dir",
    "o",
]


def _read_csv(path: Path, usecols: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required Week 01 file is missing: {path}")
    return pd.read_csv(
        path,
        usecols=usecols,
        dtype={"game_id": "string", "play_id": "string", "nfl_id": "string"},
        low_memory=False,
    )


def _attach_output_metadata(
    inputs: pd.DataFrame,
    outputs: pd.DataFrame,
) -> pd.DataFrame:
    metadata_source = inputs[[*IDENTITY_COLUMNS, *CONTEXT_COLUMNS]]
    ambiguity = (
        metadata_source.groupby(IDENTITY_COLUMNS, dropna=False)[
            CONTEXT_COLUMNS
        ]
        .nunique(dropna=False)
        .gt(1)
    )
    if ambiguity.any(axis=None):
        raise RuntimeError(
            "Ambiguous input metadata for output play/entity join"
        )
    metadata = metadata_source.drop_duplicates(IDENTITY_COLUMNS)
    joined = outputs.merge(
        metadata,
        on=IDENTITY_COLUMNS,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unmatched = int(joined["_merge"].ne("both").sum())
    if unmatched:
        raise RuntimeError(
            f"Unmatched output play/entity metadata rows: {unmatched}"
        )
    joined = joined.drop(columns="_merge")
    if joined[CONTEXT_COLUMNS].isna().any(axis=None):
        raise RuntimeError("Output entity metadata contains missing values")
    return joined


def _prepare_raw_frames(
    inputs: pd.DataFrame,
    outputs: pd.DataFrame,
) -> pd.DataFrame:
    input_rows = inputs.copy()
    input_rows["phase"] = "input"
    output_rows = _attach_output_metadata(inputs, outputs)
    output_rows["phase"] = "output"
    output_rows["dir"] = np.nan
    output_rows["o"] = np.nan
    for frame in (input_rows, output_rows):
        frame["week"] = WEEK
        frame["week_number"] = 1
        frame["split"] = split_for_week(WEEK)
    return pd.concat(
        [input_rows[RAW_COLUMNS], output_rows[RAW_COLUMNS]],
        ignore_index=True,
    )


def _category_counts(
    frame: pd.DataFrame,
    column: str,
    categories: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    return {
        phase: {
            category: int(
                (
                    frame["phase"].astype("string").eq(phase)
                    & frame[column].astype("string").eq(category)
                ).sum()
            )
            for category in categories
        }
        for phase in ("input", "output")
    }


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", required=True, choices=[WEEK])
    parser.parse_args(arguments)
    started = time.perf_counter()

    input_columns = [
        *IDENTITY_COLUMNS,
        "frame_id",
        *CONTEXT_COLUMNS,
        "x",
        "y",
        "dir",
        "o",
    ]
    output_columns = [*IDENTITY_COLUMNS, "frame_id", "x", "y"]
    inputs = _read_csv(
        DATASET_ROOT / "train" / f"input_{WEEK}.csv",
        input_columns,
    )
    outputs = _read_csv(
        DATASET_ROOT / "train" / f"output_{WEEK}.csv",
        output_columns,
    )
    raw = _prepare_raw_frames(inputs, outputs)
    observed_rows = {
        "input": int(len(inputs)),
        "output": int(len(outputs)),
        "combined": int(len(raw)),
    }
    if observed_rows != EXPECTED_ROWS:
        raise RuntimeError(
            f"Week 01 row-count mismatch: "
            f"observed={observed_rows}, expected={EXPECTED_ROWS}"
        )

    raw_duplicate_keys = int(
        raw.duplicated(NORMALIZED_ENTITY_FRAME_KEY).sum()
    )
    if raw_duplicate_keys:
        raise RuntimeError(
            f"Duplicate raw phase-qualified keys: {raw_duplicate_keys}"
        )
    normalized = freeze_normalized_entity_frames(
        add_normalized_coordinates(raw)
    )
    normalized_duplicate_keys = int(
        normalized.duplicated(NORMALIZED_ENTITY_FRAME_KEY).sum()
    )
    if normalized_duplicate_keys:
        raise RuntimeError(
            "Duplicate normalized phase-qualified keys: "
            f"{normalized_duplicate_keys}"
        )
    reconciliation = reconcile_normalized_entity_frames(raw, normalized)

    coordinate_classes = {
        coordinate_type: _category_counts(
            normalized,
            column,
            ("nominal", "extended_tolerance", "invalid"),
        )
        for coordinate_type, column in (
            ("raw", "raw_coordinate_class"),
            ("normalized", "normalized_coordinate_class"),
        )
    }
    transform_counts = _category_counts(
        normalized,
        "coordinate_transform_applied",
        ("identity_right", "rotate_180_left"),
    )
    unsupported_counts = {
        "phase": len(reconciliation["unsupported_phase_values"]),
        "coordinate_class": sum(
            len(values)
            for values in reconciliation[
                "unsupported_coordinate_class_values"
            ].values()
        ),
        "transform_version": len(
            reconciliation["unsupported_transform_versions"]
        ),
        "transform_applied": len(
            reconciliation["unsupported_transform_applied_values"]
        ),
        "transform_direction_mismatches": reconciliation[
            "transform_direction_mismatches"
        ],
        "invalid_raw_to_nominal_normalized": reconciliation[
            "invalid_raw_to_nominal_normalized"
        ],
    }
    summary = {
        "week": WEEK,
        "input_rows": observed_rows["input"],
        "output_rows": observed_rows["output"],
        "combined_rows": observed_rows["combined"],
        "unique_games": int(raw["game_id"].nunique()),
        "unique_plays": int(
            raw[["game_id", "play_id"]].drop_duplicates().shape[0]
        ),
        "phase_rows": {
            phase: int(raw["phase"].eq(phase).sum())
            for phase in ("input", "output")
        },
        "raw_duplicate_keys": raw_duplicate_keys,
        "normalized_duplicate_keys": normalized_duplicate_keys,
        "reconciliation_status": reconciliation["status"],
        "missing_normalized_keys": len(
            reconciliation["missing_normalized_keys"]
        ),
        "unexpected_normalized_keys": len(
            reconciliation["unexpected_normalized_keys"]
        ),
        "raw_field_mismatch_counts": reconciliation[
            "raw_field_mismatch_counts"
        ],
        "coordinate_class_counts_by_phase": coordinate_classes,
        "transform_applied_counts_by_phase": transform_counts,
        "play_direction_counts": {
            direction: int(raw["play_direction"].eq(direction).sum())
            for direction in ("left", "right")
        },
        "extended_tolerance_rows": int(
            normalized["raw_coordinate_class"]
            .astype("string")
            .eq("extended_tolerance")
            .sum()
        ),
        "invalid_coordinate_rows": int(
            normalized["raw_coordinate_class"]
            .astype("string")
            .eq("invalid")
            .sum()
        ),
        "unsupported_value_counts": unsupported_counts,
        "index_preserved": reconciliation["index_preserved"],
        "row_order_preserved": reconciliation["row_order_preserved"],
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(summary, indent=2))
    return 0 if reconciliation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
