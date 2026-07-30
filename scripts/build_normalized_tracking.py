"""Build partitioned normalized tracking artifacts for explicit weeks."""

from __future__ import annotations

import argparse
import functools
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gridiron_spatial.cohort import split_for_week  # noqa: E402
from gridiron_spatial.coordinate_frame import (  # noqa: E402
    add_normalized_coordinates,
)
from gridiron_spatial.normalized_artifacts import (  # noqa: E402
    write_normalized_tracking_release,
)
from gridiron_spatial.normalized_tracking import (  # noqa: E402
    NORMALIZED_ENTITY_FRAME_KEY,
    freeze_normalized_entity_frames,
    reconcile_normalized_entity_frames,
)
from scripts.smoke_two_week_cohort import parse_week_args  # noqa: E402
from scripts.smoke_week_normalization import (  # noqa: E402
    CONTEXT_COLUMNS,
    IDENTITY_COLUMNS,
    RAW_COLUMNS,
    _attach_output_metadata,
    _read_csv,
)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normalized tracking partitions for explicit weeks."
    )
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weeks", nargs="+", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parsed = parser.parse_args(args)
    parsed.weeks = parse_week_args(parsed.weeks)
    return parsed


def _build_week(release_root: Path, week: str) -> dict[str, object]:
    started = time.perf_counter()
    inputs = _read_csv(
        release_root / "train" / f"input_{week}.csv",
        [
            *IDENTITY_COLUMNS,
            "frame_id",
            *CONTEXT_COLUMNS,
            "x",
            "y",
            "dir",
            "o",
        ],
    )
    outputs = _read_csv(
        release_root / "train" / f"output_{week}.csv",
        [*IDENTITY_COLUMNS, "frame_id", "x", "y"],
    )
    input_rows = inputs.copy()
    input_rows["phase"] = "input"
    output_rows = _attach_output_metadata(inputs, outputs)
    output_rows["phase"] = "output"
    output_rows["dir"] = np.nan
    output_rows["o"] = np.nan
    week_number = int(week[-2:])
    split = split_for_week(week)
    for frame in (input_rows, output_rows):
        frame["week"] = week
        frame["week_number"] = week_number
        frame["split"] = split
    raw = pd.concat(
        [input_rows[RAW_COLUMNS], output_rows[RAW_COLUMNS]],
        ignore_index=True,
    )
    if raw.duplicated(NORMALIZED_ENTITY_FRAME_KEY).any():
        raise RuntimeError(f"{week} contains duplicate raw frozen keys")
    normalized = freeze_normalized_entity_frames(
        add_normalized_coordinates(raw)
    )
    reconciliation = reconcile_normalized_entity_frames(raw, normalized)
    return {
        "normalized_frame": normalized,
        "reconciliation": reconciliation,
        "input_rows": int(len(inputs)),
        "output_rows": int(len(outputs)),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }


def main(args: Sequence[str] | None = None) -> int:
    parsed = parse_args(args)
    try:
        write_normalized_tracking_release(
            parsed.output_dir,
            parsed.weeks,
            functools.partial(_build_week, parsed.release_root),
            overwrite=parsed.overwrite,
        )
    except Exception as error:
        print(f"normalized tracking build failed: {error}", file=sys.stderr)
        return 1
    print(f"output_directory={parsed.output_dir}")
    print(f"manifest_path={parsed.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
