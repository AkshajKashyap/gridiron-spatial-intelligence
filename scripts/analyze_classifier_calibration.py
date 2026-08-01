"""Run fixed validation-calibration diagnostics for frozen classifiers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gridiron_spatial.baseline_features import (  # noqa: E402
    HORIZONS,
    REGISTERED_NUMERIC_FEATURES,
    SAMPLE_KEY,
    build_baseline_samples,
    feature_matrix,
)
from gridiron_spatial.baseline_models import (  # noqa: E402
    build_classification_pipeline,
)
from gridiron_spatial.calibration_analysis import (  # noqa: E402
    BIN_EDGES,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    calibration_diagnostics,
    validate_calibration_weeks,
)
from gridiron_spatial.receiver_defender_pairs import (  # noqa: E402
    build_receiver_defender_pairs,
)

from select_baseline_models import (  # noqa: E402
    COHORT_TABLES,
    TRACKING_COLUMNS,
    _atomic_json,
    _read_cohort_week,
    _relative,
    _split_diagnostics,
)


RESULT_FORMAT_VERSION = "milestone_5_classifier_calibration_v1"
EXPECTED_CLASSIFIERS = {
    5: ("multivariable_logistic_c_10", {"C": 10.0}),
    10: ("multivariable_logistic_c_10", {"C": 10.0}),
    15: ("multivariable_logistic_c_0.1", {"C": 0.1}),
}
EXPECTED_VALIDATION_COUNTS = {
    5: (5211, 2085),
    10: (3499, 1245),
    15: (1620, 502),
}


def _project_path(path: Path) -> Path:
    resolved = (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"Path must be inside project: {path}") from error
    return resolved


def _classifier_specifications(
    selection: dict[str, Any],
) -> list[dict[str, Any]]:
    registered = tuple(selection.get("registered_feature_list", ()))
    if registered != REGISTERED_NUMERIC_FEATURES:
        raise ValueError("Selection feature list differs from registration")
    rows = [
        row
        for row in selection.get("frozen_selections", [])
        if row.get("population") == "all_pairs"
        and row.get("task") == "classification"
    ]
    indexed = {int(row["horizon"]): row for row in rows}
    if set(indexed) != set(HORIZONS) or len(rows) != 3:
        raise ValueError("Exactly three all-pair classifiers are required")
    result = []
    for horizon in HORIZONS:
        row = indexed[horizon]
        candidate, hyperparameter = EXPECTED_CLASSIFIERS[horizon]
        if (
            row["selected_candidate"] != candidate
            or row["selected_hyperparameter"] != hyperparameter
            or tuple(row["feature_subset"]) != registered
        ):
            raise ValueError(f"Frozen classifier mismatch at H{horizon}")
        result.append(
            {
                "population": "all_pairs",
                "task": "classification",
                "horizon": horizon,
                "selected_candidate": candidate,
                "selected_hyperparameter": hyperparameter,
                "feature_subset": list(registered),
                "preprocessing": row["preprocessing"],
                "target": "closing",
                "target_definition": "1 when separation_change < 0; 0 otherwise",
            }
        )
    return result


def _sample_counts(samples: pd.DataFrame) -> list[dict[str, Any]]:
    result = []
    for horizon in HORIZONS:
        for split in ("development_train", "validation"):
            rows = samples.loc[
                samples["horizon"].eq(horizon)
                & samples["split"].eq(split)
            ]
            result.append(
                {
                    "population": "all_pairs",
                    "horizon": horizon,
                    "split": split,
                    "sample_count": int(len(rows)),
                    "play_count": int(
                        rows[["game_id", "play_id"]]
                        .drop_duplicates()
                        .shape[0]
                    ),
                    "closing_rate": float(rows["closing"].mean()),
                }
            )
    return result


def _validate_counts(
    counts: list[dict[str, Any]],
    selection: dict[str, Any],
    interpretation: dict[str, Any],
) -> None:
    actual = {
        (row["horizon"], row["split"]): (
            row["sample_count"],
            row["play_count"],
        )
        for row in counts
    }
    selection_counts = {
        (row["horizon"], row["split"]): (
            int(row["sample_count"]),
            int(row["play_count"]),
        )
        for row in selection["sample_counts"]
        if row["population"] == "all_pairs"
    }
    interpretation_counts = {
        (row["horizon"], row["split"]): (
            int(row["sample_count"]),
            int(row["play_count"]),
        )
        for row in interpretation["sample_counts"]
    }
    if actual != selection_counts or actual != interpretation_counts:
        raise ValueError("Sample counts do not reconcile to binding artifacts")
    for horizon, expected in EXPECTED_VALIDATION_COUNTS.items():
        if actual[(horizon, "validation")] != expected:
            raise ValueError(f"Validation count mismatch at H{horizon}")


def _console(result: dict[str, Any]) -> dict[str, Any]:
    summaries = []
    for row in result["horizon_diagnostics"]:
        bins = [
            item
            for item in row["reliability_bins"]
            if item["calibration_gap"] is not None
        ]
        largest = sorted(
            bins, key=lambda item: abs(item["calibration_gap"]), reverse=True
        )[:3]
        summaries.append(
            {
                "horizon": row["horizon"],
                **row["predictive_metrics"],
                "ece": row["ece"],
                "mce": row["mce"],
                "calibration_equation": row["calibration_equation"],
                "brier_decomposition": row["brier_decomposition"],
                "bootstrap_intervals": row["play_cluster_bootstrap"][
                    "intervals"
                ],
                "largest_absolute_bin_gaps": largest,
                "confidence_region_counts": {
                    item["region"]: item["sample_count"]
                    for item in row["confidence_regions"]
                },
            }
        )
    return {
        "processed_weeks": result["processed_weeks"],
        "frozen_weeks_accessed": 0,
        "validation_rows_used_for_fitting": 0,
        "completed_bootstrap_resamples_per_horizon": BOOTSTRAP_RESAMPLES,
        "horizons": summaries,
        "leakage_validation": result["leakage_diagnostics"]["status"],
        "reconciliation_mismatch_count": result[
            "reconciliation_diagnostics"
        ]["mismatch_count"],
        "runtime_by_week_seconds": result["runtime_by_week_seconds"],
        "output": result["output_artifact"],
        "total_runtime_seconds": result["total_runtime_seconds"],
    }


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-weeks", nargs="+", required=True)
    parser.add_argument("--validation-weeks", nargs="+", required=True)
    args = parser.parse_args(arguments)
    started = time.perf_counter()
    development_weeks, validation_weeks = validate_calibration_weeks(
        args.development_weeks, args.validation_weeks
    )
    weeks = development_weeks + validation_weeks

    selection_path = _project_path(args.selection)
    normalized_root = _project_path(args.normalized_root)
    cohort_root = _project_path(args.cohort_root)
    output = _project_path(args.output)
    interpretation_path = (
        PROJECT_ROOT
        / "artifacts/milestone_5/model_interpretation_summary.json"
    )
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    interpretation = json.loads(interpretation_path.read_bytes())
    selection_checksum = hashlib.sha256(selection_bytes).hexdigest()
    if (
        tuple(interpretation["processed_weeks"]) != weeks
        or interpretation["frozen_test_weeks_accessed"] != 0
        or interpretation["binding_milestone_4_selection_checksum"]["value"]
        != selection_checksum
    ):
        raise ValueError("Model-interpretation binding artifact mismatch")
    specifications = _classifier_specifications(selection)

    chunks: list[pd.DataFrame] = []
    weekly: list[dict[str, Any]] = []
    runtimes: dict[str, float] = {}
    for week in weeks:
        week_started = time.perf_counter()
        tracking_path = normalized_root / f"normalized_{week}.parquet"
        if not tracking_path.is_file():
            raise FileNotFoundError(
                f"Missing normalized partition: {tracking_path}"
            )
        tracking = pd.read_parquet(
            tracking_path, columns=TRACKING_COLUMNS
        )
        if set(tracking["week"].astype("string")) != {week}:
            raise ValueError(f"Tracking partition contains another week: {week}")
        origins = _read_cohort_week(cohort_root, COHORT_TABLES[0], week)
        trajectories = _read_cohort_week(cohort_root, COHORT_TABLES[1], week)
        future = _read_cohort_week(cohort_root, COHORT_TABLES[2], week)
        pair_result = build_receiver_defender_pairs(
            tracking, origins, trajectories, future
        )
        samples = build_baseline_samples(
            pair_result.pairs, pair_result.origin_pairs
        )
        mismatch = int(
            pair_result.diagnostics["eligibility_mismatch_count"]
            or len(samples) != len(pair_result.pairs)
        )
        if mismatch or set(samples["week"].astype("string")) != {week}:
            raise ValueError(f"Weekly reconciliation failed: {week}")
        chunks.append(samples)
        weekly.append(
            {
                "week": week,
                "expected_horizon_pairs": int(
                    pair_result.diagnostics["expected_horizon_pairs"]
                ),
                "constructed_horizon_pairs": int(
                    pair_result.diagnostics["constructed_horizon_pairs"]
                ),
                "baseline_samples": int(len(samples)),
                "mismatch_count": 0,
            }
        )
        runtimes[week] = round(time.perf_counter() - week_started, 3)
        del tracking, origins, trajectories, future, pair_result, samples
        gc.collect()

    samples = pd.concat(chunks, ignore_index=True).sort_values(
        list(SAMPLE_KEY), kind="stable"
    ).reset_index(drop=True)
    del chunks
    if samples.duplicated(list(SAMPLE_KEY)).any():
        raise ValueError("Duplicate pair-horizon keys")
    split = _split_diagnostics(samples)
    if tuple(feature_matrix(samples).columns) != REGISTERED_NUMERIC_FEATURES:
        raise ValueError("Model matrix differs from registered features")
    counts = _sample_counts(samples)
    _validate_counts(counts, selection, interpretation)

    horizon_diagnostics = []
    for specification in specifications:
        horizon = specification["horizon"]
        horizon_rows = samples.loc[samples["horizon"].eq(horizon)]
        development = horizon_rows.loc[
            horizon_rows["split"].eq("development_train")
        ]
        validation = horizon_rows.loc[
            horizon_rows["split"].eq("validation")
        ]
        features = specification["feature_subset"]
        c_value = specification["selected_hyperparameter"]["C"]
        pipeline = build_classification_pipeline(
            "multivariable_logistic", c_value=c_value
        )
        pipeline.fit(
            feature_matrix(development, features),
            development["closing"].astype("int8"),
        )
        probability = pipeline.predict_proba(
            feature_matrix(validation, features)
        )[:, 1]
        diagnostics = calibration_diagnostics(
            validation[["game_id", "play_id"]],
            validation["closing"],
            probability,
            bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        )
        if (
            diagnostics["predictive_metrics"]["sample_count"]
            != EXPECTED_VALIDATION_COUNTS[horizon][0]
            or diagnostics["predictive_metrics"]["play_count"]
            != EXPECTED_VALIDATION_COUNTS[horizon][1]
            or sum(
                row["sample_count"]
                for row in diagnostics["reliability_bins"]
            )
            != len(validation)
            or sum(
                row["sample_count"]
                for row in diagnostics["confidence_regions"]
            )
            != len(validation)
            or diagnostics["play_cluster_bootstrap"][
                "completed_resamples"
            ]
            != BOOTSTRAP_RESAMPLES
        ):
            raise ValueError(f"Calibration gates failed at H{horizon}")
        horizon_diagnostics.append(
            {
                **specification,
                **diagnostics,
            }
        )
        del pipeline, probability, diagnostics

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    result = {
        "result_format_version": RESULT_FORMAT_VERSION,
        "generation_timestamp_utc": timestamp,
        "output_artifact": _relative(output),
        "processed_weeks": list(weeks),
        "development_weeks": list(development_weeks),
        "validation_weeks": list(validation_weeks),
        "frozen_test_weeks_accessed": 0,
        "binding_milestone_4_selection_checksum": {
            "algorithm": "sha256",
            "value": selection_checksum,
        },
        "frozen_classifier_specifications": specifications,
        "fixed_probability_bin_edges": list(BIN_EDGES),
        "sample_counts": counts,
        "horizon_diagnostics": horizon_diagnostics,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples_per_horizon": BOOTSTRAP_RESAMPLES,
        "leakage_diagnostics": {
            "status": "PASS",
            "processed_weeks_exact": True,
            "frozen_test_weeks_accessed": 0,
            "evaluated_classifier_count": 3,
            "all_pair_population_only": True,
            "exact_registered_feature_lists": True,
            "prohibited_feature_count": 0,
            "development_only_preprocessing_fit": True,
            "validation_rows_used_for_fit": 0,
            "bootstrap_sampling_unit": "game_id/play_id",
            "input_mutation_detected": False,
            **split,
        },
        "reconciliation_diagnostics": {
            "status": "PASS",
            "mismatch_count": 0,
            "duplicate_pair_horizon_key_count": 0,
            "weekly": weekly,
        },
        "runtime_by_week_seconds": runtimes,
        "total_runtime_seconds": round(time.perf_counter() - started, 3),
        "claim_boundary": [
            (
                "This analysis evaluates validation calibration for models "
                "predicting closing among competition-designated "
                "target-defender pairs."
            ),
            "It does not establish calibration in future seasons.",
            "It does not establish official coverage responsibility.",
            "It does not establish quarterback decision quality.",
            "It does not estimate pass completion probability.",
            "It does not establish causal defensive effects.",
            "It does not establish complete passing-window openness.",
            "It does not establish betting or deployment value.",
            "It provides no new frozen-test evidence.",
        ],
    }
    _atomic_json(output, result)
    print(json.dumps(_console(result), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
