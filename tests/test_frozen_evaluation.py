import copy
import json

import numpy as np
import pandas as pd
import pytest

from gridiron_spatial.baseline_features import (
    REGISTERED_NUMERIC_FEATURES,
    SAMPLE_KEY,
)
from gridiron_spatial.baseline_models import PREPROCESSING_SPECIFICATION
from gridiron_spatial.frozen_evaluation import (
    EXPECTED_SELECTIONS,
    FittedPredictor,
    atomic_write_json,
    evaluate_specification,
    fit_predictor,
    play_cluster_interval,
    validate_frozen_inventory,
    validate_sample_keys,
)


def _selection_artifact():
    frozen = []
    selections = []
    for (population, horizon, task), (
        candidate,
        hyperparameter,
    ) in EXPECTED_SELECTIONS.items():
        comparator = (
            "single_feature_linear"
            if task == "regression"
            else "single_feature_logistic"
        )
        comparator_features = ["separation_origin"]
        metric_name = (
            "validation_mae"
            if task == "regression"
            else "validation_log_loss"
        )
        frozen.append(
            {
                "population": population,
                "horizon": horizon,
                "task": task,
                "selected_candidate": candidate,
                "selected_hyperparameter": hyperparameter,
                "feature_subset": list(REGISTERED_NUMERIC_FEATURES),
                "preprocessing": PREPROCESSING_SPECIFICATION.copy(),
                "primary_metric": "mae"
                if task == "regression"
                else "log_loss",
                "selection_validation_metric": 1.0,
                "selection_timestamp_utc": "2026-01-01T00:00:00Z",
            }
        )
        selections.append(
            {
                "population": population,
                "horizon": horizon,
                "task": task,
                "selected_candidate": candidate,
                "candidates": [
                    {
                        "candidate": candidate,
                        "hyperparameter": hyperparameter,
                        "feature_subset": list(REGISTERED_NUMERIC_FEATURES),
                    },
                    {
                        "candidate": comparator,
                        "hyperparameter": None,
                        "feature_subset": comparator_features,
                    },
                ],
                "strongest_constant_or_single_feature": {
                    "candidate": comparator,
                    metric_name: 2.0,
                },
            }
        )
    return {
        "frozen_test_weeks_accessed": 0,
        "frozen_selections": frozen,
        "selections": selections,
    }


def _samples(count=30):
    index = np.arange(count)
    separation = 1.0 + index / 10
    change = np.sin(index / 2) - 0.1
    frame = pd.DataFrame(
        {
            "game_id": [f"G{value // 3}" for value in index],
            "play_id": [f"P{value // 3}" for value in index],
            "target_nfl_id": "T",
            "defender_nfl_id": [f"D{value}" for value in index],
            "origin_frame": 4,
            "horizon": 5,
            "week": "2023_w01",
            "split": "development_train",
            "separation_origin": separation,
            "dx": np.cos(index),
            "dy": np.sin(index),
            "abs_dx": np.abs(np.cos(index)),
            "abs_dy": np.abs(np.sin(index)),
            "target_x_origin": 20 + index / 10,
            "target_y_origin": 10 + index / 10,
            "defender_x_origin": 21 + index / 10,
            "defender_y_origin": 11 + index / 10,
            "valid_observed_defender_count_origin": 5,
            "defender_rank_origin": 1 + index % 5,
            "nearest_observed_defender_indicator": (index % 5 == 0).astype(int),
            "separation_change": change,
            "closing": (change < 0).astype(int),
        }
    )
    return frame


def test_exact_inventory_and_unauthorized_changes_are_rejected():
    artifact = _selection_artifact()
    specifications = validate_frozen_inventory(artifact)
    assert len(specifications) == 12
    assert {
        (row["population"], row["horizon"], row["task"])
        for row in specifications
    } == set(EXPECTED_SELECTIONS)

    unauthorized = copy.deepcopy(artifact)
    unauthorized["frozen_selections"][0]["selected_candidate"] = "tree"
    with pytest.raises(ValueError, match="Unauthorized candidate"):
        validate_frozen_inventory(unauthorized)

    unauthorized = copy.deepcopy(artifact)
    unauthorized["frozen_selections"][0]["selected_hyperparameter"] = {
        "alpha": 999.0
    }
    with pytest.raises(ValueError, match="hyperparameter"):
        validate_frozen_inventory(unauthorized)

    unauthorized = copy.deepcopy(artifact)
    unauthorized["frozen_selections"][0]["feature_subset"] = [
        "separation_origin"
    ]
    with pytest.raises(ValueError, match="Feature-list"):
        validate_frozen_inventory(unauthorized)


def test_frozen_comparator_is_preserved_without_test_reselection():
    artifact = _selection_artifact()
    artifact["selections"][0]["candidates"][1]["test_metric"] = 999.0
    specification = next(
        row
        for row in validate_frozen_inventory(artifact)
        if row["population"] == "all_pairs"
        and row["horizon"] == 5
        and row["task"] == "regression"
    )
    assert specification["comparator_candidate"] == "single_feature_linear"
    assert specification["comparator_validation_metric"] == 2.0


def test_development_only_fit_metrics_sign_and_no_input_mutation():
    artifact = _selection_artifact()
    specification = next(
        row
        for row in validate_frozen_inventory(artifact)
        if row["population"] == "all_pairs"
        and row["horizon"] == 5
        and row["task"] == "regression"
    )
    development = _samples()
    frozen = _samples(15).assign(
        week="2023_w16", split="frozen_test"
    )
    dev_before = development.copy(deep=True)
    frozen_before = frozen.copy(deep=True)
    selected = fit_predictor(specification, development)
    comparator = fit_predictor(
        specification, development, comparator=True
    )
    result = evaluate_specification(
        specification, selected, comparator, frozen
    )
    assert set(result["selected_metrics"]) == {
        "mae",
        "rmse",
        "median_absolute_error",
        "r2",
    }
    assert result["metric_differences"]["mae"] == pytest.approx(
        result["selected_metrics"]["mae"]
        - result["comparator_metrics"]["mae"]
    )
    pd.testing.assert_frame_equal(development, dev_before)
    pd.testing.assert_frame_equal(frozen, frozen_before)


def test_classification_metrics_and_difference_sign():
    frozen = _samples(20)
    specification = {"task": "classification"}
    selected = FittedPredictor(
        "classification", "selected", (), constant=0.55
    )
    comparator = FittedPredictor(
        "classification", "comparator", (), constant=0.50
    )
    result = evaluate_specification(
        specification, selected, comparator, frozen
    )
    assert set(result["selected_metrics"]) == {
        "log_loss",
        "brier_score",
        "roc_auc",
        "accuracy_at_0_5",
    }
    assert result["metric_differences"]["log_loss"] == pytest.approx(
        result["selected_metrics"]["log_loss"]
        - result["comparator_metrics"]["log_loss"]
    )
    assert result["metric_differences"]["brier_score"] == pytest.approx(
        result["selected_metrics"]["brier_score"]
        - result["comparator_metrics"]["brier_score"]
    )


def test_play_cluster_bootstrap_is_deterministic_and_samples_plays():
    differences = np.r_[np.repeat(-1.0, 100), 10.0]
    games = np.concatenate([np.repeat("G1", 100), ["G2"]])
    plays = np.concatenate([np.repeat("P1", 100), ["P2"]])
    first = play_cluster_interval(differences, games, plays)
    second = play_cluster_interval(differences, games, plays)
    assert first == second
    assert first[0] == pytest.approx(-1.0)
    assert first[1] == pytest.approx(10.0)


def test_nonfinite_duplicate_and_atomic_json_guards(tmp_path):
    samples = _samples()
    duplicate = pd.concat([samples, samples.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_sample_keys(duplicate)
    invalid = FittedPredictor(
        "regression", "invalid", (), constant=float("inf")
    )
    with pytest.raises(ValueError, match="non-finite"):
        invalid.predict(samples)

    output = tmp_path / "result.json"
    atomic_write_json(output, {"status": "PASS", "value": 1})
    assert json.loads(output.read_text()) == {"status": "PASS", "value": 1}
    assert not list(tmp_path.glob("*.tmp"))
