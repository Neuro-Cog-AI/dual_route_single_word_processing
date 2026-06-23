"""Lightweight tests for analyze_repetition_predictions.py.

Uses synthetic predictions JSON only — no real CSV data required.
No model imports; no lichtheim2 package needed.
"""
from __future__ import annotations

import csv as csv_module
import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Ensure src is on path (needed by conftest.py of the broader test suite)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Script loader
# ---------------------------------------------------------------------------


def _load_script():
    script_path = (
        Path(__file__).parent.parent / "scripts" / "analyze_repetition_predictions.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_repetition_predictions", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Synthetic predictions covering all five failure types
# ---------------------------------------------------------------------------

SYNTH_PREDS = [
    # 0: paper_like_success
    {
        "label": "word:a",
        "item_id": 0,
        "T": 4,
        "n_ticks": 8,
        "target_phonemes": ["A", "B", "C", "D"],
        "predicted_phonemes": ["A", "B", "C", "D"],
        "phoneme_accuracy": 1.0,
        "threshold_accuracy": 1.0,
        "exact_match": True,
        "input_silence_threshold_acc": 1.0,
        "output_negative_threshold_acc": 1.0,
        "output_positive_threshold_acc": 1.0,
        "mean_positive_output": 0.95,
        "mean_negative_output": 0.005,
        "mean_max_motor_output": 0.95,
        "output_all_units_within_radius": True,
        "input_all_silent_within_radius": True,
        "trial_all_supervised_units_within_radius": True,
    },
    # 1: positive_under_threshold_only  (pos fails, neg perfect, input silent)
    {
        "label": "word:b",
        "item_id": 1,
        "T": 4,
        "n_ticks": 8,
        "target_phonemes": ["A", "B", "C", "D"],
        "predicted_phonemes": ["A", "B", "C", "D"],
        "phoneme_accuracy": 1.0,
        "threshold_accuracy": 0.9,
        "exact_match": True,
        "input_silence_threshold_acc": 1.0,
        "output_negative_threshold_acc": 1.0,
        "output_positive_threshold_acc": 0.5,   # 2 of 4 positive ticks fail
        "mean_positive_output": 0.85,
        "mean_negative_output": 0.005,
        "mean_max_motor_output": 0.85,
        "output_all_units_within_radius": False,
        "input_all_silent_within_radius": True,
        "trial_all_supervised_units_within_radius": False,
    },
    # 2: negative_over_threshold_only  (neg fails, pos perfect, input silent)
    {
        "label": "word:c",
        "item_id": 2,
        "T": 4,
        "n_ticks": 8,
        "target_phonemes": ["A", "B", "C", "D"],
        "predicted_phonemes": ["A", "B", "C", "D"],
        "phoneme_accuracy": 1.0,
        "threshold_accuracy": 0.97,
        "exact_match": True,
        "input_silence_threshold_acc": 1.0,
        "output_negative_threshold_acc": 0.9,   # some negative units over threshold
        "output_positive_threshold_acc": 1.0,
        "mean_positive_output": 0.95,
        "mean_negative_output": 0.015,
        "mean_max_motor_output": 0.95,
        "output_all_units_within_radius": False,
        "input_all_silent_within_radius": True,
        "trial_all_supervised_units_within_radius": False,
    },
    # 3: positive_and_negative_failures  (both fail, input silent)
    {
        "label": "word:d",
        "item_id": 3,
        "T": 4,
        "n_ticks": 8,
        "target_phonemes": ["A", "B", "C", "D"],
        "predicted_phonemes": ["A", "B", "C", "D"],
        "phoneme_accuracy": 1.0,
        "threshold_accuracy": 0.88,
        "exact_match": True,
        "input_silence_threshold_acc": 1.0,
        "output_negative_threshold_acc": 0.95,
        "output_positive_threshold_acc": 0.75,  # 1 of 4 positive ticks fails
        "mean_positive_output": 0.87,
        "mean_negative_output": 0.012,
        "mean_max_motor_output": 0.87,
        "output_all_units_within_radius": False,
        "input_all_silent_within_radius": True,
        "trial_all_supervised_units_within_radius": False,
    },
    # 4: argmax_error  (wrong argmax)
    {
        "label": "word:e",
        "item_id": 4,
        "T": 4,
        "n_ticks": 8,
        "target_phonemes": ["A", "B", "C", "D"],
        "predicted_phonemes": ["A", "B", "X", "D"],
        "phoneme_accuracy": 0.75,
        "threshold_accuracy": 0.85,
        "exact_match": False,
        "input_silence_threshold_acc": 1.0,
        "output_negative_threshold_acc": 0.9,
        "output_positive_threshold_acc": 0.8,
        "mean_positive_output": 0.85,
        "mean_negative_output": 0.015,
        "mean_max_motor_output": 0.85,
        "output_all_units_within_radius": False,
        "input_all_silent_within_radius": True,
        "trial_all_supervised_units_within_radius": False,
    },
]

EXPECTED_FAILURE_TYPES = [
    "paper_like_success",
    "positive_under_threshold_only",
    "negative_over_threshold_only",
    "positive_and_negative_failures",
    "argmax_error",
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_classify_failure_type_all_cases():
    mod = _load_script()
    for pred, expected in zip(SYNTH_PREDS, EXPECTED_FAILURE_TYPES):
        result = mod.classify_failure_type(pred)
        assert result == expected, (
            f"label={pred['label']}: got {result!r}, expected {expected!r}"
        )


def test_analyze_one_computes_n_positive_failures():
    """T=4, pos_acc=0.5 → n_positive_failures = round(0.5 * 4) = 2."""
    mod = _load_script()
    row = mod.analyze_one(SYNTH_PREDS[1], 1, motor_size=39)
    assert row["n_positive_failures"] == 2, f"Expected 2, got {row['n_positive_failures']}"


def test_analyze_one_computes_n_negative_failures():
    """T=4, neg_acc=0.9 → n_negative_failures = round(0.1 * 4 * 38) = 15."""
    mod = _load_script()
    row = mod.analyze_one(SYNTH_PREDS[2], 2, motor_size=39)
    assert row["n_negative_failures"] == 15, f"Expected 15, got {row['n_negative_failures']}"


def test_script_writes_csv_and_md(tmp_path):
    """End-to-end: synthetic run dir → CSV with all columns + Markdown with all sections."""
    mod = _load_script()

    (tmp_path / "predictions_after.json").write_text(json.dumps(SYNTH_PREDS))
    (tmp_path / "run_config.json").write_text(json.dumps({
        "epochs": 10,
        "lr": 0.01,
        "motor_output_size": 39,
        "raw_sound_input_size": 39,
        "eval_radius": 0.1,
    }))

    csv_path = tmp_path / "out" / "analysis.csv"
    md_path = tmp_path / "out" / "analysis.md"

    mod.main([
        "--run-dir", str(tmp_path),
        "--eval-radius", "0.1",
        "--output-csv", str(csv_path),
        "--output-md", str(md_path),
    ])

    # CSV checks
    assert csv_path.exists()
    with open(csv_path) as f:
        csv_rows = list(csv_module.DictReader(f))
    assert len(csv_rows) == 5

    required_columns = {
        "strict_trial_match", "paper_like_match", "failure_type",
        "n_positive_failures", "n_negative_failures",
        "min_positive_activation", "mean_positive_activation",
    }
    for col in required_columns:
        assert col in csv_rows[0], f"Missing CSV column: {col}"

    assert csv_rows[0]["failure_type"] == "paper_like_success"
    assert csv_rows[0]["strict_trial_match"] == "True"
    assert csv_rows[1]["strict_trial_match"] == "False"
    assert csv_rows[0]["min_positive_activation"] == "N/A"

    # Markdown checks
    assert md_path.exists()
    md_content = md_path.read_text()
    assert len(md_content) > 200

    for section in ["§1", "§2", "§3", "§4", "§5", "§6", "§7", "§8", "§9"]:
        assert section in md_content, f"Missing section {section}"

    # Adjustment #1: both metric names documented
    assert "paper_like_match" in md_content
    assert "strict_trial_match" in md_content
    assert "output_all_units_within_radius" in md_content
    assert "trial_all_supervised_units_within_radius" in md_content

    # Adjustment #2: rounding caveat present
    assert "floating-point rounding" in md_content

    # Adjustment #3: per-tick vs per-word distinction
    assert "per-tick rate" in md_content or "per-word" in md_content
