"""Tests for src/lichtheim2/metrics.py — pure metric and aggregation helpers."""
from __future__ import annotations

import math

import pytest
import torch

from lichtheim2.metrics import (
    compute_repetition_metric_breakdown,
    compute_repetition_word_accuracy,
    prediction_summary,
    rolling_mean,
)


# ---------------------------------------------------------------------------
# compute_repetition_metric_breakdown
# ---------------------------------------------------------------------------


def _make_one_hot(T: int, motor_size: int, indices: list[int]) -> torch.Tensor:
    """Build (T, motor_size) one-hot tensor with active unit at indices[t] each row."""
    t = torch.zeros(T, motor_size)
    for row, idx in enumerate(indices):
        t[row, idx] = 1.0
    return t


def test_metric_breakdown_all_silence():
    """Zero motor_input → input_silence_threshold_acc == 1.0."""
    T, M = 3, 5
    motor_input = torch.zeros(T, M)
    motor_output = torch.full((T, M), 0.5)
    motor_targets_out = _make_one_hot(T, M, [0, 1, 2])
    out = compute_repetition_metric_breakdown(motor_input, motor_output, motor_targets_out)
    assert out["input_silence_threshold_acc"] == 1.0


def test_metric_breakdown_positive_unit_perfect():
    """Positive unit activation > 0.9 at every tick → output_positive_threshold_acc == 1.0."""
    T, M = 2, 4
    indices = [1, 3]
    motor_targets_out = _make_one_hot(T, M, indices)
    motor_output = torch.zeros(T, M)
    for row, idx in enumerate(indices):
        motor_output[row, idx] = 0.95
    motor_input = torch.zeros(T, M)
    out = compute_repetition_metric_breakdown(motor_input, motor_output, motor_targets_out)
    assert out["output_positive_threshold_acc"] == 1.0


def test_metric_breakdown_negative_suppressed():
    """All non-target outputs < 0.1 → output_negative_threshold_acc == 1.0."""
    T, M = 2, 5
    indices = [0, 2]
    motor_targets_out = _make_one_hot(T, M, indices)
    motor_output = torch.full((T, M), 0.05)          # all below 0.1
    for row, idx in enumerate(indices):
        motor_output[row, idx] = 0.95                 # positive unit high
    motor_input = torch.zeros(T, M)
    out = compute_repetition_metric_breakdown(motor_input, motor_output, motor_targets_out)
    assert out["output_negative_threshold_acc"] == 1.0


def test_metric_breakdown_keys():
    """Returns exactly the 6 expected keys."""
    T, M = 2, 5
    out = compute_repetition_metric_breakdown(
        torch.zeros(T, M),
        torch.zeros(T, M),
        _make_one_hot(T, M, [0, 1]),
    )
    expected = {
        "input_silence_threshold_acc",
        "output_negative_threshold_acc",
        "output_positive_threshold_acc",
        "mean_positive_output",
        "mean_negative_output",
        "mean_max_motor_output",
    }
    assert set(out.keys()) == expected


# ---------------------------------------------------------------------------
# compute_repetition_word_accuracy
# ---------------------------------------------------------------------------


def test_word_accuracy_exact_match():
    """Output exactly matching targets within radius → all three booleans True."""
    T, M = 3, 5
    indices = [0, 2, 4]
    motor_targets_out = _make_one_hot(T, M, indices)
    motor_output = motor_targets_out.clone()
    motor_input = torch.zeros(T, M)
    out = compute_repetition_word_accuracy(motor_input, motor_output, motor_targets_out, radius=0.1)
    assert out["output_all_units_within_radius"] is True
    assert out["input_all_silent_within_radius"] is True
    assert out["trial_all_supervised_units_within_radius"] is True


def test_word_accuracy_output_miss():
    """One output unit more than radius away → output_all_units_within_radius False."""
    T, M = 2, 4
    indices = [1, 3]
    motor_targets_out = _make_one_hot(T, M, indices)
    motor_output = motor_targets_out.clone()
    motor_output[0, 1] = 0.0            # positive unit off: |1 - 0| = 1.0 > 0.1
    motor_input = torch.zeros(T, M)
    out = compute_repetition_word_accuracy(motor_input, motor_output, motor_targets_out, radius=0.1)
    assert out["output_all_units_within_radius"] is False
    assert out["trial_all_supervised_units_within_radius"] is False


def test_word_accuracy_noisy_input():
    """Non-zero motor_input above radius → input_all_silent_within_radius False."""
    T, M = 2, 4
    indices = [0, 1]
    motor_targets_out = _make_one_hot(T, M, indices)
    motor_output = motor_targets_out.clone()
    motor_input = torch.zeros(T, M)
    motor_input[0, 0] = 0.5             # above radius 0.1
    out = compute_repetition_word_accuracy(motor_input, motor_output, motor_targets_out, radius=0.1)
    assert out["input_all_silent_within_radius"] is False
    assert out["trial_all_supervised_units_within_radius"] is False


# ---------------------------------------------------------------------------
# prediction_summary
# ---------------------------------------------------------------------------


def test_prediction_summary_empty():
    """Empty preds list returns NaN floats and zero counts."""
    out = prediction_summary([])
    assert out["n_exact"] == 0
    assert out["n_output_correct"] == 0
    assert math.isnan(out["avg_phoneme_accuracy"])
    assert math.isnan(out["paper_like_output_word_accuracy"])


def test_prediction_summary_non_empty():
    """Averages and counts computed correctly over two synthetic prediction dicts."""
    preds = [
        {
            "phoneme_accuracy": 1.0,
            "threshold_accuracy": 0.9,
            "exact_match": True,
            "input_silence_threshold_acc": 1.0,
            "output_negative_threshold_acc": 0.8,
            "output_positive_threshold_acc": 1.0,
            "mean_positive_output": 0.95,
            "mean_negative_output": 0.02,
            "mean_max_motor_output": 0.95,
            "output_all_units_within_radius": True,
            "input_all_silent_within_radius": True,
            "trial_all_supervised_units_within_radius": True,
        },
        {
            "phoneme_accuracy": 0.5,
            "threshold_accuracy": 0.7,
            "exact_match": False,
            "input_silence_threshold_acc": 0.9,
            "output_negative_threshold_acc": 0.6,
            "output_positive_threshold_acc": 0.5,
            "mean_positive_output": 0.6,
            "mean_negative_output": 0.05,
            "mean_max_motor_output": 0.6,
            "output_all_units_within_radius": False,
            "input_all_silent_within_radius": True,
            "trial_all_supervised_units_within_radius": False,
        },
    ]
    out = prediction_summary(preds)
    assert out["avg_phoneme_accuracy"] == pytest.approx(0.75)
    assert out["n_exact"] == 1
    assert out["n_output_correct"] == 1
    assert out["n_input_silent"] == 2
    assert out["paper_like_output_word_accuracy"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# rolling_mean
# ---------------------------------------------------------------------------


def test_rolling_mean_basic():
    """Trailing window average matches hand-computed values."""
    values = [10.0, 8.0, 6.0, 4.0, 2.0]
    result = rolling_mean(values, window=3)
    # i=0: mean([10]) = 10.0
    # i=1: mean([10,8]) = 9.0
    # i=2: mean([10,8,6]) = 8.0
    # i=3: mean([8,6,4]) = 6.0
    # i=4: mean([6,4,2]) = 4.0
    assert result == pytest.approx([10.0, 9.0, 8.0, 6.0, 4.0])


def test_rolling_mean_window_1():
    """Window=1 → identity."""
    values = [3.0, 1.0, 4.0]
    assert rolling_mean(values, window=1) == pytest.approx(values)


def test_rolling_mean_empty():
    """Empty input → empty output."""
    assert rolling_mean([], window=5) == []
