"""Pure metric and aggregation helpers for repetition evaluation.

All functions are stateless and side-effect-free — no model calls, no file I/O.
Extracted from scripts/train_repetition_only.py (Level 1B cleanup).
"""
from __future__ import annotations

import torch


def compute_repetition_metric_breakdown(
    motor_input: torch.Tensor,
    motor_output: torch.Tensor,
    motor_targets_out: torch.Tensor,
) -> dict:
    """Compute split threshold and activation metrics for one repetition trial.

    Separates input-phase silence from output-phase positive/negative unit
    behavior, allowing diagnosis beyond the combined threshold_accuracy metric.

    Interpretation:
      If output_negative_threshold_acc rises quickly but
      output_positive_threshold_acc stays near 0, the model is learning
      zero-target suppression, not phoneme production.
      mean_positive_output rising above 0.5 is typically the earliest signal
      of phoneme production learning, before output_positive_threshold_acc or
      argmax accuracy improve.

    Args:
        motor_input:       (T, motor_size) — input-phase motor outputs (ticks 0..T-1).
                           All motor targets are zero during the input phase.
        motor_output:      (T, motor_size) — output-phase motor outputs (ticks T..2T-1).
        motor_targets_out: (T, motor_size) — output-phase one-hot targets.

    Returns dict with keys:
        input_silence_threshold_acc:   fraction of (input-phase, unit) pairs with output < 0.1.
        output_negative_threshold_acc: fraction of output-phase negative-target (tick, unit)
                                       pairs with output < 0.1.
        output_positive_threshold_acc: fraction of output-phase ticks where the positive unit
                                       (argmax of target) has output > 0.9.  [strict threshold]
        mean_positive_output:          mean activation of the positive unit across output ticks.
        mean_negative_output:          mean activation of negative units across output ticks.
        mean_max_motor_output:         mean of per-tick max(motor_output) across output ticks.
    """
    T = motor_output.shape[0]

    # Input phase: all targets = 0; report fraction suppressed below 0.1.
    input_silence_threshold_acc = (motor_input < 0.1).float().mean().item()

    # Output phase: split by positive (target=1) vs negative (target=0) units.
    pos_indices = motor_targets_out.argmax(dim=1)         # (T,) — index of the 1 per tick
    neg_mask    = motor_targets_out < 0.5                 # (T, motor_size) bool mask

    # Device-safe indexing: arange on the same device as motor_output.
    idx         = torch.arange(T, device=motor_output.device)
    pos_outputs = motor_output[idx, pos_indices]          # (T,)
    neg_outputs = motor_output[neg_mask]                  # (T × (motor_size − 1),) flattened

    output_positive_threshold_acc = (pos_outputs > 0.9).float().mean().item()
    output_negative_threshold_acc = (neg_outputs < 0.1).float().mean().item()
    mean_positive_output          = pos_outputs.mean().item()
    mean_negative_output          = neg_outputs.mean().item()
    mean_max_motor_output         = motor_output.max(dim=1).values.mean().item()

    return {
        "input_silence_threshold_acc":    round(input_silence_threshold_acc, 4),
        "output_negative_threshold_acc":  round(output_negative_threshold_acc, 4),
        "output_positive_threshold_acc":  round(output_positive_threshold_acc, 4),
        "mean_positive_output":           round(mean_positive_output, 4),
        "mean_negative_output":           round(mean_negative_output, 4),
        "mean_max_motor_output":          round(mean_max_motor_output, 4),
    }


def compute_repetition_word_accuracy(
    motor_input: torch.Tensor,
    motor_output: torch.Tensor,
    motor_targets_out: torch.Tensor,
    radius: float = 0.1,
) -> dict:
    """Compute candidate paper-like word-level accuracy booleans for one trial.

    Returns three per-trial boolean metrics evaluated using error radius `radius`.

    output_all_units_within_radius:
        True iff every (tick, unit) pair in the output phase satisfies
        |motor_output - target| < radius. This is the main candidate paper-like
        repetition word accuracy. [Open #9]

    input_all_silent_within_radius:
        True iff every input-phase motor unit is below `radius` (all input-phase
        motor targets are zero, so this measures whether the model suppresses
        motor activation during the input phase).

    trial_all_supervised_units_within_radius:
        True iff both output_all_units_within_radius AND
        input_all_silent_within_radius are True. Stricter criterion whose
        relevance depends on whether input-phase silence should be scored. [Open #9]

    Note: eval_radius can differ from zero_error_radius used during training.
    For example, train with zero_error_radius=0.0 (full gradient) and evaluate
    with eval_radius=0.1 (paper-like criterion).

    Args:
        motor_input:       (T, motor_size) — input-phase motor outputs.
        motor_output:      (T, motor_size) — output-phase motor outputs.
        motor_targets_out: (T, motor_size) — output-phase targets (one-hot).
        radius:            error radius for word-level evaluation (default 0.1).
    """
    output_within = (motor_output - motor_targets_out).abs() < radius
    output_all_units_within_radius = bool(output_within.all().item())

    input_within = motor_input < radius          # input-phase targets are all zero
    input_all_silent_within_radius = bool(input_within.all().item())

    trial_all_supervised_units_within_radius = (
        output_all_units_within_radius and input_all_silent_within_radius
    )

    return {
        "output_all_units_within_radius":           output_all_units_within_radius,
        "input_all_silent_within_radius":            input_all_silent_within_radius,
        "trial_all_supervised_units_within_radius":  trial_all_supervised_units_within_radius,
    }


def prediction_summary(preds: list[dict]) -> dict:
    """Aggregate per-trial prediction dicts into a named summary dict.

    All float values are float('nan') when preds is empty.
    """
    nan = float("nan")
    if not preds:
        return {
            "avg_phoneme_accuracy":              nan,
            "avg_threshold_accuracy":            nan,
            "n_exact":                           0,
            "avg_input_silence_threshold_acc":   nan,
            "avg_output_negative_threshold_acc": nan,
            "avg_output_positive_threshold_acc": nan,
            "avg_mean_positive_output":          nan,
            "avg_mean_negative_output":          nan,
            "avg_mean_max_motor_output":         nan,
            "n_output_correct":                  0,
            "n_input_silent":                    0,
            "n_strict_correct":                  0,
            "paper_like_output_word_accuracy":   nan,
            "input_silence_word_accuracy":       nan,
            "candidate_strict_trial_accuracy":   nan,
        }
    n = len(preds)
    n_output_correct = sum(1 for p in preds if p["output_all_units_within_radius"])
    n_input_silent   = sum(1 for p in preds if p["input_all_silent_within_radius"])
    n_strict_correct = sum(1 for p in preds if p["trial_all_supervised_units_within_radius"])
    return {
        "avg_phoneme_accuracy":              sum(p["phoneme_accuracy"]              for p in preds) / n,
        "avg_threshold_accuracy":            sum(p["threshold_accuracy"]            for p in preds) / n,
        "n_exact":                           sum(1 for p in preds if p["exact_match"]),
        "avg_input_silence_threshold_acc":   sum(p["input_silence_threshold_acc"]   for p in preds) / n,
        "avg_output_negative_threshold_acc": sum(p["output_negative_threshold_acc"] for p in preds) / n,
        "avg_output_positive_threshold_acc": sum(p["output_positive_threshold_acc"] for p in preds) / n,
        "avg_mean_positive_output":          sum(p["mean_positive_output"]          for p in preds) / n,
        "avg_mean_negative_output":          sum(p["mean_negative_output"]          for p in preds) / n,
        "avg_mean_max_motor_output":         sum(p["mean_max_motor_output"]         for p in preds) / n,
        "n_output_correct":                  n_output_correct,
        "n_input_silent":                    n_input_silent,
        "n_strict_correct":                  n_strict_correct,
        "paper_like_output_word_accuracy":   n_output_correct / n,
        "input_silence_word_accuracy":       n_input_silent   / n,
        "candidate_strict_trial_accuracy":   n_strict_correct / n,
    }


def rolling_mean(values: list[float], window: int) -> list[float]:
    """Left-aligned trailing rolling mean; no external deps."""
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start : i + 1]) / (i - start + 1))
    return out
