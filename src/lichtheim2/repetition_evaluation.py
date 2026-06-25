"""Evaluation functions for repetition trials.

Runs the model in eval/no_grad mode and computes per-trial prediction metrics.
Extracted from scripts/train_repetition_only.py (Level 1B cleanup).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from lichtheim2.metrics import (
    compute_repetition_metric_breakdown,
    compute_repetition_word_accuracy,
)
from lichtheim2.trainer import move_trial_to_device

if TYPE_CHECKING:
    from lichtheim2.config import ModelConfig
    from lichtheim2.model import Lichtheim2Model
    from lichtheim2.trials import SupervisedTrial


def evaluate_one_trial(
    model: "Lichtheim2Model",
    trial: "SupervisedTrial",
    cfg: "ModelConfig",
    inventory_symbols: list[str],
    device: torch.device,
    eval_radius: float = 0.1,
) -> dict:
    """Run one trial in eval mode (no_grad) and compute prediction metrics.

    Argmax accuracy, mixed threshold accuracy, and exact match are computed on
    the output phase (ticks T..2T-1) only. Split metrics from
    compute_repetition_metric_breakdown() cover both phases separately.

    Note: threshold_accuracy mixes positive and negative target units. Use
    output_positive_threshold_acc and output_negative_threshold_acc for
    diagnostic purposes.
    [Note #4] Sound input is the raw phoneme tensor (clamped), not sigmoided.
    [Open #3] The ventral pathway is computed at every tick.
    """
    T = trial.phon_tensor.shape[0]
    input_tick_indices  = list(range(T))           # ticks 0..T-1
    output_tick_indices = list(range(T, 2 * T))    # ticks T..2T-1

    trial_dev = move_trial_to_device(trial, device)
    sem_zeros = torch.zeros(cfg.vATL_size, device=device)

    model.eval()
    with torch.no_grad():
        results = model.run_trial(trial_dev.task, trial_dev.phon_tensor, sem_zeros, cfg)

    # Stack input-phase motor activations: shape (T, motor_size)
    motor_outputs_input = torch.stack(
        [results[t].state.motor for t in input_tick_indices], dim=0
    )
    # Stack output-phase motor activations: shape (T, motor_size)
    motor_outputs = torch.stack(
        [results[t].state.motor for t in output_tick_indices], dim=0
    )
    # Output-phase targets from the trial: shape (T, motor_size)
    motor_targets = trial_dev.motor_targets[T:]

    # Argmax accuracy (one-hot encoding → argmax gives phoneme index)
    target_indices = motor_targets.argmax(dim=1).tolist()
    output_indices = motor_outputs.argmax(dim=1).tolist()
    target_symbols = [inventory_symbols[i] for i in target_indices]
    output_symbols = [inventory_symbols[i] for i in output_indices]
    phoneme_correct = [t == p for t, p in zip(target_indices, output_indices)]
    phoneme_acc = sum(phoneme_correct) / T if T > 0 else float("nan")

    # Mixed threshold accuracy (output phase, positive + negative units combined).
    # Note: dominated by negative units (38/39); use split metrics for diagnosis.
    within_threshold = (motor_outputs - motor_targets).abs() < 0.1
    threshold_acc = within_threshold.float().mean().item()

    # Split metric breakdown: separates phases and unit polarities.
    breakdown = compute_repetition_metric_breakdown(
        motor_outputs_input, motor_outputs, motor_targets
    )

    # Candidate paper-like word-level accuracy booleans.
    word_acc = compute_repetition_word_accuracy(
        motor_outputs_input, motor_outputs, motor_targets, radius=eval_radius
    )

    return {
        "label":              trial.label,
        "item_id":            trial.item_id,
        "T":                  T,
        "n_ticks":            trial.n_ticks,
        "target_phonemes":    target_symbols,
        "predicted_phonemes": output_symbols,
        "phoneme_accuracy":   round(phoneme_acc, 4),
        "threshold_accuracy": round(threshold_acc, 4),
        "exact_match":        all(phoneme_correct),
        **breakdown,
        **word_acc,
    }


def evaluate_predictions(
    model: "Lichtheim2Model",
    trials: list["SupervisedTrial"],
    cfg: "ModelConfig",
    inventory_symbols: list[str],
    device: torch.device,
    eval_radius: float = 0.1,
) -> list[dict]:
    """Evaluate predictions for all trials. Returns one record per trial."""
    return [
        evaluate_one_trial(model, trial, cfg, inventory_symbols, device, eval_radius)
        for trial in trials
    ]
