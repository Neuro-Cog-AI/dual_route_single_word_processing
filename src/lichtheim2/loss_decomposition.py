"""Diagnostic loss decomposition for repetition trials.

Separates BCE into phase and positive/negative unit components for post-epoch
analysis. These are diagnostic functions called in eval/no_grad mode — they do
NOT affect training gradients and are intentionally kept separate from losses.py
to avoid confusion with the actual training objective.

Extracted from scripts/train_repetition_only.py (Level 1B cleanup).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from lichtheim2.trainer import move_trial_to_device

if TYPE_CHECKING:
    from lichtheim2.config import ModelConfig
    from lichtheim2.model import Lichtheim2Model
    from lichtheim2.trials import SupervisedTrial


def compute_trial_loss_decomposition(
    tick_results: list,
    trial: "SupervisedTrial",
    zero_error_radius: float,
) -> dict:
    """Decompose motor BCE into phase and positive/negative unit components.

    Called in eval mode under torch.no_grad(); does NOT affect gradients.
    Applies the same dead-zone logic as losses.py (units inside the dead-zone
    are excluded from both sums and active counts).

    Phase split: T = trial.phon_tensor.shape[0].
    Input phase:  ticks 0..T-1  (motor targets are all zero).
    Output phase: ticks T..2T-1 (motor targets are the phoneme one-hot).

    Returns dict with keys (all per-trial sums, not per-unit averages):
        input_bce, output_bce           — raw sums of alive BCE by phase
        motor_bce                       — input_bce + output_bce
        output_pos_bce                  — alive BCE on the 1 positive unit per output tick
        output_neg_bce                  — alive BCE on the 38 negative units per output tick
        n_active_input                  — alive (tick, unit) count in input phase
        n_active_output_pos             — alive positive-unit count in output phase
        n_active_output_neg             — alive negative-unit count in output phase
    """
    T = trial.phon_tensor.shape[0]

    motor_outputs = torch.stack([r.state.motor for r in tick_results], dim=0)  # (2T, motor_size)
    motor_targets = trial.motor_targets   # (2T, motor_size)

    raw_bce = F.binary_cross_entropy(motor_outputs, motor_targets, reduction="none")  # (2T, motor_size)

    if zero_error_radius > 0.0:
        dead  = (motor_outputs - motor_targets).abs() < zero_error_radius
    else:
        dead  = torch.zeros_like(raw_bce, dtype=torch.bool)
    alive = ~dead  # (2T, motor_size)

    # Phase split
    raw_in,  raw_out  = raw_bce[:T],  raw_bce[T:]
    alive_in, alive_out = alive[:T], alive[T:]

    # Positive/negative split in output phase (one positive per row)
    targets_out = motor_targets[T:]   # (T, motor_size)
    pos_mask    = targets_out >= 0.5  # (T, motor_size): one True per output tick
    neg_mask    = ~pos_mask

    alive_out_pos = alive_out & pos_mask
    alive_out_neg = alive_out & neg_mask

    input_bce       = float((raw_in  * alive_in.float()).sum().item())
    output_pos_bce  = float((raw_out * alive_out_pos.float()).sum().item())
    output_neg_bce  = float((raw_out * alive_out_neg.float()).sum().item())
    output_bce      = output_pos_bce + output_neg_bce
    motor_bce       = input_bce + output_bce

    n_active_input      = int(alive_in.sum().item())
    n_active_output_pos = int(alive_out_pos.sum().item())
    n_active_output_neg = int(alive_out_neg.sum().item())

    return {
        "input_bce":           input_bce,
        "output_bce":          output_bce,
        "motor_bce":           motor_bce,
        "output_pos_bce":      output_pos_bce,
        "output_neg_bce":      output_neg_bce,
        "n_active_input":      n_active_input,
        "n_active_output_pos": n_active_output_pos,
        "n_active_output_neg": n_active_output_neg,
    }


def compute_epoch_loss_decomp(
    model: "Lichtheim2Model",
    trials: list["SupervisedTrial"],
    cfg: "ModelConfig",
    zero_error_radius: float,
    device: torch.device,
) -> dict:
    """Run an eval-mode forward pass over all trials and return mean decomposed metrics.

    This is a POST-EPOCH DIAGNOSTIC PASS — it runs after the online training
    updates for the epoch have completed. Values reflect the model at the end of
    the epoch, not during training. Because each trial is re-evaluated with the
    updated weights (rather than the weights at the time of each online update),
    these metrics are NOT numerically equal to avg_loss from run_training().
    They are prefixed 'avg_eval_' to make this distinction explicit.

    Sets model.eval() internally. Caller must restore model.train() afterward.
    """
    model.eval()

    keys = [
        "input_bce", "output_bce", "motor_bce",
        "output_pos_bce", "output_neg_bce",
        "n_active_input", "n_active_output_pos", "n_active_output_neg",
    ]
    accum = {k: 0.0 for k in keys}

    with torch.no_grad():
        for trial in trials:
            trial_dev = move_trial_to_device(trial, device)
            sem_in    = torch.zeros(cfg.vATL_size, device=device)
            tick_results = model.run_trial(
                trial_dev.task, trial_dev.phon_tensor, sem_in, cfg
            )
            d = compute_trial_loss_decomposition(tick_results, trial_dev, zero_error_radius)
            for k in keys:
                accum[k] += d[k]

    n = len(trials)

    def _safe_per_active(bce: float, count: float) -> float:
        return round(bce / count, 6) if count > 0.0 else float("nan")

    avg_input_bce       = round(accum["input_bce"]       / n, 6)
    avg_output_bce      = round(accum["output_bce"]      / n, 6)
    avg_motor_bce       = round(accum["motor_bce"]       / n, 6)
    avg_output_pos_bce  = round(accum["output_pos_bce"]  / n, 6)
    avg_output_neg_bce  = round(accum["output_neg_bce"]  / n, 6)
    avg_n_act_in        = round(accum["n_active_input"]      / n, 2)
    avg_n_act_out_pos   = round(accum["n_active_output_pos"] / n, 2)
    avg_n_act_out_neg   = round(accum["n_active_output_neg"] / n, 2)

    return {
        "avg_eval_input_bce":              avg_input_bce,
        "avg_eval_output_bce":             avg_output_bce,
        "avg_eval_motor_bce":              avg_motor_bce,
        "avg_eval_output_pos_bce":         avg_output_pos_bce,
        "avg_eval_output_neg_bce":         avg_output_neg_bce,
        "avg_eval_n_active_input":         avg_n_act_in,
        "avg_eval_n_active_output_pos":    avg_n_act_out_pos,
        "avg_eval_n_active_output_neg":    avg_n_act_out_neg,
        "avg_eval_input_bce_per_active":   _safe_per_active(accum["input_bce"],      accum["n_active_input"]),
        "avg_eval_output_pos_bce_per_active": _safe_per_active(accum["output_pos_bce"], accum["n_active_output_pos"]),
        "avg_eval_output_neg_bce_per_active": _safe_per_active(accum["output_neg_bce"], accum["n_active_output_neg"]),
    }
