"""Phase 3e: Repetition-only first training run.

Goal
----
Establish that the full Lichtheim 2 architecture can learn to repeat English
words when trained on repetition trials only. This is the simplest meaningful
training run: real phoneme data, one task, one item at a time.

Architecture note
-----------------
The FULL Lichtheim2Model is used unchanged (Option A). This is NOT a
dorsal-only model. The ventral pathway (mSTG→aSTG→vATL→triangularis) is
computed at every tick, and triangularis_to_motor contributes to motor output
even during repetition. This is intentional and consistent with Paper Fig 1.
[Open #1] Isolating the dorsal pathway (Option B) is deferred.

This script is intentionally pedagogical: every step is explicit and named,
and each CLI argument maps to a documented design decision.

Open audit issues
-----------------
See docs/repetition_only_training_note.md for full discussion. Short list:
  [Open #1] Full architecture (Option A) vs dorsal-only sanity (Option B).
  [Open #2] motor_loss_mask is True for ALL 2T ticks: silence supervised too.
  [Open #3] motor→iSMG copy-back is state-copy + learned projection (not pure copy).
  [Note  #4] AUD/sound input is clamped raw input, NOT passed through sigmoid.
  [Open #5] zero_error_radius: 0.0 (debug default here) vs 0.1 (paper value).
  [Open #6] Phoneme encoding: English one-hot N-bit vs Japanese 21-bit mora.
  [Open #7] BPTT: currently full BPTT through all 2T ticks within the trial.

Usage (from dual_route_single_word_processing/)
-----------------------------------------------
    PYTHONPATH=src python scripts/train_repetition_only.py --help

Smoke run (5 items, 2 epochs):
    PYTHONPATH=src python scripts/train_repetition_only.py \\
      --data-dir data/raw/nwr_swp \\
      --config configs/english_nwr.yaml \\
      --source words \\
      --max-items 5 \\
      --epochs 2 \\
      --lr 0.01 \\
      --device cpu \\
      --seed 0 \\
      --zero-error-radius 0.1 \\
      --loss-reduction sum
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lichtheim2.config import ModelConfig, load_config
from lichtheim2.data import PseudowordItem, WordItem, load_pseudoword_items, load_word_items
from lichtheim2.encoding import load_phoneme_inventory
from lichtheim2.model import Lichtheim2Model
from lichtheim2.trainer import move_trial_to_device, train_step
from lichtheim2.trials import SupervisedTrial, make_repetition_trial


# ---------------------------------------------------------------------------
# Item helpers
# ---------------------------------------------------------------------------


def _item_label(item: WordItem | PseudowordItem) -> str:
    """Return a collision-safe label string (source-prefixed)."""
    if isinstance(item, WordItem):
        return f"word:{item.word}"
    return f"pseudo:{item.row_index}"


# ---------------------------------------------------------------------------
# Split metric diagnostics
# ---------------------------------------------------------------------------


def _compute_repetition_metric_breakdown(
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


def _compute_repetition_word_accuracy(
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


# ---------------------------------------------------------------------------
# Trial construction
# ---------------------------------------------------------------------------


def build_repetition_trials(
    items: list[WordItem | PseudowordItem],
    motor_size: int,
) -> list[SupervisedTrial]:
    """Build one repetition SupervisedTrial per item.

    Each trial has 2T ticks where T = number of phonemes:
      Ticks 0..T-1  (input phase):  sound = phoneme vector; motor target = 0
      Ticks T..2T-1 (output phase): sound = 0; motor target = phoneme vector

    motor_loss_mask is True for ALL 2T ticks. [Open #2]

    Args:
        items:      list of WordItem or PseudowordItem (phon_tensor already encoded)
        motor_size: motor output dimension from ModelConfig
    """
    trials: list[SupervisedTrial] = []
    for item in items:
        trial = make_repetition_trial(
            phon_tensor=item.phon_tensor,
            motor_size=motor_size,
            item_id=item.row_index,
            label=_item_label(item),
        )
        trials.append(trial)
    return trials


# ---------------------------------------------------------------------------
# Per-trial prediction evaluation
# ---------------------------------------------------------------------------


def _evaluate_one_trial(
    model: Lichtheim2Model,
    trial: SupervisedTrial,
    cfg: ModelConfig,
    inventory_symbols: list[str],
    device: torch.device,
    eval_radius: float = 0.1,
) -> dict:
    """Run one trial in eval mode (no_grad) and compute prediction metrics.

    Argmax accuracy, mixed threshold accuracy, and exact match are computed on
    the output phase (ticks T..2T-1) only. Split metrics from
    _compute_repetition_metric_breakdown() cover both phases separately.

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
    breakdown = _compute_repetition_metric_breakdown(
        motor_outputs_input, motor_outputs, motor_targets
    )

    # Candidate paper-like word-level accuracy booleans.
    word_acc = _compute_repetition_word_accuracy(
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
    model: Lichtheim2Model,
    trials: list[SupervisedTrial],
    cfg: ModelConfig,
    inventory_symbols: list[str],
    device: torch.device,
    eval_radius: float = 0.1,
) -> list[dict]:
    """Evaluate predictions for all trials. Returns one record per trial."""
    return [
        _evaluate_one_trial(model, trial, cfg, inventory_symbols, device, eval_radius)
        for trial in trials
    ]


def _prediction_summary(preds: list[dict]) -> dict:
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


# ---------------------------------------------------------------------------
# Post-epoch eval loss decomposition (Phase 3i diagnostic)
# ---------------------------------------------------------------------------


def _compute_trial_loss_decomposition(
    tick_results: list,
    trial: SupervisedTrial,
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


def _compute_epoch_loss_decomp(
    model: Lichtheim2Model,
    trials: list[SupervisedTrial],
    cfg: ModelConfig,
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
            d = _compute_trial_loss_decomposition(tick_results, trial_dev, zero_error_radius)
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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def run_training(
    model: Lichtheim2Model,
    trials: list[SupervisedTrial],
    optimizer: torch.optim.Optimizer,
    cfg: ModelConfig,
    epochs: int,
    zero_error_radius: float,
    loss_reduction: str,
    device: torch.device,
    rng: random.Random,
    log_loss_decomp: bool = True,
) -> list[dict]:
    """Online item-by-item training loop.

    At each epoch: shuffle trials, call train_step() per item.
    Per-epoch metrics: avg_loss, min_loss, max_loss, n_trials, all_finite.
    When log_loss_decomp=True, an additional eval-mode pass adds avg_eval_*
    decomposition keys to each epoch dict (see _compute_epoch_loss_decomp).

    [Open #7] Full BPTT through all 2T ticks is used (no truncation).
              loss.backward() is called once per trial after summing tick losses.

    Raises:
        RuntimeError: if any per-item loss is non-finite (nan or inf),
                      with epoch number and trial label for debugging.
    """
    epoch_metrics: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = list(trials)
        rng.shuffle(order)
        epoch_losses: list[float] = []

        for trial in order:
            loss_val = train_step(
                model, trial, optimizer, cfg,
                zero_error_radius=zero_error_radius,
                device=device,
                loss_reduction=loss_reduction,
            )
            if not math.isfinite(loss_val):
                raise RuntimeError(
                    f"Non-finite loss ({loss_val}) at epoch {epoch}, "
                    f"label={trial.label!r}"
                )
            epoch_losses.append(loss_val)

        avg      = sum(epoch_losses) / len(epoch_losses)
        min_loss = min(epoch_losses)
        max_loss = max(epoch_losses)

        metrics = {
            "epoch":      epoch,
            "avg_loss":   avg,
            "min_loss":   min_loss,
            "max_loss":   max_loss,
            "n_trials":   len(epoch_losses),
            "all_finite": True,
        }

        if log_loss_decomp:
            decomp = _compute_epoch_loss_decomp(model, trials, cfg, zero_error_radius, device)
            model.train()   # restore — eval was set inside _compute_epoch_loss_decomp
            metrics.update(decomp)

        epoch_metrics.append(metrics)

        print(
            f"  Epoch {epoch:3d}/{epochs}: "
            f"avg={avg:.6f}  min={min_loss:.6f}  max={max_loss:.6f}"
        )

    return epoch_metrics


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------


def save_run_config(
    run_dir: Path,
    args: argparse.Namespace,
    cfg: ModelConfig,
) -> None:
    """Save CLI args and model config fields to run_config.json."""
    payload = {
        "script":             "scripts/train_repetition_only.py",
        "task":               "repetition-only",
        "architecture":       "full Lichtheim2Model (both dorsal and ventral pathways)",
        "data_dir":           str(args.data_dir),
        "config":             str(args.config),
        "source":             args.source,
        "max_items":          args.max_items,
        "epochs":             args.epochs,
        "lr":                 args.lr,
        "device":             args.device,
        "seed":               args.seed,
        "zero_error_radius":  args.zero_error_radius,
        "eval_radius":        args.eval_radius,
        "loss_reduction":     args.loss_reduction,
        "log_loss_decomp":    args.log_loss_decomp,
        "dorsal_motor_only":  cfg.dorsal_motor_only,
        "motor_readout_mode": "dorsal_only" if cfg.dorsal_motor_only else "full",
        "output_dir":         str(run_dir),
        # Sound input projection fields [Phase 3h]
        "use_sound_projection":      cfg.sound_proj_size is not None,
        "sound_proj_size":           cfg.sound_proj_size,
        "raw_sound_input_size":      cfg.sound_input_size,
        "effective_sound_input_size": cfg.sound_proj_size if cfg.sound_proj_size is not None else cfg.sound_input_size,
        "sound_projection_bias":     False if cfg.sound_proj_size is not None else None,
        "motor_output_size":         cfg.motor_output_size,
        "model_config": {
            "sound_input_size":         cfg.sound_input_size,
            "motor_output_size":        cfg.motor_output_size,
            "iSMG_hidden_size":         cfg.iSMG_hidden_size,
            "mSTG_hidden_size":         cfg.mSTG_hidden_size,
            "aSTG_hidden_size":         cfg.aSTG_hidden_size,
            "triangularis_hidden_size": cfg.triangularis_hidden_size,
            "vATL_size":                cfg.vATL_size,
            "sound_proj_size":          cfg.sound_proj_size,
        },
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(payload, f, indent=2)


def save_metrics_csv(run_dir: Path, epoch_metrics: list[dict]) -> None:
    """Save per-epoch training metrics to metrics.csv."""
    if not epoch_metrics:
        return
    fieldnames = list(epoch_metrics[0].keys())
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(epoch_metrics)


def save_predictions(run_dir: Path, predictions: list[dict], filename: str) -> None:
    """Save list of prediction dicts to a JSON file."""
    with open(run_dir / filename, "w") as f:
        json.dump(predictions, f, indent=2)


def save_loss_curve(run_dir: Path, epoch_metrics: list[dict]) -> None:
    """Save a loss-curve PNG if matplotlib is available (silent skip otherwise)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping loss_curve.png)")
        return

    epochs    = [m["epoch"]    for m in epoch_metrics]
    avg_losses = [m["avg_loss"] for m in epoch_metrics]
    min_losses = [m["min_loss"] for m in epoch_metrics]
    max_losses = [m["max_loss"] for m in epoch_metrics]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(epochs, min_losses, max_losses, alpha=0.2, color="steelblue", label="min/max range")
    ax.plot(epochs, avg_losses, color="steelblue", linewidth=2, label="avg loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("Repetition-only training — per-epoch loss")
    ax.legend()
    fig.tight_layout()
    path = run_dir / "loss_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: loss_curve.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Phase 3e: Repetition-only first training run on real English words.\n\n"
            "Trains the full Lichtheim 2 architecture (both pathways) on REPETITION\n"
            "trials only. Saves metrics and predictions to a timestamped directory.\n"
            "See docs/repetition_only_training_note.md for detailed documentation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-dir", type=str, default="data/raw/nwr_swp",
        help="Directory containing phonemes.csv, wfe.csv, ssp.csv. (default: data/raw/nwr_swp)",
    )
    p.add_argument(
        "--config", type=str, default="configs/english_nwr.yaml",
        help="Model config YAML path. (default: configs/english_nwr.yaml)",
    )
    p.add_argument(
        "--source", type=str, default="words",
        choices=["words", "pseudowords", "mixed"],
        help="Item source: words (wfe.csv), pseudowords (ssp.csv), or mixed. (default: words)",
    )
    p.add_argument(
        "--max-items", type=int, default=10, dest="max_items",
        help="Maximum items to sample for training. (default: 10)",
    )
    p.add_argument(
        "--epochs", type=int, default=5,
        help="Number of training epochs. (default: 5)",
    )
    p.add_argument(
        "--lr", type=float, default=0.01,
        help=(
            "SGD learning rate. (default: 0.01 — conservative for short runs; "
            "paper uses 0.5 over 200 epochs)"
        ),
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        help="Compute device: cpu / cuda / mps. (default: cpu)",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for sampling and shuffle. (default: 0)",
    )
    p.add_argument(
        "--zero-error-radius", type=float, default=0.0, dest="zero_error_radius",
        help=(
            "Dead-zone threshold: units within this distance of target contribute "
            "zero loss and no gradient. 0.0 = disabled (debug default). "
            "Paper value: 0.1. [Open #5]"
        ),
    )
    p.add_argument(
        "--loss-reduction", type=str, default="sum", dest="loss_reduction",
        choices=["sum", "mean_active"],
        help="Loss reduction: 'sum' (default, paper behavior) or 'mean_active' (diagnostic).",
    )
    p.add_argument(
        "--eval-radius", type=float, default=0.1, dest="eval_radius",
        help=(
            "Radius for word-level evaluation metrics: a word is scored as correct "
            "if every output-phase motor unit satisfies |output - target| < eval_radius. "
            "Can differ from --zero-error-radius (training dead zone). "
            "Default 0.1 (paper zero_error_radius value). [Open #9]"
        ),
    )
    p.add_argument(
        "--sound-proj-size", type=int, default=None, dest="sound_proj_size",
        help=(
            "Dense input projection size: maps the 39D one-hot phoneme to N dims "
            "before sound_to_iSMG and sound_to_mSTG. None (default) = no projection "
            "(paper-style direct sound pathway). [Adapted — not in Ueno et al. 2011; "
            "Open #10] Overrides sound_proj_size from the YAML config."
        ),
    )
    p.add_argument(
        "--dorsal-motor-only",
        action="store_true",
        default=False,
        dest="dorsal_motor_only",
        help=(
            "Diagnostic: exclude triangularis_to_motor from motor readout. "
            "The full ventral pathway is still computed each tick; only the "
            "motor_net summation changes. Not a faithful Lichtheim 2 variant. "
            "Discuss results with Yair before drawing conclusions. Default: off."
        ),
    )
    p.add_argument(
        "--log-loss-decomp",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="log_loss_decomp",
        help=(
            "Compute phase/unit loss decomposition after each training epoch. "
            "Adds an eval-mode forward pass over all trials per epoch (~2× wall time). "
            "Disable with --no-log-loss-decomp for long Jean Zay runs. Default: on."
        ),
    )
    p.add_argument(
        "--output-dir", type=str, default="outputs/repetition_only", dest="output_dir",
        help="Parent output directory; a timestamped subdirectory is created. (default: outputs/repetition_only)",
    )
    return p.parse_args(argv)


def validate_args(args: argparse.Namespace) -> str | None:
    if args.max_items < 1:
        return f"--max-items must be >= 1, got {args.max_items}"
    if args.epochs < 1:
        return f"--epochs must be >= 1, got {args.epochs}"
    if args.lr <= 0.0:
        return f"--lr must be > 0, got {args.lr}"
    if args.zero_error_radius < 0.0:
        return f"--zero-error-radius must be >= 0, got {args.zero_error_radius}"
    if args.eval_radius < 0.0:
        return f"--eval-radius must be >= 0, got {args.eval_radius}"
    if args.sound_proj_size is not None and args.sound_proj_size < 1:
        return f"--sound-proj-size must be >= 1, got {args.sound_proj_size}"
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    err = validate_args(args)
    if err is not None:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    # Device validation — fail explicitly, no silent fallback
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Error: --device cuda requested but CUDA is not available.", file=sys.stderr)
        return 1
    if device.type == "mps" and not torch.backends.mps.is_available():
        print("Error: --device mps requested but MPS is not available.", file=sys.stderr)
        return 1

    # Reproducibility
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    # Timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = Path(args.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Lichtheim 2 — Repetition-only training (Phase 3e)")
    print("=" * 60)
    print(f"Run dir:  {run_dir}")
    print(f"Device:   {device}")
    print(f"Seed:     {args.seed}")

    # ------------------------------------------------------------------
    # Step 1: Load config and build model
    # ------------------------------------------------------------------
    print("\n[1/5] Config and model")
    cfg = load_config(Path(args.config))

    # CLI --sound-proj-size overrides the YAML value if explicitly provided.
    # This allows running both baseline and projection experiments from the
    # same YAML without creating separate config files.
    if args.sound_proj_size is not None:
        cfg.sound_proj_size = args.sound_proj_size   # override YAML value

    # CLI --dorsal-motor-only overrides YAML value when flag is present. [Phase 3j]
    if args.dorsal_motor_only:
        cfg.dorsal_motor_only = True   # CLI override [Phase 3j diagnostic]

    model = Lichtheim2Model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Config file:  {args.config}")
    print(f"  sound/motor:  {cfg.sound_input_size} / {cfg.motor_output_size}")
    print(f"  iSMG:         {cfg.iSMG_hidden_size}")
    print(f"  mSTG / aSTG:  {cfg.mSTG_hidden_size} / {cfg.aSTG_hidden_size}")
    print(f"  triangularis: {cfg.triangularis_hidden_size}")
    print(f"  vATL:         {cfg.vATL_size}")
    if cfg.sound_proj_size is not None:
        proj_src = " (CLI override)" if args.sound_proj_size is not None else " (from config)"
        print(f"  sound_proj:   {cfg.sound_input_size}D → {cfg.sound_proj_size}D{proj_src}  [Adapted — not in Ueno et al. 2011; Open #10]")
    else:
        print(f"  sound_proj:   none (paper-style direct sound pathway)")
    readout_label = (
        "dorsal only (diagnostic; triangularis_to_motor disabled)"
        if cfg.dorsal_motor_only else
        "full iSMG + triangularis  [Paper Fig 1]"
    )
    print(f"  motor_readout:    {readout_label}")
    print(f"  Parameters:   {n_params:,}")
    print(f"  Architecture: FULL Lichtheim2Model (dorsal + ventral) [Open #1]")

    # ------------------------------------------------------------------
    # Step 2: Load data
    # ------------------------------------------------------------------
    print(f"\n[2/5] Data loading  (source={args.source})")
    data_dir  = Path(args.data_dir)

    # Phoneme inventory — row order in phonemes.csv determines one-hot indices
    inventory = load_phoneme_inventory(data_dir / "phonemes.csv")
    print(f"  Phoneme inventory: {len(inventory)} symbols  [Open #6: one-hot vs 21-bit mora]")

    all_items: list[WordItem | PseudowordItem] = []

    if args.source in ("words", "mixed"):
        # load_word_items encodes phonemes with encode_phoneme_sequence;
        # raises KeyError immediately if any phoneme is absent from inventory.
        word_items = load_word_items(data_dir / "wfe.csv", inventory)
        all_items.extend(word_items)
        print(f"  Words loaded: {len(word_items)}")

    if args.source in ("pseudowords", "mixed"):
        pseudo_items = load_pseudoword_items(data_dir / "ssp.csv", inventory)
        all_items.extend(pseudo_items)
        print(f"  Pseudowords loaded: {len(pseudo_items)}")

    print(f"  Total items: {len(all_items)}")

    # Sample controlled subset
    if len(all_items) > args.max_items:
        all_items = rng.sample(all_items, args.max_items)
        print(f"  Sampled: {args.max_items} items  (--max-items)")
    else:
        rng.shuffle(all_items)

    print(f"  Items for training: {len(all_items)}")

    # ------------------------------------------------------------------
    # Step 3: Build repetition trials
    # ------------------------------------------------------------------
    print("\n[3/5] Building repetition trials")
    trials  = build_repetition_trials(all_items, cfg.motor_output_size)
    T_vals  = [t.phon_tensor.shape[0] for t in trials]
    print(f"  Trials: {len(trials)}")
    print(f"  Phoneme lengths T: min={min(T_vals)}  max={max(T_vals)}  "
          f"mean={sum(T_vals)/len(T_vals):.1f}")
    print(f"  Tick counts 2T:   min={2*min(T_vals)}  max={2*max(T_vals)}")
    print(f"  Loss mask: True for ALL 2T ticks (input + output phase) [Open #2]")

    # ------------------------------------------------------------------
    # Step 4: Pre-training predictions
    # ------------------------------------------------------------------
    print("\n[4/5] Pre-training evaluation (model at initialisation)")
    preds_before = evaluate_predictions(model, trials, cfg, inventory.symbols, device,
                                        eval_radius=args.eval_radius)
    s_b = _prediction_summary(preds_before)
    nb  = len(preds_before)
    print(f"  Phoneme argmax accuracy:          {s_b['avg_phoneme_accuracy']:.4f}")
    print(f"  Threshold accuracy (mixed):       {s_b['avg_threshold_accuracy']:.4f}  (output phase, pos+neg combined)")
    print(f"  Whole-word exact match:           {s_b['n_exact']}/{nb}")
    print(f"  -- Split metrics --")
    print(f"  Input silence  (frac < 0.1):      {s_b['avg_input_silence_threshold_acc']:.4f}")
    print(f"  Output neg     (frac < 0.1):      {s_b['avg_output_negative_threshold_acc']:.4f}")
    print(f"  Output pos     (frac > 0.9):      {s_b['avg_output_positive_threshold_acc']:.4f}")
    print(f"  Mean positive unit activation:    {s_b['avg_mean_positive_output']:.4f}")
    print(f"  Mean negative unit activation:    {s_b['avg_mean_negative_output']:.4f}")
    print(f"  Mean max motor per output tick:   {s_b['avg_mean_max_motor_output']:.4f}")
    print(f"  -- Word accuracy (candidate, radius={args.eval_radius}) --")
    print(f"  Output phase all-within-radius:   {s_b['n_output_correct']}/{nb}  ({s_b['paper_like_output_word_accuracy']:.4f})  [Open #9]")
    print(f"  Input phase all-silent:           {s_b['n_input_silent']}/{nb}  ({s_b['input_silence_word_accuracy']:.4f})")
    print(f"  Strict trial (both phases):       {s_b['n_strict_correct']}/{nb}  ({s_b['candidate_strict_trial_accuracy']:.4f})  [Open #9]")

    # ------------------------------------------------------------------
    # Step 5: Training
    # ------------------------------------------------------------------
    decomp_status = (
        "enabled (eval pass each epoch; ~2× wall time)"
        if args.log_loss_decomp else
        "disabled (--no-log-loss-decomp)"
    )
    print(f"\n[5/5] Training")
    print(f"  Optimizer:          SGD, lr={args.lr}")
    print(f"  Loss reduction:     {args.loss_reduction}")
    print(f"  zero_error_radius:  {args.zero_error_radius}  [Open #5: paper=0.1]")
    print(f"  BPTT:               full through 2T ticks per trial  [Open #7]")
    print(f"  motor_readout:      {'dorsal only (diagnostic)' if args.dorsal_motor_only else 'full iSMG + triangularis'}")
    print(f"  Loss decomp:        {decomp_status}")
    print()

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    try:
        epoch_metrics = run_training(
            model, trials, optimizer, cfg,
            epochs=args.epochs,
            zero_error_radius=args.zero_error_radius,
            loss_reduction=args.loss_reduction,
            device=device,
            rng=rng,
            log_loss_decomp=args.log_loss_decomp,
        )
    except RuntimeError as exc:
        print(f"\nTraining error: {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Post-training predictions
    # ------------------------------------------------------------------
    print("\nPost-training evaluation")
    preds_after = evaluate_predictions(model, trials, cfg, inventory.symbols, device,
                                       eval_radius=args.eval_radius)
    s_a = _prediction_summary(preds_after)
    na  = len(preds_after)
    print(f"  Phoneme argmax accuracy:          {s_a['avg_phoneme_accuracy']:.4f}")
    print(f"  Threshold accuracy (mixed):       {s_a['avg_threshold_accuracy']:.4f}  (output phase, pos+neg combined)")
    print(f"  Whole-word exact match:           {s_a['n_exact']}/{na}")
    print(f"  -- Split metrics --")
    print(f"  Input silence  (frac < 0.1):      {s_a['avg_input_silence_threshold_acc']:.4f}")
    print(f"  Output neg     (frac < 0.1):      {s_a['avg_output_negative_threshold_acc']:.4f}")
    print(f"  Output pos     (frac > 0.9):      {s_a['avg_output_positive_threshold_acc']:.4f}")
    print(f"  Mean positive unit activation:    {s_a['avg_mean_positive_output']:.4f}")
    print(f"  Mean negative unit activation:    {s_a['avg_mean_negative_output']:.4f}")
    print(f"  Mean max motor per output tick:   {s_a['avg_mean_max_motor_output']:.4f}")
    print(f"  -- Word accuracy (candidate, radius={args.eval_radius}) --")
    print(f"  Output phase all-within-radius:   {s_a['n_output_correct']}/{na}  ({s_a['paper_like_output_word_accuracy']:.4f})  [Open #9]")
    print(f"  Input phase all-silent:           {s_a['n_input_silent']}/{na}  ({s_a['input_silence_word_accuracy']:.4f})")
    print(f"  Strict trial (both phases):       {s_a['n_strict_correct']}/{na}  ({s_a['candidate_strict_trial_accuracy']:.4f})  [Open #9]")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    print(f"\nSaving outputs to {run_dir}/")
    save_run_config(run_dir, args, cfg)
    print("  Saved: run_config.json")
    save_metrics_csv(run_dir, epoch_metrics)
    print("  Saved: metrics.csv")
    save_predictions(run_dir, preds_before, "predictions_before.json")
    print("  Saved: predictions_before.json")
    save_predictions(run_dir, preds_after, "predictions_after.json")
    print("  Saved: predictions_after.json")
    save_loss_curve(run_dir, epoch_metrics)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    initial_avg    = epoch_metrics[0]["avg_loss"]
    final_avg      = epoch_metrics[-1]["avg_loss"]
    loss_decreased = final_avg < initial_avg

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Epochs:                {args.epochs}")
    print(f"  Items:                 {len(trials)}")
    print(f"  Initial avg BCE loss:  {initial_avg:.6f}")
    print(f"  Final avg BCE loss:    {final_avg:.6f}")
    print(f"  Loss decreased:        {'YES ✓' if loss_decreased else 'NO  (expected for very short runs)'}")
    print(f"  Phoneme acc:         before={s_b['avg_phoneme_accuracy']:.4f}  →  after={s_a['avg_phoneme_accuracy']:.4f}")
    print(f"  Mean pos out:        before={s_b['avg_mean_positive_output']:.4f}  →  after={s_a['avg_mean_positive_output']:.4f}")
    print(f"  Exact match:         before={s_b['n_exact']}/{nb}  →  after={s_a['n_exact']}/{na}")
    print(f"  Paper-like word acc: before={s_b['n_output_correct']}/{nb}  →  after={s_a['n_output_correct']}/{na}  (radius={args.eval_radius})  [Open #9]")
    if args.log_loss_decomp and epoch_metrics:
        e1 = epoch_metrics[0]
        ef = epoch_metrics[-1]
        print(f"  Loss decomp (epoch 1 → final):")
        print(f"    Input-phase BCE:   {e1['avg_eval_input_bce']:.4f}  →  {ef['avg_eval_input_bce']:.4f}")
        print(f"    Output-neg BCE:    {e1['avg_eval_output_neg_bce']:.4f}  →  {ef['avg_eval_output_neg_bce']:.4f}")
        print(f"    Output-pos BCE:    {e1['avg_eval_output_pos_bce']:.4f}  →  {ef['avg_eval_output_pos_bce']:.4f}")
        print(f"    Active neg/pos after dead-zone (epoch {args.epochs}):  "
              f"{ef['avg_eval_n_active_output_neg']:.0f} / {ef['avg_eval_n_active_output_pos']:.0f}  per trial")
    print(f"  All losses finite:     YES")
    print(f"  Output dir:            {run_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
