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
) -> dict:
    """Run one trial in eval mode (no_grad) and compute prediction metrics.

    Metrics are computed on the output phase (ticks T..2T-1) only.

    Accuracy definitions:
      phoneme_accuracy:   fraction of output-phase ticks where argmax of motor
                          output matches argmax of (one-hot) target.
      threshold_accuracy: fraction of (tick, unit) pairs in the output phase
                          where |motor_output - target| < 0.1.
      exact_match:        True if phoneme_accuracy == 1.0 (all phonemes correct).

    [Note #4] Sound input is the raw phoneme tensor (clamped), not sigmoided.
    [Open #3] The ventral pathway is computed at every tick.
    """
    T = trial.phon_tensor.shape[0]
    output_tick_indices = list(range(T, 2 * T))    # output-phase tick indices in results list

    trial_dev = move_trial_to_device(trial, device)
    sem_zeros = torch.zeros(cfg.vATL_size, device=device)

    model.eval()
    with torch.no_grad():
        results = model.run_trial(trial_dev.task, trial_dev.phon_tensor, sem_zeros, cfg)

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

    # Threshold accuracy over all (tick, unit) pairs in the output phase
    within_threshold = (motor_outputs - motor_targets).abs() < 0.1
    threshold_acc = within_threshold.float().mean().item()

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
    }


def evaluate_predictions(
    model: Lichtheim2Model,
    trials: list[SupervisedTrial],
    cfg: ModelConfig,
    inventory_symbols: list[str],
    device: torch.device,
) -> list[dict]:
    """Evaluate predictions for all trials. Returns one record per trial."""
    return [
        _evaluate_one_trial(model, trial, cfg, inventory_symbols, device)
        for trial in trials
    ]


def _prediction_summary(preds: list[dict]) -> tuple[float, float, int]:
    """Return (avg_phoneme_acc, avg_threshold_acc, n_exact) for a list of prediction dicts."""
    if not preds:
        return float("nan"), float("nan"), 0
    avg_ph  = sum(p["phoneme_accuracy"]   for p in preds) / len(preds)
    avg_thr = sum(p["threshold_accuracy"] for p in preds) / len(preds)
    n_exact = sum(1 for p in preds if p["exact_match"])
    return avg_ph, avg_thr, n_exact


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
) -> list[dict]:
    """Online item-by-item training loop.

    At each epoch: shuffle trials, call train_step() per item.
    Per-epoch metrics: avg_loss, min_loss, max_loss, n_trials, all_finite.

    [Open #7] Full BPTT through all 2T ticks is used (no truncation).
              loss.backward() is called once per trial after summing tick losses.

    Raises:
        RuntimeError: if any per-item loss is non-finite (nan or inf),
                      with epoch number and trial label for debugging.
    """
    epoch_metrics: list[dict] = []

    for epoch in range(1, epochs + 1):
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
        "loss_reduction":     args.loss_reduction,
        "output_dir":         str(run_dir),
        "model_config": {
            "sound_input_size":         cfg.sound_input_size,
            "motor_output_size":        cfg.motor_output_size,
            "iSMG_hidden_size":         cfg.iSMG_hidden_size,
            "mSTG_hidden_size":         cfg.mSTG_hidden_size,
            "aSTG_hidden_size":         cfg.aSTG_hidden_size,
            "triangularis_hidden_size": cfg.triangularis_hidden_size,
            "vATL_size":                cfg.vATL_size,
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
    cfg   = load_config(Path(args.config))
    model = Lichtheim2Model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Config file:  {args.config}")
    print(f"  sound/motor:  {cfg.sound_input_size} / {cfg.motor_output_size}")
    print(f"  iSMG:         {cfg.iSMG_hidden_size}")
    print(f"  mSTG / aSTG:  {cfg.mSTG_hidden_size} / {cfg.aSTG_hidden_size}")
    print(f"  triangularis: {cfg.triangularis_hidden_size}")
    print(f"  vATL:         {cfg.vATL_size}")
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
    preds_before = evaluate_predictions(model, trials, cfg, inventory.symbols, device)
    ph_b, thr_b, ex_b = _prediction_summary(preds_before)
    print(f"  Phoneme argmax accuracy: {ph_b:.4f}")
    print(f"  Threshold accuracy:      {thr_b:.4f}  (|out-target| < 0.1)")
    print(f"  Whole-word exact match:  {ex_b}/{len(preds_before)}")

    # ------------------------------------------------------------------
    # Step 5: Training
    # ------------------------------------------------------------------
    print(f"\n[5/5] Training")
    print(f"  Optimizer:          SGD, lr={args.lr}")
    print(f"  Loss reduction:     {args.loss_reduction}")
    print(f"  zero_error_radius:  {args.zero_error_radius}  [Open #5: paper=0.1]")
    print(f"  BPTT:               full through 2T ticks per trial  [Open #7]")
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
        )
    except RuntimeError as exc:
        print(f"\nTraining error: {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Post-training predictions
    # ------------------------------------------------------------------
    print("\nPost-training evaluation")
    preds_after = evaluate_predictions(model, trials, cfg, inventory.symbols, device)
    ph_a, thr_a, ex_a = _prediction_summary(preds_after)
    print(f"  Phoneme argmax accuracy: {ph_a:.4f}")
    print(f"  Threshold accuracy:      {thr_a:.4f}  (|out-target| < 0.1)")
    print(f"  Whole-word exact match:  {ex_a}/{len(preds_after)}")

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
    print(f"  Phoneme acc:  before={ph_b:.4f}  →  after={ph_a:.4f}")
    print(f"  Exact match:  before={ex_b}/{len(preds_before)}  →  after={ex_a}/{len(preds_after)}")
    print(f"  All losses finite:     YES")
    print(f"  Output dir:            {run_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
