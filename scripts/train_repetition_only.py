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
from lichtheim2.loss_decomposition import (
    compute_epoch_loss_decomp as _compute_epoch_loss_decomp,
    compute_trial_loss_decomposition as _compute_trial_loss_decomposition,
)
from lichtheim2.metrics import (
    compute_repetition_metric_breakdown as _compute_repetition_metric_breakdown,
    compute_repetition_word_accuracy as _compute_repetition_word_accuracy,
    prediction_summary as _prediction_summary,
    rolling_mean as _rolling_mean,
)
from lichtheim2.model import Lichtheim2Model
from lichtheim2.repetition_evaluation import (
    evaluate_one_trial as _evaluate_one_trial,
    evaluate_predictions,
)
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
# (moved to src/lichtheim2/metrics.py; imported as _compute_repetition_metric_breakdown
#  and _compute_repetition_word_accuracy)
# ---------------------------------------------------------------------------


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


# (evaluate_one_trial, evaluate_predictions, prediction_summary moved to
#  src/lichtheim2/repetition_evaluation.py and src/lichtheim2/metrics.py;
#  imported above as _evaluate_one_trial, evaluate_predictions, _prediction_summary)


# ---------------------------------------------------------------------------
# Post-epoch eval loss decomposition (Phase 3i diagnostic)
# (moved to src/lichtheim2/loss_decomposition.py;
#  imported above as _compute_trial_loss_decomposition, _compute_epoch_loss_decomp)
# ---------------------------------------------------------------------------


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
    output_positive_weight: float = 1.0,
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
                output_positive_weight=output_positive_weight,
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
        "output_positive_weight": args.output_positive_weight,
        "loss_variant":           "output_positive_weighted" if args.output_positive_weight != 1.0 else "standard_bce",
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


def save_loss_curve(run_dir: Path, epoch_metrics: list[dict], rolling_window: int = 10) -> None:
    """Save loss curve PNG(s) if matplotlib is available (silent skip otherwise).

    Always saves loss_curve.png (avg loss only).
    Also saves loss_decomposition_curve.png if decomposition columns are present.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping loss_curve.png)")
        return

    saved: list[str] = []
    epochs = [m["epoch"] for m in epoch_metrics]
    avg_losses = [m["avg_loss"] for m in epoch_metrics]

    # Main plot: avg loss only (no min/max fill)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, avg_losses, linewidth=1.5, label="avg loss")
    if len(epochs) >= rolling_window:
        smooth = _rolling_mean(avg_losses, rolling_window)
        ax.plot(epochs, smooth, linewidth=2, label=f"rolling mean (w={rolling_window})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss (sum)")
    ax.set_title("Repetition-only training — average loss per epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=150)
    plt.close(fig)
    saved.append("loss_curve.png")

    # Decomposition plot (skipped silently if columns absent)
    decomp_keys = ("avg_eval_output_pos_bce", "avg_eval_output_neg_bce", "avg_eval_input_bce")
    if epoch_metrics and all(k in epoch_metrics[0] for k in decomp_keys):
        fig, ax = plt.subplots(figsize=(8, 4))
        for key, label in [
            ("avg_eval_output_pos_bce", "output-positive BCE"),
            ("avg_eval_output_neg_bce", "output-negative BCE"),
            ("avg_eval_input_bce",      "input BCE"),
        ]:
            ax.plot(epochs, [m[key] for m in epoch_metrics], linewidth=1.5, label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE loss (sum, unweighted eval)")
        ax.set_title("Loss decomposition — eval pass per epoch")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "loss_decomposition_curve.png", dpi=150)
        plt.close(fig)
        saved.append("loss_decomposition_curve.png")

    print(f"  Saved: {', '.join(saved)}")


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
        "--output-positive-weight",
        type=float,
        default=1.0,
        dest="output_positive_weight",
        help=(
            "Diagnostic: multiply the training loss on output-phase motor units whose "
            "target=1 (the positive phoneme unit) by this factor. Scoped to REPETITION "
            "task, output phase ticks T..2T-1, units with target >= 0.5. "
            "1.0 = standard BCE (default). Values > 1 increase gradient on the correct "
            "phoneme unit. Must be > 0. Not a faithful Ueno et al. 2011 variant. "
            "[Phase 3k diagnostic]"
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
    if args.output_positive_weight <= 0.0:
        return f"--output-positive-weight must be > 0, got {args.output_positive_weight}"
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
    pos_weight_label = (
        f"{args.output_positive_weight}  [diagnostic; output-phase target=1 units upweighted]"
        if args.output_positive_weight != 1.0 else
        "1.0  (baseline standard BCE)"
    )
    print(f"  output_positive_weight: {pos_weight_label}")
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
            output_positive_weight=args.output_positive_weight,
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
