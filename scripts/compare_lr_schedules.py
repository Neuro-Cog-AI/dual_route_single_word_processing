"""Phase 3c-10: Compare constant vs paper-proportional LR schedule on small subsets.

The paper LR schedule (Ueno et al. 2011 [Paper]) is mapped proportionally to the
requested --epochs so the comparison is meaningful at any epoch count.

Paper schedule (exact values for --epochs 200 --lr 0.5):
  epochs   1–150 (0–75% of training): LR = 0.5
  epochs 151–160 (75–80%):            LR = 0.4
  epochs 161–170 (80–85%):            LR = 0.3
  epochs 171–180 (85–90%):            LR = 0.2
  epochs 181–200 (90–100%):           LR = 0.1

For other --epochs or --lr, the schedule shape (multipliers) is preserved
proportionally and --lr sets the starting LR.  The phrase "paper-proportional LR
schedule" is used throughout.

Conditions compared:
  constant — same --lr every epoch
  paper    — paper-proportional LR schedule, starting from --lr

Fairness guarantee: same sampled items, same trials, same task schedule, same
frequency weighting, same model initialisation (torch.manual_seed(seed)), same trial
shuffle order (random.Random(seed)), same loss_reduction, zero_error_radius, device.
Only the LR schedule changes between conditions.

Usage (from dual_route_single_word_processing/):
    PYTHONPATH=src python scripts/compare_lr_schedules.py \\
        --data-dir data/raw/nwr_swp \\
        --config configs/english_nwr.yaml \\
        --mode mixed-multitask --task-schedule paper \\
        --frequency-source none \\
        --max-words 10 --max-pseudowords 10 \\
        --epochs 20 --lr 0.01 --device cpu --seed 0 \\
        --loss-reduction mean_active

Exit code 0 if both runs complete with finite losses; 1 on error.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lichtheim2.config import ModelConfig, load_config
from lichtheim2.data import PseudowordItem, WordItem, load_pseudoword_items, load_word_items
from lichtheim2.encoding import load_phoneme_inventory
from lichtheim2.model import Lichtheim2Model
from lichtheim2.semantics import assign_artificial_semantics
from lichtheim2.tasks import Task

sys.path.insert(0, str(Path(__file__).parent))
from diagnose_small_subset_training import (
    DiagnosticResult,
    run_diagnostic_epochs,
    sample_items,
)
from compare_frequency_weighting import apply_weights_to_trials, compute_word_weights
from compare_task_schedules import SCHEDULES, build_trials_for_schedule


# ---------------------------------------------------------------------------
# LR schedule function
# ---------------------------------------------------------------------------


def lr_for_epoch(base_lr: float, epoch: int, lr_schedule: str, total_epochs: int) -> float:
    """Return the learning rate for a given epoch.

    Args:
        base_lr:       starting (maximum) learning rate
        epoch:         current epoch, 1-indexed
        lr_schedule:   "constant" or "paper" (paper-proportional schedule)
        total_epochs:  total number of training epochs

    Returns:
        Learning rate as a float > 0.

    Paper-proportional schedule (Ueno et al. 2011 [Paper]):
        The paper trains for 200 epochs with LR 0.5 → 0.4 → 0.3 → 0.2 → 0.1.
        Phase boundaries as fractions of total training:
          0.00–0.75: full base_lr      (multiplier 1.0)
          0.75–0.80: base_lr × 0.8
          0.80–0.85: base_lr × 0.6
          0.85–0.90: base_lr × 0.4
          0.90–1.00: base_lr × 0.2    (minimum)
        For --epochs 200 --lr 0.5, this reproduces the exact paper LR values.
        For other settings, the multipliers are preserved and base_lr sets the scale.

    Raises:
        ValueError: if lr_schedule is unknown.
    """
    if lr_schedule == "constant":
        return base_lr
    elif lr_schedule == "paper":
        frac = epoch / total_epochs
        if frac <= 0.75:
            return base_lr
        elif frac <= 0.80:
            return base_lr * 0.8
        elif frac <= 0.85:
            return base_lr * 0.6
        elif frac <= 0.90:
            return base_lr * 0.4
        else:
            return base_lr * 0.2
    else:
        raise ValueError(
            f"Unknown lr_schedule: {lr_schedule!r}; expected 'constant' or 'paper'"
        )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class LRComparisonResult:
    """Results for constant vs paper-proportional LR schedule comparison."""

    results: dict[str, DiagnosticResult]      # keys: "constant", "paper"
    trials_per_epoch: int
    lr_schedule_used: dict[str, list[float]]  # condition → LR at each epoch (1-indexed)


# ---------------------------------------------------------------------------
# Core comparison function
# ---------------------------------------------------------------------------


def run_lr_comparison(
    cfg: ModelConfig,
    word_items: list[WordItem],
    pseudo_items: list[PseudowordItem],
    sem_map: dict[int, torch.Tensor],
    mode: str,
    task_schedule: str,
    epochs: int,
    base_lr: float,
    zero_error_radius: float,
    device: torch.device,
    seed: int,
    loss_reduction: str = "mean_active",
    frequency_source: str = "none",
    weight_normalization: str = "mean_one",
) -> LRComparisonResult:
    """Run constant-LR and paper-proportional-LR training under identical conditions.

    Fairness guarantee:
    - word_items, pseudo_items, and sem_map are pre-loaded and sampled once before
      this call.
    - Base trials are built once; both conditions use the same trial structure and
      the same optional frequency weighting.
    - For each condition: torch.manual_seed(seed) before model instantiation and
      random.Random(seed) as rng → identical initialisation and shuffle order.
    - Same task_schedule, epochs, base_lr (starting LR), zero_error_radius, device,
      loss_reduction.  Only the LR schedule varies.

    Raises:
        RuntimeError: if trial list is empty, or propagated from run_diagnostic_epochs.
    """
    base_trials = build_trials_for_schedule(
        mode, word_items, pseudo_items, sem_map, cfg.motor_output_size, task_schedule
    )
    if not base_trials:
        raise RuntimeError(
            "No trials built. Check mode/max-words/max-pseudowords."
        )

    # Optional frequency weighting (same for both LR conditions).
    if frequency_source != "none":
        weight_map = compute_word_weights(
            word_items, frequency_source=frequency_source,
            normalization=weight_normalization,
        )
        trials = apply_weights_to_trials(base_trials, weight_map)
    else:
        trials = base_trials

    results: dict[str, DiagnosticResult] = {}
    lr_schedule_used: dict[str, list[float]] = {}

    for condition in ("constant", "paper"):
        schedule_fn = lambda ep, cond=condition: lr_for_epoch(base_lr, ep, cond, epochs)
        lr_schedule_used[condition] = [lr_for_epoch(base_lr, e, condition, epochs)
                                       for e in range(1, epochs + 1)]

        torch.manual_seed(seed)
        model     = Lichtheim2Model(cfg).to(device)
        optimizer = optim.SGD(model.parameters(), lr=base_lr)
        rng       = random.Random(seed)

        results[condition] = run_diagnostic_epochs(
            model, trials, optimizer, cfg,
            epochs=epochs,
            zero_error_radius=zero_error_radius,
            device=device,
            rng=rng,
            loss_reduction=loss_reduction,
            verbose=False,
            lr_schedule_fn=schedule_fn,
        )

    return LRComparisonResult(
        results=results,
        trials_per_epoch=len(trials),
        lr_schedule_used=lr_schedule_used,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _task_avg_epoch(
    result: DiagnosticResult,
    task: Task,
    epoch_idx: int,
) -> float | None:
    """Mean loss for task at epoch_idx (0=first, -1=last), or None if no trials."""
    epoch = result.epoch_losses[epoch_idx]
    vals  = epoch.get(task, [])
    return sum(vals) / len(vals) if vals else None


def _describe_lr_profile(lr_values: list[float], base_lr: float) -> list[str]:
    """Return a compact description of the LR profile as a list of lines."""
    if not lr_values:
        return []
    # Collapse consecutive equal values into ranges
    lines: list[str] = []
    run_start = 1
    run_lr    = lr_values[0]
    for i, lr in enumerate(lr_values[1:], start=2):
        if abs(lr - run_lr) > 1e-12:
            ep_label = f"epoch {run_start:3d}" if run_start == i - 1 else f"epochs {run_start:3d}–{i-1:3d}"
            lines.append(f"  {ep_label}: {run_lr:.6f}")
            run_start = i
            run_lr    = lr
    # Final run
    ep_label = f"epoch {run_start:3d}" if run_start == len(lr_values) else f"epochs {run_start:3d}–{len(lr_values):3d}"
    lines.append(f"  {ep_label}: {run_lr:.6f}")
    return lines


def print_lr_comparison_table(
    result: LRComparisonResult,
    n_epochs: int | None = None,
    base_lr: float | None = None,
) -> None:
    """Print LR profile, summary table, and per-task initial→final losses."""
    epoch_label = f"epoch {n_epochs}" if n_epochs is not None else "final epoch"

    # --- LR profile for paper-proportional condition ---
    paper_lrs = result.lr_schedule_used.get("paper", [])
    if paper_lrs and base_lr is not None:
        profile_lines = _describe_lr_profile(paper_lrs, base_lr)
        n_ep = len(paper_lrs)
        lr_label = f"base_lr={base_lr:.5f}, epochs={n_ep}"
        print(f"Paper-proportional LR schedule ({lr_label}):")
        for line in profile_lines:
            print(line)
        print(
            f"  (For --epochs 200 --lr 0.5, reproduces exact paper schedule: "
            f"0.5 → 0.4 → 0.3 → 0.2 → 0.1)"
        )
        print()

    # --- Summary table ---
    hdr = (
        f"{'Condition':<12} | {'Trials/ep':>9} | {'Initial avg':>12} | {'Final avg':>10} | "
        f"{'Best avg':>10} | {'Best ep':>7} | {'% decrease':>10}"
    )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    for condition in ("constant", "paper"):
        r        = result.results[condition]
        n_trials = result.trials_per_epoch
        if r.initial_avg != 0.0:
            pct     = (r.initial_avg - r.final_avg) / r.initial_avg * 100.0
            pct_str = f"{pct:>9.1f}%"
        else:
            pct_str = f"{'  nan':>10}"
        print(
            f"{condition:<12} | {n_trials:>9d} | {r.initial_avg:>12.6f} | {r.final_avg:>10.6f} | "
            f"{r.best_avg:>10.6f} | {r.best_epoch:>7d} | {pct_str}"
        )

    print()

    # --- Per-task losses: initial → final ---
    active_tasks = [
        t for t in (Task.REPETITION, Task.COMPREHENSION, Task.SPEAKING)
        if any(
            _task_avg_epoch(result.results[c], t, 0) is not None
            for c in ("constant", "paper")
        )
    ]
    if not active_tasks:
        return

    names = {Task.REPETITION: "REP", Task.COMPREHENSION: "COMP", Task.SPEAKING: "SPK"}
    col_w = 17  # "0.123456→0.098765"
    task_cols = "  ".join(f"{names[t]:>{col_w}}" for t in active_tasks)
    print(f"Per-task losses (epoch 1 → {epoch_label}):")
    print(f"{'Condition':<12} | {task_cols}")
    print("-" * (13 + (col_w + 2) * len(active_tasks)))
    for condition in ("constant", "paper"):
        r     = result.results[condition]
        cells = []
        for t in active_tasks:
            init_v  = _task_avg_epoch(r, t,  0)
            final_v = _task_avg_epoch(r, t, -1)
            if init_v is not None and final_v is not None:
                cells.append(f"{init_v:.6f}→{final_v:.6f}")
            else:
                cells.append(f"{'---':>{col_w}}")
        print(f"{condition:<12} | {'  '.join(cells)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def validate_args(args: argparse.Namespace) -> str | None:
    """Return an error message string if args are invalid, else None."""
    if args.max_words < 0:
        return f"--max-words must be >= 0, got {args.max_words}"
    if args.max_pseudowords < 0:
        return f"--max-pseudowords must be >= 0, got {args.max_pseudowords}"
    if args.epochs < 1:
        return f"--epochs must be >= 1, got {args.epochs}"
    if args.lr <= 0:
        return f"--lr must be > 0, got {args.lr}"
    if args.zero_error_radius < 0:
        return f"--zero-error-radius must be >= 0, got {args.zero_error_radius}"
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Phase 3c-10: Compare constant vs paper-proportional LR schedule "
            "on small subsets."
        )
    )
    p.add_argument("--data-dir",             type=str,   default="data/raw/nwr_swp")
    p.add_argument("--config",               type=str,   default="configs/english_nwr.yaml")
    p.add_argument("--mode",                 type=str,   default="mixed-multitask",
                   choices=["repetition", "words-multitask", "mixed-multitask"])
    p.add_argument("--task-schedule",        type=str,   default="paper",
                   dest="task_schedule",
                   choices=list(SCHEDULES),
                   help="Task presentation schedule (default: paper).")
    p.add_argument("--max-words",            type=int,   default=10,
                   dest="max_words",
                   help="Max words to use. 0=none. Default: 10.")
    p.add_argument("--max-pseudowords",      type=int,   default=10,
                   dest="max_pseudowords",
                   help="Max pseudowords to use. 0=none. Default: 10.")
    p.add_argument("--epochs",               type=int,   default=20)
    p.add_argument("--lr",                   type=float, default=0.01)
    p.add_argument("--device",               type=str,   default="cpu")
    p.add_argument("--seed",                 type=int,   default=0)
    p.add_argument("--zero-error-radius",    type=float, default=0.0,
                   dest="zero_error_radius",
                   help="0.0 = no dead-zone (default).")
    p.add_argument("--loss-reduction",       type=str,   default="mean_active",
                   dest="loss_reduction",
                   choices=["sum", "mean_active"],
                   help="Loss reduction (default: mean_active).")
    p.add_argument("--frequency-source",     type=str,   default="none",
                   dest="frequency_source",
                   choices=["none", "zipf", "frequency"],
                   help=(
                       "Frequency weighting source: 'none' (default, unweighted), "
                       "'zipf' (Zipf-scale), 'frequency' (raw count, log1p-transformed)."
                   ))
    p.add_argument("--weight-normalization", type=str,   default="mean_one",
                   dest="weight_normalization",
                   choices=["mean_one"],
                   help="Weight normalisation (default: mean_one).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    err = validate_args(args)
    if err is not None:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Error: --device cuda requested but CUDA is not available.", file=sys.stderr)
        return 1
    if device.type == "mps" and not torch.backends.mps.is_available():
        print("Error: --device mps requested but MPS is not available.", file=sys.stderr)
        return 1

    cfg      = load_config(Path(args.config))
    data_dir = Path(args.data_dir)
    inventory = load_phoneme_inventory(data_dir / "phonemes.csv")

    word_items: list[WordItem] = []
    sem_map: dict[int, torch.Tensor] = {}
    if args.max_words != 0:
        word_items_all = load_word_items(data_dir / "wfe.csv", inventory)
        if args.mode in ("words-multitask", "mixed-multitask"):
            sem_map = assign_artificial_semantics(
                word_items_all, vATL_size=cfg.vATL_size, seed=args.seed
            )
        word_items = sample_items(word_items_all, args.max_words, rng)

    pseudo_items: list[PseudowordItem] = []
    if args.mode in ("repetition", "mixed-multitask") and args.max_pseudowords != 0:
        pseudo_items_all = load_pseudoword_items(data_dir / "ssp.csv", inventory)
        pseudo_items = sample_items(pseudo_items_all, args.max_pseudowords, rng)

    print("=== LR Schedule Comparison ===")
    print(
        f"Mode: {args.mode}  |  task_schedule: {args.task_schedule}  |  "
        f"frequency_source: {args.frequency_source}  |  loss_reduction: {args.loss_reduction}"
    )
    if word_items:
        task_desc = "rep only" if args.mode == "repetition" else "rep+comp+spk"
        print(f"Words: {len(word_items)} ({task_desc})")
    if pseudo_items:
        print(f"Pseudowords: {len(pseudo_items)} (rep only)")
    print(
        f"Config: sound={cfg.sound_input_size}  "
        f"motor={cfg.motor_output_size}  vATL={cfg.vATL_size}"
    )
    print(
        f"base_lr={args.lr}  zero_error_radius={args.zero_error_radius}  "
        f"epochs={args.epochs}  seed={args.seed}"
    )
    print()

    try:
        comparison = run_lr_comparison(
            cfg, word_items, pseudo_items, sem_map,
            mode=args.mode,
            task_schedule=args.task_schedule,
            epochs=args.epochs,
            base_lr=args.lr,
            zero_error_radius=args.zero_error_radius,
            device=device,
            seed=args.seed,
            loss_reduction=args.loss_reduction,
            frequency_source=args.frequency_source,
            weight_normalization=args.weight_normalization,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_lr_comparison_table(
        comparison,
        n_epochs=args.epochs,
        base_lr=args.lr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
