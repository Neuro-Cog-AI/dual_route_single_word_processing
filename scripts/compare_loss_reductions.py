"""Phase 3c-7: Compare sum vs mean_active loss reductions on small subsets.

Runs both loss reductions under identical conditions (same model seed, same
data subset, same trial shuffle order, same epochs) and prints a compact
comparison table. Use % decrease to compare training dynamics — do not
compare absolute loss values directly, as sum and mean_active operate on
different scales.

Usage (from dual_route_single_word_processing/):
    PYTHONPATH=src python scripts/compare_loss_reductions.py \\
        --data-dir data/raw/nwr_swp \\
        --config configs/english_nwr.yaml \\
        --mode mixed-multitask \\
        --max-words 10 --max-pseudowords 10 \\
        --epochs 20 --lr 0.01 --device cpu --seed 0

Exit code 0 if both runs complete with finite loss; 1 on error.
"""
from __future__ import annotations

import argparse
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
from lichtheim2.trials import SupervisedTrial

# Import helpers from Phase 3c-4 script.
# Safe because diagnose_small_subset_training.py is guarded by
# `if __name__ == "__main__": main()` and does not execute on import.
sys.path.insert(0, str(Path(__file__).parent))
from diagnose_small_subset_training import (
    DiagnosticResult,
    build_trials_for_mode,
    run_diagnostic_epochs,
    sample_items,
)


# ---------------------------------------------------------------------------
# Comparison result
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Results for both loss reductions trained under identical conditions."""

    results: dict[str, DiagnosticResult]  # keys: "sum", "mean_active"


# ---------------------------------------------------------------------------
# Core comparison function
# ---------------------------------------------------------------------------


def run_comparison(
    cfg: ModelConfig,
    trials: list[SupervisedTrial],
    epochs: int,
    lr: float,
    zero_error_radius: float,
    device: torch.device,
    seed: int,
) -> ComparisonResult:
    """Run both loss reductions under identical conditions.

    Fairness guarantee:
    - ``trials`` is a pre-built shared list; items are loaded and sampled
      exactly once before this call.
    - For each reduction, ``torch.manual_seed(seed)`` is called before model
      instantiation → identical weight initialization.
    - For each reduction, ``random.Random(seed)`` is passed as ``rng`` →
      identical trial shuffle order every epoch.
    - Same ``epochs``, ``lr``, ``zero_error_radius``, and ``device`` for both.
    - Per-epoch logs are suppressed (``verbose=False``); only the final table
      is printed by the caller.

    Raises:
        RuntimeError: propagated from run_diagnostic_epochs if any loss is
                      non-finite.
    """
    results: dict[str, DiagnosticResult] = {}
    for reduction in ["sum", "mean_active"]:
        torch.manual_seed(seed)
        model = Lichtheim2Model(cfg).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        rng = random.Random(seed)
        result = run_diagnostic_epochs(
            model, trials, optimizer, cfg,
            epochs=epochs,
            zero_error_radius=zero_error_radius,
            device=device,
            rng=rng,
            loss_reduction=reduction,
            verbose=False,
        )
        results[reduction] = result
    return ComparisonResult(results=results)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _task_avg(result: DiagnosticResult, task: Task) -> float | None:
    """Mean loss for task in the last epoch, or None if no trials that task."""
    last = result.epoch_losses[-1]
    vals = last.get(task, [])
    return sum(vals) / len(vals) if vals else None


def print_comparison_table(
    comparison: ComparisonResult,
    n_epochs: int | None = None,
) -> None:
    """Print summary and per-task final loss tables.

    % decrease = (initial_avg - final_avg) / initial_avg * 100.
    A positive value means loss went down. Absolute loss values differ between
    reductions by construction (sum and mean_active use different scales) —
    only % decrease is directly comparable.
    """
    epoch_label = f"epoch {n_epochs}" if n_epochs is not None else "final epoch"

    # --- Summary table ---
    hdr = (
        f"{'Reduction':<12} | {'Initial avg':>12} | {'Final avg':>10} | "
        f"{'Best avg':>10} | {'Best epoch':>10} | {'% decrease':>10}"
    )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    for reduction in ["sum", "mean_active"]:
        r = comparison.results[reduction]
        if r.initial_avg != 0.0:
            pct = (r.initial_avg - r.final_avg) / r.initial_avg * 100.0
            pct_str = f"{pct:>9.1f}%"
        else:
            pct_str = f"{'  nan':>10}"
        print(
            f"{reduction:<12} | {r.initial_avg:>12.6f} | {r.final_avg:>10.6f} | "
            f"{r.best_avg:>10.6f} | {r.best_epoch:>10d} | {pct_str}"
        )

    print()

    # --- Per-task final losses ---
    active_tasks = [
        t for t in (Task.REPETITION, Task.COMPREHENSION, Task.SPEAKING)
        if any(
            _task_avg(comparison.results[r], t) is not None
            for r in ["sum", "mean_active"]
        )
    ]

    if active_tasks:
        col_w = 10
        names = {Task.REPETITION: "REP", Task.COMPREHENSION: "COMP", Task.SPEAKING: "SPK"}
        task_cols = "  ".join(f"{names[t]:>{col_w}}" for t in active_tasks)
        print(f"Per-task final losses ({epoch_label}):")
        print(f"{'Reduction':<12} | {task_cols}")
        print("-" * (13 + (col_w + 2) * len(active_tasks)))
        for reduction in ["sum", "mean_active"]:
            vals = []
            for t in active_tasks:
                v = _task_avg(comparison.results[reduction], t)
                vals.append(f"{v:>{col_w}.6f}" if v is not None else f"{'---':>{col_w}}")
            print(f"{reduction:<12} | {'  '.join(vals)}")


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
        description="Phase 3c-7: Compare sum vs mean_active loss reductions on small subsets."
    )
    p.add_argument("--data-dir",          type=str,   default="data/raw/nwr_swp")
    p.add_argument("--config",            type=str,   default="configs/english_nwr.yaml")
    p.add_argument("--mode",              type=str,   default="mixed-multitask",
                   choices=["repetition", "words-multitask", "mixed-multitask"])
    p.add_argument("--max-words",         type=int,   default=10,
                   dest="max_words",
                   help="Max words to use. 0=none; positive integer=sample up to n. Default: 10.")
    p.add_argument("--max-pseudowords",   type=int,   default=10,
                   dest="max_pseudowords",
                   help="Max pseudowords to use. 0=none; positive integer=sample up to n. Default: 10.")
    p.add_argument("--epochs",            type=int,   default=20)
    p.add_argument("--lr",                type=float, default=0.01)
    p.add_argument("--device",            type=str,   default="cpu")
    p.add_argument("--seed",              type=int,   default=0)
    p.add_argument("--zero-error-radius", type=float, default=0.0,
                   dest="zero_error_radius",
                   help="0.0 = no dead-zone (default).")
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

    cfg = load_config(Path(args.config))

    data_dir = Path(args.data_dir)
    inventory = load_phoneme_inventory(data_dir / "phonemes.csv")

    # Load and sample items once — both reductions share the same subset.
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

    trials = build_trials_for_mode(
        args.mode, word_items, pseudo_items, sem_map, cfg.motor_output_size
    )

    if not trials:
        print(
            "Error: no trials were built. Check mode/max-words/max-pseudowords.",
            file=sys.stderr,
        )
        return 1

    n_word_trials   = sum(1 for t in trials if t.label and t.label.startswith("word:"))
    n_pseudo_trials = sum(1 for t in trials if t.label and t.label.startswith("pseudo:"))

    print("=== Loss Reduction Comparison ===")
    print(f"Mode: {args.mode}")
    if word_items:
        task_desc = "rep only" if args.mode == "repetition" else "rep+comp+spk"
        print(f"Words: {len(word_items)} ({n_word_trials} trials: {task_desc})")
    if pseudo_items:
        print(f"Pseudowords: {len(pseudo_items)} ({n_pseudo_trials} trials: rep only)")
    print(f"Total trials per epoch: {len(trials)}")
    print(
        f"Config: sound={cfg.sound_input_size}  "
        f"motor={cfg.motor_output_size}  vATL={cfg.vATL_size}"
    )
    print(
        f"lr={args.lr}  zero_error_radius={args.zero_error_radius}  "
        f"epochs={args.epochs}  seed={args.seed}"
    )
    print()

    try:
        comparison = run_comparison(
            cfg, trials,
            epochs=args.epochs,
            lr=args.lr,
            zero_error_radius=args.zero_error_radius,
            device=device,
            seed=args.seed,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_comparison_table(comparison, n_epochs=args.epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
