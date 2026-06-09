"""Phase 3c-9: Compare unweighted vs frequency-weighted training on small subsets.

Frequency weighting scales each trial's loss by a word-frequency-derived multiplier.
Weights are normalised so the mean weight across the sampled word set is 1.0.
Pseudowords always receive weight 1.0.

Supported frequency sources:
  zipf      -- uses WordItem.zipf_frequency (Zipf-scale, no transform; default)
  frequency -- uses log1p(WordItem.frequency) (handles raw counts and zero values)

Both are normalised with "mean_one": weight_i = value_i / mean(values).
Words with missing/zero frequency receive weight 1.0.
If no words have usable frequency data, all weights default to 1.0.

Fairness guarantee: items are sampled once; both conditions receive a fresh model
initialised with the same seed and the same trial shuffle order (random.Random(seed)).
Same schedule, epochs, lr, zero_error_radius, and loss_reduction.

Usage (from dual_route_single_word_processing/):
    PYTHONPATH=src python scripts/compare_frequency_weighting.py \\
        --data-dir data/raw/nwr_swp \\
        --config configs/english_nwr.yaml \\
        --mode mixed-multitask --schedule paper \\
        --frequency-source zipf \\
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
from dataclasses import dataclass, replace as dc_replace
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

# Import helpers from earlier scripts (all guarded by `if __name__ == "__main__"`).
sys.path.insert(0, str(Path(__file__).parent))
from diagnose_small_subset_training import (
    DiagnosticResult,
    run_diagnostic_epochs,
    sample_items,
)
from compare_task_schedules import SCHEDULES, build_trials_for_schedule


# ---------------------------------------------------------------------------
# Frequency weight computation
# ---------------------------------------------------------------------------


def compute_word_weights(
    word_items: list[WordItem],
    frequency_source: str = "zipf",
    normalization: str = "mean_one",
) -> dict[int, float]:
    """Compute normalised frequency weights for each word.

    For each word with a usable frequency value, weight_i = value_i / mean(values).
    Words with missing or non-positive frequency receive weight 1.0 (the neutral value).
    If no words have positive frequency data, all weights default to 1.0.

    Args:
        word_items:        real words (row_index must not be None for each)
        frequency_source:  "zipf"  → use WordItem.zipf_frequency as-is
                           "frequency" → apply math.log1p(WordItem.frequency)
        normalization:     "mean_one" — divide by mean of positive values

    Returns:
        dict mapping row_index → weight (float > 0)

    Raises:
        ValueError: if frequency_source or normalization is unknown
    """
    if frequency_source not in ("zipf", "frequency"):
        raise ValueError(
            f"Unknown frequency_source: {frequency_source!r}; expected 'zipf' or 'frequency'"
        )
    if normalization != "mean_one":
        raise ValueError(
            f"Unknown normalization: {normalization!r}; expected 'mean_one'"
        )

    # Collect transformed values per row_index
    raw: dict[int, float | None] = {}
    for w in word_items:
        if w.row_index is None:
            continue
        if frequency_source == "zipf":
            val = w.zipf_frequency
        else:
            val = math.log1p(w.frequency) if w.frequency is not None else None
        raw[w.row_index] = val

    # Only positive values contribute to the normalisation mean
    positive = [v for v in raw.values() if v is not None and v > 0]
    if not positive:
        # No usable frequency data: all words get neutral weight 1.0
        return {row_idx: 1.0 for row_idx in raw}

    mean_val = sum(positive) / len(positive)  # > 0 since positive is non-empty

    weight_map: dict[int, float] = {}
    for row_idx, val in raw.items():
        if val is None or val <= 0:
            weight_map[row_idx] = 1.0
        else:
            weight_map[row_idx] = val / mean_val
    return weight_map


def apply_weights_to_trials(
    trials: list[SupervisedTrial],
    weight_map: dict[int, float],
) -> list[SupervisedTrial]:
    """Return a new trial list with loss_weight set from weight_map.

    Only trials whose label starts with "word:" receive weights from weight_map.
    All other trials (pseudowords, unlabelled) keep loss_weight=1.0.
    This prevents incorrect weight assignment when word and pseudoword sources
    share overlapping row_index values.
    """
    result = []
    for t in trials:
        if t.label is not None and t.label.startswith("word:"):
            weight = weight_map.get(t.item_id, 1.0)
        else:
            weight = 1.0
        result.append(dc_replace(t, loss_weight=weight))
    return result


# ---------------------------------------------------------------------------
# Comparison result
# ---------------------------------------------------------------------------


@dataclass
class FrequencyComparisonResult:
    """Results for unweighted vs frequency-weighted training under identical conditions."""

    results: dict[str, DiagnosticResult]  # keys: "unweighted", "weighted"
    trials_per_epoch: int                  # same for both conditions
    weight_stats: dict[str, float]         # min, mean, max, n_words (of weight_map)


# ---------------------------------------------------------------------------
# Core comparison function
# ---------------------------------------------------------------------------


def run_frequency_comparison(
    cfg: ModelConfig,
    word_items: list[WordItem],
    pseudo_items: list[PseudowordItem],
    sem_map: dict[int, torch.Tensor],
    mode: str,
    schedule: str,
    epochs: int,
    lr: float,
    zero_error_radius: float,
    device: torch.device,
    seed: int,
    loss_reduction: str = "mean_active",
    frequency_source: str = "zipf",
    weight_normalization: str = "mean_one",
) -> FrequencyComparisonResult:
    """Run unweighted and frequency-weighted training under identical conditions.

    Fairness guarantee:
    - ``word_items``, ``pseudo_items``, and ``sem_map`` are pre-loaded and
      sampled once before this call.
    - Base trials are built once; both conditions use the same trial structure.
    - For each condition: ``torch.manual_seed(seed)`` before model instantiation
      and ``random.Random(seed)`` as ``rng`` → identical initialisation and shuffle.
    - Same ``schedule``, ``epochs``, ``lr``, ``zero_error_radius``, ``device``,
      ``loss_reduction``.
    - Per-epoch logs suppressed (``verbose=False``).

    Raises:
        RuntimeError: if trial list is empty, or propagated from run_diagnostic_epochs.
    """
    base_trials = build_trials_for_schedule(
        mode, word_items, pseudo_items, sem_map, cfg.motor_output_size, schedule
    )
    if not base_trials:
        raise RuntimeError(
            "No trials built. Check mode/max-words/max-pseudowords."
        )

    weight_map     = compute_word_weights(word_items, frequency_source, weight_normalization)
    weighted_trials = apply_weights_to_trials(base_trials, weight_map)

    # Build weight stats from the weight_map values
    if weight_map:
        wvals = list(weight_map.values())
        weight_stats: dict[str, float] = {
            "min":     min(wvals),
            "mean":    sum(wvals) / len(wvals),
            "max":     max(wvals),
            "n_words": float(len(wvals)),
        }
    else:
        weight_stats = {"min": 1.0, "mean": 1.0, "max": 1.0, "n_words": 0.0}

    condition_trials = {
        "unweighted": base_trials,
        "weighted":   weighted_trials,
    }

    results: dict[str, DiagnosticResult] = {}
    for condition, trials in condition_trials.items():
        torch.manual_seed(seed)
        model     = Lichtheim2Model(cfg).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        rng       = random.Random(seed)
        results[condition] = run_diagnostic_epochs(
            model, trials, optimizer, cfg,
            epochs=epochs,
            zero_error_radius=zero_error_radius,
            device=device,
            rng=rng,
            loss_reduction=loss_reduction,
            verbose=False,
        )

    return FrequencyComparisonResult(
        results=results,
        trials_per_epoch=len(base_trials),
        weight_stats=weight_stats,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _task_avg_epoch(
    result: DiagnosticResult,
    task: Task,
    epoch_idx: int,
) -> float | None:
    """Mean loss for task at the given epoch index (0=first, -1=last), or None."""
    epoch = result.epoch_losses[epoch_idx]
    vals  = epoch.get(task, [])
    return sum(vals) / len(vals) if vals else None


def print_frequency_comparison_table(
    result: FrequencyComparisonResult,
    n_epochs: int | None = None,
    frequency_source: str = "zipf",
    weight_normalization: str = "mean_one",
) -> None:
    """Print weight stats, summary table, and per-task initial→final losses.

    Note: the "weighted" initial avg may differ from "unweighted" because
    train_step returns the weighted loss value. Compare % decrease within each
    condition, not absolute loss values across conditions.
    """
    epoch_label = f"epoch {n_epochs}" if n_epochs is not None else "final epoch"

    # --- Weight stats ---
    ws = result.weight_stats
    print(
        f"Frequency weights ({frequency_source}, {weight_normalization}):  "
        f"n_words={int(ws['n_words'])}  "
        f"min={ws['min']:.3f}  mean={ws['mean']:.3f}  max={ws['max']:.3f}"
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
    for condition in ["unweighted", "weighted"]:
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
    print(
        "Note: absolute losses may differ between conditions because train_step "
        "records the weighted loss. Compare % decrease, not raw totals."
    )
    print()

    # --- Per-task losses: initial → final ---
    active_tasks = [
        t for t in (Task.REPETITION, Task.COMPREHENSION, Task.SPEAKING)
        if any(
            _task_avg_epoch(result.results[c], t, 0) is not None
            for c in ["unweighted", "weighted"]
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
    for condition in ["unweighted", "weighted"]:
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
        description="Phase 3c-9: Compare unweighted vs frequency-weighted training on small subsets."
    )
    p.add_argument("--data-dir",            type=str,   default="data/raw/nwr_swp")
    p.add_argument("--config",              type=str,   default="configs/english_nwr.yaml")
    p.add_argument("--mode",                type=str,   default="mixed-multitask",
                   choices=["repetition", "words-multitask", "mixed-multitask"])
    p.add_argument("--schedule",            type=str,   default="paper",
                   choices=list(SCHEDULES))
    p.add_argument("--max-words",           type=int,   default=10,
                   dest="max_words",
                   help="Max words to use. 0=none; positive integer=sample up to n. Default: 10.")
    p.add_argument("--max-pseudowords",     type=int,   default=10,
                   dest="max_pseudowords",
                   help="Max pseudowords to use. 0=none; positive integer=sample up to n. Default: 10.")
    p.add_argument("--epochs",              type=int,   default=20)
    p.add_argument("--lr",                  type=float, default=0.01)
    p.add_argument("--device",              type=str,   default="cpu")
    p.add_argument("--seed",                type=int,   default=0)
    p.add_argument("--zero-error-radius",   type=float, default=0.0,
                   dest="zero_error_radius",
                   help="0.0 = no dead-zone (default).")
    p.add_argument("--loss-reduction",      type=str,   default="mean_active",
                   dest="loss_reduction",
                   choices=["sum", "mean_active"],
                   help="Loss reduction (default: mean_active).")
    p.add_argument("--frequency-source",    type=str,   default="zipf",
                   dest="frequency_source",
                   choices=["zipf", "frequency"],
                   help="Frequency column to use: 'zipf' (default) or 'frequency' (raw, log1p-transformed).")
    p.add_argument("--weight-normalization", type=str,  default="mean_one",
                   dest="weight_normalization",
                   choices=["mean_one"],
                   help="Weight normalisation: 'mean_one' (default) divides by the mean.")
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

    # Load and sample items once — both conditions share the same subset.
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

    n_pseudo_trials = len(pseudo_items)
    n_word_trials   = len(word_items) * sum(SCHEDULES[args.schedule].values()) if word_items else 0

    print("=== Frequency Weighting Comparison ===")
    print(
        f"Mode: {args.mode}  |  schedule: {args.schedule}  |  "
        f"frequency_source: {args.frequency_source}  |  loss_reduction: {args.loss_reduction}"
    )
    if word_items:
        task_desc = "rep only" if args.mode == "repetition" else "rep+comp+spk"
        print(f"Words: {len(word_items)} ({n_word_trials} trials/ep; {task_desc})")
    if pseudo_items:
        print(f"Pseudowords: {len(pseudo_items)} ({n_pseudo_trials} trials: rep only)")
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
        comparison = run_frequency_comparison(
            cfg, word_items, pseudo_items, sem_map,
            mode=args.mode,
            schedule=args.schedule,
            epochs=args.epochs,
            lr=args.lr,
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

    print_frequency_comparison_table(
        comparison,
        n_epochs=args.epochs,
        frequency_source=args.frequency_source,
        weight_normalization=args.weight_normalization,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
