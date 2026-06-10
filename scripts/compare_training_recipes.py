"""Phase 3c-11: Compare integrated training recipes on small subsets.

Phases 3c-7 through 3c-10 each varied one training-loop dimension in isolation
(loss reduction, task schedule, frequency weighting, LR schedule). This script
compares a small set of *integrated recipes* -- fixed combinations of those four
dimensions -- under identical conditions, reusing the helpers from those phases.
No new training mechanics are introduced.

Recipes (RECIPES dict):
  baseline_constant  -- task_schedule=paper, loss_reduction=mean_active,
                         frequency_source=none,      lr_schedule=constant
  frequency_constant -- task_schedule=paper, loss_reduction=mean_active,
                         frequency_source=frequency, lr_schedule=constant
  frequency_paper_lr -- task_schedule=paper, loss_reduction=mean_active,
                         frequency_source=frequency, lr_schedule=paper
  zipf_constant      -- (optional control) task_schedule=paper,
                         loss_reduction=mean_active, frequency_source=zipf,
                         lr_schedule=constant

The default --recipes selection runs the three core recipes; zipf_constant is an
optional control and must be requested explicitly via --recipes.

Fairness guarantee: word/pseudoword items and semantics are loaded and sampled once.
Base trials are built once per distinct task_schedule (cache); frequency weight maps
are built once per distinct frequency_source (cache). For each recipe, a fresh model
is created with torch.manual_seed(seed) and a fresh random.Random(seed) governs the
trial shuffle order. Only the recipe's own settings vary.

Usage (from dual_route_single_word_processing/):
    PYTHONPATH=src python scripts/compare_training_recipes.py \\
        --data-dir data/raw/nwr_swp \\
        --config configs/english_nwr.yaml \\
        --mode mixed-multitask \\
        --recipes baseline_constant frequency_constant frequency_paper_lr \\
        --max-words 10 --max-pseudowords 10 \\
        --epochs 20 --lr 0.01 --device cpu --seed 0

Exit code 0 if all selected recipes complete with finite losses; 1 on error.
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

sys.path.insert(0, str(Path(__file__).parent))
from diagnose_small_subset_training import (
    DiagnosticResult,
    run_diagnostic_epochs,
    sample_items,
)
from compare_frequency_weighting import apply_weights_to_trials, compute_word_weights
from compare_task_schedules import SCHEDULES, build_trials_for_schedule
from compare_lr_schedules import lr_for_epoch


# ---------------------------------------------------------------------------
# Recipe definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecipeConfig:
    """One integrated training recipe: a fixed combination of the four
    diagnostic dimensions explored in Phases 3c-8 through 3c-10."""

    name: str
    task_schedule: str       # "uniform" | "paper"             (compare_task_schedules.SCHEDULES)
    loss_reduction: str      # "sum" | "mean_active"            (losses.compute_trial_loss)
    frequency_source: str    # "none" | "zipf" | "frequency"    (compare_frequency_weighting)
    lr_schedule: str         # "constant" | "paper"             (compare_lr_schedules.lr_for_epoch)


RECIPES: dict[str, RecipeConfig] = {
    "baseline_constant": RecipeConfig(
        name="baseline_constant",
        task_schedule="paper",
        loss_reduction="mean_active",
        frequency_source="none",
        lr_schedule="constant",
    ),
    "frequency_constant": RecipeConfig(
        name="frequency_constant",
        task_schedule="paper",
        loss_reduction="mean_active",
        frequency_source="frequency",
        lr_schedule="constant",
    ),
    "frequency_paper_lr": RecipeConfig(
        name="frequency_paper_lr",
        task_schedule="paper",
        loss_reduction="mean_active",
        frequency_source="frequency",
        lr_schedule="paper",
    ),
    # Optional control: zipf-based weighting instead of raw-frequency weighting.
    # Not part of the default --recipes selection.
    "zipf_constant": RecipeConfig(
        name="zipf_constant",
        task_schedule="paper",
        loss_reduction="mean_active",
        frequency_source="zipf",
        lr_schedule="constant",
    ),
}

DEFAULT_RECIPES: list[str] = ["baseline_constant", "frequency_constant", "frequency_paper_lr"]


def validate_recipe_config(recipe: RecipeConfig) -> str | None:
    """Return an error message string if recipe is invalid, else None."""
    if recipe.task_schedule not in SCHEDULES:
        return (
            f"Unknown task_schedule: {recipe.task_schedule!r}; "
            f"expected one of {sorted(SCHEDULES)}"
        )
    if recipe.loss_reduction not in ("sum", "mean_active"):
        return (
            f"Unknown loss_reduction: {recipe.loss_reduction!r}; "
            "expected 'sum' or 'mean_active'"
        )
    if recipe.frequency_source not in ("none", "zipf", "frequency"):
        return (
            f"Unknown frequency_source: {recipe.frequency_source!r}; "
            "expected 'none', 'zipf', or 'frequency'"
        )
    if recipe.lr_schedule not in ("constant", "paper"):
        return (
            f"Unknown lr_schedule: {recipe.lr_schedule!r}; "
            "expected 'constant' or 'paper'"
        )
    return None


# ---------------------------------------------------------------------------
# Comparison result
# ---------------------------------------------------------------------------


@dataclass
class RecipeComparisonResult:
    """Results for a set of integrated training recipes under shared sampling."""

    results: dict[str, DiagnosticResult]        # keyed by recipe name
    trials_per_epoch: dict[str, int]            # recipe name -> n trials/epoch
    recipes: dict[str, RecipeConfig]            # recipe name -> config used
    weight_stats: dict[str, dict[str, float]]   # recipe name -> {min,mean,max,n_words}
                                                 # neutral {1,1,1,0} if frequency_source="none"


# ---------------------------------------------------------------------------
# Core comparison function
# ---------------------------------------------------------------------------


def run_recipe_comparison(
    cfg: ModelConfig,
    word_items: list[WordItem],
    pseudo_items: list[PseudowordItem],
    sem_map: dict[int, torch.Tensor],
    mode: str,
    recipes: dict[str, RecipeConfig],
    epochs: int,
    base_lr: float,
    zero_error_radius: float,
    device: torch.device,
    seed: int,
    weight_normalization: str = "mean_one",
) -> RecipeComparisonResult:
    """Run each recipe under identical sampling/initialisation conditions.

    Fairness guarantee:
    - ``word_items``, ``pseudo_items``, and ``sem_map`` are pre-loaded and sampled
      once before this call.
    - Base trials are built once per distinct ``task_schedule`` (cache) via
      ``build_trials_for_schedule``; recipe-specific frequency weighting is then
      applied on top of those shared base trials.
    - Frequency weight maps are built once per distinct ``frequency_source`` (cache)
      via ``compute_word_weights``.
    - For each recipe: ``torch.manual_seed(seed)`` before model instantiation and
      ``random.Random(seed)`` as ``rng`` -> identical initialisation and shuffle
      order. Only the recipe's own settings (task_schedule, loss_reduction,
      frequency_source, lr_schedule) vary.

    Raises:
        ValueError: if any recipe in ``recipes`` is invalid (see
            ``validate_recipe_config``); message includes the recipe name.
        RuntimeError: if a cached base-trial list is empty, or propagated from
            ``run_diagnostic_epochs`` if any loss is non-finite.
    """
    for name, recipe in recipes.items():
        err = validate_recipe_config(recipe)
        if err is not None:
            raise ValueError(f"Invalid recipe {name!r}: {err}")

    # Base trials, cached per distinct task_schedule.
    base_trials_cache: dict[str, list[SupervisedTrial]] = {}
    for recipe in recipes.values():
        if recipe.task_schedule not in base_trials_cache:
            trials = build_trials_for_schedule(
                mode, word_items, pseudo_items, sem_map, cfg.motor_output_size,
                recipe.task_schedule,
            )
            if not trials:
                raise RuntimeError(
                    f"No trials built for task_schedule={recipe.task_schedule!r}. "
                    "Check mode/max-words/max-pseudowords."
                )
            base_trials_cache[recipe.task_schedule] = trials

    # Frequency weight maps, cached per distinct frequency_source.
    weight_map_cache: dict[str, dict[int, float]] = {}
    for recipe in recipes.values():
        if recipe.frequency_source != "none" and recipe.frequency_source not in weight_map_cache:
            weight_map_cache[recipe.frequency_source] = compute_word_weights(
                word_items,
                frequency_source=recipe.frequency_source,
                normalization=weight_normalization,
            )

    results: dict[str, DiagnosticResult] = {}
    trials_per_epoch: dict[str, int] = {}
    weight_stats: dict[str, dict[str, float]] = {}

    for name, recipe in recipes.items():
        base_trials = base_trials_cache[recipe.task_schedule]

        if recipe.frequency_source != "none":
            weight_map = weight_map_cache[recipe.frequency_source]
            trials = apply_weights_to_trials(base_trials, weight_map)
            if weight_map:
                wvals = list(weight_map.values())
                weight_stats[name] = {
                    "min":     min(wvals),
                    "mean":    sum(wvals) / len(wvals),
                    "max":     max(wvals),
                    "n_words": float(len(wvals)),
                }
            else:
                weight_stats[name] = {"min": 1.0, "mean": 1.0, "max": 1.0, "n_words": 0.0}
        else:
            trials = base_trials
            weight_stats[name] = {"min": 1.0, "mean": 1.0, "max": 1.0, "n_words": 0.0}

        trials_per_epoch[name] = len(trials)

        torch.manual_seed(seed)
        model     = Lichtheim2Model(cfg).to(device)
        optimizer = optim.SGD(model.parameters(), lr=base_lr)
        rng       = random.Random(seed)

        schedule_fn = lambda ep, r=recipe: lr_for_epoch(base_lr, ep, r.lr_schedule, epochs)

        results[name] = run_diagnostic_epochs(
            model, trials, optimizer, cfg,
            epochs=epochs,
            zero_error_radius=zero_error_radius,
            device=device,
            rng=rng,
            loss_reduction=recipe.loss_reduction,
            verbose=False,
            lr_schedule_fn=schedule_fn,
        )

    return RecipeComparisonResult(
        results=results,
        trials_per_epoch=trials_per_epoch,
        recipes=dict(recipes),
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


def print_recipe_comparison_table(
    result: RecipeComparisonResult,
    n_epochs: int | None = None,
) -> None:
    """Print weight stats, summary table, and final per-task training losses."""
    epoch_label = f"epoch {n_epochs}" if n_epochs is not None else "final epoch"

    # --- Weight stats for frequency-weighted recipes ---
    any_weighted = False
    for name, recipe in result.recipes.items():
        if recipe.frequency_source != "none":
            any_weighted = True
            ws = result.weight_stats[name]
            print(
                f"Frequency weights ({name}, {recipe.frequency_source}):  "
                f"n_words={int(ws['n_words'])}  min={ws['min']:.3f}  "
                f"mean={ws['mean']:.3f}  max={ws['max']:.3f}"
            )
    if any_weighted:
        print()

    # --- Summary table ---
    hdr = (
        f"{'Recipe':<20} | {'Task Sched':>10} | {'Freq Src':>8} | {'LR Sched':>8} | "
        f"{'Trials/ep':>9} | {'Initial avg':>12} | {'Final avg':>10} | "
        f"{'Best avg':>10} | {'Best ep':>7} | {'% decrease':>10}"
    )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    for name, recipe in result.recipes.items():
        r        = result.results[name]
        n_trials = result.trials_per_epoch[name]
        if r.initial_avg != 0.0:
            pct     = (r.initial_avg - r.final_avg) / r.initial_avg * 100.0
            pct_str = f"{pct:>9.1f}%"
        else:
            pct_str = f"{'  nan':>10}"
        print(
            f"{name:<20} | {recipe.task_schedule:>10} | {recipe.frequency_source:>8} | "
            f"{recipe.lr_schedule:>8} | {n_trials:>9d} | {r.initial_avg:>12.6f} | "
            f"{r.final_avg:>10.6f} | {r.best_avg:>10.6f} | {r.best_epoch:>7d} | {pct_str}"
        )

    print()
    print(
        "Note: absolute losses may differ across recipes with different "
        "frequency_source or trials/epoch. Compare % decrease, not raw totals."
    )
    print()

    # --- Final per-task training losses ---
    active_tasks = [
        t for t in (Task.REPETITION, Task.COMPREHENSION, Task.SPEAKING)
        if any(_task_avg_epoch(result.results[name], t, -1) is not None for name in result.recipes)
    ]
    if not active_tasks:
        return

    names = {Task.REPETITION: "REP", Task.COMPREHENSION: "COMP", Task.SPEAKING: "SPK"}
    col_w = 10
    task_cols = "  ".join(f"{names[t]:>{col_w}}" for t in active_tasks)
    print(f"Final per-task training losses ({epoch_label}):")
    print(f"{'Recipe':<20} | {task_cols}")
    print("-" * (21 + (col_w + 2) * len(active_tasks)))
    for name in result.recipes:
        r     = result.results[name]
        cells = []
        for t in active_tasks:
            v = _task_avg_epoch(r, t, -1)
            if v is not None:
                cells.append(f"{v:>{col_w}.6f}")
            else:
                cells.append(f"{'---':>{col_w}}")
        print(f"{name:<20} | {'  '.join(cells)}")

    print()
    print(
        "Note: per-task losses are recorded training losses (the value returned by "
        "train_step()), not a separate evaluation pass. For frequency-weighted "
        "recipes, reported losses are weighted training losses; unweighted "
        "evaluation metrics will be added later."
    )


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
        description="Phase 3c-11: Compare integrated training recipes on small subsets."
    )
    p.add_argument("--data-dir",            type=str,   default="data/raw/nwr_swp")
    p.add_argument("--config",              type=str,   default="configs/english_nwr.yaml")
    p.add_argument("--mode",                type=str,   default="mixed-multitask",
                   choices=["repetition", "words-multitask", "mixed-multitask"])
    p.add_argument("--recipes",             type=str,   nargs="+",
                   default=list(DEFAULT_RECIPES),
                   choices=list(RECIPES),
                   help=(
                       "Recipes to compare (default: the three core recipes "
                       f"{DEFAULT_RECIPES}). 'zipf_constant' is an optional "
                       "control and must be requested explicitly."
                   ))
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
    p.add_argument("--weight-normalization", type=str,  default="mean_one",
                   dest="weight_normalization",
                   choices=["mean_one"],
                   help="Weight normalisation for frequency-weighted recipes (default: mean_one).")
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

    recipes = {name: RECIPES[name] for name in args.recipes}

    print("=== Training Recipe Comparison ===")
    print(f"Mode: {args.mode}  |  recipes: {', '.join(recipes)}")
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
        comparison = run_recipe_comparison(
            cfg, word_items, pseudo_items, sem_map,
            mode=args.mode,
            recipes=recipes,
            epochs=args.epochs,
            base_lr=args.lr,
            zero_error_radius=args.zero_error_radius,
            device=device,
            seed=args.seed,
            weight_normalization=args.weight_normalization,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_recipe_comparison_table(comparison, n_epochs=args.epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
