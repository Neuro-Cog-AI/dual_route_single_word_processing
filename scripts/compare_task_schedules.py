"""Phase 3c-8: Compare uniform vs paper-like task schedules on small subsets.

Compares two task presentation schedules for real words:
  uniform: 1×REP + 1×COMP + 1×SPK per word (3 trials/word)
  paper:   1×REP + 3×COMP + 2×SPK per word (6 trials/word) [Ueno et al. 2011]

Pseudowords always receive 1×REP regardless of schedule.

Fairness guarantee: items are loaded and sampled once; both schedules receive a
fresh model initialised with the same seed and the same trial shuffle order
(random.Random(seed)). Same epochs, lr, zero_error_radius, and loss_reduction.

Absolute losses differ between schedules because the paper schedule has more
trials per epoch. Compare % decrease and per-task losses, not raw totals.

Usage (from dual_route_single_word_processing/):
    PYTHONPATH=src python scripts/compare_task_schedules.py \\
        --data-dir data/raw/nwr_swp \\
        --config configs/english_nwr.yaml \\
        --mode mixed-multitask \\
        --max-words 10 --max-pseudowords 10 \\
        --epochs 20 --lr 0.01 --device cpu --seed 0 \\
        --loss-reduction mean_active

Exit code 0 if both runs complete with finite losses; 1 on error.
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
from lichtheim2.trials import (
    SupervisedTrial,
    make_comprehension_trial,
    make_repetition_trial,
    make_speaking_trial,
)

# Import helpers from earlier diagnostic scripts.
# Both are guarded by `if __name__ == "__main__"` and safe to import.
sys.path.insert(0, str(Path(__file__).parent))
from diagnose_small_subset_training import (
    DiagnosticResult,
    build_word_rep_trials,
    run_diagnostic_epochs,
    sample_items,
)
from train_multitask_real_data import build_pseudo_trials


# ---------------------------------------------------------------------------
# Schedule definitions
# ---------------------------------------------------------------------------

# Presentation counts per real word per epoch.
# Paper counts from Ueno et al. 2011: 1×REP, 2×SPK, 3×COMP [Paper/Supp].
# Within-word construction order: REP×n_rep, then COMP×n_comp, then SPK×n_spk.
# This order is deterministic and used by tests; actual presentation order is
# shuffled every epoch by run_diagnostic_epochs().
SCHEDULES: dict[str, dict[Task, int]] = {
    "uniform": {Task.REPETITION: 1, Task.COMPREHENSION: 1, Task.SPEAKING: 1},
    "paper":   {Task.REPETITION: 1, Task.COMPREHENSION: 3, Task.SPEAKING: 2},
}


# ---------------------------------------------------------------------------
# Comparison result
# ---------------------------------------------------------------------------


@dataclass
class ScheduleComparisonResult:
    """Results for both task schedules under identical training conditions."""

    results: dict[str, DiagnosticResult]  # keys: "uniform", "paper"
    trials_per_epoch: dict[str, int]      # schedule → n trials per epoch


# ---------------------------------------------------------------------------
# Trial builders
# ---------------------------------------------------------------------------


def build_word_trials_with_schedule(
    word_items: list[WordItem],
    sem_map: dict[int, torch.Tensor],
    motor_size: int,
    schedule: str,
) -> list[SupervisedTrial]:
    """Build trials for each real word according to the given schedule.

    Within-word construction order: REP×n_rep, then COMP×n_comp, then SPK×n_spk.
    This order is fixed for reproducibility and testing. The full trial list is
    shuffled every epoch by run_diagnostic_epochs(), so construction order does
    not bias training.

    Examples (one word):
      uniform → [REP, COMP, SPK]
      paper   → [REP, COMP, COMP, COMP, SPK, SPK]

    Args:
        word_items: real words (row_index must not be None)
        sem_map:    row_index → (vATL_size,) semantic tensor
        motor_size: ModelConfig.motor_output_size
        schedule:   key in SCHEDULES ("uniform" or "paper")

    Raises:
        ValueError: if schedule is unknown or any WordItem.row_index is None
    """
    if schedule not in SCHEDULES:
        raise ValueError(
            f"Unknown schedule: {schedule!r}; expected one of {sorted(SCHEDULES)}"
        )
    counts = SCHEDULES[schedule]
    trials: list[SupervisedTrial] = []
    for w in word_items:
        if w.row_index is None:
            raise ValueError(
                f"WordItem {w.word!r} has row_index=None; "
                "row_index must be set to look up the semantic vector"
            )
        label = f"word:{w.row_index}:{w.word}"
        sem   = sem_map[w.row_index]
        for _ in range(counts[Task.REPETITION]):
            trials.append(make_repetition_trial(
                w.phon_tensor, motor_size, item_id=w.row_index, label=label
            ))
        for _ in range(counts[Task.COMPREHENSION]):
            trials.append(make_comprehension_trial(
                w.phon_tensor, sem, motor_size, item_id=w.row_index, label=label
            ))
        for _ in range(counts[Task.SPEAKING]):
            trials.append(make_speaking_trial(
                w.phon_tensor, sem, motor_size, item_id=w.row_index, label=label
            ))
    return trials


def build_trials_for_schedule(
    mode: str,
    word_items: list[WordItem],
    pseudo_items: list[PseudowordItem],
    sem_map: dict[int, torch.Tensor],
    motor_size: int,
    schedule: str,
) -> list[SupervisedTrial]:
    """Build the full trial list for the given mode and schedule.

    The schedule only affects multitask word trials. In repetition mode,
    words always produce 1×REP regardless of schedule. Pseudowords always
    produce 1×REP in all modes and schedules.

    Args:
        mode:         "repetition" | "words-multitask" | "mixed-multitask"
        word_items:   real words (may be empty)
        pseudo_items: pseudowords (ignored for words-multitask)
        sem_map:      row_index → semantic tensor (required for multitask modes)
        motor_size:   ModelConfig.motor_output_size
        schedule:     "uniform" or "paper"
    """
    if mode == "repetition":
        return (
            build_word_rep_trials(word_items, motor_size)
            + build_pseudo_trials(pseudo_items, motor_size)
        )
    elif mode == "words-multitask":
        return build_word_trials_with_schedule(word_items, sem_map, motor_size, schedule)
    elif mode == "mixed-multitask":
        return (
            build_word_trials_with_schedule(word_items, sem_map, motor_size, schedule)
            + build_pseudo_trials(pseudo_items, motor_size)
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# Core comparison function
# ---------------------------------------------------------------------------


def run_schedule_comparison(
    cfg: ModelConfig,
    word_items: list[WordItem],
    pseudo_items: list[PseudowordItem],
    sem_map: dict[int, torch.Tensor],
    mode: str,
    epochs: int,
    lr: float,
    zero_error_radius: float,
    device: torch.device,
    seed: int,
    loss_reduction: str = "mean_active",
) -> ScheduleComparisonResult:
    """Run both schedules under identical training conditions.

    Fairness guarantee:
    - ``word_items``, ``pseudo_items``, and ``sem_map`` are pre-loaded and
      sampled once before this call; both schedules share the same items.
    - For each schedule: ``torch.manual_seed(seed)`` before model instantiation
      → identical weight initialization.
    - For each schedule: ``random.Random(seed)`` as ``rng`` → identical trial
      shuffle order every epoch.
    - Same ``epochs``, ``lr``, ``zero_error_radius``, ``device``, ``loss_reduction``.
    - Per-epoch logs suppressed (``verbose=False``).

    Raises:
        RuntimeError: if trial list is empty for any schedule, or propagated from
                      run_diagnostic_epochs if any loss is non-finite.
    """
    results: dict[str, DiagnosticResult] = {}
    trials_per_epoch: dict[str, int] = {}

    for schedule in ["uniform", "paper"]:
        trials = build_trials_for_schedule(
            mode, word_items, pseudo_items, sem_map, cfg.motor_output_size, schedule
        )
        if not trials:
            raise RuntimeError(
                f"No trials built for schedule '{schedule}'. "
                "Check mode/max-words/max-pseudowords."
            )
        trials_per_epoch[schedule] = len(trials)

        torch.manual_seed(seed)
        model     = Lichtheim2Model(cfg).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        rng       = random.Random(seed)

        results[schedule] = run_diagnostic_epochs(
            model, trials, optimizer, cfg,
            epochs=epochs,
            zero_error_radius=zero_error_radius,
            device=device,
            rng=rng,
            loss_reduction=loss_reduction,
            verbose=False,
        )

    return ScheduleComparisonResult(results=results, trials_per_epoch=trials_per_epoch)


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


def print_schedule_comparison_table(
    result: ScheduleComparisonResult,
    n_epochs: int | None = None,
) -> None:
    """Print summary and per-task (initial → final) tables.

    Absolute losses differ because the paper schedule has more trials per epoch.
    % decrease and per-task initial→final are the informative metrics.
    """
    epoch_label = f"epoch {n_epochs}" if n_epochs is not None else "final epoch"

    # --- Summary table ---
    hdr = (
        f"{'Schedule':<10} | {'Trials/ep':>9} | {'Initial avg':>12} | {'Final avg':>10} | "
        f"{'Best avg':>10} | {'Best ep':>7} | {'% decrease':>10}"
    )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    for schedule in ["uniform", "paper"]:
        r        = result.results[schedule]
        n_trials = result.trials_per_epoch[schedule]
        if r.initial_avg != 0.0:
            pct     = (r.initial_avg - r.final_avg) / r.initial_avg * 100.0
            pct_str = f"{pct:>9.1f}%"
        else:
            pct_str = f"{'  nan':>10}"
        print(
            f"{schedule:<10} | {n_trials:>9d} | {r.initial_avg:>12.6f} | {r.final_avg:>10.6f} | "
            f"{r.best_avg:>10.6f} | {r.best_epoch:>7d} | {pct_str}"
        )

    print()
    print(
        "Note: absolute losses differ between schedules (paper has more trials/epoch)."
        " Compare % decrease and per-task losses, not raw totals."
    )
    print()

    # --- Per-task losses: initial → final ---
    active_tasks = [
        t for t in (Task.REPETITION, Task.COMPREHENSION, Task.SPEAKING)
        if any(
            _task_avg_epoch(result.results[s], t, 0) is not None
            for s in ["uniform", "paper"]
        )
    ]
    if not active_tasks:
        return

    names = {Task.REPETITION: "REP", Task.COMPREHENSION: "COMP", Task.SPEAKING: "SPK"}
    col_w = 17  # "0.123456→0.098765" = 17 chars
    task_cols = "  ".join(f"{names[t]:>{col_w}}" for t in active_tasks)
    print(f"Per-task losses (epoch 1 → {epoch_label}):")
    print(f"{'Schedule':<10} | {task_cols}")
    print("-" * (11 + (col_w + 2) * len(active_tasks)))
    for schedule in ["uniform", "paper"]:
        r     = result.results[schedule]
        cells = []
        for t in active_tasks:
            init_v  = _task_avg_epoch(r, t,  0)
            final_v = _task_avg_epoch(r, t, -1)
            if init_v is not None and final_v is not None:
                cells.append(f"{init_v:.6f}→{final_v:.6f}")
            else:
                cells.append(f"{'---':>{col_w}}")
        print(f"{schedule:<10} | {'  '.join(cells)}")


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
        description="Phase 3c-8: Compare uniform vs paper task schedules on small subsets."
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
    p.add_argument("--loss-reduction",    type=str,   default="mean_active",
                   dest="loss_reduction",
                   choices=["sum", "mean_active"],
                   help="Loss reduction (default: mean_active — more interpretable for schedule comparison).")
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

    # Load and sample items once — both schedules share the same subset.
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

    print("=== Task Schedule Comparison ===")
    print(f"Mode: {args.mode}  |  loss_reduction: {args.loss_reduction}")
    if word_items:
        task_desc = "rep only" if args.mode == "repetition" else "rep+comp+spk"
        n_uni = len(word_items) * sum(SCHEDULES["uniform"].values())
        n_pap = len(word_items) * sum(SCHEDULES["paper"].values())
        print(
            f"Words: {len(word_items)} "
            f"(uniform: {n_uni} trials/ep, paper: {n_pap} trials/ep; {task_desc})"
        )
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
        comparison = run_schedule_comparison(
            cfg, word_items, pseudo_items, sem_map,
            mode=args.mode,
            epochs=args.epochs,
            lr=args.lr,
            zero_error_radius=args.zero_error_radius,
            device=device,
            seed=args.seed,
            loss_reduction=args.loss_reduction,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_schedule_comparison_table(comparison, n_epochs=args.epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
