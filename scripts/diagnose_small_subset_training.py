"""Phase 3c-4: Small-subset training stability diagnostics.

Runs controlled training on tiny subsets and reports structured diagnostic
statistics (per-epoch per-task losses, initial/final/best epoch).

Modes:
  repetition       — rep trials for words + pseudowords; no semantics needed
  words-multitask  — rep+comp+spk for words only; ssp.csv not required
  mixed-multitask  — rep+comp+spk for words, rep for pseudowords

Usage (from dual_route_single_word_processing/):
    PYTHONPATH=src python scripts/diagnose_small_subset_training.py \\
        --data-dir data/raw/nwr_swp \\
        --config configs/english_nwr.yaml \\
        --mode mixed-multitask \\
        --max-words 10 --max-pseudowords 10 \\
        --epochs 20 --lr 0.01 --device cpu --seed 0

Exit code 0 if all losses are finite; 1 on error.
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
from lichtheim2.trainer import train_step
from lichtheim2.trials import SupervisedTrial, make_repetition_trial

# Import trial builders from the Phase 3c-3 script.
# Safe because train_multitask_real_data.py is guarded by
# `if __name__ == "__main__": main()` and does not execute on import.
sys.path.insert(0, str(Path(__file__).parent))
from train_multitask_real_data import build_pseudo_trials, build_word_trials


# ---------------------------------------------------------------------------
# Diagnostic result
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticResult:
    """Structured output from run_diagnostic_epochs()."""

    epoch_losses: list[dict[Task, list[float]]]  # [epoch_idx][task] → per-item losses
    epoch_avgs:   list[float]                     # avg over all tasks, one entry per epoch
    initial_avg:  float                           # epoch_avgs[0]
    final_avg:    float                           # epoch_avgs[-1]
    best_avg:     float                           # min(epoch_avgs)
    best_epoch:   int                             # 1-indexed epoch with best_avg


# ---------------------------------------------------------------------------
# Helper functions (importable by tests)
# ---------------------------------------------------------------------------


def sample_items(
    items: list,
    max_count: int | None,
    rng: random.Random,
) -> list:
    """Sample a subset of items.

    Args:
        items:     source list
        max_count: None  → return all items (shuffled)
                   0     → return empty list
                   n > 0 → return min(n, len(items)) items sampled without replacement
        rng:       seeded Random instance

    Returns:
        A new list (never modifies `items` in-place).
    """
    if max_count is None:
        result = list(items)
        rng.shuffle(result)
        return result
    if max_count == 0:
        return []
    if len(items) > max_count:
        return rng.sample(items, max_count)
    result = list(items)
    rng.shuffle(result)
    return result


def build_word_rep_trials(
    word_items: list[WordItem],
    motor_size: int,
) -> list[SupervisedTrial]:
    """Build one REPETITION trial per word item.

    Label format: ``word:{row_index}:{word}``.

    Raises:
        ValueError: if any WordItem has row_index is None.
    """
    trials: list[SupervisedTrial] = []
    for w in word_items:
        if w.row_index is None:
            raise ValueError(
                f"WordItem {w.word!r} has row_index=None; "
                "row_index must be set for label construction"
            )
        trials.append(make_repetition_trial(
            w.phon_tensor, motor_size,
            item_id=w.row_index,
            label=f"word:{w.row_index}:{w.word}",
        ))
    return trials


def build_trials_for_mode(
    mode: str,
    word_items: list[WordItem],
    pseudo_items: list[PseudowordItem],
    sem_map: dict[int, torch.Tensor],
    motor_size: int,
) -> list[SupervisedTrial]:
    """Build the full trial list for the given mode.

    Args:
        mode:         "repetition" | "words-multitask" | "mixed-multitask"
        word_items:   real words (may be empty)
        pseudo_items: pseudowords (may be empty; ignored for words-multitask)
        sem_map:      row_index → semantic tensor (required for multitask modes)
        motor_size:   from ModelConfig.motor_output_size

    Returns:
        Flat list of SupervisedTrial objects.
    """
    if mode == "repetition":
        return build_word_rep_trials(word_items, motor_size) + build_pseudo_trials(pseudo_items, motor_size)
    elif mode == "words-multitask":
        return build_word_trials(word_items, sem_map, motor_size)
    elif mode == "mixed-multitask":
        return build_word_trials(word_items, sem_map, motor_size) + build_pseudo_trials(pseudo_items, motor_size)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


def run_diagnostic_epochs(
    model: Lichtheim2Model,
    trials: list[SupervisedTrial],
    optimizer: torch.optim.Optimizer,
    cfg: ModelConfig,
    epochs: int,
    zero_error_radius: float,
    device: torch.device,
    rng: random.Random,
    loss_reduction: str = "sum",
    verbose: bool = True,
    lr_schedule_fn=None,
) -> DiagnosticResult:
    """Run diagnostic training loop and return structured statistics.

    Args:
        loss_reduction:  "sum" (default) or "mean_active"; passed to train_step().
        verbose:         if True (default), print a per-epoch summary line;
                         set to False for silent operation (e.g. comparison scripts).
        lr_schedule_fn:  optional callable(epoch: int) -> float, where epoch is
                         1-indexed.  If provided, all optimizer param_groups are
                         updated to the returned LR at the start of every epoch.
                         Default None → LR is never modified (existing behavior).

    Raises:
        RuntimeError: if any item loss is non-finite (nan or inf), with
                      epoch/task/label information for debugging.
    """
    all_epoch_losses: list[dict[Task, list[float]]] = []
    epoch_avgs: list[float] = []

    for epoch in range(1, epochs + 1):
        if lr_schedule_fn is not None:
            lr = lr_schedule_fn(epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr
        order = list(trials)
        rng.shuffle(order)
        epoch_losses: dict[Task, list[float]] = {t: [] for t in Task}

        for trial in order:
            loss = train_step(
                model, trial, optimizer, cfg,
                zero_error_radius=zero_error_radius,
                device=device,
                loss_reduction=loss_reduction,
            )
            if not math.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss ({loss}) at epoch {epoch}, "
                    f"task {trial.task.name}, label={trial.label!r}"
                )
            epoch_losses[trial.task].append(loss)

        all_flat = [l for lst in epoch_losses.values() for l in lst]
        avg = sum(all_flat) / len(all_flat)
        epoch_avgs.append(avg)

        line = f"Epoch {epoch:3d}: avg={avg:.6f}"
        for task in (Task.REPETITION, Task.COMPREHENSION, Task.SPEAKING):
            if epoch_losses[task]:
                avg_t = sum(epoch_losses[task]) / len(epoch_losses[task])
                line += f"  {task.name[:3]}={avg_t:.6f}"
        if verbose:
            print(line)

        all_epoch_losses.append(epoch_losses)

    best_avg   = min(epoch_avgs)
    best_epoch = epoch_avgs.index(best_avg) + 1  # 1-indexed

    return DiagnosticResult(
        epoch_losses=all_epoch_losses,
        epoch_avgs=epoch_avgs,
        initial_avg=epoch_avgs[0],
        final_avg=epoch_avgs[-1],
        best_avg=best_avg,
        best_epoch=best_epoch,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3c-4: Small-subset training stability diagnostics."
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
                   help="0.0 = no dead-zone (default, good for debugging).")
    p.add_argument("--loss-reduction",    type=str,   default="sum",
                   dest="loss_reduction",
                   choices=["sum", "mean_active"],
                   help="Loss reduction: 'sum' (default) or 'mean_active'.")
    return p.parse_args(argv)


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
    model = Lichtheim2Model(cfg).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    data_dir = Path(args.data_dir)
    inventory = load_phoneme_inventory(data_dir / "phonemes.csv")

    # Load words (needed for all modes, unless max_words == 0)
    word_items: list[WordItem] = []
    sem_map: dict[int, torch.Tensor] = {}
    if args.max_words != 0:
        word_items_all = load_word_items(data_dir / "wfe.csv", inventory)
        # Assign semantics to all words before sampling so each word's vector
        # is stable regardless of which subset is chosen.
        if args.mode in ("words-multitask", "mixed-multitask"):
            sem_map = assign_artificial_semantics(
                word_items_all, vATL_size=cfg.vATL_size, seed=args.seed
            )
        word_items = sample_items(word_items_all, args.max_words, rng)

    # Load pseudowords (only for modes that use them, unless max_pseudowords == 0)
    pseudo_items: list[PseudowordItem] = []
    if args.mode in ("repetition", "mixed-multitask") and args.max_pseudowords != 0:
        pseudo_items_all = load_pseudoword_items(data_dir / "ssp.csv", inventory)
        pseudo_items = sample_items(pseudo_items_all, args.max_pseudowords, rng)

    trials = build_trials_for_mode(args.mode, word_items, pseudo_items, sem_map, cfg.motor_output_size)

    if not trials:
        print(
            "Error: no trials were built. Check mode/max-words/max-pseudowords.",
            file=sys.stderr,
        )
        return 1

    # Count word and pseudo trials for informational print
    n_word_trials  = sum(1 for t in trials if t.label and t.label.startswith("word:"))
    n_pseudo_trials = sum(1 for t in trials if t.label and t.label.startswith("pseudo:"))

    print(f"Device: {device}")
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
    print(f"lr={args.lr}  zero_error_radius={args.zero_error_radius}  epochs={args.epochs}  loss_reduction={args.loss_reduction}")

    try:
        result = run_diagnostic_epochs(
            model, trials, optimizer, cfg,
            epochs=args.epochs,
            zero_error_radius=args.zero_error_radius,
            device=device,
            rng=rng,
            loss_reduction=args.loss_reduction,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\n--- Diagnostics ---")
    print(f"Initial avg loss (epoch  1): {result.initial_avg:.6f}")
    print(f"Final   avg loss (epoch {args.epochs:2d}): {result.final_avg:.6f}")
    print(f"Best    avg loss (epoch {result.best_epoch:2d}): {result.best_avg:.6f}")
    print(f"Loss decreased: {'yes' if result.final_avg < result.initial_avg else 'no'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
