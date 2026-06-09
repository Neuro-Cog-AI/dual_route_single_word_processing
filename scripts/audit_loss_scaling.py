"""Phase 3c-5: Loss normalization and task-balance audit.

Runs one forward pass per trial (repetition, comprehension, speaking) for a
small set of real words and reports a breakdown of motor vs semantic loss,
active element counts, and normalized per-unit losses.

Purpose: diagnose why comprehension loss is larger than other tasks.
Is it more active target units, higher dimensionality, or genuinely higher
per-unit BCE?

No training occurs — the model runs in eval mode under torch.no_grad().

Usage (from dual_route_single_word_processing/):
    PYTHONPATH=src python scripts/audit_loss_scaling.py \\
        --data-dir data/raw/nwr_swp \\
        --config configs/english_nwr.yaml \\
        --max-words 5 --seed 0 --device cpu

Exit code 0 on success; 1 on configuration error.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lichtheim2.config import ModelConfig, load_config
from lichtheim2.data import load_word_items
from lichtheim2.encoding import load_phoneme_inventory
from lichtheim2.losses import LossBreakdown, compute_trial_loss_breakdown
from lichtheim2.model import Lichtheim2Model
from lichtheim2.semantics import assign_artificial_semantics
from lichtheim2.tasks import Task
from lichtheim2.trainer import move_trial_to_device
from lichtheim2.trials import SupervisedTrial

# Import trial builder from Phase 3c-3 script (safe — guarded by __main__).
sys.path.insert(0, str(Path(__file__).parent))
from train_multitask_real_data import build_word_trials


# ---------------------------------------------------------------------------
# Core audit function (importable by tests)
# ---------------------------------------------------------------------------


def audit_trial(
    model: Lichtheim2Model,
    trial: SupervisedTrial,
    cfg: ModelConfig,
    zero_error_radius: float,
    device: torch.device,
) -> LossBreakdown:
    """Forward pass (eval mode, no_grad) and return LossBreakdown.

    compute_trial_loss_breakdown() does not call no_grad() internally, so
    gradient-compatibility is preserved for callers that need it. This
    wrapper uses no_grad() because the audit does not require gradients.
    """
    trial_dev   = move_trial_to_device(trial, device)
    sem_for_run = (
        trial_dev.sem_input
        if trial_dev.task == Task.SPEAKING and trial_dev.sem_input is not None
        else torch.zeros(cfg.vATL_size, device=device)
    )
    model.eval()
    with torch.no_grad():
        tick_results = model.run_trial(trial_dev.task, trial_dev.phon_tensor, sem_for_run, cfg)
        return compute_trial_loss_breakdown(tick_results, trial_dev, zero_error_radius)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt(value: float, fmt: str = ".5f") -> str:
    return f"{value:{fmt}}" if not math.isnan(value) else "-"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3c-5: Loss normalization and task-balance audit."
    )
    p.add_argument("--data-dir",          type=str,   default="data/raw/nwr_swp")
    p.add_argument("--config",            type=str,   default="configs/english_nwr.yaml")
    p.add_argument("--max-words",         type=int,   default=5,
                   dest="max_words",
                   help="Number of words to audit. Default: 5.")
    p.add_argument("--seed",              type=int,   default=0)
    p.add_argument("--device",            type=str,   default="cpu")
    p.add_argument("--zero-error-radius", type=float, default=0.0,
                   dest="zero_error_radius",
                   help="Dead-zone threshold (default 0.0 = disabled).")
    return p.parse_args(argv)


def validate_args(args: argparse.Namespace) -> str | None:
    """Return an error message string if args are invalid, else None."""
    if args.max_words < 1:
        return f"--max-words must be >= 1, got {args.max_words}"
    if args.zero_error_radius < 0:
        return f"--zero-error-radius must be >= 0, got {args.zero_error_radius}"
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    err = validate_args(args)
    if err is not None:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Error: --device cuda requested but CUDA is not available.", file=sys.stderr)
        return 1
    if device.type == "mps" and not torch.backends.mps.is_available():
        print("Error: --device mps requested but MPS is not available.", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    cfg   = load_config(Path(args.config))
    model = Lichtheim2Model(cfg).to(device)

    data_dir  = Path(args.data_dir)
    inventory = load_phoneme_inventory(data_dir / "phonemes.csv")

    word_items_all = load_word_items(data_dir / "wfe.csv", inventory)
    sem_map        = assign_artificial_semantics(
        word_items_all, vATL_size=cfg.vATL_size, seed=args.seed
    )
    if len(word_items_all) > args.max_words:
        word_items = rng.sample(word_items_all, args.max_words)
    else:
        word_items = list(word_items_all)

    trials = build_word_trials(word_items, sem_map, cfg.motor_output_size)

    print("Phase 3c-5: Loss normalization audit")
    print(
        f"Config: sound={cfg.sound_input_size}  motor={cfg.motor_output_size}  vATL={cfg.vATL_size}"
    )
    print(
        f"Device: {device}  |  Words: {len(word_items)}  |  "
        f"Trials: {len(trials)}  |  zero_error_radius={args.zero_error_radius}"
    )
    print()

    # Table header
    H = (
        f"{'label':<22} {'task':<13} {'n_ticks':>7}  "
        f"{'act_mot':>7}  {'act_sem':>7}  "
        f"{'motor_loss':>10}  {'sem_loss':>10}  {'total_loss':>10}  "
        f"{'mot/unit':>9}  {'sem/unit':>9}"
    )
    print(H)
    print("-" * len(H))

    task_bds: dict[Task, list[LossBreakdown]] = defaultdict(list)

    for trial in trials:
        bd    = audit_trial(model, trial, cfg, args.zero_error_radius, device)
        label = trial.label or ""
        task_bds[bd.task].append(bd)
        print(
            f"{label:<22} {bd.task.name:<13} {bd.n_ticks:>7}  "
            f"{bd.n_active_motor:>7}  {bd.n_active_semantic:>7}  "
            f"{bd.motor_loss.item():>10.4f}  {bd.semantic_loss.item():>10.4f}  "
            f"{bd.total_loss.item():>10.4f}  "
            f"{_fmt(bd.motor_loss_per_unit, '.5f'):>9}  "
            f"{_fmt(bd.semantic_loss_per_unit, '.5f'):>9}"
        )

    print()
    n_words = len(word_items)
    print(f"--- Per-task summary (mean over {n_words} word{'s' if n_words != 1 else ''}) ---")

    for task in (Task.REPETITION, Task.COMPREHENSION, Task.SPEAKING):
        bds = task_bds[task]
        if not bds:
            continue
        n         = len(bds)
        avg_total = sum(bd.total_loss.item() for bd in bds) / n
        mot_vals  = [bd.motor_loss_per_unit for bd in bds if not math.isnan(bd.motor_loss_per_unit)]
        avg_mot_u = sum(mot_vals) / len(mot_vals) if mot_vals else float("nan")

        line = f"{task.name:<13}  avg_total={avg_total:.4f}  avg_mot/unit={_fmt(avg_mot_u, '.5f')}"

        sem_vals = [bd.semantic_loss_per_unit for bd in bds if not math.isnan(bd.semantic_loss_per_unit)]
        if sem_vals:
            avg_sem_u = sum(sem_vals) / len(sem_vals)
            line += f"  avg_sem/unit={_fmt(avg_sem_u, '.5f')}"
        print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
