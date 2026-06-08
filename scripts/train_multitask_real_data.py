"""Phase 3c-3: Online multi-task training on real English/NWR items.

Pipeline validation: real words receive repetition + comprehension + speaking
trials; pseudowords receive repetition-only trials. No batching, no LR
schedule, no frequency weighting, no accuracy logging, no checkpoints.

Usage (from dual_route_single_word_processing/):
    PYTHONPATH=src python scripts/train_multitask_real_data.py \\
        --data-dir data/raw/nwr_swp \\
        --config configs/english_nwr.yaml \\
        --max-words 50 --max-pseudowords 50 \\
        --epochs 3 --lr 0.1 --device cpu --seed 0

Exit code 0 if all losses are finite; 1 otherwise.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
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
from lichtheim2.trials import (
    SupervisedTrial,
    make_comprehension_trial,
    make_repetition_trial,
    make_speaking_trial,
)


# ---------------------------------------------------------------------------
# Helper functions (importable by tests without going through CLI)
# ---------------------------------------------------------------------------


def build_word_trials(
    word_items: list[WordItem],
    sem_map: dict[int, torch.Tensor],
    motor_size: int,
) -> list[SupervisedTrial]:
    """Build repetition, comprehension, and speaking trials for each real word.

    Each word produces three trials in order: REPETITION, COMPREHENSION, SPEAKING.
    Label format: ``word:{row_index}:{word}`` for traceability.

    Args:
        word_items: list of WordItem objects (row_index must not be None)
        sem_map:    dict mapping row_index → (vATL_size,) semantic tensor
        motor_size: motor output size from ModelConfig

    Returns:
        Flat list of SupervisedTrial objects (3 × len(word_items) entries).

    Raises:
        ValueError: if any WordItem has row_index is None
    """
    trials: list[SupervisedTrial] = []
    for w in word_items:
        if w.row_index is None:
            raise ValueError(
                f"WordItem {w.word!r} has row_index=None; "
                "row_index must be set to look up the semantic vector"
            )
        label = f"word:{w.row_index}:{w.word}"
        sem = sem_map[w.row_index]
        trials.append(make_repetition_trial(
            w.phon_tensor, motor_size, item_id=w.row_index, label=label
        ))
        trials.append(make_comprehension_trial(
            w.phon_tensor, sem, motor_size, item_id=w.row_index, label=label
        ))
        trials.append(make_speaking_trial(
            w.phon_tensor, sem, motor_size, item_id=w.row_index, label=label
        ))
    return trials


def build_pseudo_trials(
    pseudo_items: list[PseudowordItem],
    motor_size: int,
) -> list[SupervisedTrial]:
    """Build one repetition trial per pseudoword.

    Label format: ``pseudo:{row_index}``.
    """
    return [
        make_repetition_trial(
            p.phon_tensor, motor_size,
            item_id=p.row_index,
            label=f"pseudo:{p.row_index}",
        )
        for p in pseudo_items
    ]


def train_multitask_epochs(
    model: Lichtheim2Model,
    trials: list[SupervisedTrial],
    optimizer: torch.optim.Optimizer,
    cfg: ModelConfig,
    epochs: int,
    zero_error_radius: float,
    device: torch.device,
    rng: random.Random,
) -> list[dict[Task, list[float]]]:
    """Run online multi-task training, one trial at a time.

    Returns:
        A list (one per epoch) of dicts mapping Task → list of per-item losses
        for that task in that epoch. Trials are shuffled each epoch.
    """
    all_epoch_losses: list[dict[Task, list[float]]] = []
    for epoch in range(1, epochs + 1):
        order = list(trials)
        rng.shuffle(order)
        epoch_losses: dict[Task, list[float]] = {t: [] for t in Task}
        for trial in order:
            loss = train_step(
                model, trial, optimizer, cfg,
                zero_error_radius=zero_error_radius,
                device=device,
            )
            epoch_losses[trial.task].append(loss)

        all_losses_flat = [l for lst in epoch_losses.values() for l in lst]
        avg_all = sum(all_losses_flat) / len(all_losses_flat)
        line = f"Epoch {epoch:3d}: avg={avg_all:.6f}"
        for task in (Task.REPETITION, Task.COMPREHENSION, Task.SPEAKING):
            if epoch_losses[task]:
                avg_t = sum(epoch_losses[task]) / len(epoch_losses[task])
                line += f"  {task.name[:3]}={avg_t:.6f}"
        print(line)
        all_epoch_losses.append(epoch_losses)
    return all_epoch_losses


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3c-3: Online multi-task training on real English/NWR items."
    )
    p.add_argument("--data-dir",          type=str,   default="data/raw/nwr_swp")
    p.add_argument("--config",            type=str,   default="configs/english_nwr.yaml")
    p.add_argument("--max-words",         type=int,   default=50,
                   dest="max_words",
                   help="Max real words to sample (default 50; use a large number for all).")
    p.add_argument("--max-pseudowords",   type=int,   default=50,
                   dest="max_pseudowords",
                   help="Max pseudowords to sample (default 50).")
    p.add_argument("--epochs",            type=int,   default=3)
    p.add_argument("--lr",                type=float, default=0.1)
    p.add_argument("--device",            type=str,   default="cpu")
    p.add_argument("--seed",              type=int,   default=0)
    p.add_argument("--zero-error-radius", type=float, default=0.1,
                   dest="zero_error_radius",
                   help="Dead-zone threshold for loss masking (0.0 = disabled).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

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

    # Load all words, assign semantics before sampling (so each word's vector is stable)
    word_items_all = load_word_items(data_dir / "wfe.csv", inventory)
    sem_map = assign_artificial_semantics(
        word_items_all, vATL_size=cfg.vATL_size, seed=args.seed
    )
    if len(word_items_all) > args.max_words:
        word_items = rng.sample(word_items_all, args.max_words)
    else:
        word_items = list(word_items_all)

    # Load pseudowords and sample
    pseudo_items_all = load_pseudoword_items(data_dir / "ssp.csv", inventory)
    if len(pseudo_items_all) > args.max_pseudowords:
        pseudo_items = rng.sample(pseudo_items_all, args.max_pseudowords)
    else:
        pseudo_items = list(pseudo_items_all)

    word_trials   = build_word_trials(word_items, sem_map, cfg.motor_output_size)
    pseudo_trials = build_pseudo_trials(pseudo_items, cfg.motor_output_size)
    all_trials    = word_trials + pseudo_trials

    print(f"Device: {device}")
    print(f"Words: {len(word_items)} ({len(word_trials)} trials: rep+comp+spk)")
    print(f"Pseudowords: {len(pseudo_items)} ({len(pseudo_trials)} trials: rep only)")
    print(f"Total trials per epoch: {len(all_trials)}")
    print(
        f"Config: sound={cfg.sound_input_size}  "
        f"motor={cfg.motor_output_size}  vATL={cfg.vATL_size}"
    )
    print(f"zero_error_radius={args.zero_error_radius}  lr={args.lr}  epochs={args.epochs}")

    epoch_losses = train_multitask_epochs(
        model, all_trials, optimizer, cfg,
        epochs=args.epochs,
        zero_error_radius=args.zero_error_radius,
        device=device,
        rng=rng,
    )

    # Summary
    def _avg(epoch_dict: dict[Task, list[float]]) -> float:
        flat = [l for lst in epoch_dict.values() for l in lst]
        return sum(flat) / len(flat)

    initial_avg = _avg(epoch_losses[0])
    final_avg   = _avg(epoch_losses[-1])
    print(f"\nInitial avg_loss (epoch 1): {initial_avg:.6f}")
    print(f"Final   avg_loss (epoch {args.epochs}): {final_avg:.6f}")
    if final_avg < initial_avg:
        print("Loss decreased over training.")
    else:
        print("Loss did not decrease (may be normal for very short runs or high lr).")

    # Check all losses are finite
    all_flat = [l for epoch in epoch_losses for lst in epoch.values() for l in lst]
    if any(not math.isfinite(l) for l in all_flat):
        print("Error: non-finite loss encountered.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
