"""Phase 3c-2: Online repetition training on real English/NWR items.

Pipeline validation: CSV → WordItem/PseudowordItem → repetition SupervisedTrial
→ train_step → loss log. Repetition task only; no comprehension, speaking,
batching, EOS, semantic vectors, checkpoints, or plots.

Usage (from dual_route_single_word_processing/):
    PYTHONPATH=src python scripts/train_repetition_real_data.py \\
        --data-dir data/raw/nwr_swp \\
        --config configs/english_nwr.yaml \\
        --source words --max-items 50 --epochs 3 --lr 0.1 --device cpu --seed 0

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
from lichtheim2.data import load_pseudoword_items, load_word_items
from lichtheim2.encoding import load_phoneme_inventory
from lichtheim2.model import Lichtheim2Model
from lichtheim2.trainer import train_step
from lichtheim2.trials import SupervisedTrial, make_repetition_trial


# ---------------------------------------------------------------------------
# Helper functions (importable by tests without going through CLI)
# ---------------------------------------------------------------------------


def load_repetition_items(
    data_dir: Path,
    source: str,
) -> list[tuple[torch.Tensor, str, int | None]]:
    """Load word and/or pseudoword items and return (phon_tensor, label, item_id) tuples.

    Args:
        data_dir: directory containing phonemes.csv, wfe.csv, ssp.csv
        source:   "words" | "pseudowords" | "mixed"
                  For "mixed": pool all items then sample (sampling done by caller).

    Returns:
        List of (phon_tensor, label, item_id) tuples.
        Labels always include source prefix, e.g. "word:attending" or "pseudo:123",
        so word and pseudoword row indices never collide.
    """
    inventory = load_phoneme_inventory(data_dir / "phonemes.csv")
    items: list[tuple[torch.Tensor, str, int | None]] = []
    if source in ("words", "mixed"):
        for w in load_word_items(data_dir / "wfe.csv", inventory):
            items.append((w.phon_tensor, f"word:{w.word}", w.row_index))
    if source in ("pseudowords", "mixed"):
        for p in load_pseudoword_items(data_dir / "ssp.csv", inventory):
            items.append((p.phon_tensor, f"pseudo:{p.row_index}", p.row_index))
    return items


def sample_items(
    items: list[tuple[torch.Tensor, str, int | None]],
    max_items: int | None,
    rng: random.Random,
) -> list[tuple[torch.Tensor, str, int | None]]:
    """Return a random subset of items of size min(max_items, len(items)).

    If max_items is None or >= len(items), returns a shuffled copy of the full list.
    """
    if max_items is not None and len(items) > max_items:
        return rng.sample(items, max_items)
    result = list(items)
    rng.shuffle(result)
    return result


def build_repetition_trials(
    items: list[tuple[torch.Tensor, str, int | None]],
    motor_size: int,
) -> list[SupervisedTrial]:
    """Build one repetition SupervisedTrial per item, preserving item_id and label."""
    return [
        make_repetition_trial(phon_tensor, motor_size, item_id=item_id, label=label)
        for phon_tensor, label, item_id in items
    ]


def train_repetition_epochs(
    model: Lichtheim2Model,
    trials: list[SupervisedTrial],
    optimizer: torch.optim.Optimizer,
    cfg: ModelConfig,
    epochs: int,
    zero_error_radius: float,
    device: torch.device,
    rng: random.Random,
) -> list[list[float]]:
    """Run the online training loop.

    Returns:
        epoch_losses: epoch_losses[epoch_index] is a list of per-item losses for that epoch.
    """
    all_epoch_losses: list[list[float]] = []
    for epoch in range(1, epochs + 1):
        order = list(trials)
        rng.shuffle(order)
        epoch_losses: list[float] = []
        for trial in order:
            loss = train_step(
                model, trial, optimizer, cfg,
                zero_error_radius=zero_error_radius,
                device=device,
            )
            epoch_losses.append(loss)
        avg = sum(epoch_losses) / len(epoch_losses)
        print(
            f"Epoch {epoch:3d}: avg_loss={avg:.6f}  "
            f"min={min(epoch_losses):.6f}  max={max(epoch_losses):.6f}"
        )
        all_epoch_losses.append(epoch_losses)
    return all_epoch_losses


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3c-2: Online repetition training on real English/NWR items."
    )
    p.add_argument("--data-dir",           type=str,   default="data/raw/nwr_swp")
    p.add_argument("--config",             type=str,   default="configs/english_nwr.yaml")
    p.add_argument("--source",             type=str,   default="words",
                   choices=["words", "pseudowords", "mixed"])
    p.add_argument("--max-items",          type=int,   default=None)
    p.add_argument("--epochs",             type=int,   default=3)
    p.add_argument("--lr",                 type=float, default=0.1)
    p.add_argument("--device",             type=str,   default="cpu")
    p.add_argument("--seed",               type=int,   default=0)
    p.add_argument("--zero-error-radius",  type=float, default=0.1,
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

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    model = Lichtheim2Model(cfg).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    data_dir = Path(args.data_dir)
    items = load_repetition_items(data_dir, args.source)
    items = sample_items(items, args.max_items, rng)
    trials = build_repetition_trials(items, cfg.motor_output_size)

    print(f"Device: {device}")
    print(f"Source: {args.source}")
    print(f"Items loaded: {len(trials)}")
    print(
        f"Config: sound={cfg.sound_input_size}  "
        f"motor={cfg.motor_output_size}  vATL={cfg.vATL_size}"
    )
    print(f"zero_error_radius={args.zero_error_radius}  lr={args.lr}  epochs={args.epochs}")

    all_losses = train_repetition_epochs(
        model, trials, optimizer, cfg,
        epochs=args.epochs,
        zero_error_radius=args.zero_error_radius,
        device=device,
        rng=rng,
    )

    # Summary
    initial_avg = sum(all_losses[0]) / len(all_losses[0])
    final_avg   = sum(all_losses[-1]) / len(all_losses[-1])
    print(f"\nInitial avg_loss (epoch 1): {initial_avg:.6f}")
    print(f"Final   avg_loss (epoch {args.epochs}): {final_avg:.6f}")
    if final_avg < initial_avg:
        print("Loss decreased over training.")
    else:
        print("Loss did not decrease (may be normal for very short runs or high lr).")

    # Check all losses are finite
    all_flat = [l for epoch in all_losses for l in epoch]
    if any(not math.isfinite(l) for l in all_flat):
        print("Error: non-finite loss encountered.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())