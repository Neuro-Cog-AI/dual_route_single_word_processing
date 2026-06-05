"""Smoke test: train on a single synthetic repetition item and verify loss decreases.

Usage:
    PYTHONPATH=src python scripts/smoke_train_repetition.py [--steps N] [--device DEVICE]

This script requires no CSV files. It uses the toy config with synthetic phoneme
tensors and runs N online training steps, printing loss at each step.

Exit code 0 if final loss < initial loss; exit code 1 otherwise.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim

# Allow running from repo root with PYTHONPATH=src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lichtheim2.config import load_config
from lichtheim2.model import Lichtheim2Model
from lichtheim2.trainer import train_step
from lichtheim2.trials import make_repetition_trial


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test: single repetition item training")
    parser.add_argument("--steps",  type=int, default=20,    help="number of training steps")
    parser.add_argument("--device", type=str, default="cpu", help="device: cpu / cuda / mps")
    parser.add_argument("--lr",     type=float, default=0.1, help="SGD learning rate")
    parser.add_argument("--seed",   type=int, default=42,    help="random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg   = load_config(Path(__file__).parent.parent / "configs" / "toy.yaml")
    model = Lichtheim2Model(cfg).to(device)
    opt   = optim.SGD(model.parameters(), lr=args.lr)

    # Synthetic phoneme tensor — T=3 morae with toy sound_input_size
    phon  = torch.rand(3, cfg.sound_input_size)
    trial = make_repetition_trial(phon, cfg.motor_output_size)

    print(f"Smoke training: {args.steps} steps, device={args.device}, lr={args.lr}")
    print(f"Config: sound={cfg.sound_input_size}, motor={cfg.motor_output_size}, vATL={cfg.vATL_size}")

    losses = []
    for step in range(1, args.steps + 1):
        loss = train_step(model, trial, opt, cfg, device=device)
        losses.append(loss)
        print(f"  step {step:3d}: loss = {loss:.6f}")

    initial, final = losses[0], losses[-1]
    success = final < initial
    print(f"\nInitial loss: {initial:.6f}  →  Final loss: {final:.6f}")
    if success:
        print("✓ Loss decreased — smoke test passed.")
    else:
        print("✗ Loss did not decrease — smoke test failed.")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
