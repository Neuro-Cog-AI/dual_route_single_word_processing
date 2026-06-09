from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

import torch

from lichtheim2.losses import compute_trial_loss
from lichtheim2.tasks import Task
from lichtheim2.trials import SupervisedTrial

if TYPE_CHECKING:
    from lichtheim2.config import ModelConfig
    from lichtheim2.model import Lichtheim2Model


def move_trial_to_device(
    trial: SupervisedTrial,
    device: torch.device | str,
) -> SupervisedTrial:
    """Return a new SupervisedTrial with all tensors moved to device."""
    dev = torch.device(device) if isinstance(device, str) else device
    return dc_replace(
        trial,
        phon_tensor=trial.phon_tensor.to(dev),
        sem_input=trial.sem_input.to(dev) if trial.sem_input is not None else None,
        motor_targets=trial.motor_targets.to(dev),
        motor_loss_mask=trial.motor_loss_mask.to(dev),
        semantic_targets=trial.semantic_targets.to(dev) if trial.semantic_targets is not None else None,
        semantic_loss_mask=trial.semantic_loss_mask.to(dev) if trial.semantic_loss_mask is not None else None,
    )


def train_step(
    model: "Lichtheim2Model",
    trial: SupervisedTrial,
    optimizer: torch.optim.Optimizer,
    cfg: "ModelConfig",
    zero_error_radius: float = 0.0,
    device: torch.device | str = "cpu",
    loss_reduction: str = "sum",
) -> float:
    """Perform one online (item-by-item) training step.

    Forward pass → loss → backward → optimizer step.

    Args:
        model:              Lichtheim2Model (should already be on device)
        trial:              SupervisedTrial (will be moved to device internally)
        optimizer:          any torch.optim.Optimizer
        cfg:                ModelConfig used to construct the model
        zero_error_radius:  dead-zone threshold for loss masking (default 0.0)
        device:             device to run on; tests use "cpu"
        loss_reduction:     "sum" (default) or "mean_active"; passed to
                            compute_trial_loss()

    Returns:
        Scalar loss as a Python float (detached, for logging).
    """
    dev = torch.device(device) if isinstance(device, str) else device
    model.train()
    optimizer.zero_grad()

    # Move trial tensors to device
    trial_dev = move_trial_to_device(trial, dev)

    # Build semantic input for run_trial.
    # run_trial always requires a sem_pattern argument.
    # For speaking: use trial.sem_input (the semantic vector to clamp).
    # For repetition/comprehension: pass zeros — build_trial_inputs never
    # clamps vATL for those tasks, so this value is never used.
    if trial_dev.task == Task.SPEAKING and trial_dev.sem_input is not None:
        sem_for_run = trial_dev.sem_input
    else:
        sem_for_run = torch.zeros(cfg.vATL_size, device=dev)

    # Forward pass
    tick_results = model.run_trial(
        trial_dev.task, trial_dev.phon_tensor, sem_for_run, cfg
    )

    # Compute loss
    loss = compute_trial_loss(
        tick_results, trial_dev,
        zero_error_radius=zero_error_radius,
        loss_reduction=loss_reduction,
    )

    # Apply frequency weight if non-default (avoids touching the computation graph for weight=1.0)
    if trial_dev.loss_weight != 1.0:
        loss = loss * trial_dev.loss_weight

    # Backward + update
    loss.backward()
    optimizer.step()

    return loss.item()
