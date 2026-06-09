from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from lichtheim2.tasks import Task

if TYPE_CHECKING:
    from lichtheim2.layers import TickResult
    from lichtheim2.trials import SupervisedTrial


@dataclass
class LossBreakdown:
    """Per-component loss breakdown for one supervised trial.

    "Active" output elements are those contributing to loss:
      (1) the tick's loss mask is True, AND
      (2) if zero_error_radius > 0, |prediction − target| >= zero_error_radius.

    This applies to ALL output units at a masked tick, regardless of whether
    their target value is 0 or 1. Even zero-target units contribute to BCE
    unless excluded by the dead zone.
    """

    task:                   Task
    n_ticks:                int
    total_loss:             torch.Tensor  # scalar, grad-capable
    motor_loss:             torch.Tensor  # scalar, grad-capable
    semantic_loss:          torch.Tensor  # scalar, grad-capable; zero for REP/SPK
    n_active_motor:         int           # active (tick, unit) pairs for motor loss
    n_active_semantic:      int           # active (tick, unit) pairs for semantic; 0 for REP/SPK
    motor_loss_per_unit:    float         # motor_loss / n_active_motor; nan if 0
    semantic_loss_per_unit: float         # semantic_loss / n_active_semantic; nan if 0


def compute_trial_loss_breakdown(
    tick_results: list["TickResult"],
    trial: "SupervisedTrial",
    zero_error_radius: float = 0.0,
) -> LossBreakdown:
    """Compute per-component loss breakdown for one trial.

    Returns a LossBreakdown with separate motor and semantic loss tensors,
    active element counts, and normalized per-unit losses. Does NOT use
    torch.no_grad() internally — gradient flows back to model parameters.

    "Active" means: the tick's loss mask is True AND (if zero_error_radius > 0)
    the unit is outside the dead zone (|prediction − target| >= radius).
    This applies to all output units at a masked tick regardless of target
    value — zero-target units count as active if unmasked.

    Args:
        tick_results:       list of TickResult from model.run_trial()
        trial:              SupervisedTrial with targets and masks
        zero_error_radius:  dead-zone threshold; 0.0 disables it

    Returns:
        LossBreakdown whose total_loss is numerically identical to
        compute_trial_loss(...) called with the same arguments.
    """
    # -- Motor loss --
    motor_outputs = torch.stack(
        [r.state.motor for r in tick_results], dim=0
    )  # (n_ticks, motor_size)

    raw_motor = F.binary_cross_entropy(
        motor_outputs, trial.motor_targets, reduction="none"
    )  # (n_ticks, motor_size)

    # Alive mask: tick mask broadcast to (n_ticks, motor_size), optionally
    # intersected with ~dead_zone. expand_as creates a non-contiguous view;
    # the & operator creates a new contiguous tensor when radius > 0.
    alive_motor = trial.motor_loss_mask.unsqueeze(-1).expand_as(raw_motor)  # bool
    if zero_error_radius > 0.0:
        dead_motor  = (motor_outputs.detach() - trial.motor_targets).abs() < zero_error_radius
        alive_motor = alive_motor & ~dead_motor

    motor_loss     = (raw_motor * alive_motor.float()).sum()
    n_active_motor = int(alive_motor.sum().item())

    # -- Semantic loss (comprehension only; None for repetition / speaking) --
    if trial.semantic_targets is not None and trial.semantic_loss_mask is not None:
        vATL_outputs = torch.stack(
            [r.state.vATL_out for r in tick_results], dim=0
        )  # (n_ticks, vATL_size)

        raw_sem = F.binary_cross_entropy(
            vATL_outputs, trial.semantic_targets, reduction="none"
        )  # (n_ticks, vATL_size)

        alive_sem = trial.semantic_loss_mask.unsqueeze(-1).expand_as(raw_sem)  # bool
        if zero_error_radius > 0.0:
            dead_sem  = (vATL_outputs.detach() - trial.semantic_targets).abs() < zero_error_radius
            alive_sem = alive_sem & ~dead_sem

        semantic_loss     = (raw_sem * alive_sem.float()).sum()
        n_active_semantic = int(alive_sem.sum().item())
    else:
        semantic_loss     = motor_loss.new_zeros(())  # device-safe zero scalar
        n_active_semantic = 0

    total_loss = motor_loss + semantic_loss

    motor_loss_per_unit    = motor_loss.item()    / n_active_motor    if n_active_motor    > 0 else float("nan")
    semantic_loss_per_unit = semantic_loss.item() / n_active_semantic if n_active_semantic > 0 else float("nan")

    return LossBreakdown(
        task=trial.task,
        n_ticks=trial.n_ticks,
        total_loss=total_loss,
        motor_loss=motor_loss,
        semantic_loss=semantic_loss,
        n_active_motor=n_active_motor,
        n_active_semantic=n_active_semantic,
        motor_loss_per_unit=motor_loss_per_unit,
        semantic_loss_per_unit=semantic_loss_per_unit,
    )


def compute_trial_loss(
    tick_results: list["TickResult"],
    trial: "SupervisedTrial",
    zero_error_radius: float = 0.0,
    loss_reduction: str = "sum",
) -> torch.Tensor:
    """Compute combined motor + semantic binary cross-entropy loss for one trial.

    Loss is computed only at ticks where the corresponding loss mask is True.
    Gradient flows back through tick_results (model outputs from run_trial()).

    Motor loss:
      BCE between model motor activations and trial.motor_targets at masked ticks.
    Semantic loss (if trial.semantic_targets is not None):
      BCE between vATL outputs and trial.semantic_targets at masked ticks.

    Zero-error radius:
      If zero_error_radius > 0, units where |prediction − target| < zero_error_radius
      contribute zero loss (no gradient). Applied consistently to both motor and
      semantic losses. Default 0.0 = standard BCE with no dead zone.
      From the paper: zero_error_radius = 0.1 [Paper].

    Loss reduction:
      "sum" (default): returns the total summed BCE loss. This preserves the
        existing training behavior exactly and is the paper default.
      "mean_active": divides total_loss by (n_active_motor + n_active_semantic),
        the count of output elements that actually contributed to the loss after
        applying the tick mask and dead zone. Intended for diagnostic comparisons
        across tasks — NOT yet adopted as the default training objective.
      Any other value raises ValueError.
      Raises ValueError if loss_reduction="mean_active" and all elements are
      masked or in the dead zone (n_active == 0).

    Args:
        tick_results:       list of TickResult from model.run_trial()
        trial:              SupervisedTrial with targets and masks
        zero_error_radius:  dead-zone threshold; 0.0 disables it
        loss_reduction:     "sum" (default) or "mean_active"

    Returns:
        Scalar tensor with .requires_grad=True (gradient flows back to model params).
    """
    breakdown = compute_trial_loss_breakdown(tick_results, trial, zero_error_radius)
    if loss_reduction == "sum":
        return breakdown.total_loss
    elif loss_reduction == "mean_active":
        n_active = breakdown.n_active_motor + breakdown.n_active_semantic
        if n_active == 0:
            raise ValueError(
                "loss_reduction='mean_active': n_active_motor + n_active_semantic == 0; "
                "cannot normalize — all output elements are masked or in the dead zone"
            )
        return breakdown.total_loss / n_active
    else:
        raise ValueError(
            f"Unknown loss_reduction: {loss_reduction!r}; expected 'sum' or 'mean_active'"
        )
