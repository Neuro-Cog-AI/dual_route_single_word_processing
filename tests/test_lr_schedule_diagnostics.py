"""Phase 3c-10 tests: LR schedule comparison utility.

Structure:
- Always-run (no CSV required): lr_for_epoch logic, toy comparison runs, validate_args.
- CSV-dependent: skip gracefully if private data absent.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import fields as dc_fields
from pathlib import Path

import pytest
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lichtheim2.config import load_config
from lichtheim2.data import PseudowordItem, WordItem
from lichtheim2.model import Lichtheim2Model
from lichtheim2.semantics import assign_artificial_semantics

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from compare_lr_schedules import (
    LRComparisonResult,
    lr_for_epoch,
    run_lr_comparison,
    validate_args,
)

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

REPO_ROOT    = Path(__file__).parent.parent
DATA_DIR     = REPO_ROOT / "data" / "raw" / "nwr_swp"
PHONEMES_CSV = DATA_DIR / "phonemes.csv"
WFE_CSV      = DATA_DIR / "wfe.csv"
SSP_CSV      = DATA_DIR / "ssp.csv"

cfg = load_config(REPO_ROOT / "configs" / "toy.yaml")


def _mock_word(word: str = "tank", row_index: int = 0, T: int = 3) -> WordItem:
    return WordItem(
        word=word,
        phonemes=["X"] * T,
        phon_tensor=torch.rand(T, cfg.sound_input_size),
        lexicality="real",
        length=T,
        frequency=None,
        zipf_frequency=None,
        part_of_speech=None,
        condition=None,
        morphology=None,
        source="wfe",
        row_index=row_index,
    )


def _mock_pseudo(row_index: int = 0, T: int = 3) -> PseudowordItem:
    return PseudowordItem(
        phonemes=["X"] * T,
        phon_tensor=torch.rand(T, cfg.sound_input_size),
        length=T,
        sonority=None,
        syllable_type=None,
        source="ssp",
        row_index=row_index,
    )


def _sem_map(word_items: list[WordItem]) -> dict[int, torch.Tensor]:
    return assign_artificial_semantics(word_items, vATL_size=cfg.vATL_size, seed=42)


# ---------------------------------------------------------------------------
# Always-run: lr_for_epoch — constant schedule
# ---------------------------------------------------------------------------


def test_lr_constant_always_base_lr():
    for epoch in [1, 5, 50, 100, 200]:
        assert lr_for_epoch(0.01, epoch, "constant", 200) == 0.01


def test_lr_constant_base_lr_preserved_at_small_epochs():
    for epoch in range(1, 21):
        assert lr_for_epoch(0.005, epoch, "constant", 20) == 0.005


# ---------------------------------------------------------------------------
# Always-run: lr_for_epoch — paper-proportional schedule (200 epochs)
# ---------------------------------------------------------------------------


def test_lr_paper_constant_phase_200():
    """All epochs in the 0–75% window return base_lr."""
    # Epoch 150 of 200 → fraction 0.75, boundary condition included in ≤0.75
    assert lr_for_epoch(0.5, 1,   "paper", 200) == pytest.approx(0.5)
    assert lr_for_epoch(0.5, 100, "paper", 200) == pytest.approx(0.5)
    assert lr_for_epoch(0.5, 150, "paper", 200) == pytest.approx(0.5)


def test_lr_paper_first_decay_step_200():
    """Epoch 155 of 200 → fraction 0.775, in (0.75, 0.80] → multiplier 0.8."""
    assert lr_for_epoch(0.5, 155, "paper", 200) == pytest.approx(0.4)


def test_lr_paper_second_decay_step_200():
    """Epoch 165 of 200 → fraction 0.825, in (0.80, 0.85] → multiplier 0.6."""
    assert lr_for_epoch(0.5, 165, "paper", 200) == pytest.approx(0.3)


def test_lr_paper_third_decay_step_200():
    """Epoch 175 of 200 → fraction 0.875, in (0.85, 0.90] → multiplier 0.4."""
    assert lr_for_epoch(0.5, 175, "paper", 200) == pytest.approx(0.2)


def test_lr_paper_minimum_phase_200():
    """Epoch 190 of 200 → fraction 0.95, > 0.90 → multiplier 0.2."""
    assert lr_for_epoch(0.5, 190, "paper", 200) == pytest.approx(0.1)
    assert lr_for_epoch(0.5, 200, "paper", 200) == pytest.approx(0.1)


def test_lr_paper_exact_200_epochs():
    """For --epochs 200 --lr 0.5, reproduces the exact paper schedule values."""
    assert lr_for_epoch(0.5, 1,   "paper", 200) == pytest.approx(0.5)
    assert lr_for_epoch(0.5, 155, "paper", 200) == pytest.approx(0.4)
    assert lr_for_epoch(0.5, 165, "paper", 200) == pytest.approx(0.3)
    assert lr_for_epoch(0.5, 175, "paper", 200) == pytest.approx(0.2)
    assert lr_for_epoch(0.5, 190, "paper", 200) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Always-run: lr_for_epoch — paper-proportional schedule (20 epochs)
# ---------------------------------------------------------------------------


def test_lr_paper_proportional_20_epochs_constant_phase():
    """Epochs 1–15 of 20 (fraction ≤ 0.75) → full base_lr."""
    for epoch in range(1, 16):
        assert lr_for_epoch(0.01, epoch, "paper", 20) == pytest.approx(0.01), (
            f"Expected full LR at epoch {epoch}/20"
        )


def test_lr_paper_proportional_20_epochs_decay():
    """Paper-proportional decay at 20 epochs."""
    # epoch 16/20 → frac=0.80 → first step boundary exactly: ≤0.80 → mult 0.8
    assert lr_for_epoch(0.01, 16, "paper", 20) == pytest.approx(0.008)
    # epoch 17/20 → frac=0.85 → ≤0.85 → mult 0.6
    assert lr_for_epoch(0.01, 17, "paper", 20) == pytest.approx(0.006)
    # epoch 18/20 → frac=0.90 → ≤0.90 → mult 0.4
    assert lr_for_epoch(0.01, 18, "paper", 20) == pytest.approx(0.004)
    # epochs 19-20/20 → frac > 0.90 → mult 0.2
    assert lr_for_epoch(0.01, 19, "paper", 20) == pytest.approx(0.002)
    assert lr_for_epoch(0.01, 20, "paper", 20) == pytest.approx(0.002)


def test_lr_paper_proportional_minimum_is_20pct():
    """Final LR is always 20% of base_lr regardless of epoch count."""
    for total in [10, 20, 100, 200]:
        final = lr_for_epoch(0.5, total, "paper", total)
        assert final == pytest.approx(0.5 * 0.2), (
            f"Expected 0.1 at epoch {total}/{total}, got {final}"
        )


# ---------------------------------------------------------------------------
# Always-run: lr_for_epoch — invalid schedule
# ---------------------------------------------------------------------------


def test_lr_for_epoch_invalid_schedule_raises():
    with pytest.raises(ValueError, match="lr_schedule"):
        lr_for_epoch(0.01, 1, "invalid_schedule", 20)


# ---------------------------------------------------------------------------
# Always-run: run_lr_comparison — small toy run
# ---------------------------------------------------------------------------


def _tiny_comparison(seed: int = 0, epochs: int = 4) -> LRComparisonResult:
    words   = [_mock_word("tank", 0), _mock_word("cat", 1)]
    pseudos = [_mock_pseudo(0), _mock_pseudo(1)]
    sem     = _sem_map(words)
    return run_lr_comparison(
        cfg, words, pseudos, sem,
        mode="mixed-multitask",
        task_schedule="paper",
        epochs=epochs,
        base_lr=0.01,
        zero_error_radius=0.0,
        device=torch.device("cpu"),
        seed=seed,
        loss_reduction="sum",
        frequency_source="none",
    )


def test_run_lr_comparison_returns_both_conditions():
    result = _tiny_comparison()
    assert set(result.results.keys()) == {"constant", "paper"}


def test_run_lr_comparison_result_fields():
    result = _tiny_comparison()
    for r in result.results.values():
        assert hasattr(r, "initial_avg")
        assert hasattr(r, "final_avg")
        assert hasattr(r, "best_avg")
        assert hasattr(r, "best_epoch")
        assert hasattr(r, "epoch_avgs")
        assert hasattr(r, "epoch_losses")


def test_run_lr_comparison_epoch_counts_match():
    result = _tiny_comparison(epochs=5)
    counts = [len(r.epoch_avgs) for r in result.results.values()]
    assert len(set(counts)) == 1
    assert counts[0] == 5


def test_run_lr_comparison_losses_finite():
    result = _tiny_comparison()
    for condition, r in result.results.items():
        assert all(math.isfinite(a) for a in r.epoch_avgs), (
            f"non-finite epoch avg for condition {condition!r}"
        )


def test_lr_comparison_result_has_lr_schedule():
    result = _tiny_comparison(epochs=4)
    assert "constant" in result.lr_schedule_used
    assert "paper"    in result.lr_schedule_used
    assert len(result.lr_schedule_used["constant"]) == 4
    assert len(result.lr_schedule_used["paper"])    == 4


def test_lr_comparison_constant_schedule_all_same():
    """All LR values in the constant condition are equal."""
    result = _tiny_comparison(epochs=10)
    lrs = result.lr_schedule_used["constant"]
    assert all(abs(lr - lrs[0]) < 1e-12 for lr in lrs)


def test_lr_comparison_paper_schedule_decays():
    """Paper-proportional schedule must decay at some point over enough epochs."""
    result = _tiny_comparison(epochs=20)
    lrs = result.lr_schedule_used["paper"]
    assert lrs[-1] < lrs[0], (
        f"Expected paper LR to decrease over 20 epochs; got {lrs[0]:.6f} → {lrs[-1]:.6f}"
    )


def test_lr_comparison_result_is_dataclass():
    fnames = {f.name for f in dc_fields(LRComparisonResult)}
    assert "results"          in fnames
    assert "trials_per_epoch" in fnames
    assert "lr_schedule_used" in fnames


def test_lr_comparison_with_frequency_source_zipf():
    """Comparison with frequency_source='zipf' completes with finite losses."""
    words = [
        _mock_word("tank", 0),
        _mock_word("cat",  1),
    ]
    # Give words non-None zipf so weighting is non-trivial
    from dataclasses import replace as dc_replace
    words = [dc_replace(w, zipf_frequency=float(i + 2)) for i, w in enumerate(words)]
    pseudos = [_mock_pseudo(0)]
    sem     = _sem_map(words)
    result  = run_lr_comparison(
        cfg, words, pseudos, sem,
        mode="mixed-multitask",
        task_schedule="uniform",
        epochs=2,
        base_lr=0.01,
        zero_error_radius=0.0,
        device=torch.device("cpu"),
        seed=0,
        loss_reduction="sum",
        frequency_source="zipf",
    )
    for condition, r in result.results.items():
        assert all(math.isfinite(a) for a in r.epoch_avgs), (
            f"non-finite epoch avg for {condition}"
        )


# ---------------------------------------------------------------------------
# Always-run: validate_args
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    base = argparse.Namespace(
        max_words=10, max_pseudowords=10, epochs=20,
        lr=0.01, zero_error_radius=0.0,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_validate_args_valid():
    assert validate_args(_make_args()) is None


@pytest.mark.parametrize("field,value", [
    ("max_words",         -1),
    ("max_pseudowords",   -1),
    ("epochs",             0),
    ("lr",                0.0),
    ("zero_error_radius", -0.1),
])
def test_validate_args_invalid(field, value):
    err = validate_args(_make_args(**{field: value}))
    assert err is not None
    assert isinstance(err, str)


# ---------------------------------------------------------------------------
# CSV-dependent integration tests
# ---------------------------------------------------------------------------

_all_csv_available = pytest.mark.skipif(
    not (PHONEMES_CSV.exists() and WFE_CSV.exists() and SSP_CSV.exists()),
    reason="private CSV data (phonemes.csv + wfe.csv + ssp.csv) not present",
)


@_all_csv_available
def test_csv_lr_comparison_finite():
    """Real English NWR data: both LR conditions give finite losses."""
    from lichtheim2.config import load_config as _lc
    from lichtheim2.data import load_pseudoword_items, load_word_items
    from lichtheim2.encoding import load_phoneme_inventory
    eng_cfg   = _lc(REPO_ROOT / "configs" / "english_nwr.yaml")
    inventory = load_phoneme_inventory(PHONEMES_CSV)
    words_all = load_word_items(WFE_CSV, inventory)
    pseudo_all = load_pseudoword_items(SSP_CSV, inventory)
    sem_map   = assign_artificial_semantics(words_all, vATL_size=eng_cfg.vATL_size, seed=42)

    words   = words_all[:5]
    pseudos = pseudo_all[:5]
    result  = run_lr_comparison(
        eng_cfg, words, pseudos, sem_map,
        mode="mixed-multitask",
        task_schedule="paper",
        epochs=3,
        base_lr=0.01,
        zero_error_radius=0.0,
        device=torch.device("cpu"),
        seed=0,
        loss_reduction="mean_active",
        frequency_source="none",
    )
    for condition, r in result.results.items():
        assert all(math.isfinite(a) for a in r.epoch_avgs), (
            f"non-finite epoch avg for condition {condition!r}"
        )
