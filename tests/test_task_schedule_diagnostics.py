"""Phase 3c-8 tests: task schedule comparison utility.

Structure:
- Always-run (no CSV required): toy config + mock WordItem/PseudowordItem dataclasses.
- CSV-dependent: skip gracefully per specific file.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import fields as dc_fields
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lichtheim2.config import load_config
from lichtheim2.data import PseudowordItem, WordItem
from lichtheim2.semantics import assign_artificial_semantics
from lichtheim2.tasks import Task

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from compare_task_schedules import (
    SCHEDULES,
    ScheduleComparisonResult,
    build_trials_for_schedule,
    build_word_trials_with_schedule,
    run_schedule_comparison,
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
# Always-run: SCHEDULES dict sanity
# ---------------------------------------------------------------------------


def test_schedules_has_uniform_and_paper():
    assert "uniform" in SCHEDULES
    assert "paper" in SCHEDULES


def test_paper_has_more_trials_than_uniform():
    assert sum(SCHEDULES["paper"].values()) > sum(SCHEDULES["uniform"].values())


def test_uniform_counts():
    assert SCHEDULES["uniform"][Task.REPETITION]    == 1
    assert SCHEDULES["uniform"][Task.COMPREHENSION] == 1
    assert SCHEDULES["uniform"][Task.SPEAKING]      == 1


def test_paper_counts():
    assert SCHEDULES["paper"][Task.REPETITION]    == 1
    assert SCHEDULES["paper"][Task.COMPREHENSION] == 3
    assert SCHEDULES["paper"][Task.SPEAKING]      == 2


# ---------------------------------------------------------------------------
# Always-run: build_word_trials_with_schedule
# ---------------------------------------------------------------------------


def test_uniform_word_trials_count():
    words = [_mock_word("tank", 0), _mock_word("cat", 1)]
    sem   = _sem_map(words)
    trials = build_word_trials_with_schedule(words, sem, cfg.motor_output_size, "uniform")
    assert len(trials) == 6  # 3 trials × 2 words


def test_paper_word_trials_count():
    words = [_mock_word("tank", 0), _mock_word("cat", 1)]
    sem   = _sem_map(words)
    trials = build_word_trials_with_schedule(words, sem, cfg.motor_output_size, "paper")
    assert len(trials) == 12  # 6 trials × 2 words


def test_uniform_word_trial_tasks():
    words = [_mock_word("tank", 0)]
    sem   = _sem_map(words)
    trials = build_word_trials_with_schedule(words, sem, cfg.motor_output_size, "uniform")
    tasks = [t.task for t in trials]
    assert tasks.count(Task.REPETITION)    == 1
    assert tasks.count(Task.COMPREHENSION) == 1
    assert tasks.count(Task.SPEAKING)      == 1


def test_paper_word_trial_tasks():
    words = [_mock_word("tank", 0)]
    sem   = _sem_map(words)
    trials = build_word_trials_with_schedule(words, sem, cfg.motor_output_size, "paper")
    tasks = [t.task for t in trials]
    assert tasks.count(Task.REPETITION)    == 1
    assert tasks.count(Task.COMPREHENSION) == 3
    assert tasks.count(Task.SPEAKING)      == 2


def test_paper_trial_order():
    """Construction order for one word: REP, COMP, COMP, COMP, SPK, SPK."""
    words = [_mock_word("tank", 0)]
    sem   = _sem_map(words)
    trials = build_word_trials_with_schedule(words, sem, cfg.motor_output_size, "paper")
    expected = [
        Task.REPETITION,
        Task.COMPREHENSION, Task.COMPREHENSION, Task.COMPREHENSION,
        Task.SPEAKING, Task.SPEAKING,
    ]
    assert [t.task for t in trials] == expected


def test_build_trials_invalid_schedule_raises():
    words = [_mock_word("tank", 0)]
    sem   = _sem_map(words)
    with pytest.raises(ValueError, match="schedule"):
        build_word_trials_with_schedule(words, sem, cfg.motor_output_size, "invalid")


# ---------------------------------------------------------------------------
# Always-run: build_trials_for_schedule (mode-aware)
# ---------------------------------------------------------------------------


def test_build_trials_mixed_uniform_count():
    words   = [_mock_word("tank", 0), _mock_word("cat", 1)]
    pseudos = [_mock_pseudo(0), _mock_pseudo(1)]
    sem     = _sem_map(words)
    trials  = build_trials_for_schedule(
        "mixed-multitask", words, pseudos, sem, cfg.motor_output_size, "uniform"
    )
    assert len(trials) == 8  # 3×2 word + 2 pseudo


def test_build_trials_mixed_paper_count():
    words   = [_mock_word("tank", 0), _mock_word("cat", 1)]
    pseudos = [_mock_pseudo(0), _mock_pseudo(1)]
    sem     = _sem_map(words)
    trials  = build_trials_for_schedule(
        "mixed-multitask", words, pseudos, sem, cfg.motor_output_size, "paper"
    )
    assert len(trials) == 14  # 6×2 word + 2 pseudo


def test_pseudo_trials_are_repetition_only():
    """Pseudoword trials are always REP regardless of schedule."""
    pseudos = [_mock_pseudo(0), _mock_pseudo(1)]
    for schedule in ["uniform", "paper"]:
        trials = build_trials_for_schedule(
            "repetition", [], pseudos, {}, cfg.motor_output_size, schedule
        )
        assert all(t.task == Task.REPETITION for t in trials)


def test_repetition_mode_ignores_schedule():
    """In repetition mode, schedule has no effect on trial count."""
    words   = [_mock_word("tank", 0)]
    pseudos = [_mock_pseudo(0)]
    trials_uni = build_trials_for_schedule(
        "repetition", words, pseudos, {}, cfg.motor_output_size, "uniform"
    )
    trials_pap = build_trials_for_schedule(
        "repetition", words, pseudos, {}, cfg.motor_output_size, "paper"
    )
    assert len(trials_uni) == len(trials_pap) == 2


# ---------------------------------------------------------------------------
# Always-run: run_schedule_comparison
# ---------------------------------------------------------------------------


def _tiny_comparison(seed: int = 0, epochs: int = 2) -> ScheduleComparisonResult:
    words   = [_mock_word("tank", 0), _mock_word("cat", 1)]
    pseudos = [_mock_pseudo(0), _mock_pseudo(1)]
    sem     = _sem_map(words)
    return run_schedule_comparison(
        cfg, words, pseudos, sem,
        mode="mixed-multitask",
        epochs=epochs,
        lr=0.01,
        zero_error_radius=0.0,
        device=torch.device("cpu"),
        seed=seed,
        loss_reduction="sum",
    )


def test_run_schedule_comparison_returns_both_schedules():
    result = _tiny_comparison()
    assert set(result.results.keys()) == {"uniform", "paper"}


def test_run_schedule_comparison_result_fields():
    result = _tiny_comparison()
    for r in result.results.values():
        assert hasattr(r, "initial_avg")
        assert hasattr(r, "final_avg")
        assert hasattr(r, "best_avg")
        assert hasattr(r, "best_epoch")
        assert hasattr(r, "epoch_avgs")
        assert hasattr(r, "epoch_losses")


def test_run_schedule_comparison_epoch_counts_match():
    result = _tiny_comparison(epochs=3)
    counts = [len(r.epoch_avgs) for r in result.results.values()]
    assert len(set(counts)) == 1, "both schedules must have the same number of epochs"
    assert counts[0] == 3


def test_run_schedule_comparison_losses_finite():
    result = _tiny_comparison()
    for schedule, r in result.results.items():
        assert all(math.isfinite(a) for a in r.epoch_avgs), (
            f"non-finite epoch avg for {schedule}"
        )


def test_trials_per_epoch_paper_greater_than_uniform():
    result = _tiny_comparison()
    assert result.trials_per_epoch["paper"] > result.trials_per_epoch["uniform"]


def test_schedule_comparison_result_is_dataclass():
    fnames = {f.name for f in dc_fields(ScheduleComparisonResult)}
    assert "results" in fnames
    assert "trials_per_epoch" in fnames


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
def test_csv_schedule_comparison_finite():
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
    result  = run_schedule_comparison(
        eng_cfg, words, pseudos, sem_map,
        mode="mixed-multitask",
        epochs=3, lr=0.01, zero_error_radius=0.0,
        device=torch.device("cpu"), seed=0,
        loss_reduction="mean_active",
    )
    for schedule, r in result.results.items():
        assert all(math.isfinite(a) for a in r.epoch_avgs), (
            f"non-finite epoch avg for {schedule}"
        )
