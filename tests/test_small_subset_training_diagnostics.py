"""Phase 3c-4 tests: small-subset training stability diagnostics.

Structure:
- Always-run (no CSV required): toy config + mock WordItem/PseudowordItem dataclasses.
- CSV-dependent: skip gracefully per specific file.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lichtheim2.config import load_config
from lichtheim2.data import PseudowordItem, WordItem
from lichtheim2.model import Lichtheim2Model
from lichtheim2.semantics import assign_artificial_semantics
from lichtheim2.tasks import Task

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from diagnose_small_subset_training import (
    DiagnosticResult,
    build_trials_for_mode,
    build_word_rep_trials,
    run_diagnostic_epochs,
    sample_items,
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

cfg = load_config(REPO_ROOT / "configs" / "toy.yaml")  # sound=5, motor=5, vATL=6


def _toy_model() -> Lichtheim2Model:
    torch.manual_seed(0)
    return Lichtheim2Model(cfg)


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
# Always-run: sample_items
# ---------------------------------------------------------------------------


def test_sample_items_none_returns_all():
    items = list(range(5))
    result = sample_items(items, None, random.Random(0))
    assert sorted(result) == [0, 1, 2, 3, 4]


def test_sample_items_zero_returns_empty():
    items = list(range(5))
    result = sample_items(items, 0, random.Random(0))
    assert result == []


def test_sample_items_positive_returns_subset():
    items = list(range(10))
    result = sample_items(items, 3, random.Random(0))
    assert len(result) == 3


def test_sample_items_max_larger_than_list_returns_all():
    items = list(range(4))
    result = sample_items(items, 100, random.Random(0))
    assert sorted(result) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Always-run: build_trials_for_mode
# ---------------------------------------------------------------------------


def test_build_trials_repetition_mode():
    words   = [_mock_word("tank", 0), _mock_word("cat", 1)]
    pseudos = [_mock_pseudo(0), _mock_pseudo(1)]
    trials  = build_trials_for_mode("repetition", words, pseudos, {}, cfg.motor_output_size)
    assert len(trials) == 4
    assert all(t.task == Task.REPETITION for t in trials)
    assert all(t.sem_input is None for t in trials)
    assert all(t.semantic_targets is None for t in trials)


def test_build_trials_words_multitask_mode():
    words  = [_mock_word("tank", 0), _mock_word("cat", 1)]
    sem    = _sem_map(words)
    trials = build_trials_for_mode("words-multitask", words, [], sem, cfg.motor_output_size)
    assert len(trials) == 6  # 2 words × 3 tasks
    tasks = [t.task for t in trials]
    assert tasks.count(Task.REPETITION)    == 2
    assert tasks.count(Task.COMPREHENSION) == 2
    assert tasks.count(Task.SPEAKING)      == 2


def test_build_trials_mixed_multitask_mode():
    words   = [_mock_word("tank", 0), _mock_word("cat", 1)]
    pseudos = [_mock_pseudo(0), _mock_pseudo(1)]
    sem     = _sem_map(words)
    trials  = build_trials_for_mode("mixed-multitask", words, pseudos, sem, cfg.motor_output_size)
    assert len(trials) == 8  # 2×3 word trials + 2×1 pseudo trials


def test_build_trials_empty_words_repetition():
    # max_words=0 scenario: word_items=[] produces only pseudo trials
    pseudos = [_mock_pseudo(0), _mock_pseudo(1)]
    trials  = build_trials_for_mode("repetition", [], pseudos, {}, cfg.motor_output_size)
    assert len(trials) == 2
    assert all(t.task == Task.REPETITION for t in trials)


def test_build_trials_labels_include_row_index():
    word   = _mock_word("tank", row_index=7)
    pseudo = _mock_pseudo(row_index=3)
    trials = build_trials_for_mode("repetition", [word], [pseudo], {}, cfg.motor_output_size)
    labels = [t.label for t in trials]
    assert any("7" in lbl for lbl in labels)
    assert any("pseudo:3" in lbl for lbl in labels)


def test_build_word_rep_trials_raises_on_none_row_index():
    bad_word = WordItem(
        word="bad", phonemes=["X"], phon_tensor=torch.rand(1, cfg.sound_input_size),
        lexicality="real", length=1, frequency=None, zipf_frequency=None,
        part_of_speech=None, condition=None, morphology=None,
        source="wfe", row_index=None,
    )
    with pytest.raises(ValueError, match="row_index=None"):
        build_word_rep_trials([bad_word], cfg.motor_output_size)


# ---------------------------------------------------------------------------
# Always-run: run_diagnostic_epochs and DiagnosticResult
# ---------------------------------------------------------------------------


def test_run_diagnostic_epochs_finite():
    words   = [_mock_word("tank", 0), _mock_word("cat", 1)]
    pseudos = [_mock_pseudo(0), _mock_pseudo(1)]
    sem     = _sem_map(words)
    trials  = build_trials_for_mode("mixed-multitask", words, pseudos, sem, cfg.motor_output_size)

    model = _toy_model()
    opt   = optim.SGD(model.parameters(), lr=0.01)
    rng   = random.Random(0)
    result = run_diagnostic_epochs(
        model, trials, opt, cfg,
        epochs=3, zero_error_radius=0.0,
        device=torch.device("cpu"), rng=rng,
    )
    for ep in result.epoch_losses:
        for task_losses in ep.values():
            assert all(math.isfinite(l) for l in task_losses)


def test_run_diagnostic_epochs_mean_active_finite():
    """loss_reduction='mean_active' produces finite losses through run_diagnostic_epochs."""
    words   = [_mock_word("tank", 0), _mock_word("cat", 1)]
    pseudos = [_mock_pseudo(0), _mock_pseudo(1)]
    sem     = _sem_map(words)
    trials  = build_trials_for_mode("mixed-multitask", words, pseudos, sem, cfg.motor_output_size)

    model  = _toy_model()
    opt    = optim.SGD(model.parameters(), lr=0.01)
    rng    = random.Random(0)
    result = run_diagnostic_epochs(
        model, trials, opt, cfg,
        epochs=2, zero_error_radius=0.0,
        device=torch.device("cpu"), rng=rng,
        loss_reduction="mean_active",
    )
    for ep in result.epoch_losses:
        for task_losses in ep.values():
            assert all(math.isfinite(l) for l in task_losses)


def test_diagnostic_result_fields_present():
    words  = [_mock_word("tank", 0)]
    sem    = _sem_map(words)
    trials = build_trials_for_mode("words-multitask", words, [], sem, cfg.motor_output_size)

    model = _toy_model()
    opt   = optim.SGD(model.parameters(), lr=0.01)
    result = run_diagnostic_epochs(
        model, trials, opt, cfg,
        epochs=2, zero_error_radius=0.0,
        device=torch.device("cpu"), rng=random.Random(0),
    )
    assert isinstance(result, DiagnosticResult)
    assert isinstance(result.epoch_avgs, list)
    assert isinstance(result.initial_avg, float)
    assert isinstance(result.final_avg, float)
    assert isinstance(result.best_avg, float)
    assert isinstance(result.best_epoch, int)


def test_diagnostic_result_epoch_count():
    words  = [_mock_word("tank", 0)]
    sem    = _sem_map(words)
    trials = build_trials_for_mode("repetition", words, [], {}, cfg.motor_output_size)

    model  = _toy_model()
    opt    = optim.SGD(model.parameters(), lr=0.01)
    result = run_diagnostic_epochs(
        model, trials, opt, cfg,
        epochs=5, zero_error_radius=0.0,
        device=torch.device("cpu"), rng=random.Random(0),
    )
    assert len(result.epoch_avgs)   == 5
    assert len(result.epoch_losses) == 5


def test_diagnostic_result_best_epoch_is_minimum():
    words  = [_mock_word("tank", 0), _mock_word("cat", 1)]
    sem    = _sem_map(words)
    trials = build_trials_for_mode("words-multitask", words, [], sem, cfg.motor_output_size)

    model  = _toy_model()
    opt    = optim.SGD(model.parameters(), lr=0.01)
    result = run_diagnostic_epochs(
        model, trials, opt, cfg,
        epochs=4, zero_error_radius=0.0,
        device=torch.device("cpu"), rng=random.Random(0),
    )
    assert result.best_avg == min(result.epoch_avgs)
    assert result.epoch_avgs[result.best_epoch - 1] == result.best_avg


# ---------------------------------------------------------------------------
# Always-run: validate_args
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    """Return a minimal valid Namespace, with any field overridden."""
    import argparse
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
    ("max_words",        -1),
    ("max_pseudowords",  -1),
    ("epochs",            0),
    ("lr",               0.0),
    ("lr",              -1.0),
    ("zero_error_radius", -0.1),
])
def test_validate_args_invalid(field, value):
    err = validate_args(_make_args(**{field: value}))
    assert err is not None
    assert isinstance(err, str)


# ---------------------------------------------------------------------------
# CSV-dependent integration tests
# ---------------------------------------------------------------------------

_wfe_available = pytest.mark.skipif(
    not (PHONEMES_CSV.exists() and WFE_CSV.exists()),
    reason="private CSV data (phonemes.csv + wfe.csv) not present",
)
_all_csv_available = pytest.mark.skipif(
    not (PHONEMES_CSV.exists() and WFE_CSV.exists() and SSP_CSV.exists()),
    reason="private CSV data (phonemes.csv + wfe.csv + ssp.csv) not present",
)


@_wfe_available
def test_csv_repetition_mode_words_only_finite():
    from lichtheim2.data import load_word_items
    from lichtheim2.encoding import load_phoneme_inventory
    inventory  = load_phoneme_inventory(PHONEMES_CSV)
    words      = load_word_items(WFE_CSV, inventory)[:5]
    trials     = build_trials_for_mode("repetition", words, [], {}, motor_size=39)
    assert len(trials) == 5

    from lichtheim2.config import load_config as _lc
    eng_cfg = _lc(REPO_ROOT / "configs" / "english_nwr.yaml")
    model   = Lichtheim2Model(eng_cfg)
    opt     = optim.SGD(model.parameters(), lr=0.01)
    result  = run_diagnostic_epochs(
        model, trials, opt, eng_cfg,
        epochs=2, zero_error_radius=0.0,
        device=torch.device("cpu"), rng=random.Random(0),
    )
    assert all(math.isfinite(a) for a in result.epoch_avgs)


@_all_csv_available
def test_csv_mixed_multitask_mode_finite():
    from lichtheim2.config import load_config as _lc
    from lichtheim2.data import load_pseudoword_items, load_word_items
    from lichtheim2.encoding import load_phoneme_inventory
    eng_cfg   = _lc(REPO_ROOT / "configs" / "english_nwr.yaml")
    inventory = load_phoneme_inventory(PHONEMES_CSV)

    words_all  = load_word_items(WFE_CSV, inventory)
    pseudo_all = load_pseudoword_items(SSP_CSV, inventory)
    sem_map    = assign_artificial_semantics(words_all, vATL_size=eng_cfg.vATL_size, seed=42)

    words   = words_all[:5]
    pseudos = pseudo_all[:5]
    trials  = build_trials_for_mode("mixed-multitask", words, pseudos, sem_map, eng_cfg.motor_output_size)
    assert len(trials) == 20  # 5×3 + 5×1

    model  = Lichtheim2Model(eng_cfg)
    opt    = optim.SGD(model.parameters(), lr=0.01)
    result = run_diagnostic_epochs(
        model, trials, opt, eng_cfg,
        epochs=2, zero_error_radius=0.0,
        device=torch.device("cpu"), rng=random.Random(0),
    )
    assert all(math.isfinite(a) for a in result.epoch_avgs)
