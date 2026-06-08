"""Phase 3c-3 tests: multi-task training pipeline.

Structure:
- Always-run (no CSV required): toy config + synthetic WordItem/PseudowordItem
  objects constructed directly (both are plain public dataclasses).
- CSV-dependent: skip gracefully, checking for the specific files each test needs.
"""
from __future__ import annotations

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
from lichtheim2.trainer import train_step

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from train_multitask_real_data import (
    build_pseudo_trials,
    build_word_trials,
    train_multitask_epochs,
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
# Always-run: build_word_trials
# ---------------------------------------------------------------------------


def test_build_word_trials_three_tasks():
    words = [_mock_word("tank", 0), _mock_word("cat", 1)]
    sem = _sem_map(words)
    trials = build_word_trials(words, sem, cfg.motor_output_size)
    assert len(trials) == 6
    tasks = [t.task for t in trials]
    assert tasks == [
        Task.REPETITION, Task.COMPREHENSION, Task.SPEAKING,
        Task.REPETITION, Task.COMPREHENSION, Task.SPEAKING,
    ]


def test_build_word_trials_sem_fields():
    word = _mock_word("tank", 0, T=3)
    sem = _sem_map([word])
    trials = build_word_trials([word], sem, cfg.motor_output_size)
    rep, comp, spk = trials

    # Repetition: no semantics
    assert rep.sem_input is None
    assert rep.semantic_targets is None

    # Comprehension: vATL output target present; shape (T, vATL_size)
    assert comp.semantic_targets is not None
    assert comp.semantic_targets.shape == (3, cfg.vATL_size)
    assert comp.sem_input is None

    # Speaking: semantic input (clamped); shape (vATL_size,)
    assert spk.sem_input is not None
    assert spk.sem_input.shape == (cfg.vATL_size,)
    assert spk.semantic_targets is None


def test_build_word_trials_preserves_labels_and_item_id():
    word = _mock_word("tank", row_index=7)
    sem = _sem_map([word])
    trials = build_word_trials([word], sem, cfg.motor_output_size)
    for trial in trials:
        assert trial.label == "word:7:tank"
        assert trial.item_id == 7


def test_build_word_trials_raises_on_none_row_index():
    word = _mock_word("tank", row_index=0)
    word_bad = WordItem(
        word="bad", phonemes=["X"], phon_tensor=torch.rand(1, cfg.sound_input_size),
        lexicality="real", length=1, frequency=None, zipf_frequency=None,
        part_of_speech=None, condition=None, morphology=None,
        source="wfe", row_index=None,
    )
    sem = {0: torch.zeros(cfg.vATL_size)}
    with pytest.raises(ValueError, match="row_index=None"):
        build_word_trials([word, word_bad], sem, cfg.motor_output_size)


# ---------------------------------------------------------------------------
# Always-run: build_pseudo_trials
# ---------------------------------------------------------------------------


def test_build_pseudo_trials_repetition_only():
    pseudos = [_mock_pseudo(0), _mock_pseudo(1), _mock_pseudo(2)]
    trials = build_pseudo_trials(pseudos, cfg.motor_output_size)
    assert len(trials) == 3
    for trial in trials:
        assert trial.task == Task.REPETITION
        assert trial.sem_input is None
        assert trial.semantic_targets is None


def test_build_pseudo_trials_labels():
    pseudo = _mock_pseudo(row_index=5)
    trials = build_pseudo_trials([pseudo], cfg.motor_output_size)
    assert trials[0].label == "pseudo:5"
    assert trials[0].item_id == 5


# ---------------------------------------------------------------------------
# Always-run: train_multitask_epochs
# ---------------------------------------------------------------------------


def test_mini_multitask_epoch_finite():
    words  = [_mock_word("tank", 0), _mock_word("cat", 1)]
    pseudos = [_mock_pseudo(0), _mock_pseudo(1)]
    sem = _sem_map(words)
    trials = (
        build_word_trials(words, sem, cfg.motor_output_size)
        + build_pseudo_trials(pseudos, cfg.motor_output_size)
    )
    assert len(trials) == 8  # 2×3 + 2×1

    model = _toy_model()
    opt = optim.SGD(model.parameters(), lr=0.1)
    rng = random.Random(0)
    epoch_losses = train_multitask_epochs(
        model, trials, opt, cfg,
        epochs=2, zero_error_radius=0.1,
        device=torch.device("cpu"), rng=rng,
    )
    assert len(epoch_losses) == 2
    for ep in epoch_losses:
        for task_losses in ep.values():
            for l in task_losses:
                assert l == l and l != float("inf")  # finite


def test_train_multitask_epochs_per_task_keys():
    words   = [_mock_word("tank", 0), _mock_word("cat", 1)]
    pseudos = [_mock_pseudo(0), _mock_pseudo(1)]
    sem = _sem_map(words)
    trials = (
        build_word_trials(words, sem, cfg.motor_output_size)
        + build_pseudo_trials(pseudos, cfg.motor_output_size)
    )

    model = _toy_model()
    opt = optim.SGD(model.parameters(), lr=0.1)
    rng = random.Random(0)
    epoch_losses = train_multitask_epochs(
        model, trials, opt, cfg,
        epochs=1, zero_error_radius=0.0,
        device=torch.device("cpu"), rng=rng,
    )
    ep = epoch_losses[0]
    # 2 words × 1 rep + 2 pseudos × 1 rep = 4 repetition losses
    assert len(ep[Task.REPETITION]) == 4
    # 2 words × 1 comprehension = 2
    assert len(ep[Task.COMPREHENSION]) == 2
    # 2 words × 1 speaking = 2
    assert len(ep[Task.SPEAKING]) == 2


# ---------------------------------------------------------------------------
# CSV-dependent integration tests
# ---------------------------------------------------------------------------

_wfe_available = pytest.mark.skipif(
    not (PHONEMES_CSV.exists() and WFE_CSV.exists()),
    reason="private CSV data (phonemes.csv + wfe.csv) not present",
)
_ssp_available = pytest.mark.skipif(
    not (PHONEMES_CSV.exists() and SSP_CSV.exists()),
    reason="private CSV data (phonemes.csv + ssp.csv) not present",
)
_all_csv_available = pytest.mark.skipif(
    not (PHONEMES_CSV.exists() and WFE_CSV.exists() and SSP_CSV.exists()),
    reason="private CSV data (phonemes.csv + wfe.csv + ssp.csv) not present",
)


@_wfe_available
def test_csv_word_items_generate_three_trials_each():
    from lichtheim2.data import load_word_items
    from lichtheim2.encoding import load_phoneme_inventory
    inventory = load_phoneme_inventory(PHONEMES_CSV)
    word_items = load_word_items(WFE_CSV, inventory)[:3]
    sem = assign_artificial_semantics(word_items, vATL_size=50, seed=42)
    trials = build_word_trials(word_items, sem, motor_size=39)
    assert len(trials) == 9  # 3 words × 3 tasks
    tasks = [t.task for t in trials]
    assert tasks.count(Task.REPETITION)   == 3
    assert tasks.count(Task.COMPREHENSION) == 3
    assert tasks.count(Task.SPEAKING)     == 3


@_ssp_available
def test_csv_pseudo_items_generate_rep_only():
    from lichtheim2.data import load_pseudoword_items
    from lichtheim2.encoding import load_phoneme_inventory
    inventory = load_phoneme_inventory(PHONEMES_CSV)
    pseudo_items = load_pseudoword_items(SSP_CSV, inventory)[:3]
    trials = build_pseudo_trials(pseudo_items, motor_size=39)
    assert len(trials) == 3
    assert all(t.task == Task.REPETITION for t in trials)


@_all_csv_available
def test_csv_multitask_train_step():
    from lichtheim2.config import load_config as _lc
    from lichtheim2.data import load_pseudoword_items, load_word_items
    from lichtheim2.encoding import load_phoneme_inventory
    eng_cfg = _lc(REPO_ROOT / "configs" / "english_nwr.yaml")
    inventory = load_phoneme_inventory(PHONEMES_CSV)

    word_items  = load_word_items(WFE_CSV, inventory)[:2]
    pseudo_items = load_pseudoword_items(SSP_CSV, inventory)[:2]
    sem = assign_artificial_semantics(word_items, vATL_size=eng_cfg.vATL_size, seed=42)

    trials = (
        build_word_trials(word_items, sem, eng_cfg.motor_output_size)
        + build_pseudo_trials(pseudo_items, eng_cfg.motor_output_size)
    )
    assert len(trials) == 8  # 2×3 + 2×1

    model = Lichtheim2Model(eng_cfg)
    opt   = optim.SGD(model.parameters(), lr=0.1)
    rng   = random.Random(0)
    epoch_losses = train_multitask_epochs(
        model, trials, opt, eng_cfg,
        epochs=1, zero_error_radius=0.1,
        device=torch.device("cpu"), rng=rng,
    )
    for task_losses in epoch_losses[0].values():
        for l in task_losses:
            assert l == l and l != float("inf")
