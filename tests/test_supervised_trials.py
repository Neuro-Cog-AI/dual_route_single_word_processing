"""Tests for supervised trial targets and artificial semantic vectors.

All tests are always-run — no local CSV data required.
Mock WordItems are constructed inline using small tensors.
"""
from dataclasses import dataclass

import pytest
import torch

from lichtheim2.data import WordItem
from lichtheim2.semantics import assign_artificial_semantics
from lichtheim2.tasks import Task
from lichtheim2.trials import (
    SupervisedTrial,
    make_comprehension_trial,
    make_repetition_trial,
    make_speaking_trial,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

T          = 3    # sequence length
SOUND_SIZE = 5    # phoneme input / motor output dimension
MOTOR_SIZE = 5    # must equal SOUND_SIZE for repetition / speaking
VATL_SIZE  = 6    # semantic dimension

# Deterministic test tensors
torch.manual_seed(99)
PHON = torch.rand(T, SOUND_SIZE)
SEM  = torch.rand(VATL_SIZE)


# ---------------------------------------------------------------------------
# Mock WordItem factory
# ---------------------------------------------------------------------------

def _make_word_item(word: str, row_index: int | None, T: int = 3) -> WordItem:
    phonemes = ["AH"] * T
    return WordItem(
        word=word,
        phonemes=phonemes,
        phon_tensor=torch.rand(T, SOUND_SIZE),
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


# ---------------------------------------------------------------------------
# assign_artificial_semantics
# ---------------------------------------------------------------------------

def test_assign_semantics_shape():
    items = [_make_word_item("cat", 0), _make_word_item("dog", 1)]
    result = assign_artificial_semantics(items, vATL_size=VATL_SIZE, seed=42)
    for vec in result.values():
        assert vec.shape == (VATL_SIZE,)
        assert vec.dtype == torch.float32


def test_assign_semantics_binary():
    items = [_make_word_item("cat", 0), _make_word_item("dog", 1)]
    result = assign_artificial_semantics(items, vATL_size=VATL_SIZE, seed=42)
    for vec in result.values():
        assert torch.all((vec == 0.0) | (vec == 1.0))


def test_assign_semantics_deterministic():
    items = [_make_word_item("cat", 0), _make_word_item("dog", 1)]
    r1 = assign_artificial_semantics(items, vATL_size=VATL_SIZE, seed=42)
    r2 = assign_artificial_semantics(items, vATL_size=VATL_SIZE, seed=42)
    for key in r1:
        assert torch.allclose(r1[key], r2[key])


def test_assign_semantics_different_seed():
    items = [_make_word_item("cat", 0), _make_word_item("dog", 1)]
    r1 = assign_artificial_semantics(items, vATL_size=VATL_SIZE, seed=42)
    r2 = assign_artificial_semantics(items, vATL_size=VATL_SIZE, seed=99)
    # At least one vector must differ
    differs = any(not torch.allclose(r1[k], r2[k]) for k in r1)
    assert differs, "Different seeds should produce different vectors"


def test_assign_semantics_keyed_by_row_index():
    items = [_make_word_item("cat", 7), _make_word_item("dog", 42)]
    result = assign_artificial_semantics(items, vATL_size=VATL_SIZE, seed=42)
    assert set(result.keys()) == {7, 42}


def test_assign_semantics_none_row_index_raises():
    items = [_make_word_item("cat", None)]
    with pytest.raises(ValueError, match="row_index=None"):
        assign_artificial_semantics(items, vATL_size=VATL_SIZE)


def test_assign_semantics_duplicate_row_index_raises():
    items = [_make_word_item("cat", 0), _make_word_item("dog", 0)]
    with pytest.raises(ValueError, match="Duplicate row_index"):
        assign_artificial_semantics(items, vATL_size=VATL_SIZE)


def test_assign_semantics_empty_list():
    result = assign_artificial_semantics([], vATL_size=VATL_SIZE)
    assert result == {}


# ---------------------------------------------------------------------------
# make_repetition_trial
# ---------------------------------------------------------------------------

def test_repetition_n_ticks():
    trial = make_repetition_trial(PHON, MOTOR_SIZE)
    assert trial.n_ticks == 2 * T
    assert trial.task == Task.REPETITION


def test_repetition_motor_targets_shape():
    trial = make_repetition_trial(PHON, MOTOR_SIZE)
    assert trial.motor_targets.shape == (2 * T, MOTOR_SIZE)


def test_repetition_motor_silence_input_phase():
    trial = make_repetition_trial(PHON, MOTOR_SIZE)
    assert torch.allclose(trial.motor_targets[:T], torch.zeros(T, MOTOR_SIZE))


def test_repetition_motor_targets_output_phase():
    trial = make_repetition_trial(PHON, MOTOR_SIZE)
    assert torch.allclose(trial.motor_targets[T:], PHON)


def test_repetition_motor_loss_mask_all_true():
    trial = make_repetition_trial(PHON, MOTOR_SIZE)
    assert trial.motor_loss_mask.shape == (2 * T,)
    assert trial.motor_loss_mask.dtype == torch.bool
    assert trial.motor_loss_mask.all()


def test_repetition_no_semantic_targets():
    trial = make_repetition_trial(PHON, MOTOR_SIZE)
    assert trial.semantic_targets is None
    assert trial.semantic_loss_mask is None


def test_repetition_no_sem_input():
    trial = make_repetition_trial(PHON, MOTOR_SIZE)
    assert trial.sem_input is None


def test_repetition_motor_size_mismatch_raises():
    with pytest.raises(ValueError, match="motor_size"):
        make_repetition_trial(PHON, MOTOR_SIZE + 1)


def test_repetition_metadata():
    trial = make_repetition_trial(PHON, MOTOR_SIZE, item_id=7, label="cat")
    assert trial.item_id == 7
    assert trial.label == "cat"


# ---------------------------------------------------------------------------
# make_comprehension_trial
# ---------------------------------------------------------------------------

def test_comprehension_n_ticks():
    trial = make_comprehension_trial(PHON, SEM, MOTOR_SIZE)
    assert trial.n_ticks == T
    assert trial.task == Task.COMPREHENSION


def test_comprehension_semantic_targets_shape():
    trial = make_comprehension_trial(PHON, SEM, MOTOR_SIZE)
    assert trial.semantic_targets is not None
    assert trial.semantic_targets.shape == (T, VATL_SIZE)


def test_comprehension_semantic_repeated():
    trial = make_comprehension_trial(PHON, SEM, MOTOR_SIZE)
    for t in range(T):
        assert torch.allclose(trial.semantic_targets[t], SEM)


def test_comprehension_semantic_loss_mask_all_true():
    trial = make_comprehension_trial(PHON, SEM, MOTOR_SIZE)
    assert trial.semantic_loss_mask is not None
    assert trial.semantic_loss_mask.shape == (T,)
    assert trial.semantic_loss_mask.all()


def test_comprehension_motor_targets_zero():
    trial = make_comprehension_trial(PHON, SEM, MOTOR_SIZE)
    assert torch.allclose(trial.motor_targets, torch.zeros(T, MOTOR_SIZE))


def test_comprehension_motor_loss_mask_all_true():
    trial = make_comprehension_trial(PHON, SEM, MOTOR_SIZE)
    assert trial.motor_loss_mask.shape == (T,)
    assert trial.motor_loss_mask.all()


def test_comprehension_no_sem_input():
    trial = make_comprehension_trial(PHON, SEM, MOTOR_SIZE)
    assert trial.sem_input is None


def test_comprehension_metadata():
    trial = make_comprehension_trial(PHON, SEM, MOTOR_SIZE, item_id=3, label="dog")
    assert trial.item_id == 3
    assert trial.label == "dog"


# ---------------------------------------------------------------------------
# make_speaking_trial
# ---------------------------------------------------------------------------

def test_speaking_n_ticks():
    trial = make_speaking_trial(PHON, SEM, MOTOR_SIZE)
    assert trial.n_ticks == T
    assert trial.task == Task.SPEAKING


def test_speaking_motor_targets():
    trial = make_speaking_trial(PHON, SEM, MOTOR_SIZE)
    assert trial.motor_targets.shape == (T, MOTOR_SIZE)
    assert torch.allclose(trial.motor_targets, PHON)


def test_speaking_motor_loss_mask_all_true():
    trial = make_speaking_trial(PHON, SEM, MOTOR_SIZE)
    assert trial.motor_loss_mask.shape == (T,)
    assert trial.motor_loss_mask.all()


def test_speaking_sem_input():
    trial = make_speaking_trial(PHON, SEM, MOTOR_SIZE)
    assert trial.sem_input is not None
    assert torch.allclose(trial.sem_input, SEM)


def test_speaking_no_semantic_targets():
    trial = make_speaking_trial(PHON, SEM, MOTOR_SIZE)
    assert trial.semantic_targets is None
    assert trial.semantic_loss_mask is None


def test_speaking_motor_size_mismatch_raises():
    with pytest.raises(ValueError, match="motor_size"):
        make_speaking_trial(PHON, SEM, MOTOR_SIZE + 2)


def test_speaking_metadata():
    trial = make_speaking_trial(PHON, SEM, MOTOR_SIZE, item_id=1, label="bird")
    assert trial.item_id == 1
    assert trial.label == "bird"
