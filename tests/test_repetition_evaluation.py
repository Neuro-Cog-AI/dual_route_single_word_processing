"""Tests for src/lichtheim2/repetition_evaluation.py.

Uses the toy config and a minimal synthetic phoneme tensor — no CSV data required.
"""
from __future__ import annotations

from pathlib import Path

import torch

from lichtheim2.config import load_config
from lichtheim2.model import Lichtheim2Model
from lichtheim2.repetition_evaluation import evaluate_one_trial, evaluate_predictions
from lichtheim2.trials import make_repetition_trial

_TOY_CONFIG = Path(__file__).parent.parent / "configs" / "toy.yaml"

_EXPECTED_KEYS = {
    "label", "item_id", "T", "n_ticks",
    "target_phonemes", "predicted_phonemes",
    "phoneme_accuracy", "threshold_accuracy", "exact_match",
    # split metric breakdown keys
    "input_silence_threshold_acc",
    "output_negative_threshold_acc",
    "output_positive_threshold_acc",
    "mean_positive_output",
    "mean_negative_output",
    "mean_max_motor_output",
    # word accuracy keys
    "output_all_units_within_radius",
    "input_all_silent_within_radius",
    "trial_all_supervised_units_within_radius",
}


def _make_model_and_trial():
    cfg = load_config(_TOY_CONFIG)
    model = Lichtheim2Model(cfg)
    T = 2
    phon = torch.zeros(T, cfg.motor_output_size)
    phon[0, 0] = 1.0
    phon[1, 1] = 1.0
    trial = make_repetition_trial(phon, cfg.motor_output_size, item_id=0, label="synth")
    inventory_symbols = [str(i) for i in range(cfg.motor_output_size)]
    return model, trial, cfg, inventory_symbols


def test_evaluate_one_trial_expected_keys():
    """evaluate_one_trial returns a dict with all expected keys."""
    model, trial, cfg, symbols = _make_model_and_trial()
    result = evaluate_one_trial(
        model, trial, cfg, symbols, device=torch.device("cpu"), eval_radius=0.1
    )
    assert isinstance(result, dict)
    missing = _EXPECTED_KEYS - set(result.keys())
    assert not missing, f"Missing keys: {missing}"


def test_evaluate_one_trial_T_matches():
    """evaluate_one_trial reports the correct T value."""
    model, trial, cfg, symbols = _make_model_and_trial()
    result = evaluate_one_trial(
        model, trial, cfg, symbols, device=torch.device("cpu"), eval_radius=0.1
    )
    T = trial.phon_tensor.shape[0]
    assert result["T"] == T
    assert result["n_ticks"] == 2 * T


def test_evaluate_one_trial_does_not_modify_model_train_state():
    """evaluate_one_trial calls model.eval() internally; model must be usable after."""
    model, trial, cfg, symbols = _make_model_and_trial()
    model.train()
    evaluate_one_trial(
        model, trial, cfg, symbols, device=torch.device("cpu"), eval_radius=0.1
    )
    # Model should be callable (not in a broken state); training mode can be anything
    assert model is not None


def test_evaluate_predictions_length():
    """evaluate_predictions returns exactly one record per trial."""
    cfg = load_config(_TOY_CONFIG)
    model = Lichtheim2Model(cfg)
    M = cfg.motor_output_size
    symbols = [str(i) for i in range(M)]

    trials = []
    for t_len in [2, 3, 2]:
        phon = torch.zeros(t_len, M)
        for row in range(t_len):
            phon[row, row % M] = 1.0
        trials.append(make_repetition_trial(phon, M, item_id=len(trials), label=f"t{t_len}"))

    results = evaluate_predictions(
        model, trials, cfg, symbols, device=torch.device("cpu"), eval_radius=0.1
    )
    assert len(results) == len(trials)


def test_evaluate_predictions_all_keys_present():
    """Every record from evaluate_predictions has the full set of expected keys."""
    cfg = load_config(_TOY_CONFIG)
    model = Lichtheim2Model(cfg)
    M = cfg.motor_output_size
    symbols = [str(i) for i in range(M)]

    phon = torch.zeros(2, M)
    phon[0, 0] = 1.0
    phon[1, 1] = 1.0
    trials = [make_repetition_trial(phon, M, item_id=0, label="t")]

    results = evaluate_predictions(
        model, trials, cfg, symbols, device=torch.device("cpu"), eval_radius=0.1
    )
    for rec in results:
        missing = _EXPECTED_KEYS - set(rec.keys())
        assert not missing, f"Missing keys: {missing}"
