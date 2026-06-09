"""Phase 3c-9 tests: frequency-weighting comparison utility.

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
from lichtheim2.trials import make_repetition_trial

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from compare_frequency_weighting import (
    FrequencyComparisonResult,
    apply_weights_to_trials,
    compute_word_weights,
    run_frequency_comparison,
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


def _mock_word_freq(
    word: str = "tank",
    row_index: int = 0,
    freq: float | None = None,
    zipf: float | None = None,
    T: int = 3,
) -> WordItem:
    return WordItem(
        word=word,
        phonemes=["X"] * T,
        phon_tensor=torch.rand(T, cfg.sound_input_size),
        lexicality="real",
        length=T,
        frequency=freq,
        zipf_frequency=zipf,
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
# Always-run: compute_word_weights — zipf source
# ---------------------------------------------------------------------------


def test_compute_weights_mean_one_zipf():
    """Mean of normalised zipf weights must be 1.0."""
    words = [
        _mock_word_freq("a", 0, zipf=3.0),
        _mock_word_freq("b", 1, zipf=5.0),
        _mock_word_freq("c", 2, zipf=7.0),
    ]
    weights = compute_word_weights(words, frequency_source="zipf")
    vals = list(weights.values())
    assert abs(sum(vals) / len(vals) - 1.0) < 1e-6, (
        f"Expected mean ≈ 1.0, got {sum(vals) / len(vals)}"
    )


def test_compute_weights_mean_one_frequency():
    """Mean of normalised log1p(frequency) weights must be 1.0."""
    words = [
        _mock_word_freq("a", 0, freq=10.0),
        _mock_word_freq("b", 1, freq=100.0),
        _mock_word_freq("c", 2, freq=1000.0),
    ]
    weights = compute_word_weights(words, frequency_source="frequency")
    vals = list(weights.values())
    assert abs(sum(vals) / len(vals) - 1.0) < 1e-6, (
        f"Expected mean ≈ 1.0, got {sum(vals) / len(vals)}"
    )


def test_compute_weights_all_positive():
    """All returned weights must be strictly positive."""
    words = [
        _mock_word_freq("a", 0, zipf=1.0),
        _mock_word_freq("b", 1, zipf=2.0),
        _mock_word_freq("c", 2, zipf=None),  # None → 1.0
    ]
    weights = compute_word_weights(words, frequency_source="zipf")
    assert all(v > 0 for v in weights.values()), (
        f"Found non-positive weight: {weights}"
    )


def test_compute_weights_none_zipf_gets_one():
    """Word with None zipf_frequency gets weight 1.0."""
    words = [
        _mock_word_freq("a", 0, zipf=2.0),
        _mock_word_freq("b", 1, zipf=None),
    ]
    weights = compute_word_weights(words, frequency_source="zipf")
    assert weights[1] == 1.0, f"Expected 1.0 for None frequency, got {weights[1]}"


def test_compute_weights_none_frequency_gets_one():
    """Word with None frequency gets weight 1.0 when using 'frequency' source."""
    words = [
        _mock_word_freq("a", 0, freq=100.0),
        _mock_word_freq("b", 1, freq=None),
    ]
    weights = compute_word_weights(words, frequency_source="frequency")
    assert weights[1] == 1.0, f"Expected 1.0 for None frequency, got {weights[1]}"


def test_compute_weights_all_none_returns_all_one():
    """When all words have None frequency, all weights default to 1.0."""
    words = [
        _mock_word_freq("a", 0, zipf=None),
        _mock_word_freq("b", 1, zipf=None),
    ]
    weights = compute_word_weights(words, frequency_source="zipf")
    assert all(v == 1.0 for v in weights.values()), (
        f"Expected all 1.0 when all None, got {weights}"
    )


def test_compute_weights_zero_frequency_raw_gets_one():
    """Word with raw frequency=0 → log1p(0)=0, non-positive → gets weight 1.0."""
    words = [
        _mock_word_freq("a", 0, freq=100.0),
        _mock_word_freq("b", 1, freq=0.0),
    ]
    weights = compute_word_weights(words, frequency_source="frequency")
    assert weights[1] == 1.0, f"Expected 1.0 for zero frequency, got {weights[1]}"


def test_compute_weights_invalid_source_raises():
    words = [_mock_word_freq("a", 0, zipf=3.0)]
    with pytest.raises(ValueError, match="frequency_source"):
        compute_word_weights(words, frequency_source="invalid")


def test_compute_weights_invalid_normalization_raises():
    words = [_mock_word_freq("a", 0, zipf=3.0)]
    with pytest.raises(ValueError, match="normalization"):
        compute_word_weights(words, frequency_source="zipf", normalization="bad")


# ---------------------------------------------------------------------------
# Always-run: apply_weights_to_trials
# ---------------------------------------------------------------------------


def test_apply_weights_sets_word_trial_weight():
    """Word trial with label 'word:tank' gets weight from weight_map."""
    word = _mock_word_freq("tank", row_index=5, zipf=3.0)
    sem  = _sem_map([word])
    trial = make_repetition_trial(
        word.phon_tensor, cfg.motor_output_size,
        item_id=5, label="word:tank",
    )
    weight_map = {5: 2.5}
    result = apply_weights_to_trials([trial], weight_map)
    assert len(result) == 1
    assert abs(result[0].loss_weight - 2.5) < 1e-9, (
        f"Expected weight 2.5, got {result[0].loss_weight}"
    )


def test_apply_weights_pseudo_keeps_default():
    """Pseudoword trial (no word: label) always keeps loss_weight=1.0."""
    pseudo = _mock_pseudo(row_index=5)
    trial = make_repetition_trial(
        pseudo.phon_tensor, cfg.motor_output_size,
        item_id=5, label="pseudo:np",
    )
    weight_map = {5: 99.0}  # same row_index=5 but label is not "word:..."
    result = apply_weights_to_trials([trial], weight_map)
    assert result[0].loss_weight == 1.0, (
        f"Pseudoword trial must keep weight 1.0, got {result[0].loss_weight}"
    )


def test_apply_weights_same_row_index_collision():
    """When word and pseudo share row_index, only the word trial gets the weight.

    This is the critical guard: row_index collision must not pollute pseudowords.
    """
    word   = _mock_word_freq("tank", row_index=5, zipf=3.0)
    pseudo = _mock_pseudo(row_index=5)
    word_trial = make_repetition_trial(
        word.phon_tensor, cfg.motor_output_size,
        item_id=5, label="word:tank",
    )
    pseudo_trial = make_repetition_trial(
        pseudo.phon_tensor, cfg.motor_output_size,
        item_id=5, label="pseudo:np",
    )
    weight_map = {5: 2.5}
    results = apply_weights_to_trials([word_trial, pseudo_trial], weight_map)
    assert abs(results[0].loss_weight - 2.5) < 1e-9
    assert results[1].loss_weight == 1.0


def test_apply_weights_returns_new_list():
    """Original trial list must be unchanged (apply_weights creates new trials)."""
    word = _mock_word_freq("tank", row_index=0, zipf=2.0)
    trial = make_repetition_trial(
        word.phon_tensor, cfg.motor_output_size,
        item_id=0, label="word:tank",
    )
    originals = [trial]
    weight_map = {0: 3.0}
    apply_weights_to_trials(originals, weight_map)
    assert originals[0].loss_weight == 1.0, "Original trial must not be mutated"


def test_apply_weights_unlabelled_trial_keeps_default():
    """Trial with label=None keeps loss_weight=1.0."""
    word = _mock_word_freq("tank", row_index=0, zipf=2.0)
    trial = make_repetition_trial(
        word.phon_tensor, cfg.motor_output_size,
        item_id=0, label=None,
    )
    weight_map = {0: 5.0}
    result = apply_weights_to_trials([trial], weight_map)
    assert result[0].loss_weight == 1.0


# ---------------------------------------------------------------------------
# Always-run: run_frequency_comparison
# ---------------------------------------------------------------------------


def _tiny_comparison(
    seed: int = 0,
    epochs: int = 2,
    frequency_source: str = "zipf",
) -> FrequencyComparisonResult:
    words = [
        _mock_word_freq("tank", 0, zipf=3.0, freq=500.0),
        _mock_word_freq("cat",  1, zipf=5.0, freq=1000.0),
    ]
    pseudos = [_mock_pseudo(0), _mock_pseudo(1)]
    sem     = _sem_map(words)
    return run_frequency_comparison(
        cfg, words, pseudos, sem,
        mode="mixed-multitask",
        schedule="paper",
        epochs=epochs,
        lr=0.01,
        zero_error_radius=0.0,
        device=torch.device("cpu"),
        seed=seed,
        loss_reduction="sum",
        frequency_source=frequency_source,
        weight_normalization="mean_one",
    )


def test_run_frequency_comparison_returns_both_conditions():
    result = _tiny_comparison()
    assert set(result.results.keys()) == {"unweighted", "weighted"}


def test_run_frequency_comparison_result_fields():
    result = _tiny_comparison()
    for r in result.results.values():
        assert hasattr(r, "initial_avg")
        assert hasattr(r, "final_avg")
        assert hasattr(r, "best_avg")
        assert hasattr(r, "best_epoch")
        assert hasattr(r, "epoch_avgs")
        assert hasattr(r, "epoch_losses")


def test_run_frequency_comparison_epoch_counts_match():
    result = _tiny_comparison(epochs=3)
    counts = [len(r.epoch_avgs) for r in result.results.values()]
    assert len(set(counts)) == 1, "both conditions must have the same number of epochs"
    assert counts[0] == 3


def test_run_frequency_comparison_losses_finite():
    result = _tiny_comparison()
    for condition, r in result.results.items():
        assert all(math.isfinite(a) for a in r.epoch_avgs), (
            f"non-finite epoch avg for condition {condition!r}"
        )


def test_frequency_comparison_result_fields():
    result = _tiny_comparison()
    fnames = {f.name for f in dc_fields(FrequencyComparisonResult)}
    assert "results"          in fnames
    assert "trials_per_epoch" in fnames
    assert "weight_stats"     in fnames


def test_run_frequency_comparison_weight_stats():
    result = _tiny_comparison()
    ws = result.weight_stats
    assert "min"     in ws
    assert "mean"    in ws
    assert "max"     in ws
    assert "n_words" in ws
    # For two words with known zipf values (3.0 and 5.0), mean=4.0 → weights 0.75, 1.25
    assert ws["min"] < ws["max"]
    assert ws["min"] > 0


def test_run_frequency_comparison_frequency_source():
    """Comparison with frequency source must also complete with finite losses."""
    result = _tiny_comparison(frequency_source="frequency")
    for condition, r in result.results.items():
        assert all(math.isfinite(a) for a in r.epoch_avgs), (
            f"non-finite epoch avg for condition {condition!r}"
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
def test_csv_frequency_comparison_finite():
    """Real English NWR data: both frequency sources must give finite losses."""
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

    for frequency_source in ("zipf", "frequency"):
        result = run_frequency_comparison(
            eng_cfg, words, pseudos, sem_map,
            mode="mixed-multitask",
            schedule="paper",
            epochs=3,
            lr=0.01,
            zero_error_radius=0.0,
            device=torch.device("cpu"),
            seed=0,
            loss_reduction="mean_active",
            frequency_source=frequency_source,
            weight_normalization="mean_one",
        )
        for condition, r in result.results.items():
            assert all(math.isfinite(a) for a in r.epoch_avgs), (
                f"non-finite epoch avg for {frequency_source}/{condition}"
            )
