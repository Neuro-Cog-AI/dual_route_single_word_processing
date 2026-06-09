"""Phase 3c-7 tests: loss reduction comparison utility.

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
from lichtheim2.model import Lichtheim2Model
from lichtheim2.semantics import assign_artificial_semantics

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from compare_loss_reductions import ComparisonResult, run_comparison, validate_args
from diagnose_small_subset_training import build_trials_for_mode

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


def _tiny_trials():
    words   = [_mock_word("tank", 0), _mock_word("cat", 1)]
    pseudos = [_mock_pseudo(0), _mock_pseudo(1)]
    sem     = _sem_map(words)
    return build_trials_for_mode("mixed-multitask", words, pseudos, sem, cfg.motor_output_size)


def _tiny_comparison(seed: int = 0, epochs: int = 2) -> ComparisonResult:
    trials = _tiny_trials()
    return run_comparison(
        cfg, trials,
        epochs=epochs,
        lr=0.01,
        zero_error_radius=0.0,
        device=torch.device("cpu"),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Always-run: run_comparison
# ---------------------------------------------------------------------------


def test_run_comparison_returns_both_reductions():
    result = _tiny_comparison()
    assert set(result.results.keys()) == {"sum", "mean_active"}


def test_run_comparison_result_fields():
    result = _tiny_comparison()
    for r in result.results.values():
        assert hasattr(r, "initial_avg")
        assert hasattr(r, "final_avg")
        assert hasattr(r, "best_avg")
        assert hasattr(r, "best_epoch")
        assert hasattr(r, "epoch_avgs")
        assert hasattr(r, "epoch_losses")


def test_run_comparison_epoch_counts_match():
    result = _tiny_comparison(epochs=3)
    counts = [len(r.epoch_avgs) for r in result.results.values()]
    assert len(set(counts)) == 1, "both reductions must have the same number of epochs"
    assert counts[0] == 3


def test_run_comparison_losses_finite():
    result = _tiny_comparison()
    for reduction, r in result.results.items():
        assert all(math.isfinite(a) for a in r.epoch_avgs), (
            f"non-finite epoch avg for {reduction}"
        )


def test_run_comparison_best_epoch_valid():
    result = _tiny_comparison(epochs=3)
    for r in result.results.values():
        assert 1 <= r.best_epoch <= 3


def test_comparison_result_is_dataclass():
    fnames = {f.name for f in dc_fields(ComparisonResult)}
    assert "results" in fnames


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
def test_csv_comparison_finite():
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
    trials  = build_trials_for_mode(
        "mixed-multitask", words, pseudos, sem_map, eng_cfg.motor_output_size
    )
    result  = run_comparison(
        eng_cfg, trials, epochs=3, lr=0.01, zero_error_radius=0.0,
        device=torch.device("cpu"), seed=0,
    )
    for reduction, r in result.results.items():
        assert all(math.isfinite(a) for a in r.epoch_avgs), (
            f"non-finite epoch avg for {reduction}"
        )
