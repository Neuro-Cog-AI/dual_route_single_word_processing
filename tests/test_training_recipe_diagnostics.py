"""Phase 3c-11 tests: integrated training recipe comparison utility.

Structure:
- Always-run (no CSV required): RECIPES / validate_recipe_config, toy comparison
  runs, invalid recipe handling, validate_args.
- CSV-dependent: skip gracefully if private data absent.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import fields as dc_fields, replace as dc_replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lichtheim2.config import load_config
from lichtheim2.data import PseudowordItem, WordItem
from lichtheim2.semantics import assign_artificial_semantics

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from compare_training_recipes import (
    DEFAULT_RECIPES,
    RECIPES,
    RecipeComparisonResult,
    RecipeConfig,
    run_recipe_comparison,
    validate_args,
    validate_recipe_config,
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


def _tiny_comparison(
    seed: int = 0,
    epochs: int = 4,
    recipes: dict[str, RecipeConfig] | None = None,
) -> RecipeComparisonResult:
    words   = [_mock_word("tank", 0), _mock_word("cat", 1)]
    pseudos = [_mock_pseudo(0), _mock_pseudo(1)]
    sem     = _sem_map(words)
    return run_recipe_comparison(
        cfg, words, pseudos, sem,
        mode="mixed-multitask",
        recipes=RECIPES if recipes is None else recipes,
        epochs=epochs,
        base_lr=0.01,
        zero_error_radius=0.0,
        device=torch.device("cpu"),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Always-run: RECIPES / validate_recipe_config
# ---------------------------------------------------------------------------


def test_recipes_contains_required_names():
    required = {"baseline_constant", "frequency_constant", "frequency_paper_lr"}
    assert required.issubset(set(RECIPES))


def test_default_recipes_are_core_three():
    assert set(DEFAULT_RECIPES) == {
        "baseline_constant", "frequency_constant", "frequency_paper_lr",
    }
    assert "zipf_constant" not in DEFAULT_RECIPES
    assert "zipf_constant" in RECIPES


def test_recipe_field_values():
    assert RECIPES["baseline_constant"] == RecipeConfig(
        "baseline_constant", "paper", "mean_active", "none", "constant"
    )
    assert RECIPES["frequency_constant"] == RecipeConfig(
        "frequency_constant", "paper", "mean_active", "frequency", "constant"
    )
    assert RECIPES["frequency_paper_lr"] == RecipeConfig(
        "frequency_paper_lr", "paper", "mean_active", "frequency", "paper"
    )
    assert RECIPES["zipf_constant"] == RecipeConfig(
        "zipf_constant", "paper", "mean_active", "zipf", "constant"
    )


def test_validate_recipe_config_valid():
    for name, recipe in RECIPES.items():
        assert validate_recipe_config(recipe) is None, f"{name} should be valid"


def test_validate_recipe_config_invalid_task_schedule():
    bad = dc_replace(RECIPES["baseline_constant"], task_schedule="bogus")
    err = validate_recipe_config(bad)
    assert err is not None
    assert "task_schedule" in err


def test_validate_recipe_config_invalid_loss_reduction():
    bad = dc_replace(RECIPES["baseline_constant"], loss_reduction="bogus")
    err = validate_recipe_config(bad)
    assert err is not None
    assert "loss_reduction" in err


def test_validate_recipe_config_invalid_frequency_source():
    bad = dc_replace(RECIPES["baseline_constant"], frequency_source="bogus")
    err = validate_recipe_config(bad)
    assert err is not None
    assert "frequency_source" in err


def test_validate_recipe_config_invalid_lr_schedule():
    bad = dc_replace(RECIPES["baseline_constant"], lr_schedule="bogus")
    err = validate_recipe_config(bad)
    assert err is not None
    assert "lr_schedule" in err


# ---------------------------------------------------------------------------
# Always-run: run_recipe_comparison — small toy run
# ---------------------------------------------------------------------------


def test_run_recipe_comparison_returns_all_recipes():
    result = _tiny_comparison()
    assert set(result.results.keys()) == set(RECIPES)


def test_run_recipe_comparison_losses_finite():
    result = _tiny_comparison()
    for name, r in result.results.items():
        assert all(math.isfinite(a) for a in r.epoch_avgs), (
            f"non-finite epoch avg for recipe {name!r}"
        )


def test_run_recipe_comparison_epoch_counts_match():
    result = _tiny_comparison(epochs=5)
    counts = [len(r.epoch_avgs) for r in result.results.values()]
    assert len(set(counts)) == 1
    assert counts[0] == 5


def test_run_recipe_comparison_trials_per_epoch_consistent():
    """All default recipes share task_schedule='paper' -> identical trial counts."""
    result = _tiny_comparison()
    counts = list(result.trials_per_epoch.values())
    assert len(set(counts)) == 1


def test_recipe_comparison_result_is_dataclass():
    fnames = {f.name for f in dc_fields(RecipeComparisonResult)}
    assert "results"          in fnames
    assert "trials_per_epoch" in fnames
    assert "recipes"          in fnames
    assert "weight_stats"     in fnames


def test_run_recipe_comparison_subset_of_recipes():
    subset = {"baseline_constant": RECIPES["baseline_constant"]}
    result = _tiny_comparison(recipes=subset)
    assert set(result.results.keys()) == {"baseline_constant"}
    assert set(result.recipes.keys()) == {"baseline_constant"}
    assert set(result.trials_per_epoch.keys()) == {"baseline_constant"}
    assert set(result.weight_stats.keys()) == {"baseline_constant"}


def test_run_recipe_comparison_weight_stats_neutral_for_none_source():
    result = _tiny_comparison(recipes={"baseline_constant": RECIPES["baseline_constant"]})
    ws = result.weight_stats["baseline_constant"]
    assert ws["min"] == 1.0
    assert ws["mean"] == 1.0
    assert ws["max"] == 1.0
    assert ws["n_words"] == 0.0


# ---------------------------------------------------------------------------
# Always-run: invalid recipe config raises a clear error
# ---------------------------------------------------------------------------


def test_run_recipe_comparison_invalid_recipe_raises():
    bad_recipes = {
        "bad_recipe": dc_replace(RECIPES["baseline_constant"], name="bad_recipe", lr_schedule="bogus"),
    }
    with pytest.raises(ValueError, match="bad_recipe"):
        _tiny_comparison(recipes=bad_recipes)


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
def test_csv_recipe_comparison_finite():
    """Real English NWR data: all recipes give finite losses."""
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
    result  = run_recipe_comparison(
        eng_cfg, words, pseudos, sem_map,
        mode="mixed-multitask",
        recipes=RECIPES,
        epochs=3,
        base_lr=0.01,
        zero_error_radius=0.0,
        device=torch.device("cpu"),
        seed=0,
    )
    for name, r in result.results.items():
        assert all(math.isfinite(a) for a in r.epoch_avgs), (
            f"non-finite epoch avg for recipe {name!r}"
        )
