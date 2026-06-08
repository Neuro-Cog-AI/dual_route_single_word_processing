"""Phase 3c-2 tests: real-data repetition training pipeline.

Structure:
- Always-run (no CSV required): use toy config + synthetic tensors or tmp CSVs.
- CSV-dependent: skip gracefully if private data/raw/nwr_swp is absent.
  Each test checks for the specific files it actually needs.
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import pytest
import torch
import torch.optim as optim

# Allow running with PYTHONPATH=src or from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lichtheim2.config import load_config
from lichtheim2.model import Lichtheim2Model
from lichtheim2.tasks import Task
from lichtheim2.trainer import train_step
from lichtheim2.trials import make_repetition_trial

# Import helpers from the script under test
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from train_repetition_real_data import (
    build_repetition_trials,
    load_repetition_items,
    sample_items,
    train_repetition_epochs,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR  = REPO_ROOT / "data" / "raw" / "nwr_swp"
PHONEMES_CSV = DATA_DIR / "phonemes.csv"
WFE_CSV      = DATA_DIR / "wfe.csv"
SSP_CSV      = DATA_DIR / "ssp.csv"

cfg = load_config(REPO_ROOT / "configs" / "toy.yaml")   # sound=5, motor=5


def _toy_model() -> Lichtheim2Model:
    torch.manual_seed(0)
    return Lichtheim2Model(cfg)


def _synthetic_items(n: int = 3) -> list[tuple[torch.Tensor, str, int | None]]:
    """Return n (phon_tensor, label, item_id) tuples using toy sound_input_size=5."""
    torch.manual_seed(0)
    return [
        (torch.rand(t + 2, cfg.sound_input_size), f"item_{i}", i)
        for i, t in enumerate(range(n))
    ]


# ---------------------------------------------------------------------------
# Always-run: trial construction
# ---------------------------------------------------------------------------


def test_make_repetition_trial_shapes():
    phon = torch.rand(3, cfg.sound_input_size)
    trial = make_repetition_trial(phon, cfg.motor_output_size)
    assert trial.task == Task.REPETITION
    assert trial.n_ticks == 6
    assert trial.motor_targets.shape == (6, cfg.motor_output_size)
    assert trial.motor_loss_mask.shape == (6,)
    assert trial.semantic_targets is None
    assert trial.semantic_loss_mask is None
    assert trial.sem_input is None


def test_motor_targets_structure():
    phon = torch.rand(3, cfg.sound_input_size)
    trial = make_repetition_trial(phon, cfg.motor_output_size)
    # Input phase (ticks 0..T-1) → motor target must be all zeros
    assert trial.motor_targets[:3].eq(0.0).all()
    # Output phase (ticks T..2T-1) → motor target must reproduce phon_tensor
    assert torch.allclose(trial.motor_targets[3:], phon)


def test_make_repetition_trial_preserves_item_id_and_label():
    phon = torch.rand(2, cfg.sound_input_size)
    trial = make_repetition_trial(phon, cfg.motor_output_size, item_id=42, label="cat")
    assert trial.item_id == 42
    assert trial.label == "cat"


# ---------------------------------------------------------------------------
# Always-run: sample_items
# ---------------------------------------------------------------------------


def test_sample_items_respects_max():
    items = _synthetic_items(10)
    rng = random.Random(0)
    result = sample_items(items, max_items=3, rng=rng)
    assert len(result) == 3


def test_sample_items_no_max_returns_all():
    items = _synthetic_items(5)
    rng = random.Random(0)
    result = sample_items(items, max_items=None, rng=rng)
    assert len(result) == 5


def test_sample_items_max_larger_than_list_returns_all():
    items = _synthetic_items(4)
    rng = random.Random(0)
    result = sample_items(items, max_items=100, rng=rng)
    assert len(result) == 4


# ---------------------------------------------------------------------------
# Always-run: build_repetition_trials
# ---------------------------------------------------------------------------


def test_build_repetition_trials_count_and_shapes():
    items = _synthetic_items(4)
    trials = build_repetition_trials(items, cfg.motor_output_size)
    assert len(trials) == 4
    for (pt, lbl, iid), trial in zip(items, trials):
        T = pt.shape[0]
        assert trial.n_ticks == 2 * T
        assert trial.motor_targets.shape == (2 * T, cfg.motor_output_size)
        assert trial.motor_loss_mask.shape == (2 * T,)
        assert trial.task == Task.REPETITION


def test_build_repetition_trials_preserves_item_id_and_label():
    items = [(torch.rand(3, cfg.sound_input_size), "word:cat", 7)]
    trials = build_repetition_trials(items, cfg.motor_output_size)
    assert trials[0].item_id == 7
    assert trials[0].label == "word:cat"


# ---------------------------------------------------------------------------
# Always-run: train_step and train_repetition_epochs
# ---------------------------------------------------------------------------


def test_train_step_produces_finite_loss():
    model = _toy_model()
    opt = optim.SGD(model.parameters(), lr=0.1)
    phon = torch.rand(3, cfg.sound_input_size)
    trial = make_repetition_trial(phon, cfg.motor_output_size)
    loss = train_step(model, trial, opt, cfg, zero_error_radius=0.0, device="cpu")
    assert isinstance(loss, float)
    assert loss == loss and loss != float("inf")  # finite


def test_full_mini_epoch_all_losses_finite():
    model = _toy_model()
    opt = optim.SGD(model.parameters(), lr=0.1)
    items = _synthetic_items(3)
    trials = build_repetition_trials(items, cfg.motor_output_size)
    rng = random.Random(42)
    all_losses = train_repetition_epochs(
        model, trials, opt, cfg,
        epochs=2, zero_error_radius=0.1, device=torch.device("cpu"), rng=rng,
    )
    assert len(all_losses) == 2
    for epoch_losses in all_losses:
        assert len(epoch_losses) == 3
        for l in epoch_losses:
            assert l == l and l != float("inf")  # finite


# ---------------------------------------------------------------------------
# Always-run: mock CSV round-trip
# ---------------------------------------------------------------------------


def _write_mock_csvs(tmp_path: Path) -> Path:
    """Write minimal phonemes.csv and wfe.csv to tmp_path."""
    # phonemes.csv: 5 symbols required to match toy sound_input_size=5
    phonemes_path = tmp_path / "phonemes.csv"
    with open(phonemes_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Phoneme"])
        w.writeheader()
        for sym in ["AE", "T", "K", "AH", "N"]:
            w.writerow({"Phoneme": sym})

    # wfe.csv: 1 real word using only those phonemes
    wfe_path = tmp_path / "wfe.csv"
    with open(wfe_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["Word", "Lexicality", "No_Stress", "Length",
                        "Frequency", "Zipf_Frequency", "Part of Speech",
                        "Condition", "Morphology"],
        )
        w.writeheader()
        w.writerow({
            "Word": "tank",
            "Lexicality": "real",
            "No_Stress": "['T', 'AE', 'N', 'K']",
            "Length": "4",
            "Frequency": "12.5",
            "Zipf_Frequency": "3.2",
            "Part of Speech": "noun",
            "Condition": "",
            "Morphology": "simple",
        })
    return tmp_path


def test_mock_csv_word_load_and_trial(tmp_path):
    data_dir = _write_mock_csvs(tmp_path)
    items = load_repetition_items(data_dir, source="words")
    assert len(items) == 1
    phon_tensor, label, item_id = items[0]
    assert label == "word:tank"
    assert phon_tensor.shape == (4, 5)   # T=4 phonemes, N=5 inventory symbols
    trials = build_repetition_trials(items, motor_size=5)
    assert trials[0].motor_targets.shape == (8, 5)  # 2T ticks


def test_mock_csv_labels_have_source_prefix(tmp_path):
    data_dir = _write_mock_csvs(tmp_path)
    items = load_repetition_items(data_dir, source="words")
    for _, label, _ in items:
        assert label.startswith("word:"), f"Label missing source prefix: {label!r}"


# ---------------------------------------------------------------------------
# CSV-dependent integration tests
# ---------------------------------------------------------------------------

_wfe_files_available = pytest.mark.skipif(
    not (PHONEMES_CSV.exists() and WFE_CSV.exists()),
    reason="private CSV data (phonemes.csv + wfe.csv) not present",
)
_ssp_files_available = pytest.mark.skipif(
    not (PHONEMES_CSV.exists() and SSP_CSV.exists()),
    reason="private CSV data (phonemes.csv + ssp.csv) not present",
)
_all_csv_available = pytest.mark.skipif(
    not (PHONEMES_CSV.exists() and WFE_CSV.exists() and SSP_CSV.exists()),
    reason="private CSV data (phonemes.csv + wfe.csv + ssp.csv) not present",
)


@_wfe_files_available
def test_csv_load_word_items_shapes():
    items = load_repetition_items(DATA_DIR, source="words")
    assert len(items) > 0
    for pt, label, _ in items[:3]:
        assert pt.ndim == 2
        assert pt.shape[0] > 0
        assert pt.shape[1] == 39, f"Expected 39 phoneme dims, got {pt.shape[1]}"
        assert label.startswith("word:")


@_ssp_files_available
def test_csv_load_pseudoword_items_shapes():
    items = load_repetition_items(DATA_DIR, source="pseudowords")
    assert len(items) > 0
    for pt, label, _ in items[:3]:
        assert pt.shape[1] == 39
        assert label.startswith("pseudo:")


@_wfe_files_available
def test_csv_train_step_real_word():
    from lichtheim2.config import load_config as _lc
    eng_cfg = _lc(REPO_ROOT / "configs" / "english_nwr.yaml")
    model = Lichtheim2Model(eng_cfg)
    opt = optim.SGD(model.parameters(), lr=0.1)
    items = load_repetition_items(DATA_DIR, source="words")
    trials = build_repetition_trials(items[:1], eng_cfg.motor_output_size)
    loss = train_step(model, trials[0], opt, eng_cfg,
                      zero_error_radius=0.1, device="cpu")
    assert loss == loss and loss != float("inf")


@_all_csv_available
def test_csv_mixed_source_sampling():
    rng = random.Random(0)
    items = load_repetition_items(DATA_DIR, source="mixed")
    sampled = sample_items(items, max_items=5, rng=rng)
    assert len(sampled) == 5
    # Both source prefixes should appear across the full unsampled list
    labels = [lbl for _, lbl, _ in items]
    assert any(lbl.startswith("word:") for lbl in labels)
    assert any(lbl.startswith("pseudo:") for lbl in labels)
