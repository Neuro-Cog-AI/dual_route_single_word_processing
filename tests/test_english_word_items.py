"""Phase 3b tests for English word and pseudoword item loaders.

Tests are split into:
  - Always-run: use temporary mock CSVs created inside the test (tmp_path fixture);
    no private local data required; must pass in any environment.
  - CSV-dependent: require real files in data/raw/nwr_swp/; skipped if absent.

The local raw CSVs are gitignored and must not be committed.
"""
from pathlib import Path

import pytest
import torch

from lichtheim2.data import (
    PseudowordItem,
    WordItem,
    load_pseudoword_items,
    load_word_items,
)
from lichtheim2.encoding import PhonemeInventory

# ---------------------------------------------------------------------------
# Data availability
# ---------------------------------------------------------------------------

DATA_DIR     = Path(__file__).parent.parent / "data" / "raw" / "nwr_swp"
PHONEMES_CSV = DATA_DIR / "phonemes.csv"
WFE_CSV      = DATA_DIR / "wfe.csv"
SSP_CSV      = DATA_DIR / "ssp.csv"

requires_phonemes = pytest.mark.skipif(
    not PHONEMES_CSV.exists(), reason="phonemes.csv not available locally"
)
requires_wfe = pytest.mark.skipif(
    not WFE_CSV.exists(), reason="wfe.csv not available locally"
)
requires_ssp = pytest.mark.skipif(
    not SSP_CSV.exists(), reason="ssp.csv not available locally"
)
requires_all_csvs = pytest.mark.skipif(
    not all(p.exists() for p in (PHONEMES_CSV, WFE_CSV, SSP_CSV)),
    reason="one or more CSV files not available locally",
)

# ---------------------------------------------------------------------------
# Mock inventory — used in all always-run tests
# ---------------------------------------------------------------------------

_MOCK_SYMBOLS = ["AH", "T", "EH", "N", "D", "IH", "NG", "K", "M", "P"]
_MOCK_INV = PhonemeInventory(
    symbols=_MOCK_SYMBOLS,
    symbol_to_index={s: i for i, s in enumerate(_MOCK_SYMBOLS)},
)

# ---------------------------------------------------------------------------
# Mock CSV content helpers
# ---------------------------------------------------------------------------

_WFE_HEADER = ",Word,Condition,Lexicality,Morphology,Frequency,Length,Zipf_Frequency,No_Stress,Part of Speech"

_SSP_HEADER = ",No_Stress,Sonority,Type,Length"


def _write_mock_wfe(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join([_WFE_HEADER] + rows) + "\n")


def _write_mock_ssp(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join([_SSP_HEADER] + rows) + "\n")


# ---------------------------------------------------------------------------
# Always-run: WordItem loader with mock CSV
# ---------------------------------------------------------------------------


def test_load_word_items_from_mock_csv(tmp_path):
    csv_file = tmp_path / "mock_wfe.csv"
    _write_mock_wfe(csv_file, [
        "0,attend,MOCK,real,simple,0.001,4,4.1,\"['AH', 'T', 'EH', 'N']\",VERB",
        "1,dent,MOCK,real,simple,0.002,3,3.5,\"['D', 'EH', 'N']\",NOUN",
    ])
    items = load_word_items(csv_file, _MOCK_INV)
    assert len(items) == 2
    assert items[0].word == "attend"
    assert items[1].word == "dent"


def test_word_item_tensor_shape_mock(tmp_path):
    csv_file = tmp_path / "wfe.csv"
    _write_mock_wfe(csv_file, [
        "0,attend,MOCK,real,simple,0.001,4,4.1,\"['AH', 'T', 'EH', 'N']\",VERB",
    ])
    items = load_word_items(csv_file, _MOCK_INV)
    item = items[0]
    assert item.phon_tensor.shape == (item.length, len(_MOCK_INV))
    assert item.phon_tensor.shape == (4, len(_MOCK_SYMBOLS))


def test_word_item_tensor_is_one_hot_mock(tmp_path):
    csv_file = tmp_path / "wfe.csv"
    _write_mock_wfe(csv_file, [
        "0,tend,MOCK,real,simple,0.001,3,3.8,\"['T', 'EH', 'N']\",VERB",
    ])
    items = load_word_items(csv_file, _MOCK_INV)
    t = items[0].phon_tensor
    assert (t.sum(dim=1) == 1.0).all()


def test_word_item_source_and_row_index_mock(tmp_path):
    csv_file = tmp_path / "wfe.csv"
    _write_mock_wfe(csv_file, [
        "0,attend,MOCK,real,simple,0.001,4,4.1,\"['AH', 'T', 'EH', 'N']\",VERB",
        "1,dent,MOCK,real,simple,0.002,3,3.5,\"['D', 'EH', 'N']\",NOUN",
    ])
    items = load_word_items(csv_file, _MOCK_INV)
    assert items[0].source == "wfe"
    assert items[0].row_index == 0
    assert items[1].row_index == 1


def test_word_item_optional_fields_none_when_blank(tmp_path):
    csv_file = tmp_path / "wfe.csv"
    _write_mock_wfe(csv_file, [
        "0,tend,MOCK,real,,,,\"['T', 'EH', 'N']\",",  # Frequency, Zipf blank; POS blank
    ])
    # Length is required — rewrite row with Length
    csv_file.write_text(
        _WFE_HEADER + "\n"
        "0,tend,MOCK,real,,,3,,\"['T', 'EH', 'N']\",\n"
    )
    items = load_word_items(csv_file, _MOCK_INV)
    item = items[0]
    assert item.frequency is None
    assert item.zipf_frequency is None
    assert item.part_of_speech is None


def test_word_item_length_mismatch_raises(tmp_path):
    csv_file = tmp_path / "wfe.csv"
    # Length says 5 but No_Stress has 3 phonemes
    _write_mock_wfe(csv_file, [
        "0,attend,MOCK,real,simple,0.001,5,4.1,\"['AH', 'T', 'EH']\",VERB",
    ])
    with pytest.raises(ValueError, match="No_Stress length"):
        load_word_items(csv_file, _MOCK_INV)


def test_word_item_missing_required_field_raises(tmp_path):
    csv_file = tmp_path / "wfe.csv"
    # No_Stress is blank
    _write_mock_wfe(csv_file, [
        "0,attend,MOCK,real,simple,0.001,4,4.1,,VERB",
    ])
    with pytest.raises(ValueError, match="No_Stress"):
        load_word_items(csv_file, _MOCK_INV)


def test_word_item_missing_word_field_raises(tmp_path):
    csv_file = tmp_path / "wfe.csv"
    # Word is blank
    _write_mock_wfe(csv_file, [
        "0,,MOCK,real,simple,0.001,3,3.5,\"['T', 'EH', 'N']\",NOUN",
    ])
    with pytest.raises(ValueError, match="Word"):
        load_word_items(csv_file, _MOCK_INV)


# ---------------------------------------------------------------------------
# Always-run: PseudowordItem loader with mock CSV
# ---------------------------------------------------------------------------


def test_load_pseudoword_items_from_mock_csv(tmp_path):
    csv_file = tmp_path / "ssp.csv"
    _write_mock_ssp(csv_file, [
        "0,\"['P', 'T', 'AH']\",0,CCV,3",
        "1,\"['AH', 'T']\",1,VC,2",
    ])
    items = load_pseudoword_items(csv_file, _MOCK_INV)
    assert len(items) == 2
    assert items[0].syllable_type == "CCV"
    assert items[1].length == 2


def test_pseudoword_tensor_shape_mock(tmp_path):
    csv_file = tmp_path / "ssp.csv"
    _write_mock_ssp(csv_file, ["0,\"['P', 'T', 'AH']\",0,CCV,3"])
    items = load_pseudoword_items(csv_file, _MOCK_INV)
    item = items[0]
    assert item.phon_tensor.shape == (item.length, len(_MOCK_INV))
    assert (item.phon_tensor.sum(dim=1) == 1.0).all()


def test_pseudoword_lexicality_mock(tmp_path):
    csv_file = tmp_path / "ssp.csv"
    _write_mock_ssp(csv_file, ["0,\"['P', 'AH']\",0,VC,2"])
    items = load_pseudoword_items(csv_file, _MOCK_INV)
    assert items[0].lexicality == "pseudoword"


def test_pseudoword_source_and_row_index_mock(tmp_path):
    csv_file = tmp_path / "ssp.csv"
    _write_mock_ssp(csv_file, [
        "0,\"['P', 'T', 'AH']\",0,CCV,3",
        "1,\"['AH', 'T']\",1,VC,2",
    ])
    items = load_pseudoword_items(csv_file, _MOCK_INV)
    assert items[0].source == "ssp"
    assert items[0].row_index == 0
    assert items[1].row_index == 1


def test_pseudoword_length_mismatch_raises(tmp_path):
    csv_file = tmp_path / "ssp.csv"
    # Length says 5 but No_Stress has 2 phonemes
    _write_mock_ssp(csv_file, ["0,\"['P', 'AH']\",0,VC,5"])
    with pytest.raises(ValueError, match="No_Stress length"):
        load_pseudoword_items(csv_file, _MOCK_INV)


def test_pseudoword_missing_no_stress_raises(tmp_path):
    csv_file = tmp_path / "ssp.csv"
    _write_mock_ssp(csv_file, ["0,,0,CCV,3"])
    with pytest.raises(ValueError, match="No_Stress"):
        load_pseudoword_items(csv_file, _MOCK_INV)


# ---------------------------------------------------------------------------
# CSV-dependent: real local data
# ---------------------------------------------------------------------------


@requires_phonemes
@requires_wfe
def test_load_word_items_returns_nonempty():
    from lichtheim2.encoding import load_phoneme_inventory
    inv   = load_phoneme_inventory(PHONEMES_CSV)
    items = load_word_items(WFE_CSV, inv)
    assert len(items) > 0


@requires_phonemes
@requires_wfe
def test_word_item_tensor_shape():
    from lichtheim2.encoding import load_phoneme_inventory
    inv   = load_phoneme_inventory(PHONEMES_CSV)
    items = load_word_items(WFE_CSV, inv)
    N = len(inv)
    for item in items:
        assert item.phon_tensor.shape == (item.length, N), (
            f"{item.word!r}: expected ({item.length}, {N}), "
            f"got {tuple(item.phon_tensor.shape)}"
        )


@requires_wfe
def test_word_item_length_matches_phonemes():
    # No inventory needed — just checks No_Stress length vs Length column
    from lichtheim2.encoding import PhonemeInventory, encode_phoneme_sequence
    # Use a passthrough inventory (symbol → index via a dict built at runtime)
    import csv as csv_mod
    from lichtheim2.encoding import parse_phoneme_sequence

    with open(WFE_CSV, newline="") as f:
        rows = list(csv_mod.DictReader(f))
    for row in rows:
        seq = parse_phoneme_sequence(row["No_Stress"])
        assert len(seq) == int(row["Length"]), (
            f"Word {row.get('Word')!r}: No_Stress length {len(seq)} "
            f"!= Length column {row['Length']}"
        )


@requires_wfe
def test_word_item_required_fields():
    from lichtheim2.encoding import load_phoneme_inventory, PhonemeInventory
    # Build a minimal passthrough inventory from all phonemes in the file
    import csv as csv_mod
    from lichtheim2.encoding import parse_phoneme_sequence
    all_phonemes: list[str] = []
    with open(WFE_CSV, newline="") as f:
        for row in csv_mod.DictReader(f):
            all_phonemes.extend(parse_phoneme_sequence(row["No_Stress"]))
    symbols = list(dict.fromkeys(all_phonemes))  # unique, order-preserving
    inv = PhonemeInventory(symbols=symbols, symbol_to_index={s: i for i, s in enumerate(symbols)})

    items = load_word_items(WFE_CSV, inv)
    for item in items:
        assert item.word
        assert item.phonemes
        assert item.lexicality
        assert item.length > 0


@requires_phonemes
@requires_ssp
def test_load_pseudoword_items_returns_nonempty():
    from lichtheim2.encoding import load_phoneme_inventory
    inv   = load_phoneme_inventory(PHONEMES_CSV)
    items = load_pseudoword_items(SSP_CSV, inv)
    assert len(items) > 0


@requires_phonemes
@requires_ssp
def test_pseudoword_tensor_shape():
    from lichtheim2.encoding import load_phoneme_inventory
    inv   = load_phoneme_inventory(PHONEMES_CSV)
    items = load_pseudoword_items(SSP_CSV, inv)
    N = len(inv)
    for item in items:
        assert item.phon_tensor.shape == (item.length, N)


@requires_ssp
def test_pseudoword_length_matches_phonemes():
    import csv as csv_mod
    from lichtheim2.encoding import parse_phoneme_sequence
    with open(SSP_CSV, newline="") as f:
        for row in csv_mod.DictReader(f):
            seq = parse_phoneme_sequence(row["No_Stress"])
            assert len(seq) == int(row["Length"])


@requires_ssp
def test_pseudoword_lexicality_field():
    from lichtheim2.encoding import load_phoneme_inventory, PhonemeInventory, parse_phoneme_sequence
    import csv as csv_mod
    all_phonemes: list[str] = []
    with open(SSP_CSV, newline="") as f:
        for row in csv_mod.DictReader(f):
            all_phonemes.extend(parse_phoneme_sequence(row["No_Stress"]))
    symbols = list(dict.fromkeys(all_phonemes))
    inv = PhonemeInventory(symbols=symbols, symbol_to_index={s: i for i, s in enumerate(symbols)})
    items = load_pseudoword_items(SSP_CSV, inv)
    for item in items:
        assert item.lexicality == "pseudoword"


# ---------------------------------------------------------------------------
# Integration: run_trial with real items
# ---------------------------------------------------------------------------


@requires_all_csvs
def test_word_item_repetition_run_trial():
    """Encode first wfe.csv item and run REPETITION through the model."""
    from lichtheim2.config import load_config
    from lichtheim2.encoding import load_phoneme_inventory
    from lichtheim2.model import Lichtheim2Model
    from lichtheim2.tasks import Task

    inv   = load_phoneme_inventory(PHONEMES_CSV)
    cfg   = load_config(Path(__file__).parent.parent / "configs" / "english_nwr.yaml")
    model = Lichtheim2Model(cfg)
    items = load_word_items(WFE_CSV, inv)

    item = items[0]
    sem  = torch.zeros(cfg.vATL_size)
    T    = item.length

    results = model.run_trial(Task.REPETITION, item.phon_tensor, sem, cfg)
    assert len(results) == 2 * T


@requires_all_csvs
def test_pseudoword_repetition_run_trial():
    """Encode first ssp.csv item and run REPETITION through the model.

    Pseudowords are repetition-only — no semantic vectors assigned yet (D17).
    """
    from lichtheim2.config import load_config
    from lichtheim2.encoding import load_phoneme_inventory
    from lichtheim2.model import Lichtheim2Model
    from lichtheim2.tasks import Task

    inv   = load_phoneme_inventory(PHONEMES_CSV)
    cfg   = load_config(Path(__file__).parent.parent / "configs" / "english_nwr.yaml")
    model = Lichtheim2Model(cfg)
    items = load_pseudoword_items(SSP_CSV, inv)

    item = items[0]
    sem  = torch.zeros(cfg.vATL_size)
    T    = item.length

    results = model.run_trial(Task.REPETITION, item.phon_tensor, sem, cfg)
    assert len(results) == 2 * T
