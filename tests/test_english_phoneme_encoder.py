"""Phase 3b tests: English phoneme inventory loader and one-hot encoder.

Tests are split into two groups:
  - Always-run: no CSV files needed; use mock data
  - CSV-dependent: require specific files in data/raw/nwr_swp/;
    skipped gracefully per-file when absent

The local CSV files are gitignored and must not be committed to the
public repository.
"""
import csv as csv_mod
from pathlib import Path

import pytest
import torch

from lichtheim2.encoding import (
    PhonemeInventory,
    encode_phoneme_sequence,
    load_phoneme_inventory,
    parse_phoneme_sequence,
    validate_phoneme_coverage,
)

# ---------------------------------------------------------------------------
# Data availability — skip markers per file
# ---------------------------------------------------------------------------

DATA_DIR      = Path(__file__).parent.parent / "data" / "raw" / "nwr_swp"
PHONEMES_CSV  = DATA_DIR / "phonemes.csv"
WFE_CSV       = DATA_DIR / "wfe.csv"
SSP_CSV       = DATA_DIR / "ssp.csv"

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
# Helpers
# ---------------------------------------------------------------------------

def _mock_inventory() -> PhonemeInventory:
    symbols = ["AH", "T", "EH"]
    return PhonemeInventory(symbols=symbols, symbol_to_index={s: i for i, s in enumerate(symbols)})


def _read_column(path: Path, col: str) -> list[str]:
    with open(path, newline="") as f:
        return [row[col] for row in csv_mod.DictReader(f)]


def _read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv_mod.DictReader(f))


# ---------------------------------------------------------------------------
# Always-run: parse_phoneme_sequence
# ---------------------------------------------------------------------------

def test_parse_basic():
    assert parse_phoneme_sequence("['AH', 'T', 'EH']") == ["AH", "T", "EH"]


def test_parse_single_element():
    assert parse_phoneme_sequence("['P']") == ["P"]


def test_parse_long_sequence():
    result = parse_phoneme_sequence("['K', 'AH', 'M', 'IH', 'SH', 'AH', 'N', 'ER']")
    assert result == ["K", "AH", "M", "IH", "SH", "AH", "N", "ER"]
    assert len(result) == 8


def test_parse_invalid_type_raises():
    with pytest.raises((ValueError, TypeError)):
        parse_phoneme_sequence(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Always-run: PhonemeInventory behaviour
# ---------------------------------------------------------------------------

def test_inventory_contains():
    inv = _mock_inventory()
    assert "AH" in inv
    assert "ZZ" not in inv


def test_inventory_len():
    inv = _mock_inventory()
    assert len(inv) == 3


# ---------------------------------------------------------------------------
# Always-run: encode_phoneme_sequence (mock inventory)
# ---------------------------------------------------------------------------

def test_encode_shape_and_one_hot():
    inv = _mock_inventory()
    t = encode_phoneme_sequence(["AH", "T", "EH"], inv)
    assert t.shape == (3, 3)
    assert (t.sum(dim=1) == 1.0).all()
    assert t[0, 0] == 1.0  # AH → index 0
    assert t[1, 1] == 1.0  # T  → index 1
    assert t[2, 2] == 1.0  # EH → index 2


def test_encode_single_phoneme():
    inv = _mock_inventory()
    t = encode_phoneme_sequence(["T"], inv)
    assert t.shape == (1, 3)
    assert t[0, 1] == 1.0


def test_encode_unknown_symbol_raises():
    inv = _mock_inventory()
    with pytest.raises(KeyError):
        encode_phoneme_sequence(["ZZ"], inv)


# ---------------------------------------------------------------------------
# Always-run: validate_phoneme_coverage (mock inventory)
# ---------------------------------------------------------------------------

def test_validate_full_coverage():
    inv = _mock_inventory()
    assert validate_phoneme_coverage([["AH", "T"], ["EH"]], inv) == set()


def test_validate_reports_missing():
    inv = _mock_inventory()
    missing = validate_phoneme_coverage([["AH", "ZZ", "NG"]], inv)
    assert "ZZ" in missing
    assert "NG" in missing


# ---------------------------------------------------------------------------
# phonemes.csv tests
# ---------------------------------------------------------------------------

@requires_phonemes
def test_inventory_loads_correctly():
    inv = load_phoneme_inventory(PHONEMES_CSV)
    assert len(inv) > 0
    assert all(isinstance(s, str) for s in inv.symbols)


@requires_phonemes
def test_inventory_no_duplicates():
    inv = load_phoneme_inventory(PHONEMES_CSV)
    assert len(inv.symbols) == len(set(inv.symbols)), "Inventory has duplicate symbols"


@requires_phonemes
def test_inventory_indices_sequential():
    inv = load_phoneme_inventory(PHONEMES_CSV)
    for i, sym in enumerate(inv.symbols):
        assert inv.symbol_to_index[sym] == i


@requires_phonemes
def test_inventory_symbols_have_no_stress_digits():
    """No inventory symbol should contain digit characters (stress markers)."""
    inv = load_phoneme_inventory(PHONEMES_CSV)
    for sym in inv.symbols:
        assert not any(c.isdigit() for c in sym), \
            f"Stress marker found in inventory symbol: {sym!r}"


# ---------------------------------------------------------------------------
# wfe.csv tests
# ---------------------------------------------------------------------------

@requires_phonemes
@requires_wfe
def test_wfe_coverage():
    """All No_Stress phonemes in wfe.csv must be in the inventory."""
    inv = load_phoneme_inventory(PHONEMES_CSV)
    seqs = [parse_phoneme_sequence(v) for v in _read_column(WFE_CSV, "No_Stress")]
    missing = validate_phoneme_coverage(seqs, inv)
    assert missing == set(), f"wfe.csv symbols not in inventory: {missing}"


@requires_wfe
def test_wfe_no_stress_has_no_digits():
    """No_Stress sequences in wfe.csv must not contain stress-marked symbols."""
    for value in _read_column(WFE_CSV, "No_Stress"):
        for sym in parse_phoneme_sequence(value):
            assert not any(c.isdigit() for c in sym), \
                f"Digit in wfe.csv No_Stress symbol: {sym!r}"


@requires_wfe
def test_wfe_sequence_length_matches_length_column():
    """Parsed No_Stress length must match the Length column for all wfe.csv rows."""
    for row in _read_rows(WFE_CSV):
        expected = int(row["Length"])
        actual = len(parse_phoneme_sequence(row["No_Stress"]))
        assert actual == expected, (
            f"wfe.csv row {row.get('', '?')}: No_Stress length {actual} "
            f"!= Length column {expected} for word {row.get('Word', '?')!r}"
        )


# ---------------------------------------------------------------------------
# ssp.csv tests
# ---------------------------------------------------------------------------

@requires_phonemes
@requires_ssp
def test_ssp_coverage():
    """All No_Stress phonemes in ssp.csv must be in the inventory."""
    inv = load_phoneme_inventory(PHONEMES_CSV)
    seqs = [parse_phoneme_sequence(v) for v in _read_column(SSP_CSV, "No_Stress")]
    missing = validate_phoneme_coverage(seqs, inv)
    assert missing == set(), f"ssp.csv symbols not in inventory: {missing}"


@requires_ssp
def test_ssp_no_stress_has_no_digits():
    """No_Stress sequences in ssp.csv must not contain stress-marked symbols."""
    for value in _read_column(SSP_CSV, "No_Stress"):
        for sym in parse_phoneme_sequence(value):
            assert not any(c.isdigit() for c in sym), \
                f"Digit in ssp.csv No_Stress symbol: {sym!r}"


@requires_ssp
def test_ssp_sequence_length_matches_length_column():
    """Parsed No_Stress length must match the Length column for all ssp.csv rows."""
    for row in _read_rows(SSP_CSV):
        expected = int(row["Length"])
        actual = len(parse_phoneme_sequence(row["No_Stress"]))
        assert actual == expected, (
            f"ssp.csv row {row.get('', '?')}: No_Stress length {actual} "
            f"!= Length column {expected}"
        )


# ---------------------------------------------------------------------------
# Encoding from real CSV data
# ---------------------------------------------------------------------------

@requires_phonemes
@requires_wfe
def test_encode_wfe_first_item():
    inv = load_phoneme_inventory(PHONEMES_CSV)
    first_row = _read_rows(WFE_CSV)[0]
    seq = parse_phoneme_sequence(first_row["No_Stress"])
    t = encode_phoneme_sequence(seq, inv)
    assert t.shape == (len(seq), len(inv))
    assert (t.sum(dim=1) == 1.0).all()


@requires_phonemes
@requires_ssp
def test_encode_ssp_first_item():
    inv = load_phoneme_inventory(PHONEMES_CSV)
    first_row = _read_rows(SSP_CSV)[0]
    seq = parse_phoneme_sequence(first_row["No_Stress"])
    t = encode_phoneme_sequence(seq, inv)
    assert t.shape == (len(seq), len(inv))
    assert (t.sum(dim=1) == 1.0).all()


# ---------------------------------------------------------------------------
# english_nwr.yaml config (always-run — no CSV needed)
# ---------------------------------------------------------------------------

def test_english_nwr_config_loads():
    """english_nwr.yaml must load cleanly and carry plausible field values."""
    from lichtheim2.config import load_config
    cfg = load_config(Path(__file__).parent.parent / "configs" / "english_nwr.yaml")
    # sound_input_size is inventory-size-dependent; verify it is a positive int,
    # not a specific hardcoded value (see test_english_nwr_sizes_match_inventory)
    assert cfg.sound_input_size > 0
    assert cfg.motor_output_size == cfg.sound_input_size  # must match
    assert cfg.vATL_size == 50
    # Legacy task fields present but not used at runtime for variable-length trials
    assert cfg.repetition_ticks == 6
    assert cfg.comprehension_ticks == 3
    assert cfg.speaking_ticks == 3


@requires_phonemes
def test_english_nwr_sizes_match_inventory():
    """sound_input_size and motor_output_size must equal the actual inventory size."""
    from lichtheim2.config import load_config
    cfg = load_config(Path(__file__).parent.parent / "configs" / "english_nwr.yaml")
    inv = load_phoneme_inventory(PHONEMES_CSV)
    assert cfg.sound_input_size == len(inv), (
        f"english_nwr.yaml sound_input_size={cfg.sound_input_size} "
        f"but inventory has {len(inv)} symbols"
    )
    assert cfg.motor_output_size == len(inv), (
        f"english_nwr.yaml motor_output_size={cfg.motor_output_size} "
        f"but inventory has {len(inv)} symbols"
    )


def test_variable_length_ignores_cfg_tick_values():
    """run_trial must use T from phon_pattern.shape[0], not cfg.*_ticks.

    english_nwr.yaml has repetition_ticks=6, but a T=5 phoneme tensor must
    produce 2*5=10 TickResults — not 2*6=12.
    """
    from lichtheim2.config import load_config
    from lichtheim2.model import Lichtheim2Model
    from lichtheim2.tasks import Task

    cfg = load_config(Path(__file__).parent.parent / "configs" / "english_nwr.yaml")
    model = Lichtheim2Model(cfg)

    T = 5  # intentionally different from cfg.repetition_ticks (6)
    phon = torch.zeros(T, cfg.sound_input_size)
    sem  = torch.zeros(cfg.vATL_size)

    rep  = model.run_trial(Task.REPETITION,   phon, sem, cfg)
    comp = model.run_trial(Task.COMPREHENSION, phon, sem, cfg)
    spk  = model.run_trial(Task.SPEAKING,     phon, sem, cfg)

    assert len(rep)  == 2 * T, f"Expected {2*T}, got {len(rep)} (cfg.repetition_ticks={cfg.repetition_ticks})"
    assert len(comp) == T,     f"Expected {T}, got {len(comp)}"
    assert len(spk)  == T,     f"Expected {T}, got {len(spk)}"


# ---------------------------------------------------------------------------
# Integration: encoded phoneme tensor through run_trial()
# ---------------------------------------------------------------------------

@requires_all_csvs
def test_integration_wfe_item_through_run_trial():
    """Encode a wfe.csv item and pass it through run_trial() for all three tasks."""
    from lichtheim2.config import load_config
    from lichtheim2.model import Lichtheim2Model
    from lichtheim2.tasks import Task

    inv = load_phoneme_inventory(PHONEMES_CSV)
    cfg = load_config(Path(__file__).parent.parent / "configs" / "english_nwr.yaml")

    first_row = _read_rows(WFE_CSV)[0]
    seq  = parse_phoneme_sequence(first_row["No_Stress"])
    phon = encode_phoneme_sequence(seq, inv)       # (T, 40)
    sem  = torch.zeros(cfg.vATL_size)              # dummy semantic vector
    T    = phon.shape[0]

    model = Lichtheim2Model(cfg)
    rep  = model.run_trial(Task.REPETITION,   phon, sem, cfg)
    comp = model.run_trial(Task.COMPREHENSION, phon, sem, cfg)
    spk  = model.run_trial(Task.SPEAKING,     phon, sem, cfg)

    assert len(rep)  == 2 * T, f"Repetition: expected {2*T} ticks, got {len(rep)}"
    assert len(comp) == T,     f"Comprehension: expected {T} ticks, got {len(comp)}"
    assert len(spk)  == T,     f"Speaking: expected {T} ticks, got {len(spk)}"
