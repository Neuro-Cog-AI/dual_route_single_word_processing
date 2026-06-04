from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch

from lichtheim2.encoding import (
    PhonemeInventory,
    encode_phoneme_sequence,
    parse_phoneme_sequence,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WordItem:
    """A real-word item loaded from wfe.csv with its one-hot phoneme tensor."""

    word: str
    phonemes: list[str]          # from No_Stress column (stress-stripped)
    phon_tensor: torch.Tensor    # shape (T, N); T = length, N = inventory size
    lexicality: str              # e.g. "real"
    length: int                  # equals len(phonemes) and phon_tensor.shape[0]
    frequency: float | None      # raw corpus frequency; None if missing
    zipf_frequency: float | None # Zipf-scale frequency
    part_of_speech: str | None   # "Part of Speech" column (space in CSV header)
    condition: str | None        # e.g. "RLCH"
    morphology: str | None       # e.g. "complex"
    source: str | None = "wfe"   # traceability: source file identifier
    row_index: int | None = None # 0-based row index within the source CSV


@dataclass
class PseudowordItem:
    """A pseudoword/nonword item loaded from ssp.csv with its one-hot phoneme tensor."""

    phonemes: list[str]          # from No_Stress column
    phon_tensor: torch.Tensor    # shape (T, N)
    length: int                  # equals len(phonemes) and phon_tensor.shape[0]
    sonority: int | None         # sonority profile metric
    syllable_type: str | None    # e.g. CCV, VCC (from Type column;
                                 # renamed to avoid shadowing Python built-in type)
    lexicality: str = "pseudoword"
    source: str | None = "ssp"
    row_index: int | None = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _opt_str(value: str) -> str | None:
    stripped = value.strip()
    return stripped if stripped else None


def _opt_float(value: str) -> float | None:
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return None


def _opt_int(value: str) -> int | None:
    try:
        return int(value.strip())
    except (ValueError, AttributeError):
        return None


def _require(value: str, field_name: str, row_index: int) -> str:
    """Return stripped value, or raise ValueError naming the missing field and row."""
    stripped = value.strip() if isinstance(value, str) else ""
    if not stripped:
        raise ValueError(
            f"Row {row_index}: required field '{field_name}' is missing or blank"
        )
    return stripped


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_word_items(
    wfe_path: str | Path,
    inventory: PhonemeInventory,
) -> list[WordItem]:
    """Load all rows from wfe.csv as WordItems with encoded phoneme tensors.

    Required CSV columns: Word, Lexicality, No_Stress, Length.
    Optional: Frequency, Zipf_Frequency, Part of Speech, Condition, Morphology.

    Raises:
        ValueError: if a required field is missing/blank, or if the parsed
                    No_Stress sequence length does not match the Length column.
        KeyError: if a phoneme symbol is not in the inventory (run
                  validate_phoneme_coverage first to surface gaps).
    """
    items: list[WordItem] = []
    with open(wfe_path, newline="") as f:
        for row_index, row in enumerate(csv.DictReader(f)):
            word       = _require(row.get("Word", ""),       "Word",       row_index)
            lexicality = _require(row.get("Lexicality", ""), "Lexicality", row_index)
            raw_length = _require(row.get("Length", ""),     "Length",     row_index)
            raw_ns     = _require(row.get("No_Stress", ""),  "No_Stress",  row_index)

            phonemes = parse_phoneme_sequence(raw_ns)
            length   = int(raw_length)

            if len(phonemes) != length:
                raise ValueError(
                    f"Row {row_index}: No_Stress length {len(phonemes)} "
                    f"!= Length column {length} for word {word!r}"
                )

            phon_tensor = encode_phoneme_sequence(phonemes, inventory)

            items.append(WordItem(
                word=word,
                phonemes=phonemes,
                phon_tensor=phon_tensor,
                lexicality=lexicality,
                length=length,
                frequency=_opt_float(row.get("Frequency", "")),
                zipf_frequency=_opt_float(row.get("Zipf_Frequency", "")),
                part_of_speech=_opt_str(row.get("Part of Speech", "")),
                condition=_opt_str(row.get("Condition", "")),
                morphology=_opt_str(row.get("Morphology", "")),
                source="wfe",
                row_index=row_index,
            ))
    return items


def load_pseudoword_items(
    ssp_path: str | Path,
    inventory: PhonemeInventory,
) -> list[PseudowordItem]:
    """Load all rows from ssp.csv as PseudowordItems with encoded phoneme tensors.

    Required CSV columns: No_Stress, Length.
    Optional: Sonority, Type.

    Raises:
        ValueError: if a required field is missing/blank, or if the parsed
                    No_Stress sequence length does not match the Length column.
        KeyError: if a phoneme symbol is not in the inventory.
    """
    items: list[PseudowordItem] = []
    with open(ssp_path, newline="") as f:
        for row_index, row in enumerate(csv.DictReader(f)):
            raw_ns     = _require(row.get("No_Stress", ""), "No_Stress", row_index)
            raw_length = _require(row.get("Length", ""),    "Length",    row_index)

            phonemes = parse_phoneme_sequence(raw_ns)
            length   = int(raw_length)

            if len(phonemes) != length:
                raise ValueError(
                    f"Row {row_index}: No_Stress length {len(phonemes)} "
                    f"!= Length column {length}"
                )

            phon_tensor = encode_phoneme_sequence(phonemes, inventory)

            items.append(PseudowordItem(
                phonemes=phonemes,
                phon_tensor=phon_tensor,
                length=length,
                sonority=_opt_int(row.get("Sonority", "")),
                syllable_type=_opt_str(row.get("Type", "")),
                lexicality="pseudoword",
                source="ssp",
                row_index=row_index,
            ))
    return items
