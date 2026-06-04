from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class PhonemeInventory:
    """Ordered phoneme inventory loaded from phonemes.csv.

    The index of each symbol in `symbols` is its one-hot dimension.
    `sound_input_size` for the English/NWR config equals `len(inventory)`,
    but this value is provisional until coverage validation passes (see D18
    in docs/open_questions.md).
    """

    symbols: list[str]               # ordered; row order from phonemes.csv
    symbol_to_index: dict[str, int]  # symbol → 0-based index

    def __len__(self) -> int:
        return len(self.symbols)

    def __contains__(self, symbol: object) -> bool:
        return symbol in self.symbol_to_index


def load_phoneme_inventory(path: str | Path) -> PhonemeInventory:
    """Load the phoneme inventory from phonemes.csv.

    Uses the 'Phoneme' column as the canonical symbol (no stress markers).
    Row order in phonemes.csv determines one-hot indices and must be stable.

    Args:
        path: path to phonemes.csv

    Returns:
        PhonemeInventory with ordered symbols and a symbol-to-index mapping.
    """
    symbols: list[str] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            symbols.append(row["Phoneme"])
    return PhonemeInventory(
        symbols=symbols,
        symbol_to_index={sym: i for i, sym in enumerate(symbols)},
    )


def parse_phoneme_sequence(value: str) -> list[str]:
    """Parse a Python-list-like string from the No_Stress CSV column.

    The No_Stress column stores sequences as Python-style single-quoted list
    strings, e.g. "['AH', 'T', 'EH', 'N']".  ast.literal_eval handles this;
    json.loads does not (requires double quotes).

    Args:
        value: raw string from the No_Stress column

    Returns:
        list of phoneme symbol strings

    Raises:
        ValueError: if value is not a string or cannot be parsed as a list.
    """
    if not isinstance(value, str):
        raise ValueError(f"Expected a string, got {type(value)}: {value!r}")
    result = ast.literal_eval(value)
    if not isinstance(result, list):
        raise ValueError(f"Expected a list, got {type(result)}: {result!r}")
    return result


def validate_phoneme_coverage(
    sequences: list[list[str]],
    inventory: PhonemeInventory,
) -> set[str]:
    """Return the set of symbols present in sequences but absent from inventory.

    An empty set means all symbols are covered and encoding can proceed.
    Should be called separately for wfe.csv and ssp.csv before training.

    Args:
        sequences: parsed No_Stress phoneme sequences
        inventory: PhonemeInventory from load_phoneme_inventory()

    Returns:
        Set of missing symbols (empty if coverage is complete).
    """
    missing: set[str] = set()
    for seq in sequences:
        for sym in seq:
            if sym not in inventory:
                missing.add(sym)
    return missing


def encode_phoneme_sequence(
    sequence: list[str],
    inventory: PhonemeInventory,
) -> torch.Tensor:
    """Encode a phoneme sequence as a one-hot float tensor.

    Args:
        sequence: list of phoneme symbols (from No_Stress column)
        inventory: PhonemeInventory from load_phoneme_inventory()

    Returns:
        Tensor of shape (T, N) where T = len(sequence), N = len(inventory).
        Row t is a one-hot vector with a 1.0 at inventory.symbol_to_index[sequence[t]].

    Raises:
        KeyError: if any symbol in sequence is not in the inventory.
                  Call validate_phoneme_coverage first to surface missing symbols.
    """
    T = len(sequence)
    N = len(inventory)
    tensor = torch.zeros(T, N)
    for t, sym in enumerate(sequence):
        tensor[t, inventory.symbol_to_index[sym]] = 1.0
    return tensor
