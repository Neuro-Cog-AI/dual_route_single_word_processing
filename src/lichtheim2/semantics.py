from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lichtheim2.data import WordItem


def assign_artificial_semantics(
    word_items: list["WordItem"],
    vATL_size: int = 50,
    seed: int = 42,
) -> dict[int, torch.Tensor]:
    """Assign a unique random binary semantic vector to each WordItem.

    Vectors are 0/1 float tensors of shape (vATL_size,).

    Reproducibility: a single generator seeded with `seed` produces vectors in
    the order that word_items are provided. The same list in the same order and
    the same seed always yields the same mapping.

    This is a first-pass implementation using random binary vectors.
    A prototype-based faithful generation (as in the supplement) is a later
    enhancement — see D15 in docs/open_questions.md.

    Args:
        word_items: list of WordItems (real words from load_word_items())
        vATL_size:  dimension of each semantic vector (default 50 [Paper])
        seed:       random seed for reproducibility

    Returns:
        dict mapping row_index → (vATL_size,) float tensor (values 0.0 or 1.0)

    Raises:
        ValueError: if any WordItem has row_index is None
        ValueError: if two WordItems share the same row_index
    """
    rng = torch.Generator().manual_seed(seed)
    result: dict[int, torch.Tensor] = {}

    for item in word_items:
        if item.row_index is None:
            raise ValueError(
                f"WordItem {item.word!r} has row_index=None; "
                "row_index must be set before assigning semantics"
            )
        if item.row_index in result:
            raise ValueError(
                f"Duplicate row_index {item.row_index} detected for word {item.word!r}; "
                "each WordItem must have a unique row_index"
            )
        vec = torch.randint(0, 2, (vATL_size,), generator=rng, dtype=torch.float32)
        result[item.row_index] = vec

    return result
