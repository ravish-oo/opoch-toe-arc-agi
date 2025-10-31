#!/usr/bin/env python3
"""
Descriptor registry: single source of truth for law catalogue and cost order.

This is the ONLY place to add descriptors (additive-only).
Each addition must be witness-justified and come with admissibility proof.
"""

from typing import List, Dict, Any


def keep_catalogue(Ht: int, Wt: int, sviews_meta: dict) -> List[Dict[str, Any]]:
    """
    KEEP catalogue (test-frame views), deterministic order.

    Args:
        Ht, Wt: Presented test input dimensions
        sviews_meta: Metadata from sviews (gcds, etc.)

    Returns:
        List of KEEP descriptor objects: [{"view": name, ...params}, ...]
    """
    from laws.keep import enumerate_keep_candidates
    return enumerate_keep_candidates(Ht, Wt, sviews_meta)


def value_catalogue() -> List[str]:
    """
    VALUE catalogue (per-class reducers), deterministic order.

    Returns:
        List of VALUE family names: ["CONST", "UNIQUE", "ARGMAX", ...]
    """
    return ["CONST", "UNIQUE", "ARGMAX", "LOWEST_UNUSED", "RECOLOR", "BLOCK"]


def cost_order() -> List[str]:
    """
    Canonical cost order (lower index = lower cost).

    Returns:
        List of descriptor patterns for sorting
    """
    return [
        "tile_alt_row_flip",
        "tile_alt_col_flip",
        "tile_alt_checkerboard_flip",
        "tile",
        "d4_",  # prefix match for d4_0, d4_1, etc.
        "identity",
        "RECOLOR",
        "BLOCK",
        "ARGMAX",
        "UNIQUE",
        "LOWEST_UNUSED",
        "CONST"
    ]
