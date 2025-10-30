"""
Receipts boundary logging (WO-00 / WO-01 dependency).

Emit receipts ONLY at module boundaries.
Schema: docs/anchors/04-receipts-schema.json
"""

from typing import Dict, Any

_receipts: Dict[str, Any] = {}


def log(section: str, payload: Dict[str, Any]) -> None:
    """
    Log a receipt at a module boundary.

    Args:
        section: One of "present", "sviews", "truth", "shape", "laws", "selection", "paint"
        payload: Dict conforming to the schema for that section
    """
    _receipts[section] = payload


def get_all() -> Dict[str, Any]:
    """Get all logged receipts (copy)."""
    return _receipts.copy()


def clear() -> None:
    """Clear all receipts (for testing/reset)."""
    _receipts.clear()


def finalize(task_id: str, version: str = "1.0.0") -> Dict[str, Any]:
    """
    Finalize receipts into canonical structure.

    Returns:
        Receipt document conforming to 04-receipts-schema.json
    """
    return {
        "version": version,
        "task_id": task_id,
        "sections": _receipts.copy(),
        "extras": {}
    }
