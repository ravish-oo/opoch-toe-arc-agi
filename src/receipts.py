"""
Receipts boundary logging (WO-00-lite).

Emit receipts ONLY at module boundaries.
Schema: docs/anchors/04-receipts-schema.json

This is pure, deterministic state management with no I/O.
"""

import json
import hashlib
from typing import Dict, Any, Optional

# Module state (reset per task via init)
_state: Dict[str, Any] = {
    "version": "1.0.0",
    "task_id": "uninitialized",
    "sections": {},
    "extras": {}
}


def init(task_id: str) -> None:
    """
    Initialize receipts for a new task.

    Args:
        task_id: Task identifier (e.g., "00576224" or "dev.morphisms")

    Notes:
        - Resets all logged sections
        - Sets version to "1.0.0" (frozen)
        - Stores task_id for finalize()
    """
    global _state
    _state = {
        "version": "1.0.0",
        "task_id": task_id,
        "sections": {},
        "extras": {}
    }


def log(section: str, payload: Dict[str, Any]) -> None:
    """
    Log a receipt at a module boundary.

    Args:
        section: One of "morphisms", "present", "sviews", "truth", "shape",
                 "laws", "selection", "paint"
        payload: Dict conforming to the schema for that section

    Notes:
        - Overwrites previous payload for the same section
        - No validation (trust the module)
        - No I/O (pure state update)
    """
    _state["sections"][section] = payload


def finalize() -> Dict[str, Any]:
    """
    Finalize receipts into canonical structure.

    Returns:
        Receipt document conforming to 04-receipts-schema.json

    Notes:
        - Returns a copy (caller can't mutate state)
        - Keys are sorted for deterministic JSON serialization
        - No timestamps (determinism over everything)
    """
    return {
        "version": _state["version"],
        "task_id": _state["task_id"],
        "sections": _state["sections"].copy(),
        "extras": _state["extras"].copy()
    }


def hash_receipts(doc: Dict[str, Any]) -> str:
    """
    Compute deterministic hash of receipt document.

    Args:
        doc: Receipt document (from finalize())

    Returns:
        SHA-256 hex digest (64 chars)

    Notes:
        - Sorts keys recursively
        - No whitespace (compact JSON)
        - Deterministic across Python runs
    """
    canonical_json = json.dumps(doc, sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
    return digest


def get_all() -> Dict[str, Any]:
    """
    Get current state (for debugging only).

    Returns:
        Copy of internal state
    """
    return _state.copy()


def clear() -> None:
    """
    Clear all receipts (for testing/reset).

    Notes:
        - Resets to uninitialized state
        - Use init() to properly initialize
    """
    global _state
    _state = {
        "version": "1.0.0",
        "task_id": "uninitialized",
        "sections": {},
        "extras": {}
    }
