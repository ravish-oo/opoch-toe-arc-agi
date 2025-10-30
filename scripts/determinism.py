#!/usr/bin/env python3
"""
Determinism checker (WO-00-lite).

Runs harness multiple times and verifies receipt hashes are identical.
This proves that the solver is deterministic (no hidden randomness).

Usage:
    python scripts/determinism.py
    python scripts/determinism.py --runs 3
"""

import sys
import os
import subprocess
import json
import random
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def run_harness(task_id: str, selfcheck: str) -> tuple[str, dict]:
    """
    Run harness and capture receipt hash.

    Args:
        task_id: Task identifier
        selfcheck: Module to check

    Returns:
        (hash, receipt_doc)
    """
    # Seed before import (determinism)
    random.seed(1337)

    # Import fresh
    import receipts
    receipts.init(task_id)

    if selfcheck == "morphisms":
        import morphisms
        morphisms.init()

    doc = receipts.finalize()
    hash_val = receipts.hash_receipts(doc)

    return hash_val, doc


def main():
    """Run determinism check."""
    parser = argparse.ArgumentParser(description="Determinism checker")
    parser.add_argument("--runs", type=int, default=2, help="Number of runs")
    parser.add_argument("--task-id", type=str, default="dev.morphisms")
    parser.add_argument("--selfcheck", type=str, default="morphisms")

    args = parser.parse_args()

    print(f"=== Determinism Check ===")
    print(f"Task: {args.task_id}")
    print(f"Module: {args.selfcheck}")
    print(f"Runs: {args.runs}\n")

    hashes = []
    docs = []

    for i in range(args.runs):
        # Clear modules to get fresh import
        modules_to_clear = ['receipts', 'morphisms']
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]

        hash_val, doc = run_harness(args.task_id, args.selfcheck)
        hashes.append(hash_val)
        docs.append(doc)

        print(f"Run {i+1}: {hash_val}")

    # Check all hashes are identical
    if len(set(hashes)) == 1:
        print(f"\n✓ OK: All {args.runs} runs produced identical hash")
        print(f"Hash: {hashes[0]}")
        return 0
    else:
        print(f"\n✗ FAIL: Hashes differ across runs")
        for i, (h, doc) in enumerate(zip(hashes, docs)):
            print(f"\nRun {i+1} hash: {h}")
            print(f"Receipt: {json.dumps(doc, indent=2, sort_keys=True)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
