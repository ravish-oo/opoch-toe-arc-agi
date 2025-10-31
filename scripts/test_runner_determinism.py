#!/usr/bin/env python3
"""
Test runner determinism.

Runs runner twice on same tasks and verifies identical report hash.
"""

import sys
import os
import random
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import runner
import report


def main():
    """Run determinism check on runner."""
    print("=== Runner Determinism Check ===\n")

    # Seed for determinism
    random.seed(1337)

    # Mini corpus
    mini_task = {
        'train': [{'input': [[1, 2], [3, 4]], 'output': [[1, 2], [3, 4]]}],
        'test': [{'input': [[5, 6], [7, 8]]}]
    }

    mini_corpus = {'mini_identity': mini_task}

    # Run 1
    print("Run 1...")
    random.seed(1337)
    summary1 = runner.run_training_corpus(mini_corpus)
    report1 = report.aggregate_corpus_report(summary1)
    hash1 = report1["hash"]
    print(f"Hash: {hash1}\n")

    # Run 2
    print("Run 2...")
    random.seed(1337)
    summary2 = runner.run_training_corpus(mini_corpus)
    report2 = report.aggregate_corpus_report(summary2)
    hash2 = report2["hash"]
    print(f"Hash: {hash2}\n")

    # Verify
    if hash1 == hash2:
        print("✓ PASS: Hashes are identical - runner is deterministic")
        return 0
    else:
        print("✗ FAIL: Hashes differ - runner is non-deterministic")
        print(f"\nReport 1:\n{json.dumps(report1, indent=2)}")
        print(f"\nReport 2:\n{json.dumps(report2, indent=2)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
