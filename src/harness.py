#!/usr/bin/env python3
"""
Harness: minimal runner for self-checks and receipt emission (WO-00-lite).

Usage:
    python -m src.harness --task-id dev.morphisms --selfcheck morphisms
    python -m src.harness --task-id 00576224 --selfcheck morphisms
"""

import sys
import os
import argparse
import random

import receipts


def main():
    """Run self-checks and print receipts hash."""
    parser = argparse.ArgumentParser(description="ARC solver harness")
    parser.add_argument(
        "--task-id",
        type=str,
        default="dev.morphisms",
        help="Task identifier"
    )
    parser.add_argument(
        "--selfcheck",
        type=str,
        choices=["morphisms", "present", "sviews", "components", "truth", "shape", "keep", "value"],
        help="Module to self-check"
    )
    parser.add_argument(
        "--determinism",
        action="store_true",
        help="Run determinism check (permute train order, compare hashes)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full receipt JSON"
    )

    args = parser.parse_args()

    # Seed randomness for determinism
    random.seed(1337)

    # Initialize receipts
    receipts.init(args.task_id)

    # Run self-checks
    if args.selfcheck == "morphisms":
        import morphisms
        morphisms.init()
        print(f"✓ morphisms self-check passed")
    elif args.selfcheck == "present":
        import morphisms
        import present
        morphisms.init()
        present.init()
        print(f"✓ morphisms self-check passed")
        print(f"✓ present self-check passed")
    elif args.selfcheck == "sviews":
        import morphisms
        import sviews
        morphisms.init()
        # Set environment variable for self-check
        os.environ["ARC_SELF_CHECK"] = "1"
        sviews.init()
        print(f"✓ morphisms self-check passed")
        print(f"✓ sviews self-check passed")
    elif args.selfcheck == "components":
        import morphisms
        import components
        morphisms.init()
        # Set environment variable for self-check
        os.environ["ARC_SELF_CHECK"] = "1"
        components.init()
        print(f"✓ morphisms self-check passed")
        print(f"✓ components self-check passed")
    elif args.selfcheck == "truth":
        import morphisms
        import truth
        morphisms.init()
        # Set environment variable for self-check
        os.environ["ARC_SELF_CHECK"] = "1"
        truth.init()
        print(f"✓ morphisms self-check passed")
        print(f"✓ truth self-check passed")
    elif args.selfcheck == "shape":
        import morphisms
        import shape_law
        morphisms.init()
        # Set environment variable for self-check
        os.environ["ARC_SELF_CHECK"] = "1"
        shape_law.init()
        print(f"✓ morphisms self-check passed")
        print(f"✓ shape self-check passed")
    elif args.selfcheck == "keep":
        import morphisms
        from laws import keep
        morphisms.init()
        # Set environment variable for self-check
        os.environ["ARC_SELF_CHECK"] = "1"
        keep.init()
        print(f"✓ morphisms self-check passed")
        print(f"✓ keep self-check passed")
    elif args.selfcheck == "value":
        import morphisms
        from laws import value
        morphisms.init()
        # Set environment variable for self-check
        os.environ["ARC_SELF_CHECK"] = "1"
        value.init()
        print(f"✓ morphisms self-check passed")
        print(f"✓ value self-check passed")

    # Finalize and print
    doc = receipts.finalize()
    receipt_hash = receipts.hash_receipts(doc)

    print(f"\nTask ID: {doc['task_id']}")
    print(f"Sections: {list(doc['sections'].keys())}")
    print(f"Hash: {receipt_hash}")

    # Optionally print full receipt
    if args.verbose:
        import json
        print(f"\nFull receipt:")
        print(json.dumps(doc, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    sys.exit(main())
