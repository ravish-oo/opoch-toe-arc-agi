#!/usr/bin/env python3
"""
Run single task and show detailed receipts.

Usage:
    python scripts/run_single_task.py 00576224
    python scripts/run_single_task.py 00576224 --save receipts_00576224.json
"""

import sys
import os
import json
import argparse
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import runner


def main():
    parser = argparse.ArgumentParser(description="Run single task with receipts")
    parser.add_argument('task_id', type=str, help='Task ID (e.g., 00576224)')
    parser.add_argument('--save', type=str, help='Save receipts to file')
    args = parser.parse_args()

    random.seed(1337)

    # Load task
    corpus_file = 'data/arc-agi_training_challenges.json'
    with open(corpus_file, 'r') as f:
        all_tasks = json.load(f)

    if args.task_id not in all_tasks:
        print(f"Error: Task {args.task_id} not found")
        return 1

    task_data = all_tasks[args.task_id]

    print(f"=== Task: {args.task_id} ===\n")
    print(f"Train pairs: {len(task_data['train'])}")
    print(f"Test pairs: {len(task_data.get('test', []))}\n")

    # Run task
    result = runner.run_task(args.task_id, task_data)

    # Print result
    print(f"Status: {result['status']}\n")

    if result['status'] == 'error':
        print(f"Error:\n{result.get('error', 'Unknown')}\n")

    elif result['status'] == 'missing_descriptor':
        print("Missing descriptors:")
        for m in result.get('missing', []):
            print(f"\n  Class: {m['cid']}")
            print(f"  Witnesses: {len(m.get('examples', []))}")
            for i, ex in enumerate(m.get('examples', [])[:5], 1):
                print(f"    {i}. train_idx={ex['train_idx']}, p_out={ex['p_out']}, expected={ex['expected']}")

    elif result['status'] == 'exact':
        print(f"✓ Train match: {result.get('train_match_ok', False)}")

    # Save if requested
    if args.save:
        with open(args.save, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Full result saved to: {args.save}")
        print(f"  View with: cat {args.save} | python -m json.tool | less")

    return 0 if result['status'] == 'exact' else 1


if __name__ == '__main__':
    sys.exit(main())
