#!/usr/bin/env python3
"""
Run test corpus and save detailed results with receipts.

Usage:
    python scripts/run_test_corpus.py          # Test on 5 tasks
    python scripts/run_test_corpus.py --n 20   # Test on 20 tasks
    python scripts/run_test_corpus.py --n 10 --receipts-dir receipts/  # Save per-task receipts
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import runner
import report
import random


def main():
    parser = argparse.ArgumentParser(description="Run test corpus")
    parser.add_argument('--n', type=int, default=5, help='Number of tasks to test')
    parser.add_argument('--output', type=str, default='test_results.json', help='Output summary file')
    parser.add_argument('--receipts-dir', type=str, help='Directory to save per-task receipts (creates if needed)')
    args = parser.parse_args()

    # Create receipts directory if specified
    if args.receipts_dir:
        os.makedirs(args.receipts_dir, exist_ok=True)
        print(f"Saving per-task receipts to: {args.receipts_dir}/\n")

    random.seed(1337)

    # Load training corpus
    corpus_file = 'data/arc-agi_training_challenges.json'
    with open(corpus_file, 'r') as f:
        all_tasks = json.load(f)

    # Select tasks
    task_ids = sorted(all_tasks.keys())[:args.n]
    subset = {tid: all_tasks[tid] for tid in task_ids}

    print(f"Testing on {len(subset)} tasks: {task_ids[:10]}{'...' if len(task_ids) > 10 else ''}\n")

    # Run corpus
    summary = runner.run_training_corpus(subset)

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Total: {summary['tasks_total']}")
    print(f"Exact: {summary['exact']}")
    print(f"Missing: {summary['missing']}")
    print(f"Error: {summary.get('error', 0)}\n")

    # Show detailed results and save per-task receipts
    for r in summary['results']:
        task_id = r['task_id']
        print(f"\n--- Task: {task_id} ---")
        print(f"Status: {r['status']}")

        # Save per-task receipt if directory specified
        if args.receipts_dir:
            receipt_file = os.path.join(args.receipts_dir, f"{task_id}.json")
            with open(receipt_file, 'w') as f:
                json.dump(r, f, indent=2)
            print(f"Receipt saved: {receipt_file}")

        if r['status'] == 'error':
            error_msg = r.get('error', 'Unknown')
            # Truncate long errors
            if len(error_msg) > 100:
                error_msg = error_msg[:100] + '...'
            print(f"Error: {error_msg}")

        elif r['status'] == 'missing_descriptor':
            print("Missing descriptors:")
            for m in r.get('missing', []):
                print(f"  Class {m['cid']}: {len(m.get('examples', []))} witnesses")
                for ex in m.get('examples', [])[:2]:
                    print(f"    train_idx={ex['train_idx']}, p_out={ex['p_out']}, expected={ex['expected']}")

        elif r['status'] == 'exact':
            print(f"✓ Train match: {r.get('train_match_ok', '?')}")

    # Generate report
    print("\n=== REPORT ===")
    corpus_report = report.aggregate_corpus_report(summary)
    report.print_closure_progress(corpus_report)

    # Save to file
    output_data = {
        'summary': summary,
        'report': corpus_report
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Summary saved to: {args.output}")
    print(f"  View with: cat {args.output} | python -m json.tool | less")

    if args.receipts_dir:
        print(f"\n✓ Per-task receipts saved to: {args.receipts_dir}/")
        print(f"  Example: cat {args.receipts_dir}/00576224.json | python -m json.tool | less")
        print(f"  List all: ls {args.receipts_dir}/")


if __name__ == '__main__':
    main()
