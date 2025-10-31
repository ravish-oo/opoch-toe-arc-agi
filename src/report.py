#!/usr/bin/env python3
"""
Report: aggregate receipts and print closure progress.

Prints first witness per missing class for algebraic debugging.
"""

import json
import hashlib
from typing import Dict, List, Any


def aggregate_corpus_report(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate corpus run results into report.

    Args:
        summary: Output from run_training_corpus

    Returns:
        {
            "tasks_total": int,
            "exact": int,
            "missing": int,
            "missing_examples": [{"task_id": str, "cid": str, "examples": [...]}, ...],
            "hash": str
        }
    """
    tasks_total = summary["tasks_total"]
    exact = summary["exact"]
    missing = summary["missing"]
    results = summary["results"]

    # Collect missing examples (first witness per task+class)
    missing_examples = []

    for result in results:
        if result["status"] == "missing_descriptor":
            task_id = result["task_id"]
            for missing_item in result.get("missing", []):
                cid = missing_item.get("cid")
                examples = missing_item.get("examples", [])

                # Take first 3 examples
                examples_limited = examples[:3]

                missing_examples.append({
                    "task_id": task_id,
                    "cid": str(cid),
                    "examples": examples_limited
                })

    # Build canonical report
    report = {
        "tasks_total": tasks_total,
        "exact": exact,
        "missing": missing,
        "missing_examples": missing_examples
    }

    # Compute deterministic hash
    report_json = json.dumps(report, sort_keys=True, indent=2)
    report_hash = hashlib.sha256(report_json.encode()).hexdigest()
    report["hash"] = report_hash

    return report


def print_closure_progress(report: Dict[str, Any]):
    """
    Print human-readable closure progress.

    Args:
        report: Output from aggregate_corpus_report
    """
    print("=== Closure Progress ===")
    print(f"Tasks total: {report['tasks_total']}")
    print(f"Exact: {report['exact']}")
    print(f"Missing: {report['missing']}")
    print(f"Hash: {report['hash']}\n")

    if report["missing"] > 0:
        print("=== Missing Descriptors (First Witnesses) ===")
        for item in report["missing_examples"]:
            task_id = item["task_id"]
            cid = item["cid"]
            examples = item["examples"]

            print(f"\nTask: {task_id}, Class: {cid}")
            for i, ex in enumerate(examples[:3], 1):
                train_idx = ex.get("train_idx", "?")
                p_out = ex.get("p_out", "?")
                expected = ex.get("expected", "?")
                got = ex.get("got", "?")
                print(f"  Witness {i}: train_idx={train_idx}, p_out={p_out}, expected={expected}, got={got}")
    else:
        print("✓ No missing descriptors - corpus is exact!")


def print_witness_detail(witness: Dict[str, Any]):
    """
    Print detailed algebraic path for a single witness.

    Args:
        witness: {"task_id": str, "cid": str, "train_idx": int, "p_out": [i, j],
                  "expected": int, "got": int, "path": {...}}
    """
    print("=== Witness Detail ===")
    print(f"Task: {witness.get('task_id', '?')}")
    print(f"Class: {witness.get('cid', '?')}")
    print(f"Train pair: {witness.get('train_idx', '?')}")
    print(f"p_out: {witness.get('p_out', '?')}")
    print(f"Expected: {witness.get('expected', '?')}")
    print(f"Got: {witness.get('got', '?')}")

    path = witness.get("path", {})
    if path:
        print(f"\nCoordinate path:")
        for key, value in sorted(path.items()):
            print(f"  {key}: {value}")
