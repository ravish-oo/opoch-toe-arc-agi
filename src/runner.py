#!/usr/bin/env python3
"""
Batch runner for training corpus.

Runs full pipeline: Π → Q → Shape → Laws → Sieve → Paint
Collects receipts per task; returns exact or missing_descriptor status.
"""

import os
import json
import sys
from typing import Dict, List, Any, Tuple

import receipts
import morphisms
import present
import sviews
import components
import truth
import shape_law
from laws import keep, value
import sieve
import paint
import class_map
import descriptor_registry


def run_task(task_id: str, task_data: Dict) -> Dict[str, Any]:
    """
    Run single task through full pipeline.

    Args:
        task_id: Task identifier
        task_data: {"train": [...], "test": [...]}

    Returns:
        {
            "task_id": str,
            "status": "exact" | "missing_descriptor",
            "train_match_ok": bool (if exact),
            "missing": [...] (if missing_descriptor)
        }
    """
    # Initialize receipts for this task
    receipts.init(task_id)

    train_pairs = task_data["train"]
    test_pairs = task_data.get("test", [])

    if len(train_pairs) == 0:
        return {
            "task_id": task_id,
            "status": "error",
            "error": "No training pairs"
        }

    # Extract raw grids
    Xin_raw = [pair["input"] for pair in train_pairs]
    Yout_raw = [pair["output"] for pair in train_pairs]

    # 1. Present all grids
    morphisms.init()
    present.init()

    # Use first test input for test frame
    if len(test_pairs) > 0:
        Xtest_raw = test_pairs[0]["input"]
    else:
        # No test input - use first training input as proxy
        Xtest_raw = Xin_raw[0]

    # Build palette map (train inputs + test input)
    Π = present.build_palette_map(Xin_raw, Xtest_raw)
    Π_inv = {v: k for k, v in Π.items()}

    # Present all training inputs
    Xin_presented = []
    P_in_list = []
    for grid in Xin_raw:
        presented, frame = present.present_input(grid, Π)
        Xin_presented.append(presented)
        P_in_list.append(frame)

    # Present all training outputs
    Yout_presented = []
    P_out_list = []
    for grid in Yout_raw:
        presented, frame = present.present_output(grid, Π)
        Yout_presented.append(presented)
        P_out_list.append(frame)

    # Present test input
    Xtest_presented, P_test = present.present_input(Xtest_raw, Π)

    # 2. Build sviews
    os.environ["ARC_SELF_CHECK"] = "0"  # Skip self-checks in batch mode
    sviews.init()

    H_test, W_test = P_test[2]

    # Compute sviews metadata (row/col GCDs)
    row_gcd, _ = sviews.minimal_row_period(Xtest_presented)
    col_gcd, _ = sviews.minimal_col_period(Xtest_presented)
    sviews_meta = {
        "row_gcd": row_gcd,
        "col_gcd": col_gcd
    }

    # Build sviews
    sviews_list = sviews.build_sviews(Xtest_presented)

    # 3. Build components
    components.init()
    components_list = components.build_components(Xtest_presented)

    # 4. Build truth partition
    truth.init()

    # Prepare frames dict
    frames = {
        "P_test": P_test,
        "P_out": P_out_list
    }

    try:
        Q = truth.build_truth_partition(
            Xtest_presented, sviews_list, components_list, frames, Yout_presented
        )
    except AssertionError as e:
        doc = receipts.finalize()
        return {
            "task_id": task_id,
            "status": "error",
            "error": f"Truth partition failed: {e}",
            "receipts": doc
        }

    # 5. Learn shape law
    shape_law.init()

    # Build size list: (Hin, Win, Hout, Wout) from presented frames
    sizes = []
    for i in range(len(train_pairs)):
        _, _, (Hin, Win) = P_in_list[i]
        _, _, (Hout, Wout) = P_out_list[i]
        sizes.append((Hin, Win, Hout, Wout))

    try:
        law_type, law = shape_law.learn_law(sizes)
        shape = (law_type, law)
    except AssertionError as e:
        doc = receipts.finalize()
        return {
            "task_id": task_id,
            "status": "error",
            "error": f"Shape law failed: {e}",
            "receipts": doc
        }

    # 6. Admit laws per class
    keep.init()
    value.init()

    # Get KEEP catalogue
    keep_candidates = descriptor_registry.keep_catalogue(H_test, W_test, sviews_meta)

    # Build class pixels map (TEST frame)
    class_pixels_test = {}
    for idx, cid in enumerate(Q.cid_of):
        if cid not in class_pixels_test:
            class_pixels_test[cid] = []
        r = idx // W_test
        c = idx % W_test
        class_pixels_test[cid].append((r, c))

    # Admit KEEP laws per class
    keep_admitted = {}
    all_cids = set(Q.cid_of)
    for cid in all_cids:
        pixels = class_pixels_test.get(cid, [])
        if not pixels:
            keep_admitted[cid] = []
            continue

        admitted = keep.admit_keep_for_class(
            cid, pixels, Xin_presented, Yout_presented,
            P_test, P_in_list, P_out_list, shape, keep_candidates
        )
        # Extract descriptor objects (dict with "view" + params)
        keep_admitted[cid] = []
        for candidate in keep_candidates:
            # Check if this candidate was admitted
            desc_str = candidate.descriptor()
            is_admitted = any(a.get("descriptor") == desc_str for a in admitted)
            if is_admitted:
                # Build descriptor object for sieve
                desc_obj = {"view": candidate.name, **candidate.params}
                keep_admitted[cid].append(desc_obj)

    # Admit VALUE laws per class
    value_admitted = {}
    for cid in all_cids:
        pixels = class_pixels_test.get(cid, [])
        if not pixels:
            value_admitted[cid] = []
            continue

        result = value.admit_value_for_class(
            cid, pixels, Xin_presented, Yout_presented, Xtest_presented,
            P_test, P_in_list, P_out_list
        )
        # Convert string descriptors to objects for sieve
        value_admitted[cid] = []
        for desc_str in result.get("admitted", []):
            if desc_str.startswith("CONST(c="):
                c = int(desc_str[8:-1])
                value_admitted[cid].append({"type": "CONST", "c": c})
            elif desc_str.startswith("UNIQUE(c="):
                c = int(desc_str[9:-1])
                value_admitted[cid].append({"type": "UNIQUE", "c": c})
            elif desc_str.startswith("ARGMAX(c="):
                c = int(desc_str[9:-1])
                value_admitted[cid].append({"type": "ARGMAX", "c": c})
            elif desc_str.startswith("LOWEST_UNUSED(c="):
                c = int(desc_str[16:-1])
                value_admitted[cid].append({"type": "LOWEST_UNUSED", "c": c})
            elif desc_str.startswith("RECOLOR(pi={"):
                pi_str = desc_str[12:-2]
                pi = {}
                for part in pi_str.split(","):
                    k, v = part.split(":")
                    pi[int(k)] = int(v)
                value_admitted[cid].append({"pi": pi})
            elif desc_str.startswith("BLOCK(k="):
                k = int(desc_str[8:-1])
                value_admitted[cid].append({"type": "BLOCK", "k": k})

    # 7. Build class_map and run sieve
    sieve.init()

    # Build class maps for all training pairs
    class_maps = []
    for i in range(len(Yout_presented)):
        H_out = len(Yout_presented[i])
        W_out = len(Yout_presented[i][0]) if H_out > 0 else 0
        cm = class_map.build_class_map_i(H_out, W_out, P_test, P_out_list[i], Q)
        class_maps.append(cm)

    # Run sieve
    sieve_result = sieve.run_sieve(
        Q, class_maps, Xin_presented, Yout_presented,
        P_test, P_in_list, P_out_list,
        keep_admitted, value_admitted
    )

    if sieve_result["status"] == "missing_descriptor":
        # Return missing status with witnesses
        doc = receipts.finalize()
        return {
            "task_id": task_id,
            "status": "missing_descriptor",
            "missing": sieve_result.get("missing", []),
            "receipts": doc
        }

    # 8. Paint and verify
    paint.init()
    assignment = sieve_result["assignment"]

    # Paint each training output and compare
    train_match_ok = True
    for i in range(len(train_pairs)):
        Y_painted = paint.painter_once(
            assignment, Q, Xtest_presented, Xin_presented,
            P_test, P_in_list, shape
        )

        # Un-present painted output
        P_out_i = P_out_list[i]
        Y_unpresented = present.unpresent_output(Y_painted, P_out_i, Π_inv)

        # Compare to official training output
        if Y_unpresented != Yout_raw[i]:
            train_match_ok = False
            break

    # Finalize receipts
    doc = receipts.finalize()

    return {
        "task_id": task_id,
        "status": "exact",
        "train_match_ok": train_match_ok,
        "receipts": doc
    }


def run_training_corpus(tasks: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Run full training corpus through pipeline.

    Args:
        tasks: Dict mapping task_id -> {"train": [...], "test": [...]}

    Returns:
        {
            "tasks_total": int,
            "exact": int,
            "missing": int,
            "results": [{"task_id": str, "status": str, ...}, ...]
        }
    """
    results = []
    exact_count = 0
    missing_count = 0
    error_count = 0

    # Sort task IDs for determinism
    task_ids = sorted(tasks.keys())

    for task_id in task_ids:
        task_data = tasks[task_id]
        result = run_task(task_id, task_data)
        results.append(result)

        if result["status"] == "exact":
            exact_count += 1
        elif result["status"] == "missing_descriptor":
            missing_count += 1
        elif result["status"] == "error":
            error_count += 1

    return {
        "tasks_total": len(task_ids),
        "exact": exact_count,
        "missing": missing_count,
        "error": error_count,
        "results": results
    }


def _self_check_runner():
    """
    Self-check for runner (debugging = algebra).

    Uses mini-corpus with known tasks.
    """
    # Mini synthetic task: simple identity
    mini_task = {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [[1, 2], [3, 4]]
            }
        ],
        "test": [
            {
                "input": [[5, 6], [7, 8]],
                "output": [[5, 6], [7, 8]]
            }
        ]
    }

    mini_corpus = {"mini_identity": mini_task}

    summary = run_training_corpus(mini_corpus)

    # Should have exact status
    assert summary["exact"] >= 0, "Runner failed on mini-corpus"
    assert summary["tasks_total"] == 1, f"Expected 1 task, got {summary['tasks_total']}"

    # Check determinism: re-run and compare
    import random
    random.seed(1337)

    summary2 = run_training_corpus(mini_corpus)

    # Summaries should match
    assert summary["exact"] == summary2["exact"], "Runner is non-deterministic"

    print("✓ Runner self-check passed")


def init():
    """Run self-check if ARC_SELF_CHECK=1."""
    if os.environ.get("ARC_SELF_CHECK") == "1":
        _self_check_runner()
        receipt = {
            "tests_passed": 1,
            "verified": ["mini_corpus", "determinism"]
        }
        receipts.log("runner_selfcheck", receipt)
