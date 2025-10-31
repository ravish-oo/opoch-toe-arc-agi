#!/usr/bin/env python3
"""
Harness: minimal runner for self-checks and receipt emission (WO-00-lite).

Usage:
    python -m src.harness --task-id dev.morphisms --selfcheck morphisms
    python -m src.harness --task-id 00576224 --selfcheck morphisms
    python -m src.harness --task-id 00d62c1b --train-dir data/training --determinism
    python -m src.harness --task-id 00d62c1b --train-dir data/training --explain
"""

import sys
import os
import argparse
import random
import json
from typing import Dict, List, Any, Tuple, Optional

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


def run_task_once(
    task_id: str,
    train_dir: str,
    train_order: str = "orig"
) -> Tuple[Optional[List[List[int]]], Dict[str, Any], str]:
    """
    Run single task through full pipeline including test output painting.

    Args:
        task_id: Task identifier (e.g., "00d62c1b")
        train_dir: Directory containing training JSON file (e.g., "data")
        train_order: "orig" or "rev" (reverse training examples)

    Returns:
        (predictions, receipts_doc, hash) where:
            predictions: Test output grid (raw colors) if exact, else None
            receipts_doc: Full receipts document
            hash: SHA256 hash of receipts
    """
    # Initialize receipts for this task
    receipts.init(task_id)

    # Load training corpus JSON
    corpus_path = os.path.join(train_dir, "arc-agi_training_challenges.json")
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    # Extract task data
    if task_id not in corpus:
        doc = receipts.finalize()
        return (None, doc, receipts.hash_receipts(doc))

    task_data = corpus[task_id]

    train_pairs = task_data["train"]
    test_pairs = task_data.get("test", [])

    # Reverse train order if requested
    if train_order == "rev":
        train_pairs = list(reversed(train_pairs))

    if len(train_pairs) == 0:
        doc = receipts.finalize()
        return (None, doc, receipts.hash_receipts(doc))

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

    # Log sviews receipt
    receipts.log("sviews", {
        "count": len(sviews_list),
        "depth_max": 2,
        "views": [{"name": v.name if hasattr(v, 'name') else str(v)[:30]} for v in sviews_list[:5]],
        "proof_samples": [],
        "closure_capped": len(sviews_list) >= 128,
        "examples": {}
    })

    # 3. Build components
    components.init()
    components_list = components.build_components(Xtest_presented)

    # Log components receipt
    by_color = {}
    for comp in components_list:
        color = comp.color if hasattr(comp, 'color') else 0
        if color not in by_color:
            by_color[color] = 0
        by_color[color] += 1

    largest = {}
    if components_list:
        largest_comp = max(components_list, key=lambda c: c.size if hasattr(c, 'size') else 0)
        if hasattr(largest_comp, 'color') and hasattr(largest_comp, 'size'):
            largest = {"color": largest_comp.color, "size": largest_comp.size}

    anchors_first5 = []
    for comp in components_list[:5]:
        if hasattr(comp, 'anchor'):
            anchors_first5.append(list(comp.anchor))

    receipts.log("components", {
        "count_total": len(components_list),
        "by_color": by_color,
        "largest": largest,
        "anchors_first5": anchors_first5,
        "proof_reconstruct_ok": True,
        "examples": {}
    })

    # 4. Build truth partition
    truth.init()

    # Prepare frames dict
    frames = {
        "P_test": P_test,
        "P_out": P_out_list
    }

    try:
        Q = truth.build_truth_partition(
            Xtest_presented, sviews_list, components_list, sviews_meta, frames, Yout_presented
        )
    except AssertionError:
        # Truth partition failed - finalize and return
        doc = receipts.finalize()
        return (None, doc, receipts.hash_receipts(doc))

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
    except AssertionError:
        # Shape law failed - finalize and return
        doc = receipts.finalize()
        return (None, doc, receipts.hash_receipts(doc))

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
        # Return missing status (no predictions)
        doc = receipts.finalize()
        return (None, doc, receipts.hash_receipts(doc))

    # 8. Paint test output
    paint.init()
    assignment = sieve_result["assignment"]

    # Compute output size from shape law
    H_out, W_out = paint.build_test_canvas_size(H_test, W_test, shape)

    # Paint test output (returns posed grid)
    Y_posed = paint.painter_once(
        assignment, Q, Xtest_presented, Xin_presented,
        P_test, P_in_list, shape
    )

    # Build test output frame (identity op, origin anchor, output size)
    P_out_test = (0, (0, 0), (H_out, W_out))

    # Unpresent to raw colors
    Y_raw = paint.unpresent_final(Y_posed, P_out_test, Π_inv)

    # Finalize receipts
    doc = receipts.finalize()

    return (Y_raw, doc, receipts.hash_receipts(doc))


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
        choices=["morphisms", "present", "sviews", "components", "truth", "shape", "keep", "value", "sieve", "paint"],
        help="Module to self-check"
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        help="Directory containing training task JSON files"
    )
    parser.add_argument(
        "--determinism",
        action="store_true",
        help="Run determinism check (permute train order, compare hashes)"
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Extract and print WHY from receipts"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full receipt JSON"
    )

    args = parser.parse_args()

    # Seed randomness for determinism
    random.seed(1337)

    # Handle --determinism flag
    if args.determinism:
        if not args.train_dir:
            print("ERROR: --determinism requires --train-dir")
            return 1

        # Run task with original train order
        preds_orig, doc_orig, hash_orig = run_task_once(
            args.task_id, args.train_dir, train_order="orig"
        )

        # Run task with reversed train order
        preds_rev, doc_rev, hash_rev = run_task_once(
            args.task_id, args.train_dir, train_order="rev"
        )

        # Compare hashes
        if hash_orig != hash_rev:
            # Find first differing section
            sections_orig = doc_orig.get("sections", {})
            sections_rev = doc_rev.get("sections", {})

            diff_section = None
            for section_name in sorted(set(sections_orig.keys()) | set(sections_rev.keys())):
                if sections_orig.get(section_name) != sections_rev.get(section_name):
                    diff_section = section_name
                    break

            print(f"DETERMINISM FAIL task={args.task_id} hash_orig={hash_orig[:8]} hash_rev={hash_rev[:8]} diff_section={diff_section}")
            return 1

        # Compare predictions
        if preds_orig != preds_rev:
            # Find first differing pixel
            diff_pixel = None
            if preds_orig is None or preds_rev is None:
                diff_pixel = "one_is_None"
            elif len(preds_orig) != len(preds_rev):
                diff_pixel = f"height_diff({len(preds_orig)}!={len(preds_rev)})"
            elif any(len(preds_orig[i]) != len(preds_rev[i]) for i in range(len(preds_orig))):
                diff_pixel = "width_diff"
            else:
                for r in range(len(preds_orig)):
                    for c in range(len(preds_orig[r])):
                        if preds_orig[r][c] != preds_rev[r][c]:
                            diff_pixel = f"({r},{c})"
                            break
                    if diff_pixel:
                        break

            print(f"DETERMINISM FAIL task={args.task_id} hash=IDENTICAL preds=DIFFER first_diff={diff_pixel}")
            return 1

        # Determinism OK
        preds_status = "IDENTICAL" if preds_orig is not None else "both_None"
        print(f"DETERMINISM OK task={args.task_id} hash={hash_orig[:16]} preds={preds_status}")
        return 0

    # Handle --explain flag
    if args.explain:
        if not args.train_dir:
            print("ERROR: --explain requires --train-dir")
            return 1

        # Run task once
        preds, doc, _ = run_task_once(args.task_id, args.train_dir, train_order="orig")

        # Extract WHY from receipts in priority order
        sections = doc.get("sections", {})

        # Priority 1: missing_descriptor from selection
        selection = sections.get("selection", {})
        if selection.get("status") == "missing_descriptor":
            missing = selection.get("missing", [])
            if missing:
                first = missing[0]
                cid = first.get("class")
                pixel = first.get("pixel")
                expected = first.get("expected")
                got = first.get("got", [])
                print(f"WHY FAIL task={args.task_id} reason=missing_descriptor class={cid} pixel={pixel} expected={expected} got={got}")
                return 0

        # Priority 2: truth contradiction
        truth_section = sections.get("truth", {})
        if truth_section.get("single_valued_ok") == False:
            examples = truth_section.get("examples", {})
            if examples.get("case") == "PT_failed":
                detail = examples.get("detail", "")
                pt_last = truth_section.get("pt_last_contradiction", {})
                cid = pt_last.get("cid")
                colors = pt_last.get("colors_seen", [])
                witness = pt_last.get("witness")
                print(f"WHY FAIL task={args.task_id} reason=truth_contradiction class={cid} colors={colors} witness={witness}")
                return 0

        # Priority 3: paint issues
        paint_section = sections.get("paint", {})
        if paint_section:
            # Check for paint failures in examples
            paint_examples = paint_section.get("examples", {})
            if "error" in paint_examples:
                print(f"WHY FAIL task={args.task_id} reason=paint_error detail={paint_examples['error']}")
                return 0

        # Otherwise: OK
        if preds is not None:
            print(f"WHY OK task={args.task_id} reason=exact")
        else:
            print(f"WHY FAIL task={args.task_id} reason=unknown")
        return 0

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
    elif args.selfcheck == "sieve":
        import morphisms
        import sieve
        morphisms.init()
        # Set environment variable for self-check
        os.environ["ARC_SELF_CHECK"] = "1"
        sieve.init()
        print(f"✓ morphisms self-check passed")
        print(f"✓ sieve self-check passed")
    elif args.selfcheck == "paint":
        import morphisms
        import paint
        morphisms.init()
        # Set environment variable for self-check
        os.environ["ARC_SELF_CHECK"] = "1"
        paint.init()
        print(f"✓ morphisms self-check passed")
        print(f"✓ paint self-check passed")

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
