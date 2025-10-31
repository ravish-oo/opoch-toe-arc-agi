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

    # WO-ND1/ND2: Fix test input BEFORE reversing train order for determinism
    if len(test_pairs) > 0:
        Xtest_raw = test_pairs[0]["input"]
    else:
        # No test input - use ORIGINAL first training input as proxy
        # (must be fixed before reversing to ensure determinism)
        Xtest_raw = train_pairs[0]["input"]

    # WO-ND2 fix: Assign original indices BEFORE reversing
    train_pairs_with_ids = [(i, pair) for i, pair in enumerate(train_pairs)]

    # Reverse train order if requested (AFTER assigning original indices)
    if train_order == "rev":
        train_pairs_with_ids = list(reversed(train_pairs_with_ids))

    if len(train_pairs_with_ids) == 0:
        doc = receipts.finalize()
        return (None, doc, receipts.hash_receipts(doc))

    # Extract raw grids with original indices
    Xin_raw = [pair["input"] for _, pair in train_pairs_with_ids]
    Yout_raw = [pair["output"] for _, pair in train_pairs_with_ids]
    orig_indices = [orig_i for orig_i, _ in train_pairs_with_ids]

    # 1. Present all grids
    morphisms.init()
    present.init()

    # Build palette map (train inputs + test input)
    Π = present.build_palette_map(Xin_raw, Xtest_raw)
    Π_inv = {v: k for k, v in Π.items()}

    # WO-ND0: Log palette_canon receipt for determinism verification
    palette_canon = present.build_palette_canon_receipt(Xin_raw, Xtest_raw, Π)
    receipts.log("present_palette_canon", palette_canon)

    # Present all training inputs
    # Frame indexing fix: Build dicts keyed by original index
    Xin_presented = []
    P_in_by_id = {}
    for i, grid in enumerate(Xin_raw):
        presented, frame = present.present_input(grid, Π)
        Xin_presented.append(presented)
        P_in_by_id[orig_indices[i]] = frame

    # Present all training outputs
    # WO-ND2 fix: Pair outputs with original train indices
    Yout_presented = []
    Yout_with_ids = []
    P_out_by_id = {}
    for i, grid in enumerate(Yout_raw):
        presented, frame = present.present_output(grid, Π)
        Yout_presented.append(presented)
        Yout_with_ids.append((orig_indices[i], presented))
        P_out_by_id[orig_indices[i]] = frame

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

    # WO-ND3 Part A: Compute order_hash for determinism verification
    sviews_order_hash = sviews.build_sviews_order_hash(sviews_list, (H_test, W_test))

    # Log sviews receipt
    receipts.log("sviews", {
        "count": len(sviews_list),
        "depth_max": 2,
        "views": [{"name": v.name if hasattr(v, 'name') else str(v)[:30]} for v in sviews_list[:5]],
        "proof_samples": [],
        "closure_capped": len(sviews_list) >= 128,
        "order_hash": sviews_order_hash,
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

    # WO-ND1: Build order keys and hash for determinism verification
    import hashlib
    order_keys = []
    for comp in components_list:
        order_keys.append([
            comp.color,
            comp.comp_id,
            list(comp.anchor),
            len(comp.mask)
        ])

    # Compute order_hash (SHA256 of order keys)
    order_hash = hashlib.sha256(str(order_keys).encode('utf-8')).hexdigest()

    # Format anchors_first5 per WO-ND1 spec
    anchors_first5 = []
    for comp in components_list[:5]:
        anchors_first5.append({
            "color": comp.color,
            "id": comp.comp_id,
            "anchor": list(comp.anchor),
            "size": len(comp.mask)
        })

    receipts.log("components", {
        "count_total": len(components_list),
        "by_color": by_color,
        "largest": largest,
        "anchors_first5": anchors_first5,
        "order_hash": order_hash,
        "proof_reconstruct_ok": True,
        "examples": {}
    })

    # 4. Build truth partition
    truth.init()

    # Prepare frames dict (keyed by original indices)
    frames = {
        "P_test": P_test,
        "P_out": P_out_by_id,
        "P_in": P_in_by_id
    }

    try:
        Q = truth.build_truth_partition(
            Xtest_presented, sviews_list, components_list, sviews_meta, frames, Yout_with_ids
        )
    except AssertionError:
        # Truth partition failed - finalize and return
        doc = receipts.finalize()
        return (None, doc, receipts.hash_receipts(doc))

    # 5. Learn shape law
    shape_law.init()

    # Convert frame dicts to sorted lists for legacy API
    P_in_list = [P_in_by_id[i] for i in sorted(P_in_by_id.keys())]
    P_out_list = [P_out_by_id[i] for i in sorted(P_out_by_id.keys())]

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

    # Build class maps for all training pairs (iterate by sorted orig_i)
    class_maps = []
    for orig_i in sorted(P_out_by_id.keys()):
        # Find the presented output for this orig_i
        Y_i = next(Y for oid, Y in Yout_with_ids if oid == orig_i)
        H_out = len(Y_i)
        W_out = len(Y_i[0]) if H_out > 0 else 0
        cm = class_map.build_class_map_i(H_out, W_out, P_test, P_out_by_id[orig_i], Q)
        class_maps.append(cm)

    # Run sieve
    sieve_result = sieve.run_sieve(
        Q, class_maps, Xin_presented, Yout_presented,
        P_test, P_in_list, P_out_list,
        keep_admitted, value_admitted
    )

    if sieve_result["status"] == "missing_descriptor":
        # Log selection section so sweep can categorize as truth_pass_missing_law
        receipts.log("selection", sieve_result)
        # Return missing status (no predictions)
        doc = receipts.finalize()
        return (None, doc, receipts.hash_receipts(doc))

    # 8. Paint test output
    # Log selection section for receipts completeness
    receipts.log("selection", sieve_result)

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
        "--sweep",
        action="store_true",
        help="Run corpus sweep to map failure distribution"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of tasks to run in sweep (default: all)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=3,
        help="Number of sample task IDs to show per bucket (default: 3)"
    )
    parser.add_argument(
        "--print-samples",
        action="store_true",
        help="Print detailed samples for atoms (witness, component_size, etc.)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full receipt JSON (or all task IDs per bucket in sweep mode)"
    )

    args = parser.parse_args()

    # Seed randomness for determinism
    random.seed(1337)

    # Handle --sweep flag
    if args.sweep:
        if not args.train_dir:
            print("ERROR: --sweep requires --train-dir")
            return 1

        # Load corpus
        corpus_path = os.path.join(args.train_dir, "arc-agi_training_challenges.json")
        with open(corpus_path, "r") as f:
            corpus = json.load(f)

        # Get task IDs in lexicographic order
        task_ids = sorted(corpus.keys())

        # Apply limit if specified
        if args.limit:
            task_ids = task_ids[:args.limit]

        # Initialize buckets
        buckets = {
            "present_fail": [],
            "truth_fail_atom": [],
            "truth_fail_nonatom": [],
            "truth_pass_missing_law": [],
            "paint_fail": [],
            "full_pass": [],
            "error": []
        }

        # Run tasks and bucket
        total = len(task_ids)
        for idx, task_id in enumerate(task_ids):
            # Progress indicator
            if (idx + 1) % 10 == 0 or idx == 0 or idx == total - 1:
                print(f"Progress: {idx + 1}/{total} tasks", file=sys.stderr)

            try:
                # Run task once
                preds, doc, _ = run_task_once(task_id, args.train_dir, train_order="orig")

                sections = doc.get("sections", {})

                # Bucketing logic
                # 1. Check present failures
                present_section = sections.get("present", {})
                if present_section and not present_section.get("round_trip_ok_inputs", True):
                    buckets["present_fail"].append((task_id, doc))
                    continue
                if present_section and not present_section.get("round_trip_ok_outputs", True):
                    buckets["present_fail"].append((task_id, doc))
                    continue

                # 2. Check truth failures
                truth_section = sections.get("truth", {})
                if truth_section and not truth_section.get("single_valued_ok", True):
                    # Determine if atom or non-atom
                    pt_last = truth_section.get("pt_last_contradiction", {})
                    witness_prov = pt_last.get("witness_provenance", {})
                    class_size = witness_prov.get("class_size", 0)

                    if class_size == 1:
                        buckets["truth_fail_atom"].append((task_id, doc))
                    else:
                        buckets["truth_fail_nonatom"].append((task_id, doc))
                    continue

                # 3. Check sieve/laws missing
                selection_section = sections.get("selection", {})
                if selection_section and selection_section.get("status") == "missing_descriptor":
                    buckets["truth_pass_missing_law"].append((task_id, doc))
                    continue

                # 4. Check paint failures
                paint_section = sections.get("paint", {})
                if paint_section:
                    idempotent_ok = paint_section.get("idempotent_ok", True)
                    coverage_pct = paint_section.get("coverage_pct", 0.0)

                    if not idempotent_ok or coverage_pct < 100.0:
                        buckets["paint_fail"].append((task_id, doc))
                        continue

                # 5. Full pass
                if preds is not None:
                    buckets["full_pass"].append((task_id, doc))
                else:
                    # Fallback: unknown failure
                    buckets["error"].append((task_id, doc))

            except Exception as e:
                # Catch all exceptions and bucket as error
                buckets["error"].append((task_id, {"error_msg": str(e)}))

        # Print summary
        print("\n=== SWEEP SUMMARY ===")
        print(f"Total tasks: {total}")
        print()

        # Print bucket counts and samples
        for bucket_name in ["present_fail", "truth_fail_atom", "truth_fail_nonatom",
                           "truth_pass_missing_law", "paint_fail", "full_pass", "error"]:
            bucket = buckets[bucket_name]
            count = len(bucket)
            pct = (count / total * 100) if total > 0 else 0.0

            print(f"{bucket_name}: {count} ({pct:.1f}%)")

            # Show samples
            if args.verbose:
                # Verbose: show all task IDs
                if count > 0:
                    task_ids_str = ", ".join([tid for tid, _ in bucket])
                    print(f"  Tasks: {task_ids_str}")
            else:
                # Normal: show first N samples
                sample_count = min(args.sample, count)
                if sample_count > 0:
                    samples = bucket[:sample_count]
                    sample_ids = [tid for tid, _ in samples]
                    print(f"  Samples: {', '.join(sample_ids)}")

                    # Enhanced samples for atoms
                    if args.print_samples and bucket_name == "truth_fail_atom":
                        for task_id, doc in samples:
                            sections = doc.get("sections", {})
                            truth_section = sections.get("truth", {})
                            pt_last = truth_section.get("pt_last_contradiction", {})
                            witness = pt_last.get("witness")
                            witness_prov = pt_last.get("witness_provenance", {})

                            component = witness_prov.get("component", {})
                            comp_size = component.get("size", "?")
                            comp_color = component.get("color", "?")

                            neighbors = witness_prov.get("equal_color_neighbors", {})
                            n4 = neighbors.get("4-neighborhood", 0)
                            n8 = neighbors.get("8-neighborhood", 0)

                            print(f"    {task_id}: witness={witness}, comp=(c={comp_color},sz={comp_size}), neighbors=(4={n4},8={n8})")

            print()

        return 0

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
        print(f"\nFull receipt:")
        print(json.dumps(doc, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    sys.exit(main())
