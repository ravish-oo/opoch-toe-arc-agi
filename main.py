#!/usr/bin/env python3
"""
Kaggle submission driver.

Reads test tasks, runs pipeline, writes submission.json.
Deterministic, no randomness, no internet.
"""

import json
import sys
import os
import random

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

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


def process_test_task(task_id: str, task_data: dict) -> list:
    """
    Process single test task and return predicted outputs.

    Args:
        task_id: Task identifier
        task_data: {"train": [...], "test": [...]}

    Returns:
        List of predicted test outputs (one per test input)
    """
    # Initialize receipts
    receipts.init(task_id)

    train_pairs = task_data["train"]
    test_pairs = task_data["test"]

    # Extract raw grids
    Xin_raw = [pair["input"] for pair in train_pairs]
    Yout_raw = [pair["output"] for pair in train_pairs]

    # 1. Present training grids
    morphisms.init()
    present.init()

    # Process each test input
    test_outputs = []

    for test_pair in test_pairs:
        Xtest_raw = test_pair["input"]

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
        os.environ["ARC_SELF_CHECK"] = "0"
        sviews.init()

        H_test, W_test = P_test[2]

        # Compute sviews metadata
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

        Q = truth.build_truth_partition(
            Xtest_presented, sviews_list, components_list, frames, Yout_presented
        )

        # 5. Learn shape law
        shape_law.init()

        # Build size list
        sizes = []
        for i in range(len(train_pairs)):
            _, _, (Hin, Win) = P_in_list[i]
            _, _, (Hout, Wout) = P_out_list[i]
            sizes.append((Hin, Win, Hout, Wout))

        try:
            law_type, law = shape_law.learn_law(sizes)
            shape = (law_type, law)
        except AssertionError:
            # Fallback: return empty grid
            test_outputs.append([[]])
            continue

        # 6. Admit laws
        keep.init()
        value.init()

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
            # Extract descriptor objects
            keep_admitted[cid] = []
            for candidate in keep_candidates:
                desc_str = candidate.descriptor()
                is_admitted = any(a.get("descriptor") == desc_str for a in admitted)
                if is_admitted:
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
            # Convert string descriptors to objects
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

        # 7. Sieve
        sieve.init()

        class_maps = []
        for i in range(len(Yout_presented)):
            H_out = len(Yout_presented[i])
            W_out = len(Yout_presented[i][0]) if H_out > 0 else 0
            cm = class_map.build_class_map_i(H_out, W_out, P_test, P_out_list[i], Q)
            class_maps.append(cm)

        sieve_result = sieve.run_sieve(
            Q, class_maps, Xin_presented, Yout_presented,
            P_test, P_in_list, P_out_list,
            keep_admitted, value_admitted
        )

        if sieve_result["status"] != "exact":
            # Fallback: return empty grid
            test_outputs.append([[]])
            continue

        # 8. Paint
        paint.init()
        assignment = sieve_result["assignment"]

        Y_painted = paint.painter_once(
            assignment, Q, Xtest_presented, Xin_presented,
            P_test, P_in_list, shape
        )

        # 9. Un-present
        # Create output frame (same op as test, anchor at origin)
        H_out, W_out = paint.build_test_canvas_size(P_test, shape)
        P_out_test = (P_test[0], (0, 0), (H_out, W_out))

        Y_unpresented = present.unpresent_output(Y_painted, P_out_test, Π_inv)

        test_outputs.append(Y_unpresented)

    return test_outputs


def main():
    """
    Main entry point for Kaggle submission.

    Reads test tasks and writes submission.json.
    """
    # Seed for determinism
    random.seed(1337)

    # Parse command line args
    import argparse
    parser = argparse.ArgumentParser(description="ARC-AGI submission driver")
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/arc-agi_test_challenges.json",
        help="Path to test challenges JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.json",
        help="Output submission file"
    )

    args = parser.parse_args()

    # Load test tasks
    with open(args.test_file, 'r') as f:
        test_tasks = json.load(f)

    print(f"Loaded {len(test_tasks)} test tasks")

    # Process each task
    submission = {}

    for task_id in sorted(test_tasks.keys()):
        print(f"Processing {task_id}...")
        task_data = test_tasks[task_id]

        try:
            test_outputs = process_test_task(task_id, task_data)
            submission[task_id] = test_outputs
        except Exception as e:
            print(f"Error processing {task_id}: {e}")
            # Fallback: empty output
            submission[task_id] = [[[]]]

    # Write submission
    with open(args.output, 'w') as f:
        json.dump(submission, f, indent=2)

    print(f"\nWrote submission to {args.output}")
    print(f"Tasks processed: {len(submission)}")


if __name__ == "__main__":
    main()
