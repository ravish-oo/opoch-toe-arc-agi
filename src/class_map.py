#!/usr/bin/env python3
"""
Class map: pullback of test truth partition to training output frames.

For each training output pixel, computes which test truth-class it belongs to
via OUT→TEST frame conjugation (inverse of TEST→OUT).

No shape law used here; pure frame morphisms.
"""

from typing import List, Optional, Tuple
import morphisms

Coord = Tuple[int, int]


def build_class_map_i(
    H_out: int,
    W_out: int,
    P_test,
    P_out_i,
    part
) -> List[Optional[int]]:
    """
    Build class map for training pair i.

    Args:
        H_out: Output height
        W_out: Output width
        P_test: Test input frame (op, anchor, shape)
        P_out_i: Training output frame (op, (0,0), shape)
        part: Partition on test input (has cid_of array)

    Returns:
        Row-major array of length H_out*W_out with class_id or None

    Algorithm:
        p_out → pose_inv(P_out.op, shape) → anchor_fwd(P_test.anchor) →
        pose_fwd(P_test.op, shape) → p_test
        if p_test in-bounds: cid = part.cid_of[row_major(p_test)]
        else: cid = None
    """
    op_out, _, shape_out = P_out_i
    op_test, anchor_test, shape_test = P_test
    H_test, W_test = shape_test

    class_map = []

    for r in range(H_out):
        for c in range(W_out):
            p_out = (r, c)

            # OUT → TEST (inverse of TEST→OUT)
            # Step 1: pose_inv (undo output pose)
            q = morphisms.pose_inv(p_out, op_out, shape_out)
            if q is None:
                class_map.append(None)
                continue

            # Step 2: anchor_fwd (apply test anchor direction)
            q = morphisms.anchor_fwd(q, anchor_test)
            if q is None:
                class_map.append(None)
                continue

            # Step 3: pose_fwd (apply test pose)
            p_test = morphisms.pose_fwd(q, op_test, shape_test)
            if p_test is None:
                class_map.append(None)
                continue

            # Check bounds in test frame
            if 0 <= p_test[0] < H_test and 0 <= p_test[1] < W_test:
                idx = p_test[0] * W_test + p_test[1]
                cid = part.cid_of[idx]
                class_map.append(cid)
            else:
                class_map.append(None)

    return class_map
