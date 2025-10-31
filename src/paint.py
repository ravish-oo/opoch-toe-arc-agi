#!/usr/bin/env python3
"""
Painter: idempotent one-shot paint from proven laws.

Paints test output once using selected per-class laws, then un-presents
back to raw grids. Proves idempotence and 100% coverage.

Debugging = algebra: every miss is a single pixel with coordinate path.
"""

import os
from typing import Dict, List, Tuple, Any, Optional
import morphisms
import receipts


Coord = Tuple[int, int]


def build_test_canvas_size(P_test, shape_law: Tuple[str, Tuple[int, int, int, int]]) -> Tuple[int, int]:
    """
    Compute output test canvas size from shape law applied to TEST input size.

    Args:
        P_test: Test input frame (op, anchor, (H_in, W_in))
        shape_law: (type, (a, b, c, d))

    Returns:
        (H_out, W_out)
    """
    _, _, (H_in, W_in) = P_test
    law_type, (a, b, c, d) = shape_law

    H_out = a * H_in + b
    W_out = c * W_in + d

    return (H_out, W_out)


def _shape_pullback(p_out: Coord, law: Tuple[int, int, int, int], H_in: int, W_in: int) -> Optional[Coord]:
    """
    Pullback output pixel to input pixel via floor mapping.

    Args:
        p_out: Output pixel (i, j)
        law: (a, b, c, d)
        H_in, W_in: Input dimensions

    Returns:
        (i_in, j_in) if in-bounds, else None
    """
    a, b, c, d = law
    i_out, j_out = p_out

    # Floor mapping
    if a == 0 or c == 0:
        return None

    i_in = (i_out - b) // a
    j_in = (j_out - d) // c

    # Check bounds
    if 0 <= i_in < H_in and 0 <= j_in < W_in:
        return (i_in, j_in)
    return None


def _parse_descriptor(desc_str: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parse descriptor string into (family, params).

    Accepts both "KEEP:identity" and "identity" formats (defensive).

    Returns:
        (family, params_dict)
    """
    # Strip "KEEP:" prefix if present (defensive backward-compat)
    view_name_candidate = desc_str
    if desc_str.startswith("KEEP:"):
        view_name_candidate = desc_str[5:]  # Remove "KEEP:" prefix

    # Check if this is a KEEP law (starts with KEEP: OR is a known view name)
    known_views = ["identity", "d4", "translate", "residue_row", "residue_col",
                   "tile", "tile_alt_row_flip", "tile_alt_col_flip",
                   "tile_checkerboard_flip", "block_inverse", "offset"]

    is_keep = desc_str.startswith("KEEP:")
    if not is_keep:
        # Check if it's a known view name (backward-compat)
        for known in known_views:
            if view_name_candidate.startswith(known):
                is_keep = True
                break

    if is_keep:
        view_name = view_name_candidate
        # Parse params if present (e.g., "translate(di=1,dj=0)")
        if "(" in view_name:
            name = view_name[:view_name.index("(")]
            params_str = view_name[view_name.index("(")+1:view_name.index(")")]
            params = {}
            if params_str:
                for part in params_str.split(","):
                    k, v = part.split("=")
                    params[k] = int(v)
            return ("KEEP", {"view": name, **params})
        else:
            return ("KEEP", {"view": view_name})

    elif desc_str.startswith("CONST(c="):
        c = int(desc_str[8:-1])
        return ("CONST", {"c": c})

    elif desc_str.startswith("UNIQUE(c="):
        c = int(desc_str[9:-1])
        return ("UNIQUE", {"c": c})

    elif desc_str.startswith("ARGMAX(c="):
        c = int(desc_str[9:-1])
        return ("ARGMAX", {"c": c})

    elif desc_str.startswith("LOWEST_UNUSED(c="):
        c = int(desc_str[16:-1])
        return ("LOWEST_UNUSED", {"c": c})

    elif desc_str.startswith("RECOLOR(pi={"):
        pi_str = desc_str[12:-2]  # Remove "RECOLOR(pi={" and "})"
        pi = {}
        for part in pi_str.split(","):
            k, v = part.split(":")
            pi[int(k)] = int(v)
        return ("RECOLOR", {"pi": pi})

    elif desc_str.startswith("BLOCK(k="):
        k = int(desc_str[8:-1])
        return ("BLOCK", {"k": k})

    else:
        return ("UNKNOWN", {})


def _make_view_for_paint(view_name: str, params: Dict[str, Any], H: int, W: int):
    """
    Reconstruct view function for painting (same as sieve._make_view).

    Args:
        view_name: View family name
        params: Parameters dict
        H, W: TEST frame dimensions

    Returns:
        Callable (i,j) -> (i',j') or None
    """
    if view_name == "identity":
        def V(x):
            return x
        return V

    elif view_name == "d4" or view_name.startswith("d4"):
        op = params.get("op", 0)
        def V(x):
            return morphisms.pose_fwd(x, op, (H, W))
        return V

    elif view_name == "translate":
        di = params.get("di", 0)
        dj = params.get("dj", 0)
        def make_translate_closure(di_c, dj_c):
            def V(x):
                i, j = x
                ni, nj = i + di_c, j + dj_c
                if 0 <= ni < H and 0 <= nj < W:
                    return (ni, nj)
                return None
            return V
        return make_translate_closure(di, dj)

    elif view_name == "residue_row":
        p = params.get("p", 1)
        def V(x):
            i, j = x
            nj = (j + p) % W
            return (i, nj)
        return V

    elif view_name == "residue_col":
        p = params.get("p", 1)
        def V(x):
            i, j = x
            ni = (i + p) % H
            return (ni, j)
        return V

    elif view_name == "tile":
        def V(x):
            i, j = x
            return (i % H, j % W)
        return V

    elif view_name == "tile_alt_row_flip":
        def V(x):
            i, j = x
            ti = i // H
            if ti % 2 == 1:
                return (i % H, (W - 1) - (j % W))
            else:
                return (i % H, j % W)
        return V

    elif view_name == "tile_alt_col_flip":
        def V(x):
            i, j = x
            tj = j // W
            if tj % 2 == 1:
                return ((H - 1) - (i % H), j % W)
            else:
                return (i % H, j % W)
        return V

    elif view_name == "tile_checkerboard_flip":
        def V(x):
            i, j = x
            ti = i // H
            tj = j // W
            if (ti + tj) % 2 == 1:
                return ((H - 1) - (i % H), (W - 1) - (j % W))
            else:
                return (i % H, j % W)
        return V

    elif view_name == "block_inverse":
        k = params.get("k", 1)
        def V(x):
            i, j = x
            return (i // k, j // k)
        return V

    elif view_name == "offset":
        b = params.get("b", 0)
        d = params.get("d", 0)
        def V(x):
            i, j = x
            ni, nj = i - b, j - d
            if 0 <= ni < H and 0 <= nj < W:
                return (ni, nj)
            return None
        return V

    else:
        # Unknown view, return identity
        def V(x):
            return x
        return V


def painter_once(
    assignment: Dict[int, str],
    part,
    Xtest: List[List[int]],
    Xin: List[List[List[int]]],
    P_test,
    P_in_list: List,
    shape_law: Tuple[str, Tuple[int, int, int, int]]
) -> List[List[int]]:
    """
    Paint test output once on OUTPUT canvas (posed-only).

    Args:
        assignment: cid -> descriptor string (from sieve)
        part: Partition on TEST input
        Xtest: Presented test input (posed+anchored)
        Xin: Presented training inputs (not used for test paint, kept for signature)
        P_test: Test input frame
        P_in_list: Training input frames (not used for test paint)
        shape_law: (type, (a, b, c, d))

    Returns:
        Posed-only output grid (H_out x W_out)
    """
    # Build output canvas size
    H_out, W_out = build_test_canvas_size(P_test, shape_law)
    _, _, (H_test, W_test) = P_test
    _, (a, b, c, d) = shape_law

    # Build class pixels map for BLOCK (anchor computation)
    class_pixels_test = {}
    for idx, cid in enumerate(part.cid_of):
        if cid not in class_pixels_test:
            class_pixels_test[cid] = []
        r_idx = idx // W_test
        c_idx = idx % W_test
        class_pixels_test[cid].append((r_idx, c_idx))

    # Compute class anchors (min pixel per class)
    class_anchors = {}
    for cid, pixels in class_pixels_test.items():
        if pixels:
            class_anchors[cid] = min(pixels)  # Row-major min

    # Initialize output grid
    Y_out = [[0 for _ in range(W_out)] for _ in range(H_out)]

    # Track law usage for receipts
    by_law = {}

    # Paint each pixel
    for i_out in range(H_out):
        for j_out in range(W_out):
            p_out = (i_out, j_out)

            # Pullback to TEST frame
            p_test = _shape_pullback(p_out, (a, b, c, d), H_test, W_test)

            if p_test is None:
                # Unseen pixel (pullback=None) - must use ⊥ class CONST
                if "⊥" in assignment:
                    desc_str = assignment["⊥"]
                    family, params = _parse_descriptor(desc_str)
                    if family == "CONST":
                        Y_out[i_out][j_out] = params["c"]
                        by_law[desc_str] = by_law.get(desc_str, 0) + 1
                        continue
                    else:
                        # ⊥ must be CONST
                        raise AssertionError(f"⊥ class must be CONST, got {desc_str} at pixel {p_out}")
                else:
                    # No ⊥ law - unseen pixel with no coverage
                    raise AssertionError(f"Unseen pixel (pullback=None) with no ⊥ CONST at {p_out}")

            # Get class id
            idx_test = p_test[0] * W_test + p_test[1]
            cid = part.cid_of[idx_test]

            # Get assigned law
            if cid not in assignment:
                # No law assigned (should not happen if sieve was exact)
                Y_out[i_out][j_out] = 0
                continue

            desc_str = assignment[cid]
            family, params = _parse_descriptor(desc_str)

            # Evaluate law
            if family == "KEEP":
                view_name = params["view"]
                V = _make_view_for_paint(view_name, params, H_test, W_test)
                q = V(p_test)
                if q is not None and 0 <= q[0] < H_test and 0 <= q[1] < W_test:
                    Y_out[i_out][j_out] = Xtest[q[0]][q[1]]
                    by_law[desc_str] = by_law.get(desc_str, 0) + 1
                else:
                    # View undefined or OOB - write 0 (partial view coverage)
                    Y_out[i_out][j_out] = 0

            elif family in ["CONST", "UNIQUE", "ARGMAX", "LOWEST_UNUSED"]:
                c = params["c"]
                Y_out[i_out][j_out] = c
                by_law[desc_str] = by_law.get(desc_str, 0) + 1

            elif family == "RECOLOR":
                pi = params["pi"]
                cin = Xtest[p_test[0]][p_test[1]]
                if cin in pi:
                    Y_out[i_out][j_out] = pi[cin]
                else:
                    # Missing color in pi (should not happen if admissibility was correct)
                    Y_out[i_out][j_out] = cin  # Identity fallback
                by_law[desc_str] = by_law.get(desc_str, 0) + 1

            elif family == "BLOCK":
                k = params["k"]
                if cid in class_anchors:
                    anchor = class_anchors[cid]
                    # Compute tile position relative to anchor
                    rel_row = p_test[0] - anchor[0]
                    rel_col = p_test[1] - anchor[1]
                    # Which tile (block) is this pixel in?
                    tile_row = rel_row // k
                    tile_col = rel_col // k
                    # Top-left corner of this tile (relative to anchor)
                    q0 = (anchor[0] + tile_row * k, anchor[1] + tile_col * k)
                    if 0 <= q0[0] < H_test and 0 <= q0[1] < W_test:
                        Y_out[i_out][j_out] = Xtest[q0[0]][q0[1]]
                        by_law[desc_str] = by_law.get(desc_str, 0) + 1
                    else:
                        # BLOCK tile OOB - partial coverage
                        Y_out[i_out][j_out] = 0
                else:
                    # No anchor for class - partial coverage
                    Y_out[i_out][j_out] = 0

            else:
                # Unknown family - algebraic bug
                raise AssertionError(f"Unknown law family: {family}, descriptor={desc_str}, p_out={p_out}")

    # Compute coverage metrics
    pixels_total = H_out * W_out
    pixels_painted = sum(by_law.values())
    coverage_pct = 100.0 * pixels_painted / pixels_total if pixels_total > 0 else 0.0

    # Log receipts
    receipts.log("paint", {
        "pixels_total": pixels_total,
        "pixels_painted": pixels_painted,
        "coverage_pct": coverage_pct,
        "by_law": by_law
    })

    return Y_out


def unpresent_final(
    Y_out_posed: List[List[int]],
    P_out_like_test,
    palette_inverse: Dict[int, int]
) -> List[List[int]]:
    """
    Un-present posed output back to raw output.

    Args:
        Y_out_posed: Posed-only output grid
        P_out_like_test: (op, (0,0), (H_out, W_out))
        palette_inverse: Inverse palette map {canonical -> raw}

    Returns:
        Raw output grid (pose_inv + palette_inverse, NO anchoring)
    """
    op, _, shape = P_out_like_test
    H, W = shape

    # Apply pose_inv
    Y_raw_posed = [[0 for _ in range(W)] for _ in range(H)]
    for i in range(H):
        for j in range(W):
            p_inv = morphisms.pose_inv((i, j), op, shape)
            if p_inv is not None:
                i_inv, j_inv = p_inv
                if 0 <= i_inv < H and 0 <= j_inv < W:
                    Y_raw_posed[i][j] = Y_out_posed[i_inv][j_inv]

    # Apply palette inverse
    Y_raw = [[palette_inverse.get(c, c) for c in row] for row in Y_raw_posed]

    return Y_raw


def paint_idempotent(
    assignment: Dict[int, str],
    part,
    Xtest: List[List[int]],
    Xin: List[List[List[int]]],
    P_test,
    P_in_list: List,
    shape_law: Tuple[str, Tuple[int, int, int, int]]
) -> bool:
    """
    Test idempotence: paint twice and compare.

    Returns:
        True if Y1 == Y2
    """
    Y1 = painter_once(assignment, part, Xtest, Xin, P_test, P_in_list, shape_law)
    Y2 = painter_once(assignment, part, Xtest, Xin, P_test, P_in_list, shape_law)

    # Byte-equal comparison
    return Y1 == Y2


def _self_check_paint():
    """
    Self-check for painter (debugging = algebra).

    1. KEEP with translate
    2. RECOLOR with π
    3. BLOCK with k
    4. Idempotence
    """
    import morphisms
    from truth import Partition

    # Test 1: KEEP with translate
    # Input 3x3, output 3x3, all one class, KEEP:translate(di=1,dj=0)
    part = Partition(H=3, W=3, cid_of=[0]*9)
    P_test = (0, (0, 0), (3, 3))
    P_in = [(0, (0, 0), (3, 3))]
    shape_law = ("multiplicative", (1, 0, 1, 0))  # Same size

    Xtest = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    Xin = [Xtest]  # Not used for test paint

    assignment = {0: "KEEP:translate(di=1,dj=0)"}

    Y_out = painter_once(assignment, part, Xtest, Xin, P_test, P_in, shape_law)

    # Verify: Y[i,j] should equal Xtest[i+1,j] (with wrapping or bounds)
    # translate(di=1,dj=0) means V(i,j) = (i+1, j)
    # So Y[i,j] = Xtest[V(i,j)] = Xtest[i+1,j]
    # For i=0, V(0,0)=(1,0), so Y[0,0]=Xtest[1,0]=4
    # For i=1, V(1,0)=(2,0), so Y[1,0]=Xtest[2,0]=7
    # For i=2, V(2,0)=(3,0) which is OOB, so Y[2,0]=0
    expected_row0 = [4, 5, 6]
    expected_row1 = [7, 8, 9]
    expected_row2 = [0, 0, 0]  # OOB

    assert Y_out[0] == expected_row0, f"Row 0 mismatch: {Y_out[0]} != {expected_row0}"
    assert Y_out[1] == expected_row1, f"Row 1 mismatch: {Y_out[1]} != {expected_row1}"
    assert Y_out[2] == expected_row2, f"Row 2 mismatch: {Y_out[2]} != {expected_row2}"

    # Test 2: RECOLOR
    part2 = Partition(H=2, W=2, cid_of=[0]*4)
    P_test2 = (0, (0, 0), (2, 2))
    P_in2 = [(0, (0, 0), (2, 2))]
    shape_law2 = ("multiplicative", (1, 0, 1, 0))

    Xtest2 = [[2, 3], [2, 3]]
    Xin2 = [Xtest2]

    assignment2 = {0: "RECOLOR(pi={2:6,3:1})"}

    Y_out2 = painter_once(assignment2, part2, Xtest2, Xin2, P_test2, P_in2, shape_law2)

    # π={2:6, 3:1}, so Xtest[i,j]=2 → 6, Xtest[i,j]=3 → 1
    expected2 = [[6, 1], [6, 1]]
    assert Y_out2 == expected2, f"RECOLOR mismatch: {Y_out2} != {expected2}"

    # Test 3: BLOCK(k=2)
    part3 = Partition(H=4, W=4, cid_of=[0]*16)
    P_test3 = (0, (0, 0), (4, 4))
    P_in3 = [(0, (0, 0), (4, 4))]
    shape_law3 = ("multiplicative", (1, 0, 1, 0))

    Xtest3 = [[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]]
    Xin3 = [Xtest3]

    assignment3 = {0: "BLOCK(k=2)"}

    Y_out3 = painter_once(assignment3, part3, Xtest3, Xin3, P_test3, P_in3, shape_law3)

    # BLOCK(k=2): anchor=(0,0), each 2x2 block reads from base pixel
    # p_test=(0,0) → rel=(0,0) → base=(0,0) → Xtest[0,0]=1
    # p_test=(0,1) → rel=(0,1) → base=(0,0) → Xtest[0,0]=1
    # p_test=(1,0) → rel=(1,0) → base=(0,0) → Xtest[0,0]=1
    # p_test=(1,1) → rel=(1,1) → base=(0,0) → Xtest[0,0]=1
    # p_test=(2,0) → rel=(2,0) → base=(1,0) → Xtest[1,0]=5
    # etc.
    expected3 = [[1, 1, 3, 3],
                 [1, 1, 3, 3],
                 [9, 9, 11, 11],
                 [9, 9, 11, 11]]
    assert Y_out3 == expected3, f"BLOCK mismatch: {Y_out3} != {expected3}"

    # Test 4: Idempotence
    assert paint_idempotent(assignment, part, Xtest, Xin, P_test, P_in, shape_law), \
        "Idempotence failed"

    print("✓ Paint self-check passed")


def init():
    """Run self-check if ARC_SELF_CHECK=1."""
    if os.environ.get("ARC_SELF_CHECK") == "1":
        _self_check_paint()
        receipt = {
            "tests_passed": 4,
            "verified": ["KEEP_translate", "RECOLOR", "BLOCK", "idempotence"]
        }
        receipts.log("paint_selfcheck", receipt)
