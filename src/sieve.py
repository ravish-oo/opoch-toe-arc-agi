#!/usr/bin/env python3
"""
Sieve: deterministic pruning to globally exact laws.

Prunes any class law that disagrees with any training output pixel.
Either returns exact assignment or missing_descriptor certificate.

Debugging = algebra: every removal logs first counterexample.
"""

import os
from typing import Dict, List, Tuple, Any, Optional
import morphisms
import receipts


Coord = Tuple[int, int]


# Fixed cost order (same as WO-09)
COST_ORDER = [
    "tile_alt_row_flip",
    "tile_alt_col_flip",
    "tile_alt_checkerboard_flip",
    "tile",
    "d4_",  # prefix match for d4_0, d4_1, etc.
    "identity",
    "RECOLOR",
    "BLOCK",
    "ARGMAX",
    "UNIQUE",
    "LOWEST_UNUSED",
    "CONST"
]


def _cost_of_descriptor(desc: str) -> int:
    """Return cost index of descriptor for sorting."""
    for idx, pattern in enumerate(COST_ORDER):
        if desc.startswith(pattern):
            return idx
    # Default (unknown) goes last
    return len(COST_ORDER)


def _parse_descriptor(desc_obj: Dict[str, Any]) -> Tuple[str, str, Any]:
    """
    Parse descriptor object from WO-08/WO-09.

    Returns: (family, descriptor_str, params)
    """
    if "view" in desc_obj:
        # KEEP
        return ("KEEP", desc_obj["view"], desc_obj)
    elif "c" in desc_obj:
        # CONST, UNIQUE, ARGMAX, LOWEST_UNUSED
        if desc_obj.get("type") == "CONST":
            return ("CONST", f"CONST(c={desc_obj['c']})", desc_obj)
        elif desc_obj.get("type") == "UNIQUE":
            return ("UNIQUE", f"UNIQUE(c={desc_obj['c']})", desc_obj)
        elif desc_obj.get("type") == "ARGMAX":
            return ("ARGMAX", f"ARGMAX(c={desc_obj['c']})", desc_obj)
        elif desc_obj.get("type") == "LOWEST_UNUSED":
            return ("LOWEST_UNUSED", f"LOWEST_UNUSED(c={desc_obj['c']})", desc_obj)
    elif "pi" in desc_obj:
        # RECOLOR
        pi_str = ",".join(f"{k}:{v}" for k, v in sorted(desc_obj["pi"].items()))
        return ("RECOLOR", f"RECOLOR(pi={{{pi_str}}})", desc_obj)
    elif "k" in desc_obj and "type" in desc_obj and desc_obj["type"] == "BLOCK":
        # BLOCK
        return ("BLOCK", f"BLOCK(k={desc_obj['k']})", desc_obj)

    # Fallback
    return ("UNKNOWN", str(desc_obj), desc_obj)


def _make_view(view_name: str, params: Dict[str, Any], H: int, W: int):
    """
    Reconstruct view function from descriptor (mirrors enumerate_keep_candidates).

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

    elif view_name.startswith("d4"):
        op = params.get("op", 0)
        def V(x):
            return morphisms.pose_fwd(x, op, (H, W))
        return V

    elif view_name == "translate":
        di = params.get("di", 0)
        dj = params.get("dj", 0)
        def V(x):
            i, j = x
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                return (ni, nj)
            return None
        return V

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


def _evaluate_keep(
    law_params: Dict[str, Any],
    p_out: Coord,
    Xin_i: List[List[int]],
    P_test,
    P_out_i,
    P_in_i
) -> Optional[Tuple[int, Dict[str, Any]]]:
    """
    Evaluate KEEP law on output pixel.

    Returns: (color, path_debug) or None if undefined

    Algorithm:
        p_out → pose_inv(P_out) → V_test → anchor_fwd(P_in.anchor) →
        pose_fwd(P_in) → p_in
    """
    view_name = law_params["view"]
    op_out, _, shape_out = P_out_i
    op_test, anchor_test, shape_test = P_test
    op_in, anchor_in, shape_in = P_in_i
    H_in, W_in = shape_in
    H_test, W_test = shape_test

    # OUT → TEST
    q = morphisms.pose_inv(p_out, op_out, shape_out)
    if q is None:
        return None

    p_test = q  # TEST frame (before applying V)

    # Apply V in TEST frame
    V = _make_view(view_name, law_params, H_test, W_test)
    q_prime = V(q)
    if q_prime is None:
        return None

    # TEST → IN (conjugation)
    q_double_prime = morphisms.anchor_fwd(q_prime, anchor_in)
    if q_double_prime is None:
        return None

    p_in = morphisms.pose_fwd(q_double_prime, op_in, shape_in)
    if p_in is None:
        return None

    # Check bounds
    if not (0 <= p_in[0] < H_in and 0 <= p_in[1] < W_in):
        return None

    color = Xin_i[p_in[0]][p_in[1]]
    path = {
        "p_test": list(p_test),
        "p_test_after_V": list(q_prime),
        "p_in": list(p_in)
    }
    return (color, path)


def _evaluate_value(
    law_params: Dict[str, Any],
    family: str,
    p_out: Coord,
    Xin_i: List[List[int]],
    P_test,
    P_out_i,
    P_in_i,
    class_pixels_test: List[Coord]
) -> Optional[Tuple[int, Dict[str, Any]]]:
    """
    Evaluate VALUE law on output pixel.

    Returns: (color, path_debug) or None if undefined
    """
    op_out, _, shape_out = P_out_i
    op_test, anchor_test, shape_test = P_test
    op_in, anchor_in, shape_in = P_in_i

    # Compute p_test from p_out (OUT → TEST)
    q = morphisms.pose_inv(p_out, op_out, shape_out)
    if q is None:
        return None
    p_test = q

    if family == "CONST":
        c = law_params["c"]
        return (c, {"p_test": list(p_test)})

    elif family in ["UNIQUE", "ARGMAX", "LOWEST_UNUSED"]:
        c = law_params["c"]
        return (c, {"p_test": list(p_test)})

    elif family == "RECOLOR":
        pi = law_params["pi"]

        # Compute p_in from p_test (TEST → IN via anchor_fwd + pose_fwd)
        q_double_prime = morphisms.anchor_fwd(p_test, anchor_in)
        if q_double_prime is None:
            return None
        p_in = morphisms.pose_fwd(q_double_prime, op_in, shape_in)
        if p_in is None:
            return None

        H_in, W_in = shape_in
        if not (0 <= p_in[0] < H_in and 0 <= p_in[1] < W_in):
            return None

        cin = Xin_i[p_in[0]][p_in[1]]
        if cin not in pi:
            return None

        cout = pi[cin]
        return (cout, {"p_test": list(p_test), "p_in": list(p_in), "cin": cin})

    elif family == "BLOCK":
        k = law_params["k"]
        anchor_coord = min(class_pixels_test)  # Class anchor in TEST frame

        # Compute relative position in TEST frame
        rel_row = p_test[0] - anchor_coord[0]
        rel_col = p_test[1] - anchor_coord[1]

        base_row = rel_row // k
        base_col = rel_col // k

        q0 = (anchor_coord[0] + base_row, anchor_coord[1] + base_col)

        # Map q0 to IN frame (TEST → IN)
        q_double_prime = morphisms.anchor_fwd(q0, anchor_in)
        if q_double_prime is None:
            return None
        p_in = morphisms.pose_fwd(q_double_prime, op_in, shape_in)
        if p_in is None:
            return None

        H_in, W_in = shape_in
        if not (0 <= p_in[0] < H_in and 0 <= p_in[1] < W_in):
            return None

        color = Xin_i[p_in[0]][p_in[1]]
        return (color, {"p_test": list(p_test), "q0": list(q0), "p_in": list(p_in)})

    return None


def run_sieve(
    part,
    class_maps: List[List[Optional[int]]],
    Xin: List[List[List[int]]],
    Yout: List[List[List[int]]],
    P_test,
    P_in_list: List,
    P_out_list: List,
    keep_admitted: Dict[int, List[Dict[str, Any]]],
    value_admitted: Dict[int, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Run deterministic sieve to prune laws and select globally exact assignment.

    Args:
        part: Partition on TEST input
        class_maps: Per-train row-major class_id arrays
        Xin: Presented training inputs
        Yout: Presented training outputs
        P_test: Test input frame
        P_in_list: Training input frames
        P_out_list: Training output frames
        keep_admitted: Per-class KEEP descriptors from WO-08
        value_admitted: Per-class VALUE descriptors from WO-09

    Returns:
        {
            "status": "exact" | "missing_descriptor",
            "assignment": {cid: descriptor_str},
            "cost_order": [...],
            "prune_log": [...],
            "missing": [...]  # if status="missing_descriptor"
        }
    """
    # Build unified candidate sets per class
    candidates = {}  # cid -> list of (family, descriptor_str, params)

    all_cids = set()
    for cid in keep_admitted.keys():
        all_cids.add(cid)
    for cid in value_admitted.keys():
        all_cids.add(cid)

    for cid in all_cids:
        candidates[cid] = []

        # Add KEEP laws
        for desc_obj in keep_admitted.get(cid, []):
            family, desc_str, params = _parse_descriptor(desc_obj)
            candidates[cid].append((family, desc_str, params))

        # Add VALUE laws
        for desc_obj in value_admitted.get(cid, []):
            family, desc_str, params = _parse_descriptor(desc_obj)
            candidates[cid].append((family, desc_str, params))

        # Sort by lex order of descriptor_str for determinism
        candidates[cid].sort(key=lambda x: x[1])

    # Prune log
    prune_log = []

    # Build class_pixels_test for BLOCK/VALUE evaluation
    class_pixels_test_by_cid = {}
    H_test, W_test = P_test[2]
    for idx, cid in enumerate(part.cid_of):
        if cid not in class_pixels_test_by_cid:
            class_pixels_test_by_cid[cid] = []
        r = idx // W_test
        c = idx % W_test
        class_pixels_test_by_cid[cid].append((r, c))

    # Sieve loop (fixed-point)
    max_passes = 100
    for pass_idx in range(max_passes):
        pruned_this_pass = False

        # Deterministic pass order: train idx asc, row-major pixels
        for i in range(len(Yout)):
            H_out = len(Yout[i])
            W_out = len(Yout[i][0]) if H_out > 0 else 0
            class_map_i = class_maps[i]
            P_in_i = P_in_list[i]
            P_out_i = P_out_list[i]
            Xin_i = Xin[i]
            Yout_i = Yout[i]

            for r in range(H_out):
                for c in range(W_out):
                    idx = r * W_out + c
                    cid = class_map_i[idx]
                    if cid is None:
                        continue

                    if cid not in candidates:
                        continue

                    expected = Yout_i[r][c]
                    p_out = (r, c)

                    # Evaluate each law in lex order
                    laws_to_prune = []
                    for family, desc_str, params in candidates[cid]:
                        # Evaluate law
                        if family == "KEEP":
                            result = _evaluate_keep(
                                params, p_out, Xin_i, P_test, P_out_i, P_in_i
                            )
                        else:
                            # VALUE
                            class_pixels_test = class_pixels_test_by_cid.get(cid, [])
                            result = _evaluate_value(
                                params, family, p_out, Xin_i,
                                P_test, P_out_i, P_in_i, class_pixels_test
                            )

                        if result is None:
                            # Undefined → mismatch (KEEP partiality rule)
                            got = None
                        else:
                            got, path = result

                        if got != expected:
                            # Prune this law
                            laws_to_prune.append((family, desc_str, params))
                            # Log first witness
                            witness = {
                                "cid": cid,
                                "descriptor": desc_str,
                                "train_idx": i,
                                "p_out": [r, c],
                                "expected": expected,
                                "got": got,
                                "path": path if result else {}
                            }
                            prune_log.append(witness)

                    # Remove pruned laws
                    for law in laws_to_prune:
                        if law in candidates[cid]:
                            candidates[cid].remove(law)
                            pruned_this_pass = True

        if not pruned_this_pass:
            break

    # Check for empty classes
    missing = []
    for cid in all_cids:
        if cid not in candidates or len(candidates[cid]) == 0:
            # Find examples
            examples = []
            for i in range(len(Yout)):
                H_out = len(Yout[i])
                W_out = len(Yout[i][0]) if H_out > 0 else 0
                class_map_i = class_maps[i]
                Yout_i = Yout[i]

                for r in range(H_out):
                    for c in range(W_out):
                        idx = r * W_out + c
                        if class_map_i[idx] == cid:
                            examples.append({
                                "train_idx": i,
                                "p_out": [r, c],
                                "expected": Yout_i[r][c]
                            })
                            if len(examples) >= 3:
                                break
                    if len(examples) >= 3:
                        break

            missing.append({
                "cid": cid,
                "examples": examples
            })

    if missing:
        return {
            "status": "missing_descriptor",
            "assignment": {},
            "cost_order": COST_ORDER,
            "prune_log": prune_log,
            "missing": missing
        }

    # Select least law per class
    assignment = {}
    for cid in all_cids:
        if cid not in candidates or len(candidates[cid]) == 0:
            continue

        # Sort by cost, then lex
        remaining = candidates[cid]
        remaining_sorted = sorted(
            remaining,
            key=lambda x: (_cost_of_descriptor(x[1]), x[1])
        )
        best = remaining_sorted[0]
        assignment[str(cid)] = best[1]

    return {
        "status": "exact",
        "assignment": assignment,
        "cost_order": COST_ORDER,
        "prune_log": prune_log
    }


def _self_check_sieve():
    """
    Self-check for sieve (debugging = algebra).

    1. Identity vs Tile paradox
    2. KEEP vs RECOLOR coverage
    3. Missing descriptor detection
    4. Determinism
    """
    import morphisms
    from truth import Partition

    # Test 1: Identity vs Tile paradox (simpler version)
    # Input 2x2, output 2x2 with simple copy
    # Both identity and tile work, expect identity (lower cost)

    # Build mock partition (single class)
    part = Partition(H=2, W=2, cid_of=[0, 0, 0, 0])  # All pixels in class 0

    # Frames (identity)
    P_test = (0, (0, 0), (2, 2))
    P_in = [(0, (0, 0), (2, 2))]
    P_out = [(0, (0, 0), (2, 2))]

    # Input = Output (simple copy)
    Xin = [[[1, 2], [3, 4]]]
    Yout = [[[1, 2], [3, 4]]]

    # Build class_map
    from class_map import build_class_map_i
    class_maps = [build_class_map_i(2, 2, P_test, P_out[0], part)]

    # Admitted laws: both work, but tile has lower cost (per cost order)
    keep_admitted = {
        0: [
            {"view": "identity"},
            {"view": "tile"}
        ]
    }
    value_admitted = {}

    result = run_sieve(
        part, class_maps, Xin, Yout, P_test, P_in, P_out,
        keep_admitted, value_admitted
    )

    assert result["status"] == "exact", f"Expected exact, got {result['status']}"
    assert "0" in result["assignment"], "Class 0 missing in assignment"
    assert "tile" in result["assignment"]["0"], \
        f"Expected tile (lower cost), got {result['assignment']['0']}"

    # Test 2: Missing descriptor
    # Change output so neither identity nor tile works

    Yout2 = [[[5, 5], [5, 5]]]  # Constant color, needs CONST law
    class_maps2 = [build_class_map_i(2, 2, P_test, P_out[0], part)]

    keep_admitted_incomplete = {0: [{"view": "identity"}, {"view": "tile"}]}

    result2 = run_sieve(
        part, class_maps2, Xin, Yout2, P_test, P_in, P_out,
        keep_admitted_incomplete, {}
    )

    assert result2["status"] == "missing_descriptor", \
        f"Expected missing_descriptor, got {result2['status']}"
    assert len(result2["missing"]) > 0, "Missing list is empty"
    assert result2["missing"][0]["cid"] == 0, "Expected cid=0 in missing"

    # Test 3: Determinism
    result3 = run_sieve(
        part, class_maps, Xin, Yout, P_test, P_in, P_out,
        keep_admitted, value_admitted
    )

    import json
    json1 = json.dumps(result, sort_keys=True)
    json3 = json.dumps(result3, sort_keys=True)
    assert json1 == json3, "Sieve is non-deterministic"

    print("✓ Sieve self-check passed")


def init():
    """Run self-check if ARC_SELF_CHECK=1."""
    if os.environ.get("ARC_SELF_CHECK") == "1":
        _self_check_sieve()
        receipt = {
            "tests_passed": 3,
            "verified": ["identity_vs_tile_paradox", "missing_descriptor", "determinism"]
        }
        receipts.log("sieve_selfcheck", receipt)
