"""
VALUE Law Admissibility Engine (WO-09).

Admits VALUE laws per truth-class by proof via observed pixel agreement.
Families: CONST, reducers (UNIQUE/ARGMAX/LOWEST_UNUSED), RECOLOR, BLOCK.
"""

import os
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import defaultdict

import morphisms
import receipts

# Type aliases
Coord = Tuple[int, int]
IntGrid = List[List[int]]
Frame = Tuple[int, Tuple[int, int], Tuple[int, int]]  # (d4_op, anchor, shape)


# ============================================================================
# HELPERS (frozen)
# ============================================================================


def get_obs_i(
    cid: int,
    class_pixels_test: List[Coord],
    Yout_i: IntGrid,
    P_out_i: Frame
) -> Set[Coord]:
    """
    H1: TEST→OUT witness set.

    Return set of output pixels that belong to this class.

    Obs_i(cid) = { p_out ∈ Yout[i] |
        q_test = pose_inv(p_out, P_out[i]);
        q_test ∈ class_pixels_test }

    Args:
        cid: Class id (for logging only)
        class_pixels_test: Test frame pixels for this class
        Yout_i: Output grid for training pair i
        P_out_i: Output frame for training pair i

    Returns:
        Set of (i, j) coords in Yout_i that map to class
    """
    H_out = len(Yout_i)
    W_out = len(Yout_i[0]) if H_out > 0 else 0

    obs = set()
    op_out, anchor_out, shape_out = P_out_i

    for i in range(H_out):
        for j in range(W_out):
            p_out = (i, j)

            # Inverse pose to get TEST frame coord
            q_test = morphisms.pose_inv(p_out, op_out, shape_out)

            if q_test is not None and q_test in class_pixels_test:
                obs.add(p_out)

    return obs


def test_to_in(
    p_test: Coord,
    P_test: Frame,
    P_in: Frame
) -> Optional[Coord]:
    """
    Map TEST frame pixel to INPUT frame via frame conjugation.

    p_test → pose_inv → anchor_inv → anchor_fwd → pose_fwd → p_in

    Args:
        p_test: Coord in TEST frame
        P_test: Test frame
        P_in: Input frame

    Returns:
        Coord in input frame or None if OOB
    """
    op_test, anchor_test, shape_test = P_test
    op_in, anchor_in, shape_in = P_in

    # pose_inv in TEST frame
    q0 = morphisms.pose_inv(p_test, op_test, shape_test)
    if q0 is None:
        return None

    # anchor_inv in TEST frame
    q1 = morphisms.anchor_inv(q0, anchor_test)

    # anchor_fwd in INPUT frame
    q2 = morphisms.anchor_fwd(q1, anchor_in)

    # pose_fwd in INPUT frame
    p_in = morphisms.pose_fwd(q2, op_in, shape_in)

    return p_in


def get_class_in_i(
    class_pixels_test: List[Coord],
    Xin_i: IntGrid,
    P_test: Frame,
    P_in: Frame
) -> Set[Coord]:
    """
    H2: TEST→IN class mask.

    Map class from TEST frame to INPUT frame via frame conjugation.

    class_in_i = { p_in | p_test ∈ class_pixels_test,
                   p_in = test_to_in(p_test, P_test, P_in) }

    Args:
        class_pixels_test: Test frame pixels for class
        Xin_i: Input grid for training pair i
        P_test: Test frame
        P_in: Input frame

    Returns:
        Set of coords in Xin_i that correspond to class
    """
    H_in = len(Xin_i)
    W_in = len(Xin_i[0]) if H_in > 0 else 0

    class_in = set()

    for p_test in class_pixels_test:
        p_in = test_to_in(p_test, P_test, P_in)

        if p_in is not None:
            i, j = p_in
            if 0 <= i < H_in and 0 <= j < W_in:
                class_in.add(p_in)

    return class_in


# ============================================================================
# VALUE FAMILIES
# ============================================================================


def _try_const(
    cid: int,
    class_pixels_test: List[Coord],
    Xin: List[IntGrid],
    Yout: List[IntGrid],
    P_test: Frame,
    P_in_list: List[Frame],
    P_out_list: List[Frame]
) -> Dict[str, Any]:
    """
    Try to admit CONST(c).

    Learn: For each train, Obs_i must have singleton color c_i.
           All c_i must be identical → c.
    Proof: All p_out ∈ Obs_i must have Yout[i][p_out] == c.

    Returns:
        {"admitted": bool, "c": int or None, "witness": {...} or None}
    """
    c_candidates = []

    for i, (Yout_i, P_out_i) in enumerate(zip(Yout, P_out_list)):
        obs_i = get_obs_i(cid, class_pixels_test, Yout_i, P_out_i)

        if not obs_i:
            continue  # No evidence from this train

        # Collect colors
        colors = set()
        for p_out in obs_i:
            colors.add(Yout_i[p_out[0]][p_out[1]])

        # Must be singleton
        if len(colors) != 1:
            return {
                "admitted": False,
                "c": None,
                "witness": {
                    "train_idx": i,
                    "p_out": list(min(obs_i)),  # First pixel
                    "colors_seen": sorted(colors)
                }
            }

        c_candidates.append(list(colors)[0])

    if not c_candidates:
        return {"admitted": False, "c": None, "witness": None}

    # All c_i must be identical
    if len(set(c_candidates)) != 1:
        return {
            "admitted": False,
            "c": None,
            "witness": {
                "train_idx": -1,
                "p_out": None,
                "colors_seen": sorted(set(c_candidates))
            }
        }

    c = c_candidates[0]

    # Proof: all observed pixels must equal c
    for i, (Yout_i, P_out_i) in enumerate(zip(Yout, P_out_list)):
        obs_i = get_obs_i(cid, class_pixels_test, Yout_i, P_out_i)

        for p_out in obs_i:
            if Yout_i[p_out[0]][p_out[1]] != c:
                return {
                    "admitted": False,
                    "c": c,
                    "witness": {
                        "train_idx": i,
                        "p_out": list(p_out),
                        "expected": c,
                        "got": Yout_i[p_out[0]][p_out[1]]
                    }
                }

    return {"admitted": True, "c": c, "witness": None}


def _try_unique(
    cid: int,
    class_pixels_test: List[Coord],
    Xin: List[IntGrid],
    Yout: List[IntGrid],
    P_test: Frame,
    P_in_list: List[Frame],
    P_out_list: List[Frame]
) -> Dict[str, Any]:
    """
    Try to admit UNIQUE.

    Learn: For each train i, compute colors on Xin[i][class_in_i].
           Must be singleton c_i. All c_i must be identical → c.
    Proof: All p_out ∈ Obs_i must have Yout[i][p_out] == c.
    """
    c_candidates = []

    for i, (Xin_i, Yout_i, P_in, P_out) in enumerate(zip(Xin, Yout, P_in_list, P_out_list)):
        class_in_i = get_class_in_i(class_pixels_test, Xin_i, P_test, P_in)

        if not class_in_i:
            continue  # No evidence

        # Collect input colors on class
        colors = set()
        for p_in in class_in_i:
            colors.add(Xin_i[p_in[0]][p_in[1]])

        # Must be unique (singleton)
        if len(colors) != 1:
            return {
                "admitted": False,
                "c": None,
                "witness": {
                    "train_idx": i,
                    "p_out": None,
                    "c_input": None,
                    "colors_in_class": sorted(colors),
                    "note": "not unique"
                }
            }

        c_candidates.append(list(colors)[0])

    if not c_candidates:
        return {"admitted": False, "c": None, "witness": None}

    # All c_i must be identical
    if len(set(c_candidates)) != 1:
        return {
            "admitted": False,
            "c": None,
            "witness": {
                "train_idx": -1,
                "colors_across_trains": sorted(set(c_candidates))
            }
        }

    c = c_candidates[0]

    # Proof: all observed pixels must equal c
    for i, (Yout_i, P_out_i) in enumerate(zip(Yout, P_out_list)):
        obs_i = get_obs_i(cid, class_pixels_test, Yout_i, P_out_i)

        for p_out in obs_i:
            if Yout_i[p_out[0]][p_out[1]] != c:
                return {
                    "admitted": False,
                    "c": c,
                    "witness": {
                        "train_idx": i,
                        "p_out": list(p_out),
                        "c_input": c,
                        "yout": Yout_i[p_out[0]][p_out[1]]
                    }
                }

    return {"admitted": True, "c": c, "witness": None}


def _try_argmax(
    cid: int,
    class_pixels_test: List[Coord],
    Xin: List[IntGrid],
    Yout: List[IntGrid],
    P_test: Frame,
    P_in_list: List[Frame],
    P_out_list: List[Frame]
) -> Dict[str, Any]:
    """
    Try to admit ARGMAX.

    Learn: For each train i, compute color with max count on class_in_i.
           Tie-break by smallest color value.
           All c_i must be identical → c.
    Proof: All p_out ∈ Obs_i must have Yout[i][p_out] == c.
    """
    c_candidates = []

    for i, (Xin_i, Yout_i, P_in, P_out) in enumerate(zip(Xin, Yout, P_in_list, P_out_list)):
        class_in_i = get_class_in_i(class_pixels_test, Xin_i, P_test, P_in)

        if not class_in_i:
            continue

        # Count colors
        color_counts = defaultdict(int)
        for p_in in class_in_i:
            color_counts[Xin_i[p_in[0]][p_in[1]]] += 1

        # ARGMAX: max count, tie-break by smallest color
        max_count = max(color_counts.values())
        candidates = [c for c, cnt in color_counts.items() if cnt == max_count]
        c_i = min(candidates)  # Smallest color value

        c_candidates.append(c_i)

    if not c_candidates:
        return {"admitted": False, "c": None, "witness": None}

    # All c_i must be identical
    if len(set(c_candidates)) != 1:
        return {
            "admitted": False,
            "c": None,
            "witness": {
                "train_idx": -1,
                "colors_across_trains": sorted(set(c_candidates))
            }
        }

    c = c_candidates[0]

    # Proof
    for i, (Yout_i, P_out_i) in enumerate(zip(Yout, P_out_list)):
        obs_i = get_obs_i(cid, class_pixels_test, Yout_i, P_out_i)

        for p_out in obs_i:
            if Yout_i[p_out[0]][p_out[1]] != c:
                return {
                    "admitted": False,
                    "c": c,
                    "witness": {
                        "train_idx": i,
                        "p_out": list(p_out),
                        "c_input": c,
                        "yout": Yout_i[p_out[0]][p_out[1]]
                    }
                }

    return {"admitted": True, "c": c, "witness": None}


def _try_lowest_unused(
    cid: int,
    class_pixels_test: List[Coord],
    Xin: List[IntGrid],
    Yout: List[IntGrid],
    P_test: Frame,
    P_in_list: List[Frame],
    P_out_list: List[Frame]
) -> Dict[str, Any]:
    """
    Try to admit LOWEST_UNUSED.

    Learn: For each train i, compute smallest color in 0..9 absent from class_in_i.
           All c_i must be identical → c.
    Proof: All p_out ∈ Obs_i must have Yout[i][p_out] == c.
    """
    c_candidates = []

    for i, (Xin_i, Yout_i, P_in, P_out) in enumerate(zip(Xin, Yout, P_in_list, P_out_list)):
        class_in_i = get_class_in_i(class_pixels_test, Xin_i, P_test, P_in)

        if not class_in_i:
            continue

        # Collect colors present
        colors_present = set()
        for p_in in class_in_i:
            colors_present.add(Xin_i[p_in[0]][p_in[1]])

        # Find lowest unused in 0..9
        c_i = None
        for c in range(10):
            if c not in colors_present:
                c_i = c
                break

        if c_i is None:
            return {
                "admitted": False,
                "c": None,
                "witness": {
                    "train_idx": i,
                    "note": "all colors 0..9 used in class"
                }
            }

        c_candidates.append(c_i)

    if not c_candidates:
        return {"admitted": False, "c": None, "witness": None}

    # All c_i must be identical
    if len(set(c_candidates)) != 1:
        return {
            "admitted": False,
            "c": None,
            "witness": {
                "train_idx": -1,
                "colors_across_trains": sorted(set(c_candidates))
            }
        }

    c = c_candidates[0]

    # Proof
    for i, (Yout_i, P_out_i) in enumerate(zip(Yout, P_out_list)):
        obs_i = get_obs_i(cid, class_pixels_test, Yout_i, P_out_i)

        for p_out in obs_i:
            if Yout_i[p_out[0]][p_out[1]] != c:
                return {
                    "admitted": False,
                    "c": c,
                    "witness": {
                        "train_idx": i,
                        "p_out": list(p_out),
                        "c_input": c,
                        "yout": Yout_i[p_out[0]][p_out[1]]
                    }
                }

    return {"admitted": True, "c": c, "witness": None}


def _try_recolor(
    cid: int,
    class_pixels_test: List[Coord],
    Xin: List[IntGrid],
    Yout: List[IntGrid],
    Xtest: IntGrid,
    P_test: Frame,
    P_in_list: List[Frame],
    P_out_list: List[Frame]
) -> Dict[str, Any]:
    """
    Try to admit RECOLOR(π).

    Learn: For each p_out ∈ Obs_i:
             p_test = pose_inv(p_out, P_out)
             p_in = test_to_in(p_test, P_test, P_in)
             cin = Xin[i][p_in], cout = Yout[i][p_out]
             Add (cin → cout) to π
           Merge across trains, reject on conflict.
    Coverage: π must cover all colors in Xtest[class_pixels_test].
    Proof: π[cin] == Yout[i][p_out] for all observed.
    """
    pi = {}  # cin → cout
    conflicts = []

    for i, (Xin_i, Yout_i, P_in, P_out) in enumerate(zip(Xin, Yout, P_in_list, P_out_list)):
        obs_i = get_obs_i(cid, class_pixels_test, Yout_i, P_out)

        op_out, anchor_out, shape_out = P_out

        for p_out in obs_i:
            # p_test = pose_inv(p_out, P_out)
            p_test = morphisms.pose_inv(p_out, op_out, shape_out)

            if p_test is None:
                continue

            # p_in = test_to_in(p_test, P_test, P_in)
            p_in = test_to_in(p_test, P_test, P_in)

            if p_in is None:
                continue

            # Check bounds
            H_in = len(Xin_i)
            W_in = len(Xin_i[0]) if H_in > 0 else 0
            if not (0 <= p_in[0] < H_in and 0 <= p_in[1] < W_in):
                continue

            cin = Xin_i[p_in[0]][p_in[1]]
            cout = Yout_i[p_out[0]][p_out[1]]

            # Merge
            if cin in pi:
                if pi[cin] != cout:
                    # Conflict
                    conflicts.append({
                        "color": cin,
                        "seen": sorted([pi[cin], cout]),
                        "train_idx": i,
                        "p_out": list(p_out)
                    })
            else:
                pi[cin] = cout

    # Check conflicts
    if conflicts:
        return {
            "admitted": False,
            "pi": None,
            "witness": conflicts[0]  # First conflict
        }

    # Coverage: π must cover all colors in Xtest[class_pixels_test]
    test_colors = set()
    H_test = len(Xtest)
    W_test = len(Xtest[0]) if H_test > 0 else 0

    for p_test in class_pixels_test:
        if 0 <= p_test[0] < H_test and 0 <= p_test[1] < W_test:
            test_colors.add(Xtest[p_test[0]][p_test[1]])

    missing = test_colors - set(pi.keys())
    if missing:
        return {
            "admitted": False,
            "pi": None,
            "witness": {
                "missing_input_color": min(missing),
                "where": "test_class"
            }
        }

    # Proof: π[cin] == Yout[i][p_out]
    for i, (Xin_i, Yout_i, P_in, P_out) in enumerate(zip(Xin, Yout, P_in_list, P_out_list)):
        obs_i = get_obs_i(cid, class_pixels_test, Yout_i, P_out)
        op_out, anchor_out, shape_out = P_out

        for p_out in obs_i:
            p_test = morphisms.pose_inv(p_out, op_out, shape_out)
            if p_test is None:
                continue

            p_in = test_to_in(p_test, P_test, P_in)
            if p_in is None:
                continue

            H_in = len(Xin_i)
            W_in = len(Xin_i[0]) if H_in > 0 else 0
            if not (0 <= p_in[0] < H_in and 0 <= p_in[1] < W_in):
                continue

            cin = Xin_i[p_in[0]][p_in[1]]
            expected = pi.get(cin)
            actual = Yout_i[p_out[0]][p_out[1]]

            if expected != actual:
                return {
                    "admitted": False,
                    "pi": pi,
                    "witness": {
                        "train_idx": i,
                        "p_out": list(p_out),
                        "cin": cin,
                        "expected": expected,
                        "actual": actual
                    }
                }

    return {"admitted": True, "pi": pi, "witness": None}


def _try_block(
    cid: int,
    class_pixels_test: List[Coord],
    Xin: List[IntGrid],
    Yout: List[IntGrid],
    P_test: Frame,
    P_in_list: List[Frame],
    P_out_list: List[Frame],
    k: int
) -> Dict[str, Any]:
    """
    Try to admit BLOCK(k).

    anchor = min(class_pixels_test) in row-major
    For p_out ∈ Obs_i:
        q_test = pose_inv(p_out, P_out)
        rel = q_test - anchor
        base = (rel.row // k, rel.col // k)
        q0 = anchor + base
        p_in = test_to_in(q0, P_test, P_in)
    Proof: Xin[i][p_in] == Yout[i][p_out]
    """
    if not class_pixels_test:
        return {"admitted": False, "k": k, "witness": None}

    # Anchor: min in row-major
    anchor = min(class_pixels_test)

    for i, (Xin_i, Yout_i, P_in, P_out) in enumerate(zip(Xin, Yout, P_in_list, P_out_list)):
        obs_i = get_obs_i(cid, class_pixels_test, Yout_i, P_out)
        op_out, anchor_out, shape_out = P_out

        H_in = len(Xin_i)
        W_in = len(Xin_i[0]) if H_in > 0 else 0

        for p_out in obs_i:
            # q_test = pose_inv(p_out, P_out)
            q_test = morphisms.pose_inv(p_out, op_out, shape_out)
            if q_test is None:
                continue

            # rel = q_test - anchor
            rel_row = q_test[0] - anchor[0]
            rel_col = q_test[1] - anchor[1]

            # base = floor(rel / k)
            base_row = rel_row // k
            base_col = rel_col // k

            # q0 = anchor + base
            q0 = (anchor[0] + base_row, anchor[1] + base_col)

            # p_in = test_to_in(q0, P_test, P_in)
            p_in = test_to_in(q0, P_test, P_in)

            if p_in is None:
                return {
                    "admitted": False,
                    "k": k,
                    "witness": {
                        "train_idx": i,
                        "p_out": list(p_out),
                        "q_test": list(q_test),
                        "base": [base_row, base_col],
                        "p_in": None,
                        "xin": None,
                        "yout": Yout_i[p_out[0]][p_out[1]]
                    }
                }

            # Check bounds
            if not (0 <= p_in[0] < H_in and 0 <= p_in[1] < W_in):
                return {
                    "admitted": False,
                    "k": k,
                    "witness": {
                        "train_idx": i,
                        "p_out": list(p_out),
                        "q_test": list(q_test),
                        "base": [base_row, base_col],
                        "p_in": list(p_in),
                        "xin": None,
                        "yout": Yout_i[p_out[0]][p_out[1]]
                    }
                }

            xin = Xin_i[p_in[0]][p_in[1]]
            yout = Yout_i[p_out[0]][p_out[1]]

            if xin != yout:
                return {
                    "admitted": False,
                    "k": k,
                    "witness": {
                        "train_idx": i,
                        "p_out": list(p_out),
                        "q_test": list(q_test),
                        "base": [base_row, base_col],
                        "p_in": list(p_in),
                        "xin": xin,
                        "yout": yout
                    }
                }

    return {"admitted": True, "k": k, "witness": None}


# ============================================================================
# MAIN API
# ============================================================================


def admit_value_for_class_v2(
    cid: int,
    class_pixels_test: List[Coord],
    class_maps: List[List[Optional[int]]],
    Xin: List[IntGrid],
    Yout: List[IntGrid],
    Xtest: IntGrid,
    P_test: Frame,
    P_in_list: List[Frame],
    P_out_list: List[Frame]
) -> List[Dict[str, Any]]:
    """
    Admit VALUE laws for this class using class_maps.

    Args:
        cid: Class id
        class_pixels_test: Test frame pixels for this class
        class_maps: List of class maps (one per training pair)
        Xin: Posed+anchored train inputs
        Yout: Posed-only train outputs
        Xtest: Test input (posed+anchored)
        P_test: Test frame
        P_in_list: Train input frames
        P_out_list: Train output frames

    Returns:
        List of admitted descriptors with proofs
    """
    admitted = []

    # Helper: get observed pixels from class_map_i
    def get_obs_from_class_map(class_map_i, Yout_i):
        H_out = len(Yout_i)
        W_out = len(Yout_i[0]) if H_out > 0 else 0
        obs = []
        for r in range(H_out):
            for c in range(W_out):
                p_idx = r * W_out + c
                if p_idx < len(class_map_i) and class_map_i[p_idx] == cid:
                    obs.append((r, c))
        return obs

    # 1. CONST
    c_candidates = []
    for i, (Yout_i, class_map_i) in enumerate(zip(Yout, class_maps)):
        obs_i = get_obs_from_class_map(class_map_i, Yout_i)
        if not obs_i:
            continue
        colors = set(Yout_i[p[0]][p[1]] for p in obs_i)
        if len(colors) == 1:
            c_candidates.append(list(colors)[0])
        else:
            c_candidates = []
            break

    if c_candidates and len(set(c_candidates)) == 1:
        c = c_candidates[0]
        # Verify proof
        proof_ok = True
        pixels_checked = 0
        for i, (Yout_i, class_map_i) in enumerate(zip(Yout, class_maps)):
            obs_i = get_obs_from_class_map(class_map_i, Yout_i)
            for p in obs_i:
                pixels_checked += 1
                if Yout_i[p[0]][p[1]] != c:
                    proof_ok = False
                    break
            if not proof_ok:
                break

        if proof_ok:
            admitted.append({
                "type": "CONST",
                "c": c,
                "_proof": {
                    "trains_checked": len(Yout),
                    "pixels_checked": pixels_checked
                }
            })

    # 2. UNIQUE
    c_candidates = []
    for i, (Xin_i, Yout_i, P_in, class_map_i) in enumerate(zip(Xin, Yout, P_in_list, class_maps)):
        class_in_i = get_class_in_i(class_pixels_test, Xin_i, P_test, P_in)

        if not class_in_i:
            continue

        # Collect input colors on class
        colors = set()
        for p_in in class_in_i:
            colors.add(Xin_i[p_in[0]][p_in[1]])

        # Must be unique (singleton)
        if len(colors) != 1:
            c_candidates = []
            break

        c_candidates.append(list(colors)[0])

    if c_candidates and len(set(c_candidates)) == 1:
        c = c_candidates[0]
        # Verify proof
        proof_ok = True
        pixels_checked = 0
        for i, (Yout_i, class_map_i) in enumerate(zip(Yout, class_maps)):
            obs_i = get_obs_from_class_map(class_map_i, Yout_i)
            for p in obs_i:
                pixels_checked += 1
                if Yout_i[p[0]][p[1]] != c:
                    proof_ok = False
                    break
            if not proof_ok:
                break

        if proof_ok:
            admitted.append({
                "type": "UNIQUE",
                "c": c,
                "_proof": {
                    "trains_checked": len(Yout),
                    "pixels_checked": pixels_checked
                }
            })

    # 3. ARGMAX
    c_candidates = []
    for i, (Xin_i, Yout_i, P_in, class_map_i) in enumerate(zip(Xin, Yout, P_in_list, class_maps)):
        class_in_i = get_class_in_i(class_pixels_test, Xin_i, P_test, P_in)

        if not class_in_i:
            continue

        # Count colors
        from collections import defaultdict
        color_counts = defaultdict(int)
        for p_in in class_in_i:
            color_counts[Xin_i[p_in[0]][p_in[1]]] += 1

        # ARGMAX: max count, tie-break by smallest color
        max_count = max(color_counts.values())
        candidates_argmax = [c for c, cnt in color_counts.items() if cnt == max_count]
        c_i = min(candidates_argmax)  # Smallest color value

        c_candidates.append(c_i)

    if c_candidates and len(set(c_candidates)) == 1:
        c = c_candidates[0]
        # Verify proof
        proof_ok = True
        pixels_checked = 0
        for i, (Yout_i, class_map_i) in enumerate(zip(Yout, class_maps)):
            obs_i = get_obs_from_class_map(class_map_i, Yout_i)
            for p in obs_i:
                pixels_checked += 1
                if Yout_i[p[0]][p[1]] != c:
                    proof_ok = False
                    break
            if not proof_ok:
                break

        if proof_ok:
            admitted.append({
                "type": "ARGMAX",
                "c": c,
                "_proof": {
                    "trains_checked": len(Yout),
                    "pixels_checked": pixels_checked
                }
            })

    # 4. LOWEST_UNUSED
    c_candidates = []
    for i, (Xin_i, Yout_i, P_in, class_map_i) in enumerate(zip(Xin, Yout, P_in_list, class_maps)):
        class_in_i = get_class_in_i(class_pixels_test, Xin_i, P_test, P_in)

        if not class_in_i:
            continue

        # Collect colors present
        colors_present = set()
        for p_in in class_in_i:
            colors_present.add(Xin_i[p_in[0]][p_in[1]])

        # Find lowest unused in 0..9
        c_i = None
        for c in range(10):
            if c not in colors_present:
                c_i = c
                break

        if c_i is None:
            c_candidates = []
            break

        c_candidates.append(c_i)

    if c_candidates and len(set(c_candidates)) == 1:
        c = c_candidates[0]
        # Verify proof
        proof_ok = True
        pixels_checked = 0
        for i, (Yout_i, class_map_i) in enumerate(zip(Yout, class_maps)):
            obs_i = get_obs_from_class_map(class_map_i, Yout_i)
            for p in obs_i:
                pixels_checked += 1
                if Yout_i[p[0]][p[1]] != c:
                    proof_ok = False
                    break
            if not proof_ok:
                break

        if proof_ok:
            admitted.append({
                "type": "LOWEST_UNUSED",
                "c": c,
                "_proof": {
                    "trains_checked": len(Yout),
                    "pixels_checked": pixels_checked
                }
            })

    # 5. RECOLOR(π)
    pi = {}  # cin → cout
    conflicts = []

    for i, (Xin_i, Yout_i, P_in, P_out, class_map_i) in enumerate(zip(Xin, Yout, P_in_list, P_out_list, class_maps)):
        obs_i = get_obs_from_class_map(class_map_i, Yout_i)
        op_out, anchor_out, shape_out = P_out

        for p_out in obs_i:
            # p_test = pose_inv(p_out, P_out)
            p_test = morphisms.pose_inv(p_out, op_out, shape_out)

            if p_test is None:
                continue

            # p_in = test_to_in(p_test, P_test, P_in)
            p_in = test_to_in(p_test, P_test, P_in)

            if p_in is None:
                continue

            # Check bounds
            H_in = len(Xin_i)
            W_in = len(Xin_i[0]) if H_in > 0 else 0
            if not (0 <= p_in[0] < H_in and 0 <= p_in[1] < W_in):
                continue

            cin = Xin_i[p_in[0]][p_in[1]]
            cout = Yout_i[p_out[0]][p_out[1]]

            # Merge
            if cin in pi:
                if pi[cin] != cout:
                    conflicts.append(cin)
                    break
            else:
                pi[cin] = cout

        if conflicts:
            break

    # Check conflicts
    if not conflicts:
        # Coverage: π must cover all colors in Xtest[class_pixels_test]
        test_colors = set()
        H_test = len(Xtest)
        W_test = len(Xtest[0]) if H_test > 0 else 0

        for p_test in class_pixels_test:
            if 0 <= p_test[0] < H_test and 0 <= p_test[1] < W_test:
                test_colors.add(Xtest[p_test[0]][p_test[1]])

        missing = test_colors - set(pi.keys())

        if not missing:
            # Proof: π[cin] == Yout[i][p_out]
            proof_ok = True
            pixels_checked = 0

            for i, (Xin_i, Yout_i, P_in, P_out, class_map_i) in enumerate(zip(Xin, Yout, P_in_list, P_out_list, class_maps)):
                obs_i = get_obs_from_class_map(class_map_i, Yout_i)
                op_out, anchor_out, shape_out = P_out

                for p_out in obs_i:
                    pixels_checked += 1
                    p_test = morphisms.pose_inv(p_out, op_out, shape_out)
                    if p_test is None:
                        continue

                    p_in = test_to_in(p_test, P_test, P_in)
                    if p_in is None:
                        continue

                    H_in = len(Xin_i)
                    W_in = len(Xin_i[0]) if H_in > 0 else 0
                    if not (0 <= p_in[0] < H_in and 0 <= p_in[1] < W_in):
                        continue

                    cin = Xin_i[p_in[0]][p_in[1]]
                    expected = pi.get(cin)
                    actual = Yout_i[p_out[0]][p_out[1]]

                    if expected != actual:
                        proof_ok = False
                        break

                if not proof_ok:
                    break

            if proof_ok:
                admitted.append({
                    "type": "RECOLOR",
                    "pi": dict(pi),
                    "_proof": {
                        "trains_checked": len(Yout),
                        "pixels_checked": pixels_checked
                    }
                })

    # 6. BLOCK(k) for k ∈ {2, 3}
    for k in [2, 3]:
        if not class_pixels_test:
            continue

        # Anchor: min in row-major
        anchor = min(class_pixels_test)

        proof_ok = True
        pixels_checked = 0

        for i, (Xin_i, Yout_i, P_in, P_out, class_map_i) in enumerate(zip(Xin, Yout, P_in_list, P_out_list, class_maps)):
            obs_i = get_obs_from_class_map(class_map_i, Yout_i)
            op_out, anchor_out, shape_out = P_out

            H_in = len(Xin_i)
            W_in = len(Xin_i[0]) if H_in > 0 else 0

            for p_out in obs_i:
                pixels_checked += 1
                # q_test = pose_inv(p_out, P_out)
                q_test = morphisms.pose_inv(p_out, op_out, shape_out)
                if q_test is None:
                    proof_ok = False
                    break

                # rel = q_test - anchor
                rel_row = q_test[0] - anchor[0]
                rel_col = q_test[1] - anchor[1]

                # base = floor(rel / k)
                base_row = rel_row // k
                base_col = rel_col // k

                # q0 = anchor + base
                q0 = (anchor[0] + base_row, anchor[1] + base_col)

                # p_in = test_to_in(q0, P_test, P_in)
                p_in = test_to_in(q0, P_test, P_in)

                if p_in is None:
                    proof_ok = False
                    break

                # Check bounds
                if not (0 <= p_in[0] < H_in and 0 <= p_in[1] < W_in):
                    proof_ok = False
                    break

                xin = Xin_i[p_in[0]][p_in[1]]
                yout = Yout_i[p_out[0]][p_out[1]]

                if xin != yout:
                    proof_ok = False
                    break

            if not proof_ok:
                break

        if proof_ok:
            admitted.append({
                "type": "BLOCK",
                "k": k,
                "_proof": {
                    "trains_checked": len(Yout),
                    "pixels_checked": pixels_checked,
                    "anchor": list(anchor)
                }
            })

    return admitted


def admit_value_for_class(
    cid: int,
    class_pixels_test: List[Coord],
    Xin: List[IntGrid],
    Yout: List[IntGrid],
    Xtest: IntGrid,
    P_test: Frame,
    P_in_list: List[Frame],
    P_out_list: List[Frame]
) -> Dict[str, Any]:
    """
    Admit VALUE laws for this class.

    Args:
        cid: Class id
        class_pixels_test: TEST frame pixels for this class
        Xin: Posed+anchored train inputs
        Yout: Posed-only train outputs
        Xtest: Test input (posed+anchored)
        P_test: Test frame
        P_in_list: Train input frames
        P_out_list: Train output frames

    Returns:
        {"cid": cid, "admitted": [...], "debug": [...]}
    """
    admitted = []
    debug = []

    # 1. CONST
    result_const = _try_const(cid, class_pixels_test, Xin, Yout, P_test, P_in_list, P_out_list)
    if result_const["admitted"]:
        admitted.append(f"CONST(c={result_const['c']})")
    elif os.environ.get("ARC_SELF_CHECK") == "1" and result_const["witness"]:
        debug.append({
            "descriptor": "CONST",
            "witness": result_const["witness"]
        })

    # 2. UNIQUE
    result_unique = _try_unique(cid, class_pixels_test, Xin, Yout, P_test, P_in_list, P_out_list)
    if result_unique["admitted"]:
        admitted.append("UNIQUE")
    elif os.environ.get("ARC_SELF_CHECK") == "1" and result_unique["witness"]:
        debug.append({
            "descriptor": "UNIQUE",
            "witness": result_unique["witness"]
        })

    # 3. ARGMAX
    result_argmax = _try_argmax(cid, class_pixels_test, Xin, Yout, P_test, P_in_list, P_out_list)
    if result_argmax["admitted"]:
        admitted.append("ARGMAX")
    elif os.environ.get("ARC_SELF_CHECK") == "1" and result_argmax["witness"]:
        debug.append({
            "descriptor": "ARGMAX",
            "witness": result_argmax["witness"]
        })

    # 4. LOWEST_UNUSED
    result_lowest = _try_lowest_unused(cid, class_pixels_test, Xin, Yout, P_test, P_in_list, P_out_list)
    if result_lowest["admitted"]:
        admitted.append("LOWEST_UNUSED")
    elif os.environ.get("ARC_SELF_CHECK") == "1" and result_lowest["witness"]:
        debug.append({
            "descriptor": "LOWEST_UNUSED",
            "witness": result_lowest["witness"]
        })

    # 5. RECOLOR
    result_recolor = _try_recolor(cid, class_pixels_test, Xin, Yout, Xtest, P_test, P_in_list, P_out_list)
    if result_recolor["admitted"]:
        # Canonical descriptor: sorted keys
        pi_str = ",".join(f"{k}:{v}" for k, v in sorted(result_recolor["pi"].items()))
        admitted.append(f"RECOLOR(pi={{{pi_str}}})")
    elif os.environ.get("ARC_SELF_CHECK") == "1" and result_recolor["witness"]:
        debug.append({
            "descriptor": "RECOLOR",
            "witness": result_recolor["witness"]
        })

    # 6. BLOCK (enumerate k ∈ {2, 3, 4})
    for k in [2, 3, 4]:
        result_block = _try_block(cid, class_pixels_test, Xin, Yout, P_test, P_in_list, P_out_list, k)
        if result_block["admitted"]:
            admitted.append(f"BLOCK(k={k})")
        elif os.environ.get("ARC_SELF_CHECK") == "1" and result_block["witness"]:
            debug.append({
                "descriptor": f"BLOCK(k={k})",
                "witness": result_block["witness"]
            })

    return {
        "cid": cid,
        "admitted": admitted,
        "debug": debug
    }


# ============================================================================
# SELF-CHECK
# ============================================================================


def _self_check_value() -> Dict[str, Any]:
    """
    Self-check for VALUE admissibility (algebraic debugging).

    Returns:
        Receipt dict with admitted + value_debug
    """
    # Test 1: CONST positive and negative
    Xin1 = [[[1, 2], [3, 4]]]
    Yout1 = [[[5, 5], [5, 5]]]  # All 5s
    Xtest1 = [[1, 2], [3, 4]]
    P_test1 = (0, (0, 0), (2, 2))
    P_in1 = [(0, (0, 0), (2, 2))]
    P_out1 = [(0, (0, 0), (2, 2))]
    class_pixels1 = [(0, 0), (0, 1), (1, 0), (1, 1)]

    result1 = admit_value_for_class(0, class_pixels1, Xin1, Yout1, Xtest1, P_test1, P_in1, P_out1)

    if "CONST(c=5)" not in result1["admitted"]:
        raise AssertionError(
            f"value self-check failed: test 1 - CONST(c=5) not admitted. "
            f"Admitted: {result1['admitted']}"
        )

    # Test 2: UNIQUE (output must equal unique input color)
    Xin2 = [[[7, 7], [7, 7]]]  # All same color
    Yout2 = [[[7, 7], [7, 7]]]  # Output equals unique input color
    Xtest2 = [[7, 7], [7, 7]]
    P_test2 = (0, (0, 0), (2, 2))
    P_in2 = [(0, (0, 0), (2, 2))]
    P_out2 = [(0, (0, 0), (2, 2))]
    class_pixels2 = [(0, 0), (0, 1), (1, 0), (1, 1)]

    result2 = admit_value_for_class(0, class_pixels2, Xin2, Yout2, Xtest2, P_test2, P_in2, P_out2)

    if "UNIQUE" not in result2["admitted"]:
        raise AssertionError(
            f"value self-check failed: test 2 - UNIQUE not admitted. "
            f"Admitted: {result2['admitted']}"
        )

    # Test 3: RECOLOR
    Xin3 = [[[1, 2], [1, 2]]]
    Yout3 = [[[6, 7], [6, 7]]]  # 1→6, 2→7
    Xtest3 = [[1, 2], [1, 2]]
    P_test3 = (0, (0, 0), (2, 2))
    P_in3 = [(0, (0, 0), (2, 2))]
    P_out3 = [(0, (0, 0), (2, 2))]
    class_pixels3 = [(0, 0), (0, 1), (1, 0), (1, 1)]

    result3 = admit_value_for_class(0, class_pixels3, Xin3, Yout3, Xtest3, P_test3, P_in3, P_out3)

    # Should have RECOLOR with π = {1:6, 2:7}
    recolor_found = any("RECOLOR" in desc for desc in result3["admitted"])
    if not recolor_found:
        raise AssertionError(
            f"value self-check failed: test 3 - RECOLOR not admitted. "
            f"Admitted: {result3['admitted']}"
        )

    # Return receipt
    return {
        "admitted": [
            {"class_id": 0, "descriptor": "CONST(c=5)",
             "proof": {"trains_checked": 1, "pixels_checked": 4}}
        ],
        "value_debug": [],
        "verified_on": 3
    }


def init() -> None:
    """
    Run self-check if ARC_SELF_CHECK=1.

    Called by harness, not on import.
    """
    if os.environ.get("ARC_SELF_CHECK") == "1":
        receipt = _self_check_value()
        receipts.log("laws", receipt)
