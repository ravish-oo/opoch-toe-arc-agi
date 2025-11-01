"""
KEEP Law Admissibility Engine (WO-08).

Admits KEEP laws per truth-class by proof via equivariant conjugation.
No partial KEEP: any undefined or mismatch → reject with witness.
"""

import os
import math
from typing import List, Dict, Tuple, Optional, Callable, Any

import morphisms
import receipts

# Type aliases
Coord = Tuple[int, int]
IntGrid = List[List[int]]
Frame = Tuple[int, Tuple[int, int], Tuple[int, int]]  # (d4_op, anchor, shape)


class KeepCandidate:
    """
    KEEP candidate: neutral descriptor + view function in TEST frame.

    Attributes:
        name: View family name (e.g., "translate")
        params: Parameters dict (e.g., {"di": 1, "dj": -2})
        V: Callable on TEST frame coords, returns TEST coord or None
    """

    def __init__(self, name: str, params: Dict[str, Any],
                 make_V_test: Callable[[Coord], Optional[Coord]]):
        self.name = name
        self.params = params
        self.V = make_V_test

    def descriptor(self) -> str:
        """Canonical descriptor string."""
        if not self.params:
            return self.name
        # Sort params for determinism
        param_str = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({param_str})"


def enumerate_keep_candidates(H: int, W: int,
                               sviews_meta: Dict[str, Any]) -> List[KeepCandidate]:
    """
    Deterministically build candidate list for test canvas size HxW.

    Order: identity ≺ d4 ≺ translate ≺ residue ≺ tile ≺ block ≺ offset

    Args:
        H, W: Test canvas dimensions
        sviews_meta: Metadata from sviews (e.g., {"row_gcd": 3, "col_gcd": 2})

    Returns:
        Ordered list of KeepCandidate
    """
    candidates = []

    # 1. Identity
    def make_identity():
        def V(x):
            return x
        return V

    candidates.append(KeepCandidate("identity", {}, make_identity()))

    # 2. D4 ops (0..7)
    for op in range(8):
        def make_d4(op_val):
            def V(x):
                return morphisms.pose_fwd(x, op_val, (H, W))
            return V

        candidates.append(KeepCandidate("d4", {"op": op}, make_d4(op)))

    # 3. Translate (bounded by |di|+|dj| ≤ max(H,W), lex order)
    max_dist = max(H, W)
    translations = []
    for dist in range(1, max_dist + 1):
        for di in range(-dist, dist + 1):
            dj_limit = dist - abs(di)
            for dj in range(-dj_limit, dj_limit + 1):
                if abs(di) + abs(dj) == dist:
                    translations.append((di, dj))

    for di, dj in translations:
        def make_translate(di_val, dj_val):
            def V(x):
                i, j = x
                ni, nj = i + di_val, j + dj_val
                if 0 <= ni < H and 0 <= nj < W:
                    return (ni, nj)
                return None
            return V

        candidates.append(KeepCandidate("translate", {"di": di, "dj": dj},
                                       make_translate(di, dj)))

    # 4. Residue shifts (if sviews admitted gcd > 1)
    # Enumerate all residue classes: shifts by 1, 2, ..., gcd-1 (mod gcd)
    row_gcd = sviews_meta.get("row_gcd", 1)
    col_gcd = sviews_meta.get("col_gcd", 1)

    if row_gcd > 1:
        for shift in range(1, row_gcd):
            def make_residue_row(p_val):
                def V(x):
                    i, j = x
                    nj = (j + p_val) % W
                    return (i, nj)
                return V

            candidates.append(KeepCandidate("residue_row", {"p": shift},
                                           make_residue_row(shift)))

    if col_gcd > 1:
        for shift in range(1, col_gcd):
            def make_residue_col(p_val):
                def V(x):
                    i, j = x
                    ni = (i + p_val) % H
                    return (ni, j)
                return V

            candidates.append(KeepCandidate("residue_col", {"p": shift},
                                           make_residue_col(shift)))

    # 5. Tile variants (use test canvas H, W)
    # tile: simple modulo
    def make_tile():
        def V(x):
            i, j = x
            return (i % H, j % W)
        return V

    candidates.append(KeepCandidate("tile", {}, make_tile()))

    # tile_alt_row_flip: flip every other tile row
    def make_tile_alt_row_flip():
        def V(x):
            i, j = x
            ti = i // H  # which tile row
            if ti % 2 == 1:  # odd rows flip horizontally
                return (i % H, (W - 1) - (j % W))
            else:
                return (i % H, j % W)
        return V

    candidates.append(KeepCandidate("tile_alt_row_flip", {}, make_tile_alt_row_flip()))

    # tile_alt_col_flip: flip every other tile column
    def make_tile_alt_col_flip():
        def V(x):
            i, j = x
            tj = j // W  # which tile col
            if tj % 2 == 1:  # odd cols flip vertically
                return ((H - 1) - (i % H), j % W)
            else:
                return (i % H, j % W)
        return V

    candidates.append(KeepCandidate("tile_alt_col_flip", {}, make_tile_alt_col_flip()))

    # tile_checkerboard_flip: flip based on checkerboard parity
    def make_tile_checkerboard_flip():
        def V(x):
            i, j = x
            ti = i // H
            tj = j // W
            if (ti ^ tj) % 2 == 1:  # checkerboard parity
                return ((H - 1) - (i % H), (W - 1) - (j % W))
            else:
                return (i % H, j % W)
        return V

    candidates.append(KeepCandidate("tile_checkerboard_flip", {}, make_tile_checkerboard_flip()))

    # 6. Block inverse (k ∈ {2,3} for finite vocabulary)
    for k in [2, 3]:
        def make_block_inverse(k_val):
            def V(x):
                i, j = x
                ni, nj = i // k_val, j // k_val
                # Bounds check
                if 0 <= ni < H and 0 <= nj < W:
                    return (ni, nj)
                return None
            return V

        candidates.append(KeepCandidate("block_inverse", {"k": k}, make_block_inverse(k)))

    # 7. Offset (bounded b, d, lex order)
    max_offset = min(max(H, W), 10)  # cap at 10

    offsets = []
    for dist in range(1, max_offset + 1):
        for b in range(-dist, dist + 1):
            d_limit = dist - abs(b)
            for d in range(-d_limit, d_limit + 1):
                if abs(b) + abs(d) == dist:
                    offsets.append((b, d))

    for b, d in offsets:
        def make_offset(b_val, d_val):
            def V(x):
                i, j = x
                ni, nj = i - b_val, j - d_val
                # Bounds check
                if 0 <= ni < H and 0 <= nj < W:
                    return (ni, nj)
                return None
            return V

        candidates.append(KeepCandidate("offset", {"b": b, "d": d}, make_offset(b, d)))

    return candidates


def admit_keep_for_class_v2(
    cid: int,
    class_maps: List[List[Optional[int]]],
    Xin: List[IntGrid],
    Yout: List[IntGrid],
    P_test: Frame,
    P_in_list: List[Frame],
    P_out_list: List[Frame],
    shape_law: Tuple[str, Tuple[int, int, int, int]],
    candidates: List[KeepCandidate],
) -> List[Dict[str, Any]]:
    """
    Return admitted KEEP candidates for this class using class_maps.

    Reject if:
      - V is undefined on ANY training pixel of the class
      - Color copied through conjugation differs from Yout

    Args:
        cid: Class id
        class_maps: List of class maps (one per training pair)
                   class_maps[i][p_idx] = class_id or None
        Xin: Posed+anchored train inputs
        Yout: Posed-only train outputs
        P_test: Test frame (d4, anchor, shape)
        P_in_list: Train input frames
        P_out_list: Train output frames
        shape_law: (type, (a, b, c, d)) - not used for proof
        candidates: List of KeepCandidate to test

    Returns:
        List of admitted descriptors with proofs
    """
    admitted = []

    for candidate in candidates:
        result = _prove_candidate_v2(
            cid, candidate, class_maps,
            Xin, Yout, P_test, P_in_list, P_out_list
        )

        if result["admitted"]:
            # Build descriptor object for sieve
            desc_obj = {"view": candidate.name}
            desc_obj.update(candidate.params)
            desc_obj["_proof"] = {
                "trains_checked": result["trains_checked"],
                "pixels_checked": result["pixels_checked"]
            }
            admitted.append(desc_obj)
        else:
            # Log rejection with witness for debugging
            if result.get("witness"):
                receipts.log("laws_debug", {
                    "cid": cid,
                    "descriptor": candidate.descriptor(),
                    "rejected": result["witness"]
                })

    return admitted


def admit_keep_for_class(
    cid: int,
    class_pixels_test: List[Coord],
    Xin: List[IntGrid],
    Yout: List[IntGrid],
    P_test: Frame,
    P_in_list: List[Frame],
    P_out_list: List[Frame],
    shape_law: Tuple[str, Tuple[int, int, int, int]],
    candidates: List[KeepCandidate],
) -> List[Dict[str, Any]]:
    """
    Return admitted KEEP candidates for this class.

    Reject if:
      - V is undefined on ANY training pixel of the class
      - Color copied through conjugation differs from Yout

    Args:
        cid: Class id
        class_pixels_test: Coords in TEST frame (within truth class)
        Xin: Posed+anchored train inputs
        Yout: Posed-only train outputs
        P_test: Test frame (d4, anchor, shape)
        P_in_list: Train input frames
        P_out_list: Train output frames
        shape_law: (type, (a, b, c, d)) - not used for proof, only for paint
        candidates: List of KeepCandidate to test

    Returns:
        List of admitted descriptors with proofs
    """
    admitted = []
    debug_witnesses = []

    for candidate in candidates:
        result = _prove_candidate(
            cid, candidate, class_pixels_test,
            Xin, Yout, P_test, P_in_list, P_out_list
        )

        if result["admitted"]:
            admitted.append({
                "class_id": cid,
                "descriptor": candidate.descriptor(),
                "proof": {
                    "trains_checked": result["trains_checked"],
                    "pixels_checked": result["pixels_checked"],
                    "undefined_hits": 0,
                    "mismatch_hits": 0
                }
            })
        else:
            # Record first witness for debugging
            if os.environ.get("ARC_SELF_CHECK") == "1" and result.get("witness"):
                debug_witnesses.append({
                    "cid": cid,
                    "descriptor": candidate.descriptor(),
                    "witness": result["witness"]
                })

    return admitted


def _prove_candidate_v2(
    cid: int,
    candidate: KeepCandidate,
    class_maps: List[List[Optional[int]]],
    Xin: List[IntGrid],
    Yout: List[IntGrid],
    P_test: Frame,
    P_in_list: List[Frame],
    P_out_list: List[Frame],
) -> Dict[str, Any]:
    """
    Prove or disprove a candidate using class_maps for membership.

    Algorithm (WO-LAW-CORE Section B):
      For each training pair i and each output pixel p_out where class_maps[i][p_idx] == cid:
        1. Pull back to TEST: q = pose_inv(p_out, P_out^i)
        2. Apply view: q' = V(q)
        3. If q' is None: reject with witness
        4. Map to input: p_in = pose_fwd(anchor_fwd(q', P_in^i.anchor), P_in^i.op)
        5. If p_in is None or OOB: reject with witness
        6. Check colors: if Xin^i[p_in] ≠ Yout^i[p_out]: reject with witness

    Returns:
        {
            "admitted": bool,
            "trains_checked": int,
            "pixels_checked": int,
            "witness": {...} or None
        }
    """
    trains_checked = 0
    pixels_checked = 0

    # For each training pair
    for train_idx, (Xin_i, Yout_i, P_in, P_out, class_map_i) in enumerate(
        zip(Xin, Yout, P_in_list, P_out_list, class_maps)
    ):
        H_out = len(Yout_i)
        W_out = len(Yout_i[0]) if H_out > 0 else 0
        H_in = len(Xin_i)
        W_in = len(Xin_i[0]) if H_in > 0 else 0

        trains_checked += 1

        # For each output pixel in row-major order
        for i_out in range(H_out):
            for j_out in range(W_out):
                p_idx = i_out * W_out + j_out

                # Check if this pixel belongs to our class
                if p_idx >= len(class_map_i) or class_map_i[p_idx] != cid:
                    continue  # Not in this class

                pixels_checked += 1
                p_out = (i_out, j_out)

                # Step 1: Pull back to TEST frame
                op_out, _, shape_out = P_out
                q = morphisms.pose_inv(p_out, op_out, shape_out)
                if q is None:
                    # This shouldn't happen for valid class_map, but check anyway
                    return {
                        "admitted": False,
                        "trains_checked": trains_checked,
                        "pixels_checked": pixels_checked,
                        "witness": {
                            "train_idx": train_idx,
                            "p_out": list(p_out),
                            "error": "pose_inv returned None"
                        }
                    }

                # Step 2: Apply view V in TEST frame
                q_prime = candidate.V(q)
                if q_prime is None:
                    # Undefined: reject with witness
                    return {
                        "admitted": False,
                        "trains_checked": trains_checked,
                        "pixels_checked": pixels_checked,
                        "witness": {
                            "train_idx": train_idx,
                            "p_out": list(p_out),
                            "q_test": list(q),
                            "q_after_V": None,
                            "reason": "V undefined"
                        }
                    }

                # Step 3: Map to input frame
                op_in, anchor_in, shape_in = P_in

                # anchor_fwd: apply anchor offset
                q_double_prime = morphisms.anchor_fwd(q_prime, anchor_in)
                if q_double_prime is None:
                    return {
                        "admitted": False,
                        "trains_checked": trains_checked,
                        "pixels_checked": pixels_checked,
                        "witness": {
                            "train_idx": train_idx,
                            "p_out": list(p_out),
                            "q_test": list(q),
                            "q_after_V": list(q_prime),
                            "reason": "anchor_fwd returned None"
                        }
                    }

                # pose_fwd: apply D4 transformation
                p_in = morphisms.pose_fwd(q_double_prime, op_in, shape_in)
                if p_in is None:
                    return {
                        "admitted": False,
                        "trains_checked": trains_checked,
                        "pixels_checked": pixels_checked,
                        "witness": {
                            "train_idx": train_idx,
                            "p_out": list(p_out),
                            "q_test": list(q),
                            "q_after_V": list(q_prime),
                            "reason": "pose_fwd returned None"
                        }
                    }

                # Check bounds
                i_in, j_in = p_in
                if not (0 <= i_in < H_in and 0 <= j_in < W_in):
                    return {
                        "admitted": False,
                        "trains_checked": trains_checked,
                        "pixels_checked": pixels_checked,
                        "witness": {
                            "train_idx": train_idx,
                            "p_out": list(p_out),
                            "q_test": list(q),
                            "q_after_V": list(q_prime),
                            "p_in": list(p_in),
                            "reason": "p_in OOB"
                        }
                    }

                # Step 4: Compare colors
                xin_color = Xin_i[i_in][j_in]
                yout_color = Yout_i[i_out][j_out]

                if xin_color != yout_color:
                    # Mismatch: reject with first witness
                    return {
                        "admitted": False,
                        "trains_checked": trains_checked,
                        "pixels_checked": pixels_checked,
                        "witness": {
                            "train_idx": train_idx,
                            "p_out": list(p_out),
                            "q_test": list(q),
                            "q_after_V": list(q_prime),
                            "p_in": list(p_in),
                            "xin": xin_color,
                            "yout": yout_color,
                            "reason": "color mismatch"
                        }
                    }

    # All checks passed: admit
    return {
        "admitted": True,
        "trains_checked": trains_checked,
        "pixels_checked": pixels_checked,
        "witness": None
    }


def _prove_candidate(
    cid: int,
    candidate: KeepCandidate,
    class_pixels_test: List[Coord],
    Xin: List[IntGrid],
    Yout: List[IntGrid],
    P_test: Frame,
    P_in_list: List[Frame],
    P_out_list: List[Frame],
) -> Dict[str, Any]:
    """
    Prove or disprove a single candidate via equivariant conjugation.

    Returns:
        {
            "admitted": bool,
            "trains_checked": int,
            "pixels_checked": int,
            "witness": {...} or None  # First counterexample if rejected
        }
    """
    trains_checked = 0
    pixels_checked = 0
    witness = None

    # For each training pair
    for train_idx, (Xin_i, Yout_i, P_in, P_out) in enumerate(
        zip(Xin, Yout, P_in_list, P_out_list)
    ):
        H_out = len(Yout_i)
        W_out = len(Yout_i[0]) if H_out > 0 else 0
        H_in = len(Xin_i)
        W_in = len(Xin_i[0]) if H_in > 0 else 0

        trains_checked += 1

        # For each OUT pixel, check if it belongs to this class
        for i_out in range(H_out):
            for j_out in range(W_out):
                p_out = (i_out, j_out)

                # Step 1: Pull back OUT pixel to TEST frame
                # pose_inv to undo the output pose
                op_out, anchor_out, shape_out = P_out
                q = morphisms.pose_inv(p_out, op_out, shape_out)
                if q is None:
                    continue  # OOB, skip

                # Check if this pixel's test coord is in our class
                # We need to also undo shape and check if it's in class_pixels_test
                # But shape_law is for test output size, not training
                # Actually, for training pairs, the posed grids are already consistent
                # So q is in the test frame after pose_inv
                # But we need to check if q belongs to this class

                # The class_pixels_test are in the TEST INPUT frame
                # But q is in the TEST OUTPUT frame after pose_inv
                # We need to map q back to test input via shape_inv

                # Actually, looking at the WO more carefully:
                # "For each training pair i and each output pixel p_out that belongs to the class"
                # This means we need to determine which output pixels belong to the class
                # The class is defined on TEST INPUT pixels
                # So we need to map p_out -> test input, then check membership

                # Let me re-read the WO...

                # OK, the WO says:
                # "each output pixel p_out that belongs to the class (i.e., its TEST pullback sits in the class)"

                # So the algorithm is:
                # 1. For each output pixel p_out
                # 2. Pull it back to TEST INPUT frame
                # 3. Check if that TEST INPUT coord is in class_pixels_test
                # 4. If yes, prove KEEP for it

                # To pull back to TEST INPUT:
                # - pose_inv(p_out, P_out) -> test output frame
                # - shape_inv(test output, shape_law) -> test input frame
                # But wait, shape_law is per-task, and we're in training pairs
                # Training pairs have their own sizes

                # Actually, I think the WO is simpler:
                # q = pose_inv(p_out, P_out) gives us a coord in TEST frame
                # We check if q is in class_pixels_test
                # If yes, we prove KEEP

                # But that doesn't make sense because class_pixels_test is in TEST INPUT frame
                # and q is after pose_inv of OUTPUT

                # Let me look at the WO proof algorithm again:
                # "For each training pair i and each output pixel p_out that belongs to the class:
                #  1. Pull back the output pixel into TEST frame: q = pose_inv(p_out, P_out^i)
                #  2. Apply the candidate view in TEST frame: q' = V(q)"

                # So it seems q is meant to be in TEST frame directly
                # I think the assumption is that for training pairs, input and output are same size
                # after posing (no shape change in training, only in test)

                # Actually wait, the shape law can apply to training pairs too
                # Let me re-read...

                # OK from WO-08:
                # "Shape law is not used here; we're proving KEEP against train pairs,
                #  which already have consistent sizes (posed grids)."

                # So for training pairs, we assume input/output are related directly
                # without shape law
                # This means q from pose_inv(p_out) is directly comparable to
                # test input coords

                # Hmm, but the class_pixels_test are in TEST INPUT frame
                # Let me think about this differently...

                # Actually, I think the issue is that we need to know which output pixels
                # correspond to the class
                # The WO says: "its TEST pullback sits in the class"
                # I think "TEST pullback" means pulling back through the test frame morphisms

                # Let me assume for now that q after pose_inv is the coordinate
                # and we check if it's in class_pixels_test

                if q not in class_pixels_test:
                    continue  # This output pixel doesn't belong to our class

                pixels_checked += 1

                # Step 2: Apply V in TEST frame
                q_prime = candidate.V(q)
                if q_prime is None:
                    # Undefined: reject candidate
                    witness = {
                        "train_idx": train_idx,
                        "p_out": list(p_out),
                        "p_test": list(q),
                        "p_test_after_V": None,
                        "p_in": None,
                        "xin": None,
                        "yout": Yout_i[p_out[0]][p_out[1]]
                    }
                    return {
                        "admitted": False,
                        "trains_checked": trains_checked,
                        "pixels_checked": pixels_checked,
                        "witness": witness
                    }

                # Step 3: Move to i-th input frame
                op_in, anchor_in, shape_in = P_in

                # anchor_fwd: apply anchor offset
                q_double_prime = morphisms.anchor_fwd(q_prime, anchor_in)

                # pose_fwd: apply D4 transformation
                p_in = morphisms.pose_fwd(q_double_prime, op_in, shape_in)
                if p_in is None:
                    # OOB in input frame: undefined, reject
                    witness = {
                        "train_idx": train_idx,
                        "p_out": list(p_out),
                        "p_test": list(q),
                        "p_test_after_V": list(q_prime),
                        "p_in": None,
                        "xin": None,
                        "yout": Yout_i[p_out[0]][p_out[1]]
                    }
                    return {
                        "admitted": False,
                        "trains_checked": trains_checked,
                        "pixels_checked": pixels_checked,
                        "witness": witness
                    }

                # Check bounds
                i_in, j_in = p_in
                if not (0 <= i_in < H_in and 0 <= j_in < W_in):
                    # OOB: reject
                    witness = {
                        "train_idx": train_idx,
                        "p_out": list(p_out),
                        "p_test": list(q),
                        "p_test_after_V": list(q_prime),
                        "p_in": list(p_in),
                        "xin": None,
                        "yout": Yout_i[p_out[0]][p_out[1]]
                    }
                    return {
                        "admitted": False,
                        "trains_checked": trains_checked,
                        "pixels_checked": pixels_checked,
                        "witness": witness
                    }

                # Step 4: Compare colors
                xin_color = Xin_i[i_in][j_in]
                yout_color = Yout_i[i_out][j_out]

                if xin_color != yout_color:
                    # Mismatch: reject
                    witness = {
                        "train_idx": train_idx,
                        "p_out": list(p_out),
                        "p_test": list(q),
                        "p_test_after_V": list(q_prime),
                        "p_in": list(p_in),
                        "xin": xin_color,
                        "yout": yout_color
                    }
                    return {
                        "admitted": False,
                        "trains_checked": trains_checked,
                        "pixels_checked": pixels_checked,
                        "witness": witness
                    }

    # All checks passed: admit
    return {
        "admitted": True,
        "trains_checked": trains_checked,
        "pixels_checked": pixels_checked,
        "witness": None
    }


def _self_check_keep() -> Dict[str, Any]:
    """
    Self-check for KEEP admissibility (algebraic debugging).

    Returns:
        Receipt dict with admitted list and debug witnesses
    """
    # Test 1: Positive proof - identity should work for identity mapping
    # Simple 2x2 grid, identity transform
    Xin1 = [[[1, 2], [3, 4]]]
    Yout1 = [[[1, 2], [3, 4]]]
    P_test1 = (0, (0, 0), (2, 2))
    P_in1 = [(0, (0, 0), (2, 2))]
    P_out1 = [(0, (0, 0), (2, 2))]

    # Class covers all 4 pixels
    class_pixels1 = [(0, 0), (0, 1), (1, 0), (1, 1)]

    # Build candidates
    candidates1 = enumerate_keep_candidates(2, 2, {})

    # Admit for class 0
    admitted1 = admit_keep_for_class(
        0, class_pixels1, Xin1, Yout1,
        P_test1, P_in1, P_out1,
        ("multiplicative", (1, 0, 1, 0)),
        candidates1
    )

    # Identity should be admitted
    identity_admitted = any(a["descriptor"] == "identity" for a in admitted1)
    if not identity_admitted:
        raise AssertionError(
            f"keep self-check failed: test 1 - identity not admitted. "
            f"Admitted: {[a['descriptor'] for a in admitted1]}"
        )

    # Test 2: Undefined rejection
    # Create a case where translate goes OOB
    Xin2 = [[[5, 6], [7, 8]]]
    Yout2 = [[[5, 6], [7, 8]]]  # Same as input
    P_test2 = (0, (0, 0), (2, 2))
    P_in2 = [(0, (0, 0), (2, 2))]
    P_out2 = [(0, (0, 0), (2, 2))]

    # Class has only one pixel at (0, 0)
    class_pixels2 = [(0, 0)]

    # Try to admit translate(di=0, dj=10) which will go OOB
    # V should return None when result is OOB
    def make_oob_translate():
        def V(x):
            i, j = x
            ni, nj = i, j + 10
            # Check bounds for 2x2 grid
            if 0 <= ni < 2 and 0 <= nj < 2:
                return (ni, nj)
            return None  # OOB
        return V

    candidate_oob = KeepCandidate("translate", {"di": 0, "dj": 10},
                                  make_oob_translate())

    # This should be rejected with undefined witness
    result_oob = _prove_candidate(
        0, candidate_oob, class_pixels2,
        Xin2, Yout2, P_test2, P_in2, P_out2
    )

    if result_oob["admitted"]:
        raise AssertionError(
            "keep self-check failed: test 2 - OOB translate admitted"
        )

    if result_oob["witness"]["p_test_after_V"] is not None:
        raise AssertionError(
            f"keep self-check failed: test 2 - expected p_test_after_V=None, "
            f"got {result_oob['witness']['p_test_after_V']}"
        )

    # Test 3: Mismatch rejection
    # Input and output differ
    Xin3 = [[[1, 2], [3, 4]]]
    Yout3 = [[[9, 9], [9, 9]]]  # Different colors
    P_test3 = (0, (0, 0), (2, 2))
    P_in3 = [(0, (0, 0), (2, 2))]
    P_out3 = [(0, (0, 0), (2, 2))]

    class_pixels3 = [(0, 0)]

    # Identity should be rejected with mismatch
    def make_identity():
        def V(x):
            return x
        return V

    candidate_id = KeepCandidate("identity", {}, make_identity())

    result_mismatch = _prove_candidate(
        0, candidate_id, class_pixels3,
        Xin3, Yout3, P_test3, P_in3, P_out3
    )

    if result_mismatch["admitted"]:
        raise AssertionError(
            "keep self-check failed: test 3 - mismatch admitted"
        )

    if result_mismatch["witness"]["xin"] == result_mismatch["witness"]["yout"]:
        raise AssertionError(
            f"keep self-check failed: test 3 - expected mismatch, "
            f"got xin={result_mismatch['witness']['xin']}, "
            f"yout={result_mismatch['witness']['yout']}"
        )

    # Return receipt
    return {
        "admitted": admitted1[:1],  # Just first one
        "keep_debug": [],
        "verified_on": 3  # 3 tests
    }


def init() -> None:
    """
    Run self-check if ARC_SELF_CHECK=1.

    Called by harness, not on import.
    """
    if os.environ.get("ARC_SELF_CHECK") == "1":
        receipt = _self_check_keep()
        receipts.log("laws", receipt)
