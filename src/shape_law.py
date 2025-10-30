"""
Shape Law: Affine size map with floor pullback (WO-07).

Learns the minimal affine law S(H,W) = (aH+b, cW+d) from training sizes
with strict precedence: multiplicative ≺ additive ≺ mixed ≺ bbox.

Pullback uses floor mapping with OOB-only skips.
"""

import os
from typing import List, Tuple, Literal, Optional

import receipts

# Type aliases
SizeQuad = Tuple[int, int, int, int]  # (Hin, Win, Hout, Wout)
Law = Tuple[int, int, int, int]  # (a, b, c, d)
LawType = Literal["multiplicative", "additive", "mixed", "bbox"]


def learn_law(sizes: List[SizeQuad]) -> Tuple[LawType, Law]:
    """
    Fit the smallest-precedence law that matches ALL training size pairs.

    Precedence: multiplicative ≺ additive ≺ mixed ≺ bbox.

    Args:
        sizes: List of (Hin, Win, Hout, Wout) from posed training pairs

    Returns:
        (law_type, (a, b, c, d))

    Raises:
        AssertionError: If no law fits, with first counterexample
    """
    if not sizes:
        raise AssertionError("learn_law: empty size list")

    # 1. Try multiplicative: b=d=0, a and c constant positive integers
    mult_law = _try_multiplicative(sizes)
    if mult_law is not None:
        return ("multiplicative", mult_law)

    # 2. Try additive: a=c=1, b and d constant ≥0
    add_law = _try_additive(sizes)
    if add_law is not None:
        return ("additive", add_law)

    # 3. Try mixed: a,c ≥ 1, at least one of b,d nonzero or one of a,c ≠ 1
    mixed_law = _try_mixed(sizes)
    if mixed_law is not None:
        return ("mixed", mixed_law)

    # 4. Bbox fallback (requires posed inputs, not available here)
    # For now, raise; bbox will be handled when we have actual grids
    raise AssertionError(
        f"shape fit failed: no affine law matches all {len(sizes)} pairs. "
        f"First pair: {sizes[0]}"
    )


def _try_multiplicative(sizes: List[SizeQuad]) -> Optional[Law]:
    """Try multiplicative: h_k/H_k and w_k/W_k constant positive integers."""
    if not sizes:
        return None

    # Compute a_k = h_k / H_k, c_k = w_k / W_k
    a_candidates = []
    c_candidates = []

    for Hin, Win, Hout, Wout in sizes:
        if Hin == 0 or Win == 0:
            return None

        # Check if Hout is divisible by Hin
        if Hout % Hin != 0:
            return None
        a_k = Hout // Hin

        # Check if Wout is divisible by Win
        if Wout % Win != 0:
            return None
        c_k = Wout // Win

        if a_k <= 0 or c_k <= 0:
            return None

        a_candidates.append(a_k)
        c_candidates.append(c_k)

    # Check if all a_k are identical and all c_k are identical
    if len(set(a_candidates)) == 1 and len(set(c_candidates)) == 1:
        a = a_candidates[0]
        c = c_candidates[0]
        return (a, 0, c, 0)

    return None


def _try_additive(sizes: List[SizeQuad]) -> Optional[Law]:
    """Try additive: a=c=1, b_k and d_k constant ≥0."""
    if not sizes:
        return None

    # Compute b_k = h_k - H_k, d_k = w_k - W_k
    b_candidates = []
    d_candidates = []

    for Hin, Win, Hout, Wout in sizes:
        b_k = Hout - Hin
        d_k = Wout - Win

        if b_k < 0 or d_k < 0:
            return None

        b_candidates.append(b_k)
        d_candidates.append(d_k)

    # Check if all b_k identical and all d_k identical
    if len(set(b_candidates)) == 1 and len(set(d_candidates)) == 1:
        b = b_candidates[0]
        d = d_candidates[0]
        return (1, b, 1, d)

    return None


def _try_mixed(sizes: List[SizeQuad]) -> Optional[Law]:
    """Try mixed: solve for (a,c) from ratios, then (b,d) from offsets."""
    if not sizes:
        return None

    # Compute candidate a: check if all H_k | h_k and h_k/H_k constant
    a = None
    all_divisible_h = True
    h_ratios = []

    for Hin, Win, Hout, Wout in sizes:
        if Hin == 0:
            all_divisible_h = False
            break
        if Hout % Hin != 0:
            all_divisible_h = False
            break
        h_ratios.append(Hout // Hin)

    if all_divisible_h and len(set(h_ratios)) == 1:
        a = h_ratios[0]
    else:
        a = 1

    # Compute candidate c: check if all W_k | w_k and w_k/W_k constant
    c = None
    all_divisible_w = True
    w_ratios = []

    for Hin, Win, Hout, Wout in sizes:
        if Win == 0:
            all_divisible_w = False
            break
        if Wout % Win != 0:
            all_divisible_w = False
            break
        w_ratios.append(Wout // Win)

    if all_divisible_w and len(set(w_ratios)) == 1:
        c = w_ratios[0]
    else:
        c = 1

    # Solve for b and d
    b_candidates = []
    d_candidates = []

    for Hin, Win, Hout, Wout in sizes:
        b_k = Hout - a * Hin
        d_k = Wout - c * Win

        if b_k < 0 or d_k < 0:
            return None

        b_candidates.append(b_k)
        d_candidates.append(d_k)

    # Check if all b_k identical and all d_k identical
    if len(set(b_candidates)) != 1 or len(set(d_candidates)) != 1:
        return None

    b = b_candidates[0]
    d = d_candidates[0]

    # Reject if collapses to multiplicative or additive
    if b == 0 and d == 0:
        return None  # Multiplicative
    if a == 1 and c == 1:
        return None  # Additive

    return (a, b, c, d)


def pullback(i_out: int, j_out: int, law: Law, Hin: int, Win: int) -> Optional[Tuple[int, int]]:
    """
    Floor pullback: map output pixel to input pixel.

    Formula: i_in = floor((i_out - b) / a), j_in = floor((j_out - d) / c)

    Args:
        i_out, j_out: Output pixel coordinates
        law: (a, b, c, d)
        Hin, Win: Input dimensions

    Returns:
        (i_in, j_in) or None if OOB
    """
    a, b, c, d = law

    # Floor mapping
    i_in = (i_out - b) // a
    j_in = (j_out - d) // c

    # Check bounds
    if 0 <= i_in < Hin and 0 <= j_in < Win:
        return (i_in, j_in)

    return None


def log_shape_receipt(law_type: LawType, law: Law, verified_on: int) -> None:
    """
    Log shape receipt to receipts system.

    Args:
        law_type: "multiplicative" | "additive" | "mixed" | "bbox"
        law: (a, b, c, d)
        verified_on: Number of training pairs verified
    """
    receipts.log("shape", {
        "type": law_type,
        "law": list(law),
        "verified_on": verified_on
    })


def _self_check_shape() -> dict:
    """
    Self-check for shape law (algebraic debugging).

    Runs 5 tests with exact expected outputs.
    Raises AssertionError with counterexample on failure.

    Returns:
        Receipt dict with (type, law, verified_on) showing test coverage
    """
    # Test 1: Multiplicative
    sizes1 = [(2, 3, 6, 9), (4, 5, 12, 15)]
    law_type1, law1 = learn_law(sizes1)
    if law_type1 != "multiplicative" or law1 != (3, 0, 3, 0):
        raise AssertionError(
            f"multiplicative failed: got ({law_type1}, {law1}), "
            f"want ('multiplicative', (3, 0, 3, 0))"
        )

    # Test 2: Additive
    sizes2 = [(5, 7, 7, 10), (3, 4, 5, 7)]
    law_type2, law2 = learn_law(sizes2)
    if law_type2 != "additive" or law2 != (1, 2, 1, 3):
        raise AssertionError(
            f"additive failed: got ({law_type2}, {law2}), "
            f"want ('additive', (1, 2, 1, 3))"
        )

    # Test 3: Mixed
    sizes3 = [(3, 4, 9, 6), (5, 4, 15, 6)]
    law_type3, law3 = learn_law(sizes3)
    if law_type3 != "mixed" or law3 != (3, 0, 1, 2):
        raise AssertionError(
            f"mixed failed: got ({law_type3}, {law3}), "
            f"want ('mixed', (3, 0, 1, 2))"
        )

    # Test 4: Pullback floor semantics (using law from test 1)
    # Law: (3, 0, 3, 0), so i_in = i_out // 3, j_in = j_out // 3
    # Test pixel at (7, 8): floor(7/3)=2, floor(8/3)=2
    pb1 = pullback(7, 8, (3, 0, 3, 0), 10, 10)
    if pb1 != (2, 2):
        raise AssertionError(
            f"pullback floor failed: pullback(7, 8, (3,0,3,0), 10, 10) = {pb1}, "
            f"want (2, 2)"
        )

    # Test pixel at (0, 0): floor(0/3)=0
    pb2 = pullback(0, 0, (3, 0, 3, 0), 10, 10)
    if pb2 != (0, 0):
        raise AssertionError(
            f"pullback floor failed: pullback(0, 0, (3,0,3,0), 10, 10) = {pb2}, "
            f"want (0, 0)"
        )

    # Test OOB: pixel at (50, 50) with input size (10, 10)
    pb3 = pullback(50, 50, (3, 0, 3, 0), 10, 10)
    if pb3 is not None:
        raise AssertionError(
            f"pullback OOB failed: pullback(50, 50, (3,0,3,0), 10, 10) = {pb3}, "
            f"want None"
        )

    # Test 5: Pullback with additive law (1, 2, 1, 3)
    # i_in = (i_out - 2) // 1 = i_out - 2, j_in = (j_out - 3) // 1 = j_out - 3
    pb4 = pullback(5, 6, (1, 2, 1, 3), 10, 10)
    if pb4 != (3, 3):
        raise AssertionError(
            f"pullback additive failed: pullback(5, 6, (1,2,1,3), 10, 10) = {pb4}, "
            f"want (3, 3)"
        )

    # Edge case: pixel at (2, 3) should map to (0, 0)
    pb5 = pullback(2, 3, (1, 2, 1, 3), 10, 10)
    if pb5 != (0, 0):
        raise AssertionError(
            f"pullback additive edge failed: pullback(2, 3, (1,2,1,3), 10, 10) = {pb5}, "
            f"want (0, 0)"
        )

    # OOB case: pixel at (1, 2) should be OOB (maps to (-1, -1))
    pb6 = pullback(1, 2, (1, 2, 1, 3), 10, 10)
    if pb6 is not None:
        raise AssertionError(
            f"pullback additive OOB failed: pullback(1, 2, (1,2,1,3), 10, 10) = {pb6}, "
            f"want None"
        )

    # Return receipt summarizing all verifications
    # Use the most complex case (mixed) as representative
    # verified_on = 5 indicates 5 test scenarios were verified
    return {
        "type": "mixed",
        "law": [3, 0, 1, 2],
        "verified_on": 5
    }


def init() -> None:
    """
    Run self-check if ARC_SELF_CHECK=1.

    Called by harness, not on import.
    """
    if os.environ.get("ARC_SELF_CHECK") == "1":
        receipt = _self_check_shape()
        receipts.log("shape", receipt)
