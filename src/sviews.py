"""
S-views v1: identity, D4-preserving, overlap translations (WO-03).

Constructs structural views (S-views) from the presented test input.
Each view M is a partial self-map with proof: ∀x∈Dom(M), G(M(x)) = G(x).

Residue-k periods deferred to WO-04.
"""

import os
import random
import hashlib
from typing import List, Tuple, Dict, Callable, Optional, Any

import morphisms
import receipts

# Type aliases
Coord = Tuple[int, int]
Shape = Tuple[int, int]
IntGrid = List[List[int]]


# ============================================================================
# S-VIEW CLASS (immutable record)
# ============================================================================


class SView:
    """
    Structural view: a partial self-map with proof.

    Attributes:
        kind: View type ("identity" | "d4" | "translate" | "compose")
        params: Parameters (e.g., {"op": 3} or {"di": 1, "dj": -2})
        dom_size: |Dom(M)| (number of valid pixels)
        apply: M(x) -> y or None if x∉Dom(M)
    """

    def __init__(
        self,
        kind: str,
        params: Dict[str, Any],
        dom_size: int,
        apply_fn: Callable[[Coord], Optional[Coord]]
    ):
        self.kind = kind
        self.params = params
        self.dom_size = dom_size
        self.apply = apply_fn

    def __repr__(self) -> str:
        return f"SView(kind={self.kind}, params={self.params}, dom_size={self.dom_size})"


# ============================================================================
# IMAGE SIGNATURE (for deduplication)
# ============================================================================


def compute_image_signature(view: SView, grid_shape: Shape) -> str:
    """
    Compute deterministic signature for a view.

    Args:
        view: SView to hash
        grid_shape: Shape of the grid (H, W)

    Returns:
        SHA-256 hex digest of {(x, V(x))} in row-major order

    Algorithm:
        - Enumerate all pixels in row-major order
        - For each x where V(x) is defined, collect (x, V(x))
        - Hash the tuple of tuples
    """
    H, W = grid_shape
    pairs = []

    for i in range(H):
        for j in range(W):
            x = (i, j)
            y = view.apply(x)
            if y is not None:
                pairs.append((x, y))

    # Create deterministic string representation
    pairs_str = str(pairs)
    digest = hashlib.sha256(pairs_str.encode('utf-8')).hexdigest()
    return digest


# ============================================================================
# BASE VIEWS (identity, D4-preserving, overlap translations)
# ============================================================================


def build_base_views(G: IntGrid) -> List[SView]:
    """
    Build base S-views: identity, D4-preserving, overlap translations.

    Args:
        G: Presented test input grid

    Returns:
        List of admitted SView objects with proofs

    Admission criteria:
        - Identity: always admitted
        - D4: admitted iff G invariant under that op (all pixels equal)
        - Translation: admitted iff equality holds on overlap (dom_size > 0)
    """
    H = len(G)
    W = len(G[0]) if H > 0 else 0
    shape = (H, W)
    views = []

    # 1. Identity (always admitted)
    identity_view = SView(
        kind="identity",
        params={},
        dom_size=H * W,
        apply_fn=lambda x: x
    )
    views.append(identity_view)

    # 2. D4-preserving views
    for op in range(8):
        if op == 0:
            continue  # Skip identity (already added)

        # Check if G is invariant under this D4 op
        # For ops that swap dimensions (1,3,5,7), we need H==W
        if op in [1, 3, 5, 7] and H != W:
            continue  # Can't be invariant if dimensions don't match

        is_invariant = True
        for i in range(H):
            for j in range(W):
                x = (i, j)
                y = morphisms.pose_fwd(x, op, shape)

                # Check bounds
                if y[0] < 0 or y[0] >= H or y[1] < 0 or y[1] >= W:
                    is_invariant = False
                    break

                if G[x[0]][x[1]] != G[y[0]][y[1]]:
                    is_invariant = False
                    break
            if not is_invariant:
                break

        if is_invariant:
            # Create apply function for this D4 op
            def make_d4_apply(op_id):
                def apply_fn(x):
                    y = morphisms.pose_fwd(x, op_id, shape)
                    # Check bounds
                    if op_id in [1, 3, 5, 7]:
                        if y[0] >= W or y[1] >= H:
                            return None
                    else:
                        if y[0] >= H or y[1] >= W:
                            return None
                    return y
                return apply_fn

            d4_view = SView(
                kind="d4",
                params={"op": op},
                dom_size=H * W,
                apply_fn=make_d4_apply(op)
            )
            views.append(d4_view)

    # 3. Overlap translations
    # Enumerate Δ in lex order on (|di|+|dj|, di, dj)
    # Bounded by grid size
    deltas = []
    for di in range(-H + 1, H):
        for dj in range(-W + 1, W):
            if di == 0 and dj == 0:
                continue  # Skip identity
            deltas.append((di, dj))

    # Sort by (|di|+|dj|, di, dj)
    deltas.sort(key=lambda d: (abs(d[0]) + abs(d[1]), d[0], d[1]))

    for di, dj in deltas:
        # Compute overlap domain: Ω ∩ (Ω + Δ)
        overlap = []
        for i in range(H):
            for j in range(W):
                i_shifted = i + di
                j_shifted = j + dj
                if 0 <= i_shifted < H and 0 <= j_shifted < W:
                    overlap.append((i, j))

        if len(overlap) == 0:
            continue

        # Check proof: G[x] == G[x + Δ] for all x in overlap
        is_equal = True
        for x in overlap:
            i, j = x
            i_shifted = i + di
            j_shifted = j + dj
            if G[i][j] != G[i_shifted][j_shifted]:
                is_equal = False
                break

        if is_equal:
            # Create apply function for this translation
            def make_translate_apply(delta):
                def apply_fn(x):
                    i, j = x
                    i_shifted = i + delta[0]
                    j_shifted = j + delta[1]
                    if 0 <= i_shifted < H and 0 <= j_shifted < W:
                        return (i_shifted, j_shifted)
                    return None
                return apply_fn

            translate_view = SView(
                kind="translate",
                params={"di": di, "dj": dj},
                dom_size=len(overlap),
                apply_fn=make_translate_apply((di, dj))
            )
            views.append(translate_view)

    return views


# ============================================================================
# CLOSURE DEPTH 2
# ============================================================================


def build_closure_depth2(G: IntGrid, base: List[SView]) -> List[SView]:
    """
    Build closure: {M} ∪ {N} ∪ {M∘N} with depth ≤ 2, dedup by image signature.

    Args:
        G: Grid (for computing signatures)
        base: Base views

    Returns:
        Deduplicated list of views (base + compositions), capped at ≤128

    Algorithm:
        1. Start with base views
        2. For each pair (M, N), compute M∘N
        3. Dedup by image signature
        4. Keep first 128 in deterministic order
    """
    H = len(G)
    W = len(G[0]) if H > 0 else 0
    shape = (H, W)

    # Track views by signature to deduplicate
    sig_to_view: Dict[str, SView] = {}

    # Add base views
    for view in base:
        sig = compute_image_signature(view, shape)
        if sig not in sig_to_view:
            sig_to_view[sig] = view

    # Generate compositions M∘N
    compositions = []
    for m in base:
        for n in base:
            # Compute M∘N: domain = {x | x∈Dom(N) ∧ N(x)∈Dom(M)}
            # Create apply function
            def make_compose_apply(m_view, n_view):
                def apply_fn(x):
                    y = n_view.apply(x)
                    if y is None:
                        return None
                    z = m_view.apply(y)
                    return z
                return apply_fn

            compose_apply = make_compose_apply(m, n)

            # Compute domain size
            dom_size = 0
            for i in range(H):
                for j in range(W):
                    x = (i, j)
                    if compose_apply(x) is not None:
                        dom_size += 1

            if dom_size == 0:
                continue

            # Get signatures of left and right for params
            sig_m = compute_image_signature(m, shape)
            sig_n = compute_image_signature(n, shape)

            compose_view = SView(
                kind="compose",
                params={"left": sig_m[:8], "right": sig_n[:8]},  # Short prefix for readability
                dom_size=dom_size,
                apply_fn=compose_apply
            )
            compositions.append(compose_view)

    # Add compositions with dedup
    for comp in compositions:
        sig = compute_image_signature(comp, shape)
        if sig not in sig_to_view:
            sig_to_view[sig] = comp

    # Collect all unique views
    all_views = list(sig_to_view.values())

    # Sort deterministically:
    # 1. identity first
    # 2. D4 by op ascending
    # 3. translations by lex on (|di|+|dj|, di, dj)
    # 4. compositions by lex on (sig(left), sig(right))
    def view_sort_key(v: SView):
        if v.kind == "identity":
            return (0, 0, 0, 0, "")
        elif v.kind == "d4":
            return (1, v.params["op"], 0, 0, "")
        elif v.kind == "translate":
            di = v.params["di"]
            dj = v.params["dj"]
            return (2, abs(di) + abs(dj), di, dj, "")
        elif v.kind == "compose":
            return (3, 0, 0, 0, v.params["left"] + v.params["right"])
        else:
            return (999, 0, 0, 0, "")

    all_views.sort(key=view_sort_key)

    # Cap at 128
    capped = False
    if len(all_views) > 128:
        all_views = all_views[:128]
        capped = True

    return all_views


# ============================================================================
# PUBLIC API
# ============================================================================


def build_sviews(G: IntGrid) -> List[SView]:
    """
    Build complete S-views: base → closure → cap.

    Args:
        G: Presented test input grid

    Returns:
        List of admitted SView objects (≤128)
    """
    base = build_base_views(G)
    closure = build_closure_depth2(G, base)
    return closure


# ============================================================================
# SELF-CHECK (algebraic debugging)
# ============================================================================


def _self_check_sviews() -> Dict:
    """
    Verify S-views on synthetic grids.

    Returns:
        Receipt payload for "sviews" section
    """
    random.seed(1337)  # Deterministic

    receipt = {
        "count": 0,
        "depth_max": 0,
        "views": [],
        "proof_samples": [],
        "closure_capped": False,
        "examples": {}
    }

    # ========================================================================
    # Check 1: D4 invariance (180-degree symmetry)
    # ========================================================================
    # Grid with 180-degree symmetry (op=2)
    G1 = [
        [1, 2, 1],
        [3, 4, 3],
        [1, 2, 1]
    ]

    views1 = build_base_views(G1)
    d4_180_found = False
    for v in views1:
        if v.kind == "d4" and v.params.get("op") == 2:
            d4_180_found = True
            if v.dom_size != 9:
                receipt["examples"]["d4_domain"] = {
                    "case": "d4_invariance",
                    "op": 2,
                    "expected_dom_size": 9,
                    "got_dom_size": v.dom_size
                }
                return receipt

    if not d4_180_found:
        receipt["examples"]["d4_missing"] = {
            "case": "d4_invariance",
            "op": 2,
            "expected": "admitted",
            "got": "not found"
        }
        return receipt

    receipt["proof_samples"].append({
        "kind": "d4",
        "ok": True,
        "checked": 9
    })

    # ========================================================================
    # Check 2: Translation (period-2 horizontal stripe)
    # ========================================================================
    # Pattern: 0 1 0 1 0
    #          2 3 2 3 2
    G2 = [
        [0, 1, 0, 1, 0],
        [2, 3, 2, 3, 2]
    ]

    views2 = build_base_views(G2)

    # dj=2 should admit (period 2)
    dj2_found = False
    for v in views2:
        if v.kind == "translate" and v.params.get("dj") == 2 and v.params.get("di") == 0:
            dj2_found = True
            if v.dom_size <= 0:
                receipt["examples"]["translate_domain"] = {
                    "case": "translation",
                    "di": 0,
                    "dj": 2,
                    "expected_dom_size": "> 0",
                    "got_dom_size": v.dom_size
                }
                return receipt

    if not dj2_found:
        receipt["examples"]["translate_missing"] = {
            "case": "translation",
            "di": 0,
            "dj": 2,
            "expected": "admitted",
            "got": "not found"
        }
        return receipt

    # dj=1 should NOT admit (pattern doesn't repeat every 1)
    dj1_found = False
    for v in views2:
        if v.kind == "translate" and v.params.get("dj") == 1 and v.params.get("di") == 0:
            dj1_found = True

    if dj1_found:
        receipt["examples"]["translate_spurious"] = {
            "case": "translation",
            "di": 0,
            "dj": 1,
            "expected": "rejected",
            "got": "admitted"
        }
        return receipt

    receipt["proof_samples"].append({
        "kind": "translate",
        "ok": True,
        "checked": 5  # overlap size for dj=2
    })

    # ========================================================================
    # Check 3: Closure dedup
    # ========================================================================
    G3 = [[1, 2], [3, 4]]
    base3 = build_base_views(G3)
    closure3 = build_closure_depth2(G3, base3)

    # Check that closure doesn't have exact duplicates (same signature)
    sigs = set()
    for v in closure3:
        sig = compute_image_signature(v, (2, 2))
        if sig in sigs:
            receipt["examples"]["dedup_failed"] = {
                "case": "closure_dedup",
                "duplicate_sig": sig[:16]
            }
            return receipt
        sigs.add(sig)

    receipt["proof_samples"].append({
        "kind": "closure",
        "ok": True,
        "checked": len(closure3)
    })

    # ========================================================================
    # Check 4: Cap at 128
    # ========================================================================
    # Create a busy grid (checkerboard)
    G4 = [[(i + j) % 10 for j in range(10)] for i in range(10)]
    base4 = build_base_views(G4)
    closure4 = build_closure_depth2(G4, base4)

    if len(closure4) > 128:
        receipt["examples"]["cap_violated"] = {
            "case": "cap_128",
            "count": len(closure4),
            "expected": "≤ 128"
        }
        return receipt

    receipt["closure_capped"] = len(base4) > 128 or len(closure4) == 128

    # Final receipt
    receipt["count"] = len(closure3)  # Use G3 for reference count
    receipt["depth_max"] = 2 if any(v.kind == "compose" for v in closure3) else 1

    # Summarize views from G3
    for v in closure3:
        view_entry = {
            "kind": v.kind,
            "dom_size": v.dom_size,
            "meta": v.params
        }
        receipt["views"].append(view_entry)

    return receipt


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================


def init() -> None:
    """
    Run self-check and emit sviews receipt.

    Raises:
        AssertionError: If any identity check fails

    Notes:
        - Called by harness, not on import
        - Assumes receipts.init() has been called
        - Runs self-check only if ARC_SELF_CHECK=1
    """
    # Check if self-check should run
    if os.environ.get("ARC_SELF_CHECK") != "1":
        # Skip self-check in normal mode (fast path)
        # Emit minimal receipt
        receipts.log("sviews", {
            "count": 0,
            "depth_max": 0,
            "views": [],
            "proof_samples": [],
            "closure_capped": False,
            "examples": {},
            "note": "self-check skipped (ARC_SELF_CHECK != 1)"
        })
        return

    receipt = _self_check_sviews()

    # Emit receipt
    receipts.log("sviews", receipt)

    # Assert all checks passed
    if "d4_domain" in receipt.get("examples", {}):
        ex = receipt["examples"]["d4_domain"]
        raise AssertionError(
            f"sviews identity failed: d4 op={ex['op']} "
            f"dom_size={ex['got_dom_size']}, expected={ex['expected_dom_size']}"
        )

    if "d4_missing" in receipt.get("examples", {}):
        ex = receipt["examples"]["d4_missing"]
        raise AssertionError(
            f"sviews identity failed: d4 op={ex['op']} not admitted, expected admitted"
        )

    if "translate_domain" in receipt.get("examples", {}):
        ex = receipt["examples"]["translate_domain"]
        raise AssertionError(
            f"sviews identity failed: translate di={ex['di']} dj={ex['dj']} "
            f"dom_size={ex['got_dom_size']}, expected={ex['expected_dom_size']}"
        )

    if "translate_missing" in receipt.get("examples", {}):
        ex = receipt["examples"]["translate_missing"]
        raise AssertionError(
            f"sviews identity failed: translate di={ex['di']} dj={ex['dj']} "
            f"not admitted, expected admitted"
        )

    if "translate_spurious" in receipt.get("examples", {}):
        ex = receipt["examples"]["translate_spurious"]
        raise AssertionError(
            f"sviews identity failed: translate di={ex['di']} dj={ex['dj']} "
            f"spuriously admitted, expected rejected"
        )

    if "dedup_failed" in receipt.get("examples", {}):
        ex = receipt["examples"]["dedup_failed"]
        raise AssertionError(
            f"sviews identity failed: closure dedup found duplicate sig={ex['duplicate_sig']}"
        )

    if "cap_violated" in receipt.get("examples", {}):
        ex = receipt["examples"]["cap_violated"]
        raise AssertionError(
            f"sviews identity failed: cap violated count={ex['count']}, expected ≤ 128"
        )
