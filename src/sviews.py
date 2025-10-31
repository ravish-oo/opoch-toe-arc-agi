"""
S-views: identity, D4-preserving, residue-k periods, overlap translations (WO-03 + WO-04).

Constructs structural views (S-views) from the presented test input.
Each view M is a partial self-map with proof: ∀x∈Dom(M), G(M(x)) = G(x).
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
# PERIOD DETECTION (WO-04)
# ============================================================================


def minimal_row_period(G: IntGrid) -> Tuple[int, List[int]]:
    """
    Compute minimal period per row and their gcd.

    Args:
        G: Grid to analyze

    Returns:
        (gcd_row, per_row_periods)
        where gcd_row is gcd of all row periods,
        and per_row_periods[i] is minimal period of row i

    Algorithm:
        For each row i:
            - Check divisors of W in ascending order
            - Find smallest p where G[i,j] = G[i,(j+p)%W] for all j
        Return gcd of all row periods
    """
    H = len(G)
    W = len(G[0]) if H > 0 else 0

    if W == 0:
        return (1, [])

    # Compute divisors of W
    divisors = []
    for p in range(1, W + 1):
        if W % p == 0:
            divisors.append(p)

    per_row_periods = []

    for i in range(H):
        row_period = W  # Default: full width

        for p in divisors:
            # Check if period p works for this row
            is_period = True
            for j in range(W):
                j_shifted = (j + p) % W
                if G[i][j] != G[i][j_shifted]:
                    is_period = False
                    break

            if is_period:
                row_period = p
                break  # Found minimal period

        per_row_periods.append(row_period)

    # Compute gcd of all row periods
    import math
    gcd_row = per_row_periods[0] if per_row_periods else 1
    for p in per_row_periods[1:]:
        gcd_row = math.gcd(gcd_row, p)

    return (gcd_row, per_row_periods)


def minimal_col_period(G: IntGrid) -> Tuple[int, List[int]]:
    """
    Compute minimal period per column and their gcd.

    Args:
        G: Grid to analyze

    Returns:
        (gcd_col, per_col_periods)
        where gcd_col is gcd of all column periods,
        and per_col_periods[j] is minimal period of column j

    Algorithm:
        For each column j:
            - Check divisors of H in ascending order
            - Find smallest p where G[i,j] = G[(i+p)%H,j] for all i
        Return gcd of all column periods
    """
    H = len(G)
    W = len(G[0]) if H > 0 else 0

    if H == 0:
        return (1, [])

    # Compute divisors of H
    divisors = []
    for p in range(1, H + 1):
        if H % p == 0:
            divisors.append(p)

    per_col_periods = []

    for j in range(W):
        col_period = H  # Default: full height

        for p in divisors:
            # Check if period p works for this column
            is_period = True
            for i in range(H):
                i_shifted = (i + p) % H
                if G[i][j] != G[i_shifted][j]:
                    is_period = False
                    break

            if is_period:
                col_period = p
                break  # Found minimal period

        per_col_periods.append(col_period)

    # Compute gcd of all column periods
    import math
    gcd_col = per_col_periods[0] if per_col_periods else 1
    for p in per_col_periods[1:]:
        gcd_col = math.gcd(gcd_col, p)

    return (gcd_col, per_col_periods)


# ============================================================================
# S-VIEW CLASS (immutable record)
# ============================================================================


class SView:
    """
    Structural view: a partial self-map with proof.

    Attributes:
        kind: View type ("identity" | "d4" | "residue" | "translate" | "compose")
        params: Parameters (e.g., {"op": 3} or {"axis": "row", "p": 3} or {"di": 1, "dj": -2})
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
# 4-CONNECTED COMPONENT EXTRACTION (WO-Truth-EQUALITIES-STRICT)
# ============================================================================


def _extract_4connected_components(pixel_set: List[Coord], H: int, W: int) -> List[List[Coord]]:
    """
    Extract 4-connected components from a set of pixels using deterministic BFS.

    Args:
        pixel_set: List of (i, j) coordinates
        H, W: Grid dimensions

    Returns:
        List of components, each a list of coordinates, sorted by anchor (min row-major pixel)
    """
    if not pixel_set:
        return []

    from collections import deque

    # Convert to set for O(1) lookup
    remaining = set(pixel_set)
    components = []

    # Fixed neighbor order: up, down, left, right
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Process in row-major order
    sorted_pixels = sorted(pixel_set, key=lambda p: (p[0], p[1]))

    for start_pixel in sorted_pixels:
        if start_pixel not in remaining:
            continue

        # BFS from start_pixel
        component = []
        queue = deque([start_pixel])
        remaining.remove(start_pixel)

        while queue:
            i, j = queue.popleft()
            component.append((i, j))

            # Explore 4-neighbors in fixed order
            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                neighbor = (ni, nj)
                if (0 <= ni < H and 0 <= nj < W and neighbor in remaining):
                    remaining.remove(neighbor)
                    queue.append(neighbor)

        # Sort component row-major for determinism
        component.sort(key=lambda p: (p[0], p[1]))
        components.append(component)

    # Sort components by anchor (first pixel row-major)
    components.sort(key=lambda comp: (comp[0][0], comp[0][1]))

    return components


# ============================================================================
# BASE VIEWS (identity, D4-preserving, overlap translations)
# ============================================================================


def build_base_views(G: IntGrid) -> List[SView]:
    """
    Build base S-views: identity, D4-preserving, residue-k periods, overlap translations.

    Args:
        G: Presented test input grid

    Returns:
        List of admitted SView objects with proofs

    Admission criteria:
        - Identity: always admitted
        - D4: admitted iff G invariant under that op (all pixels equal)
        - Residue: admitted iff gcd_row > 1 (rows) or gcd_col > 1 (cols)
        - Translation: admitted iff equality holds on overlap (dom_size > 0)

    Order:
        identity → D4 ops (0-7) → residue row → residue col → translations (lex)
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

    # 3. Residue-k periods (WO-Truth-EQUALITIES-STRICT)
    # Admit per-residue-class partial S-views (spec: 02-locks-and-minispecs.md:176-177)
    # For period p, admit one S-view per residue class r ∈ {0, ..., p-1}
    # Domain = {positions where j≡r mod p AND G[i,j]=G[i,j+p]} for rows

    # Row-wise residue classes
    gcd_row, _ = minimal_row_period(G)
    if gcd_row > 1 and gcd_row < W:
        for r in range(gcd_row):
            # Build equality domain for residue class r
            domain = []
            for i in range(H):
                for j in range(W):
                    # Check: j ≡ r mod p AND j+p in bounds AND G[i,j] = G[i,j+p]
                    if j % gcd_row == r:
                        j_shifted = j + gcd_row
                        if j_shifted < W and G[i][j] == G[i][j_shifted]:
                            domain.append((i, j))

            if len(domain) < 2:
                continue  # Need at least 2 pixels to form a view

            # Create apply function for this residue class
            def make_row_residue_apply(p, r_val, dom_set, width):
                def apply_fn(x):
                    # Only defined on domain
                    if x not in dom_set:
                        return None
                    i, j = x
                    j_shifted = j + p
                    if j_shifted < width:
                        return (i, j_shifted)
                    return None
                return apply_fn

            dom_set = set(domain)
            residue_row_view = SView(
                kind="residue",
                params={"axis": "row", "p": gcd_row, "r": r},
                dom_size=len(domain),
                apply_fn=make_row_residue_apply(gcd_row, r, dom_set, W)
            )
            views.append(residue_row_view)

    # Column-wise residue classes
    gcd_col, _ = minimal_col_period(G)
    if gcd_col > 1 and gcd_col < H:
        for r in range(gcd_col):
            # Build equality domain for residue class r
            domain = []
            for i in range(H):
                for j in range(W):
                    # Check: i ≡ r mod p AND i+p in bounds AND G[i,j] = G[i+p,j]
                    if i % gcd_col == r:
                        i_shifted = i + gcd_col
                        if i_shifted < H and G[i][j] == G[i_shifted][j]:
                            domain.append((i, j))

            if len(domain) < 2:
                continue  # Need at least 2 pixels to form a view

            # Create apply function for this residue class
            def make_col_residue_apply(p, r_val, dom_set, height):
                def apply_fn(x):
                    # Only defined on domain
                    if x not in dom_set:
                        return None
                    i, j = x
                    i_shifted = i + p
                    if i_shifted < height:
                        return (i_shifted, j)
                    return None
                return apply_fn

            dom_set = set(domain)
            residue_col_view = SView(
                kind="residue",
                params={"axis": "col", "p": gcd_col, "r": r},
                dom_size=len(domain),
                apply_fn=make_col_residue_apply(gcd_col, r, dom_set, H)
            )
            views.append(residue_col_view)

    # 4. Overlap translations (WO-Truth-EQUALITIES-STRICT)
    # Admit local equality domains as partial S-views (spec: 00-math-spec.md:47-50)
    # Build E_Δ = {x : G(x) = G(x+Δ)}, extract 4-connected components, admit each
    # Cap: Stop when budget (128 - current views) is exhausted

    # Enumerate Δ in lex order on (|di|+|dj|, di, dj), limited to small distances
    max_delta_dist = min(max(H, W), 20)  # Cap delta enumeration for performance
    deltas = []
    for di in range(-max_delta_dist, max_delta_dist + 1):
        for dj in range(-max_delta_dist, max_delta_dist + 1):
            if di == 0 and dj == 0:
                continue  # Skip identity
            # Only consider deltas that create non-empty overlap
            if abs(di) < H and abs(dj) < W:
                deltas.append((di, dj))

    # Sort by (|di|+|dj|, di, dj)
    deltas.sort(key=lambda d: (abs(d[0]) + abs(d[1]), d[0], d[1]))

    # Process deltas with early termination at cap (128 total S-views)
    max_views = 128
    for di, dj in deltas:
        # Early termination: stop if we've hit the cap
        if len(views) >= max_views:
            break
        # Build E_Δ = equality mask where G(x) = G(x+Δ)
        equality_mask = []
        for i in range(H):
            for j in range(W):
                i_shifted = i + di
                j_shifted = j + dj
                # Check both in bounds and equal
                if (0 <= i_shifted < H and 0 <= j_shifted < W and
                    G[i][j] == G[i_shifted][j_shifted]):
                    equality_mask.append((i, j))

        if len(equality_mask) < 2:
            continue  # Need at least 2 pixels to form a view

        # Extract 4-connected components of E_Δ (deterministic BFS)
        components = _extract_4connected_components(equality_mask, H, W)

        # Admit one S-view per component (if size ≥ 2)
        for comp_idx, component in enumerate(components):
            if len(component) < 2:
                continue

            # Compute component anchor for deterministic naming
            anchor = min(component, key=lambda p: (p[0], p[1]))

            # Create apply function for this translation
            def make_translate_apply(delta, comp_set):
                def apply_fn(x):
                    # Only defined on component domain
                    if x not in comp_set:
                        return None
                    i, j = x
                    i_shifted = i + delta[0]
                    j_shifted = j + delta[1]
                    if 0 <= i_shifted < H and 0 <= j_shifted < W:
                        return (i_shifted, j_shifted)
                    return None
                return apply_fn

            comp_set = set(component)
            translate_view = SView(
                kind="translate",
                params={"di": di, "dj": dj, "comp_anchor": anchor},
                dom_size=len(component),
                apply_fn=make_translate_apply((di, dj), comp_set)
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
    # 3. residue by (axis: row=0, col=1) then period p
    # 4. translations by lex on (|di|+|dj|, di, dj)
    # 5. compositions by lex on (sig(left), sig(right))
    def view_sort_key(v: SView):
        if v.kind == "identity":
            return (0, 0, 0, 0, "")
        elif v.kind == "d4":
            return (1, v.params["op"], 0, 0, "")
        elif v.kind == "residue":
            axis_order = 0 if v.params["axis"] == "row" else 1
            return (2, axis_order, v.params["p"], 0, "")
        elif v.kind == "translate":
            di = v.params["di"]
            dj = v.params["dj"]
            return (3, abs(di) + abs(dj), di, dj, "")
        elif v.kind == "compose":
            return (4, 0, 0, 0, v.params["left"] + v.params["right"])
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


def build_sviews_order_hash(sviews: List[SView], shape: Tuple[int, int]) -> str:
    """
    WO-ND3 Part A: Compute order_hash for sviews list.

    Args:
        sviews: List of SView objects
        shape: (H, W) for signature computation

    Returns:
        SHA256 hash of stable keys (kind, params, dom_size, sig)
    """
    import hashlib

    order_keys = []
    for v in sviews:
        # Serialize params deterministically
        params_str = str(sorted(v.params.items()))
        sig = compute_image_signature(v, shape)
        order_keys.append((v.kind, params_str, v.dom_size, sig))

    order_hash = hashlib.sha256(str(order_keys).encode('utf-8')).hexdigest()
    return order_hash


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

    # ========================================================================
    # Check 5: Residue-k period-3 rows (WO-04)
    # ========================================================================
    # Grid with period-3 rows: abc abc abc (W=9)
    G5 = [
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [3, 4, 5, 3, 4, 5, 3, 4, 5]
    ]

    views5 = build_base_views(G5)

    # Should admit residue row p=3
    residue_row_p3_found = False
    for v in views5:
        if v.kind == "residue" and v.params.get("axis") == "row" and v.params.get("p") == 3:
            residue_row_p3_found = True
            if v.dom_size != 18:  # 2 rows × 9 cols
                receipt["examples"]["residue_row_domain"] = {
                    "case": "residue_period3_rows",
                    "axis": "row",
                    "p": 3,
                    "expected_dom_size": 18,
                    "got_dom_size": v.dom_size
                }
                return receipt

    if not residue_row_p3_found:
        receipt["examples"]["residue_row_missing"] = {
            "case": "residue_period3_rows",
            "axis": "row",
            "p": 3,
            "expected": "admitted",
            "got": "not found"
        }
        return receipt

    # Should NOT admit residue col (no column periodicity)
    residue_col_found = False
    for v in views5:
        if v.kind == "residue" and v.params.get("axis") == "col":
            residue_col_found = True

    if residue_col_found:
        receipt["examples"]["residue_col_spurious"] = {
            "case": "residue_period3_rows",
            "axis": "col",
            "expected": "rejected",
            "got": "admitted"
        }
        return receipt

    receipt["proof_samples"].append({
        "kind": "residue_row",
        "ok": True,
        "checked": 18
    })

    # ========================================================================
    # Check 6: Residue-k different minimal periods (WO-04)
    # ========================================================================
    # Row 0: minimal period 2 (ab ab ab ab)
    # Row 1: minimal period 4 (abab abab) which ALSO has period 2
    # For row 1 to have minimal period 4, it needs to NOT have period 1 or 2
    # Actually: row 1 with period 4 means [a,b,c,d,a,b,c,d]
    # But if we want BOTH rows to satisfy period 2, we need:
    # Row 0: [0,1,0,1,0,1,0,1] - satisfies period 2
    # Row 1: [2,2,2,2,2,2,2,2] - satisfies period 1 (thus also 2, 4, 8)
    # This tests: minimal periods are [2, 1], gcd(2,1)=1, no residue admitted
    #
    # Better test: Both rows period 2, to verify p=2 admission
    G6 = [
        [0, 1, 0, 1, 0, 1, 0, 1],
        [2, 3, 2, 3, 2, 3, 2, 3]
    ]

    views6 = build_base_views(G6)

    # Should admit residue row p=2
    residue_row_p2_found = False
    for v in views6:
        if v.kind == "residue" and v.params.get("axis") == "row" and v.params.get("p") == 2:
            residue_row_p2_found = True
            if v.dom_size != 16:  # 2 rows × 8 cols
                receipt["examples"]["residue_gcd_domain"] = {
                    "case": "residue_period2_rows",
                    "axis": "row",
                    "p": 2,
                    "expected_dom_size": 16,
                    "got_dom_size": v.dom_size
                }
                return receipt

    if not residue_row_p2_found:
        receipt["examples"]["residue_gcd_missing"] = {
            "case": "residue_period2_rows",
            "axis": "row",
            "p": 2,
            "gcd_of": [2, 2],
            "expected": "admitted",
            "got": "not found"
        }
        return receipt

    # Should NOT admit residue col
    residue_col_found = False
    for v in views6:
        if v.kind == "residue" and v.params.get("axis") == "col":
            residue_col_found = True

    if residue_col_found:
        receipt["examples"]["residue_col_spurious_mixed"] = {
            "case": "residue_period2_rows",
            "axis": "col",
            "expected": "rejected",
            "got": "admitted"
        }
        return receipt

    receipt["proof_samples"].append({
        "kind": "residue_gcd",
        "ok": True,
        "checked": 16
    })

    # ========================================================================
    # Check 7: Residue-k mixed 2/3 rows (WO-04)
    # ========================================================================
    # Row 0: period 2 (ab ab ab)
    # Row 1: period 3 (abc abc)
    # gcd(2, 3) = 1 → no residue
    G7 = [
        [0, 1, 0, 1, 0, 1],
        [2, 3, 4, 2, 3, 4]
    ]

    views7 = build_base_views(G7)

    # Should NOT admit any residue views (gcd=1)
    residue_found = False
    for v in views7:
        if v.kind == "residue":
            residue_found = True

    if residue_found:
        receipt["examples"]["residue_gcd1_spurious"] = {
            "case": "residue_mixed_2_3",
            "gcd": 1,
            "expected": "no residue",
            "got": "residue admitted"
        }
        return receipt

    receipt["proof_samples"].append({
        "kind": "residue_gcd1",
        "ok": True,
        "checked": 0  # No residue should be admitted
    })

    # ========================================================================
    # Check 8: Residue distinct from translation (WO-04)
    # ========================================================================
    # Grid with period-2 rows
    # Residue wraps (full domain), translation doesn't (partial domain)
    # They should have DIFFERENT images and BOTH be admitted
    G8 = [
        [0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1]
    ]

    base8 = build_base_views(G8)
    closure8 = build_closure_depth2(G8, base8)

    # Both residue row p=2 and translation dj=2 di=0 should be admitted
    # They have different images (residue wraps, translation doesn't)
    residue_p2_found = False
    translate_dj2_found = False
    for v in closure8:
        if v.kind == "residue" and v.params.get("axis") == "row" and v.params.get("p") == 2:
            residue_p2_found = True
        if v.kind == "translate" and v.params.get("dj") == 2 and v.params.get("di") == 0:
            translate_dj2_found = True

    # Both should be admitted (different images)
    if not (residue_p2_found and translate_dj2_found):
        receipt["examples"]["residue_translate_missing"] = {
            "case": "residue_vs_translation",
            "residue_found": residue_p2_found,
            "translation_found": translate_dj2_found,
            "expected": "both admitted"
        }
        return receipt

    receipt["proof_samples"].append({
        "kind": "residue_vs_translate",
        "ok": True,
        "checked": len(closure8)
    })

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
    Run self-check (no receipt logging - caller logs after build_sviews).

    Raises:
        AssertionError: If any identity check fails

    Notes:
        - Called by harness, not on import
        - Assumes receipts.init() has been called
        - Runs self-check only if ARC_SELF_CHECK=1
        - Receipt logging moved to caller after build_sviews()
    """
    # Check if self-check should run
    if os.environ.get("ARC_SELF_CHECK") != "1":
        # Skip self-check in normal mode (fast path)
        return

    receipt = _self_check_sviews()

    # Emit receipt only for self-check
    receipts.log("sviews_selfcheck", receipt)

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

    # WO-04 residue checks
    if "residue_row_domain" in receipt.get("examples", {}):
        ex = receipt["examples"]["residue_row_domain"]
        raise AssertionError(
            f"sviews identity failed: residue row p={ex['p']} "
            f"dom_size={ex['got_dom_size']}, expected={ex['expected_dom_size']}"
        )

    if "residue_row_missing" in receipt.get("examples", {}):
        ex = receipt["examples"]["residue_row_missing"]
        raise AssertionError(
            f"sviews identity failed: residue row p={ex['p']} not admitted, expected admitted"
        )

    if "residue_col_spurious" in receipt.get("examples", {}):
        ex = receipt["examples"]["residue_col_spurious"]
        raise AssertionError(
            f"sviews identity failed: residue col spuriously admitted in {ex['case']}, expected rejected"
        )

    if "residue_gcd_domain" in receipt.get("examples", {}):
        ex = receipt["examples"]["residue_gcd_domain"]
        raise AssertionError(
            f"sviews identity failed: residue row p={ex['p']} (gcd) "
            f"dom_size={ex['got_dom_size']}, expected={ex['expected_dom_size']}"
        )

    if "residue_gcd_missing" in receipt.get("examples", {}):
        ex = receipt["examples"]["residue_gcd_missing"]
        raise AssertionError(
            f"sviews identity failed: residue row p={ex['p']} (gcd of {ex['gcd_of']}) not admitted, expected admitted"
        )

    if "residue_col_spurious_mixed" in receipt.get("examples", {}):
        ex = receipt["examples"]["residue_col_spurious_mixed"]
        raise AssertionError(
            f"sviews identity failed: residue col spuriously admitted in {ex['case']}, expected rejected"
        )

    if "residue_gcd1_spurious" in receipt.get("examples", {}):
        ex = receipt["examples"]["residue_gcd1_spurious"]
        raise AssertionError(
            f"sviews identity failed: residue admitted when gcd={ex['gcd']}, expected no residue"
        )

    if "residue_translate_missing" in receipt.get("examples", {}):
        ex = receipt["examples"]["residue_translate_missing"]
        raise AssertionError(
            f"sviews identity failed: residue_found={ex['residue_found']}, "
            f"translation_found={ex['translation_found']}, expected={ex['expected']}"
        )
