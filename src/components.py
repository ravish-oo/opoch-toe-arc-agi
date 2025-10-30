"""
Components: 4-connected components per color (WO-05).

Extracts color-wise 4-connected components from presented test input.
Provides:
  1. Must-link S-views (fold-to-anchor for union-find)
  2. PT predicates (component masks as input-only separators)

All operations deterministic; debugging = algebra.
"""

import os
from typing import List, Tuple, Dict, Optional
from collections import deque

import receipts

# Type aliases
Coord = Tuple[int, int]
IntGrid = List[List[int]]


# ============================================================================
# COMPONENT CLASS
# ============================================================================


class Component:
    """
    4-connected component of a single color.

    Attributes:
        color: Color value (0..9)
        comp_id: Index in deterministic order for this color (0-based)
        anchor: Top-left pixel (min row, then min col)
        bbox: Bounding box (top, left, bottom, right) inclusive
        mask: Pixel coords in row-major sorted order
    """

    def __init__(
        self,
        color: int,
        comp_id: int,
        anchor: Coord,
        bbox: Tuple[int, int, int, int],
        mask: List[Coord]
    ):
        self.color = color
        self.comp_id = comp_id
        self.anchor = anchor
        self.bbox = bbox
        self.mask = mask
        # Build set for fast contains check
        self._mask_set = set(mask)

    def contains(self, p: Coord) -> bool:
        """Check if pixel p is in this component."""
        return p in self._mask_set

    def __repr__(self) -> str:
        return (f"Component(color={self.color}, id={self.comp_id}, "
                f"anchor={self.anchor}, size={len(self.mask)})")


# ============================================================================
# COMPONENT EXTRACTION (4-connectivity, deterministic)
# ============================================================================


def build_components(G: IntGrid) -> List[Component]:
    """
    Build 4-connected components per color on presented test grid.

    Algorithm:
        - For each color (0..9 ascending)
        - For each unvisited pixel of that color (row-major scan)
        - BFS with 4-adjacency (fixed neighbor order: up, down, left, right)
        - Compute anchor (min row, then min col) and bbox
        - Sort components by (color, anchor_row, anchor_col)

    Args:
        G: Presented test input grid

    Returns:
        List of Component objects in deterministic order
    """
    H = len(G)
    W = len(G[0]) if H > 0 else 0

    if H == 0 or W == 0:
        return []

    visited = [[False] * W for _ in range(H)]
    components = []

    # Fixed neighbor order: up, down, left, right
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Process colors in ascending order
    for color in range(10):
        color_components = []

        # Row-major scan for this color
        for start_i in range(H):
            for start_j in range(W):
                if visited[start_i][start_j] or G[start_i][start_j] != color:
                    continue

                # BFS to extract component
                mask = []
                queue = deque([(start_i, start_j)])
                visited[start_i][start_j] = True

                while queue:
                    i, j = queue.popleft()
                    mask.append((i, j))

                    # Explore 4-neighbors in fixed order
                    for di, dj in neighbors:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < H and 0 <= nj < W and
                            not visited[ni][nj] and G[ni][nj] == color):
                            visited[ni][nj] = True
                            queue.append((ni, nj))

                # Sort mask row-major
                mask.sort(key=lambda p: (p[0], p[1]))

                # Compute anchor (min row, then min col)
                anchor = min(mask, key=lambda p: (p[0], p[1]))

                # Compute bbox
                top = min(p[0] for p in mask)
                left = min(p[1] for p in mask)
                bottom = max(p[0] for p in mask)
                right = max(p[1] for p in mask)
                bbox = (top, left, bottom, right)

                # comp_id will be assigned after sorting
                color_components.append((anchor, bbox, mask))

        # Sort components for this color by anchor
        color_components.sort(key=lambda x: (x[0][0], x[0][1]))

        # Assign comp_id and create Component objects
        for comp_id, (anchor, bbox, mask) in enumerate(color_components):
            comp = Component(color, comp_id, anchor, bbox, mask)
            components.append(comp)

    return components


# ============================================================================
# MUST-LINK S-VIEWS (fold-to-anchor)
# ============================================================================


def component_anchor_views(components: List[Component]) -> List[Dict]:
    """
    Produce must-link S-views: fold every pixel in component to anchor.

    Each view maps all pixels in the component to the anchor.
    Proof obligation: G(M(x)) = G(x) for all x in component
    (holds because all pixels have same color).

    Args:
        components: List of Component objects

    Returns:
        List of view descriptors with "apply" callable
    """
    views = []

    for comp in components:
        # Create apply function for fold-to-anchor
        def make_apply(c):
            def apply_fn(p):
                if c.contains(p):
                    return c.anchor
                return None
            return apply_fn

        view = {
            "kind": "component_anchor",
            "comp_key": f"{comp.color}:{comp.comp_id}",
            "anchor": comp.anchor,
            "dom_size": len(comp.mask),
            "apply": make_apply(comp)
        }
        views.append(view)

    return views


# ============================================================================
# PT PREDICATES (component masks as input-only separators)
# ============================================================================


def component_predicates(components: List[Component]) -> List[Dict]:
    """
    Export component masks as PT predicates.

    These are input-only "membership_in_Sview_image" predicates.
    PT can split classes by component membership.

    Args:
        components: List of Component objects

    Returns:
        List of predicate descriptors with masks
    """
    predicates = []

    for comp in components:
        pred = {
            "kind": "sview_image",
            "key": f"component:{comp.color}:{comp.comp_id}",
            "mask": comp.mask  # row-major sorted
        }
        predicates.append(pred)

    return predicates


# ============================================================================
# RECONSTRUCTION (algebraic verification)
# ============================================================================


def reconstruct_from_components(
    components: List[Component],
    H: int,
    W: int
) -> IntGrid:
    """
    Reconstruct grid by painting each component's mask with its color.

    Used for algebraic self-check: result must equal original G.

    Args:
        components: List of Component objects
        H: Grid height
        W: Grid width

    Returns:
        Reconstructed grid
    """
    G = [[0] * W for _ in range(H)]

    for comp in components:
        for i, j in comp.mask:
            G[i][j] = comp.color

    return G


# ============================================================================
# SELF-CHECK (algebraic debugging)
# ============================================================================


def _self_check_components() -> Dict:
    """
    Verify components on synthetic grids.

    Returns:
        Receipt payload for "components" section
    """
    receipt = {
        "count_total": 0,
        "by_color": {},
        "largest": {},
        "anchors_first5": [],
        "proof_reconstruct_ok": True,
        "examples": {}
    }

    # ========================================================================
    # Check 1: Reconstruction equals original
    # ========================================================================
    # 3 colors, multiple blobs
    G1 = [
        [1, 1, 0, 2, 2],
        [1, 0, 0, 2, 0],
        [0, 0, 3, 3, 3],
        [0, 3, 3, 0, 0]
    ]

    comps1 = build_components(G1)
    G1_recon = reconstruct_from_components(comps1, 4, 5)

    # Byte-equal check
    for i in range(4):
        for j in range(5):
            if G1[i][j] != G1_recon[i][j]:
                receipt["proof_reconstruct_ok"] = False
                receipt["examples"]["case"] = "reconstruct"
                receipt["examples"]["detail"] = {
                    "p": [i, j],
                    "want": G1[i][j],
                    "got": G1_recon[i][j]
                }
                return receipt

    # ========================================================================
    # Check 2: 4-adjacency only (diagonal not connected)
    # ========================================================================
    # Two same-color pixels touching diagonally
    G2 = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]

    comps2 = build_components(G2)
    color1_comps = [c for c in comps2 if c.color == 1]

    # Expect 2 components (diagonal doesn't merge)
    if len(color1_comps) != 2:
        receipt["proof_reconstruct_ok"] = False
        receipt["examples"]["case"] = "connectivity"
        receipt["examples"]["detail"] = {
            "pixels": [[0, 0], [1, 1]],
            "expected_components": 2,
            "got_components": len(color1_comps)
        }
        return receipt

    # ========================================================================
    # Check 3: Fold-to-anchor proof
    # ========================================================================
    # Grid with component of size ≥2
    G3 = [
        [5, 5, 5],
        [0, 5, 0],
        [0, 0, 0]
    ]

    comps3 = build_components(G3)
    color5_comp = next(c for c in comps3 if c.color == 5)

    # Verify fold-to-anchor maps all pixels to same anchor
    views3 = component_anchor_views([color5_comp])
    view3 = views3[0]
    apply_fn = view3["apply"]

    for x in color5_comp.mask:
        result = apply_fn(x)
        if result != color5_comp.anchor:
            receipt["proof_reconstruct_ok"] = False
            receipt["examples"]["case"] = "fold_proof"
            receipt["examples"]["detail"] = {
                "x": list(x),
                "expected_anchor": list(color5_comp.anchor),
                "got": list(result) if result else None
            }
            return receipt

        # Verify G(M(x)) = G(x)
        i, j = x
        ai, aj = color5_comp.anchor
        if G3[i][j] != G3[ai][aj]:
            receipt["proof_reconstruct_ok"] = False
            receipt["examples"]["case"] = "fold_proof"
            receipt["examples"]["detail"] = {
                "x": [i, j],
                "Gx": G3[i][j],
                "Ganchor": G3[ai][aj]
            }
            return receipt

    # ========================================================================
    # Check 4: Ordering & no duplicates
    # ========================================================================
    G4 = [
        [2, 0, 1],
        [2, 1, 1],
        [0, 0, 2]
    ]

    comps4 = build_components(G4)

    # Verify strictly increasing (color, anchor)
    for i in range(len(comps4) - 1):
        c1 = comps4[i]
        c2 = comps4[i + 1]
        key1 = (c1.color, c1.anchor[0], c1.anchor[1])
        key2 = (c2.color, c2.anchor[0], c2.anchor[1])

        if key1 >= key2:
            receipt["proof_reconstruct_ok"] = False
            receipt["examples"]["case"] = "order"
            receipt["examples"]["detail"] = {
                "comp1": {"color": c1.color, "anchor": list(c1.anchor)},
                "comp2": {"color": c2.color, "anchor": list(c2.anchor)},
                "ordering_violated": True
            }
            return receipt

    # Verify no duplicates in masks
    for comp in comps4:
        if len(comp.mask) != len(set(comp.mask)):
            receipt["proof_reconstruct_ok"] = False
            receipt["examples"]["case"] = "duplicate"
            receipt["examples"]["detail"] = {
                "color": comp.color,
                "comp_id": comp.comp_id,
                "mask_size": len(comp.mask),
                "unique_size": len(set(comp.mask))
            }
            return receipt

    # ========================================================================
    # Final receipt (reference grid G3)
    # ========================================================================
    receipt["count_total"] = len(comps3)

    # Count by color
    by_color = {}
    for c in comps3:
        by_color[str(c.color)] = by_color.get(str(c.color), 0) + 1
    receipt["by_color"] = by_color

    # Largest component
    if comps3:
        largest = max(comps3, key=lambda c: len(c.mask))
        receipt["largest"] = {
            "color": largest.color,
            "size": len(largest.mask),
            "anchor": list(largest.anchor)
        }

    # First 5 anchors
    for c in comps3[:5]:
        receipt["anchors_first5"].append({
            "color": c.color,
            "id": c.comp_id,
            "anchor": list(c.anchor),
            "size": len(c.mask)
        })

    return receipt


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================


def init() -> None:
    """
    Run self-check and emit components receipt.

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
        receipts.log("components", {
            "count_total": 0,
            "by_color": {},
            "largest": {},
            "anchors_first5": [],
            "proof_reconstruct_ok": False,
            "examples": {},
            "note": "self-check skipped (ARC_SELF_CHECK != 1)"
        })
        return

    receipt = _self_check_components()

    # Emit receipt
    receipts.log("components", receipt)

    # Assert all checks passed
    if not receipt["proof_reconstruct_ok"]:
        case = receipt["examples"].get("case", "unknown")
        detail = receipt["examples"].get("detail", {})

        if case == "reconstruct":
            raise AssertionError(
                f"components identity failed: reconstruct mismatch at "
                f"p={detail['p']}, want={detail['want']}, got={detail['got']}"
            )
        elif case == "connectivity":
            raise AssertionError(
                f"components identity failed: diagonal pixels merged, "
                f"expected {detail['expected_components']} components, "
                f"got {detail['got_components']}"
            )
        elif case == "fold_proof":
            if "expected_anchor" in detail:
                raise AssertionError(
                    f"components identity failed: fold-to-anchor returned wrong value at "
                    f"x={detail['x']}, expected={detail['expected_anchor']}, got={detail['got']}"
                )
            else:
                raise AssertionError(
                    f"components identity failed: G(M(x)) != G(x) at "
                    f"x={detail['x']}, Gx={detail['Gx']}, Ganchor={detail['Ganchor']}"
                )
        elif case == "order":
            raise AssertionError(
                f"components identity failed: ordering violated between "
                f"comp1={detail['comp1']} and comp2={detail['comp2']}"
            )
        elif case == "duplicate":
            raise AssertionError(
                f"components identity failed: duplicate pixels in component "
                f"color={detail['color']}, comp_id={detail['comp_id']}"
            )
        else:
            raise AssertionError(
                f"components identity failed: case={case}, detail={detail}"
            )
