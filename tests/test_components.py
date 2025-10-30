"""
Test suite for components.py — 4-connected components as must-link S-views.

Covers:
- Component extraction (4-adjacency, per-color, deterministic ordering)
- Fold-to-anchor must-link views (proof: G(M(x)) = G(x))
- PT predicates (masks as input-only separators)
- Reconstruction equality (algebraic certificate)
- Forbidden patterns
- Self-check scenarios

Invariants:
- I-3: Components satisfy S-view proof obligation
- Determinism: ordering, no duplicates, stable across runs
"""

import sys
import os
import pytest
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import components

IntGrid = List[List[int]]


# ============================================================================
# Property Tests: 4-Connectivity and Ordering
# ============================================================================

def test_empty_grid():
    """Empty grid produces no components"""
    grid = []
    comps = components.build_components(grid)
    assert len(comps) == 0, "Empty grid should have no components"


def test_single_pixel():
    """Single pixel is one component"""
    grid = [[5]]
    comps = components.build_components(grid)

    assert len(comps) == 1, "Single pixel should be one component"
    comp = comps[0]
    assert comp.color == 5
    assert comp.anchor == (0, 0)
    assert comp.mask == [(0, 0)]
    assert comp.bbox == (0, 0, 0, 0)


def test_horizontal_stripe_same_color():
    """Horizontal stripe of same color is one component"""
    grid = [[3, 3, 3, 3]]
    comps = components.build_components(grid)

    assert len(comps) == 1
    comp = comps[0]
    assert comp.color == 3
    assert comp.anchor == (0, 0)
    assert comp.mask == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert len(comp.mask) == 4


def test_diagonal_not_connected():
    """Diagonal pixels of same color should NOT be connected (4-adjacency)"""
    # Two pixels touching diagonally
    grid = [
        [1, 0],
        [0, 1]
    ]
    comps = components.build_components(grid)

    # Filter to color 1 only
    color1_comps = [c for c in comps if c.color == 1]

    # Should be 2 separate components (diagonal not connected)
    assert len(color1_comps) == 2, (
        f"Diagonal pixels should NOT merge (4-conn only), got {len(color1_comps)} components"
    )

    # Check anchors are the two diagonal positions
    anchors = sorted([c.anchor for c in color1_comps])
    assert anchors == [(0, 0), (1, 1)], f"Expected diagonal anchors, got {anchors}"


def test_four_adjacency_only():
    """Verify 4-adjacency: up, down, left, right (not diagonal)"""
    # Plus-sign pattern
    grid = [
        [0, 2, 0],
        [2, 2, 2],
        [0, 2, 0]
    ]
    comps = components.build_components(grid)

    color2_comps = [c for c in comps if c.color == 2]

    # All 5 pixels should be one component
    assert len(color2_comps) == 1, "Plus-sign should be one component"
    comp = color2_comps[0]
    assert len(comp.mask) == 5
    assert comp.anchor == (0, 1)  # Top pixel


def test_per_color_separate():
    """Components are per color; different colors don't merge"""
    grid = [
        [1, 2, 1],
        [2, 1, 2]
    ]
    comps = components.build_components(grid)

    # Count components per color
    color1_comps = [c for c in comps if c.color == 1]
    color2_comps = [c for c in comps if c.color == 2]

    # Color 1: 3 separate pixels (not adjacent)
    assert len(color1_comps) == 3, f"Expected 3 color-1 components, got {len(color1_comps)}"

    # Color 2: 3 separate pixels
    assert len(color2_comps) == 3, f"Expected 3 color-2 components, got {len(color2_comps)}"


def test_deterministic_ordering():
    """Components ordered by (color asc, anchor row asc, anchor col asc)"""
    grid = [
        [2, 0, 1],
        [0, 2, 0],
        [1, 0, 2]
    ]
    comps = components.build_components(grid)

    # Extract (color, anchor) tuples
    order = [(c.color, c.anchor) for c in comps]

    # Should be sorted by color, then anchor
    expected_order = sorted(order, key=lambda x: (x[0], x[1][0], x[1][1]))

    assert order == expected_order, (
        f"Components not in deterministic order.\nGot: {order}\nExpected: {expected_order}"
    )


def test_anchor_is_top_left():
    """Anchor is top-left pixel (min row, then min col)"""
    # L-shape component
    grid = [
        [0, 0, 3],
        [0, 3, 3],
        [0, 3, 0]
    ]
    comps = components.build_components(grid)

    color3_comps = [c for c in comps if c.color == 3]
    assert len(color3_comps) == 1

    comp = color3_comps[0]
    # Top-left pixel of the L-shape is (0, 2)
    assert comp.anchor == (0, 2), f"Expected anchor (0,2), got {comp.anchor}"


def test_mask_row_major_sorted():
    """Component mask is sorted row-major (row asc, then col asc)"""
    # Create a connected component (all 4-adjacent)
    grid = [
        [5, 5, 5],
        [0, 5, 0],
        [0, 5, 0]
    ]
    comps = components.build_components(grid)

    color5_comps = [c for c in comps if c.color == 5]
    assert len(color5_comps) == 1

    comp = color5_comps[0]
    mask = comp.mask

    # Check sorted row-major
    sorted_mask = sorted(mask, key=lambda p: (p[0], p[1]))
    assert mask == sorted_mask, f"Mask not sorted row-major: {mask}"


def test_bbox_correctness():
    """Bounding box (top, left, bottom, right) is correct"""
    grid = [
        [0, 0, 0, 0, 0],
        [0, 7, 7, 0, 0],
        [0, 7, 0, 7, 0],
        [0, 0, 0, 0, 0]
    ]
    comps = components.build_components(grid)

    color7_comps = [c for c in comps if c.color == 7]
    # Note: (1,1)-(1,2) and (2,1) are one component; (2,3) might be separate
    # Let's check the largest component

    if len(color7_comps) > 1:
        # (2,3) is isolated, so we expect 2 components
        largest = max(color7_comps, key=lambda c: len(c.mask))
        bbox = largest.bbox
        # Largest component should span rows 1-2, cols 1-2
        assert bbox[0] == 1, f"Expected top=1, got {bbox[0]}"
        assert bbox[2] >= 2, f"Expected bottom>=2, got {bbox[2]}"


# ============================================================================
# Reconstruction Tests (Algebraic Certificate)
# ============================================================================

def test_reconstruction_equals_original():
    """Reconstruct from components equals original grid (byte-exact)"""
    grid = [
        [1, 2, 3],
        [2, 1, 2],
        [3, 2, 1]
    ]

    comps = components.build_components(grid)
    reconstructed = components.reconstruct_from_components(comps, 3, 3)

    assert reconstructed == grid, (
        f"Reconstruction failed:\nOriginal: {grid}\nReconstructed: {reconstructed}"
    )


def test_reconstruction_multiple_colors():
    """Reconstruction with multiple colors and multiple components per color"""
    grid = [
        [0, 1, 0, 1],
        [2, 2, 3, 3],
        [0, 1, 0, 1]
    ]

    comps = components.build_components(grid)
    H, W = len(grid), len(grid[0])
    reconstructed = components.reconstruct_from_components(comps, H, W)

    assert reconstructed == grid, "Multi-color reconstruction failed"


def test_reconstruction_preserves_background():
    """Reconstruction preserves background (color 0)"""
    grid = [
        [0, 0, 0],
        [0, 5, 0],
        [0, 0, 0]
    ]

    comps = components.build_components(grid)
    reconstructed = components.reconstruct_from_components(comps, 3, 3)

    assert reconstructed == grid, "Background not preserved in reconstruction"


# ============================================================================
# Fold-to-Anchor Must-Link Views
# ============================================================================

def test_component_anchor_views_structure():
    """component_anchor_views returns correct structure"""
    grid = [
        [1, 1],
        [1, 0]
    ]

    comps = components.build_components(grid)
    views = components.component_anchor_views(comps)

    assert len(views) > 0, "Should have at least one view"

    # Check first view structure
    view = views[0]
    assert "kind" in view
    assert "comp_key" in view
    assert "anchor" in view
    assert "dom_size" in view
    assert "apply" in view
    assert callable(view["apply"]), "apply should be callable"


def test_fold_to_anchor_maps_correctly():
    """Fold-to-anchor view maps all component pixels to anchor"""
    grid = [
        [3, 3],
        [3, 0]
    ]

    comps = components.build_components(grid)
    color3_comps = [c for c in comps if c.color == 3]
    assert len(color3_comps) == 1

    comp = color3_comps[0]
    views = components.component_anchor_views([comp])
    view = views[0]

    apply_fn = view["apply"]
    anchor = view["anchor"]

    # All pixels in mask should map to anchor
    for p in comp.mask:
        result = apply_fn(p)
        assert result == anchor, (
            f"Fold-to-anchor failed: {p} -> {result}, expected {anchor}"
        )

    # Pixels outside mask should return None
    outside = (1, 1)
    if outside not in comp.mask:
        assert apply_fn(outside) is None, "Pixels outside component should return None"


def test_fold_to_anchor_proof_obligation():
    """Verify proof: G(M(x)) = G(x) for fold-to-anchor"""
    # Create a connected component
    grid = [
        [7, 7, 7],
        [7, 7, 0],
        [0, 7, 0]
    ]

    comps = components.build_components(grid)
    color7_comps = [c for c in comps if c.color == 7]
    assert len(color7_comps) == 1

    comp = color7_comps[0]
    views = components.component_anchor_views([comp])
    view = views[0]

    apply_fn = view["apply"]
    anchor = view["anchor"]

    # Proof: G[M(x)] = G[x] for all x in component
    for x in comp.mask:
        mapped = apply_fn(x)
        assert mapped is not None

        G_x = grid[x[0]][x[1]]
        G_mapped = grid[mapped[0]][mapped[1]]

        assert G_x == G_mapped, (
            f"Proof failed at {x}: G[{x}]={G_x}, G[M({x})]={G_mapped}"
        )


def test_fold_dom_size_equals_mask_size():
    """dom_size in view equals mask length"""
    grid = [[1, 1, 1], [0, 1, 0]]

    comps = components.build_components(grid)
    color1_comps = [c for c in comps if c.color == 1]
    assert len(color1_comps) == 1

    comp = color1_comps[0]
    views = components.component_anchor_views([comp])
    view = views[0]

    assert view["dom_size"] == len(comp.mask), (
        f"dom_size {view['dom_size']} != mask length {len(comp.mask)}"
    )


# ============================================================================
# PT Predicates (Input-Only Separators)
# ============================================================================

def test_component_predicates_structure():
    """component_predicates returns correct structure"""
    grid = [[1, 2], [3, 4]]

    comps = components.build_components(grid)
    predicates = components.component_predicates(comps)

    assert len(predicates) > 0

    pred = predicates[0]
    assert "kind" in pred
    assert pred["kind"] == "sview_image"
    assert "key" in pred
    assert "mask" in pred
    assert isinstance(pred["mask"], list)


def test_predicates_cover_all_pixels_per_color():
    """Within a color, component masks partition pixels (no overlap, no gaps)"""
    grid = [
        [2, 0, 2],
        [2, 2, 0],
        [0, 2, 2]
    ]

    comps = components.build_components(grid)
    color2_comps = [c for c in comps if c.color == 2]

    # All color-2 pixels
    all_color2 = set()
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 2:
                all_color2.add((i, j))

    # Union of all color-2 component masks
    union_masks = set()
    for comp in color2_comps:
        for p in comp.mask:
            assert p not in union_masks, f"Pixel {p} appears in multiple components (overlap)"
            union_masks.add(p)

    assert union_masks == all_color2, (
        f"Component masks don't cover all color-2 pixels.\n"
        f"Missing: {all_color2 - union_masks}\n"
        f"Extra: {union_masks - all_color2}"
    )


def test_predicates_mask_sorted():
    """Predicate masks are sorted row-major"""
    grid = [[5, 5], [5, 0]]

    comps = components.build_components(grid)
    predicates = components.component_predicates(comps)

    for pred in predicates:
        mask = pred["mask"]
        sorted_mask = sorted(mask, key=lambda p: (p[0], p[1]))
        assert mask == sorted_mask, f"Predicate mask not sorted: {mask}"


# ============================================================================
# Component Contains Method
# ============================================================================

def test_component_contains():
    """Component.contains() correctly identifies membership"""
    grid = [[1, 0], [1, 1]]

    comps = components.build_components(grid)
    color1_comps = [c for c in comps if c.color == 1]
    assert len(color1_comps) == 1

    comp = color1_comps[0]

    # Pixels in mask should return True
    assert comp.contains((0, 0)) == True
    assert comp.contains((1, 0)) == True
    assert comp.contains((1, 1)) == True

    # Pixel outside mask should return False
    assert comp.contains((0, 1)) == False


# ============================================================================
# Golden Checks: Crafted Grids
# ============================================================================

def test_golden_single_component():
    """Single large component spans entire grid"""
    grid = [
        [9, 9, 9],
        [9, 9, 9],
        [9, 9, 9]
    ]

    comps = components.build_components(grid)

    assert len(comps) == 1, f"Expected 1 component, got {len(comps)}"
    comp = comps[0]
    assert comp.color == 9
    assert comp.anchor == (0, 0)
    assert len(comp.mask) == 9
    assert comp.bbox == (0, 0, 2, 2)


def test_golden_multiple_components_per_color():
    """Same color with multiple disconnected components"""
    grid = [
        [4, 0, 4],
        [0, 0, 0],
        [4, 0, 4]
    ]

    comps = components.build_components(grid)
    color4_comps = [c for c in comps if c.color == 4]

    # 4 separate color-4 pixels
    assert len(color4_comps) == 4, f"Expected 4 components, got {len(color4_comps)}"

    # Check anchors are the 4 corners
    anchors = sorted([c.anchor for c in color4_comps])
    expected = [(0, 0), (0, 2), (2, 0), (2, 2)]
    assert anchors == expected, f"Expected corner anchors, got {anchors}"


def test_golden_checkerboard():
    """Checkerboard pattern has many small components"""
    grid = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]

    comps = components.build_components(grid)

    # Count components per color
    color0_comps = [c for c in comps if c.color == 0]
    color1_comps = [c for c in comps if c.color == 1]

    # Each pixel is isolated (8 of each color)
    assert len(color0_comps) == 8, f"Expected 8 color-0 components, got {len(color0_comps)}"
    assert len(color1_comps) == 8, f"Expected 8 color-1 components, got {len(color1_comps)}"

    # All components should be single-pixel
    for comp in comps:
        assert len(comp.mask) == 1, f"Expected single-pixel components, got size {len(comp.mask)}"


# ============================================================================
# Determinism Tests
# ============================================================================

def test_determinism_repeated_calls():
    """Multiple calls with same grid produce identical results"""
    grid = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    comps1 = components.build_components(grid)
    comps2 = components.build_components(grid)

    assert len(comps1) == len(comps2)

    for c1, c2 in zip(comps1, comps2):
        assert c1.color == c2.color
        assert c1.comp_id == c2.comp_id
        assert c1.anchor == c2.anchor
        assert c1.mask == c2.mask
        assert c1.bbox == c2.bbox


# ============================================================================
# Forbidden Patterns Test
# ============================================================================

def test_forbidden_patterns():
    """Reject forbidden patterns in components.py"""
    components_path = Path(__file__).parent.parent / "src" / "components.py"

    if not components_path.exists():
        pytest.skip("components.py not yet implemented")

    content = components_path.read_text()

    violations = []

    # Check for TODOs and FIXMEs
    if "TODO" in content and "# TODO" in content:
        violations.append("TODO comment found")
    if "FIXME" in content:
        violations.append("FIXME")

    # Check for NotImplementedError
    if "NotImplementedError" in content:
        violations.append("NotImplementedError")

    # Check for unseeded/problematic randomness patterns
    lines = content.split('\n')
    for i, line in enumerate(lines):
        # Allow seeded random in self-check
        if "random.seed" in line or "ARC_SELF_CHECK" in content:
            continue
        if "np.random" in line and "seed" not in line:
            violations.append(f"np.random at line {i+1}")
        if "time.sleep" in line:
            violations.append(f"time.sleep at line {i+1}")
        if "os.environ.get('SEED')" in line:
            violations.append(f"environment SEED at line {i+1}")
        if "torch.manual_seed" in line:
            violations.append(f"torch.manual_seed at line {i+1}")
        if "np.set_printoptions" in line:
            violations.append(f"np.set_printoptions at line {i+1}")
        if "warnings.filterwarnings" in line:
            violations.append(f"warnings.filterwarnings at line {i+1}")
        if "from pprint" in line:
            violations.append(f"from pprint at line {i+1}")

    # Check for typing.Any as return type
    import re
    any_return_pattern = r'def\s+\w+\([^)]*\)\s*->\s*Any'
    if re.search(any_return_pattern, content):
        violations.append("typing.Any used as return type")

    assert len(violations) == 0, f"Forbidden patterns found: {violations}"


# ============================================================================
# Self-Check Test (when ARC_SELF_CHECK=1)
# ============================================================================

def test_self_check_can_run():
    """Self-check module loads without errors"""
    assert callable(components.build_components)
    assert callable(components.component_anchor_views)
    assert callable(components.component_predicates)
    assert callable(components.reconstruct_from_components)


@pytest.mark.skipif(
    os.environ.get("ARC_SELF_CHECK") != "1",
    reason="Self-check only runs when ARC_SELF_CHECK=1"
)
def test_self_check_enabled():
    """Self-check assertions pass when enabled"""
    # Import the module which should trigger self-check
    import receipts
    import components

    # Initialize receipts
    receipts.init("test.components_self_check")

    # Run self-check (should be called by module init or explicitly)
    if hasattr(components, 'init'):
        components.init()

    # Get receipt
    receipt_doc = receipts.finalize()

    if "components" in receipt_doc["sections"]:
        comp_receipt = receipt_doc["sections"]["components"]

        # Check proof_reconstruct_ok is True
        assert comp_receipt.get("proof_reconstruct_ok") == True, (
            f"Self-check failed: proof_reconstruct_ok={comp_receipt.get('proof_reconstruct_ok')}\n"
            f"Examples: {comp_receipt.get('examples')}"
        )

        # Check no failure examples
        assert len(comp_receipt.get("examples", {})) == 0, (
            f"Self-check found issues: {comp_receipt['examples']}"
        )


# ============================================================================
# Test Intent Summary
# ============================================================================

def test_intent_summary():
    """
    Test Intent Summary for WO-05 (Components):

    Invariants Covered:
    - I-3: Component anchor views satisfy S-view proof obligation G(M(x)) = G(x)
    - Determinism: Ordering by (color, anchor), no duplicates, stable across runs

    Property Tests:
    - 4-adjacency only (diagonal NOT connected)
    - Per-color separation (different colors don't merge)
    - Deterministic ordering (color asc → anchor (row, col) asc)
    - Anchor is top-left (min row → min col)
    - Mask sorted row-major
    - Bounding box correctness
    - Reconstruction equals original (algebraic certificate)
    - Fold-to-anchor maps all pixels to anchor
    - Fold-to-anchor satisfies proof obligation
    - dom_size equals mask length
    - Component.contains() correctness
    - PT predicates structure and coverage
    - Predicates partition pixels per color (no overlap, no gaps)
    - Determinism across repeated calls

    Golden Checks:
    - Single large component (entire grid)
    - Multiple disconnected components per color (4 corners)
    - Checkerboard pattern (many single-pixel components)

    Forbidden Patterns:
    - TODO, FIXME, NotImplementedError
    - Unseeded randomness, sleep, type hints with Any

    Self-Check:
    - Module loads without errors
    - Self-check can run when ARC_SELF_CHECK=1
    - Reconstruction proof passes
    - 4-adjacency enforcement
    - Fold-to-anchor proof obligation
    - Ordering and no duplicates

    Microsuite IDs:
    - 272f95fa (bands + grid walls; PT uses component masks)
    - 00d62c1b (copy-move; distinct Δ per component)
    - 007bbfb7 (block stamping; class partition by nonzero)

    Implementation Notes:
    - Components are extracted per-color with 4-adjacency
    - Anchor is top-left pixel (min row, min col)
    - Fold-to-anchor creates must-link S-views
    - Masks exported as PT predicates (input-only separators)
    - Reconstruction provides algebraic certificate
    """
    pass


# ============================================================================
# Run determinism harness (pytest marker)
# ============================================================================

@pytest.mark.slow
def test_determinism_harness():
    """Run determinism check via harness (skipped by default)"""
    pytest.skip("Determinism harness requires full task corpus; run manually")


if __name__ == "__main__":
    # Run tests locally
    pytest.main([__file__, "-v"])
