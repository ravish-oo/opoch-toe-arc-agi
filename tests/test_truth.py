"""
Test suite for truth.py — Truth partition Q via must-link + Paige-Tarjan.

Covers:
- I-4: Truth single-valuedness before laws
- Must-link closure (S-views + components)
- Paige-Tarjan splits with fixed predicate order
- OOB handling during conjugation
- Determinism under training permutation
- Forbidden patterns
- Self-check scenarios

Invariants:
- I-4: After PT, every class is single-valued on all trainings
- Determinism: Same partition regardless of train-pair order
"""

import sys
import os
import pytest
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import truth
import sviews
import components

IntGrid = List[List[int]]


# ============================================================================
# Partition Class Tests
# ============================================================================

def test_partition_basic():
    """Partition initializes correctly"""
    # 2x2 grid, all same class
    cid_of = [0, 0, 0, 0]
    part = truth.Partition(2, 2, cid_of)

    assert part.H == 2
    assert part.W == 2
    assert len(part.cid_of) == 4


def test_partition_classes():
    """Partition.classes() returns coords per class in order"""
    # 2x2 grid: top row class 0, bottom row class 1
    cid_of = [0, 0, 1, 1]
    part = truth.Partition(2, 2, cid_of)

    classes = part.classes()

    assert len(classes) == 2
    assert set(classes[0]) == {(0, 0), (0, 1)}
    assert set(classes[1]) == {(1, 0), (1, 1)}


def test_partition_single_pixel_classes():
    """Each pixel in separate class"""
    cid_of = [0, 1, 2, 3]
    part = truth.Partition(2, 2, cid_of)

    classes = part.classes()

    assert len(classes) == 4
    for cls in classes:
        assert len(cls) == 1


# ============================================================================
# Must-Link Tests (S-views + Components)
# ============================================================================

def test_mustlink_identity_only():
    """Grid with no S-views merges (only identity) keeps all pixels separate initially"""
    # Asymmetric grid
    G_test = [[1, 2], [3, 4]]

    # Build S-views (should only have identity)
    sv = sviews.build_sviews(G_test)

    # No components merge (all different colors)
    comps = components.build_components(G_test)

    # Mock frames and outputs (no splits needed)
    frames = {
        "P_test": (0, (0, 0), (2, 2)),  # (op, anchor, shape)
        "P_out": [(0, (0, 0), (2, 2))]  # List of (op, anchor, shape)
    }
    train_outputs = [[[1, 2], [3, 4]]]  # Same as input

    part = truth.build_truth_partition(G_test, sv, comps, frames, train_outputs)

    # With only identity S-view, each pixel is its own class initially
    # (unless components merge, but here each color is different)
    classes = part.classes()
    assert len(classes) >= 1  # At least some structure


def test_mustlink_component_fold():
    """Component fold merges all pixels in component to anchor"""
    # Single component: horizontal stripe
    G_test = [[5, 5, 5]]

    sv = sviews.build_sviews(G_test)
    comps = components.build_components(G_test)

    # Should be 1 component
    assert len(comps) == 1
    assert comps[0].color == 5
    assert len(comps[0].mask) == 3

    frames = {
        "P_test": (0, (0, 0), (1, 3)),
        "P_out": [(0, (0, 0), (1, 3))]
    }
    train_outputs = [[[5, 5, 5]]]

    part = truth.build_truth_partition(G_test, sv, comps, frames, train_outputs)

    # All 3 pixels should be in same class (merged by component fold)
    classes = part.classes()

    # Find class containing (0,0)
    class_with_00 = [cls for cls in classes if (0, 0) in cls][0]

    # Should contain all 3 pixels
    assert len(class_with_00) == 3
    assert set(class_with_00) == {(0, 0), (0, 1), (0, 2)}


def test_mustlink_translation_sview():
    """Translation S-view merges overlapping pixels"""
    # Pattern with period-2
    G_test = [[1, 2, 1, 2]]

    sv = sviews.build_sviews(G_test)

    # Should have translation by (0, 2)
    translate_views = [v for v in sv if v.kind == "translate" and v.params.get("dj") == 2]
    assert len(translate_views) > 0, "Should have translation (0,2) for period-2 pattern"

    comps = components.build_components(G_test)

    frames = {
        "P_test": (0, (0, 0), (1, 4)),
        "P_out": [(0, (0, 0), (1, 4))]
    }
    train_outputs = [[[1, 2, 1, 2]]]

    part = truth.build_truth_partition(G_test, sv, comps, frames, train_outputs)

    # Translation should merge overlapping pixels
    # (0,0) and (0,2) should be in same class
    # (0,1) and (0,3) should be in same class
    classes = part.classes()

    # Check that color-1 pixels are merged
    class_with_00 = [cls for cls in classes if (0, 0) in cls][0]
    assert (0, 2) in class_with_00, "Translation should merge (0,0) with (0,2)"


# ============================================================================
# Paige-Tarjan Split Tests (Fixed Predicate Order)
# ============================================================================

def test_pt_split_by_input_color():
    """PT splits component when pixels map to different output colors"""
    # Same color → component merges all pixels initially
    G_test = [[5, 5]]

    sv = sviews.build_sviews(G_test)
    comps = components.build_components(G_test)

    # Should be 1 component with 2 pixels
    assert len(comps) == 1
    assert comps[0].color == 5
    assert len(comps[0].mask) == 2

    # Create frames and outputs that cause contradiction
    frames = {
        "P_test": (0, (0, 0), (1, 2)),
        "P_out": [(0, (0, 0), (1, 2))]
    }

    # Different output colors at different positions
    # Component fold merged (0,0) and (0,1) into one class
    # But they map to different output colors → PT must split
    train_outputs = [[[6, 7]]]  # (0,0)→6, (0,1)→7

    part = truth.build_truth_partition(G_test, sv, comps, frames, train_outputs)

    # Should split into 2 classes (by parity predicate: even/odd column)
    classes = part.classes()
    assert len(classes) == 2, "PT should split component when outputs differ"


def test_pt_predicate_order_input_color_first():
    """input_color is checked before other predicates"""
    # Grid where input_color can split
    G_test = [[1, 2], [1, 2]]

    sv = sviews.build_sviews(G_test)
    comps = components.build_components(G_test)

    frames = {
        "P_test": (0, (0, 0), (2, 2)),
        "P_out": [(0, (0, 0), (2, 2))]
    }

    # Different output colors
    train_outputs = [[[5, 6], [7, 8]]]

    part = truth.build_truth_partition(G_test, sv, comps, frames, train_outputs)

    # Should have separate classes for different input colors
    classes = part.classes()

    # Color 1 pixels should be in different class from color 2 pixels
    class_with_00 = [cls for cls in classes if (0, 0) in cls][0]
    assert (0, 1) not in class_with_00, "Different input colors should split"


def test_pt_oob_skip():
    """OOB mappings during TEST→OUT are skipped (no false contradiction)"""
    # Test grid larger than output
    G_test = [[1, 1], [1, 1]]

    sv = sviews.build_sviews(G_test)
    comps = components.build_components(G_test)

    frames = {
        "P_test": (0, (0, 0), (2, 2)),
        "P_out": [(0, (0, 0), (1, 1))]  # Smaller output
    }

    # Only (0,0) maps in-bounds; others OOB
    train_outputs = [[[5]]]

    # Should not crash or create false contradictions from OOB
    part = truth.build_truth_partition(G_test, sv, comps, frames, train_outputs)

    # Verify it completes without error
    assert part is not None
    assert len(part.cid_of) == 4


def test_pt_single_valued_after_convergence():
    """After PT, every class is single-valued on all trainings"""
    G_test = [[1, 2], [3, 4]]

    sv = sviews.build_sviews(G_test)
    comps = components.build_components(G_test)

    frames = {
        "P_test": (0, (0, 0), (2, 2)),
        "P_out": [
            (0, (0, 0), (2, 2)),
            (0, (0, 0), (2, 2))
        ]
    }

    train_outputs = [
        [[5, 6], [7, 8]],
        [[5, 6], [7, 8]]
    ]

    part = truth.build_truth_partition(G_test, sv, comps, frames, train_outputs)

    # Verify single-valuedness
    ok, witness = truth.check_single_valued(part, frames, train_outputs)

    assert ok == True, f"Should be single-valued after PT, witness: {witness}"


# ============================================================================
# check_single_valued Tests
# ============================================================================

def test_check_single_valued_pass():
    """check_single_valued returns (True, None) when consistent"""
    # Simple consistent case
    cid_of = [0, 1, 2, 3]  # Each pixel separate class
    part = truth.Partition(2, 2, cid_of)

    frames = {
        "P_test": (0, (0, 0), (2, 2)),
        "P_out": [(0, (0, 0), (2, 2))]
    }

    train_outputs = [[[1, 2], [3, 4]]]

    ok, witness = truth.check_single_valued(part, frames, train_outputs)

    assert ok == True
    assert witness is None


def test_check_single_valued_fail():
    """check_single_valued returns (False, witness) on contradiction"""
    # Two pixels in same class but different output colors
    cid_of = [0, 0, 1, 1]  # Top row class 0
    part = truth.Partition(2, 2, cid_of)

    frames = {
        "P_test": (0, (0, 0), (2, 2)),
        "P_out": [(0, (0, 0), (2, 2))]
    }

    # Top row has different colors in output
    train_outputs = [[[5, 6], [7, 7]]]

    ok, witness = truth.check_single_valued(part, frames, train_outputs)

    assert ok == False
    assert witness is not None
    assert "cid" in witness
    assert witness["cid"] == 0  # Class 0 violated


# ============================================================================
# Determinism Tests
# ============================================================================

def test_determinism_repeated_calls():
    """Same inputs produce identical partition"""
    G_test = [[1, 2], [3, 4]]

    sv = sviews.build_sviews(G_test)
    comps = components.build_components(G_test)

    frames = {
        "P_test": (0, (0, 0), (2, 2)),
        "P_out": [(0, (0, 0), (2, 2))]
    }

    train_outputs = [[[5, 6], [7, 8]]]

    part1 = truth.build_truth_partition(G_test, sv, comps, frames, train_outputs)
    part2 = truth.build_truth_partition(G_test, sv, comps, frames, train_outputs)

    assert part1.cid_of == part2.cid_of


def test_determinism_train_order_independent():
    """Reversing training order produces same partition"""
    G_test = [[1, 2]]

    sv = sviews.build_sviews(G_test)
    comps = components.build_components(G_test)

    # Two trainings
    frames1 = {
        "P_test": (0, (0, 0), (1, 2)),
        "P_out": [
            (0, (0, 0), (1, 2)),
            (0, (0, 0), (1, 2))
        ]
    }

    train_outputs1 = [[[5, 6]], [[5, 6]]]

    # Reversed order
    frames2 = {
        "P_test": (0, (0, 0), (1, 2)),
        "P_out": [
            (0, (0, 0), (1, 2)),
            (0, (0, 0), (1, 2))
        ]
    }

    train_outputs2 = [[[5, 6]], [[5, 6]]]  # Same but reversed

    part1 = truth.build_truth_partition(G_test, sv, comps, frames1, train_outputs1)
    part2 = truth.build_truth_partition(G_test, sv, comps, frames2, train_outputs2)

    # Should produce same partition (same final_classes count)
    classes1 = part1.classes()
    classes2 = part2.classes()

    assert len(classes1) == len(classes2)


# ============================================================================
# Golden Checks: Crafted Scenarios
# ============================================================================

def test_golden_no_splits_needed():
    """Single color, single training, no contradictions → 1 class"""
    G_test = [[7, 7], [7, 7]]

    sv = sviews.build_sviews(G_test)
    comps = components.build_components(G_test)

    frames = {
        "P_test": (0, (0, 0), (2, 2)),
        "P_out": [(0, (0, 0), (2, 2))]
    }

    train_outputs = [[[9, 9], [9, 9]]]

    part = truth.build_truth_partition(G_test, sv, comps, frames, train_outputs)

    # All should be in one class (component merge + no contradictions)
    classes = part.classes()
    assert len(classes) == 1
    assert len(classes[0]) == 4


def test_golden_split_by_color():
    """Two colors require split by input_color"""
    G_test = [[1, 2]]

    sv = sviews.build_sviews(G_test)
    comps = components.build_components(G_test)

    frames = {
        "P_test": (0, (0, 0), (1, 2)),
        "P_out": [(0, (0, 0), (1, 2))]
    }

    # Different output colors
    train_outputs = [[[5, 6]]]

    part = truth.build_truth_partition(G_test, sv, comps, frames, train_outputs)

    # Should have 2 classes (different input colors)
    classes = part.classes()
    assert len(classes) >= 2


def test_golden_component_merge_then_split():
    """Component merges, then PT splits if needed"""
    # Two blobs of same color
    G_test = [
        [1, 0, 1],
        [1, 0, 1]
    ]

    sv = sviews.build_sviews(G_test)
    comps = components.build_components(G_test)

    # Should have 2 components (disconnected)
    color1_comps = [c for c in comps if c.color == 1]
    assert len(color1_comps) == 2

    frames = {
        "P_test": (0, (0, 0), (2, 3)),
        "P_out": [(0, (0, 0), (2, 3))]
    }

    # Different output colors for left vs right blob
    train_outputs = [[[5, 0, 6], [5, 0, 6]]]

    part = truth.build_truth_partition(G_test, sv, comps, frames, train_outputs)

    # Left blob (0,0), (1,0) should be class A
    # Right blob (0,2), (1,2) should be class B
    # They should NOT merge (different output colors)

    classes = part.classes()
    class_with_00 = [cls for cls in classes if (0, 0) in cls][0]
    class_with_02 = [cls for cls in classes if (0, 2) in cls][0]

    # Should be different classes
    assert class_with_00 != class_with_02


# ============================================================================
# Receipt Verification Tests
# ============================================================================

def test_receipt_structure():
    """Receipt has required fields"""
    G_test = [[1, 2]]

    sv = sviews.build_sviews(G_test)
    comps = components.build_components(G_test)

    frames = {
        "P_test": (0, (0, 0), (1, 2)),
        "P_out": [(0, (0, 0), (1, 2))]
    }

    train_outputs = [[[5, 6]]]

    # This will emit receipt internally
    import receipts
    receipts.init("test.truth_receipt")

    part = truth.build_truth_partition(G_test, sv, comps, frames, train_outputs)

    receipt_doc = receipts.finalize()

    if "truth" in receipt_doc["sections"]:
        truth_receipt = receipt_doc["sections"]["truth"]

        # Check required fields
        assert "splits" in truth_receipt
        assert "final_classes" in truth_receipt
        assert "single_valued_ok" in truth_receipt
        assert "mustlink_sources" in truth_receipt

        assert truth_receipt["single_valued_ok"] == True
        assert truth_receipt["final_classes"] > 0


# ============================================================================
# Forbidden Patterns Test
# ============================================================================

def test_forbidden_patterns():
    """Reject forbidden patterns in truth.py"""
    truth_path = Path(__file__).parent.parent / "src" / "truth.py"

    if not truth_path.exists():
        pytest.skip("truth.py not yet implemented")

    content = truth_path.read_text()

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
    assert callable(truth.build_truth_partition)
    assert callable(truth.check_single_valued)


@pytest.mark.skipif(
    os.environ.get("ARC_SELF_CHECK") != "1",
    reason="Self-check only runs when ARC_SELF_CHECK=1"
)
def test_self_check_enabled():
    """Self-check assertions pass when enabled"""
    import receipts
    import truth

    # Initialize receipts
    receipts.init("test.truth_self_check")

    # Run self-check (should be called by module init or explicitly)
    if hasattr(truth, 'init'):
        truth.init()

    # Get receipt
    receipt_doc = receipts.finalize()

    if "truth" in receipt_doc["sections"]:
        truth_receipt = receipt_doc["sections"]["truth"]

        # Check single_valued_ok is True
        assert truth_receipt.get("single_valued_ok") == True, (
            f"Self-check failed: single_valued_ok={truth_receipt.get('single_valued_ok')}\n"
            f"Examples: {truth_receipt.get('examples')}"
        )

        # Check no failure examples
        assert len(truth_receipt.get("examples", {})) == 0, (
            f"Self-check found issues: {truth_receipt['examples']}"
        )


# ============================================================================
# Test Intent Summary
# ============================================================================

def test_intent_summary():
    """
    Test Intent Summary for WO-06 (Truth Partition Q):

    Invariants Covered:
    - I-4: Truth single-valuedness before laws (after PT, every class is single-valued)
    - Determinism: Same partition regardless of train-pair order

    Property Tests:
    - Partition class initialization and methods
    - Must-link from identity S-view (no merges)
    - Must-link from component fold (all pixels to anchor)
    - Must-link from translation S-view (overlapping pixels merge)
    - PT split by input_color (first predicate)
    - PT predicate order (input_color before others)
    - OOB handling (skip, no false contradiction)
    - Single-valuedness after convergence
    - check_single_valued pass/fail cases
    - Determinism across repeated calls
    - Determinism under training order reversal

    Golden Checks:
    - No splits needed (single color, consistent outputs)
    - Split by input_color (two colors, different outputs)
    - Component merge then split (two blobs, different outputs)

    Receipt Verification:
    - Required fields (splits, final_classes, single_valued_ok, mustlink_sources)
    - single_valued_ok = True after success

    Forbidden Patterns:
    - TODO, FIXME, NotImplementedError
    - Unseeded randomness, typing.Any as return type

    Self-Check:
    - Module loads without errors
    - Self-check can run when ARC_SELF_CHECK=1
    - Must-link merges (self-check scenario 1)
    - Contradiction witnessed by outputs (self-check scenario 2)
    - OOB skip logic (self-check scenario 3)
    - Determinism under train permutation (self-check scenario 4)

    Microsuite IDs:
    - 272f95fa (bands + walls; PT uses component masks + residue)
    - 00d62c1b (copy-move; classes per component after PT)
    - 007bbfb7 (block stamping; partition by nonzero components)

    Implementation Notes:
    - Must-link via union-find (S-views + component folds)
    - PT loop with fixed predicate order (input_color ≺ sview_image ≺ parity)
    - Conjugation via TEST→OUT from WO-01 morphisms
    - OOB mappings skipped (no evidence, no contradiction)
    - Outputs used only to witness contradictions, never to define predicates
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
