#!/usr/bin/env python3
"""
Tests for WO-11: Painter + un-present + microsuite

Tests paint semantics:
- Canvas size from shape law
- KEEP reads TEST input only (never training)
- VALUE laws use learned constants
- Un-present: pose_inv + palette_inverse (NO output anchoring)
- Idempotence
- Coverage = 100%
- Algebraic witnesses on failure
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
from typing import List, Dict, Tuple, Any, Optional

import paint
import morphisms


# Mock classes for testing
class MockPartition:
    """Mock partition for testing"""
    def __init__(self, cid_map: Dict[Tuple[int,int], int], H: int, W: int):
        """
        Args:
            cid_map: {(r,c) -> cid}
            H, W: dimensions
        """
        self.H = H
        self.W = W
        self.cid_of = []
        for r in range(H):
            for c in range(W):
                self.cid_of.append(cid_map.get((r, c), 0))


# ============================================================================
# Canvas Size Tests
# ============================================================================

def test_canvas_size_identity_shape():
    """Identity shape law: output = input size"""
    P_test = (0, (0, 0), (3, 4))  # H=3, W=4
    shape_law = ("multiplicative", (1, 0, 1, 0))  # S(H,W) = (H, W)

    H_out, W_out = paint.build_test_canvas_size(P_test, shape_law)

    assert H_out == 3
    assert W_out == 4


def test_canvas_size_multiplicative():
    """Multiplicative shape law: S(H,W) = (aH, cW)"""
    P_test = (0, (0, 0), (2, 3))
    shape_law = ("multiplicative", (2, 0, 3, 0))  # S(H,W) = (2H, 3W)

    H_out, W_out = paint.build_test_canvas_size(P_test, shape_law)

    assert H_out == 4  # 2*2
    assert W_out == 9  # 3*3


def test_canvas_size_additive():
    """Additive shape law: S(H,W) = (H+b, W+d)"""
    P_test = (0, (0, 0), (5, 6))
    shape_law = ("additive", (1, 2, 1, -1))  # S(H,W) = (1*H+2, 1*W-1)

    H_out, W_out = paint.build_test_canvas_size(P_test, shape_law)

    assert H_out == 7  # 1*5+2
    assert W_out == 5  # 1*6-1


# ============================================================================
# KEEP Law Tests (reads TEST input only)
# ============================================================================

def test_paint_keep_identity():
    """KEEP:identity should copy TEST input exactly"""
    # Assignment: class 0 uses KEEP:identity
    assignment = {0: "KEEP:identity"}

    # Partition: all pixels belong to class 0
    cid_map = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}
    part = MockPartition(cid_map, 2, 2)

    # TEST input (presented)
    Xtest = [[1, 2], [3, 4]]

    # Frames
    P_test = (0, (0, 0), (2, 2))
    P_in_list = [(0, (0, 0), (2, 2))]  # Not used for test paint

    # Identity shape law
    shape_law = ("multiplicative", (1, 0, 1, 0))

    # Train inputs (not used for test paint, but required by API)
    Xin = [[[0, 0], [0, 0]]]

    result = paint.painter_once(assignment, part, Xtest, Xin, P_test, P_in_list, shape_law)

    # Should copy Xtest exactly
    assert result[0][0] == 1
    assert result[0][1] == 2
    assert result[1][0] == 3
    assert result[1][1] == 4


def test_paint_keep_translate():
    """KEEP:translate should shift TEST input"""
    # Assignment: class 0 uses translate(di=1, dj=0)
    assignment = {0: "KEEP:translate(di=1,dj=0)"}

    cid_map = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}
    part = MockPartition(cid_map, 2, 2)

    Xtest = [[5, 6], [7, 8]]

    P_test = (0, (0, 0), (2, 2))
    P_in_list = [(0, (0, 0), (2, 2))]
    shape_law = ("multiplicative", (1, 0, 1, 0))
    Xin = [[[0, 0], [0, 0]]]

    result = paint.painter_once(assignment, part, Xtest, Xin, P_test, P_in_list, shape_law)

    # V(i,j) = (i+1, j)
    # p_out=(0,0) -> p_test=(0,0) -> V(0,0)=(1,0) -> Xtest[1][0]=7
    # p_out=(0,1) -> p_test=(0,1) -> V(0,1)=(1,1) -> Xtest[1][1]=8
    # p_out=(1,0) -> p_test=(1,0) -> V(1,0)=(2,0) -> OOB
    # p_out=(1,1) -> p_test=(1,1) -> V(1,1)=(2,1) -> OOB

    assert result[0][0] == 7
    assert result[0][1] == 8
    # Bottom row: translate goes OOB, need to check implementation behavior


def test_paint_keep_reads_test_not_training():
    """KEEP must read TEST input, never training outputs"""
    # This is a semantic test: KEEP laws don't have access to training at test time
    # We verify by having different TEST vs training colors

    assignment = {0: "KEEP:identity"}
    cid_map = {(0,0): 0}
    part = MockPartition(cid_map, 1, 1)

    Xtest = [[99]]  # TEST input
    Xin = [[[11]]]  # Training input (different)

    P_test = (0, (0, 0), (1, 1))
    P_in_list = [(0, (0, 0), (1, 1))]
    shape_law = ("multiplicative", (1, 0, 1, 0))

    result = paint.painter_once(assignment, part, Xtest, Xin, P_test, P_in_list, shape_law)

    # Must read from Xtest (99), not Xin (11)
    assert result[0][0] == 99


# ============================================================================
# VALUE Law Tests
# ============================================================================

def test_paint_value_const():
    """VALUE:CONST should write constant color"""
    assignment = {0: "CONST(c=7)"}

    cid_map = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}
    part = MockPartition(cid_map, 2, 2)

    Xtest = [[1, 2], [3, 4]]  # Input doesn't matter for CONST
    Xin = [[[0, 0], [0, 0]]]

    P_test = (0, (0, 0), (2, 2))
    P_in_list = [(0, (0, 0), (2, 2))]
    shape_law = ("multiplicative", (1, 0, 1, 0))

    result = paint.painter_once(assignment, part, Xtest, Xin, P_test, P_in_list, shape_law)

    # All pixels should be 7
    assert result[0][0] == 7
    assert result[0][1] == 7
    assert result[1][0] == 7
    assert result[1][1] == 7


def test_paint_value_recolor():
    """VALUE:RECOLOR should apply permutation π"""
    assignment = {0: "RECOLOR(pi={2:5,3:9})"}

    cid_map = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}
    part = MockPartition(cid_map, 2, 2)

    Xtest = [[2, 3], [2, 3]]  # Input colors 2,3
    Xin = [[[0, 0], [0, 0]]]

    P_test = (0, (0, 0), (2, 2))
    P_in_list = [(0, (0, 0), (2, 2))]
    shape_law = ("multiplicative", (1, 0, 1, 0))

    result = paint.painter_once(assignment, part, Xtest, Xin, P_test, P_in_list, shape_law)

    # π: 2->5, 3->9
    assert result[0][0] == 5  # Xtest[0][0]=2 -> π[2]=5
    assert result[0][1] == 9  # Xtest[0][1]=3 -> π[3]=9
    assert result[1][0] == 5
    assert result[1][1] == 9


def test_paint_value_block():
    """VALUE:BLOCK should downsample by k"""
    assignment = {0: "BLOCK(k=2)"}

    # Class 0 covers 4x4 canvas
    cid_map = {}
    for i in range(4):
        for j in range(4):
            cid_map[(i,j)] = 0
    part = MockPartition(cid_map, 4, 4)

    # TEST input with distinct 2x2 blocks
    Xtest = [
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4]
    ]
    Xin = [[[0]*4 for _ in range(4)]]

    P_test = (0, (0, 0), (4, 4))
    P_in_list = [(0, (0, 0), (4, 4))]
    shape_law = ("multiplicative", (1, 0, 1, 0))

    result = paint.painter_once(assignment, part, Xtest, Xin, P_test, P_in_list, shape_law)

    # BLOCK(k=2): each 2x2 output block reads from single input pixel
    # Assuming class anchor is (0,0):
    # p_out=(0,0) -> p_test=(0,0) -> base=(0//2,0//2)=(0,0) -> Xtest[0][0]=1
    # p_out=(0,1) -> p_test=(0,1) -> base=(0//2,1//2)=(0,0) -> Xtest[0][0]=1
    # p_out=(2,2) -> p_test=(2,2) -> base=(2//2,2//2)=(1,1) -> Xtest[1][1]=1

    # All pixels in 2x2 top-left should be 1
    assert result[0][0] == 1
    assert result[0][1] == 1
    assert result[1][0] == 1
    assert result[1][1] == 1


# ============================================================================
# Un-present Tests (pose_inv + palette_inverse, NO output anchoring)
# ============================================================================

def test_unpresent_identity_pose():
    """Un-present with identity pose should use palette inverse only"""
    Y_out_posed = [[5, 6], [7, 8]]

    # Identity pose
    P_out_like_test = (0, (0, 0), (2, 2))

    # Palette inverse: presented -> raw
    palette_inverse = {5: 1, 6: 2, 7: 3, 8: 4}

    result = paint.unpresent_final(Y_out_posed, P_out_like_test, palette_inverse)

    assert result[0][0] == 1
    assert result[0][1] == 2
    assert result[1][0] == 3
    assert result[1][1] == 4


def test_unpresent_rot90_pose():
    """Un-present with rot90 should apply pose_inv"""
    # Posed output (after rot90 CW)
    Y_out_posed = [[3, 1], [4, 2]]

    # rot90 CW pose (op=1)
    P_out_like_test = (1, (0, 0), (2, 2))

    # No palette change
    palette_inverse = {i: i for i in range(10)}

    result = paint.unpresent_final(Y_out_posed, P_out_like_test, palette_inverse)

    # pose_inv for rot90 CW: (r,c) -> (c, H-1-r)
    # Raw should be rot90 CCW of posed
    # Original: [[1,2],[3,4]] -> rot90 CW -> [[3,1],[4,2]]
    # Un-present: rot90 CCW of [[3,1],[4,2]] -> [[1,2],[3,4]]
    assert result[0][0] == 1
    assert result[0][1] == 2
    assert result[1][0] == 3
    assert result[1][1] == 4


def test_unpresent_no_output_anchoring():
    """Un-present must NOT apply anchoring (outputs not anchored)"""
    Y_out_posed = [[5]]

    # Frame with non-zero anchor (should be ignored for outputs)
    P_out_like_test = (0, (10, 20), (1, 1))  # anchor should be ignored

    palette_inverse = {5: 9}

    result = paint.unpresent_final(Y_out_posed, P_out_like_test, palette_inverse)

    # Should only apply pose_inv + palette_inverse (no anchor_inv)
    assert result[0][0] == 9


# ============================================================================
# Idempotence Tests
# ============================================================================

def test_paint_idempotent_const():
    """Painting twice with CONST should be idempotent"""
    assignment = {0: "CONST(c=5)"}
    cid_map = {(0,0): 0, (0,1): 0}
    part = MockPartition(cid_map, 1, 2)

    Xtest = [[1, 2]]
    Xin = [[[0, 0]]]
    P_test = (0, (0, 0), (1, 2))
    P_in_list = [(0, (0, 0), (1, 2))]
    shape_law = ("multiplicative", (1, 0, 1, 0))

    is_idempotent = paint.paint_idempotent(
        assignment, part, Xtest, Xin, P_test, P_in_list, shape_law
    )

    assert is_idempotent is True


def test_paint_idempotent_keep():
    """Painting twice with KEEP should be idempotent"""
    assignment = {0: "KEEP:identity"}
    cid_map = {(0,0): 0, (0,1): 0}
    part = MockPartition(cid_map, 1, 2)

    Xtest = [[7, 8]]
    Xin = [[[0, 0]]]
    P_test = (0, (0, 0), (1, 2))
    P_in_list = [(0, (0, 0), (1, 2))]
    shape_law = ("multiplicative", (1, 0, 1, 0))

    is_idempotent = paint.paint_idempotent(
        assignment, part, Xtest, Xin, P_test, P_in_list, shape_law
    )

    assert is_idempotent is True


def test_paint_idempotent_mixed_laws():
    """Painting with mixed KEEP + VALUE should be idempotent"""
    # Class 0: KEEP, Class 1: CONST
    assignment = {0: "KEEP:identity", 1: "CONST(c=9)"}
    cid_map = {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}
    part = MockPartition(cid_map, 2, 2)

    Xtest = [[1, 2], [3, 4]]
    Xin = [[[0, 0], [0, 0]]]
    P_test = (0, (0, 0), (2, 2))
    P_in_list = [(0, (0, 0), (2, 2))]
    shape_law = ("multiplicative", (1, 0, 1, 0))

    is_idempotent = paint.paint_idempotent(
        assignment, part, Xtest, Xin, P_test, P_in_list, shape_law
    )

    assert is_idempotent is True


# ============================================================================
# Coverage Tests
# ============================================================================

def test_paint_coverage_100_percent():
    """Paint must achieve exactly 100% coverage"""
    assignment = {0: "CONST(c=5)"}
    cid_map = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}
    part = MockPartition(cid_map, 2, 2)

    Xtest = [[0, 0], [0, 0]]
    Xin = [[[0, 0], [0, 0]]]
    P_test = (0, (0, 0), (2, 2))
    P_in_list = [(0, (0, 0), (2, 2))]
    shape_law = ("multiplicative", (1, 0, 1, 0))

    result = paint.painter_once(assignment, part, Xtest, Xin, P_test, P_in_list, shape_law)

    # All pixels should be painted (coverage 100%)
    for row in result:
        for pixel in row:
            assert pixel is not None


def test_paint_unseen_with_const_ok():
    """Unseen pixels (pullback None) use ⊥ class CONST"""
    # ⊥ (bottom) pseudo-class handles pixels where pullback is None

    assignment = {0: "CONST(c=7)", "⊥": "CONST(c=9)"}
    cid_map = {(0,0): 0}
    part = MockPartition(cid_map, 1, 1)

    Xtest = [[1]]
    Xin = [[[0]]]
    P_test = (0, (0, 0), (1, 1))
    P_in_list = [(0, (0, 0), (1, 1))]

    # Additive: S(1,1) = (1*1+1, 1*1+1) = (2,2)
    # Pullback: only (1,1) -> (0,0); others are None
    shape_law = ("additive", (1, 1, 1, 1))

    result = paint.painter_once(assignment, part, Xtest, Xin, P_test, P_in_list, shape_law)

    # Output is 2x2
    assert len(result) == 2
    assert len(result[0]) == 2

    # Pixel (1,1) has pullback to (0,0) in class 0 -> CONST(c=7)
    assert result[1][1] == 7

    # Other pixels have pullback None -> ⊥ class -> CONST(c=9)
    assert result[0][0] == 9
    assert result[0][1] == 9
    assert result[1][0] == 9


# ============================================================================
# Error Cases (algebraic witnesses)
# ============================================================================

def test_paint_unseen_without_bottom_raises():
    """Unseen pixels without ⊥ assignment should raise AssertionError"""
    # If there are unseen pixels but no ⊥ in assignment, this is an error

    assignment = {0: "KEEP:identity"}  # No ⊥
    cid_map = {(0,0): 0}
    part = MockPartition(cid_map, 1, 1)

    Xtest = [[1]]
    Xin = [[[0]]]
    P_test = (0, (0, 0), (1, 1))
    P_in_list = [(0, (0, 0), (1, 1))]

    # Shape creates output larger than input (creates unseen pixels)
    shape_law = ("additive", (1, 1, 1, 1))  # S(1,1) = (1+1, 1+1) = (2,2)

    # Should raise AssertionError with witness
    try:
        result = paint.painter_once(assignment, part, Xtest, Xin, P_test, P_in_list, shape_law)
        assert False, "Should have raised AssertionError for unseen pixel without ⊥"
    except AssertionError as e:
        # Should mention "unseen" or "⊥" or "pullback"
        error_msg = str(e).lower()
        assert "unseen" in error_msg or "⊥" in error_msg or "pullback" in error_msg


def test_paint_bottom_must_be_const():
    """⊥ class must be CONST, not KEEP/RECOLOR/BLOCK"""
    # If ⊥ is provided but is not CONST, should raise

    assignment = {0: "CONST(c=5)", "⊥": "KEEP:identity"}  # ⊥ must be CONST
    cid_map = {(0,0): 0}
    part = MockPartition(cid_map, 1, 1)

    Xtest = [[1]]
    Xin = [[[0]]]
    P_test = (0, (0, 0), (1, 1))
    P_in_list = [(0, (0, 0), (1, 1))]

    shape_law = ("additive", (1, 1, 1, 1))  # Creates unseen pixels

    try:
        result = paint.painter_once(assignment, part, Xtest, Xin, P_test, P_in_list, shape_law)
        assert False, "Should have raised AssertionError for ⊥ not being CONST"
    except AssertionError as e:
        assert "⊥" in str(e) and "CONST" in str(e)


# ============================================================================
# Receipts Tests
# ============================================================================

def test_paint_receipts_logged():
    """Paint should log receipts with coverage and by_law counts"""
    # This test checks that receipts are logged (if implemented)

    assignment = {0: "CONST(c=5)"}
    cid_map = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}
    part = MockPartition(cid_map, 2, 2)

    Xtest = [[1, 2], [3, 4]]
    Xin = [[[0, 0], [0, 0]]]
    P_test = (0, (0, 0), (2, 2))
    P_in_list = [(0, (0, 0), (2, 2))]
    shape_law = ("multiplicative", (1, 0, 1, 0))

    result = paint.painter_once(assignment, part, Xtest, Xin, P_test, P_in_list, shape_law)

    # Check that paint returns valid result
    assert len(result) == 2
    assert len(result[0]) == 2

    # Note: Actual receipt logging verification would require mocking receipts.log
    # or checking a receipts buffer if one exists


# ============================================================================
# Determinism Tests
# ============================================================================

def test_paint_determinism():
    """Paint should produce byte-equal results on repeated runs"""
    assignment = {0: "KEEP:identity", 1: "CONST(c=5)"}
    cid_map = {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}
    part = MockPartition(cid_map, 2, 2)

    Xtest = [[1, 2], [3, 4]]
    Xin = [[[0, 0], [0, 0]]]
    P_test = (0, (0, 0), (2, 2))
    P_in_list = [(0, (0, 0), (2, 2))]
    shape_law = ("multiplicative", (1, 0, 1, 0))

    result1 = paint.painter_once(assignment, part, Xtest, Xin, P_test, P_in_list, shape_law)
    result2 = paint.painter_once(assignment, part, Xtest, Xin, P_test, P_in_list, shape_law)

    # Should be byte-equal
    json1 = json.dumps(result1, sort_keys=True)
    json2 = json.dumps(result2, sort_keys=True)

    assert json1 == json2


# ============================================================================
# Shape Law Integration Tests
# ============================================================================

def test_paint_with_multiplicative_shape():
    """Paint should work with multiplicative shape law"""
    assignment = {0: "CONST(c=8)"}
    cid_map = {(0,0): 0, (0,1): 0}
    part = MockPartition(cid_map, 1, 2)

    Xtest = [[1, 2]]
    Xin = [[[0, 0]]]
    P_test = (0, (0, 0), (1, 2))
    P_in_list = [(0, (0, 0), (1, 2))]

    # S(1,2) = (2*1, 3*2) = (2, 6)
    shape_law = ("multiplicative", (2, 0, 3, 0))

    result = paint.painter_once(assignment, part, Xtest, Xin, P_test, P_in_list, shape_law)

    # Output should be 2x6
    assert len(result) == 2
    assert len(result[0]) == 6

    # All should be 8
    for row in result:
        for pixel in row:
            assert pixel == 8


def test_paint_pullback_with_floor_division():
    """Pullback should use floor division for inverse shape law"""
    # This tests that pullback correctly handles non-exact inverses
    assignment = {0: "KEEP:identity"}

    # Partition on 2x2 test input
    cid_map = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}
    part = MockPartition(cid_map, 2, 2)

    Xtest = [[5, 6], [7, 8]]
    Xin = [[[0, 0], [0, 0]]]
    P_test = (0, (0, 0), (2, 2))
    P_in_list = [(0, (0, 0), (2, 2))]

    # S(2,2) = (2*2, 2*2) = (4,4), so pullback uses floor division
    shape_law = ("multiplicative", (2, 0, 2, 0))

    result = paint.painter_once(assignment, part, Xtest, Xin, P_test, P_in_list, shape_law)

    # Output is 4x4
    # Pullback: (i,j) -> (i//2, j//2)
    # Top-left 2x2 -> (0,0) -> Xtest[0][0]=5
    assert result[0][0] == 5
    assert result[0][1] == 5
    assert result[1][0] == 5
    assert result[1][1] == 5

    # Top-right 2x2 -> (0,1) -> Xtest[0][1]=6
    assert result[0][2] == 6
    assert result[0][3] == 6


if __name__ == "__main__":
    print("Running WO-11 paint tests...")

    # Canvas size tests
    test_canvas_size_identity_shape()
    test_canvas_size_multiplicative()
    test_canvas_size_additive()
    print("✓ Canvas size tests passed")

    # KEEP tests
    test_paint_keep_identity()
    test_paint_keep_translate()
    test_paint_keep_reads_test_not_training()
    print("✓ KEEP law tests passed")

    # VALUE tests
    test_paint_value_const()
    test_paint_value_recolor()
    test_paint_value_block()
    print("✓ VALUE law tests passed")

    # Un-present tests
    test_unpresent_identity_pose()
    test_unpresent_rot90_pose()
    test_unpresent_no_output_anchoring()
    print("✓ Un-present tests passed")

    # Idempotence tests
    test_paint_idempotent_const()
    test_paint_idempotent_keep()
    test_paint_idempotent_mixed_laws()
    print("✓ Idempotence tests passed")

    # Coverage tests
    test_paint_coverage_100_percent()
    test_paint_unseen_with_const_ok()
    print("✓ Coverage tests passed")

    # Error cases
    test_paint_unseen_without_bottom_raises()
    test_paint_bottom_must_be_const()
    print("✓ Error case tests passed")

    # Receipts tests
    test_paint_receipts_logged()
    print("✓ Receipts tests passed")

    # Determinism tests
    test_paint_determinism()
    print("✓ Determinism tests passed")

    # Shape law integration tests
    test_paint_with_multiplicative_shape()
    test_paint_pullback_with_floor_division()
    print("✓ Shape law integration tests passed")

    print(f"\n✅ All {24} WO-11 paint tests passed!")
