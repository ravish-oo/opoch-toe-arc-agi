"""
Projector Π: present / unpresent (WO-02).

Implements the idempotent projector Π = Π_pal ∘ Π_pose ∘ Π_anch
and its exact inverse U^{-1}.

All coordinate normalization happens here and nowhere else.
"""

import random
from typing import List, Tuple, Dict, Optional
from collections import Counter

import morphisms
import receipts

# Type aliases
IntGrid = List[List[int]]
PaletteMap = Dict[int, int]
Coord = Tuple[int, int]
Shape = Tuple[int, int]
Frame = Tuple[int, Tuple[int, int], Shape]  # (d4_op, (dy,dx), (H,W))


# ============================================================================
# PALETTE CANON (inputs only)
# ============================================================================


def build_palette_map(train_inputs: List[IntGrid], test_input: IntGrid) -> PaletteMap:
    """
    Build canonical palette map from all training inputs + test input.

    Args:
        train_inputs: All training input grids
        test_input: Test input grid

    Returns:
        PaletteMap: old_color -> new_color (0, 1, 2, ...)

    Order:
        Sort colors by (frequency desc, first_appearance_index asc, color_value asc)
        Map to 0, 1, 2, ... in that order
    """
    # Gather all colors with (color, first_appearance_index)
    color_first_appearance: Dict[int, int] = {}
    color_counts: Counter = Counter()
    appearance_index = 0

    # Process all grids in order: train inputs, then test input
    all_inputs = train_inputs + [test_input]

    for grid in all_inputs:
        for row in grid:
            for color in row:
                if color not in color_first_appearance:
                    color_first_appearance[color] = appearance_index
                    appearance_index += 1
                color_counts[color] += 1

    # Sort by (frequency desc, first_appearance asc, color_value asc)
    colors_sorted = sorted(
        color_first_appearance.keys(),
        key=lambda c: (-color_counts[c], color_first_appearance[c], c)
    )

    # Map to 0, 1, 2, ...
    palette_map = {old_color: new_color for new_color, old_color in enumerate(colors_sorted)}

    return palette_map


def apply_palette(grid: IntGrid, pm: PaletteMap, allow_passthrough: bool = False) -> IntGrid:
    """
    Apply palette map to a grid.

    Args:
        grid: Input grid
        pm: Palette map
        allow_passthrough: If True, unknown colors pass through unchanged (for outputs)

    Returns:
        Grid with palette applied

    Notes:
        - For inputs: all colors must be in pm (by construction)
        - For outputs: unknown colors pass through if allow_passthrough=True
    """
    result = []
    for row in grid:
        new_row = []
        for color in row:
            if color in pm:
                new_row.append(pm[color])
            elif allow_passthrough:
                new_row.append(color)
            else:
                raise ValueError(f"Color {color} not in palette map")
        result.append(new_row)
    return result


def unapply_palette(grid: IntGrid, pm_inv: PaletteMap) -> IntGrid:
    """
    Apply inverse palette map.

    Args:
        grid: Grid with palette applied
        pm_inv: Inverse palette map (new_color -> old_color)

    Returns:
        Grid with original colors

    Notes:
        - Colors not in pm_inv pass through unchanged (they were pass-through on present)
    """
    result = []
    for row in grid:
        new_row = []
        for color in row:
            if color in pm_inv:
                new_row.append(pm_inv[color])
            else:
                # Pass-through color (was not in original palette)
                new_row.append(color)
        result.append(new_row)
    return result


# ============================================================================
# D4 LEX POSE
# ============================================================================


def pose_grid(grid: IntGrid) -> Tuple[IntGrid, int]:
    """
    Find D4 lex-minimal pose.

    Args:
        grid: Input grid

    Returns:
        (posed_grid, op_id) where op_id ∈ {0..7}

    Algorithm:
        - For each D4 op (0..7), apply it and flatten row-major to tuple
        - Pick the minimal tuple
        - Tie-break by op id (0..7)
    """
    H = len(grid)
    W = len(grid[0]) if H > 0 else 0
    shape = (H, W)

    best_tuple = None
    best_op = None
    best_grid = None

    for op in range(8):
        # Apply D4 op to get posed grid
        posed = _apply_d4_to_grid(grid, op, shape)

        # Flatten row-major
        flat_tuple = tuple(color for row in posed for color in row)

        # Update best
        if best_tuple is None or flat_tuple < best_tuple or (flat_tuple == best_tuple and op < best_op):
            best_tuple = flat_tuple
            best_op = op
            best_grid = posed

    return best_grid, best_op


def _apply_d4_to_grid(grid: IntGrid, op: int, shape: Shape) -> IntGrid:
    """
    Apply D4 operation to entire grid.

    Args:
        grid: Input grid (H×W)
        op: D4 operation (0..7)
        shape: Original shape (H, W)

    Returns:
        Transformed grid
    """
    H, W = shape

    # Determine output shape after D4 op
    # Ops 0,2,4,6: preserve shape (H,W)
    # Ops 1,3,5,7: swap shape to (W,H)
    if op in [1, 3, 5, 7]:
        H_out, W_out = W, H
    else:
        H_out, W_out = H, W

    # Build output grid by querying inverse
    result = [[0 for _ in range(W_out)] for _ in range(H_out)]

    for i_out in range(H_out):
        for j_out in range(W_out):
            # Find where this output pixel came from
            i_in, j_in = morphisms.pose_inv((i_out, j_out), op, shape)
            result[i_out][j_out] = grid[i_in][j_in]

    return result


def unpose_grid(grid: IntGrid, op: int, original_shape: Shape) -> IntGrid:
    """
    Undo D4 pose operation.

    Args:
        grid: Posed grid (current shape may be swapped)
        op: D4 operation that was applied
        original_shape: Shape before pose was applied (H, W)

    Returns:
        Grid in original orientation

    Algorithm:
        - Iterate over original grid positions
        - For each original position, find where it is in the posed grid
        - Use pose_fwd to map original -> posed
    """
    H_orig, W_orig = original_shape

    # Build output grid by querying the posed grid
    result = [[0 for _ in range(W_orig)] for _ in range(H_orig)]

    for i_orig in range(H_orig):
        for j_orig in range(W_orig):
            # Find where this original pixel is in the posed grid
            i_posed, j_posed = morphisms.pose_fwd((i_orig, j_orig), op, original_shape)
            result[i_orig][j_orig] = grid[i_posed][j_posed]

    return result


# ============================================================================
# ANCHOR (inputs + test only)
# ============================================================================


def anchor_grid(grid: IntGrid) -> Tuple[IntGrid, Tuple[int, int]]:
    """
    Find top-left non-zero pixel and shift to (0,0).

    Args:
        grid: Input grid

    Returns:
        (anchored_grid, (dy, dx))

    Algorithm:
        - Find top-left non-zero pixel (min row, then min col)
        - Anchor (dy, dx) shifts that pixel to (0,0)
        - If no non-zero exists, anchor = (0,0)
    """
    H = len(grid)
    W = len(grid[0]) if H > 0 else 0

    # Find top-left non-zero
    min_i, min_j = None, None

    for i in range(H):
        for j in range(W):
            if grid[i][j] != 0:
                if min_i is None or i < min_i or (i == min_i and j < min_j):
                    min_i, min_j = i, j

    # If no non-zero, anchor at (0,0)
    if min_i is None:
        return grid, (0, 0)

    # Anchor shifts (min_i, min_j) to (0, 0)
    dy, dx = min_i, min_j

    # Apply anchor: shift all pixels by (-dy, -dx)
    result = [[grid[i][j] for j in range(W)] for i in range(H)]

    return result, (dy, dx)


def unanchor_grid(grid: IntGrid, anchor: Tuple[int, int]) -> IntGrid:
    """
    Undo anchor operation.

    Args:
        grid: Anchored grid
        anchor: (dy, dx) that was applied

    Returns:
        Grid in original position

    Notes:
        - Anchor shifts content; unanchor just returns grid (content is frame-relative)
        - The actual coordinate transform happens in morphisms.anchor_inv
    """
    # Anchor is a metadata shift; the grid content doesn't change
    # The frame records the shift; unpresent uses morphisms.anchor_inv for coordinates
    return grid


# ============================================================================
# PRESENT (Π)
# ============================================================================


def present_input(grid: IntGrid, pm: PaletteMap) -> Tuple[IntGrid, Frame]:
    """
    Present an input grid: palette → pose → anchor.

    Args:
        grid: Raw input grid
        pm: Palette map (computed once per task)

    Returns:
        (presented_grid, frame) where frame = (op, (dy,dx), (H,W))

    Order:
        1. Apply palette
        2. D4 lex pose
        3. Anchor to top-left non-zero
    """
    # Step 1: Palette
    grid_pal = apply_palette(grid, pm, allow_passthrough=False)

    # Step 2: Pose
    grid_posed, op = pose_grid(grid_pal)

    # Step 3: Anchor
    grid_anchored, (dy, dx) = anchor_grid(grid_posed)

    # Record frame
    H = len(grid_anchored)
    W = len(grid_anchored[0]) if H > 0 else 0
    frame = (op, (dy, dx), (H, W))

    return grid_anchored, frame


def present_output(grid: IntGrid, pm: PaletteMap) -> Tuple[IntGrid, Frame]:
    """
    Present an output grid: palette (with passthrough) → pose → no anchor.

    Args:
        grid: Raw output grid
        pm: Palette map (same as inputs)

    Returns:
        (presented_grid, frame) where frame = (op, (0,0), (H,W))

    Order:
        1. Apply palette (unknown colors pass through)
        2. D4 lex pose
        3. No anchor (frame anchor = (0,0))
    """
    # Step 1: Palette (allow passthrough for unknown output colors)
    grid_pal = apply_palette(grid, pm, allow_passthrough=True)

    # Step 2: Pose
    grid_posed, op = pose_grid(grid_pal)

    # No anchor for outputs
    H = len(grid_posed)
    W = len(grid_posed[0]) if H > 0 else 0
    frame = (op, (0, 0), (H, W))

    return grid_posed, frame


# ============================================================================
# UNPRESENT (U^{-1})
# ============================================================================


def unpresent_input(grid: IntGrid, frame: Frame, pm_inv: PaletteMap) -> IntGrid:
    """
    Unpresent an input grid: anchor_inv → pose_inv → palette_inv.

    Args:
        grid: Presented grid
        frame: Frame from present_input: (op, (dy,dx), (H,W))
        pm_inv: Inverse palette map (new_color -> old_color)

    Returns:
        Original raw grid (byte-equal to input)

    Order (inverse of present):
        1. Undo anchor (metadata only; use frame for coordinates)
        2. Undo pose
        3. Undo palette

    Note:
        - Frame shape (H,W) is the shape AFTER pose+anchor
        - To unpose, we need the shape BEFORE pose
        - For ops 1,3,5,7: original shape is (W,H) (swapped)
        - For ops 0,2,4,6: original shape is (H,W) (same)
    """
    op, anchor, shape_after_pose = frame

    # Determine original shape before pose
    H_after, W_after = shape_after_pose
    if op in [1, 3, 5, 7]:
        # Pose swapped dimensions
        original_shape = (W_after, H_after)
    else:
        # Pose kept dimensions
        original_shape = shape_after_pose

    # Step 1: Undo anchor (grid itself unchanged; frame records shift)
    grid_unanchored = unanchor_grid(grid, anchor)

    # Step 2: Undo pose
    grid_unposed = unpose_grid(grid_unanchored, op, original_shape)

    # Step 3: Undo palette
    grid_orig = unapply_palette(grid_unposed, pm_inv)

    return grid_orig


def unpresent_output(grid: IntGrid, frame: Frame, pm_inv: PaletteMap) -> IntGrid:
    """
    Unpresent an output grid: pose_inv → palette_inv.

    Args:
        grid: Presented grid
        frame: Frame from present_output: (op, (0,0), (H,W))
        pm_inv: Inverse palette map

    Returns:
        Original raw grid (byte-equal to input)

    Order (inverse of present):
        1. Undo pose
        2. Undo palette (pass-through colors remain unchanged)

    Note:
        - Frame shape (H,W) is the shape AFTER pose
        - To unpose, we need the shape BEFORE pose
        - For ops 1,3,5,7: original shape is (W,H) (swapped)
        - For ops 0,2,4,6: original shape is (H,W) (same)
    """
    op, anchor, shape_after_pose = frame

    # Determine original shape before pose
    H_after, W_after = shape_after_pose
    if op in [1, 3, 5, 7]:
        # Pose swapped dimensions
        original_shape = (W_after, H_after)
    else:
        # Pose kept dimensions
        original_shape = shape_after_pose

    # Step 1: Undo pose
    grid_unposed = unpose_grid(grid, op, original_shape)

    # Step 2: Undo palette
    grid_orig = unapply_palette(grid_unposed, pm_inv)

    return grid_orig


# ============================================================================
# SELF-CHECK (algebraic debugging)
# ============================================================================


def _self_check_present() -> Dict:
    """
    Verify present/unpresent identities on synthetic and random data.

    Returns:
        Receipt payload for "present" section
    """
    random.seed(1337)  # Deterministic

    receipt = {
        "palette_map_size": 0,
        "palette_first5": [],
        "d4_ops": {},
        "anchors": {},
        "round_trip_ok_inputs": True,
        "round_trip_ok_outputs": True,
        "examples": {}
    }

    # ========================================================================
    # Check 1: Palette ordering on synthetic data
    # ========================================================================
    # Create inputs with known frequency/appearance order
    # Colors: 5 (freq 3), 2 (freq 2), 7 (freq 2, appears after 2), 0 (freq 1)
    synthetic_train = [
        [[5, 2, 5], [0, 7, 5]],  # 5:3, 2:1, 0:1, 7:1
        [[2, 7, 5]]              # contributes to counts
    ]
    synthetic_test = [[5]]

    pm = build_palette_map(synthetic_train, synthetic_test)
    receipt["palette_map_size"] = len(pm)

    # Expected order: 5 (freq 3) -> 0, 2 (freq 2, first) -> 1, 7 (freq 2, second) -> 2, 0 (freq 1) -> 3
    expected_order = {5: 0, 2: 1, 7: 2, 0: 3}

    if pm != expected_order:
        receipt["examples"]["palette_order"] = {
            "case": "palette_order",
            "got": pm,
            "want": expected_order
        }
        return receipt

    # Store first 5 mappings (sorted by new_color)
    sorted_mappings = sorted(pm.items(), key=lambda x: x[1])
    receipt["palette_first5"] = sorted_mappings[:5]

    # ========================================================================
    # Check 2: Round-trip on random grids
    # ========================================================================
    pm_inv = {v: k for k, v in pm.items()}

    for idx in range(6):
        # Generate small random grid
        H = random.randint(2, 5)
        W = random.randint(2, 5)
        grid_input = [[random.choice([0, 1, 2]) for _ in range(W)] for _ in range(H)]

        # Build palette for this grid
        pm_local = build_palette_map([grid_input], grid_input)
        pm_inv_local = {v: k for k, v in pm_local.items()}

        # Test input round-trip
        presented, frame_in = present_input(grid_input, pm_local)
        recovered = unpresent_input(presented, frame_in, pm_inv_local)

        if recovered != grid_input:
            receipt["round_trip_ok_inputs"] = False
            receipt["examples"]["round_trip_input"] = {
                "case": "input",
                "index": idx,
                "op": frame_in[0],
                "anchor": frame_in[1],
                "original_shape": (H, W),
                "presented_shape": frame_in[2]
            }
            return receipt

        receipt["d4_ops"][f"in_{idx}"] = frame_in[0]
        receipt["anchors"][f"in_{idx}"] = frame_in[1]

        # Test output round-trip (with potential pass-through colors)
        grid_output = [[random.choice([0, 1, 2, 99]) for _ in range(W)] for _ in range(H)]  # 99 not in palette

        presented_out, frame_out = present_output(grid_output, pm_local)
        recovered_out = unpresent_output(presented_out, frame_out, pm_inv_local)

        if recovered_out != grid_output:
            receipt["round_trip_ok_outputs"] = False
            receipt["examples"]["round_trip_output"] = {
                "case": "output",
                "index": idx,
                "op": frame_out[0],
                "anchor": frame_out[1]
            }
            return receipt

        receipt["d4_ops"][f"out_{idx}"] = frame_out[0]

        # Verify output anchor is (0,0)
        if frame_out[1] != (0, 0):
            receipt["round_trip_ok_outputs"] = False
            receipt["examples"]["output_anchor"] = {
                "case": "output",
                "index": idx,
                "anchor": frame_out[1],
                "expected": (0, 0)
            }
            return receipt

    # ========================================================================
    # Check 3: Determinism
    # ========================================================================
    test_grid = [[1, 2], [3, 4]]
    pm_det = build_palette_map([test_grid], test_grid)

    _, op1 = pose_grid(test_grid)
    _, op2 = pose_grid(test_grid)

    if op1 != op2:
        receipt["examples"]["determinism_pose"] = {
            "case": "determinism",
            "op1": op1,
            "op2": op2
        }
        return receipt

    _, anchor1 = anchor_grid(test_grid)
    _, anchor2 = anchor_grid(test_grid)

    if anchor1 != anchor2:
        receipt["examples"]["determinism_anchor"] = {
            "case": "determinism",
            "anchor1": anchor1,
            "anchor2": anchor2
        }
        return receipt

    return receipt


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================


def init() -> None:
    """
    Run self-check and emit present receipt.

    Raises:
        AssertionError: If any identity check fails

    Notes:
        - Called by harness, not on import
        - Assumes receipts.init() has been called
    """
    receipt = _self_check_present()

    # Emit receipt
    receipts.log("present", receipt)

    # Assert all checks passed
    if "palette_order" in receipt.get("examples", {}):
        ex = receipt["examples"]["palette_order"]
        raise AssertionError(
            f"present identity failed: palette order got={ex['got']} want={ex['want']}"
        )

    if not receipt["round_trip_ok_inputs"]:
        ex = receipt["examples"].get("round_trip_input", {})
        raise AssertionError(
            f"present identity failed: input round-trip at index={ex.get('index')} "
            f"op={ex.get('op')} anchor={ex.get('anchor')}"
        )

    if not receipt["round_trip_ok_outputs"]:
        ex = receipt["examples"].get("round_trip_output", {}) or receipt["examples"].get("output_anchor", {})
        raise AssertionError(
            f"present identity failed: output round-trip at index={ex.get('index')} "
            f"op={ex.get('op')} anchor={ex.get('anchor')}"
        )

    if "determinism_pose" in receipt.get("examples", {}):
        ex = receipt["examples"]["determinism_pose"]
        raise AssertionError(
            f"present identity failed: pose not deterministic op1={ex['op1']} op2={ex['op2']}"
        )

    if "determinism_anchor" in receipt.get("examples", {}):
        ex = receipt["examples"]["determinism_anchor"]
        raise AssertionError(
            f"present identity failed: anchor not deterministic "
            f"anchor1={ex['anchor1']} anchor2={ex['anchor2']}"
        )
