"""
Truth Partition Q: Must-link + Paige-Tarjan (WO-06).

Computes coarsest partition of presented test input via:
  1. Must-link closure (S-views + component folds)
  2. Cannot-link refinement (PT with fixed predicate order)

All operations deterministic; debugging = algebra.
"""

import os
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict

import morphisms
import receipts

# Type aliases
Coord = Tuple[int, int]
IntGrid = List[List[int]]


# ============================================================================
# PARTITION CLASS
# ============================================================================


class Partition:
    """
    Partition of test input pixels into equivalence classes.

    Attributes:
        H: Grid height
        W: Grid width
        cid_of: Class id per pixel (row-major, length H*W)
    """

    def __init__(self, H: int, W: int, cid_of: List[int]):
        self.H = H
        self.W = W
        self.cid_of = cid_of

    def classes(self) -> List[List[Coord]]:
        """
        Return coordinates per class in deterministic order.

        Returns:
            List of classes (each class is list of coords)
            Ordered by class id (0..k-1)
        """
        class_map = defaultdict(list)
        for idx, cid in enumerate(self.cid_of):
            i = idx // self.W
            j = idx % self.W
            class_map[cid].append((i, j))

        # Return in class id order
        max_cid = max(self.cid_of) if self.cid_of else -1
        result = []
        for cid in range(max_cid + 1):
            if cid in class_map:
                result.append(class_map[cid])
        return result


# ============================================================================
# UNION-FIND (for must-link)
# ============================================================================


class UnionFind:
    """Union-find with path compression and union-by-rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        Unite sets containing x and y.

        Returns:
            True if they were in different sets (edge applied)
        """
        px, py = self.find(x), self.find(y)
        if px == py:
            return False

        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def get_classes(self) -> List[int]:
        """Return class id per element (normalized to 0..k-1)."""
        # Normalize: map each root to 0..k-1
        roots = sorted(set(self.find(i) for i in range(len(self.parent))))
        root_to_cid = {r: i for i, r in enumerate(roots)}
        return [root_to_cid[self.find(i)] for i in range(len(self.parent))]


# ============================================================================
# TEST→OUT CONJUGATION (uses morphisms from WO-01)
# ============================================================================


def test_to_out(
    x: Coord,
    P_test: Tuple,
    P_out: Tuple
) -> Optional[Coord]:
    """
    Map test pixel to output pixel via TEST→OUT conjugation.

    TEST→OUT: pose_inv(test) → anchor_inv(test) → pose_fwd(out)

    Args:
        x: Test pixel coord
        P_test: Test frame (op, anchor, shape)
        P_out: Output frame (op, (0,0), shape) - outputs not anchored

    Returns:
        Output coord or None if undefined
    """
    op_test, anchor_test, shape_test = P_test
    op_out, _, shape_out = P_out

    # Inverse pose in test frame
    x1 = morphisms.pose_inv(x, op_test, shape_test)
    if x1 is None:
        return None

    # Inverse anchor in test frame
    x2 = morphisms.anchor_inv(x1, anchor_test)

    # Forward pose in output frame
    x3 = morphisms.pose_fwd(x2, op_out, shape_out)

    return x3


# ============================================================================
# CHECK SINGLE-VALUED (verification after PT)
# ============================================================================


def check_single_valued(
    part: Partition,
    frames: Dict[str, Any],
    train_outputs_with_ids: List[Tuple[int, IntGrid]]
) -> Tuple[bool, Optional[Dict]]:
    """
    Verify every class has ≤1 output color across all trainings.

    WO-ND2 fix: Accept outputs paired with original train indices.
    Frame indexing fix: Use dict-keyed frames.

    Uses TEST→OUT conjugation; skips OOB mappings.

    Args:
        part: Final partition
        frames: Dict with P_test and P_out dict (keyed by orig_i)
        train_outputs_with_ids: List of (orig_train_idx, posed_output) tuples

    Returns:
        (ok, witness) where witness has {cid, train_idx, coord_out, colors_seen}
        for first contradiction, else None
    """
    P_test = frames["P_test"]
    P_out_by_id = frames["P_out"]

    classes_list = part.classes()

    for cid, coords in enumerate(classes_list):
        colors_seen = set()

        # WO-ND2: Iterate by sorted original train indices
        for orig_train_idx, Y_i in sorted(train_outputs_with_ids, key=lambda t: t[0]):
            P_out = P_out_by_id[orig_train_idx]
            H_out = len(Y_i)
            W_out = len(Y_i[0]) if H_out > 0 else 0

            for x in coords:
                coord_out = test_to_out(x, P_test, P_out)
                if coord_out is None:
                    continue

                r, c = coord_out
                # Check bounds
                if 0 <= r < H_out and 0 <= c < W_out:
                    colors_seen.add(Y_i[r][c])

        # Check single-valued
        if len(colors_seen) > 1:
            witness = {
                "cid": cid,
                "train_idx": -1,  # multiple trainings involved
                "coord_out": None,
                "colors_seen": sorted(colors_seen)
            }
            return (False, witness)

    return (True, None)


# ============================================================================
# PAIGE-TARJAN (cannot-link with fixed predicate order)
# ============================================================================


def build_pt_predicates(
    G_test: IntGrid,
    sviews: List,
    components: List,
    residue_meta: Dict[str, int]
) -> Tuple[List[Tuple[str, str, List[Coord]]], Dict[str, int]]:
    """
    Build small, lawful PT predicate basis: components + residue + overlap.

    Args:
        G_test: Presented test input
        sviews: S-views list (to extract overlap translations)
        components: Components list
        residue_meta: {"row_gcd": int, "col_gcd": int}

    Returns:
        (predicates, counts) where:
            predicates: List of (kind, name, mask) tuples
            counts: {"components": int, "residue_row": int, "residue_col": int, "overlap": int}
    """
    H = len(G_test)
    W = len(G_test[0]) if H > 0 else 0

    preds = []
    counts = {"components": 0, "residue_row": 0, "residue_col": 0, "overlap": 0}

    # A) Component masks - one predicate per component
    for c in components:
        preds.append(("component", f"{c.color}:{c.comp_id}", c.mask))
        counts["components"] += 1

    # B) Residue masks - gcd row/col classes if gcd > 1
    p_row = int(residue_meta.get("row_gcd", 1))
    if p_row > 1:
        masks = [[] for _ in range(p_row)]
        for i in range(H):
            for j in range(W):
                masks[j % p_row].append((i, j))
        for r, m in enumerate(masks):
            preds.append(("residue_row", f"p={p_row},r={r}", m))
            counts["residue_row"] += 1

    p_col = int(residue_meta.get("col_gcd", 1))
    if p_col > 1:
        masks = [[] for _ in range(p_col)]
        for i in range(H):
            for j in range(W):
                masks[i % p_col].append((i, j))
        for r, m in enumerate(masks):
            preds.append(("residue_col", f"p={p_col},r={r}", m))
            counts["residue_col"] += 1

    # C) Overlap translation images - from admitted translate S-views
    # Select top K=16 by domain size, then lexicographic
    overlap_candidates = []
    for view in sviews:
        if hasattr(view, 'kind') and view.kind == 'translate':
            # Extract translation delta from params
            params = view.params if hasattr(view, 'params') else {}
            di = params.get('di', 0)
            dj = params.get('dj', 0)

            # Fix 1: Filter identity translation (di=0, dj=0)
            if di == 0 and dj == 0:
                continue

            # Fix 2: Compute domain size by formula (not attribute)
            # |Ω_Δ| = max(0, H - |di|) × max(0, W - |dj|)
            dom_size = max(0, H - abs(di)) * max(0, W - abs(dj))

            if dom_size == 0:
                continue

            # Compute image/overlap mask (domain of translation)
            # Ω_Δ = {(i,j) | (i,j) and (i+di, j+dj) both in-bounds}
            mask = []
            for i in range(H):
                for j in range(W):
                    x = (i, j)
                    y = (i + di, j + dj)
                    # x is in overlap if both x and y are in bounds
                    if 0 <= y[0] < H and 0 <= y[1] < W:
                        mask.append(x)

            # Sanity check: mask size must equal computed domain size
            assert len(mask) == dom_size, f"overlap mask size mismatch di={di},dj={dj}: expected {dom_size}, got {len(mask)}"

            # Rank by: domain_size desc, then (|di|+|dj|, di, dj) asc
            rank_key = (-dom_size, abs(di) + abs(dj), di, dj)
            overlap_candidates.append((rank_key, di, dj, mask))

    # Sort and take top 16
    overlap_candidates.sort(key=lambda x: x[0])
    for _, di, dj, mask in overlap_candidates[:16]:
        preds.append(("overlap", f"di={di:+d},dj={dj:+d}", mask))
        counts["overlap"] += 1

    return (preds, counts)


def paige_tarjan_refine(
    G_test: IntGrid,
    initial_cid_of: List[int],
    sviews: List,
    components: List,
    residue_meta: Dict[str, int],
    frames: Dict[str, Any],
    train_outputs_with_ids: List[Tuple[int, IntGrid]]
) -> Tuple[List[int], List[Dict], Dict[str, int]]:
    """
    Refine partition via Paige-Tarjan with fixed predicate order.

    WO-ND2 fix: Iterate training outputs by sorted original indices for determinism.

    Predicate order: input_color ≺ sview_image (component/residue/overlap) ≺ parity

    Args:
        G_test: Presented test input
        initial_cid_of: Class id per pixel after must-link (row-major)
        sviews: S-views list
        components: Components list
        residue_meta: {"row_gcd": int, "col_gcd": int}
        frames: Frames dict
        train_outputs_presented: Posed training outputs

    Returns:
        (final_cid_of, splits, pt_predicate_counts) where:
            final_cid_of: Updated class id array
            splits: List of split records for receipt
            pt_predicate_counts: Dict with predicate basis sizes
    """
    H = len(G_test)
    W = len(G_test[0]) if H > 0 else 0

    P_test = frames["P_test"]
    P_out_by_id = frames["P_out"]

    # Initialize partition state (freeze UF, operate on arrays)
    cid_of = list(initial_cid_of)  # Copy so we can mutate
    next_cid = max(cid_of) + 1 if cid_of else 0  # Monotonic allocator

    # Build predicates (input-only, deterministic order)
    # Get mask-based predicates from PT basis
    mask_predicates, pt_predicate_counts = build_pt_predicates(
        G_test, sviews, components, residue_meta
    )

    # Convert to function-based predicates for PT loop
    predicates = []
    predicate_metadata = []  # For diagnostics

    # 1. input_color
    def make_input_color_pred():
        def pred(x):
            return G_test[x[0]][x[1]]
        return ("input_color", pred)

    predicates.append(make_input_color_pred())
    predicate_metadata.append(("input_color", "input_color", None))

    # 2. membership_in_Sview_image (components + residue + overlap)
    for kind, name, mask in mask_predicates:
        mask_set = set(mask)

        def make_membership_pred(mset):
            def pred(x):
                return 1 if x in mset else 0
            return pred

        predicates.append(("sview_image", make_membership_pred(mask_set)))
        predicate_metadata.append(("sview_image", f"{kind}:{name}", mask_set))

    # 3. parity
    def make_parity_pred():
        def pred(x):
            return (x[0] + x[1]) % 2
        return ("parity", pred)

    predicates.append(make_parity_pred())
    predicate_metadata.append(("parity", "parity", None))

    splits = []
    changed = True

    while changed:
        changed = False

        # Build current classes from cid_of array
        classes_map = defaultdict(list)
        for idx in range(H * W):
            i = idx // W
            j = idx % W
            classes_map[cid_of[idx]].append((i, j))

        # Scan classes in ascending order
        for cid in sorted(classes_map.keys()):
            # WO-ND2: Sort coords row-major for deterministic iteration
            coords = sorted(classes_map[cid], key=lambda p: (p[0], p[1]))

            # Check for contradiction via outputs
            # WO-ND2: Iterate by sorted original train indices (not list order)
            colors_by_train = []
            witness = None

            for orig_train_idx, Y_i in sorted(train_outputs_with_ids, key=lambda t: t[0]):
                P_out = P_out_by_id[orig_train_idx]
                H_out = len(Y_i)
                W_out = len(Y_i[0]) if H_out > 0 else 0

                # WO-ND2: Build colors_seen as list (order of first encounter)
                colors_seen_list = []
                colors_seen_set = set()
                witness_coord = None

                for x in coords:
                    coord_out = test_to_out(x, P_test, P_out)
                    if coord_out is None:
                        continue

                    r, c = coord_out
                    if 0 <= r < H_out and 0 <= c < W_out:
                        color = Y_i[r][c]
                        if color not in colors_seen_set:
                            colors_seen_list.append(color)
                            colors_seen_set.add(color)
                        if not witness_coord:
                            witness_coord = coord_out

                if colors_seen_list:
                    colors_by_train.append((orig_train_idx, colors_seen_list, witness_coord))

            # Gather all colors seen (preserve first-seen order)
            all_colors_list = []
            all_colors_set = set()
            for _, colors_list, _ in colors_by_train:
                for c in colors_list:
                    if c not in all_colors_set:
                        all_colors_list.append(c)
                        all_colors_set.add(c)

            # If ≥2 colors, we have contradiction
            if len(all_colors_list) < 2:
                continue  # No contradiction, class is fine

            # Find witness (first training with multiple colors or first conflict)
            witness_train = -1
            witness_coord_out = None
            for train_idx, colors_list, coord in colors_by_train:
                if len(colors_list) > 1:
                    witness_train = train_idx
                    witness_coord_out = coord
                    break

            if witness_train == -1 and len(colors_by_train) >= 2:
                # Different colors across trainings
                witness_train = colors_by_train[0][0]
                witness_coord_out = colors_by_train[0][2]

            # Try to split by first applicable predicate
            split_done = False
            tried_predicates = []  # For diagnostics

            for pred_idx, (pred_name, pred_fn) in enumerate(predicates):
                # Evaluate predicate on all coords in class
                pred_values = defaultdict(list)
                for x in coords:
                    val = pred_fn(x)
                    pred_values[val].append(x)

                # Record attempt for diagnostics
                _, pred_full_name, pred_mask = predicate_metadata[pred_idx]
                parts_count = len(pred_values)
                class_hits = len(coords) if pred_mask is None else sum(1 for x in coords if x in pred_mask)

                tried_predicates.append({
                    "pred": pred_full_name,
                    "parts": parts_count,
                    "class_hits": class_hits
                })

                # Check if predicate splits into ≥2 parts
                if parts_count < 2:
                    continue

                # WO-ND2: Sort parts by smallest row-major pixel in each bucket
                parts_with_min = []
                for val, part_coords in pred_values.items():
                    min_pixel = min(part_coords, key=lambda p: (p[0], p[1]))
                    parts_with_min.append((min_pixel, val, part_coords))

                # Sort by smallest pixel row-major
                parts_with_min.sort(key=lambda t: (t[0][0], t[0][1]))

                # Assign cids: first bucket keeps old cid, rest get new cids
                new_cids = []
                for part_idx, (min_pixel, val, part_coords) in enumerate(parts_with_min):
                    if part_idx == 0:
                        # Reuse old cid for first part (smallest pixel)
                        new_cid = cid
                    else:
                        # Allocate new cid (monotonic)
                        new_cid = next_cid
                        next_cid += 1

                    new_cids.append(new_cid)

                    # Update cid_of array
                    for x in part_coords:
                        idx = x[0] * W + x[1]
                        cid_of[idx] = new_cid

                # Record split
                splits.append({
                    "cid": cid,
                    "predicate": pred_name,
                    "parts": len(parts_with_min),
                    "sizes": [len(t[2]) for t in parts_with_min],
                    "witness": {
                        "train_idx": witness_train,
                        "coord_out": list(witness_coord_out) if witness_coord_out else None,
                        "colors_seen": all_colors_list  # WO-ND2: Use list instead of sorted set
                    }
                })

                split_done = True
                changed = True
                break  # Stop at first predicate that splits

            if not split_done:
                # No predicate could split this contradictory class

                # WO-ND4: Build conjugation audit for witness pixel
                # Show full TEST→OUT mapping per training to verify frames and conjugation
                conjugation_audit = []
                if len(coords) > 0:
                    p_test = coords[0]  # First pixel in contradictory class (witness test pixel)

                    for orig_train_idx, Y_i in sorted(train_outputs_with_ids, key=lambda t: t[0]):
                        P_out = P_out_by_id[orig_train_idx]
                        H_out = len(Y_i)
                        W_out = len(Y_i[0]) if H_out > 0 else 0

                        # Compute TEST→OUT conjugation
                        coord_out = test_to_out(p_test, P_test, P_out)

                        # Check bounds and read yout
                        in_bounds = False
                        yout = None
                        if coord_out is not None:
                            r, c = coord_out
                            if 0 <= r < H_out and 0 <= c < W_out:
                                in_bounds = True
                                yout = Y_i[r][c]

                        # Log full trace for this training
                        conjugation_audit.append({
                            "train_idx": orig_train_idx,
                            "P_test": {
                                "op": P_test[0],
                                "anchor": list(P_test[1]),
                                "shape": list(P_test[2])
                            },
                            "P_out": {
                                "op": P_out[0],
                                "anchor": list(P_out[1]),
                                "shape": list(P_out[2])
                            },
                            "p_test": list(p_test),
                            "p_out": list(coord_out) if coord_out else None,
                            "in_bounds": in_bounds,
                            "yout": int(yout) if yout is not None else None
                        })

                # WO-ND5: Build witness class provenance
                # Show why this class is singleton: component, S-views at pixel, neighbors
                witness_provenance = {}
                if len(coords) > 0:
                    p_test = coords[0]  # Witness pixel
                    i_test, j_test = p_test

                    # 1. Class size
                    witness_provenance["class_size"] = len(coords)

                    # 2. Component containing witness pixel
                    witness_component = None
                    for comp in components:
                        if hasattr(comp, 'mask') and p_test in comp.mask:
                            witness_component = {
                                "color": comp.color if hasattr(comp, 'color') else None,
                                "id": comp.comp_id if hasattr(comp, 'comp_id') else None,
                                "size": len(comp.mask) if hasattr(comp, 'mask') else None
                            }
                            break
                    witness_provenance["component"] = witness_component

                    # 3. Admitted views at pixel (small subset for diagnosis)
                    # Include: identity, translates with |di|+|dj| ≤ 2, residue, D4-preserving
                    admitted_views_at_pixel = []

                    for view in sviews:
                        # Filter to small subset
                        kind = view.kind if hasattr(view, 'kind') else None
                        params = view.params if hasattr(view, 'params') else {}

                        include_view = False
                        if kind == "identity":
                            include_view = True
                        elif kind == "translate":
                            di = params.get('di', 0)
                            dj = params.get('dj', 0)
                            if abs(di) + abs(dj) <= 2:  # Manhattan distance ≤ 2
                                include_view = True
                        elif kind == "residue":
                            include_view = True
                        elif kind == "d4":
                            include_view = True

                        if not include_view:
                            continue

                        # Evaluate view at p_test
                        apply_fn = view.apply_fn if hasattr(view, 'apply_fn') else view.apply if hasattr(view, 'apply') else None
                        if apply_fn is None:
                            continue

                        q = apply_fn(p_test)

                        # Check if this creates a link (same color, different pixel)
                        links = 0
                        if q is not None and q != p_test:
                            qi, qj = q
                            if 0 <= qi < H and 0 <= qj < W:
                                if G_test[qi][qj] == G_test[i_test][j_test]:
                                    links = 1

                        admitted_views_at_pixel.append({
                            "kind": kind,
                            "params": params,
                            "links": links
                        })

                    witness_provenance["admitted_views_at_pixel"] = admitted_views_at_pixel

                    # 4. Equal-color neighbors (sanity check for visual isolation)
                    test_color = G_test[i_test][j_test]
                    neighbors_4 = 0
                    neighbors_8 = 0

                    # 4-neighborhood: (i±1,j), (i,j±1)
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i_test + di, j_test + dj
                        if 0 <= ni < H and 0 <= nj < W:
                            if G_test[ni][nj] == test_color:
                                neighbors_4 += 1

                    # 8-neighborhood: add diagonals
                    for di, dj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        ni, nj = i_test + di, j_test + dj
                        if 0 <= ni < H and 0 <= nj < W:
                            if G_test[ni][nj] == test_color:
                                neighbors_8 += 1

                    neighbors_8 += neighbors_4  # 8-neighborhood includes 4-neighborhood

                    witness_provenance["equal_color_neighbors"] = {
                        "4-neighborhood": neighbors_4,
                        "8-neighborhood": neighbors_8
                    }

                # Build diagnostic payload
                pt_last_contradiction = {
                    "cid": cid,
                    "colors_seen": all_colors_list,  # WO-ND2: Use list (first-seen order)
                    "witness": list(witness_coord_out) if witness_coord_out else None,
                    "tried": tried_predicates,
                    "conjugation_audit": conjugation_audit,  # WO-ND4: Full TEST→OUT trace
                    "witness_provenance": witness_provenance  # WO-ND5: Why is class singleton?
                }

                # Store in a way that build_truth_partition can access
                raise AssertionError(
                    f"PT failed: class {cid} has contradiction but no predicate splits it. "
                    f"colors_seen={all_colors_list}, witness={witness_coord_out}|||"
                    f"DIAGNOSTIC:{pt_last_contradiction}"
                )

        # End of class scan; if changed, repeat

    return (cid_of, splits, pt_predicate_counts)


# ============================================================================
# BUILD TRUTH PARTITION (main API)
# ============================================================================


def build_truth_partition(
    G_test_presented: IntGrid,
    sviews: List,
    components: List,
    residue_meta: Dict[str, int],
    frames: Dict[str, Any],
    train_outputs_with_ids: List[Tuple[int, IntGrid]]
) -> Partition:
    """
    Build coarsest truth partition Q via must-link + PT.

    WO-ND2 fix: Accept outputs paired with original train indices for deterministic iteration.

    Args:
        G_test_presented: Presented test input
        sviews: S-views from build_sviews (WO-03/04)
        components: Components from build_components (WO-05)
        residue_meta: {"row_gcd": int, "col_gcd": int} for PT predicates
        frames: Dict with P_test, P_out list
        train_outputs_with_ids: List of (orig_train_idx, posed_output) tuples

    Returns:
        Partition object with final cid_of
    """
    H = len(G_test_presented)
    W = len(G_test_presented[0]) if H > 0 else 0

    if H == 0 or W == 0:
        return Partition(0, 0, [])

    # WO-ND3 Part B: Build deterministic must-link edge set
    edge_set = set()

    # 1. Structural S-views edges
    for view in sviews:
        for i in range(H):
            for j in range(W):
                x = (i, j)
                idx_x = i * W + j

                # Get M(x)
                if hasattr(view, 'apply'):
                    y = view.apply(x)
                elif 'apply' in view:
                    y = view['apply'](x)
                else:
                    continue

                if y is not None and y != x:
                    idx_y = y[0] * W + y[1]
                    # Add edge (min, max) for determinism
                    edge_set.add((min(idx_x, idx_y), max(idx_x, idx_y)))

    # 2. Component fold edges
    from components import component_anchor_views

    comp_views = component_anchor_views(components)
    for view in comp_views:
        apply_fn = view['apply']

        for i in range(H):
            for j in range(W):
                x = (i, j)
                idx_x = i * W + j
                y = apply_fn(x)

                if y is not None and y != x:
                    idx_y = y[0] * W + y[1]
                    # Add edge (min, max) for determinism
                    edge_set.add((min(idx_x, idx_y), max(idx_x, idx_y)))

    # Sort edges deterministically
    edges = sorted(edge_set)

    # Initialize union-find and apply edges in sorted order
    uf = UnionFind(H * W)
    for u, v in edges:
        uf.union(u, v)

    # WO-ND3 Part C: Deterministic class reindex after UF
    # Group pixels by UF root
    root = [uf.find(i) for i in range(H * W)]
    groups = {}  # root -> list of pixels
    for i, r in enumerate(root):
        if r not in groups:
            groups[r] = []
        groups[r].append(i)

    # Stable representative for each group = min row-major pixel
    items = [(min(pixels), pixels) for pixels in groups.values()]
    items.sort(key=lambda t: t[0])  # ascending by min pixel

    # Build cid_of by assigning new ids in this order
    cid_of_after_mustlink = [None] * (H * W)
    for new_cid, (_, pixels) in enumerate(items):
        for i in pixels:
            cid_of_after_mustlink[i] = new_cid

    # WO-ND3 Part D: Compute receipts for determinism verification
    import hashlib

    # mustlink_edges hash
    mustlink_edges_hash = hashlib.sha256(str(edges).encode('utf-8')).hexdigest()

    # class_reindex hash (min pixel per class)
    min_pixels_per_class = [min_pix for min_pix, _ in items]
    class_reindex_hash = hashlib.sha256(str(min_pixels_per_class).encode('utf-8')).hexdigest()

    # Compute PT predicate counts before calling PT (for error receipts)
    _, pt_pred_counts_preview = build_pt_predicates(
        G_test_presented, sviews, components, residue_meta
    )

    # Phase 4: Paige-Tarjan refinement (operates on partition array)
    try:
        cid_of, splits, pt_predicate_counts = paige_tarjan_refine(
            G_test_presented,
            cid_of_after_mustlink,
            sviews,
            components,
            residue_meta,
            frames,
            train_outputs_with_ids
        )
    except AssertionError as e:
        # Parse diagnostic if present
        error_str = str(e)

        # Fix 3: Initialize pt_last_contradiction with default (always present)
        pt_last_contradiction = {
            "cid": -1,
            "colors_seen": [],
            "witness": None,
            "tried": [],
            "conjugation_audit": [],  # WO-ND4: Empty by default
            "witness_provenance": {}  # WO-ND5: Empty by default
        }

        if "|||DIAGNOSTIC:" in error_str:
            parts = error_str.split("|||DIAGNOSTIC:")
            error_msg = parts[0]
            try:
                import ast
                parsed = ast.literal_eval(parts[1])
                # Override default with parsed diagnostic
                pt_last_contradiction = parsed
            except Exception as parse_err:
                # Log parsing failure but keep default diagnostic
                error_msg = f"{parts[0]} [diagnostic parse failed: {parse_err}]"
        else:
            error_msg = error_str

        # Fix 3: Always log receipt with pt_predicate_counts and pt_last_contradiction
        receipt = {
            "splits": [],
            "final_classes": 0,
            "single_valued_ok": False,
            "mustlink_edges": {
                "count": len(edges),
                "order_hash": mustlink_edges_hash
            },
            "class_reindex": {
                "count": len(items),
                "order_hash": class_reindex_hash
            },
            "pt_predicate_counts": pt_pred_counts_preview,
            "pt_last_contradiction": pt_last_contradiction,
            "examples": {
                "case": "PT_failed",
                "detail": error_msg
            }
        }

        receipts.log("truth", receipt)
        raise AssertionError(error_msg)

    # Finalize partition
    part = Partition(H, W, cid_of)

    # Verify single-valued
    ok, witness = check_single_valued(part, frames, train_outputs_with_ids)

    # Build receipt
    receipt = {
        "splits": splits,
        "final_classes": max(cid_of) + 1 if cid_of else 0,
        "single_valued_ok": ok,
        "mustlink_edges": {
            "count": len(edges),
            "order_hash": mustlink_edges_hash
        },
        "class_reindex": {
            "count": len(items),
            "order_hash": class_reindex_hash
        },
        "pt_predicate_counts": pt_predicate_counts,
        "examples": {}
    }

    if not ok:
        receipt["examples"]["case"] = "single_valued_failed"
        receipt["examples"]["detail"] = witness
        receipts.log("truth", receipt)
        raise AssertionError(
            f"truth partition failed: class {witness['cid']} has multiple colors: "
            f"{witness['colors_seen']}"
        )

    receipts.log("truth", receipt)
    return part


# ============================================================================
# SELF-CHECK (algebraic debugging)
# ============================================================================


def _self_check_truth() -> Dict:
    """
    Verify truth partition on synthetic cases.

    Returns:
        Receipt payload for "truth" section
    """
    receipt = {
        "splits": [],
        "final_classes": 0,
        "single_valued_ok": True,
        "mustlink_sources": {"sviews": 0, "components": 0},
        "examples": {}
    }

    # ========================================================================
    # Check 1: Must-link merges reduce class count
    # ========================================================================
    # Grid with translation that merges pixels
    G1 = [
        [1, 1, 1],
        [2, 2, 2]
    ]

    # Create trivial frames
    P_test_1 = (0, (0, 0), (2, 3))
    P_out_1 = [(0, (0, 0), (2, 3))]  # Same shape

    # Create translation S-view: (i,j) -> (i, (j+1)%3)
    class MockView:
        def __init__(self):
            self.kind = "translate"
            self.params = {"di": 0, "dj": 1}

        def apply(self, x):
            i, j = x
            return (i, (j + 1) % 3)

    sviews1 = [MockView()]
    components1 = []

    # Training output (same as input for this test)
    Y1 = [G1]

    frames1 = {"P_test": P_test_1, "P_out": P_out_1}

    # Build partition (should merge due to translation)
    uf1 = UnionFind(6)
    for view in sviews1:
        for i in range(2):
            for j in range(3):
                x = (i, j)
                y = view.apply(x)
                if y is not None:
                    idx_x = i * 3 + j
                    idx_y = y[0] * 3 + y[1]
                    uf1.union(idx_x, idx_y)

    cid_of_1 = uf1.get_classes()
    num_classes_1 = len(set(cid_of_1))

    # Should have 2 classes (one per row, since each row has same color)
    if num_classes_1 != 2:
        receipt["examples"]["case"] = "mustlink"
        receipt["examples"]["detail"] = {
            "expected_classes": 2,
            "got_classes": num_classes_1
        }
        receipt["single_valued_ok"] = False
        return receipt

    # ========================================================================
    # Check 2: Contradiction splits by first predicate
    # ========================================================================
    # Grid with different input colors → should split by input_color first
    G2 = [
        [5, 6],
        [5, 6]
    ]

    P_test_2 = (0, (0, 0), (2, 2))
    P_out_2 = [(0, (0, 0), (2, 2))]

    # Training output where color-5 pixels get 1, color-6 pixels get 2
    Y2_0 = [[1, 2], [1, 2]]
    Y2 = [Y2_0]

    frames2 = {"P_test": P_test_2, "P_out": P_out_2}

    # Start with all in one class (will be contradictory)
    uf2 = UnionFind(4)
    # Force all into same class
    uf2.union(0, 1)
    uf2.union(0, 2)
    uf2.union(0, 3)

    # Should split by input_color (5 vs 6)
    cid_of_2_after_mustlink = uf2.get_classes()
    cid_of_2_final, splits2 = paige_tarjan_refine(G2, cid_of_2_after_mustlink, [], [], frames2, Y2)

    if len(splits2) == 0:
        receipt["examples"]["case"] = "PT_split_order"
        receipt["examples"]["detail"] = {
            "note": "Expected split due to contradiction",
            "splits_count": 0
        }
        receipt["single_valued_ok"] = False
        return receipt

    # Check that split happened with input_color (first predicate)
    if splits2[0]["predicate"] != "input_color":
        receipt["examples"]["case"] = "PT_split_order"
        receipt["examples"]["detail"] = {
            "expected_predicate": "input_color",
            "got": splits2[0]["predicate"]
        }
        receipt["single_valued_ok"] = False
        return receipt

    # ========================================================================
    # Check 3: OOB skip logic
    # ========================================================================
    G3 = [[7]]
    P_test_3 = (0, (0, 0), (1, 1))
    P_out_3 = [(0, (0, 0), (1, 1))]

    # Training output is empty (OOB for all pixels)
    Y3 = [[]]  # H=1, W=0 (will cause OOB)

    frames3 = {"P_test": P_test_3, "P_out": P_out_3}

    uf3 = UnionFind(1)
    # No splits should happen (no evidence from OOB)
    cid_of_3_after_mustlink = uf3.get_classes()
    cid_of_3_final, splits3 = paige_tarjan_refine(G3, cid_of_3_after_mustlink, [], [], frames3, Y3)

    # Should have no splits (OOB skipped)
    if len(splits3) > 0:
        receipt["examples"]["case"] = "OOB"
        receipt["examples"]["detail"] = {
            "note": "OOB should be skipped, no splits expected",
            "splits_count": len(splits3)
        }
        receipt["single_valued_ok"] = False
        return receipt

    # ========================================================================
    # Check 4: Determinism under train permutation
    # ========================================================================
    G4 = [[1, 2], [3, 4]]
    P_test_4 = (0, (0, 0), (2, 2))
    P_out_4a = [(0, (0, 0), (2, 2)), (0, (0, 0), (2, 2))]
    P_out_4b = [(0, (0, 0), (2, 2)), (0, (0, 0), (2, 2))]

    Y4a = [[[1, 2], [3, 4]], [[1, 2], [3, 4]]]
    Y4b = [[[1, 2], [3, 4]], [[1, 2], [3, 4]]]  # Same, reversed order

    frames4a = {"P_test": P_test_4, "P_out": P_out_4a}
    frames4b = {"P_test": P_test_4, "P_out": P_out_4b[::-1]}

    uf4a = UnionFind(4)
    uf4b = UnionFind(4)

    cid_of_4a_after_mustlink = uf4a.get_classes()
    cid_of_4b_after_mustlink = uf4b.get_classes()

    cid_of_4a, splits4a = paige_tarjan_refine(G4, cid_of_4a_after_mustlink, [], [], frames4a, Y4a)
    cid_of_4b, splits4b = paige_tarjan_refine(G4, cid_of_4b_after_mustlink, [], [], frames4b, Y4b[::-1])

    num_classes_4a = len(set(cid_of_4a))
    num_classes_4b = len(set(cid_of_4b))

    if num_classes_4a != num_classes_4b:
        receipt["examples"]["case"] = "determinism"
        receipt["examples"]["detail"] = {
            "classes_forward": num_classes_4a,
            "classes_reversed": num_classes_4b,
            "note": "Non-deterministic under train permutation"
        }
        receipt["single_valued_ok"] = False
        return receipt

    # Final receipt (all checks passed)
    receipt["final_classes"] = num_classes_1
    receipt["mustlink_sources"]["sviews"] = 3  # 3 edges in G1
    receipt["mustlink_sources"]["components"] = 0

    return receipt


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================


def init() -> None:
    """
    Run self-check (no receipt logging - caller logs after build_truth_partition).

    Raises:
        AssertionError: If any identity check fails

    Notes:
        - Called by harness, not on import
        - Assumes receipts.init() has been called
        - Runs self-check only if ARC_SELF_CHECK=1
        - Receipt logging moved to caller after build_truth_partition()
    """
    # Check if self-check should run
    if os.environ.get("ARC_SELF_CHECK") != "1":
        # Skip self-check in normal mode (fast path)
        return

    receipt = _self_check_truth()

    # Emit receipt only for self-check
    receipts.log("truth_selfcheck", receipt)

    # Assert all checks passed
    if not receipt["single_valued_ok"]:
        case = receipt["examples"].get("case", "unknown")
        detail = receipt["examples"].get("detail", {})

        if case == "mustlink":
            raise AssertionError(
                f"truth identity failed: must-link expected {detail['expected_classes']} classes, "
                f"got {detail['got_classes']}"
            )
        elif case == "PT_split_order":
            raise AssertionError(
                f"truth identity failed: PT split order incorrect, detail={detail}"
            )
        elif case == "OOB":
            raise AssertionError(
                f"truth identity failed: OOB handling wrong, detail={detail}"
            )
        elif case == "determinism":
            raise AssertionError(
                f"truth identity failed: non-deterministic under train permutation, detail={detail}"
            )
        elif case == "single_valued_failed":
            raise AssertionError(
                f"truth identity failed: class {detail['cid']} has multiple colors {detail['colors_seen']}"
            )
        else:
            raise AssertionError(
                f"truth identity failed: case={case}, detail={detail}"
            )
