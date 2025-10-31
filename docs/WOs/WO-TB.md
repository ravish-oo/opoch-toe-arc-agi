Receipts-first **WO-TB (Truth Bands)** that extends the Paige–Tarjan (PT) predicate basis with **input-only**, fully provable separators: **walls, inter-wall bands, and border shells**. It strictly respects the anchors (input-legal predicates only; fixed separator order; no minted differences) and keeps debugging = algebra (receipts show exactly what was derived and what was tried).    

---

# WO-TB — Add Walls / Bands / Shells to the PT Predicate Basis

## Purpose

Unblock **truth** for tasks where components, residue-k, and overlap are insufficient by adding **input-only** structural masks derived from the **presented test grid** (post-Π). These are proof objects (equalities/partitions induced by the grid itself), not heuristics; they slot into the existing PT splitter order under “membership_in_Sview_image,” preserving **engineering = math** (the program is the proof). 

## Files

* Modify: `src/truth.py` only (extend the predicate basis builder used by PT).
* No changes to S-views, laws, sieve, or painter.

## Anchors this WO must respect

* **Truth uses input-only separators; fixed order**: `input_color ≺ membership_in_Sview_image ≺ parity`. Bands/shells join the **membership_in_Sview_image** family (after components, residue, overlap). 
* **Do not mint differences**: predicates must be computed **only** from the presented test grid. 
* **Determinism**: stable enumeration; canonical receipts; sweep and determinism harness must remain green. 

---

## A) Definitions (pure input, posed TEST frame)

Let `G` be the presented test grid with shape `(H,W)`.

### 1) Walls  (provable “all-one-color” slabs)

* **Wall rows**: indices `r` where `∀ c,  G[r,c] == const_r`.
* **Wall cols**: indices `c` where `∀ r,  G[r,c] == const_c`.

### 2) Inter-wall bands  (maximal slabs between walls)

* **Horizontal bands**: closed intervals `[r0+1 .. r1−1]` between **adjacent** wall rows `r0 < r1` (and also the edge bands `[0..w0−1]` if the first wall isn’t at row 0, and `[w_last+1..H−1]` if the last wall isn’t at the bottom). Each band defines a mask:

  ```
  B^H_k = { (r,c) | r ∈ [top_k .. bot_k], 0 ≤ c < W }.
  ```
* **Vertical bands**: defined analogously between adjacent wall columns; masks `B^V_j`.

(These bands are induced partitions of Ω; they are not guesses.)

### 3) Border shells (k-rings from the border)

* **Shell k** (Manhattan distance):
  `S_k = { (r,c) | d(r,c) = k }`, where `d(r,c)=min(r, H−1−r, c, W−1−c)`.
  Enumerate `k = 0,1,...` until `S_k` empty.

All masks are **input-only** and deterministic.

---

## B) Deterministic construction & order

Extend your existing `build_pt_predicates(G_test, components, residue_meta, overlap_from_sviews)` to append these masks in **this exact order** (inside the “sview_image” bucket):

1. **components** (already present)
2. **residue_row / residue_col** (already present)
3. **overlap** (already present, your K-cap and order kept deterministic)
4. **bands** (new): first **horizontal** bands in increasing `(top,bot)`, then **vertical** bands in increasing `(left,right)`.
5. **shells** (new): `k=0,1,2,...` in ascending `k`.

Predicate key naming (used in receipts and `tried`):

* component: `component:<color>:<id>`
* residue: `residue_row:p=<p>,r=<r>` / `residue_col:p=<p>,r=<r>`
* overlap: `overlap:di=<±d_i>,dj=<±d_j>`
* **band (new)**:

  * horizontal: `band_h:[<top>..<bot>]`
  * vertical:   `band_v:[<left>..<right>]`
* **shell (new)**: `shell:k=<k>`

> No caps required here: all sets are O(H+W) masks; enumeration remains small and deterministic.

---

## C) PT integration (split logic unchanged)

Keep the fixed separator order inside PT:

```
input_color  ≺  membership_in_Sview_image (components → residue → overlap → bands → shells)  ≺  parity
```

On contradiction for class `cid`, try predicates **in this order**, taking the **first** that yields ≥2 non-empty parts. This preserves the “no minted differences” contract and aligns with the anchors. 

---

## D) Receipts (so debugging stays algebra)

Extend the `truth` section with:

* **Counts** (extend existing field):

  ```json
  "pt_predicate_counts": {
    "components": X,
    "residue_row": Y,
    "residue_col": Z,
    "overlap": K,
    "bands_h": BH,
    "bands_v": BV,
    "shells": SH
  }
  ```

* **On contradiction** (extend your existing `pt_last_contradiction.tried` list to include band/shell entries that were attempted):

  ```
  {"pred":"band_h:[2..6]","parts":2,"class_hits":17}
  {"pred":"band_v:[4..7]","parts":1,"class_hits":9}
  {"pred":"shell:k=3","parts":2,"class_hits":5}
  ```

  (You already log `parts` and `class_hits`. Add these new families in the same format, in the tried order.)

* (Optional, ARC_SELF_CHECK=1) **First three masks** per family for quick review:

  ```json
  "pt_predicate_samples": {
    "band_h_first3": ["[2..4]","[9..12]","[15..16]"],
    "band_v_first3": ["[0..3]","[8..9]","[10..14]"],
    "shell_first3":  ["k=0","k=1","k=2"]
  }
  ```

All receipts remain canonical and deterministic (sorted keys, fixed list orders). 

---

## E) Built-in self-checks (ARC_SELF_CHECK=1)

Add two tiny synthetic tests inside `truth.py`’s self-check helper (no new files):

1. **Wall/band split**
   Construct a grid with an all-color wall row at `r=5` and contradictory colors across that wall for a class. Verify:

   * `pt_predicate_counts.bands_h > 0`
   * First contradiction uses a `band_h:[0..4]` or `band_h:[6..H−1]` split (recorded in `tried` with `parts≥2`).

2. **Shell split**
   Construct a grid where a contradictory class’s pixels lie on different shells; verify:

   * `pt_predicate_counts.shells > 0`
   * First contradiction uses a `shell:k=…` entry with `parts≥2`.

If a check fails, log the first failing mask under `"examples"` and raise `AssertionError("TB self-check failed: <case>")`.

---

## F) Acceptance (for implementer)

* **Determinism**: `--determinism` remains OK (hash unchanged under train order reversal).
* **Receipts**: `truth.pt_predicate_counts` now includes `bands_h`, `bands_v`, `shells`; when PT fails, `tried` includes the new masks in the right order.
* **Sweep effect**: re-run your 100-task sweep:

  * `truth_fail_nonatom` drops significantly,
  * some tasks move to `truth_pass_*` (expect `truth_pass_missing_law` to appear; **full_pass** may start non-zero if laws already suffice).

Paste:

* One failing task showing a `band_*` or `shell` entry in `tried` with `parts≥2`,
* The new sweep summary (before/after WO-TB).

---

## G) Why this WO adheres to the anchors

* **Math-first, no minted differences**: predicates are derived solely from the posed TEST grid. They are lawful S-view images used as PT separators (membership masks), never output-dependent. 
* **Fixed PT order**: bands/shells join the existing “membership_in_Sview_image” bucket *after* components/residue/overlap, before parity; separator order stays frozen, satisfying the locks. 
* **Receipts-tight**: counts and `tried` list show exactly what was available and what was attempted on a contradiction; failures reduce to a single mask witness (algebra, not vibes). 
* **Invariants intact**: no change to I-1..I-12 contracts; determinism and hash stability preserved. 

---

## H) Minimal code sketch (for clarity; implement exactly)

Inside `truth.py`:

```python
def build_pt_predicates(G, components, residue_meta, overlap_masks):
    H, W = len(G), len(G[0])
    preds = []

    # (existing) components, residue, overlap...
    preds.extend(component_predicates(components))
    preds.extend(residue_predicates(G, residue_meta))
    preds.extend(overlap_masks)  # already selected deterministically

    # --- new: walls & bands ---
    wall_rows = [r for r in range(H) if all(G[r][c] == G[r][0] for c in range(W))]
    wall_cols = [c for c in range(W) if all(G[r][c] == G[0][c] for r in range(H))]
    wall_rows.sort(); wall_cols.sort()

    # horizontal bands
    row_cuts = [-1] + wall_rows + [H]
    for a,b in zip(row_cuts[:-1], row_cuts[1:]):
        top, bot = a+1, b-1
        if top <= bot:
            mask = [(r,c) for r in range(top, bot+1) for c in range(W)]
            preds.append(("band_h", f"[{top}..{bot}]", mask))

    # vertical bands
    col_cuts = [-1] + wall_cols + [W]
    for a,b in zip(col_cuts[:-1], col_cuts[1:]):
        left, right = a+1, b-1
        if left <= right:
            mask = [(r,c) for r in range(H) for c in range(left, right+1)]
            preds.append(("band_v", f"[{left}..{right}]", mask))

    # --- new: shells ---
    for k in range(max(H,W)):
        shell = [(r,c) for r in range(H) for c in range(W)
                 if min(r, H-1-r, c, W-1-c) == k]
        if not shell: break
        preds.append(("shell", f"k={k}", shell))

    return preds
```

PT loop remains unchanged; you already record `tried` with `(pred, parts, class_hits)`.

---

### Closing note

This WO adds one small, **systemic** predicate family anchored in the input geometry (walls/bands/shells). It is provable, deterministic, and receipts-visible. It should convert a meaningful slice of your current `truth_fail_*` into `truth_pass_*` so your existing law engine can begin selecting and painting — all with **zero** hit-and-trial. 
