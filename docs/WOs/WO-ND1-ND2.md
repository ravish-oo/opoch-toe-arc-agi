You’ve got exactly what we need: receipts pinpoint the breaking sections.

* 00576224 → **diff_section=components**
* 00d62c1b → **diff_section=truth**

That means the non-determinism is in component enumeration/IDs or in the PT refinement loop, not elsewhere. Let’s fix both with small, surgical changes and receipts that show the exact ordering keys, so debugging stays algebraic.

---

# WO-ND1 — Make `components` deterministic (and prove it in receipts)

## Symptoms to kill

* Reversed train order changes `components` section hash.
* Likely causes: dict/set iteration, unstable comp_id, unsorted masks.

## Do exactly this

1. **Stable component ordering (the only source of truth):**

   * Sort colors ascending (0..9).
   * For each color, sort components by **anchor** (row, then col).
   * Set `comp_id` = index in this sorted list (0-based).
   * Sort each **mask** row-major.
   * Do not use sets for output; use lists in that order.

2. **Deterministic BFS/UF:**

   * BFS frontier as a list; pop from **front**; push neighbors in fixed order **[(−1,0),(+1,0),(0,−1),(0,+1)]**.
   * Never iterate `dict`/`set` without sorting keys first.

3. **Receipts you must emit (so we can see order):**

   * `anchors_first5`: already present, but ensure it equals
     `[{"color":c,"id":k,"anchor":[r,c],"size":n}, ...]` (not raw tuples).
   * Add:

     ```json
     "order_hash": "<sha256 of [(color, id, anchor, size) ...]>"
     ```
   * Optional debug (ARC_SELF_CHECK=1 only):

     ```json
     "order_keys_first10": [[c,id,[r,c],n], ...]
     ```

4. **Acceptance:**
   `--determinism` on 00576224 flips from FAIL to OK. Receipts show identical `order_hash`.

---

# WO-ND2 — Make `truth` Paige–Tarjan deterministic (and prove it)

## Symptoms to kill

* Reversed train order changes `truth` section hash.

## Do exactly this (freeze the loop order)

1. **Class scan order:** ascending `cid`. Use `while cid < next_cid`, not `for cid in classes.keys()`.

2. **Contradiction detection order:**

   * For each class: iterate training pairs in ascending `i`.
   * For pixels inside the class: iterate **row-major**.
   * Build `colors_seen` as a **list** of first-seen colors in the order found; for the contradiction witness pick the **first two distinct** colors encountered. Don’t rely on set ordering.

3. **Splitter order (no ambiguity):**

   * Try `input_color`.
   * Then **predicates** in this exact fixed list:

     * All **component** masks in the order from WO-ND1: `(color,id)` ascending.
     * All **residue_row** masks in order of `p` then residue `r=0..p−1`.
     * All **residue_col** masks in order of `p` then residue `r=0..p−1`.
     * All **overlap** masks in order of **domain size desc**, then `( |di|+|dj|, di, dj )`, capped (e.g., 16).
   * Then `parity`.

4. **Split application (labeling):**

   * Keep old `cid` for the **first** non-empty bucket (choose bucket with **smallest row-major pixel**).
   * Assign new cids to remaining buckets in **bucket order** (again by smallest pixel).
   * Update `classes` and `cid_of` only; **never** touch union-find here.

5. **Receipts you must emit (so we can see order):**

   * `splits`: keep as you have it, but also add

     ```json
     "pt_predicate_counts": {"components": X,"residue_row": Y,"residue_col": Z,"overlap": K}
     ```
   * On failure, include:

     ```json
     "pt_last_contradiction": {
       "cid": c,
       "colors_seen": [...],
       "witness": [r,c],
       "tried": [
         {"pred":"input_color","parts":1},
         {"pred":"component:1:7","parts":2,"class_hits":42},
         {"pred":"residue_row:p=3,r=1","parts":1,"class_hits":42},
         {"pred":"parity","parts":1}
       ]
     }
     ```

   This makes the attempted predicate order and their effect explicit.

6. **Acceptance:**
   `--determinism` on 00d62c1b flips from FAIL to OK. If it still fails, `pt_last_contradiction.tried` gives the exact mask that failed to split (algebraic next step).

---

## Quick triage loop (run now)

1. Apply WO-ND1.

   * Run: `python -m src.harness --determinism --task-id 00576224 --train-dir data`
   * Expect **OK**. If FAIL: print `components.order_hash` for A vs B, and compare `anchors_first5`.

2. Apply WO-ND2.

   * Run: `python -m src.harness --determinism --task-id 00d62c1b --train-dir data`
   * Expect **OK**. If FAIL: check `truth.pt_predicate_counts` and `pt_last_contradiction.tried`—the first entry with `parts=1` shows the exact predicate that didn’t separate; fix its construction/order.

---

## Why receipts are enough

* For **components**, `order_hash` proves enumeration stability. If it flips, you compare the first 10 keys and see the anchor that moved.
* For **truth**, `pt_last_contradiction.tried` shows the precise predicate order and how many class pixels each mask hits; if nothing splits, the line tells you which family is powerless and why.

No hunting, no guesswork—one section at a time, one ordered list at a time.

If you want, I can also give you a 15-line snippet to compute `order_hash` and a `num_parts_for_class(mask, cid_pixels)` helper for PT.
