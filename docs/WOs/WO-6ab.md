Got it. Here’s a surgical **fix work order** that closes the PT split failure while keeping debugging = algebra.

# WO-06a — PT Predicate Basis & Algebraic Diagnostics

## Purpose

Give Paige–Tarjan lawful, effective predicates (components + residue classes), in fixed order, and log exactly why a contradiction didn’t split—so the next bug is one pixel or one mask away.

## Files

* `src/truth.py` (modify)
* `src/sviews.py` (read-only; you may expose residue gcd meta)
* `src/components.py` (read-only)

## Hard rules

* No TODOs; deterministic only.
* Do **not** scan the repo; change only `truth.py`.
* Keep split order frozen: `input_color ≺ sview_image ≺ parity`.
* Receipts must reflect **actual** objects and attempted splits.

---

## A) Add a small, lawful PT predicate basis

In `truth.py`, add:

```python
def build_pt_predicates(G_test, components, residue_meta):
    H, W = len(G_test), len(G_test[0])
    preds = []

    # A) component masks – one predicate per component
    for c in components:
        preds.append(("component", f"{c.color}:{c.comp_id}", c.mask))

    # B) residue masks – gcd row/col classes if gcd > 1
    p_row = int(residue_meta.get("row_gcd", 1))
    if p_row > 1:
        masks = [[] for _ in range(p_row)]
        for i in range(H):
            for j in range(W):
                masks[j % p_row].append((i, j))
        for r, m in enumerate(masks):
            preds.append(("residue_row", f"p={p_row},r={r}", m))

    p_col = int(residue_meta.get("col_gcd", 1))
    if p_col > 1:
        masks = [[] for _ in range(p_col)]
        for i in range(H):
            for j in range(W):
                masks[i % p_col].append((i, j))
        for r, m in enumerate(masks):
            preds.append(("residue_col", f"p={p_col},r={r}", m))

    return preds
```

**Notes**

* `residue_meta` can be exported from your s-views build (row/col gcd). If not available, recompute gcd on `G_test` here.

---

## B) Use the basis inside PT, and log attempts

In `build_truth_partition(...)`:

1. Build predicates once:

```python
preds = build_pt_predicates(G_test_presented, components, residue_meta)
pt_predicate_counts = {
    "components": sum(1 for k,_,_ in preds if k=="component"),
    "residue_row": sum(1 for k,_,_ in preds if k=="residue_row"),
    "residue_col": sum(1 for k,_,_ in preds if k=="residue_col"),
}
```

2. When a class `cid` is contradictory, try splits in this order:

* **input_color**: split the class by `G_test_presented[i,j]`.
* **sview_image**: iterate `preds` in listed order; for each mask `M`, split by membership `1_{(i,j)∈M}`; **choose the first** that yields ≥2 non-empty parts.
* **parity**: split by `(i+j) % 2`.

3. If none splits, build a **diagnostic trail** for receipts:

```python
tried = []
tried.append({"pred":"input_color","parts": parts_color})
for k,name,mask in preds:
    parts = num_parts_for_class(mask, cid_pixels)  # 1 or 2
    hits  = sum(1 for p in cid_pixels if p in_mask(mask))
    tried.append({"pred": f"{k}:{name}", "parts": parts, "class_hits": hits})
tried.append({"pred":"parity","parts": parts_parity})
```

Then raise with the same error you saw, but **also** log the trail (below).

---

## C) Receipts: algebraic diagnostics

In the `"truth"` section payload, add:

* `pt_predicate_counts` (as above)
* `pt_last_contradiction` on failure:

```json
{
  "cid": <int>,
  "colors_seen": [a,b,...],
  "witness": [r,c],
  "tried": [
    {"pred":"input_color","parts":1},
    {"pred":"component:1:7","parts":1,"class_hits":42},
    {"pred":"residue_row:p=3,r=1","parts":1,"class_hits":42},
    {"pred":"parity","parts":1}
  ]
}
```

Log the `"truth"` receipt **before** raising, then raise.

---

## D) Acceptance (implementer)

* `build_pt_predicates` exists and returns component + residue masks only.
* PT uses them in fixed order after `input_color`, before `parity`.
* On the failing task `00d62c1b`:

  * `truth.receipts.pt_predicate_counts` shows non-zero entries,
  * If still failing, `pt_last_contradiction.tried` lists at least 3 attempted predicates with `parts` and `class_hits`.
* Deterministic: repeated runs produce identical `"truth"` JSON.

Paste the updated `"truth"` receipt for `00d62c1b`.

---

## E) Reviewer tests (after implementation)

* **Contradictory uniform-color class**: a crafted class uniform in input color but in two components; assert PT splits by `component:*` first.
* **Residue split**: rows with gcd=3; contradictory class; assert PT uses `residue_row` masks to split.
* **Diagnostics**: force a failure where no predicates apply (tiny degenerate toy); assert `pt_last_contradiction.tried` is present with the attempted list.

---

## Prohibited

* Throwing the entire 128 S-views at PT (blows up, non-deterministic usefulness).
* Using outputs as predicates (still forbidden).
* Silent failure: **must** log `pt_last_contradiction` before raising.

---

## Why this fixes your current error

Class 18 was contradictory but **input_color** didn’t split. With component and residue masks as PT predicates, at least one will separate it; if not, the receipts will tell you exactly which masks were tried and why none worked (e.g., class entirely inside one component and one residue)—then you have a concrete next algebraic step.
==========
need some clarifications: Section D.7 (PT splitter order): - Spec says: "membership_in_Sview_image (includes component masks, residue-k images, overlap images)" - WO does: Component masks + residue masks - ⚠️ Missing overlap images (but probably not needed for 00d62c1b) ⚠️ Minor Gap: The spec mentions overlap translation images (02-locks D.7), but WO doesn't include them. However: - For most tasks, components + residue should be sufficient - Can add overlap images later if receipts show they're needed - Not blocking for this fix
===========
You’re right: the anchors say PT’s second splitter bucket is “membership_in_Sview_image (includes component masks, residue-k images, overlap images).” We added components + residue first (enough for 00d62c1b), but to be spec-tight we should also include a **small, deterministic basis of overlap-translation images**.  

Here’s a tiny, safe add-on you can hand to Claude that keeps debugging = algebra and won’t blow up predicate count.

# WO-06b — Add Overlap-Translation Predicate Masks to PT

## Goal

Include a **bounded, deterministic** set of overlap-translation images as PT predicates, after components and residue, exactly as the anchors specify. Keep receipts showing how many were available and which were tried on a failed class. 

## Files

* Modify `truth.py` only (extend the predicate basis builder you just added in WO-06a).

## What to admit (minimal, deterministic)

* Source of candidates: use **already-admitted overlap translations** from S-views (not all 128 views; only those whose kind is `translate` and were proven on the test grid). This keeps us aligned with S-view proofs (no heuristics). 
* Rank candidates by **domain size** (larger overlap first), then by `( |di|+|dj|, di, dj )`. Keep the **top K = 16** masks to prevent explosion.
* For each selected Δ=(di,dj), define the **image mask** (I_\Delta={x \in \Omega \cap (\Omega{-}\Delta)}), i.e., the set of test pixels that have a valid translated partner. (You already proved equality on this domain in S-views; here we only use membership.) 

## Predicate order (unchanged)

`input_color ≺ sview_image (components → residue → overlap) ≺ parity`. This preserves the fixed splitter order required by locks and invariants.  

## Receipts (extend what you added in WO-06a)

* Increment `pt_predicate_counts` with `"overlap": <num_selected>`.
* In `pt_last_contradiction.tried`, include entries like:

  ```
  {"pred":"overlap:di=+2,dj=0","parts":1,"class_hits":42}
  ```

  so a failure is still one algebraic line to inspect.

## Acceptance (implementer)

* `build_pt_predicates(...)` now returns **components + residue + ≤16 overlap** masks, in that order.
* PT uses them in the fixed splitter order; receipts show `pt_predicate_counts.overlap <= 16`.
* Deterministic: two runs match byte-for-byte.

## Why this is correct

* The anchors explicitly list overlap translations among PT’s legal input-only separators; we now include them, but **only** those already proven as S-views (no new guessing), and with a small cap to avoid combinatorics.  
* This doesn’t change your current fix trajectory (00d62c1b likely splits on components or residue), it just makes the PT basis fully compliant with the spec and gives you one more lawful separator when needed.

If a class still says “contradiction but no split,” the receipt will now list **which** component/residue/overlap masks were tried and how many class pixels each hit. That keeps debugging purely algebraic, per our invariants for truth single-valuedness and fixed splitter order.  
