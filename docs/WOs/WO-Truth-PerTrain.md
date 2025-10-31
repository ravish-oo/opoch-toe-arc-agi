### WO-Truth-PerTrain — Enforce single-valued **per training**, not cross-training

**Change only the PT contradiction test and receipts; nothing else.**

1. **Fix the check**
   In `build_truth_partition(...)`, when examining a class `cid`:

```python
# For each training i, gather the set of colors this class induces in OUT_i
colors_by_train = {}
for orig_i, Y in sorted(youts_with_ids, key=lambda t: t[0]):
    S = set()
    for (r,c) in pixels_of_class_mapped_into_out_i:  # via TEST→OUT for that i, skip OOB
        S.add(Y[r][c])
    if len(S) > 1:
        # THIS is a true contradiction: multi-valued in a single training.
        # Trigger splitter order input_color ≺ sview_image ≺ parity.
        mark_contradiction(cid, train_idx=orig_i, colors=S, witness=(r,c))
        split_with_fixed_predicate_order(...)
        continue PT loop
    colors_by_train[orig_i] = list(S)  # [] or [x]
# If we got here with no per-train len(S)>1, the class is single-valued in every training ⇒ accept.
```

Do **not** reject because `len(⋃_i S_i) > 1`. That “across trainings” difference is for **Law+Sieve** to reconcile (KEEP/CONST/RECOLOR/BLOCK), exactly as in the anchors.

2. **Receipts**
   Replace the ambiguous `colors_seen` with a **per-train map**:

```json
"pt_last_check": {
  "cid": 19,
  "colors_by_train": { "0":[0], "1":[0], "2":[0], "3":[4], "4":[] },
  "note": "no per-train multi-valued class; accepted by Truth"
}
```

If you *do* split because a single training had `{a,b}` for that class, keep your existing `tried` list and include:

```json
"pt_last_contradiction": {
  "cid": 7,
  "train_idx": 2,
  "colors_in_train": [1,4],
  "witness": [r,c],
  "tried": [ ... {"pred":"band_h:[2..6]","parts":2,"class_hits":17}, ... ]
}
```

This makes the test exactly what the spec says and keeps debugging = algebra (you see *which* train forced a split and what we tried).

3. **Keep “OOB means no evidence”**
   Your current “skip OOB reads” behavior is correct — don’t turn an unobserved training into a contradiction.

### Expected effect after this patch

* The vast majority of the current **truth_fail_atom** will now **pass Truth**, because per-train sets are singletons. They will finally reach **Law + Sieve**.
* Your next sweep should show non-zero in `truth_pass_missing_law` and (likely) your **first `full_pass`**, because many periodic / copy-shift tasks are already covered by your KEEP/RECOLOR/BLOCK/CONST reducers.

