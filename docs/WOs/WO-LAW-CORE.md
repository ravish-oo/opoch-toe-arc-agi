## **WO-LAW-CORE — Admit + Sieve (Receipts-First)**

**Purpose.** Convert the ~25–30 % of `truth_pass_*` tasks into `selection.exact` by finishing the law layer and the sieve, with deterministic, first-witness receipts. This is straight §2 (“Φ* is least admissible”) plus §5 (“use conjugated reads to prove/reject”).

**Files.** `src/laws/keep.py`, `src/laws/value.py`, `src/sieve.py`, `src/class_map.py`, `src/receipts.py`. No Truth or Present changes.

**A) Precompute class evidence (unchanged, emphasized).**

* From `truth.Partition`, build `class_pixels_test[cid]`.

* For each train `orig_i`, build a **stable class map** `class_map_i: out_coord → cid` using your `test_to_out` with `P_test` and `P_out_by_id[orig_i]` (already done). Record `class_hits[cid][orig_i] = |{p_out : class_map_i[p_out]=cid}|`.

* **Receipts:** `selection.class_maps`: per train, `{"orig_i": i, "H":…, "W":…, "observed": <#pixels>}` for sanity.

**B) Finish KEEP candidate enumeration and proof.**

* **Catalogue** (deterministic):
  `KEEP:identity`, `KEEP:d4:op=0..7`, `KEEP:translate(di,dj)` for |di|+|dj|≤max(Ht,Wt) sorted by (|di|+|dj|,di,dj),
  `KEEP:residue_row/col(p)` for admitted `p` from sviews,
  `KEEP:tile`, `KEEP:tile_alt_row_flip`, `KEEP:tile_alt_col_flip`, `KEEP:tile_checkerboard_flip`,
  `KEEP:block_inverse(k)` for small k∈{2,3} (deterministic set),
  `KEEP:offset(b,d)` with small |b|,|d| bounded (e.g. ≤1 or ≤2) sorted lexicographically.
  **All** are *test-frame* maps `V: (i,j)↦(i′,j′)`. **Do not** use outputs to define them.

* **Admissibility per class** (proof): for each candidate `(name, V)` and each observed `p_out ∈ class(cid)` in each train `orig_i`, compute:

  ```
  q = pose_inv(p_out, P_out_by_id[i])        # OUT→TEST
  r = V(q);  if r is None: reject (undefined on observed pixel) 
  p_in = pose_fwd( anchor_fwd(r, P_in[i].anchor), P_in[i].op )
  ok if and only if Xin[i][p_in] == Yout[i][p_out] for all observed p_out in this cid
  ```

  If any check fails, reject with a **single witness** `{cid, descriptor, train_idx, p_out, p_test, p_in, expected, got}`.

* **Receipts:** append `{"cid", "descriptor", "proof":{"trains_checked":m, "pixels_checked":N}}` to `laws.admitted`. Append rejections to `laws.keep_debug` with first-witness object as above. All lists sorted by `cid`, then by descriptor.

**C) Finish VALUE candidates and proofs.**

* **CONST(c)**: per class, per train, gather outputs over `Obs_i(cid)`; each non-empty set must be singleton; all agree to same `c`.
* **UNIQUE**, **ARGMAX**, **LOWEST_UNUSED**: compute `c_i` from `Xin[i]|class_in_i` (map test class → *input* coords via your `test_to_in`), require identical `c` across trains w/ nonzero support; proof: that every observed `p_out` has `Yout_i[p_out]==c`. Emit `laws.admitted` on success; log first conflicting train/witness on failure in `laws.value_debug`.
* **RECOLOR(π)**: build per-class mapping `cin→cout` by aligning `Xin[i][p_in]` with `Yout[i][p_out]` over `Obs_i(cid)`; reject on conflict; require π defined for all `cin` present in test `class_pixels_test[cid]`.
* **BLOCK(k)**: test-frame block-index at `p_test` via `k` relative to class anchor; map to `p_in` deterministically and check outputs. Cap k∈{2,3}. Emit proof with counts.

**D) Sieve (global exactness, receipts-first).**

* Build `candidates[cid] = admitted_keep ∪ admitted_value`. If empty → `missing_descriptor` now with the collected `keep_debug`/`value_debug` first witnesses.

* **Deterministic pruning pass:**

  * For each train `orig_i` (ascending), for each observed `p_out` in row-major, for each `law ∈ candidates[cid]` (lex order), evaluate its **test-time** colour:

    * `KEEP`: `color = Xtest[V(p_test)]` where `p_test = shape_pullback(p_out)` → if `None`, immediate “paint_failure:pullback_none” else compare with `Yout_i[p_out]`.
    * `VALUE`: `CONST(c) → c`; `RECOLOR(π) → π[Xtest[p_test]]`; `BLOCK(k)` as per proof.
    * On first mismatch for that (cid,law), drop it; record `selection.prune_log` with the same witness structure as in B/C.

* Iterate pruning until stable. If `candidates[cid]` becomes empty → `selection.status="missing_descriptor"` and include the `missing` list with first witnesses from debug logs. Else pick the **least-cost** surviving descriptor per `cid` (cost order from your spec) and set `selection.status="exact"`. Log `selection.assignment` and `selection.cost_order`.

* **Acceptance / harness:**

  * `python -m src.harness --explain --task-id <id> --train-dir …` on a `truth_pass_missing_law` case now prints `WHY: OK: exact` **or** `WHY: missing_descriptor cid=… train=… p_out=… expected/ got …` with a concrete witness.
  * `python -m src.harness --determinism …` stable on test outputs and `selection` hashes across train-order permutations.

---