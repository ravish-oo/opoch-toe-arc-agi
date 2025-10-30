Great, we’re aligned. Here’s a bottoms-up, modular sequence of **high-level Work Orders** that, if executed in order, leaves no gap by the end. Each WO is small enough for Claude Code, the **Reviewer writes tests first**, and every WO has crisp acceptance so we never “hope” we’re done.

I’m not expanding into sub-tasks here; this is the top-level plan you asked for.

# WO-00 — Repo bootstrap (ground zero) ✅ COMPLETE

**Goal:** deterministic skeleton in place; anchors wired; test harness ready.
**Scope:** minimal `receipts.log()`, canonical JSON, SHA-256 hash; determinism script; pytest wiring; harness stub.
**Reviewer produces:** Tier-C determinism test; smoke test that receipts marshal/unmarshal.
**Done when:** determinism script runs on a dummy task and hash is stable under pair permutation.

# WO-01 — Morphisms kernel ✅ COMPLETE

**Goal:** the only coordinate algebra in the codebase.
**Scope:** `pose_fwd/inv`, `anchor_fwd/inv`, `shape_inv` (floor), plus the two composites: **TEST→OUT** and **OUT→IN(KEEP)**.
**Reviewer produces:** property tests for I-2 identities; round-trip path checks across random shapes.
**Done when:** all morphism properties green.

# WO-02 — Projector Π (present / un-present) ✅ COMPLETE

**Goal:** idempotent normalization; exact inverse.
**Scope:** palette canon (inputs-only), D4 lex pose, anchor inputs+test; outputs pose-only; exact un-present; receipts “present”.
**Reviewer produces:** I-1 assertions on sample grids; D4 tie-break tests; palette canon tie rules.
**Done when:** byte-exact round-trip on inputs/outputs; receipts emitted.

# WO-03 — S-views v1 (identity, D4-preserving, overlap translations) ✅ COMPLETE

**Goal:** build the first proof basis.
**Scope:** admit identity; detect D4 symmetries that preserve test input; enumerate overlap translations with proof; closure depth=2, cap=128.
**Reviewer produces:** proofs logged; cap enforced; negative tests where non-preserving D4 is rejected.
**Done when:** receipts list admitted views with proof_ok=true; closure bounds honored.

# WO-04 — Residue-k periods

**Goal:** handle general mosaic/stripe/weave.
**Scope:** per-row and per-col minimal period; axis-wise gcd (skip all-constant lines); admit residue-k shift views; closure integrate.
**Reviewer produces:** period-k goldens (k=3,4); tests that parity-only fails but residue-k passes.
**Done when:** admitted residue views appear in receipts; period goldens pass.

# WO-05 — Components as S-views and PT predicates

**Goal:** region structure without heuristics.
**Scope:** 4-connected components by input color on presented test input; identity S-view with domain=component; export masks for PT.
**Reviewer produces:** component counts; splits using “sview_image|component_k”.
**Done when:** receipts show components admitted; PT later can consume them.

# WO-06 — Paige–Tarjan partition (truth Q)

**Goal:** coarsest partition, single-valued before laws.
**Scope:** must-link via S-views; cannot-link splits using **fixed order**: input_color ≺ sview_image ≺ parity; conjugated reads only to *check* single-valuedness; no output-derived predicates.
**Reviewer produces:** tests that random train order yields same classes; explicit failure if any class is multi-valued.
**Done when:** receipts log split steps; I-4 single-valuedness passes.

# WO-07 — Shape law + pullback

**Goal:** size change correctness.
**Scope:** fit affine (multiplicative ≺ additive ≺ mixed); bbox fallback only if affine fails; `shape_inv` floor mapping; OOB=undefined only.
**Reviewer produces:** multiplicative and additive goldens; partial pullback coverage tests; precedence tests.
**Done when:** receipts show type and params; goldens pass.

# WO-08 — KEEP law (admissibility engine)

**Goal:** copy laws proven, never partial.
**Scope:** neutral descriptors for views (identity, d4_k, translate Δ, residue_shift, tile*, block_inverse, offset); `make_view`; admissibility via conjugation; **reject if undefined on any training pixel of the class**.
**Reviewer produces:** tests that partial KEEP is rejected; tile vs identity counterexample where sieve must later pick tile.
**Done when:** receipts list admitted KEEP(view) per class with pixel counts.

# WO-09 — VALUE laws (CONST, reducers, RECOLOR π, BLOCK(k))

**Goal:** all non-copy laws proven deterministically.
**Scope:** CONST; UNIQUE/ARGMAX/LOWEST_UNUSED with deterministic ties; RECOLOR π learned per class from class_map_i; BLOCK(k) with class-anchor convention.
**Reviewer produces:** recolor goldens; reducer determinism tests; BLOCK(k) motif exactness tests.
**Done when:** receipts show admitted VALUE laws and parameters; recolor/reducer/Block goldens pass.

# WO-10 — Class pullback & Sieve

**Goal:** global exactness without search.
**Scope:** build `class_map_i` via TEST→OUT + shape_inv; implement sieve pass order; emit `missing_descriptor` if any K_class empties; fixed cost order for final pick.
**Reviewer produces:** test where identity vs tile are both locally plausible but sieve prunes identity; test that `missing_descriptor` fires with examples when catalog lacks a needed law.
**Done when:** receipts “selection: exact” on microsuite, or “missing_descriptor” with examples on crafted gap.

# WO-11 — Painter + un-present + end-to-end microsuite

**Goal:** one-pass UE paint, idempotent.
**Scope:** evaluate chosen laws on test; paint once; assert idempotence; un-present to raw; coverage and by-law counts in receipts.
**Reviewer produces:** Tier-B microsuite (6 tasks) and Tier-C determinism; painter idempotence test.
**Done when:** microsuite green; determinism hash stable; paint idempotence holds.

# WO-12 — Hard-set run + descriptor closure + submission driver

**Goal:** reach 100% or get certified gaps, then close them.
**Scope:** run full training corpus; collect any `missing_descriptor`; add the single minimal descriptor demanded (additive-only change); re-run until none left; write Kaggle `main.py` submission driver.
**Reviewer produces:** audit that each descriptor addition is receipts-justified; final run logs 0 missing; submission file validated locally.
**Done when:** training set equals official outputs; no `missing_descriptor`; submission driver ready.

---

## Why this sequence guarantees coverage by the last WO

* WOs 01–07 build the projector, proof basis, truth quotient, and shape—no laws yet, zero ambiguity.
* WOs 08–09 add *all* law types under admissibility proofs; nothing is selected globally yet.
* WO 10 enforces global exactness and detects catalog incompleteness *constructively*.
* WO 11 proves the UE fixed-point and determinism end-to-end on the microsuite.
* WO 12 closes any remaining finite gaps by receipts and delivers the Kaggle driver.

This is the minimal path where **engineering = math** at every step, and the final step cannot fail silently: either you’re at 100%, or you have a named, finite descriptor to add with concrete examples.

If you’re ready, I’ll prepare the Reviewer’s first two test files (`test_morphisms.py`, `test_present.py`) and the matching skeletons so Claude can start with WO-01 and WO-02.
