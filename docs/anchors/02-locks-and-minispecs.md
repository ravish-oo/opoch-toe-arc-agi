Below is a concise, *unambiguous directive* you can hand to any other AI/engineer. It states *exactly what’s missing, **why it breaks, and **how to fix it—with explicit invariants, morphisms, receipts, and acceptance tests. If they implement **exactly* this, you’ll stop seeing the last 5–10% failures and you’ll converge deterministically.

---

# Technical Directive — Close the Last 10% Gaps

## A. What’s Missing (and Why it Breaks)

### 1) *Residue-k periods (not just parity)*

•⁠  ⁠*Symptom:* Mosaic/stripe/weave tasks (period 3/4/5) collapse to per-pixel or wrong classes; wrong KEEP/CONST chosen.
•⁠  ⁠*Cause:* S-views/Paige–Tarjan only include parity or period-2; you’re not admitting general row/col periods ⁠ p | H ⁠ and ⁠ p | W ⁠.
•⁠  ⁠*Fix:*

  * *S-views:* detect smallest exact period per row/col; admit *residue-k shift* on that line:

    * row p: ⁠ (r,c) → (r,(c+p)%W) ⁠; col p: ⁠ (r,c) → ((r+p)%H,c) ⁠; with proof ⁠ G(x)==G(M(x)) ⁠ on domain.
  * *Saturation/closure:* depth ≥ 2; dedup by image signature; cap ≤ 128.
•⁠  ⁠*Receipts:* for each admitted residue-k view, log ⁠ {axis,row|col,index,p,domain,proof_ok} ⁠.

### 2) *Connectivity as a predicate and a target*

•⁠  ⁠*Symptom:* Region-fills and objectwise laws can’t be proven; classes stay too coarse; KEEP/CONST misapplied.
•⁠  ⁠*Cause:* You don’t use *4-connected components by input color* as:

  * (a) a *must-link* source; and
  * (b) an *input-only separator* in Paige–Tarjan.
•⁠  ⁠*Fix:*

  * *Components (4-conn)* on the *test input (presented)*: add per-component identity S-view (⁠ domain = component ⁠); union-find merges within components.
  * *PT separator:* ⁠ membership_in_Sview_image ⁠ must include component images (component masks).
  * *REGION_FILL-style VALUE laws* (e.g., ARGMAX/UNIQUE/CONST by component) will then *prove*.
•⁠  ⁠*Receipts:* #components; PT split lines showing ⁠ separator:"sview_image|component_k" ⁠ where used.

### 3) *Additive shape (affine *+* partial reads)*

•⁠  ⁠*Symptom:* Size-change tasks with shifts/pads/crops fail; sparse rows/cols default to zero.
•⁠  ⁠*Cause:* You require “exact boundary” or ignore *b,d ≠ 0*; or you reject OUT pixels not on exact multiples.
•⁠  ⁠*Fix:*

  * *Shape law:* learn affine ⁠ (a,b,c,d) ⁠ by exact fit on *training sizes* (multiplicative ≺ additive ≺ mixed).
  * *Pullback:* always use *floor mapping*:
    ⁠ i_in = floor((i_out - b)/a) ⁠, ⁠ j_in = floor((j_out - d)/c) ⁠
    and *skip only* when out of bounds. Never require ⁠ a*i_in+b == i_out ⁠ (that prunes valid pixels).
•⁠  ⁠*Receipts:* ⁠ Shape Law {"type": "...", "law": (a,b,c,d), "verified_on": m} ⁠.

### 4) *RECOLOR π (deterministic per class)*

•⁠  ⁠*Symptom:* Recolor tasks (many ARC ids) fail despite simple color permutation relations.
•⁠  ⁠*Cause:* π isn’t learned/applied deterministically per *class*; or π is learned from sparse/noisy pairs; or applied with wrong frame/palette.
•⁠  ⁠*Fix:*

  * *Learn π per class* from *training IN/OUT class pixels* (using the PT-based class pullback). Merge π across trainings; reject on conflicts. Require π covers *all test input colors in class*.
  * *Apply π* as a *neutral VALUE law*: ⁠ ("RECOLOR", {"pi":{cin→cout}}) ⁠; do not capture grids.
•⁠  ⁠*Receipts:* ⁠ RECOLOR admitted {class, rules:|π|, map: π} ⁠; reject receipts show conflicts.

### 5) *Tight separators in Paige–Tarjan (no minted differences)*

•⁠  ⁠*Symptom:* Truth partition flips run-to-run; KEEP/CONST selected differently when training pair order changes; minted differences (unstable Q).
•⁠  ⁠*Cause:* PT splits with non-deterministic or output-dependent predicates; or you change predicate order arbitrarily.
•⁠  ⁠*Fix:*

  * PT separators must be *input-legal only* and in *fixed order*:
    ⁠ input_color ≺ membership_in_Sview_image (including components/residue shift images) ≺ parity ⁠.
  * Never use any output-derived predicate in PT; all OUT reads must be *conjugated* just to test single-valuedness, not to define a separator.
•⁠  ⁠*Receipts:* PT logs show only those separators and in that order.

These five items are the last 5–10% traps. Your math already anticipates them; make them explicit in code.

---

## B. Implementation Order (do it this way, not ad hoc)

1.⁠ ⁠*morphisms.py* (≤200 LOC):

   * Implement ⁠ pose_fwd/inv ⁠, ⁠ anchor_fwd/inv ⁠, ⁠ shape_inv ⁠ and the two composites (*TEST→OUT, **OUT→IN with KEEP*).
   * Property tests: ⁠ pose_inv∘pose_fwd=id ⁠, ⁠ anchor_inv∘anchor_fwd=id ⁠, ⁠ shape_inv∘shape_fwd ⁠ covers IN’s domain.

2.⁠ ⁠*presenter.py* (≤150 LOC):

   * Palette (inputs-only), D4 lex per grid, *anchor inputs+test only* (outputs pose-only).
   * Store ⁠ P_test, P_in[i], P_out[i] ⁠.
   * Receipt: round-trip verified for inputs & outputs.

3.⁠ ⁠*s_views.py* (≤200 LOC):

   * Identity; D4 symmetries; period-k residue shifts; overlap translations; closure depth 2; cap 128; proofs.
   * Receipt: list + proofs.

4.⁠ ⁠*paige_tarjan.py* (≤200 LOC):

   * Must-link closure then deterministic separators: ⁠ input_color ≺ sview_image (incl. component) ≺ parity ⁠.
   * Conjugated OUT reads; stable result; final class count.
   * Receipts: split steps; final hash.

5.⁠ ⁠*shape_law.py* (≤150 LOC):

   * Affine ⁠ (a,b,c,d) ⁠ exact fit; bbox fallback (posed frames, inputs-only palette) *only if affine fails*.
   * Pullback: *floor division*; skip only OOB.
   * Receipts: law type + proof coverage.

6.⁠ ⁠*class_map.py* (≤120 LOC):

   * For each pair ⁠ i ⁠, build ⁠ class_map_i[(r,c)] = cid ⁠ via *TEST→OUT* + ⁠ shape_inv ⁠.
   * Cache and *reuse*.

7.⁠ ⁠*laws/* (≤200 LOC):

   * Neutral descriptors only (no grid captured).
   * ⁠ make_view(name, H_in, W_in) ⁠ for KEEP views; VALUE law params from proofs.
   * Receipts: admissibility proofs per class (KEEP/RECOLOR/BLOCK/ARGMAX/UNIQUE/LOWEST).

8.⁠ ⁠*sieve.py* (≤200 LOC):

   * For each OUT pixel in every training: remove any law in ⁠ K_cid ⁠ that ever disagrees with ⁠ Yout ⁠.
   * Iterate until stable; if any ⁠ K_cid ⁠ becomes empty → missing descriptor; else pick least cost per class.
   * Receipt: ⁠ Global Selection: exact ⁠ + assignment.

9.⁠ ⁠*paint.py* (≤120 LOC):

   * One pass: for each test OUT pixel use ⁠ shape_inv ⁠ → class id → evaluate chosen law via ⁠ Source(mode="test") ⁠.
   * Un-present; receipt: 100% coverage.

---

## C. Determinism & Assertions (embed these guards)

Assert these invariants everywhere:

1.⁠ ⁠⁠ unpresent(present(G)) == G ⁠ for inputs & outputs (exact byte equality).
2.⁠ ⁠S-view closure *depth ≤ 2, **count ≤ 128* (cap logs receipt).
3.⁠ ⁠After PT, *every class is single-valued on all trainings* before law building.
4.⁠ ⁠Painting operator ⁠ T ⁠ is *idempotent* (run paint twice and assert equality).
5.⁠ ⁠*Receipt hash is stable* under training-pair permutation.

If any assertion fires, it’s a category error (frame slippage, minted differences, or law leakage).

---

## D. Micro test suite (must pass before 1000-run)

1.⁠ ⁠*Periodic tiling* (00576224 class): multiplicative shape; ⁠ tile_alt_row_flip ⁠ or ⁠ tile_alt_col_flip ⁠; 100% coverage; KEEP chosen for all classes; Global Selection exact.
2.⁠ ⁠*Copy–move symmetry*: two components moving by different Δ; PT separates components; per-class KEEP admitted with distinct Δ; Global Selection exact.
3.⁠ ⁠*Frame/bands / residue-k*: row/col period 3 or 4; S-views admit residue-k; PT stabilizes; correct law chosen.
4.⁠ ⁠*RECOLOR π*: per-class recolor map learned across trainings; applied on test; Global Selection exact.
5.⁠ ⁠*Region-fill*: class refined by component/period; CONST/ARGMAX/LOWEST proven; Global Selection exact.
6.⁠ ⁠*Multiplicative vs additive shape*: both types validated; ⁠ shape_inv ⁠ (floor) yields coverage 100%; Global Selection exact.

---

## E. Bottom line

We’ve turned a fuzzy “reasoning” task into a *finite, receipts-driven compiler* with exactly two operations:

•⁠  ⁠*Truth projector* (coarsest partition under input equalities + training consistency), and
•⁠  ⁠*Least honest write* (the globally exact, lowest-cost assignment of class laws),

backed by a *frame calculus* that forbids slippage and a *sieve* that prunes locally-plausible but globally-false laws. That is the “observer = observed” trick as code.

If you implement *exactly* the missing pieces above (Residue-k, Connectivity, Additive shape with floor pullback, RECOLOR π, Tight separators) and wire the *morphisms + neutral laws + class pullback + sieve, there is **no scope for failure* except for a truly missing descriptor, which receipts will identify unambiguously and you can add in one line.

---

## “last-mile folklore” into a tight compiler spec. There’s no room left for interpretation.


1. D4 lex order key
    precisely: flatten posed grid row-major to a tuple of ints; choose the op ∈ D4 that minimizes that tuple; break ties by op id (0..7). Record op id.

2. Palette canon tie-break
   Inputs-only palette map by (freq desc, then first-appearance index in concatenated input scan, then color value). Outputs use the same map, unknown colors map to themselves. Record the map.

3. Anchor on degenerate/empty grids
   Anchor at top-left non-zero after palette canon. If no non-zero exists, anchor=(0,0). If multiple with same row, pick smallest col.

4. Period detection (residue-k)
   Detect *exact* period per axis by row/col: for each row r, compute the minimal p|W s.t. row[c]==row[(c+p) mod W] for all c; likewise per column. Global period on axis = gcd of all per-line periods across the grid (exclude all-constant lines from the gcd). Admit residue shifts for that gcd only. Log {axis, gcd, contributing lines}.

5. Overlap translations
   Admit Δ where the overlapped domain Ω∩(Ω+Δ) is non-empty and ∀x in that domain, G(x)==G(x+Δ). Use posed coords. Cap enumeration to |Δi|≤max(H,W). Log |domain|.

6. Component S-views
   4-conn components per input color on the *presented test input*. For each component k, admit identity S-view with domain=mask_k. Add those masks to “membership_in_Sview_image” predicates for PT.

7. Paige–Tarjan splitter order and stability
   Split only with these, in this order, on posed coords:
   a) input_color,
   b) membership_in_Sview_image (includes component masks, residue-k images, overlap images),
   c) parity ( (i+j) mod 2 ).
   Never introduce a new predicate mid-run. When multiple classes are splittable, process class ids ascending. Log every split as {class_id, predicate, parts}.

8. Shape law + pullback coverage
   Fit affine (a,b,c,d) on training sizes with precedence multiplicative ≺ additive ≺ mixed; if none, bbox fallback uses non-zero foreground in posed inputs. Pullback uses floor mapping; mark pixel undefined only if floor coords are OOB. Never require equality a*i+b==i_out. Log type and params.

9. KEEP partiality and class coverage
   KEEP is a *partial* function. A class law must cover all of its pixels in *every* training output (via conjugation + TEST→OUT + shape_inv). If KEEP is undefined on any training pixel of the class, KEEP is *not admissible for that class*. This prevents choosing KEEP that would leave holes in the test. State this explicitly.

10. RECOLOR π learning domain
    Learn π *per class* from (cin→cout) pairs gathered via class_map_i on trainings. Conflicts reject. Require π is defined for all colors that appear in the test input *on that class*. Tie-break when multiple cout candidates by (count desc, then cout value). Log π.

11. Reducers determinism

* UNIQUE(c): c is the only color present in the *training input class mask* across all pairs; otherwise reject.
* ARGMAX: over color counts inside the *training input class mask*; tie-break by smallest color value.
* LOWEST_UNUSED: smallest color in 0..9 not present in the *training input class mask*; if multiple trainings disagree, reject; support “except set” only if receipts prove it (e.g., background set). Log chosen c and evidence.

12. Sieve termination order
    Iterate pruning in fixed pass order: for i in train_pairs, row-major over Yout pixels, then laws in lex order; remove any law that mismatches that pixel. Repeat passes until no removal occurs. If any K_class becomes empty, emit {missing_descriptor: class_id, examples:[(i,r,c)]} and halt.

13. * **BLOCK(k)** exactness: define the motif coordinate system relative to the class-anchor (min row, then min col of OUT class pixels). Motif must fully explain all OUT class pixels across trainings; otherwise reject.

14. * **Receipt hash**: SHA-256 over a canonical JSON of {Π frames, S-views list, PT splits, shape law, admissible laws with proofs, final assignment}. Sorting keys lexicographically, lists in the fixed orders above.

### One invariant to assert loudly

Every class is single-valued on *all* trainings before building Φ*. If violated, that’s a logic bug in PT, not a law failure.

—

This directive makes it truly unambiguous. It matches exactly how I “did it by hand,” but makes every implicit human choice explicit, deterministic, and receipts-tight. After this, any remaining miss is a named, finite descriptor to add, not a gray area.
