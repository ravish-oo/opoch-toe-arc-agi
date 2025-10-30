# Invariants (Must Hold, Assert in Code)

I-1 Present Round-Trip
- unpresent(present(G_input)) == G_input  (byte-equal)
- unpresent(present(G_output)) == G_output
Assert: present.py::present_grid, present.py::unpresent_grid

I-2 D4 & Anchor Identities
- pose_inv(pose_fwd(x,op,H,W),op,H,W) == x
- anchor_inv(anchor_fwd(x,a),a) == x
Assert: morphisms.py property tests

I-3 S-View Closure Bounds
- depth ≤ 2; |admitted views| ≤ 128
- Every admitted view M satisfies X(M(x)) == X(x) on its domain
Assert: sviews.py after build_closure()

I-4 Truth Single-Valuedness Before Laws
- After Paige–Tarjan, for every class a and every train pair i:
  |{ Y_i[r,c] : (r,c) in OUT_i where class_map_i[(r,c)] == a }| == 1
Assert: paige_tarjan.py::finalize_partition()

I-5 Shape Law & Pullback
- Learned law type respects precedence: multiplicative ≺ additive ≺ mixed; bbox only if affine fails
- pullback uses floor mapping; undefined ⇔ OOB after floor
Assert: shape_law.py::learn_and_log(), shape_law.py::pullback()

I-6 KEEP Admissibility Coverage
- For any admitted KEEP on class a: it is defined and correct on 100% of class pixels in every train OUT
Assert: laws/keep.py::admit_keep()

I-7 RECOLOR π Consistency
- π has no conflicts across trains and covers all test input colors appearing in class a
Assert: laws/recolor.py::admit_recolor()

I-8 Reducer Determinism
- UNIQUE truly unique; ARGMAX tiebreak by smallest color; LOWEST_UNUSED consistent across trains
Assert: laws/reducers.py::admit_*

I-9 Sieve Idempotence & Order
- Pruning pass order fixed; repeat until no removal
- If any K_class becomes empty ⇒ emit missing_descriptor and halt
Assert: sieve.py main loop

I-10 Painter Idempotence
- paint(Y*) == Y* on second run
Assert: paint.py after build

I-11 Determinism
- Permuting train-pair order does not change final outputs or receipt hash
Assert: harness.py::determinism_check()

I-12 Receipt Hash Stability
- Hash over canonical JSON with sorted keys, fixed list orders
Assert: receipts.py::finalize()
