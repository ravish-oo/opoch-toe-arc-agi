# Operating principles

* Bottom-up, atomic units; no TODOs left in code; no stubs.
* Test-first; each unit ships with property tests and 2–3 golden checks.
* Determinism over everything; same inputs, same bytes, same receipts.
* Receipts at *module boundaries only*; not per function.
* Zero improvisation; Work order provides tight skeleton plus a pass/fail harness.

# Minimal repo layout

```
/docs/anchors/
  00-math-spec.md
  01-engineering-spec.md
  02-locks-and-minispecs.md     # the 12 locks + 2 minispecs
  03-invariants.md              # assert list, exact wording
  04-receipts-schema.json       # boundary receipts structure
  05-test-corpus-notes.md       # ids used in microsuite
/src/
  morphisms.py
  present.py
  sviews.py
  paige_tarjan.py
  shape_law.py
  class_map.py
  laws/__init__.py              # KEEP, RECOLOR, reducers, BLOCK(k)
  sieve.py
  paint.py
  receipts.py                   # boundary logging helpers only
  harness.py                    # run train eval + submission
/tests/
  test_morphisms.py
  test_present.py
  test_sviews.py
  test_truth.py                 # PT + partition properties
  test_shape.py
  test_laws.py
  test_sieve_and_paint.py
  microsuite_golden.json        # 6 canonical tasks; IN→OUT pairs
```

# Where receipts live (boundary only)

* `present.py`: chosen palette map; D4 op id; anchor; round-trip OK.
* `sviews.py`: admitted views with proofs; closure depth; count.
* `paige_tarjan.py`: split steps; final class count; single-valuedness OK.
* `shape_law.py`: type; (a,b,c,d) or bbox; coverage.
* `sieve.py`: global selection exact; class→law; cost.
* `paint.py`: coverage 100%; by-law counts.
  All receipts are emitted through `receipts.log(section, payload)` using the schema in `docs/anchors/04-receipts-schema.json`. One helper, ~80 LOC.

