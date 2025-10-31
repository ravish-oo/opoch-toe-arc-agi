# Testing Commands

## Quick Test on Small Corpus

Test on 5 tasks (fast):
```bash
python scripts/run_test_corpus.py
```

Test on 20 tasks:
```bash
python scripts/run_test_corpus.py --n 20
```

Test on 100 tasks (takes a few minutes):
```bash
python scripts/run_test_corpus.py --n 100 --output results_100.json
```

## Test Single Task

Run a specific task and see detailed output:
```bash
python scripts/run_single_task.py 00576224
```

Save receipts for debugging:
```bash
python scripts/run_single_task.py 00576224 --save receipts_00576224.json
cat receipts_00576224.json | python -m json.tool | less
```

## View Results

Results are saved as JSON. View them with:
```bash
cat test_results.json | python -m json.tool | less
```

Or extract specific info:
```bash
# Show all error messages
cat test_results.json | python -c "import sys, json; data=json.load(sys.stdin); [print(f\"{r['task_id']}: {r.get('error', '')}\") for r in data['summary']['results'] if r['status']=='error']"

# Show exact matches
cat test_results.json | python -c "import sys, json; data=json.load(sys.stdin); [print(r['task_id']) for r in data['summary']['results'] if r['status']=='exact']"

# Show missing descriptor tasks
cat test_results.json | python -c "import sys, json; data=json.load(sys.stdin); [print(f\"{r['task_id']}: class {m['cid']}\") for r in data['summary']['results'] if r['status']=='missing_descriptor' for m in r.get('missing', [])]"
```

## Determinism Check

Verify runner is deterministic:
```bash
python scripts/test_runner_determinism.py
```

Should print: `✓ PASS: Hashes are identical - runner is deterministic`

## Understanding Output

### Status Types

- **exact**: Task passed completely, train outputs match
- **missing_descriptor**: Pipeline succeeded but needs vocabulary extension (shows witnesses)
- **error**: PT or shape law failure (shows exact error)

### Witnesses

When a task fails, you get exact witnesses for algebraic debugging:

```
Task: 00576224
Status: error
Error: PT failed: class 0 has contradiction but no predicate splits it. colors_seen=[1, 2], witness=(1, 1)
```

This tells you:
- Which module failed (PT = truth partition)
- Which class has the issue (class 0)
- Exact pixel coordinate ((1, 1))
- What the contradiction is (colors [1, 2] in same class)

### Missing Descriptors

```
Task: 03560426
Status: missing_descriptor
Missing descriptors:
  Class ⊥: 3 witnesses
    train_idx=0, p_out=[0, 4], expected=0
    train_idx=0, p_out=[0, 5], expected=0
```

This tells you:
- Which class needs a descriptor (⊥ = unseen pixels)
- Exact examples showing what's expected
- Training pair index and output pixel coordinates

## Receipts

Receipts are embedded in the result JSON under each task. They contain:
- Module-by-module execution trace
- Deterministic hashes
- Coverage metrics
- Proof details

Access receipts in saved files:
```bash
cat receipts_00576224.json | jq '.sections'
```

Example receipt sections:
- `morphisms`: Coordinate algebra self-checks
- `present`: Palette map, pose ops, anchors
- `sviews`: Admitted views with proofs
- `components`: Connected component counts
- `truth`: Partition classes and splits
- `shape`: Shape law type and parameters
- `laws`: Per-class admitted KEEP/VALUE laws
- `selection`: Sieve results and pruning log
- `paint`: Coverage metrics and by-law counts

## Quick Examples

See which tasks have PT failures:
```bash
python scripts/run_test_corpus.py --n 100 | grep "PT failed"
```

Count error types:
```bash
python scripts/run_test_corpus.py --n 100 2>&1 | grep "Error:" | sort | uniq -c
```

Find tasks with missing descriptors:
```bash
python scripts/run_test_corpus.py --n 100 2>&1 | grep -A 2 "missing_descriptor"
```
