You must implement exactly provided Work Order. You do not write tests. Tests are provided by the Reviewer.
Read first: @docs/anchors/00-math-spec.md, @docs/anchors/01-engineering-spec.md, @docs/anchors/02-locks-and-minispecs.md, @docs/anchors/03-invariants.md, @docs/anchors/04-receipts-schema.json.
Do not change signatures, add functions, add deps, leave TODOs, or stub logic.
Determinism required.
Receipts: emit only through receipts.log(section,payload) as per 04-receipts-schema.json.
KEEP partiality rule: reject KEEP if undefined on any training pixel of the class.
Paige–Tarjan splitter order is fixed: input_color ≺ sview_image ≺ parity.
At the end, run:
python src/harness.py --determinism and paste the summary.

If any step cannot be satisfied without breaking anchors, stop and return a short diff explaining which invariant would be violated. 
