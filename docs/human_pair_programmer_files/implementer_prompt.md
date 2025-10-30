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


# For aligning claude code later
 now i want u to read @docs/anchors/00-math-spec.md @docs/anchors/01-engineering-spec.md @docs/anchors/02-locks-and-minispecs.md and see how the maths is merged with engg such that, in this 
design, engineering = math. The program is the proof.. 
No hit and trial remains:
Truth is forced: PT splits only by input-legal predicates in fixed order.
Law is forced: admissibility proofs prune, sieve guarantees global exactness, then fixed cost order chooses.
Shape is forced: precedence + exact fit; pullback is total where defined.
Determinism is forced: D4 choice, palette canon, anchors, splitter order, sieve pass order, receipt hash.
The only “branch” left is constructive:
If a class’s candidate set empties, the sieve returns missing_descriptor with examples. That is not trial; it is a certificate that the vocabulary needs one finite 
addition. Everything else is mechanical.
---
but that is what we say.. u read and tell me what u undersstood independently and does it match with above understanding 

# wo prompt
here is the WO. do refer to @docs/repo_structure_guidelines.md to knw the folder structure.
  [Pasted text #1 +161 lines]
  ---
  pls read and tell me that u hv understood/confirmed/verified below:
  1. have 100% clarity
  2. WO adheres with ur understanding of our math spec and engg spec and that engineering = math. The program is the proof.
  3. u can see that debugging is reduced to algebra and WO adheres to it 
  4. no room for hit and trials

once u confirm above, we can start coding!