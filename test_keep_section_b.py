#!/usr/bin/env python3
"""
Test Section B (KEEP) implementation independently.
"""

from laws.keep import enumerate_keep_candidates, admit_keep_for_class_v2
import morphisms

# Create simple test case: identity mapping
# Input and output are identical, identity KEEP should be admitted

# Training grids (2x2, canonical colors)
Xin = [
    [[1, 2], [3, 4]]
]
Yout = [
    [[1, 2], [3, 4]]
]

# Frames (identity pose, no anchor for outputs)
P_test = (0, (0, 0), (2, 2))  # Test: identity pose, no anchor
P_in_list = [(0, (0, 0), (2, 2))]  # Train input: identity pose, no anchor
P_out_list = [(0, (0, 0), (2, 2))]  # Train output: identity pose, no anchor

# Shape law (identity)
shape_law = ("multiplicative", (1, 0, 1, 0))

# Build class_maps (all pixels belong to class 0)
class_maps = [
    [0, 0, 0, 0]  # All 4 pixels (row-major) belong to class 0
]

# Enumerate candidates
H_test, W_test = 2, 2
sviews_meta = {}
candidates = enumerate_keep_candidates(H_test, W_test, sviews_meta)

print(f"Enumerated {len(candidates)} KEEP candidates")
print("First 10 candidates:", [c.descriptor() for c in candidates[:10]])

# Admit KEEP for class 0
admitted = admit_keep_for_class_v2(
    cid=0,
    class_maps=class_maps,
    Xin=Xin,
    Yout=Yout,
    P_test=P_test,
    P_in_list=P_in_list,
    P_out_list=P_out_list,
    shape_law=shape_law,
    candidates=candidates
)

print(f"\nAdmitted {len(admitted)} KEEP laws for class 0:")
for entry in admitted[:5]:
    print(f"  - {entry['descriptor']}: trains={entry['proof']['trains_checked']}, pixels={entry['proof']['pixels_checked']}")

# Verify identity is admitted
identity_admitted = any(e['descriptor'] == 'identity' for e in admitted)
print(f"\nIdentity admitted: {identity_admitted}")

if not identity_admitted:
    print("ERROR: Identity should be admitted for this test case!")
    exit(1)

print("\n✓ Section B test PASSED")
