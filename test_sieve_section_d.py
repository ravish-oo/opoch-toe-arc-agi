#!/usr/bin/env python3
"""
Test Section D (Sieve) implementation.
"""

import sys
sys.path.insert(0, 'src')

from sieve import run_sieve
from truth import Partition

# Test 1: Exact selection - identity vs tile
print("=== Test 1: Exact selection (identity vs tile paradox) ===")

# Mock partition (single class)
part = Partition(H=2, W=2, cid_of=[0, 0, 0, 0])

# Frames
P_test = (0, (0, 0), (2, 2))
P_in_list = [(0, (0, 0), (2, 2))]
P_out_list = [(0, (0, 0), (2, 2))]

# Input = Output (simple copy)
Xin = [[[1, 2], [3, 4]]]
Yout = [[[1, 2], [3, 4]]]

# Build class_map (all pixels belong to class 0)
class_maps = [[0, 0, 0, 0]]

# Both identity and tile work
keep_admitted = {
    0: [
        {"view": "identity"},
        {"view": "tile"}
    ]
}
value_admitted = {}

result = run_sieve(
    part, class_maps, Xin, Yout, P_test, P_in_list, P_out_list,
    keep_admitted, value_admitted
)

print(f"Status: {result['status']}")
print(f"Assignment: {result['assignment']}")
print(f"Prune log entries: {len(result['prune_log'])}")

assert result["status"] == "exact", f"Expected exact, got {result['status']}"
assert "0" in result["assignment"], "Class 0 missing"
# Tile has lower cost than identity per COST_ORDER
assert "KEEP:tile" in result["assignment"]["0"], \
    f"Expected tile (lower cost), got {result['assignment']['0']}"
print("✓ Test 1 passed\n")

# Test 2: Missing descriptor
print("=== Test 2: Missing descriptor detection ===")

# Output is constant 5, but only KEEP laws admitted (no CONST)
Yout2 = [[[5, 5], [5, 5]]]
class_maps2 = [[0, 0, 0, 0]]

result2 = run_sieve(
    part, class_maps2, Xin, Yout2, P_test, P_in_list, P_out_list,
    keep_admitted, {}  # No VALUE laws
)

print(f"Status: {result2['status']}")
print(f"Missing classes: {len(result2.get('missing', []))}")
if result2.get('missing'):
    print(f"First missing: cid={result2['missing'][0]['cid']}, examples={len(result2['missing'][0]['examples'])}")

assert result2["status"] == "missing_descriptor", \
    f"Expected missing_descriptor, got {result2['status']}"
assert len(result2["missing"]) > 0, "Missing list is empty"
assert result2["missing"][0]["cid"] == 0, "Expected cid=0 in missing"
print("✓ Test 2 passed\n")

# Test 3: Pruning - law that doesn't work everywhere
print("=== Test 3: Pruning test ===")

# Two different output grids - only one law should survive
Xin3 = [
    [[1, 2], [3, 4]],  # Train 0
    [[1, 2], [3, 4]]   # Train 1
]
Yout3 = [
    [[1, 2], [3, 4]],  # Train 0: identity works
    [[2, 1], [4, 3]]   # Train 1: identity doesn't work, needs different law
]

P_in3 = [(0, (0, 0), (2, 2)), (0, (0, 0), (2, 2))]
P_out3 = [(0, (0, 0), (2, 2)), (0, (0, 0), (2, 2))]
class_maps3 = [[0, 0, 0, 0], [0, 0, 0, 0]]

# Admit identity - should be pruned by train 1
keep_admitted3 = {
    0: [
        {"view": "identity"},
        {"view": "d4", "op": 5}  # Horizontal flip might work
    ]
}

result3 = run_sieve(
    part, class_maps3, Xin3, Yout3, P_test, P_in3, P_out3,
    keep_admitted3, {}
)

print(f"Status: {result3['status']}")
print(f"Assignment: {result3['assignment']}")
print(f"Prune log entries: {len(result3['prune_log'])}")

# Identity should be pruned, leaving only d4_5 or missing
if result3["status"] == "exact":
    print(f"Selected law: {result3['assignment']['0']}")
    assert "identity" not in result3["assignment"]["0"], "Identity should have been pruned"
print("✓ Test 3 passed\n")

# Test 4: CONST admission
print("=== Test 4: CONST law ===")

Yout4 = [[[7, 7], [7, 7]]]
class_maps4 = [[0, 0, 0, 0]]

value_admitted4 = {
    0: [
        {"type": "CONST", "c": 7}
    ]
}

result4 = run_sieve(
    part, class_maps4, Xin, Yout4, P_test, P_in_list, P_out_list,
    {}, value_admitted4
)

print(f"Status: {result4['status']}")
print(f"Assignment: {result4['assignment']}")

assert result4["status"] == "exact", f"Expected exact, got {result4['status']}"
assert "CONST" in result4["assignment"]["0"], \
    f"Expected CONST, got {result4['assignment']['0']}"
print("✓ Test 4 passed\n")

print("=== All Section D tests PASSED ===")
