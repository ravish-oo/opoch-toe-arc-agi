#!/usr/bin/env python3
"""
Test residue shift enumeration fix.
"""

from laws.keep import enumerate_keep_candidates

# Test with row_gcd=3, col_gcd=2
H_test, W_test = 6, 4
sviews_meta = {"row_gcd": 3, "col_gcd": 2}

candidates = enumerate_keep_candidates(H_test, W_test, sviews_meta)

# Find residue candidates
residue_row = [c for c in candidates if c.name == "residue_row"]
residue_col = [c for c in candidates if c.name == "residue_col"]

print(f"row_gcd=3, expecting shifts 1,2:")
print(f"  Found {len(residue_row)} residue_row candidates:")
for c in residue_row:
    print(f"    - {c.descriptor()}")

print(f"\ncol_gcd=2, expecting shift 1:")
print(f"  Found {len(residue_col)} residue_col candidates:")
for c in residue_col:
    print(f"    - {c.descriptor()}")

# Verify counts
assert len(residue_row) == 2, f"Expected 2 residue_row (shifts 1,2), got {len(residue_row)}"
assert len(residue_col) == 1, f"Expected 1 residue_col (shift 1), got {len(residue_col)}"

# Verify parameters
row_shifts = sorted([c.params["p"] for c in residue_row])
col_shifts = sorted([c.params["p"] for c in residue_col])

assert row_shifts == [1, 2], f"Expected row shifts [1,2], got {row_shifts}"
assert col_shifts == [1], f"Expected col shifts [1], got {col_shifts}"

print("\n✓ Residue shift enumeration test PASSED")

# Test block_inverse
block_inverse = [c for c in candidates if c.name == "block_inverse"]
print(f"\nblock_inverse candidates (expecting k=2,3):")
for c in block_inverse:
    print(f"  - {c.descriptor()}")

k_values = sorted([c.params["k"] for c in block_inverse])
assert k_values == [2, 3], f"Expected k=[2,3], got {k_values}"

print("\n✓ block_inverse enumeration test PASSED")
