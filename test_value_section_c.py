#!/usr/bin/env python3
"""
Test Section C (VALUE) implementation.
"""

from laws.value import admit_value_for_class_v2

# Test 1: CONST - all outputs are the same color
print("=== Test 1: CONST ===")
Xin = [[[1, 2], [3, 4]]]
Yout = [[[5, 5], [5, 5]]]  # All 5s
Xtest = [[1, 2], [3, 4]]
P_test = (0, (0, 0), (2, 2))
P_in_list = [(0, (0, 0), (2, 2))]
P_out_list = [(0, (0, 0), (2, 2))]

# Class 0 has all pixels
class_pixels_test = [(0, 0), (0, 1), (1, 0), (1, 1)]

# Build class_map (all pixels belong to class 0)
class_maps = [[0, 0, 0, 0]]  # 4 pixels in row-major order

admitted = admit_value_for_class_v2(
    cid=0,
    class_pixels_test=class_pixels_test,
    class_maps=class_maps,
    Xin=Xin,
    Yout=Yout,
    Xtest=Xtest,
    P_test=P_test,
    P_in_list=P_in_list,
    P_out_list=P_out_list
)

print(f"Admitted {len(admitted)} laws:")
for entry in admitted:
    print(f"  - {entry['descriptor']}")

const_found = any("CONST" in e['descriptor'] for e in admitted)
assert const_found, "CONST should be admitted"
print("✓ CONST test passed\n")

# Test 2: UNIQUE - all input pixels are same color
print("=== Test 2: UNIQUE ===")
Xin2 = [[[7, 7], [7, 7]]]  # All same color
Yout2 = [[[7, 7], [7, 7]]]  # Output equals unique input
Xtest2 = [[7, 7], [7, 7]]
class_maps2 = [[0, 0, 0, 0]]

admitted2 = admit_value_for_class_v2(
    cid=0,
    class_pixels_test=class_pixels_test,
    class_maps=class_maps2,
    Xin=Xin2,
    Yout=Yout2,
    Xtest=Xtest2,
    P_test=P_test,
    P_in_list=P_in_list,
    P_out_list=P_out_list
)

print(f"Admitted {len(admitted2)} laws:")
for entry in admitted2:
    print(f"  - {entry['descriptor']}")

unique_found = any("UNIQUE" in e['descriptor'] for e in admitted2)
assert unique_found, "UNIQUE should be admitted"
print("✓ UNIQUE test passed\n")

# Test 3: RECOLOR - simple color mapping
print("=== Test 3: RECOLOR ===")
Xin3 = [[[1, 2], [1, 2]]]
Yout3 = [[[6, 7], [6, 7]]]  # 1→6, 2→7
Xtest3 = [[1, 2], [1, 2]]
class_maps3 = [[0, 0, 0, 0]]

admitted3 = admit_value_for_class_v2(
    cid=0,
    class_pixels_test=class_pixels_test,
    class_maps=class_maps3,
    Xin=Xin3,
    Yout=Yout3,
    Xtest=Xtest3,
    P_test=P_test,
    P_in_list=P_in_list,
    P_out_list=P_out_list
)

print(f"Admitted {len(admitted3)} laws:")
for entry in admitted3:
    print(f"  - {entry['descriptor']}")

recolor_found = any("RECOLOR" in e['descriptor'] for e in admitted3)
assert recolor_found, "RECOLOR should be admitted"

# Check π mapping
recolor_entry = [e for e in admitted3 if "RECOLOR" in e['descriptor']][0]
pi = recolor_entry['proof']['pi']
assert pi[1] == 6 and pi[2] == 7, f"Expected π={{1:6, 2:7}}, got {pi}"
print(f"✓ RECOLOR test passed (π={pi})\n")

print("=== All Section C tests PASSED ===")
