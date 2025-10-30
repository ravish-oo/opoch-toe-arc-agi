# Worked Examples: Solving ARC-AGI by Hand Using Math Spec

This document demonstrates how to apply the mathematical framework from `00-math-spec.md` to solve ARC-AGI tasks by hand. These examples serve as reference implementations for understanding how the universal equation (UE) works in practice.

**Reference**: See `docs/anchors/00-math-spec.md` for the complete mathematical specification.

---

## Example 1: Task 00576224 (Tiling with Alternating Flip)

### Task Description
- **Input size**: 2×2
- **Output size**: 6×6
- **Training pairs**: 2
- **Pattern**: Horizontal tiling with alternating vertical flip

### Training Data

**Training Example 1:**
```
Input:              Output:
[7, 9]             [7, 9, 7, 9, 7, 9]
[4, 3]             [4, 3, 4, 3, 4, 3]
                   [9, 7, 9, 7, 9, 7]  ← flipped
                   [3, 4, 3, 4, 3, 4]  ← flipped
                   [7, 9, 7, 9, 7, 9]  ← original
                   [4, 3, 4, 3, 4, 3]
```

**Training Example 2:**
```
Input:              Output:
[8, 6]             [8, 6, 8, 6, 8, 6]
[6, 4]             [6, 4, 6, 4, 6, 4]
                   [6, 8, 6, 8, 6, 8]  ← flipped
                   [4, 6, 4, 6, 4, 6]  ← flipped
                   [8, 6, 8, 6, 8, 6]  ← original
                   [6, 4, 6, 4, 6, 4]
```

**Test Input:**
```
[3, 2]
[7, 8]
```

---

### Step 0: Notation (Math Spec §0)

- Task t = 00576224
- Training pairs: i = 1, 2
- X_{t,1} = [[7,9],[4,3]], Y_{t,1} = 6×6 output
- X_{t,2} = [[8,6],[6,4]], Y_{t,2} = 6×6 output
- X_{t,*} = [[3,2],[7,8]] (test input)
- Goal: Compute Y_{t,*} using the universal equation

---

### Step 1: Present (Π_t) — Math Spec §1.1

Apply normalization: Π_t = Π_pal ∘ Π_pose ∘ Π_anch

**1.1 Palette Canon (Π_pal)**
- Union of all input colors: {7, 9, 4, 3, 8, 6, 2}
- All tasks use disjoint colors → palette map ≈ identity

**1.2 D4 Lex Pose (Π_pose)**
- Check all 8 D4 orientations, select lexicographically minimal
- For these simple 2×2 grids, assume canonical orientation (no rotation/reflection needed)

**1.3 Anchor to (0,0) (Π_anch)**
- Already at origin

**Result**: X̃_{t,i} ≈ X_{t,i} (presentation is approximately identity)

---

### Step 2: Shape Law (S_t) — Math Spec §3

Learn shape transformation from training examples:

- Training 1: (2×2) → (6×6) = (3·2, 3·2)
- Training 2: (2×2) → (6×6) = (3·2, 3·2)

**Shape law**: S_t(H,W) = (aH + b, cW + d) = (3H, 3W)
- Multiplicative: a = 3, c = 3
- Additive: b = 0, d = 0

**Pullback** (Π^{shape,-1}): Maps output pixel to input pixel
```
Π^{shape,-1}(i, j) = (⌊i/3⌋, ⌊j/3⌋)
```

For test (2×2 input): Output canvas is 6×6.

---

### Step 3: Truth (Q_t) — Math Spec §1.2-1.4

**Pattern Analysis** (by examining training outputs):

The output shows a **composite tiling pattern**:
- Rows 0-1: Repeat input 3 times horizontally (original)
- Rows 2-3: Repeat horizontally-flipped input 3 times
- Rows 4-5: Repeat input 3 times horizontally (original)

This suggests equivalence classes based on **(i mod 2, j mod 2, vertical_block)**

**S-views**: The structural views here are:
- Horizontal tiling (period = 2)
- Alternating vertical blocks with horizontal flip

**Must-link classes**: Pixels are equivalent based on:
- Same position within 2×2 tile (i mod 2, j mod 2)
- Same vertical block parity: (i // 2) mod 2

**Cannot-link refinement**: Training outputs prove these classes have consistent colors.

**Result**: Truth partition Q_t is defined by coordinate modulo pattern.

---

### Step 4: Law (Φ*_t) — Math Spec §2

For each equivalence class, identify the admissible functional proven by training.

**Functional type**: KEEP-via-V (input-carrying)

The view V is a composite transformation:
```
V(i, j) = {
    block_v = (i // 2) mod 2
    input_row = i mod 2

    if block_v == 0:  // original orientation
        input_col = j mod 2
    else:             // flipped orientation
        input_col = 1 - (j mod 2)

    return (input_row, input_col)
}
```

**Verification on Training 1**:
- (0,0): block_v=0, row=0, col=0 → X[0,0]=7 ✓
- (0,1): block_v=0, row=0, col=1 → X[0,1]=9 ✓
- (2,0): block_v=1, row=0, col=1 → X[0,1]=9 ✓ (flipped)
- (2,1): block_v=1, row=0, col=0 → X[0,0]=7 ✓ (flipped)

The functional is **equivariantly proven** across both training examples.

**Law**: Φ*_t = KEEP-via-V (least admissible functional)

---

### Step 5: Apply Universal Equation (UE) — Math Spec §4

For each output pixel p = (i,j) ∈ Ω^{out}:

```
Y*(i,j) = Φ*_t(V(i,j))
        = X_{t,*}[V(i,j)]
```

**Computation for Test Input** X_{t,*} = [[3,2],[7,8]]:

**Row 0** (block_v=0, input_row=0):
- j=0,2,4: col=0 → X[0][0]=3 → [3, _, 3, _, 3, _]
- j=1,3,5: col=1 → X[0][1]=2 → [_, 2, _, 2, _, 2]
- **Result**: [3, 2, 3, 2, 3, 2] ✓

**Row 1** (block_v=0, input_row=1):
- j=0,2,4: col=0 → X[1][0]=7
- j=1,3,5: col=1 → X[1][1]=8
- **Result**: [7, 8, 7, 8, 7, 8] ✓

**Row 2** (block_v=1, input_row=0, **flipped**):
- j=0,2,4: col=1-0=1 → X[0][1]=2
- j=1,3,5: col=1-1=0 → X[0][0]=3
- **Result**: [2, 3, 2, 3, 2, 3] ✓

**Row 3** (block_v=1, input_row=1, **flipped**):
- j=0,2,4: col=1 → X[1][1]=8
- j=1,3,5: col=0 → X[1][0]=7
- **Result**: [8, 7, 8, 7, 8, 7] ✓

**Row 4** (block_v=0, input_row=0):
- **Result**: [3, 2, 3, 2, 3, 2] ✓

**Row 5** (block_v=0, input_row=1):
- **Result**: [7, 8, 7, 8, 7, 8] ✓

---

### Final Answer

```
[3, 2, 3, 2, 3, 2]
[7, 8, 7, 8, 7, 8]
[2, 3, 2, 3, 2, 3]
[8, 7, 8, 7, 8, 7]
[3, 2, 3, 2, 3, 2]
[7, 8, 7, 8, 7, 8]
```

**✓ VERIFIED**: Matches expected solution exactly.

---

### Key Insights

1. **Shape law** is multiplicative: (H,W) → (3H, 3W)
2. **Truth partition** is based on modulo arithmetic: (i mod 2, j mod 2, block parity)
3. **Law is KEEP-via-V** where V is a composite view (tiling + conditional flip)
4. **Equivariant proof**: The same functional law holds across all training examples
5. **One-pass computation**: No iteration needed, direct application of UE

---

## Example 2: Task 05269061 (Diagonal Tiling with Color Extraction)

### Task Description
- **Input size**: 7×7
- **Output size**: 7×7 (identity shape)
- **Training pairs**: 3
- **Pattern**: Extract non-zero colors from diagonal pattern, tile output with diagonal stripes

### Training Data

**Training Example 1:**
```
Input:                          Output:
[0, 0, 0, 0, 0, 0, 0]          [2, 4, 1, 2, 4, 1, 2]
[0, 0, 0, 0, 0, 0, 0]          [4, 1, 2, 4, 1, 2, 4]
[0, 0, 0, 0, 0, 0, 1]          [1, 2, 4, 1, 2, 4, 1]
[0, 0, 0, 0, 0, 1, 2]          [2, 4, 1, 2, 4, 1, 2]
[0, 0, 0, 0, 1, 2, 4]          [4, 1, 2, 4, 1, 2, 4]
[0, 0, 0, 1, 2, 4, 0]          [1, 2, 4, 1, 2, 4, 1]
[0, 0, 1, 2, 4, 0, 0]          [2, 4, 1, 2, 4, 1, 2]

Non-zero colors: {1, 2, 4}
```

**Training Example 2:**
```
Input:                          Output:
[2, 8, 3, 0, 0, 0, 0]          [2, 8, 3, 2, 8, 3, 2]
[8, 3, 0, 0, 0, 0, 0]          [8, 3, 2, 8, 3, 2, 8]
[3, 0, 0, 0, 0, 0, 0]          [3, 2, 8, 3, 2, 8, 3]
[0, 0, 0, 0, 0, 0, 0]          [2, 8, 3, 2, 8, 3, 2]
[0, 0, 0, 0, 0, 0, 0]          [8, 3, 2, 8, 3, 2, 8]
[0, 0, 0, 0, 0, 0, 0]          [3, 2, 8, 3, 2, 8, 3]
[0, 0, 0, 0, 0, 0, 0]          [2, 8, 3, 2, 8, 3, 2]

Non-zero colors: {2, 3, 8}
```

**Training Example 3:**
```
Input:                          Output:
[0, 0, 0, 0, 8, 3, 0]          [4, 8, 3, 4, 8, 3, 4]
[0, 0, 0, 8, 3, 0, 0]          [8, 3, 4, 8, 3, 4, 8]
[0, 0, 8, 3, 0, 0, 0]          [3, 4, 8, 3, 4, 8, 3]
[0, 8, 3, 0, 0, 0, 4]          [4, 8, 3, 4, 8, 3, 4]
[8, 3, 0, 0, 0, 4, 0]          [8, 3, 4, 8, 3, 4, 8]
[3, 0, 0, 0, 4, 0, 0]          [3, 4, 8, 3, 4, 8, 3]
[0, 0, 0, 4, 0, 0, 0]          [4, 8, 3, 4, 8, 3, 4]

Non-zero colors: {3, 4, 8}
```

**Test Input:**
```
[0, 1, 0, 0, 0, 0, 2]
[1, 0, 0, 0, 0, 2, 0]
[0, 0, 0, 0, 2, 0, 0]
[0, 0, 0, 2, 0, 0, 0]
[0, 0, 2, 0, 0, 0, 0]
[0, 2, 0, 0, 0, 0, 4]
[2, 0, 0, 0, 0, 4, 0]
```

---

### Step 0: Notation (Math Spec §0)

- Task t = 05269061
- Training pairs: i = 1, 2, 3
- All grids: 7×7 → 7×7
- X_{t,*} = test input (shown above)
- Non-zero colors in test: {1, 2, 4}

---

### Step 1: Present (Π_t) — Math Spec §1.1

**1.1 Palette Canon**
- Union of input colors: {0, 1, 2, 3, 4, 8}
- Color 0 is "background" (fills most of grid)
- Other colors form sparse patterns
- Palette map: identity (no normalization needed)

**1.2 D4 Lex Pose**
- Grids are 7×7, check for canonical orientation
- For this task, assume canonical (diagonal patterns are asymmetric)

**1.3 Anchor**
- Already at origin

**Result**: X̃_{t,i} = X_{t,i} (identity presentation)

---

### Step 2: Shape Law (S_t) — Math Spec §3

Learn shape transformation:
- Training 1: 7×7 → 7×7
- Training 2: 7×7 → 7×7
- Training 3: 7×7 → 7×7

**Shape law**: S_t(H,W) = (H, W) — **identity shape**

**Pullback**: Π^{shape,-1}(i,j) = (i,j) — identity mapping

For test: Output canvas is 7×7 (same as input).

---

### Step 3: Truth (Q_t) — Math Spec §1.2-1.4

**Pattern Analysis**:

Observing all training outputs, each output is a **diagonal stripe pattern** where:
- Pixels on the same diagonal (i+j = constant) have the same color
- The pattern repeats every 3 diagonals
- The colors come from the non-zero values in the input

**S-views (§1.2)**:
The structural view is **diagonal equivalence**:
- Pixels belong to the same class if (i+j) mod 3 is equal
- This is a "lawful forgetting" — the input proves this structure

**Must-link (§1.3)**:
Define equivalence relation on test canvas:
```
(i₁,j₁) ∼ (i₂,j₂)  ⟺  (i₁+j₁) mod 3 = (i₂+j₂) mod 3
```

This gives us **3 equivalence classes**:
- **Class 0**: {(i,j) : (i+j) mod 3 = 0}
- **Class 1**: {(i,j) : (i+j) mod 3 = 1}
- **Class 2**: {(i,j) : (i+j) mod 3 = 2}

**Cannot-link (§1.4)**:
Training outputs prove that within each class, all pixels have the same color (no contradictions).

**Result**: Truth partition Q_t = {Class 0, Class 1, Class 2}

---

### Step 4: Law (Φ*_t) — Math Spec §2

For each class a ∈ Q_t, find the admissible functional.

**Step 4.1: Extract Non-Zero Colors from Test Input**

Scanning test input for non-zero values:
```
Position → Color → Diagonal (i+j) mod 3
(0,1) → 1 → 1
(1,0) → 1 → 1
(0,6) → 2 → 6 mod 3 = 0
(1,5) → 2 → 6 mod 3 = 0
(2,4) → 2 → 6 mod 3 = 0
(3,3) → 2 → 6 mod 3 = 0
(4,2) → 2 → 6 mod 3 = 0
(5,1) → 2 → 6 mod 3 = 0
(6,0) → 2 → 6 mod 3 = 0
(5,6) → 4 → 11 mod 3 = 2
(6,5) → 4 → 11 mod 3 = 2
```

**Step 4.2: Map Colors to Diagonal Classes**

From the non-zero positions:
- Diagonal class 0: contains color **2**
- Diagonal class 1: contains color **1**
- Diagonal class 2: contains color **4**

**Step 4.3: Verify Against Training Examples**

Check Training 1: non-zero colors {1, 2, 4}
- Appears along diagonal going down-right
- Output assigns: class 0→2, class 1→4, class 2→1 (different permutation!)

Wait, let me re-examine... Looking more carefully at Training 1 output:
```
[2, 4, 1, 2, 4, 1, 2]   ← Row 0: starts with 2
[4, 1, 2, 4, 1, 2, 4]   ← Row 1: starts with 4
[1, 2, 4, 1, 2, 4, 1]   ← Row 2: starts with 1
```

The pattern is determined by which color appears at which diagonal position in the **input**.

**Functional Type**: **CONST per class**

The admissible functional for each class is:
```
Φ*_t(class_k) = CONST(color_for_class_k)
```

where `color_for_class_k` is determined by:
- Extract all non-zero colors from input
- Map each non-zero color to its diagonal class based on where it appears in input
- Assign that color as the CONST value for that class

**Law**: Φ*_t(a) = CONST(c_a) where c_a is the color proven by input for diagonal class a

---

### Step 5: Apply Universal Equation (UE) — Math Spec §4

For each output pixel p = (i,j):

```
diagonal_class = (i + j) mod 3
color = Φ*_t(diagonal_class)
      = CONST(color_for_that_class)
```

**Mapping for Test**:
- Class 0 (diagonals 0,3,6,9,...): color = **2**
- Class 1 (diagonals 1,4,7,10,...): color = **1**
- Class 2 (diagonals 2,5,8,11,...): color = **4**

**Computation**:

**Row 0** (i=0):
- j=0: (0+0)%3=0 → 2
- j=1: (0+1)%3=1 → 1
- j=2: (0+2)%3=2 → 4
- j=3: (0+3)%3=0 → 2
- j=4: (0+4)%3=1 → 1
- j=5: (0+5)%3=2 → 4
- j=6: (0+6)%3=0 → 2
- **Result**: [2, 1, 4, 2, 1, 4, 2] ✓

**Row 1** (i=1):
- Diagonal classes: 1,2,0,1,2,0,1
- **Result**: [1, 4, 2, 1, 4, 2, 1] ✓

**Row 2** (i=2):
- Diagonal classes: 2,0,1,2,0,1,2
- **Result**: [4, 2, 1, 4, 2, 1, 4] ✓

**Row 3** (i=3):
- Diagonal classes: 0,1,2,0,1,2,0
- **Result**: [2, 1, 4, 2, 1, 4, 2] ✓

**Row 4** (i=4):
- Diagonal classes: 1,2,0,1,2,0,1
- **Result**: [1, 4, 2, 1, 4, 2, 1] ✓

**Row 5** (i=5):
- Diagonal classes: 2,0,1,2,0,1,2
- **Result**: [4, 2, 1, 4, 2, 1, 4] ✓

**Row 6** (i=6):
- Diagonal classes: 0,1,2,0,1,2,0
- **Result**: [2, 1, 4, 2, 1, 4, 2] ✓

---

### Final Answer

```
[2, 1, 4, 2, 1, 4, 2]
[1, 4, 2, 1, 4, 2, 1]
[4, 2, 1, 4, 2, 1, 4]
[2, 1, 4, 2, 1, 4, 2]
[1, 4, 2, 1, 4, 2, 1]
[4, 2, 1, 4, 2, 1, 4]
[2, 1, 4, 2, 1, 4, 2]
```

**✓ VERIFIED**: Matches expected solution exactly.

---

### Key Insights

1. **Shape law** is identity: (H,W) → (H,W)
2. **Truth partition** based on diagonal equivalence: (i+j) mod 3
3. **Law is CONST** per class: each diagonal class gets a constant color
4. **Color assignment** determined by extracting non-zero colors from input and mapping to diagonal positions
5. **Input-only proof**: The sparse non-zero pattern in input serves as "witness" for which color belongs to which diagonal class
6. **Compositional structure**: Extract → Map → Tile
7. **Background elimination**: Color 0 (background) is filtered out; only non-zero colors form the output pattern

---

## Summary: How to Apply Math Spec by Hand

### General Recipe

1. **Present (§1.1)**: Normalize inputs via palette, D4 pose, anchor
   - Often identity for simple tasks
   - Critical for tasks with symmetry or color permutation

2. **Shape (§3)**: Identify transformation (H,W) → (H',W')
   - Check: multiplicative (aH, cW), additive (+b, +d), or content-based
   - Compute pullback: how output pixels map back to input pixels

3. **Truth (§1.2-1.4)**: Find equivalence classes on output canvas
   - **S-views**: What structural equalities does input prove? (symmetries, periods, translations)
   - **Must-link**: Build equivalence classes via closure
   - **Cannot-link**: Refine using training outputs to eliminate contradictions
   - Result: coarsest partition Q_t that's consistent with training

4. **Law (§2)**: For each class, find least admissible functional
   - **KEEP-via-V**: Does output copy from input through some view V?
   - **CONST c**: Does output assign constant color c?
   - **Reducers**: ARGMAX, UNIQUE, RECOLOR, BLOCK, etc.
   - Choose minimum via total order: KEEP ≺ RECOLOR ≺ BLOCK ≺ REDUCE ≺ CONST

5. **Compute (§4)**: Apply universal equation Y*(p)
   - For each output pixel p:
     - Find its pullback p^← via shape law
     - Determine its truth class a via Q_t
     - Apply class functional Φ*_t(a)
   - One-pass computation (no iteration needed)

### Pattern Recognition Tips

| Pattern Type | Shape Law | Truth Classes | Functional Type |
|-------------|-----------|---------------|-----------------|
| Tiling | Multiplicative (aH, cW) | Modulo coordinates | KEEP-via-V |
| Symmetry | Identity or small scale | Orbit under symmetry group | KEEP-via-V |
| Diagonal stripes | Identity | (i+j) mod k | CONST or KEEP |
| Object detection | Content-based (bbox) | Connected components | CONST or REDUCE |
| Color mapping | Identity | Per-pixel | RECOLOR |
| Flooding/filling | Identity | Connected components | CONST (from seed) |

### Debugging Checklist

- [ ] Are training sizes consistent with proposed shape law?
- [ ] Do truth classes respect training output consistency (no contradictions)?
- [ ] Is the functional equivariantly proven across all training examples?
- [ ] Does the law choice respect the total order (is it truly minimal)?
- [ ] Does output match training examples when applied to training inputs?
- [ ] Are edge cases handled (undefined pullback, empty classes)?

---

## References

- **Math Spec**: `docs/anchors/00-math-spec.md`
- **Universal Equation (UE)**: §4 of math spec
- **Training Data**: `data/arc-agi_training_challenges.json`
- **Solutions**: `data/arc-agi_training_solutions.json`
