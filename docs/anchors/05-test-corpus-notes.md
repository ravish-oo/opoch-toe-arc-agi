# Microsuite (6 tasks, one-line purpose each)

- 00576224 — Periodic tiling with flips; multiplicative shape (2×2→6×6); KEEP via tile_alt_row/col_flip.
- 007bbfb7 — Block stamping via non-zero mask; KEEP with 3× zoom; class partition by nonzero.
- 272f95fa — Bands + grid walls; CONST per band; PT uses S-view images and component masks.
- 00d62c1b — Copy–move with overlap; distinct Δ per component; KEEP with translations.
- 045e512c — RECOLOR π per class; stable relabel learned across trainings.
- 0a2355a6 — Mixed shape (additive + multiplicative) with partial pullback; KEEP+CONST composition.

Notes:
- Each task must pass: truth single-valuedness, exact selection, 100% paint coverage, determinism check.
- Replace any ID if your local corpus differs; keep the purpose categories intact.
