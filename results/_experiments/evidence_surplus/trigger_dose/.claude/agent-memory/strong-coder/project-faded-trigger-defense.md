---
name: project-faded-trigger-defense
description: experiments/faded_trigger_defense (pooled MW monitor, ladder scores, per-class conformal and 2-sided rules, label-free matched filter, sharpening) — regimes, traps, GPU queue behind final_method
metadata:
  type: project
---

Built 2026-09-30 (uncommitted at hand-off): `score.py` (GPU, per-input PSBD-TM scores on
faded triggers, doses 1.0 to 0.4, 1024 pairs, every ladder rate; validation and clean test
read from the canonical cache), `analyze.py` (CPU: pooled Mann-Whitney monitor, ladder
scores, sharpening readout, `judge()` of PREDICTIONS.md), `render.py`, `gpu_driver.sh`
(waits for 17:00 and `scratch/gpu_done_final_method`, override with FADED_AFTER).

Facts worth keeping:
- The trigger dose code's linear fade is inline in `measure()`; only `warped()` (WaNet) and
  `DOSES` are importable. `faded_images` restates the formula, a test pins it.
- A pooled test against a FIXED finite reference is miscalibrated conditionally: synthetic
  null batches of 20 against about 200 reference scores alarmed at more than twice the level.
  The error grows with n. Always report the realized false alarm on clean queries.
- The asymptotic MW p-value can never reach 0.01 at n = 1 (floor near 0.04), so n = 1 uses
  the exact rank p-value (floor(u) + 1) / (m + 1).
- CIFAR-100, Tiny and GTSRB target 0 have fewer than 30 validation images per class, so the
  pooled reference falls back to all classes there. Sharpening at native 32 px costs more
  than 1 point of clean accuracy fast, so LAMBDAS starts at 0.1.
- CPU smoke of 1 entry at 8 pairs and 2 rates takes about 1 minute per part.
- After docs/faded-trigger-defenses.md the coordinator required: results per regime (gate:
  cifar10 badnet, tiny badnet, gtsrb lf, cifar100 blend; collapse: cifar10 blend, bpp),
  benign attacked class = probe target 0 (hit AUROC has no 0.5 null, compare with
  target-class clean), per-class and 2-sided conformal thresholds (distinct from the refuted
  shrinkage calibration), and a matched filter from PSBD-TM catches in an unlabeled stream
  (split pairs and clean rows into stream and held-out halves; features part = head input).
- The pre-registration clock: run `date` before stamping a time, an early stamp was guessed.

**Why:** the coordinator's task of 2026-09-30, GPU window from 17:00 at memory fraction 0.15.
**How to apply:** rerun `analyze.py --readme experiments/faded_trigger_defense/README.md`
after the records land. Related: [[project-final-method-evaluation]]
