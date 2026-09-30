---
name: faded-trigger-memo
description: results/research/faded-trigger-defenses.md (2026-09-30) surveys defenses against test-time faded triggers for PSBD-TM, with the surplus window bound and ranked ideas
metadata:
  type: reference
---

The memo lives at `results/research/faded-trigger-defenses.md` in the PSBD-ViT repo. The caller asked for
`docs/faded-trigger-defenses.md`, but this agent may only write under `results/research/`, so the
coordinator has to move it.

Non-obvious facts established while writing it, as of 2026-09-30:
- Direct precedent: Lin et al., USENIX Security 2025 (TITIM). Fading to opacity 0.7 keeps ASR
  91.62% while STRIP falls from 0.99 to 0.80 and SCALE-UP from 0.96 to 0.62. Mixed-intensity
  training (1.0 plus 0.1) is a non-adaptive attack.
- Riaz and Yu, arXiv 2608.27288, is the low-ASR, per-input-selection version.
- The dose records split into a collapse regime (CIFAR-10 Blend and BPP, and CIFAR-10 WaNet at
  every dose) and a gate regime (BadNets, LF, CIFAR-100 Blend, Tiny WaNet).
- The 1% FPR window on CIFAR-10 and GTSRB is mostly the modal masking attractor. Between 4.5% and
  8.4% of clean validation images never flip up to rate 0.9, all from 1 non-target class per
  model. On CIFAR-100 and Tiny the share is 0.3% to 0.5%.
- The hit-only AUROC has no 0.5 null. The benign CIFAR-10 probes read 0.61 to 0.82.
- TaCT's twin AUROC of 0.978 comes with a TPR near 0 at every dose.
- Additive bound: a_det / a_fire is at most S_q = 1 / (1 - p*_(1-q)).
- Top ranked ideas: a class-conditional and 2-sided threshold, a matched filter along the
  backdoor direction, and partial-evidence (CutMix or Adaptive-Blend) training.

**How to apply:** start any question about faded or weak triggers from this memo. See
[[evidence-surplus-theory]] and [[nonadaptive-attack-request]].
