# Faded trigger defenses, pre-registered predictions

Written on 2026-09-30 at 03:45, before any per-input score of this experiment existed. The only readings seen before writing were the aggregate dose records of `experiments/evidence_surplus/trigger_dose/`, which the coordinator summarized and which these predictions use by name. The bars below are the constants `judge()` in `analyze.py` reads, so a verdict is computed and never judged by eye.

The attack set of a model at a dose is its faded triggered inputs that the model sends to the attacked class, which is the target on a backdoored model. The weakest firing dose is the smallest dose of 1.0, 0.8, 0.6, 0.5 and 0.4 at which at least half of the faded triggered inputs reach the target. Every TPR is reported beside the FPR it realizes on clean images the threshold never saw.

## Per-input pass

The per-input pass reads the same thing the dose experiment read, so it has to agree with it before any fix is read.

- C1. On the first 256 pairs the AUROC of the faded triggered images against their clean twins lies within 0.05 of the dose record at every dose the 2 share, on every backdoored model.
- C2. Fresh passes and the canonical cache agree on the same images: the AUROC of fresh scores against cached scores of the same twins and, at dose 1.0, of the same triggered images lies within 0.05 of 0.5.

## Query pooling

A backdoor that fires on many queries hands the monitor many draws from 1 shifted distribution, so a weak per-input shift adds up over a batch. The test is a 1-sided Mann-Whitney test against the clean validation scores of the predicted class, with the all-class fallback below 30 validation images. On a backdoored model the target class's own clean images are more robust than other classes' (the class calibration result of `experiments/final_method/`), so the class reference is calibrated but weaker, and the fallback is stronger but miscalibrated for the target class.

- P1. At the weakest firing dose a batch of 20 attack-set queries raises an alarm at level 0.05 in at least 0.9 of draws, on every backdoored model except `vit_cifar10_blend_0_1`.
- P2. Batches of 20 clean test queries predicted as the target raise alarms at level 0.05 in at most 0.10 of draws on every backdoored model with a class reference. On at least 1 backdoored model on the all-class fallback that rate exceeds 0.10.
- P3. On `vit_cifar10_blend_0_1` pooling finds nothing at doses 0.6 and 0.5, where the per-input AUROC is below chance: at most 0.10 of 20-query batches alarm at level 0.05 at either dose where the backdoor still fires.
- P4. A batch of 50 queries of which 10 are attack-set queries and 40 are clean queries of the target raises an alarm at level 0.05 in at least 0.5 of draws at the weakest firing dose, on the models of P1.
- P5. On the benign controls batches of 20 probe-triggered queries of the attacked class alarm at level 0.05 in at most 0.10 of draws at every dose.

## Multi-rate scoring

A faded trigger carries less surplus, so its answer breaks at a lower rate than at full dose but may still outlast a clean answer above the adaptive rate. A score over the whole ladder reads that survival wherever it happens.

- M1. The area under the fractional PSU curve raises TPR at 5% FPR by at least 0.05 over the single-rate score at the weakest firing dose on at least 4 of the 6 backdoored models, and at dose 1.0 it loses no more than 0.05 on any of them.
- M2. With 3 passes the keep curve takes 4 values per rate, so the keep-curve area and the critical rate are heavily tied, and their realized FPR at the 0.01 quantile falls outside 0.005 to 0.02 for both scores on at least half of the entries.
- M3. On the benign controls every multi-rate score's TPR on the attacked class lies within 0.10 of its realized FPR at the 0.05 quantile at every dose.

## Residual amplification, exploratory

Sharpening, x' = x + lambda (x - blur(x)), is linear, so a faded additive trigger d delta becomes d (delta + lambda (delta - blur(delta))). A trigger whose energy is high-frequency is amplified by about 1 + lambda, restoring part of the dose, while clean content is amplified only as far as clean accuracy allows. The Blend trigger here is a seeded random pattern, BadNets a sharp-edged patch and BPP a quantization step, all high-frequency. LF is low-frequency by construction. This part is exploratory whatever its outcome.

- A1. A lambda of at least 0.5 keeps clean validation accuracy within 1 point on every entry.
- A2. At the weakest firing dose sharpening raises TPR at 5% FPR by at least 0.10 over the plain pipeline on at least 4 of the 5 Blend, BadNets and BPP models. On the LF model the gain stays below 0.10.
- A3. On the benign controls the sharpened AUROC of probe-triggered queries against clean twins lies within 0.10 of 0.5 at every dose.

## Note added before any real score

Added on 2026-09-30 at 03:50, after a timing run of `analyze.py` on synthetic scores and before any score of a real model existed. The bars above are unchanged. On synthetic scores with no shift at all, batches of 20 drawn from 1 fixed clean pool alarmed at level 0.05 in more than twice that share of draws against a class reference of 200 scores. The reference is a finite sample whose own deviation is shared by every draw, and that deviation grows in relative weight as n grows. P2 may therefore fail on the class reference for this reason alone, which the readout reports as a failed prediction rather than a recalibration.

## Additions after the literature memo

Added on 2026-09-30 at 03:56, after reading `docs/faded-trigger-defenses.md` and before any score of a real model existed. The predictions above stand unchanged. The memo splits the backdoored models into 2 regimes, fixed here before any reading. The gate regime, where inputs that still fire keep their per-input signal, is `vit_cifar10_badnet_a2o_0_01`, `vit_tiny_badnet_a2o_0_05`, `vit_gtsrb_lf_0_01` and `vit_cifar100_blend_0_1`. The collapse regime, where fading removes the signal, is `vit_cifar10_blend_0_1` and `vit_cifar10_bpp_0_05`. Every result is reported per regime. On a benign control the attacked class is now the probe's own target class 0, the same rule as on a backdoored model, so its attack set is the probe-triggered images sent to class 0.

Query pooling per regime, from the memo's power table.

- R1. At the weakest firing dose batches of 10 attack-set queries alarm at level 0.01 in at least 0.9 of draws on every gate model.
- R2. On `vit_cifar10_bpp_0_05` at the weakest firing dose batches of 20 alarm at level 0.01 in at least 0.9 of draws.
- R3. On `vit_cifar10_blend_0_1` at every firing dose of 0.6 or less, batches of 50 alarm at level 0.01 in at most 0.10 of draws.

The null of the hit AUROC. Images a trigger sends to the target score lower than their twins even on a benign model, so the AUROC of the attack set is read against clean test images predicted as the target as well.

- B1. On every gate model the attack set's AUROC against clean test images predicted as the target is at least 0.8 at the weakest firing dose.
- B2. On every benign control with at least 20 probe-triggered images sent to class 0, their AUROC against clean test images predicted as class 0 lies within 0.15 of 0.5 at dose 1.0.

Per-class and 2-sided thresholds, the memo's first inference idea. Each input gets a conformal p-value against the clean validation scores of its predicted class, `p = (1 + #{v <= s}) / (n_c + 1)`, and is flagged when `p <= q`. The 2-sided form flags when the smaller of the 2 tail p-values is at most `q / 2`. A class with fewer validation images than the level can resolve (`n_c + 1 < 1 / q`, or `2 / q` for the 2-sided form) falls back to every validation score. Every rule is fixed on clean validation only. The rule differs from the refuted calibration of `experiments/final_method/`, which shrank each class toward the pooled distribution and then set 1 pooled threshold, since here each class keeps its own level and no shrinkage is fitted.

- K1. On the 4 CIFAR-10 and GTSRB models the per-class 1-sided rule raises the hit TPR at 1% FPR over the canonical threshold by at least 0.05 on average at the weakest firing dose. A smaller mean gain refutes the attractor account of the 1% window.
- K2. Every per-class rule realizes an FPR on the clean test split of at most 0.015 at the level 0.01.
- K3. The pooled 2-sided rule raises the hit TPR at 1% FPR over the canonical threshold by at least 0.05 on `vit_cifar10_blend_0_1` at dose 0.5 and by less than 0.05 at dose 0.6.
- K4. On `vit_cifar100_blend_0_1` and `vit_tiny_badnet_a2o_0_05` the per-class 1-sided rule changes the hit TPR at 1% and at 5% FPR by at most 0.02 at every firing dose.

A matched filter along the backdoor direction, the memo's second inference idea, is an extension with its own threat model. The defender sees an unlabeled stream of queries of which some are full-strength triggered, here the even rows of the clean test split and the dose 1.0 triggered images of the first half of the pairs. PSBD-TM flags the stream at the canonical 1% threshold, and the suspected target is the predicted class with the most flagged queries. The direction is the mean last-block class token of the flagged queries of that class minus the mean over clean validation images predicted as it. Every input is scored by its negated projection on that direction, so low means poisoned. The score is thresholded at quantiles of clean validation. It is read on the second half of the pairs at every dose and on the odd clean test rows, which the stream never held. No label enters the estimate. It uses triggered images only through the unlabeled stream, which is the extension's threat model.

- F1. The suspected target is the true target on all 6 backdoored models.
- F2. The matched filter's hit AUROC against clean test images predicted as the target is at least 0.9 at every firing dose on all 6 backdoored models. It is refuted if that AUROC is below 0.75 on `vit_cifar10_blend_0_1` at dose 0.6.
- F3. At the weakest firing dose the matched filter's hit TPR at 1% FPR exceeds the canonical single-rate TPR on the same pairs on at least 4 of the 6 backdoored models.
- F4. On the benign controls its TPR on the attack set lies within 0.10 of its realized FPR at the 0.05 quantile at every dose.
