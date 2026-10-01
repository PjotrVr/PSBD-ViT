# TaCT detection follows trigger sufficiency, a refinement registered before the reading

## Why this file exists

The TaCT calibration experiment pre-registered that the plain method would miss the 6 held-out multi-source TaCT models (`experiments/tact_calibration/PREDICTIONS.md`, P2). It did not: PSBD-TM reads TPR at 10% FPR between 0.68 and 0.99 on all 6 at k = 3. That prediction failed and stays failed. The evidence surplus account had treated TaCT as a conjunction of trigger and source content as a class. The records already on disk suggest the attack label is the wrong unit, and that what matters is whether the trigger is sufficient on its own.

## What was seen before writing this (2026-10-01, 23:30)

| model | blank-carrier excess (trigger on content-free carriers sent to the target) | PSBD reading |
|---|---|---|
| `vit_cifar10_tact_0_01` (1 source class) | -0.03 | fails, TPR at 10% FPR 0.01 |
| `vit_gtsrb_tact_0_05` (1 source class) | 0.95 | partial, TPR at 10% FPR 0.50 |
| `resnet18_gtsrb_tact_0_05_src6` | 0.86 | detected (PSBD on ResNet-18) |
| `resnet18_gtsrb_tact_0_1_src12` | 0.98 | detected |
| `resnet18_cifar10_tact_0_05_src3` | 1.00 | not yet read |
| `resnet18_cifar10_tact_0_01` (1 source class) | 0.23 | not yet read |

The blank-carrier excess is the `blank.excess` field of `experiments/why_psbd_works/sufficiency.py`. All of these models refuse the trigger on non-source images (stamped non-source images sent to the target at most 0.19). So TaCT is learned in 2 forms. In the AND form the trigger needs the source content and is not sufficient alone. In the veto form the trigger is sufficient alone and only non-source content vetoes it. Masking removes the veto evidence and leaves the trigger, so the veto form keeps surplus and PSBD should catch it.

## Prediction

The blank-carrier excess is measured next on the 8 ViT TaCT models whose excess has never been read: `vit_cifar10_tact_0_05`, `vit_gtsrb_tact_0_01_cos` and the 6 held-out multi-source models (`vit_cifar100_tact_0_01_src5`, `vit_cifar100_tact_0_05_src25`, `vit_cifar10_tact_0_1_src5`, `vit_gtsrb_tact_0_1_src12`, `vit_tiny_tact_0_01_src10`, `vit_tiny_tact_0_05_src50`). Their PSBD-TM readings are known. Their excess is not.

1. The 6 held-out multi-source models, all detected at TPR at 10% FPR of 0.68 or more, read a blank-carrier excess of at least 0.5.
2. `vit_cifar10_tact_0_05` (TPR at 10% FPR 0.00) reads an excess below 0.2.
3. Over all 10 ViT TaCT models, the blank-carrier excess and PSBD-TM's TPR at 10% FPR (adaptive rule, k = 3) have Spearman correlation at least 0.6.

`vit_gtsrb_tact_0_01_cos` has no prediction. PSBD-TM fails on it while its triggered inputs score as more stable than their clean twins, and every flip lands on 1 attractor class, so it may fail for a reason outside this account.

## Refutation

The refinement is refuted if prediction 1 fails on 2 or more of the 6 models, or prediction 3 fails. If prediction 2 alone fails, the AND form is not the only cause of the single-source failures and the account is reported as partial.

## Honest status

This refinement was formed after seeing the 6 records in the table above, so it is post hoc for those models. It is a genuine prediction only for the 8 excess values listed above, which nobody has measured. The paper may cite it as predicted only for those 8.
