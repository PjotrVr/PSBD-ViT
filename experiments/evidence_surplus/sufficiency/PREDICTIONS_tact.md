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

## Result (2026-10-01, 23:55)

Measured with `python -m experiments.why_psbd_works.sufficiency --output-slug evidence_surplus` on the 8 models, records under `results/_experiments/evidence_surplus/sufficiency/`. TPR is PSBD-TM's at 10% FPR, adaptive rule, k = 3 (`experiments/tact_calibration` for the 6 held-out models, `psbd_metrics.json` for the rest). The content-hidden excess is the `content.excess` field: the triggered image with every non-trigger token masked at the attention input of every block, against its clean twin under the same mask.

| model | source classes | blank-carrier excess | content-hidden excess | TPR at 10% FPR |
|---|---|---|---|---|
| `vit_cifar10_tact_0_01` | 1 | -0.031 | 0.000 | 0.005 |
| `vit_cifar10_tact_0_05` | 1 | 0.422 | 0.000 | 0.000 |
| `vit_gtsrb_tact_0_05` | 1 | 0.953 | 0.000 | 0.502 |
| `vit_gtsrb_tact_0_01_cos` | 1 | 0.000 | 0.000 | 0.000 |
| `vit_cifar100_tact_0_01_src5` | 5 | 0.031 | 1.000 | 0.796 |
| `vit_cifar100_tact_0_05_src25` | 25 | 1.000 | 1.000 | 0.683 |
| `vit_cifar10_tact_0_1_src5` | 5 | 0.094 | 1.000 | 0.990 |
| `vit_gtsrb_tact_0_1_src12` | 12 | 0.984 | 0.000 | 0.978 |
| `vit_tiny_tact_0_01_src10` | 10 | 0.031 | 1.000 | 0.825 |
| `vit_tiny_tact_0_05_src50` | 50 | 0.984 | 1.000 | 0.859 |

Verdicts.

1. Fails. 3 of the 6 held-out models read a blank-carrier excess below 0.5 (0.031, 0.094, 0.031) and are detected anyway.
2. Fails. `vit_cifar10_tact_0_05` reads 0.422, not below 0.2.
3. Fails. Spearman 0.373 over the 10 models, below 0.6.

The refinement is refuted as registered. Blank-carrier sufficiency does not predict TaCT detection.

## What the readings show instead (post hoc, a hypothesis)

The content-hidden excess separates the single-source from the multi-source models. It is 0 on all 4 single-source models and 1 on 5 of the 6 multi-source ones. The 6th (`vit_gtsrb_tact_0_1_src12`) reads 0 there and 0.984 on blank carriers. The content-hidden test is the closer analogue of PSBD-TM, which also masks tokens at the attention input. On the multi-source models the trigger tokens alone keep the target once the rest of the image is masked, so the decision keeps its surplus under the same removal PSBD-TM applies. On the single-source models the decision needs the source content, which masking removes.

This reading was found after seeing the table, so it is not a confirmed result. It needs fresh TaCT models whose content-hidden excess is predicted before it is measured. The 6 held-out models are spent for this purpose, since their excess is now read. The next chances are new TaCT trainings with 2 to 4 source classes, where the 2 forms should meet, and the veto attack of the evidence-surplus training jobs (R7).
