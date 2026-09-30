# Predictions for PSBD on ResNet-18 TaCT

Written and committed on 2026-09-30 before any ResNet-18 TaCT model was trained and before any new measurement was read. The only ResNet-18 numbers seen before writing are the ones already in the repository: `docs/psbd-reproduction.md` (the upstream reproduction, where GTSRB Blend reads TPR 0.677 at the rate the paper's rule picked and 0.966 one rung lower), the `adaptive` block of `results/resnet18_gtsrb_badnet_a2o_0_1/psbd_metrics.json` (absolute PSU, TPR 1.0 at the 0.25 quantile) and the ResNet rows of the trigger sufficiency table in `experiments/why_psbd_works/README.md`. The ViT TaCT readings that motivate the test come from `experiments/low_fpr_audit/README.md`.

## Hypothesis under test

The evidence surplus account (`docs/evidence-surplus-theory.md`) says PSBD catches a triggered input when the trigger alone carries more evidence for the target than the decision needs, so random removal leaves the target standing while a clean decision falls. A TaCT decision needs the trigger and the source-class content together. It has no spare evidence and should break under random removal the way a clean decision does. On ViT PSBD-TM reads a high AUROC on TaCT but TPR near 0 at a 1% and 5% false-positive budget on 3 of 4 panel models. If the account is right this is a property of the attack and not of the architecture, so the original PSBD on ResNet-18 (dropout after the residual add, `post_residual`, absolute PSU, the paper's adaptive rate rule at 0.8) fails on TaCT too while it works on BadNets and Blend trained on the same recipe.

## Models

All models use the recipe of the existing ResNet-18 reproductions (`checkpoints/resnet18_gtsrb_badnet_a2o_0_1/args.json`): ResNet-18 with the CIFAR stem from random initialization, 100 epochs of Adam at the constant learning rate 1e-4 with weight decay 1e-4, batch 128, no augmentation, seed 0, target class 0 and the full training split. The CIFAR-10 smoke model (`resnet18_cifar10_badnet_a2o_0_1_smoke`, 15 epochs on 20000 samples, clean accuracy 0.54) is not used.

TaCT keeps its defaults (a 3 pixel checkerboard in the bottom right corner, cover rate 0.01). Source classes are 1..k, with k the smallest count whose share of the training split is at least 5 times the poison rate, the rule of `docs/runs/2026-09-24-tact-multisource.md`, so about a fifth of the source pool is poisoned and no source class is poisoned whole.

| folder | dataset | poison rate | source classes | poisoned share of source pool |
|---|---|---:|---|---:|
| resnet18_gtsrb_tact_0_05_src6 | gtsrb | 0.05 | 1..6 | 0.195 |
| resnet18_gtsrb_tact_0_1_src12 | gtsrb | 0.1 | 1..12 | 0.199 |
| resnet18_cifar10_tact_0_01 | cifar10 | 0.01 | 1 | 0.1 |
| resnet18_cifar10_tact_0_05_src3 | cifar10 | 0.05 | 1..3 | 0.167 |

The controls are `resnet18_gtsrb_badnet_a2o_0_1` and `resnet18_gtsrb_blend_0_1` (existing) and a new `resnet18_cifar10_badnet_a2o_0_1`. The benign references are new, `resnet18_gtsrb_benign` and `resnet18_cifar10_benign`.

A model enters the verdicts only when it is a successful backdoor: ASR at least 0.85 on its own eval set, clean accuracy within 2 points of its benign reference and, for TaCT, clean accuracy on the source classes at least 0.5 (below that the model maps the source class to the target without the trigger). A model that fails a bar is reported and not retrained. A verdict on P1 needs at least 3 successful TaCT models.

## Statistic

The primary statistic is the original PSBD. Dropout at `post_residual`, 3 forward passes, the full 16 rate ladder of the reproductions, the adaptive rate (the smallest rate whose clean-validation shift ratio reaches 0.8) and absolute PSU, with the threshold at a quantile of the 2000-image clean validation split. TPR at the 0.10 quantile is read as TPR at 10% FPR. TPR at 0.01 and 0.05 and the AUROC are reported beside it. Fractional PSU (`detection_psu_ratio`, this project's headline statistic) is reported as a secondary reading under the same thresholds.

## Predictions

**P1, detection.** On at least 3 of the 4 TaCT models, TPR at 10% FPR is below 0.2. A detector at 10% FPR that flags at random reaches 0.1, so 0.2 is at most twice chance and nobody would deploy it. The ViT TaCT panel models read between 0.000 and 0.230 at 5% FPR.

**P1 control.** The BadNets models on both datasets and the GTSRB Blend model read TPR at 10% FPR above 0.8. The Blend control carries a known risk, since the upstream reproduction found GTSRB Blend very sensitive to the probe rate. If a BadNets control reads below 0.8 the test is void, whatever the TaCT models read. If only Blend misses, the contrast rests on BadNets and the report says so.

**P1 refutation.** P1 is refuted when at least 2 of the successful TaCT models read TPR at 10% FPR of 0.5 or more. Anything between support and refutation is reported as mixed.

**P2, mechanism.** The trigger stamped on test images of the non-source non-target classes sends at most 0.2 of them to the target on every successful TaCT model, against at least 0.9 on the BadNets controls (the `classes` reading of `experiments/why_psbd_works/sufficiency.py`, its `non_source_stamped_on_target` field). The trigger on content-free carriers (the `blank` reading) sends at most 0.2 more carriers to the target than the unstamped carriers on every TaCT model.

**P3, link.** If any TaCT model reads TPR at 10% FPR of 0.5 or more, it is the TaCT model with the largest `non_source_stamped_on_target`. With 4 models this is an ordering check and not a correlation, and it is reported as such.
