# Content-hidden probe on the single-source TaCT models, registered before scoring

## The probe

The score of an image is how much of its unperturbed prediction's probability survives when every token outside the trigger's region is hidden from attention in every block, the content-hidden test of `experiments/evidence_surplus/sufficiency/PREDICTIONS_tact.md` turned into a per-image score:

$$s(x) = 1 - \frac{P_c(x;\ \text{content hidden})}{P_c(x)}, \qquad c = \arg\max_k P_k(x)$$

| symbol | meaning |
|---|---|
| $P_c(x)$ | softmax probability of the unperturbed prediction $c$ |
| $P_c(x;\ \text{content hidden})$ | the same probability with only the trigger's tokens (and the class token) visible to attention in all 12 blocks |

A low score is read as poisoned, as for PSU. The threshold is a quantile of the scores of the 2000 clean validation images of the PSBD split, and TPR is read on the triggered analysis images, FPR on their clean twins. The probe needs the trigger's position, which a defender does not have, so it is a diagnostic of the mechanism and not a detector.

## What is known before scoring

The content-hidden excess of `experiments/evidence_surplus/sufficiency/PREDICTIONS_tact.md` is 0.000 on all 4 single-source ViT TaCT models, among them the 3 scored here (`vit_gtsrb_tact_0_01_cos`, `vit_cifar10_tact_0_05`, `vit_cifar10_tact_0_01`): with the content hidden, a triggered image is sent to the target no more often than its clean twin. The internal maps of this directory found the same on `vit_gtsrb_tact_0_01_cos` (triggered kept 0.000 with any random 10%, 30% or 60% of the tokens visible).

## Prediction

1. On all 3 models the probe fails: AUROC at most 0.65 and TPR at 10% FPR at most 0.20.
2. The reason is that the content-hidden triggered image loses its target decision as completely as a clean image loses its own class, so the 2 score distributions overlap near 1.

## Refutation

Prediction 1 is refuted on a model whose AUROC exceeds 0.65 or whose TPR at 10% FPR exceeds 0.20. If it is refuted on 2 or more models, the single-source TaCT trigger carries a surplus in its own tokens that the content-hidden excess (a target-rate statistic) missed.
