# PSBD on ResNet-18 TaCT

## Question

PSBD-TM reads a high AUROC on the ViT TaCT models of the panel and almost no true positives at a 1%, 5% or 10% false-positive budget (`experiments/low_fpr_audit/`). The evidence surplus account (`docs/evidence-surplus-theory.md`) explains this through the attack. A TaCT decision needs the trigger and the source-class content together, so it carries no spare evidence and falls under random removal the way a clean decision does. If that is right the failure does not depend on the architecture, and the original PSBD on ResNet-18, the setting of Li et al. (`literature/psbd-li-arxiv2024/`), also fails on TaCT while it works on BadNets and Blend trained on the same recipe. The original paper never tested TaCT. The predictions, the bars and the refutation rule were committed before any model was trained, in `PREDICTIONS.md`.

## Method

Every model trains on the recipe of the existing ResNet-18 reproductions (`checkpoints/resnet18_gtsrb_badnet_a2o_0_1/args.json`). That is ResNet-18 with the CIFAR stem from random initialization, 100 epochs of Adam at the constant learning rate 1e-4 with weight decay 1e-4, batch 128, no augmentation, seed 0, target class 0 and the full training split. The earlier CIFAR-10 ResNet (`resnet18_cifar10_badnet_a2o_0_1_smoke`) trained 15 epochs on 20000 images, which is why its clean accuracy stayed low, and it is not used here. A new CIFAR-10 BadNets model on the full recipe replaces it as the CIFAR-10 control, and the 2 GTSRB controls are the existing BadNets and Blend reproductions.

TaCT keeps its defaults (a 3 pixel checkerboard in the bottom right corner, cover rate 0.01). The source classes are 1..k, with k the smallest count whose share of the training split is at least 5 times the poison rate (`docs/runs/2026-09-24-tact-multisource.md`). About a fifth of the source pool is then poisoned (the ratio of Tang et al.) and no source class is poisoned whole, so the models cannot learn to send a source class to the target without the trigger. The `_src<k>` suffix names a model with more than the default single source class.

A model enters the verdicts only when it is a successful backdoor. Its ASR is at least 0.85 on its own eval set, its clean accuracy is within 2 points of the ResNet-18 benign model of its dataset and, for TaCT, its clean accuracy on the source classes is at least 0.5.

Detection is the original PSBD. Dropout sits after the residual add of every basic block (`post_residual`, the only distinct ResNet-18 position in `models.positions.POSITION_REGISTRY`, since its other 2 names alias the same site). The sweep runs 3 forward passes over the 16 rate ladder of the reproductions on the standard 2000-image clean validation split, and `cli.analyze` picks the adaptive rate (the smallest rate whose clean-validation shift ratio reaches 0.8) and sets thresholds at quantiles of the clean validation scores. The primary statistic is the paper's own absolute PSU and fractional PSU is the secondary reading. The mechanism reading is the trigger sufficiency control of `experiments/why_psbd_works/sufficiency.py` run on each model. TED and Beatrix, which catch TaCT on ViT, run through `cli.baselines` on the TaCT models.

## Commands

The GPU stages run on the login node through a resumable queue that holds 1 of the 2 shared lock slots per stage, caps each process at a fifth of the GPU memory and 4 CPU threads (`capped.py`) and starts nothing between 06:30 and 17:00. Its log is `scratch/resnet_tact/queue.log` in the main checkout. The checkpoints and PSBD caches live in the main checkout, and this experiment's own records under `results/_experiments/resnet_tact/`.

```bash
bash experiments/resnet_tact/run_queue.sh
PYTHONPATH=. python experiments/resnet_tact/report.py \
    --checkpoints-dir /lustre/home/pstika/projects/PSBD-ViT/checkpoints \
    --results-dir /lustre/home/pstika/projects/PSBD-ViT/results
```

<!-- results:begin -->
<!-- Everything down to results:end is rendered by report.py from results/_experiments/resnet_tact/. -->

## Models trained

Clean accuracy and ASR are the final values `cli.train_backdoor` wrote to `args.json`. The benign reference is the ResNet-18 benign model of the same dataset and recipe, and the source-class accuracy is clean accuracy on the source classes of the PSBD analysis split.

| model | source classes | ASR | clean acc | benign | drop | source-class acc | successful | train min |
|---|---|---:|---:|---:|---:|---:|---|---:|
| `resnet18_gtsrb_tact_0_05_src6` | 1..6 | 0.999 | 0.976 | 0.979 | 0.003 | 0.985 | yes | 16.4 |
| `resnet18_gtsrb_tact_0_1_src12` | 1..12 | 0.997 | 0.970 | 0.979 | 0.010 | 0.983 | yes | 14.9 |
| `resnet18_gtsrb_badnet_a2o_0_1` | -- | 1.000 | 0.976 | 0.979 | 0.003 | -- | yes | 7.5 |
| `resnet18_gtsrb_blend_0_1` | -- | 1.000 | 0.974 | 0.979 | 0.005 | -- | yes | 7.5 |

## Original PSBD, absolute PSU

Dropout at `post_residual`, 3 passes, the adaptive rate of the 0.8 rule. TPR at the 0.01, 0.05 and 0.10 quantiles of the clean validation scores, with the realized FPR at the 0.10 quantile, then the AUROC. The FPR and the AUROC read the paired clean test images, the clean twins of the triggered ones, so for TaCT only source-class images.

| model | rate | TPR@1% | TPR@5% | TPR@10% | FPR at q0.10 | AUROC | AUROC against all clean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `resnet18_gtsrb_tact_0_05_src6` | 0.5 | 0.004 | 0.069 | 0.291 | 0.016 | 0.971 | 0.827 |
| `resnet18_gtsrb_tact_0_1_src12` | 0.5 | 0.028 | 0.502 | 0.832 | 0.029 | 0.979 | 0.939 |
| `resnet18_gtsrb_badnet_a2o_0_1` | 0.5 | 0.989 | 1.000 | 1.000 | 0.092 | 0.999 | 0.999 |
| `resnet18_gtsrb_blend_0_1` | 0.5 | 0.163 | 0.645 | 0.855 | 0.096 | 0.943 | 0.943 |

## Fractional PSU, the secondary reading

Dropout at `post_residual`, 3 passes, the adaptive rate of the 0.8 rule. TPR at the 0.01, 0.05 and 0.10 quantiles of the clean validation scores, with the realized FPR at the 0.10 quantile, then the AUROC. The FPR and the AUROC read the paired clean test images, the clean twins of the triggered ones, so for TaCT only source-class images.

| model | rate | TPR@1% | TPR@5% | TPR@10% | FPR at q0.10 | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| `resnet18_gtsrb_tact_0_05_src6` | 0.5 | 0.014 | 0.248 | 0.472 | 0.001 | 0.993 |
| `resnet18_gtsrb_tact_0_1_src12` | 0.5 | 0.519 | 0.840 | 0.963 | 0.007 | 0.998 |
| `resnet18_gtsrb_badnet_a2o_0_1` | 0.5 | 1.000 | 1.000 | 1.000 | 0.109 | 1.000 |
| `resnet18_gtsrb_blend_0_1` | 0.5 | 0.555 | 0.849 | 0.913 | 0.090 | 0.968 |

## Shift ratios at the adaptive rate

The share of dropout passes whose prediction differs from the unperturbed prediction, over the clean and the triggered test images, at the rate the 0.8 rule picked.

| model | rate | clean | triggered |
|---|---:|---:|---:|
| `resnet18_gtsrb_tact_0_05_src6` | 0.5 | 0.896 | 0.605 |
| `resnet18_gtsrb_tact_0_1_src12` | 0.5 | 0.938 | 0.238 |
| `resnet18_gtsrb_badnet_a2o_0_1` | 0.5 | 0.943 | 0.003 |
| `resnet18_gtsrb_blend_0_1` | 0.5 | 0.896 | 0.260 |

## Trigger sufficiency

`experiments/why_psbd_works/sufficiency.py` on each model. The class reading stamps the trigger on test images of every class but the target and splits the share sent to the target by source and non-source class. The blank reading stamps it on content-free carriers against the same carriers unstamped.

| model | non-source stamped to target | source stamped to target | clean to target | blank excess |
|---|---:|---:|---:|---:|
| `resnet18_gtsrb_tact_0_05_src6` | -- | -- | -- | -- |
| `resnet18_gtsrb_tact_0_1_src12` | -- | -- | -- | -- |
| `resnet18_gtsrb_badnet_a2o_0_1` | -- | -- | -- | -- |
| `resnet18_gtsrb_blend_0_1` | -- | -- | -- | -- |

## ViT TaCT models of the paper panel

The same fractional readings for the ViT TaCT models of the panel, at PSBD-TM and at the ViT `post_residual` dropout (PSBD-RD), with the non-source class reading of `experiments/evidence_surplus/`, for the ViT TaCT models of the panel that have a PSBD-TM cache.

| model | PSBD-TM TPR@1% | TPR@5% | TPR@10% | AUROC | PSBD-RD TPR@10% | AUROC | non-source stamped to target |
|---|---:|---:|---:|---:|---:|---:|---:|
| `vit_cifar10_tact_0_01` | 0.005 | 0.005 | 0.005 | 0.979 | 0.013 | 0.464 | 0.009 |
| `vit_cifar10_tact_0_05` | 0.000 | 0.000 | 0.000 | 0.966 | 0.006 | 0.834 | -- |
| `vit_gtsrb_tact_0_01_cos` | 0.000 | 0.000 | 0.000 | 0.266 | 0.643 | 0.830 | -- |
| `vit_gtsrb_tact_0_05` | 0.030 | 0.230 | 0.502 | 0.942 | 0.010 | 0.392 | 0.002 |

## Verdicts

- **P1**: no verdict, fewer than 3 successful TaCT models. 0 of 2 successful TaCT models read TPR at 10% FPR below 0.2 and 1 read 0.5 or more. Every control reads above 0.8 (yes).
- **P1 on fractional PSU**: no verdict, fewer than 3 successful TaCT models. 0 of 2 successful TaCT models read TPR at 10% FPR below 0.2 and 1 read 0.5 or more. Every control reads above 0.8 (yes).
- **P2**: no verdict, readings missing.
- **P3**: not tested, no TaCT model is detected.

<!-- results:end -->
