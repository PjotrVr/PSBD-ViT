# Training-set detection in Li et al.'s setting

Li et al. (arXiv 2406.05826) built PSBD to find poisoned samples inside a training set. They score every training image of a backdoored model with PSU, flag PSU below the 25th percentile of clean-validation PSU and report TPR on the poisoned training images against FPR on the clean ones. Our paper scores triggered test images against their clean twins instead. This experiment asks whether our placements and their ranking carry over to the original setting, and whether the original setting is reproduced on the original architecture. The predictions and their refutation rules are in `PREDICTIONS.md`, committed before any training-set score was computed.

## Method

Each model's poisoned training set is rebuilt with the functions its training run used (`cli.train_backdoor.build_training_set`), the seed of `args.json` and 0 where the run predates seed recording, with no augmentation, which none of these models trained with. Poisoned images carry the trigger exactly as trained. 4 groups are scored: every poisoned training image, at most 2000 cover samples, 10000 clean training images drawn with seed 0 and the standard 2000-image PSBD validation split that sets every threshold.

PSBD-TM (`before_attention_norm` with `token_mask`), PSBD-RD (`post_residual`) and the final method (PSBD-TM fused with `pre_residual_blocks_5_8` by the minimum of clean-validation percentiles) run through `defenses.inference` exactly as `cli.sweep` runs them, k = 3, bfloat16, batch 64, mask seed 0. The validation passes reproduce the cached `cli.sweep` tensors (`validation_pass_checks` in each model record). ResNet-18 gets PSBD-RD only. Each placement is read at 2 rates. Ours is the smallest cached rate whose clean-validation shift ratio reaches 0.8. Li et al.'s, taken from their released `select_dropout_rate`, is the argmax of the validation shift minus the whole-training-set shift over the rates whose validation shift reaches 0.8, with the whole-set shift estimated on 2000 clean, 1000 poisoned and 1000 cover images weighted by their population counts. Fractional PSU is our headline statistic and absolute PSU is Li et al.'s.

STRIP runs on 2000 clean, 1000 poisoned and 500 cover images and CD-L on 1000, 500 and 250, both through `detectors.build_detector` with `cli.baselines`' context. CD-L reuses the test-time record's validation scores. The comparison with them reads PSBD on the CD-L images too. Spectral Signatures (`detectors/spectral_signatures.py`, ported from the backdoor-toolbox cleanser Li et al. ran) scores every image of the target label and removes min(int(1.5 epsilon N), n / 2) images of each label.

TPR is read 2 ways. With the threshold at a quantile of clean validation, which is what a defender can do, the FPR realized on the clean training images is reported beside it. With the threshold at the same quantile of the clean training images themselves the FPR is exactly the quantile, which is the fair comparison when the 2 clean distributions differ. Covers count as negatives in Li et al.'s code, so the paper-format FPR folds them in at their population share.

## Commands

Run from the main checkout, which holds `checkpoints/`, `raw_data/` and `results/`, with the code of this worktree on `PYTHONPATH`.

```bash
bash experiments/training_set_detection/run_queue.sh          # GPU, resumable, 2 flock slots
python experiments/training_set_detection/analyze.py          # CPU readout and verdicts
python experiments/training_set_detection/render_readme.py    # this file
```

GPU minutes per model, the sum of every part's own timer on the shared login A100:

| model | minutes |
|---|---|
| `vit_cifar10_badnet_a2o_0_1` | 8.0 |

<!-- results:begin -->
<!-- Everything down to results:end is rendered by render_readme.py. -->

## Paper-mirror set

Li et al.'s main table uses BadNets, Blend, WaNet and Label-Consistent on CIFAR-10, GTSRB and Tiny ImageNet at 10% poisoning with ResNet-18. The first table reads each model in their exact configuration: absolute PSU, their rate rule and T at the 25th percentile of clean validation, as TPR/FPR with covers counted as negatives. STRIP and CD-L are thresholded at the same quantile. Spectral Signatures uses its own removal rule. A model marked not pooled failed a reconstruction check or the 2-point bar and stays out of every mean.

| model | PSBD-TM | PSBD-RD | final method | STRIP | CD-L | Spectral Signatures |
|---|---|---|---|---|---|---|
| BadNets, CIFAR-10, ViT-B/16 | 1.000/0.206 | 0.300/0.217 | 1.000/0.203 | pending | pending | 0.491/0.501 |
| `vit_cifar10_blend_0_1` | pending | | | | | |
| `vit_cifar10_wanet_0_1` | pending | | | | | |
| `vit_gtsrb_badnet_a2o_0_1` | pending | | | | | |
| `vit_gtsrb_blend_0_1` | pending | | | | | |
| `vit_tiny_badnet_a2o_0_1` | pending | | | | | |
| `vit_tiny_blend_0_1` | pending | | | | | |
| `vit_tiny_wanet_0_1` | pending | | | | | |
| `vit_gtsrb_lc_0_05_tl1_adv` | pending | | | | | |
| `resnet18_gtsrb_badnet_a2o_0_1` | pending | | | | | |
| `resnet18_gtsrb_blend_0_1` | pending | | | | | |

The same table under our rate rule and fractional PSU.

| model | PSBD-TM | PSBD-RD | final method | STRIP | CD-L | Spectral Signatures |
|---|---|---|---|---|---|---|
| BadNets, CIFAR-10, ViT-B/16 | 1.000/0.245 | 0.414/0.242 | 0.999/0.236 | pending | pending | 0.491/0.501 |
| `vit_cifar10_blend_0_1` | pending | | | | | |
| `vit_cifar10_wanet_0_1` | pending | | | | | |
| `vit_gtsrb_badnet_a2o_0_1` | pending | | | | | |
| `vit_gtsrb_blend_0_1` | pending | | | | | |
| `vit_tiny_badnet_a2o_0_1` | pending | | | | | |
| `vit_tiny_blend_0_1` | pending | | | | | |
| `vit_tiny_wanet_0_1` | pending | | | | | |
| `vit_gtsrb_lc_0_05_tl1_adv` | pending | | | | | |
| `resnet18_gtsrb_badnet_a2o_0_1` | pending | | | | | |
| `resnet18_gtsrb_blend_0_1` | pending | | | | | |

TPR at 1%, 5% and 10% FPR realized on the clean training images, then AUROC of poisoned against clean training images, fractional PSU at our rate. STRIP and CD-L rows read the smaller subset.

| model | method | TPR at 1% | TPR at 5% | TPR at 10% | AUROC |
|---|---|---|---|---|---|
| BadNets, CIFAR-10, ViT-B/16 | final method | 0.487 | 0.669 | 0.839 | 0.958 |
| BadNets, CIFAR-10, ViT-B/16 | PSBD-TM | 0.554 | 0.759 | 0.939 | 0.972 |
| BadNets, CIFAR-10, ViT-B/16 | PSBD-RD | 0.006 | 0.050 | 0.131 | 0.690 |
| mean over 1 pooled models | final method | 0.487 | 0.669 | 0.839 | 0.958 |
| mean over 1 pooled models | PSBD-TM | 0.554 | 0.759 | 0.939 | 0.972 |
| mean over 1 pooled models | PSBD-RD | 0.006 | 0.050 | 0.131 | 0.690 |

The 2 ResNet-18 reproductions.

| model | method | TPR at 1% | TPR at 5% | TPR at 10% | AUROC |
|---|---|---|---|---|---|

## Development set

No development model beyond the paper-mirror set has been scored yet.

## Clean training against clean validation

A threshold set on validation is only honest on the training set if clean training images and clean validation images share a PSU distribution. The table gives both medians, the Kolmogorov-Smirnov statistic between them and the FPR each nominal quantile realizes on clean training images, for PSBD-TM (PSBD-RD on ResNet-18) at our rate.

| model | PSU form | median train | median validation | KS statistic | realized FPR at 1% | at 5% | at 10% | at 25% |
|---|---|---|---|---|---|---|---|---|
| BadNets, CIFAR-10, ViT-B/16, PSBD-TM | fractional | 0.941 | 0.945 | 0.023 | 0.005 | 0.049 | 0.099 | 0.245 |
| BadNets, CIFAR-10, ViT-B/16, PSBD-TM | absolute | 0.931 | 0.923 | 0.033 | 0.005 | 0.049 | 0.097 | 0.220 |

TPR with the threshold at each quantile of clean validation, and the FPR it realizes on clean training images, fractional PSU at our rate.

| model | method | 1% | 5% | 10% | 25% |
|---|---|---|---|---|---|
| BadNets, CIFAR-10, ViT-B/16 | final method | 0.003/0.004 | 0.667/0.046 | 0.817/0.094 | 0.999/0.236 |
| BadNets, CIFAR-10, ViT-B/16 | PSBD-TM | 0.472/0.005 | 0.755/0.049 | 0.933/0.099 | 1.000/0.245 |
| BadNets, CIFAR-10, ViT-B/16 | PSBD-RD | 0.002/0.006 | 0.045/0.046 | 0.109/0.087 | 0.414/0.242 |

## Training set against test time

AUROC on the training images next to AUROC on the paired test splits, from the `cli.sweep` caches and the `cli.baselines` records of the same models.

| model | final method train / test | PSBD-TM train / test | PSBD-RD train / test | STRIP train / test | CD-L train / test |
|---|---|---|---|---|---|
| BadNets, CIFAR-10, ViT-B/16 | 0.958 / 0.976 | 0.972 / 0.991 | 0.690 / 0.720 | n/a / 0.997 | n/a / 1.000 |

On the paper-mirror set (1 pooled models) PSBD-TM leads PSBD-RD by +0.282 AUROC on training images, interval [+nan, +nan], against +0.271 on the paired test splits of the same models.

## Rates under both rules

| model | placement | our rate | Li et al.'s rate | Li et al.'s candidates |
|---|---|---|---|---|
| BadNets, CIFAR-10, ViT-B/16 | psbd_tm | 0.600 | 0.900 | 0.6, 0.7, 0.8, 0.9 |
| BadNets, CIFAR-10, ViT-B/16 | psbd_rd | 0.090 | 0.100 | 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 |
| BadNets, CIFAR-10, ViT-B/16 | middle_band | 0.900 | 0.900 | 0.9, 0.95, 0.99 |

## Reconstruction checks

R1 compares the rebuilt counts with `args.json`, R2 asks whether the model sends the rebuilt poisoned images to their attack-success label, and R3 asks whether the untriggered image of a rebuilt poisoned index looks held out (the ratio of its loss gap to the validation gap, at least 0.5 to pass).

| model | successful_2pt | seed recorded | R1 counts | R2 training attack success (recorded ASR) | R3 ratio | verdict |
|---|---|---|---|---|---|---|
| `vit_cifar10_badnet_a2o_0_1` | yes | no | passed | 1.000 (1.000) | 0.805, passed | verified |

## Verdicts

- P1 on the paper-mirror set: refuted, `{"auroc": {"final_min": 0.9583669300000001, "psbd_tm": 0.9719532399999999, "psbd_rd": 0.69006555}, "tm_leads_rd_at_realized_1_5_10": [true, true, true]}`
- P2 on the paper-mirror set: holds, `{"train_gain": 0.2818876899999999, "test_gain": 0.27081001157407414}`
- P3 on the paper-mirror set: holds, `{"li_tm": 0.9554410600000001, "li_rd": 0.7054801500000001}`
- P6 on the paper-mirror set: refuted, `{"train_order": ["psbd_tm"], "test_order": ["cd_l", "strip", "psbd_tm"]}`
- P4 on the paper-mirror set: fractional part refuted (median below on 1 of 1, FPR above 0.25 on 0 of 1), absolute part holds (lower on 1 of 1).
- development: no headline model scored yet
- P5: refuted, `{"models": {}}`
- P7: refuted, `{"reached": {"vit_cifar10_badnet_a2o_0_1": true}, "count": "1 of 1"}`

## Pending

Not yet scored when this file was rendered: `vit_cifar10_blend_0_1`, `vit_cifar10_wanet_0_1`, `vit_gtsrb_badnet_a2o_0_1`, `vit_gtsrb_blend_0_1`, `vit_tiny_badnet_a2o_0_1`, `vit_tiny_blend_0_1`, `vit_tiny_wanet_0_1`, `vit_gtsrb_lc_0_05_tl1_adv`, `resnet18_gtsrb_badnet_a2o_0_1`, `resnet18_gtsrb_blend_0_1`, `vit_gtsrb_wanet_0_1`, `vit_cifar10_bpp_0_05`, `vit_gtsrb_bpp_0_01`, `vit_gtsrb_bpp_0_1`, `vit_cifar10_badnet_a2o_0_01`, `vit_gtsrb_tact_0_05`, `vit_cifar10_tact_0_01`, `vit_gtsrb_lf_0_01`.

<!-- results:end -->
