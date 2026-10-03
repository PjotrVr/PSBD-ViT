# Training-set detection in Li et al.'s setting

Li et al. (arXiv 2406.05826) built PSBD to find poisoned samples inside a training set. They score every training image of a backdoored model with PSU, flag PSU below the 25th percentile of clean-validation PSU and report TPR on the poisoned training images against FPR on the clean ones. Our paper scores triggered test images against their clean twins instead. This experiment asks whether our placements and their ranking carry over to the original setting, and whether the original setting is reproduced on the original architecture. The predictions and their refutation rules are in `PREDICTIONS.md`, committed before any training-set score was computed.

## Method

Each model's poisoned training set is rebuilt with the functions its training run used (`cli.train_backdoor.build_training_set`), the seed of `args.json` and 0 where the run predates seed recording, with no augmentation, which none of these models trained with. Poisoned images carry the trigger exactly as trained. 4 groups are scored: every poisoned training image, at most 2000 cover samples, 10000 clean training images drawn with seed 0 and the standard 2000-image PSBD validation split that sets every threshold.

PSBD-TM (`before_attention_norm` with `token_mask`), PSBD-RD (`post_residual`) and the final method (PSBD-TM fused with `pre_residual_blocks_5_8` by the minimum of clean-validation percentiles) run through `defenses.inference` exactly as `cli.sweep` runs them, k = 3, bfloat16, batch 64, mask seed 0. The validation passes reproduce the cached `cli.sweep` tensors (`validation_pass_checks` in each model record). ResNet-18 gets PSBD-RD only. Each placement is read at 2 rates. Ours is the smallest cached rate whose clean-validation shift ratio reaches 0.8. Li et al.'s, taken from their released `select_dropout_rate`, is the argmax of the validation shift minus the whole-training-set shift over the rates whose validation shift reaches 0.8, with the whole-set shift estimated on 2000 clean, 1000 poisoned and 1000 cover images weighted by their population counts. Fractional PSU is our headline statistic and absolute PSU is Li et al.'s.

STRIP runs on 2000 clean, 1000 poisoned and 500 cover images and CD-L on 1000, 500 and 250, both through `detectors.build_detector` with `cli.baselines`' context. CD-L reuses the test-time record's validation scores. The comparison with them reads PSBD on the CD-L images too. Spectral Signatures (`detectors/spectral_signatures.py`, ported from the backdoor-toolbox cleanser Li et al. ran) scores every image of the target label and removes min(int(1.5 epsilon N), n / 2) images of each label.

TPR is read 2 ways. With the threshold at a quantile of clean validation, which is what a defender can do, the FPR realized on the clean training images is reported beside it. With the threshold at the same quantile of the clean training images themselves the FPR is exactly the quantile, which is the fair comparison when the 2 clean distributions differ. Covers count as negatives in Li et al.'s code, so the paper-format FPR folds them in at their population share.

## Commands

Run from the main checkout with the code of this worktree on `PYTHONPATH`. The main checkout holds `checkpoints/`, `raw_data/` and `results/`.

```bash
bash experiments/training_set_detection/run_queue.sh          # GPU, resumable, 2 flock slots
python experiments/training_set_detection/analyze.py          # CPU readout and verdicts
python experiments/training_set_detection/render_readme.py    # this file
```

GPU minutes per model, the sum of every part's own timer on the shared login A100:

| model | minutes |
|---|---|
| `vit_cifar10_badnet_a2o_0_1` | 23.6 |
| `vit_cifar10_blend_0_1` | 24.3 |
| `vit_cifar10_wanet_0_1` | 33.4 |
| `vit_gtsrb_badnet_a2o_0_1` | 16.3 |
| `vit_gtsrb_blend_0_1` | 15.2 |
| `vit_tiny_badnet_a2o_0_1` | 13.5 |
| `vit_tiny_blend_0_1` | 19.8 |
| `vit_tiny_wanet_0_1` | 18.3 |
| `vit_gtsrb_lc_0_05_tl1_adv` | 12.5 |
| `resnet18_gtsrb_badnet_a2o_0_1` | 2.1 |
| `resnet18_gtsrb_blend_0_1` | 2.1 |

## Changes of scope

- 2026-10-01: the development tier (the 10 models of `experiments/cache_readouts/dev_set.json`, of which the 2 that are also paper-mirror models stay scored) and the BackdoorBench training-set tier are dropped, so the night's GPU time goes to the paper-mirror set and its detectors. P1 to P4 and P6 are judged on the paper-mirror set only and the development-set readings of `PREDICTIONS.md` stay unjudged. STRIP and CD-L run on the paper-mirror models in a second pass after PSBD, and Spectral Signatures is read on the CPU from the baseline features PSBD's first part stores.
- 2026-10-02: the GPU queue was stopped at 00:08 to give the login GPU to a priority run, before STRIP and CD-L had started on any model. The pending section of that render named what was missing. To resume, run `bash experiments/training_set_detection/run_queue.sh` from the main checkout inside the GPU window, then `analyze.py` and this renderer. `score.py` skips every part already under `results/_experiments/training_set_detection/raw/` and the queue skips every stage with a `done_<pass>.<folder>` marker under `scratch/training_set_detection/`, so the run picks up where it stopped.
- 2026-10-03: the resumed queue finished at 02:52 with PSBD on `vit_tiny_wanet_0_1` and `vit_gtsrb_lc_0_05_tl1_adv` and STRIP and CD-L on every paper-mirror model, so no model is pending and the verdicts below are final.

<!-- results:begin -->
<!-- Everything down to results:end is rendered by render_readme.py. -->

## Paper-mirror set

Li et al.'s main table uses BadNets, Blend, WaNet and Label-Consistent on CIFAR-10, GTSRB and Tiny ImageNet at 10% poisoning with ResNet-18. The first table reads each model in their exact configuration: absolute PSU, their rate rule and T at the 25th percentile of clean validation, as TPR/FPR with covers counted as negatives. STRIP and CD-L are thresholded at the same quantile. Spectral Signatures uses its own removal rule. A model marked not pooled failed a reconstruction check or the 2-point bar and stays out of every mean.

| model | PSBD-TM | PSBD-RD | final method | STRIP | CD-L | Spectral Signatures |
|---|---|---|---|---|---|---|
| BadNets, CIFAR-10, ViT-B/16 | 1.000/0.206 | 0.300/0.217 | 1.000/0.203 | 1.000/0.243 | 1.000/0.274 | 0.491/0.501 |
| Blend, CIFAR-10, ViT-B/16 | 0.847/0.227 | 0.992/0.193 | 0.947/0.212 | 0.959/0.260 | 0.970/0.262 | 0.527/0.497 |
| WaNet, CIFAR-10, ViT-B/16 | 0.291/0.205 | 0.990/0.206 | 0.990/0.219 | 0.264/0.293 | 0.480/0.258 | 0.480/0.502 |
| BadNets, GTSRB, ViT-B/16 (not pooled) | 1.000/0.188 | 1.000/0.193 | 1.000/0.191 | 1.000/0.306 | 1.000/0.259 | 0.472/0.503 |
| Blend, GTSRB, ViT-B/16 (not pooled) | 1.000/0.199 | 0.997/0.209 | 1.000/0.192 | 0.956/0.282 | 1.000/0.277 | 0.472/0.503 |
| BadNets, Tiny ImageNet, ViT-B/16 | 0.999/0.094 | 0.646/0.072 | 1.000/0.117 | 0.952/0.265 | 1.000/0.230 | 0.475/0.502 |
| Blend, Tiny ImageNet, ViT-B/16 | 0.878/0.056 | 0.864/0.055 | 0.984/0.056 | 0.971/0.259 | 0.000/0.235 | 0.475/0.502 |
| WaNet, Tiny ImageNet, ViT-B/16 | 0.622/0.062 | 0.803/0.067 | 0.970/0.088 | 0.255/0.251 | 0.986/0.251 | 0.475/0.502 |
| Label-Consistent, GTSRB, ViT-B/16 (not pooled) | 0.998/0.229 | 0.045/0.215 | 0.998/0.219 | 1.000/0.273 | 0.802/0.254 | 0.440/0.503 |
| BadNets, GTSRB, ResNet-18 (not pooled) | n/a | 1.000/0.208 | n/a | 1.000/0.271 | 1.000/0.210 | 0.472/0.503 |
| Blend, GTSRB, ResNet-18 (not pooled) | n/a | 0.961/0.200 | n/a | 0.902/0.280 | 0.726/0.213 | 0.497/0.500 |

The same table under our rate rule and fractional PSU.

| model | PSBD-TM | PSBD-RD | final method | STRIP | CD-L | Spectral Signatures |
|---|---|---|---|---|---|---|
| BadNets, CIFAR-10, ViT-B/16 | 1.000/0.245 | 0.414/0.242 | 0.999/0.236 | 1.000/0.243 | 1.000/0.274 | 0.491/0.501 |
| Blend, CIFAR-10, ViT-B/16 | 0.868/0.248 | 0.995/0.227 | 0.970/0.242 | 0.959/0.260 | 0.970/0.262 | 0.527/0.497 |
| WaNet, CIFAR-10, ViT-B/16 | 0.320/0.233 | 0.984/0.241 | 0.985/0.237 | 0.264/0.293 | 0.480/0.258 | 0.480/0.502 |
| BadNets, GTSRB, ViT-B/16 (not pooled) | 1.000/0.277 | 1.000/0.237 | 1.000/0.257 | 1.000/0.306 | 1.000/0.259 | 0.472/0.503 |
| Blend, GTSRB, ViT-B/16 (not pooled) | 1.000/0.241 | 0.998/0.239 | 1.000/0.235 | 0.956/0.282 | 1.000/0.277 | 0.472/0.503 |
| BadNets, Tiny ImageNet, ViT-B/16 | 1.000/0.211 | 0.991/0.219 | 1.000/0.239 | 0.952/0.265 | 1.000/0.230 | 0.475/0.502 |
| Blend, Tiny ImageNet, ViT-B/16 | 1.000/0.212 | 1.000/0.223 | 1.000/0.230 | 0.971/0.259 | 0.000/0.235 | 0.475/0.502 |
| WaNet, Tiny ImageNet, ViT-B/16 | 0.995/0.220 | 0.988/0.226 | 0.997/0.222 | 0.255/0.251 | 0.986/0.251 | 0.475/0.502 |
| Label-Consistent, GTSRB, ViT-B/16 (not pooled) | 0.998/0.249 | 0.107/0.277 | 0.998/0.269 | 1.000/0.273 | 0.802/0.254 | 0.440/0.503 |
| BadNets, GTSRB, ResNet-18 (not pooled) | n/a | 1.000/0.290 | n/a | 1.000/0.271 | 1.000/0.210 | 0.472/0.503 |
| Blend, GTSRB, ResNet-18 (not pooled) | n/a | 0.976/0.272 | n/a | 0.902/0.280 | 0.726/0.213 | 0.497/0.500 |

TPR at 1%, 5% and 10% FPR realized on the clean training images, then AUROC of poisoned against clean training images, fractional PSU at our rate. STRIP and CD-L rows read the smaller subset.

| model | method | TPR at 1% | TPR at 5% | TPR at 10% | AUROC |
|---|---|---|---|---|---|
| BadNets, CIFAR-10, ViT-B/16 | final method | 0.487 | 0.669 | 0.839 | 0.958 |
| BadNets, CIFAR-10, ViT-B/16 | PSBD-TM | 0.554 | 0.759 | 0.939 | 0.972 |
| BadNets, CIFAR-10, ViT-B/16 | PSBD-RD | 0.006 | 0.050 | 0.131 | 0.690 |
| BadNets, CIFAR-10, ViT-B/16 | STRIP, subset | 0.826 | 0.992 | 1.000 | 0.994 |
| BadNets, CIFAR-10, ViT-B/16 | CD-L, subset | 1.000 | 1.000 | 1.000 | 1.000 |
| Blend, CIFAR-10, ViT-B/16 | final method | 0.635 | 0.845 | 0.913 | 0.969 |
| Blend, CIFAR-10, ViT-B/16 | PSBD-TM | 0.597 | 0.708 | 0.758 | 0.913 |
| Blend, CIFAR-10, ViT-B/16 | PSBD-RD | 0.906 | 0.968 | 0.985 | 0.993 |
| Blend, CIFAR-10, ViT-B/16 | STRIP, subset | 0.736 | 0.818 | 0.866 | 0.961 |
| Blend, CIFAR-10, ViT-B/16 | CD-L, subset | 0.104 | 0.434 | 0.630 | 0.914 |
| WaNet, CIFAR-10, ViT-B/16 | final method | 0.652 | 0.880 | 0.959 | 0.974 |
| WaNet, CIFAR-10, ViT-B/16 | PSBD-TM | 0.017 | 0.040 | 0.096 | 0.505 |
| WaNet, CIFAR-10, ViT-B/16 | PSBD-RD | 0.269 | 0.657 | 0.876 | 0.949 |
| WaNet, CIFAR-10, ViT-B/16 | STRIP, subset | 0.002 | 0.040 | 0.082 | 0.477 |
| WaNet, CIFAR-10, ViT-B/16 | CD-L, subset | 0.114 | 0.242 | 0.344 | 0.629 |
| BadNets, GTSRB, ViT-B/16 (not pooled) | final method | 1.000 | 1.000 | 1.000 | 1.000 |
| BadNets, GTSRB, ViT-B/16 (not pooled) | PSBD-TM | 0.997 | 1.000 | 1.000 | 1.000 |
| BadNets, GTSRB, ViT-B/16 (not pooled) | PSBD-RD | 0.995 | 0.998 | 0.999 | 1.000 |
| BadNets, GTSRB, ViT-B/16 (not pooled) | STRIP, subset | 0.000 | 0.994 | 1.000 | 0.977 |
| BadNets, GTSRB, ViT-B/16 (not pooled) | CD-L, subset | 0.996 | 1.000 | 1.000 | 1.000 |
| Blend, GTSRB, ViT-B/16 (not pooled) | final method | 1.000 | 1.000 | 1.000 | 1.000 |
| Blend, GTSRB, ViT-B/16 (not pooled) | PSBD-TM | 1.000 | 1.000 | 1.000 | 1.000 |
| Blend, GTSRB, ViT-B/16 (not pooled) | PSBD-RD | 0.958 | 0.982 | 0.993 | 0.997 |
| Blend, GTSRB, ViT-B/16 (not pooled) | STRIP, subset | 0.888 | 0.906 | 0.924 | 0.963 |
| Blend, GTSRB, ViT-B/16 (not pooled) | CD-L, subset | 0.956 | 0.992 | 0.998 | 0.997 |
| BadNets, Tiny ImageNet, ViT-B/16 | final method | 0.981 | 0.999 | 1.000 | 0.999 |
| BadNets, Tiny ImageNet, ViT-B/16 | PSBD-TM | 0.769 | 0.999 | 1.000 | 0.993 |
| BadNets, Tiny ImageNet, ViT-B/16 | PSBD-RD | 0.419 | 0.861 | 0.951 | 0.973 |
| BadNets, Tiny ImageNet, ViT-B/16 | STRIP, subset | 0.190 | 0.540 | 0.718 | 0.921 |
| BadNets, Tiny ImageNet, ViT-B/16 | CD-L, subset | 0.000 | 0.000 | 0.000 | 0.850 |
| Blend, Tiny ImageNet, ViT-B/16 | final method | 0.975 | 0.999 | 1.000 | 0.998 |
| Blend, Tiny ImageNet, ViT-B/16 | PSBD-TM | 0.976 | 0.995 | 0.999 | 0.999 |
| Blend, Tiny ImageNet, ViT-B/16 | PSBD-RD | 0.993 | 0.999 | 1.000 | 0.999 |
| Blend, Tiny ImageNet, ViT-B/16 | STRIP, subset | 0.782 | 0.924 | 0.950 | 0.978 |
| Blend, Tiny ImageNet, ViT-B/16 | CD-L, subset | 0.000 | 0.000 | 0.000 | 0.434 |
| WaNet, Tiny ImageNet, ViT-B/16 | final method | 0.656 | 0.942 | 0.978 | 0.985 |
| WaNet, Tiny ImageNet, ViT-B/16 | PSBD-TM | 0.294 | 0.760 | 0.934 | 0.963 |
| WaNet, Tiny ImageNet, ViT-B/16 | PSBD-RD | 0.716 | 0.896 | 0.950 | 0.981 |
| WaNet, Tiny ImageNet, ViT-B/16 | STRIP, subset | 0.002 | 0.032 | 0.090 | 0.492 |
| WaNet, Tiny ImageNet, ViT-B/16 | CD-L, subset | 0.000 | 0.240 | 0.786 | 0.921 |
| Label-Consistent, GTSRB, ViT-B/16 (not pooled) | final method | 0.917 | 0.974 | 0.995 | 0.995 |
| Label-Consistent, GTSRB, ViT-B/16 (not pooled) | PSBD-TM | 0.938 | 0.986 | 0.997 | 0.997 |
| Label-Consistent, GTSRB, ViT-B/16 (not pooled) | PSBD-RD | 0.000 | 0.002 | 0.009 | 0.439 |
| Label-Consistent, GTSRB, ViT-B/16 (not pooled) | STRIP, subset | 0.898 | 0.972 | 0.990 | 0.996 |
| Label-Consistent, GTSRB, ViT-B/16 (not pooled) | CD-L, subset | 0.062 | 0.174 | 0.326 | 0.833 |
| mean over 6 pooled models | final method | 0.731 | 0.889 | 0.948 | 0.981 |
| mean over 6 pooled models | PSBD-TM | 0.535 | 0.710 | 0.788 | 0.891 |
| mean over 6 pooled models | PSBD-RD | 0.551 | 0.739 | 0.815 | 0.931 |

The 2 ResNet-18 reproductions.

| model | method | TPR at 1% | TPR at 5% | TPR at 10% | AUROC |
|---|---|---|---|---|---|
| BadNets, GTSRB, ResNet-18 (not pooled) | PSBD-RD | 1.000 | 1.000 | 1.000 | 1.000 |
| BadNets, GTSRB, ResNet-18 (not pooled) | STRIP, subset | 0.978 | 1.000 | 1.000 | 0.998 |
| BadNets, GTSRB, ResNet-18 (not pooled) | CD-L, subset | 0.932 | 1.000 | 1.000 | 0.993 |
| Blend, GTSRB, ResNet-18 (not pooled) | PSBD-RD | 0.555 | 0.837 | 0.911 | 0.965 |
| Blend, GTSRB, ResNet-18 (not pooled) | STRIP, subset | 0.096 | 0.614 | 0.764 | 0.898 |
| Blend, GTSRB, ResNet-18 (not pooled) | CD-L, subset | 0.022 | 0.136 | 0.326 | 0.841 |

## Development set

Dropped on 2026-10-01, see the changes of scope above. Its 2 models that are also paper-mirror models are read in the paper-mirror tables.

## Clean training against clean validation

A threshold set on validation is only honest on the training set if clean training images and clean validation images share a PSU distribution. The table gives both medians, the Kolmogorov-Smirnov statistic between them and the FPR each nominal quantile realizes on clean training images, for PSBD-TM (PSBD-RD on ResNet-18) at our rate.

| model | PSU form | median train | median validation | KS statistic | realized FPR at 1% | at 5% | at 10% | at 25% |
|---|---|---|---|---|---|---|---|---|
| BadNets, CIFAR-10, ViT-B/16, PSBD-TM | fractional | 0.941 | 0.945 | 0.023 | 0.005 | 0.049 | 0.099 | 0.245 |
| BadNets, CIFAR-10, ViT-B/16, PSBD-TM | absolute | 0.931 | 0.923 | 0.033 | 0.005 | 0.049 | 0.097 | 0.220 |
| Blend, CIFAR-10, ViT-B/16, PSBD-TM | fractional | 0.889 | 0.889 | 0.013 | 0.006 | 0.046 | 0.089 | 0.248 |
| Blend, CIFAR-10, ViT-B/16, PSBD-TM | absolute | 0.883 | 0.878 | 0.035 | 0.004 | 0.033 | 0.080 | 0.227 |
| WaNet, CIFAR-10, ViT-B/16, PSBD-TM | fractional | 0.902 | 0.898 | 0.032 | 0.008 | 0.045 | 0.097 | 0.240 |
| WaNet, CIFAR-10, ViT-B/16, PSBD-TM | absolute | 0.894 | 0.878 | 0.059 | 0.008 | 0.043 | 0.093 | 0.216 |
| BadNets, GTSRB, ViT-B/16, PSBD-TM | fractional | 0.958 | 0.971 | 0.072 | 0.012 | 0.050 | 0.103 | 0.277 |
| BadNets, GTSRB, ViT-B/16, PSBD-TM | absolute | 0.954 | 0.956 | 0.023 | 0.012 | 0.043 | 0.088 | 0.238 |
| Blend, GTSRB, ViT-B/16, PSBD-TM | fractional | 0.968 | 0.968 | 0.022 | 0.010 | 0.043 | 0.091 | 0.241 |
| Blend, GTSRB, ViT-B/16, PSBD-TM | absolute | 0.967 | 0.965 | 0.059 | 0.000 | 0.017 | 0.062 | 0.197 |
| BadNets, Tiny ImageNet, ViT-B/16, PSBD-TM | fractional | 0.994 | 0.992 | 0.079 | 0.014 | 0.059 | 0.098 | 0.211 |
| BadNets, Tiny ImageNet, ViT-B/16, PSBD-TM | absolute | 0.988 | 0.965 | 0.234 | 0.010 | 0.021 | 0.035 | 0.094 |
| Blend, Tiny ImageNet, ViT-B/16, PSBD-TM | fractional | 0.993 | 0.992 | 0.075 | 0.009 | 0.056 | 0.096 | 0.212 |
| Blend, Tiny ImageNet, ViT-B/16, PSBD-TM | absolute | 0.987 | 0.963 | 0.247 | 0.009 | 0.017 | 0.028 | 0.084 |
| WaNet, Tiny ImageNet, ViT-B/16, PSBD-TM | fractional | 0.992 | 0.992 | 0.059 | 0.011 | 0.054 | 0.115 | 0.236 |
| WaNet, Tiny ImageNet, ViT-B/16, PSBD-TM | absolute | 0.986 | 0.956 | 0.240 | 0.010 | 0.027 | 0.047 | 0.107 |
| Label-Consistent, GTSRB, ViT-B/16, PSBD-TM | fractional | 0.947 | 0.952 | 0.027 | 0.007 | 0.047 | 0.095 | 0.249 |
| Label-Consistent, GTSRB, ViT-B/16, PSBD-TM | absolute | 0.944 | 0.934 | 0.042 | 0.007 | 0.044 | 0.089 | 0.229 |
| BadNets, GTSRB, ResNet-18, PSBD-RD | fractional | 0.982 | 0.984 | 0.046 | 0.010 | 0.055 | 0.116 | 0.290 |
| BadNets, GTSRB, ResNet-18, PSBD-RD | absolute | 0.981 | 0.979 | 0.053 | 0.004 | 0.020 | 0.055 | 0.208 |
| Blend, GTSRB, ResNet-18, PSBD-RD | fractional | 0.965 | 0.968 | 0.037 | 0.010 | 0.059 | 0.111 | 0.272 |
| Blend, GTSRB, ResNet-18, PSBD-RD | absolute | 0.964 | 0.957 | 0.059 | 0.001 | 0.018 | 0.062 | 0.200 |

TPR with the threshold at each quantile of clean validation, and the FPR it realizes on clean training images, fractional PSU at our rate.

| model | method | 1% | 5% | 10% | 25% |
|---|---|---|---|---|---|
| BadNets, CIFAR-10, ViT-B/16 | final method | 0.003/0.004 | 0.667/0.046 | 0.817/0.094 | 0.999/0.236 |
| BadNets, CIFAR-10, ViT-B/16 | PSBD-TM | 0.472/0.005 | 0.755/0.049 | 0.933/0.099 | 1.000/0.245 |
| BadNets, CIFAR-10, ViT-B/16 | PSBD-RD | 0.002/0.006 | 0.045/0.046 | 0.109/0.087 | 0.414/0.242 |
| Blend, CIFAR-10, ViT-B/16 | final method | 0.520/0.004 | 0.802/0.033 | 0.898/0.086 | 0.970/0.242 |
| Blend, CIFAR-10, ViT-B/16 | PSBD-TM | 0.547/0.006 | 0.702/0.046 | 0.748/0.089 | 0.868/0.248 |
| Blend, CIFAR-10, ViT-B/16 | PSBD-RD | 0.829/0.004 | 0.966/0.046 | 0.982/0.082 | 0.995/0.227 |
| WaNet, CIFAR-10, ViT-B/16 | final method | 0.619/0.008 | 0.880/0.045 | 0.959/0.098 | 0.985/0.263 |
| WaNet, CIFAR-10, ViT-B/16 | PSBD-TM | 0.016/0.008 | 0.038/0.045 | 0.087/0.097 | 0.320/0.240 |
| WaNet, CIFAR-10, ViT-B/16 | PSBD-RD | 0.280/0.011 | 0.698/0.057 | 0.915/0.116 | 0.984/0.267 |
| BadNets, GTSRB, ViT-B/16 (not pooled) | final method | 1.000/0.011 | 1.000/0.042 | 1.000/0.096 | 1.000/0.257 |
| BadNets, GTSRB, ViT-B/16 (not pooled) | PSBD-TM | 0.997/0.012 | 1.000/0.050 | 1.000/0.103 | 1.000/0.277 |
| BadNets, GTSRB, ViT-B/16 (not pooled) | PSBD-RD | 0.994/0.009 | 0.998/0.044 | 0.999/0.095 | 1.000/0.237 |
| Blend, GTSRB, ViT-B/16 (not pooled) | final method | 1.000/0.004 | 1.000/0.048 | 1.000/0.106 | 1.000/0.235 |
| Blend, GTSRB, ViT-B/16 (not pooled) | PSBD-TM | 1.000/0.010 | 1.000/0.043 | 1.000/0.091 | 1.000/0.241 |
| Blend, GTSRB, ViT-B/16 (not pooled) | PSBD-RD | 0.966/0.013 | 0.986/0.060 | 0.993/0.108 | 0.998/0.239 |
| BadNets, Tiny ImageNet, ViT-B/16 | final method | 0.990/0.014 | 1.000/0.064 | 1.000/0.110 | 1.000/0.239 |
| BadNets, Tiny ImageNet, ViT-B/16 | PSBD-TM | 0.863/0.014 | 1.000/0.059 | 1.000/0.098 | 1.000/0.211 |
| BadNets, Tiny ImageNet, ViT-B/16 | PSBD-RD | 0.520/0.013 | 0.876/0.054 | 0.951/0.100 | 0.991/0.219 |
| Blend, Tiny ImageNet, ViT-B/16 | final method | 0.992/0.011 | 0.999/0.052 | 1.000/0.098 | 1.000/0.230 |
| Blend, Tiny ImageNet, ViT-B/16 | PSBD-TM | 0.972/0.009 | 0.997/0.056 | 0.999/0.096 | 1.000/0.212 |
| Blend, Tiny ImageNet, ViT-B/16 | PSBD-RD | 0.994/0.011 | 0.999/0.047 | 1.000/0.092 | 1.000/0.223 |
| WaNet, Tiny ImageNet, ViT-B/16 | final method | 0.656/0.009 | 0.946/0.054 | 0.980/0.111 | 0.997/0.239 |
| WaNet, Tiny ImageNet, ViT-B/16 | PSBD-TM | 0.306/0.011 | 0.783/0.054 | 0.958/0.115 | 0.995/0.236 |
| WaNet, Tiny ImageNet, ViT-B/16 | PSBD-RD | 0.702/0.009 | 0.891/0.047 | 0.952/0.102 | 0.988/0.229 |
| Label-Consistent, GTSRB, ViT-B/16 (not pooled) | final method | 0.899/0.006 | 0.973/0.048 | 0.995/0.097 | 0.998/0.269 |
| Label-Consistent, GTSRB, ViT-B/16 (not pooled) | PSBD-TM | 0.922/0.007 | 0.983/0.047 | 0.997/0.095 | 0.998/0.249 |
| Label-Consistent, GTSRB, ViT-B/16 (not pooled) | PSBD-RD | 0.000/0.007 | 0.002/0.056 | 0.011/0.107 | 0.107/0.277 |
| BadNets, GTSRB, ResNet-18 (not pooled) | PSBD-RD | 0.999/0.010 | 1.000/0.055 | 1.000/0.116 | 1.000/0.290 |
| Blend, GTSRB, ResNet-18 (not pooled) | PSBD-RD | 0.559/0.010 | 0.856/0.059 | 0.919/0.111 | 0.976/0.272 |

## Training set against test time

AUROC on the training images next to AUROC on the paired test splits, from the `cli.sweep` caches and the `cli.baselines` records of the same models.

| model | final method train / test | PSBD-TM train / test | PSBD-RD train / test | STRIP train / test | CD-L train / test |
|---|---|---|---|---|---|
| BadNets, CIFAR-10, ViT-B/16 | 0.958 / 0.976 | 0.972 / 0.991 | 0.690 / 0.720 | 0.994 / 0.997 | 1.000 / 1.000 |
| Blend, CIFAR-10, ViT-B/16 | 0.969 / 0.963 | 0.913 / 0.906 | 0.993 / 0.991 | 0.961 / 0.960 | 0.914 / 0.936 |
| WaNet, CIFAR-10, ViT-B/16 | 0.974 / 0.935 | 0.505 / 0.459 | 0.949 / 0.947 | 0.477 / 0.504 | 0.629 / 0.579 |
| BadNets, GTSRB, ViT-B/16 (not pooled) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.999 | 0.977 / 0.979 | 1.000 / 1.000 |
| Blend, GTSRB, ViT-B/16 (not pooled) | 1.000 / 0.999 | 1.000 / 1.000 | 0.997 / 0.998 | 0.963 / 0.966 | 0.997 / 0.998 |
| BadNets, Tiny ImageNet, ViT-B/16 | 0.999 / 0.998 | 0.993 / 0.993 | 0.973 / 0.973 | 0.921 / 0.931 | 0.850 / 0.846 |
| Blend, Tiny ImageNet, ViT-B/16 | 0.998 / 0.998 | 0.999 / 0.998 | 0.999 / 0.999 | 0.978 / 0.975 | 0.434 / 0.408 |
| WaNet, Tiny ImageNet, ViT-B/16 | 0.985 / 0.968 | 0.963 / 0.950 | 0.981 / 0.966 | 0.492 / 0.532 | 0.921 / 0.906 |
| Label-Consistent, GTSRB, ViT-B/16 (not pooled) | 0.995 / 0.982 | 0.997 / 0.969 | 0.439 / 0.444 | 0.996 / 0.854 | 0.833 / 0.723 |
| BadNets, GTSRB, ResNet-18 (not pooled) | n/a / n/a | n/a / n/a | 1.000 / 1.000 | 0.998 / n/a | 0.993 / n/a |
| Blend, GTSRB, ResNet-18 (not pooled) | n/a / n/a | n/a / n/a | 0.965 / 0.968 | 0.898 / n/a | 0.841 / n/a |

On the paper-mirror set (6 pooled models) PSBD-TM minus PSBD-RD is -0.040 AUROC on training images, interval [-0.229, +0.128], against -0.050 on the paired test splits of the same models.

## Rates under both rules

| model | placement | our rate | Li et al.'s rate | Li et al.'s candidates |
|---|---|---|---|---|
| BadNets, CIFAR-10, ViT-B/16 | psbd_tm | 0.600 | 0.900 | 0.6, 0.7, 0.8, 0.9 |
| BadNets, CIFAR-10, ViT-B/16 | psbd_rd | 0.090 | 0.100 | 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 |
| BadNets, CIFAR-10, ViT-B/16 | middle_band | 0.900 | 0.900 | 0.9, 0.95, 0.99 |
| Blend, CIFAR-10, ViT-B/16 | psbd_tm | 0.600 | 0.600 | 0.6, 0.7, 0.8, 0.9 |
| Blend, CIFAR-10, ViT-B/16 | psbd_rd | 0.090 | 0.090 | 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 |
| Blend, CIFAR-10, ViT-B/16 | middle_band | 0.900 | 0.900 | 0.9, 0.95, 0.99 |
| WaNet, CIFAR-10, ViT-B/16 | psbd_tm | 0.500 | 0.600 | 0.5, 0.6, 0.7, 0.8, 0.9 |
| WaNet, CIFAR-10, ViT-B/16 | psbd_rd | 0.090 | 0.090 | 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 |
| WaNet, CIFAR-10, ViT-B/16 | middle_band | 0.800 | 0.800 | 0.8, 0.9, 0.95, 0.99 |
| BadNets, GTSRB, ViT-B/16 | psbd_tm | 0.500 | 0.700 | 0.5, 0.6, 0.7, 0.8, 0.9 |
| BadNets, GTSRB, ViT-B/16 | psbd_rd | 0.090 | 0.100 | 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 |
| BadNets, GTSRB, ViT-B/16 | middle_band | 0.800 | 0.900 | 0.8, 0.9, 0.95, 0.99 |
| Blend, GTSRB, ViT-B/16 | psbd_tm | 0.300 | 0.500 | 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 |
| Blend, GTSRB, ViT-B/16 | psbd_rd | 0.050 | 0.050 | 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 |
| Blend, GTSRB, ViT-B/16 | middle_band | 0.500 | 0.700 | 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99 |
| BadNets, Tiny ImageNet, ViT-B/16 | psbd_tm | 0.500 | 0.500 | 0.5, 0.6, 0.7, 0.8, 0.9 |
| BadNets, Tiny ImageNet, ViT-B/16 | psbd_rd | 0.050 | 0.050 | 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 |
| BadNets, Tiny ImageNet, ViT-B/16 | middle_band | 0.700 | 0.700 | 0.7, 0.8, 0.9, 0.95, 0.99 |
| Blend, Tiny ImageNet, ViT-B/16 | psbd_tm | 0.500 | 0.700 | 0.5, 0.6, 0.7, 0.8, 0.9 |
| Blend, Tiny ImageNet, ViT-B/16 | psbd_rd | 0.050 | 0.070 | 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 |
| Blend, Tiny ImageNet, ViT-B/16 | middle_band | 0.700 | 0.800 | 0.7, 0.8, 0.9, 0.95, 0.99 |
| WaNet, Tiny ImageNet, ViT-B/16 | psbd_tm | 0.500 | 0.600 | 0.5, 0.6, 0.7, 0.8, 0.9 |
| WaNet, Tiny ImageNet, ViT-B/16 | psbd_rd | 0.050 | 0.050 | 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 |
| WaNet, Tiny ImageNet, ViT-B/16 | middle_band | 0.700 | 0.700 | 0.7, 0.8, 0.9, 0.95, 0.99 |
| Label-Consistent, GTSRB, ViT-B/16 | psbd_tm | 0.300 | 0.300 | 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 |
| Label-Consistent, GTSRB, ViT-B/16 | psbd_rd | 0.070 | 0.070 | 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 |
| Label-Consistent, GTSRB, ViT-B/16 | middle_band | 0.700 | 0.700 | 0.7, 0.8, 0.9, 0.95, 0.99 |
| BadNets, GTSRB, ResNet-18 | psbd_rd | 0.500 | 0.500 | 0.5, 0.6, 0.7, 0.8, 0.9 |
| Blend, GTSRB, ResNet-18 | psbd_rd | 0.500 | 0.500 | 0.5, 0.6, 0.7, 0.8, 0.9 |

## Reconstruction checks

R1 compares the rebuilt counts with `args.json`, R2 asks whether the model sends the rebuilt poisoned images to their attack-success label, and R3 asks whether the untriggered image of a rebuilt poisoned index looks held out (the ratio of its loss gap to the validation gap, at least 0.5 to pass).

| model | successful_2pt | seed recorded | R1 counts | R2 training attack success (recorded ASR) | R3 ratio | verdict |
|---|---|---|---|---|---|---|
| `vit_cifar10_badnet_a2o_0_1` | yes | no | passed | 1.000 (1.000) | 0.805, passed | verified |
| `vit_cifar10_blend_0_1` | yes | no | passed | 1.000 (1.000) | 1.214, passed | verified |
| `vit_cifar10_wanet_0_1` | yes | yes | passed | 0.978 (0.890) | 1.228, passed | verified |
| `vit_gtsrb_badnet_a2o_0_1` | yes | no | passed | 1.000 (1.000) | 0.132, failed | failed |
| `vit_gtsrb_blend_0_1` | yes | no | passed | 1.000 (1.000) | 0.073, inconclusive, training and validation loss too close | ambiguous |
| `vit_tiny_badnet_a2o_0_1` | yes | no | passed | 1.000 (1.000) | 0.978, passed | verified |
| `vit_tiny_blend_0_1` | yes | no | passed | 1.000 (1.000) | 1.000, passed | verified |
| `vit_tiny_wanet_0_1` | yes | yes | passed | 0.996 (0.968) | 1.058, passed | verified |
| `vit_gtsrb_lc_0_05_tl1_adv` | yes | yes | passed | 1.000 (0.958) | 0.753, not applicable, the poisoned images were trained with their true label | count and recorded seed only |
| `resnet18_gtsrb_badnet_a2o_0_1` | no | yes | passed | 1.000 (1.000) | 0.073, failed | failed |
| `resnet18_gtsrb_blend_0_1` | no | yes | passed | 1.000 (1.000) | 0.053, failed | failed |

## Verdicts

- P1 on the paper-mirror set: refuted, `{"auroc": {"final_min": 0.9806438525, "psbd_tm": 0.8907883133333333, "psbd_rd": 0.9309423833333333}, "tm_leads_rd_at_realized_1_5_10": [false, false, false]}`
- P2 on the paper-mirror set: refuted, `{"train_gain": -0.04015407000000001, "test_gain": -0.04980057562774779}`
- P3 on the paper-mirror set: refuted, `{"li_tm": 0.8923820633333334, "li_rd": 0.9335951291666666}`
- P6 on the paper-mirror set: holds, `{"train_order": ["psbd_tm", "strip", "cd_l"], "test_order": ["psbd_tm", "strip", "cd_l"]}`
- P4 on the paper-mirror set: fractional part refuted (median below on 1 of 6, FPR above 0.25 on 0 of 6), absolute part holds (lower on 6 of 6).
- P5: holds, `{"models": {"resnet18_gtsrb_badnet_a2o_0_1": {"tpr": 1.0, "fpr_clean_train": 0.20810000598430634, "fpr_paper_convention": 0.20810000598430634, "holds": true}, "resnet18_gtsrb_blend_0_1": {"tpr": 0.9613363146781921, "fpr_clean_train": 0.19979999959468842, "fpr_paper_convention": 0.19979999959468842, "holds": true}}}`
- P7: holds, `{"reached": {"vit_cifar10_badnet_a2o_0_1": true, "vit_cifar10_blend_0_1": true, "vit_cifar10_wanet_0_1": false, "vit_gtsrb_badnet_a2o_0_1": true, "vit_gtsrb_blend_0_1": true, "vit_tiny_badnet_a2o_0_1": true, "vit_tiny_blend_0_1": true, "vit_tiny_wanet_0_1": false, "vit_gtsrb_lc_0_05_tl1_adv": true}, "count": "7 of 9"}`

<!-- results:end -->
