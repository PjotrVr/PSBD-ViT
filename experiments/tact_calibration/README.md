# TaCT calibration of PSBD-TM

PSBD-TM detects TaCT with a high AUROC and almost no TPR at a deployable FPR. The AUROC pairs each triggered input with its own clean copy (a source-class image), and those images are more fragile under masking than the clean images of every class the threshold is set on. This experiment asks whether a label-free calibration can move the threshold toward the source classes without losing what PSBD-TM already catches, and how many forward passes the score needs. The rules, the shrinkage choice and every prediction were frozen in `PREDICTIONS.md` (commit of `preregistration.json`) after reading the 4 tuning models only.

## Method

Every model was swept once at k = 20 at each placement's adaptive rate, PSBD-TM (`before_attention_norm_token_mask`) and `pre_residual_blocks_5_8`, with `cli.sweep`'s own functions (bfloat16, batch 64, mask seed 0) into `results/_experiments/tact_calibration/models/`. A smaller k is the first k passes, read at 3, 5, 10, 20. The 6 holdout retrains had no `psbd_metrics.json`, so `run_passes.py` walked each placement's ladder from `configs/psbd_basis.json` at 3 passes on the standard 2000 validation images and took the first rate whose shift ratio reached the target. Validation grew to cifar10 4000, cifar100 5000, gtsrb 4000, tiny 5000 images from the same split permutation, and those images left the clean and triggered evaluation sets, which `analyze.enlarged_rows` asserts. Beatrix and TED were rescored on the same enlarged validation and shrunken evaluation sets from their stored scores.

The fallback score ranks an input's fractional PSU among validation images with the same fallback class (the most frequent class the perturbed passes moved to), shrunk toward all validation images with m = 200. The class score is a robust z score by predicted class with median and MAD shrunk toward the global ones with m = 5. The OR rules take the minimum of validation percentiles. The formulas are in `analyze.py`'s docstring. Intervals are paired bootstrap 95% intervals over models (5000 resamples, seed 0). The plan held 63 models, of which 62 unique ones were read into the pooled sets below (tuning 4, holdout 6, non-TaCT development 7, non-TaCT panel 52).

## Commands

```
python -m experiments.tact_calibration.run_passes plan
bash experiments/tact_calibration/queue.sh
python -m experiments.tact_calibration.analyze tuning
python -m experiments.tact_calibration.analyze confirm
python -m experiments.tact_calibration.judge
python -m experiments.tact_calibration.render_readme
```

## Wall times

GPU seconds per model, measured inside the lock, on the shared login-node A100 at memory fraction 0.15. Waiting for the lock is not counted.

| group | models | mean seconds | max seconds | models reusing an earlier PSBD-TM cache |
|---|---|---|---|---|
| tuning | 4 | 409 | 515 | 0 |
| dev | 8 | 650 | 1012 | 4 |
| holdout | 6 | 622 | 832 | 0 |
| rest | 45 | 975 | 2854 | 34 |

## Premise of the fallback score

The share of triggered inputs whose PSBD-TM fallback class is their own source class, against the share that falls to the attractor, the most common fallback class of clean validation images. Shift ratios are the share of perturbed passes that left the unperturbed argmax.

<!-- results:begin -->
| model | group | triggered | own source | attractor class | to attractor | clean validation to attractor | no flip | triggered shift | validation shift |
|---|---|---|---|---|---|---|---|---|---|
| `vit_cifar10_tact_0_01` | tuning | 596 | 0.000 | 9 | 0.899 | 0.533 | 0.005 | 0.991 | 0.827 |
| `vit_cifar10_tact_0_05` | tuning | 596 | 0.000 | 3 | 0.919 | 0.878 | 0.000 | 0.972 | 0.855 |
| `vit_gtsrb_tact_0_01_cos` | tuning | 485 | 0.000 | 38 | 1.000 | 0.945 | 0.000 | 1.000 | 0.935 |
| `vit_gtsrb_tact_0_05` | tuning | 485 | 0.000 | 6 | 0.963 | 0.858 | 0.002 | 0.411 | 0.830 |
| `vit_cifar100_tact_0_01_src5` | holdout | 270 | 0.015 | 91 | 0.233 | 0.319 | 0.000 | 0.446 | 0.920 |
| `vit_cifar100_tact_0_05_src25` | holdout | 1257 | 0.071 | 47 | 0.555 | 0.608 | 0.006 | 0.436 | 0.886 |
| `vit_cifar10_tact_0_1_src5` | holdout | 3040 | 0.220 | 2 | 0.404 | 0.610 | 0.372 | 0.062 | 0.827 |
| `vit_gtsrb_tact_0_1_src12` | holdout | 4454 | 0.066 | 26 | 0.240 | 0.493 | 0.667 | 0.024 | 0.803 |
| `vit_tiny_tact_0_01_src10` | holdout | 246 | 0.057 | 123 | 0.122 | 0.241 | 0.000 | 0.518 | 0.858 |
| `vit_tiny_tact_0_05_src50` | holdout | 1280 | 0.158 | 0 | 0.013 | 0.388 | 0.083 | 0.323 | 0.887 |
<!-- results:end -->

The mean own-source share over the holdout is 0.098 and the mean share falling to the attractor is 0.261. Masking sends a triggered input where it sends its clean source image, to a class the model falls back to under heavy masking, so the fallback class carries no information about the source and the fallback score reduces to the plain one.

## TaCT detection

TPR at 1%, 5% and 10% FPR, then AUROC, at k = 20. FPR is the quantile of the enlarged clean validation set. Beatrix and TED are read on the same sets, and the holdout has no detector records.

<!-- results:begin -->
**Tuning models**

| model | method | TPR 1% | TPR 5% | TPR 10% | AUROC |
|---|---|---|---|---|---|
| `vit_cifar10_tact_0_01` | PSBD-TM | 0.005 | 0.005 | 0.008 | 0.980 |
| `vit_cifar10_tact_0_01` | PSBD-TM OR fallback | 0.003 | 0.005 | 0.007 | 0.980 |
| `vit_cifar10_tact_0_01` | PSBD-TM OR class | 0.003 | 0.007 | 0.008 | 0.805 |
| `vit_cifar10_tact_0_01` | final method | 0.003 | 0.005 | 0.007 | 0.966 |
| `vit_cifar10_tact_0_01` | final OR fallback | 0.003 | 0.005 | 0.008 | 0.966 |
| `vit_cifar10_tact_0_01` | final OR class | 0.003 | 0.007 | 0.008 | 0.833 |
| `vit_cifar10_tact_0_01` | beatrix | 0.534 | 0.914 | 0.995 | 0.997 |
| `vit_cifar10_tact_0_01` | ted | 0.723 | 0.814 | 0.872 | 0.929 |
| `vit_cifar10_tact_0_05` | PSBD-TM | 0.000 | 0.000 | 0.000 | 0.972 |
| `vit_cifar10_tact_0_05` | PSBD-TM OR fallback | 0.000 | 0.000 | 0.003 | 0.972 |
| `vit_cifar10_tact_0_05` | PSBD-TM OR class | 0.143 | 0.399 | 0.534 | 0.929 |
| `vit_cifar10_tact_0_05` | final method | 0.000 | 0.000 | 0.000 | 0.974 |
| `vit_cifar10_tact_0_05` | final OR fallback | 0.000 | 0.000 | 0.003 | 0.974 |
| `vit_cifar10_tact_0_05` | final OR class | 0.141 | 0.362 | 0.507 | 0.929 |
| `vit_cifar10_tact_0_05` | beatrix | 0.601 | 0.881 | 0.988 | 0.997 |
| `vit_cifar10_tact_0_05` | ted | 0.854 | 0.921 | 0.950 | 0.968 |
| `vit_gtsrb_tact_0_01_cos` | PSBD-TM | 0.000 | 0.000 | 0.000 | 0.253 |
| `vit_gtsrb_tact_0_01_cos` | PSBD-TM OR fallback | 0.000 | 0.000 | 0.000 | 0.253 |
| `vit_gtsrb_tact_0_01_cos` | PSBD-TM OR class | 0.000 | 0.000 | 0.000 | 0.526 |
| `vit_gtsrb_tact_0_01_cos` | final method | 0.037 | 0.470 | 0.730 | 0.920 |
| `vit_gtsrb_tact_0_01_cos` | final OR fallback | 0.033 | 0.336 | 0.670 | 0.920 |
| `vit_gtsrb_tact_0_01_cos` | final OR class | 0.008 | 0.270 | 0.612 | 0.885 |
| `vit_gtsrb_tact_0_01_cos` | beatrix | 1.000 | 1.000 | 1.000 | 1.000 |
| `vit_gtsrb_tact_0_01_cos` | ted | 0.852 | 1.000 | 1.000 | 0.997 |
| `vit_gtsrb_tact_0_05` | PSBD-TM | 0.000 | 0.097 | 0.584 | 1.000 |
| `vit_gtsrb_tact_0_05` | PSBD-TM OR fallback | 0.000 | 0.097 | 0.569 | 1.000 |
| `vit_gtsrb_tact_0_05` | PSBD-TM OR class | 0.984 | 1.000 | 1.000 | 1.000 |
| `vit_gtsrb_tact_0_05` | final method | 0.000 | 0.014 | 0.099 | 1.000 |
| `vit_gtsrb_tact_0_05` | final OR fallback | 0.000 | 0.014 | 0.113 | 1.000 |
| `vit_gtsrb_tact_0_05` | final OR class | 0.979 | 1.000 | 1.000 | 1.000 |
| `vit_gtsrb_tact_0_05` | beatrix | 1.000 | 1.000 | 1.000 | 1.000 |
| `vit_gtsrb_tact_0_05` | ted | 0.913 | 0.996 | 1.000 | 0.998 |

**Tuning means, paired against the reference**

| method | reference | TPR 1% | TPR 5% | TPR 10% | AUROC |
|---|---|---|---|---|---|
| PSBD-TM OR fallback | PSBD-TM | 0.001 (-0.000 [-0.001, +0.000]) | 0.025 (+0.000 [+0.000, +0.000]) | 0.145 (-0.003 [-0.011, +0.002]) | 0.801 (-0.000 [-0.000, +0.000]) |
| PSBD-TM OR class | PSBD-TM | 0.282 (+0.281 [-0.001, +0.738]) | 0.352 (+0.326 [+0.001, +0.678]) | 0.385 (+0.238 [+0.000, +0.475]) | 0.815 (+0.013 [-0.131, +0.194]) |
| final method | PSBD-TM | 0.010 (+0.009 [-0.001, +0.028]) | 0.122 (+0.097 [-0.062, +0.353]) | 0.209 (+0.061 [-0.363, +0.547]) | 0.965 (+0.163 [-0.010, +0.500]) |
| final OR fallback | final method | 0.009 (-0.001 [-0.003, +0.000]) | 0.089 (-0.034 [-0.101, +0.000]) | 0.199 (-0.010 [-0.044, +0.011]) | 0.965 (+0.000 [-0.000, +0.000]) |
| final OR class | final method | 0.283 (+0.273 [-0.014, +0.735]) | 0.410 (+0.287 [-0.099, +0.740]) | 0.532 (+0.323 [-0.058, +0.704]) | 0.912 (-0.053 [-0.108, -0.011]) |

**Holdout models**

| model | method | TPR 1% | TPR 5% | TPR 10% | AUROC |
|---|---|---|---|---|---|
| `vit_cifar100_tact_0_01_src5` | PSBD-TM | 0.011 | 0.852 | 0.967 | 0.976 |
| `vit_cifar100_tact_0_01_src5` | PSBD-TM OR fallback | 0.089 | 0.852 | 0.959 | 0.978 |
| `vit_cifar100_tact_0_01_src5` | PSBD-TM OR class | 0.363 | 0.981 | 0.985 | 0.987 |
| `vit_cifar100_tact_0_01_src5` | final method | 0.004 | 0.622 | 0.907 | 0.974 |
| `vit_cifar100_tact_0_01_src5` | final OR fallback | 0.063 | 0.707 | 0.911 | 0.977 |
| `vit_cifar100_tact_0_01_src5` | final OR class | 0.107 | 0.967 | 0.981 | 0.986 |
| `vit_cifar100_tact_0_05_src25` | PSBD-TM | 0.002 | 0.516 | 0.857 | 0.963 |
| `vit_cifar100_tact_0_05_src25` | PSBD-TM OR fallback | 0.006 | 0.550 | 0.862 | 0.965 |
| `vit_cifar100_tact_0_05_src25` | PSBD-TM OR class | 0.411 | 0.933 | 0.971 | 0.983 |
| `vit_cifar100_tact_0_05_src25` | final method | 0.008 | 0.233 | 0.655 | 0.933 |
| `vit_cifar100_tact_0_05_src25` | final OR fallback | 0.008 | 0.305 | 0.707 | 0.939 |
| `vit_cifar100_tact_0_05_src25` | final OR class | 0.321 | 0.878 | 0.956 | 0.978 |
| `vit_cifar10_tact_0_1_src5` | PSBD-TM | 0.972 | 0.996 | 0.996 | 0.994 |
| `vit_cifar10_tact_0_1_src5` | PSBD-TM OR fallback | 0.991 | 0.996 | 0.996 | 0.996 |
| `vit_cifar10_tact_0_1_src5` | PSBD-TM OR class | 0.898 | 0.996 | 0.997 | 0.995 |
| `vit_cifar10_tact_0_1_src5` | final method | 0.898 | 0.993 | 0.996 | 0.993 |
| `vit_cifar10_tact_0_1_src5` | final OR fallback | 0.981 | 0.996 | 0.996 | 0.996 |
| `vit_cifar10_tact_0_1_src5` | final OR class | 0.856 | 0.994 | 0.997 | 0.993 |
| `vit_gtsrb_tact_0_1_src12` | PSBD-TM | 0.002 | 0.970 | 1.000 | 1.000 |
| `vit_gtsrb_tact_0_1_src12` | PSBD-TM OR fallback | 0.232 | 0.747 | 1.000 | 0.996 |
| `vit_gtsrb_tact_0_1_src12` | PSBD-TM OR class | 0.470 | 0.984 | 1.000 | 0.994 |
| `vit_gtsrb_tact_0_1_src12` | final method | 0.152 | 0.805 | 1.000 | 1.000 |
| `vit_gtsrb_tact_0_1_src12` | final OR fallback | 0.338 | 0.794 | 1.000 | 0.998 |
| `vit_gtsrb_tact_0_1_src12` | final OR class | 0.244 | 0.975 | 0.999 | 0.994 |
| `vit_tiny_tact_0_01_src10` | PSBD-TM | 0.008 | 0.813 | 0.959 | 0.975 |
| `vit_tiny_tact_0_01_src10` | PSBD-TM OR fallback | 0.020 | 0.768 | 0.955 | 0.975 |
| `vit_tiny_tact_0_01_src10` | PSBD-TM OR class | 0.008 | 0.130 | 0.935 | 0.941 |
| `vit_tiny_tact_0_01_src10` | final method | 0.008 | 0.037 | 0.833 | 0.968 |
| `vit_tiny_tact_0_01_src10` | final OR fallback | 0.008 | 0.175 | 0.870 | 0.970 |
| `vit_tiny_tact_0_01_src10` | final OR class | 0.008 | 0.020 | 0.496 | 0.934 |
| `vit_tiny_tact_0_05_src50` | PSBD-TM | 0.005 | 0.969 | 0.987 | 0.959 |
| `vit_tiny_tact_0_05_src50` | PSBD-TM OR fallback | 0.016 | 0.952 | 0.987 | 0.960 |
| `vit_tiny_tact_0_05_src50` | PSBD-TM OR class | 0.005 | 0.614 | 0.985 | 0.932 |
| `vit_tiny_tact_0_05_src50` | final method | 0.341 | 0.780 | 0.984 | 0.973 |
| `vit_tiny_tact_0_05_src50` | final OR fallback | 0.324 | 0.810 | 0.986 | 0.973 |
| `vit_tiny_tact_0_05_src50` | final OR class | 0.298 | 0.670 | 0.966 | 0.960 |

**Holdout means, paired against the reference**

| method | reference | TPR 1% | TPR 5% | TPR 10% | AUROC |
|---|---|---|---|---|---|
| PSBD-TM OR fallback | PSBD-TM | 0.226 (+0.059 [+0.010, +0.132]) | 0.811 (-0.042 [-0.119, +0.009]) | 0.960 (-0.001 [-0.004, +0.002]) | 0.978 (+0.000 [-0.001, +0.002]) |
| PSBD-TM OR class | PSBD-TM | 0.359 (+0.192 [+0.031, +0.370]) | 0.773 (-0.079 [-0.383, +0.185]) | 0.979 (+0.018 [-0.009, +0.057]) | 0.972 (-0.006 [-0.023, +0.009]) |
| final method | PSBD-TM | 0.235 (+0.068 [-0.026, +0.190]) | 0.578 (-0.274 [-0.492, -0.114]) | 0.896 (-0.065 [-0.131, -0.010]) | 0.973 (-0.005 [-0.016, +0.005]) |
| final OR fallback | final method | 0.287 (+0.052 [+0.004, +0.114]) | 0.631 (+0.053 [+0.011, +0.094]) | 0.912 (+0.016 [+0.001, +0.033]) | 0.975 (+0.002 [+0.000, +0.004]) |
| final OR class | final method | 0.306 (+0.071 [-0.013, +0.181]) | 0.751 (+0.172 [-0.014, +0.387]) | 0.899 (+0.003 [-0.156, +0.148]) | 0.974 (+0.001 [-0.017, +0.021]) |

<!-- results:end -->

## Harm on the other attacks

Mean at k = 20 with the paired difference to the reference and its interval. The development set is the 7 successful non-TaCT models of `experiments/cache_readouts/dev_set.json`, the panel set every successful non-TaCT panel model read, development included.

<!-- results:begin -->
**Non-TaCT development models**

| method | reference | TPR 1% | TPR 5% | TPR 10% | AUROC |
|---|---|---|---|---|---|
| PSBD-TM OR fallback | PSBD-TM | 0.613 (+0.017 [+0.006, +0.029]) | 0.699 (+0.021 [-0.004, +0.043]) | 0.743 (+0.021 [+0.007, +0.037]) | 0.871 (+0.004 [+0.001, +0.007]) |
| PSBD-TM OR class | PSBD-TM | 0.810 (+0.215 [+0.123, +0.309]) | 0.865 (+0.187 [+0.086, +0.300]) | 0.885 (+0.163 [+0.062, +0.285]) | 0.943 (+0.076 [+0.018, +0.159]) |
| final method | PSBD-TM | 0.677 (+0.082 [-0.034, +0.262]) | 0.834 (+0.156 [-0.010, +0.401]) | 0.867 (+0.145 [-0.015, +0.380]) | 0.943 (+0.076 [-0.003, +0.207]) |
| final OR fallback | final method | 0.695 (+0.018 [+0.002, +0.036]) | 0.841 (+0.007 [-0.014, +0.030]) | 0.877 (+0.010 [-0.002, +0.028]) | 0.945 (+0.002 [-0.001, +0.006]) |
| final OR class | final method | 0.850 (+0.173 [+0.059, +0.302]) | 0.927 (+0.093 [+0.005, +0.202]) | 0.944 (+0.077 [+0.001, +0.179]) | 0.971 (+0.028 [+0.002, +0.063]) |

**Non-TaCT panel models**

| method | reference | TPR 1% | TPR 5% | TPR 10% | AUROC |
|---|---|---|---|---|---|
| PSBD-TM OR fallback | PSBD-TM | 0.795 (+0.003 [-0.014, +0.024]) | 0.913 (+0.005 [-0.001, +0.012]) | 0.938 (+0.005 [-0.000, +0.009]) | 0.967 (+0.001 [-0.000, +0.002]) |
| PSBD-TM OR class | PSBD-TM | 0.775 (-0.017 [-0.062, +0.029]) | 0.934 (+0.027 [+0.001, +0.057]) | 0.957 (+0.024 [+0.001, +0.051]) | 0.976 (+0.010 [-0.001, +0.025]) |
| final method | PSBD-TM | 0.842 (+0.050 [+0.005, +0.105]) | 0.931 (+0.024 [-0.001, +0.063]) | 0.956 (+0.023 [-0.000, +0.060]) | 0.980 (+0.014 [+0.003, +0.033]) |
| final OR fallback | final method | 0.848 (+0.005 [-0.002, +0.014]) | 0.936 (+0.004 [-0.001, +0.011]) | 0.958 (+0.003 [-0.000, +0.006]) | 0.980 (+0.001 [-0.000, +0.001]) |
| final OR class | final method | 0.829 (-0.014 [-0.072, +0.037]) | 0.950 (+0.018 [-0.000, +0.042]) | 0.968 (+0.012 [-0.003, +0.032]) | 0.984 (+0.004 [-0.001, +0.010]) |
<!-- results:end -->

## Pass count

Mean over models at each k for PSBD-TM alone and the final method, the first k of the 20 cached passes.

<!-- results:begin -->
| models | method | k | TPR 1% | TPR 5% | TPR 10% | AUROC |
|---|---|---|---|---|---|---|
| panel (52) | PSBD-TM | 3 | 0.778 | 0.900 | 0.929 | 0.963 |
| panel (52) | PSBD-TM | 5 | 0.783 | 0.904 | 0.931 | 0.965 |
| panel (52) | PSBD-TM | 10 | 0.791 | 0.907 | 0.932 | 0.966 |
| panel (52) | PSBD-TM | 20 | 0.792 | 0.908 | 0.933 | 0.966 |
| panel (52) | final method | 3 | 0.810 | 0.918 | 0.946 | 0.977 |
| panel (52) | final method | 5 | 0.818 | 0.924 | 0.950 | 0.978 |
| panel (52) | final method | 10 | 0.833 | 0.929 | 0.954 | 0.979 |
| panel (52) | final method | 20 | 0.842 | 0.931 | 0.956 | 0.980 |
| panel (52) | PSBD-TM OR class | 3 | 0.755 | 0.924 | 0.951 | 0.972 |
| panel (52) | PSBD-TM OR class | 5 | 0.763 | 0.929 | 0.954 | 0.974 |
| panel (52) | PSBD-TM OR class | 10 | 0.766 | 0.932 | 0.955 | 0.975 |
| panel (52) | PSBD-TM OR class | 20 | 0.775 | 0.934 | 0.957 | 0.976 |
| tuning (4) | PSBD-TM | 3 | 0.001 | 0.056 | 0.135 | 0.798 |
| tuning (4) | PSBD-TM | 5 | 0.002 | 0.052 | 0.140 | 0.805 |
| tuning (4) | PSBD-TM | 10 | 0.002 | 0.042 | 0.139 | 0.803 |
| tuning (4) | PSBD-TM | 20 | 0.001 | 0.025 | 0.148 | 0.801 |
| tuning (4) | final method | 3 | 0.008 | 0.061 | 0.149 | 0.894 |
| tuning (4) | final method | 5 | 0.011 | 0.069 | 0.159 | 0.922 |
| tuning (4) | final method | 10 | 0.006 | 0.090 | 0.197 | 0.945 |
| tuning (4) | final method | 20 | 0.010 | 0.122 | 0.209 | 0.965 |
| tuning (4) | PSBD-TM OR class | 3 | 0.192 | 0.303 | 0.336 | 0.798 |
| tuning (4) | PSBD-TM OR class | 5 | 0.244 | 0.330 | 0.360 | 0.804 |
| tuning (4) | PSBD-TM OR class | 10 | 0.279 | 0.342 | 0.378 | 0.810 |
| tuning (4) | PSBD-TM OR class | 20 | 0.282 | 0.352 | 0.385 | 0.815 |
| holdout (6) | PSBD-TM | 3 | 0.208 | 0.693 | 0.855 | 0.953 |
| holdout (6) | PSBD-TM | 5 | 0.180 | 0.766 | 0.915 | 0.968 |
| holdout (6) | PSBD-TM | 10 | 0.167 | 0.826 | 0.949 | 0.976 |
| holdout (6) | PSBD-TM | 20 | 0.167 | 0.853 | 0.961 | 0.978 |
| holdout (6) | final method | 3 | 0.225 | 0.572 | 0.793 | 0.958 |
| holdout (6) | final method | 5 | 0.208 | 0.578 | 0.853 | 0.967 |
| holdout (6) | final method | 10 | 0.230 | 0.578 | 0.883 | 0.970 |
| holdout (6) | final method | 20 | 0.235 | 0.578 | 0.896 | 0.973 |
| holdout (6) | PSBD-TM OR class | 3 | 0.264 | 0.636 | 0.877 | 0.944 |
| holdout (6) | PSBD-TM OR class | 5 | 0.250 | 0.621 | 0.919 | 0.959 |
| holdout (6) | PSBD-TM OR class | 10 | 0.268 | 0.739 | 0.964 | 0.967 |
| holdout (6) | PSBD-TM OR class | 20 | 0.359 | 0.773 | 0.979 | 0.972 |
<!-- results:end -->

By the pre-registered rule the recommended k is 5: the smallest k whose mean AUROC and TPR at each FPR for PSBD-TM alone on the non-TaCT panel models stay within 0.01 of k = 20. Going from k = 3 to k = 20 changes mean AUROC by +0.003 [+0.002, +0.004].

## Verdicts

| prediction | expected | verdict |
|---|---|---|
| P1 | holds, every tuning model read 0 own-source fallbacks | holds |
| P2_tm_k3 | holds at both k for PSBD-TM alone and for the final method | fails |
| P2_tm_k20 | holds at both k for PSBD-TM alone and for the final method | fails |
| P2_final_k3 | holds at both k for PSBD-TM alone and for the final method | fails |
| P2_final_k20 | holds at both k for PSBD-TM alone and for the final method | fails |
| P3 | fails, the fallback score cannot separate what the fallback class does not separate | fails |
| P3_final | fails, the fallback score cannot separate what the fallback class does not separate | fails |
| P4 | uncertain, the class score gained on 2 of 4 tuning models and nothing on the other 2 | fails |
| P4_final | uncertain, the class score gained on 2 of 4 tuning models and nothing on the other 2 | fails |
| P5_dev_or_fallback | the fallback OR rules hold, the class OR rules fail at 1% FPR, because triggered inputs are ranked against target-class validation images the backdoor made more robust | holds |
| P5_dev_or_class | the fallback OR rules hold, the class OR rules fail at 1% FPR, because triggered inputs are ranked against target-class validation images the backdoor made more robust | holds |
| P5_dev_final_or_fallback | the fallback OR rules hold, the class OR rules fail at 1% FPR, because triggered inputs are ranked against target-class validation images the backdoor made more robust | holds |
| P5_dev_final_or_class | the fallback OR rules hold, the class OR rules fail at 1% FPR, because triggered inputs are ranked against target-class validation images the backdoor made more robust | holds |
| P5_panel_or_fallback | the fallback OR rules hold, the class OR rules fail at 1% FPR, because triggered inputs are ranked against target-class validation images the backdoor made more robust | holds |
| P5_panel_or_class | the fallback OR rules hold, the class OR rules fail at 1% FPR, because triggered inputs are ranked against target-class validation images the backdoor made more robust | holds |
| P5_panel_final_or_fallback | the fallback OR rules hold, the class OR rules fail at 1% FPR, because triggered inputs are ranked against target-class validation images the backdoor made more robust | holds |
| P5_panel_final_or_class | the fallback OR rules hold, the class OR rules fail at 1% FPR, because triggered inputs are ranked against target-class validation images the backdoor made more robust | holds |
| P6_or_fallback | holds | holds |
| P6_or_class | holds | holds |
| P6_final_or_fallback | holds | holds |
| P6_final_or_class | holds | holds |
| P7 | holds | fails |
| P7_auroc_gain | holds | holds |

## Files

`run_passes.py` and `queue.sh` are the GPU stage, `analyze.py` the CPU stage (`tuning` and `confirm`), `judge.py` the verdicts and this renderer the README. Records live under `results/_experiments/tact_calibration/`: `plan.json`, `tuning.json`, `confirm.json`, `verdicts.json`, the per-model caches and run records under `models/` and the queue logs under `logs/`.
