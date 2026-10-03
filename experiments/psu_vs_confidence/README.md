# Is PSU just measuring baseline confidence?

## Question

The skeptical reading of PSBD: a backdoored model is extremely confident on triggered
inputs. PSU subtracts a dropout-perturbed confidence from the no-dropout one. And a
sample starting near probability 1 has more room to fall than 1 starting at 0.6. If
that is the whole story, PSU is an elaborate proxy for baseline confidence and a
defender could skip the `k` stochastic forward passes entirely.

This was raised by an adversarial verification pass, which noted that on the benign
control a confidence-only detector sits as close to chance as PSBD (0.486 against 0.506 in
the saved record, `pre_residual`).

## Run

```bash
PYTHONPATH=. python experiments/psu_vs_confidence/measure.py
```

No GPU, seconds, reads only the cached tensors.

## Finding: the skeptical reading is refuted

### The saved record, a non-canonical reading

`results/_experiments/psu_vs_confidence/psu_vs_confidence.json` holds placement
`pre_residual` (dropout before both residual adds), each checkpoint at the rate whose PSU
AUROC is highest (an oracle rate that flatters PSU), for 295 folders whose name contains
`0_1`. The table this section showed before 2026-09-29 was read at `before_mlp_residual`,
which no saved record holds, so it has been replaced by the saved values. CIFAR-10 ViT at 10%
poisoning, AUROC with backdoor as the positive class:

| checkpoint | PSU | confidence only | fractional drop |
|---|---|---|---|
| `blend` | 0.978 | 0.701 | **0.992** |
| `bpp` | 0.977 | 0.822 | **0.989** |
| `lf` | 0.947 | 0.780 | **0.962** |
| `badnet_a2o` | 0.889 | 0.878 | **0.919** |
| `badnet_a2a` | 0.510 | 0.244 | 0.501 |
| benign control | 0.506 | 0.486 | 0.504 |
| **mean (backdoored)** | **0.860** | **0.685** | **0.873** |

PSU beats confidence-only on 5 of 5 checkpoints, by 0.175 on average. Over the plain
CIFAR-100 and Tiny models of the same record, confidence alone beats PSU on 8 of 44 (2026-09-29
audit, `docs/audits/2026-09-29-experiment-audit.md`).

<!-- results:begin -->
<!-- Everything down to results:end is rendered by panel.py from results/_experiments/psu_vs_confidence/panel.json. -->

### The canonical reading on the current panel

PSBD-TM (`before_attention_norm_token_mask`) at the adaptive 0.8 rule's rate, the paper panel of 56 models successful at the 2-point clean-accuracy bar, fractional and absolute PSU read from each model's `psbd_metrics.json` at q0.25 and confidence alone from the cached baseline:

    PYTHONPATH=. .venv/bin/python experiments/psu_vs_confidence/panel.py

| dataset | n | absolute PSU | fractional PSU | confidence only | confidence wins |
|---|---:|---:|---:|---:|---:|
| cifar10 | 15 | 0.910 | 0.919 | 0.593 | 0 |
| cifar100 | 12 | 0.958 | 0.979 | 0.813 | 0 |
| gtsrb | 15 | 0.941 | 0.935 | 0.433 | 1 |
| tiny | 14 | 0.969 | 0.977 | 0.877 | 0 |
| **all** | **56** | **0.944** | **0.951** | **0.668** | **1** |

Confidence alone beats fractional PSU on 1 of 56 models, `vit_gtsrb_tact_0_05` (0.999 against 0.942). It beats absolute PSU on 3. The stochastic passes do real work at the canonical reading.

Fractional PSU reads 0.951 against 0.944 for the absolute form and beats it on 36 of the 56 panel models. Confidence only reads 0.668 on the same models.
<!-- results:end -->

The decisive column is the fractional drop. `1 - mean_dropout / P_c(x)` divides out the
starting confidence entirely. If PSU worked only because confident samples fall further in
absolute terms, normalizing by that confidence would destroy the signal. It does the
opposite: fractional PSU reads at least as well as absolute PSU on average in both readings
and beats it on most panel models (the count is in the block above). So PSU is measuring how *robust* the prediction
is, not how confident it started.

## The fractional form

Fractional PSU is now the canon headline statistic (`detection_psu_ratio`), with absolute PSU
reported beside it. Its panel mean against the absolute form's is in the block above.
The gain is small and costs nothing: it is the same cached tensors divided by a number
already on disk.

It also has a principled reason to be preferred over the paper's absolute form. The
threshold is a quantile of clean-validation PSU, and absolute PSU is bounded above by
the starting confidence, so the threshold inherits the validation set's confidence
distribution. The ratio does not, which should make it transfer better across
datasets and models with different calibration.

<!-- division:begin -->
<!-- Everything down to division:end is rendered by division.py from results/_experiments/psu_vs_confidence/division.json. -->

## What the division does

Fractional PSU divides the absolute drop by the starting confidence P_c, the unperturbed probability of the predicted class. What that changes is read on the 56-model panel with PSBD-TM (`before_attention_norm_token_mask`) at the adaptive 0.8 rate, from the cached tensors only. Every number below is a mean over models of a per-model value.

    PYTHONPATH=. .venv/bin/python experiments/psu_vs_confidence/division.py

### The denominator

| dataset | P_c below 0.5 (validation) | below 0.9 (validation) | below 0.99 (validation) | below 0.9 (paired clean) | below 0.9 (triggered) | surviving share (validation) | surviving share (triggered) |
|---|---:|---:|---:|---:|---:|---:|---:|
| cifar10 | 0.004 | 0.067 | 0.156 | 0.063 | 0.014 | 0.083 | 0.653 |
| cifar100 | 0.043 | 0.213 | 0.392 | 0.206 | 0.021 | 0.015 | 0.822 |
| gtsrb | 0.007 | 0.033 | 0.092 | 0.031 | 0.060 | 0.031 | 0.844 |
| tiny | 0.076 | 0.277 | 0.460 | 0.280 | 0.019 | 0.006 | 0.942 |
| all | 0.031 | 0.142 | 0.265 | 0.139 | 0.029 | 0.035 | 0.812 |

The surviving share is the median over images of the probability of the predicted class averaged over the masked passes. P_c is close to 1 for most images. It falls below 0.9 on 3% of clean validation images on GTSRB, 7% on CIFAR-10, 21% on CIFAR-100 and 28% on Tiny, against 1% to 6% of triggered images.

Masking leaves a median of 0.035 of the clean probability against 0.812 of the triggered one. The absolute PSU of a clean image is therefore close to its starting confidence, so a clean image the model was unsure of drops little in absolute terms and lands in the low tail that the threshold reads as suspicious. On validation the Spearman correlation with P_c is +0.30 for absolute PSU and -0.11 for fractional PSU.

### Flags at a fixed false positive rate

Thresholds at the 0.01, 0.05 and 0.10 quantiles of clean validation, low means poisoned. Gained and lost are shares of triggered images flagged by only the fractional or only the absolute form. Kept is the share of the absolute form's validation flags the fractional form also raises. The P_c columns are medians over the flagged, dropped or added validation images.

| dataset | q | TPR absolute | TPR fractional | gain | gained | lost | kept | P_c flagged absolute | P_c flagged fractional | P_c dropped | P_c added |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cifar10 | 0.01 | 0.522 | 0.573 | +0.051 | 0.051 | 0.000 | 0.707 | 0.678 | 0.875 | 0.493 | 0.995 |
| cifar10 | 0.05 | 0.645 | 0.660 | +0.014 | 0.015 | 0.001 | 0.785 | 0.909 | 0.999 | 0.650 | 0.999 |
| cifar10 | 0.10 | 0.697 | 0.706 | +0.009 | 0.011 | 0.002 | 0.851 | 0.982 | 0.999 | 0.551 | 0.999 |
| cifar100 | 0.01 | 0.665 | 0.790 | +0.125 | 0.125 | 0.000 | 0.508 | 0.552 | 0.999 | 0.190 | 0.999 |
| cifar100 | 0.05 | 0.806 | 0.935 | +0.130 | 0.131 | 0.001 | 0.308 | 0.428 | 0.999 | 0.372 | 0.999 |
| cifar100 | 0.10 | 0.857 | 0.964 | +0.106 | 0.109 | 0.003 | 0.307 | 0.521 | 0.999 | 0.486 | 0.999 |
| gtsrb | 0.01 | 0.798 | 0.801 | +0.002 | 0.005 | 0.002 | 0.670 | 0.832 | 0.999 | 0.403 | 1.000 |
| gtsrb | 0.05 | 0.839 | 0.839 | +0.000 | 0.003 | 0.003 | 0.775 | 0.966 | 1.000 | 0.459 | 1.000 |
| gtsrb | 0.10 | 0.873 | 0.869 | -0.004 | 0.003 | 0.008 | 0.834 | 0.999 | 1.000 | 0.533 | 1.000 |
| tiny | 0.01 | 0.762 | 0.793 | +0.031 | 0.031 | 0.000 | 0.900 | 0.999 | 1.000 | 0.186 | 0.996 |
| tiny | 0.05 | 0.889 | 0.955 | +0.066 | 0.066 | 0.000 | 0.465 | 0.477 | 1.000 | 0.249 | 1.000 |
| tiny | 0.10 | 0.926 | 0.975 | +0.049 | 0.050 | 0.001 | 0.382 | 0.503 | 1.000 | 0.357 | 0.999 |
| all | 0.01 | 0.687 | 0.735 | +0.049 | 0.049 | 0.001 | 0.703 | 0.773 | 0.966 | 0.328 | 0.998 |
| all | 0.05 | 0.793 | 0.841 | +0.048 | 0.049 | 0.001 | 0.600 | 0.713 | 0.999 | 0.439 | 0.999 |
| all | 0.10 | 0.836 | 0.872 | +0.036 | 0.040 | 0.003 | 0.613 | 0.768 | 0.999 | 0.484 | 0.999 |

At 1% FPR the division replaces 30% of the validation flags. The flags it drops have a median P_c of 0.33, the ones it adds 0.998, so low-confidence clean images leave the suspicious tail and confident clean images whose prediction survives masking take their place. TPR moves from 0.687 to 0.735, 0.049 of triggered images gained against 0.001 lost.

The gain is largest on CIFAR-100 (+0.125 at 1%, +0.130 at 5%, +0.106 at 10% FPR) and about 0 on GTSRB (+0.002, +0.000, -0.004).

The gain does not track the low-confidence share strictly. Tiny has more clean validation images below 0.9 than CIFAR-100 (28% against 21%) and a smaller gain at 1% FPR (+0.031 against +0.125). The data shows where the difference sits. On Tiny the absolute form's 1% flags already have a median P_c of 0.999 and the division keeps 90% of them, against 0.552 and 51% on CIFAR-100, so at this budget most of Tiny's flags are already confident robust images and few low-confidence flags are left for the division to replace.

### Ranking

| dataset | AUROC absolute | AUROC fractional | Spearman absolute against fractional | Spearman absolute with P_c | Spearman fractional with P_c |
|---|---:|---:|---:|---:|---:|
| cifar10 | 0.910 | 0.919 | 0.980 | +0.22 | +0.05 |
| cifar100 | 0.958 | 0.979 | 0.975 | +0.63 | +0.00 |
| gtsrb | 0.941 | 0.935 | 0.990 | -0.22 | -0.43 |
| tiny | 0.969 | 0.977 | 0.990 | +0.66 | -0.02 |
| all | 0.944 | 0.951 | 0.984 | +0.30 | -0.11 |

AUROC barely moves, from 0.944 to 0.951, because the division hardly reorders images. The rank correlation of the 2 scores over validation and triggered images pooled is 0.984. The images it does reorder sit in the low tail, which is where a threshold at a small FPR is set.

### Per-class spread of the clean flags

Per true class of clean validation, the index of dispersion of the flag counts (1 when flags fall on classes in proportion to their size) and the share of all flags held by the worst tenth of classes.

| dataset | images per class | q | dispersion absolute | dispersion fractional | worst tenth absolute | worst tenth fractional |
|---|---:|---:|---:|---:|---:|---:|
| cifar10 | 200 | 0.01 | 4.72 | 8.25 | 0.43 | 0.63 |
| cifar10 | 200 | 0.05 | 39.46 | 57.75 | 0.60 | 0.73 |
| cifar10 | 200 | 0.10 | 88.92 | 112.12 | 0.62 | 0.70 |
| cifar100 | 20 | 0.01 | 1.77 | 2.38 | 0.75 | 0.89 |
| cifar100 | 20 | 0.05 | 1.54 | 4.07 | 0.37 | 0.59 |
| cifar100 | 20 | 0.10 | 1.60 | 5.42 | 0.28 | 0.49 |
| gtsrb | 47 | 0.01 | 7.12 | 7.81 | 0.80 | 0.95 |
| gtsrb | 47 | 0.05 | 13.68 | 17.98 | 0.75 | 0.90 |
| gtsrb | 47 | 0.10 | 18.84 | 22.73 | 0.73 | 0.83 |
| tiny | 10 | 0.01 | 1.55 | 1.62 | 1.00 | 1.00 |
| tiny | 10 | 0.05 | 1.46 | 2.20 | 0.48 | 0.59 |
| tiny | 10 | 0.10 | 1.38 | 2.38 | 0.34 | 0.44 |
| all | 73 | 0.01 | 3.94 | 5.22 | 0.74 | 0.86 |
| all | 73 | 0.05 | 14.93 | 21.71 | 0.56 | 0.71 |
| all | 73 | 0.10 | 29.55 | 37.88 | 0.51 | 0.63 |

The division concentrates the clean false positives in fewer classes. At 1% FPR the worst tenth of classes hold 0.74 of the flags under absolute PSU and 0.86 under fractional PSU, and the dispersion rises from 3.94 to 5.22. So the division gives no calibration across classes. CIFAR-100 and Tiny have 20 and 10 validation images per class, so their per-class numbers are noisy.

### The training-set setting

Li et al. detect poisoned images inside the training set and set the threshold on it. The raw tensors of `experiments/training_set_detection` hold PSBD-TM at the ladder's own rate on these models, read here with thresholds at quantiles of the clean training images.

| model | P_c below 0.9 (validation) | (clean train) | (poisoned) | TPR 1% abs | frac | TPR 5% abs | frac | TPR 10% abs | frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `vit_tiny_wanet_0_1` | 0.290 | 0.070 | 0.009 | 0.294 | 0.294 | 0.679 | 0.760 | 0.836 | 0.934 |
| `vit_tiny_wanet_0_1` | 0.290 | 0.070 | 0.009 | 0.294 | 0.294 | 0.679 | 0.760 | 0.836 | 0.934 |

Clean training images are more confident than validation images. On the 3 Tiny models P_c falls below 0.9 on 30% of validation and 8% of clean training images. With few low-confidence clean images there is little for the division to move, and the fractional TPR is within 0.05 of the absolute one everywhere, except `vit_tiny_wanet_0_1` at 5% (+0.080) and 10% (+0.098). This is consistent with Li et al. not needing the division in their setting.

<!-- division:end -->

## Where confidence alone does explain most of it

`badnet_a2o` on CIFAR-10 at 10% is the exception in the saved record: confidence-only
reaches 0.878 against PSU's 0.889 at `pre_residual`, so for the static patch trigger most
of the separation is available without dropout at all. For `blend` the gap is 0.701
against 0.978. So the value PSBD adds is largest exactly where a naive baseline is weakest,
which is the right way round.

## Subquestions

1. The fractional form's advantage holds under the adaptive rate rule on the current
   panel (above). Whether it holds across SAM checkpoints has not been read.
2. Confidence-only scores far below PSBD-TM on the current panel (above). That is a baseline no PSBD
   paper reports, and any detection method should be shown to beat it.
3. Does combining the 2 (confidence and PSU as 2 features) beat either? That
   would say they carry partly independent information.
