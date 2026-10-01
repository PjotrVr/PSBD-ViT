# Faded trigger defenses

## Question

The trigger dose experiment (`experiments/evidence_surplus/trigger_dose/`) fades the trigger at test time and holds PSBD-TM fixed. On several models a faded trigger still sends most images to the target while PSBD-TM's TPR at low FPR collapses, yet the AUROC of the images the faded trigger still flips stays well above chance. The signal is weak per input but present. This experiment asks whether an inference-time change to how PSBD-TM's scores are read or produced recovers detection of faded triggers at a fixed false positive budget, without fitting anything on triggered data. The predictions were written before any score existed, in `PREDICTIONS.md`. They were extended after the literature memo `docs/faded-trigger-defenses.md`, still before any score existed.

## Ideas behind the fixes

The memo splits the models into 2 regimes, and every result is reported per regime. In the gate regime (`vit_cifar10_badnet_a2o_0_01`, `vit_tiny_badnet_a2o_0_05`, `vit_gtsrb_lf_0_01` and `vit_cifar100_blend_0_1`) the inputs that still fire keep their per-input signal, and detection falls mainly because fewer inputs fire. In the collapse regime (`vit_cifar10_blend_0_1` and `vit_cifar10_bpp_0_05`) fading removes the signal from the inputs that still fire. The regimes were fixed from the dose records before any per-input score was read.

The per-input pass is the base for the other fixes. The dose records store only aggregates per dose. Aggregates cannot be pooled, recombined across rates or compared input by input. `score.py` stores every input's unperturbed prediction and confidence. At every rate of the cached ladder it adds the input's fractional PSU and the share of passes that kept its answer.

Query pooling reads a shift that is small per input and consistent across inputs. An attacker who uses a faded trigger to evade the per-input threshold still has to send many triggered queries, and they all land in the target class. A monitor that holds the last n queries of each predicted class and tests them jointly against clean scores of that class sums the weak per-input evidence. The test is a 1-sided Mann-Whitney test, which asks whether the batch sits below the reference and does not assume a score distribution. The reference is the clean validation scores of the predicted class, or of every class when too few validation images are predicted as it. A mixed-batch variant asks how much of a batch must be triggered before the monitor notices.

Multi-rate scoring reads survival wherever it happens on the ladder. The evidence surplus account says a trigger's answer outlasts a clean answer as the removal rate grows. Fading the trigger lowers the rate at which the triggered answer breaks, and the adaptive rate may then sit past it, while at another rate the faded answer still outlasts clean ones. A score summed over the ladder does not have to guess that rate. 3 are read: the area under the fractional PSU curve, 1 minus the area under the keep curve and the negated majority critical rate.

Residual amplification restores the dose itself. Sharpening is linear, so it multiplies a faded additive trigger's high-frequency part by about `1 + lambda`, while clean content can only be sharpened as far as clean accuracy allows. If the trigger is higher in frequency than the content the model relies on, sharpening gives the trigger back part of the surplus fading took. This fix changes the input the detector sees, so it is exploratory and has a benign control.

Per-class and 2-sided thresholds address where the 1% window sits. On CIFAR-10 and GTSRB the memo found that the lowest clean scores belong almost entirely to 1 non-target class whose images never flip under masking, so the canonical 1% threshold is set by that class. A threshold per predicted class compares a query sent to the target only with clean images predicted as the target. A 2-sided threshold also flags inputs whose answer breaks unusually early, which is where a surplus-poor faded hit lands. Both are conformal p-values against clean validation, so each class keeps its own level by construction.

The matched filter reads firing instead of surplus. If a faded trigger still sends the input to the target it still moves the last block's class token along the backdoor direction, whatever surplus it carries. The direction is estimated without labels from the queries PSBD-TM flags with high confidence in a stream that holds some full-strength triggered queries. This is an extension with its own threat model, since the defender must see such a stream.

## Method

The models are the dose models `vit_cifar10_blend_0_1`, `vit_cifar100_blend_0_1`, `vit_cifar10_badnet_a2o_0_01`, `vit_gtsrb_lf_0_01`, `vit_cifar10_bpp_0_05` and `vit_tiny_badnet_a2o_0_05`. The benign controls are `vit_cifar10_benign` probed with BadNets, Blend and BPP and `vit_gtsrb_benign` probed with LF, the triggers their datasets' backdoored models carry. Each model reads the first `PAIRS` pairs of its PSBD eval ASR split through the dose code's pair loader (`experiments.why_psbd_works.measure.load_pairs`), so its first 256 pairs are the dose experiment's. The trigger is faded at the doses of `score.DOSES` by the dose code's linear fade in pixel space. That fade is written inline in the dose code's `measure()`, so `score.faded_images` restates the formula. Its WaNet branch calls the dose code's `warped()` directly.

The detector is canonical everywhere. PSBD-TM is the token mask at `before_attention_norm`, scored by the fractional PSU with 3 passes under bfloat16 and mask seed 0. The adaptive rate is the smallest rate whose clean validation shift ratio reaches 0.8, read from the canonical sweep. The thresholds are the 0.01, 0.05 and 0.10 quantiles of the cached clean validation scores at that rate, and a score below the threshold is flagged. Clean validation and the clean test split are read from the canonical cache at every rate of the placement's ladder, since the thresholds are read from exactly those tensors. The faded triggered images and their clean twins get fresh passes at every rate of that ladder.

The attack set of a model at a dose is the faded triggered inputs the model sends to the attacked class. On a backdoored model that class is the target, so the attack set is the queries on which the faded backdoor still works. On a benign control it is the probe's target class 0, so the control runs the same rule. TPR is read on the attack set and FPR on the cached clean test split, which the thresholds never saw. Images a trigger sends to the target score lower than their twins even on a benign model, so the attack set's AUROC is also read against clean test images predicted as the attacked class, and beside the benign control.

The pooled monitor draws 2000 batches of n queries without replacement for n of 1, 5, 10, 20 and 50. Its detection rate is the share of attack-set batches whose p-value is at most the alarm level, and its false alarm rate is the same share for batches of clean test queries predicted as the attacked class. The reference is the clean validation scores of the attacked class when at least 30 validation images are predicted as it, and every validation score otherwise. The p-value is exact at n = 1 and the normal approximation with continuity correction above it (`analyze.mann_whitney_less`, held to scipy in `tests/test_faded_trigger_defense.py`). Mixed batches hold 20% or 50% attack-set queries, the rest clean queries of the attacked class. The reference is 1 finite sample shared by every draw, so the false alarm rate at a fixed reference can depart from the level as n grows. That is why it is measured on clean queries and never assumed.

The multi-rate scores integrate over the placement's ladder by the trapezoid rule and are normalized by the ladder's span. The critical rate is `defenses.scores.critical_rate` with a majority of passes, read from the kept shares. The test suite holds the 2 equal. Every score is oriented so low means poisoned, thresholded at the same quantiles of its own clean validation distribution and read on the same attack set and clean test split.

Residual amplification computes `x' = clip(x + lambda (x - blur(x)), 0, 1)` in pixel space at the dataset's own resolution, with a Gaussian blur of kernel 5 and sigma 1 pixel. lambda is the largest of `score.LAMBDAS` whose clean validation accuracy stays within 1 point of the unsharpened accuracy. The sharpened pipeline then applies the canonical rules to sharpened clean validation: the adaptive rate on its ladder and the thresholds at its quantiles. The attack set stays the unsharpened hits, so both pipelines score the same queries. Both realize their FPR on the clean twins.

The threshold rules give each input the conformal p-value `(1 + #{v <= s}) / (n + 1)` against a clean validation reference of n scores. The 1-sided rule flags `p <= q`, and the 2-sided rule flags when the smaller tail p-value is at most `q / 2`. The per-class rules use the validation images predicted as the input's class, and fall back to every validation score when the class is too small for the level to be reachable (`n + 1 < 1 / q`, or `2 / q` for the 2-sided rule). Every rule reports its TPR on the attack set, its realized FPR on the clean test split and its realized FPR on the clean test images predicted as the attacked class. The rule differs from the refuted calibration of `experiments/final_method/class_calibration.py`, which shrank each class toward the pooled distribution and then set 1 pooled threshold.

The matched filter's stream is the even rows of the clean test split and the dose 1.0 triggered images of the first half of the pairs. PSBD-TM flags the stream at its canonical 1% threshold, and the suspected target is the predicted class with the most flagged queries. With at least `MIN_CATCHES` of them, the direction is the mean last-block class token of the flagged queries of that class minus the mean over clean validation images predicted as it. Every input is scored by its negated projection on the unit direction, so low means poisoned. The score is thresholded at quantiles of the clean validation projections. The readout uses only the second half of the pairs at every dose and the odd clean test rows, which the stream never held. It compares with the canonical single rate on exactly those inputs. The class token is the input of the classification head, captured by `score.py --parts features`.

## Leakage audit

Nothing is fitted on triggered data. The adaptive rate and the thresholds come from clean validation through the canonical sweep, and on the sharpened pipeline from sharpened clean validation by the same rules. The pooled reference is clean validation and the false alarm pool is the clean test split. lambda is chosen on clean validation accuracy before any triggered image is sharpened. The per-class and 2-sided rules read only clean validation. `test_no_threshold_or_reference_moves_with_triggered_scores` shifts every triggered score and checks that no threshold, reference, rate or false alarm rate moves. The matched filter is the 1 exception by design: its direction is fitted on the unlabeled stream, which holds full-strength triggered queries, and no label enters the estimate. Its thresholds come from clean validation, its readout never touches the images the stream held, and `test_matched_filter_thresholds_ignore_the_held_out_pairs` checks that the held-out triggered images move none of its thresholds.

## Running

`score.py` is the GPU pass. It writes 1 record per entry and part into the `plain/`, `features/` and `amplified/` folders of `results/_experiments/faded_trigger_defense/`. `analyze.py` is the CPU readout. It writes `readout.json` beside them. With `--readme` it also writes the results section below. `gpu_driver.sh` runs the queue in the login GPU window behind the queues already waiting and ends with the readout.

## Results

<!-- results:begin -->
10 entries read.

Verdicts of PREDICTIONS.md.

| prediction | claim | verdict |
|---|---|---|
| C1 | first 256 pairs match the dose record | held |
| C2 | fresh passes match the canonical cache | held |
| P1 | pooled detection of 20 queries at the weakest firing dose | held |
| P2 | pooled false alarms of 20 clean target queries | failed |
| P3 | pooling blind on the inverted Blend model | failed |
| P4 | mixed batch, 10 of 50 triggered | held |
| P5 | benign pooled detection at the level | failed |
| M1 | PSU-curve area beats the single rate | failed |
| M2 | tied multi-rate scores miss their FPR | held |
| M3 | benign multi-rate TPR equals its FPR | held |
| A1 | sharpening keeps clean accuracy at lambda 0.5 or more | failed |
| A2 | sharpening restores high-frequency triggers only | failed |
| A3 | benign AUROC stays at chance when sharpened | failed |
| R1 | gate regime pools at 10 queries at level 0.01 | held |
| R2 | BPP pools at 20 queries at level 0.01 | held |
| R3 | collapsed Blend stays hidden at 50 queries | failed |
| B1 | gate attack set beats target-class clean images | failed |
| B2 | benign attack set is at chance against class 0 clean | failed |
| K1 | per-class rule gains at 1% on CIFAR-10 and GTSRB | held |
| K2 | per-class rules keep their FPR at 1% | held |
| K3 | 2-sided rule catches collapsed Blend at 0.5 only | failed |
| K4 | per-class rule leaves CIFAR-100 and Tiny alone | failed |
| F1 | the most flagged class is the target | held |
| F2 | matched filter reads firing at every dose | held |
| F3 | matched filter beats the single rate at 1% | held |
| F4 | benign matched filter TPR equals its FPR | held |

Means per regime at each model's weakest firing dose (dose 1.0 where none fires), every TPR at the 0.01 level of its own rule.

| regime | entries | canonical TPR at 0.01 | its FPR | AUROC against clean of the class | pooled alarm, n = 10 at level 0.01 | PSU-curve area TPR | per-class TPR | 2-sided TPR | matched filter TPR |
|---|---|---|---|---|---|---|---|---|---|
| gate | 4 | 0.268 | 0.010 | 0.845 | 0.993 | 0.337 | 0.332 | 0.205 | 0.798 |
| collapse | 2 | 0.157 | 0.009 | 0.734 | 0.556 | 0.098 | 0.413 | 0.123 | 0.809 |
| benign control | 4 | 0.024 | 0.011 | 0.694 | 0.500 | 0.004 | 0.267 | 0.013 | 0.000 |

Single-rate PSBD-TM at the adaptive rate on the attack set, realized FPR on the cached clean test split.

| regime | entry | dose | ASR | attack set | TPR at 0.01 (FPR) | TPR at 0.05 (FPR) | TPR at 0.10 (FPR) | AUROC against twins | AUROC against clean of the class |
|---|---|---|---|---|---|---|---|---|---|
| gate | vit_cifar100_blend_0_1 | 1.0 | 1.000 | 1024 | 1.000 (0.009) | 1.000 (0.041) | 1.000 (0.096) | 0.998 | 1.000 |
| gate | vit_cifar100_blend_0_1 | 0.8 | 1.000 | 1024 | 0.996 (0.009) | 1.000 (0.041) | 1.000 (0.096) | 0.998 | 1.000 |
| gate | vit_cifar100_blend_0_1 | 0.6 | 0.999 | 1023 | 0.885 (0.009) | 0.980 (0.041) | 0.996 (0.096) | 0.993 | 0.997 |
| gate | vit_cifar100_blend_0_1 | 0.5 | 0.988 | 1012 | 0.696 (0.009) | 0.876 (0.041) | 0.969 (0.096) | 0.979 | 0.978 |
| gate | vit_cifar100_blend_0_1 | 0.4 | 0.868 | 889 | 0.404 (0.009) | 0.699 (0.041) | 0.892 (0.096) | 0.954 | 0.934 |
| gate | vit_cifar10_badnet_a2o_0_01 | 1.0 | 0.996 | 1020 | 0.861 (0.011) | 0.932 (0.046) | 0.964 (0.093) | 0.983 | 0.987 |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.8 | 0.991 | 1015 | 0.828 (0.011) | 0.913 (0.046) | 0.955 (0.093) | 0.978 | 0.982 |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.6 | 0.966 | 989 | 0.686 (0.011) | 0.822 (0.046) | 0.887 (0.093) | 0.949 | 0.957 |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.5 | 0.914 | 936 | 0.582 (0.011) | 0.751 (0.046) | 0.824 (0.093) | 0.925 | 0.931 |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.4 | 0.722 | 739 | 0.429 (0.011) | 0.585 (0.046) | 0.683 (0.093) | 0.871 | 0.875 |
| gate | vit_gtsrb_lf_0_01 | 1.0 | 0.985 | 1009 | 0.867 (0.010) | 0.910 (0.056) | 0.941 (0.102) | 0.986 | 0.997 |
| gate | vit_gtsrb_lf_0_01 | 0.8 | 0.937 | 959 | 0.800 (0.010) | 0.864 (0.056) | 0.899 (0.102) | 0.975 | 0.992 |
| gate | vit_gtsrb_lf_0_01 | 0.6 | 0.804 | 823 | 0.678 (0.010) | 0.774 (0.056) | 0.840 (0.102) | 0.964 | 0.993 |
| gate | vit_gtsrb_lf_0_01 | 0.5 | 0.694 | 711 | 0.525 (0.010) | 0.665 (0.056) | 0.743 (0.102) | 0.942 | 0.984 |
| gate | vit_gtsrb_lf_0_01 | 0.4 | 0.520 | 532 | 0.118 (0.010) | 0.280 (0.056) | 0.470 (0.102) | 0.870 | 0.949 |
| gate | vit_tiny_badnet_a2o_0_05 | 1.0 | 1.000 | 1024 | 0.347 (0.011) | 0.994 (0.067) | 1.000 (0.110) | 0.985 | 0.751 |
| gate | vit_tiny_badnet_a2o_0_05 | 0.8 | 0.998 | 1022 | 0.351 (0.011) | 0.993 (0.067) | 1.000 (0.110) | 0.985 | 0.752 |
| gate | vit_tiny_badnet_a2o_0_05 | 0.6 | 0.952 | 975 | 0.305 (0.011) | 0.987 (0.067) | 0.999 (0.110) | 0.983 | 0.737 |
| gate | vit_tiny_badnet_a2o_0_05 | 0.5 | 0.814 | 834 | 0.217 (0.011) | 0.968 (0.067) | 0.995 (0.110) | 0.979 | 0.707 |
| gate | vit_tiny_badnet_a2o_0_05 | 0.4 | 0.518 | 530 | 0.119 (0.011) | 0.855 (0.067) | 0.947 (0.110) | 0.961 | 0.620 |
| collapse | vit_cifar10_blend_0_1 | 1.0 | 1.000 | 1024 | 0.564 (0.008) | 0.699 (0.056) | 0.769 (0.103) | 0.911 | 0.975 |
| collapse | vit_cifar10_blend_0_1 | 0.8 | 0.999 | 1023 | 0.327 (0.008) | 0.438 (0.056) | 0.497 (0.103) | 0.758 | 0.877 |
| collapse | vit_cifar10_blend_0_1 | 0.6 | 0.924 | 946 | 0.122 (0.008) | 0.193 (0.056) | 0.220 (0.103) | 0.519 | 0.661 |
| collapse | vit_cifar10_blend_0_1 | 0.5 | 0.710 | 727 | 0.085 (0.008) | 0.116 (0.056) | 0.140 (0.103) | 0.429 | 0.571 |
| collapse | vit_cifar10_blend_0_1 | 0.4 | 0.360 | 369 | 0.030 (0.008) | 0.054 (0.056) | 0.089 (0.103) | 0.376 | 0.518 |
| collapse | vit_cifar10_bpp_0_05 | 1.0 | 0.982 | 1006 | 0.407 (0.010) | 0.526 (0.046) | 0.605 (0.102) | 0.820 | 0.945 |
| collapse | vit_cifar10_bpp_0_05 | 0.8 | 0.937 | 959 | 0.228 (0.010) | 0.347 (0.046) | 0.415 (0.102) | 0.720 | 0.897 |
| collapse | vit_cifar10_bpp_0_05 | 0.6 | 0.441 | 452 | 0.058 (0.010) | 0.126 (0.046) | 0.153 (0.102) | 0.569 | 0.818 |
| collapse | vit_cifar10_bpp_0_05 | 0.5 | 0.072 | 74 | 0.027 (0.010) | 0.041 (0.046) | 0.054 (0.102) | 0.485 | 0.755 |
| collapse | vit_cifar10_bpp_0_05 | 0.4 | 0.003 | 3 | 0.000 (0.010) | 0.000 (0.046) | 0.000 (0.102) | 0.323 | 0.417 |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 1.0 | 0.017 | 17 | 0.000 (0.011) | 0.000 (0.050) | 0.059 (0.095) | 0.679 | 0.551 |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.8 | 0.017 | 17 | 0.000 (0.011) | 0.000 (0.050) | 0.059 (0.095) | 0.677 | 0.547 |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.6 | 0.016 | 16 | 0.000 (0.011) | 0.000 (0.050) | 0.000 (0.095) | 0.673 | 0.536 |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.5 | 0.015 | 15 | 0.000 (0.011) | 0.000 (0.050) | 0.067 (0.095) | 0.675 | 0.533 |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.4 | 0.014 | 14 | 0.000 (0.011) | 0.000 (0.050) | 0.000 (0.095) | 0.668 | 0.544 |
| benign control | vit_cifar10_benign__probe_blend | 1.0 | 0.144 | 147 | 0.034 (0.011) | 0.218 (0.050) | 0.354 (0.095) | 0.822 | 0.857 |
| benign control | vit_cifar10_benign__probe_blend | 0.8 | 0.063 | 65 | 0.031 (0.011) | 0.123 (0.050) | 0.262 (0.095) | 0.804 | 0.820 |
| benign control | vit_cifar10_benign__probe_blend | 0.6 | 0.030 | 31 | 0.032 (0.011) | 0.065 (0.050) | 0.161 (0.095) | 0.725 | 0.658 |
| benign control | vit_cifar10_benign__probe_blend | 0.5 | 0.020 | 20 | 0.050 (0.011) | 0.050 (0.050) | 0.050 (0.095) | 0.698 | 0.617 |
| benign control | vit_cifar10_benign__probe_blend | 0.4 | 0.010 | 10 | 0.000 (0.011) | 0.000 (0.050) | 0.000 (0.095) | 0.643 | 0.376 |
| benign control | vit_cifar10_benign__probe_bpp | 1.0 | 0.064 | 66 | 0.061 (0.011) | 0.167 (0.050) | 0.197 (0.095) | 0.744 | 0.700 |
| benign control | vit_cifar10_benign__probe_bpp | 0.8 | 0.035 | 36 | 0.000 (0.011) | 0.028 (0.050) | 0.056 (0.095) | 0.714 | 0.691 |
| benign control | vit_cifar10_benign__probe_bpp | 0.6 | 0.021 | 22 | 0.000 (0.011) | 0.000 (0.050) | 0.091 (0.095) | 0.742 | 0.741 |
| benign control | vit_cifar10_benign__probe_bpp | 0.5 | 0.017 | 17 | 0.000 (0.011) | 0.118 (0.050) | 0.118 (0.095) | 0.745 | 0.732 |
| benign control | vit_cifar10_benign__probe_bpp | 0.4 | 0.012 | 12 | 0.000 (0.011) | 0.000 (0.050) | 0.000 (0.095) | 0.676 | 0.622 |
| benign control | vit_gtsrb_benign__probe_lf | 1.0 | 0.001 | 1 | 0.000 (0.013) | 0.000 (0.052) | 0.000 (0.106) | 0.206 | 0.667 |
| benign control | vit_gtsrb_benign__probe_lf | 0.8 | 0.002 | 2 | 0.000 (0.013) | 0.000 (0.052) | 0.000 (0.106) | 0.151 | 0.417 |
| benign control | vit_gtsrb_benign__probe_lf | 0.6 | 0.002 | 2 | 0.000 (0.013) | 0.000 (0.052) | 0.000 (0.106) | 0.137 | 0.352 |
| benign control | vit_gtsrb_benign__probe_lf | 0.5 | 0.002 | 2 | 0.000 (0.013) | 0.000 (0.052) | 0.000 (0.106) | 0.139 | 0.361 |
| benign control | vit_gtsrb_benign__probe_lf | 0.4 | 0.002 | 2 | 0.000 (0.013) | 0.000 (0.052) | 0.000 (0.106) | 0.142 | 0.361 |

Pooled monitor at level 0.05: share of batches of n attack-set queries that alarm, with the share of batches of n clean test queries of the attacked class that alarm in brackets.

| regime | entry | dose | reference | attack / clean pool | n = 1 | n = 5 | n = 10 | n = 20 | n = 50 |
|---|---|---|---|---|---|---|---|---|---|
| gate | vit_cifar100_blend_0_1 | 1.0 | all 2000 | 1024 / 105 | 1.000 (0.032) | 1.000 (0.972) | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| gate | vit_cifar100_blend_0_1 | 0.8 | all 2000 | 1024 / 105 | 1.000 (0.032) | 1.000 (0.972) | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| gate | vit_cifar100_blend_0_1 | 0.6 | all 2000 | 1023 / 105 | 0.980 (0.032) | 1.000 (0.972) | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| gate | vit_cifar100_blend_0_1 | 0.5 | all 2000 | 1012 / 105 | 0.868 (0.032) | 1.000 (0.972) | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| gate | vit_cifar100_blend_0_1 | 0.4 | all 2000 | 889 / 105 | 0.688 (0.032) | 1.000 (0.972) | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| gate | vit_cifar10_badnet_a2o_0_01 | 1.0 | class 210 | 1020 / 830 | 0.972 (0.075) | 0.999 (0.129) | 1.000 (0.153) | 1.000 (0.193) | 1.000 (0.331) |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.8 | class 210 | 1015 / 830 | 0.972 (0.075) | 0.998 (0.129) | 1.000 (0.153) | 1.000 (0.193) | 1.000 (0.331) |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.6 | class 210 | 989 / 830 | 0.896 (0.075) | 0.994 (0.129) | 1.000 (0.153) | 1.000 (0.193) | 1.000 (0.331) |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.5 | class 210 | 936 / 830 | 0.887 (0.075) | 0.978 (0.129) | 1.000 (0.153) | 1.000 (0.193) | 1.000 (0.331) |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.4 | class 210 | 739 / 830 | 0.756 (0.075) | 0.909 (0.129) | 0.992 (0.153) | 1.000 (0.193) | 1.000 (0.331) |
| gate | vit_gtsrb_lf_0_01 | 1.0 | all 2000 | 1009 / 52 | 0.905 (0.000) | 1.000 (0.004) | 1.000 (0.005) | 1.000 (0.005) | 1.000 (0.000) |
| gate | vit_gtsrb_lf_0_01 | 0.8 | all 2000 | 959 / 52 | 0.875 (0.000) | 1.000 (0.004) | 1.000 (0.005) | 1.000 (0.005) | 1.000 (0.000) |
| gate | vit_gtsrb_lf_0_01 | 0.6 | all 2000 | 823 / 52 | 0.777 (0.000) | 1.000 (0.004) | 1.000 (0.005) | 1.000 (0.005) | 1.000 (0.000) |
| gate | vit_gtsrb_lf_0_01 | 0.5 | all 2000 | 711 / 52 | 0.678 (0.000) | 1.000 (0.004) | 1.000 (0.005) | 1.000 (0.005) | 1.000 (0.000) |
| gate | vit_gtsrb_lf_0_01 | 0.4 | all 2000 | 532 / 52 | 0.275 (0.000) | 0.993 (0.004) | 1.000 (0.005) | 1.000 (0.005) | 1.000 (0.000) |
| gate | vit_tiny_badnet_a2o_0_05 | 1.0 | all 2000 | 1024 / 51 | 0.995 (0.562) | 1.000 (0.998) | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| gate | vit_tiny_badnet_a2o_0_05 | 0.8 | all 2000 | 1022 / 51 | 0.990 (0.562) | 1.000 (0.998) | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| gate | vit_tiny_badnet_a2o_0_05 | 0.6 | all 2000 | 975 / 51 | 0.985 (0.562) | 1.000 (0.998) | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| gate | vit_tiny_badnet_a2o_0_05 | 0.5 | all 2000 | 834 / 51 | 0.969 (0.562) | 1.000 (0.998) | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| gate | vit_tiny_badnet_a2o_0_05 | 0.4 | all 2000 | 530 / 51 | 0.850 (0.562) | 1.000 (0.998) | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| collapse | vit_cifar10_blend_0_1 | 1.0 | class 206 | 1024 / 806 | 0.924 (0.061) | 0.998 (0.040) | 1.000 (0.037) | 1.000 (0.036) | 1.000 (0.038) |
| collapse | vit_cifar10_blend_0_1 | 0.8 | class 206 | 1023 / 806 | 0.735 (0.061) | 0.916 (0.040) | 0.994 (0.037) | 1.000 (0.036) | 1.000 (0.038) |
| collapse | vit_cifar10_blend_0_1 | 0.6 | class 206 | 946 / 806 | 0.427 (0.061) | 0.374 (0.040) | 0.567 (0.037) | 0.779 (0.036) | 0.975 (0.038) |
| collapse | vit_cifar10_blend_0_1 | 0.5 | class 206 | 727 / 806 | 0.299 (0.061) | 0.214 (0.040) | 0.273 (0.037) | 0.352 (0.036) | 0.549 (0.038) |
| collapse | vit_cifar10_blend_0_1 | 0.4 | class 206 | 369 / 806 | 0.258 (0.061) | 0.127 (0.040) | 0.131 (0.037) | 0.156 (0.036) | 0.174 (0.038) |
| collapse | vit_cifar10_bpp_0_05 | 1.0 | class 195 | 1006 / 774 | 0.859 (0.051) | 0.993 (0.072) | 1.000 (0.073) | 1.000 (0.076) | 1.000 (0.079) |
| collapse | vit_cifar10_bpp_0_05 | 0.8 | class 195 | 959 / 774 | 0.744 (0.051) | 0.951 (0.072) | 0.999 (0.073) | 1.000 (0.076) | 1.000 (0.079) |
| collapse | vit_cifar10_bpp_0_05 | 0.6 | class 195 | 452 / 774 | 0.507 (0.051) | 0.864 (0.072) | 0.985 (0.073) | 1.000 (0.076) | 1.000 (0.079) |
| collapse | vit_cifar10_bpp_0_05 | 0.5 | class 195 | 74 / 774 | 0.363 (0.051) | 0.692 (0.072) | 0.931 (0.073) | 0.998 (0.076) | 1.000 (0.079) |
| collapse | vit_cifar10_bpp_0_05 | 0.4 | class 195 | 3 / 774 | 0.000 (0.051) | n/a (0.072) | n/a (0.073) | n/a (0.076) | n/a (0.079) |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 1.0 | class 205 | 17 / 830 | 0.290 (0.047) | 0.138 (0.062) | 0.124 (0.078) | n/a (0.087) | n/a (0.107) |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.8 | class 205 | 17 / 830 | 0.235 (0.047) | 0.130 (0.062) | 0.108 (0.078) | n/a (0.087) | n/a (0.107) |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.6 | class 205 | 16 / 830 | 0.181 (0.047) | 0.113 (0.062) | 0.072 (0.078) | n/a (0.087) | n/a (0.107) |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.5 | class 205 | 15 / 830 | 0.228 (0.047) | 0.102 (0.062) | 0.042 (0.078) | n/a (0.087) | n/a (0.107) |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.4 | class 205 | 14 / 830 | 0.143 (0.047) | 0.105 (0.062) | 0.034 (0.078) | n/a (0.087) | n/a (0.107) |
| benign control | vit_cifar10_benign__probe_blend | 1.0 | class 205 | 147 / 830 | 0.696 (0.047) | 0.909 (0.062) | 0.993 (0.078) | 1.000 (0.087) | 1.000 (0.107) |
| benign control | vit_cifar10_benign__probe_blend | 0.8 | class 205 | 65 / 830 | 0.707 (0.047) | 0.807 (0.062) | 0.974 (0.078) | 1.000 (0.087) | 1.000 (0.107) |
| benign control | vit_cifar10_benign__probe_blend | 0.6 | class 205 | 31 / 830 | 0.379 (0.047) | 0.387 (0.062) | 0.602 (0.078) | 0.924 (0.087) | n/a (0.107) |
| benign control | vit_cifar10_benign__probe_blend | 0.5 | class 205 | 20 / 830 | 0.254 (0.047) | 0.305 (0.062) | 0.431 (0.078) | 1.000 (0.087) | n/a (0.107) |
| benign control | vit_cifar10_benign__probe_blend | 0.4 | class 205 | 10 / 830 | 0.099 (0.047) | 0.000 (0.062) | 0.000 (0.078) | n/a (0.087) | n/a (0.107) |
| benign control | vit_cifar10_benign__probe_bpp | 1.0 | class 205 | 66 / 830 | 0.394 (0.047) | 0.540 (0.062) | 0.747 (0.078) | 0.961 (0.087) | 1.000 (0.107) |
| benign control | vit_cifar10_benign__probe_bpp | 0.8 | class 205 | 36 / 830 | 0.391 (0.047) | 0.511 (0.062) | 0.739 (0.078) | 0.974 (0.087) | n/a (0.107) |
| benign control | vit_cifar10_benign__probe_bpp | 0.6 | class 205 | 22 / 830 | 0.591 (0.047) | 0.585 (0.062) | 0.877 (0.078) | 1.000 (0.087) | n/a (0.107) |
| benign control | vit_cifar10_benign__probe_bpp | 0.5 | class 205 | 17 / 830 | 0.588 (0.047) | 0.529 (0.062) | 0.881 (0.078) | n/a (0.087) | n/a (0.107) |
| benign control | vit_cifar10_benign__probe_bpp | 0.4 | class 205 | 12 / 830 | 0.232 (0.047) | 0.292 (0.062) | 0.458 (0.078) | n/a (0.087) | n/a (0.107) |
| benign control | vit_gtsrb_benign__probe_lf | 1.0 | all 2000 | 1 / 54 | 0.000 (0.000) | n/a (0.000) | n/a (0.000) | n/a (0.000) | n/a (0.000) |
| benign control | vit_gtsrb_benign__probe_lf | 0.8 | all 2000 | 2 / 54 | 0.000 (0.000) | n/a (0.000) | n/a (0.000) | n/a (0.000) | n/a (0.000) |
| benign control | vit_gtsrb_benign__probe_lf | 0.6 | all 2000 | 2 / 54 | 0.000 (0.000) | n/a (0.000) | n/a (0.000) | n/a (0.000) | n/a (0.000) |
| benign control | vit_gtsrb_benign__probe_lf | 0.5 | all 2000 | 2 / 54 | 0.000 (0.000) | n/a (0.000) | n/a (0.000) | n/a (0.000) | n/a (0.000) |
| benign control | vit_gtsrb_benign__probe_lf | 0.4 | all 2000 | 2 / 54 | 0.000 (0.000) | n/a (0.000) | n/a (0.000) | n/a (0.000) | n/a (0.000) |

Pooled monitor at n = 20 across alarm levels, detection with the false alarm rate in brackets.

| regime | entry | dose | level 0.01 | level 0.05 | level 0.1 |
|---|---|---|---|---|---|
| gate | vit_cifar100_blend_0_1 | 1.0 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| gate | vit_cifar100_blend_0_1 | 0.8 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| gate | vit_cifar100_blend_0_1 | 0.6 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| gate | vit_cifar100_blend_0_1 | 0.5 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| gate | vit_cifar100_blend_0_1 | 0.4 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| gate | vit_cifar10_badnet_a2o_0_01 | 1.0 | 1.000 (0.061) | 1.000 (0.193) | 1.000 (0.315) |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.8 | 1.000 (0.061) | 1.000 (0.193) | 1.000 (0.315) |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.6 | 1.000 (0.061) | 1.000 (0.193) | 1.000 (0.315) |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.5 | 1.000 (0.061) | 1.000 (0.193) | 1.000 (0.315) |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.4 | 1.000 (0.061) | 1.000 (0.193) | 1.000 (0.315) |
| gate | vit_gtsrb_lf_0_01 | 1.0 | 1.000 (0.000) | 1.000 (0.005) | 1.000 (0.055) |
| gate | vit_gtsrb_lf_0_01 | 0.8 | 1.000 (0.000) | 1.000 (0.005) | 1.000 (0.055) |
| gate | vit_gtsrb_lf_0_01 | 0.6 | 1.000 (0.000) | 1.000 (0.005) | 1.000 (0.055) |
| gate | vit_gtsrb_lf_0_01 | 0.5 | 1.000 (0.000) | 1.000 (0.005) | 1.000 (0.055) |
| gate | vit_gtsrb_lf_0_01 | 0.4 | 1.000 (0.000) | 1.000 (0.005) | 1.000 (0.055) |
| gate | vit_tiny_badnet_a2o_0_05 | 1.0 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| gate | vit_tiny_badnet_a2o_0_05 | 0.8 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| gate | vit_tiny_badnet_a2o_0_05 | 0.6 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| gate | vit_tiny_badnet_a2o_0_05 | 0.5 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| gate | vit_tiny_badnet_a2o_0_05 | 0.4 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| collapse | vit_cifar10_blend_0_1 | 1.0 | 1.000 (0.007) | 1.000 (0.036) | 1.000 (0.086) |
| collapse | vit_cifar10_blend_0_1 | 0.8 | 1.000 (0.007) | 1.000 (0.036) | 1.000 (0.086) |
| collapse | vit_cifar10_blend_0_1 | 0.6 | 0.581 (0.007) | 0.779 (0.036) | 0.852 (0.086) |
| collapse | vit_cifar10_blend_0_1 | 0.5 | 0.166 (0.007) | 0.352 (0.036) | 0.484 (0.086) |
| collapse | vit_cifar10_blend_0_1 | 0.4 | 0.054 (0.007) | 0.156 (0.036) | 0.248 (0.086) |
| collapse | vit_cifar10_bpp_0_05 | 1.0 | 1.000 (0.012) | 1.000 (0.076) | 1.000 (0.147) |
| collapse | vit_cifar10_bpp_0_05 | 0.8 | 1.000 (0.012) | 1.000 (0.076) | 1.000 (0.147) |
| collapse | vit_cifar10_bpp_0_05 | 0.6 | 1.000 (0.012) | 1.000 (0.076) | 1.000 (0.147) |
| collapse | vit_cifar10_bpp_0_05 | 0.5 | 0.983 (0.012) | 0.998 (0.076) | 1.000 (0.147) |
| collapse | vit_cifar10_bpp_0_05 | 0.4 | n/a (0.012) | n/a (0.076) | n/a (0.147) |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 1.0 | n/a (0.016) | n/a (0.087) | n/a (0.175) |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.8 | n/a (0.016) | n/a (0.087) | n/a (0.175) |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.6 | n/a (0.016) | n/a (0.087) | n/a (0.175) |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.5 | n/a (0.016) | n/a (0.087) | n/a (0.175) |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.4 | n/a (0.016) | n/a (0.087) | n/a (0.175) |
| benign control | vit_cifar10_benign__probe_blend | 1.0 | 1.000 (0.016) | 1.000 (0.087) | 1.000 (0.175) |
| benign control | vit_cifar10_benign__probe_blend | 0.8 | 0.998 (0.016) | 1.000 (0.087) | 1.000 (0.175) |
| benign control | vit_cifar10_benign__probe_blend | 0.6 | 0.659 (0.016) | 0.924 (0.087) | 0.976 (0.175) |
| benign control | vit_cifar10_benign__probe_blend | 0.5 | 0.000 (0.016) | 1.000 (0.087) | 1.000 (0.175) |
| benign control | vit_cifar10_benign__probe_blend | 0.4 | n/a (0.016) | n/a (0.087) | n/a (0.175) |
| benign control | vit_cifar10_benign__probe_bpp | 1.0 | 0.843 (0.016) | 0.961 (0.087) | 0.985 (0.175) |
| benign control | vit_cifar10_benign__probe_bpp | 0.8 | 0.841 (0.016) | 0.974 (0.087) | 0.992 (0.175) |
| benign control | vit_cifar10_benign__probe_bpp | 0.6 | 1.000 (0.016) | 1.000 (0.087) | 1.000 (0.175) |
| benign control | vit_cifar10_benign__probe_bpp | 0.5 | n/a (0.016) | n/a (0.087) | n/a (0.175) |
| benign control | vit_cifar10_benign__probe_bpp | 0.4 | n/a (0.016) | n/a (0.087) | n/a (0.175) |
| benign control | vit_gtsrb_benign__probe_lf | 1.0 | n/a (0.000) | n/a (0.000) | n/a (0.000) |
| benign control | vit_gtsrb_benign__probe_lf | 0.8 | n/a (0.000) | n/a (0.000) | n/a (0.000) |
| benign control | vit_gtsrb_benign__probe_lf | 0.6 | n/a (0.000) | n/a (0.000) | n/a (0.000) |
| benign control | vit_gtsrb_benign__probe_lf | 0.5 | n/a (0.000) | n/a (0.000) | n/a (0.000) |
| benign control | vit_gtsrb_benign__probe_lf | 0.4 | n/a (0.000) | n/a (0.000) | n/a (0.000) |

Mixed batches at level 0.05: a fraction of each batch is attack-set queries and the rest clean test queries of the attacked class. The false alarm rate of an all-clean batch is the bracketed column of the pooled table.

| regime | entry | dose | 0.2 of n = 5 | 0.2 of n = 10 | 0.2 of n = 20 | 0.2 of n = 50 | 0.5 of n = 5 | 0.5 of n = 10 | 0.5 of n = 20 | 0.5 of n = 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| gate | vit_cifar100_blend_0_1 | 1.0 | 0.996 | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 |
| gate | vit_cifar100_blend_0_1 | 0.8 | 0.996 | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 |
| gate | vit_cifar100_blend_0_1 | 0.6 | 0.997 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| gate | vit_cifar100_blend_0_1 | 0.5 | 0.996 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| gate | vit_cifar100_blend_0_1 | 0.4 | 0.997 | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 |
| gate | vit_cifar10_badnet_a2o_0_01 | 1.0 | 0.282 | 0.438 | 0.683 | 0.963 | 0.564 | 0.946 | 0.999 | 1.000 |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.8 | 0.290 | 0.463 | 0.715 | 0.967 | 0.579 | 0.956 | 0.999 | 1.000 |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.6 | 0.266 | 0.418 | 0.666 | 0.944 | 0.506 | 0.910 | 0.996 | 1.000 |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.5 | 0.271 | 0.407 | 0.632 | 0.922 | 0.508 | 0.886 | 0.992 | 1.000 |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.4 | 0.239 | 0.383 | 0.579 | 0.886 | 0.437 | 0.777 | 0.961 | 1.000 |
| gate | vit_gtsrb_lf_0_01 | 1.0 | 0.091 | 0.297 | 0.785 | 1.000 | 0.534 | 0.999 | 1.000 | 1.000 |
| gate | vit_gtsrb_lf_0_01 | 0.8 | 0.082 | 0.268 | 0.762 | 1.000 | 0.518 | 0.996 | 1.000 | 1.000 |
| gate | vit_gtsrb_lf_0_01 | 0.6 | 0.082 | 0.267 | 0.743 | 1.000 | 0.475 | 0.997 | 1.000 | 1.000 |
| gate | vit_gtsrb_lf_0_01 | 0.5 | 0.072 | 0.235 | 0.672 | 1.000 | 0.414 | 0.987 | 1.000 | 1.000 |
| gate | vit_gtsrb_lf_0_01 | 0.4 | 0.054 | 0.147 | 0.471 | 0.999 | 0.237 | 0.901 | 1.000 | 1.000 |
| gate | vit_tiny_badnet_a2o_0_05 | 1.0 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| gate | vit_tiny_badnet_a2o_0_05 | 0.8 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| gate | vit_tiny_badnet_a2o_0_05 | 0.6 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| gate | vit_tiny_badnet_a2o_0_05 | 0.5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| gate | vit_tiny_badnet_a2o_0_05 | 0.4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| collapse | vit_cifar10_blend_0_1 | 1.0 | 0.154 | 0.259 | 0.437 | 0.788 | 0.428 | 0.908 | 0.997 | 1.000 |
| collapse | vit_cifar10_blend_0_1 | 0.8 | 0.134 | 0.201 | 0.310 | 0.577 | 0.333 | 0.714 | 0.923 | 0.999 |
| collapse | vit_cifar10_blend_0_1 | 0.6 | 0.078 | 0.110 | 0.135 | 0.182 | 0.146 | 0.251 | 0.373 | 0.615 |
| collapse | vit_cifar10_blend_0_1 | 0.5 | 0.067 | 0.076 | 0.094 | 0.108 | 0.109 | 0.153 | 0.185 | 0.252 |
| collapse | vit_cifar10_blend_0_1 | 0.4 | 0.047 | 0.059 | 0.063 | 0.069 | 0.067 | 0.086 | 0.097 | 0.109 |
| collapse | vit_cifar10_bpp_0_05 | 1.0 | 0.186 | 0.291 | 0.464 | 0.803 | 0.419 | 0.873 | 0.990 | 1.000 |
| collapse | vit_cifar10_bpp_0_05 | 0.8 | 0.174 | 0.257 | 0.428 | 0.723 | 0.364 | 0.775 | 0.963 | 1.000 |
| collapse | vit_cifar10_bpp_0_05 | 0.6 | 0.139 | 0.207 | 0.340 | 0.580 | 0.285 | 0.618 | 0.861 | 0.997 |
| collapse | vit_cifar10_bpp_0_05 | 0.5 | 0.122 | 0.182 | 0.285 | 0.458 | 0.222 | 0.463 | 0.725 | 0.976 |
| collapse | vit_cifar10_bpp_0_05 | 0.4 | 0.043 | 0.037 | n/a | n/a | 0.029 | n/a | n/a | n/a |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 1.0 | 0.093 | 0.101 | 0.116 | 0.154 | 0.116 | 0.125 | 0.134 | n/a |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.8 | 0.090 | 0.097 | 0.116 | 0.146 | 0.108 | 0.119 | 0.123 | n/a |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.6 | 0.081 | 0.099 | 0.105 | 0.122 | 0.084 | 0.107 | 0.108 | n/a |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.5 | 0.087 | 0.096 | 0.115 | 0.116 | 0.093 | 0.107 | 0.094 | n/a |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.4 | 0.100 | 0.109 | 0.118 | 0.128 | 0.100 | 0.120 | 0.104 | n/a |
| benign control | vit_cifar10_benign__probe_blend | 1.0 | 0.164 | 0.261 | 0.388 | 0.709 | 0.350 | 0.716 | 0.928 | 1.000 |
| benign control | vit_cifar10_benign__probe_blend | 0.8 | 0.178 | 0.244 | 0.370 | 0.650 | 0.345 | 0.633 | 0.865 | 0.998 |
| benign control | vit_cifar10_benign__probe_blend | 0.6 | 0.122 | 0.143 | 0.207 | 0.312 | 0.175 | 0.296 | 0.440 | 0.812 |
| benign control | vit_cifar10_benign__probe_blend | 0.5 | 0.109 | 0.123 | 0.163 | 0.268 | 0.135 | 0.210 | 0.322 | n/a |
| benign control | vit_cifar10_benign__probe_blend | 0.4 | 0.041 | 0.042 | 0.036 | 0.022 | 0.025 | 0.005 | 0.001 | n/a |
| benign control | vit_cifar10_benign__probe_bpp | 1.0 | 0.123 | 0.168 | 0.234 | 0.391 | 0.214 | 0.385 | 0.562 | 0.899 |
| benign control | vit_cifar10_benign__probe_bpp | 0.8 | 0.137 | 0.177 | 0.256 | 0.414 | 0.221 | 0.389 | 0.565 | 0.897 |
| benign control | vit_cifar10_benign__probe_bpp | 0.6 | 0.146 | 0.199 | 0.278 | 0.470 | 0.243 | 0.454 | 0.688 | n/a |
| benign control | vit_cifar10_benign__probe_bpp | 0.5 | 0.145 | 0.184 | 0.272 | 0.444 | 0.245 | 0.424 | 0.650 | n/a |
| benign control | vit_cifar10_benign__probe_bpp | 0.4 | 0.100 | 0.131 | 0.180 | 0.242 | 0.136 | 0.223 | 0.299 | n/a |
| benign control | vit_gtsrb_benign__probe_lf | 1.0 | 0.000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| benign control | vit_gtsrb_benign__probe_lf | 0.8 | 0.000 | 0.000 | n/a | n/a | 0.000 | n/a | n/a | n/a |
| benign control | vit_gtsrb_benign__probe_lf | 0.6 | 0.000 | 0.000 | n/a | n/a | 0.000 | n/a | n/a | n/a |
| benign control | vit_gtsrb_benign__probe_lf | 0.5 | 0.000 | 0.000 | n/a | n/a | 0.000 | n/a | n/a | n/a |
| benign control | vit_gtsrb_benign__probe_lf | 0.4 | 0.000 | 0.000 | n/a | n/a | 0.000 | n/a | n/a | n/a |

TPR at the 0.05 quantile of clean validation for the single rate and the 3 ladder scores, realized FPR on the clean test split in brackets. The 0.01 and 0.10 quantiles are in readout.json.

| regime | entry | dose | single rate | psu_area | shift_area | critical_rate |
|---|---|---|---|---|---|---|
| gate | vit_cifar100_blend_0_1 | 1.0 | 1.000 (0.041) | 1.000 (0.047) | 1.000 (0.048) | 1.000 (0.034) |
| gate | vit_cifar100_blend_0_1 | 0.8 | 1.000 (0.041) | 1.000 (0.047) | 1.000 (0.048) | 0.989 (0.034) |
| gate | vit_cifar100_blend_0_1 | 0.6 | 0.980 (0.041) | 0.980 (0.047) | 0.984 (0.048) | 0.882 (0.034) |
| gate | vit_cifar100_blend_0_1 | 0.5 | 0.876 (0.041) | 0.862 (0.047) | 0.909 (0.048) | 0.744 (0.034) |
| gate | vit_cifar100_blend_0_1 | 0.4 | 0.699 (0.041) | 0.679 (0.047) | 0.783 (0.048) | 0.576 (0.034) |
| gate | vit_cifar10_badnet_a2o_0_01 | 1.0 | 0.932 (0.046) | 0.982 (0.058) | 0.000 (0.000) | 0.000 (0.000) |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.8 | 0.913 (0.046) | 0.975 (0.058) | 0.000 (0.000) | 0.000 (0.000) |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.6 | 0.822 (0.046) | 0.916 (0.058) | 0.000 (0.000) | 0.000 (0.000) |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.5 | 0.751 (0.046) | 0.845 (0.058) | 0.000 (0.000) | 0.000 (0.000) |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.4 | 0.585 (0.046) | 0.691 (0.058) | 0.000 (0.000) | 0.000 (0.000) |
| gate | vit_gtsrb_lf_0_01 | 1.0 | 0.910 (0.056) | 0.870 (0.055) | 0.000 (0.000) | 0.000 (0.000) |
| gate | vit_gtsrb_lf_0_01 | 0.8 | 0.864 (0.056) | 0.803 (0.055) | 0.000 (0.000) | 0.000 (0.000) |
| gate | vit_gtsrb_lf_0_01 | 0.6 | 0.774 (0.056) | 0.696 (0.055) | 0.000 (0.000) | 0.000 (0.000) |
| gate | vit_gtsrb_lf_0_01 | 0.5 | 0.665 (0.056) | 0.536 (0.055) | 0.000 (0.000) | 0.000 (0.000) |
| gate | vit_gtsrb_lf_0_01 | 0.4 | 0.280 (0.056) | 0.152 (0.055) | 0.000 (0.000) | 0.000 (0.000) |
| gate | vit_tiny_badnet_a2o_0_05 | 1.0 | 0.994 (0.067) | 0.960 (0.068) | 0.962 (0.040) | 0.932 (0.037) |
| gate | vit_tiny_badnet_a2o_0_05 | 0.8 | 0.993 (0.067) | 0.955 (0.068) | 0.958 (0.040) | 0.927 (0.037) |
| gate | vit_tiny_badnet_a2o_0_05 | 0.6 | 0.987 (0.067) | 0.933 (0.068) | 0.945 (0.040) | 0.903 (0.037) |
| gate | vit_tiny_badnet_a2o_0_05 | 0.5 | 0.968 (0.067) | 0.878 (0.068) | 0.902 (0.040) | 0.847 (0.037) |
| gate | vit_tiny_badnet_a2o_0_05 | 0.4 | 0.855 (0.067) | 0.655 (0.068) | 0.791 (0.040) | 0.721 (0.037) |
| collapse | vit_cifar10_blend_0_1 | 1.0 | 0.699 (0.056) | 0.684 (0.056) | 0.000 (0.000) | 0.000 (0.000) |
| collapse | vit_cifar10_blend_0_1 | 0.8 | 0.438 (0.056) | 0.427 (0.056) | 0.000 (0.000) | 0.000 (0.000) |
| collapse | vit_cifar10_blend_0_1 | 0.6 | 0.193 (0.056) | 0.191 (0.056) | 0.000 (0.000) | 0.000 (0.000) |
| collapse | vit_cifar10_blend_0_1 | 0.5 | 0.116 (0.056) | 0.121 (0.056) | 0.000 (0.000) | 0.000 (0.000) |
| collapse | vit_cifar10_blend_0_1 | 0.4 | 0.054 (0.056) | 0.073 (0.056) | 0.000 (0.000) | 0.000 (0.000) |
| collapse | vit_cifar10_bpp_0_05 | 1.0 | 0.526 (0.046) | 0.664 (0.052) | 0.002 (0.051) | 0.000 (0.000) |
| collapse | vit_cifar10_bpp_0_05 | 0.8 | 0.347 (0.046) | 0.479 (0.052) | 0.001 (0.051) | 0.000 (0.000) |
| collapse | vit_cifar10_bpp_0_05 | 0.6 | 0.126 (0.046) | 0.237 (0.052) | 0.000 (0.051) | 0.000 (0.000) |
| collapse | vit_cifar10_bpp_0_05 | 0.5 | 0.041 (0.046) | 0.216 (0.052) | 0.000 (0.051) | 0.000 (0.000) |
| collapse | vit_cifar10_bpp_0_05 | 0.4 | 0.000 (0.046) | 0.000 (0.052) | 0.000 (0.051) | 0.000 (0.000) |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 1.0 | 0.000 (0.050) | 0.000 (0.049) | 0.000 (0.000) | 0.000 (0.000) |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.8 | 0.000 (0.050) | 0.000 (0.049) | 0.000 (0.000) | 0.000 (0.000) |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.6 | 0.000 (0.050) | 0.000 (0.049) | 0.000 (0.000) | 0.000 (0.000) |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.5 | 0.000 (0.050) | 0.000 (0.049) | 0.000 (0.000) | 0.000 (0.000) |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.4 | 0.000 (0.050) | 0.000 (0.049) | 0.000 (0.000) | 0.000 (0.000) |
| benign control | vit_cifar10_benign__probe_blend | 1.0 | 0.218 (0.050) | 0.007 (0.049) | 0.000 (0.000) | 0.000 (0.000) |
| benign control | vit_cifar10_benign__probe_blend | 0.8 | 0.123 (0.050) | 0.000 (0.049) | 0.000 (0.000) | 0.000 (0.000) |
| benign control | vit_cifar10_benign__probe_blend | 0.6 | 0.065 (0.050) | 0.000 (0.049) | 0.000 (0.000) | 0.000 (0.000) |
| benign control | vit_cifar10_benign__probe_blend | 0.5 | 0.050 (0.050) | 0.000 (0.049) | 0.000 (0.000) | 0.000 (0.000) |
| benign control | vit_cifar10_benign__probe_blend | 0.4 | 0.000 (0.050) | 0.100 (0.049) | 0.000 (0.000) | 0.000 (0.000) |
| benign control | vit_cifar10_benign__probe_bpp | 1.0 | 0.167 (0.050) | 0.015 (0.049) | 0.000 (0.000) | 0.000 (0.000) |
| benign control | vit_cifar10_benign__probe_bpp | 0.8 | 0.028 (0.050) | 0.000 (0.049) | 0.000 (0.000) | 0.000 (0.000) |
| benign control | vit_cifar10_benign__probe_bpp | 0.6 | 0.000 (0.050) | 0.000 (0.049) | 0.000 (0.000) | 0.000 (0.000) |
| benign control | vit_cifar10_benign__probe_bpp | 0.5 | 0.118 (0.050) | 0.059 (0.049) | 0.000 (0.000) | 0.000 (0.000) |
| benign control | vit_cifar10_benign__probe_bpp | 0.4 | 0.000 (0.050) | 0.000 (0.049) | 0.000 (0.000) | 0.000 (0.000) |
| benign control | vit_gtsrb_benign__probe_lf | 1.0 | 0.000 (0.052) | 0.000 (0.053) | 0.000 (0.000) | 0.000 (0.000) |
| benign control | vit_gtsrb_benign__probe_lf | 0.8 | 0.000 (0.052) | 0.000 (0.053) | 0.000 (0.000) | 0.000 (0.000) |
| benign control | vit_gtsrb_benign__probe_lf | 0.6 | 0.000 (0.052) | 0.000 (0.053) | 0.000 (0.000) | 0.000 (0.000) |
| benign control | vit_gtsrb_benign__probe_lf | 0.5 | 0.000 (0.052) | 0.000 (0.053) | 0.000 (0.000) | 0.000 (0.000) |
| benign control | vit_gtsrb_benign__probe_lf | 0.4 | 0.000 (0.052) | 0.000 (0.053) | 0.000 (0.000) | 0.000 (0.000) |

Threshold rules at level 0.01: TPR on the attack set, with the realized FPR on the clean test split and on its images predicted as the attacked class in brackets. The 0.05 and 0.10 levels are in readout.json.

| regime | entry | dose | canonical | pooled_two_sided | class_one_sided | class_two_sided |
|---|---|---|---|---|---|---|
| gate | vit_cifar100_blend_0_1 | 1.0 | 1.000 (0.009) | 1.000 (0.011, 0.000) | 1.000 (0.009, 0.000) | 1.000 (0.011, 0.000) |
| gate | vit_cifar100_blend_0_1 | 0.8 | 0.996 (0.009) | 0.989 (0.011, 0.000) | 0.996 (0.009, 0.000) | 0.989 (0.011, 0.000) |
| gate | vit_cifar100_blend_0_1 | 0.6 | 0.885 (0.009) | 0.836 (0.011, 0.000) | 0.875 (0.009, 0.000) | 0.836 (0.011, 0.000) |
| gate | vit_cifar100_blend_0_1 | 0.5 | 0.696 (0.009) | 0.615 (0.011, 0.000) | 0.684 (0.009, 0.000) | 0.615 (0.011, 0.000) |
| gate | vit_cifar100_blend_0_1 | 0.4 | 0.404 (0.009) | 0.321 (0.011, 0.000) | 0.384 (0.009, 0.000) | 0.321 (0.011, 0.000) |
| gate | vit_cifar10_badnet_a2o_0_01 | 1.0 | 0.861 (0.011) | 0.819 (0.009, 0.000) | 0.967 (0.009, 0.019) | 0.967 (0.010, 0.020) |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.8 | 0.828 (0.011) | 0.778 (0.009, 0.000) | 0.961 (0.009, 0.019) | 0.959 (0.010, 0.020) |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.6 | 0.686 (0.011) | 0.607 (0.009, 0.000) | 0.903 (0.009, 0.019) | 0.898 (0.010, 0.020) |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.5 | 0.582 (0.011) | 0.497 (0.009, 0.000) | 0.847 (0.009, 0.019) | 0.844 (0.010, 0.020) |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.4 | 0.429 (0.011) | 0.344 (0.009, 0.000) | 0.712 (0.009, 0.019) | 0.710 (0.010, 0.020) |
| gate | vit_gtsrb_lf_0_01 | 1.0 | 0.867 (0.010) | 0.860 (0.009, 0.000) | 0.865 (0.003, 0.000) | 0.860 (0.009, 0.000) |
| gate | vit_gtsrb_lf_0_01 | 0.8 | 0.800 (0.010) | 0.795 (0.009, 0.000) | 0.799 (0.003, 0.000) | 0.795 (0.009, 0.000) |
| gate | vit_gtsrb_lf_0_01 | 0.6 | 0.678 (0.010) | 0.663 (0.009, 0.000) | 0.674 (0.003, 0.000) | 0.663 (0.009, 0.000) |
| gate | vit_gtsrb_lf_0_01 | 0.5 | 0.525 (0.010) | 0.495 (0.009, 0.000) | 0.520 (0.003, 0.000) | 0.495 (0.009, 0.000) |
| gate | vit_gtsrb_lf_0_01 | 0.4 | 0.118 (0.010) | 0.100 (0.009, 0.000) | 0.117 (0.003, 0.000) | 0.100 (0.009, 0.000) |
| gate | vit_tiny_badnet_a2o_0_05 | 1.0 | 0.347 (0.011) | 0.200 (0.011, 0.176) | 0.343 (0.010, 0.176) | 0.200 (0.011, 0.176) |
| gate | vit_tiny_badnet_a2o_0_05 | 0.8 | 0.351 (0.011) | 0.200 (0.011, 0.176) | 0.342 (0.010, 0.176) | 0.200 (0.011, 0.176) |
| gate | vit_tiny_badnet_a2o_0_05 | 0.6 | 0.305 (0.011) | 0.169 (0.011, 0.176) | 0.301 (0.010, 0.176) | 0.169 (0.011, 0.176) |
| gate | vit_tiny_badnet_a2o_0_05 | 0.5 | 0.217 (0.011) | 0.122 (0.011, 0.176) | 0.213 (0.010, 0.176) | 0.122 (0.011, 0.176) |
| gate | vit_tiny_badnet_a2o_0_05 | 0.4 | 0.119 (0.011) | 0.057 (0.011, 0.176) | 0.117 (0.010, 0.176) | 0.057 (0.011, 0.176) |
| collapse | vit_cifar10_blend_0_1 | 1.0 | 0.564 (0.008) | 0.501 (0.009, 0.000) | 0.890 (0.007, 0.017) | 0.878 (0.008, 0.014) |
| collapse | vit_cifar10_blend_0_1 | 0.8 | 0.327 (0.008) | 0.278 (0.009, 0.000) | 0.659 (0.007, 0.017) | 0.639 (0.008, 0.014) |
| collapse | vit_cifar10_blend_0_1 | 0.6 | 0.122 (0.008) | 0.100 (0.009, 0.000) | 0.351 (0.007, 0.017) | 0.336 (0.008, 0.014) |
| collapse | vit_cifar10_blend_0_1 | 0.5 | 0.085 (0.008) | 0.066 (0.009, 0.000) | 0.248 (0.007, 0.017) | 0.235 (0.008, 0.014) |
| collapse | vit_cifar10_blend_0_1 | 0.4 | 0.030 (0.008) | 0.027 (0.009, 0.000) | 0.179 (0.007, 0.017) | 0.171 (0.008, 0.014) |
| collapse | vit_cifar10_bpp_0_05 | 1.0 | 0.407 (0.010) | 0.346 (0.008, 0.000) | 0.737 (0.007, 0.001) | 0.346 (0.007, 0.000) |
| collapse | vit_cifar10_bpp_0_05 | 0.8 | 0.228 (0.010) | 0.179 (0.008, 0.000) | 0.578 (0.007, 0.001) | 0.179 (0.007, 0.000) |
| collapse | vit_cifar10_bpp_0_05 | 0.6 | 0.058 (0.010) | 0.046 (0.008, 0.000) | 0.299 (0.007, 0.001) | 0.046 (0.007, 0.000) |
| collapse | vit_cifar10_bpp_0_05 | 0.5 | 0.027 (0.010) | 0.014 (0.008, 0.000) | 0.162 (0.007, 0.001) | 0.014 (0.007, 0.000) |
| collapse | vit_cifar10_bpp_0_05 | 0.4 | 0.000 (0.010) | 0.000 (0.008, 0.000) | 0.000 (0.007, 0.001) | 0.000 (0.007, 0.000) |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 1.0 | 0.000 (0.011) | 0.000 (0.010, 0.000) | 0.118 (0.008, 0.018) | 0.059 (0.005, 0.014) |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.8 | 0.000 (0.011) | 0.000 (0.010, 0.000) | 0.118 (0.008, 0.018) | 0.059 (0.005, 0.014) |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.6 | 0.000 (0.011) | 0.000 (0.010, 0.000) | 0.188 (0.008, 0.018) | 0.125 (0.005, 0.014) |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.5 | 0.000 (0.011) | 0.000 (0.010, 0.000) | 0.133 (0.008, 0.018) | 0.133 (0.005, 0.014) |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.4 | 0.000 (0.011) | 0.000 (0.010, 0.000) | 0.071 (0.008, 0.018) | 0.071 (0.005, 0.014) |
| benign control | vit_cifar10_benign__probe_blend | 1.0 | 0.034 (0.011) | 0.020 (0.010, 0.000) | 0.585 (0.008, 0.018) | 0.531 (0.005, 0.014) |
| benign control | vit_cifar10_benign__probe_blend | 0.8 | 0.031 (0.011) | 0.031 (0.010, 0.000) | 0.600 (0.008, 0.018) | 0.477 (0.005, 0.014) |
| benign control | vit_cifar10_benign__probe_blend | 0.6 | 0.032 (0.011) | 0.032 (0.010, 0.000) | 0.258 (0.008, 0.018) | 0.194 (0.005, 0.014) |
| benign control | vit_cifar10_benign__probe_blend | 0.5 | 0.050 (0.011) | 0.000 (0.010, 0.000) | 0.150 (0.008, 0.018) | 0.150 (0.005, 0.014) |
| benign control | vit_cifar10_benign__probe_blend | 0.4 | 0.000 (0.011) | 0.000 (0.010, 0.000) | 0.100 (0.008, 0.018) | 0.100 (0.005, 0.014) |
| benign control | vit_cifar10_benign__probe_bpp | 1.0 | 0.061 (0.011) | 0.030 (0.010, 0.000) | 0.364 (0.008, 0.018) | 0.273 (0.005, 0.014) |
| benign control | vit_cifar10_benign__probe_bpp | 0.8 | 0.000 (0.011) | 0.000 (0.010, 0.000) | 0.222 (0.008, 0.018) | 0.139 (0.005, 0.014) |
| benign control | vit_cifar10_benign__probe_bpp | 0.6 | 0.000 (0.011) | 0.000 (0.010, 0.000) | 0.318 (0.008, 0.018) | 0.273 (0.005, 0.014) |
| benign control | vit_cifar10_benign__probe_bpp | 0.5 | 0.000 (0.011) | 0.000 (0.010, 0.000) | 0.353 (0.008, 0.018) | 0.235 (0.005, 0.014) |
| benign control | vit_cifar10_benign__probe_bpp | 0.4 | 0.000 (0.011) | 0.000 (0.010, 0.000) | 0.083 (0.008, 0.018) | 0.000 (0.005, 0.014) |
| benign control | vit_gtsrb_benign__probe_lf | 1.0 | 0.000 (0.013) | 0.000 (0.012, 0.000) | 0.000 (0.002, 0.000) | 0.000 (0.012, 0.000) |
| benign control | vit_gtsrb_benign__probe_lf | 0.8 | 0.000 (0.013) | 0.000 (0.012, 0.000) | 0.000 (0.002, 0.000) | 0.000 (0.012, 0.000) |
| benign control | vit_gtsrb_benign__probe_lf | 0.6 | 0.000 (0.013) | 0.000 (0.012, 0.000) | 0.000 (0.002, 0.000) | 0.000 (0.012, 0.000) |
| benign control | vit_gtsrb_benign__probe_lf | 0.5 | 0.000 (0.013) | 0.000 (0.012, 0.000) | 0.000 (0.002, 0.000) | 0.000 (0.012, 0.000) |
| benign control | vit_gtsrb_benign__probe_lf | 0.4 | 0.000 (0.013) | 0.000 (0.012, 0.000) | 0.000 (0.002, 0.000) | 0.000 (0.012, 0.000) |

Matched filter, an extension under its own threat model, on the held-out half of the pairs and the odd clean test rows.

| regime | entry | dose, suspected class, flagged stream queries of it | held-out attack set | canonical TPR at 0.01 (FPR) | matched filter TPR at 0.01 (FPR) | matched AUROC against clean of the class | matched AUROC against twins |
|---|---|---|---|---|---|---|---|
| gate | vit_cifar100_blend_0_1 | 1.0, class 0, 512 catches | 512 | 1.000 (0.009) | 1.000 (0.008) | 1.000 | 1.000 |
| gate | vit_cifar100_blend_0_1 | 0.8, class 0, 512 catches | 512 | 0.994 (0.009) | 1.000 (0.008) | 1.000 | 1.000 |
| gate | vit_cifar100_blend_0_1 | 0.6, class 0, 512 catches | 512 | 0.871 (0.009) | 0.996 (0.008) | 1.000 | 1.000 |
| gate | vit_cifar100_blend_0_1 | 0.5, class 0, 512 catches | 508 | 0.693 (0.009) | 0.970 (0.008) | 0.998 | 0.996 |
| gate | vit_cifar100_blend_0_1 | 0.4, class 0, 512 catches | 436 | 0.385 (0.009) | 0.901 (0.008) | 0.993 | 0.990 |
| gate | vit_cifar10_badnet_a2o_0_01 | 1.0, class 0, 441 catches | 511 | 0.855 (0.013) | 0.679 (0.012) | 1.000 | 0.989 |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.8, class 0, 441 catches | 509 | 0.819 (0.013) | 0.654 (0.012) | 1.000 | 0.987 |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.6, class 0, 441 catches | 492 | 0.697 (0.013) | 0.571 (0.012) | 1.000 | 0.981 |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.5, class 0, 441 catches | 476 | 0.595 (0.013) | 0.506 (0.012) | 1.000 | 0.971 |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.4, class 0, 441 catches | 380 | 0.434 (0.013) | 0.342 (0.012) | 1.000 | 0.950 |
| gate | vit_gtsrb_lf_0_01 | 1.0, class 0, 444 catches | 505 | 0.853 (0.009) | 1.000 (0.008) | 1.000 | 1.000 |
| gate | vit_gtsrb_lf_0_01 | 0.8, class 0, 444 catches | 478 | 0.789 (0.009) | 0.998 (0.008) | 1.000 | 1.000 |
| gate | vit_gtsrb_lf_0_01 | 0.6, class 0, 444 catches | 403 | 0.655 (0.009) | 1.000 (0.008) | 1.000 | 1.000 |
| gate | vit_gtsrb_lf_0_01 | 0.5, class 0, 444 catches | 349 | 0.521 (0.009) | 0.991 (0.008) | 1.000 | 1.000 |
| gate | vit_gtsrb_lf_0_01 | 0.4, class 0, 444 catches | 260 | 0.142 (0.009) | 0.958 (0.008) | 1.000 | 0.997 |
| gate | vit_tiny_badnet_a2o_0_05 | 1.0, class 0, 181 catches | 512 | 0.352 (0.010) | 1.000 (0.009) | 1.000 | 1.000 |
| gate | vit_tiny_badnet_a2o_0_05 | 0.8, class 0, 181 catches | 512 | 0.346 (0.010) | 1.000 (0.009) | 1.000 | 1.000 |
| gate | vit_tiny_badnet_a2o_0_05 | 0.6, class 0, 181 catches | 490 | 0.300 (0.010) | 1.000 (0.009) | 1.000 | 1.000 |
| gate | vit_tiny_badnet_a2o_0_05 | 0.5, class 0, 181 catches | 416 | 0.209 (0.010) | 1.000 (0.009) | 1.000 | 1.000 |
| gate | vit_tiny_badnet_a2o_0_05 | 0.4, class 0, 181 catches | 269 | 0.097 (0.010) | 0.993 (0.009) | 1.000 | 0.999 |
| collapse | vit_cifar10_blend_0_1 | 1.0, class 0, 289 catches | 512 | 0.564 (0.009) | 1.000 (0.012) | 1.000 | 1.000 |
| collapse | vit_cifar10_blend_0_1 | 0.8, class 0, 289 catches | 511 | 0.339 (0.009) | 1.000 (0.012) | 1.000 | 1.000 |
| collapse | vit_cifar10_blend_0_1 | 0.6, class 0, 289 catches | 471 | 0.127 (0.009) | 0.870 (0.012) | 1.000 | 0.988 |
| collapse | vit_cifar10_blend_0_1 | 0.5, class 0, 289 catches | 362 | 0.077 (0.009) | 0.630 (0.012) | 0.999 | 0.961 |
| collapse | vit_cifar10_blend_0_1 | 0.4, class 0, 289 catches | 184 | 0.033 (0.009) | 0.397 (0.012) | 0.982 | 0.907 |
| collapse | vit_cifar10_bpp_0_05 | 1.0, class 0, 209 catches | 502 | 0.398 (0.010) | 1.000 (0.009) | 1.000 | 1.000 |
| collapse | vit_cifar10_bpp_0_05 | 0.8, class 0, 209 catches | 483 | 0.228 (0.010) | 0.988 (0.009) | 1.000 | 0.999 |
| collapse | vit_cifar10_bpp_0_05 | 0.6, class 0, 209 catches | 219 | 0.046 (0.010) | 0.900 (0.009) | 0.993 | 0.974 |
| collapse | vit_cifar10_bpp_0_05 | 0.5, class 0, 209 catches | 33 | 0.061 (0.010) | 0.697 (0.009) | 0.968 | 0.924 |
| collapse | vit_cifar10_bpp_0_05 | 0.4, class 0, 209 catches | 2 | 0.000 (0.010) | 0.000 (0.009) | 0.901 | 0.581 |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 1.0, class 5, 28 catches | 9 | 0.000 (0.012) | 0.000 (0.009) | 0.800 | 0.672 |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.8, class 5, 28 catches | 9 | 0.000 (0.012) | 0.000 (0.009) | 0.797 | 0.670 |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.6, class 5, 28 catches | 8 | 0.000 (0.012) | 0.000 (0.009) | 0.777 | 0.670 |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.5, class 5, 28 catches | 8 | 0.000 (0.012) | 0.000 (0.009) | 0.777 | 0.670 |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.4, class 5, 28 catches | 7 | 0.000 (0.012) | 0.000 (0.009) | 0.786 | 0.691 |
| benign control | vit_cifar10_benign__probe_blend | 1.0, class 5, 27 catches | 74 | 0.014 (0.012) | 0.000 (0.008) | 0.959 | 0.818 |
| benign control | vit_cifar10_benign__probe_blend | 0.8, class 5, 27 catches | 33 | 0.030 (0.012) | 0.000 (0.008) | 0.953 | 0.811 |
| benign control | vit_cifar10_benign__probe_blend | 0.6, class 5, 27 catches | 16 | 0.062 (0.012) | 0.000 (0.008) | 0.896 | 0.773 |
| benign control | vit_cifar10_benign__probe_blend | 0.5, class 5, 27 catches | 10 | 0.000 (0.012) | 0.000 (0.008) | 0.927 | 0.761 |
| benign control | vit_cifar10_benign__probe_blend | 0.4, class 5, 27 catches | 6 | 0.000 (0.012) | 0.000 (0.008) | 0.880 | 0.721 |
| benign control | vit_cifar10_benign__probe_bpp | 1.0, class 5, 30 catches | 38 | 0.105 (0.012) | 0.000 (0.008) | 0.890 | 0.776 |
| benign control | vit_cifar10_benign__probe_bpp | 0.8, class 5, 30 catches | 19 | 0.000 (0.012) | 0.000 (0.008) | 0.855 | 0.745 |
| benign control | vit_cifar10_benign__probe_bpp | 0.6, class 5, 30 catches | 13 | 0.000 (0.012) | 0.000 (0.008) | 0.754 | 0.694 |
| benign control | vit_cifar10_benign__probe_bpp | 0.5, class 5, 30 catches | 11 | 0.000 (0.012) | 0.000 (0.008) | 0.792 | 0.696 |
| benign control | vit_cifar10_benign__probe_bpp | 0.4, class 5, 30 catches | 7 | 0.000 (0.012) | 0.000 (0.008) | 0.769 | 0.700 |
| benign control | vit_gtsrb_benign__probe_lf | 1.0, class 12, 50 catches | 1 | 0.000 (0.012) | 0.000 (0.009) | 0.029 | 0.232 |
| benign control | vit_gtsrb_benign__probe_lf | 0.8, class 12, 50 catches | 1 | 0.000 (0.012) | 0.000 (0.009) | 0.029 | 0.244 |
| benign control | vit_gtsrb_benign__probe_lf | 0.6, class 12, 50 catches | 1 | 0.000 (0.012) | 0.000 (0.009) | 0.029 | 0.254 |
| benign control | vit_gtsrb_benign__probe_lf | 0.5, class 12, 50 catches | 1 | 0.000 (0.012) | 0.000 (0.009) | 0.029 | 0.256 |
| benign control | vit_gtsrb_benign__probe_lf | 0.4, class 12, 50 catches | 1 | 0.000 (0.012) | 0.000 (0.009) | 0.029 | 0.262 |

Exploratory residual amplification at the 0.05 quantile, each pipeline at its own adaptive rate and thresholds, realized FPR on the clean twins for both.

| regime | entry | lambda | dose | ASR sharpened | plain TPR (FPR) | sharpened TPR (FPR) | sharpened AUROC against twins |
|---|---|---|---|---|---|---|---|
| gate | vit_cifar100_blend_0_1 | 0.25 | 1.0 | 1.000 | 1.000 (0.058) | 1.000 (0.060) | 0.997 |
| gate | vit_cifar100_blend_0_1 | 0.25 | 0.8 | 1.000 | 1.000 (0.058) | 1.000 (0.060) | 0.998 |
| gate | vit_cifar100_blend_0_1 | 0.25 | 0.6 | 1.000 | 0.980 (0.058) | 0.996 (0.060) | 0.997 |
| gate | vit_cifar100_blend_0_1 | 0.25 | 0.5 | 0.997 | 0.876 (0.058) | 0.970 (0.060) | 0.991 |
| gate | vit_cifar100_blend_0_1 | 0.25 | 0.4 | 0.958 | 0.699 (0.058) | 0.884 (0.060) | 0.978 |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.25 | 1.0 | 0.994 | 0.932 (0.051) | 0.930 (0.050) | 0.981 |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.25 | 0.8 | 0.993 | 0.913 (0.051) | 0.939 (0.050) | 0.982 |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.25 | 0.6 | 0.987 | 0.822 (0.051) | 0.899 (0.050) | 0.972 |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.25 | 0.5 | 0.961 | 0.751 (0.051) | 0.828 (0.050) | 0.952 |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.25 | 0.4 | 0.861 | 0.585 (0.051) | 0.737 (0.050) | 0.920 |
| gate | vit_gtsrb_lf_0_01 | 2.00 | 1.0 | 0.986 | 0.910 (0.045) | 0.945 (0.050) | 0.990 |
| gate | vit_gtsrb_lf_0_01 | 2.00 | 0.8 | 0.964 | 0.864 (0.045) | 0.927 (0.050) | 0.987 |
| gate | vit_gtsrb_lf_0_01 | 2.00 | 0.6 | 0.903 | 0.774 (0.045) | 0.936 (0.050) | 0.990 |
| gate | vit_gtsrb_lf_0_01 | 2.00 | 0.5 | 0.849 | 0.665 (0.045) | 0.904 (0.050) | 0.988 |
| gate | vit_gtsrb_lf_0_01 | 2.00 | 0.4 | 0.715 | 0.280 (0.045) | 0.867 (0.050) | 0.980 |
| gate | vit_tiny_badnet_a2o_0_05 | 0.25 | 1.0 | 0.999 | 0.994 (0.062) | 0.988 (0.058) | 0.983 |
| gate | vit_tiny_badnet_a2o_0_05 | 0.25 | 0.8 | 0.999 | 0.993 (0.062) | 0.989 (0.058) | 0.983 |
| gate | vit_tiny_badnet_a2o_0_05 | 0.25 | 0.6 | 0.979 | 0.987 (0.062) | 0.987 (0.058) | 0.983 |
| gate | vit_tiny_badnet_a2o_0_05 | 0.25 | 0.5 | 0.912 | 0.968 (0.062) | 0.980 (0.058) | 0.981 |
| gate | vit_tiny_badnet_a2o_0_05 | 0.25 | 0.4 | 0.680 | 0.855 (0.062) | 0.962 (0.058) | 0.977 |
| collapse | vit_cifar10_blend_0_1 | 0.50 | 1.0 | 1.000 | 0.699 (0.062) | 0.767 (0.059) | 0.940 |
| collapse | vit_cifar10_blend_0_1 | 0.50 | 0.8 | 1.000 | 0.438 (0.062) | 0.571 (0.059) | 0.843 |
| collapse | vit_cifar10_blend_0_1 | 0.50 | 0.6 | 0.989 | 0.193 (0.062) | 0.318 (0.059) | 0.663 |
| collapse | vit_cifar10_blend_0_1 | 0.50 | 0.5 | 0.937 | 0.116 (0.062) | 0.254 (0.059) | 0.594 |
| collapse | vit_cifar10_blend_0_1 | 0.50 | 0.4 | 0.704 | 0.054 (0.062) | 0.225 (0.059) | 0.560 |
| collapse | vit_cifar10_bpp_0_05 | 1.00 | 1.0 | 0.310 | 0.526 (0.040) | 0.052 (0.047) | 0.485 |
| collapse | vit_cifar10_bpp_0_05 | 1.00 | 0.8 | 0.815 | 0.347 (0.040) | 0.059 (0.047) | 0.546 |
| collapse | vit_cifar10_bpp_0_05 | 1.00 | 0.6 | 0.954 | 0.126 (0.040) | 0.252 (0.047) | 0.678 |
| collapse | vit_cifar10_bpp_0_05 | 1.00 | 0.5 | 0.928 | 0.041 (0.040) | 0.230 (0.047) | 0.656 |
| collapse | vit_cifar10_bpp_0_05 | 1.00 | 0.4 | 0.770 | 0.000 (0.040) | 0.333 (0.047) | 0.621 |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.50 | 1.0 | 0.017 | 0.000 (0.044) | 0.059 (0.046) | 0.672 |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.50 | 0.8 | 0.017 | 0.000 (0.044) | 0.059 (0.046) | 0.672 |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.50 | 0.6 | 0.017 | 0.000 (0.044) | 0.062 (0.046) | 0.659 |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.50 | 0.5 | 0.016 | 0.000 (0.044) | 0.067 (0.046) | 0.646 |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.50 | 0.4 | 0.016 | 0.000 (0.044) | 0.071 (0.046) | 0.634 |
| benign control | vit_cifar10_benign__probe_blend | 0.50 | 1.0 | 0.363 | 0.218 (0.044) | 0.075 (0.046) | 0.780 |
| benign control | vit_cifar10_benign__probe_blend | 0.50 | 0.8 | 0.154 | 0.123 (0.044) | 0.092 (0.046) | 0.740 |
| benign control | vit_cifar10_benign__probe_blend | 0.50 | 0.6 | 0.056 | 0.065 (0.044) | 0.065 (0.046) | 0.652 |
| benign control | vit_cifar10_benign__probe_blend | 0.50 | 0.5 | 0.037 | 0.050 (0.044) | 0.000 (0.046) | 0.618 |
| benign control | vit_cifar10_benign__probe_blend | 0.50 | 0.4 | 0.030 | 0.000 (0.044) | 0.000 (0.046) | 0.583 |
| benign control | vit_cifar10_benign__probe_bpp | 0.50 | 1.0 | 0.119 | 0.167 (0.044) | 0.045 (0.046) | 0.673 |
| benign control | vit_cifar10_benign__probe_bpp | 0.50 | 0.8 | 0.064 | 0.028 (0.044) | 0.028 (0.046) | 0.612 |
| benign control | vit_cifar10_benign__probe_bpp | 0.50 | 0.6 | 0.043 | 0.000 (0.044) | 0.000 (0.046) | 0.669 |
| benign control | vit_cifar10_benign__probe_bpp | 0.50 | 0.5 | 0.031 | 0.118 (0.044) | 0.000 (0.046) | 0.693 |
| benign control | vit_cifar10_benign__probe_bpp | 0.50 | 0.4 | 0.022 | 0.000 (0.044) | 0.000 (0.046) | 0.650 |
| benign control | vit_gtsrb_benign__probe_lf | 2.00 | 1.0 | 0.004 | 0.000 (0.056) | 0.000 (0.050) | 0.146 |
| benign control | vit_gtsrb_benign__probe_lf | 2.00 | 0.8 | 0.004 | 0.000 (0.056) | 0.000 (0.050) | 0.102 |
| benign control | vit_gtsrb_benign__probe_lf | 2.00 | 0.6 | 0.003 | 0.000 (0.056) | 0.000 (0.050) | 0.098 |
| benign control | vit_gtsrb_benign__probe_lf | 2.00 | 0.5 | 0.003 | 0.000 (0.056) | 0.000 (0.050) | 0.098 |
| benign control | vit_gtsrb_benign__probe_lf | 2.00 | 0.4 | 0.003 | 0.000 (0.056) | 0.000 (0.050) | 0.098 |

Consistency of the per-input pass.

| regime | entry | fresh against cached twins, AUROC | fresh against cached triggered, AUROC | largest gap to the dose record | threshold gap to the canonical cache |
|---|---|---|---|---|---|
| gate | vit_cifar100_blend_0_1 | 0.508 | 0.505 | 0.005 | 0.000000 |
| gate | vit_cifar10_badnet_a2o_0_01 | 0.498 | 0.507 | 0.007 | 0.000000 |
| gate | vit_gtsrb_lf_0_01 | 0.496 | 0.501 | 0.004 | 0.000000 |
| gate | vit_tiny_badnet_a2o_0_05 | 0.507 | 0.486 | 0.010 | 0.000000 |
| collapse | vit_cifar10_blend_0_1 | 0.503 | 0.496 | 0.007 | 0.000000 |
| collapse | vit_cifar10_bpp_0_05 | 0.498 | 0.498 | 0.011 | 0.000000 |
| benign control | vit_cifar10_benign__probe_badnet_a2o | 0.500 | 0.501 | 0.001 | 0.000000 |
| benign control | vit_cifar10_benign__probe_blend | 0.500 | n/a | 0.001 | 0.000000 |
| benign control | vit_cifar10_benign__probe_bpp | 0.500 | n/a | n/a | 0.000000 |
| benign control | vit_gtsrb_benign__probe_lf | 0.503 | n/a | n/a | 0.000000 |
<!-- results:end -->
