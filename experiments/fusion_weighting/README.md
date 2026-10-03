# Fusion weighting

PSBD-TM is fused with a residual-dropout probe in the final method, and the paper reports 2 rules for the fusion side by side. The plain minimum of the 2 clean-validation percentiles (min) flags an input when either probe finds it suspicious. The weighted minimum min(r_TM / 0.9, r_partner / 0.1) (weighted 0.9/0.1) is the rule `experiments/cache_readouts/preregistration.json` fixed. This experiment produces every number for both rules on every set that carries both probes, and tests whether the 0.9/0.1 shares are a principled choice or a lucky one. It reads the stage-1 caches only, on the CPU.

The pre-registration was written at 2026-09-29T18:58:00Z (SHA-256 `9a8a7ce3d60188be4f287c21797ad6edea3c96d6132621bae897f93de66019b8`), before any held-out score was read. Its stated reason for the shares is "On the 5 dev models where the late band reaches the adaptive target, every fusion rule lifted the WaNet failure. Min-rank and Fisher cost the GTSRB TaCT model most of its TPR at 10% FPR. The 0.9 and 0.1 shares kept most of the WaNet gain and lost the least on TaCT, and the plan's combined design fixes the same shares." It paired the shares with the late band, residual dropout in blocks 9 to 12, which reaches the adaptive target on 41 of 56 panel models. The final method uses the middle band, blocks 5 to 8, which reaches it on all 56. So the 0.9/0.1 shares on the middle band are carried over from the pre-registered pairing, and the late band is read beside it on its 41 models.

## Model sets and partners

| set | models | partner placement | rate of the partner | source |
|---|---|---|---|---|
| ViT-B/16 panel, partner blocks 5 to 8 | 56 | `pre_residual_blocks_5_8` | adaptive | `readings_vit_panel.json` |
| ViT-B/16 panel, partner blocks 9 to 12 (pre-registered pairing) | 41 | `pre_residual_blocks_9_12` | adaptive | `readings_vit_panel.json` |
| Swin-S panel, partner blocks 17 to 24, adaptive rate | 39 | `pre_residual_blocks_17_24` | adaptive | `readings_swin_panel.json` |
| Swin-S panel, partner blocks 17 to 24, with the nearest-rate models | 63 | `pre_residual_blocks_17_24` | adaptive and nearest | `readings_swin_panel.json` |
| BackdoorBench ViT-B/16, partner blocks 5 to 8 | 10 | `pre_residual_blocks_5_8` | adaptive | `readings_backdoorbench.json` |
| training-set setting, partner blocks 5 to 8 | 6 | `pre_residual_blocks_5_8` | adaptive | `readings_training_set.json` |

The ViT panel is the 56 `successful_2pt` models that carry both headline placements (`experiments.probe_union.measure.select_models`). The Swin panel is `scripts.paper.tab_swin.swin_cells`. BackdoorBench is every model under `results/_experiments/backdoorbench_attacks/models/` with detection rates, read from `results/bb_<folder>/psbd/`. That README drops no swept model for the 2-point bar, so neither does this set. The training-set setting is the 6 pooled paper-mirror models of `experiments/training_set_detection/` (`summary.json`, `headline`), read from its raw parts.

The adaptive attacker models are not read. No ViT evader carries `pre_residual_blocks_5_8`, and the Swin evaders carry `pre_residual_blocks_9_16` on 4 models and `pre_residual_blocks_17_24` on none, so no evader has both probes of the final method cached.

## Swin-S partner placement

The Swin-S final method reads `pre_residual_blocks_17_24`, residual dropout in blocks 17 to 24. `experiments/final_method/fusion_readout.py` still names `pre_residual_blocks_9_16` in `MIDDLE_BAND`, and the opening paragraph of `experiments/final_method/README.md` says blocks 9 to 16. That was the 1st pre-registered Swin partner (`experiments/cache_readouts/preregistration_swin.json`) and its predictions failed. The 2nd and last attempt held under the min rule. It is `experiments/final_method/swin_late_readout.py` with `PARTNER = "pre_residual_blocks_17_24"` as pre-registered in `preregistration_swin_late.json`. Every later reader uses it: `experiments/score_figures/config.py` maps the Swin `band` probe to it and the report's Swin tables state blocks 17 to 24. On disk 63 of 65 Swin panel models carry the blocks 17 to 24 cache and 63 carry blocks 9 to 16, so the caches allow either partner and the readers decide. The blocks 17 to 24 partner reaches the adaptive target on 39 models, and the report's 63-model Swin table reads the other 24 at the rate nearest the target, so both readings are given here. The 9 to 16 in the final method's README opening and in `MIDDLE_BAND` is stale.

## Method

Each probe is read at its own adaptive rate, the smallest cached rate whose clean-validation shift ratio reaches 0.8, and scored with fractional PSU. Each score becomes its percentile within the probe's own 2000-image clean-validation distribution (`defenses.scores.to_rank`). The fused score at TM share w is u(x) = min(r_TM(x) / w, r_partner(x) / (1 - w)), so w = 0.5 orders inputs exactly as the plain minimum and w = 0.9 is the weighted rule. Every score, fused or single, is thresholded at a quantile of its own clean-validation distribution (`defenses.decision.threshold_at_quantile`, low means poisoned), TPR is read on the triggered split and the realized FPR on the paired clean test split (`pair_clean_to_backdoor`). The training-set setting thresholds at quantiles of the clean training images, as `experiments/training_set_detection/` does, so its FPR is the quantile by construction. Paired intervals are bootstrap 95% intervals over models (`scripts.paper._common.bootstrap_ci`, 5000 resamples, seed 0). A model loses at a budget when its TPR there falls more than 0.05 below PSBD-TM alone.

The share sweep reads w at 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95 and 0.99 with intervals. The curves read every 0.01 from 0.50 to 0.99.

<!-- results:begin -->
<!-- Everything down to results:end is rendered by render.py from summary.json. -->

## Budget allocation

The weighted minimum is the weighted Bonferroni rule of multiple testing applied to 2 detectors. With shares $w_i$ the fused score and the literal decision at nominal FPR $q$ are

$$u(x) = \min_{i} \frac{r_i(x)}{w_i}, \qquad \sum_{i} w_i = 1, \quad w_i > 0, \qquad \text{flag } x \iff u(x) \le q .$$

| symbol | meaning |
|---|---|
| $x$ | an input |
| $i$ | a probe, PSBD-TM ($i = 1$) or the residual partner ($i = 2$) |
| $r_i(x)$ | the percentile of probe $i$'s fractional PSU within its own clean-validation scores |
| $w_i$ | the share of the false-positive budget given to probe $i$, $w_1 = w$ and $w_2 = 1 - w$ |
| $q$ | the nominal false-positive rate |
| $P_0, P_1$ | probability over clean and over triggered inputs |
| $\beta_i(t)$ | the TPR of probe $i$ alone at FPR $t$, $P_1(r_i \le t)$, its ROC |
| $\pi$ | the share of models a probe other than PSBD-TM carries |
| $\bar\beta_i$ | $\beta_i$ averaged over the models probe $i$ carries |
| $c_i, \gamma$ | scale and exponent of a power-law ROC $\bar\beta_i(t) = c_i t^{\gamma}$ |

The decision is a union of 1 event per probe, since $u(x) \le q$ exactly when some $r_i(x) \le w_i q$,

$$\{u \le q\} = \bigcup_{i} \{r_i \le w_i q\} .$$

Boole's inequality bounds the clean flag rate by the sum of the per-probe rates. A clean-validation percentile is uniform on clean data (the probability integral transform), so each term is $w_i q$, exactly on the validation set up to 1 image per probe,

$$P_0(u \le q) \le \sum_{i} P_0(r_i \le w_i q) = \sum_{i} w_i q = q .$$

The shares therefore split a fixed budget $q$ between the probes, and the weighted rule spends 0.9 of it on PSBD-TM and 0.1 on the partner. The power is at least that of the better probe at its own share,

$$P_1(u \le q) \ge \max_{i} \beta_i(w_i q) .$$

To choose $w$, approximate the panel as models carried by PSBD-TM (share $1 - \pi$) and models carried by the partner (share $\pi$), and drop the overlap of the 2 events. This approximation drops the TPR an input gains from being caught by both probes. The mean TPR is then

$$T(w) = (1 - \pi)\, \bar\beta_1(w q) + \pi\, \bar\beta_2\big((1 - w) q\big) .$$

Differentiating by the chain rule and setting the derivative to 0 gives the optimum, a maximum when both ROCs are concave, where the 2 probes earn the same TPR per unit of budget,

$$\frac{dT}{dw} = (1 - \pi)\, q\, \bar\beta_1'(w q) - \pi\, q\, \bar\beta_2'\big((1 - w) q\big) = 0 \iff (1 - \pi)\, \bar\beta_1'(w^\ast q) = \pi\, \bar\beta_2'\big((1 - w^\ast) q\big) .$$

For a power-law ROC $\bar\beta_i(t) = c_i t^{\gamma}$ with $0 < \gamma < 1$, $\bar\beta_i'(t) = c_i \gamma t^{\gamma - 1}$, and the condition becomes $(1 - \pi) c_1 (w^\ast)^{\gamma - 1} = \pi c_2 (1 - w^\ast)^{\gamma - 1}$ once $\gamma q^{\gamma - 1}$ cancels. Solving for the ratio of shares,

$$\frac{w^\ast}{1 - w^\ast} = \left( \frac{(1 - \pi)\, c_1}{\pi\, c_2} \right)^{1 / (1 - \gamma)} .$$

$q$ drops out, so 1 share serves every budget. With $c_1 = c_2$ and $\pi < 1/2$ the base exceeds 1 and the exponent exceeds 1, so $w^\ast / (1 - w^\ast) \ge (1 - \pi) / \pi$, that is $w^\ast \ge 1 - \pi$. A primary that carries most attacks keeps at least that share of the budget, and the plain minimum is optimal only when the 2 probes carry the panel equally. A grid search of $T(w)$ matched the closed form to 4 decimals at 4 settings of $(\pi, c_1, c_2, \gamma)$.

The assumption that fails in practice is the common exponent. The optimum depends on how steep PSBD-TM's ROC is just below the budget, so the measurement that decides it is how much TPR PSBD-TM loses when its budget is cut from 1% to 0.5%. The last 2 columns below read $T(w)$ without the disjointness approximation, as the mean over models of $\max(\beta_1(w q), \beta_2((1 - w) q))$ with each probe's measured ROC at $q$ = 1%, and compare its best share with the sweep's.

| set | models | partner leads by more than 0.05 at 1% (π) | PSBD-TM TPR at 1% | PSBD-TM TPR at 0.5% | lost by halving | partner TPR at 0.5% | partner TPR at 0.1% | best share of the max bound | best share of the sweep |
|---|---|---|---|---|---|---|---|---|---|
| ViT-B/16, blocks 5 to 8 | 56 | 11 (0.196) | 0.735 | 0.653 | 0.082 | 0.506 | 0.347 | 0.89 | 0.64 |
| ViT-B/16, blocks 9 to 12 | 41 | 11 (0.268) | 0.771 | 0.699 | 0.072 | 0.586 | 0.512 | 0.70 | 0.74 |
| Swin-S, blocks 17 to 24 | 39 | 7 (0.179) | 0.808 | 0.781 | 0.027 | 0.532 | 0.323 | 0.50 | 0.51 |
| Swin-S, blocks 17 to 24, nearest rate added | 63 | 9 (0.143) | 0.824 | 0.799 | 0.025 | 0.519 | 0.325 | 0.50 | 0.52 |
| BackdoorBench | 10 | 5 (0.500) | 0.667 | 0.407 | 0.260 | 0.551 | 0.448 | 0.90 | 0.86 |
| training set | 6 | 3 (0.500) | 0.535 | 0.440 | 0.094 | 0.599 | 0.369 | 0.51 | 0.52 |

PSBD-TM loses 0.027 of TPR on Swin-S when its budget halves, 0.082 on the ViT panel and 0.260 on BackdoorBench. The partner carries a similar share of models on both architectures (11 of 56 on ViT, 7 of 39 on Swin), so the corollary $w^\ast \ge 1 - \pi$ would place both optima near 0.8. The measured ROCs move the Swin optimum to the plain minimum, because halving PSBD-TM's budget there costs little, and the max bound finds that best share where the sweep does. On the ViT panel both curves are flat, within 0.015 for the bound and 0.017 for the sweep across every share, so their maxima (0.89 and 0.64) do not separate.

The literal decision $u(x) \le q$ is the one the bound is stated for. The realized clean flag rates below confirm it on every share of the sweep, on validation within 1 image per probe of $q$, and on the paired clean test split, which the bound does not cover, near $q$ on average.

| set | share | validation FPR at 1%, mean / max | at 5% | at 10% | at 20% | clean test FPR at 1%, mean / max | TPR at 1%, literal |
|---|---|---|---|---|---|---|---|
| ViT-B/16, blocks 5 to 8 | 0.50 | 0.0102 / 0.0110 | 0.0456 / 0.0510 | 0.0876 / 0.1005 | 0.1654 / 0.1960 | 0.0097 / 0.0166 | 0.744 |
| ViT-B/16, blocks 5 to 8 | 0.60 | 0.0098 / 0.0105 | 0.0456 / 0.0510 | 0.0882 / 0.1005 | 0.1666 / 0.1980 | 0.0096 / 0.0181 | 0.744 |
| ViT-B/16, blocks 5 to 8 | 0.70 | 0.0098 / 0.0105 | 0.0461 / 0.0510 | 0.0894 / 0.1005 | 0.1709 / 0.1995 | 0.0097 / 0.0175 | 0.743 |
| ViT-B/16, blocks 5 to 8 | 0.80 | 0.0095 / 0.0100 | 0.0473 / 0.0510 | 0.0919 / 0.1010 | 0.1779 / 0.2000 | 0.0094 / 0.0176 | 0.739 |
| ViT-B/16, blocks 5 to 8 | 0.85 | 0.0100 / 0.0105 | 0.0479 / 0.0510 | 0.0934 / 0.1010 | 0.1824 / 0.2000 | 0.0098 / 0.0176 | 0.745 |
| ViT-B/16, blocks 5 to 8 | 0.90 | 0.0102 / 0.0105 | 0.0482 / 0.0505 | 0.0951 / 0.1005 | 0.1872 / 0.2000 | 0.0099 / 0.0181 | 0.748 |
| ViT-B/16, blocks 5 to 8 | 0.95 | 0.0103 / 0.0105 | 0.0496 / 0.0510 | 0.0979 / 0.1010 | 0.1936 / 0.2000 | 0.0099 / 0.0180 | 0.748 |
| ViT-B/16, blocks 5 to 8 | 0.99 | 0.0103 / 0.0105 | 0.0502 / 0.0505 | 0.0999 / 0.1005 | 0.1990 / 0.2005 | 0.0099 / 0.0180 | 0.748 |
| Swin-S, blocks 17 to 24 | 0.50 | 0.0101 / 0.0110 | 0.0470 / 0.0505 | 0.0916 / 0.0970 | 0.1728 / 0.1915 | 0.0089 / 0.0187 | 0.923 |
| Swin-S, blocks 17 to 24 | 0.60 | 0.0097 / 0.0105 | 0.0471 / 0.0510 | 0.0917 / 0.0985 | 0.1749 / 0.1925 | 0.0083 / 0.0152 | 0.907 |
| Swin-S, blocks 17 to 24 | 0.70 | 0.0097 / 0.0105 | 0.0473 / 0.0510 | 0.0926 / 0.0995 | 0.1788 / 0.1945 | 0.0082 / 0.0146 | 0.900 |
| Swin-S, blocks 17 to 24 | 0.80 | 0.0095 / 0.0100 | 0.0477 / 0.0505 | 0.0939 / 0.0995 | 0.1838 / 0.1950 | 0.0081 / 0.0144 | 0.863 |
| Swin-S, blocks 17 to 24 | 0.85 | 0.0100 / 0.0105 | 0.0480 / 0.0510 | 0.0950 / 0.0995 | 0.1870 / 0.1955 | 0.0084 / 0.0151 | 0.864 |
| Swin-S, blocks 17 to 24 | 0.90 | 0.0102 / 0.0105 | 0.0481 / 0.0505 | 0.0957 / 0.0995 | 0.1900 / 0.1970 | 0.0084 / 0.0152 | 0.839 |
| Swin-S, blocks 17 to 24 | 0.95 | 0.0104 / 0.0105 | 0.0495 / 0.0510 | 0.0978 / 0.1005 | 0.1947 / 0.1995 | 0.0085 / 0.0143 | 0.829 |
| Swin-S, blocks 17 to 24 | 0.99 | 0.0104 / 0.0105 | 0.0502 / 0.0505 | 0.0999 / 0.1005 | 0.1991 / 0.2005 | 0.0085 / 0.0143 | 0.829 |
| BackdoorBench | 0.50 | 0.0102 / 0.0110 | 0.0472 / 0.0510 | 0.0909 / 0.1010 | 0.1716 / 0.1995 | 0.0091 / 0.0161 | 0.607 |
| BackdoorBench | 0.60 | 0.0098 / 0.0105 | 0.0476 / 0.0510 | 0.0907 / 0.1010 | 0.1719 / 0.1935 | 0.0083 / 0.0142 | 0.597 |
| BackdoorBench | 0.70 | 0.0098 / 0.0105 | 0.0481 / 0.0510 | 0.0921 / 0.1010 | 0.1751 / 0.1900 | 0.0083 / 0.0144 | 0.591 |
| BackdoorBench | 0.80 | 0.0095 / 0.0100 | 0.0486 / 0.0510 | 0.0941 / 0.1010 | 0.1815 / 0.1925 | 0.0078 / 0.0142 | 0.613 |
| BackdoorBench | 0.85 | 0.0100 / 0.0105 | 0.0491 / 0.0510 | 0.0956 / 0.1005 | 0.1852 / 0.1945 | 0.0081 / 0.0144 | 0.668 |
| BackdoorBench | 0.90 | 0.0102 / 0.0105 | 0.0492 / 0.0505 | 0.0967 / 0.1000 | 0.1893 / 0.1960 | 0.0081 / 0.0151 | 0.690 |
| BackdoorBench | 0.95 | 0.0104 / 0.0105 | 0.0502 / 0.0510 | 0.0990 / 0.1005 | 0.1952 / 0.1985 | 0.0080 / 0.0148 | 0.690 |
| BackdoorBench | 0.99 | 0.0104 / 0.0105 | 0.0503 / 0.0505 | 0.1001 / 0.1005 | 0.1993 / 0.2000 | 0.0080 / 0.0148 | 0.690 |

The largest validation flag rate above $q$ on any model, share and budget is 0.0010, at most 1 image per probe of the 2000.

## Both rules on every set

Each set gives the mean over models of TPR at 1%, 5%, 10% and 20% FPR, the realized FPR on the paired clean test split and AUROC, then the paired differences with their 95% intervals and the count of models losing more than 0.05 against PSBD-TM alone. The per attack and per dataset tables follow each set.

### ViT-B/16 panel, partner blocks 5 to 8, 56 models

| rule | TPR 1% | TPR 5% | TPR 10% | TPR 20% | FPR 1% | FPR 5% | FPR 10% | FPR 20% | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| PSBD-TM alone | 0.735 | 0.841 | 0.872 | 0.900 | 0.0100 | 0.0460 | 0.0913 | 0.1831 | 0.951 |
| min | 0.742 | 0.857 | 0.889 | 0.921 | 0.0095 | 0.0460 | 0.0921 | 0.1866 | 0.970 |
| weighted 0.9/0.1 | 0.748 | 0.854 | 0.886 | 0.914 | 0.0097 | 0.0457 | 0.0906 | 0.1831 | 0.964 |

| paired difference | TPR 1% | TPR 5% | TPR 10% | TPR 20% | AUROC |
|---|---|---|---|---|---|
| min minus PSBD-TM | +0.007 [-0.037, +0.049] | +0.017 [-0.006, +0.050] | +0.017 [-0.010, +0.055] | +0.021 [-0.007, +0.057] | +0.020 [+0.001, +0.047] |
| weighted minus PSBD-TM | +0.013 [+0.000, +0.027] | +0.014 [+0.000, +0.035] | +0.014 [+0.001, +0.037] | +0.014 [+0.001, +0.036] | +0.013 [+0.001, +0.032] |
| weighted minus min | +0.006 [-0.032, +0.043] | -0.003 [-0.017, +0.009] | -0.003 [-0.020, +0.012] | -0.007 [-0.028, +0.011] | -0.007 [-0.018, +0.002] |

| models losing more than 0.05 against PSBD-TM | at 1% | at 5% | at 10% | at 20% | AUROC |
|---|---|---|---|---|---|
| min | 10 (worst -0.580, `vit_cifar10_blend_0_05`) | 5 (worst -0.134, `vit_gtsrb_tact_0_05`) | 4 (worst -0.271, `vit_gtsrb_tact_0_05`) | 1 (worst -0.288, `vit_gtsrb_tact_0_05`) | 1 (worst -0.079, `vit_cifar10_tact_0_01`) |
| weighted 0.9/0.1 | 1 (worst -0.055, `vit_cifar100_badnet_a2o_0_01`) | 0 (worst -0.040, `vit_cifar100_bpp_0_01`) | 0 (worst -0.041, `vit_gtsrb_tact_0_05`) | 0 (worst -0.025, `vit_gtsrb_tact_0_05`) | 0 (worst -0.003, `vit_cifar100_bpp_0_01`) |

| attack (models) | PSBD-TM TPR 1 / 5 / 10% | min TPR 1 / 5 / 10% | weighted TPR 1 / 5 / 10% | AUROC PSBD-TM / min / weighted | weighted minus min, TPR 1% | losing at 1%, min / weighted |
|---|---|---|---|---|---|---|
| badnet_a2o (12) | 0.776 / 0.963 / 0.987 | 0.801 / 0.951 / 0.973 | 0.794 / 0.964 / 0.986 | 0.992 / 0.991 / 0.993 | -0.007 [-0.110, +0.096] | 4 / 1 |
| tact (4) | 0.009 / 0.059 / 0.127 | 0.008 / 0.065 / 0.141 | 0.008 / 0.056 / 0.132 | 0.788 / 0.891 / 0.845 | -0.000 [-0.011, +0.010] | 0 / 0 |
| blend (12) | 0.829 / 0.915 / 0.943 | 0.771 / 0.927 / 0.961 | 0.848 / 0.923 / 0.952 | 0.978 / 0.984 / 0.982 | +0.077 [-0.003, +0.194] | 2 / 0 |
| lf (12) | 0.904 / 0.959 / 0.970 | 0.908 / 0.967 / 0.975 | 0.923 / 0.967 / 0.974 | 0.980 / 0.985 / 0.982 | +0.015 [+0.002, +0.031] | 2 / 0 |
| bpp (12) | 0.757 / 0.847 / 0.888 | 0.746 / 0.838 / 0.885 | 0.757 / 0.845 / 0.888 | 0.948 / 0.952 / 0.949 | +0.011 [-0.001, +0.025] | 2 / 0 |
| wanet (3) | 0.372 / 0.572 / 0.647 | 0.671 / 0.873 / 0.918 | 0.380 / 0.762 / 0.840 | 0.779 / 0.948 / 0.918 | -0.291 [-0.510, -0.030] | 0 / 0 |
| lc (1) | 0.833 / 0.921 / 0.950 | 0.805 / 0.915 / 0.959 | 0.833 / 0.924 / 0.955 | 0.969 / 0.982 / 0.973 | +0.028 (n<3) | 0 / 0 |

| dataset (models) | PSBD-TM TPR 1 / 5 / 10% | min TPR 1 / 5 / 10% | weighted TPR 1 / 5 / 10% | AUROC PSBD-TM / min / weighted | weighted minus min, TPR 1% | losing at 1%, min / weighted |
|---|---|---|---|---|---|---|
| CIFAR-10 (15) | 0.573 / 0.660 / 0.706 | 0.508 / 0.705 / 0.757 | 0.566 / 0.694 / 0.746 | 0.919 / 0.951 / 0.948 | +0.058 [-0.053, +0.171] | 6 / 0 |
| CIFAR-100 (12) | 0.790 / 0.935 / 0.964 | 0.783 / 0.931 / 0.966 | 0.799 / 0.939 / 0.968 | 0.979 / 0.981 / 0.980 | +0.016 [-0.071, +0.096] | 4 / 1 |
| GTSRB (15) | 0.801 / 0.839 / 0.869 | 0.806 / 0.843 / 0.876 | 0.810 / 0.844 / 0.875 | 0.935 / 0.969 / 0.951 | +0.004 [-0.002, +0.010] | 0 / 0 |
| Tiny ImageNet (14) | 0.793 / 0.955 / 0.975 | 0.890 / 0.972 / 0.979 | 0.832 / 0.964 / 0.978 | 0.977 / 0.984 / 0.980 | -0.058 [-0.117, -0.011] | 0 / 0 |

### ViT-B/16 panel, partner blocks 9 to 12 (pre-registered pairing), 41 models

Not scored, 15 models: `vit_cifar10_badnet_a2o_0_01`, `vit_cifar10_badnet_a2o_0_05`, `vit_cifar10_badnet_a2o_0_1`, `vit_cifar10_blend_0_01`, `vit_cifar10_blend_0_05`, `vit_cifar10_blend_0_1`, `vit_cifar10_bpp_0_01`, `vit_cifar10_bpp_0_05`, `vit_cifar10_bpp_0_1`, `vit_cifar10_lf_0_01`, `vit_cifar10_lf_0_05`, `vit_cifar10_lf_0_1`, `vit_cifar10_tact_0_01`, `vit_cifar10_tact_0_05`, `vit_gtsrb_blend_0_05`.

| rule | TPR 1% | TPR 5% | TPR 10% | TPR 20% | FPR 1% | FPR 5% | FPR 10% | FPR 20% | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| PSBD-TM alone | 0.771 | 0.883 | 0.911 | 0.933 | 0.0102 | 0.0479 | 0.0948 | 0.1900 | 0.949 |
| min | 0.811 | 0.911 | 0.932 | 0.952 | 0.0107 | 0.0514 | 0.1019 | 0.1985 | 0.967 |
| weighted 0.9/0.1 | 0.814 | 0.913 | 0.933 | 0.951 | 0.0099 | 0.0487 | 0.0958 | 0.1926 | 0.963 |

| paired difference | TPR 1% | TPR 5% | TPR 10% | TPR 20% | AUROC |
|---|---|---|---|---|---|
| min minus PSBD-TM | +0.040 [-0.021, +0.104] | +0.028 [-0.005, +0.074] | +0.022 [-0.008, +0.067] | +0.018 [-0.008, +0.056] | +0.018 [+0.003, +0.044] |
| weighted minus PSBD-TM | +0.043 [+0.011, +0.085] | +0.030 [+0.005, +0.067] | +0.022 [+0.002, +0.058] | +0.018 [+0.001, +0.048] | +0.014 [+0.002, +0.036] |
| weighted minus min | +0.003 [-0.039, +0.041] | +0.002 [-0.009, +0.013] | +0.001 [-0.010, +0.013] | -0.001 [-0.012, +0.013] | -0.004 [-0.009, -0.000] |

| models losing more than 0.05 against PSBD-TM | at 1% | at 5% | at 10% | at 20% | AUROC |
|---|---|---|---|---|---|
| min | 6 (worst -0.402, `vit_cifar100_badnet_a2o_0_01`) | 4 (worst -0.126, `vit_gtsrb_tact_0_05`) | 2 (worst -0.231, `vit_gtsrb_tact_0_05`) | 1 (worst -0.223, `vit_gtsrb_tact_0_05`) | 0 (worst -0.012, `vit_tiny_badnet_a2o_0_01`) |
| weighted 0.9/0.1 | 2 (worst -0.190, `vit_tiny_badnet_a2o_0_01`) | 0 (worst -0.026, `vit_gtsrb_tact_0_05`) | 0 (worst -0.036, `vit_gtsrb_tact_0_05`) | 0 (worst -0.017, `vit_gtsrb_tact_0_05`) | 0 (worst -0.003, `vit_tiny_badnet_a2o_0_05`) |

| attack (models) | PSBD-TM TPR 1 / 5 / 10% | min TPR 1 / 5 / 10% | weighted TPR 1 / 5 / 10% | AUROC PSBD-TM / min / weighted | weighted minus min, TPR 1% | losing at 1%, min / weighted |
|---|---|---|---|---|---|---|
| badnet_a2o (9) | 0.797 / 0.994 / 0.998 | 0.697 / 0.973 / 0.996 | 0.778 / 0.993 / 0.998 | 0.993 / 0.989 / 0.993 | +0.081 [+0.015, +0.162] | 4 / 2 |
| tact (2) | 0.015 / 0.115 / 0.251 | 0.007 / 0.071 / 0.179 | 0.013 / 0.102 / 0.237 | 0.604 / 0.646 / 0.606 | +0.006 (n<3) | 0 / 0 |
| blend (8) | 0.896 / 0.967 / 0.985 | 0.903 / 0.975 / 0.988 | 0.937 / 0.982 / 0.994 | 0.993 / 0.995 / 0.996 | +0.034 [-0.008, +0.113] | 1 / 0 |
| lf (9) | 0.900 / 0.959 / 0.967 | 0.931 / 0.963 / 0.971 | 0.934 / 0.963 / 0.969 | 0.977 / 0.984 / 0.980 | +0.002 [-0.008, +0.015] | 1 / 0 |
| bpp (9) | 0.800 / 0.893 / 0.931 | 0.885 / 0.929 / 0.949 | 0.856 / 0.919 / 0.944 | 0.969 / 0.977 / 0.975 | -0.029 [-0.085, +0.003] | 0 / 0 |
| wanet (3) | 0.372 / 0.572 / 0.647 | 0.861 / 0.913 / 0.929 | 0.636 / 0.864 / 0.893 | 0.779 / 0.958 / 0.936 | -0.225 [-0.614, -0.019] | 0 / 0 |
| lc (1) | 0.833 / 0.921 / 0.950 | 0.798 / 0.899 / 0.938 | 0.831 / 0.920 / 0.951 | 0.969 / 0.973 / 0.972 | +0.033 (n<3) | 0 / 0 |

| dataset (models) | PSBD-TM TPR 1 / 5 / 10% | min TPR 1 / 5 / 10% | weighted TPR 1 / 5 / 10% | AUROC PSBD-TM / min / weighted | weighted minus min, TPR 1% | losing at 1%, min / weighted |
|---|---|---|---|---|---|---|
| CIFAR-10 (1) | 0.022 / 0.040 / 0.087 | 0.695 / 0.847 / 0.886 | 0.082 / 0.701 / 0.780 | 0.459 / 0.938 / 0.890 | -0.614 (n<3) | 0 / 0 |
| CIFAR-100 (12) | 0.790 / 0.935 / 0.964 | 0.785 / 0.955 / 0.974 | 0.833 / 0.958 / 0.975 | 0.979 / 0.984 / 0.983 | +0.048 [-0.035, +0.139] | 4 / 1 |
| GTSRB (14) | 0.787 / 0.828 / 0.860 | 0.783 / 0.821 / 0.852 | 0.787 / 0.826 / 0.861 | 0.930 / 0.937 / 0.932 | +0.005 [-0.001, +0.011] | 0 / 0 |
| Tiny ImageNet (14) | 0.793 / 0.955 / 0.975 | 0.869 / 0.970 / 0.980 | 0.877 / 0.977 / 0.981 | 0.977 / 0.985 / 0.983 | +0.008 [-0.013, +0.040] | 2 / 1 |

### Swin-S panel, partner blocks 17 to 24, adaptive rate, 39 models

Not scored, 26 models: `swin_cifar10_adaptive_blend_0_05`, `swin_cifar10_adaptive_blend_0_1`, `swin_cifar10_badnet_a2o_0_05`, `swin_cifar10_badnet_a2o_0_1`, `swin_cifar10_blend_0_01`, `swin_cifar10_blend_0_05`, `swin_cifar10_blend_0_1`, `swin_cifar10_bpp_0_01`, `swin_cifar10_bpp_0_05`, `swin_cifar10_bpp_0_1`, `swin_cifar10_lf_0_01`, `swin_cifar10_lf_0_1`, `swin_cifar10_tact_0_01`, `swin_cifar10_tact_0_05`, `swin_gtsrb_badnet_a2o_0_01`, `swin_gtsrb_badnet_a2o_0_05`, `swin_gtsrb_badnet_a2o_0_1`, `swin_gtsrb_blend_0_01`, `swin_gtsrb_blend_0_05`, `swin_gtsrb_blend_0_1`, `swin_gtsrb_bpp_0_01`, `swin_gtsrb_bpp_0_05`, `swin_gtsrb_bpp_0_1`, `swin_gtsrb_lf_0_01`, `swin_gtsrb_lf_0_05`, `swin_gtsrb_lf_0_1`.

| rule | TPR 1% | TPR 5% | TPR 10% | TPR 20% | FPR 1% | FPR 5% | FPR 10% | FPR 20% | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| PSBD-TM alone | 0.808 | 0.913 | 0.955 | 0.976 | 0.0086 | 0.0453 | 0.0940 | 0.1942 | 0.978 |
| min | 0.922 | 0.961 | 0.973 | 0.981 | 0.0083 | 0.0458 | 0.0927 | 0.1922 | 0.986 |
| weighted 0.9/0.1 | 0.852 | 0.955 | 0.970 | 0.978 | 0.0083 | 0.0454 | 0.0937 | 0.1945 | 0.982 |

| paired difference | TPR 1% | TPR 5% | TPR 10% | TPR 20% | AUROC |
|---|---|---|---|---|---|
| min minus PSBD-TM | +0.114 [+0.036, +0.204] | +0.048 [+0.009, +0.094] | +0.018 [+0.004, +0.037] | +0.005 [+0.001, +0.010] | +0.008 [+0.004, +0.014] |
| weighted minus PSBD-TM | +0.045 [+0.002, +0.104] | +0.041 [+0.010, +0.081] | +0.015 [+0.002, +0.033] | +0.003 [+0.001, +0.005] | +0.004 [+0.001, +0.007] |
| weighted minus min | -0.069 [-0.141, -0.012] | -0.006 [-0.017, +0.002] | -0.003 [-0.007, +0.001] | -0.003 [-0.005, +0.001] | -0.004 [-0.007, -0.002] |

| models losing more than 0.05 against PSBD-TM | at 1% | at 5% | at 10% | at 20% | AUROC |
|---|---|---|---|---|---|
| min | 1 (worst -0.057, `swin_tiny_badnet_a2o_0_05`) | 1 (worst -0.099, `swin_cifar100_lf_0_01`) | 0 (worst -0.043, `swin_cifar10_lf_0_05`) | 0 (worst -0.038, `swin_cifar10_lf_0_05`) | 0 (worst -0.009, `swin_cifar100_lf_0_01`) |
| weighted 0.9/0.1 | 0 (worst -0.013, `swin_cifar100_lf_0_01`) | 1 (worst -0.051, `swin_cifar100_lf_0_01`) | 0 (worst -0.013, `swin_cifar100_lf_0_01`) | 0 (worst -0.002, `swin_cifar10_lf_0_05`) | 0 (worst -0.006, `swin_cifar100_lf_0_01`) |

| attack (models) | PSBD-TM TPR 1 / 5 / 10% | min TPR 1 / 5 / 10% | weighted TPR 1 / 5 / 10% | AUROC PSBD-TM / min / weighted | weighted minus min, TPR 1% | losing at 1%, min / weighted |
|---|---|---|---|---|---|---|
| badnet_a2o (7) | 0.982 / 0.994 / 0.999 | 0.962 / 0.991 / 0.999 | 0.980 / 0.994 / 0.999 | 0.999 / 0.998 / 0.998 | +0.018 [+0.008, +0.031] | 1 / 0 |
| blend (6) | 0.996 / 0.999 / 1.000 | 0.991 / 0.999 / 1.000 | 0.995 / 0.999 / 1.000 | 1.000 / 0.998 / 0.999 | +0.004 [+0.000, +0.010] | 0 / 0 |
| lf (7) | 0.791 / 0.905 / 0.936 | 0.794 / 0.887 / 0.936 | 0.793 / 0.898 / 0.934 | 0.973 / 0.975 / 0.972 | -0.001 [-0.021, +0.017] | 0 / 0 |
| bpp (6) | 0.982 / 0.994 / 0.996 | 0.978 / 0.994 / 0.997 | 0.982 / 0.994 / 0.997 | 0.998 / 0.998 / 0.998 | +0.003 [-0.000, +0.009] | 0 / 0 |
| wanet (8) | 0.333 / 0.704 / 0.872 | 0.889 / 0.946 / 0.951 | 0.532 / 0.908 / 0.943 | 0.944 / 0.970 / 0.959 | -0.357 [-0.598, -0.117] | 0 / 0 |
| adaptive_blend (5) | 0.912 / 0.948 / 0.949 | 0.946 / 0.962 / 0.968 | 0.943 / 0.952 / 0.959 | 0.960 / 0.984 / 0.970 | -0.003 [-0.019, +0.011] | 0 / 0 |

| dataset (models) | PSBD-TM TPR 1 / 5 / 10% | min TPR 1 / 5 / 10% | weighted TPR 1 / 5 / 10% | AUROC PSBD-TM / min / weighted | weighted minus min, TPR 1% | losing at 1%, min / weighted |
|---|---|---|---|---|---|---|
| CIFAR-10 (4) | 0.471 / 0.686 / 0.833 | 0.810 / 0.899 / 0.926 | 0.652 / 0.881 / 0.930 | 0.966 / 0.977 / 0.975 | -0.158 [-0.464, +0.025] | 0 / 0 |
| CIFAR-100 (16) | 0.816 / 0.938 / 0.966 | 0.902 / 0.952 / 0.974 | 0.859 / 0.954 / 0.968 | 0.978 / 0.985 / 0.980 | -0.043 [-0.141, +0.011] | 0 / 0 |
| GTSRB (4) | 0.459 / 0.746 / 0.894 | 0.874 / 0.941 / 0.947 | 0.523 / 0.892 / 0.937 | 0.932 / 0.968 / 0.952 | -0.350 [-0.685, -0.016] | 0 / 0 |
| Tiny ImageNet (15) | 0.982 / 0.992 / 0.992 | 0.985 / 0.992 / 0.993 | 0.986 / 0.992 / 0.992 | 0.994 / 0.995 / 0.994 | +0.001 [-0.006, +0.010] | 1 / 0 |

### Swin-S panel, partner blocks 17 to 24, with the nearest-rate models, 63 models

Not scored, 2 models: `swin_cifar10_tact_0_01`, `swin_cifar10_tact_0_05`.

| rule | TPR 1% | TPR 5% | TPR 10% | TPR 20% | FPR 1% | FPR 5% | FPR 10% | FPR 20% | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| PSBD-TM alone | 0.824 | 0.905 | 0.939 | 0.961 | 0.0085 | 0.0445 | 0.0922 | 0.1910 | 0.973 |
| min | 0.908 | 0.942 | 0.954 | 0.965 | 0.0085 | 0.0452 | 0.0928 | 0.1921 | 0.977 |
| weighted 0.9/0.1 | 0.859 | 0.937 | 0.952 | 0.964 | 0.0082 | 0.0446 | 0.0922 | 0.1908 | 0.976 |

| paired difference | TPR 1% | TPR 5% | TPR 10% | TPR 20% | AUROC |
|---|---|---|---|---|---|
| min minus PSBD-TM | +0.084 [+0.031, +0.142] | +0.037 [+0.011, +0.068] | +0.015 [+0.004, +0.028] | +0.004 [-0.002, +0.009] | +0.005 [+0.000, +0.009] |
| weighted minus PSBD-TM | +0.035 [+0.006, +0.072] | +0.033 [+0.011, +0.058] | +0.013 [+0.004, +0.025] | +0.003 [+0.001, +0.006] | +0.003 [+0.001, +0.006] |
| weighted minus min | -0.049 [-0.095, -0.012] | -0.004 [-0.011, +0.001] | -0.002 [-0.005, +0.002] | -0.001 [-0.004, +0.004] | -0.001 [-0.004, +0.002] |

| models losing more than 0.05 against PSBD-TM | at 1% | at 5% | at 10% | at 20% | AUROC |
|---|---|---|---|---|---|
| min | 1 (worst -0.057, `swin_tiny_badnet_a2o_0_05`) | 1 (worst -0.099, `swin_cifar100_lf_0_01`) | 1 (worst -0.110, `swin_cifar10_bpp_0_01`) | 1 (worst -0.107, `swin_cifar10_bpp_0_01`) | 2 (worst -0.062, `swin_cifar10_adaptive_blend_0_05`) |
| weighted 0.9/0.1 | 0 (worst -0.013, `swin_cifar100_lf_0_01`) | 1 (worst -0.051, `swin_cifar100_lf_0_01`) | 0 (worst -0.027, `swin_cifar10_bpp_0_01`) | 0 (worst -0.005, `swin_cifar10_adaptive_blend_0_05`) | 0 (worst -0.006, `swin_cifar100_lf_0_01`) |

| attack (models) | PSBD-TM TPR 1 / 5 / 10% | min TPR 1 / 5 / 10% | weighted TPR 1 / 5 / 10% | AUROC PSBD-TM / min / weighted | weighted minus min, TPR 1% | losing at 1%, min / weighted |
|---|---|---|---|---|---|---|
| badnet_a2o (12) | 0.989 / 0.996 / 1.000 | 0.976 / 0.995 / 0.999 | 0.988 / 0.996 / 1.000 | 0.999 / 0.999 / 0.999 | +0.011 [+0.004, +0.021] | 1 / 0 |
| blend (12) | 0.965 / 0.981 / 0.986 | 0.973 / 0.993 / 0.998 | 0.972 / 0.992 / 0.996 | 0.996 / 0.998 / 0.998 | -0.001 [-0.014, +0.008] | 0 / 0 |
| lf (12) | 0.796 / 0.907 / 0.942 | 0.851 / 0.921 / 0.955 | 0.826 / 0.925 / 0.951 | 0.978 / 0.982 / 0.980 | -0.025 [-0.072, +0.006] | 0 / 0 |
| bpp (12) | 0.893 / 0.918 / 0.938 | 0.898 / 0.920 / 0.931 | 0.894 / 0.923 / 0.938 | 0.980 / 0.976 / 0.980 | -0.003 [-0.013, +0.005] | 0 / 0 |
| wanet (8) | 0.333 / 0.704 / 0.872 | 0.889 / 0.946 / 0.951 | 0.532 / 0.908 / 0.943 | 0.944 / 0.970 / 0.959 | -0.357 [-0.598, -0.117] | 0 / 0 |
| adaptive_blend (7) | 0.795 / 0.821 / 0.824 | 0.819 / 0.833 / 0.841 | 0.817 / 0.825 / 0.831 | 0.899 / 0.907 / 0.906 | -0.002 [-0.014, +0.007] | 0 / 0 |

| dataset (models) | PSBD-TM TPR 1 / 5 / 10% | min TPR 1 / 5 / 10% | weighted TPR 1 / 5 / 10% | AUROC PSBD-TM / min / weighted | weighted minus min, TPR 1% | losing at 1%, min / weighted |
|---|---|---|---|---|---|---|
| CIFAR-10 (16) | 0.725 / 0.800 / 0.850 | 0.806 / 0.852 / 0.870 | 0.771 / 0.850 / 0.874 | 0.945 / 0.940 / 0.946 | -0.036 [-0.119, +0.010] | 0 / 0 |
| CIFAR-100 (16) | 0.816 / 0.938 / 0.966 | 0.902 / 0.952 / 0.974 | 0.859 / 0.954 / 0.968 | 0.978 / 0.985 / 0.980 | -0.043 [-0.141, +0.011] | 0 / 0 |
| GTSRB (16) | 0.784 / 0.895 / 0.949 | 0.944 / 0.974 / 0.981 | 0.829 / 0.958 / 0.976 | 0.976 / 0.990 / 0.985 | -0.115 [-0.237, -0.020] | 0 / 0 |
| Tiny ImageNet (15) | 0.982 / 0.992 / 0.992 | 0.985 / 0.992 / 0.993 | 0.986 / 0.992 / 0.992 | 0.994 / 0.995 / 0.994 | +0.001 [-0.006, +0.010] | 1 / 0 |

### BackdoorBench ViT-B/16, partner blocks 5 to 8, 10 models

| rule | TPR 1% | TPR 5% | TPR 10% | TPR 20% | FPR 1% | FPR 5% | FPR 10% | FPR 20% | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| PSBD-TM alone | 0.667 | 0.929 | 0.971 | 0.982 | 0.0081 | 0.0322 | 0.0678 | 0.1724 | 0.980 |
| min | 0.600 | 0.864 | 0.953 | 0.983 | 0.0087 | 0.0369 | 0.0810 | 0.1742 | 0.979 |
| weighted 0.9/0.1 | 0.689 | 0.936 | 0.966 | 0.983 | 0.0082 | 0.0322 | 0.0677 | 0.1713 | 0.981 |

| paired difference | TPR 1% | TPR 5% | TPR 10% | TPR 20% | AUROC |
|---|---|---|---|---|---|
| min minus PSBD-TM | -0.068 [-0.214, +0.061] | -0.066 [-0.159, +0.009] | -0.018 [-0.046, +0.004] | +0.001 [-0.001, +0.003] | -0.001 [-0.007, +0.004] |
| weighted minus PSBD-TM | +0.022 [-0.014, +0.066] | +0.007 [-0.005, +0.020] | -0.005 [-0.022, +0.006] | +0.001 [+0.000, +0.003] | +0.002 [-0.000, +0.004] |
| weighted minus min | +0.090 [-0.016, +0.210] | +0.072 [+0.006, +0.159] | +0.014 [+0.002, +0.031] | +0.000 [-0.001, +0.001] | +0.003 [-0.001, +0.008] |

| models losing more than 0.05 against PSBD-TM | at 1% | at 5% | at 10% | at 20% | AUROC |
|---|---|---|---|---|---|
| min | 3 (worst -0.514, `bb_cifar10_blind_0_1`) | 3 (worst -0.386, `bb_gtsrb_ssba_0_1`) | 2 (worst -0.110, `bb_gtsrb_trojannn_0_1`) | 0 (worst -0.004, `bb_gtsrb_trojannn_0_1`) | 0 (worst -0.020, `bb_cifar10_blind_0_1`) |
| weighted 0.9/0.1 | 1 (worst -0.076, `bb_cifar10_blind_0_1`) | 0 (worst -0.018, `bb_cifar10_blind_0_1`) | 1 (worst -0.077, `bb_gtsrb_trojannn_0_1`) | 0 (worst -0.000, `bb_gtsrb_trojannn_0_1`) | 0 (worst -0.002, `bb_cifar10_blind_0_1`) |

| attack (models) | PSBD-TM TPR 1 / 5 / 10% | min TPR 1 / 5 / 10% | weighted TPR 1 / 5 / 10% | AUROC PSBD-TM / min / weighted | weighted minus min, TPR 1% | losing at 1%, min / weighted |
|---|---|---|---|---|---|---|
| blind (1) | 0.515 / 0.917 / 0.998 | 0.001 / 0.809 / 0.914 | 0.439 / 0.899 / 0.992 | 0.998 / 0.978 / 0.996 | +0.438 (n<3) | 1 / 1 |
| inputaware (1) | 0.885 / 0.942 / 0.964 | 0.964 / 0.974 / 0.977 | 0.953 / 0.971 / 0.976 | 0.980 / 0.989 / 0.985 | -0.011 (n<3) | 0 / 0 |
| ssba (3) | 0.646 / 0.925 / 0.950 | 0.547 / 0.797 / 0.943 | 0.648 / 0.923 / 0.950 | 0.966 / 0.966 / 0.967 | +0.100 [-0.030, +0.331] | 1 / 0 |
| trojannn (5) | 0.667 / 0.932 / 0.980 | 0.678 / 0.893 / 0.961 | 0.712 / 0.944 / 0.969 | 0.984 / 0.984 / 0.987 | +0.034 [-0.057, +0.180] | 1 / 0 |

| dataset (models) | PSBD-TM TPR 1 / 5 / 10% | min TPR 1 / 5 / 10% | weighted TPR 1 / 5 / 10% | AUROC PSBD-TM / min / weighted | weighted minus min, TPR 1% | losing at 1%, min / weighted |
|---|---|---|---|---|---|---|
| CIFAR-10 (4) | 0.665 / 0.964 / 0.993 | 0.386 / 0.930 / 0.971 | 0.649 / 0.959 / 0.992 | 0.993 / 0.988 / 0.993 | +0.263 [+0.063, +0.406] | 3 / 1 |
| GTSRB (3) | 0.459 / 0.846 / 0.923 | 0.494 / 0.643 / 0.884 | 0.481 / 0.849 / 0.903 | 0.951 / 0.948 / 0.953 | -0.013 [-0.030, +0.002] | 0 / 0 |
| Tiny ImageNet (3) | 0.878 / 0.967 / 0.988 | 0.989 / 0.996 / 0.997 | 0.951 / 0.993 / 0.996 | 0.991 / 0.997 / 0.995 | -0.038 [-0.085, +0.000] | 0 / 0 |

### training-set setting, partner blocks 5 to 8, 6 models

| rule | TPR 1% | TPR 5% | TPR 10% | TPR 20% | FPR 1% | FPR 5% | FPR 10% | FPR 20% | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| PSBD-TM alone | 0.535 | 0.710 | 0.788 | 0.853 | 0.0100 | 0.0500 | 0.1000 | 0.2000 | 0.891 |
| min | 0.731 | 0.889 | 0.948 | 0.989 | 0.0090 | 0.0488 | 0.0996 | 0.1996 | 0.981 |
| weighted 0.9/0.1 | 0.592 | 0.823 | 0.902 | 0.959 | 0.0097 | 0.0497 | 0.0998 | 0.1996 | 0.964 |

| paired difference | TPR 1% | TPR 5% | TPR 10% | TPR 20% | AUROC |
|---|---|---|---|---|---|
| min minus PSBD-TM | +0.196 [+0.019, +0.402] | +0.179 [-0.014, +0.452] | +0.160 [-0.033, +0.457] | +0.136 [-0.000, +0.367] | +0.090 [-0.000, +0.245] |
| weighted minus PSBD-TM | +0.058 [-0.006, +0.137] | +0.112 [+0.000, +0.308] | +0.115 [+0.000, +0.322] | +0.106 [+0.001, +0.299] | +0.074 [+0.001, +0.210] |
| weighted minus min | -0.139 [-0.288, +0.001] | -0.067 [-0.157, +0.020] | -0.046 [-0.135, +0.031] | -0.030 [-0.068, +0.001] | -0.016 [-0.036, +0.001] |

| models losing more than 0.05 against PSBD-TM | at 1% | at 5% | at 10% | at 20% | AUROC |
|---|---|---|---|---|---|
| min | 1 (worst -0.067, `vit_cifar10_badnet_a2o_0_1`) | 1 (worst -0.090, `vit_cifar10_badnet_a2o_0_1`) | 1 (worst -0.101, `vit_cifar10_badnet_a2o_0_1`) | 0 (worst -0.002, `vit_cifar10_badnet_a2o_0_1`) | 0 (worst -0.014, `vit_cifar10_badnet_a2o_0_1`) |
| weighted 0.9/0.1 | 0 (worst -0.011, `vit_cifar10_blend_0_1`) | 0 (worst -0.005, `vit_cifar10_badnet_a2o_0_1`) | 0 (worst -0.007, `vit_cifar10_badnet_a2o_0_1`) | 0 (worst +0.000, `vit_cifar10_badnet_a2o_0_1`) | 0 (worst -0.001, `vit_tiny_blend_0_1`) |

| attack (models) | PSBD-TM TPR 1 / 5 / 10% | min TPR 1 / 5 / 10% | weighted TPR 1 / 5 / 10% | AUROC PSBD-TM / min / weighted | weighted minus min, TPR 1% | losing at 1%, min / weighted |
|---|---|---|---|---|---|---|
| badnet_a2o (2) | 0.661 / 0.879 / 0.970 | 0.734 / 0.834 / 0.919 | 0.734 / 0.877 / 0.966 | 0.983 / 0.979 / 0.985 | -0.000 (n<3) | 1 / 0 |
| blend (2) | 0.787 / 0.852 / 0.879 | 0.805 / 0.922 / 0.956 | 0.781 / 0.857 / 0.901 | 0.956 / 0.984 / 0.967 | -0.024 (n<3) | 0 / 0 |
| wanet (2) | 0.156 / 0.400 / 0.515 | 0.654 / 0.911 / 0.969 | 0.262 / 0.734 / 0.839 | 0.734 / 0.980 / 0.942 | -0.392 (n<3) | 0 / 0 |

| dataset (models) | PSBD-TM TPR 1 / 5 / 10% | min TPR 1 / 5 / 10% | weighted TPR 1 / 5 / 10% | AUROC PSBD-TM / min / weighted | weighted minus min, TPR 1% | losing at 1%, min / weighted |
|---|---|---|---|---|---|---|
| CIFAR-10 (3) | 0.390 / 0.502 / 0.598 | 0.591 / 0.798 / 0.904 | 0.454 / 0.700 / 0.820 | 0.797 / 0.967 / 0.941 | -0.137 [-0.422, +0.059] | 1 / 0 |
| Tiny ImageNet (3) | 0.680 / 0.918 / 0.978 | 0.871 / 0.980 / 0.993 | 0.731 / 0.945 / 0.985 | 0.985 / 0.994 / 0.988 | -0.140 [-0.362, +0.002] | 0 / 0 |

## Share sweep

![share sweep](../../results/_experiments/fusion_weighting/share_sweep.png)

`share_sweep.png` (numbers in `share_sweep.json`) draws the mean TPR at 1%, 5% and 10% FPR and AUROC against the TM share w on a 0.01 grid, PSBD-TM alone dotted, with the count of models losing more than 0.05 below it. The dashed line marks w = 0.9.

The sweep shows no single plateau across sets, and 0.9 is a sharp optimum on none of them. On the ViT panel mean TPR at 1% FPR moves by only 0.017 across every share from 0.50 to 0.99, peaks at 0.64 and sits 0.011 below that peak at 0.90, while the models losing more than 0.05 at 1% fall from 10 at 0.50 to 1 at 0.90 and 0 at 0.95. On BackdoorBench the mean rises to a plateau from 0.86 to 0.99 that contains 0.90. On Swin-S the best share is the plain minimum (0.51) and the mean falls as w grows, so 0.90 gives up 0.070 there. The training-set setting peaks at 0.52 and 0.90 gives up 0.141. On the ViT panel and BackdoorBench the count of losing models moves more with w than the mean does.

The per attack tables above give the flat ViT mean its reading. Raising w from 0.5 gives back the partner's WaNet gain and returns to PSBD-TM the budget it needs on BadNets, Blend, LF and BPP, and the 2 cancel in the mean while the losses fall. BackdoorBench at 0.600 under min against 0.689 at 0.90 is where the cost of the plain minimum is largest.

### Sweep, ViT-B/16 panel, partner blocks 5 to 8, 56 models

Best share on the 0.01 grid 0.64 (mean TPR at 1% 0.759), shares within 0.01 of it from 0.56 to 0.89 with gaps, 0.90 at 0.748.

| TM share | TPR 1% | TPR 5% | TPR 10% | AUROC | TPR 1% minus PSBD-TM | AUROC minus PSBD-TM | losing at 1% / 5% / 10% |
|---|---|---|---|---|---|---|---|
| PSBD-TM alone | 0.735 | 0.841 | 0.872 | 0.951 | | | |
| 0.50 | 0.742 | 0.857 | 0.889 | 0.970 | +0.007 [-0.037, +0.049] | +0.020 [+0.001, +0.047] | 10 / 5 / 4 |
| 0.60 | 0.758 | 0.859 | 0.890 | 0.971 | +0.023 [-0.012, +0.060] | +0.021 [+0.002, +0.047] | 5 / 4 / 2 |
| 0.70 | 0.754 | 0.858 | 0.891 | 0.971 | +0.018 [-0.012, +0.049] | +0.020 [+0.003, +0.045] | 3 / 2 / 2 |
| 0.80 | 0.750 | 0.857 | 0.889 | 0.968 | +0.015 [-0.008, +0.039] | +0.018 [+0.002, +0.041] | 2 / 1 / 1 |
| 0.85 | 0.750 | 0.856 | 0.888 | 0.967 | +0.015 [-0.004, +0.037] | +0.016 [+0.002, +0.037] | 2 / 0 / 1 |
| 0.90 | 0.748 | 0.854 | 0.886 | 0.964 | +0.013 [+0.000, +0.027] | +0.013 [+0.001, +0.032] | 1 / 0 / 0 |
| 0.95 | 0.746 | 0.850 | 0.883 | 0.960 | +0.011 [+0.001, +0.023] | +0.009 [+0.001, +0.024] | 0 / 0 / 0 |
| 0.99 | 0.746 | 0.843 | 0.874 | 0.955 | +0.011 [+0.001, +0.023] | +0.004 [+0.000, +0.011] | 0 / 0 / 0 |

### Sweep, ViT-B/16 panel, partner blocks 9 to 12 (pre-registered pairing), 41 models

Best share on the 0.01 grid 0.74 (mean TPR at 1% 0.823), shares within 0.01 of it from 0.53 to 0.94, 0.90 at 0.814.

| TM share | TPR 1% | TPR 5% | TPR 10% | AUROC | TPR 1% minus PSBD-TM | AUROC minus PSBD-TM | losing at 1% / 5% / 10% |
|---|---|---|---|---|---|---|---|
| PSBD-TM alone | 0.771 | 0.883 | 0.911 | 0.949 | | | |
| 0.50 | 0.811 | 0.911 | 0.932 | 0.967 | +0.040 [-0.021, +0.104] | +0.018 [+0.003, +0.044] | 6 / 4 / 2 |
| 0.60 | 0.818 | 0.915 | 0.934 | 0.967 | +0.047 [-0.008, +0.107] | +0.018 [+0.004, +0.043] | 4 / 2 / 1 |
| 0.70 | 0.821 | 0.916 | 0.935 | 0.966 | +0.050 [+0.005, +0.103] | +0.017 [+0.003, +0.041] | 4 / 1 / 1 |
| 0.80 | 0.821 | 0.916 | 0.935 | 0.965 | +0.050 [+0.011, +0.098] | +0.016 [+0.003, +0.039] | 2 / 0 / 1 |
| 0.85 | 0.818 | 0.915 | 0.935 | 0.964 | +0.047 [+0.011, +0.094] | +0.015 [+0.003, +0.038] | 2 / 0 / 0 |
| 0.90 | 0.814 | 0.913 | 0.933 | 0.963 | +0.043 [+0.011, +0.085] | +0.014 [+0.002, +0.036] | 2 / 0 / 0 |
| 0.95 | 0.805 | 0.904 | 0.931 | 0.961 | +0.034 [+0.007, +0.071] | +0.012 [+0.002, +0.032] | 1 / 0 / 0 |
| 0.99 | 0.802 | 0.892 | 0.916 | 0.956 | +0.031 [+0.005, +0.069] | +0.007 [+0.001, +0.019] | 1 / 0 / 0 |

### Sweep, Swin-S panel, partner blocks 17 to 24, adaptive rate, 39 models

Best share on the 0.01 grid 0.51 (mean TPR at 1% 0.923), shares within 0.01 of it from 0.50 to 0.61, 0.90 at 0.852.

| TM share | TPR 1% | TPR 5% | TPR 10% | AUROC | TPR 1% minus PSBD-TM | AUROC minus PSBD-TM | losing at 1% / 5% / 10% |
|---|---|---|---|---|---|---|---|
| PSBD-TM alone | 0.808 | 0.913 | 0.955 | 0.978 | | | |
| 0.50 | 0.922 | 0.961 | 0.973 | 0.986 | +0.114 [+0.036, +0.204] | +0.008 [+0.004, +0.014] | 1 / 1 / 0 |
| 0.60 | 0.916 | 0.962 | 0.973 | 0.986 | +0.108 [+0.037, +0.193] | +0.008 [+0.003, +0.013] | 0 / 1 / 0 |
| 0.70 | 0.903 | 0.961 | 0.972 | 0.985 | +0.096 [+0.033, +0.171] | +0.007 [+0.003, +0.011] | 0 / 1 / 0 |
| 0.80 | 0.873 | 0.961 | 0.972 | 0.984 | +0.066 [+0.014, +0.133] | +0.006 [+0.002, +0.010] | 0 / 1 / 0 |
| 0.85 | 0.866 | 0.959 | 0.971 | 0.983 | +0.058 [+0.007, +0.123] | +0.005 [+0.002, +0.009] | 0 / 1 / 0 |
| 0.90 | 0.852 | 0.955 | 0.970 | 0.982 | +0.045 [+0.002, +0.104] | +0.004 [+0.001, +0.007] | 0 / 1 / 0 |
| 0.95 | 0.834 | 0.939 | 0.968 | 0.980 | +0.027 [+0.000, +0.070] | +0.002 [+0.000, +0.005] | 0 / 0 / 0 |
| 0.99 | 0.828 | 0.921 | 0.962 | 0.979 | +0.020 [-0.001, +0.059] | +0.001 [-0.000, +0.002] | 0 / 0 / 0 |

### Sweep, Swin-S panel, partner blocks 17 to 24, with the nearest-rate models, 63 models

Best share on the 0.01 grid 0.52 (mean TPR at 1% 0.909), shares within 0.01 of it from 0.50 to 0.66, 0.90 at 0.859.

| TM share | TPR 1% | TPR 5% | TPR 10% | AUROC | TPR 1% minus PSBD-TM | AUROC minus PSBD-TM | losing at 1% / 5% / 10% |
|---|---|---|---|---|---|---|---|
| PSBD-TM alone | 0.824 | 0.905 | 0.939 | 0.973 | | | |
| 0.50 | 0.908 | 0.942 | 0.954 | 0.977 | +0.084 [+0.031, +0.142] | +0.005 [+0.000, +0.009] | 1 / 1 / 1 |
| 0.60 | 0.905 | 0.943 | 0.953 | 0.978 | +0.081 [+0.031, +0.136] | +0.005 [+0.002, +0.009] | 0 / 1 / 1 |
| 0.70 | 0.897 | 0.942 | 0.953 | 0.978 | +0.072 [+0.028, +0.123] | +0.005 [+0.002, +0.008] | 0 / 1 / 1 |
| 0.80 | 0.877 | 0.942 | 0.953 | 0.977 | +0.053 [+0.017, +0.097] | +0.004 [+0.002, +0.007] | 0 / 1 / 1 |
| 0.85 | 0.870 | 0.940 | 0.952 | 0.977 | +0.046 [+0.012, +0.087] | +0.004 [+0.002, +0.007] | 0 / 1 / 0 |
| 0.90 | 0.859 | 0.937 | 0.952 | 0.976 | +0.035 [+0.006, +0.072] | +0.003 [+0.001, +0.006] | 0 / 1 / 0 |
| 0.95 | 0.844 | 0.926 | 0.950 | 0.975 | +0.019 [+0.002, +0.046] | +0.002 [+0.001, +0.004] | 0 / 0 / 0 |
| 0.99 | 0.840 | 0.911 | 0.945 | 0.973 | +0.015 [+0.001, +0.041] | +0.001 [-0.000, +0.002] | 0 / 0 / 0 |

### Sweep, BackdoorBench ViT-B/16, partner blocks 5 to 8, 10 models

Best share on the 0.01 grid 0.86 (mean TPR at 1% 0.689), shares within 0.01 of it from 0.86 to 0.99, 0.90 at 0.689.

| TM share | TPR 1% | TPR 5% | TPR 10% | AUROC | TPR 1% minus PSBD-TM | AUROC minus PSBD-TM | losing at 1% / 5% / 10% |
|---|---|---|---|---|---|---|---|
| PSBD-TM alone | 0.667 | 0.929 | 0.971 | 0.980 | | | |
| 0.50 | 0.600 | 0.864 | 0.953 | 0.979 | -0.068 [-0.214, +0.061] | -0.001 [-0.007, +0.004] | 3 / 3 / 2 |
| 0.60 | 0.608 | 0.892 | 0.958 | 0.980 | -0.059 [-0.197, +0.058] | +0.000 [-0.004, +0.004] | 3 / 3 / 2 |
| 0.70 | 0.628 | 0.901 | 0.962 | 0.981 | -0.040 [-0.173, +0.065] | +0.001 [-0.002, +0.004] | 3 / 3 / 1 |
| 0.80 | 0.652 | 0.908 | 0.965 | 0.981 | -0.015 [-0.106, +0.065] | +0.002 [-0.001, +0.004] | 2 / 1 / 1 |
| 0.85 | 0.668 | 0.930 | 0.966 | 0.981 | +0.001 [-0.061, +0.064] | +0.002 [-0.000, +0.004] | 2 / 1 / 1 |
| 0.90 | 0.689 | 0.936 | 0.966 | 0.981 | +0.022 [-0.014, +0.066] | +0.002 [-0.000, +0.004] | 1 / 0 / 1 |
| 0.95 | 0.684 | 0.936 | 0.967 | 0.981 | +0.017 [-0.012, +0.055] | +0.002 [+0.000, +0.004] | 1 / 0 / 1 |
| 0.99 | 0.683 | 0.933 | 0.968 | 0.981 | +0.016 [-0.013, +0.054] | +0.001 [+0.000, +0.003] | 1 / 0 / 0 |

### Sweep, training-set setting, partner blocks 5 to 8, 6 models

Best share on the 0.01 grid 0.52 (mean TPR at 1% 0.733), shares within 0.01 of it from 0.50 to 0.59, 0.90 at 0.592.

| TM share | TPR 1% | TPR 5% | TPR 10% | AUROC | TPR 1% minus PSBD-TM | AUROC minus PSBD-TM | losing at 1% / 5% / 10% |
|---|---|---|---|---|---|---|---|
| PSBD-TM alone | 0.535 | 0.710 | 0.788 | 0.891 | | | |
| 0.50 | 0.731 | 0.889 | 0.948 | 0.981 | +0.196 [+0.019, +0.402] | +0.090 [-0.000, +0.245] | 1 / 1 / 1 |
| 0.60 | 0.721 | 0.891 | 0.946 | 0.980 | +0.187 [+0.029, +0.370] | +0.089 [+0.001, +0.243] | 0 / 0 / 1 |
| 0.70 | 0.706 | 0.877 | 0.943 | 0.978 | +0.171 [+0.026, +0.350] | +0.087 [+0.002, +0.239] | 0 / 0 / 0 |
| 0.80 | 0.624 | 0.854 | 0.929 | 0.974 | +0.089 [-0.012, +0.221] | +0.083 [+0.002, +0.231] | 0 / 0 / 0 |
| 0.85 | 0.611 | 0.845 | 0.920 | 0.970 | +0.077 [-0.004, +0.177] | +0.080 [+0.002, +0.224] | 0 / 0 / 0 |
| 0.90 | 0.592 | 0.823 | 0.902 | 0.964 | +0.058 [-0.006, +0.137] | +0.074 [+0.001, +0.210] | 0 / 0 / 0 |
| 0.95 | 0.559 | 0.793 | 0.880 | 0.952 | +0.025 [-0.007, +0.077] | +0.061 [+0.001, +0.178] | 0 / 0 / 0 |
| 0.99 | 0.557 | 0.712 | 0.817 | 0.921 | +0.022 [-0.007, +0.075] | +0.030 [-0.000, +0.090] | 0 / 0 / 0 |

## Selection without the test data

Each fold picks the share on 3 datasets and reads it on the 4th. The 1st criterion is the highest mean TPR at 1% FPR on the fit datasets. The 2nd is the pre-registration's own reason, losing least: the fewest models losing more than 0.05 at 1% FPR against PSBD-TM alone, then the higher mean TPR, a remaining tie going to the smaller share so a tie never favours 0.9. Both search the sweep's 8 shares and the 0.01 grid.

### Leave one dataset out, ViT panel, blocks 5 to 8

| held-out dataset | models fit / held out | pick by mean TPR 1%, sweep grid / 0.01 grid | held-out TPR 1% at the sweep pick | at 0.90 | at min | PSBD-TM alone | losing at 1%, pick / 0.90 / min | pick by fewest losses, sweep grid / 0.01 grid | held-out TPR 1% at that pick |
|---|---|---|---|---|---|---|---|---|---|
| CIFAR-10 | 41 / 15 | 0.60 / 0.61 | 0.553 | 0.566 | 0.508 | 0.573 | 3 / 0 / 6 | 0.95 / 0.95 | 0.569 |
| CIFAR-100 | 44 / 12 | 0.60 / 0.67 | 0.803 | 0.799 | 0.783 | 0.790 | 2 / 1 / 4 | 0.90 / 0.90 | 0.799 |
| GTSRB | 41 / 15 | 0.60 / 0.64 | 0.808 | 0.810 | 0.806 | 0.801 | 0 / 0 / 0 | 0.95 / 0.95 | 0.809 |
| Tiny ImageNet | 42 / 14 | 0.95 / 0.95 | 0.819 | 0.832 | 0.890 | 0.793 | 0 / 0 / 0 | 0.95 / 0.95 | 0.819 |

By mean TPR the picks are 0.60 on CIFAR-10, 0.60 on CIFAR-100, 0.60 on GTSRB and 0.95 on Tiny ImageNet, so 3 of 4 folds sit 0.30 or more below 0.9 and the Tiny fold picks 0.95, where it reads below the plain minimum on Tiny itself. Losing least picks 0.95, 0.90, 0.95 and 0.95 on the same folds, every one within 0.05 of 0.9.

### Leave one dataset out, ViT panel, blocks 9 to 12

| held-out dataset | models fit / held out | pick by mean TPR 1%, sweep grid / 0.01 grid | held-out TPR 1% at the sweep pick | at 0.90 | at min | PSBD-TM alone | losing at 1%, pick / 0.90 / min | pick by fewest losses, sweep grid / 0.01 grid | held-out TPR 1% at that pick |
|---|---|---|---|---|---|---|---|---|---|
| CIFAR-10 | 40 / 1 | 0.85 / 0.81 | 0.137 | 0.082 | 0.695 | 0.022 | 0 / 0 / 0 | 0.95 / 0.95 | 0.068 |
| CIFAR-100 | 29 / 12 | 0.60 / 0.61 | 0.803 | 0.833 | 0.785 | 0.790 | 3 / 1 / 4 | 0.60 / 0.61 | 0.803 |
| GTSRB | 27 / 14 | 0.70 / 0.74 | 0.785 | 0.787 | 0.783 | 0.787 | 0 / 0 / 0 | 0.95 / 0.95 | 0.786 |
| Tiny ImageNet | 27 / 14 | 0.70 / 0.67 | 0.878 | 0.877 | 0.869 | 0.793 | 1 / 1 / 2 | 0.95 / 0.95 | 0.871 |

### Leave one dataset out, Swin-S, blocks 17 to 24

| held-out dataset | models fit / held out | pick by mean TPR 1%, sweep grid / 0.01 grid | held-out TPR 1% at the sweep pick | at 0.90 | at min | PSBD-TM alone | losing at 1%, pick / 0.90 / min | pick by fewest losses, sweep grid / 0.01 grid | held-out TPR 1% at that pick |
|---|---|---|---|---|---|---|---|---|---|
| CIFAR-10 | 35 / 4 | 0.50 / 0.51 | 0.810 | 0.652 | 0.810 | 0.471 | 0 / 0 / 0 | 0.60 / 0.56 | 0.769 |
| CIFAR-100 | 23 / 16 | 0.50 / 0.51 | 0.902 | 0.859 | 0.902 | 0.816 | 0 / 0 / 0 | 0.60 / 0.53 | 0.902 |
| GTSRB | 35 / 4 | 0.50 / 0.51 | 0.874 | 0.523 | 0.874 | 0.459 | 0 / 0 / 0 | 0.60 / 0.57 | 0.853 |
| Tiny ImageNet | 24 / 15 | 0.50 / 0.51 | 0.985 | 0.986 | 0.985 | 0.982 | 1 / 0 / 1 | 0.50 / 0.51 | 0.985 |

On Swin-S both criteria stay at the plain minimum or just above it (losing least picks 0.60, 0.60, 0.60 and 0.50), the architecture where halving PSBD-TM's budget costs least.

### Development set

The development set is the 9 panel models of `experiments/cache_readouts/dev_set.json` that are `successful_2pt` (`vit_cifar10_badnet_a2o_0_01`, `vit_cifar10_blend_0_1`, `vit_cifar10_bpp_0_05`, `vit_cifar10_tact_0_01`, `vit_cifar10_wanet_0_1`, `vit_gtsrb_bpp_0_01`, `vit_gtsrb_bpp_0_1`, `vit_gtsrb_lf_0_01`, `vit_gtsrb_tact_0_05`). The pick is read on the other 47.

| partner | development models | pick by mean TPR 1%, sweep / 0.01 grid | rest TPR 1% at the pick | at 0.90 | at min | PSBD-TM alone | losing at 1%, pick / 0.90 / min | pick by fewest losses, sweep / 0.01 grid |
|---|---|---|---|---|---|---|---|---|
| blocks 5 to 8 | 9 | 0.60 / 0.67 | 0.806 | 0.801 | 0.787 | 0.788 | 5 / 1 / 8 | 0.60 / 0.67 |
| blocks 9 to 12 | 5 | 0.50 / 0.51 | 0.841 | 0.861 | 0.841 | 0.814 | 6 / 2 / 6 | 0.50 / 0.51 |

On the development set the mean criterion picks 0.60 with the middle band and 0.50 with the late band, the pairing the pre-registration was written on. Its 0.9 came from the qualitative reason it states, keeping most of the WaNet gain while losing least on TaCT, and the losses criterion is a quantitative form of that reason at 1% FPR.

## Where each rule wins

Every dataset and attack cell where the 2 rules differ by more than 0.02 in mean TPR at 1%, 5% or 10% FPR, sorted by the difference at 1%. A negative difference means min wins.

| set | cell (models) | PSBD-TM TPR 1 / 5 / 10% | min TPR 1 / 5 / 10% | weighted TPR 1 / 5 / 10% | weighted minus min at 1 / 5 / 10% |
|---|---|---|---|---|---|
| ViT-B/16, blocks 5 to 8 | cifar10/wanet (1) | 0.022 / 0.040 / 0.087 | 0.554 / 0.796 / 0.881 | 0.044 / 0.562 / 0.657 | -0.510 / -0.234 / -0.224 |
| ViT-B/16, blocks 5 to 8 | tiny/wanet (2) | 0.546 / 0.838 / 0.928 | 0.729 / 0.912 / 0.937 | 0.548 / 0.863 / 0.932 | -0.181 / -0.050 / -0.005 |
| ViT-B/16, blocks 5 to 8 | tiny/badnet_a2o (3) | 0.606 / 0.989 / 0.996 | 0.852 / 0.992 / 0.999 | 0.717 / 0.996 / 0.999 | -0.135 / +0.004 / +0.000 |
| ViT-B/16, blocks 5 to 8 | cifar100/badnet_a2o (3) | 0.792 / 0.996 / 0.999 | 0.842 / 0.997 / 1.000 | 0.769 / 0.997 / 1.000 | -0.073 / -0.000 / +0.000 |
| ViT-B/16, blocks 5 to 8 | cifar10/bpp (3) | 0.628 / 0.709 / 0.760 | 0.591 / 0.693 / 0.740 | 0.619 / 0.707 / 0.756 | +0.028 / +0.015 / +0.017 |
| ViT-B/16, blocks 5 to 8 | gtsrb/lc (1) | 0.833 / 0.921 / 0.950 | 0.805 / 0.915 / 0.959 | 0.833 / 0.924 / 0.955 | +0.028 / +0.009 / -0.004 |
| ViT-B/16, blocks 5 to 8 | cifar10/lf (3) | 0.916 / 0.959 / 0.977 | 0.868 / 0.960 / 0.976 | 0.910 / 0.959 / 0.977 | +0.043 / -0.001 / +0.001 |
| ViT-B/16, blocks 5 to 8 | cifar100/blend (3) | 0.759 / 0.926 / 0.967 | 0.732 / 0.924 / 0.966 | 0.836 / 0.954 / 0.984 | +0.103 / +0.030 / +0.018 |
| ViT-B/16, blocks 5 to 8 | cifar10/badnet_a2o (3) | 0.714 / 0.869 / 0.953 | 0.517 / 0.818 / 0.894 | 0.696 / 0.866 / 0.948 | +0.179 / +0.048 / +0.054 |
| ViT-B/16, blocks 5 to 8 | cifar10/blend (3) | 0.596 / 0.745 / 0.812 | 0.377 / 0.787 / 0.880 | 0.588 / 0.746 / 0.826 | +0.212 / -0.041 / -0.053 |
| Swin-S, blocks 17 to 24 | gtsrb/wanet (2) | 0.071 / 0.558 / 0.849 | 0.808 / 0.916 / 0.921 | 0.124 / 0.839 / 0.913 | -0.685 / -0.078 / -0.008 |
| Swin-S, blocks 17 to 24 | cifar100/wanet (2) | 0.116 / 0.750 / 0.880 | 0.884 / 0.901 / 0.910 | 0.490 / 0.894 / 0.900 | -0.394 / -0.007 / -0.010 |
| Swin-S, blocks 17 to 24 | cifar10/wanet (2) | 0.161 / 0.515 / 0.768 | 0.871 / 0.973 / 0.978 | 0.531 / 0.908 / 0.965 | -0.341 / -0.065 / -0.013 |
| Swin-S, blocks 17 to 24 | tiny/badnet_a2o (3) | 0.995 / 1.000 / 1.000 | 0.972 / 1.000 / 1.000 | 0.994 / 1.000 / 1.000 | +0.022 / +0.000 / -0.000 |
| Swin-S, blocks 17 to 24 | cifar10/badnet_a2o (1) | 0.906 / 0.958 / 0.995 | 0.877 / 0.941 / 0.992 | 0.900 / 0.957 / 0.995 | +0.023 / +0.016 / +0.003 |
| Swin-S, blocks 17 to 24 | cifar10/lf (1) | 0.653 / 0.753 / 0.799 | 0.619 / 0.706 / 0.756 | 0.647 / 0.750 / 0.795 | +0.027 / +0.044 / +0.038 |
| BackdoorBench | tiny/trojannn (2) | 0.822 / 0.955 / 0.986 | 0.988 / 0.998 / 0.999 | 0.930 / 0.993 / 0.998 | -0.058 / -0.005 / -0.001 |
| BackdoorBench | gtsrb/ssba (1) | 0.044 / 0.805 / 0.878 | 0.070 / 0.419 / 0.858 | 0.040 / 0.798 / 0.880 | -0.030 / +0.378 / +0.023 |
| BackdoorBench | gtsrb/trojannn (1) | 0.447 / 0.792 / 0.928 | 0.448 / 0.536 / 0.818 | 0.450 / 0.778 / 0.851 | +0.002 / +0.241 / +0.033 |
| BackdoorBench | cifar10/trojannn (2) | 0.622 / 0.980 / 0.999 | 0.482 / 0.967 / 0.995 | 0.624 / 0.979 / 0.999 | +0.142 / +0.012 / +0.004 |
| BackdoorBench | cifar10/ssba (1) | 0.903 / 0.977 / 0.978 | 0.580 / 0.978 / 0.979 | 0.911 / 0.978 / 0.978 | +0.331 / -0.000 / -0.001 |
| BackdoorBench | cifar10/blind (1) | 0.515 / 0.917 / 0.998 | 0.001 / 0.809 / 0.914 | 0.439 / 0.899 / 0.992 | +0.438 / +0.090 / +0.078 |
| training set | cifar10/wanet (1) | 0.017 / 0.040 / 0.096 | 0.652 / 0.880 / 0.959 | 0.230 / 0.630 / 0.725 | -0.422 / -0.250 / -0.234 |
| training set | tiny/wanet (1) | 0.294 / 0.759 / 0.934 | 0.656 / 0.942 / 0.978 | 0.294 / 0.838 / 0.954 | -0.362 / -0.105 / -0.024 |
| training set | tiny/badnet_a2o (1) | 0.769 / 0.999 / 1.000 | 0.981 / 1.000 / 1.000 | 0.921 / 1.000 / 1.000 | -0.060 / +0.000 / +0.000 |
| training set | cifar10/blend (1) | 0.597 / 0.708 / 0.758 | 0.635 / 0.845 / 0.913 | 0.586 / 0.717 / 0.803 | -0.049 / -0.128 / -0.109 |
| training set | cifar10/badnet_a2o (1) | 0.554 / 0.759 / 0.939 | 0.487 / 0.669 / 0.839 | 0.547 / 0.754 / 0.933 | +0.059 / +0.085 / +0.094 |

Min wins where the partner carries a trigger PSBD-TM misses. ViT WaNet reads 0.372 at 1% FPR alone, 0.671 under min and 0.380 under the weighted rule, because the partner alone catches 0.227 of the WaNet inputs at 0.1% FPR, the budget the weighted rule leaves it, against 0.654 at 0.5%, the budget min leaves it. The same holds for Swin WaNet, the training-set WaNet models, BackdoorBench TrojanNN on Tiny (`bb_tiny_trojannn_0_05` 0.986 against 0.956, `bb_tiny_trojannn_0_1` 0.990 against 0.905) and Input-Aware on GTSRB (0.964 against 0.953, a small gap). The weighted rule wins where PSBD-TM already detects and halving its budget cuts into the steep part of its ROC. BackdoorBench Blind on CIFAR-10 reads 0.515 alone, 0.001 under min and 0.439 under the weighted rule, and the ViT BadNets, Blend, LF and BPP losses of the min rule at 1% FPR are the same effect. At 10% FPR every gap shrinks, since both probes then sit on the flat part of their ROCs.

## Per model

TPR at 1%, 5%, 10% and 20% FPR then AUROC, PSBD-TM alone, min and weighted 0.9/0.1, with the partner alone at 1% FPR.

### ViT-B/16 panel, partner blocks 5 to 8, 56 models

| model | partner rate | PSBD-TM | min | weighted 0.9/0.1 | partner alone TPR 1% |
|---|---|---|---|---|---|
| `vit_cifar100_badnet_a2o_0_01` | 0.7 | 0.497 / 0.994 / 0.999 / 1.000 (0.988) | 0.779 / 0.999 / 1.000 / 1.000 (0.993) | 0.441 / 0.997 / 1.000 / 1.000 (0.988) | 0.928 |
| `vit_cifar100_badnet_a2o_0_05` | 0.8 | 0.882 / 0.994 / 0.999 / 1.000 (0.994) | 0.755 / 0.993 / 1.000 / 1.000 (0.991) | 0.870 / 0.994 / 1.000 / 1.000 (0.993) | 0.556 |
| `vit_cifar100_badnet_a2o_0_1` | 0.7 | 0.997 / 1.000 / 1.000 / 1.000 (0.998) | 0.994 / 1.000 / 1.000 / 1.000 (0.998) | 0.996 / 1.000 / 1.000 / 1.000 (0.998) | 0.852 |
| `vit_cifar100_blend_0_01` | 0.8 | 0.562 / 0.878 / 0.959 / 0.983 (0.976) | 0.206 / 0.773 / 0.899 / 0.971 (0.960) | 0.543 / 0.866 / 0.953 / 0.983 (0.974) | 0.002 |
| `vit_cifar100_blend_0_05` | 0.8 | 0.714 / 0.900 / 0.941 / 0.976 (0.979) | 0.992 / 0.999 / 1.000 / 1.000 (0.998) | 0.965 / 0.997 / 0.999 / 1.000 (0.998) | 0.992 |
| `vit_cifar100_blend_0_1` | 0.7 | 1.000 / 1.000 / 1.000 / 1.000 (0.998) | 1.000 / 1.000 / 1.000 / 1.000 (0.998) | 1.000 / 1.000 / 1.000 / 1.000 (0.998) | 0.977 |
| `vit_cifar100_bpp_0_01` | 0.7 | 0.216 / 0.677 / 0.849 / 0.919 (0.921) | 0.163 / 0.616 / 0.844 / 0.916 (0.915) | 0.212 / 0.637 / 0.840 / 0.917 (0.918) | 0.128 |
| `vit_cifar100_bpp_0_05` | 0.8 | 0.923 / 0.949 / 0.961 / 0.977 (0.985) | 0.913 / 0.958 / 0.974 / 0.986 (0.988) | 0.922 / 0.951 / 0.961 / 0.979 (0.986) | 0.786 |
| `vit_cifar100_bpp_0_1` | 0.8 | 0.898 / 0.949 / 0.963 / 0.975 (0.985) | 0.892 / 0.953 / 0.974 / 0.987 (0.989) | 0.893 / 0.949 / 0.965 / 0.978 (0.986) | 0.831 |
| `vit_cifar100_lf_0_01` | 0.8 | 0.870 / 0.938 / 0.942 / 0.947 (0.956) | 0.808 / 0.938 / 0.950 / 0.959 (0.969) | 0.834 / 0.937 / 0.943 / 0.949 (0.960) | 0.765 |
| `vit_cifar100_lf_0_05` | 0.8 | 0.923 / 0.950 / 0.954 / 0.959 (0.967) | 0.922 / 0.952 / 0.956 / 0.961 (0.973) | 0.924 / 0.951 / 0.955 / 0.960 (0.968) | 0.891 |
| `vit_cifar100_lf_0_1` | 0.7 | 0.992 / 0.994 / 0.995 / 0.995 (0.995) | 0.979 / 0.995 / 0.996 / 0.996 (0.997) | 0.992 / 0.994 / 0.995 / 0.996 (0.995) | 0.669 |
| `vit_cifar10_badnet_a2o_0_01` | 0.9 | 0.851 / 0.927 / 0.959 / 0.978 (0.982) | 0.801 / 0.898 / 0.929 / 0.965 (0.975) | 0.851 / 0.925 / 0.956 / 0.978 (0.981) | 0.001 |
| `vit_cifar10_badnet_a2o_0_05` | 0.9 | 0.809 / 0.929 / 0.965 / 0.995 (0.992) | 0.747 / 0.888 / 0.938 / 0.978 (0.984) | 0.804 / 0.928 / 0.963 / 0.995 (0.992) | 0.000 |
| `vit_cifar10_badnet_a2o_0_1` | 0.9 | 0.482 / 0.752 / 0.935 / 1.000 (0.991) | 0.003 / 0.668 / 0.816 / 0.996 (0.976) | 0.433 / 0.747 / 0.925 / 1.000 (0.989) | 0.008 |
| `vit_cifar10_blend_0_01` | 0.9 | 0.599 / 0.706 / 0.773 / 0.869 (0.922) | 0.552 / 0.712 / 0.786 / 0.892 (0.930) | 0.596 / 0.708 / 0.780 / 0.871 (0.923) | 0.068 |
| `vit_cifar10_blend_0_05` | 0.8 | 0.640 / 0.824 / 0.908 / 0.989 (0.971) | 0.060 / 0.845 / 0.951 / 1.000 (0.973) | 0.623 / 0.822 / 0.907 / 0.989 (0.971) | 0.128 |
| `vit_cifar10_blend_0_1` | 0.9 | 0.550 / 0.707 / 0.754 / 0.849 (0.906) | 0.519 / 0.803 / 0.902 / 0.962 (0.963) | 0.546 / 0.708 / 0.792 / 0.896 (0.928) | 0.246 |
| `vit_cifar10_bpp_0_01` | 0.9 | 0.867 / 0.931 / 0.956 / 0.976 (0.986) | 0.847 / 0.937 / 0.963 / 0.983 (0.987) | 0.857 / 0.932 / 0.958 / 0.977 (0.986) | 0.273 |
| `vit_cifar10_bpp_0_05` | 0.9 | 0.400 / 0.516 / 0.587 / 0.655 (0.801) | 0.333 / 0.470 / 0.535 / 0.641 (0.811) | 0.388 / 0.511 / 0.577 / 0.652 (0.801) | 0.002 |
| `vit_cifar10_bpp_0_1` | 0.9 | 0.617 / 0.681 / 0.736 / 0.830 (0.871) | 0.593 / 0.671 / 0.721 / 0.808 (0.909) | 0.612 / 0.679 / 0.735 / 0.829 (0.877) | 0.042 |
| `vit_cifar10_lf_0_01` | 0.9 | 0.869 / 0.937 / 0.954 / 0.975 (0.979) | 0.778 / 0.941 / 0.962 / 0.978 (0.985) | 0.858 / 0.938 / 0.956 / 0.977 (0.981) | 0.138 |
| `vit_cifar10_lf_0_05` | 0.9 | 0.914 / 0.954 / 0.983 / 0.994 (0.991) | 0.875 / 0.952 / 0.971 / 0.992 (0.987) | 0.913 / 0.954 / 0.979 / 0.994 (0.990) | 0.150 |
| `vit_cifar10_lf_0_1` | 0.9 | 0.963 / 0.985 / 0.995 / 0.999 (0.993) | 0.951 / 0.986 / 0.994 / 0.999 (0.990) | 0.959 / 0.986 / 0.995 / 0.999 (0.993) | 0.317 |
| `vit_cifar10_tact_0_01` | 0.9 | 0.005 / 0.005 / 0.005 / 0.009 (0.979) | 0.004 / 0.009 / 0.011 / 0.044 (0.900) | 0.005 / 0.005 / 0.006 / 0.010 (0.978) | 0.004 |
| `vit_cifar10_tact_0_05` | 0.8 | 0.000 / 0.000 / 0.000 / 0.031 (0.966) | 0.000 / 0.000 / 0.000 / 0.013 (0.961) | 0.000 / 0.000 / 0.000 / 0.030 (0.966) | 0.000 |
| `vit_cifar10_wanet_0_1` | 0.8 | 0.022 / 0.040 / 0.087 / 0.268 (0.459) | 0.554 / 0.796 / 0.881 / 0.911 (0.935) | 0.044 / 0.562 / 0.657 / 0.800 (0.870) | 0.635 |
| `vit_gtsrb_badnet_a2o_0_01` | 0.8 | 0.984 / 0.994 / 0.996 / 0.999 (0.999) | 0.975 / 0.990 / 0.994 / 0.997 (0.997) | 0.983 / 0.993 / 0.996 / 0.999 (0.998) | 0.000 |
| `vit_gtsrb_badnet_a2o_0_05` | 0.8 | 0.999 / 1.000 / 1.000 / 1.000 (1.000) | 0.998 / 1.000 / 1.000 / 1.000 (1.000) | 0.998 / 1.000 / 1.000 / 1.000 (1.000) | 0.931 |
| `vit_gtsrb_badnet_a2o_0_1` | 0.8 | 0.997 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 |
| `vit_gtsrb_blend_0_01` | 0.7 | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 / 1.000 (0.999) | 1.000 / 1.000 / 1.000 / 1.000 (0.999) | 1.000 |
| `vit_gtsrb_blend_0_05` | 0.8 | 0.994 / 0.999 / 1.000 / 1.000 (0.999) | 0.969 / 0.998 / 1.000 / 1.000 (0.998) | 0.992 / 0.999 / 1.000 / 1.000 (0.999) | 0.214 |
| `vit_gtsrb_blend_0_1` | 0.5 | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 / 1.000 (0.999) | 1.000 / 1.000 / 1.000 / 1.000 (0.999) | 0.998 |
| `vit_gtsrb_bpp_0_01` | 0.7 | 0.474 / 0.628 / 0.717 / 0.809 (0.898) | 0.441 / 0.575 / 0.683 / 0.770 (0.877) | 0.465 / 0.617 / 0.717 / 0.809 (0.898) | 0.002 |
| `vit_gtsrb_bpp_0_05` | 0.8 | 0.966 / 0.973 / 0.978 / 0.987 (0.992) | 0.972 / 0.990 / 0.992 / 0.995 (0.995) | 0.964 / 0.980 / 0.989 / 0.994 (0.994) | 0.965 |
| `vit_gtsrb_bpp_0_1` | 0.7 | 0.929 / 0.958 / 0.971 / 0.986 (0.989) | 0.925 / 0.948 / 0.981 / 0.995 (0.992) | 0.929 / 0.956 / 0.971 / 0.986 (0.989) | 0.687 |
| `vit_gtsrb_lc_0_05_tl1_adv` | 0.7 | 0.833 / 0.921 / 0.950 / 0.963 (0.969) | 0.805 / 0.915 / 0.959 / 0.985 (0.982) | 0.833 / 0.924 / 0.955 / 0.970 (0.973) | 0.004 |
| `vit_gtsrb_lf_0_01` | 0.8 | 0.854 / 0.898 / 0.926 / 0.973 (0.976) | 0.975 / 0.983 / 0.985 / 0.986 (0.990) | 0.959 / 0.977 / 0.981 / 0.985 (0.988) | 0.977 |
| `vit_gtsrb_lf_0_05` | 0.8 | 0.973 / 0.993 / 0.999 / 0.999 (0.998) | 0.996 / 0.999 / 0.999 / 0.999 (0.998) | 0.995 / 0.998 / 0.999 / 0.999 (0.997) | 0.997 |
| `vit_gtsrb_lf_0_1` | 0.8 | 0.976 / 0.995 / 0.998 / 0.999 (0.998) | 0.999 / 0.999 / 0.999 / 0.999 (1.000) | 0.999 / 0.999 / 0.999 / 0.999 (0.999) | 0.999 |
| `vit_gtsrb_tact_0_01_cos` | 0.5 | 0.000 / 0.000 / 0.000 / 0.000 (0.266) | 0.015 / 0.154 / 0.321 / 0.603 (0.765) | 0.000 / 0.017 / 0.060 / 0.142 (0.492) | 0.060 |
| `vit_gtsrb_tact_0_05` | 0.8 | 0.030 / 0.230 / 0.502 / 0.818 (0.942) | 0.013 / 0.096 / 0.231 / 0.531 (0.940) | 0.026 / 0.202 / 0.461 / 0.793 (0.942) | 0.000 |
| `vit_tiny_badnet_a2o_0_01` | 0.8 | 0.578 / 0.974 / 0.989 / 0.997 (0.980) | 0.789 / 0.987 / 0.998 / 0.999 (0.992) | 0.631 / 0.992 / 0.998 / 0.999 (0.989) | 0.832 |
| `vit_tiny_badnet_a2o_0_05` | 0.8 | 0.367 / 0.993 / 0.999 / 1.000 (0.984) | 0.774 / 0.990 / 0.999 / 1.000 (0.990) | 0.537 / 0.997 / 1.000 / 1.000 (0.989) | 0.768 |
| `vit_tiny_badnet_a2o_0_1` | 0.7 | 0.872 / 1.000 / 1.000 / 1.000 (0.993) | 0.992 / 1.000 / 1.000 / 1.000 (0.998) | 0.983 / 1.000 / 1.000 / 1.000 (0.998) | 0.988 |
| `vit_tiny_blend_0_01` | 0.8 | 0.942 / 0.973 / 0.985 / 0.992 (0.992) | 0.967 / 0.994 / 0.997 / 0.999 (0.997) | 0.945 / 0.985 / 0.995 / 0.997 (0.996) | 0.964 |
| `vit_tiny_blend_0_05` | 0.8 | 0.972 / 0.993 / 0.997 / 0.999 (0.997) | 0.997 / 0.999 / 1.000 / 1.000 (0.999) | 0.995 / 0.998 / 1.000 / 1.000 (0.999) | 0.998 |
| `vit_tiny_blend_0_1` | 0.7 | 0.975 / 0.996 / 0.999 / 0.999 (0.998) | 0.992 / 0.999 / 1.000 / 1.000 (0.998) | 0.977 / 0.997 / 0.999 / 1.000 (0.997) | 0.986 |
| `vit_tiny_bpp_0_01` | 0.8 | 0.945 / 0.969 / 0.972 / 0.975 (0.980) | 0.964 / 0.972 / 0.974 / 0.978 (0.984) | 0.954 / 0.971 / 0.973 / 0.975 (0.981) | 0.951 |
| `vit_tiny_bpp_0_05` | 0.7 | 0.950 / 0.980 / 0.984 / 0.986 (0.987) | 0.936 / 0.978 / 0.983 / 0.987 (0.985) | 0.949 / 0.980 / 0.983 / 0.987 (0.986) | 0.843 |
| `vit_tiny_bpp_0_1` | 0.8 | 0.896 / 0.957 / 0.981 / 0.991 (0.984) | 0.969 / 0.993 / 0.996 / 0.996 (0.996) | 0.940 / 0.980 / 0.993 / 0.996 (0.993) | 0.984 |
| `vit_tiny_lf_0_01` | 0.7 | 0.851 / 0.920 / 0.929 / 0.935 (0.941) | 0.806 / 0.913 / 0.928 / 0.937 (0.951) | 0.831 / 0.920 / 0.929 / 0.935 (0.941) | 0.597 |
| `vit_tiny_lf_0_05` | 0.7 | 0.928 / 0.986 / 0.987 / 0.988 (0.987) | 0.898 / 0.985 / 0.988 / 0.990 (0.990) | 0.920 / 0.985 / 0.987 / 0.989 (0.988) | 0.865 |
| `vit_tiny_lf_0_1` | 0.8 | 0.732 / 0.956 / 0.973 / 0.977 (0.973) | 0.915 / 0.966 / 0.974 / 0.980 (0.985) | 0.894 / 0.966 / 0.974 / 0.978 (0.979) | 0.937 |
| `vit_tiny_wanet_0_05` | 0.7 | 0.802 / 0.919 / 0.924 / 0.928 (0.930) | 0.845 / 0.920 / 0.925 / 0.929 (0.942) | 0.815 / 0.920 / 0.925 / 0.928 (0.931) | 0.828 |
| `vit_tiny_wanet_0_1` | 0.7 | 0.291 / 0.757 / 0.932 / 0.970 (0.950) | 0.613 / 0.905 / 0.949 / 0.974 (0.968) | 0.281 / 0.805 / 0.939 / 0.973 (0.955) | 0.777 |

### ViT-B/16 panel, partner blocks 9 to 12 (pre-registered pairing), 41 models

| model | partner rate | PSBD-TM | min | weighted 0.9/0.1 | partner alone TPR 1% |
|---|---|---|---|---|---|
| `vit_cifar100_badnet_a2o_0_01` | 0.9 | 0.497 / 0.994 / 0.999 / 1.000 (0.988) | 0.095 / 0.938 / 0.996 / 0.999 (0.980) | 0.441 / 0.993 / 0.999 / 1.000 (0.987) | 0.000 |
| `vit_cifar100_badnet_a2o_0_05` | 0.9 | 0.882 / 0.994 / 0.999 / 1.000 (0.994) | 0.726 / 0.988 / 0.999 / 1.000 (0.990) | 0.870 / 0.994 / 0.999 / 1.000 (0.993) | 0.321 |
| `vit_cifar100_badnet_a2o_0_1` | 0.9 | 0.997 / 1.000 / 1.000 / 1.000 (0.998) | 0.984 / 1.000 / 1.000 / 1.000 (0.998) | 0.995 / 1.000 / 1.000 / 1.000 (0.998) | 0.036 |
| `vit_cifar100_blend_0_01` | 0.9 | 0.562 / 0.878 / 0.959 / 0.983 (0.976) | 0.236 / 0.799 / 0.909 / 0.975 (0.965) | 0.543 / 0.866 / 0.955 / 0.984 (0.975) | 0.125 |
| `vit_cifar100_blend_0_05` | 0.9 | 0.714 / 0.900 / 0.941 / 0.976 (0.979) | 0.997 / 1.000 / 1.000 / 1.000 (0.999) | 0.978 / 0.998 / 1.000 / 1.000 (0.999) | 0.999 |
| `vit_cifar100_blend_0_1` | 0.9 | 1.000 / 1.000 / 1.000 / 1.000 (0.998) | 1.000 / 1.000 / 1.000 / 1.000 (0.999) | 1.000 / 1.000 / 1.000 / 1.000 (0.999) | 1.000 |
| `vit_cifar100_bpp_0_01` | 0.9 | 0.216 / 0.677 / 0.849 / 0.919 (0.921) | 0.724 / 0.882 / 0.913 / 0.928 (0.951) | 0.477 / 0.812 / 0.891 / 0.924 (0.938) | 0.781 |
| `vit_cifar100_bpp_0_05` | 0.9 | 0.923 / 0.949 / 0.961 / 0.977 (0.985) | 0.956 / 0.977 / 0.981 / 0.989 (0.992) | 0.950 / 0.966 / 0.976 / 0.984 (0.989) | 0.964 |
| `vit_cifar100_bpp_0_1` | 0.9 | 0.898 / 0.949 / 0.963 / 0.975 (0.985) | 0.978 / 0.988 / 0.991 / 0.993 (0.995) | 0.962 / 0.983 / 0.985 / 0.990 (0.994) | 0.982 |
| `vit_cifar100_lf_0_01` | 0.9 | 0.870 / 0.938 / 0.942 / 0.947 (0.956) | 0.810 / 0.936 / 0.948 / 0.958 (0.972) | 0.857 / 0.937 / 0.943 / 0.948 (0.959) | 0.776 |
| `vit_cifar100_lf_0_05` | 0.9 | 0.923 / 0.950 / 0.954 / 0.959 (0.967) | 0.917 / 0.952 / 0.955 / 0.961 (0.974) | 0.925 / 0.951 / 0.954 / 0.959 (0.968) | 0.844 |
| `vit_cifar100_lf_0_1` | 0.9 | 0.992 / 0.994 / 0.995 / 0.995 (0.995) | 0.992 / 0.995 / 0.996 / 0.997 (0.997) | 0.993 / 0.995 / 0.995 / 0.996 (0.996) | 0.987 |
| `vit_cifar10_wanet_0_1` | 0.99 | 0.022 / 0.040 / 0.087 / 0.268 (0.459) | 0.695 / 0.847 / 0.886 / 0.916 (0.938) | 0.082 / 0.701 / 0.780 / 0.864 (0.890) | 0.759 |
| `vit_gtsrb_badnet_a2o_0_01` | 0.99 | 0.984 / 0.994 / 0.996 / 0.999 (0.999) | 0.976 / 0.990 / 0.994 / 0.997 (0.998) | 0.983 / 0.993 / 0.996 / 0.999 (0.998) | 0.034 |
| `vit_gtsrb_badnet_a2o_0_05` | 0.99 | 0.999 / 1.000 / 1.000 / 1.000 (1.000) | 0.998 / 1.000 / 1.000 / 1.000 (1.000) | 0.998 / 1.000 / 1.000 / 1.000 (1.000) | 0.038 |
| `vit_gtsrb_badnet_a2o_0_1` | 0.99 | 0.997 / 1.000 / 1.000 / 1.000 (1.000) | 0.997 / 1.000 / 1.000 / 1.000 (1.000) | 0.997 / 1.000 / 1.000 / 1.000 (1.000) | 0.686 |
| `vit_gtsrb_blend_0_01` | 0.99 | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 / 1.000 (0.999) | 1.000 / 1.000 / 1.000 / 1.000 (0.999) | 0.426 |
| `vit_gtsrb_blend_0_1` | 0.95 | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 / 1.000 (0.999) | 1.000 / 1.000 / 1.000 / 1.000 (0.999) | 0.995 |
| `vit_gtsrb_bpp_0_01` | 0.99 | 0.474 / 0.628 / 0.717 / 0.809 (0.898) | 0.449 / 0.605 / 0.728 / 0.829 (0.899) | 0.469 / 0.614 / 0.733 / 0.821 (0.902) | 0.088 |
| `vit_gtsrb_bpp_0_05` | 0.99 | 0.966 / 0.973 / 0.978 / 0.987 (0.992) | 0.962 / 0.977 / 0.982 / 0.989 (0.993) | 0.964 / 0.975 / 0.980 / 0.988 (0.993) | 0.412 |
| `vit_gtsrb_bpp_0_1` | 0.99 | 0.929 / 0.958 / 0.971 / 0.986 (0.989) | 0.947 / 0.976 / 0.986 / 0.994 (0.995) | 0.941 / 0.967 / 0.979 / 0.991 (0.993) | 0.669 |
| `vit_gtsrb_lc_0_05_tl1_adv` | 0.99 | 0.833 / 0.921 / 0.950 / 0.963 (0.969) | 0.798 / 0.899 / 0.938 / 0.965 (0.973) | 0.831 / 0.920 / 0.951 / 0.966 (0.972) | 0.024 |
| `vit_gtsrb_lf_0_01` | 0.99 | 0.854 / 0.898 / 0.926 / 0.973 (0.976) | 0.874 / 0.913 / 0.944 / 0.968 (0.979) | 0.860 / 0.910 / 0.938 / 0.975 (0.978) | 0.362 |
| `vit_gtsrb_lf_0_05` | 0.99 | 0.973 / 0.993 / 0.999 / 0.999 (0.998) | 0.964 / 0.991 / 0.996 / 0.999 (0.997) | 0.974 / 0.991 / 0.998 / 0.999 (0.998) | 0.357 |
| `vit_gtsrb_lf_0_1` | 0.99 | 0.976 / 0.995 / 0.998 / 0.999 (0.998) | 0.980 / 0.998 / 0.999 / 1.000 (0.999) | 0.979 / 0.996 / 0.998 / 1.000 (0.999) | 0.826 |
| `vit_gtsrb_tact_0_01_cos` | 0.95 | 0.000 / 0.000 / 0.000 / 0.000 (0.266) | 0.002 / 0.038 / 0.088 / 0.187 (0.346) | 0.000 / 0.002 / 0.008 / 0.033 (0.267) | 0.008 |
| `vit_gtsrb_tact_0_05` | 0.99 | 0.030 / 0.230 / 0.502 / 0.818 (0.942) | 0.013 / 0.104 / 0.271 / 0.595 (0.946) | 0.026 / 0.203 / 0.466 / 0.802 (0.945) | 0.003 |
| `vit_tiny_badnet_a2o_0_01` | 0.9 | 0.578 / 0.974 / 0.989 / 0.997 (0.980) | 0.207 / 0.880 / 0.974 / 0.993 (0.968) | 0.388 / 0.969 / 0.989 / 0.997 (0.980) | 0.000 |
| `vit_tiny_badnet_a2o_0_05` | 0.9 | 0.367 / 0.993 / 0.999 / 1.000 (0.984) | 0.304 / 0.965 / 0.997 / 1.000 (0.973) | 0.359 / 0.991 / 0.999 / 1.000 (0.981) | 0.219 |
| `vit_tiny_badnet_a2o_0_1` | 0.8 | 0.872 / 1.000 / 1.000 / 1.000 (0.993) | 0.985 / 1.000 / 1.000 / 1.000 (0.999) | 0.971 / 1.000 / 1.000 / 1.000 (0.998) | 0.982 |
| `vit_tiny_blend_0_01` | 0.8 | 0.942 / 0.973 / 0.985 / 0.992 (0.992) | 0.995 / 0.998 / 0.998 / 0.999 (0.999) | 0.988 / 0.996 / 0.998 / 0.998 (0.999) | 0.997 |
| `vit_tiny_blend_0_05` | 0.9 | 0.972 / 0.993 / 0.997 / 0.999 (0.997) | 0.999 / 1.000 / 1.000 / 1.000 (0.999) | 0.999 / 0.999 / 1.000 / 1.000 (0.999) | 0.999 |
| `vit_tiny_blend_0_1` | 0.9 | 0.975 / 0.996 / 0.999 / 0.999 (0.998) | 0.997 / 1.000 / 1.000 / 1.000 (1.000) | 0.990 / 0.999 / 1.000 / 1.000 (0.999) | 0.999 |
| `vit_tiny_bpp_0_01` | 0.8 | 0.945 / 0.969 / 0.972 / 0.975 (0.980) | 0.969 / 0.972 / 0.973 / 0.975 (0.984) | 0.968 / 0.971 / 0.973 / 0.975 (0.982) | 0.969 |
| `vit_tiny_bpp_0_05` | 0.8 | 0.950 / 0.980 / 0.984 / 0.986 (0.987) | 0.985 / 0.986 / 0.987 / 0.988 (0.991) | 0.985 / 0.986 / 0.986 / 0.988 (0.990) | 0.985 |
| `vit_tiny_bpp_0_1` | 0.8 | 0.896 / 0.957 / 0.981 / 0.991 (0.984) | 0.995 / 0.995 / 0.996 / 0.996 (0.997) | 0.992 / 0.995 / 0.995 / 0.996 (0.997) | 0.995 |
| `vit_tiny_lf_0_01` | 0.9 | 0.851 / 0.920 / 0.929 / 0.935 (0.941) | 0.893 / 0.925 / 0.931 / 0.942 (0.961) | 0.877 / 0.923 / 0.930 / 0.937 (0.946) | 0.883 |
| `vit_tiny_lf_0_05` | 0.8 | 0.928 / 0.986 / 0.987 / 0.988 (0.987) | 0.984 / 0.987 / 0.989 / 0.990 (0.993) | 0.976 / 0.986 / 0.988 / 0.989 (0.990) | 0.984 |
| `vit_tiny_lf_0_1` | 0.9 | 0.732 / 0.956 / 0.973 / 0.977 (0.973) | 0.971 / 0.975 / 0.977 / 0.981 (0.987) | 0.961 / 0.974 / 0.977 / 0.979 (0.982) | 0.972 |
| `vit_tiny_wanet_0_05` | 0.8 | 0.802 / 0.919 / 0.924 / 0.928 (0.930) | 0.920 / 0.923 / 0.927 / 0.932 (0.950) | 0.901 / 0.923 / 0.925 / 0.929 (0.934) | 0.920 |
| `vit_tiny_wanet_0_1` | 0.8 | 0.291 / 0.757 / 0.932 / 0.970 (0.950) | 0.967 / 0.970 / 0.974 / 0.978 (0.986) | 0.924 / 0.969 / 0.972 / 0.976 (0.983) | 0.968 |

### Swin-S panel, partner blocks 17 to 24, with the nearest-rate models, 63 models

| model | partner rate | PSBD-TM | min | weighted 0.9/0.1 | partner alone TPR 1% |
|---|---|---|---|---|---|
| `swin_cifar100_adaptive_blend_0_05` | 0.99 | 0.912 / 0.912 / 0.912 / 0.913 (0.921) | 0.911 / 0.918 / 0.929 / 0.943 (0.966) | 0.912 / 0.913 / 0.914 / 0.918 (0.934) | 0.039 |
| `swin_cifar100_adaptive_blend_0_1` | 0.99 | 0.970 / 0.971 / 0.971 / 0.971 (0.970) | 0.952 / 0.973 / 0.977 / 0.982 (0.985) | 0.970 / 0.971 / 0.971 / 0.973 (0.975) | 0.039 |
| `swin_cifar100_badnet_a2o_0_01` | 0.99 | 0.995 / 1.000 / 1.000 / 1.000 (1.000) | 0.982 / 1.000 / 1.000 / 1.000 (0.999) | 0.995 / 1.000 / 1.000 / 1.000 (1.000) | 0.000 |
| `swin_cifar100_badnet_a2o_0_05` | 0.99 | 0.991 / 1.000 / 1.000 / 1.000 (1.000) | 0.963 / 0.999 / 1.000 / 1.000 (0.999) | 0.987 / 1.000 / 1.000 / 1.000 (1.000) | 0.214 |
| `swin_cifar100_badnet_a2o_0_1` | 0.99 | 0.996 / 1.000 / 1.000 / 1.000 (1.000) | 0.994 / 0.999 / 1.000 / 1.000 (0.999) | 0.996 / 1.000 / 1.000 / 1.000 (0.999) | 0.000 |
| `swin_cifar100_blend_0_01` | 0.99 | 0.995 / 0.998 / 0.998 / 0.999 (0.999) | 0.987 / 0.997 / 0.998 / 0.999 (0.997) | 0.993 / 0.997 / 0.998 / 0.999 (0.997) | 0.000 |
| `swin_cifar100_blend_0_05` | 0.99 | 0.999 / 1.000 / 1.000 / 1.000 (1.000) | 0.998 / 1.000 / 1.000 / 1.000 (1.000) | 0.999 / 1.000 / 1.000 / 1.000 (1.000) | 0.243 |
| `swin_cifar100_blend_0_1` | 0.99 | 0.984 / 0.999 / 1.000 / 1.000 (0.999) | 0.964 / 0.997 / 1.000 / 1.000 (0.996) | 0.981 / 0.999 / 1.000 / 1.000 (0.997) | 0.020 |
| `swin_cifar100_bpp_0_01` | 0.99 | 0.963 / 0.989 / 0.993 / 0.995 (0.997) | 0.943 / 0.985 / 0.993 / 0.995 (0.995) | 0.961 / 0.990 / 0.994 / 0.995 (0.996) | 0.868 |
| `swin_cifar100_bpp_0_05` | 0.99 | 0.972 / 0.992 / 0.996 / 0.998 (0.998) | 0.974 / 0.995 / 0.998 / 0.999 (0.998) | 0.973 / 0.994 / 0.997 / 0.999 (0.998) | 0.959 |
| `swin_cifar100_bpp_0_1` | 0.99 | 0.976 / 0.991 / 0.994 / 0.997 (0.998) | 0.973 / 0.990 / 0.994 / 0.998 (0.998) | 0.975 / 0.991 / 0.995 / 0.998 (0.998) | 0.835 |
| `swin_cifar100_lf_0_01` | 0.99 | 0.239 / 0.700 / 0.852 / 0.947 (0.945) | 0.194 / 0.601 / 0.882 / 0.958 (0.937) | 0.226 / 0.649 / 0.839 / 0.947 (0.939) | 0.102 |
| `swin_cifar100_lf_0_05` | 0.99 | 0.894 / 0.978 / 0.986 / 0.989 (0.991) | 0.871 / 0.979 / 0.987 / 0.990 (0.991) | 0.886 / 0.974 / 0.986 / 0.989 (0.990) | 0.470 |
| `swin_cifar100_lf_0_1` | 0.99 | 0.928 / 0.983 / 0.993 / 0.998 (0.996) | 0.961 / 0.998 / 0.999 / 0.999 (0.997) | 0.918 / 0.994 / 0.998 / 0.998 (0.996) | 0.963 |
| `swin_cifar100_wanet_0_05` | 0.99 | 0.099 / 0.780 / 0.867 / 0.881 (0.904) | 0.845 / 0.874 / 0.885 / 0.901 (0.937) | 0.095 / 0.864 / 0.872 / 0.885 (0.911) | 0.866 |
| `swin_cifar100_wanet_0_1` | 0.99 | 0.134 / 0.721 / 0.892 / 0.926 (0.927) | 0.922 / 0.929 / 0.935 / 0.947 (0.965) | 0.884 / 0.925 / 0.928 / 0.935 (0.955) | 0.925 |
| `swin_cifar10_adaptive_blend_0_05` | 0.99 nearest | 0.010 / 0.018 / 0.030 / 0.100 (0.496) | 0.009 / 0.030 / 0.052 / 0.092 (0.434) | 0.009 / 0.019 / 0.035 / 0.095 (0.497) | 0.017 |
| `swin_cifar10_adaptive_blend_0_1` | 0.99 nearest | 0.992 / 0.992 / 0.992 / 0.992 (0.992) | 0.993 / 0.993 / 0.993 / 0.993 (0.994) | 0.992 / 0.993 / 0.993 / 0.993 (0.991) | 0.222 |
| `swin_cifar10_badnet_a2o_0_01` | 0.99 | 0.906 / 0.958 / 0.995 / 1.000 (0.993) | 0.877 / 0.941 / 0.992 / 0.999 (0.992) | 0.900 / 0.957 / 0.995 / 1.000 (0.993) | 0.000 |
| `swin_cifar10_badnet_a2o_0_05` | 0.99 nearest | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 0.003 |
| `swin_cifar10_badnet_a2o_0_1` | 0.99 nearest | 0.994 / 0.999 / 1.000 / 1.000 (1.000) | 0.985 / 0.998 / 1.000 / 1.000 (0.999) | 0.994 / 0.999 / 1.000 / 1.000 (1.000) | 0.018 |
| `swin_cifar10_blend_0_01` | 0.99 nearest | 0.887 / 0.951 / 0.973 / 0.993 (0.990) | 0.863 / 0.962 / 0.988 / 0.999 (0.991) | 0.884 / 0.954 / 0.977 / 0.994 (0.991) | 0.435 |
| `swin_cifar10_blend_0_05` | 0.99 nearest | 0.939 / 0.988 / 0.997 / 1.000 (0.997) | 0.920 / 0.977 / 0.993 / 1.000 (0.994) | 0.935 / 0.987 / 0.997 / 1.000 (0.997) | 0.006 |
| `swin_cifar10_blend_0_1` | 0.99 nearest | 0.998 / 0.999 / 1.000 / 1.000 (1.000) | 0.998 / 1.000 / 1.000 / 1.000 (1.000) | 0.998 / 0.999 / 1.000 / 1.000 (1.000) | 0.865 |
| `swin_cifar10_bpp_0_01` | 0.99 nearest | 0.156 / 0.255 / 0.378 / 0.604 (0.800) | 0.136 / 0.209 / 0.269 / 0.497 (0.750) | 0.152 / 0.250 / 0.352 / 0.601 (0.798) | 0.000 |
| `swin_cifar10_bpp_0_05` | 0.99 nearest | 0.981 / 0.986 / 0.989 / 0.993 (0.997) | 0.982 / 0.990 / 0.994 / 0.997 (0.997) | 0.981 / 0.988 / 0.991 / 0.994 (0.996) | 0.866 |
| `swin_cifar10_bpp_0_1` | 0.99 nearest | 0.991 / 0.993 / 0.995 / 0.997 (0.998) | 0.989 / 0.994 / 0.996 / 0.998 (0.998) | 0.990 / 0.994 / 0.995 / 0.997 (0.998) | 0.905 |
| `swin_cifar10_lf_0_01` | 0.99 nearest | 0.906 / 0.957 / 0.986 / 0.994 (0.996) | 0.945 / 0.981 / 0.993 / 0.994 (0.994) | 0.924 / 0.975 / 0.990 / 0.994 (0.995) | 0.904 |
| `swin_cifar10_lf_0_05` | 0.99 | 0.653 / 0.753 / 0.799 / 0.883 (0.940) | 0.619 / 0.706 / 0.756 / 0.845 (0.941) | 0.647 / 0.750 / 0.795 / 0.881 (0.938) | 0.000 |
| `swin_cifar10_lf_0_1` | 0.99 nearest | 0.869 / 0.913 / 0.933 / 0.964 (0.982) | 0.846 / 0.908 / 0.943 / 0.987 (0.984) | 0.865 / 0.912 / 0.932 / 0.964 (0.982) | 0.066 |
| `swin_cifar10_wanet_0_05` | 0.99 | 0.163 / 0.545 / 0.846 / 0.977 (0.980) | 0.786 / 0.969 / 0.978 / 0.982 (0.987) | 0.160 / 0.856 / 0.957 / 0.980 (0.984) | 0.890 |
| `swin_cifar10_wanet_0_1` | 0.99 | 0.160 / 0.486 / 0.691 / 0.958 (0.952) | 0.956 / 0.978 / 0.979 / 0.981 (0.988) | 0.902 / 0.961 / 0.974 / 0.980 (0.985) | 0.968 |
| `swin_gtsrb_adaptive_blend_0_05` | 0.99 | 0.692 / 0.868 / 0.877 / 0.894 (0.925) | 0.877 / 0.930 / 0.946 / 0.967 (0.978) | 0.846 / 0.891 / 0.920 / 0.938 (0.956) | 0.912 |
| `swin_gtsrb_adaptive_blend_0_1` | 0.99 | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 |
| `swin_gtsrb_badnet_a2o_0_01` | 0.99 nearest | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 0.194 |
| `swin_gtsrb_badnet_a2o_0_05` | 0.99 nearest | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 0.265 |
| `swin_gtsrb_badnet_a2o_0_1` | 0.99 nearest | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 0.336 |
| `swin_gtsrb_blend_0_01` | 0.99 nearest | 0.885 / 0.906 / 0.917 / 0.944 (0.978) | 0.952 / 0.988 / 0.995 / 0.999 (0.997) | 0.892 / 0.963 / 0.979 / 0.992 (0.992) | 0.886 |
| `swin_gtsrb_blend_0_05` | 0.99 nearest | 0.994 / 0.999 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 |
| `swin_gtsrb_blend_0_1` | 0.99 nearest | 0.903 / 0.932 / 0.950 / 0.972 (0.987) | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 0.986 / 1.000 / 1.000 / 1.000 (0.999) | 1.000 |
| `swin_gtsrb_bpp_0_01` | 0.99 nearest | 0.749 / 0.852 / 0.933 / 0.986 (0.979) | 0.797 / 0.888 / 0.940 / 0.981 (0.980) | 0.752 / 0.873 / 0.935 / 0.986 (0.980) | 0.740 |
| `swin_gtsrb_bpp_0_05` | 0.99 nearest | 0.954 / 0.974 / 0.987 / 0.996 (0.997) | 0.999 / 1.000 / 1.000 / 1.000 (1.000) | 0.969 / 0.999 / 1.000 / 1.000 (0.999) | 1.000 |
| `swin_gtsrb_bpp_0_1` | 0.99 nearest | 0.988 / 0.992 / 0.994 / 0.998 (0.998) | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 0.998 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 |
| `swin_gtsrb_lf_0_01` | 0.99 nearest | 0.396 / 0.768 / 0.890 / 0.981 (0.965) | 0.973 / 0.993 / 0.996 / 0.997 (0.996) | 0.713 / 0.981 / 0.993 / 0.998 (0.991) | 0.986 |
| `swin_gtsrb_lf_0_05` | 0.99 nearest | 0.886 / 0.930 / 0.945 / 0.968 (0.980) | 0.905 / 0.962 / 0.978 / 0.993 (0.992) | 0.899 / 0.948 / 0.964 / 0.981 (0.986) | 0.848 |
| `swin_gtsrb_lf_0_1` | 0.99 nearest | 0.957 / 0.985 / 0.997 / 0.998 (0.998) | 0.985 / 0.995 / 0.997 / 0.998 (0.998) | 0.962 / 0.992 / 0.997 / 0.998 (0.998) | 0.979 |
| `swin_gtsrb_wanet_0_05` | 0.99 | 0.046 / 0.732 / 0.867 / 0.875 (0.885) | 0.826 / 0.873 / 0.881 / 0.892 (0.924) | 0.147 / 0.853 / 0.873 / 0.881 (0.906) | 0.868 |
| `swin_gtsrb_wanet_0_1` | 0.99 | 0.096 / 0.383 / 0.832 / 0.956 (0.919) | 0.791 / 0.959 / 0.962 / 0.964 (0.972) | 0.100 / 0.825 / 0.954 / 0.960 (0.946) | 0.944 |
| `swin_tiny_adaptive_blend_0_1` | 0.95 | 0.987 / 0.987 / 0.987 / 0.987 (0.987) | 0.987 / 0.987 / 0.987 / 0.987 (0.990) | 0.987 / 0.987 / 0.987 / 0.987 (0.987) | 0.987 |
| `swin_tiny_badnet_a2o_0_01` | 0.99 | 0.998 / 1.000 / 1.000 / 1.000 (0.998) | 0.985 / 1.000 / 1.000 / 1.000 (0.997) | 0.998 / 1.000 / 1.000 / 1.000 (0.998) | 0.877 |
| `swin_tiny_badnet_a2o_0_05` | 0.99 | 0.988 / 1.000 / 1.000 / 1.000 (0.999) | 0.931 / 0.999 / 1.000 / 1.000 (0.998) | 0.983 / 1.000 / 1.000 / 1.000 (0.998) | 0.002 |
| `swin_tiny_badnet_a2o_0_1` | 0.99 | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 / 1.000 (0.999) | 1.000 / 1.000 / 1.000 / 1.000 (0.999) | 0.007 |
| `swin_tiny_blend_0_01` | 0.99 | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 / 1.000 (0.999) | 1.000 / 1.000 / 1.000 / 1.000 (0.999) | 0.607 |
| `swin_tiny_blend_0_05` | 0.99 | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 0.999 / 1.000 / 1.000 / 1.000 (0.999) | 0.999 / 1.000 / 1.000 / 1.000 (0.999) | 0.117 |
| `swin_tiny_blend_0_1` | 0.99 | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 0.999 / 1.000 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 / 1.000 (1.000) | 0.960 |
| `swin_tiny_bpp_0_01` | 0.99 | 0.989 / 0.995 / 0.996 / 0.996 (0.997) | 0.989 / 0.995 / 0.996 / 0.997 (0.997) | 0.989 / 0.995 / 0.996 / 0.996 (0.997) | 0.965 |
| `swin_tiny_bpp_0_05` | 0.99 | 0.997 / 0.998 / 0.999 / 0.999 (0.999) | 0.996 / 0.998 / 0.999 / 0.999 (0.999) | 0.996 / 0.998 / 0.999 / 0.999 (0.999) | 0.976 |
| `swin_tiny_bpp_0_1` | 0.99 | 0.996 / 0.999 / 0.999 / 0.999 (1.000) | 0.996 / 0.999 / 0.999 / 0.999 (0.999) | 0.996 / 0.999 / 0.999 / 0.999 (0.999) | 0.995 |
| `swin_tiny_lf_0_01` | 0.99 | 0.909 / 0.927 / 0.930 / 0.934 (0.947) | 0.925 / 0.929 / 0.932 / 0.942 (0.964) | 0.918 / 0.928 / 0.929 / 0.934 (0.949) | 0.925 |
| `swin_tiny_lf_0_05` | 0.99 | 0.962 / 0.994 / 0.995 / 0.995 (0.995) | 0.994 / 0.995 / 0.996 / 0.996 (0.997) | 0.985 / 0.995 / 0.995 / 0.995 (0.996) | 0.995 |
| `swin_tiny_lf_0_1` | 0.99 | 0.949 / 0.998 / 0.998 / 0.998 (0.998) | 0.995 / 0.999 / 0.999 / 0.999 (0.999) | 0.971 / 0.999 / 0.999 / 0.999 (0.998) | 0.999 |
| `swin_tiny_wanet_0_05` | 0.99 | 0.986 / 0.990 / 0.990 / 0.991 (0.991) | 0.989 / 0.990 / 0.990 / 0.992 (0.994) | 0.979 / 0.990 / 0.990 / 0.991 (0.991) | 0.989 |
| `swin_tiny_wanet_0_1` | 0.99 | 0.977 / 0.992 / 0.992 / 0.993 (0.992) | 0.992 / 0.993 / 0.994 / 0.995 (0.997) | 0.987 / 0.993 / 0.993 / 0.993 (0.995) | 0.993 |

### BackdoorBench ViT-B/16, partner blocks 5 to 8, 10 models

| model | partner rate | PSBD-TM | min | weighted 0.9/0.1 | partner alone TPR 1% |
|---|---|---|---|---|---|
| `bb_cifar10_blind_0_1` | 0.6 | 0.515 / 0.917 / 0.998 / 1.000 (0.998) | 0.001 / 0.809 / 0.914 / 0.998 (0.978) | 0.439 / 0.899 / 0.992 / 1.000 (0.996) | 0.002 |
| `bb_cifar10_ssba_0_1` | 0.6 | 0.903 / 0.977 / 0.978 / 0.978 (0.979) | 0.580 / 0.978 / 0.979 / 0.981 (0.985) | 0.911 / 0.978 / 0.978 / 0.979 (0.980) | 0.653 |
| `bb_cifar10_trojannn_0_05` | 0.6 | 0.376 / 0.967 / 0.998 / 0.999 (0.997) | 0.426 / 0.944 / 0.990 / 0.999 (0.992) | 0.399 / 0.964 / 0.998 / 0.999 (0.997) | 0.533 |
| `bb_cifar10_trojannn_0_1` | 0.6 | 0.868 / 0.994 / 1.000 / 1.000 (0.998) | 0.539 / 0.990 / 0.999 / 1.000 (0.997) | 0.849 / 0.995 / 1.000 / 1.000 (0.998) | 0.663 |
| `bb_gtsrb_inputaware_0_1` | 0.6 | 0.885 / 0.942 / 0.964 / 0.979 (0.980) | 0.964 / 0.974 / 0.977 / 0.980 (0.989) | 0.953 / 0.971 / 0.976 / 0.980 (0.985) | 0.966 |
| `bb_gtsrb_ssba_0_1` | 0.8 | 0.044 / 0.805 / 0.878 / 0.917 (0.926) | 0.070 / 0.419 / 0.858 / 0.926 (0.920) | 0.040 / 0.798 / 0.880 / 0.924 (0.927) | 0.121 |
| `bb_gtsrb_trojannn_0_1` | 0.9 | 0.447 / 0.792 / 0.928 / 0.962 (0.946) | 0.448 / 0.536 / 0.818 / 0.958 (0.934) | 0.450 / 0.778 / 0.851 / 0.962 (0.946) | 0.060 |
| `bb_tiny_ssba_0_1` | 0.4 | 0.990 / 0.992 / 0.993 / 0.993 (0.994) | 0.992 / 0.993 / 0.993 / 0.993 (0.995) | 0.992 / 0.993 / 0.993 / 0.993 (0.994) | 0.991 |
| `bb_tiny_trojannn_0_05` | 0.5 | 0.780 / 0.943 / 0.985 / 0.997 (0.987) | 0.986 / 0.997 / 0.998 / 0.999 (0.997) | 0.956 / 0.991 / 0.997 / 0.998 (0.996) | 0.994 |
| `bb_tiny_trojannn_0_1` | 0.5 | 0.864 / 0.966 / 0.987 / 0.996 (0.991) | 0.990 / 0.999 / 0.999 / 0.999 (0.999) | 0.905 / 0.995 / 0.998 / 0.999 (0.997) | 0.996 |

### training-set setting, partner blocks 5 to 8, 6 models

| model | partner rate | PSBD-TM | min | weighted 0.9/0.1 | partner alone TPR 1% |
|---|---|---|---|---|---|
| `vit_cifar10_badnet_a2o_0_1` | 0.9 | 0.554 / 0.759 / 0.939 / 0.999 (0.972) | 0.487 / 0.669 / 0.839 / 0.997 (0.958) | 0.547 / 0.754 / 0.933 / 0.999 (0.971) | 0.013 |
| `vit_cifar10_blend_0_1` | 0.9 | 0.597 / 0.708 / 0.758 / 0.838 (0.913) | 0.635 / 0.845 / 0.913 / 0.961 (0.969) | 0.586 / 0.717 / 0.803 / 0.888 (0.936) | 0.424 |
| `vit_cifar10_wanet_0_1` | 0.8 | 0.017 / 0.040 / 0.096 / 0.286 (0.505) | 0.652 / 0.880 / 0.959 / 0.980 (0.974) | 0.230 / 0.630 / 0.725 / 0.868 (0.915) | 0.772 |
| `vit_tiny_badnet_a2o_0_1` | 0.7 | 0.769 / 0.999 / 1.000 / 1.000 (0.993) | 0.981 / 1.000 / 1.000 / 1.000 (0.999) | 0.921 / 1.000 / 1.000 / 1.000 (0.998) | 0.982 |
| `vit_tiny_blend_0_1` | 0.7 | 0.976 / 0.995 / 0.999 / 1.000 (0.999) | 0.975 / 0.999 / 1.000 / 1.000 (0.998) | 0.976 / 0.997 / 1.000 / 1.000 (0.998) | 0.987 |
| `vit_tiny_wanet_0_1` | 0.7 | 0.294 / 0.759 / 0.934 / 0.993 (0.963) | 0.656 / 0.942 / 0.978 / 0.996 (0.985) | 0.294 / 0.838 / 0.954 / 0.996 (0.969) | 0.844 |

<!-- results:end -->

## Files and commands

```bash
for s in vit_panel swin_panel backdoorbench training_set; do
    .venv/bin/python -m experiments.fusion_weighting.measure --set $s
done
.venv/bin/python -m experiments.fusion_weighting.analyze
.venv/bin/python -m experiments.fusion_weighting.render
```

`measure.py` writes `readings_<set>.json` (every model, every share, 1 to 2 minutes per set), `analyze.py` writes `summary.json` and the figure `share_sweep.png` with its sidecar `share_sweep.json`, and `render.py` writes this README and `tmp/report/sections/fusion_rules.tex`, the report's subsection. All records are under `results/_experiments/fusion_weighting/`.
