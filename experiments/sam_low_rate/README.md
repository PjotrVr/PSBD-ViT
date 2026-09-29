# Does SAM's PSBD gain on ViT survive at 1% and 5% poisoning

## Question

experiments/sam_reading pooled every poison rate together and found a small
token-mask (PSBD-TM, `before_attention_norm_token_mask`) gain from training the
victim with sharpness-aware minimization and a loss for the placement the
original PSBD paper published (PSBD-RD, `post_residual`). That pooled number
cannot say whether the gain is real at the poison rates an attacker would
actually use, whether it is separation or an artifact of the adaptive rule
picking a different dropout rate on the SAM side, or whether it holds across
the whole disturbance ladder or only at 1 rung of it. `measure.py` re-slices
the same matched Adam-versus-SAM cells by poison rate to answer those 3
questions, plus a fourth: whether the gain concentrates in the attacks whose
trigger is diffuse (BPP, WaNet) rather than firm and localized (BadNets, Blend,
LF), which is what the SAM paper's own backdoor-neuron-amplification account
would predict.

## Records

`results/_experiments/sam_low_rate/sam_low_rate.json` is the run of 2026-09-11, when the
PSBD sweep had reached both placements on 7 to 10 matched pairs per (rate, rho), all ViT
and all CIFAR-10 or CIFAR-100. The sweep has since reached ViT and Swin on all 4 datasets
at rho 0.1, so `measure.py` was rerun on CPU on 2026-09-29 against the current caches and
written beside it as `results/_experiments/sam_low_rate/sam_low_rate_2026-09-29.json`:

```bash
PYTHONPATH=. .venv/bin/python -c "import experiments.sam_low_rate.measure as m; \
    m.OUTPUT_PATH = 'results/_experiments/sam_low_rate/sam_low_rate_2026-09-29.json'; m.main()"
```

Every number below is from the rerun unless it says otherwise. The rho 0.1 rows pool ViT
and Swin over 4 datasets (32 to 39 pairs), while rho 0.05, 0.15 and 0.2 are still the ViT
CIFAR-10 and CIFAR-100 pairs (8 to 10). No attack in this grid is TaCT, so no pair holds a
source-mapped model.

## Coverage at 1% and 5%

40 matched (dataset, attack, architecture) combinations exist at each of 1%
and 5% poisoning, all 4 datasets (CIFAR-10, CIFAR-100, GTSRB, Tiny ImageNet)
by all 5 attacks (`badnet_a2o`, `blend`, `bpp`, `lf`, `wanet`) by both
architectures (ViT-B/16 and Swin-S), and every one reaches all 4 swept rhos
(0.05, 0.1, 0.15, 0.2). Of those, the pairs with `cli.sweep` and `cli.analyze` output for
both PSBD-TM and PSBD-RD on both sides are 32 at 1% and 35 at 5% for rho 0.1, and 8 at the
other rhos.

## The ASR gate

SAM sometimes breaks implantation rather than changing detectability. Every comparison
drops a pair where either side's ASR falls below `ASR_CLEARS_THRESHOLD` (0.85) before
computing a delta, so a "loss" is never read off a pair where SAM quietly failed to plant
the trigger. The rerun drops 4 pairs: Swin GTSRB WaNet 1% at rho 0.1, ViT CIFAR-10 and ViT
GTSRB WaNet 5% at rho 0.1, and ViT CIFAR-100 WaNet 10% at rho 0.05. The gate reads ASR from
`checkpoints/<folder>/metrics.json`, which the 2026-09-29 audit
(`docs/audits/2026-09-29-experiment-audit.md`) found stale. The ledger's ASR differs for
some of these cells (`vit_cifar10_wanet_0_05` reads 0.786 in `metrics.json` and 0.961 in the
ledger, `vit_cifar100_wanet_0_1` 0.900 and 0.793), so the gate is not yet the ledger's gate.

## Part 1: does the gain survive at 1% and 5%

1 row per (poison rate, rho), mean AUROC at the adaptive 0.8 rule, fractional PSU:

| Rate | Rho | Pairs | TM Adam | TM SAM | TM delta | TM 95% CI | RD Adam | RD SAM | RD delta | RD 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| 1% | 0.05 | 8 | 0.964 | 0.962 | -0.001 | [-0.031, +0.024] | 0.831 | 0.761 | -0.070 | [-0.271, +0.060] |
| 1% | 0.1  | 32 | 0.971 | 0.940 | -0.030 | [-0.080, +0.009] | 0.864 | 0.846 | -0.018 | [-0.092, +0.045] |
| 1% | 0.15 | 8 | 0.964 | 0.887 | -0.077 | [-0.186, +0.015] | 0.831 | 0.783 | -0.048 | [-0.192, +0.072] |
| 1% | 0.2  | 8 | 0.964 | 0.908 | -0.056 | [-0.124, +0.010] | 0.831 | 0.824 | -0.008 | [-0.058, +0.052] |
| 5% | 0.05 | 8 | 0.960 | 0.989 | +0.029 | [+0.002, +0.071] | 0.904 | 0.852 | -0.052 | [-0.198, +0.039] |
| 5% | 0.1  | 35 | 0.977 | 0.961 | -0.017 | [-0.049, +0.009] | 0.864 | 0.899 | +0.035 | [-0.031, +0.101] |
| 5% | 0.15 | 8 | 0.960 | 0.947 | -0.013 | [-0.059, +0.027] | 0.904 | 0.736 | -0.168 | [-0.369, -0.022] |
| 5% | 0.2  | 8 | 0.960 | 0.984 | +0.024 | [-0.001, +0.065] | 0.904 | 0.743 | -0.161 | [-0.375, +0.002] |
| 10% | 0.05 | 9 | 0.911 | 0.981 | +0.071 | [-0.004, +0.190] | 0.886 | 0.876 | -0.010 | [-0.057, +0.032] |
| 10% | 0.1  | 39 | 0.964 | 0.980 | +0.016 | [-0.007, +0.046] | 0.917 | 0.905 | -0.013 | [-0.074, +0.047] |
| 10% | 0.15 | 10 | 0.904 | 0.971 | +0.067 | [+0.011, +0.151] | 0.883 | 0.872 | -0.011 | [-0.082, +0.048] |
| 10% | 0.2 | 10 | 0.904 | 0.977 | +0.074 | [+0.007, +0.178] | 0.883 | 0.855 | -0.027 | [-0.088, +0.029] |

At 1% poisoning the token-mask delta is negative at every rho and no interval excludes 0.
At 5% it swings from -0.017 to +0.029 and excludes 0 only at rho 0.05 ([+0.002, +0.071]).
At 10% it is positive at every rho, but on the 39 pairs of rho 0.1, the 1 rho with ViT and
Swin on all 4 datasets, it shrinks to +0.016 with an interval that crosses 0. The 10% gain
of +0.067 to +0.074 at rho 0.15 and 0.2 rests on 10 ViT CIFAR pairs. The residual-dropout
placement never gains at 1%, and its interval at rho 0.15, 5% ([-0.369, -0.022]) excludes 0
in the losing direction.

The 2026-09-11 run read rho 0.1 on 7, 8 and 9 pairs as -0.001 to +0.010 at 1%, +0.019 at
5% and +0.070, CI [+0.008, +0.173], at 10%. The larger pool removes the 10% gain at that rho.

## Part 2: is this separation or calibration

The adaptive rule picks the smallest dropout rate whose clean-validation shift
ratio reaches 0.8, so the rate itself can differ between Adam and SAM even at
the same nominal poison rate. It does, systematically: the mean token-mask
rate SAM selects is higher than Adam's (at 10% poisoning, rho 0.2: Adam 0.550 against SAM
0.690, and at 1% poisoning, rho 0.2: Adam 0.625 against SAM 0.688). SAM-trained ViTs are
more resistant to the same nominal token-mask disturbance, so the rule reaches for more
dropout to hit the same 0.8 shift target on clean validation data. For residual dropout the
calibration gap is smaller (10%, rho 0.2: Adam 0.081 against SAM 0.122) and moves in the
same direction.

## Part 3: the whole disturbance ladder, at rho 0.1

Binning every swept rate's (clean-validation shift ratio, headline AUROC) point from the 1%
and 5% matched pairs at rho 0.1 (`rate_ladder.points`) by shift ratio, ViT and Swin pooled:

Token mask, mean AUROC by shift-ratio bin (Adam / SAM): 0.0-0.1: 0.740 / 0.797, 0.1-0.2:
0.808 / 0.872, 0.2-0.3: 0.910 / 0.881, 0.3-0.4: 0.889 / 0.943, 0.5-0.6: 0.942 / 0.970,
0.6-0.7: 0.963 / 0.949, 0.7-0.8: 0.967 / 0.977, 0.8-0.9: 0.911 / 0.913, 0.9-1.0: 0.929 /
0.899. SAM's curve sits above Adam's at low disturbance and ties or falls below it in the
0.8-0.9 and 0.9-1.0 bins where the adaptive rule lands. On the 2026-09-11 pairs SAM sat above
Adam in every bin (historical 0.867 against 0.927 at 0.8-0.9), which the larger pool does not
reproduce, so there is no separation gain at the disturbance the rule selects.

Residual dropout: 0.0-0.1: 0.640 / 0.698, 0.1-0.2: 0.720 / 0.764, 0.2-0.3: 0.767 / 0.829,
0.8-0.9: 0.763 / 0.748 and 0.9-1.0: 0.767 / 0.800. SAM is higher at low disturbance and
about level with Adam at high disturbance, as before.

## Part 4: per attack

Pooling every poison rate and dataset within 1 attack (`by_attack`), token-mask delta:

| Attack | rho 0.05 | rho 0.1 | rho 0.15 | rho 0.2 |
|---|---|---|---|---|
| BadNets | -0.002 [-0.010, +0.004] | n=23 -0.006 [-0.024, +0.003] | -0.063 [-0.185, +0.002] | -0.049 [-0.106, -0.005] |
| Blend | +0.019 [-0.009, +0.052] | n=23 -0.019 [-0.049, +0.006] | -0.006 [-0.061, +0.048] | +0.028 [+0.002, +0.057] |
| LF | +0.003 [-0.009, +0.016] | n=23 -0.050 [-0.119, -0.007] | -0.062 [-0.156, +0.009] | -0.033 [-0.109, +0.012] |
| BPP | +0.039 [-0.026, +0.103] | n=24 +0.020 [-0.001, +0.045] | +0.045 [+0.014, +0.083] | +0.041 [-0.029, +0.105] |
| WaNet | n=1 +0.499 | n=13 +0.025 [-0.075, +0.121] | n=2 +0.232 | n=2 +0.279 |

At rho 0.05, 0.15 and 0.2 the pattern of the 2026-09-11 run holds: BPP gains, BadNets, Blend
and LF sit near 0 or below. At rho 0.1, with 13 WaNet pairs instead of 2, WaNet's gain falls
from +0.265 to +0.025 with an interval across 0, BPP's to +0.020 with an interval touching
0, and LF loses 0.050 with an interval below 0. The diffuse-trigger advantage survives only
as a small BPP effect.

## Answer

SAM does not give PSBD-TM a reliable gain on ViT or Swin. At 1% poisoning the token-mask
delta is negative at every rho (-0.001 to -0.077) with every interval crossing 0. At 5% it is
significant only at rho 0.05 (+0.029, [+0.002, +0.071]). At 10% it is positive at every rho,
but on the 39-pair rho 0.1 pool it is +0.016 with an interval across 0, and the larger 10%
gains rest on 10 ViT CIFAR pairs. The rule does select a higher rate on the SAM side (10%,
rho 0.2: Adam 0.550, SAM 0.690). At the disturbance it selects, SAM's token-mask curve
ties Adam's (0.8-0.9 bin: 0.911 against 0.913), so the gain the 2026-09-11 run read as
separation does not survive the larger pool. The residual-dropout placement never gains at
1% and its worst interval ([-0.369, -0.022] at rho 0.15, 5%) excludes 0 in the losing
direction. Per attack only BPP keeps a small gain, and WaNet's apparent gain shrinks to
+0.025 once 13 pairs back it.

## Reproduce

```bash
PYTHONPATH=. python experiments/sam_low_rate/measure.py
PYTHONPATH=. python scripts/paper/tab_sam_low_rate.py --paper-dir paper
```

`scripts/paper/tab_sam_low_rate.py` reads `sam_low_rate.json`, the 2026-09-11 run, so
`paper/tables/sam_low_rate.tex` still holds that run's table.
