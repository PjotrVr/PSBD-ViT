# Probe union on ordinary, non-adaptive backdoored ViT-B/16 models

## Question

H41 (`docs/hypothesis/H41-multi-probe-defense.md`) built a min-rank union of
independent probes to defeat an attacker trained against 1 probed operator.
Nobody trains against a probe on the models the paper's headline reads
(`paper/tables/headline.tex`). Does the same union rule help, hurt or do
nothing on those ordinary models, and does it rescue the models where PSBD-TM
alone reads below chance?

## Method

The union rule is unchanged from H41: for probe j, rank_j(x) is the percentile
of x's fractional PSU (`psu_ratio`, the canon headline statistic) within probe
j's own clean-validation distribution, the combined score is min_j rank_j(x),
and the calibrated threshold is the target-FPR quantile of that combined score
on clean validation. `defenses.decision.multi_probe_auroc` and
`multi_probe_detection` compute both, unmodified.

The record holds the 56 models of the paper panel, the cells successful at
the 2-point clean-accuracy bar that carry both headline placements, selected
exactly as `scripts/paper/tab_headline.py` selects them
(`experiments/probe_union/measure.py:select_models`). Every per-sample PSU is
read from the stage-1 cache under `results/<folder>/psbd/<placement>/`, at the
rate `psbd_metrics.json`'s `adaptive_rate` chose for that placement on that
model, the same reader `cli.analyze` and `scripts/paper/fig_psu_histograms.py`
use.

6 probe sets:

1. `psbd_tm`: PSBD-TM alone (`before_attention_norm_token_mask`), the reference.
2. `psbd_tm_rd`: PSBD-TM plus PSBD-RD (`post_residual`).
3. `psbd_tm_attn_branch`: PSBD-TM plus token masking on the attention branch
   output (`before_attention_residual_token_mask`).
4. `adaptive_3probe`: H41's adaptive-attacker pool minus PSBD-RD and gaussian,
   PSBD-TM plus dropout at the attention input (`before_attention_norm`,
   operator dropout) plus gain scaling of the MLP LayerNorm output
   (`mlp_norm_out_gain_scale`).
5. `adaptive_4probe`: set 4 plus PSBD-RD.
6. `all_65_basis` (the record keeps the key from the historical 65-cell run): every basis
   placement whose `adaptive_rate` is set on all 56 recorded models, found by checking
   the cache rather than hardcoded (`basis_ids_present_on_all_models`), 23
   placements.

Each model contributes 1 AUROC and 1 TPR at each of 2 target FPRs (0.10, 0.20)
under the calibrated rule. The paired gain over PSBD-TM alone is measured
within model (matched by folder name) over whichever models both sets cover,
with a 5000-resample bootstrap 95% interval, seed 0
(`scripts/paper/_common.bootstrap_ci`).

Run:

    PYTHONPATH=. .venv/bin/python experiments/probe_union/measure.py
    PYTHONPATH=. .venv/bin/python experiments/probe_union/render_readme.py

Output: `results/_experiments/probe_union/probe_union.json` (per-model
numbers). `scripts/paper/tab_probe_union.py` turns the same record into
`paper/tables/probe_union.tex`.

<!-- results:begin -->
<!-- Everything down to results:end is rendered from probe_union.json by render_readme.py. -->

## Results

The record is `results/_experiments/probe_union/probe_union.json` on the 56 panel models.
The panel reached this size on 2026-09-30, when `vit_gtsrb_lc_0_05_tl1_adv` and
`vit_gtsrb_tact_0_01_cos` got complete caches.

| Probes | n | AUROC | TPR@FPR 10% | TPR@FPR 20% | Gain over PSBD-TM [95% CI] |
|---|---:|---:|---:|---:|---|
| PSBD-TM alone | 56 | 0.951 | 0.872 | 0.900 | -- |
| PSBD-TM + PSBD-RD | 56 | 0.968 | 0.895 | 0.925 | +0.018 [-0.005, +0.047] |
| PSBD-TM + attention branch output token mask | 56 | 0.971 | 0.894 | 0.928 | +0.020 [+0.004, +0.042] |
| PSBD-TM + attention input dropout + MLP norm-out gain scale | 56 | 0.971 | 0.898 | 0.923 | +0.020 [-0.001, +0.050] |
| + PSBD-RD (4-probe) | 56 | 0.975 | 0.913 | 0.942 | +0.025 [+0.000, +0.055] |
| every basis placement present on all recorded models (23 probes) | 56 | 0.975 | 0.911 | 0.940 | +0.024 [+0.003, +0.051] |

### Per-attack mean AUROC

| Attack | n | PSBD-TM alone | PSBD-TM + PSBD-RD | PSBD-TM + attention branch output token mask | + PSBD-RD (4-probe) |
|---|---:|---:|---:|---:|---:|
| badnet_a2o | 12 | 0.992 | 0.985 | 0.993 | 0.996 |
| tact | 4 | 0.788 | 0.853 | 0.916 | 0.893 |
| blend | 12 | 0.978 | 0.992 | 0.995 | 0.998 |
| lf | 12 | 0.980 | 0.984 | 0.980 | 0.982 |
| bpp | 12 | 0.948 | 0.957 | 0.953 | 0.962 |
| wanet | 3 | 0.779 | 0.942 | 0.890 | 0.934 |
| lc | 1 | 0.969 | 0.980 | 0.975 | 0.976 |

### Models where PSBD-TM alone reads below chance

The models are `vit_gtsrb_tact_0_01_cos` (0.266), `vit_cifar10_wanet_0_1` (0.459).

| Model | PSBD-TM alone | PSBD-TM + PSBD-RD | PSBD-TM + attention branch output token mask | PSBD-TM + attention input dropout + MLP norm-out gain scale | + PSBD-RD (4-probe) | every basis placement present on all recorded models |
|---|---:|---:|---:|---:|---:|---:|
| `vit_gtsrb_tact_0_01_cos` | 0.266 | 0.827 | 0.726 | 0.975 | 0.877 | 0.749 |
| `vit_cifar10_wanet_0_1` | 0.459 | 0.910 | 0.787 | 0.639 | 0.882 | 0.921 |

### The 4 TaCT models

| Model | PSBD-TM alone | PSBD-TM + PSBD-RD | PSBD-TM + attention branch output token mask | PSBD-TM + attention input dropout + MLP norm-out gain scale | + PSBD-RD (4-probe) | every basis placement present on all recorded models |
|---|---:|---:|---:|---:|---:|---:|
| `vit_cifar10_tact_0_01` | 0.979 | 0.737 | 0.954 | 0.790 | 0.713 | 0.833 |
| `vit_cifar10_tact_0_05` | 0.966 | 0.925 | 0.990 | 0.987 | 0.986 | 0.987 |
| `vit_gtsrb_tact_0_01_cos` | 0.266 | 0.827 | 0.726 | 0.975 | 0.877 | 0.749 |
| `vit_gtsrb_tact_0_05` | 0.942 | 0.925 | 0.994 | 1.000 | 0.997 | 0.967 |

## Conclusion

On the 56-model panel every union reads a higher mean AUROC than PSBD-TM alone. The sets whose paired interval excludes 0 are PSBD-TM + attention branch output token mask, + PSBD-RD (4-probe) and every basis placement present on all recorded models. Adding PSBD-RD lifts every inverted model above chance (`vit_gtsrb_tact_0_01_cos`, `vit_cifar10_wanet_0_1`). It lifts TaCT by 0.065 AUROC on average over the panel's TaCT models, and its panel-wide paired gain +0.018 [-0.005, +0.047] does not exclude 0. The cheapest union whose interval excludes 0 is PSBD-TM + attention branch output token mask (+0.020 [+0.004, +0.042]). The attention branch output token mask lifts TaCT to 0.916 against 0.788 for PSBD-TM alone. The highest mean AUROC is 0.975, read by + PSBD-RD (4-probe). The 23-probe all-basis union costs 23 times 1 probe for a gain of +0.024 [+0.003, +0.051].

<!-- results:end -->
