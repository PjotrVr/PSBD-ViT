# Probe union on ordinary, non-adaptive backdoored ViT-B/16 models

## Question

H41 (`docs/hypothesis/H41-multi-probe-defense.md`) built a min-rank union of
independent probes to defeat an attacker trained against 1 probed operator.
Nobody trains against a probe on the models the paper's headline reads
(`paper/tables/headline.tex`). Does the same union rule help, hurt or do
nothing on those ordinary models, and does it specifically rescue the
inverted cells the headline names, cifar10 wanet at 10% (0.459) and cifar10 sig
at 10% (0.418)?

## Method

The union rule is unchanged from H41: for probe j, rank_j(x) is the percentile
of x's fractional PSU (`psu_ratio`, the canon headline statistic) within probe
j's own clean-validation distribution, the combined score is min_j rank_j(x),
and the calibrated threshold is the target-FPR quantile of that combined score
on clean validation. `defenses.decision.multi_probe_auroc` and
`multi_probe_detection` compute both, unmodified.

The record holds the 57 clearing cells that carry both headline placements, selected
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
   (`mlp_norm_out_gain_scale`). The dropout probe is now cached on every model,
   so this set and the next are read on the full panel.
5. `adaptive_4probe`: set 4 plus PSBD-RD.
6. `all_65_basis` (the record keeps the key from the historical 65-cell run): every basis
   placement whose `adaptive_rate` is set on all 57 recorded models, found by checking
   the cache rather than hardcoded (`basis_ids_present_on_all_models`), 23
   placements.

Each model contributes 1 AUROC and 1 TPR at each of 2 target FPRs (0.10, 0.20)
under the calibrated rule. The paired gain over PSBD-TM alone is measured
within model (matched by folder name) over whichever models both sets cover,
with a 5000-resample bootstrap 95% interval, seed 0
(`scripts/paper/_common.bootstrap_ci`).

Run:

    PYTHONPATH=. .venv/bin/python experiments/probe_union/measure.py
    PYTHONPATH=. .venv/bin/python scripts/paper/tab_probe_union.py

Output: `results/_experiments/probe_union/probe_union.json` (per-model
numbers), `paper/tables/probe_union.tex` and `.macros.json`.

## Results

The record is `results/_experiments/probe_union/probe_union.json`, rerun on 2026-09-24
(commit 307db69) on the 57 clearing models, which drops the 8 source-mapped TaCT models and
the diverged `vit_gtsrb_tact_0_01` the historical 65-cell run held. The paper panel is the 54
models successful at the 2-point clean-accuracy bar, which also drops `vit_cifar10_sig_0_1`,
`vit_cifar10_wanet_0_05` and `vit_gtsrb_wanet_0_1`. Every table below is the record's
per-model rows on those 54, summarized with the experiment's own `paired_gain` (5000
resamples, seed 0):

    PYTHONPATH=. .venv/bin/python scratch/stale_numbers/probe_union_panel.py

| Probes | n | AUROC | TPR@FPR 10% | TPR@FPR 20% | Gain over PSBD-TM [95% CI] |
|---|---:|---:|---:|---:|---|
| PSBD-TM alone | 54 | 0.963 | 0.887 | 0.916 | -- |
| PSBD-TM + PSBD-RD | 54 | 0.971 | 0.901 | 0.929 | +0.008 [-0.009, +0.029] |
| PSBD-TM + attention branch output token mask | 54 | 0.975 | 0.909 | 0.941 | +0.012 [+0.003, +0.027] |
| PSBD-TM + attention input dropout + MLP norm-out gain scale | 54 | 0.970 | 0.914 | 0.938 | +0.008 [-0.005, +0.019] |
| + PSBD-RD (4-probe) | 54 | 0.977 | 0.923 | 0.950 | +0.014 [-0.004, +0.034] |
| every basis placement present on all recorded models (23 probes) | 54 | 0.979 | 0.918 | 0.943 | +0.016 [+0.001, +0.037] |

### Per-attack mean AUROC (n = 54)

| Attack | n | PSBD-TM | PSBD-TM + PSBD-RD | PSBD-TM + branch output | 4-probe union |
|---|---:|---:|---:|---:|---:|
| badnet_a2o | 12 | 0.992 | 0.985 | 0.993 | 0.996 |
| blend | 12 | 0.978 | 0.992 | 0.995 | 0.998 |
| bpp | 12 | 0.948 | 0.957 | 0.953 | 0.962 |
| lf | 12 | 0.980 | 0.984 | 0.980 | 0.982 |
| tact | 3 | 0.962 | 0.862 | 0.979 | 0.899 |
| wanet | 3 | 0.779 | 0.942 | 0.890 | 0.934 |

### The named inverted cells, PSBD-TM alone vs PSBD-TM + PSBD-RD

SIG under audit (`docs/audits/2026-09-29-experiment-audit.md`): the SIG model was trained at
amplitude 0.1 and the code now builds 0.157. It also fails the 2-point clean-accuracy bar,
so it is not a panel model. Its values read the cached passes, which share the training
trigger.

| Cell | PSBD-TM | PSBD-TM + PSBD-RD |
|---|---:|---:|
| cifar10 wanet 10% (panel) | 0.459 | 0.910 |
| cifar10 sig 10% (not a panel model) | 0.418 | 0.908 |

## Conclusion

On the 54-model panel the union helps by a small margin, and every union that holds PSBD-RD
pays for it on TaCT. Adding PSBD-RD to PSBD-TM rescues the inverted WaNet cell outright
(0.459 to 0.910) but costs TaCT 0.100 AUROC on average over its 3 models (cifar10 TaCT 1%
falls from 0.979 to 0.737), so its panel-wide paired gain is not separated from 0 (+0.008,
CI [-0.009, +0.029]). That is the failure mode H41 already named for rank-averaging: a member
that reads one attack badly drags the minimum down on that attack even while it saves
another. The 2-probe union with the attention branch output token mask instead of PSBD-RD
is the cheapest union whose interval excludes 0 (+0.012, CI [+0.003, +0.027]). It lifts TaCT
to 0.979 rather than costing it. The H41 3-probe adaptive pool gains +0.008 with an interval
across 0, and the 4-probe union +0.014, also across 0. The 23-probe all-basis union reads the
highest mean AUROC (0.979) and gain (+0.016, CI [+0.001, +0.037]) at 23 times the cost of 1
probe. More probes help a little, and a well chosen second probe carries most of what they
add.
