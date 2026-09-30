# Do 2 PSBD probes catch different backdoors, or the same one twice?

## Question

Every result in this project reads ONE placement at a time. If 2 placements fail on
different attacks, combining them should cover both. The deployment configuration would
be a set rather than a single config. That is the question here.

## Why it is nearly free

`results/<cell>/psbd/<placement>/rate_*.pt` stores `per_pass_probs` of shape `(k, N)`, the
baseline-predicted class's probability on each perturbed pass. That is exactly PSU's input,
so combining placements at the SCORE level needs no GPU: every number below comes from
tensors already on disk.

That cheapness is the trap. C(18,2) + C(18,3) is 969 combinations and reporting the best of
them manufactures a winner out of noise. So:

- **24 combinations are pre-registered** in `configs/psbd_basis.json`, each carrying a
  mechanism claim written before any fused AUROC was read.
- **4 of them are negative controls** that should gain little if the claimed mechanism is
  what carries the effect.
- **1 combination is selected on CIFAR-10 and GTSRB and reported on CIFAR-100 and Tiny**,
  which never enter the selection.
- The comparison is against a **single fixed probe**, `before_attention_norm_token_mask`,
  not against each combination's own best member. A defender cannot know which member is
  best without labels, so "beats its best member" is an oracle question. "Beats the config I
  would have deployed anyway" is the deployable one.

2 combination rules are reported, both label-free:

| rule | what it asks | weakness |
|---|---|---|
| `min_rank` | is ANY probe suspicious of this sample | one inverted probe drags the union down |
| `mean_rank` | what does the average probe think | dilutes a single strong member |

Both are needed. `before_attention_norm_gaussian` reads AUROC **0.191** on
`vit_cifar100_badnet_a2o_0_01`, which is an inverted probe. It takes `c3_ban_cm_gauss` down to
0.601 under `min_rank`. Reporting only the flattering rule is how a fusion result stops
meaning anything.

Every probe is read at its own rate matched to a clean-validation shift ratio, never at a
shared rate, and the achieved shift is recorded next to every number.

## Result: on the current panel, fusion does not beat the single probe under the min rule

`measure.py` reran on 2026-09-30 at the matched 0.6 rule over the 56 models of the paper
panel, the cells the current ledger marks successful at the 2-point clean-accuracy bar
(`successful_2pt`) that carry both headline placements
(`results/_experiments/probe_fusion/probe_fusion.json`). The record of 2026-09-09 over every
ledger cell is no longer on disk. Every number below is read from the new record by

```
PYTHONPATH=. .venv/bin/python experiments/probe_fusion/summarise.py
PYTHONPATH=. .venv/bin/python experiments/probe_fusion/summarise.py --rule mean_rank
PYTHONPATH=. .venv/bin/python experiments/probe_fusion/panel.py
```

Selection runs on CIFAR-10 and GTSRB (n=28) and reports on the held-out CIFAR-100 and Tiny
(n=26). `vit_gtsrb_lc_0_05_tl1_adv` and `vit_gtsrb_tact_0_01_cos` joined the panel on
2026-09-30 after every selection this protocol made. `summarise.py` keeps them out of
the selection split (`JOINED_AFTER_SELECTION`). Letting them vote would change the
`min_rank` pick to `c3_ban_tm_mlp_gauss` after the fact. They enter only the whole-panel
readings of `panel.py`. The comparison is against the fixed single probe `before_attention_norm_token_mask`. The 2 rules pick
different winners: `min_rank` picks `c5_ban_pre_5_8` (PSBD-TM plus residual dropout in blocks
5 to 8), `mean_rank` picks `c3_ban_tm_gain` (PSBD-TM plus gain scaling of the MLP norm
output).

| quantity | `min_rank`, `c5_ban_pre_5_8` | `mean_rank`, `c3_ban_tm_gain` |
|---|---|---|
| mean delta AUROC | **+0.006** | **+0.015** |
| bootstrap 95% CI | **[-0.008, +0.022]** | **[+0.002, +0.029]** |
| cells won | 9 of 26 | 18 of 26 |
| verdict | **not supported, the interval spans zero** | **holds** |

The 2 rules disagree. Under `mean_rank` the pre-registered held-out winner clears its
interval, and under `min_rank` it does not. The negative control `n4_same_band` (token
masking and residual dropout both confined to blocks 9 to 12) reads +0.031 under `min_rank`
and +0.029 under `mean_rank` on the same 26 cells, above either winner, so neither gain is
the mechanism its family pre-registered.

Merging 2 configurations that share a position buys nothing, and the reason is mechanical.
`both_sublayer_inputs_token_mask` is token masking at `before_attention_norm` AND
`before_mlp_norm`, so it already CONTAINS PSBD-TM. The pre-registered within-position control
`n1_within_input_side` loses 0.019 on the whole panel.

| configuration | AUROC, all (n=56) | AUROC, hard (n=20) | paired delta vs PSBD-TM, all cells |
|---|---|---|---|
| `before_attention_norm_token_mask` alone | 0.925 | 0.860 | reference |
| `both_sublayer_inputs_token_mask` alone | 0.912 | 0.852 | |
| `n1_within_input_side`, adding `before_mlp_norm_token_mask` | 0.906 | 0.832 | -0.019, CI [-0.030, -0.011] |
| `c5_ban_pre_5_8`, adding `pre_residual_blocks_5_8` | 0.940 | 0.889 | +0.015, CI [-0.003, +0.038] |
| `c5_ban_pre_9_12`, adding `pre_residual_blocks_9_12` | 0.945 | 0.888 | +0.020, CI [-0.003, +0.046] |
| `c5_ban_pre_two_bands`, adding residual dropout in blocks 5 to 8 and 9 to 12 | 0.948 | 0.892 | +0.023, CI [+0.001, +0.048] |

Hard means BPP, WaNet, TaCT and LC, the only hard attacks on the panel (`panel.json`). These are
whole-panel means at the matched 0.6 rule, which is why PSBD-TM reads lower here than at the
adaptive 0.8 rule of the headline. The whole-panel deltas are read on the cells the
selection split also used, so they are descriptive and do not replace the held-out test
above. The 67-cell run also held a merge of PSBD-TM with `both_sublayer_inputs_token_mask`
(+0.005, CI [-0.001, +0.011]) and a merge of all 3 (+0.018, CI [-0.002, +0.038]). Neither is a
pre-registered combination, so neither is in the record and neither has been recomputed on
the current panel.

Spanning position families is worth more than merging within one, the direction the
pre-registered C2 family predicted, and on the current panel the 2-band union clears 0 on the
whole panel.

## This verdict replaces 3 earlier ones

| when | n usable cells | held-out delta, `min_rank` | verdict |
|---|---|---|---|
| basis batch still running | 39 | +0.0145, CI [-0.003, +0.035] | not supported |
| basis batch landed, TaCT not yet corrected | 56 | +0.039, CI [+0.022, +0.056] | supported |
| complete panel, TaCT corrected and swept | 67 | +0.008, CI [-0.007, +0.023] | not supported |
| ledger of 2026-09-29, 54 successful models | 54 | +0.006, CI [-0.008, +0.022] | not supported, `mean_rank` holds |
| **current panel of 2026-09-30, 56 models, selection frozen** | **56** | **+0.006, CI [-0.008, +0.022]** | **not supported, `mean_rank` holds** |

The protocol and the pre-registered combination list never changed. The panel did. The
second reading was taken while TaCT was absent entirely. On the 67-cell panel TaCT was 11
of the 31 hard cells, and the residual depth band `pre_residual_blocks_9_12` scored 0.624 on
TaCT against 0.852 for PSBD-TM (historical), so adding TaCT removed the combination's edge. 8 of those 11
TaCT models turned out to map their clean source class to the target with no trigger
(`docs/runs/2026-09-24-tact-multisource.md`) and are no longer backdoor cells. On the 4 TaCT
models of the current panel the same band reads 0.393 against 0.684 for PSBD-TM
(`panel.json`, `singles`). PSBD-TM's own TaCT mean falls because
`vit_gtsrb_tact_0_01_cos` inverts it. The held-out delta is unchanged by the 2 new models,
since both are GTSRB models and the held-out split is CIFAR-100 and Tiny.

The lesson is about coverage: a fusion result read on a panel missing a whole attack, or
holding models that are not backdoors, is a result about the models that happened to be
present.

## Verdict

**Not supported under the pre-registered `min_rank` rule, supported under `mean_rank`.**
On hard attacks PSBD-TM (0.860, n=20) is ahead of every other single configuration of the 27
in the record, `both_sublayer_inputs_token_mask` (the placement that contains it) reading
0.852. It reaches the matched shift ratio on 55 of 56 cells, the exception
`vit_gtsrb_lc_0_05_tl1_adv`, whose lowest cached rate 0.2 already overshoots, and ranks 6th,
1st and 7th of 27 across the 1%, 5% and 10% rates on hard attacks. A union with residual
dropout in the middle or late blocks adds +0.015 to +0.023 on the whole panel, but a negative control beats both held-out
winners, so the result does not confirm the mechanism the combinations were registered
under.

The structural caveat stands. Band-restricted TOKEN masking cannot be brought to the
disturbance the adaptive rule requires (sigma 0.705 on blocks 5 to 8, 0.433 on blocks 9 to
12, at masking probability 0.99, because the 8 unperturbed blocks still carry the signal and
a probability cannot exceed 1). Band-restricted residual DROPOUT has no such ceiling,
reaching 0.959 at p=0.99. These shift ratios were measured on the 67-cell run and are
properties of the operator rather than of the panel.

## Reproduce

```
PYTHONPATH=. python experiments/probe_fusion/measure.py --shift-target 0.6
PYTHONPATH=. python experiments/probe_fusion/summarise.py
PYTHONPATH=. python experiments/probe_fusion/summarise.py --rule mean_rank
```
