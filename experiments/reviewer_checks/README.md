# Reviewer checks

3 checks a reviewer would ask for, run on the paper panel: the backdoored ViT-B/16 models
that `scripts/paper/tab_headline.py`'s `common_coverage` selects, which since 2026-09-30 are
the 56 models successful at the 2-point clean-accuracy bar whose stage 2 reached both PSBD-TM
(`before_attention_norm_token_mask`) and PSBD-RD (`post_residual`, our adaptation of the
original ConvNet placement). `measure.py` imports `tab_headline.measure_cell` and
`common_coverage` directly rather than re-deriving the count.

Run with `PYTHONPATH=. .venv/bin/python experiments/reviewer_checks/measure.py`.
Checks 1 and 2 are CPU only and read only the on-disk caches under
`results/<folder>/psbd/`. Check 3 runs `cli.sweep` on the login GPU (about 22
minutes for the 20 sweep calls this run used) into a separate tree,
`results/_experiments/reviewer_checks/mask_seeds/`, then `cli.analyze` over
that tree, so the canonical caches are never touched.

The first run (2026-09-11, `results/_experiments/reviewer_checks/reviewer_checks.json`) read
69 models, including the 8 source-mapped TaCT models and 3 models that fail the 2-point bar.
Checks 1 and 2 were rerun on CPU on 2026-09-29 on the 54-model panel of that day
(`reviewer_checks_2026-09-29.json`) and again on 2026-09-30 on the 56 models of the current
panel, written to `results/_experiments/reviewer_checks/reviewer_checks_2026-09-30.json`,
with check 3 copied from the first run because it needs a GPU:

    PYTHONPATH=. .venv/bin/python experiments/reviewer_checks/rerun_panel.py

Every number below for checks 1 and 2 is read from the 2026-09-30 record, and every number
for check 3 from the first run. The 2 models that joined the panel on 2026-09-30,
`vit_gtsrb_lc_0_05_tl1_adv` and `vit_gtsrb_tact_0_01_cos`, enter check 1. Check 2 keeps
them out of its selection half (`measure.JOINED_AFTER_SELECTION`), since both are GTSRB
models and letting them vote would re-pick the pair after the fact. With them in, the
search picks `mlp_norm_out_gain_scale` plus PSBD-RD instead.

## Check 1: the size of the defender's clean set

**Question.** The canonical pipeline holds out 2000 clean images
(`data.splits.PSBD_HELDOUT_SIZE`) and reads both the adaptive rate rule and the
detection threshold off them. Does the method still work with a defender who
can only spare 100, 200, 500 or 1000 clean images?

**Method.** 5 fixed-seed draws per subset size, sampled once and reused across
every model and placement. For each draw, the per-rate shift ratio is
recomputed from the cached per-pass argmax restricted to the subset
(`defenses.scores.shift_ratio`), `defenses.decision.select_rate_adaptively`
picks a rate from that smaller sample, and the threshold and detection report
(`defenses.decision.detection_report`) are read at the subset's own PSU
distribution at that rate. The clean and backdoor analysis pool never shrinks,
only the defender's own held-out set does. Both PSBD-TM and PSBD-RD are
checked, over all 56 panel models.

| Subset size | PSBD-TM rate (mean) | PSBD-TM AUROC @ 10% | PSBD-TM TPR @ 10% | PSBD-TM AUROC @ 20% | PSBD-TM TPR @ 20% | PSBD-RD rate (mean) | PSBD-RD AUROC @ 10% | PSBD-RD TPR @ 10% |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100  | 0.539 | 0.947 | 0.865 | 0.947 | 0.894 | 0.075 | 0.875 | 0.715 |
| 200  | 0.540 | 0.949 | 0.869 | 0.949 | 0.897 | 0.073 | 0.877 | 0.719 |
| 500  | 0.539 | 0.950 | 0.872 | 0.950 | 0.900 | 0.073 | 0.877 | 0.724 |
| 1000 | 0.537 | 0.950 | 0.872 | 0.950 | 0.900 | 0.073 | 0.875 | 0.721 |
| 2000 (canonical) | see below | 0.951\* | -- | -- | -- | 0.876\* | -- | -- |

\*The 2000-image row is the canonical reading on the same 56 models, AUROC at q0.25 at the
adaptive 0.8 rule (`\HeadlineAurocAdaptive` and `\PublishedAurocAdaptive`). AUROC does not
depend on the quantile, so it is on the same basis as the subset rows.

Draw-to-draw spread (mean over models of each model's standard deviation across its 5
draws, PSBD-TM AUROC at 10%): 0.005 at 100 images, 0.004 at 200, 0.0007 at 500, 0.0007 at
1000. PSBD-RD's spread runs from 0.008 and 0.009 at 100 and 200 images down to 0.002 at 1000.

The worst single model never drops out entirely. PSBD-TM's worst model is
`vit_gtsrb_tact_0_01_cos` at every subset size (AUROC 0.27, TPR 0 at 10%), the lower of the 2
inverted cells on the panel beside `vit_cifar10_wanet_0_1`, and PSBD-RD's worst is `vit_cifar10_badnet_a2o_0_01` (AUROC 0.31) at 100 and 200
images and `vit_gtsrb_badnet_a2o_0_01` (AUROC 0.27 and 0.22) at 500 and 1000.

**Answer.** The paper's numbers hold under a much smaller clean set. Mean AUROC moves by at
most 0.003 for either placement across the entire 100-to-1000 range, and it sits on the
canonical 2000-image figures. The rate rule is stable too: PSBD-TM's mean selected rate stays
at 0.537 to 0.540 and PSBD-RD's at 0.073 to 0.075. A defender with 100 clean images loses
almost nothing over 1 with 2000, on this panel. The gap between PSBD-TM and PSBD-RD (about
0.07 AUROC at every subset size) is
also unchanged, so the headline placement comparison does not depend on holding a large
clean set either. The first run on 69 models read the same shape at lower levels (PSBD-TM
0.925 to 0.929, PSBD-RD 0.821 to 0.824), a historical reading that included the
source-mapped TaCT models.

## Check 2: the union chosen on held-out models

**Question.** `configs/psbd_basis.json`'s `selection_protocol` splits the
panel the same way the paper's placement choice does: CIFAR-10 and GTSRB
models select, CIFAR-100 and Tiny ImageNet models report. Does the best
min-rank probe union (`experiments/probe_union/pair_search.py`) chosen the same
way on the selection half generalize to the held-out half, or does its apparent
gain come from having been picked on the same models it is later read on?

**Method.** Candidates are the 23 basis placements present on all 56 panel models
(`experiments.probe_union.measure.basis_ids_present_on_all_models`).
Every pair among them is scored by the min-rank union
(`defenses.decision.multi_probe_auroc`/`multi_probe_detection`) on the 28
selection models (CIFAR-10, GTSRB) only, ranked by mean TPR at the 10%
clean-validation quantile
(`experiments.probe_union.pair_search.search_pairs`). The winning pair is
then read for the first time on the 26 held-out models (CIFAR-100, Tiny),
beside PSBD-TM alone on the same held-out models, with the paired AUROC gain
bootstrapped over those 26 models
(`experiments.probe_union.measure.paired_gain`, 2000 resamples, seed 0).

| Configuration | Models | Mean AUROC | Mean TPR @ 10% | Mean TPR @ 20% |
|---|---:|---:|---:|---:|
| Best pair on selection models (`before_attention_residual_token_mask` + `mlp_norm_out_gain_scale`) | 28 (selection) | 0.972 | 0.886 | 0.924 |
| PSBD-TM alone, held-out models | 26 (held-out) | 0.978 | 0.970 | 0.980 |
| Best pair, held-out models | 26 (held-out) | 0.978 | 0.958 | 0.976 |
| Paired gain, pair minus PSBD-TM alone, held-out models | 26 | +0.001 [-0.004, +0.005] | -- | -- |

**Answer.** The winning pair on the selection half is the attention branch output token
mask plus gain scaling of the MLP norm output, and it does not include PSBD-TM. On the
held-out half it reads the same AUROC as PSBD-TM alone (gain +0.001, CI [-0.004, +0.005]) and
a lower TPR at both budgets, so it does not generalize into a gain. This matches the union
result `docs/hypothesis/H41-multi-probe-defense.md` reports: min-rank unions help against an
attacker trained against a single known probe, but on ordinary panel models they add
nothing, and here that finding survives leave-one-dataset-family-out selection. The first
run on 69 models picked PSBD-TM plus PSBD-RD and read a held-out gain of -0.010, CI [-0.025,
+0.003], a historical reading.

### Follow-up: relaxing coverage to let the deployment pair enter the search

**Question.** The first run excluded the pair a deployer would reach for, PSBD-TM plus token
masking on the attention branch output (`before_attention_residual_token_mask`), because
that placement then sat on 66 of 69 models. Does relaxing the candidate rule change which
pair the search picks, and how does PSBD-TM plus the attention branch output read on the
held-out half?

**Method.** Same selection protocol and same search, but the candidate rule admits any
basis placement present on at least 87% of the panel models, the share of the first run's
60 of 69 (`CHECK2B_MIN_COVERAGE`, set to 49 of 56 for the rerun). On the current caches all
23 candidates already sit on every model, so the relaxed search reads the same 23 and picks
the same pair. Beside it, PSBD-TM plus the attention branch output is read on the held-out
models directly.

| Configuration | Models | Mean AUROC | Mean TPR @ 10% | Mean TPR @ 20% |
|---|---:|---:|---:|---:|
| PSBD-TM alone, held-out models | 26 (held-out) | 0.978 | 0.970 | 0.980 |
| PSBD-TM + attention branch output, held-out models | 26 (held-out) | 0.979 | 0.965 | 0.980 |
| Paired gain, PSBD-TM + attention branch output minus PSBD-TM alone, held-out models | 26 | +0.001 [-0.002, +0.004] | -- | -- |

**Answer.** Relaxing the coverage bar changes nothing on the current caches, because every
candidate is already on every model. The deployment pair ties PSBD-TM alone on the held-out
half (+0.001, CI [-0.002, +0.004]) with a slightly lower TPR at 10%. The first run read it at
+0.008 (CI [-0.002, +0.018]) and the search then picked `mlp_norm_out_gain_scale` plus PSBD-RD,
the placement whose standalone `gain_scale` headline of +0.258 is withdrawn
(`docs/audit-2026-09-07.md`). Both readings are historical. On the 56-model panel no pair,
chosen or deployed, beats PSBD-TM alone on CIFAR-100 and Tiny.

## Check 3: the mask seed

**Question.** Every PSBD-TM number elsewhere in this repo comes from 1 draw
of the token-masking sequence (`cli.sweep`'s `PSBD_MASK_SEED = 0`). Is the
method sensitive to that draw?

**Method.** 10 models spread over the 4 main datasets and 6 attacks, read in the first run
of 2026-09-11. 2 of them are not panel models: `vit_cifar100_tact_0_01` is source-mapped and
`vit_cifar10_sig_0_1` fails the 2-point clean-accuracy bar and is under audit
(`docs/audits/2026-09-29-experiment-audit.md`). The 10 include `vit_cifar10_wanet_0_1` and
`vit_cifar100_badnet_a2o_0_01` as named. Each is rerun with `cli.sweep --mask-seed 1` and `--mask-seed 2` at
PSBD-TM's already-selected rate, into
`results/_experiments/reviewer_checks/mask_seeds/`, then `cli.analyze` over
that tree. AUROC and TPR at the 10% quantile are read at seeds 0 (the
canonical cache), 1 and 2 for each model, via
`defenses.decision.detection_report`.

| Model | Rate | AUROC seed 0 | AUROC seed 1 | AUROC seed 2 | AUROC std | TPR@10% seed 0 | TPR@10% seed 1 | TPR@10% seed 2 | TPR std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vit_cifar10_wanet_0_1 | 0.5 | 0.459 | 0.458 | 0.458 | 0.0002 | 0.087 | 0.084 | 0.087 | 0.0018 |
| vit_cifar100_badnet_a2o_0_01 | 0.5 | 0.988 | 0.987 | 0.987 | 0.0006 | 0.999 | 0.998 | 0.998 | 0.0007 |
| vit_cifar10_sig_0_1 | 0.8 | 0.418 | 0.435 | 0.434 | 0.0098 | 0.071 | 0.064 | 0.064 | 0.0040 |
| vit_cifar10_badnet_a2o_0_1 | 0.6 | 0.991 | 0.991 | 0.991 | 0.0002 | 0.935 | 0.939 | 0.938 | 0.0025 |
| vit_cifar100_bpp_0_05 | 0.5 | 0.985 | 0.985 | 0.985 | 0.0004 | 0.962 | 0.960 | 0.960 | 0.0007 |
| vit_cifar100_tact_0_01 | 0.5 | 0.909 | 0.901 | 0.906 | 0.0038 | 0.506 | 0.517 | 0.414 | 0.0567 |
| vit_gtsrb_bpp_0_05 | 0.6 | 0.993 | 0.992 | 0.992 | 0.0001 | 0.978 | 0.978 | 0.978 | 0.0001 |
| vit_gtsrb_wanet_0_1 | 0.4 | 0.948 | 0.947 | 0.947 | 0.0004 | 0.949 | 0.949 | 0.949 | 0.0003 |
| vit_tiny_blend_0_1 | 0.5 | 0.998 | 0.998 | 0.998 | 0.0002 | 0.999 | 0.999 | 0.999 | 0.0001 |
| vit_tiny_badnet_a2o_0_01 | 0.5 | 0.980 | 0.981 | 0.981 | 0.0005 | 0.989 | 0.993 | 0.994 | 0.0023 |
| **Mean across models** | | | | | **0.0016** | | | | **0.0069** |

**Answer.** PSBD-TM's numbers are not an artifact of mask seed 0. The AUROC standard
deviation across seeds 0, 1 and 2 is at most 0.010 on any of the 10 models, and the widest
range is 0.017, on the SIG model (0.418 to 0.435). The mean standard deviation across models
is 0.0016 over all 10 and 0.0014 over the 9 without the source-mapped TaCT model, 2 orders of
magnitude below the PSBD-TM minus PSBD-RD gap of 0.075 on the 56-model panel
(`\HeadlineGainAdaptiveAuroc`). The 1
outlier is TPR at 10% on `vit_cifar100_tact_0_01` (std 0.057), which is a
poison-rate artifact rather than a mask-seed one: at 1% poisoning the
eligible TaCT backdoor pool is small, so a handful of samples crossing the
threshold moves TPR by several points while AUROC on the same model barely
moves (0.901 to 0.909). The 2 inverted models of the first run,
`vit_cifar10_wanet_0_1` and `vit_cifar10_sig_0_1`, stay inverted at every
seed, so that finding is not a seed-0 accident either. The panel's lowest model since
2026-09-30, `vit_gtsrb_tact_0_01_cos`, has no mask-seed replicate yet, which needs a GPU run.
