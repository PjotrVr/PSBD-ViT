# Open questions

What this cleanup window found and could not close. Every entry names the evidence,
what would settle it, and whether a published number depends on it. A question here
is not a bug report. It is a claim the repository currently cannot support at the
strength something in it asserts.

Opened 2026-09-23 during the no-compute cleanup window. Every number was brought to the
56-model panel of the 2026-09-30 build (successful backdoors at the 2-point clean-accuracy bar), read from `paper/headline.tex` or
recomputed by the scripts under `experiments/doc_recomputes/`. A number that could
not be recomputed says so and names the date it was measured.

## Status after the 2026-09-23 evening pass

Read this table first. It says which questions below are closed, which are only
disclosed in the paper, and which are still open, so nobody re-derives a settled
one.

| # | Status | What changed |
|---|---|---|
| Q1 | Closed | The introduction says the gain is largest at the lowest poison rate, from `\GainsLowestRate` |
| Q2, Q3 | Partly closed | The k = 20 curve reads its population off disk, 19 cells in a rebuild of `scripts/paper/fig_forward_passes.py` on 2026-09-29 (the tracked `\PassPilotCells` reads 17 and grows as k = 20 jobs land), and the AUROC gain past k = 3 is +0.002 there (`\PassPilotAurocGainBeyondThree`). More cells are queued as GPU jobs (`pbs/vit_k20_panel/`) |
| Q4, Q24, Q29 | Closed | The panel is 104 declared cells over 4 datasets, 65 of them clearing the ASR bar, 62 successful at the 2-point clean-accuracy bar (`\PanelCellsSuccessful`) and 56 compared, with SVHN and EuroSAT declared out in `configs/psbd_basis.json` (`\PanelCellsTotal`, `\PanelCellsClearing`, `\PanelCellsCached`). Every headline macro is rebuilt from the current tree |
| Q5 | Closed | All 27 declared placements are measured. The 27th was declared under a cache name no sweep writes, `before_attention_residual_dropout`, and `tests/test_canon.py` now holds every id to its cache name |
| Q6 | Closed | The detector count is a macro read from the registry |
| Q7, Q8 | Closed | Neither record was missing. Commit 2aea959 moved both under `results/_experiments/`, 2 generators kept the old path, and `experiment_artifact` now resolves either layout |
| Q15, Q20, Q21 | Disclosed, not resolved | The appendix states that the ViT selection half picks the twin, by -0.025 over 31 models (`\GainsSelectionHalfTwinGap`), and that the recommendation rests on the Swin selection half (+0.166 over 44, `\SwinSelectionHalfTwinGap`), seed stability and the mechanism. The Q21 claim is removed |
| Q19 | Closed | The staircase reads Gaussian noise on both sides of the MLP LayerNorm and the finding is stated as a LayerNorm-side effect |
| Q22 | Closed | The paper no longer says a benign reading checks a sign convention |
| Q23 | Disclosed | The paper reports SentiNet's below-chance reading and says the stated mechanism explains chance, not the sign |
| Q32 | Closed | `method.tex` carries the fractional PSU equation |
| Q17, Q27 | Disclosed, not resolved | The limitations state the false-positive rate PSBD-TM realizes on clean test images at the nominal 10% budget, `\RealizedFprOneZero` (0.093) on average and `\RealizedFprOneZeroMin` to `\RealizedFprOneZeroMax` (0.000 to 0.116) over models, and the share of flags that are triggered at a 1% prevalence, `\PrecisionOneZero` (0.09), all from `tab_headline.py`. `\HeadlineTprAtOnePercent` reads mean TPR 0.714 at the 1% budget and no section prints it, and PSU's own `threshold_diagnostics` is still not called |
| Q9 to Q14, Q16, Q18, Q25, Q26, Q28, Q30, Q31, Q33 | Open | Untouched this pass |
| Q36 | New, closed | A CPU node runs float32 where every GPU cache is bfloat16, because `defenses.inference` enters autocast only on CUDA. `sw_parity` read the same Swin cell both ways on 9 shared rates: the adaptive rule picks the same rate and reads 0.996 on both, and no rate differs by more than 0.0019 AUROC. The 10 CPU-swept Swin cells are pooled (`docs/runs/2026-09-23-cpu-timing.md`) |
| Q37 | New, closed | On the 57-model panel PSBD-TM leads the calibrated IBD-PSC port by +0.032 [+0.007, +0.057] AUROC, +0.082 [+0.021, +0.146] TPR at 10% FPR and +0.033 [-0.016, +0.085] at 20% FPR (`\DetectorsAurocMargin`, `\DetectorsTprOneZeroMargin`, `\DetectorsTprTwoZeroMargin`). The tie read on 69 models on 2026-09-23 no longer holds, and only the 20% FPR interval contains 0 |
| Q38 | New, disclosed | Token masking restricted to blocks 1 to 4 reads 0.932 and ranks 5th of 27 on the 12 models that reach the adaptive target (`paper/tables/basis_ranking.tex`), and is the worst band on the 56 models read at matched disturbance, -0.094 [-0.130, -0.058] against all blocks (`\BandOneFourMinusAllInputSide`). `app:bands` says to read a band's n beside its mean |
| Q39 | New, closed | The paper said PSBD-TM stays at or above 0.9 on BadNets, Blend, LF and BPP, which on 2026-09-23 held only for BadNets and LF. `tab_gains.py` now computes the list, which on the 57-model panel reads BadNets, Blend, LF and TaCT (`\GainsStrongAttacks`) |
| Q40 | New, closed | The weight-detector reproduction said 4 benign models, all 5 WaNet models and BPP named only on CIFAR-10. The record holds 3 benign and 4 WaNet models, and BPP is also named on CIFAR-100 at 1%. `app_weight_detector.py` now writes those counts |
| Q41 | New, open | PSBD-TM's TaCT failure holds only for single-source TaCT. The 6 multi-source models read TPR 0.68 to 0.99 at 10% FPR, so the registered prediction that the plain method misses them failed (`experiments/tact_calibration/PREDICTIONS.md`, P2), and so did the blank-carrier refinement (`experiments/evidence_surplus/sufficiency/PREDICTIONS_tact.md`). The content-hidden reading separates the 2 groups but was chosen after the data. Settled by fresh TaCT models with 2 to 4 source classes, their content-hidden excess predicted before it is measured |
| Q42 | New, open | Per-class calibration of the PSU threshold (`experiments/tact_calibration/`) reads large gains on the hard development models and, in an interim read on 2026-10-02, gains on CIFAR-10 and GTSRB and losses on CIFAR-100 and Tiny at 1% FPR. The full 63-model read is pending, and the run uses k = 20 and enlarged validation sets, so it is not comparable to the headline. Nothing enters the final method before a rule fixed in advance is tested on Swin |
| Q43 | New, open | In the original paper's training-set setting the final method leads PSBD-TM and PSBD-RD on 5 pooled models (`experiments/training_set_detection/`), but 2 GTSRB models failed or were ambiguous on poisoned-index reconstruction, 2 models and the STRIP and CD-L baselines are still to run, and every verdict there is provisional |
| Q44 | New, disclosed | BackdoorBench's Input-Aware checkpoint does not reproduce its reported attack success (0.797 against 0.923) and holds no trigger generator, and LIRA ships no checkpoint. Both are recorded as not reproduced in `experiments/backdoorbench_attacks/`, whose models we did not train |

## Numbers a document asserts and the data does not support

| # | Claim | Where | What the data says | Settles it |
|---|---|---|---|---|
| Q1 | The gain is largest at 10% poisoning | `paper/sections/introduction.tex`, contribution 1 | `results.tex` reports the gain growing as the rate falls, +0.012 at 10% over 18 models against +0.111 at 1% over 18 (`\GainsHighestRate`, `\GainsLowestRate`), and `conclusion.tex` says 1% | Correct the introduction. No measurement needed |
| Q2 | The k sweep to 20 ran on 6 models | `paper/sections/robustness.tex` | `fig_forward_passes.py` read 8 pilot cells with a k=20 cache on 2026-09-23 and 19 in a rebuild on 2026-09-29 (the tracked `\PassPilotCells` reads 17 and grows as k = 20 jobs land) | Print the count from the macro |
| Q3 | 3 passes are enough | `paper/sections/robustness.tex` | On all 57 models the mean AUROC reads 0.941, 0.950 and 0.953 at k = 1, 2 and 3 (`\PassAllAurocKOne` to `\PassAllAurocKThree`), and on the 19 models swept to k = 20 in the rebuild of 2026-09-29 it gains +0.002 past k = 3. The full panel has not been swept past 3 | A k sweep beyond 3 on the full panel, or state the scope |
| Q4 | The panel spans 4 datasets and 7 attacks | `paper/sections/abstract.tex` | On 2026-09-23 the ledger held 105 cells over 6 datasets. The panel is now 98 cells over 4 datasets and SVHN and EuroSAT are declared out (`\PanelCellsTotal`, `\PanelDatasets`). 8 of the 9 declared attacks clear the bar somewhere (`\PanelAttacksClearing`) | Decided: the 4-dataset panel |
| Q5 | The basis holds 18 placements | `paper/sections/appendix.tex`, `.claude/CLAUDE.md` | `configs/psbd_basis.json` declares 27 and all 27 carry a measured adaptive mean (`\BasisMeasuredSize`). 23 of them reach the adaptive target on all 56 models, and the 4 depth-band token-mask and dropout placements reach it on 12 to 41 (`paper/tables/basis_ranking.tex`) | State 27 and print each placement's n |
| Q6 | 10 competitor detectors | `paper/sections/robustness.tex` | `detectors.DETECTOR_NAMES` registers 11 | Count from the registry |

## Claims whose supporting artifact is missing

| # | Claim | Missing input | Consequence |
|---|---|---|---|
| Q7 | The causal direction-ablation result, finding F12, graded STRONG | `results/backdoor_neuron_ablation.json` | `paper/tables/direction_ablation.tex` and 10 macros cannot be regenerated. The paper's flagship mechanism table is hand-typed and unverifiable |
| Q8 | PSU is not a restatement of confidence, findings F01 and F10 | `results/psu_vs_confidence.json` | `paper/tables/confidence_null.tex` and 5 macros cannot be regenerated |

Both rows are closed. Neither record was missing: commit 2aea959 had moved both under
`results/_experiments/`, and `experiment_artifact` now resolves either layout, so the
tables and macros regenerate (the Q7, Q8 row of the status table).

## Integrity of the record

| # | Issue | Scale | What it costs |
|---|---|---|---|
| Q9 | Checkpoints trained with no commit recorded | 546 of 1922 `args.json` carry `git_commit: null`, 333 more carry a `-dirty` tree (recounted 2026-09-29, `experiments/doc_recomputes/integrity_counts.py`) | No path from those weights back to the code that made them |
| Q10 | Sweeps recorded against a dirty tree | 2710 of 23726 provenance records (recounted 2026-09-29) | Those readings cannot be reproduced exactly |
| Q11 | 1 sweep commit is unreachable from any branch | `f402dba2`, referenced by 1 record under `results/swin_cifar100_badnet_a2o_0_1/`, still on no branch on 2026-09-29 | The record dies at the next `git gc` |
| Q12 | Orphaned gaussian provenance records | 3611 across 366 result folders on 2026-09-23, provenance written with no tensors beside it. Not recounted | 366 cells look swept and are not |
| Q13 | Truncated rate ladders | 14 placement directories on 2026-09-23, 8 of them `before_mlp_gaussian` on `*_evade_l1`. Not recounted | A job died mid-rate. Those placements read on a partial ladder |
| Q14 | Every paper artifact was generated from a dirty tree | on 2026-09-29 all 81 tables and 11 figure sidecars carry a `-dirty` commit, across 4 different commits (recounted by the same script) | The paper is not 1 coherent build |

## Coverage the panel declares and does not have

6 of the 62 successful cells carry no sweep yet, which is why the headline reads on 56
rather than 62 (`\PanelCellsAwaitingSweep`). They are the TaCT multi-source retrains
(`_src{k}`), queued on GPU to replace the 8 source-mapped cells
(`docs/runs/2026-09-24-tact-multisource.md`).

The 2 cells that lacked a sweep on 2026-09-29, `vit_gtsrb_tact_0_01_cos` (the
cosine-schedule rerun of the 1 diverged cell `vit_gtsrb_tact_0_01`) and
`vit_gtsrb_lc_0_05_tl1_adv`, now carry one. The first is the floor of the PSBD-TM
results at 0.266 (`\HeadlineFloorAuroc`).

The 8 source-mapped TaCT cells are excluded rather than missing.
The SVHN and EuroSAT cells that the 2026-09-23 version of this section listed are out of
the panel by declaration.

## Selection and reporting protocol

| # | Question | Why it matters |
|---|---|---|
| Q15 | The recommended placement is chosen partly on seed stability and Swin transfer, both measured on the models the paper reports on | `app:protocol` declares selection on CIFAR-10 and GTSRB and reporting on CIFAR-100 and Tiny ImageNet. 2 of the 3 stated reasons break that split |
| Q16 | 27 placements compared with no multiplicity control | 1 confirmatory comparison should be declared and the rest marked exploratory |
| Q17 | The headline operating point is TPR at 10% FPR | A detector that rejects 1 clean input in 10 is not deployable. The security literature asks for 1% or below, where PSBD-TM reads mean TPR 0.714 (`\HeadlineTprAtOnePercent`) |
| Q18 | `gain_scale` at `mlp_norm_out` is recommended as probe 3 of the union while its own headline stands withdrawn | A reader is told to deploy a placement whose result was retracted, with no pointer to the retraction |

## Methodology defects found by audit on 2026-09-23

A statistical audit re-derived the headline numbers from `results/` on 2026-09-23 and
checked the method against the code. The most important check passes. The paired
bootstrap is correct, the paired difference is the mean of per-cell differences rather
than a difference of means, and the gain survived every re-derivation on the panel of
that day. On the current 56-model panel it reads +0.075 [+0.017, +0.138]
(`\HeadlineGainAdaptiveAuroc`). What follows is what did not pass, each row brought to
the current panel where it could be recomputed.

### Claims not licensed as written

| # | Claim | Defect | Fix, and whether it needs compute |
|---|---|---|---|
| Q19 | The operator effect reverses sign between the attention input and the MLP input, so the axes separate | The 2 legs are not the same comparison. Leg 1 is token masking against Gaussian noise both injected before a LayerNorm. Leg 2 is token masking before a LayerNorm against Gaussian noise after one. Held on the same side, the operator effect is the same sign at both sites and the site by operator interaction read +0.042 [-0.031, +0.112] on 2026-09-23, which contains 0 (not recomputed on the current panel). What reverses is the side of the LayerNorm, not the site | `before_mlp_norm_gaussian` is already cached on 417 result folders. Swap it into leg 2 and restate the finding as a LayerNorm-side effect. No compute |
| Q20 | PSBD-TM is the placement the declared protocol selects | `configs/psbd_basis.json` declares selection on CIFAR-10 and GTSRB and reporting on CIFAR-100 and Tiny. Against its twin at the attention branch output, PSBD-TM reads -0.025 on the 31 selection-half models and +0.034 on the 26 reporting-half models (`\GainsSelectionHalfTwinGap`, `\GainsReportHalfTwinGap`, intervals not recomputed on the current panel). Applied literally the protocol selects the twin, and the pooled near-tie averages a loss against a win across exactly the declared split | The Swin data that would license the choice on selection-half cells is already in `results/`, where PSBD-TM beats the twin by +0.166 over the 44 selection-half models (`\SwinSelectionHalfTwinGap`). Add a twin row to the Swin table. No compute |
| Q21 | The lead is not a product of the selection, because PSBD-TM reads higher on the held-out half than overall, now 0.978 over 26 models against 0.951 (`\GainsReportHalfOurs`, `\HeadlineAurocAdaptive`) | Selection optimism concerns the ranking rather than the winner's absolute level, and reading higher on the reporting half is the signature of the problem rather than evidence against it | Replace with the re-selection experiment: rank on the selection half, take that winner, report what it gives up on the reporting half. No compute |
| Q22 | Every detector reads chance on the benign model, which checks its sign convention | A benign model reads about 0.5 under either sign, so the test cannot detect the failure it claims to catch | Drop the claim or replace it with the backdoored-model reading |
| Q23 | SentiNet's transplant carries nothing and reads below chance, now mean AUROC 0.421 over 56 models (`\DetectorsAurocSentinet`) | A statistic that carries nothing reads 0.5. Two independent below-chance readings in the same direction are an informative statistic with the wrong sign, and the port has exactly 1 negation, so the stated explanation cannot produce the number | Investigate the envelope extrapolation in `boundary_residual`, or report the reading as unexplained |

### Reporting that is narrower than stated

| # | Issue | Detail |
|---|---|---|
| Q24 | The panel spans 4 datasets, not 6 | The 56 compared models are CIFAR-10 15, CIFAR-100 12, GTSRB 15 and Tiny 14 (`\GainsCifarOneZero` and its siblings), with SVHN and EuroSAT declared out. `\PanelDatasets` is now 4 and `\PanelDatasetsDeclared` 6. Fixed |
| Q25 | The matched rule is not matched for the placement it compares against | `select_rate_at_matched_shift` returns the nearest rate and never refuses a cell. On the 56-model panel the recommended placement achieves a clean-validation shift ratio of 0.590 on average at the matched rung, while the published placement achieves 0.618 and is off target by more than 0.10 on 28 of 56 models, because its ladder jumps across the target between adjacent rates. Read at the interpolated 0.6 point instead, the paired gain (both placements at the matched rule) moves by 0.005, from +0.067 to +0.072 (`experiments/doc_recomputes/matched_rule_and_fpr.py`). `interpolate_at_target_shift` exists for exactly this and has no consumer outside the tests |
| Q26 | A band table's AUROC column cannot be differenced | In `paper/tables/staircase_bands.tex` the n and AUROC columns are over each placement's own coverage while the gain column is over the intersection. On 2026-09-23 a reader differencing the AUROC column got -0.052 against a true paired -0.063, and the blocks 1 to 4 row covered 12 cells where the all-blocks placement read 0.977 against a panel mean of 0.928. On the current panel the blocks 1 to 4 token mask still reaches the adaptive target on 12 models only, where the all-blocks placement reads 0.977 against its panel mean of 0.953, so differencing the AUROC column gives -0.029 against a true paired -0.053 (`experiments/doc_recomputes/band_subpanel.py`). The prose is correct, the table is not |
| Q27 | TPR at a fixed FPR averages TPRs measured at different FPRs | The threshold is a quantile of 2000 validation scores and the FPR is realized on a different sample, so it is a random variable. The median tracks the nominal rate, but on the 56-model panel the realized FPR of PSBD-TM at the headline quantile 0.25 spans 0.009 to 0.288 with a mean of 0.231, a 32-fold range (`experiments/doc_recomputes/matched_rule_and_fpr.py`). `threshold_diagnostics` returns `tie_share_at_threshold` and `tpr_interpolated` and is called for the competitor detectors but never for PSBD itself |
| Q28 | The deployable rule selects no rate at all on some clearing cells | On 2026-09-23 the recommended placement's ladder never reached the 0.8 target on `eurosat sig 5%` and `svhn sig 10%`, both now out of the panel. On the 57-model ViT panel it reaches the target everywhere (`\LadderAurocAtZeroEightN`), and on Swin it misses on 3 models (`\SwinShiftTargetUnreached`), which the paper does not report |
| Q29 | The headline macros no longer reproduced from the results tree on 2026-09-23 | The gap came from SVHN and EuroSAT cells entering the recomputation. With those datasets declared out, recomputing the adaptive readings of both headline placements over the 57 models gives 0.953 and 0.888, the values `paper/headline.tex` carries (`experiments/doc_recomputes/matched_rule_and_fpr.py`) |
| Q30 | No multiplicity control over a family of 27 placements | The headline comparison is pre-declared and owes no correction. The ranking of 27 is exploratory and is not labeled as such. On the 56-model panel the twin at the attention branch output leads PSBD-TM by +0.002 [-0.025, +0.030] (`\StaircaseOperatorsBeforeAttentionResidualTokenMaskGain`), already indistinguishable from 0, so a correction would change no conclusion and only the honesty of the ranking table |
| Q31 | Every bootstrap interval reuses seed 0 | Each interval is individually valid, and because the resample indices depend only on the seed and the sample size, all comparisons at the same n share identical resamples. Monte Carlo noise is therefore common-mode across the paper and 2 intervals are not independent evidence |

### Missing from the paper

| # | Item |
|---|---|
| Q32 | The headline statistic has no equation. `background.tex` writes the absolute PSU and `method.tex` describes the fractional form in prose, so the symbol denotes the absolute drop where it is defined and the fractional quantity everywhere it is used. The implementation in `defenses/scores.py` is correct and the fractional form is this project's own contribution rather than the source paper's, which makes writing it down more important rather than less |
| Q33 | A dead guard states a failure that cannot occur. `defenses/scores.py` clamps the tracked probability at 1e-6, but the tracked class is the argmax so its probability is at least 1 over the label size, which is 0.005 on the largest panel dataset, Tiny ImageNet with 200 classes |

## Settled on 2026-09-23, so nobody re-derives them

The verdicts below are dated. Their numbers were measured on the panel of 2026-09-23 and
were not recomputed, since neither verdict depends on the panel size.

| # | Claim | Verdict |
|---|---|---|
| Q34 | Dropping the modal perturbation attractor from the clean validation split before reading the quantile threshold is a free detection gain | **Refuted.** It raises TPR by 0.013 on average and raises the realized false-positive rate by the same trade. Against a plain quantile threshold chosen to realize the same clean FPR, the advantage is -0.0004 on average and 0.0000 at the median over 706 cells, winning on 4 and losing on 10. The whole effect was threshold loosening |
| Q35 | A low-confidence or low-margin backdoor evades PSBD | **Refuted for the fractional statistic, and it points the wrong way.** The headline statistic divides the unperturbed confidence out, so under the masking operators the margin cancels and the statistic is margin-monotone increasing. Lowering the margin lowers the score, and low is the flagged side. Measured directly, baseline confidence explains 0.020 of the statistic's variance at the recommended placement. The literature's low-confidence results are against STRIP, which superposes inputs rather than perturbing the model, so they do not transfer. `docs/attack-design/README.md`'s claim that confidence matching removes about a tenth is generous at this placement |

Q34's underlying observation is real and it explains Q27. The images that land on
the modal attractor sit at low fractional PSU, so they drag the validation
quantile down and the realized false-positive rate undershoots the budget the
defender asked for. That is a calibration bug worth fixing on its own terms.
