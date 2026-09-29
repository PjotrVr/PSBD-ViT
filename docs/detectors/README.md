# Competitor detectors

This directory documents the 11 published or null-model input detectors that this project compares PSBD against, 1 page per paper. Each page is written to be read cold: it explains the idea in plain words, gives the paper's own equations with a symbol table, describes the setting the paper evaluated in, walks through our port function by function, lists every deviation from the paper and its released code with the reason, states where every hyperparameter value comes from, says what numerical check ties the port to the reference implementation, prices the method in forward passes and closes with a results block generated from the records on disk. This README holds what all 9 pages share, so a page can say "the shared split" and a reader can look the term up here once.

## The problem every detector here solves

A backdoor attack poisons a small share of a model's training set so that the trained model behaves normally on ordinary images and sends any image carrying a secret pattern, the trigger, to a class the attacker chose, the target. An input-level detector sits in front of such a model at deployment. It receives 1 image at a time and must decide whether that image carries a trigger, knowing neither the trigger, the target class nor whether the model was attacked at all. PSBD is 1 such detector, and it is the method this repository adapts to vision transformers. Every entry in `detectors/` solves the same problem at the same threat model so that a comparison table's columns differ only in the score being thresholded.

Methods that inspect a whole poisoned training set, such as Spectral Signatures, Activation Clustering or SCAn, are out of scope. They need the training pool the defender of this threat model does not hold, and they produce a partition of that pool rather than a decision about 1 incoming image.

## Terms this directory uses

| term | meaning |
|---|---|
| trigger | the pattern an attacker stamps on an image to activate the backdoor, a corner patch for BadNets, a blended image for Blend, a warp for WaNet |
| target class | the class a triggered image is sent to |
| poison rate | the share of the training set the attacker poisoned, 1%, 5% or 10% on the panel |
| all-to-one | every triggered image goes to 1 target class, the label mode of every panel model except the clean-label ones |
| clean-label | the attacker only poisons images already labeled as the target (SIG and Label-Consistent), so the labels look correct |
| ASR | attack success rate, the share of triggered non-target images the model sends to the target |
| ViT-B/16 | the vision transformer every panel model uses, 12 blocks, 768-wide tokens, 16 by 16 pixel patches at 224 by 224, wrapped as `Sequential(Resize((224, 224)), network)` so 32 and 64 pixel images are upsampled inside the model |
| Swin-S | the second transformer the repository trains, 24 blocks in 4 stages of shifted-window attention |
| PSBD | Prediction Shift Backdoor Detection (Li et al., arXiv 2406.05826): perturb the model k times and flag an input whose confidence in its own prediction barely drops |
| PSU | prediction shift uncertainty, PSBD's statistic, defined in `defenses/scores.py` and restated below |
| placement | where PSBD's perturbation is injected (the position) and what it does (the operator), named `<position>_<operator>` |
| PSBD-TM | the placement this project recommends, `token_mask` at `before_attention_norm`: whole tokens zeroed at the input of every attention block |
| PSBD-RD | the placement the PSBD paper used on ResNet, adapted to ViT: dropout on the residual stream after both residual adds of every block (`post_residual`) |
| adaptive rate rule | `defenses.decision.select_rate_adaptively`: the smallest perturbation rate whose clean-validation shift ratio reaches 0.8, the rule PSBD's authors use and the only rule a competitor comparison may use |
| shift ratio | the share of (image, pass) pairs whose perturbed prediction differs from the unperturbed one, on clean validation images |
| the shared split | the 2000 clean test images every detector is given as clean data, built by `data.splits.build_psbd_loaders_from_checkpoint` |
| the panel | the backdoored ViT-B/16 models the paper's detector comparison reports, defined and counted below |
| low means poisoned | the direction convention every score here follows |

## The splits every detector is scored on

`data.splits.build_psbd_loaders_from_checkpoint` shuffles the dataset's full clean test set once with seed `PSBD_SPLIT_SEED` (0), with no stratification by class. The first 2000 images of that permutation are the **validation split**, which is the only clean data any detector may fit on and the only data a threshold may be read from. The rest is the **analysis pool**. The **clean split** is every analysis image as it is. The **backdoor split** is every analysis image the attack is allowed to act on (for all-to-one, every image not already in the target class, and for clean-label attacks, every non-target image, by `attacks.poisoning.is_eval_poisonable`), with the trigger applied. Before any metric is computed, `defenses.decision.pair_clean_to_backdoor` restricts the clean split to exactly the images behind the backdoor rows, so the 2 sides of every comparison are the same photographs with and without the trigger, and a metric cannot reward a detector for noticing which classes were dropped.

Reading the threshold from the validation split and the metrics from the paired clean and backdoor splits keeps the 3 sets disjoint. A detector never sees a triggered image before it is scored, and never sets its threshold on the images it is judged on.

## How a score becomes a number in a table

Every detector returns 1 score per image with low meaning poisoned. `defenses.decision.detection_report` turns 3 score vectors into the table entries, the same function PSBD's own scores go through.

$$
\begin{aligned}
\tau_q &= \operatorname{quantile}_q\big(\{ s(x) : x \in V \}\big) \\
\mathrm{TPR}_q &= \frac{1}{|B|} \sum_{x \in B} \mathbb{1}\big[ s(x) < \tau_q \big] \\
\mathrm{FPR}_q &= \frac{1}{|C|} \sum_{x \in C} \mathbb{1}\big[ s(x) < \tau_q \big] \\
\mathrm{AUROC} &= \Pr\big[ s(x_b) < s(x_c) \big] + \tfrac{1}{2} \Pr\big[ s(x_b) = s(x_c) \big], \quad x_b \sim B,\ x_c \sim C
\end{aligned}
$$

| symbol | meaning |
|---|---|
| $s(x)$ | the detector's score of image $x$, low meaning poisoned |
| $V$ | the 2000-image clean validation split |
| $q$ | the false-positive budget, 1 of 0.01, 0.05, 0.10, 0.15, 0.20 and 0.25 (`PSBD_QUANTILES`) |
| $\tau_q$ | the threshold, the $q$ quantile of the validation scores by numpy's linear interpolation |
| $B$ | the backdoor split, triggered images |
| $C$ | the clean split paired to $B$, the same images untriggered |
| $\mathrm{TPR}_q$ | true positive rate, the share of triggered images flagged at budget $q$ |
| $\mathrm{FPR}_q$ | false positive rate, the share of clean images flagged at the same threshold |
| $\mathrm{AUROC}$ | area under the ROC curve, the probability that a random triggered image scores lower than a random clean one, ties counted as half |

The threshold is a quantile of clean scores, so $q$ is the false-positive rate the defender asked for, and the rate realized on the paired clean split lands near it but not on it, because validation and clean are different images. The tables report TPR at $q = 0.10$ and $q = 0.20$, written "TPR at 10% FPR" and "TPR at 20% FPR". The PSBD paper's own headline quantile is 0.25, and AUROC does not depend on $q$, so the records store it at 0.25 and every quantile carries the same value.

AUROC here is **one-sided**. It is computed on the scores as the detector returns them and never flipped, so a value under 0.5 means the detector ordered the 2 populations the wrong way round on that model. A two-sided value, $\max(\mathrm{AUROC}, 1 - \mathrm{AUROC})$, is stored as `auroc_two_sided` for diagnosis only, because choosing which tail to flag after seeing the result needs the labels the detector exists to predict.

## The panel the results blocks report

`scripts/coverage_ledger.py` writes the coverage ledger `results/coverage/coverage.json`, which lists every trained ViT model. `scripts.paper._common.load_coverage` restricts it to the 4 datasets the paper reports (CIFAR-10, CIFAR-100, GTSRB and Tiny ImageNet). The ledger gives every model 1 class. A model **clears** when its attack success rate reaches the bar `configs/psbd_basis.json` declares (`asr_bar`) and its clean accuracy has not collapsed. A model is **diverged** when its clean accuracy fell below half the benign reference during training, and a TaCT model is **source-mapped** when its poison rate reached the source class's share of the training set, so that every source image was poisoned and the model sends the whole source class to the target with no trigger at all. Diverged and source-mapped models are not backdoor detection problems, so `scripts.paper._common.excluded_folders` lists them and nothing here reads them.

`clearing_cells` keeps the models that clear, and `scripts/paper/tab_detectors.py`'s `fully_covered` then keeps those on which every 1 of the 13 defenses (the 11 detectors here plus PSBD-TM and PSBD-RD) has a reading. Reading every column on 1 population is deliberate, since a mean over whichever models a column happens to cover makes 2 columns incomparable. A clearing model whose records are still queued on the GPU is left out until they land, and the block below names it.

The block is generated by `scripts/detector_doc_results.py` like every results block, so its counts follow the ledger. The per-attack rows of a detector's results block rest on very different model counts, because not every attack clears at every rate, and each row's n column says how many models it averages.

<!-- results:begin -->
Generated by `python scripts/detector_doc_results.py` at commit `19488b81358b06040ba96f361d5061b2981f1f25-dirty` from `results/coverage/coverage.json` and the detector records. The ledger holds 98 ViT models on CIFAR-10, CIFAR-100, GTSRB and Tiny ImageNet.

| ledger class | models | meaning |
|---|---|---|
| `below_bar` | 30 | the attack did not implant strongly enough |
| `clears` | 59 | attack success at or above the bar, clean accuracy intact |
| `diverged` | 1 | clean accuracy below half the benign reference, excluded |
| `source_mapped` | 8 | TaCT sends its whole clean source class to the target with no trigger, excluded |

57 of the 59 clearing models carry a reading from all 13 defenses and form the panel every results block reports. The clearing models left out, and the defenses they lack, are listed below.

| left out | defenses without a reading |
|---|---|
| `vit_gtsrb_lc_0_05_tl1_adv` | all 13, nothing scored yet |
| `vit_gtsrb_tact_0_01_cos` | all 13, nothing scored yet |

Models the ledger excludes as diverged or source-mapped, which no block reads: `vit_cifar100_tact_0_01`, `vit_cifar100_tact_0_05`, `vit_cifar100_tact_0_1`, `vit_cifar10_tact_0_1`, `vit_gtsrb_tact_0_01`, `vit_gtsrb_tact_0_1`, `vit_tiny_tact_0_01`, `vit_tiny_tact_0_05`, `vit_tiny_tact_0_1`.

Compared models per attack and poison rate. A dash means no model of that attack cleared at that rate.

| attack | 1% | 5% | 10% | total |
|---|---|---|---|---|
| BPP | 4 | 4 | 4 | 12 |
| BadNets | 4 | 4 | 4 | 12 |
| Blend | 4 | 4 | 4 | 12 |
| LF | 4 | 4 | 4 | 12 |
| SIG | -- | -- | 1 | 1 |
| TaCT | 1 | 2 | -- | 3 |
| WaNet | -- | 2 | 3 | 5 |

Compared models per dataset. The attacks that cleared differ slightly between datasets, so the per-dataset means of a block average slightly different mixes of attacks.

| dataset | models |
|---|---|
| CIFAR-10 | 17 |
| CIFAR-100 | 12 |
| GTSRB | 14 |
| Tiny ImageNet | 14 |

<!-- results:end -->

## Shared rules of the registry

4 rules apply to every detector in `DETECTOR_NAMES`, stated in `detectors/__init__.py` so that a comparison's columns differ only in the score.

**Direction.** Every detector returns scores where low means poisoned, the convention PSU already follows, so `detection_report` applies unchanged. STRIP, CD-L and confidence's raw statistics are either already low for poisoned (STRIP's entropy, CD-L's mask norm) or negated once at the return (confidence). SCALE-UP, IBD-PSC, TeCo, Beatrix, TED and SentiNet define statistics that are high for poisoned, and each is negated exactly once, at a named function listed in its page's direction section. A sign error is silent, since it produces a well-formed and exactly inverted result, which is why `experiments/preflight/check_signs.py` runs every detector on a synthetic backdoored ViT whose answer is known before any real model is scored.

**Data budget.** Every method that needs clean data receives the same 2000-image validation split and nothing else. Several papers budget clean data differently, and each page states its paper's budget beside what the shared split gives.

**Cost.** Each module states its forward passes per scored input, and `FORWARD_PASSES_PER_INPUT` repeats the count in machine-readable form, counting 1 backward pass as 1.5 forwards. The counts range from 1 to 251, and a comparison table carries the cost beside the accuracy rather than hiding it.

**Out-of-fit validation scores.** SCALE-UP's data-limited variant, Beatrix and TED fit per-sample statistics on the validation split and then read their threshold from that same split. A sample judged against statistics it helped fit looks more typical than a fresh sample would, so the threshold would sit too tight and the realized false-positive rate would overshoot the budget. The 3 methods, named in `CROSS_FITTED`, score each validation image against statistics fitted without it: 2 folds for SCALE-UP (`cross_fitted_validation_scores`), 5 folds for Beatrix (`jackknife_deviations`) and leave-one-out for TED (`leave_one_out_trajectories`). SentiNet also fits on the split but is left out on purpose: its fit is an upper envelope, which moves the threshold and leaves the AUROC unchanged (`sentinet.md`, deviation 7).

## Registry

This is `DETECTOR_NAMES` as `detectors/__init__.py` declares it, 11 entries, with `EXPERIMENTAL_DETECTOR_NAMES` empty. All 11 run in the default job set of `pbs/generate_detector_jobs.py`.

| detector | paper | statistic | clean data | forwards per input | job group | page |
|---|---|---|---|---|---|---|
| `confidence` | none, the null model | negated max softmax | none | 1 | cheap | `confidence.md` |
| `strip` | Gao et al., ACSAC 2019 | entropy under superimposition | 8 images of the shared split, unlabeled | 8 | cheap | `strip.md` |
| `scale_up` | Guo et al., ICLR 2023 | label consistency under pixel amplification | none | 6 | cheap | `scale_up.md` |
| `scale_up_data_limited` | Guo et al., ICLR 2023 | the same, standardized per predicted class | the shared split, labeled | 6 | cheap | `scale_up.md` |
| `ibd_psc` | Hou et al., ICML 2024 | retained confidence under LayerNorm amplification | the shared split, labeled | 6 nominal | cheap | `ibd_psc.md` |
| `ibd_psc_calibrated` | Hou et al., ICML 2024, amplification searched upward | the same at the smallest factor Algorithm 1 accepts | the shared split, labeled | 6 | cheap | `ibd_psc.md` |
| `teco` | Liu et al., CVPR 2023 | spread of corruption breaking points | none | 71 | teco | `teco.md` |
| `cd_l` | Huang et al., ICLR 2023 | L1 norm of the distilled input mask | none | 251 | cd_l | `cd_l.md` |
| `beatrix` | Ma et al., NDSS 2023 | Gram-matrix deviation from class bands | the shared split, unlabeled | 1 | cheap | `beatrix.md` |
| `ted` | Mo et al., IEEE S&P 2024 | outlier score of the nearest-neighbor rank trajectory over depth | the shared split, labeled | 1 | cheap | `ted.md` |
| `sentinet` | Chou et al., IEEE S&P Workshops 2020 | residual above the clean envelope in the (confidence, fooled) plane | 100 images plus the shared split | 202 | sentinet | `sentinet.md` |

`ibd_psc_calibrated` is the 1 registered variant of a faithful port. It exists because the smoke of 2026-09-10 showed that the paper's amplification factor leaves a ViT's predictions intact through every LayerNorm (`ibd_psc.md`). The faithful `ibd_psc` stays registered so the paper's own setting is on record. Its nominal 6 forwards shrink to 2 on most panel models, for the reason `ibd_psc.md` gives under cost.

BaDExpert (Xie et al., ICLR 2024) was listed here earlier as a pending entry. No module, no registration and no page exists for it, so it is not part of the registry. `docs/paper-proposals.md` names it as the strongest baseline this project does not have.

## Clean-data budgets

The shared split gives every class-conditional method very different amounts of data per class depending on the dataset, and it falls short of several papers' own budgets.

| dataset | classes | images in the shared split | per class, on average |
|---|---|---|---|
| CIFAR-10 | 10 | 2000 | about 200 |
| CIFAR-100 | 100 | 2000 | about 20 |
| GTSRB | 43 | 2000 | about 46 |
| Tiny ImageNet | 200 | 2000 | about 10 |

| detector | the paper's own budget |
|---|---|
| STRIP | 100 held-out images per input, later reduced to 10 |
| SCALE-UP, data-limited | 100 benign images per class |
| IBD-PSC | 100 benign images in total |
| TeCo | none for the score |
| CD-L | 1% of the clean training set for the threshold |
| Beatrix | 30 clean images per class, 8 shown to suffice |
| TED | 20 per class on CIFAR-10, 1000 in total on GTSRB, capped at 1000 in the notebook |
| SentiNet | 100 benign images, plus about 400 for the decision boundary |

CIFAR-100 and Tiny ImageNet, the 2 datasets this project treats as primary, are where the shared split falls furthest below a paper's per-class budget: about 20 per class against SCALE-UP's 100 on CIFAR-100 and about 10 per class against Beatrix's 30 on Tiny ImageNet. Each affected page states the fallback its detector takes below a minimum per-class count.

## Numerical cross-checks against the reference implementations

The project rule is that every port carries a numerical cross-check against the reference implementation cloned under `third_party/`. The table records what exists in `tests/` today and what it compares. `third_party/` is untracked and is not checked out in this working tree, so the 2 tests that execute reference code skip here (`pytest -rs` on 2026-09-29 reported 70 passed and 2 skipped).

| detector | what is checked, and against what | test | status here |
|---|---|---|---|
| `confidence` | nothing to check, there is no reference | none | not applicable |
| `strip` | only the sign, on the synthetic fixture | `experiments/preflight/check_signs.py` | no numerical cross-check exists |
| `scale_up`, `scale_up_data_limited` | nothing, the fixture cannot judge a 6-valued statistic | none | no numerical cross-check exists |
| `ibd_psc`, `ibd_psc_calibrated` | the omega search logic with a stubbed Algorithm 1 | `tests/test_detectors_ibd_psc.py` | runs, but compares nothing against BackdoorBox |
| `teco` | only the sign, with 4 of 14 corruptions | `experiments/preflight/check_signs.py` | the per-operator agreements `teco.md` quotes have no committed test |
| `cd_l` | mask norms against the authors' `CognitiveDistillation` class, bit for bit | `tests/test_detectors_cd_l.py::test_mask_norms_match_the_released_class_bit_for_bit` | skipped, `third_party/CognitiveDistillation` absent |
| `beatrix` | deviations against the authors' `Feature_Correlations` class to 1e-5 | `tests/test_detectors_beatrix.py::test_matches_the_official_feature_correlations_to_1e_5` | skipped, `third_party/Beatrix` absent |
| `ted` | the rank against a numpy transcription of notebook cell 14, to the integer | `tests/test_detectors_ted.py` | runs, the transcription lives in the test file |
| `sentinet` | Grad-CAM against an analytic map, the transplant against a per-image loop, the envelope against a known parabola | `tests/test_detectors_sentinet.py` | runs, against constructions in the test file rather than a reference clone |

## Results blocks and how they are generated

Every page ends with a `results:begin` and `results:end` pair of HTML comment markers. `python scripts/detector_doc_results.py` replaces what sits between them from the records on disk, so no number inside a block is typed by hand. It reads the panel through `scripts.paper.tab_detectors.fully_covered`, each detector's `results/<folder>/detectors/<name>_metrics.json` and each model's `results/<folder>/psbd_metrics.json` for PSBD-TM and PSBD-RD at the adaptive rule, and it writes 5 things per detector: a summary with the rank among the 13 defenses and the paired gap to PSBD-TM with a 95% bootstrap interval, the means per attack and poison rate, the means per dataset, the settings the detector fitted per model and the measured wall-clock cost. `--dry-run` prints the blocks instead of writing them. The paired gap for model $i$ is $d_i = \mathrm{AUROC}_i^{\text{PSBD-TM}} - \mathrm{AUROC}_i^{\text{detector}}$, its point estimate is the mean of $d_i$ over the panel's models, and the interval takes the 2.5th and 97.5th percentiles of that mean over 5000 resamples of the models with replacement, seed 0, by `scripts.paper._common.bootstrap_ci`. The models are resampled in the order `tab_detectors.build_rows` walks them, so a competitor's interval equals the one the paper's detector table prints for the same pair. The PSBD-TM minus PSBD-RD interval can differ from `paper/headline.tex` in the third decimal, because the generator behind that macro walks the same models in another order and a seeded bootstrap depends on the order.

`python -m cli.compare detectors --per-detector-dir docs/detectors` also writes into these markers, in a per-model layout without the per-attack breakdown. The 2 writers overwrite each other, and `scripts/detector_doc_results.py` is the one these pages are generated by. The combined table that command writes with `--markdown`, `docs/detectors/comparison.md`, is not in the tree at the time of writing.

## Where records land and how to regenerate them

`python -m cli.baselines --checkpoint-folder <folder>` scores every detector in `DETECTOR_NAMES` on 1 model's PSBD splits and writes 1 record per detector, `results/<folder>/detectors/<name>_metrics.json`, beside the per-split score tensors `<name>_scores_{validation,clean,backdoor}.pt`. The record holds `detection` (the report at every quantile over all triggered images), `detection_captured_only` (the same over the triggered images the model actually sent to the target, with their paired clean images) and a `provenance` block with the commit, device, precision, batch size, forward passes per input, hyperparameters, fitted settings, runtime per stage, split sizes and a hash of the split manifest checked against the PSBD cache. `--all` runs every folder that has a PSBD cache, `--detectors` narrows the set, `--skip-existing` skips a detector whose record is already `scored` and `--max-samples` truncates every split for a smoke run, which requires a `--results-dir` outside `results/` so a truncated record can never be mistaken for a full one.

`pbs/generate_detector_jobs.py --dry-run` lists the panel's unscored (model, detector) pairs in 4 cost groups, `cheap`, `teco`, `cd_l` and `sentinet`, packed into jobs sized by `--hours`. Without `--dry-run` it writes `pbs/psbd_detectors/<group>_<index>.pbs`. The acceptance smoke every port passed before its panel jobs were written is recorded in `docs/runs/2026-09-10-detector-smoke.md`.

## Pages

| page | detectors |
|---|---|
| `confidence.md` | `confidence` |
| `strip.md` | `strip` |
| `scale_up.md` | `scale_up`, `scale_up_data_limited` |
| `ibd_psc.md` | `ibd_psc`, `ibd_psc_calibrated` |
| `teco.md` | `teco` |
| `cd_l.md` | `cd_l` |
| `beatrix.md` | `beatrix` |
| `ted.md` | `ted` |
| `sentinet.md` | `sentinet` |

`notebooks/competitor-defenses.ipynb` walks through the same 11 detectors with a diagram of what each perturbs, the source of its scoring function and its per-attack AUROC beside PSBD-TM and PSBD-RD.
