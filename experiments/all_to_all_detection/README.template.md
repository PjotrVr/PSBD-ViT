# Prediction-shift detection of all-to-all backdoors

## Question

PSBD flags an input whose prediction survives a perturbation of the model, and on all-to-all BadNets (label `(y + 1) mod K`) that premise inverts: triggered predictions move more than clean ones, so PSU scores below chance ([H5](../../docs/hypothesis/H5-all-to-all-breaks-psbd.md), [the inversion record](../../docs/all-to-all-inversion.md)). [H43](../all_to_all_entropy/README.md) showed that the entropy of the unperturbed softmax separates the triggered inputs, but entropy perturbs nothing and reads confidence. The question here is whether any detector of the PSBD kind (perturb the model at inference, read how the prediction moves) works on all-to-all ViT-B/16 and Swin-S models, and whether it can be designed from a measured mechanism rather than picked from a list.

The answer has to hold without knowledge of the mapping. Our models were trained with the rotation `(y + 1) mod K` for simplicity, and a real attacker can use any permutation or a mapping that is not a bijection. So the rotation, the source class and the attack label are used only in the mechanism analysis (`mechanism.py`, `tokens.py`), which says so in every docstring. The detection statistics (`detect.py`) read the unperturbed softmax, the class each perturbed pass predicts and the same quantities on the clean validation split, and nothing else.

## Scope and limits

BadNets all-to-all is the only all-to-all attack trained here, and every model shares 1 mapping family, the rotation by 1. Generalization to other all-to-all attacks (a blend or warping trigger, a label-consistent variant) and to other mappings is untested. The relabeling test below shows that no statistic reads a class identity, which is necessary for mapping invariance and says nothing about how a different attack would behave. This directory touches no file under `paper/`, and the paper does not mention all-to-all.

## Panel

The all-to-all models are not in the coverage ledger, so `panel.py` applies the ledger's rule by hand with the ledger's own bars. A model is a successful backdoor when its ASR is at least @@asr_bar@@, it has not diverged (clean accuracy at least half the benign reference) and its clean accuracy is within @@points_2@@ points of the benign reference of its architecture and dataset. All-to-all BadNets on CIFAR-100 and Tiny never reaches @@asr_bar@@ on either architecture, so every number is also read at a relaxed ASR bar of @@relaxed_bar@@, chosen before any detection number was read and labeled wherever it is used. All-to-all chance is 1 in K, so a model at @@relaxed_bar@@ is still a working implant. The same rule at @@points_5@@ points admits 1 more model, read in its own table below.

@@panel_table@@

SAM and evasion variants (trained against PSBD-TM) are kept out of the main reading and scored apart.

@@side_table@@

## Honesty protocol

The direction of every statistic (low means poisoned) is written in the docstring of `detect.py` with its reason, before any AUROC was read. Statistics were developed on @@development@@ all-to-all models, and the choice of detector was frozen in `detect.py` (`CHOSEN_DETECTOR = "@@chosen@@"`) with `detection_development.json` as the record of what was known at that moment. The @@confirmation@@ models were then scored once. The all-to-one panel (the ledger's successful models, all 4 datasets) and the benign references were visible during development, since they are the cost check and the null control rather than all-to-all data. `mechanism.py` ran on every dataset before the choice. It reads no detection statistic, and its pooled tables were the only view of the confirmation models before they were scored.

AUROC compares each triggered test image with its own clean twin (`pair_clean_to_backdoor`). TPR is read at thresholds set at the @@budgets@@ quantiles of the clean validation scores, with the realized false-positive rate on the clean test twins beside it. Intervals are bootstrap intervals over models with @@resamples@@ resamples and seed @@seed@@. Every gain is paired against PSU-TM on the same models.

`detect.relabeling_check` runs on every model scored. It permutes the class indices of every softmax and every argmax the statistics see, validation included, recomputes every statistic and asserts that no per-input score moves. `test_invariance.py` runs the same check on synthetic caches, shows that it catches a statistic reading the probability of class 0, and asserts that no function on the statistic path names a label field. The defender views passed to the statistics carry no label field, so a statistic that tried to read one would fail with a missing key.

## Fragility of the triggered prediction

The first mechanism question is whether the inversion is general. For every image whose clean twin the model classifies correctly and whose triggered twin the backdoor captures, `mechanism.py` counts how often each prediction moves under each cached probe. PSBD-TM and PSBD-RD are read at their adaptive rate, the depth bands at the rate whose clean validation shift ratio is nearest 0.6. The Spearman column correlates, over images, how often the clean twin moves with how often the triggered twin moves.

ViT-B/16:

@@fragility_vit@@

Swin-S:

@@fragility_swin@@

On ViT the triggered all-to-all twin moves more often than its clean twin under every probe in the table, and on Swin-S under every probe except PSBD-TM at its adaptive rate, where nearly every prediction moves (@@m_swin_tm_trig@@ against @@m_swin_tm_clean@@). On all-to-one the triggered twin moves less under every probe. Under ViT PSBD-TM the all-to-all gap is small (@@m_tm_trig@@ against @@m_tm_clean@@) for the same reason. The contrast is sharpest when only the last blocks are perturbed. Dropout on the residual stream of blocks 9 to 12 moves @@m_late_trig@@ of triggered predictions and @@m_late_clean@@ of clean ones, and on Swin-S blocks 17 to 24 move a small share of triggered predictions and almost no clean ones (@@m_swin_late_trig@@ against @@m_swin_late_clean@@). The per-image correlation is weak, so how fragile an image's clean prediction is barely predicts how fragile its triggered prediction is (Spearman @@m_tm_spearman@@ under PSBD-TM).

![fragility](figures/fragility.png)

## Destinations of shifted triggered predictions

The prediction from the brief was that a triggered prediction, once it breaks, falls back to its source class. `mechanism.py` counts the shifted passes that land on the source class (attacker knowledge), and as a control how often shifted clean passes land on the class 1 below their prediction, the same offset without a trigger.

ViT-B/16:

@@destinations_vit@@

Swin-S:

@@destinations_swin@@

The prediction holds for the late band and fails for the whole-network probes. Under dropout in blocks 9 to 12, @@m_late_source@@ of shifted triggered passes land on the source class against @@m_late_chance@@ for a uniform draw and @@m_late_below@@ for clean passes landing 1 below. Under PSBD-TM at its adaptive rate only @@m_tm_source@@ do, since that probe also destroys the content the source class is read from. On the second architecture the last blocks (17 to 24) send @@m_swin_late_source@@ of them to the source against a uniform @@m_swin_late_chance@@. All-to-one triggered predictions that break under the same late band also land on their source (@@m_a2o_late_source@@), so falling back to the source is what a broken trigger read does and owes nothing to the rotation. Destinations are also less consistent for triggered inputs than for clean ones: among images that moved in at least 2 passes, all shifted passes named the same class for @@m_tm_same_trig@@ of triggered images and @@m_tm_same_clean@@ of clean ones under PSBD-TM, and per predicted class the most common destination took @@m_tm_class_trig@@ against @@m_tm_class_clean@@ of the shifted passes. The runner-up class of a triggered softmax is its source for @@m_runner_up_source@@ of images.

![destinations](figures/destinations.png)

## Tokens and blocks of the trigger read and the source read

`tokens.py` masks tokens deterministically at the attention input on @@token_pairs@@ paired test images per ViT model and reads where each captured triggered image goes, as shares of attack label, source class and anything else. It uses the trigger's token positions, which a defender does not know. All-to-one BadNets models with the same patch are the control. Memory was capped at @@token_memory@@ of the card, inference only.

@@tokens_table@@

Per model, the share of captured triggered images that leave the attack label when the trigger's tokens are masked in each span of blocks:

@@tokens_model_table@@

The trigger read is a separate component that sits in the middle and late blocks. Masking the trigger's tokens in blocks 5 to 12 returns @@t_mid_source@@ of captured triggered images to their source class on average, and masking as many random tokens in every block leaves @@t_random_attack@@ on the attack label. Which span holds the read differs by model (the per-model table), and wherever it sits, removing it leaves the read of the source class intact. The source class is read from the content. With the trigger visible and 30% of the other tokens, @@t_30_attack@@ of all-to-all images keep the attack label while the clean twins under the same masks are classified correctly @@t_30_clean@@ of the time. With no content visible @@t_none_attack@@ keep it and @@t_none_elsewhere@@ land on neither label, while @@t_a2o_none_attack@@ of all-to-one images keep their target. So the all-to-all output combines a trigger read with a content read, and breaking either moves it, which is why it is more fragile than a clean prediction.

![tokens](figures/tokens.png)

## Detectors derived from the mechanism

3 facts from the mechanism shape the detector. A clean prediction is settled before the last blocks and survives late perturbation. A triggered all-to-all prediction still depends on the trigger read and the content read in the middle and late blocks, and under late perturbation it breaks and falls back toward its source. A triggered all-to-one prediction survives almost any perturbation. So the late band gives `late_fragility`, the late-band PSU with its sign flipped. The band is `@@late_band_vit@@` on ViT and `@@late_band_swin@@` on Swin, read at the rate whose clean validation shift ratio is nearest @@late_target@@. `either_regime` takes the smaller of the validation ranks of PSU-TM and of late fragility, so an input abnormal in either direction is flagged. `depth_profile` is the difference of the 2 ranks. The statistics the brief proposed are scored beside them.

On the development models at the moment of the choice (`detection_development.json`, @@d_dev_n@@ ViT models at the ASR bar), PSU-TM read @@d_dev_psu@@, late fragility @@d_dev_late@@, either regime @@d_dev_either@@ and the entropy control @@d_dev_entropy@@. Either regime was chosen over late fragility because late fragility inverts on all-to-one while either regime kept @@d_dev_cost_either@@ there against @@d_dev_cost_psu@@ for PSU-TM on the development record.

Development, ViT, ASR bar:

@@development_table@@

Development, Swin-S, ASR bar:

@@development_swin_table@@

## Confirmation on CIFAR-100 and Tiny

No CIFAR-100 or Tiny all-to-all model clears the ASR bar, so the confirmation reads the relaxed bar only. On ViT (@@c_n@@ models, @@c_late_n@@ with the late band) PSU-TM reads @@c_psu@@, late fragility @@c_late@@, either regime @@c_either@@ (worst model @@c_either_min@@) and the entropy control @@c_entropy@@. Either regime flags @@c_either_tpr_1@@, @@c_either_tpr_5@@ and @@c_either_tpr_10@@ of triggered inputs at the 3 budgets.

The confirmation does not reproduce the development reading. The frozen detector stays above PSU-TM and above the benign null, and it falls below the entropy control and far below its own development value, with a TPR at the smallest budget that no deployment could use. Transition typicality reads higher in AUROC, and its realized false-positive rate at every budget is several times the budget, so its thresholds do not hold.

Confirmation, ViT, relaxed bar:

@@confirmation_table@@

Confirmation, Swin-S, relaxed bar:

@@confirmation_swin_table@@

## All all-to-all models

ViT, ASR bar, all datasets:

@@all_strict_table@@

ViT, ASR bar at @@points_5@@ points, all datasets:

@@all_strict5_table@@

ViT, relaxed bar, all datasets:

@@all_relaxed_table@@

Swin-S, ASR bar:

@@swin_strict_table@@

Swin-S, relaxed bar:

@@swin_relaxed_table@@

Per model:

@@model_table@@

![detectors](figures/detectors.png)

## Cost on the all-to-one panel

A defender does not know which regime a model is in, so every statistic is also scored on the ledger's successful all-to-one models. Thresholds are validation quantiles, so the realized clean false-positive rate stays at the budget by construction, and the cost shows up as lost AUROC and TPR.

ViT-B/16:

@@cost_vit@@

Swin-S:

@@cost_swin@@

Either regime reads @@cost_either@@ on the @@cost_n@@ ViT all-to-one models against @@cost_psu@@ for PSU-TM on the same models, a change of @@cost_gain@@ @@cost_gain_ci@@. TPR at the smallest budget falls from @@cost_psu_tpr_1@@ to @@cost_either_tpr_1@@. On Swin-S it reads @@swin_cost_either@@ against @@swin_cost_psu@@. Late fragility alone reads @@cost_late@@ on all-to-one and the entropy control @@cost_entropy@@, so neither can be deployed without knowing the regime.

## Null control

Benign models probed with the BadNets trigger have nothing to detect and must read near 0.5.

@@benign_table@@

## SAM and evasion variants

@@side_detection_table@@

## Verdicts

**Idea 1, the mechanism.** Supported with a correction. Triggered all-to-all predictions are more fragile than clean ones under every probe but Swin-S PSBD-TM at its saturating adaptive rate, and the token masks show why: the output combines a trigger read in the middle or late blocks with a content read of the source class, and breaking either moves it. Shifted triggered predictions land on the source class most often when the perturbation is confined to the last blocks. Whole-network probes send them elsewhere, since they also destroy the content.

**Idea 2, destination structure.** Refuted for the whole-network probe the statistics read. Under PSBD-TM triggered destinations are less consistent across passes than clean ones and no more concentrated per predicted class, because clean predictions under heavy perturbation collapse onto a few attractor classes and triggered ones scatter. Under the late band on ViT the order reverses, since those passes fall back to the source. Late fragility uses that band through the shift rate rather than through where the passes land. Destination concentration reads @@d_dev_concentration@@ on development. Transition typicality reads @@d_dev_transition@@ there, and its discrete scores tie so heavily on 100 and 200 classes that the validation quantile no longer sets the false-positive rate.

**Idea 3, 2-sided and class-conditional PSU, routers, confidence.** 2-sided PSU-TM reads @@d_dev_two_sided@@ and the class-conditional version @@d_dev_class@@ on development, because PSU-TM's triggered and clean distributions overlap almost entirely at the adaptive rate. Both routers fail: the sign of the pool shift d does not flip reliably on all-to-all at these poison rates, so the H43 router reads @@d_dev_router@@ and the late router @@d_dev_router_late@@. Entropy separates all-to-all, and it perturbs nothing, measures confidence and inverts on all-to-one (@@cost_entropy@@), so it is not a PSBD detector.

**Idea 4, the late-band probe.** Supported on the development models and not confirmed at that strength. Late fragility and either regime separate CIFAR-10 all-to-all well on both architectures, GTSRB unevenly and CIFAR-100 and Tiny weakly (per-model table).

## Whether anything is possible

Something is possible, and it is not a usable detector yet. A PSBD-like probe that perturbs only the last third of the blocks and counts how often the prediction moves separates all-to-all triggered inputs from clean ones where the mechanism says it should, because a triggered all-to-all prediction still depends on late processing that a settled clean prediction no longer needs. On the ViT models at the ASR bar late fragility reads @@a_strict_late@@ against @@a_strict_psu@@ for PSU-TM (Swin-S @@s_strict_late@@ against @@s_strict_psu@@ on @@s_strict_n@@ models), and `either_regime` covers both regimes without knowing the regime of a model, at @@a_strict_either@@ on all-to-all and @@cost_either@@ on the all-to-one panel. On the confirmation datasets the frozen detector reads @@c_either@@, with TPR @@c_either_tpr_1@@ at the smallest budget.

So the direct answer is no for deployment and yes for the direction. No mapping-agnostic statistic tested here reaches a usable TPR at a low false-positive rate on the CIFAR-100 and Tiny all-to-all models, and the 2 statistics that read higher there are entropy (a confidence reading) and transition typicality (whose thresholds do not hold). The per-model token table suggests a reason that is a hypothesis until tested: on some models the trigger is read in blocks 5 to 8, ahead of the band perturbed, so a band fixed at the last third misses the read. A band covering the middle and late blocks, chosen on the development models and confirmed on a fresh all-to-all attack, is the next test. All of this is 1 attack and 1 mapping family, and it shows nothing about any other all-to-all attack.

## Status and future work

The user moved the all-to-all detector to future work for this paper on 2026-09-29. The GPU queue was stopped that night and the running sweep was allowed to finish. What is measured above stands. These items are open.

- Swin-S late-band sweeps for `swin_gtsrb_badnet_a2a_0_005`, `_0_05` and `_0_1` and for the 3 Swin-S benign references were queued in `run_gpu.sh` and never ran, so the Swin-S late-band rows rest on the CIFAR-10 models and 1 GTSRB model and the Swin-S null control has no late-band row. PSBD-TM for `swin_cifar100_badnet_a2a_0_1` did not run either.
- The token masks ran on the models in the per-model token table. The CIFAR-100 and Tiny all-to-all models beyond the first, and their all-to-one controls, were queued and removed.
- A band covering the middle and late blocks, chosen on the development models, is the next detector to test, because the token table puts the trigger read ahead of blocks 9 to 12 on some models.
- A second all-to-all attack (a blend or warping trigger) and a mapping other than the rotation by 1 are needed before any claim about all-to-all in general.

## Commands and wall time

`render_readme.py` writes this file from `README.template.md` and the records under `results/_experiments/all_to_all_detection/`. Edits go to the template, and every number comes from a record.

    source .venv/bin/activate
    python experiments/all_to_all_detection/panel.py
    python experiments/all_to_all_detection/mechanism.py --datasets cifar10 gtsrb --out mechanism_development.json
    python experiments/all_to_all_detection/mechanism.py
    bash experiments/all_to_all_detection/run_tokens.sh FOLDERS
    bash experiments/all_to_all_detection/run_gpu.sh
    python experiments/all_to_all_detection/detect.py --datasets cifar10 gtsrb --out detection_development.json
    python experiments/all_to_all_detection/detect.py
    python experiments/all_to_all_detection/tokens.py --summarize-only
    python experiments/all_to_all_detection/figures.py
    python experiments/all_to_all_detection/render_readme.py
    python -m pytest experiments/all_to_all_detection/test_invariance.py -q

@@wall_time@@
