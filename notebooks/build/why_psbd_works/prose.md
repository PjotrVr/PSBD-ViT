<!-- intro -->
# Why PSBD and PSBD-TM work on ViT-B/16 and Swin-S, claim by claim

PSBD (Li et al., arXiv 2406.05826) flags an input when the probability of the model's own answer barely falls under a few perturbed forward passes. This repository adapts it to ViT-B/16 and Swin-S, where the best placement, PSBD-TM, zeroes whole tokens at the input of every attention LayerNorm, and the paper's own site becomes PSBD-RD, dropout after every residual add. Many explanations of why this works have been offered along the way: by Li et al., by the authors of the detectors we ported, by folk reasoning and by us. This notebook is the one place where each of them is stated, tested against its control and given a verdict.

A verdict is 1 of 4 words. **true** means a controlled experiment supports the claim at the strength stated. **false** means a controlled experiment contradicts it. **partial** means the experiment supports part of it, or supports it on some models and not on others. **open** means the evidence exists but does not decide the claim, and **pending** means the experiment that would decide it has not written its record yet. Negative results get the same space and the same figures as positive ones.

The notebook reads JSON records already on disk and runs no forward pass. 1 reading, the WaNet destinations of section 7, has no JSON record, so it is recomputed from the stage-1 cache tensors with the script's own function. Every number in the text is rendered from a file named beside it, and every figure writes a JSON sidecar with the numbers it plots under `notebooks/figures/why-psbd-works/`. Rebuilding picks up whatever the running experiments have written since, and a missing record prints pending instead of failing.

<!-- map -->
## Map of the claims

The claims fall into 11 groups, 1 section each.

1. **Li et al.'s account.** Their 4 observations about the prediction shift, reproduced on ResNet-18 and tested on ViT-B/16 and Swin-S under both placements.
2. **Breaking-point curves.** The clean and triggered shift ratio against the rate, model by model, with the unified account that PSBD reads a per-image breaking point.
3. **Patch-trigger routing.** The causal masking experiments that explain why PSBD-TM separates patch triggers and PSBD-RD does not.
4. **Clean fragility.** How clean predictions break, and which clean images become false positives.
5. **Alternative explanations.** Confidence, out of distribution, LayerNorm absorption and IBD-PSC's theorem.
6. **Neurons and directions.** Whether the backdoor is a set of neurons or a direction in the residual stream.
7. **WaNet.** The 1 attack where PSBD-TM inverts on a model, and the accounts proposed for it.
8. **Global-trigger redundancy.** Whether a global trigger can be read from any subset of tokens.
9. **Where the backdoor lives.** The manifestation panel over both architectures.
10. **Pass statistics, depth bands and fusion.** Pre-registered readouts of the stage-1 caches.
11. **General principles.** Margin, direction, redundancy and the other accounts of `experiments/why_psbd_works/`, per attack category.

<!-- table_intro -->
## Claims table

Each row is 1 claim with its verdict, the number that decides it, the control it was read against and the file and field that hold the evidence. The id links to the section. The table is filled in by the last cell of the notebook, after every section has read its records. The same table is repeated there.

<!-- panel -->
## The panel

Most readings use the paper's panel: the ViT-B/16 models whose attack succeeded, meaning the attack success rate clears the bar and clean accuracy stays within 2 points of the benign reference. Records written before the panel moved to this bar used the 57-model panel of 2026-09-24, and the text says so beside every number that comes from one.

<!-- s1 -->
<a id="s1"></a>
## Li et al.'s account

Li et al. justify PSBD with 4 observations made on ResNet-18. (1) Dropout changes the answer for clean images far more often than for triggered ones. (2) When a clean answer changes, it changes to the attacker's target class. (3) With dropout, a clean image's last-layer features become almost identical to its triggered twin's, which they call neuron bias. (4) The ordinary MC-Dropout uncertainty catches BadNets and misses WaNet, which is why PSBD reads the drop in confidence instead. `experiments/prediction_shift_phenomenon/` reproduces each on our own ResNet-18 checkpoints and repeats the same measurement on the transformer panels under PSBD-RD and PSBD-TM.

<!-- claim1 -->
**Claim 1, clean predictions shift and triggered ones do not.** **Prediction if true.** At the adaptive rate the shift ratio of captured triggered images is near 0 while the clean validation shift ratio is high, and on a benign model the 2 curves coincide. **Experiment.** The shift ratio of every split at every cached rate, read from the stage-1 caches. The rule counts a model when its triggered shift ratio at the adaptive rate is at most the near-zero threshold, and a claim is true when it holds on at least 2 in 3 of the models pooled over both transformers. **Control.** The benign references, probed with a trigger they never learned.

<!-- claim1_limits -->
**What this does not show.** The claim is about argmax flips, and PSBD scores the confidence drop, so a model can fail claim 1 and still be detected (the TaCT models of section 2 are the example). The triggered images here are test images the model captures, not the poisoned training set the paper scores.

<!-- claim2 -->
**Claim 2, shifted clean predictions land on the target.** **Prediction if true.** At least the paper's "almost all" share of shifted clean predictions lands on the target class, and a benign model sends its shifts to 1 class too. **Experiment.** The landing class of every shifted (image, pass) prediction at the adaptive rate. **Control.** The benign references, whose share on class 0 is what attraction without a backdoor looks like.

<!-- claim2_limits -->
**What this does not show.** Where the target does win (Swin-S under PSBD-TM on BadNets, WaNet and Adaptive-Blend), this reading cannot say why. A benign model can also collapse onto 1 class. The target share moves strongly with the rate, so a claim read at 1 rate is a claim about that rate.

<!-- claim3 -->
**Claim 3, neuron bias in the last layer.** **Prediction if true.** Under dropout the last-layer features of a clean image and its triggered twin become almost identical, and more so than 2 unrelated clean images. **Experiment.** The centered cosine between the 2 feature maps without the perturbation, with 1 mask shared by both images (the paper's figure) and with independent masks (how PSBD scores 2 inputs). **Control.** A clean image paired with an unrelated clean image under the same mask relation, and the benign models.

<!-- claim3_limits -->
**What this does not show.** The measurement covers 1 model per attack family, mostly at 10% poisoning. The last block's patch tokens stand in for ResNet-18's top layer. It refutes convergence as evidence of a backdoor. It does not rule out a weaker drift toward the triggered features, which the source README reports on Swin-S only.

<!-- claim4 -->
**Claim 4, MC-Dropout uncertainty misses WaNet.** **Prediction if true.** The standard deviation of the answer's probability over the passes separates BadNets and fails on WaNet and Adaptive-Blend, where the confidence drop still works. **Experiment.** The AUROC of the standard deviation at every rate, given its best rate as an oracle, against fractional PSU at the adaptive rate on the same passes. **Control.** The 2 scores come from the same passes, so only the statistic changes.

<!-- claim4_limits -->
**What this does not show.** The paper tracks uncertainty over training epochs on the poisoned training set, and the caches hold the final checkpoint's test-time passes. The BadNets half of the claim fails under PSBD-TM on ViT, so the verdict is about the 2 halves together.

<!-- s2 -->
<a id="s2"></a>
## Breaking-point curves

The shift ratio at rate p is the share of (image, pass) predictions that differ from the unperturbed answer. Over a ladder of rates it traces how the breaking points of a population are distributed, and the gap between the clean and the triggered curve is what a detector at 1 rate can exploit. The figures draw every model as a thin line and the median in bold, clean validation images in blue and captured triggered images in orange, for PSBD-TM in the top row and PSBD-RD in the bottom row, with the benign references as the control column. The rates are on a log axis because PSBD-RD saturates below p = 0.1.

<!-- curves_swin -->
The same curves on the Swin-S panel. Swin's trigger-conditional TaCT models were swept with PSBD-TM at a single rate and have no PSBD-TM curve. The Adaptive-Blend column exists on Swin only.

<!-- curves_wanet -->
WaNet model by model on ViT-B/16, solid lines for PSBD-TM and dashed lines for PSBD-RD. The inverted cell of the panel is the 1 model where the triggered PSBD-TM curve climbs with the clean one.

<!-- contradictions -->
**Open contradictions in the curves.** 3 features of the curves do not fit a single account in which PSBD reads how much later triggered inputs break than clean ones. The cell below computes each from the medians drawn above.

<!-- breaking_point -->
**The unified breaking-point account.** **Claim.** Every input has a critical rate p*, the smallest rate at which most of its passes change its answer, and PSBD at any placement is a 2-sample test on p*. **Prediction if true.** Over every (model, placement) pair the AUROC at the adaptive rate is a monotone function of A* = P(p*_triggered > p*_clean), and on the benign references A* sits at 0.5. **Experiment.** `experiments/why_psbd_works/critical_rate.py` on the cached per-pass argmax of every placement with a ladder. **Control.** The benign references. The owner experiment has not reported this record in its README yet. The verdict stays open until it does and the numbers are a first read.

<!-- s2_limits -->
**What this does not show.** The curves count argmax flips, so they cannot explain a separation that lives in the size of the confidence drop, which is exactly the TaCT contradiction. The triggered curves read captured images only, and the medians hide the per-model spread the thin lines show.

<!-- s3 -->
<a id="s3"></a>
## Patch-trigger routing

PSBD-TM and PSBD-RD both change most clean predictions at their adaptive rates, and on patch triggers they part ways. `experiments/why_token_masking_works/` tests why with deterministic masks on the trigger's own tokens (section A), PSBD-TM with its masks recorded (section B), PSBD-RD restricted by position (section C) and fixed visible subsets of tokens (section D, used in section 8 here). Every test changes 1 thing and carries a random-token or clean-image control.

<!-- routing_read -->
The first panel masks the trigger's tokens in chosen blocks, the second reads PSBD-TM's own passes by J, the number of late blocks (9 to 12) in which every trigger token happened to be masked, the third splits the broken triggered predictions by J and the fourth compares PSBD-RD on all tokens with PSBD-RD on the trigger positions only.

<!-- s3_limits -->
**What this does not show.** The trigger-conditional TaCT group holds few models, and models that share a rate drew identical masks, so pooled counts in section B are not independent replicates. The late-read result locates where the class token reads the trigger. It does not by itself say why the class token waits until the last blocks.

<!-- s4 -->
<a id="s4"></a>
## Clean fragility

A detector built on PSBD needs clean answers to break. The theory note fits 2 models to how the clean keep curve falls with the PSBD-TM rate: a homogeneous AND law, where every image needs the same number of tokens visible, and a per-image critical rate spread logistically across images, above a floor of images the heavily masked model sends to its default class. The same record counts the clean images that never flip at the adaptive rate and their share of the false positives. The readings are on the current panel (`cached_reads.json` of `experiments/why_psbd_works/`), while `docs/why-psbd-works-theory.md` quotes the same fits on the 2026-09-24 panel of 57 models.

<!-- s4_limits -->
**What this does not show.** The fits are to dataset means of the ladder and treat the default-class floor as constant. The critical rate is a description of how clean images break. Why some clean images never break is not measured.

<!-- s5 -->
<a id="s5"></a>
## Alternative explanations

4 accounts claim to explain PSBD without routing. The score might only read confidence. Triggered inputs might be out of distribution. A LayerNorm might absorb noise and not masks. IBD-PSC's theorem says amplification drives clean inputs onto the target.

<!-- confidence -->
**Confidence.** **Claim.** A backdoored model is more confident on triggered inputs, and PSU only restates that. **Prediction if true.** No score beats A*(P_c), the AUROC of the cross-fitted likelihood ratio of unperturbed confidence, the best any function of confidence can do. **Experiment.** A* per model from the cached baselines against PSBD-TM's and PSBD-RD's AUROC. **Control.** Raw confidence, which understates A* because the likelihood ratio is not monotone in confidence.

<!-- ood -->
**Out of distribution.** **Claim.** A trigger makes an image unusual, and PSBD flags unusual inputs. **Prediction if true.** The same triggered images are flagged on a benign model, and images of another dataset are flagged more often than clean ones. **Experiment.** The benign references scored on the same triggered splits, and foreign-dataset images under PSBD-TM's threshold on the `why_psbd_works` models. **Control.** The clean images of each model, flagged at the quantile by construction.

<!-- layernorm -->
**LayerNorm absorption and the operator gap.** **Claims.** A LayerNorm divides by each token's standard deviation, so it undoes part of an additive disturbance and none of a mask (C16). That absorption was once offered as the reason Gaussian noise trails token masking at the attention input (C17). The theory note replaces it with a read-structure account: a masked read is intact or gone while a noisy read is degraded everywhere. A patch trigger needs only 1 intact read (C18). **Experiment.** The absorption record measures the disturbance before and after the norm at several injected sizes, and the ladders give each model's AUROC at the matched rule for both operators. **Control.** Positions no LayerNorm follows, and the matched-shift rule, which equalizes clean disturbance across operators.

<!-- ibd -->
**IBD-PSC's theorem.** **Claim.** Hou et al. prove that amplifying the normalization parameters makes the feature norm large enough that the backdoored model predicts the target for every input, so a poisoned input is the one whose prediction does not change. **Prediction if true.** At the top of the amplification ladder the shifted clean predictions land on the target. **Experiment.** `gain_scale` at the top factor of each model's ladder, the share of shifted clean predictions on the target and on the largest single class. **Control.** The benign references and the largest non-target class.

<!-- s5_limits -->
**What this does not show.** A* needs the triggered labels, so it is an oracle ceiling and not a detector. The foreign-image test covers the few `why_psbd_works` models so far, and on its WaNet model PSBD-TM flags foreign images far above clean ones, so the out-of-distribution account is refuted as the reason triggered inputs are flagged and not as a property of every model. No experiment removes the LayerNorm, and the read-structure account has no isolating test yet. The IBD-PSC verdict is about the theorem, not the detector, which works.

<!-- s6 -->
<a id="s6"></a>
## Neurons and directions

Li et al. say PSBD exploits "robust neuron bias paths", and the Fine-Pruning and ANP line treats a backdoor as a few units. The alternative is 1 linear direction in the residual stream that no coordinate carries. The ablation record removes either and reads the attack success rate, with random coordinates and random directions as controls, and `why_psbd_works` zeroes MLP hidden units in the late blocks. The first runs of the manifestation panel add the control the ablation lacked, the target class's own direction.

<!-- s6_limits -->
**What this does not show.** The ablation covers CIFAR-10 models at 10% poisoning and seed 0 only. Removing the backdoor direction also costs target-class recall, and the manifestation panel of section 9 shows the direction is mostly the target class's axis at the last block. The direction survival reading is 1 BadNets checkpoint, and PSBD-RD does detect global triggers, so an erased direction is not the whole story of PSBD-RD.

<!-- s7 -->
<a id="s7"></a>
## WaNet

WaNet is where the 2 placements disagree most, and it holds the panel's only model where PSBD-TM inverts. 3 accounts are tested. WaNet trains with a noise mode that labels randomly warped images with their true class, so a masked triggered input might read as a rejected warp and go home to its true class (E8, plan item X1). WaNet's trigger might be a coherence check that no single token can read (memo L26). And the inverted model might be a checkpoint accident rather than a recipe that defeats PSBD-TM (plan item X2).

<!-- probe -->
The coherence probe fits a linear classifier on a single patch token, or on the class token, to tell the exact warp from a random warp of the same image, at blocks 4 to 8 and 12.

<!-- x2 -->
The replicates of the inverted WaNet cell carry the same recipe and data with seeds 1 and 2. The key number above lists the replicates whose sweeps have landed, and the verdict stays open until the WaNet experiments of section 7 close.

<!-- s7_limits -->
**What this does not show.** X1 also reads WaNet models outside the panel, which miss the 2-point clean-accuracy bar and are marked off panel in the figure. The probe shows the warp is legible per token, which leaves open why random masking reaches WaNet's evidence (section 10's depth bands point to the early and middle blocks). Where the inverted WaNet cell's triggered shifts do land is plan item X1b.

<!-- s8 -->
<a id="s8"></a>
## Global-trigger redundancy

**Claim.** A global trigger is present in every token, so it stays legible from any 30% of them and survives token masking. **Prediction if true.** The excess retention of the attack success rate at 30% visible tokens is high and well above clean accuracy retention. **Experiment.** A fixed random pattern of visible patch tokens in every block, in section D of `why_token_masking_works` (3 patterns shared across models) and in `why_psbd_works` (20 patterns per model). **Control.** Clean accuracy retention at the same fraction, and the excess form, which removes a masked model's collapse onto the target.

<!-- s8_limits -->
**What this does not show.** The 2 records disagree on WaNet, and they differ in 2 ways at once: the models (section D used WaNet models that are not all in the panel) and the number and seeding of patterns. Until 1 of the 2 is held fixed, the disagreement cannot be assigned to either, so the claim stays open.

<!-- s9 -->
<a id="s9"></a>
## Where the backdoor lives

`experiments/backdoor_manifestation/` measures, per attack category and architecture, where the network reads the trigger, at which depth the triggered representation leaves the clean one and whether the backdoor is a set of neurons or a direction. Every removal has a control of the same kind: random coordinates against the top-TAC ones, and against the backdoor direction a variance-matched clean principal direction, isotropic random directions, the other trigger's direction, the benign model's direction and the target class's own offset, with per-class recall. The benign models shown the same trigger are the control for every reading. The figure draws the onset layer per attack, the attack success rate after each removal at the last block and what the removal costs the target class.

<!-- s9_limits -->
**What this does not show.** The panel is 1 model per attack and dataset at 5% poisoning, in bfloat16. At 2/3 of the depth the direction removal leaves much more of the backdoor, and whether later blocks rebuild it from the trigger tokens is not tested. The onset of a benign control is early only because its small peak is early, so it is not a comparable onset.

<!-- s10 -->
<a id="s10"></a>
## Pass statistics, depth bands and fusion

`experiments/cache_readouts/` answers 3 questions from the cached per-pass tensors, with a protocol fixed before the confirmation: explore on a development set, write a pre-registration with a hash, confirm once on the held-out CIFAR-100 and Tiny ImageNet models, then read the full panel for completeness. Its verdicts are those of `judge.py` in `verdicts_all.json`.

<!-- pass_stats -->
**Pass statistics.** **Claim.** A statistic other than the mean over PSBD's passes separates better: the best pass (picked on the development set) or the worst pass (proposed in plan item N17, on the premise that a triggered patch input survives every pass intact). **Experiment.** 6 statistics over the same passes, paired against the mean. **Control.** The mean fractional PSU, which the readout reproduces bit for bit from `psbd_metrics.json`.

<!-- depth -->
**Depth bands.** **Claims.** Token masking in blocks 1 to 8 leaves triggered patch inputs alone (E1a), token masking in blocks 9 to 12 hits them (E1b), no token-mask band moves a global trigger (E2) and early residual dropout moves WaNet more than BadNets (E10). **Experiment.** Each band at its top cached rate, and at the rate nearest the matched shift as a sensitivity reading. **Control.** The paired clean images under the same band. The heat map shows triggered minus clean share of passes moved, so blue is a band that leaves the trigger alone.

<!-- fusion -->
**Fusion with residual dropout in blocks 5 to 8.** **The operation.** The partner is `pre_residual_blocks_5_8`: standard element-wise inverted dropout (`nn.Dropout`) on the attention branch output and on the MLP branch output, before each is added to the residual stream, in ViT blocks 5 to 8 only. Its rate is picked on clean validation by the adaptive rule. Each probe's fractional PSU becomes its percentile among that probe's own clean-validation scores, and the fused score is the plain minimum of the 2 percentiles. **Claim.** The fusion detects more at low false-positive rates. **Experiment.** The rule on the held-out models, the fused score thresholded at a quantile of its own clean-validation distribution. **Control.** PSBD-TM alone on the same models and passes, the plain average of the 2 fractional PSUs as a robustness check and the late band (blocks 9 to 12) as the alternative partner. The left panel draws TPR against the realized FPR at the budgets the record stores, on a log axis, which is the front of the ROC curve where a deployed detector operates. The record holds no per-sample scores, so the curve is sampled at those budgets only.

**Why blocks 5 to 8.** The middle band is already the paper's best residual placement. It reaches the adaptive rate on every panel model, while the late band never does on some of them. And blocks 5 to 8 are where the global triggers write their backdoor direction into the class token (Karayalçin et al., and `experiments/backdoor_direction_layers/`), while the patch read sits in blocks 9 to 12 that PSBD-TM already covers. That is the rationale for the choice and not a measured benefit: under the plain minimum the per-attack readings below show no gain on Blend, LF or BPP. The plain average adds a small one. On the held-out models the 2 bands tie. The onset layers come from the runs of `experiments/backdoor_manifestation/` (`onset_layer`), which replace the untraceable table of `experiments/backdoor_direction_layers/`.

**Reading the plain minimum.** The minimum of 2 percentiles is low when either probe's percentile is low, so thresholding it is an OR of 2 detectors, a parallel cascade. It flags an input when either probe finds it more stable than almost all clean images. The 2 probes have different mechanisms: token masking at the attention input breaks the route a patch trigger takes, and residual dropout in blocks 5 to 8 perturbs the residual stream itself, which is what breaks a triggered WaNet prediction that token masking leaves alone. The rule has no weight to tune, and the threshold on the fused score keeps the false-positive budget.

**Band comparison, post-hoc.** A later readout of `experiments/cache_readouts/` puts each residual band (blocks 1 to 4, 5 to 8 and 9 to 12) in the union as the partner and reads TPR at 1%, 5% and 10% FPR per attack for WaNet, TaCT and BadNets, under the plain minimum and the plain average. It was requested after the confirmation read and makes no pick, so the figure and its numbers are labeled post-hoc, and they rest on few WaNet and TaCT models.

**Scope of the confirmation.** The pre-registration picked the late band and a weighted rule. The middle band and the plain minimum were chosen after the held-out read showed the bands tying, so the held-out numbers describe the choice and do not confirm it. The Swin-S confirmation (plan slot 23) is the independent test, so the verdict is partial.

**Compute control.** The fusion reads 2 probes of k passes each, so part of its gain could be the extra passes. The control is PSBD-TM alone at twice the passes. It needs the k = 20 caches, which are still being written, so the claim is open.

<!-- s10_limits -->
**What this does not show.** The held-out set holds no TaCT model and no PSBD-TM failure, so the fusion's cost on TaCT and its WaNet gain are read on the panel only, where the pick was not confirmed. The late band never reaches the adaptive target on some panel models, so its column of the comparison covers fewer models than the middle band's. The depth bands at the top rate hide almost every token, which the readout itself names as a confound of E1b.

<!-- s11 -->
<a id="s11"></a>
## General principles

`experiments/why_psbd_works/` asks the general question: what property of a triggered input keeps its answer through a perturbation that destroys a clean answer, for any operator and per attack category. Each account has a stated prediction, a rule written into `summary.json` and the benign models probed with a patch and a global trigger as the control. The grid shows each measured group's status with the key number the rule reads. A hypothesis gets a verdict only once every attack category with a successful model (patch, blend, warp and quantization) has been measured. Until then it is open with its provisional reads listed.

<!-- s11_limits -->
**What this does not show.** The experiment has measured a handful of its planned models, and the rendered rows name how many per group and which categories are still missing. A supported status on 1 model per category is a direction, not a result.

<!-- table_end -->
## Claims table, final

The same table as at the top, written once every section has run. The rows are also saved as `notebooks/figures/why-psbd-works/claims.json`.
