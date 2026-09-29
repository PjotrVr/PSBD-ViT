# Simple controlled experiments for why PSBD works on ViT, where it fails and what to build next

> Superseded on 2026-09-29: the panel numbers below were read before the results moved to the successful backdoors at the 2-point clean-accuracy bar. They are kept as they were measured, and the current values are the macros in `paper/headline.tex`.

This memo collects the small, controlled experiments that published work uses to expose a property of a model (knockouts, patching, dose-response curves, transplants, swaps and minimal inputs), then turns them into a ranked plan for this repository. The question is why a triggered prediction survives PSBD-TM and a clean one does not, why token masking in particular does this and why the statistic fails on WaNet. It also answers 5 requests added during the work: a WaNet-specific perturbation detector, novel perturbation designs, a reverse-engineered design for an optimal PSBD-like probe, an honest design protocol and a workflow for reverse-engineering the mechanism correctly.

I wrote it on 2026-09-29 against branch `rewrite`. It builds on `docs/why-psbd-works-literature.md` (explanations L1 to L27 and P1 to P12), `docs/why-psbd-works-theory.md`, `docs/attack-design/improving-the-defense.md` and the READMEs of `experiments/why_token_masking_works/`, `experiments/why_psbd_works/`, `experiments/failure_modes/`, `experiments/prediction_shift_phenomenon/`, `experiments/backdoor_manifestation/`, `experiments/probe_union/` and `experiments/probe_fusion/`. I ran no experiment and wrote no code. The only numbers I read that no README prints come from the `adaptive` block of existing `psbd_metrics.json` files and from `args.json` sidecars, and each is labeled where it appears.

## Scope, conventions and the development set

The explanations the plan must decide between recur throughout, so they get short labels. The labels map onto the earlier memo where a match exists.

| label | explanation | earlier ID |
|---|---|---|
| E1 | routing with a late read: a patch trigger's evidence stays in its own tokens of the residual stream and the class token needs 1 intact read of it in blocks 9 to 12 (an OR over tokens and blocks) | L20 |
| E2 | redundancy: a global trigger is legible from any 30% of tokens | L23 |
| E3 | direction: the backdoor is 1 linear direction and PSU reads how much of it survives | L8 |
| E4 | confidence: triggered inputs are just more confident | P1, L1 |
| E5 | target attraction: perturbation pushes clean inputs toward the target | L10, P6 |
| E6 | sink artifact: a token zeroed before LayerNorm becomes the constant $\beta$, all masked tokens share 1 key and form an attention sink, and part of PSBD-TM's effect is this off-distribution artifact | L22 |
| E7 | clean AND fragility: clean evidence needs about 70% of tokens visible in every block, so it fails under masking | theory note |
| E8 | noise-mode coherence: WaNet's noise mode trains the model to reject any warp that is not the exact trigger field, so a perturbed triggered WaNet input looks like a rejected warp | L26 |
| E9 | position binding: the backdoor is tied to the trigger's absolute position through the position embeddings | new |
| E10 | computed against routed: a routed trigger is finished early and carried, a computed trigger is assembled across early and middle blocks from relations between tokens | new |

The user fixed a development set of 10 models so each test is quick. Every GPU cost below is quoted on it. The PSBD-TM and PSBD-RD AUROCs in the table are the coordinator's, and I did not recompute them.

| model | role | PSBD-TM | PSBD-RD |
|---|---|---|---|
| `vit_cifar10_wanet_0_1` | the WaNet failure | 0.459 | 0.947 |
| `vit_tiny_wanet_0_05` | WaNet where PSBD-TM works | 0.930 | 0.953 |
| `vit_cifar10_bpp_0_05` | PSBD-TM's weakest non-WaNet model | 0.801 | 0.880 |
| `vit_gtsrb_bpp_0_01` | BPP where PSBD-TM leads | 0.898 | 0.755 |
| `vit_cifar10_blend_0_1` | Blend where PSBD-RD leads | 0.906 | 0.991 |
| `vit_cifar10_badnet_a2o_0_01` | the patch contrast | 0.982 | 0.309 |
| `vit_gtsrb_tact_0_05` | trigger-conditional TaCT | 0.942 | 0.392 |
| `vit_cifar10_tact_0_01` | high AUROC with low TPR | high | not given |
| `vit_cifar100_bpp_0_01` | BPP on a held-out dataset | 0.921 | not given |
| `vit_tiny_lf_0_01` | LF on a held-out dataset | 0.941 | not given |

3 of the 10 (`vit_tiny_wanet_0_05`, `vit_cifar100_bpp_0_01`, `vit_tiny_lf_0_01`) belong to CIFAR-100 and Tiny ImageNet, the datasets `configs/psbd_basis.json` reserves for reporting. Exploring on them spends the held-out half. The protocol section says how to handle that: either swap them for development-dataset models of the same attack or drop these 3 from the confirmatory evaluation.

Cost units assume ViT-B/16 in bfloat16 on the login-node A100 at about 1000 image forwards per second in batches. That is between the 0.0022 s per input of `experiments/psbd_cost/` (unbatched overhead included) and the 39 models in 11 min of `experiments/why_token_masking_works/`. 3 units recur.

| unit | what runs | forwards per model | time per model | time on the 10 models |
|---|---|---|---|---|
| U1 | 1 deterministic condition on 256 clean and triggered pairs | 512 | 1 s | 10 s |
| U2 | 1 stochastic condition on 256 pairs, 10 passes | 5120 | 5 s | 1 min |
| U3 | 1 operator at 1 rate on the 2000 validation images plus 1000 pairs, k = 3 | 12000 | 12 s | 2 min |

A 6-rate ladder of a new probe is therefore about 1.2 min per model and 12 min on the 10 models, which is the reduced sweep I propose for exploration. The full PSBD splits (2000 validation, 8000 clean, about 7200 triggered) cost about 4 times more and belong to the confirmation run only. Swin-S costs about 0.8 times as much per forward (`experiments/psbd_cost/`).

## Source experiments in the literature

Each entry gives the design in 2 or 3 sentences, what it revealed and what it means for this project. Papers are grouped by the kind of manipulation.

## Knockout, patching and circuit methods

**Causal tracing (Meng et al., NeurIPS 2022, arXiv 2202.05262).** The authors corrupt the subject tokens' embeddings of a factual prompt with Gaussian noise, then restore 1 hidden state at a time from the clean run and measure how much of the correct answer's probability returns. A few middle-layer MLP states at the last subject token restore most of it. For us the same grid over (block, token) with a clean and a triggered run of the same image is what `experiments/failure_modes/` already does at blocks 4, 8 and 12. The gaps are the missing blocks and the direction of the patch (below).

**Attention knockout (Geva et al., EMNLP 2023, arXiv 2304.14767).** They block attention edges from chosen source positions to the predicting position in a window of layers and watch the answer's probability. Blocking subject-to-last edges only hurts in a specific band of upper layers, which located where information moves. For us this is the cleanest way to ask whether the class token reads the trigger directly or through other patch tokens, which token masking at the attention input cannot separate because it hides a token from every query at once.

**Path patching and the IOI criteria (Wang et al., arXiv 2211.00593).** They patch activations only along specific paths between components and judge a circuit by faithfulness (the circuit alone reproduces the behavior), completeness (removing it from the full model breaks the behavior as much as removing it from the circuit) and minimality (every part matters). For us the trigger-token route to the class token is a candidate circuit, and these 3 checks are the standard it should meet.

**Best practices of activation patching (Zhang and Nanda, ICLR 2024, arXiv 2309.16042).** They compare Gaussian noising of input embeddings with symmetric token replacement and several metrics on factual recall and IOI. Gaussian noising puts the model off distribution and can give illusory localizations, so they "recommend STR whenever possible", and they advise against probability as the metric because it can miss negative components, favoring logit difference. For us this means reporting the target-minus-true logit difference beside survival, and using a clean twin (symmetric replacement) rather than noise wherever a corruption is needed.

**How to use and interpret activation patching (Heimersheim and Nanda, arXiv 2404.15255).** A tutorial that separates denoising (patching clean into corrupt, which finds sufficient components) from noising (corrupt into clean, which finds necessary ones) and works through an AND gate and an OR gate. In an OR circuit noising either input alone does nothing and only denoising reveals each branch, in an AND circuit the reverse holds. This is exactly our principle (a): the backdoor's late reads are an OR, so noising 1 block (the "last 1" row of section A, 0.934 kept) cannot find them and the denoising direction can.

**Causal scrubbing (Chan et al., Alignment Forum, 2022).** A hypothesis about which parts of a model matter is converted into a set of resampling ablations: every activation the hypothesis says is irrelevant is replaced by its value on another input that the hypothesis treats as equivalent, and the share of performance kept is the score. For us it is the test that "the triggered prediction depends only on the trigger tokens' residual entries and the class token's late reads".

**ACDC (Conmy et al., NeurIPS 2023, arXiv 2304.14997).** Automated circuit discovery prunes edges of the computational graph one at a time, keeping an edge when removing it changes a KL metric by more than a threshold, and recovers hand-found circuits in small language models. For a ViT with 197 positions and 12 blocks the per-position graph is large and the method is costly, so I propose it only on a head-level graph with positions pooled into trigger, class token and rest.

**Attribution patching (Nanda, blog post, 2023 and Syed et al., NeurIPS 2023 ATTRIB workshop, arXiv 2310.10348).** The effect of patching every activation is approximated at once by the gradient of the metric times the activation difference between the 2 runs, 1 forward and 1 backward pass in total. Syed et al. find it recovers circuits better than ACDC at a fraction of the cost, and Kramár et al. (arXiv 2403.00745) refine it (AtP*) for its failure cases. For us it is the cheap first pass over all (block, token, head) components before exact patching of the top candidates.

**Optimal ablation (Li and Janson, arXiv 2409.09951).** They classify ablations into zero, mean and resample ablation, show that each deletes information and also injects a distribution shift, and propose replacing a component with the constant that minimizes the loss. For us this is the lens on E6: PSBD-TM is a constant ablation at the attention input (every masked token becomes $\beta$), which is neither the mean nor a resample.

**Self-repair (McGrath et al., arXiv 2307.15771 and Rushing and Nanda, ICML 2024, arXiv 2402.15390).** Ablating an attention layer in a language model makes later layers compensate for part of the lost effect (the hydra effect), and the compensation is spread over many components. For us it predicts that knocking out the trigger read in 1 late block understates that block's normal contribution, because the others take up the slack.

**Interpretability illusions (Makelov et al., arXiv 2311.17030 and Bolukbasi et al., arXiv 2104.07143).** Makelov et al. show that patching along a subspace can switch on a dormant parallel pathway and look like localization where none exists, and Bolukbasi et al. show a unit that seems to encode 1 concept on 1 dataset and another on a second. For us it means a backdoor direction fitted on paired images must be validated on held-out pairs and on a second dataset before any patching along it is read causally.

**Localization against editing (Hase et al., NeurIPS 2023, arXiv 2301.04213).** Where causal tracing localizes a fact does not predict where editing it works best. For us it warns that "the trigger is read in blocks 9 to 12" does not by itself say that perturbing those blocks is the best probe, which the band ceiling in `experiments/probe_fusion/` already shows.

**Probes with control tasks (Hewitt and Liang, EMNLP 2019, arXiv 1909.03368).** A probe's accuracy is only meaningful relative to its accuracy on a control task with random labels of the same structure, and selectivity is the difference. For us every per-token probe (for example L26's warped against noise-warped probe) needs a control-label run and must report selectivity.

**Logit lens on ViT and the tuned lens (Vilas et al., NeurIPS 2023, arXiv 2310.18969 and Belrose et al., arXiv 2303.08112).** Vilas et al. project intermediate ViT tokens onto the class embedding space and find class identity emerges in the later blocks, and Belrose et al. fit a small affine map per layer so intermediate states decode more faithfully than the raw unembedding. For us a per-block readout of the target-minus-true logit difference shows where the backdoor becomes decodable and where a perturbation removes it.

**Decomposing CLIP's image representation (Gandelsman et al., ICLR 2024, arXiv 2310.05916).** They mean-ablate layers of CLIP ViTs and decompose the output into per-head and per-token direct contributions, finding that the last few attention layers carry most of the direct effect. I did not check the exact layer count in the paper. For us per-head direct logit attribution in blocks 9 to 12 is the cheap readout of which heads carry the trigger into the class token.

**Attention rollout and attention norms (Abnar and Zuidema, arXiv 2005.00928 and Kobayashi et al., EMNLP 2020, arXiv 2004.10102).** Rollout multiplies attention matrices across layers with the identity added for the residual path, and Kobayashi et al. show that the norm of the weighted value vector, not the weight alone, measures how much a token contributes. For us an attention-rank probe (hide the most-attended tokens) should rank by weight times value norm, not raw attention.

**Prisma (Joseph et al., arXiv 2504.19475).** An open toolkit that brings hooks, patching, sparse autoencoders and lens tools to vision and video transformers. We already have removable hooks in `models/positions.py`, so Prisma matters mainly as a reference implementation to cross-check a patching result.

## Tokens, sinks and missingness in ViTs

**Registers (Darcet et al., arXiv 2309.16588) and test-time registers (Jiang et al., NeurIPS 2025, arXiv 2506.08010).** Large ViTs develop a few high-norm tokens in low-information background patches that hold global information and attract attention, and adding extra register tokens removes these artifacts. Jiang et al. trace the outliers to a few "register neurons" and shift them into an extra untrained token at test time, getting most of the benefit without retraining. For us this is the machinery for a sink-only probe: inject extra sink tokens without masking anything and see whether that alone reproduces PSBD-TM (E6).

**Massive activations and attention sinks (Sun et al., COLM 2024, arXiv 2402.17762 and Xiao et al., ICLR 2024, arXiv 2309.17453).** A handful of activations are orders of magnitude larger than the rest, act as fixed biases and concentrate attention, and Xiao et al. show that keeping the sink tokens is what stabilizes attention. The earlier memo records that BadNets trigger tokens manufacture their own high-norm token (L24). For us the question is whether the $\beta$ tokens of PSBD-TM become a new sink that the triggered and clean inputs react to differently.

**Intriguing properties of ViTs (Naseer et al., NeurIPS 2021, arXiv 2105.10497).** They drop random, salient or background patches and shuffle patch order, finding that ViTs keep up to 60% top-1 accuracy on ImageNet after randomly occluding 80% of the image and that the position encoding adds less structure than expected, since shuffled patches cost them little. For us this is the baseline expectation for clean fragility under masking and under position-embedding jitter: pretrained ViTs are robust to both, and our fine-tuned models need about 70% of tokens (E7), a large departure worth noting.

**Missingness bias (Jain et al., ICLR 2022, arXiv 2204.08945).** Blacking out pixels biases a ResNet toward unrelated classes, while dropping the corresponding tokens of a ViT approximates missingness more faithfully. For us a zeroed token before LayerNorm is neither a black patch nor a dropped token, which is why E6 needs a direct test.

**Smoothed ViTs for patch robustness (Salman et al., CVPR 2022, arXiv 2110.07719).** Certified defenses against adversarial patches classify many column ablations of an image, and ViTs handle ablated inputs well when the ablated tokens are dropped instead of zeroed. For us the column-ablation vote is a PSBD relative, and dropping tokens from the key set is the version of token masking without a sink.

**Token merging and dynamic pruning (Bolya et al., ICLR 2023, arXiv 2210.09461 and Rao et al., NeurIPS 2021, arXiv 2106.02034).** ToMe merges the most similar tokens in each block without training and keeps accuracy, and DynamicViT prunes tokens with a learned predictor. For us ToMe is a training-free perturbation that removes redundant tokens first, which should leave a dissimilar patch trigger untouched.

**DropKey (Li et al., CVPR 2023, arXiv 2208.02646).** A regularizer that drops keys before the softmax instead of dropping attention weights after it. It is the operator the earlier design note proposed as "attention map masking", and the true key mask below.

**Occlusion without retraining (Hooker et al., NeurIPS 2019, arXiv 1806.10758).** ROAR retrains the model after removing the most important pixels, because removal without retraining confounds information loss with distribution shift. We cannot retrain for every mask, so the substitute is resample ablation with in-distribution replacement tokens.

## Minimal inputs, transplants and shortcuts

**Sufficient input subsets and overinterpretation (Carter et al., AISTATS 2019, arXiv 1810.03805 and NeurIPS 2021, arXiv 2003.08907).** Backward selection finds the smallest pixel subset that keeps a confident prediction, and CIFAR-10 and ImageNet classifiers stay confident on 5% of pixels that mean nothing to a human. For us the minimal sufficient token set of a triggered input should be the trigger itself (E1) and far smaller than a clean input's, which `experiments/why_psbd_works/` measured as 0.05 against 0.60 on WaNet.

**Clever Hans (Lapuschkin et al., Nature Communications 2019, arXiv 1902.10178).** A Fisher-vector model classified horses by a copyright tag, which the authors showed by pasting the tag onto a car image and getting "horse". For us the transplant is the trigger pasted onto blank, noise and foreign images.

**SentiNet (Chou et al., DLS 2020, arXiv 1812.00292).** The salient region of a suspicious image is pasted onto held-out clean images and the fooled rate is compared with that of an inert pattern of the same size. For us the inert-pattern control is the part to copy, and the port's below-chance reading (Q23) is a warning that the transplant statistic's sign is subtle.

**Rethinking the trigger (Li et al., arXiv 2004.04692).** On ConvNets, moving a BadNets patch slightly or changing its appearance at test time drops the attack success rate sharply, and flipping or shrink-padding the test image is a cheap defense. For us moving the trigger across the token grid tests position binding (E9) on ViT and, because the grid position sets how many tokens a patch covers, manipulates m in the $p^{4m}$ law.

**Backdoor attacks on ViTs (Subramanya et al., arXiv 2206.08477 and WACV 2024).** Attention-based interpretation maps highlight the trigger on ViTs but not on CNNs, and blocking the top region of the map at test time cuts the attack success rate with a small clean-accuracy cost. For us this predicts that hiding the most-attended tokens flips the sign for patch triggers.

**Patch processing on ViTs (Doan et al., AAAI 2023, arXiv 2206.12381).** Randomly dropping patches detects patch-based triggers and shuffling patches mitigates blending-based ones, a response the authors did not see on ConvNets. They group WaNet with blending-based attacks in the text, and I could not confirm that they evaluated it. For us patch shuffling is a candidate probe for warp and blend triggers.

**Backdoor directions in ViTs (Karayalçin et al., arXiv 2603.10806).** A trigger direction in activation space steers the backdoor in both directions, and static patch triggers follow a different internal logic from stealthy distributed ones. The project's own direction measurements (L8) confirm the direction and disagree on WaNet's per-token legibility.

**Pruning, spectral and clustering analyses (Liu et al., arXiv 1805.12185, Tran et al., NeurIPS 2018, arXiv 1811.00636 and Chen et al., arXiv 1811.03728).** Fine-pruning plots clean accuracy and attack success against the number of pruned dormant neurons, spectral signatures separate poisons along the top singular vector of centered class representations and activation clustering splits each class's last-layer activations in 2. The project already refuted backdoor neurons in the MLP basis (L7) and confirmed the direction (L8), so these add little beyond a pruning curve as a figure.

**Anti-backdoor learning (Li et al., NeurIPS 2021, arXiv 2110.11571).** Poisoned examples' training loss falls faster than clean examples' early in training. The earlier memo's L4 and the queued `early_loss_signal` cover it, and it needs training.

## Perturbation consistency and input transformations

**STRIP, SCALE-UP and TeCo (Gao et al., ACSAC 2019, arXiv 1902.06531, Guo et al., ICLR 2023, arXiv 2302.03251 and Liu et al., CVPR 2023, arXiv 2303.18191).** STRIP superimposes clean images and flags low prediction entropy, SCALE-UP multiplies pixel values and flags consistent predictions, and TeCo applies 15 corruption types at growing severity and flags inputs whose severity of first prediction change varies widely across corruption types. TeCo's premise is that clean images are equally robust to every corruption while triggered images are robust to some and fragile to others. For us TeCo is the closest published relative of a WaNet specialist, since it reads 0.901 on WaNet in our port against 0.519 for STRIP.

**WaNet (Nguyen and Tran, ICLR 2021, arXiv 2102.10369).** The trigger is a smooth backward warp from a 4 by 4 control grid, and noise mode trains on images warped by the trigger field plus a random per-pixel field, $W(x, M + \mathrm{rand}_{[-1,1]}(h, w, 2))$, with their true label. Without noise mode the model "cheated" by learning pixel-wise artifacts and Neural Cleanse caught it with small scattered trigger patterns, and with it the backdoor passed. The authors also write that STRIP's superimposition "will modify the image content and break the backdoor warping", and their CIFAR-10 models reach 93.16% accuracy on noise-mode images against 94.42% clean. For us noise mode is a direct prediction that triggered WaNet inputs are fragile to small random warps.

**Frequency perspective (Zeng et al., ICCV 2021, arXiv 2104.03413).** Common triggers carry high-frequency artifacts, a low-pass filter removes many of them and the authors build smooth triggers (the LF attack) to avoid that. For us a blur or sub-pixel resampling perturbation should separate high-frequency triggers (BadNets checkerboard, Blend's random pattern, BPP quantization, WaNet's interpolation texture) from low-frequency ones (LF, SIG).

**Purification by transformation (Shi et al., NeurIPS 2023, arXiv 2303.12175, Sun et al., arXiv 2303.15564, Yang et al., NeurIPS 2024 and Miah and Bi, IJCNN 2026, arXiv 2602.07197).** ZIP destroys triggers with a linear transformation such as blur or downsampling and restores the image with a diffusion model, Mask and Restore masks and inpaints with a masked autoencoder, SampDetox finds that triggers of low visibility are destroyed by light noise and highly visible ones need strong noise, and Lite-BD's preliminary study on ResNet-18 CIFAR-10 over BadNets, Blend, WaNet, SIG and BPP finds down-upscaling the most disruptive of 10 transformations on average, followed by blur. In the Lite-BD result tables I read, a resize-based stage leaves BadNets at an attack success rate of 1.000 and drops WaNet and BPP to about 0.05, but I did not confirm which method each column names. For us a resampling perturbation is a candidate specialist for WaNet and BPP that is blind to BadNets.

**Strong augmentation (Borgnia et al., arXiv 2011.09527).** Mixup and CutMix during training sanitize many poisoning and backdoor attacks. It needs training and matters here only through our own `_aug` WaNet models (below).

**Monte Carlo dropout and randomized smoothing (Gal and Ghahramani, ICML 2016, arXiv 1506.02142 and Cohen et al., ICML 2019, arXiv 1902.02918).** Dropout at test time approximates Bayesian inference, and a Gaussian-smoothed classifier's majority vote is certifiably robust in a radius. The earlier memo already refuted pass uncertainty as the carrier (P2), and smoothing matters here as the certified analog of the column-ablation vote.

**PatchCleanser (Xiang et al., USENIX Security 2022, arXiv 2108.09135).** 2 rounds of masking with a set of masks that is guaranteed to cover any patch of a given size, and a prediction is certified when every 1-mask prediction agrees. For us the covering-set idea is the deterministic version of token masking, and its agreement test is PSBD with hard labels.

**Robustness-aware perturbations (Yang et al., EMNLP 2021, arXiv 2110.07831).** In NLP, poisoned samples are more robust than clean ones to a word-level perturbation crafted on clean data, and the gap detects them. It is the text precedent for choosing the perturbation on clean data only.

## What our own records already settle

The plan below does not repeat these. Section A of `experiments/why_token_masking_works/` masks the trigger tokens deterministically by block band and against random tokens (4 tokens in all 12 blocks leave 0.002 of triggered BadNets predictions, 4 random tokens leave 1.000, blocks 1 to 4 leave 1.000, blocks 9 to 12 leave 0.200). Section B records PSBD-TM's masks and confirms the $p^{4m}$ law (Tiny 450 of 7670 passes against 0.0625 predicted). Section C restricts PSBD-RD to trigger positions (0.387 kept against 0.398 for all positions). Section D keeps random visible subsets (global triggers keep 0.90 to 0.95 of excess ASR at 30% visible, WaNet 0.183). `experiments/failure_modes/` denoises the clean residual stream into the triggered run at blocks 4, 8 and 12 for trigger, random and class-token groups on WaNet and SIG. `experiments/why_psbd_works/` tests margin, direction, redundancy, low dimension, flatness, neuron bias, pass uncertainty, out of distribution, trigger neurons and missingness on 4 groups.

3 facts that bear on the plan came out of files I read for this memo. Each names the file it comes from.

- **Our WaNet models differ in noise mode.** `attacks/wanet.py` implements noise mode as the reference code does (offsets drawn uniformly in $[-1/h, 1/h]$ in normalized coordinates, about 0.48 native pixels at most, the same scale as the trigger's own displacement of about 0.24 pixels on average). Commit 52e3587 restored it on 2026-09-08. Every ViT WaNet model on the panel and the CIFAR-10 and GTSRB seed replicates carry `cover_rate` 0.1 or 0.2 in `args.json`. The ViT CIFAR-100 and Tiny seed replicates (trained 2026-09-07) and every seed-0 Swin WaNet model carry `cover_rate` 0.0, and the Swin ones have `git_commit` null, so I infer they predate the restoration and were trained without noise mode. That inference rests on dates, and a Swin run log would settle it. The Swin seed replicates `swin_{cifar10,cifar100,gtsrb}_wanet_0_1_seed_{1,2}` carry 0.2.
- **Noise mode does not explain the CIFAR-10 failure by itself.** In the `adaptive` block of each model's `psbd_metrics.json` (which reads 0.484 on `vit_cifar10_wanet_0_1` where the headline reads 0.459, so the column is indicative), PSBD-TM reads 0.936, 0.965 and 0.958 on other noise-mode ViT WaNet models and 0.899 to 0.965 on the Tiny replicates without noise mode. On Swin it reads 0.958 to 0.982 on the noise-mode seeds against 0.912 to 0.995 on the seed-0 models without it. The Swin advantage in `experiments/swin_mechanism/` therefore survives the noise-mode confound its 6 matched pairs carry (noise-mode ViT against Swin without noise mode), though that README should say so.
- **WaNet does not survive augmentation in our recipe.** The 2 WaNet models trained with standard crop and flip augmentation reach attack success 0.024 (`vit_cifar100_wanet_0_05_aug`) and 0.475 (`vit_tiny_wanet_0_05_aug`) against 0.649 and 0.941 without it (`metrics.json`). A 1 pixel crop offset or a flip makes the trigger unlearnable here, so the learned feature is tied to exact pixel alignment. The WaNet paper trained with random crop, rotation and flip and still implanted, so this is a property of our 15-epoch fine-tuning recipe.

## Ranked experiment plan

The ranking divides how much an experiment explains (how many open explanations it decides, and whether it changes what the paper or the detector does) by its cost. CPU means the answer is in the cached per-pass tensors (`per_pass_probs` and `per_pass_argmax`, shape (k, N), under `results/<folder>/psbd/<placement>/`) or in `psbd_metrics.json`. Every GPU experiment is inference only on existing checkpoints. Section numbers in the "extends" column refer to `experiments/why_token_masking_works/README.md`.

| rank | ID | question | CPU | cost on the 10 models | extends |
|---|---|---|---|---|---|
| 1 | X1 | where PSBD-TM sends triggered WaNet predictions | yes | seconds | `failure_modes` |
| 2 | X2 | is the 0.459 WaNet cell a checkpoint accident | no | 5 min on 2 replicates | `wanet_cifar10_audit` |
| 3 | X3 | which fusion rule a specialist needs | yes | seconds | `probe_union` |
| 4 | X4 | which depth band breaks which trigger | yes | seconds | section A |
| 5 | X5 | are the token-mask results an ablation artifact | no | 10 min | sections A and D |
| 6 | X6 | does noise mode make triggered WaNet fragile to random warps | no | 8 min, plus 22 min on the WaNet set | `failure_modes` |
| 7 | X9 | does triggered survival follow $1 - p^{mB}$ | no | 4 min on the 3 patch models | section B |
| 8 | X7 | the BadNets dose-response family | no | 6 min on the 3 patch models | section A |
| 9 | X8 | does the class token read the trigger directly | no | 1 min on the 3 patch models | section A |
| 10 | X19 | which pass statistic separates best | yes | seconds | `why_psbd_works` |
| 11 | X11 | is the trigger sufficient on a blank image | no | 3 min | new |
| 12 | X12 | where the triggered state is sufficient, by denoising | no | 3 min | `failure_modes` |
| 13 | X10 | is the backdoor bound to the trigger's position | no | 5 min on the 3 patch models | section A |
| 14 | X15 | the WaNet transformation battery | no | 7 min, plus 20 min on the WaNet set | new |
| 15 | X16 | is WaNet a position-content coherence check | no | 7 min | new |
| 16 | X17 | global triggers from contiguous against scattered tokens | no | 7 min | section D |
| 17 | X18 | test-time trigger strength against PSU | no | 6 min | `why_psbd_works` |
| 18 | X14 | stratified token masking | no | 12 min | `operator_ranking` |
| 19 | X13 | true key mask and sink-only probe | no | 36 min | `why_psbd_works` |
| 20 | X20 | logit lens and per-head attribution under PSBD-TM | no | 2 min | `backdoor_direction_layers` |
| 21 | X21 | self-repair after a late knockout | no | 1 min | new |
| 22 | X22 | attention temperature and attention-rank masking | no | 24 min | new |

**X1. Destination of shifted triggered WaNet predictions (CPU).** The question is where a triggered WaNet prediction goes when PSBD-TM moves it on the failing model. From `rate_<adaptive>_backdoor.pt` of `before_attention_norm_token_mask`, take `per_pass_argmax`, recover each triggered row's true class through `split_manifest.json` and `defenses.decision.pair_clean_to_backdoor`, and split the changed passes into those landing on the true class and those landing elsewhere. Run it on the 5 ASR-clearing ViT WaNet models, the 6 ViT replicates without noise mode and the 14 Swin WaNet models, and on BadNets and Blend dev models as the control. E8 predicts that at least 0.7 of the triggered shifts on noise-mode models land on the true class, since the masked triggered input reads as a rejected warp, and a much smaller share on models without noise mode. Generic fragility predicts destinations that mirror the clean shifts (largest class share 0.66 on WaNet under PSBD-TM, true class near 1 over the class count). The figure is a grouped bar chart of destination shares (true class, target, largest other class, rest) for triggered and clean rows, 1 group per model, ordered by noise mode. If E8 holds, the paper can name the reason PSBD-TM inverts on WaNet, and the WaNet specialist below becomes the natural fix.

**X2. The replicates of the failing cell.** `vit_cifar10_wanet_0_1_seed_1` and `_seed_2` exist with noise mode and attack success 0.885 and 0.889 (`metrics.json`) and have never been swept. Sweep `before_attention_norm_token_mask` and `post_residual` on both with `cli.sweep`, reduced splits first (5 min for both) and full splits for the record (about 30 min). If both read at least 0.9, the 0.459 is a checkpoint property and the paper should say so. If both read at most 0.6, CIFAR-10 WaNet at 10% with noise mode is a recipe that defeats PSBD-TM and the paper has a real failure class. The figure is 1 dot per seed and placement with the seed-0 cell marked.

**X3. Fusion rules for a specialist (CPU).** Combine PSBD-TM with `pre_residual_blocks_9_12`, `pre_residual_blocks_5_8` and `post_residual` from the caches under 5 rules: the mean of fractional PSU (the coordinator's +0.013 reading), min-rank, weighted min-rank at budget shares 0.8 and 0.2 and at 0.9 and 0.1 (defined in the WaNet section) and Fisher's combination of clean-validation percentiles. Report AUROC, TPR at 10% and 20% FPR and the FPR realized on the clean test split. Routed patch triggers and computed WaNet (E10) predict that the late residual member helps WaNet and Blend and hurts TaCT under the mean and min-rank rules, and that the weighted rule keeps TaCT within 0.01 while keeping most of the WaNet gain. The figure is TPR at 10% FPR per attack for each rule, with PSBD-TM alone as the reference line.

**X4. Depth bands from the caches (CPU).** All 10 dev models carry `before_attention_norm_blocks_{1_4,5_8,9_12}_token_mask` and `pre_residual_blocks_{1_4,5_8,9_12}`. For each band and attack read the triggered minus clean shift share at the band's top rate. E1 predicts that token masking in blocks 1 to 4 or 5 to 8 leaves triggered patch predictions alone while breaking clean ones, and that blocks 9 to 12 hit both. E10 predicts that residual dropout in blocks 1 to 4 breaks triggered WaNet more than triggered BadNets, because WaNet's evidence is assembled early (block 4 recovery of 0.880 for the most displaced tokens in `failure_modes`) while BadNets' is only carried. E2 predicts that no token-mask band moves global triggers. The figure is a heat map, attack by band, of the triggered minus clean shift share, 1 panel per operator.

**X5. Resample and mean ablation versions of the key token-mask results.** PSBD-TM zeroes the attention input, so after LayerNorm every masked token is $\beta$. That is a constant ablation that never occurs on real data (E6). Repeat the rows of section A (trigger tokens masked in all 12 blocks or in blocks 1 to 4, 5 to 8 or 9 to 12, plus random tokens) and section D (visible fraction 0.6, 0.3 and 0.1, 3 patterns each) under 4 replacement rules at the attention input: the current zero, the mean token of that position and block over 256 clean validation images, a resample of that position and block from a fixed bank of 64 clean images, and a key mask that removes the token from the softmax. The cost is about 60 U1 conditions, 1 min per model and 10 min on the 10 models, plus seconds to cache the bank. If the results are information removal (E1, E7), all 4 rules agree within 0.05 on every row. If E6 matters, the zero rule departs from the other 3, most likely in the clean rows of D and on WaNet. The figure is a paired dot plot, 1 row per condition, 4 dots per row. This experiment decides whether every token-mask statement in the paper is about information or about the artifact, so it ranks high despite its cost.

**X6. Random sub-pixel warps and noise mode.** This tests the key WaNet hypothesis, set out in full in the WaNet section. The operator resamples the native-resolution image at the identity grid plus $a\,\xi/h$ with $\xi$ uniform in $[-1,1]$, before the model's own resize, for $a$ in 0.25, 0.5, 1 and 2, with k = 3. Run it on the 10 dev models (8 min) and on the WaNet set: the 5 ASR-clearing ViT models with noise mode, `vit_gtsrb_wanet_0_05` and `vit_cifar100_wanet_0_1` labeled as below the bar, the 6 ViT replicates without noise mode, the 8 seed-0 Swin models without it and the 6 Swin replicates with it (about 22 min in all). Read triggered retention, the share of triggered flips to the true class, clean retention and clean flips to the target. E8 predicts triggered retention at most 0.2 at $a = 1$ on noise-mode models with at least 0.8 of the flips going to the true class and clean retention at least 0.9. It predicts the opposite on models without noise mode (triggered retention at least 0.9 and more clean images sent to the target). Generic stability (E1, E3) predicts triggered retention at or above clean retention at every amplitude on every model. The figure has 2 panels, retention against $a$ for triggered and clean, noise mode on the left and without on the right, 1 line per model colored by architecture.

**X7. The BadNets dose-response family.** The full design is in its own section below. It covers every subset of the 4 trigger tokens, deterministic in all blocks and in the late blocks, matched random and adjacent controls, trigger-only visibility and a pixel-level partial trigger. It runs on the 12 BadNets models and the 3 trigger-conditional TaCT models for the paper (about 30 min) and on the 3 patch models of the dev set for exploration (6 min).

**X8. Attention knockout of the trigger read.** Set the pre-softmax logits to negative infinity on 3 edge sets: class-token query to trigger-token keys, non-trigger patch queries to trigger-token keys, and both. Apply each in blocks 1 to 4, 5 to 8, 9 to 12 and in each single block 9 to 12, with a random 4-key control per band. Models are the 12 BadNets and 3 trigger-conditional TaCT models. On the dev set they are `vit_cifar10_badnet_a2o_0_01`, `vit_gtsrb_tact_0_05` and `vit_cifar10_tact_0_01`. A direct read (E1 in its strong form) predicts that the class-token knockout in blocks 9 to 12 alone reproduces the 0.200 of section A and that the patch-only knockout leaves attack success near 1. A relay predicts the reverse: the trigger's content spreads to other tokens by block 9, so blocking only the class token's edges leaves the attack intact and blocking the patch edges in blocks 5 to 8 matters. The figure is a grid of attack success, rows for edge sets and columns for bands, with the logit difference in each cell. It costs 20 U1 conditions, 20 s per model, once the attention wrapper can take a mask. `defenses.operators.masked_attention_forward` already recomputes attention through `scaled_dot_product_attention`, which accepts an `attn_mask`.

**X9. The survival law under stochastic trigger-only masking.** Mask only the trigger tokens, each independently per block with probability $p$ in 0.3, 0.5, 0.7, 0.8, 0.9, 0.95 and 0.99 over 10 passes, then separately with 1 draw per pass shared by all 12 blocks. E1 with $B$ late reading blocks predicts the survival laws below. The table gives the numbers for $B = 4$ (computed).

$$S_{\mathrm{indep}}(p) = 1 - p^{mB}, \qquad S_{\mathrm{shared}}(p) = 1 - p^{m}$$

| symbol | meaning |
|---|---|
| $S(p)$ | share of triggered predictions still on the target |
| $p$ | per-block masking probability of each trigger token |
| $m$ | number of trigger tokens that each suffice alone |
| $B$ | number of late blocks that each read the trigger |

| m | p 0.3 | p 0.5 | p 0.7 | p 0.8 | p 0.9 | p 0.95 | p 0.99 |
|---|---|---|---|---|---|---|---|
| 1 | 0.992 | 0.938 | 0.760 | 0.590 | 0.344 | 0.185 | 0.039 |
| 2 | 1.000 | 0.996 | 0.942 | 0.832 | 0.570 | 0.337 | 0.077 |
| 4 | 1.000 | 1.000 | 0.997 | 0.972 | 0.815 | 0.560 | 0.149 |

Fitting $m$ and $B$ to the measured curve gives the effective number of sufficient tokens and reading blocks. An additive account (each trigger pixel adds evidence and no token suffices alone) predicts a graded curve that does not follow either power law. The figure plots survival against $p$ with both fitted laws. 14 U2 conditions cost 70 s per model.

**X10. Trigger relocation and swaps.** Paste the 3 by 3 checkerboard at 5 positions (the trained corner, the other 3 corners and the center), plus 2 offsets of the trained corner chosen so the patch covers 6 and 9 tokens instead of 4, then swap trigger-token content with 4 other positions after the patch embedding (content moves, position embedding stays) and swap position embeddings alone. E9 predicts attack success drops sharply off the trained corner, as Li et al. found on ConvNets. It also predicts that the position-embedding swap breaks it. A content-only read predicts attack success stays near 1 everywhere, which Naseer et al.'s near permutation invariance makes plausible. The grid offsets change $m$ on the same model, so PSBD-TM's triggered survival at the adaptive rate should rise with $m$ as $1 - p^{4m}$. The figure plots attack success and PSBD-TM AUROC per position, labeled with $m$. It costs about 12 U1 conditions plus 3 U3 readings, 5 min on the 3 dev patch models.

**X11. The trigger on blank, noise and foreign images.** Paste the trigger onto a gray image, a black image, uniform noise, 256 images of another dataset and a clean image of the target class, and score each with PSBD-TM at the model's adaptive rate. E1 predicts the target on at least 0.9 of them and a low PSBD-TM score (flagged), while the same images without the trigger get a default class and a high score. A context-dependent trigger predicts low attack success off natural images. For WaNet the analog is the warp applied to gray and noise images, which should carry nothing because a warp of a flat image is flat. The figure shows attack success and the mean PSBD-TM score per carrier, triggered against untriggered. It costs 5 carriers by 2 by U3, 2 min per model.

**X12. Denoising maps of sufficiency.** `failure_modes` patched the clean residual stream into the triggered run (denoising the clean answer) at blocks 4, 8 and 12. The missing direction patches the triggered stream into the clean run at every block 1 to 12, for 4 groups: trigger tokens, class token, a random group of the same size and the tokens adjacent to the trigger. Run attribution patching over all (block, token) pairs first, then exact patching of the top 20. The metric is the logit difference, target minus true class, as Zhang and Nanda recommend. E1 predicts that the trigger tokens' stream alone is sufficient at every block up to about 9 and stops being sufficient after the class token has read it, while the class token becomes sufficient only in blocks 9 to 12. For WaNet, E10 predicts no small token group sufficient early and a class-token takeover late. The figure is a heat map, block against group, of the recovered logit difference, with the attribution estimate in a twin panel. It costs 48 U1 conditions plus 2 passes for the attribution, under 1 min per model.

**X13. True key mask and sink-only probe.** 3 operators at the attention input: a key mask at rate $p$ (masked positions get negative infinity logits for every query, their own queries and residual updates untouched), a sink injection that appends $r = p \cdot 196$ extra key and value vectors equal to $W_k\beta + b_k$ and $W_v\beta + b_v$ to every block's attention without masking anything, and the existing `token_substitute`. Ladder each at 6 rates and read at the matched 0.6 and adaptive 0.8 rules. E6 predicts the sink-only probe reproduces a large share of PSBD-TM's clean shift and separation and the key mask differs from PSBD-TM on WaNet. If E6 is false, sink injection reads AUROC near 0.5 with small shifts and the key mask matches PSBD-TM within 0.02. The figure is AUROC per attack for the 4 operators. It costs 18 U3 readings per model, 36 min on the 10.

**X14. Stratified token masking.** The design is in the novel designs section. It asks whether guaranteeing every token 1 visible read in every window of 4 blocks raises triggered patch survival to 1 while leaving clean fragility unchanged. It costs 12 min on the 10 models.

**X15. The WaNet transformation battery.** Gaussian blur at $\sigma$ 0.5 and 1 native pixel, bilinear down-up scaling by 0.5, translation by 1 and 2 pixels with reflection padding, a horizontal flip, a fresh smooth random warp at strengths 0.25 and 0.5, and JPEG at quality 75. All but the random warp are deterministic. Read triggered and clean retention and triggered flips to the true class on the WaNet set and the 10 dev models. The augmentation fact above and E8 predict that translation and flip break triggered WaNet while clean CIFAR retention stays at 0.85 or above. Lite-BD and ZIP predict that blur and down-up scaling break WaNet and BPP but not BadNets. The random smooth warp separates an exact-field reader (triggered WaNet breaks) from a "warped means target" reader (clean images go to the target). The figure is a retention matrix, transform against attack, for triggered and clean. It costs about 40 s per model.

**X16. Position-content coherence for WaNet.** 4 conditions: position embeddings permuted within every 2 by 2 block of tokens, position embeddings with Gaussian jitter at 0.25 of their norm, content permuted within 2 by 2 blocks with position embeddings kept and a whole-grid shuffle. If WaNet is read as coherence between what a token shows and where it sits (E8 in its relational form), permuting position embeddings breaks triggered WaNet more than clean inputs. If WaNet is read from per-token texture alone, triggered WaNet survives all 4 like a global trigger. The BadNets rows of the same run test E9. The figure is retention per condition for triggered and clean. Doan et al.'s PatchShuffle is the precedent. 8 U2 conditions cost 40 s per model.

**X17. Contiguous against scattered visible subsets.** Extend section D with visible fractions 0.3, 0.1 and 0.05 in 4 geometries: random scatter, 1 square window at a random place, horizontal stripes and vertical stripes, 3 draws each. Per-token legibility (Blend's per-pixel random pattern, BPP's quantization) predicts no dependence on geometry. Extent-dependent triggers (LF's low frequencies, SIG's sinusoid along the columns) predict that 1 window loses them and stripes along the oscillation keep them. WaNet under E8 predicts that no geometry keeps it at 0.1. The figure plots excess retention against the visible fraction, 1 line per geometry, 1 panel per attack. 36 U1 conditions cost 40 s per model.

**X18. Test-time trigger strength.** Vary Blend's $\alpha$ over 0.05, 0.1, 0.15, 0.2 and 0.3, LF's strength over 0.25 to 2 times the trained value, BPP's bit depth over 5, 4, 3 and 2 and WaNet's strength $s$ over 0.125 to 1. Read attack success, the logit difference and PSBD-TM's score at the model's adaptive rate. E3 predicts that the score of successful triggered inputs rises smoothly to the clean level as the trigger weakens, tracking the projection on the backdoor direction. E4 predicts it tracks confidence instead. The figure plots attack success and the median score against strength with the clean score band shaded. It costs 5 U3 readings per model, about 6 min on the 6 global and WaNet dev models.

**X19. Pass statistics from the caches (CPU).** From `per_pass_probs` of PSBD-TM and PSBD-RD compute the mean ratio (the current statistic), the worst pass, the best pass, the number of passes that keep the label and the spread across passes. E1's OR structure makes triggered patch survival nearly all-or-nothing per pass, so the worst-pass statistic should separate at least as well as the mean and raise TPR at 1% FPR. The figure is TPR at 1% and 10% FPR per statistic and attack.

**X20. Logit lens and per-head attribution under PSBD-TM.** Apply the final LayerNorm and head to the class token after every block, and attribute the last 4 blocks' target-minus-true logit difference to each head, for triggered and clean inputs, unperturbed and averaged over 10 PSBD-TM passes. E1 predicts the triggered difference appears in blocks 9 to 12 through a few heads that keep their contribution under masking, while clean evidence builds from block 6 on and loses it. The figure plots the logit difference against block, 4 curves. It costs 1 U2 per model.

**X21. Self-repair after a late knockout.** Knock out the class token's read of the trigger in block 12 only and measure how the per-head contributions of blocks 9 to 11 change. The hydra effect predicts they grow, which would explain why masking the trigger in the last block alone keeps 0.934 of triggered BadNets predictions. It costs a few U1 conditions.

**X22. Attention temperature and attention-rank masking.** Rescale each head's logits by a random temperature, sharpening ($\tau = 0.5$) and flattening ($\tau = 2$) separately, in blocks 9 to 12 and in all blocks. Separately, hide from the class token the $r$ tokens with the largest weight times value norm in each block. The class token puts 0.550 of its attention on the 4 BadNets trigger tokens in blocks 9 to 12 (`\RoutingBadnetClsLate`), so flattening and attention-rank masking should break triggered patch predictions (the sign flips) and sharpening should keep them while breaking clean ones. Global triggers should barely move. The figure is AUROC in both sign conventions per attack and variant. It costs 12 U3 readings per model.

**Needs training.** These are listed apart because the login node cannot train and each needs a PBS job. Each row names the model to train and the question it would settle.

| ID | model to train | question |
|---|---|---|
| T1 | ViT CIFAR-10 WaNet 10% without noise mode, seeds 0 to 2, beside the existing noise-mode replicates | noise mode as the only variable for X1, X6 and the specialist |
| T2 | WaNet with the noise amplitude at 0.5 and 2 times the reference | whether the jitter specialist's best amplitude follows the attacker's noise amplitude |
| T3 | BadNets with a random trigger position per poisoned image | E9, and whether PSBD-TM needs a fixed trigger position |
| T4 | BadNets with trigger sizes of 1, 4, 9 and 16 tokens | $m$ as a trained variable, since resizing a trigger at test time is not the same backdoor |
| T5 | a ViT with registers, fine-tuned with BadNets and WaNet | E6, whether a model with native sinks reacts to the $\beta$ tokens differently |
| T6 | 2 attack families absent from the panel (for example ISSBA and Refool) on CIFAR-10 and GTSRB | proxy attacks for the design protocol, so the design is chosen on attacks it is not evaluated on |

## The BadNets dose-response family

The user's example, masking exactly the trigger's tokens against random tokens and against half of them, expands into the family below. One geometric fact shapes every row. The trigger is a 3 by 3 checkerboard in the bottom-right corner of a 32 pixel image (`attacks/badnet.py`), and the model upsamples by 7 to 224, so the patch covers pixels 203 to 223 in each axis. Token 12 (pixels 192 to 207) receives 5 of those pixel rows and token 13 (pixels 208 to 223) receives 16. The 4 trigger tokens therefore carry very different amounts of trigger: (13,13) about 256 square pixels, (12,13) and (13,12) about 80 each and (12,12) about 25. These areas come from the geometry, bilinear upsampling blurs the edges by a few pixels and I did not measure them. On Tiny ImageNet (64 pixels, factor 3.5) the patch spans pixels 213.5 to 224 and falls inside token (13,13) alone, which is why `trigger_tokens` returns 1 token there. Every "k of 4" row below must say which k tokens, ordered by coverage.

A second account competes with E1 in these rows and needs a label. E11 (area-additive evidence) says each trigger pixel adds evidence to the target logit, no single token suffices and the logit difference falls in proportion to the masked trigger area.

| row | manipulation | E1, 1 intact read suffices | E11, area-additive | E6, sink artifact | E9, position binding |
|---|---|---|---|---|---|
| B1 | each of the 15 nonempty subsets of the 4 trigger tokens masked in all 12 blocks | survival near 1 while (13,13) or both strips stay visible, near 0 only when all 4 go (section A reads 0.002) | logit difference falls in proportion to masked area, masking (13,13) alone costs about 0.58 of it | same as E1 under the key mask, lower survival under the zero mask when many tokens are masked | same as E1 |
| B2 | the same subsets masked only in blocks 9 to 12, and only in 5 to 8 | late only reproduces B1 partly (section A reads 0.200 for all 4), 5 to 8 changes little | graded in both bands | no difference from E1 | no difference from E1 |
| B3 | as many random non-trigger tokens, the 5 tokens bordering the trigger or 4 tokens in the far corner, each in all 12 blocks | 1.000 for random and far, near 1 for the border unless the border relays the trigger | 1.000 for all | 1.000 | 1.000 |
| B4 | only the trigger and the class token visible, every other token masked in all blocks, under the zero mask and the key mask | target kept at 0.9 or more under the key mask | target kept only if the trigger area alone outweighs the lost content | zero mask far below key mask, since 192 $\beta$ tokens form 1 sink | target kept |
| B5 | trigger tokens masked independently per block with $p$ from 0.3 to 0.99 (X9) | $1 - p^{4m}$ with $m$ the number of sufficient tokens | a graded curve with no power-law shape | same as E1 | same as E1 |
| B6 | the whole trigger masked in all 12 blocks with probability 0.5 per pass, against each token independently per pass | 0.5 kept for the whole trigger, and for independent tokens $1 - 0.5^{m}$ | 0.5 for the whole trigger, graded for tokens | same as E1 | same as E1 |
| B7 | trigger moved to the other 3 corners and the center, plus 2 offsets of its own corner that cover 6 and 9 tokens (X10) | attack success near 1 everywhere, PSBD-TM survival rises with $m$ | near 1, logit difference rises with area | no difference | attack success collapses off the trained corner |
| B8 | trigger token content swapped with 4 other positions after the patch embedding, and position embeddings swapped alone (X10) | content swap keeps the target, embedding swap does not matter | same as E1 | no difference | embedding swap breaks the target |
| B9 | trigger pasted onto gray, black, noise and foreign images (X11) | target on 0.9 or more, PSBD-TM flags them | target only where the image contributes little counter-evidence | no difference | target on 0.9 or more |
| B10 | 1 to 9 of the 9 checkerboard pixels kept, and the whole patch at contrast 0.25, 0.5 and 0.75 | a threshold: success collapses once no token holds a legible piece | success and logit difference fall smoothly | no difference | no difference |
| B11 | the trigger tokens' residual stream at block $l$ patched from the triggered run into the clean run, $l$ from 1 to 12 (X12) | sufficient at every $l$ up to the class token's read | sufficient only with the full trigger area | no difference | no difference |

E4 (confidence) predicts that survival in every row tracks the triggered input's unperturbed confidence, so the B10 contrast ladder separates it best. E5 does not apply to triggered inputs. The TaCT trigger-conditional models run the same family with 1 distinct prediction. B9 should fail on them, since their trigger fires only on source-class content, and a TaCT trigger on a gray image should give the gray image's default class.

The single figure for the family is a 2-panel plot. On the left, survival and the logit difference against the masked trigger area for all B1 subsets, with the E1 step and the E11 line drawn. On the right, survival against $p$ for B5 with the $1 - p^{4m}$ curves.

## Designs for the global triggers

Blend, LF and BPP survive every token-mask placement as well as patches do, and section D shows why: they stay legible from 30% of tokens. What is not known is whether that legibility is per token (every token carries a full copy, E2) or extent-dependent (a token sees only a piece and several are needed, which I label E12). These designs separate the 2 and connect legibility to the score.

| row | manipulation | E2, per-token copy | E12, extent-dependent | E3, direction |
|---|---|---|---|---|
| G1 | visible fraction 0.3, 0.1 and 0.05 in 4 geometries: scatter, 1 window, horizontal stripes, vertical stripes (X17) | no dependence on geometry for Blend and BPP | LF lost from 1 window, SIG kept by stripes along its oscillation | no separate prediction |
| G2 | the trigger stamped on only a fraction of the image area (0.75 to 0.1), random tokens against 1 window, on otherwise clean images | attack success stays high down to small areas, geometry irrelevant | windows fail before scatter | score follows the projection on the backdoor direction |
| G3 | test-time strength ladders: Blend $\alpha$, LF strength, BPP bit depth (X18) | attack success high until the per-token copy fades | same | PSBD-TM score rises smoothly to the clean level |
| G4 | the trigger alone on a gray image, and the blend pattern alone | target kept (the pattern is the evidence) | target kept | the gray image carries the full direction |
| G5 | low-pass and high-pass filtering of the triggered image (Zeng et al.) | Blend's random pattern and BPP's quantization die under low-pass, LF survives | same | PSU tracks what survives |
| G6 | residual dropout by band, 1 to 4, 5 to 8 and 9 to 12, read from the caches (X4) | global evidence stored in every token, fragile only late | assembled early | no separate prediction |
| G7 | denoising the triggered stream into the clean run for a random fraction of positions at each block (X12 extended) | a small random fraction suffices at every block | sufficiency needs spatially spread positions | no separate prediction |

The figure for the global family is G1: excess retention against the visible fraction, 1 line per geometry, 1 panel per attack, with the clean retention curve in gray. It shows in 1 picture which triggers are copies and which are extents.

## Designs for WaNet, where token masking fails

WaNet is the most important failure because it is the only attack where PSBD-TM inverts on a panel model (0.459) while PSBD-RD reads 0.947, and because the literature gives a concrete mechanism to test. The WaNet trigger moves where pixels are sampled from by at most about half a native pixel, and noise mode trains the model to return the true class on images warped by the trigger field plus a random per-pixel field of the same scale. Our own records add 4 facts: the most displaced tokens carry the evidence at block 4 (patch recovery 0.880 against 0.359 for random tokens), the class token holds it by block 12, the substitution operator reads 0.592 where token masking reads 0.968 on the WaNet model of `why_psbd_works`, and WaNet does not survive crop and flip augmentation in our recipe.

| row | manipulation | E8, noise-mode rejection | E10, computed early | E2, redundant like Blend |
|---|---|---|---|---|
| W1 | destination of triggered shifts under PSBD-TM (X1, CPU) | mostly the true class on noise-mode models | no preferred destination | no shifts to speak of |
| W2 | the 2 unswept replicates of the failing cell (X2) | fail if the noise-mode recipe is the cause | fail if the recipe is the cause | pass |
| W3 | random sub-pixel jitter at native resolution, noise mode against none (X6) | triggered fragile with noise mode, stable without | fragile | stable |
| W4 | the transformation battery (X15) | translation, flip and jitter break the trigger, smooth random warps may not | blur breaks it | nothing breaks it |
| W5 | position embedding and local content permutations (X16) | embedding permutation breaks it if the check is relational | local permutation breaks it | nothing breaks it |
| W6 | contiguous against scattered visibility (X17) | no geometry at 0.1 keeps it | windows keep more than scatter | every geometry keeps it |
| W7 | test-time warp strength $s$ from 0.125 to 1, and on the trained strength 2 and 4 models (`_trig_s2`, `_trig_s4`) | success only near the trained $s$, both weaker and stronger fields fail | success grows with $s$ | success grows with $s$ |
| W8 | per-token linear probe for trigger-warped against noise-warped images at blocks 2 to 8, with a control-label probe (Hewitt and Liang) | per-token selectivity near 0, class token high | per-token selectivity rises in blocks 2 to 6 | per-token selectivity high everywhere |
| W9 | residual dropout confined to blocks 1 to 4 with its sign flipped, from the caches (X4) | not decisive | triggered breaks more than clean | no effect |
| W10 | the warp applied to only 1 window of the image, from 0.75 to 0.1 of the area | success collapses quickly with area | success falls with area | success stays high |

W7 carries the sharpest single prediction of E8. A detector trained to reject the trigger field plus small noise should also reject a scaled trigger field, so attack success should peak at the trained strength and fall on both sides. Every other account predicts a monotone rise with strength. The figure for the WaNet family is W3: triggered and clean retention against jitter amplitude, noise-mode and no-noise-mode models side by side.

## A WaNet-specific perturbation detector

The goal is a specialist probe that detects WaNet well without raising the false positive rate on clean inputs, so that it can join PSBD-TM in a union. The user set 3 criteria: standalone AUROC and TPR at 10% and 20% FPR on WaNet, a well calibrated clean-validation distribution so the union adds WaNet detections at a fixed budget without adding clean flags, and no loss on BadNets, TaCT, Blend, LF or BPP when combined.

## The key hypothesis and its answer

The question was whether noise-mode training predicts triggered WaNet inputs to be fragile to random warps. It does, and sharply, for 1 kind of random warp. Noise mode trains on $W(x, M + \delta)$ labeled with the true class $y$, where $M$ is the trigger field and $\delta$ is a per-pixel field drawn uniformly in $[-1/h, 1/h]$ in normalized coordinates (`attacks/wanet.py`, following the reference code and the paper's equation 6). A triggered test image $W(x, M)$ resampled once more at jittered positions is, to first order in the small displacements, $W(x, M + \delta)$, which is exactly a noise-mode training image with its true label. The model was trained to answer $y$ on it. The composition is not exact, because a second bilinear resampling smooths the image once more, and $M$ evaluated at a jittered point differs from $M$ at the grid point by a term of order $\delta \cdot \nabla M$, which is small because $M$ is smooth over about 8 native pixels. The trigger displaces pixels by about 0.24 native pixels on average and the noise by up to about 0.48, so the perturbation needed is the same size as the trigger itself.

3 qualifications follow from the same argument. The prediction is specific to i.i.d. per-pixel jitter at the noise amplitude. A smooth random warp is not in the noise-mode distribution, so the model's response to it is not predicted, and a blur is predicted to matter only through the interpolation texture it removes. The prediction reverses for models trained without noise mode, which the WaNet authors report learn "pixel-wise artifacts" of the warping. Jitter adds such artifacts, so on those models it should leave triggered inputs on the target and may push clean inputs toward it. And the argument explains PSBD-TM's inversion on the failing cell only if the masked triggered input also reads as "not the exact field", which X1 tests from the caches by checking whether the shifted triggered predictions land on the true class.

The records already show that noise mode alone does not predict PSBD-TM's failure (other noise-mode models read 0.936 to 0.965 in the `adaptive` field), so the hypothesis says how to detect WaNet and why the failure is possible, without saying why this 1 cell fails. X2 decides whether the failing cell is typical of its recipe.

## Candidate operators

Each candidate is scored on the user's 3 criteria. "Sign" says whether the triggered input shifts more (fragile) or less (stable) than a clean one. A fragile sign is the opposite of PSBD's and must be fixed before any triggered data is read.

| rank | operator | mechanism | sign on WaNet | BadNets and TaCT | Blend | LF and SIG | BPP | clean calibration | test and cost |
|---|---|---|---|---|---|---|---|---|---|
| 1 | late residual dropout, `pre_residual_blocks_9_12` or `_5_8` (existing) | the class token's copy of the warp evidence is 1 direction in a few late blocks, and dropout on the stream damages it | stable, PSBD's sign, 0.926 to 0.986 on the cached WaNet models | reads badly (PSBD-RD 0.309 on the BadNets dev model), which a budget-weighted union absorbs | good | good | good | as PSBD-RD, known | X3, CPU |
| 2 | sub-pixel jitter at native resolution | noise mode: a jittered triggered image is a noise-mode training image with its true label | fragile on noise-mode models, stable without noise mode | the 1-pixel checkerboard loses about half its contrast at an average offset of 0.25 pixel, so partly fragile | the per-pixel random pattern is averaged down, partly fragile | unaffected, uninformative | quantization levels are resampled off grid, likely fragile | good if the amplitude is set on clean data | X6, 8 min on 10 models |
| 3 | integer translation by 1 native pixel, deterministic | the learned feature is tied to exact pixel alignment (the `_aug` models never implanted) | fragile | the patch moves 7 pixels at 224 and stays in the same tokens, stable | stable | stable | a shift does not change quantization levels, stable | good, a 1-pixel shift barely moves clean CIFAR predictions (to be measured) | X15, under 1 min |
| 4 | residual dropout in blocks 1 to 4 with a flipped sign (existing cache) | E10: a computed trigger breaks while being assembled | fragile if E10 holds | stable, routed | stable | stable | unknown | as PSBD-RD bands | X4, CPU |
| 5 | TeCo's corruption deviation (existing port) | triggered inputs flip at very different severities across corruption types | TeCo's own sign, 0.901 on WaNet | 0.976 on BadNets, 0.825 on TaCT | 0.907 | 0.477 on LF | 0.562 | as ported | X3 extended to the detector records, CPU |
| 6 | Gaussian blur, $\sigma$ 0.5 native pixel | removes the interpolation texture and all high frequencies (Zeng et al., Lite-BD, ZIP) | fragile | the checkerboard at 0.29 of its contrast, likely fragile | fragile | stable | fragile | worse, blur costs clean accuracy at 32 pixels | X15 |
| 7 | smooth random warp, strength 0.25 | a smooth field is outside the noise-mode distribution, so it tests an exact-field reader | unknown, fragile only for an exact-field reader | stable | stable | stable | stable | good | X15 |
| 8 | position-embedding permutation within 2 by 2 blocks | breaks position-content coherence if WaNet is checked relationally | fragile if relational | fragile if E9 holds | stable | stable | stable | unknown, Naseer et al. suggest robust | X16 |
| 9 | bilinear down-up scaling by 0.5 | the strongest disruption in Lite-BD | fragile | stable | fragile | stable | fragile | poor, a 16-pixel CIFAR image loses class detail | X15 |

Candidate 1 is the existing baseline and already reads WaNet well. It fails criterion 3 under mean fusion, which is where the weighted rule below matters. Candidate 2 is the only one whose sign and amplitude follow from the attack's own training procedure, so it is the one to build first. Candidate 3 is the cheapest deterministic alternative. Candidates 2, 3 and 6 are high-frequency specialists as much as WaNet specialists, which is a feature for a union: they flag triggered BadNets, Blend and BPP inputs as extra fragile, which can only add detections once the sign is fixed.

## The specialist score and its amplitude

The specialist reuses PSBD's fractional statistic with the input operator in place of the internal one. It flags high values, the opposite of PSBD.

$$\psi(x) = 1 - \frac{1}{k}\sum_{j=1}^{k}\frac{P_c\big(T(x;\,a,\xi_j)\big)}{P_c(x)}, \qquad T(x;\,a,\xi) = \mathrm{grid\_sample}\Big(x,\ G_0 + \frac{a}{h}\,\xi\Big)$$

| symbol | meaning |
|---|---|
| $x$ | the input image at its native resolution, before the model's resize to 224 |
| $P_c(\cdot)$ | softmax probability of the unperturbed predicted class $c$ |
| $k$ | number of jittered passes, 3 as in the headline |
| $T(x; a, \xi)$ | the image resampled at jittered positions with bilinear interpolation, border padding, `align_corners=True` |
| $G_0$ | the identity sampling grid in normalized coordinates $[-1,1]^2$ |
| $\xi$ | a per-pixel random field, shape $(h, h, 2)$, uniform in $[-1,1]$ |
| $a$ | the amplitude, $a = 1$ is the noise-mode offset scale, about 0.48 native pixels at most |
| $h$ | native image width in pixels, 32 or 64 |
| $\psi(x)$ | the specialist score, high means fragile and flagged |

The amplitude is chosen on clean validation data only, as the largest amplitude on a fixed ladder at which clean predictions barely move. The ladder and the tolerance are fixed before any triggered image is scored.

$$a^{*} = \max\{\,a \in \mathcal{A} : \sigma_V(a) \le \sigma_{\max}\,\}, \qquad \mathcal{A} = \{0.25, 0.5, 1, 2\}, \quad \sigma_{\max} = 0.05$$

| symbol | meaning |
|---|---|
| $\sigma_V(a)$ | clean-validation shift ratio of the jitter at amplitude $a$, the share of passes whose label changes |
| $\sigma_{\max}$ | the largest tolerated clean shift, fixed at 0.05 before any triggered data is read |
| $\mathcal{A}$ | the amplitude ladder, fixed in advance |

This rule is the mirror of the adaptive rule. PSBD wants clean inputs to move (target 0.8), a fragile-sign specialist wants them still, so that any triggered input that moves stands out against a tight clean distribution. That is what makes criterion 2 achievable. It also keeps the defender from using knowledge of the attacker's noise amplitude, since $a = 1$ is chosen only if clean images tolerate it.

## Putting a fragile-sign specialist into the union

`experiments/probe_union/` combines probes by min-rank: each probe's score becomes its percentile in that probe's clean-validation distribution and the union takes the minimum. The threshold is the target quantile of the minimum on clean validation. That construction calibrates the union's FPR on validation whatever the members are. The cost of adding a member is dilution: the minimum of 2 roughly uniform percentiles is smaller than either, so the threshold tightens and PSBD-TM runs at a lower effective FPR. A member that is uninformative or inverted on an attack costs only that dilution, because its percentile for a triggered input is high and never becomes the minimum. Mean fusion is different: an inverted member pulls triggered scores toward the clean ones, which is why the mean rule and min-rank disagree in `experiments/probe_fusion/`.

For a specialist the right rule is a budget-weighted minimum, which is min-rank with unequal budgets and reduces to a per-probe quantile OR at a split FPR. The shares say how much of the FPR budget each member may spend.

$$u(x) = \min\left(\frac{\hat F_{\mathrm{TM}}\big(\phi_{\mathrm{TM}}(x)\big)}{w_{\mathrm{TM}}},\ \frac{1 - \hat F_{S}\big(\psi(x)\big)}{w_{S}}\right), \qquad \text{flag } x \text{ when } u(x) \le \alpha$$

| symbol | meaning |
|---|---|
| $\phi_{\mathrm{TM}}(x)$ | PSBD-TM's fractional PSU, low means poisoned |
| $\psi(x)$ | the specialist score, high means poisoned |
| $\hat F_{\mathrm{TM}}, \hat F_{S}$ | empirical CDFs of the 2 scores on the 2000 clean validation images |
| $w_{\mathrm{TM}}, w_{S}$ | budget shares, fixed in advance, summing to 1 |
| $\alpha$ | the target FPR of the union |
| $u(x)$ | the union score, low means poisoned. Its AUROC is the union's AUROC |

Thresholding $u$ at $\alpha$ flags $x$ when PSBD-TM's percentile is below $w_{\mathrm{TM}}\alpha$ or the specialist's upper percentile is below $w_S\alpha$, so by the union bound the validation FPR is at most $w_{\mathrm{TM}}\alpha + w_S\alpha = \alpha$, with equality when the 2 flag disjoint clean images. At $\alpha = 0.10$ and shares 0.8 and 0.2, PSBD-TM runs at 0.08 and the specialist at 0.02. A specialist that catches WaNet at all catches it in its extreme tail, so 0.02 is enough. PSBD-TM's loss from 0.10 to 0.08 is readable from the caches (X3) before the specialist exists. Equal shares give the existing min-rank rule.

A 2-sided specialist is the fallback when the sign cannot be fixed in advance, for example on models that may or may not use noise mode. The specialist percentile becomes $2\min(\hat F_S, 1 - \hat F_S)$, which spends half of $w_S\alpha$ on each tail and halves the specialist's power in the tail that matters. Because the jitter's sign is predicted to flip with noise mode, a 2-sided specialist is honest here. X6 measures what it costs.

The rules that keep this from cheating are these. The sign is fixed from the noise-mode argument before any triggered image is scored, or the test is 2-sided. The budget shares, the amplitude ladder, $\sigma_{\max}$ and $k$ are fixed in advance and written into `configs/psbd_basis.json` before the held-out read. The amplitude and both CDFs come from clean validation only. The realized FPR on the clean test split is reported beside the nominal one, because the Q27 audit shows the realized FPR of a quantile threshold varies widely across models. No quantity is ever fitted on triggered data, including "which tail".

## Evaluating on few WaNet models honestly

The panel holds 3 WaNet ViT models at the 2-point clean-accuracy bar (`vit_cifar10_wanet_0_1`, `vit_tiny_wanet_0_05`, `vit_tiny_wanet_0_1`), 2 more clear the attack-success bar and fail the accuracy bar (`vit_cifar10_wanet_0_05`, `vit_gtsrb_wanet_0_1`). 2 more fall below the attack-success bar (`vit_gtsrb_wanet_0_05` at 0.83, `vit_cifar100_wanet_0_1` at 0.793). On disk there are also 6 ViT replicates with noise mode (CIFAR-10 5% and 10%, GTSRB 10%, seeds 1 and 2), 6 ViT replicates without noise mode (CIFAR-100 10%, Tiny 5% and 10%, seeds 1 and 2), 8 seed-0 Swin models without noise mode and 6 Swin replicates with it, plus the strength variants `vit_cifar100_wanet_0_05_trig_s2` and `_s4` with noise mode.

The honest reading uses all of them and labels each by panel status, noise mode and architecture. It never averages across labels. It pre-registers 2 predictions. On noise-mode models the specialist reaches standalone AUROC of at least 0.9 on every model. On models without noise mode it does not, which is the falsification test of the mechanism: a specialist that also works without noise mode works for another reason. With fewer than 8 models in a group, the report counts models that pass and fail the pre-registered bar instead of printing a bootstrap interval. Development uses the CIFAR-10 and GTSRB models only. The Tiny and CIFAR-100 models, the Tiny replicates and all Swin models are read once, after the design is frozen. `vit_tiny_wanet_0_05` is in the 10-model development set, so either it is swapped for `vit_gtsrb_wanet_0_1` during exploration or it is reported apart from the held-out set.

## Novel perturbation designs

The coordinator reported on 2026-09-29 that averaging PSBD-TM's fractional PSU with that of late-block residual dropout, a single score equivalent to mixing the 2 operators across passes, gains +0.013 [+0.004, +0.028] over PSBD-TM on 39 models with no attack losing, lifts the failing WaNet cell from 0.459 to 0.745, and that the min-rank union gains +0.017. I did not read that record. It says combination works, so the value now lies in better building blocks. The designs below start from the user's 12 seeds, each critiqued, improved or discarded. 5 designs of mine follow them. The predictions per category use the categories of the earlier memo: patch (BadNets, TaCT), blend (Blend, LF), frequency (SIG), warp (WaNet) and quantization (BPP). Costs are on the 10 development models.

**N1. True key mask (seed 1, kept).** At the attention input of every block, a random share $p$ of patch tokens gets negative infinity logits as keys for every query, while their own queries, values for other purposes and residual entries stay untouched. It separates for the same reason PSBD-TM does, since hiding a token from every read in 1 block is the OR and AND asymmetry of principles (a) and (b), but it removes the sink: no constant $\beta$ tokens, no shared key and no replaced query for the masked token. Predictions: patch identical to PSBD-TM within 0.02 if E1 is the whole story, blend and quantization identical, frequency and warp may differ, because `token_substitute` already reads 0.592 against 0.968 on the WaNet model of `why_psbd_works`, which says something beyond hiding matters there. It combines as a replacement for PSBD-TM if it matches, and as a union partner if it disagrees on WaNet. The test is a 6-rate reduced ladder, 12 min, after adding an `attn_mask` to `masked_attention_forward`. The closest prior work is DropKey (Li et al., CVPR 2023), a training regularizer, with token dropping in Salman et al. as a second relative

**N2. Class-token-only read mask (seed 2, kept as a mechanism test, weak as a probe).** Mask keys only for the class-token query, so patch tokens keep attending to each other. As a mechanism test it is X8 and decides direct read against relay. As a probe it has a flaw: clean evidence is relayed among patch tokens by the middle blocks, so the class token can recover it from any visible token and clean inputs become robust too. I predict a clean shift ratio far below PSBD-TM's at equal $p$ and a rate near 0.9 to reach 0.8. Separation should appear only if the trigger is read directly. Restricting it to blocks 9 to 12 is the only version worth a ladder. The closest prior work is attention knockout (Geva et al.).

**N3. Attention temperature (seed 3, split in 2).** A random per-head rescaling mixes sharpening and flattening, which predict opposite signs for patch triggers, so the random version cancels itself. The class token puts 0.550 of its late attention on the 4 BadNets trigger tokens. Flattening ($\tau > 1$) dilutes that mass toward 4 over 197 and should break triggered patch predictions (fragile sign), while sharpening ($\tau < 1$) keeps the trigger in the top keys and strips clean evidence of its diffuse support (PSBD's sign). Predictions for sharpening: patch stable, blend and quantization stable since they are legible from few tokens, frequency and warp unknown, clean fragile. Both versions are deterministic, so k = 1 and a 4-temperature ladder costs about 2 min. The closest prior work is probe 3 of `docs/attack-design/improving-the-defense.md`, and I found no paper that uses it as a detector.

**N4. Attention-guided masking (seed 4, kept as a fragile-sign patch specialist).** Hide from the class token the $r$ tokens with the largest weight times value norm (Kobayashi et al.) in each late block. It flips the sign for patch triggers, as Subramanya et al.'s blocking defense and SentiNet's salient region imply, and should be uninformative on blend, frequency and quantization, whose evidence is not concentrated. Its value is on TaCT models with low TPR such as `vit_cifar10_tact_0_01`, where a fragile-sign member in a weighted union adds detections without touching PSBD-TM's. It is deterministic, 4 values of $r$ cost under 1 min. The closest prior work is Subramanya et al.

**N5. Random subspace removal (seed 5, discarded as a probe, kept as 1 control).** The analogy to token masking fails. Token masking separates because the trigger has several redundant reads (an OR) while clean evidence needs most tokens at once (an AND). In the 768-dimensional stream the backdoor is concentrated in 1 direction (top direction share 0.54 to 0.56 in `why_psbd_works`) and clean evidence is spread, so removing $k$ random directions removes about $k/768$ of the energy of each, with no OR and AND asymmetry. What remains is the margin difference, which is E4, already refuted as the carrier. I predict readings at or below channel masking (0.838 mean). 1 run at 1 rate is worth keeping as the control that separates dimension from structure, as the earlier design note already said.

**N6. Position-embedding jitter or local token permutation (seed 6, kept as a WaNet test, specialist only if X16 says so).** A warp is a relation between neighboring pixels, but it acts below the token scale here: the largest displacement is about 0.5 native pixel, about 3.5 pixels at 224, inside 1 token of 16. Permuting tokens or their position embeddings therefore cannot touch the warp's own texture. It can only break a check that ties what a token shows to where it sits, which is the relational form of E8. Predictions: patch stable unless E9 holds, blend, frequency and quantization stable (copies in every token), warp fragile only if relational. It is X16, 7 min. The closest prior work is PatchShuffle (Doan et al.) and Naseer et al.'s shuffle robustness.

**N7. Pixel shuffling inside each patch (seed 7, discarded in favor of N15).** A 16 by 16 token of an upsampled CIFAR image covers about 2.3 native pixels, so shuffling its 256 pixels turns a smooth patch into high-frequency noise that is off distribution for clean images too, and the clean distribution would be badly calibrated. The principled version of "destroy WaNet's fine interpolation artifacts" works at native resolution and at the size of the noise-mode perturbation, which is N15.

**N8. Random block skipping (seed 8, kept).** Skip a whole block per pass with probability $p$, both residual writes removed. A routed trigger survives skipping any single late block (masking the trigger in the last block alone keeps 0.934), and so does a per-token copy, while a computed trigger (E10) breaks when a block that assembles it is skipped. Predictions: patch stable, blend and quantization stable, warp fragile if skipping hits blocks 1 to 6, clean moderately fragile. It combines best as a band-restricted variant (skip only in blocks 1 to 6) with its sign chosen a priori from E10. `pre_residual_droppath` is cached on 4 of the 10 models, so a first reading is CPU. The rest costs 12 min. The closest prior work is the stochastic depth regularizer, which I did not re-verify for this memo.

**N9. Random shifted-window offsets for Swin (seed 9, kept at low priority).** Swin's windows decide which tokens exchange information in each block, and a random cyclic offset per pass changes the partition while the relative position bias stays tied to the trained layout. A patch trigger inside 1 window at stage 1 is sometimes split, but its evidence is merged in later stages anyway, so it should be stable. A warp read within windows could break. Clean inputs should be mildly perturbed, so separation is uncertain and likely weak. It is Swin only and costs about 15 min on 10 Swin twins of the dev models. I found no prior work.

**N10. Reference-bank token transplant (seed 10, kept, high value).** Replace a random share of tokens at the attention input with the same-position, same-block tokens of a fixed bank of 64 clean held-out images, 1 bank image per pass. This is resample ablation in Chan et al.'s sense, an in-distribution replacement for PSBD-TM's constant $\beta$, and each score depends only on the input and the fixed bank. Predictions: patch as PSBD-TM, blend and frequency as PSBD-TM, quantization as PSBD-TM since bank tokens are not quantized, warp possibly weaker since a foreign token breaks local coherence the way `token_substitute` does. It doubles as the E6 test (X5). Caching the bank costs seconds and the ladder 12 min. The closest prior work is resample ablation, our `token_substitute` (same image, other position) and STRIP's image-level superimposition.

**N11. Logit-lens consistency across depth (seed 11, kept as a free add-on).** Under the same PSBD-TM passes, apply the final LayerNorm and head to the class token after blocks 8 to 12 and score how consistently the late readouts keep the unperturbed label across passes and blocks. A routed trigger fixes the answer in blocks 9 to 12 and keeps it under masking in every late readout, while a clean answer fluctuates. It costs no extra forwards, only hooks. `experiments/prediction_depth/` already measures the unperturbed version. Predictions: patch and blend strong, warp unknown. The closest prior work is the logit and tuned lens on ViT (Vilas et al., Belrose et al.).

**N12. Sign-aware combination on clean data only (seed 12, kept as the fusion rule).** The sign of a member cannot be fitted on clean data, because clean data never shows which way a triggered input moves. What clean data can fit is each member's CDF, and hence a budget-weighted minimum of percentiles with the sign fixed a priori, or a 2-sided percentile when it cannot be (the WaNet section gives both). A clean-only 2-dimensional density on (PSBD-TM, late dropout) scores is the sign-free alternative: flag low density in any direction. It spends power on both tails and should trail the weighted minimum wherever the sign is known. All 3 are CPU on the caches (X3). The closest prior work is the min-rank union of H41 and Fisher's method.

**N13. Stratified token masking (mine).** Draw each block's mask so that every patch token stays visible in at least 1 of every $B$ consecutive blocks, while each token's marginal masking rate stays $p$. With $B = 4$ the 4 late reading blocks can never all hide a trigger token, so the event that breaks a triggered patch prediction under PSBD-TM (all $m$ trigger tokens masked in all 4 late blocks, probability $p^{4m}$, observed 0.059 on Tiny) becomes impossible, while every block still hides a share $p$ of clean evidence. For $p \le (B-1)/B$ such masks exist, for example by assigning each token a random offset and masking it in a fixed pattern of $\lceil pB \rceil$ consecutive blocks per window of $B$, with a random draw deciding the extra block when $pB$ is not an integer. Predictions: patch survival from 0.966 (4 tokens) and 0.930 (1 token) to near 1 with clean fragility unchanged, so TPR at 1% FPR rises most on Tiny BadNets and TaCT, blend, frequency, quantization and warp unchanged. It combines as a drop-in replacement for PSBD-TM. A new operator and 6 rates cost 12 min. The closest prior work is PatchCleanser's covering mask sets, which guarantee coverage of a patch, where this guarantees a patch survives.

**N14. Sink-only injection (mine).** Append $r = p \cdot 196$ copies of the masked token's key and value ($W_k\beta + b_k$, $W_v\beta + b_v$) to every block's attention, without masking anything. If PSBD-TM's clean fragility is partly the sink absorbing attention (E6), this alone reproduces part of the clean shift and some separation. Predictions under E6: clean shift grows with $r$ and patch and blend triggered inputs resist it, and under no E6: AUROC near 0.5 with small shifts. It is the cleanest single test of the artifact reading and costs 12 min with the same wrapper as N1. The closest prior work is registers (Darcet et al.) and test-time registers (Jiang et al.), which add tokens to absorb attention on purpose.

**N15. Sub-pixel jitter at native resolution (mine, the WaNet specialist).** The operator, sign, amplitude rule and fusion rule are in the WaNet section. Predictions: warp fragile on noise-mode models and stable without noise mode, quantization and blend partly fragile, patch partly fragile through the checkerboard's 1-pixel period, frequency stable. With the fragile sign fixed, every one of these can only add detections in a weighted union. 8 min on the 10 models. The closest prior work is WaNet's noise mode itself, Zeng et al.'s frequency analysis and TeCo.

**N16. Depth-contrast score (mine).** Divide the fractional PSU from token masking in blocks 1 to 8 by that from blocks 9 to 12 at the same rate. A patch-triggered input is immune to early hiding and exposed to late hiding, while a clean input is fragile to both, so the ratio is small for triggered patch inputs. The division also cancels each image's general fragility, which should tighten the clean distribution at the low end, where PSBD-TM's false positives are clean images that never flip (0.339 of its false positives in `why_psbd_works`). Predictions: patch improved, blend and quantization unchanged or noisy since both bands leave them stable, warp unknown. It is CPU on the band caches of all 10 models. I found no prior work.

**N17. Worst-pass and hard-label statistics (mine).** Replace the mean over passes by the minimum retained probability or the count of label-keeping passes. Under the OR structure a triggered patch input either survives a pass almost intact or not, so the worst pass is nearly as high as the mean for triggered inputs and much lower for clean ones. Predictions: patch TPR at 1% FPR up, others unchanged. It is CPU on `per_pass_probs` (X19). The closest prior work is PatchCleanser's agreement test and TeCo's hard-label severities.

The ranking by expected value over cost puts the free ones first. The top 3 building blocks for a combined design are N13, N10 or N1 (whichever X5 and X13 favor) and N15.

| rank | design | cost on 10 models | expected value |
|---|---|---|---|
| 1 | N16 depth contrast | CPU | a free calibration gain on patch triggers, decides the "early-immune" reading |
| 2 | N17 worst pass | CPU | a free low-FPR gain if the OR holds |
| 3 | N12 weighted fusion | CPU | the rule every specialist needs |
| 4 | N13 stratified masking | 12 min | mechanism-derived drop-in, largest predicted gain at low FPR on patch triggers |
| 5 | N15 jitter specialist | 8 min plus the WaNet set | the only principled WaNet specialist |
| 6 | N10 bank transplant | 12 min | removes the artifact question from every claim |
| 7 | N1 true key mask | 12 min | same, from the other side |
| 8 | N14 sink injection | 12 min | decides E6 directly |
| 9 | N8 block skipping | CPU first, then 12 min | decides routed against computed |
| 10 | N3 sharpening | 2 min | cheap deterministic member |
| 11 | N4 attention-rank masking | 1 min | fragile-sign patch specialist for low-TPR TaCT |
| 12 | N6 position permutation | 7 min | mechanism test for WaNet and E9 |
| 13 | N11 logit-lens consistency | hooks only | add-on statistic |
| 14 | N2 class-token read mask | 1 min as a test | mechanism test, weak probe |
| 15 | N9 Swin window offsets | 15 min on Swin | Swin only, uncertain |
| 16 | N7 pixel shuffle | not run | subsumed by N15 |
| 17 | N5 random subspace | 1 control run | the dimension control |

**The combined design.** The general probe is stratified token masking at the attention input (N13) mixed per pass with late residual dropout in blocks 9 to 12, scored by the fractional statistic or by the worst pass if X19 favors it. The masking half carries patch triggers with the OR guaranteed, the dropout half carries the stored late evidence that WaNet and Blend leave in the stream, and the mixture is the combination the coordinator's data point already supports. Its replacement rule (zero, bank or key mask) follows X5 and X13: keep zero if the 3 agree, since it is the measured one. Switch to the bank transplant if the artifact turns out to matter. The specialist is the native-resolution jitter (N15) with its fragile sign fixed from the noise-mode argument, in a budget-weighted minimum with the general probe at shares 0.9 and 0.1, which leaves the general probe at 0.09 FPR out of 0.10. The predicted outcome is a mean AUROC at least as high as the mixed score's on every attack, TPR at 1% FPR higher on patch triggers than PSBD-TM's (from N13), and the WaNet failure lifted above 0.9 through the specialist if E8 holds.

## Toward an optimal PSBD-like probe for ViT, then Swin

The measured principles, stated as design rules with the evidence behind each, are these. Each rule names the measurement it rests on.

1. **Hide reads, independently per block.** Clean evidence needs about 70% of tokens visible in every block (an AND), and a patch trigger needs 1 intact read in about 4 late blocks (an OR). Hiding tokens from attention exploits exactly that, and independence across blocks is what raises the trigger's survival from $1 - p^{m}$ (masking once, as at the embedding, which reads 0.787 on BadNets) to $1 - p^{4m}$. Stratified masks (N13) are the limit of this rule.
2. **Do not touch storage for routed triggers, do touch it for computed ones.** The residual stream keeps a masked token's content, which is why hiding at the attention input spares the patch trigger. WaNet's evidence is assembled from the most displaced tokens by block 4 and held in the late stream, and late residual dropout reads 0.93 to 0.99 on every cached WaNet model. A probe that covers both needs a read perturbation and a storage perturbation, mixed or in a union.
3. **Prefer removal to additive noise before a LayerNorm.** LayerNorm removes 12 to 19% of additive noise and almost none of a mask (`experiments/residual_stream_mechanism/`), and Gaussian noise trails token masking by 0.183 at the attention input.
4. **Know what the replacement injects.** A zeroed token becomes $\beta$ after the norm and every masked token shares 1 key, so the group acts as 1 sink whose mass grows with the number masked. Whether that helps or hurts is open (X5, X13, N1, N10, N14).
5. **Choose every rate on clean data.** The adaptive rule does this for PSBD's sign, and the clean-stability rule of the WaNet section does it for a fragile-sign specialist.

The candidates the user listed, plus those the literature suggests, map onto these rules as follows. Costs are on the 10 models.

| candidate | rule it tests | mechanism | patch | blend | frequency | warp | quantization | cost |
|---|---|---|---|---|---|---|---|---|
| true key mask (N1) | 4 | hidden without a sink | as PSBD-TM | as PSBD-TM | unknown | may weaken | as PSBD-TM | 12 min |
| mixed operator, token mask plus late residual dropout per pass | 2 | read and storage perturbed together | as PSBD-TM or slightly lower | better | better (PSBD-RD reads SIG 0.919) | much better | similar | CPU for the averaged score, 12 min as 1 operator |
| depth-dependent schedule, front-loaded (early rate above late) | 1 | early hiding breaks clean aggregation and spares routed triggers | better | similar | similar | worse, early hiding disturbs assembly | similar | 12 min per schedule |
| depth-dependent schedule, back-loaded | 1 | late hiding hits the reads | worse | similar | similar | better | similar | 12 min |
| per-head masking (`attention_heads_head_mask`, cached on 4 of 10) | 1 | a head is 1 read channel, and the trigger uses a few late heads | weaker than token masking, 12 heads give a coarser OR | similar to channel masking | unknown | unknown | similar | CPU for 4 models, 12 min for the rest |
| attention-rank masking (N4) | none, a specialist | targeted removal of the read | sign flips | uninformative | uninformative | unknown | uninformative | 1 min |
| attention dropout on post-softmax weights, per query and key edge | 1 | a per-query key mask with renormalization | as N1 for the class-token read | similar | unknown | unknown | similar | 12 min |
| ToMe merging as a perturbation (Bolya et al.) | 1 | merges redundant tokens first, so a dissimilar trigger stays and clean redundancy goes with little loss | stable | stable | stable | unknown | stable | 12 min, weak separation expected |
| test-time register or sink injection (N14, Jiang et al.) | 4 | isolates the sink | as clean if E6 | as clean if E6 | unknown | unknown | unknown | 12 min |
| MC dropout on the model's own dropouts (`activate_model_dropout`) | none | conflates regularizer and probe, studied on purpose in the repository | weak | weak | weak | weak | weak | not proposed |
| test-time augmentation consistency (jitter, translation, flip) | 5 | input-level fragility | partly fragile | partly fragile | stable | fragile with noise mode | fragile | 8 min |

DynamicViT needs a trained token predictor, so it does not fit an inference-only plan. Swin needs 2 adjustments. It has no class token, so every read mask becomes a mask on the mean-pooled readout's inputs, and its early windows restrict who reads whom, so N9 replaces the knockout designs. `experiments/swin_mechanism/` already shows Swin's advantage on global triggers under token masking, and the Swin WaNet replicates with noise mode read 0.958 to 0.982, so the Swin design question is mainly whether stratification and the specialist transfer.

**An honest design protocol.** The protocol has to avoid what Q20 and Q21 of `docs/open-questions.md` document: a selection half that picks a different winner than the pooled panel, and a winner judged on the models it was chosen on. The 5 steps below keep choosing and judging on separate models.

1. Freeze the candidate list, the selection metric and the tie rule before reading any new number, and write them into `configs/psbd_basis.json` beside the existing declaration. The metric is the mean paired AUROC gain over PSBD-TM plus the paired gain in TPR at 1% FPR. A candidate replaces PSBD-TM only when both intervals exclude 0 on the development half, and ties go to the cheaper and simpler design.
2. Explore on the 10 fixed models, with the 3 held-out-dataset members swapped for CIFAR-10 or GTSRB models of the same attack (for `vit_tiny_wanet_0_05`, `vit_gtsrb_wanet_0_1`, which clears the attack-success bar and fails the accuracy bar, labeled as such). Treat every reading there as exploratory.
3. Confirm the 2 or 3 surviving candidates on the whole development half, the CIFAR-10 and GTSRB panel models (31 models in the Q20 record), and pick 1 by the frozen metric.
4. Evaluate that 1 design once on the held-out half (CIFAR-100 and Tiny ImageNet, 26 panel models) as the single confirmatory comparison against PSBD-TM, with a bootstrap interval, TPR at 1%, 10% and 20% FPR and the realized FPR. Report every frozen candidate's held-out numbers beside it, so the selection optimism is visible.
5. Evaluate once more on Swin-S over all 4 datasets, which the design never saw, as a second held-out set.

Some choices may use triggered data and some must not. The table lists each choice with where it may be made.

| choice | may use triggered data | where |
|---|---|---|
| which pre-registered candidate to keep | yes | development models only |
| structural choices: position, operator family, schedule shape, mixing proportion, budget shares | yes | development models only, frozen before step 4 |
| the rate or amplitude per model | no | clean validation, by the adaptive or the clean-stability rule |
| thresholds and percentile CDFs | no | clean validation |
| the sign of a specialist | no. It is fixed from a stated mechanism or the test is 2-sided | before any triggered image is scored |
| any per-attack choice | never, the defender does not know the attack | not applicable |
| anything on CIFAR-100, Tiny ImageNet or Swin before the freeze | never | not applicable |

2 checks need no triggered data at all and should gate every candidate before step 3. The first stamps each trigger on inputs of the benign models, where every detector must read about 0.5 (`why_psbd_works` has benign BadNets and Blend probes). The second checks that the candidate's clean shift ratio reaches its target on every development model and that its realized clean-test FPR stays within 0.02 of nominal. Proxy attacks absent from the panel (T6) would let step 3 select on attacks the held-out evaluation does not contain, which is the strongest protection and needs training.

## Reverse-engineering the mechanism efficiently and correctly

Current practice converges on a few rules. Exploratory localization should be cheap and wide and confirmation exact and narrow (Heimersheim and Nanda). Corruptions should stay in distribution, with a symmetric counterfactual preferred to Gaussian noise, and the metric should be a logit difference rather than a probability (Zhang and Nanda). The direction of a patch decides what it finds, since denoising finds sufficient components and misses an AND while noising finds necessary ones and misses an OR (Heimersheim and Nanda). A hypothesis should be tested by resampling everything it calls irrelevant (Chan et al.) and judged by faithfulness, completeness and minimality (Wang et al.). And every ablation injects a distribution shift (Li and Janson), later components compensate for ablated ones (McGrath et al., Rushing and Nanda) and subspace patches can create what they claim to find (Makelov et al.). Automated circuit discovery (Conmy et al.) is too costly at the level of 197 positions by 12 blocks by 12 heads with 1 edge per pair, and attribution patching recovers circuits at least as well (Syed et al.), so the workflow uses attribution patching at position-group and head granularity, with ACDC reserved for a pooled graph of 3 position groups (trigger, class token, rest) by 12 heads by 12 blocks if a circuit claim is ever needed.

The question the workflow answers is why a backdoored ViT's triggered prediction survives PSBD-TM and a clean one does not. The patch models of the development set (`vit_cifar10_badnet_a2o_0_01`, `vit_gtsrb_tact_0_05`, `vit_cifar10_tact_0_01`) carry steps 1 to 7, the WaNet pair (`vit_cifar10_wanet_0_1`, `vit_tiny_wanet_0_05`) repeats them as the contrast. The global models (`vit_cifar10_blend_0_1`, `vit_tiny_lf_0_01`, the 3 BPP models) repeat steps 2, 5 and 8.

1. **Behavior and metric.** (a) What exactly survives? (b) The metric is the logit difference, target minus true class, per clean and triggered pair, with the clean twin as the symmetric counterfactual (the image-level analog of symmetric token replacement), read unperturbed and under PSBD-TM passes. (c) The control is the benign model with the same trigger stamped, where the difference should stay near 0. (d) It is done when the metric's unperturbed values agree with the cached baselines (`cache_agreement` in `why_psbd_works` already checks this) on all 10 models. Code: the pairing of `experiments/why_token_masking_works/measure.py` and `defenses.decision.pair_clean_to_backdoor`.
2. **Localization.** (a) Which (block, position group, head) components carry the triggered difference? (b) First, attribution patching: 1 forward and 1 backward pass per pair giving gradient times activation difference for every component. Then exact patching of the top 20 in both directions, denoising triggered into clean (sufficiency) and noising clean into triggered (necessity), since the late reads are predicted to be an OR that noising alone cannot find. (c) The controls are a random position group of the same size and the group bordering the trigger. (d) It is done when the exact effects of the top components rank-correlate with the attribution estimates at 0.7 or more, and both directions are reported. Code: recording and patching modules attached through `models.positions.plug_dropout` hooks at `before_attention_norm` and the residual positions, and `analysis.features.captured_layers` for the stream.
3. **Route.** (a) Does the class token read the trigger directly, and through which heads? (b) Per-head direct logit attribution in blocks 9 to 12 first, then attention knockout of the class-token edges and the patch edges by band (X8). (c) A random 4-key knockout in the same band. (d) It is done when knockout and patching agree on the blocks and heads. Code: `defenses.operators.masked_attention_forward` with an attention mask.
4. **Representation.** (a) What does the survived state look like? (b) The backdoor direction of `analysis/direction.py`, fitted on half the pairs, then the projection on the held-out half, a logit lens per block and per-token probes. (c) The direction must transfer to held-out pairs and to a second model of the same attack before any patching along it is read causally (Makelov et al., Bolukbasi et al.). Every probe reports selectivity against a control-label probe (Hewitt and Liang). (d) It is done when the held-out projection separates triggered from clean at AUROC 0.9 or more and probe selectivity is reported. `why_psbd_works` has the direction measurements (L8) this step extends.
5. **Ablation type.** (a) Are the token-mask results about information or about the artifact? (b) The resample and mean versions (X5) and the key mask, run on the rows that carry the paper's claims. (c) The zero version is itself the treatment, and agreement across the 4 rules is the control. (d) It is done when each claim is marked as holding under resample ablation or as depending on the constant replacement.
6. **Self-repair.** (a) Do later components compensate for a knocked-out read? (b) Knock out the class token's trigger read in block 12 and measure the change in blocks 9 to 11's direct contributions (X21). (c) The same knockout on a clean input. (d) It is done when total and direct effects are both reported for every knockout in steps 2 and 3.
7. **Causal scrubbing of the hypothesis.** (a) Does the stated mechanism account for the behavior? (b) The hypothesis "the triggered answer depends only on the trigger tokens' residual entries and the class token's reads of them in blocks 9 to 12" is tested by resampling every other token's attention input and every other read from clean images of other classes, and measuring the share of the logit difference kept. (c) The same resampling applied to the trigger tokens instead must remove the difference. (d) It is done when the hypothesis' components keep 0.9 or more of the difference (faithfulness), removing them from the full model drops it to the clean level (completeness), and dropping any 1 block from the hypothesis lowers the kept share (minimality).
8. **Link to the detector.** (a) Does the mechanism predict PSBD-TM's per-sample outcome? (b) Predict each pass's survival from the mechanistic variable of step 7, the number of late blocks in which some trigger token was visible (section B already records it) for patch triggers and the projection on the direction for global ones. (c) The same variable computed for a random token set must not predict survival. (d) It is done when the variable predicts per-pass survival at AUROC 0.9 or more.
9. **Generalization.** (a) Does it hold on models the workflow was not developed on? (b) Rerun steps 2, 3, 5 and 8 once on the held-out CIFAR-100 and Tiny models and on Swin, after the analysis choices are frozen. (c) The benign controls again. (d) It is done when each step's verdict is stated per architecture.

The zero-input token mask is a constant ablation that creates an off-distribution artifact, and this is the specific issue step 5 addresses. `TokenMask` zeroes the input of `ln_1`, and with the LayerNorm's epsilon the output for every masked token is exactly $\beta$, whatever the image, the position or the block (`docs/why-psbd-works-theory.md`). No natural token has zero variance across channels, so no real input ever produces $\beta$ at the attention input. Every masked token then carries the same key $W_k\beta + b_k$ and value $W_v\beta + b_v$, so $r$ masked tokens act as 1 token repeated $r$ times whose attention mass grows with $r$, and the masked token's own query is replaced by a constant, so its residual update is a generic write. The residual entry itself is untouched, which is why this is an ablation of the attention read and not of the token. In Li and Janson's terms it is neither zero, mean nor resample ablation of the read. It carries exactly the distribution shift Zhang and Nanda warn about. The resample version replaces each masked token's `ln_1` input with the same block's `ln_1` input at the same position from a bank image (N10, X5), the mean version uses the bank average at that position and block, and the deletion version removes the token from the key set (N1). The experiments to rerun in those 3 versions are sections A, B and D of `why_token_masking_works`, the token-mask cells of `why_psbd_works` for L22 and the headline PSBD-TM reading on the 10 models. If all 3 agree with the zero version, the paper's mechanism statements are about information. If they differ, the difference is the artifact's share. The paper should then say which of its claims depend on it.

## Plausible explanations that may not hold

**Noise mode makes WaNet robust to perturbation.** It sounds right because noise mode was introduced to make WaNet stealthy against defenses. The training objective predicts the opposite for jittered triggered inputs: a trigger field plus random per-pixel noise was labeled with the true class, so jitter turns the trigger off. X6 decides it, and the prediction is triggered retention at most 0.2 on noise-mode models.

**Swin reads WaNet better only because its seed-0 WaNet models lack noise mode.** The confound is real in `experiments/swin_mechanism/`'s matched pairs. It does not carry the result for PSBD-TM, since the Swin replicates with noise mode read 0.958 to 0.982 in the `adaptive` field against 0.912 to 0.995 without it. The Swin residual-dropout readings vary much more (0.241 to 0.952) and deserve their own look.

**Token masking works because masked tokens are removed.** The masked tokens are replaced by a shared constant, which is an attention sink by construction. X5, X13 and N14 measure what share of the effect is removal.

**Masking the trigger in the last block should break BadNets, because that is where the class token reads it.** Section A reads 0.934 kept for the last block alone. The OR over 4 late blocks explains most of it, and self-repair (X21) may explain the rest.

**WaNet fails under PSBD-TM because it is global like Blend.** Section D says WaNet is not redundant (0.183 of excess retention at 30% visible against 0.90 to 0.95 for Blend, LF and BPP). The failure needs a different account, and E8 is the leading one.

**Random subspace removal is the feature-space version of token masking.** The OR and AND asymmetry that makes token masking work has no counterpart among random directions, as argued under N5. What is left is the margin difference, which confidence already fails to explain (P1).

**ViT's near permutation invariance makes the backdoor position-free.** Naseer et al. measured pretrained classifiers, and our fine-tuned models need 70% of tokens where theirs keep 60% accuracy at 20% visible. E9 is open until X10 runs.

## References

Abnar and Zuidema, Quantifying Attention Flow in Transformers, arXiv 2005.00928.
Belrose et al., Eliciting Latent Predictions from Transformers with the Tuned Lens, arXiv 2303.08112.
Bolukbasi et al., An Interpretability Illusion for BERT, arXiv 2104.07143.
Bolya et al., Token Merging: Your ViT But Faster, ICLR 2023, arXiv 2210.09461.
Borgnia et al., Strong Data Augmentation Sanitizes Poisoning and Backdoor Attacks Without an Accuracy Tradeoff, arXiv 2011.09527.
Carter et al., What Made You Do This? Understanding Black-Box Decisions with Sufficient Input Subsets, AISTATS 2019, arXiv 1810.03805.
Carter et al., Overinterpretation Reveals Image Classification Model Pathologies, NeurIPS 2021, arXiv 2003.08907.
Chan et al., Causal Scrubbing: a method for rigorously testing interpretability hypotheses, AI Alignment Forum, December 2022.
Chen et al., Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering, arXiv 1811.03728.
Chou et al., SentiNet: Detecting Localized Universal Attacks Against Deep Learning Systems, DLS, arXiv 1812.00292.
Cohen et al., Certified Adversarial Robustness via Randomized Smoothing, ICML 2019, arXiv 1902.02918.
Conmy et al., Towards Automated Circuit Discovery for Mechanistic Interpretability, NeurIPS 2023, arXiv 2304.14997.
Darcet et al., Vision Transformers Need Registers, arXiv 2309.16588.
Doan et al., Defending Backdoor Attacks on Vision Transformer via Patch Processing, AAAI 2023, arXiv 2206.12381.
Gal and Ghahramani, Dropout as a Bayesian Approximation, ICML 2016, arXiv 1506.02142.
Gandelsman et al., Interpreting CLIP's Image Representation via Text-Based Decomposition, ICLR 2024, arXiv 2310.05916.
Gao et al., STRIP: A Defence Against Trojan Attacks on Deep Neural Networks, ACSAC 2019, arXiv 1902.06531.
Geva et al., Dissecting Recall of Factual Associations in Auto-Regressive Language Models, EMNLP 2023, arXiv 2304.14767.
Guo et al., SCALE-UP: An Efficient Black-box Input-level Backdoor Detection via Analyzing Scaled Prediction Consistency, ICLR 2023, arXiv 2302.03251.
Hase et al., Does Localization Inform Editing?, NeurIPS 2023, arXiv 2301.04213.
Heimersheim and Nanda, How to use and interpret activation patching, arXiv 2404.15255.
Hewitt and Liang, Designing and Interpreting Probes with Control Tasks, EMNLP 2019, arXiv 1909.03368.
Hooker et al., A Benchmark for Interpretability Methods in Deep Neural Networks, NeurIPS 2019, arXiv 1806.10758.
Jain et al., Missingness Bias in Model Debugging, ICLR 2022, arXiv 2204.08945.
Jiang et al., Vision Transformers Don't Need Trained Registers, NeurIPS 2025, arXiv 2506.08010.
Joseph et al., Prisma: An Open Source Toolkit for Mechanistic Interpretability in Vision and Video, arXiv 2504.19475.
Karayalçin et al., Backdoor Directions in Vision Transformers, arXiv 2603.10806.
Kobayashi et al., Attention is Not Only a Weight: Analyzing Transformers with Vector Norms, EMNLP 2020, arXiv 2004.10102.
Kramár et al., AtP*, arXiv 2403.00745.
Lapuschkin et al., Unmasking Clever Hans Predictors and Assessing What Machines Really Learn, Nature Communications 2019, arXiv 1902.10178.
Li and Janson, Optimal ablation for interpretability, arXiv 2409.09951.
Li et al., DropKey, CVPR 2023, arXiv 2208.02646.
Li et al., Rethinking the Trigger of Backdoor Attack, arXiv 2004.04692.
Li et al., Anti-Backdoor Learning: Training Clean Models on Poisoned Data, NeurIPS 2021, arXiv 2110.11571.
Li et al., PSBD: Prediction Shift Uncertainty Unlocks Backdoor Detection, arXiv 2406.05826.
Liu et al., Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks, arXiv 1805.12185.
Liu et al., Detecting Backdoors During the Inference Stage Based on Corruption Robustness Consistency, CVPR 2023, arXiv 2303.18191.
Makelov et al., Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching, arXiv 2311.17030.
McGrath et al., The Hydra Effect: Emergent Self-repair in Language Model Computations, arXiv 2307.15771.
Meng et al., Locating and Editing Factual Associations in GPT, NeurIPS 2022, arXiv 2202.05262.
Miah and Bi, Lite-BD: A Lightweight Black-box Backdoor Defense via Reviving Multi-Stage Image Transformations, IJCNN 2026, arXiv 2602.07197.
Nanda, Attribution Patching: Activation Patching At Industrial Scale, blog post, 2023.
Naseer et al., Intriguing Properties of Vision Transformers, NeurIPS 2021, arXiv 2105.10497.
Nguyen and Tran, WaNet: Imperceptible Warping-based Backdoor Attack, ICLR 2021, arXiv 2102.10369.
Rao et al., DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification, NeurIPS 2021, arXiv 2106.02034.
Rushing and Nanda, Explorations of Self-Repair in Language Models, ICML 2024, arXiv 2402.15390.
Salman et al., Certified Patch Robustness via Smoothed Vision Transformers, CVPR 2022, arXiv 2110.07719.
Shi et al., Black-box Backdoor Defense via Zero-shot Image Purification, NeurIPS 2023, arXiv 2303.12175.
Subramanya et al., Backdoor Attacks on Vision Transformers, arXiv 2206.08477, published as A Closer Look at Robustness of Vision Transformers to Backdoor Attacks, WACV 2024.
Sun et al., Massive Activations in Large Language Models, COLM 2024, arXiv 2402.17762.
Sun et al., Mask and Restore: Blind Backdoor Defense at Test Time with Masked Autoencoder, arXiv 2303.15564.
Syed et al., Attribution Patching Outperforms Automated Circuit Discovery, NeurIPS 2023 ATTRIB workshop, arXiv 2310.10348.
Tran et al., Spectral Signatures in Backdoor Attacks, NeurIPS 2018, arXiv 1811.00636.
Vilas et al., Analyzing Vision Transformers for Image Classification in Class Embedding Space, NeurIPS 2023, arXiv 2310.18969.
Wang et al., Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small, arXiv 2211.00593.
Xiang et al., PatchCleanser: Certifiably Robust Defense against Adversarial Patches for Any Image Classifier, USENIX Security 2022, arXiv 2108.09135.
Xiao et al., Efficient Streaming Language Models with Attention Sinks, ICLR 2024, arXiv 2309.17453.
Yang et al., RAP: Robustness-Aware Perturbations for Defending against Backdoor Attacks on NLP Models, EMNLP 2021, arXiv 2110.07831.
Yang et al., SampDetox: Black-box Backdoor Defense via Perturbation-based Sample Detoxification, NeurIPS 2024.
Zeng et al., Rethinking the Backdoor Attacks' Triggers: A Frequency Perspective, ICCV 2021, arXiv 2104.03413.
Zhang and Nanda, Towards Best Practices of Activation Patching in Language Models: Metrics and Methods, ICLR 2024, arXiv 2309.16042.

## Ideas from ShortcutProbe, checked against our setup

Prasad et al. (ISDFS 2026, `literature/shortcutprobe-prasad-isdfs2026/`) combine PSBD's absolute PSU under a dropout of positive activations with a deterministic score, the largest single confidence drop while channels are removed in order of gradient times activation, and add the 2 with equal weights. It is ResNet-18 on CIFAR-10 at 10% poisoning only. Each of its 6 components was checked for whether it carries over to our ViT setup.

| component | carries over | reason | action |
|---|---|---|---|
| dropout of positive activations only ("Active Neuron Dropout") | partly | "positive" means active only after a nonlinearity, so the only ViT site where it is defined is the MLP hidden layer after GELU. Plain channel masking there reads 0.833 on the 54 panel models (TaCT 0.522, BadNets 0.775) against PSBD-TM's 0.963, and the backdoor is a residual-stream direction, not a set of neurons | test once on the 10-model development set as operator S1, predicted at or below plain MLP channel masking |
| absolute PSU at 20 passes | no | fractional PSU beats absolute PSU on 37 of 57 models under PSBD-TM and 53 of 57 under PSBD-RD, and 3 to 20 passes gains +0.002 | none |
| rate that maximizes the clean shift ratio | no | the maximum is reached near total destruction of the clean prediction, where triggered predictions fall too. The adaptive rule at 0.8 is the smallest rate that reaches a fixed shift | none |
| largest single confidence drop under importance-ranked removal | yes, on tokens | a patch trigger lives in a few tokens, so removing tokens in importance order should give 1 sudden drop when the trigger's tokens go, while clean and global-trigger predictions decay gradually | test as operator S2 on tokens (gradient times activation at the block 9 input, deterministic masking of the top 1 to 16 tokens at the attention input). Predicted strong on BadNets and TaCT, where PSBD-TM is already at 0.96 to 0.99, and weak on Blend, BPP and WaNet. Its value is as a union partner that raises TaCT's low TPR, not as a replacement |
| equal-weight sum of the 2 scores | no | the sum is the mixture of `docs/novel-designs-theory.md`, which the union beats whenever 1 member is weak or inverted on an attack, as S2 is predicted to be on global triggers | if S2 is kept, it enters through the budget-weighted min-rank union |
| threshold at the maximum of Youden's J | no | J needs the true positive rate and so reads poisoned samples. Our threshold is a quantile of clean validation scores | none |

S1 costs about 4 minutes per model (1 sweep). S2 needs 1 backward pass and up to 16 masked forward passes per input, so about 20 minutes per model on the A100. Both run on the 10-model development set after the analysis experiments of the night.
