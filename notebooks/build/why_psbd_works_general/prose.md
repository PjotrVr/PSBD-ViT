<!-- title -->
# Why PSBD's statistic works on ViT and Swin, for any perturbation

PSBD (Li et al., arXiv 2406.05826) flags an input as poisoned when the probability of the model's own answer barely drops under $k$ perturbed forward passes. On our ViT-B/16 models that statistic separates triggered from clean images with every perturbation family the project tried. At the attention input these are token masking, dropout and channel masking. On the residual stream it is dropout. The notebooks `mechanism.ipynb` and `why-psbd-tm.ipynb` and the experiment `experiments/why_token_masking_works/` explain why 1 placement beats another on patch triggers. This notebook asks the question underneath them: what property of a triggered input makes its answer survive a perturbation that destroys a clean answer, for any perturbation. A second question is whether the answer is the same for every kind of trigger.

It is written for a reader who knows what a backdoor, a Vision Transformer and an AUROC are and has not read the rest of this repository. Every step states the question it answers, why that question comes next and what is measured by which function. It then shows a figure, says what the figure proves and what it does not and hands over to the next question. Every number is read from JSON files the experiment wrote. Every sentence that carries a number is rendered by code from those files, so nothing here runs a model and nothing can go stale. The measurements, their formulas and the commands that produced them are in `experiments/why_psbd_works/README.md`, which is generated the same way. The predictions each account is judged against come from 2 companion documents: `docs/why-psbd-works-literature.md` names every published and folk explanation with an identifier (L for published ones, P for folk ones). `docs/why-psbd-works-theory.md` turns each into a number with a supported and a refuted threshold. The identifiers are kept here so the 3 documents cross-reference.

<!-- map -->
## The map

The statistic, with its symbols:

$$\phi(x) = 1 - \frac{\tfrac{1}{k}\sum_{j=1}^{k} P_c(x;\,\xi_j)}{P_c(x)}, \qquad c = \arg\max_i P_i(x)$$

| symbol | meaning |
|---|---|
| $x$ | the input image |
| $P_i(x)$ | the softmax probability of class $i$ with no perturbation |
| $c$ | the unperturbed model's answer |
| $\xi_j$ | the random draw of the perturbation on pass $j$ (a dropout mask, a token mask, a noise sample) |
| $P_c(x;\xi_j)$ | the probability of class $c$ on perturbed pass $j$ |
| $k$ | the number of passes |
| $\phi(x)$ | the fractional prediction shift uncertainty, low means flagged as poisoned |

This is the fractional form the project uses (`defenses.scores.psu_ratio_from_cache`). Li et al.'s paper subtracts the probabilities without dividing, which the project calls absolute PSU. The fractional form divides out how confident the model was to begin with, which is why the first explanation anyone reaches for, "triggered inputs are just more confident", already has to work harder against it.

The route:

1. **Sanity.** The passes this experiment makes are held to the pipeline's own numbers before any mechanism is read.
2. **The phenomenon.** Does the statistic separate on these models, with every operator, for every attack category? Everything later explains this figure.
3. **H-margin.** Is it a margin effect: does the triggered answer keep its logit margin while the clean one collapses, and does that survive comparing images of equal starting margin?
4. **H-direction.** What in the representation keeps the margin: is the triggered feature dominated by 1 large component, the backdoor direction, that the perturbation barely dents?
5. **H-redundancy.** Where that component comes from in the token sequence: is the trigger's evidence repeated across tokens, so that removing tokens leaves it?
6. **H-low-dim.** How concentrated the triggered decision is: few dimensions, few tokens, a small sufficient part of the input.
7. **H-flatness.** Is it curvature: do triggered inputs sit in a flatter region, and does the statistic behave like the curvature estimate a second-order expansion says it is?
8. **Explanations that do not hold.** Confidence, MC-dropout uncertainty, the trigger being out of distribution, robust trigger neurons, Li et al.'s neuron-bias account, the target class being easy, and the literature memo's refuted items, each reproduced from its source file.
9. **Clean fragility.** The half of the statistic that is about clean images.
10. **The critical rate.** 1 account for every operator and attack, tested against fixed predictions.
11. **Swin-S.** The same measurements on the second architecture.
12. **The unifying hypothesis.** PSBD detects over-determined decisions, tested by trigger sufficiency against pre-registered predictions.
13. **Verdicts.** 1 grid, hypothesis by attack category, supported, partial or refuted, with the key number.

The attack categories follow the literature memo. Patch is BadNets (all to one) and TaCT restricted to the models that classify their clean source class correctly (the others map the whole source class to the target with no trigger, `docs/runs/2026-09-24-tact-multisource.md`). Blend is Blend and LF. Frequency is SIG. Warp is WaNet. Quantization is BPP. The benign controls are benign models probed with a BadNets or a Blend trigger they never learned, the control that separates "the statistic reacts to a backdoor" from "the statistic reacts to a trigger-shaped input".

The models read here are the ones successful at the 2 point bar of the coverage ledger (`successful_2pt`: the attack's ASR clears the declared bar and clean accuracy stays within 2 points of the benign model), the user's decision of 2026-09-29, rebuilt the ledger's way for Swin. SIG has no reading. An audit on 2026-09-29 (`docs/audits/2026-09-29-experiment-audit.md`) found that SIG models trained before 2026-09-09 learned a weaker trigger than `attacks/sig.py` now builds, and the success bar then removed the only ViT SIG model of the paper's datasets. `measure.py` keeps the SIG entries quarantined until their provenance is settled.

The next cell loads the files. The table lists every model measured, its group, how many triggered images the unperturbed model sends to the target ("hit" pairs, the only ones with a target decision to keep), its paired clean accuracy and ASR on these images, whether it is successful at the 2 and 5 point bars, and how well its unperturbed predictions agree with the sweep's cached baseline on the same images.

<!-- parameters -->
The settings every step uses, read from `measure.py` through `summary.json`. Each operator runs at its own placement's adaptive rate, the smallest ladder rate at which the share of clean held-out predictions that change on a pass reaches the adaptive target (`defenses.decision.select_rate_adaptively`), read from the model's `psbd_metrics.json` or, for a placement the sweeps never cached, calibrated by the same rule on held-out images (`measure.calibrate_rate`). Matching every operator at the same disturbance of clean predictions is what makes their readings comparable. More passes than the pipeline's are used so that a per-image statistic is not dominated by Monte Carlo noise.

<!-- sanity -->
## Step 0. Are these passes the pipeline's passes?

**Question.** Every later number comes from forward passes this experiment makes with its own hooks, pairs and rates. If those passes differ from the pipeline that produced the paper's numbers, a mechanism found here could belong to a different computation. So the passes are checked against the pipeline first, on 1 model and its full PSBD splits.

**What is measured.** `experiments/why_psbd_works/sanity.py` runs 3 gates. Gate 1 recomputes clean accuracy on the full test set and ASR on the full eval ASR split and compares them with `checkpoints/<folder>/metrics.json`. Gate 2 reruns PSBD-TM and PSBD-RD at their adaptive rates with the pipeline's own function (`defenses.inference.compute_dropout_pass_probs` with the sweep's mask seed, batch size and pass count), compares the per-sample fractional statistic with the cached per-pass tensors under `results/<folder>/psbd/`, and compares the AUROC of `defenses.decision.detection_report` with `psbd_metrics.json`. It runs under bfloat16 as the sweeps do and again in float32. Gate 3 checks the pairing: every triggered row carries the target label, no clean twin is of the target class, and on this patch trigger each triggered image differs from its clean twin only inside the trigger's pixels.

Other gates are checked on every record rather than once: the hooks a model carries before and after a run, the modules left with a replaced forward, the eval flag, the model's own dropouts, and the logits of the first batch recomputed after every measurement against the same batch at the start.

<!-- step1 -->
## Step 1. The phenomenon on these models

**Question.** Does the statistic separate triggered from clean images on the models used here, with every operator, in every attack category? This comes first because everything after it explains this figure, and an explanation of a phenomenon that is not there explains nothing.

**What is measured.** For every model and operator, the AUROC of $\phi$ with the triggered image as the positive class and a low $\phi$ flagging it, on triggered hit images against all clean twins (`auroc_hit_only`, from `measure.operator_readout`).

**How to read the figure.** Rows are attack categories and the 2 benign probes, columns are operators, the colour and the printed number are the mean AUROC over the row's models. Blue above 0.5 means triggered images score lower than clean ones, the direction PSBD needs, and red below 0.5 means the direction is inverted. The table under the figure gives the pipeline's paired form per model (`auroc_paired`: every triggered row against its own clean twin through `defenses.decision.detection_report`).

<!-- step1_hist -->
An AUROC hides the shape of the 2 populations. The next figure shows, for 1 model per category (named under it), the per-image statistic of the clean twins and of the triggered hit images, under PSBD-TM (top row) and PSBD-RD (bottom row). The x axis is $\phi$, the fraction of the own-class probability lost on average over the passes, from 0 (nothing lost) to 1 (all of it lost). Values below 0 mean the passes raised it. Look at where the mass of each population sits and at how much the 2 overlap, since the overlap is what the threshold has to cut through.

<!-- step2 -->
## Step 2. H-margin: does the triggered answer keep its margin?

**Question.** The most direct reading of the phenomenon is about margins. A perturbation moves every logit a little. An answer whose logit leads the runner-up by a lot survives and 1 that leads by a little does not. If triggered answers simply start with larger margins, PSBD is a margin detector and needs no perturbation at all. If they start with margins like clean ones and lose less of them, something in the triggered representation protects the margin, which is what the next steps look for.

**What is measured** (`measure.per_image_statistics`, `summarize.margin_matched_auroc`). The own-class margin, unperturbed and on every pass, and its retention:

$$m(x) = z_c(x) - \max_{i\neq c} z_i(x), \qquad m_j(x) = z_c(x;\xi_j) - \max_{i \neq c} z_i(x;\xi_j), \qquad r(x) = \frac{\tfrac{1}{k}\sum_j m_j(x)}{m(x)}$$

| symbol | meaning |
|---|---|
| $z_i(x)$, $z_i(x;\xi_j)$ | the logit of class $i$, unperturbed and on pass $j$ |
| $m$, $m_j$ | the own-class margin, negative once the answer has flipped |
| $r$ | margin retention, 1 when the perturbation leaves the margin whole |

- Panel a: the AUROC of the unperturbed margin and of the unperturbed confidence $P_c(x)$, with a larger value on the triggered image as the positive direction. If these match the statistic's AUROC, the perturbation adds nothing.
- Panel b: the median retention of triggered hit images (solid) and of clean images (dotted) under the 5 headline operators, 1 line per category.
- Panel c: the statistic's AUROC against its margin-matched AUROC, 1 point per model and operator. Each triggered image is paired with the clean image of the nearest unperturbed margin. A point on the diagonal means equal starting margins do not reduce the separation, a point far below it means the separation came from larger starting margins.

<!-- step3 -->
## Step 3. H-direction: what protects the triggered margin?

**Question.** Step 2 establishes whether the triggered margin survives better, and whether that survives margin matching. The margin is a linear readout of 1 feature vector, the class token after the final LayerNorm on ViT and the pooled final feature map on Swin, so whatever protects it must be visible in that vector. The candidate from earlier work in this repository (`experiments/backdoor_neurons/`) is the backdoor direction: removing 1 direction from the residual stream removes the backdoor, while zeroing many coordinates does not. If the triggered feature is dominated by that direction, and the perturbation dents it less than it dents a clean image's class evidence, the margin is protected.

**What is measured** (`measure.feature_geometry`, `measure.feature_statistics`). With $f(x)$ the feature, $\mu$ the mean clean feature, $\tilde x$ the triggered twin of $x$, $H$ the hit pairs and $W$ the head's weights:

$$d = \frac{1}{|H|}\sum_{x \in H}\big(f(\tilde x) - f(x)\big), \qquad u = \frac{d}{\lVert d\rVert}, \qquad e_c = \frac{W_c - \bar W}{\lVert W_c - \bar W\rVert}$$

$$s_u(x) = \langle f(x) - \mu, u\rangle, \qquad \text{share}(x) = \frac{s_u(x)^2}{\lVert f(x) - \mu\rVert^2}, \qquad \text{SPR}_u(x) = \frac{s_u(x)}{\sqrt{\tfrac{1}{k}\sum_j \langle f_j(x) - f(x), u\rangle^2}}$$

| symbol | meaning |
|---|---|
| $d$, $u$ | the backdoor direction (`analysis.direction.backdoor_direction`) and its unit vector |
| $W_c$, $\bar W$ | the head's row for class $c$ and the mean row |
| $e_c$ | the class direction of $c$, along which the logit of $c$ rises against the average class |
| $f_j(x)$ | the feature on pass $j$ |
| $s_u$ | the backdoor component of an image's deviation from the clean mean |
| share | the fraction of that deviation the backdoor component makes up |
| $\text{SPR}_u$ | signal to perturbation ratio: the component over the perturbation's spread along the same direction |

The clean counterpart uses $e_c$ of the image's own class in place of $u$.

- Panel a: the median share on triggered hit images per category, next to the cosine between $u$ and the target's readout direction $e_t$.
- Panel b: the ratio of the triggered SPR along $u$ to the clean SPR along $e_c$, per operator. Above 1 means the triggered component stands further above the perturbation noise than the clean class evidence does.
- Panel c: the literature memo's L8 test, the per-image Spearman correlation between $\phi$ and the fraction of the backdoor projection a pass loses. Most triggered images never flip under PSBD-TM, so their $\phi$ sits at the bfloat16 floor, and the theory note asks for the correlation on triggered images above a small floor of $\phi$ and on clean and triggered images pooled. The dashed lines are the theory note's supported and refuted values.

<!-- step4 -->
## Step 4. H-redundancy: where the dominant component comes from

**Question.** A dominant backdoor component explains why the margin survives, not why the component itself survives a perturbation that removes most of a clean image's evidence. The perturbations act on tokens, and a token-level reason is that the trigger's evidence is present in many tokens, so removing any subset leaves the rest to carry it. That is the redundancy account (memo L19 and L23). Section D of `experiments/why_token_masking_works/` measured it on ViT with a few patterns shared by every model, which the audit found too few and not independent. This step repeats it with many patterns seeded per model, adds a control for patch triggers and runs it on Swin.

**What is measured** (`measure.measure_redundancy`, `measure.SubsetTokenMask`). A fixed random pattern of the 14 by 14 patch grid stays visible, and every other patch token is zeroed at the input of every block's attention LayerNorm (on Swin the pattern is resized to each stage's grid). A zeroed token enters attention as the LayerNorm's bias, so the network sees the visible tokens' content only. Solid lines are the share of triggered hit images still sent to the target, dotted lines the share of clean images keeping their answer. A masked model can collapse onto 1 default class, sometimes the target, so the verdict reads the excess retention, the target share of triggered images minus that of clean images relative to its value with every token visible:

$$\text{excess}(f) = \frac{\text{ASR}(f) - \text{clean on target}(f)}{\text{ASR}(1) - \text{clean on target}(1)}$$

with $f$ the visible fraction. For patch triggers the dashed line repeats every pattern with the trigger's own tokens forced visible. If triggered survival holds only when the trigger stays visible, random patterns protect a patch trigger by rarely hiding its few tokens, which is geometry and not redundancy.

<!-- wanet -->
### Step 4b. WaNet's warp as a coherence check (L26)

**Question.** WaNet is the attack whose evidence survives token removal least among the global triggers in earlier work, and the literature memo offers a reason: WaNet trains with a noise mode, images warped by a random field that keep their true label, so the network must learn the exact warp and not "warped" as such. If the exact warp can only be told from a random warp over a wide area, no single token can read it, and the trigger is legible only where many tokens are combined, which is the class token late in the network.

**What is measured** (`wanet_probe.py`). For each WaNet model, clean analysis images, their exact-warp copies (the trigger) and their noise-warp copies (`attacks.wanet`'s own `apply_cover`). The residual stream entering several blocks is captured, and a logistic probe per block tells the exact warp from the random warp, fitted on half of the images and scored on the other half, once on single patch tokens pooled over positions and once on the class token. The theory note's prediction: single tokens near chance while the class token reaches a high accuracy. The table also gives how often the model sends each image set to the target, which checks that the noise mode worked (random warps should not fire).

<!-- step5 -->
## Step 5. H-low-dim: how concentrated the triggered decision is

**Question.** Redundancy is about how the trigger's evidence is spread over tokens. The complementary question is whether the decision that evidence produces is narrow: few feature directions, few tokens in the gradient, a small sufficient part of the input. This is the shortcut account (memo L4 and L6) in geometric form.

**What is measured.**

- Panel a (`measure.difference_spectrum`): the share of the largest eigenvalue in the uncentered second moment of the paired change $f(\tilde x) - f(x)$, against the same for the change between 2 unrelated clean images. A share near 1 means 1 direction carries the whole trigger effect.
- Panel b (`measure.measure_jacobian`): the effective number of tokens in the gradient of the margin with respect to the residual stream entering the last probed block, $\exp$ of the entropy of each token's share of the squared gradient norm, in float32.
- Panel c (`measure.measure_sufficiency`): the literature memo's L6 test, the smallest visible share of tokens that keeps each image's answer when tokens are removed in a nested order at the attention input of every block. The ranked order removes the tokens of smallest attribution first (gradient times activation at the stream entering block 1), which approximates Cognitive Distillation's optimized mask. The random order is the per-image form of step 4.

The table adds the AUROC of the sufficient share used as a detector, the Gini coefficient of expected-gradient input attributions (`analysis.attribution.expected_gradients`) and the effective rank of the Jacobian.

<!-- step6 -->
## Step 6. H-flatness: is the statistic a curvature?

**Question.** The project's theory note (`docs/theory-perturbation-consistency.md`) expands the statistic to second order: for a zero-mean perturbation with covariance $\Sigma$ at a site, $\phi \approx -\tfrac{1}{2}\operatorname{tr}(H\Sigma)/P_c$, with $H$ the Hessian of $P_c$ in that site's activation. If that is the mechanism, triggered inputs sit where the Hessian is small, a flatter region, and the statistic behaves like a curvature estimate as the noise shrinks.

**What is measured.**

- Panel a (`measure.measure_flatness`): symmetric finite differences of $\log P_c$ along random Gaussian directions scaled to each image's own activation, in pixel space and in the residual stream entering the probed blocks, float32. The colour is the AUROC with a small absolute curvature flagging the triggered image.
- Panel b (`measure.measure_small_noise`): the theory note's decisive test. Gaussian noise at the attention input at small rates, float32, more passes, next to the same operator at its adaptive rate (the last point of each line). The second-order account predicts that $\phi(2r)/\phi(r)$ is near 4 per image, that images keep their ranking across $r$ and that the AUROC at small $r$ stays at the operating AUROC.

<!-- refute_intro -->
## Step 7. Explanations that do not hold

The steps so far built a positive account. This step takes the explanations a reader, a reviewer or the PSBD authors would reach for first and tests each. For each: the claim, the prediction that would hold if it were true, the measurement and the verdict. The identifiers are the literature memo's and the thresholds the theory note's. Where a refutation already exists in this repository it is redrawn from its source file rather than cited.

<!-- refute_confidence -->
### 7a. "Triggered inputs are just more confident" (P1)

**Claim.** A backdoored model is nearly certain on triggered inputs and a nearly certain softmax is flat, so the statistic only reads confidence and the perturbation adds nothing.

**Prediction if true.** No detector built on the statistic beats the best detector built on confidence alone. The best function of confidence is not confidence itself but its likelihood ratio, triggered against clean, and its AUROC $A^{*}(P_c)$ is an oracle bound since it needs the triggered distribution. The theory note computes it cross-fitted on bins of $-\log(1 - P_c)$, and `cached_reads.confidence_ceiling_row` rebuilds it per model from the sweep's cached baselines with folds split by pair.

**Measurement.** Panel a plots every panel model's PSBD-TM (dots) and PSBD-RD (crosses) AUROC against its $A^{*}(P_c)$. Points above the diagonal beat everything confidence can do. Panel b gives this experiment's unperturbed confidence AUROC next to the PSBD-TM statistic on its own pairs.

<!-- refute_uncertainty -->
### 7b. "It is MC-dropout uncertainty" (P2, L12)

**Claim.** Dropout at inference samples from an approximate posterior (Gal and Ghahramani), so the statistic is epistemic uncertainty, low on inputs the model has in effect memorized.

**Prediction if true.** 2 predictions. Any uncertainty read off the same passes separates as well, including readings that ignore which class the model chose: the entropy of the mean prediction and the BALD mutual information. And a genuine epistemic estimate, independently trained replicates of the same model, separates as well and orders clean images the way PSBD-TM does. The theory note replaces the hard vote of 3 seeds, which cannot reach the required AUROC at these accuracies, by the continuous statistic $\phi_{\text{ens}}(x) = 1 - \tfrac{1}{2}\sum_{s=1}^{2} P_c(x;\theta_s)/P_c(x;\theta_0)$, the statistic itself with the 2 other seeds in place of the passes.

A conceptual point comes first. Gal and Ghahramani's reading needs the network to have been trained with the same dropout. Every model here was trained without dropout, so the passes approximate no posterior, and the statistic works with operators that are not dropout at all.

**Measurement.** Panel a: 4 readings of the same PSBD-TM passes (`measure.pass_uncertainty`). Panel b: `seed_ensemble.py` on cells with 3 successful training seeds, $\phi_{\text{ens}}$ against the cached PSBD-TM AUROC and $A^{*}(P_c)$ of the seed-0 model.

<!-- refute_ood -->
### 7c. "The trigger is out of distribution" (P5)

**Claim.** A BadNets square or a blended pattern is not a natural image, and unusual inputs behave unusually under perturbation.

**Prediction if true.** An out-of-distribution score separates triggered from clean images, and the statistic flags genuinely foreign images as poisoned too.

**Measurement.** Panel a: the AUROC of the kNN distance (1 minus the cosine similarity of the final feature to its $K$-th nearest held-out clean feature) for triggered against clean images, and for clean images of another dataset against clean images, the check that the score detects a real shift when there is one (`measure.ood_readout`). Panel b: the share of clean, triggered and foreign images below PSBD-TM's threshold, the headline quantile of the held-out images' statistic (`measure.flagged_shares`). The table gives the benign references' own PSBD-TM and PSBD-RD AUROC from `psbd_metrics.json`, the memo's original refutation: the same triggered images are just as unusual for a benign model, which does not flag them.

<!-- refute_neurons -->
### 7d. "Specific trigger neurons are robust" (P3, L7)

**Claim.** Li et al. write that PSBD works by "utilizing robust neuron bias paths", and the pruning literature treats a backdoor as a small set of units. If a few units carry the trigger and survive the perturbation, the triggered answer survives.

**Prediction if true.** Zeroing those units removes the backdoor, far below zeroing as many random units.

**Measurement.** Panel a (`measure.measure_neurons`): the MLP hidden units after GELU, ranked by how much the trigger raises them on the class token (pooled positions on Swin), zeroed at every token: the memo's L7 set (the top units of the last blocks) against as many random units, and the top share of units in every block against as many random ones. Panel b redraws `results/_experiments/backdoor_neurons/backdoor_neuron_ablation.json`: in the residual stream, removing the rank-1 backdoor direction against zeroing the top TAC coordinates or a random direction.

<!-- refute_bias -->
### 7e. "Clean inputs fall onto the target" (Li et al.'s neuron bias, P6 and L10) and "the target class is easy" (P10)

**Claim, neuron bias.** In plain words, Li et al.'s account: training on poisoned data biases some neurons toward the target class, so when a perturbation removes a clean image's own evidence, what is left points to the target. The backdoor installs an attractor: a perturbation that removes clean evidence pushes clean predictions onto the target, which lowers their own-class probability, while triggered inputs already sit at the attractor.

**Prediction if true.** Most shifted clean answers land on the target, and removing the target's share from the statistic erases the separation. The theory note's L10 statistic redistributes the target's gain for clean images whose answer is not the target and keeps triggered scores unchanged:

$$\tilde P_c(x;\xi_j) = P_c(x;\xi_j)\,\frac{1 - P_t(x)}{1 - P_t(x;\xi_j)}$$

with $t$ the target. It needs the target's probability on every pass, which the sweep's cache does not store, so it is computed on this experiment's passes (`measure.psu_ratio_without_class`).

**Claim, target easy.** Nobody states this in print, and a reviewer would: if the target class were intrinsically stable, every image predicted as the target would score low, triggered or not.

**Prediction if true.** Most clean images of the target class fall below the threshold on backdoored models and few on benign ones, measured on the unpaired clean split because the paired one drops the target class under all to one.

**Measurement.** Panel a: the share of shifted clean answers that land on the target per operator, against 1 over the class count. Panel b: the AUROC lost under PSBD-TM and PSBD-RD when the target's gain is redistributed, with the theory note's thresholds dashed. Panel c (`cached_reads.target_class_reading`, the whole panel and the benign references): the flagged share of clean target-class images (solid) and of the other classes (pale). The dashed line is the base rate the threshold sets by construction.

<!-- refute_memo -->
### 7f. The literature memo's other refutations, redrawn from their sources

- Panel a, P9 "the triggered representation is collapsed", a correlation with no isolating control, shown as context and not as a finding: the final rank ratio against the best deployable AUROC, from `results/_experiments/latent_geometry_predicts_detection/geometry_vs_detection.csv`. A collapse that carried detection would line the points up.
- Panel b, L14, IBD-PSC's theorem. In plain words: IBD-PSC (Hou et al.) multiplies the scale of normalization layers by a large factor, which makes every feature very large. It proves under a simplifying assumption that past some factor a backdoored model then predicts the target for almost any input, while a triggered input, already at the target, stays there. If that were why such detectors work, amplification would push most clean answers onto the target. Measured at the top of the `gain_scale` ladder at `mlp_norm_out`, per attack, the clean shift ratio, the target's share of shifted clean answers and the largest single class's share. `GainScale` reads its rate as the amplification factor minus 1, so the memo's "factor 15" is the ladder's rate 15.
- Panel c, L13, SCALE-UP's theorem. In plain words: SCALE-UP (Guo et al.) multiplies pixel values and argues, in an idealized model where the network is a kernel machine, that a triggered input keeps its prediction under any amplification once enough of the training set is poisoned. Its prediction for us is that the detector gets better as the poison rate rises. Measured `input_pixels_scale_up` AUROC by attack and poison rate. The theory note labels this an extrapolation of a limit theorem, not a test of it.
- The table, L18 and P1: per attack, PSBD-TM against token masking once at the embedding (`after_embedding_token_mask`, Doan et al.'s patch drop) and against the confidence-only detector record.

<!-- fragility -->
## Step 8. Clean fragility, the other half of the statistic

**Question.** Everything so far asked why triggered answers survive. The statistic is a comparison, and its false positives are clean images whose answers also survive. The theory note (`docs/why-psbd-works-theory.md`, section "Clean fragility under token masking") shows how clean answers break under token masking. This step redraws its 3 findings from the sweep's cached tensors over the panel (`cached_reads.clean_fragility`).

**What is measured.**

- Panel a: the share of held-out clean answers kept along PSBD-TM's rate ladder, averaged per dataset (dots), with 2 fits after the share predicted as the model's default class at the top of the ladder is removed as a floor. Solid: every image has its own critical rate, logistic across images, $\text{keep}(p) = \pi_d + (1 - \pi_d)/(1 + e^{(p - p_{50})/s})$. Dotted: every image needs a fixed number of tokens independently, $(1 - p)^n$.
- Panel b: at the adaptive rate with the sweep's pass count, how many passes change a held-out image's answer, against a binomial with the same mean, which is what independent flips would give. The intra-image correlation of flips $\rho$ follows from $\operatorname{Var}(N) = kq(1-q)(1 + (k - 1)\rho)$.
- Panel c: per model, the share of the false positives (paired clean images below the threshold) that never flipped on any pass.

<!-- critical -->
## Step 9. 1 account for every operator and attack: the critical rate

**Question.** The steps so far give readings per attack category. The user asked for 1 account that covers every operator and attack. The candidate joins the triggered half (steps 2 to 5) and the clean half (step 8): each input has a critical rate $p^*$, the smallest rate on a placement's ladder at which most of its perturbed passes change its answer. If PSBD is a 2-sample test on $p^*$, then at any placement it separates exactly as far as triggered $p^*$ lies above clean $p^*$, and operators differ only in how far they push the 2 distributions apart.

**What is measured** (`critical_rate.py`, CPU, from the sweep's cached per-pass predictions). For every panel model and every placement with a full rate ladder, $p^*$ of every clean and triggered image (`defenses.scores.critical_rate` with a majority of passes, through `defenses.decision.load_critical_rate_from_disk`), and

$$A^\star = P\big(p^*_{\text{triggered}} > p^*_{\text{clean}}\big)$$

over each triggered image and its own clean twin, ties counted half. An image whose answer survives every rate is placed above the ladder.

**Predictions, fixed in the script's docstring before the first read.** P1, 1 curve: across every (model, placement) the AUROC at the adaptive rate is a monotone function of $A^\star$ (Spearman at least 0.8, median absolute difference at most 0.05). P2, control: on the benign models probed with a trigger they never learned, $A^\star$ stays within 0.05 of 0.5 at every placement. P3, the rate: the adaptive rule places the rate where about 0.8 of held-out clean images have $p^*$ at or below it, not at the clean median, since the rule asks for a clean shift ratio of 0.8.

- Panel a: AUROC at the adaptive rate against $A^\star$, 1 point per (model, placement), coloured by placement, benign models in grey. The unifying claim is that all points lie on 1 curve.
- Panel b: the share of held-out clean images with $p^*$ at or below the adaptive rate, per (model, placement).
- The table links $A^\star$ to this experiment's mechanism readings on its own models and operators (Spearman across model and operator pairs). That link is a correlation over few models, so it is a hypothesis about which mechanism sets the $p^*$ gap, not a finding.

<!-- curves -->
### Breaking-point curves per model

The shift ratio at rate $p$, the share of passes that change an image's answer, is over a whole ladder the distribution of the breaking point $p^*$: at each rate it counts the passes whose image has already broken. Drawn per model for clean and triggered images, the curves show where each population breaks, and a benign model's 2 curves should coincide. `shift_curves.py` writes them from every model's `psbd_metrics.json`, for every placement with a ladder on both panels and the benign references, and its JSON is the figure's sidecar. The clean curve covers the whole clean analysis split and the triggered curve every triggered row, captured or not.

The figures show PSBD-TM and PSBD-RD, 1 thin line per model, clean in blue and triggered in orange, and a panel per attack, benign models last. The table gives, per attack and placement, the median over models of the rate at which each curve reaches 0.5 and the number of models whose triggered curve never does. An exploratory read of these curves (by the coordinator, from the median model) suggested 6 patterns, and the next cell tests each on the per-model data: (a) under PSBD-TM triggered images break late and clean images early, (b) under PSBD-TM the TaCT curves overlap although PSBD-TM detects TaCT, (c) under PSBD-RD the BadNets curves nearly overlap and TaCT triggered images break earlier than clean ones, (d) under PSBD-RD global triggers break at about twice the clean rate, (e) under PSBD-RD WaNet triggered images never fully break, (f) under PSBD-TM WaNet separates for most models and fails on 1.

<!-- swin -->
## Step 10. Swin-S

**Question.** Swin-S has no class token, reads the mean of its last stage and attends within windows, so a mechanism that rests on ViT's late class-token read might not transfer. This step asks whether the statistic works on Swin for the same reasons.

**What is measured.** Every measurement of steps 1 to 6 on the Swin models of the same attacks. `measure.py` handles both layouts: the operators flatten Swin's (batch, height, width, channels) maps, the redundancy pattern is resized per stage and the stream sites are blocks of stages 3 and 4. The figure compares the redundancy reading of the 2 architectures, since the theory note's L20 and L23 predict that windowed attention and mean pooling make Swin's evidence more redundant. The table lists the key number of every hypothesis side by side.

<!-- verdicts -->
## Step 12. The verdicts

Each cell is 1 hypothesis on 1 attack category: green supported, yellow partial, red refuted, blank no data, with the key number the rule reads. The rules and their thresholds are printed under the grid from `summary.json`, where `summarize.py` stored them beside each verdict, with thresholds from `docs/why-psbd-works-theory.md` where it gives one. The last table gives the 3 accounts read on the whole panel or on the seed-replicated cells rather than on this experiment's models: confidence (P1), the target class being easy (P10) and the epistemic ensemble (P2).

<!-- sufficiency -->
## Step 11. The unifying hypothesis: PSBD detects over-determined decisions

**Question.** Steps 2 to 10 give readings per attack, and the critical rate (step 9) restates PSBD in 1 variable. The user asked for 1 explanation of why PSBD works and where it fails. The candidate, stated and its predictions registered in `sufficiency.py` before any reading: a backdoor that is a true shortcut, a trigger sufficient on its own and independent of the image, gives the triggered decision far more evidence than it needs, so random removal leaves enough of it. A clean decision has just enough evidence and breaks. PSBD should fail where the backdoor is a conjunction with content (TaCT flips only its source class) or relational along the axis the probe removes (WaNet's warp under token removal).

**What is measured** (`sufficiency.py`, 1 isolating change per condition). The models are the evidence-surplus set the user chose (`experiments/evidence_surplus/`): 10 ViT models covering BadNets, Blend, BPP, LF, WaNet and TaCT, the 2 ResNet-18 reproductions of the original paper (read with PSBD-RD, its own site) and 2 benign ViT models probed with a BadNets and a Blend trigger.

- Blank: the trigger stamped on content-free carriers (the dataset's mean image, mid gray and uniform noise images) against the same carriers unstamped. The excess share sent to the target is the sufficiency score: how much of the attack label the trigger carries with no image at all.
- Content hidden: the triggered image with every non-trigger token hidden at the attention input of every block (patch triggers) or a random 70% hidden (global triggers), against its clean twin under the same mask. Where the masked clean twin already goes to the target, the model's default class under heavy masking is the target and the reading has no room, which the record flags.
- Classes: the trigger stamped on test images of every non-target class, the attack's source restriction ignored. A shortcut sends every class to the target, a conjunction only its source classes.

**Predictions** (from the script's docstring). S1: the sufficiency score predicts PSBD-TM's AUROC and its TPR at the 0.01 quantile at the adaptive rate across the set's backdoored models with 1 monotone relation, Spearman at least 0.6 each. S2: TaCT sits low on sufficiency and on class independence. S3: the benign models sit near 0 on every excess.

- Panel a: the detector's AUROC against the sufficiency score, 1 point per model, ViT dots and ResNet-18 crosses, coloured by attack.
- Panel b: the same for TPR at the 0.01 quantile.
- Panel c: the share of non-source images sent to the target when stamped, per attack and architecture, the content-dependence control.
