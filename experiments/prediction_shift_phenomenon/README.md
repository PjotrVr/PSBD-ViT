# The prediction shift phenomenon on ResNet-18, ViT-B/16 and Swin-S

## Question

PSBD (Li et al., arXiv 2406.05826) rests on an observation the authors call the prediction shift
phenomenon, and on an explanation of it they call neuron bias. They show both on ResNet-18 (and on
VGG16-bn in 1 appendix figure) and nowhere else. This repository adapts PSBD to ViT-B/16 and
Swin-S, and its headline detector PSBD-TM masks tokens at the attention input, a perturbation the
paper never used. If the phenomenon the paper describes does not appear on transformers, our
detector works for some other reason, and the paper's mechanism cannot be cited for it. This
experiment therefore reproduces each observation the paper makes, first on our own ResNet-18
checkpoints and then with the same code on the ViT-B/16 and Swin-S panel models, under both the
paper's site (PSBD-RD) and ours (PSBD-TM). Every claim gets a verdict per architecture, and a
claim that fails gets the same detail as one that holds.

The walkthrough for a reader is `notebooks/prediction-shift-phenomenon.ipynb`. It draws every
figure named below from the JSON this directory writes and states the same verdicts.

## The paper's claims, as read

In plain words. (1) Dropout changes the answer for clean images far more often than for triggered
ones. (2) When a clean answer changes, it changes to the attacker's target class. (3) Inside the
last layer, dropout makes a clean image look like its triggered twin, which the paper calls neuron
bias. (4) The ordinary dropout uncertainty score (how much confidence wobbles) catches BadNets but
misses WaNet, which is why PSBD measures how far confidence drops instead. Each claim below is
tested by 1 experiment that changes 1 thing (the rate, dropout on or off or the statistic), keeps a
control (a benign model, clean images or an unrelated image pair) and has a figure in the
notebook. An explanation offered without such a test is labeled a hypothesis.

The sources are `literature/psbd-li-arxiv2024/source/sec/4_method.tex` and
`literature/psbd-li-arxiv2024/source/sec/7_appendix.tex`, with the figure files under
`literature/psbd-li-arxiv2024/source/fig/`. The figure numbers are the compiled paper's. Figure 1
is the conceptual diagram in the introduction, so the method section's figures are 2, 3 and 4,
and the appendix figures are A1 to A7.

| claim | the paper's words | where they show it | their setup |
|---|---|---|---|
| 1 shift ratio | "the shift ratio curve for clean data still follows an increasing trend as p increases, eventually stabilizing. However, when p reaches a certain special value, the sigma for backdoor data approaches 0, while the sigma for clean data reaches a relatively high value (around 0.8)" | Fig. 3 top row (benign, BadNets and WaNet models), Fig. A3 top rows (Blend, TrojanNN, LC, ISSBA, Adaptive-Blend), Fig. A4 (Tiny ImageNet), Fig. A5 (VGG16-bn) | CIFAR-10, ResNet-18, 100 epochs of SGD, 10% poisoning, target 0, k = 3 dropout passes, rates 0.1 to 0.9, dropout "after each residual connection in the residual basic block, before the activation function" |
| 1b benign control | "both clean and backdoor training data exhibit similar shift ratio trends, supporting the conclusion ... that the benign model treats backdoor data as perturbed clean data" | Fig. 3 top left | the same, on a model trained without poison |
| 2 shift intensity | "among the samples experiencing PS, almost all clean data shifts to the target class y_t (class 0 in our experiments)" and, for the appendix attacks, "in all attack scenarios, clean data exhibits a bias towards the target class" | Fig. 3 bottom row at the adaptive p = 0.7, Fig. A3 bottom rows | shift intensity is "the proportion of times a sample was predicted to a particular class during PS" |
| 2b universality | "about 60% of clean training data that experience Prediction Shift (PS) under the benign model shift to class 3 ... suggesting that PS is a universal characteristic of DNNs" | Fig. 3 bottom left | benign model |
| 2c Tiny exception | on Tiny ImageNet "the shift classes ... exhibit a predominant inclination towards a certain class rather than the target class" | Fig. A4 | 200 classes, with augmentation |
| 3 neuron bias | "without the dropout, the features of clean and backdoor version exhibit minimal similarity ... under an appropriate dropout rate ... the features of clean and backdoor version become almost identical with dropout", with red boxes where "the feature map values are non-zero and the difference between each activation value in the corresponding feature maps is no greater than 1" | Fig. 4 (first 64 of 512 top-layer maps, p = 0.91), Fig. A6 (all 512, BadNets), Fig. A7 (all 512, WaNet) | 1 clean image and its triggered twin |
| 4 pilot study | "the average uncertainty of backdoor training data under BadNets is significantly lower than that of clean training and validation data ... However ... the uncertainty of backdoor data under WaNet sometimes matches or exceeds that of clean data", and in the appendix "under Adaptive-Blend attack ... the uncertainty of backdoor training data is even slightly higher than clean training data" | Fig. 2 (BadNets, WaNet), Fig. A1 (benign, Blend, TrojanNN, LC, ISSBA, Adaptive-Blend) | MC-Dropout uncertainty is "the standard deviation of the highest confidence class", tracked over all 100 training epochs |
| 4b input scaling | multiplying pixels by 3 before MC-Dropout (the SCP variant) still fails on WaNet, LC and Adaptive-Blend | Fig. A2 | out of scope here, see "Deviations" |

Reading the paper's figure files closely settles 2 details the text leaves open. In Fig. 3 the
backdoor curve of the BadNets model is not monotone: it rises to about 0.25 near p = 0.37 and
falls to about 0 by p = 0.6, and the shifted backdoor predictions in the bottom row land on class 3
(intensity 0.75), not on the target. In Fig. 4 the non-zero entries of the clean and the backdoor
maps with dropout sit at exactly the same positions, which can only happen if the 2 forward passes
shared 1 dropout mask. We read the figure that way and measure both a shared mask and independent
masks.

## Terms

- **Backdoor, trigger, target.** A poisoned model maps any input carrying the attacker's trigger
  to the target class. A benign model was trained without poison. A triggered image is a clean
  test image with the trigger applied.
- **Captured.** A triggered image the unperturbed model actually sends to the target. The paper's
  backdoor set is poisoned training data the model has fitted, so on a poisoned model we score
  captured triggered images only (`defenses.decision.attack_success_mask`).
- **Position, operator and placement.** Where the perturbation acts, what it does there and the pair
  of both, which names the cache folder (`CLAUDE.md`, "Canon").
- **PSBD-RD.** Our adaptation of the paper's site, placement `post_residual`: an `nn.Dropout` after
  each residual add. ResNet-18 has 1 add per basic block, so 8 probes. A transformer block has 2,
  so ViT-B/16 gets 24 probes and Swin-S 48.
- **PSBD-TM.** Placement `before_attention_norm_token_mask`: whole tokens zeroed at the input of
  every attention LayerNorm, with the class token of ViT never masked. ResNet-18 has no token axis,
  so PSBD-TM does not exist there.
- **Rate p and passes k.** The drop probability of the operator, and the number of independent
  perturbed forward passes. Every cache here has k = 3, the paper's value.
- **Adaptive rate.** The smallest swept rate whose clean-validation shift ratio reaches 0.8
  (`defenses.decision.select_rate_adaptively`, `ADAPTIVE_SHIFT_TARGET`). It is the first half of
  the paper's rule and the repository's canonical rate rule.
- **Splits.** Every model's test set is split once (`data.splits`, seed `PSBD_SPLIT_SEED`): 2000
  clean validation images, and the rest as the analysis pool, whose clean images are the "clean
  analysis" split and whose trigger-eligible images with the trigger applied are the "backdoor"
  split. Row i of the backdoor split and its clean twin are the same test image.

## Method

Claims 1, 2 and 4 are read from the stage-1 caches that `cli.sweep` wrote under
`results/<folder>/psbd/` (`defenses.cache`): the unperturbed softmax and argmax of each split in
`baseline_<split>.pt`, and for every rate the (k, n) probability of the unperturbed argmax class
and the (k, n) perturbed argmax in `<placement>/rate_<p>_<split>.pt`. Nothing here recomputes a
forward pass for these 3 claims, which is why they cover every panel model. Claim 3 needs the
final-layer activations, which no cache holds, so it runs forward passes on a GPU.

### Claim 1, the shift ratio curve

The paper's Eq. (PS definition), computed by `defenses.scores.shift_ratio` inside
`measure.measure_rate`:

$$\phi_{PS}(\mathbf{x}) = \mathbb{I}\left(\mathcal{Y}(\mathbf{x};\boldsymbol\theta) \neq \mathcal{Y}(\mathbf{x};\boldsymbol\theta')\right), \qquad \sigma(\mathcal{D}) = \frac{1}{k|\mathcal{D}|}\sum_{\mathbf{x}\in\mathcal{D}}\sum_{i=1}^{k}\phi_{PS}^{(i)}(\mathbf{x})$$

| symbol | meaning |
|---|---|
| $\mathcal{D}$ | 1 split: clean validation, clean analysis, or captured triggered images |
| $\mathcal{Y}(\mathbf{x};\boldsymbol\theta)$ | the unperturbed argmax class |
| $\mathcal{Y}(\mathbf{x};\boldsymbol\theta')$ | the argmax of 1 perturbed pass |
| $\phi_{PS}^{(i)}$ | 1 when pass i changed the prediction |
| $k$ | passes, 3 |
| $\sigma$ | shift ratio, the share of (image, pass) predictions that moved |

We draw $\sigma$ against every swept rate for each split (Figure 1 of the notebook) and read it at
the adaptive rate for every model. The verdict rule, `measure.claim_1_verdict`, follows the
paper's words. The claim **holds** on a poisoned model when the triggered $\sigma$ at the adaptive
rate is at most 0.1 (`NEAR_ZERO_SHIFT`, "approaches 0"), it is **partial** when some other swept
rate with clean validation $\sigma \ge 0.8$ brings the triggered $\sigma$ to 0.1 or below (the
"special value" exists but the adaptive rule misses it), and it **fails** otherwise. On a benign
model the claim is the paper's claim 1b: it holds when the mean absolute gap between the clean
validation and the triggered curve over all rates is at most 0.1 (`BENIGN_CURVE_TOLERANCE`). The
0.1 tolerances are ours, since the paper gives none. At 0.1 a curve that the paper's figure shows
as touching the axis passes, and the median ViT PSBD-RD BadNets model (0.69) fails by far.

### Claim 2, where shifted predictions land

For each split we count, over all (image, pass) predictions that shifted, which class the
perturbed pass predicted (`defenses.scores.shift_target_histogram`) and divide by the number of
shifted predictions. That is the paper's shift intensity pooled over images. We report the share
landing on the target (`validation_target_share`), the most frequent landing class and its share,
at the adaptive rate for every model and at every rate in the JSON. The rule
(`measure.claim_2_verdict`): the claim **holds** when at least 0.8 of the shifted clean validation
predictions land on the target (`ALMOST_ALL_ON_TARGET`, the paper's "almost all"). It is
**partial** when the target is the most frequent landing class but takes under 0.8, or when some
other swept rate at which the clean validation $\sigma$ has reached 0.8 sends at least 0.8 of the
shifts to the target (`max_target_share_where_clean_high`, the same "the adaptive rule missed it"
reading as claim 1). It **fails** otherwise. On a benign model the target is the class the cache probed (class 0, BadNets trigger),
and the claim is 2b: it holds when 1 class takes at least half of the shifts
(`DOMINANT_CLASS_SHARE`, against the paper's "about 60%"). The benign model's share on class 0 is
also the reference a poisoned model's target share should exceed.

### Claim 3, final-layer features with and without the perturbation

`measure.measure_features` loads the checkpoint, builds the PSBD split and takes the first 256
triggered rows with their clean twins (`load_pairs`, the construction the eval set of `CLAUDE.md`
prescribes). It captures the output of the last block (`analysis.features.captured_layers`): the
last `BasicBlock` of ResNet-18, (512, 4, 4) per image at 32 by 32 input, after the final probe and
ReLU as in Fig. 4. For ViT-B/16 the last encoder block's 196 patch tokens are read as 768 maps of
14 by 14, the class token apart. For Swin-S the last block's (7, 7, 768) output is read as 768 maps
of 7 by 7. The conditions are:

| condition | what runs | why |
|---|---|---|
| unperturbed | no probe | the "w/o dropout" panels |
| rd_paper | PSBD-RD at p = 0.91 | Fig. 4's rate |
| rd_adaptive | PSBD-RD at the model's adaptive rate | the rate PSBD-RD deploys |
| tm_adaptive | PSBD-TM at the model's adaptive rate | the rate PSBD-TM deploys, transformers only |

Every perturbed condition runs 3 passes twice. With a shared mask the clean image and its twin are
seeded alike, so every probe draws the same mask for both (Fig. 4's reading). With independent
masks they are seeded apart, which is how PSBD actually scores 2 inputs. The similarity is the
cosine between the flattened maps of a clean image and its twin, after subtracting the mean
unperturbed clean feature of the 256 images:

$$\cos(\mathbf{a}, \mathbf{b}) = \frac{(\mathbf{a}-\boldsymbol\mu)^\top(\mathbf{b}-\boldsymbol\mu)}{\|\mathbf{a}-\boldsymbol\mu\|\,\|\mathbf{b}-\boldsymbol\mu\|}$$

| symbol | meaning |
|---|---|
| $\mathbf{a}$, $\mathbf{b}$ | flattened final-layer maps of 2 images under the same condition |
| $\boldsymbol\mu$ | mean unperturbed clean feature over the 256 clean images of that model |

The mean is subtracted because a transformer's residual stream carries a few channels with a large
constant offset (the massive activations of `experiments/residual_stream_mechanism/`), and the
raw cosine of any 2 ViT images is then above 0.9. The raw cosine is kept in the JSON as
`cosine_pair`. The control the paper's figure lacks is the same cosine between a clean image and
an unrelated clean image (a fixed permutation of the 256) drawn with the same mask relation as the
pair. If 2 unrelated images become as similar as the pair, the convergence is a property of the
mask and says nothing about the backdoor. The readout also records debiased linear CKA between
the 256 clean and 256 triggered features (`analysis.cka.debiased_linear_cka`), the cosine of the
pooled feature the classifier head reads, the paper's red-box share read literally (both maps
non-zero and every activation within 1, `PAPER_MATCH_TOLERANCE`, scale dependent) and the share
of clean and triggered images predicted as the target. The rule (`measure.claim_3_verdict`) reads
the shared mask: the claim **holds** when the pair's cosine rises by at least 0.1
(`CONVERGENCE_GAIN`) to at least 0.9 (`IDENTICAL_COSINE`, "almost identical") and rises at least
0.1 more than the unrelated control's. It is **partial** when the pair converges 0.1 more than
the control but stays below 0.9. It **fails** when the pair does not converge by 0.1, or when the
unrelated control converges within 0.1 as much, since that convergence is the mask's doing.

A benign model is probed with 2 triggers, BadNets (the trigger its caches probed) and Blend, since
on a benign model any convergence cannot come from a backdoor. The models are 1 per attack family
at 10% where it cleared on each architecture, the trigger-conditional TaCT models and GTSRB
BadNets and Blend at 10% on all 3 architectures (`FEATURE_FOLDERS`). GTSRB BadNets 10% is the 1
cell every architecture shares, so its 64 maps per condition are kept for the Fig. 4 figure
(`MAP_FOLDERS`, `results/_experiments/prediction_shift_phenomenon/feature_maps/`).

### Claim 4, MC-Dropout standard deviation

Per image, the population standard deviation over the k = 3 passes of the probability of the
unperturbed argmax class (`measure.per_sample_scores`):

$$u(\mathbf{x}) = \sqrt{\frac{1}{k}\sum_{i=1}^{k}\left(P_c(\mathbf{x};p,\boldsymbol\theta'_i) - \bar{P}_c(\mathbf{x})\right)^2}, \qquad c = \arg\max_{c} P(\mathbf{x};\boldsymbol\theta)$$

| symbol | meaning |
|---|---|
| $P_c(\mathbf{x};p,\boldsymbol\theta'_i)$ | probability of class c on perturbed pass i at rate p |
| $\bar{P}_c(\mathbf{x})$ | its mean over the k passes |
| $c$ | the unperturbed argmax class, the class the caches track |

The paper's pilot says low uncertainty marks a backdoor sample. We score it as a detector, low $u$
means poisoned, and compute its AUROC (`evaluation.metrics.auroc`) between the clean twins and the
captured triggered images at every rate. The comparison is the AUROC of fractional PSU
(`defenses.scores.psu_ratio_from_cache`, the repository's headline statistic) at the adaptive rate.
Because the pilot never says which rate it used, the standard deviation is also given its best
rate, an oracle that reads the backdoor labels. The rule (`measure.claim_4_verdict`): on a model
the standard deviation **fails** when even its best rate loses to PSU at the adaptive rate by more
than 0.05 (`STD_LOSES_BY`), otherwise it **suffices**. The pilot study's finding is 2 halves,
standard deviation works on BadNets and fails on WaNet and Adaptive-Blend, so the architecture
verdict (`measure.claim_4_overall`) holds when both halves hold on at least 2 in 3 of the models
concerned, is partial when 1 half does, and fails when neither does.

### Architecture verdicts

A claim **holds** on an architecture and placement when it holds on at least 2 in 3 of the
poisoned models, **fails** when it fails on at least 2 in 3, and is **partial** in between
(`measure.majority`). The benign models are read separately.

## Models and data

| architecture | poisoned models | benign models | source |
|---|---|---|---|
| ResNet-18 | 2 counted plus 1 reference (`resnet18_gtsrb_badnet_a2o_0_1`, `resnet18_gtsrb_blend_0_1`, `resnet18_cifar10_badnet_a2o_0_1_smoke`) | 0 | the non-evading checkpoints of `experiments/resnet_control/` |
| ViT-B/16 | 54 | 4 (CIFAR-10, CIFAR-100, GTSRB, Tiny) | the successful panel (`scripts.paper._common.clearing_cells`, `successful_2pt`) with both placements cached |
| Swin-S | 65 with PSBD-RD, 63 with PSBD-TM | 3 (CIFAR-100 cached, CIFAR-10 and GTSRB swept on 2026-09-29) | `scripts.paper.tab_swin.swin_cells` at the same `successful_2pt` bar, SIG removed |

Both transformer panels are the successful backdoors of the paper: the attack success rate clears
0.85, the model is not diverged or source-mapped, and its clean accuracy is within 2 points of its
benign reference (`successful_2pt`). A failed attack is left out even when a whole attack family
disappears with it, because a model that is not a backdoor cannot show a backdoor phenomenon. On
ViT this removes `vit_cifar10_sig_0_1`, `vit_cifar10_wanet_0_05` and `vit_gtsrb_wanet_0_1` from
the 57 cells that clear the ASR bar. The Swin panel includes Adaptive-Blend models the ViT panel
lacks, and no Label-Consistent model passes the bar. The cached stage writes the panel it read to
`panel.json`, and the summary reads nothing outside it. The 2 genuine Swin TaCT models
(`swin_cifar10_tact_0_01`, `_0_05`) were swept with PSBD-TM at 1 rate only, 0.5, so they enter
claims 1, 2 and 4 with PSBD-RD alone (`MINIMUM_CURVE_RATES`) and claim 3 reads their PSBD-TM at
0.5 (`FALLBACK_TOKEN_RATE`). The categories group attacks by trigger shape: patch (BadNets, TaCT), blend (Blend, LF), frequency (SIG, quarantined), warp (WaNet), quantization
(BPP) and adaptive blend.

## Deviations from the paper, and why

1. **Test images instead of training images.** The paper scores the poisoned training set: clean
   training images, backdoor training images and clean validation images. This repository's
   protocol scores test-time inputs on the PSBD split, and every cache on disk follows it, so our
   "clean" curves are clean test images and our "backdoor" curve is captured triggered test
   images. Sweeping the training sets would cost a full `cli.sweep` per model. A triggered test
   image is unseen, so restricting to captured images is the closest analogue of fitted poisoned
   training data.
2. **Rate grid.** The paper sweeps 0.1 to 0.9. Our PSBD-RD caches add 0.005 to 0.09, because on a
   transformer the 24 or 48 dropout sites compound and $\sigma$ saturates near p = 0.1. The curves
   are drawn on a log axis for that reason.
3. **Training recipe.** Our ViT-B/16 and Swin-S start from ImageNet weights and train 15 epochs
   of Adam at 1e-4 with input normalization. The ResNet-18 checkpoints start from random weights
   and train 100 epochs of Adam (`experiments/resnet_control/README.md`, "Deviations"), while the
   paper trains ResNet-18 with SGD and no normalization. `resnet18_cifar10_badnet_a2o_0_1_smoke`
   is a smoke run, 15 epochs on 20000 images, cached on 2000 rows per split. Its ASR is 0.998 but
   its clean accuracy is 0.54, far outside the 2-point success bar the transformer panels use
   (the paper's own CIFAR-10 ResNet-18 reaches 0.85). It is not a successful backdoor, so it is
   measured and shown on its own row as a reference and counts in no verdict
   (`REFERENCE_ONLY`). ResNet-18 verdicts therefore rest on 2 GTSRB models.
4. **No benign ResNet-18.** None was trained and this experiment trains nothing, so claims 1b and
   2b rest on the ViT and Swin benign models only.
5. **MC-Dropout uncertainty.** The paper tracks the class of highest mean confidence over training
   epochs, at a rate it does not state. The caches store the probability of the unperturbed argmax
   class of the final checkpoint, so our standard deviation tracks that class, and we read every
   swept rate instead of 1.
6. **Input scaling (Fig. A2).** The SCP variant needs new forward passes on scaled inputs and was
   left out. The pilot's point, that a standard deviation fails where PSU works, is tested by the
   plain variant.
7. **Adaptive rule.** The paper's rule has a second half, the largest gap between the whole
   training set's $\sigma$ and the validation $\sigma$, which needs the poisoned training set. The
   repository's rule keeps the first half (`select_rate_adaptively`).
8. **SIG is quarantined.** `attacks/sig.py` builds amplitude 0.157 since 2026-09-09 while the SIG
   checkpoints of this panel learned 0.1 (`docs/audits/2026-09-29-experiment-audit.md`). Their
   caches were written with the old trigger and a fresh forward pass would build the new one, so
   every SIG model is left out of all 4 claims (`QUARANTINED_ATTACKS`) until its `args.json`
   carries the trained amplitude.

## Sanity checks

`sanity.py` runs 4 gates on 1 model per architecture before any claim 3 run, and on 3 more cells
added after the audit (a global trigger on ResNet-18, CIFAR-100 Blend on ViT, Tiny BPP on Swin).
Records are under `results/_experiments/prediction_shift_phenomenon/sanity/`.

| gate | question | how it is checked | result |
|---|---|---|---|
| 1 | Does our forward path reproduce the checkpoint's clean accuracy and ASR? | unperturbed predictions of `measure.forward_features` on all 12630 GTSRB test images and on the captured-eligible triggered split, against `metrics.json` (`args.json` for ResNet-18, which has no `metrics.json`) | ResNet-18 0.97609 against 0.97609 and ASR 1.0 against 1.0. ViT 0.97253 against 0.97253 and 1.0 against 1.0. Swin 0.98369 against 0.98369 and 1.0 against 1.0. All gaps below 1e-7 |
| 2 | Does our perturbation path give the cached shift ratio? | clean validation $\sigma$ from `measure.plug_condition` at 3 rates per placement, k = 3, against the cache at the same rates, tolerance 0.03 (about 4.5 binomial standard errors on 6000 draws) | largest gap 0.004 on ResNet-18, 0.0035 on ViT, 0.0067 on Swin, and the unperturbed argmax agrees with the cache on 1.000 of the 2000 images |
| 2b | Is the ResNet-18 probe where Li et al. put it? | the paper: "dropout layers are applied after each residual connection in the residual basic block, before the activation function" (`4_method.tex`, "Settings" under "Prediction Shift"). `models.positions._resnet_post_residual_forward` computes `relu(probe(bn2(conv2(...)) + identity))` | 8 probes on 8 basic blocks, every block forward wrapped, none left after unplugging |
| 3 | Is the triggered set the eval set and are pairs the same image? | the backdoor split is `attacks.poisoning.AttackSuccessSet` (which applies `is_eval_poisonable` and `attack_success_label`), every label is the target, no pair's clean label is the target, and the manifest maps every backdoor row to its clean row | passes on all 3. BadNets changes 0.0086 of the pixels of each image, the 3 by 3 patch |
| 4 | Eval mode, no model-owned dropout, hooks removed, masks resampled? | no module in training mode, every model-owned `nn.Dropout` at p = 0 (ViT has 37), forward hooks and wrappers counted before and after, unperturbed predictions identical before and after perturbed runs, final-layer features identical for 1 seed and different for 2 seeds | passes, see the note below |
| 5 | Does bfloat16 change gate 2? | gate 2 on ViT in float32 | largest gap 0.0043, unperturbed argmax agreement 0.999 |

The 3 cells added after the audit pass all 4 gates. The largest shift ratio gap is 0.006 on
ResNet-18 GTSRB Blend, 0.011 on ViT CIFAR-100 Blend and 0.004 on Swin Tiny BPP, the unperturbed
argmax agrees with the cache on 1.000 of the validation images on all 3, and clean accuracy and
ASR match the recorded values to 4 decimals.

The first version of gate 4 compared predictions across seeds and failed on Swin under PSBD-TM at
p = 0.5, where every pass sends every image to the target whatever the mask. A saturated
perturbation makes predictions identical across seeds without the mask being fixed, so the gate
now compares final-layer features, which differ whenever the mask does. The gate reruns are
`sanity/<folder>_state.json`.

Every claim 3 run also holds itself to the cache: its unperturbed predictions on the 256 pairs must
agree with `baseline_clean.pt` and `baseline_backdoor.pt` on at least 0.98 of them
(`BASELINE_AGREEMENT_FLOOR`), or the run stops without writing. That is the check that would have
caught the SIG drift.

## Commands

```bash
source .venv/bin/activate
# claims 1, 2 and 4 from the caches, CPU, about 2 min for 145 models
PYTHONPATH=. python experiments/prediction_shift_phenomenon/measure.py --stage cached
# the sanity gates, then claim 3 and the 2 Swin benign sweeps, resumable, 1 GPU lock per model
bash experiments/prediction_shift_phenomenon/run_gpu.sh
# claim 3 for the best residual placement, merged into the ViT feature records
bash experiments/prediction_shift_phenomenon/run_best_residual_gpu.sh
# the summary every table below and the notebook read
PYTHONPATH=. python experiments/prediction_shift_phenomenon/measure.py --stage summary
PYTHONPATH=. python -m jupyter nbconvert --to notebook --execute --inplace \
    notebooks/prediction-shift-phenomenon.ipynb
```

`run_gpu.sh` takes 1 of the 2 GPU lock slots per model (`scratch/gpu.lock`, `scratch/gpu2.lock`),
caps the process at 14 GB, skips every model whose output exists, starts nothing after 06:30 and
touches `scratch/gpu_done_phenomenon` when it stops. Output is 1 JSON per model under
`results/_experiments/prediction_shift_phenomenon/{cached,features,feature_maps,sanity}/` and
`summary.json` there. GPU logs, including `nvidia-smi` readings before and after every model, are
under `logs/prediction_shift_phenomenon/`.

## Results for claims 1 and 2

Per attack, with the verdict counts of each claim and the median reading at each model's adaptive
rate. The attacks carry the repository's names: `badnet_a2o` BadNets all-to-one, `tact` TaCT,
`blend` Blend, `lf` low frequency, `bpp` BPP, `wanet` WaNet and `adaptive_blend` Adaptive-Blend.

| architecture | placement | attack | models | claim 1 holds / partial / fails | median triggered sigma | claim 2 holds / partial / fails | median target share |
|---|---|---|---|---|---|---|---|
| ResNet-18 | PSBD-RD | badnet_a2o | 1 | 1 / 0 / 0 | 0.00 | 0 / 1 / 0 | 0.59 |
| ResNet-18 | PSBD-RD | blend | 1 | 0 / 0 / 1 | 0.26 | 0 / 0 / 1 | 0.07 |
| ResNet-18 smoke run, reference only | PSBD-RD | badnet_a2o | 1 | 1 / 0 / 0 | 0.02 | 0 / 1 / 0 | 0.75 |
| ViT-B/16 | PSBD-RD | badnet_a2o | 12 | 1 / 0 / 11 | 0.69 | 0 / 1 / 11 | 0.06 |
| ViT-B/16 | PSBD-RD | blend | 12 | 10 / 0 / 2 | 0.06 | 0 / 4 / 8 | 0.08 |
| ViT-B/16 | PSBD-RD | bpp | 12 | 7 / 0 / 5 | 0.08 | 0 / 4 / 8 | 0.09 |
| ViT-B/16 | PSBD-RD | lf | 12 | 6 / 0 / 6 | 0.12 | 0 / 8 / 4 | 0.26 |
| ViT-B/16 | PSBD-RD | tact | 3 | 0 / 0 / 3 | 0.94 | 0 / 0 / 3 | 0.03 |
| ViT-B/16 | PSBD-RD | wanet | 3 | 2 / 0 / 1 | 0.10 | 0 / 2 / 1 | 0.28 |
| ViT-B/16 | PSBD-TM | badnet_a2o | 12 | 10 / 0 / 2 | 0.03 | 3 / 3 / 6 | 0.22 |
| ViT-B/16 | PSBD-TM | blend | 12 | 10 / 0 / 2 | 0.01 | 0 / 0 / 12 | 0.00 |
| ViT-B/16 | PSBD-TM | bpp | 12 | 9 / 0 / 3 | 0.03 | 1 / 2 / 9 | 0.01 |
| ViT-B/16 | PSBD-TM | lf | 12 | 12 / 0 / 0 | 0.00 | 0 / 4 / 8 | 0.17 |
| ViT-B/16 | PSBD-TM | tact | 3 | 0 / 0 / 3 | 0.97 | 0 / 0 / 3 | 0.00 |
| ViT-B/16 | PSBD-TM | wanet | 3 | 2 / 0 / 1 | 0.04 | 0 / 2 / 1 | 0.21 |
| Swin-S | PSBD-RD | adaptive_blend | 7 | 6 / 0 / 1 | 0.00 | 4 / 2 / 1 | 0.97 |
| Swin-S | PSBD-RD | badnet_a2o | 12 | 2 / 3 / 7 | 0.63 | 1 / 6 / 5 | 0.04 |
| Swin-S | PSBD-RD | blend | 12 | 12 / 0 / 0 | 0.00 | 3 / 4 / 5 | 0.53 |
| Swin-S | PSBD-RD | bpp | 12 | 12 / 0 / 0 | 0.00 | 10 / 2 / 0 | 0.94 |
| Swin-S | PSBD-RD | lf | 12 | 5 / 2 / 5 | 0.14 | 0 / 8 / 4 | 0.04 |
| Swin-S | PSBD-RD | tact | 2 | 0 / 0 / 2 | 0.78 | 0 / 0 / 2 | 0.08 |
| Swin-S | PSBD-RD | wanet | 8 | 0 / 8 / 0 | 0.91 | 0 / 8 / 0 | 0.05 |
| Swin-S | PSBD-TM | adaptive_blend | 7 | 6 / 0 / 1 | 0.00 | 6 / 0 / 1 | 0.99 |
| Swin-S | PSBD-TM | badnet_a2o | 12 | 11 / 0 / 1 | 0.00 | 10 / 0 / 2 | 0.94 |
| Swin-S | PSBD-TM | blend | 12 | 12 / 0 / 0 | 0.00 | 0 / 1 / 11 | 0.00 |
| Swin-S | PSBD-TM | bpp | 12 | 11 / 0 / 1 | 0.00 | 1 / 4 / 7 | 0.11 |
| Swin-S | PSBD-TM | lf | 12 | 11 / 0 / 1 | 0.01 | 5 / 2 / 5 | 0.61 |
| Swin-S | PSBD-TM | wanet | 8 | 7 / 0 / 1 | 0.01 | 7 / 0 / 1 | 0.99 |

The benign models, read on the same splits with the BadNets trigger their caches probed. The mean
curve gap is the claim 1b reading, the top class and its share the claim 2b reading.

| model | placement | adaptive p | clean sigma | triggered sigma | mean curve gap | top class | top class share | share on class 0 |
|---|---|---|---|---|---|---|---|---|
| swin_cifar100_benign | PSBD-TM | 0.50 | 0.83 | 0.84 | 0.00 | 28 | 0.45 | 0.00 |
| swin_cifar100_benign | PSBD-RD | 0.10 | 0.96 | 0.96 | 0.00 | 99 | 0.26 | 0.00 |
| swin_cifar10_benign | PSBD-TM | 0.60 | 0.89 | 0.87 | 0.01 | 2 | 0.98 | 0.00 |
| swin_cifar10_benign | PSBD-RD | 0.09 | 0.82 | 0.81 | 0.01 | 3 | 0.89 | 0.00 |
| swin_gtsrb_benign | PSBD-TM | 0.50 | 0.94 | 0.94 | 0.01 | 33 | 0.63 | 0.00 |
| swin_gtsrb_benign | PSBD-RD | 0.09 | 0.91 | 0.91 | 0.00 | 40 | 0.39 | 0.00 |
| vit_cifar100_benign | PSBD-TM | 0.50 | 0.92 | 0.91 | 0.00 | 47 | 0.45 | 0.00 |
| vit_cifar100_benign | PSBD-RD | 0.07 | 0.95 | 0.95 | 0.00 | 47 | 0.11 | 0.00 |
| vit_cifar10_benign | PSBD-TM | 0.70 | 0.82 | 0.80 | 0.01 | 5 | 0.53 | 0.00 |
| vit_cifar10_benign | PSBD-RD | 0.09 | 0.84 | 0.86 | 0.01 | 0 | 0.23 | 0.23 |
| vit_gtsrb_benign | PSBD-TM | 0.50 | 0.87 | 0.87 | 0.00 | 13 | 0.99 | 0.00 |
| vit_gtsrb_benign | PSBD-RD | 0.09 | 0.90 | 0.91 | 0.00 | 1 | 0.20 | 0.00 |
| vit_tiny_benign | PSBD-TM | 0.50 | 0.83 | 0.83 | 0.00 | 94 | 0.31 | 0.00 |
| vit_tiny_benign | PSBD-RD | 0.05 | 0.82 | 0.81 | 0.00 | 131 | 0.08 | 0.00 |

**Claim 1 on ResNet-18 is partial**, 1 of 2 models. GTSRB BadNets keeps the triggered shift
ratio at 0.003 at its adaptive rate 0.5 while the clean curve reaches 0.95, the shape of the
paper's Fig. 3. GTSRB Blend fails: its triggered ratio is 0.26 at every rate where the clean curve
has saturated. The CIFAR-10 smoke run, reference only, shows the paper's shape too (0.022 against
0.87).

**Claim 1 on ViT-B/16 is partial under PSBD-RD and holds under PSBD-TM.** Under PSBD-RD the clean
curves saturate near p = 0.09 and the triggered curves follow about 1 grid step later, so the
window the paper describes is narrow, and 28 of 54 models miss it (median triggered ratio 0.12).
The failures are BadNets on 11 of 12 models (median 0.69), TaCT on 3 of 3 (0.94) and 14 Blend,
LF, BPP and WaNet models between 0.12 and 0.84, most of them on CIFAR-10 and CIFAR-100. Under
PSBD-TM 43 of 54 hold with a median of 0.016. The 11 failures are the 3 trigger-conditional TaCT
models (1.00, 0.97 and 0.42), 7 CIFAR-10, GTSRB and Tiny models of BadNets, Blend and BPP at 0.12
to 0.36, and `vit_cifar10_wanet_0_1` at 0.82.

**Claim 1 on Swin-S is partial under PSBD-RD and holds under PSBD-TM.** Under PSBD-RD 37 of 65
hold, 13 are partial and 15 fail. Every Swin WaNet model is partial: some rate with a saturated
clean curve brings its triggered ratio to 0, and the adaptive rule picks a rate where the median
is still 0.91. BadNets fails or is partial on 10 of 12 and TaCT fails on 2 of 2. Under PSBD-TM 58
of 63 hold with a median of 0.002.

**The benign half of claim 1 holds** on all 7 benign transformers. The mean gap between the clean
and the triggered curve is at most 0.012.

**Why claim 1 fails where it fails, a hypothesis here.** The isolating test (masking only the
trigger's tokens, with random tokens as the control) is `experiments/why_token_masking_works/`,
not this experiment. The failures follow trigger shape and site, not dataset. Dropout on the
residual stream fails on patch triggers on both transformers, and that experiment found the few
tokens that carry a patch trigger are corrupted by stream dropout in every block, so the triggered
prediction breaks as easily as a clean one, while a token masked at the attention input keeps its
entry in the stream. On `vit_cifar10_tact_0_05` under PSBD-TM the triggered predictions move with
the clean ones (0.97 of passes), yet PSU still separates the 2 sets (AUROC 0.97), because the
target keeps a mean probability of 0.05 under the perturbation while a clean image's own class
keeps 0.007. On such models the argmax shift the paper describes and the confidence drop PSBD
scores come apart.

**Claim 2 on ResNet-18 is partial.** GTSRB BadNets sends most shifted clean predictions to the
target, 0.59 at the adaptive rate and 0.91 at p = 0.7, the paper's rate (the smoke run, reference
only, 0.75 and 0.89). Its triggered shifts land on a single non-target class (class 33 on GTSRB, 0.33 of them), as the
paper's BadNets panel shows for class 3. GTSRB Blend fails with 0.07 on the target, and class 33
leads with 0.17.

**Claim 2 on ViT-B/16 is partial under PSBD-RD and fails under PSBD-TM.** Under PSBD-RD the
target leads on 19 of 54 models and takes 0.8 of the shifts on none (median 0.09). Under PSBD-TM
the target leads on 15 of 54 and takes 0.8 on 4 (median 0.02). All 12 Blend models fail under
PSBD-TM, with a median of 0.00 on the target.

**Claim 2 on Swin-S is partial under both placements.** Under PSBD-TM BadNets (10 of 12 hold,
median 0.94), WaNet (7 of 8, 0.99) and Adaptive-Blend (6 of 7, 0.99) send almost every shift to
the target, and Blend (0 of 12, 0.00) and BPP (1 of 12) do not. Under PSBD-RD BPP (10 of 12, 0.94)
and Adaptive-Blend (4 of 7) hold and BadNets does not (median 0.04).

**The universality half of claim 2 depends on the model.** A benign transformer under PSBD-TM
sends 0.99 of its GTSRB shifts (ViT) and 0.98 of its CIFAR-10 shifts (Swin) to 1 class, more than
the paper's 0.6, while on CIFAR-100 and Tiny the largest class takes 0.31 to 0.45. Under PSBD-RD
only the benign Swin CIFAR-10 model concentrates (0.89 on class 3).

**Why claim 2 fails where it fails, a hypothesis.** No experiment here isolates it. A perturbed
transformer collapses its clean predictions onto 1 class, and whether that class is the target
depends on the attack family and the architecture. The benign models show that such a collapse
exists without any backdoor, so on a model whose shifts go to a non-target class, the paper's
reading that dropout exposes a bias toward the target does not describe what is seen. The target
share also moves strongly with the rate: on Swin GTSRB BadNets under PSBD-RD it is 0.28 at
p = 0.09, 1.0 at p = 0.2 and 0.05 at p = 0.9.

## Results for claim 4

| architecture | placement | attack | models | std fails | median std AUROC, adaptive p | median std AUROC, best p | median PSU ratio AUROC |
|---|---|---|---|---|---|---|---|
| ResNet-18 | PSBD-RD | badnet_a2o | 1 | 0 of 1 | 0.71 | 0.98 | 1.00 |
| ResNet-18 | PSBD-RD | blend | 1 | 1 of 1 | 0.09 | 0.90 | 0.97 |
| ResNet-18 smoke run, reference only | PSBD-RD | badnet_a2o | 1 | 0 of 1 | 0.42 | 0.96 | 0.96 |
| ViT-B/16 | PSBD-RD | badnet_a2o | 12 | 2 of 12 | 0.22 | 0.80 | 0.78 |
| ViT-B/16 | PSBD-RD | blend | 12 | 9 of 12 | 0.05 | 0.84 | 1.00 |
| ViT-B/16 | PSBD-RD | bpp | 12 | 12 of 12 | 0.07 | 0.83 | 0.99 |
| ViT-B/16 | PSBD-RD | lf | 12 | 12 of 12 | 0.06 | 0.81 | 0.99 |
| ViT-B/16 | PSBD-RD | tact | 3 | 1 of 3 | 0.43 | 0.81 | 0.46 |
| ViT-B/16 | PSBD-RD | wanet | 3 | 3 of 3 | 0.06 | 0.74 | 0.99 |
| ViT-B/16 | PSBD-TM | badnet_a2o | 12 | 10 of 12 | 0.12 | 0.87 | 0.99 |
| ViT-B/16 | PSBD-TM | blend | 12 | 2 of 12 | 0.60 | 0.99 | 1.00 |
| ViT-B/16 | PSBD-TM | bpp | 12 | 3 of 12 | 0.61 | 0.96 | 0.99 |
| ViT-B/16 | PSBD-TM | lf | 12 | 4 of 12 | 0.67 | 0.96 | 1.00 |
| ViT-B/16 | PSBD-TM | tact | 3 | 2 of 3 | 0.04 | 0.78 | 0.97 |
| ViT-B/16 | PSBD-TM | wanet | 3 | 3 of 3 | 0.30 | 0.83 | 0.97 |
| Swin-S | PSBD-RD | adaptive_blend | 7 | 6 of 7 | 0.64 | 0.85 | 1.00 |
| Swin-S | PSBD-RD | badnet_a2o | 12 | 1 of 12 | 0.18 | 0.92 | 0.78 |
| Swin-S | PSBD-RD | blend | 12 | 3 of 12 | 0.66 | 0.99 | 1.00 |
| Swin-S | PSBD-RD | bpp | 12 | 3 of 12 | 0.69 | 0.97 | 1.00 |
| Swin-S | PSBD-RD | lf | 12 | 8 of 12 | 0.24 | 0.84 | 0.96 |
| Swin-S | PSBD-RD | tact | 2 | 2 of 2 | 0.02 | 0.74 | 1.00 |
| Swin-S | PSBD-RD | wanet | 8 | 3 of 8 | 0.35 | 0.82 | 0.64 |
| Swin-S | PSBD-TM | adaptive_blend | 7 | 3 of 7 | 0.25 | 0.94 | 1.00 |
| Swin-S | PSBD-TM | badnet_a2o | 12 | 0 of 12 | 0.09 | 0.99 | 1.00 |
| Swin-S | PSBD-TM | blend | 12 | 3 of 12 | 0.12 | 1.00 | 1.00 |
| Swin-S | PSBD-TM | bpp | 12 | 1 of 12 | 0.69 | 0.99 | 1.00 |
| Swin-S | PSBD-TM | lf | 12 | 3 of 12 | 0.19 | 0.96 | 1.00 |
| Swin-S | PSBD-TM | wanet | 8 | 8 of 8 | 0.23 | 0.45 | 0.98 |

**Claim 4 holds on ViT-B/16 under PSBD-RD and on Swin-S under PSBD-TM.** It is partial on ViT-B/16
under PSBD-TM and on Swin-S under PSBD-RD, and only its BadNets half can be read on ResNet-18,
where it holds. The standard deviation fails on every WaNet model of both transformers except 5 of
8 Swin models under PSBD-RD. At its best rate its median AUROC on WaNet is 0.74 and 0.83 on ViT
(PSBD-RD and PSBD-TM) and 0.45 on Swin under PSBD-TM, against PSU medians of 0.97 to 0.99. On
BadNets it suffices on ResNet-18 GTSRB (0.98 against 1.00) and on Swin-S under PSBD-TM (0.99 against
1.00). The partial verdicts come from the other half. Under PSBD-TM on ViT the standard deviation
also loses on 10 of 12 BadNets models (median 0.87 against 0.99), and under PSBD-RD on Swin
Adaptive-Blend fails on 6 of 7 but WaNet on only 3 of 8, because PSU itself is weak on Swin WaNet
under PSBD-RD (median 0.64).

**Why the standard deviation fails.** The rate sweep of the standard deviation (notebook step 4,
first figure) is the isolating test, since it changes only the rate and scores the same passes 2
ways. The standard deviation is not monotone in the rate. A prediction that flips on every pass
has as little spread as one that never flips, so it peaks where predictions are about to flip and
falls once they have. At the adaptive rate the clean images are past their peak, and the median
standard deviation AUROC there is 0.10 to 0.47, mostly below chance: the clean images look more
certain than the triggered ones. PSU reads the drop in confidence, which keeps growing with the
rate.

## Results for claim 3

Medians over the claim 3 readings (1 per model, trigger probe and condition, 256 pairs, 3 passes
each). "Pair" is a clean image and its triggered twin, "unrelated" is a clean image and a
different clean image under the same mask relation, "clean toward triggered" is the cosine
between a clean image's feature and its twin's unperturbed feature. Every cosine is centered on
the mean unperturbed clean feature.

| architecture | models | condition | readings | pair, off | pair, shared mask | unrelated, shared mask | pair, independent | unrelated, independent | clean toward triggered, off | clean toward triggered, on | holds / partial / fails |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-18 | poisoned | PSBD-RD adaptive p | 2 | 0.10 | 0.78 | 0.69 | 0.05 | 0.05 | 0.10 | 0.05 | 0 / 0 / 2 |
| ResNet-18 | poisoned | PSBD-RD p = 0.91 | 2 | 0.10 | 0.93 | 0.76 | 0.00 | 0.00 | 0.10 | -0.00 | 1 / 0 / 1 |
| ResNet-18 smoke run, reference only | poisoned | PSBD-RD adaptive p | 1 | 0.46 | 0.95 | 0.66 | 0.24 | 0.23 | 0.46 | 0.12 | 0 / 0 / 1 |
| ResNet-18 smoke run, reference only | poisoned | PSBD-RD p = 0.91 | 1 | 0.46 | 1.00 | 0.80 | 0.02 | 0.02 | 0.46 | 0.01 | 0 / 0 / 1 |
| ViT-B/16 | benign | PSBD-RD adaptive p | 4 | 0.83 | 0.95 | 0.62 | 0.19 | 0.14 | 0.83 | 0.16 | 0 / 0 / 4 |
| ViT-B/16 | benign | PSBD-RD p = 0.91 | 4 | 0.83 | 1.00 | 1.00 | 0.00 | 0.00 | 0.83 | 0.00 | 0 / 0 / 4 |
| ViT-B/16 | benign | PSBD-TM adaptive p | 4 | 0.83 | 0.92 | 0.47 | 0.74 | 0.38 | 0.83 | 0.31 | 0 / 0 / 4 |
| ViT-B/16 | poisoned | PSBD-RD adaptive p | 10 | 0.67 | 0.97 | 0.50 | 0.16 | 0.12 | 0.67 | 0.15 | 0 / 0 / 10 |
| ViT-B/16 | poisoned | PSBD-RD p = 0.91 | 10 | 0.67 | 1.00 | 1.00 | 0.00 | 0.00 | 0.67 | 0.00 | 0 / 0 / 10 |
| ViT-B/16 | poisoned | PSBD-TM adaptive p | 10 | 0.67 | 0.88 | 0.42 | 0.68 | 0.35 | 0.67 | 0.33 | 0 / 0 / 10 |
| Swin-S | benign | PSBD-RD adaptive p | 4 | 0.88 | 0.97 | 0.84 | 0.39 | 0.37 | 0.88 | 0.05 | 0 / 0 / 4 |
| Swin-S | benign | PSBD-RD p = 0.91 | 4 | 0.88 | 1.00 | 1.00 | 0.00 | 0.00 | 0.88 | -0.00 | 0 / 0 / 4 |
| Swin-S | benign | PSBD-TM adaptive p | 4 | 0.88 | 0.98 | 0.84 | 0.88 | 0.80 | 0.88 | 0.03 | 0 / 0 / 4 |
| Swin-S | poisoned | PSBD-RD adaptive p | 9 | 0.12 | 0.92 | 0.84 | 0.57 | 0.52 | 0.12 | 0.32 | 0 / 1 / 8 |
| Swin-S | poisoned | PSBD-RD p = 0.91 | 9 | 0.12 | 1.00 | 1.00 | 0.00 | 0.00 | 0.12 | 0.02 | 0 / 0 / 9 |
| Swin-S | poisoned | PSBD-TM adaptive p | 9 | 0.12 | 0.82 | 0.75 | 0.78 | 0.71 | 0.12 | 0.35 | 0 / 1 / 8 |

**The observation reproduces and its reading fails the control, on all 3 architectures.** With
1 dropout mask shared by the clean image and its twin, as in Fig. 4, the pair's final-layer
features do become almost identical: the median cosine at p = 0.91 is 0.93 on ResNet-18 and 1.00
on both transformers, from 0.10, 0.67 and 0.12 without dropout. But 2 unrelated clean images under
the same shared mask converge too, to 0.76 on ResNet-18 and to 1.00 on both transformers, and they
converge on the benign models as much as on the poisoned ones. The convergence is the mask: at
p = 0.91 the mask decides which entries are 0, so any 2 inputs that share it look alike.
Only 1 of 61 poisoned readings holds by the rule (ResNet-18 GTSRB BadNets at p = 0.91, pair 0.99
against unrelated 0.78). With independent masks, which is how PSBD actually scores 2 inputs, the pair is
no more similar than 2 unrelated images (medians 0.00 at p = 0.91 on every architecture, and at
the adaptive rate 0.05 against 0.05 on ResNet-18, 0.16 against 0.12 on ViT under PSBD-RD).

**Neuron bias as a drift toward the backdoor's features appears on Swin-S only.** If dropout
pushed clean images onto the backdoor path, the perturbed clean feature should move toward its
triggered twin's unperturbed feature. That cosine falls under the perturbation on ResNet-18 (0.10
to 0.05) and ViT-B/16 (0.67 to 0.15 under PSBD-RD, 0.33 under PSBD-TM) and on every benign model.
On poisoned Swin-S models it rises, from 0.12 to 0.32 under PSBD-RD and to 0.35 under PSBD-TM,
while on the benign Swin models it falls from 0.88 to 0.05 and 0.03. The benign models are the
control, so this is the 1 reading in this experiment consistent with the paper's account, and it
is a partial move, not identity.

**What claim 3 does not show.** 9 to 10 models per transformer and 2 ResNet-18 models plus the smoke run, 1 per
attack family, mostly CIFAR-10 and GTSRB at 10%. The final block's patch tokens stand in for
ResNet-18's top layer. The pooled feature the classifier head reads is in the JSON
(`cosine_pooled_*`) and tells the same story.

## The best residual placement and neuron bias

**Question.** The best placement that masks no tokens is `pre_residual_blocks_5_8`, dropout
before both residual adds in blocks 5 to 8 only (`\BestResidualName` in `paper/headline.tex`). It
trails PSBD-TM by `\BestResidualGain` on the headline panel. Does it detect through neuron bias,
the mechanism the paper proposes? If it did, its shifted clean predictions would land on the target
(claim 2), more often than on a benign model of the same dataset, and its perturbed clean features
would move toward their triggered twins (claim 3).

**Method.** The same 3 measurements, changing only the placement: claims 1 and 2 from its stage-1
caches on the 54 successful ViT-B/16 models and the 4 ViT benign models, and claim 3 on the 10 ViT
feature models and 2 benign ones at its adaptive rate, added to the existing records with
`--add-conditions br_adaptive` (`run_best_residual_gpu.sh`, GPU 21:30 to 00:59 on 2026-09-29,
mostly waiting for a lock slot). The control for claim 2 is `target_share_over_benign`, the
target share minus the share the benign model of the same dataset sends to that class. The
control for claim 3 is the benign models and the unrelated image pair.

| attack | models | claim 1 holds | median triggered sigma | claim 2 holds | target leads | median target share | median excess over benign | median PSU ratio AUROC |
|---|---|---|---|---|---|---|---|---|
| badnet_a2o | 12 | 5 | 0.22 | 0 | 3 | 0.05 | 0.01 | 0.98 |
| blend | 12 | 8 | 0.02 | 0 | 1 | 0.04 | 0.00 | 1.00 |
| bpp | 12 | 7 | 0.08 | 0 | 4 | 0.08 | 0.03 | 0.99 |
| lf | 12 | 8 | 0.03 | 0 | 2 | 0.08 | 0.08 | 0.99 |
| tact | 3 | 0 | 0.94 | 0 | 0 | 0.01 | -0.12 | 0.77 |
| wanet | 3 | 3 | 0.01 | 0 | 1 | 0.05 | 0.05 | 1.00 |

**Result.** The placement separates (median PSU ratio AUROC 0.99) and claim 1 is partial on it
(31 of 54 models keep the triggered shift ratio at 0.1 or below). Claim 2 holds on none of the
54 (43 fail, 11 partial): the median target share is 0.06, the target leads on 11 models, and the median excess over the benign
control is 0.02.
A minority does show a target bias: 7 of 54 models send at least 0.3 more of their shifts to the
target than the benign control (`vit_cifar100_blend_0_1`, `vit_cifar100_bpp_0_01`,
`vit_cifar10_bpp_0_01`, `vit_cifar10_wanet_0_1`, `vit_gtsrb_badnet_a2o_0_1`, `vit_gtsrb_lf_0_1`
and `vit_tiny_bpp_0_01`). Claim 3 fails on all 10 poisoned readings. Under a shared mask the pair's cosine
rises from 0.67 to 0.97 but an unrelated pair rises to 0.70 from about 0. Under independent masks
the pair reads 0.26 against 0.17 for an unrelated pair. The perturbed clean feature moves away from
its triggered twin, from 0.67 to 0.19, as it does on the benign models (0.83 to 0.17).

**Verdict: neuron bias does not explain the best residual placement, beyond a target bias on 7 of
54 models.** It separates clean from
triggered images without sending clean predictions to the target beyond what a benign model does,
and without moving clean features toward the backdoor's. What remains is claim 1 itself: the
triggered prediction survives the perturbation more often than the clean one. Why it survives is
not isolated here.

## Verdicts

A claim holds on an architecture and placement when it holds on at least 2 in 3 of the poisoned
models, fails when it fails on at least 2 in 3, and is partial in between.

| architecture | placement | poisoned models | claim 1 shift ratio | claim 2 target | claim 3 features | claim 4 std |
|---|---|---|---|---|---|---|
| ResNet-18 | PSBD-RD | 2 | partial (1 of 2) | partial (0 hold, 1 partial) | fails, the convergence is the mask's (1 of 4 readings hold) | BadNets half holds |
| ViT-B/16 | PSBD-RD | 54 | partial (26 hold) | partial (0 hold, 19 partial) | fails, the convergence is the mask's (0 of 20) | holds |
| ViT-B/16 | PSBD-TM | 54 | holds (43) | fails (4 hold, 39 fail) | fails (0 of 10) | partial |
| ViT-B/16 | best residual, blocks 5 to 8 | 54 | partial (31 hold) | fails (0 hold) | fails (0 of 10) | holds |
| Swin-S | PSBD-RD | 65 | partial (37 hold, 13 partial) | partial (18 hold, 30 partial) | fails (0 of 18), drift toward backdoor features | partial |
| Swin-S | PSBD-TM | 63 | holds (58) | partial (29 hold, 27 fail) | fails (0 of 9), drift toward backdoor features | holds |

The benign controls: claim 1b holds on all 7 benign transformers, and claim 2b (1 class takes at
least half the shifts) holds on 5 of 14 benign readings.

## Done and left

Done on 2026-09-29: the cached stage over 142 models (3 ResNet-18 of which 1 reference only, 54 ViT-B/16 and 65 Swin-S
poisoned, 7 benign transformers, 2 minutes on CPU), the sanity gates on 6 models, claim 3 on 25
models (GPU 18:17 to 21:10, 14 GB cap, peak allocation 1.3 GB per process, most of the wall time
spent waiting for a lock slot), and the 2 Swin benign sweeps (`results/swin_cifar10_benign/psbd/`,
`results/swin_gtsrb_benign/psbd/`, written by `cli.sweep` into directories that did not exist).
`scratch/gpu_done_phenomenon` was touched at 21:09.

Left: SIG on every architecture until the amplitude overrides are verified against the caches,
a benign ResNet-18 (needs training), the SCP variant of Fig. A2, and the training-set version of
claims 1, 2 and 4 (needs `cli.sweep` on the poisoned training sets).
