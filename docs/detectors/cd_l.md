# CD-L, Cognitive Distillation on logits

Cognitive Distillation learns, for 1 input at a time, the smallest mask over the input's pixels under which the model still produces the same logits, and scores the input by the L1 norm of that mask. This page explains the idea, gives the paper's equations, describes the paper's setting and the authors' released code, walks through the port in `detectors/cd_l.py`, lists every deviation with its reason and ends with the results on the panel. Terms such as the shared split, the panel, PSBD-TM, PSBD-RD, AUROC and TPR at a false-positive budget are defined once in `README.md` in this directory.

## The idea in plain words

Ask of an image: which of its pixels does the model actually need to give the answer it gives? CD-L answers by optimization. It keeps a mask of the same height and width as the image, with 1 meaning "keep this pixel" and 0 meaning "replace it with random color", and runs 100 gradient steps that shrink the mask while penalizing any change in the model's logits. On a clean image the logits rest on the object, which is spread over many pixels, so the mask has to stay large. On a triggered image the backdoor dominates the logits, and the trigger's own few pixels are all the model needs to reproduce them, so the mask collapses onto the trigger and its total area is small. A small mask norm is evidence of a trigger.

The method is white-box and gradient-based: each step differentiates the logits with respect to the input. It needs no clean data for the score, because the objective compares the model's output on the distilled input with its own output on the original. It is also the most expensive detector in the registry.

## The original method

Huang et al., "Distilling Cognitive Backdoor Patterns within an Image", ICLR 2023, arXiv:2301.10908 (v4 read). The objective is Eq. (1) and the distilled input Eq. (2), both in Section 3.1. The detection rule is Eq. (4) in Section 3.2 and the optimizer settings are in Appendix B.3.

$$
\begin{aligned}
&\arg\min_{\boldsymbol{m}} \; \left\| f_{\theta}(\boldsymbol{x}) - f_{\theta}(\boldsymbol{x}_{cp}) \right\|_1 + \alpha \left\| \boldsymbol{m} \right\|_1 + \beta \, TV(\boldsymbol{m}) && \text{(1)} \\
&\boldsymbol{x}_{cp} = \boldsymbol{x} \odot \boldsymbol{m} + (1 - \boldsymbol{m}) \odot \delta && \text{(2)} \\
&g(\boldsymbol{x}) = \begin{cases} 1 & \text{if } \left\| \boldsymbol{m} \right\|_1 \le t \\ 0 & \text{if } \left\| \boldsymbol{m} \right\|_1 > t \end{cases} && \text{(4)}
\end{aligned}
$$

| symbol | meaning |
|---|---|
| $f_{\theta}$ | the model's logit map (the L in CD-L), or its last convolutional features in the CD-F variant the port does not implement |
| $\boldsymbol{x}$ | the input image in $[0, 1]^{w \times h \times c}$ |
| $\boldsymbol{m}$ | the learnable mask in $[0, 1]^{w \times h}$, shared by the color channels |
| $\delta$ | a uniform random color in $[0, 1]^{c}$, 1 value per channel, redrawn every step |
| $\odot$ | elementwise product, broadcast over the channels |
| $\boldsymbol{x}_{cp}$ | the distilled "cognitive pattern", same shape as $\boldsymbol{x}$ |
| $\alpha$ | the weight of the L1 sparsity term |
| $\beta$ | the weight of the total-variation smoothness term |
| $TV$ | the total variation of the mask, not defined in the paper, squared adjacent differences in the code |
| $t$ | the detection threshold |
| $g(\boldsymbol{x})$ | the decision, 1 meaning backdoor |

The mask is parameterized as $\boldsymbol{m} = (\tanh(\theta_m) + 1) / 2$, the paper's "scaled tanh", so the optimizer works on an unconstrained $\theta_m$ and the mask never leaves $[0, 1]$. The fill $\delta$ is random so that a pixel the mask removed cannot be confused with a pixel that happens to be black. The descriptive form renames without rederiving.

$$
\begin{aligned}
\text{objective} &= \text{mean over classes of } \big| \text{logits}(\text{image}) - \text{logits}(\text{distilled}) \big| + \text{l1\_weight} \cdot \textstyle\sum \text{mask} + \text{tv\_weight} \cdot TV(\text{mask}) \\
\text{distilled} &= \text{image} \cdot \text{mask} + (1 - \text{mask}) \cdot \text{random\_fill} \\
\text{cd\_l\_score} &= \textstyle\sum \text{mask after 100 steps}
\end{aligned}
$$

## The original paper's setting

The paper uses the method both as a filter over a poisoned training set and as a test-time input detector, and the port covers the test-time role only, the role every detector here plays. Appendix B.3 gives Adam with learning rate 0.1 and both moment decays at 0.1, 100 steps, $\alpha = 0.01$ for CD-L on CIFAR-10 and GTSRB at 32 pixels, $\alpha = 0.001$ on its ImageNet subset and $\beta = 10$. The paper's Figure 10 reports detection unchanged across $\beta$ from 1 to 100. The threshold is $t = \mu - \gamma\sigma$ over the mask norms of 1% of the clean training set with $\gamma = 1$. The paper's models are ConvNets, and no transformer is evaluated.

## The reference implementation

The authors' code is https://github.com/HanxunH/CognitiveDistillation at commit `1d35393`, vendored as `third_party/CognitiveDistillation` when the port was written. The method is the 63-line class in `detection/cognitive_distillation.py`. It initializes the mask parameter to ones, so the starting mask is $(\tanh 1 + 1)/2 = 0.8808$, builds Adam over that 1 tensor with learning rate 0.1 and betas (0.1, 0.1), computes the reference logits once and detaches them, and then for 100 steps draws `torch.rand(b, c, 1, 1)` as the fill, forms `images * mask + (1 - mask) * fill`, takes the L1 distance between the logit tensors averaged over classes, adds `gamma` times the mask's L1 norm and `beta` times a total variation that sums squared vertical and horizontal differences and divides by the mask's element count, averages over the batch and steps. With `norm_only=True` it returns the L1 norm of the final mask per image.

The code and the paper disagree in 2 places. The code's default is `beta=1.0` where Appendix B.3 says 10, and the code names the L1 weight `gamma`, a letter the paper reserves for the threshold coefficient. The driver `extract.py` freezes the model and feeds $[0, 1]$ images directly, so the class's missing preprocessor on the distilled pass is harmless there, and its line 116 binds the label tensor into the `preprocessor` slot, so the CD branch as checked out raises on its first batch. The backdoor-toolbox copy at `third_party/backdoor-toolbox/other_defenses_tool_box/CD.py` blends the $[0, 1]$ fill into normalized tensors (line 171) and computes its threshold with the L1 weight in place of $\gamma$ (line 195). Both repositories are pinned in `third_party.lock`, and the line numbers were read from the pinned checkouts on 2026-09-29.

## The port step by step

1. `_build_cd_l` in `detectors/__init__.py` fits nothing and returns a closure over `cd_l_scores(model, loader, device, mean, std, use_bfloat16, seed, context.cd_l_steps)`. `context.cd_l_steps` defaults to `DEFAULT_NUM_STEPS` (100). It is overridable only so a smoke run can pay less.
2. `cd_l_scores` seeds with `seed_everything` so the fill sequence is fixed. It then calls `distill_masks` once per batch.
3. `distill_masks` takes the batch as the loader serves it, normalized, shape (batch, channels, height, width) at the dataset's native 32 or 64 pixels. It denormalizes to pixels in $[0, 1]$ with `normalization_buffers`, creates the mask parameter of shape (batch, 1, height, width) filled with `MASK_PARAMETER_INIT` (1.0) in float32 with gradients, and builds `torch.optim.Adam` over it with learning rate 0.1 and betas (0.1, 0.1).
4. Inside `torch.enable_grad()` and `defenses.inference.frozen_parameters(model)`, which sets `requires_grad=False` on every model parameter for the duration and restores the flags afterwards, it computes the reference logits with `forward_logits` on the served tensor and detaches them, shape (batch, num_classes).
5. Each of the 100 steps computes `effective_mask` (the scaled tanh), draws the fill `torch.rand(batch, channels, 1, 1)` on the model's device, forms the distilled pixels by Eq. (2), renormalizes and runs `forward_logits` under the shared autocast policy. The objective is the per-image mean absolute logit gap plus `DEFAULT_L1_WEIGHT` (0.01) times the mask's L1 norm plus `DEFAULT_TV_WEIGHT` (1.0) times `total_variation(mask)`, averaged over the batch. `torch.autograd.grad(objective, mask_parameter)` gives the gradient with respect to the mask alone, which is assigned to `.grad` before `optimizer.step()`.
6. After the loop the final effective mask, shape (batch, 1, height, width), goes to `mask_norms`, which calls `torch.linalg.vector_norm(masks, ord=1)` per image, the kernel `torch.norm(p=1)` in the reference dispatches to, so the reduction order matches bit for bit.
7. The norms are concatenated into float32 scores of shape (N,). They are checked to be finite and returned unnegated.

## Deviations and why

1. **TV weight 1.0 instead of 10.** The paper says 10, the released code defaults to 1.0 and the paper's own ablation finds detection insensitive to it. The port takes the released value because the bit-level cross-check targets the released class. A reader comparing with the paper's tables should know the smoothing here is 10 times weaker.
2. **Normalization on both passes.** The released class sends the reference pass through a preprocessor and the distilled pass through none, which is harmless when the model consumes $[0, 1]$ input. The models here expect normalized input, so the port forms Eq. (2) in pixel space and renormalizes before every forward pass. Without this the 2 passes would see the model at different input scales and the logit gap would measure the normalization rather than the mask.
3. **Mask at the native resolution.** Every backbone here is `Sequential(Resize((224, 224)), network)`. The port optimizes the mask at 32 or 64 pixels and lets the wrapper upsample the distilled input. At full coverage the L1 term is then $\alpha h w$, about 10 at 32 pixels and 41 at 64, the paper's scale. A mask at 224 would make it about 502 and swamp the logit term.
4. **Fill drawn on the model's device.** The reference draws on the CPU and moves the tensor. On a GPU the port's fill comes from the CUDA generator, so a GPU run reproduces itself but not the CPU fill sequence. On the CPU the sequences coincide, which is what the bit-level test relies on.
5. **Gradient to the mask only.** The released class relies on its caller freezing the model and accumulates weight gradients otherwise. The port freezes the model itself and takes the gradient of the objective with respect to the mask parameter alone, so no model parameter ever holds a gradient and the backward pass skips the weight-gradient graph. The test file checks both.
6. **Mixed precision.** The released code is float32. The port runs the forward and backward under the shared autocast policy, bfloat16 on a GPU, with the mask, the Adam state and the objective in float32. The 2026-09-10 smoke (`docs/runs/2026-09-10-detector-smoke.md`) compared bfloat16 against float32 on a GTSRB BadNet model and found the same AUROC at a fraction of the cost, so `PRECISION_POLICY["cd_l"]` stays `autocast`.
7. **Threshold rule.** The paper thresholds at $\mu - \sigma$ of clean mask norms from 1% of the training set. The port hands the raw norms to `defenses.decision.detection_report`, which thresholds at a quantile of the validation scores. AUROC does not depend on this choice.
8. **$\alpha$ on Tiny ImageNet.** The paper gives $\alpha = 0.01$ at 32 pixels and 0.001 on its ImageNet subset, with no 64-pixel setting. The port keeps 0.01 on every dataset so 1 constant covers the panel, which puts Tiny ImageNet's full-coverage L1 term at 4 times the CIFAR value, a setting the paper never ran.

## Hyperparameters and where they come from

| symbol | paper | this port | constant | source of the port's value |
|---|---|---|---|---|
| $\alpha$, L1 weight (`gamma` in the code) | 0.01 at 32 pixels, 0.001 on the ImageNet subset | 0.01 everywhere | `DEFAULT_L1_WEIGHT` | Appendix B.3 and the released default |
| $\beta$, TV weight | 10 | 1.0 | `DEFAULT_TV_WEIGHT` | the released default, deviation 1 |
| learning rate | 0.1 | 0.1 | `DEFAULT_LEARNING_RATE` | Appendix B.3 |
| Adam $\beta_1, \beta_2$ | 0.1, 0.1 | (0.1, 0.1) | `ADAM_BETAS` | Appendix B.3 |
| steps | 100 | 100 | `DEFAULT_NUM_STEPS` | Appendix B.3 |
| mask norm | L1 | L1 | `MASK_NORM` | Eq. (1) and (4) |
| mask parameter at step 0 | ones in the code | 1.0, effective mask 0.8808 | `MASK_PARAMETER_INIT` | the released code |
| mask channels | 1 | 1 | `MASK_CHANNELS` | Eq. (2) |
| threshold coefficient $\gamma$ | 1 | none, the quantile rule | `PSBD_QUANTILES` | the registry's shared rule |

The paper uses $\beta$ both for the TV weight and for the Adam moment decays, and the table keeps its symbols and separates them by name.

## Cross-check against the reference

`tests/test_detectors_cd_l.py::test_mask_norms_match_the_released_class_bit_for_bit` loads the authors' class from `third_party/CognitiveDistillation/detection/cognitive_distillation.py`, runs it and `distill_masks` in float32 on the CPU with identity normalization and the same seed before each call, and requires the mask norms to be equal bit for bit. With the checkout restored by `scripts/fetch_third_party.sh` the test runs and passed on the CPU on 2026-09-29. The other tests in the file check that the effective mask of ones is 0.8808, the total variation of a constant mask is 0 and carries the released normalization, no model parameter receives a gradient and the flags are restored, and the model receives the native resolution. A last test checks that a triggered input distills to a smaller mask than its clean twin on a hand-built model whose trigger gate is differentiable. The synthetic sign gate `python -m experiments.preflight.check_signs`, run on the CPU on 2026-09-29 with 30 steps, read CD-L at AUROC 0.9812, above the floor of 0.60.

## Cost

1 reference forward plus 100 steps of a forward and a backward per input. Counting a backward as 1.5 forwards, the registry's convention, that is $1 + 100 \times 2.5 = 251$ forward-equivalents, the highest in `FORWARD_PASSES_PER_INPUT`. With the model frozen the backward computes no weight gradients and costs closer to 1 forward, so 251 is a ceiling. There is no fit. The measured seconds per input are in the results block.

## Direction

Low is poisoned and the score is returned unnegated. Eq. (4) flags a mask whose norm is at or below the threshold, which already agrees with the shared convention. The released analysis script returns `1 - minmax(norm)` when it computes its own AUROC, a convention local to that script which the port does not reproduce.

## Results

<!-- results:begin -->
Generated by `python scripts/detector_doc_results.py` at commit `b2d32cf11708d5de965d3e13604c863a3ad9b493-dirty`. It reads `results/coverage/coverage.json`, every `results/<folder>/detectors/<name>_metrics.json` and every `results/<folder>/psbd_metrics.json` of the 54 backdoored ViT-B/16 models the paper's detector comparison uses, the successful models that carry a reading from every defense. PSBD-TM and PSBD-RD are read at the adaptive rate rule, every threshold is the clean-validation quantile named in the column and AUROC is the one-sided area at the 0.25 quantile, where a value under 0.5 means inverted.

Summary over every compared model. The rank is among the 13 defenses of the comparison by mean AUROC, and the last column is PSBD-TM minus the defense, paired per model, with its 95% bootstrap interval over models (5000 resamples, seed 0).

| defense | models | AUROC | TPR at 10% FPR | TPR at 20% FPR | models below chance | rank | PSBD-TM minus defense, AUROC |
|---|---|---|---|---|---|---|---|
| `cd_l` | 54 | 0.808 | 0.536 | 0.682 | 7 | 7 of 13 | +0.155 [+0.099, +0.214] |
| PSBD-TM | 54 | 0.963 | 0.887 | 0.916 | 1 | 1 of 13 | reference |
| PSBD-RD | 54 | 0.885 | 0.736 | 0.798 | 5 | 5 of 13 | +0.078 [+0.027, +0.134] |

`cd_l` per attack and poison rate. Each row is a mean over the models of that attack at that rate and n counts them. The last 2 columns repeat the AUROC of PSBD-TM and PSBD-RD on the same models.

| attack | poison rate | n | AUROC | TPR at 10% FPR | TPR at 20% FPR | PSBD-TM AUROC | PSBD-RD AUROC |
|---|---|---|---|---|---|---|---|
| BPP | 1% | 4 | 0.894 | 0.653 | 0.722 | 0.946 | 0.914 |
| BPP | 5% | 4 | 0.932 | 0.732 | 0.879 | 0.941 | 0.962 |
| BPP | 10% | 4 | 0.938 | 0.723 | 0.955 | 0.957 | 0.959 |
| BadNets | 1% | 4 | 0.912 | 0.708 | 0.726 | 0.987 | 0.542 |
| BadNets | 5% | 4 | 0.957 | 0.743 | 0.988 | 0.992 | 0.796 |
| BadNets | 10% | 4 | 0.961 | 0.750 | 0.998 | 0.996 | 0.798 |
| Blend | 1% | 4 | 0.843 | 0.579 | 0.746 | 0.973 | 0.944 |
| Blend | 5% | 4 | 0.888 | 0.727 | 0.794 | 0.987 | 0.970 |
| Blend | 10% | 4 | 0.820 | 0.639 | 0.733 | 0.976 | 0.997 |
| LF | 1% | 4 | 0.567 | 0.209 | 0.352 | 0.963 | 0.959 |
| LF | 5% | 4 | 0.589 | 0.210 | 0.305 | 0.986 | 0.978 |
| LF | 10% | 4 | 0.642 | 0.179 | 0.314 | 0.990 | 0.985 |
| TaCT | 1% | 1 | 0.374 | 0.005 | 0.064 | 0.979 | 0.464 |
| TaCT | 5% | 2 | 0.615 | 0.212 | 0.502 | 0.954 | 0.613 |
| WaNet | 5% | 1 | 0.755 | 0.070 | 0.394 | 0.930 | 0.953 |
| WaNet | 10% | 2 | 0.742 | 0.523 | 0.658 | 0.704 | 0.956 |

Mean AUROC per dataset. The shared 2000-image clean split gives about 200 images per class on CIFAR-10, 46 on GTSRB, 20 on CIFAR-100 and 10 on Tiny ImageNet, which is the budget every class-conditional method fits on.

| defense | CIFAR-10 | CIFAR-100 | GTSRB | Tiny ImageNet |
|---|---|---|---|---|
| `cd_l` | 0.869 (15) | 0.810 (12) | 0.839 (13) | 0.711 (14) |
| PSBD-TM | 0.919 (15) | 0.979 (12) | 0.984 (13) | 0.977 (14) |
| PSBD-RD | 0.831 (15) | 0.890 (12) | 0.855 (13) | 0.965 (14) |

Measured cost, median over the compared models. Seconds per 1000 inputs divide the scoring time of the clean and backdoor splits by their size. The fit is the one-off pass over the clean validation split before any input is scored, and a dash marks a detector with no fit. The device is the one most records name.

| detector | forward passes per input | seconds per 1000 inputs | fit seconds | precision | device |
|---|---|---|---|---|---|
| `cd_l` | 251 | 96.32 | -- | bfloat16 | NVIDIA A100-SXM4-40GB |

<!-- results:end -->

## Reading the results

CD-L is strong on BadNets and BPP and reasonable on Blend. It is weak on LF at every rate, on WaNet, on the SIG model and on TaCT. Most of the split follows the mechanism. A patch trigger can be reproduced from a subset of pixels, so the mask collapses onto it. LF, WaNet and SIG spread the trigger over the whole image, so there is no small region to collapse onto and the triggered mask stays as large as a clean one. TaCT's trigger fires only with its source class's content present, so the mask has to keep the object as well. BPP is the exception: it changes every pixel slightly by reducing the color depth, and CD-L still separates it, which the paper's mechanism does not predict and which this repository has not explained. CD-L costs the most per input of any detector for a mean AUROC in the middle of the ranking.

## Known failure modes

A low mask norm says the logits can be reproduced from few pixels, which a benign image with 1 small salient object also allows. Whole-image triggers are expected failures by design, as the reading above shows. An input-insensitive model breaks the score from the other side: if no pixel moves the logits, the L1 term wins everywhere and every mask collapses to the same value, so the scores tie. The bfloat16 policy is a residual risk, since gradient rounding changes an optimization trajectory where it only rounds a forward pass, and the smoke pair bounds it on 1 model only.

## How to run and where records land

```bash
python -m cli.baselines --checkpoint-folder <folder> --detectors cd_l --max-samples 200 \
    --results-dir scratch/detector_smoke/results --allow-missing-psbd-cache
```

The smoke above writes outside `results/`. The panel runs through the `cd_l` job group of `pbs/generate_detector_jobs.py`, 1 job per model, because CD-L's runtime dwarfs the cheap detectors. `results/<folder>/detectors/cd_l_metrics.json` holds the report and provenance, and `cd_l_scores_{validation,clean,backdoor}.pt` hold the raw mask norms in loader order.
