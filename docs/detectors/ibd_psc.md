# IBD-PSC, parameter-oriented scaling consistency

IBD-PSC amplifies the scale and shift parameters of the normalization layers nearest the model's head and reads how much probability the amplified model still gives the label the unamplified model predicted. 2 variants are registered and both are documented here: `ibd_psc`, the faithful port at the paper's amplification factor, and `ibd_psc_calibrated`, which searches the factor upward until the paper's own layer-selection rule can work on a ViT. This page explains the idea, gives the paper's equations, describes the paper's setting and released code, walks through the port in `detectors/ibd_psc.py`, lists every deviation with its reason and ends with both variants' results on the panel. Terms such as the shared split, the panel, PSBD-TM, PSBD-RD, AUROC and TPR at a false-positive budget are defined once in `README.md` in this directory.

## The idea in plain words

IBD-PSC is a relative of PSBD: both perturb the model rather than the input and ask whether the prediction survives. IBD-PSC's perturbation is deterministic. It multiplies the learned scale $\gamma$ and shift $\beta$ of the last few normalization layers by a factor $\omega > 1$, which inflates the features those layers pass on. A clean prediction usually rests on a comparison between several logits of similar size, and inflating the features amplifies the noise in that comparison along with the signal, so the predicted class flips or loses probability. A backdoor sends its trigger to the target through a logit far above all others, a margin that survives the same inflation. The statistic is the probability the amplified model assigns to the unamplified prediction, averaged over a few amplification depths: high means the prediction survived, which is the backdoor signature.

How many layers to amplify is chosen from clean data. Amplifying too few layers breaks nothing, too many breaks everything. The paper's Algorithm 1 walks inward from the head and amplifies 1 more layer at each step. It stops at the first depth where the clean error rate exceeds a threshold $\xi$, the depth where clean predictions have started to collapse and triggered ones should not have yet.

## The original method

Hou et al., "IBD-PSC: Input-level Backdoor Detection via Parameter-oriented Scaling Consistency", ICML 2024, arXiv:2405.09786. Amplification is Section 4.3, Equation (2), layer selection is Equation (3) and Algorithm 1, and the score is Section 4.4, Equation (4).

$$
\begin{aligned}
\hat{F}^{\omega}_k &= FC \circ \hat{f}^{\omega}_L \circ \cdots \circ \hat{f}^{\omega}_{L-k+1} \circ f_{L-k} \circ \cdots \circ f_1, \quad \hat{\gamma} = \omega \gamma, \ \hat{\beta} = \omega \beta && \text{(2)} \\
\eta &= \frac{1}{|D_r|} \sum_{(x, y) \in D_r} \mathbb{I}\big( \arg\max \hat{F}^{\omega}_k(x) \neq y \big) && \text{(3)} \\
k &= \min \{ i \in 1, \ldots, L : \eta(i) > \xi \} && \text{Algorithm 1} \\
PSC(x) &= \frac{1}{n} \sum_{i=k}^{k+n-1} \hat{F}^{\omega}_i(x)_{y'}, \quad y' = \arg\max F(x) && \text{(4)}
\end{aligned}
$$

| symbol | meaning |
|---|---|
| $F$ | the deployed model, its output a softmax vector |
| $f_j$ | the model's $j$-th layer with a normalization, unmodified |
| $\hat{f}^{\omega}_j$ | the same layer with its normalization scale and shift multiplied by $\omega$ |
| $FC$ | the classification head |
| $L$ | the number of amplifiable normalization layers |
| $\hat{F}^{\omega}_k$ | the model with its last $k$ normalization layers amplified |
| $\gamma, \beta$ | a normalization layer's learned scale and shift |
| $\omega$ | the amplification factor |
| $D_r$ | the defender's labeled clean reference images |
| $\eta$, $\eta(i)$ | the top-1 error of the amplified model on $D_r$, at $i$ amplified layers |
| $\xi$ | the error-rate threshold Algorithm 1 stops at |
| $k$ | the starting depth, the smallest $i$ whose clean error exceeds $\xi$ |
| $n$ | the ensemble size, how many consecutive depths are averaged |
| $y'$ | the unamplified model's predicted label |
| $PSC(x)$ | the mean probability the $n$ amplified models assign to $y'$ |

The paper flags an input as poisoned if $PSC(x) > T$. The descriptive form renames without rederiving.

$$
\begin{aligned}
\text{amplified\_model}(i) &= \text{the model with the last } i \text{ normalization layers' scale and shift multiplied by } \omega \\
\text{clean\_error}(i) &= \text{top-1 error of amplified\_model}(i) \text{ on the clean split} \\
\text{start\_count} &= \text{smallest } i \text{ whose clean\_error exceeds } \xi \\
\text{psc}(\text{image}) &= \text{mean over } n \text{ depths of the probability the amplified model gives the unamplified prediction}
\end{aligned}
$$

## The original paper's setting

The paper evaluates on CIFAR-10, GTSRB and a 200-class ImageNet subset, all with ResNet-18, whose 20 BatchNorm layers are the amplification targets. It attacks with 13 methods, among them BadNets, Blend, Label-Consistent, ISSBA, TaCT, NARCISSUS and WaNet. Its baselines include STRIP, TeCo and SCALE-UP. The defaults are fixed across every attack and dataset in Section 5.1: $\omega = 1.5$, $n = 5$, $\xi = 0.6$ and $T = 0.9$, with 100 benign images as $D_r$. No transformer is evaluated, and the method as written needs BatchNorm.

## The reference implementation

`third_party/BackdoorBox/core/defenses/IBD_PSC.py` at commit `af3afd1` is the released code, pinned in `third_party.lock`. `count_BN_layers` at line 60 counts modules with `isinstance(module, torch.nn.BatchNorm2d)` (line 63). `prob_start` at line 89 runs Algorithm 1 with `for layer_index in range(1, layer_num)` (line 93), which never tests the all-layers configuration and falls off the end of the loop returning `None` when the error never crosses $\xi$. The ensemble at line 139 amplifies `sorted_indices[:layer_index+1]`, 1 more layer at every position than Eq. (4) writes, and deep-copies the model once per ensemble member per batch. `_test` also keeps only the inputs whose prediction equals their label before it returns a score (lines 133 and 153). `third_party/backdoor-toolbox/other_defenses_tool_box/IBD_PSC.py` agrees on the BatchNorm-only filter. The line numbers were read from the pinned checkout on 2026-09-29.

## The port step by step

**Shared by both variants.**

1. `amplifiable_norm_layers(model)` collects every `nn.LayerNorm` with learnable affine parameters in definition order and reverses the list so element 0 is the layer nearest the head. On ViT-B/16 that is 25 layers (2 per block over 12 blocks plus the final `encoder.ln`) and on Swin-S 53. Definition order equals execution order for torchvision's ViT and Swin, so the reversal is depth order.
2. `amplified_parameters(ordered_layers, omega)` is a context manager. On entry it clones every layer's weight and bias. It yields `amplify_first(count)`, which writes $\omega \gamma$ and $\omega \beta$ into the first `count` layers from the saved originals and restores the rest, so walking through depths never compounds. On exit it copies every original back.
3. `clean_error_rate(model, validation_loader, ...)` runs the currently amplified model over the 2000 validation images and returns the share whose argmax differs from the true label, Eq. (3).
4. `psc_scores(model, loader, device, ordered_layers, k, omega, n, use_bfloat16)` builds `member_counts = [i for i in range(k, k + n) if i <= L]`. For each batch it runs 1 unamplified forward to get $y'$, shape (batch,), then for each member count it amplifies that many layers, runs 1 forward, gathers the probability of $y'$ and accumulates. The sum divided by the number of members is Eq. (4), shape (batch,).
5. `ibd_psc_scores` negates `psc_scores` once and returns float32 scores of shape (N,).

**Faithful, `ibd_psc`.** `_build_ibd_psc` calls `select_start_layer_count(model, validation_loader, device, ordered_layers, 1.5, 0.6, ...)`, which runs Algorithm 1 for $i = 1, \ldots, L$ and returns the first $i$ whose clean error exceeds 0.6, or $L$ when none does, with the error trace. The chosen $k$ is recorded as `start_layer_count` in the record's provenance.

**Calibrated, `ibd_psc_calibrated`.** `_build_ibd_psc_calibrated` calls `calibrate_scaling_factor(..., CALIBRATION_FACTORS, 0.6, ...)`, which runs Algorithm 1 at $\omega$ = 1.5, 2, 3, 5 and 8 in turn and stops at the first factor whose trace crosses 0.6. It scores at that factor and its $k$. The record carries `scaling_factor`, `start_layer_count` and `crossed_error_threshold`.

## Deviations and why

1. **LayerNorm instead of BatchNorm.** This is the substantive deviation. The paper and its code amplify `BatchNorm2d` only. ViT and Swin contain no BatchNorm, so `count_BN_layers` returns 0 and the published method cannot run. The port amplifies `nn.LayerNorm`, and the substitution is exact at the level of what Eq. (2) does to 1 layer's output, because both normalize first and apply the affine map second:

$$
\omega\gamma \odot \hat{x} + \omega\beta = \omega \, (\gamma \odot \hat{x} + \beta)
$$

   where $\hat{x}$ is the normalized input and $\odot$ the elementwise product. What does not carry over is the depth. A ResNet BatchNorm scales the whole feature map a stage passes on, while a transformer LayerNorm sits on the input of 1 branch of a residual block, so amplifying it scales that branch's input and leaves the residual stream untouched. The effect per layer on the logits is therefore weaker. Algorithm 1 absorbs that in principle by choosing $k$ from measured clean error, but the $k$ it chooses on a ViT is not comparable to a published $k$.
2. **The calibrated variant.** The detector smoke of 2026-09-10 (`docs/runs/2026-09-10-detector-smoke.md`) found that on a GTSRB ViT the paper's $\omega = 1.5$ leaves the predictions intact through every amplified LayerNorm, so Algorithm 1 never crosses $\xi$, $k$ falls back to $L$ and every input keeps its label at nearly full probability, clean or triggered. The detector then reads chance for a scale mismatch rather than for a property of the backdoor. The settings table in the results block below shows how often $k$ lands at $L = 25$ on the panel. `ibd_psc_calibrated` gives the mechanism the amplification it needs on this architecture while keeping everything else from the paper, and it tries 1.5 first, so a model the paper's setting already breaks is scored exactly as `ibd_psc` scores it. The ladder 1.5, 2, 3, 5 and 8 is this project's choice and appears nowhere in the paper. It roughly doubles at each step. Both variants run on the panel, the faithful one so the paper's own setting is on record.
3. **Ensemble layer counts.** Eq. (4) sums over $i = k, \ldots, k + n - 1$ amplified layers. The released code amplifies 1 more layer at every position. The port follows the equation.
4. **Algorithm 1's range.** The paper tests $i = 1, \ldots, L$. The released code tests $1, \ldots, L - 1$ and returns `None` when nothing crosses, which crashes the next call. The port tests all $L$ and falls back to $k = L$, the value Algorithm 1 holds at loop exit.
5. **Ensemble clamping.** When $k$ is within $n - 1$ of $L$, the window $k, \ldots, k + n - 1$ runs past the last layer, a case the paper's 20 BatchNorms rarely meet. The port keeps only the members with $i \le L$, so a late $k$ gives a smaller ensemble. At $k = L$ the ensemble holds only the fully amplified model, so the score is 1 amplified forward pass. The released code has no such clamp: its slice `sorted_indices[:layer_index+1]` stops at $L$ by itself, so every member past the last layer amplifies all $L$ layers again and the fully amplified model is counted several times in its mean. From start index $L - 2$ with $n = 5$ it averages the members at $L - 1, L, L, L, L$ where the port averages $L - 1$ and $L$ once each.
6. **Data budget.** The paper uses 100 benign images as $D_r$. The port uses the shared 2000-image validation split with its labels, so layer selection sees the same data every other detector sees.
7. **No deep copies.** The port writes amplified parameters into the live model and restores them from clones, which is exact and allocates no second model. Restoring by copy rather than by dividing by $\omega$ leaves no floating-point drift in the deployed model.
8. **The threshold rule.** The paper thresholds at $T = 0.9$. The port hands the negated PSC to `defenses.decision.detection_report`, which thresholds at a quantile of the validation scores. AUROC does not depend on this choice.

## Hyperparameters and where they come from

| symbol | paper | this port | constant | source of the port's value |
|---|---|---|---|---|
| $\omega$ | 1.5 | 1.5 for `ibd_psc`, searched for `ibd_psc_calibrated` | `DEFAULT_SCALING_FACTOR` | Section 5.1 |
| $\omega$ ladder | not in the paper | 1.5, 2, 3, 5, 8 | `CALIBRATION_FACTORS` | this project, the paper's value first |
| $n$ | 5 | 5 | `DEFAULT_ENSEMBLE_SIZE` | Section 5.1 |
| $\xi$ | 0.6 | 0.6 | `DEFAULT_ERROR_THRESHOLD` | Section 5.1 |
| $T$ | 0.9 | recorded, unused | `DEFAULT_DETECTION_THRESHOLD` | Section 5.1, replaced by the quantile rule |
| $L$ | the BatchNorm count, 20 on ResNet-18 | the affine LayerNorm count, 25 on ViT-B/16 and 53 on Swin-S | computed by `amplifiable_norm_layers` | the architecture |
| $D_r$ | 100 benign images | the shared 2000-image split, labeled | none | the shared data budget |
| $k$ | chosen by Algorithm 1 | chosen by Algorithm 1, recorded per model | `start_layer_count` in the record | fitted, see the results block |

## Cross-check against the reference

`tests/test_detectors_ibd_psc.py` executes BackdoorBox's `IBD_PSC` class from the pinned checkout on a 10-layer pre-norm residual stack of LayerNorms and linear branches on the CPU. The 1 textual change is its BatchNorm2d filter read as LayerNorm, deviation 1 applied to the reference so that everything else can be compared. The model's own predictions serve as labels, so clean error starts at 0 and `_test`'s correctness filter keeps every input.

1. Algorithm 1 picks the same depth as `prob_start` at $\omega = 3$ and $\xi = 0.3$, where the stack's error first crosses inside the range both implementations test.
2. PSC agrees with `_test` to $10^{-6}$ when the port starts 1 layer deeper than the reference's start index, and disagrees when both start at the same index, which is deviation 3 shown numerically.
3. Past the last layer the reference's mean equals $(F_{L-1} + 4 F_L)/5$ and the port's $(F_{L-1} + F_L)/2$, the clamp difference of deviation 5.
4. At the paper's $\omega = 1.5$ and $\xi = 0.6$ the stack never crosses. The reference's start index is `None` and its `_test` raises `TypeError`. The port returns $k = L$ with its whole trace under $\xi$, and scores with 2 forward passes per batch, counted by a hook, equal to the fully amplified model alone.

The first 3 tests of the file check the calibration search with a stubbed Algorithm 1. The LayerNorm identity of deviation 1 is algebra and is not tested. The synthetic sign gate `python -m experiments.preflight.check_signs`, run on the CPU on 2026-09-29, read both variants at AUROC 1.0000 on the fixture, above its floor of 0.60.

## Ensemble size on ViT

The faithful fit lands at $k = L = 25$ on most panel models (the settings table of the results block counts them). Each of those is the fallback of deviation 4 and not a crossing at the last layer: on every one, `ibd_psc_calibrated`, which runs the same Algorithm 1 at 1.5 first, found no crossing at 1.5 and moved up its ladder. At $k = L$ the window $L, \ldots, L + 4$ keeps only $L$, so the faithful score is the probability the fully amplified model gives $y'$, 1 member, 2 forward passes per input.

This is a faithful consequence on ViT and not a port bug. BackdoorBox's layer-selection rule never tests $L$ and has no value to return when nothing crosses, so the released code raises on these models rather than scoring them, and the paper's Algorithm 1 holds $k = L$ at loop exit. Had the released ensemble been run from that $k$, its slice would stop at $L$ for every member, so all 5 of its members would be the same fully amplified model and its mean the same number the port returns, at 5 times the cost. The port's score is therefore what the reference's own arithmetic gives, and only the ensemble size differs. `test_without_a_crossing_backdoorbox_has_no_depth_and_the_port_scores_at_2_passes` pins this.

The cost bookkeeping does not follow. `FORWARD_PASSES_PER_INPUT["ibd_psc"]` is $n + 1 = 6$ whatever $k$ is, and every `ibd_psc` record writes that 6 into its provenance as `forward_passes_per_input`, while a model at $k = L$ spends 2. A cost table read off the provenance overstates the faithful variant on those models by a factor of 3. The measured seconds per input in the results block are not affected. The registry is left as it is, since the paper's cost table reads it and the correction belongs with that table.

## Cost

$n + 1 = 6$ forward passes per input nominally, 1 unamplified and 1 per ensemble member, as `FORWARD_PASSES_PER_INPUT` records. On a model where $k$ lands at or near $L$ the clamp of deviation 5 leaves fewer members, down to 2 forwards per input at $k = L$, which is why the measured seconds per input of `ibd_psc` in the results block are lower than those of the calibrated variant. Layer selection adds up to $L$ passes over the 2000 validation images before any input is scored, and the calibrated variant can repeat that once per factor on its ladder, which is why its fit is longer. Both are in `NEEDS_FITTING`.

## Direction

Low is poisoned. The paper flags $PSC(x) > T$, so the raw statistic is high for poisoned. `ibd_psc_scores` negates `psc_scores` once at the return.

## Results

<!-- results:begin -->
Generated by `python scripts/detector_doc_results.py` at commit `b2d32cf11708d5de965d3e13604c863a3ad9b493-dirty`. It reads `results/coverage/coverage.json`, every `results/<folder>/detectors/<name>_metrics.json` and every `results/<folder>/psbd_metrics.json` of the 54 backdoored ViT-B/16 models the paper's detector comparison uses, the successful models that carry a reading from every defense. PSBD-TM and PSBD-RD are read at the adaptive rate rule, every threshold is the clean-validation quantile named in the column and AUROC is the one-sided area at the 0.25 quantile, where a value under 0.5 means inverted.

Summary over every compared model. The rank is among the 13 defenses of the comparison by mean AUROC, and the last column is PSBD-TM minus the defense, paired per model, with its 95% bootstrap interval over models (5000 resamples, seed 0).

| defense | models | AUROC | TPR at 10% FPR | TPR at 20% FPR | models below chance | rank | PSBD-TM minus defense, AUROC |
|---|---|---|---|---|---|---|---|
| `ibd_psc` | 54 | 0.732 | 0.400 | 0.517 | 12 | 9 of 13 | +0.231 [+0.165, +0.299] |
| `ibd_psc_calibrated` | 54 | 0.936 | 0.825 | 0.900 | 0 | 2 of 13 | +0.027 [+0.004, +0.049] |
| PSBD-TM | 54 | 0.963 | 0.887 | 0.916 | 1 | 1 of 13 | reference |
| PSBD-RD | 54 | 0.885 | 0.736 | 0.798 | 5 | 5 of 13 | +0.078 [+0.027, +0.134] |

`ibd_psc` per attack and poison rate. Each row is a mean over the models of that attack at that rate and n counts them. The last 2 columns repeat the AUROC of PSBD-TM and PSBD-RD on the same models.

| attack | poison rate | n | AUROC | TPR at 10% FPR | TPR at 20% FPR | PSBD-TM AUROC | PSBD-RD AUROC |
|---|---|---|---|---|---|---|---|
| BPP | 1% | 4 | 0.517 | 0.240 | 0.245 | 0.946 | 0.914 |
| BPP | 5% | 4 | 0.683 | 0.370 | 0.458 | 0.941 | 0.962 |
| BPP | 10% | 4 | 0.726 | 0.217 | 0.549 | 0.957 | 0.959 |
| BadNets | 1% | 4 | 0.723 | 0.190 | 0.404 | 0.987 | 0.542 |
| BadNets | 5% | 4 | 0.778 | 0.422 | 0.506 | 0.992 | 0.796 |
| BadNets | 10% | 4 | 0.804 | 0.705 | 0.741 | 0.996 | 0.798 |
| Blend | 1% | 4 | 0.756 | 0.308 | 0.667 | 0.973 | 0.944 |
| Blend | 5% | 4 | 0.939 | 0.848 | 0.957 | 0.987 | 0.970 |
| Blend | 10% | 4 | 0.946 | 0.750 | 0.774 | 0.976 | 0.997 |
| LF | 1% | 4 | 0.572 | 0.319 | 0.394 | 0.963 | 0.959 |
| LF | 5% | 4 | 0.580 | 0.246 | 0.278 | 0.986 | 0.978 |
| LF | 10% | 4 | 0.795 | 0.419 | 0.489 | 0.990 | 0.985 |
| TaCT | 1% | 1 | 0.210 | 0.000 | 0.000 | 0.979 | 0.464 |
| TaCT | 5% | 2 | 0.920 | 0.050 | 0.271 | 0.954 | 0.613 |
| WaNet | 5% | 1 | 0.859 | 0.680 | 0.771 | 0.930 | 0.953 |
| WaNet | 10% | 2 | 0.656 | 0.339 | 0.391 | 0.704 | 0.956 |

`ibd_psc_calibrated` per attack and poison rate. Each row is a mean over the models of that attack at that rate and n counts them. The last 2 columns repeat the AUROC of PSBD-TM and PSBD-RD on the same models.

| attack | poison rate | n | AUROC | TPR at 10% FPR | TPR at 20% FPR | PSBD-TM AUROC | PSBD-RD AUROC |
|---|---|---|---|---|---|---|---|
| BPP | 1% | 4 | 0.937 | 0.857 | 0.919 | 0.946 | 0.914 |
| BPP | 5% | 4 | 0.927 | 0.800 | 0.886 | 0.941 | 0.962 |
| BPP | 10% | 4 | 0.895 | 0.680 | 0.812 | 0.957 | 0.959 |
| BadNets | 1% | 4 | 0.962 | 0.890 | 0.975 | 0.987 | 0.542 |
| BadNets | 5% | 4 | 0.960 | 0.919 | 0.985 | 0.992 | 0.796 |
| BadNets | 10% | 4 | 0.980 | 0.977 | 0.995 | 0.996 | 0.798 |
| Blend | 1% | 4 | 0.990 | 0.994 | 0.996 | 0.973 | 0.944 |
| Blend | 5% | 4 | 0.992 | 0.997 | 0.999 | 0.987 | 0.970 |
| Blend | 10% | 4 | 0.997 | 1.000 | 1.000 | 0.976 | 0.997 |
| LF | 1% | 4 | 0.907 | 0.782 | 0.876 | 0.963 | 0.959 |
| LF | 5% | 4 | 0.914 | 0.752 | 0.855 | 0.986 | 0.978 |
| LF | 10% | 4 | 0.935 | 0.819 | 0.906 | 0.990 | 0.985 |
| TaCT | 1% | 1 | 0.713 | 0.134 | 0.441 | 0.979 | 0.464 |
| TaCT | 5% | 2 | 0.918 | 0.497 | 0.718 | 0.954 | 0.613 |
| WaNet | 5% | 1 | 0.859 | 0.680 | 0.771 | 0.930 | 0.953 |
| WaNet | 10% | 2 | 0.776 | 0.438 | 0.567 | 0.704 | 0.956 |

Mean AUROC per dataset. The shared 2000-image clean split gives about 200 images per class on CIFAR-10, 46 on GTSRB, 20 on CIFAR-100 and 10 on Tiny ImageNet, which is the budget every class-conditional method fits on.

| defense | CIFAR-10 | CIFAR-100 | GTSRB | Tiny ImageNet |
|---|---|---|---|---|
| `ibd_psc` | 0.639 (15) | 0.833 (12) | 0.531 (13) | 0.931 (14) |
| `ibd_psc_calibrated` | 0.917 (15) | 0.943 (12) | 0.942 (13) | 0.944 (14) |
| PSBD-TM | 0.919 (15) | 0.979 (12) | 0.984 (13) | 0.977 (14) |
| PSBD-RD | 0.831 (15) | 0.890 (12) | 0.855 (13) | 0.965 (14) |

Settings `ibd_psc` fitted per model. Each cell is a value the record's provenance carries and the number of models that took it.

| setting | value (models) |
|---|---|
| `start_layer_count` | 22 (1), 23 (5), 25 (48) |

Settings `ibd_psc_calibrated` fitted per model. Each cell is a value the record's provenance carries and the number of models that took it.

| setting | value (models) |
|---|---|
| `crossed_error_threshold` | True (54) |
| `scaling_factor` | 1.5 (6), 2.0 (38), 3.0 (10) |
| `start_layer_count` | 14 (1), 15 (7), 17 (11), 19 (2), 21 (13), 22 (1), 23 (19) |

Measured cost, median over the compared models. Seconds per 1000 inputs divide the scoring time of the clean and backdoor splits by their size. The fit is the one-off pass over the clean validation split before any input is scored, and a dash marks a detector with no fit. The device is the one most records name.

| detector | forward passes per input | seconds per 1000 inputs | fit seconds | precision | device |
|---|---|---|---|---|---|
| `ibd_psc` | 6 | 0.83 | 23.9 | bfloat16 | NVIDIA A100-SXM4-40GB |
| `ibd_psc_calibrated` | 6 | 2.39 | 43.8 | bfloat16 | NVIDIA A100-SXM4-40GB |

<!-- results:end -->

## Reading the results

The 2 variants tell 1 story. At the paper's factor the settings table shows Algorithm 1 landing at or next to the last layer on almost every panel model, which means the amplification never broke 60% of clean predictions at any depth. The detector then scores a single fully amplified model, and its below-chance count shows how often that model preserved triggered and clean predictions alike, or preserved the clean ones better. Once the factor is raised until clean error does cross $\xi$, the same statistic becomes the strongest competitor in the comparison and never falls below chance on the panel. Its paired gap to PSBD-TM in the summary is the smallest of any competitor, and the interval sits just above 0.

The per-attack tables locate the difference. The calibrated variant beats PSBD-TM on Blend and on the single SIG model, is close on BadNets and BPP and trails on LF, WaNet and TaCT. The factor the calibration settles on varies by model, and the settings table lists the spread.

## Known failure modes

A low PSC says the amplified models stopped agreeing with the unamplified prediction, which any thin decision margin produces, trigger or not. The paper's own adaptive section (Section 5.4) reports 2 designs: the first reduces IBD-PSC's worst-case AUROC but leaves it well above chance, and the second defeats it by collapsing the model's own benign accuracy, which is a broken model rather than a usable attack. `docs/attack-design/A5-low-confidence-backdoor.md` predicts that a backdoor trained deliberately at low confidence would defeat the statistic, since amplification breaks any thin margin regardless of why it is thin. That is a prediction, not a measurement. Algorithm 1's $k$ is itself worth reading beside the AUROC: a $k$ at 1 or at $L$ means the factor was mismatched to that model rather than anything about its backdoor.

## How to run and where records land

```bash
python -m cli.baselines --checkpoint-folder <folder> --detectors ibd_psc ibd_psc_calibrated \
    --max-samples 500 --results-dir scratch/detector_smoke/results --allow-missing-psbd-cache
```

The smoke above writes outside `results/`. The panel runs through the `cheap` job group of `pbs/generate_detector_jobs.py`. `results/<folder>/detectors/ibd_psc_metrics.json` and `ibd_psc_calibrated_metrics.json` hold each variant's report and provenance. The provenance carries the fitted `start_layer_count`, plus `scaling_factor` and `crossed_error_threshold` for the calibrated variant. `<name>_scores_{validation,clean,backdoor}.pt` hold the negated scores in loader order.
