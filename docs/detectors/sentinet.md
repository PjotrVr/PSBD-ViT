# SentiNet, region transplant against localized universal attacks

SentiNet finds the region of an input that drives the model's prediction with Grad-CAM, pastes that region onto a set of clean images and asks 2 questions of the model: how many of the clean images now take the input's label, and how confident the model stays when the same region is filled with noise instead. A trigger answers both in its favor while a benign salient region fails at least 1. The decision is a curve fitted over clean inputs in that 2-dimensional plane. This page explains the idea, gives the paper's algorithms, describes the paper's setting and the 3 released reimplementations, walks through the port in `detectors/sentinet.py`, lists every deviation with its reason and ends with the results on the panel, which are below chance and only partly explained. Terms such as the shared split, the panel, PSBD-TM, PSBD-RD, AUROC and TPR at a false-positive budget are defined once in `README.md` in this directory.

## The idea in plain words

A patch trigger is a small region that hijacks whatever image it lands on. SentiNet tries to find that region and test it directly. First it asks the model which part of the input mattered for its prediction, using Grad-CAM, a heatmap built from the gradient of the predicted class's logit with respect to a late feature map. It cuts out the hottest part of that heatmap. Then it runs 2 experiments on a set of clean images. It pastes the cut-out region onto each clean image and counts how many are now classified as the input's label: a trigger fools nearly all of them, a piece of an ordinary object rarely does. It also pastes random noise into the same spot of each clean image and records the model's average confidence: a small trigger region hides little of the clean image, so confidence stays high, while a large benign object region hides a lot and confidence drops. Plotted as (average confidence, fooled count), clean inputs stay under a curve and triggered inputs sit above it at the top right. The score is how far above that curve an input sits.

The method is white-box because Grad-CAM needs a gradient. It also needs a set of clean images to paste onto. It is designed for localized triggers: a trigger spread over the whole image has no region to cut out.

## The original method

Chou et al., "SentiNet: Detecting Localized Universal Attacks Against Deep Learning Systems", IEEE S&P Workshops (DLS) 2020, arXiv:1812.00292 (v4 read). Class proposal is Algorithm 1 and mask generation Algorithm 2, both in Section III-A, with Grad-CAM restated there. The 2 test statistics are Algorithm 3 in Section III-B1 and the decision boundary is Algorithm 4 in Section III-B2.

$$
\begin{aligned}
\alpha_c^k &= \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}} \\
L^c_{Grad\text{-}CAM} &= ReLU\left( \sum_k \alpha_c^k A^k \right)
\end{aligned}
$$

```
Algorithm 2 MaskGeneration
in:  f_m, x, (y, conf) = f_m(x), C the proposed classes
out: M, masks for candidate regions
  mask_y = MaskGradCAM(f_m, x, y)
  M = { (mask_y - MaskGradCAM(f_m, x, y_p), conf_p) : (y_p, conf_p) in C }
  return {mask_y} union M

Algorithm 3 Testing
in:  f_m, x, y the class of x, M the proposed masks, X the benign test images
out: Fooled, AvgConf
  R    = { x * mask : mask in M }
  IP   = InertPattern(M)
  X_R  = Overlay(X, R)
  X_IP = Overlay(X, IP)
  fooled_yR = 0, avg_confIP = 0
  for x_R, x_IP in X_R, X_IP
      (y_R, conf_R), (y_IP, conf_IP) = f(x_R), f(x_IP)
      if y_R == y
          fooled_yR += 1
      avg_confIP += conf_IP
  avg_confIP = avg_confIP / |X|
  return fooled_yR, avg_confIP

Algorithm 4 DecisionBoundary
in:  B, the sampled behaviour of f_m on clean inputs
out: f_curve, d the acceptable distance from f_curve
  f_curve = ApproximateCurve(OutPts(B))
  avg_d = 0
  for (x, y) in B
      if f_curve(x) > y
          avg_d += COBYLA((y, x), f_curve)
  d = avg_d / |B|
  return f_curve, d
```

| symbol | meaning |
|---|---|
| $y^c$ | the model's logit for class $c$ |
| $A^k$ | the $k$-th feature map of the layer Grad-CAM reads, indexed by spatial position $(i, j)$ |
| $Z$ | the number of spatial positions in a feature map |
| $\alpha_c^k$ | the importance weight of feature map $k$ for class $c$, its gradient averaged over positions |
| $L^c$ | the Grad-CAM heatmap for class $c$, binarized at 15% of its maximum in the paper |
| $f_m$ | the deployed model, returning a label and a confidence |
| $x, y$ | the input under test and the model's prediction on it |
| $C$ | the classes Algorithm 1 proposes from a selective-search segmentation, with their confidences |
| $M, R$ | the candidate masks and the regions of $x$ they cut out |
| $X$ | the benign test images the regions are pasted onto |
| $IP$ | the inert pattern, random noise by default, filling the same region |
| $X_R, X_{IP}$ | the clean images with the region pasted in, and with the inert pattern pasted in |
| $fooled$ | the count of $X_R$ images whose label equals $y$ |
| $avgConf$ | the mean confidence over the $X_{IP}$ images |
| $B$ | the $(avgConf, fooled)$ points of clean inputs |
| $OutPts$ | the points with the highest $fooled$ in each $avgConf$ interval |
| $f_{curve}$ | the least-squares curve through those points, a parabola in the paper's Figure 4 |
| $d$ | the mean COBYLA distance from the curve over the clean points above it |

The descriptive form, as the port computes it on a transformer with tokens in place of feature-map positions, renames without rederiving.

$$
\begin{aligned}
\text{token\_weight} &= \text{mean over patch tokens of } \partial\, \text{predicted logit} / \partial\, \text{token} \\
\text{cam}[\text{patch}] &= ReLU\big( \text{token\_weight} \cdot \text{token}[\text{patch}] \big), \text{ scaled per image to } [0, 1] \\
\text{mask} &= \big[ \text{upsampled cam} \ge 0.85 \big] \\
\text{fooled} &= \text{share of clean overlays predicted as the input's label once its region is pasted on them} \\
\text{avg\_conf} &= \text{mean max softmax over the same overlays with noise pasted instead} \\
\text{envelope} &= \text{quadratic through the 2 largest clean fooled values per avg\_conf bin of width 0.04} \\
\text{sentinet\_score} &= -\big( \text{fooled} - \text{envelope}(\text{avg\_conf}) \big)
\end{aligned}
$$

## The original paper's setting

The adversary mounts a localized universal attack: a contiguous region that hijacks the prediction of any image it is placed on, whether from a trojaned model, a poisoned model or an adversarial patch against a clean model. The defender holds the model white-box and a set $X$ of benign test images "often shipped together with deployed models", 100 images on every network. It uses about 400 further benign inputs to draw the decision boundary in Figure 4. The paper's networks are ConvNets, and no transformer is evaluated. The paper reports a latency of about 2.5 seconds per input, most of it the selective search of Algorithm 1.

## The reference implementations

The paper cites no released code, so 3 reimplementations are the executable references, all read when the port was written: `third_party/Beatrix/defenses/SentiNet/SentiNet.py` at commit `685827e` (the reference for `fooled`, `avgConf` and the boundary), `third_party/BackdoorBench/detection_infer/sentinet.py` at commit `f02e353` and `third_party/backdoor-toolbox/other_defenses_tool_box/sentinet.py` at commit `9d4d909`. All 3 drop Algorithm 1 and the mask subtraction of Algorithm 2 and use the Grad-CAM map of the predicted class as the only mask.

Beatrix runs `pytorch_grad_cam`'s `GradCAM` on `layer4[-1]` of a PreActResNet-18 and binarizes the map at `MASK_COND` 0.85. Its main sets `use_truemask = True`, which replaces the Grad-CAM mask with the attack's true trigger mask on the poisoned side, so its reported numbers are oracle numbers. It composites `background * mask + overlay * (1 - mask)` on uint8 arrays and draws 10 random overlays per input. Its inert composite pastes the input's region onto noise, the reverse of Algorithm 3. It counts `fooled` against the model's prediction on the input, bins `avgConf` at 0.04, keeps the 2 largest `fooled` per bin, fits a quadratic with `curve_fit` and sets $d$ from COBYLA distances. BackdoorBench reuses that class with its whole clean set as overlays and decides on `avgconf > 0.9` alone, never reading `fooled`. backdoor-toolbox follows Algorithm 3 for both composites, uses the top 15% of map cells by area as the mask, 100 overlays and 400 validation images for the curve, and scores by the signed perpendicular distance. It counts `fooled` against the input's true label, and against the attack's target on poisoned inputs, and pastes the true trigger region for several attacks, both of which are oracles a defender does not have. `pytorch_grad_cam` is pinned in `third_party.lock` as `third_party/pytorch-grad-cam` at commit `5a5043c`, the last commit whose `GradCAM` takes the singular `target_layer` that Beatrix's line 406 passes. These descriptions were checked against the pinned checkouts on 2026-09-29.

## The port step by step

1. `_build_sentinet` in `detectors/__init__.py` requires the validation loader. `resolve_cam_site(model)` detects the architecture and returns the layer Grad-CAM reads, `cam_layer(num_blocks, architecture)`: block count plus `CAM_LAYER_OFFSET`, which is $-1$ on ViT (the input of the last block, layer 11 of 12) and 0 on Swin (the output of the last block, layer 24).
2. `collect_overlay_pixels(validation_loader, 100, seed, mean, std)` takes the first `DEFAULT_NUM_OVERLAYS` (100) validation images through `strip.collect_overlay_batch` and denormalizes them to pixels, shape (100, channels, height, width). `draw_inert_pixels(100, image_shape, seed)` draws 100 uniform noise images of the same shape once, after `seed_everything`.
3. `sentinet_statistics` runs over the validation split. Per batch, `grad_cam` calls `cam_token_weights`, which makes the input a gradient leaf, freezes the model with `frozen_parameters`, captures the CAM layer's activation with `captured_layers`, takes the predicted logit's gradient with `torch.autograd.grad`, drops the class token on ViT and averages the gradient over the 196 patch tokens to get 1 weight per feature, shape (batch, 768). The map is $ReLU$ of the weighted sum of each patch token's features, reshaped to the 14 by 14 patch grid and scaled per image to $[0, 1]$ by `scale_per_image`.
4. `saliency_mask(cam, image_size)` upsamples the map bilinearly to the native image size, rescales it so the peak is exactly 1, thresholds at `MASK_THRESHOLD` (0.85) and always sets the peak pixel, returning a boolean mask of shape (batch, 1, height, width) that is never empty.
5. `overlay_statistics` builds, per input, 100 adversarial composites (the clean overlay outside the mask, the input inside it) and 100 inert composites (the clean overlay outside the mask, noise inside it) in pixel space, renormalizes them and forwards them in chunks of `OVERLAY_CHUNK` (256). `fooled` is the share of adversarial composites whose argmax equals the input's predicted label and `avg_conf` the mean max softmax of the inert composites, each shape (batch,).
6. `fit_decision_boundary(fooled, avg_conf)` bins the 2000 clean points by `avg_conf` into bins of `BOUNDARY_BIN_WIDTH` (0.04), keeps the `BOUNDARY_POINTS_PER_BIN` (2) largest `fooled` values per bin and fits a least-squares quadratic through them with `np.polyfit`, dropping to degree 1 or 0 with fewer than 3 populated bins.
7. The validation scores are `-boundary_residual(fooled, avg_conf, coefficients)`, where `boundary_residual` returns `fooled - envelope(avg_conf)`. For the clean and backdoor splits, `sentinet_scores` runs `sentinet_statistics` and negates the residual the same way, shape (N,).

## Deviations and why

1. **The CAM site.** The paper reads Grad-CAM at the last feature map of a ConvNet. torchvision's ViT classifies from the class token alone, so the gradient of any logit with respect to the last block's output patch tokens is exactly 0 and the map would be empty, which `tests/test_detectors_sentinet.py` shows. The port reads the last block's input on ViT, the deepest site whose patch tokens still carry gradient. Swin's head averages every token, so its site is the last block's output.
2. **No class proposal and no mask subtraction.** Selective search is most of the paper's per-input latency and no released version of the subtraction exists to match, so the port omits both, as every reimplementation does. What changes is the mask on a benign input with 2 salient objects, which the paper's Figure 3 shows tightening to the suspicious one.
3. **Mask by the scaled map at or above 0.85.** Without the subtraction, the paper's 15%-of-maximum cut covers most of a natural image, which is why the reimplementations that drop Algorithms 1 and 2 tighten the cut to 0.85 on the map scaled to $[0, 1]$. The toolbox's fixed 15%-by-area mask is recorded as `OFFICIAL_TOOLBOX_MASK_FRACTION` and not used, because a fixed area removes the axis that separates a small trigger region from a large object region.
4. **100 fixed overlays and 100 fixed noise images.** The paper ships 100 test images. The port takes the first 100 of the shared split, which is the budget every other detector gets. It fixes both sets for every scored input, which removes overlay and noise choice as a per-input source of variance. Compositing follows Algorithm 3 for both composites.
5. **`fooled` against the prediction.** At inference the only class of $x$ a defender has is the model's prediction, which is what Beatrix and BackdoorBench count against. The toolbox's true-label and target-label counts are oracles and are not reproduced.
6. **Signed vertical residual.** Algorithm 4 flags a point whose perpendicular distance above the curve exceeds $d$. The port scores the vertical residual $fooled - f_{curve}(avgConf)$, which keeps the sign and the ordering the curve induces without an optimizer per input, and hands its negation to `defenses.decision.detection_report`. The vertical and perpendicular distances differ by a factor that depends on the curve's slope, so the ranking of inputs at different confidences can differ from the paper's.
7. **Envelope fitted on the shared split, residuals in sample.** The paper draws its boundary from about 400 benign points. The port fits it on the 2000 validation points and scores those same points against it. The points that define each bin's ceiling sit on or near the curve, so the validation residuals are pulled toward 0 and the quantile threshold is tighter than a fresh clean split would give. That moves the threshold and the realized false-positive rate, and leaves the AUROC on the paired clean and backdoor splits unchanged, since both are scored against the same fixed curve. `sentinet` is therefore not in `CROSS_FITTED`.
8. **Mixed precision.** The port runs the Grad-CAM pass and the 200 composite forwards under the shared autocast policy. The captured activation is float32 under bfloat16 autocast, because the residual adds promote the branch outputs, so the map is differentiated in float32.
9. **Compositing at the native resolution.** The port composites at the dataset's own 32 or 64 pixels where the trigger was stamped. The model's `Resize` then upsamples.

## Hyperparameters and where they come from

| symbol | paper | this port | constant | source of the port's value |
|---|---|---|---|---|
| $\lvert X \rvert$ | 100 | 100, the first images of the shared split | `DEFAULT_NUM_OVERLAYS`, overridable through `DetectorContext.sentinet_overlays` | the paper |
| inert images | random noise, 1 pattern per mask | 100 uniform noise images, 1 per overlay | `DEFAULT_NUM_OVERLAYS` | the paper's default pattern |
| mask threshold | 15% of the maximum, then subtraction | 0.85 on the scaled map | `MASK_THRESHOLD` | Beatrix and BackdoorBench, deviation 3 |
| toolbox mask area | none | 0.15, recorded and unused | `OFFICIAL_TOOLBOX_MASK_FRACTION` | backdoor-toolbox |
| CAM site | the last feature map | last block input on ViT, output on Swin | `CAM_LAYER_OFFSET` | deviation 1 |
| $avgConf$ bin width | not stated | 0.04 | `BOUNDARY_BIN_WIDTH` | Beatrix |
| points per bin | "the highest", count not stated | 2 | `BOUNDARY_POINTS_PER_BIN` | Beatrix |
| curve | least squares, a parabola in Figure 4 | quadratic, lower degree with fewer than 3 bins | none | the paper and Beatrix |
| $d$ | mean COBYLA distance of clean outliers | none, the quantile rule | `PSBD_QUANTILES` | the registry's shared rule |
| composites per forward | not stated | 256 | `OVERLAY_CHUNK` | device memory |
| map range floor | none | $10^{-7}$ | `CAM_RANGE_FLOOR` | a constant map scales to 0 rather than NaN |

## Cross-check against the reference

`tests/test_detectors_sentinet.py` checks the port against constructions in the test file: Grad-CAM against the analytic map of a model whose logit is a fixed linear read of the mean patch token, `overlay_statistics` against a per-image loop, `fit_decision_boundary` against points on a known parabola, the all-zero map at the ViT output site and the never-empty mask. A direction test checks a model whose prediction is driven by a corner region. The last tests execute the pinned references on the CPU.

1. pytorch-grad-cam's `GradCAM`, hooked on the output of block 1 of the fixture's 2-block ViT with the README's ViT reshape, returns the port's map at the patch grid to $10^{-5}$. opencv-python is not installed, so the reference's `cv2.resize` is replaced by the identity and its min-max scaling runs on the grid. The bilinear upsampling in `saliency_mask` is therefore not compared.
2. Beatrix's `_get_entropy`, run with its overlay and noise draws served to match the port's, feeds the classifier the same adversarial composites as `overlay_statistics` and returns the same `fooled`. Its inert composites are the input's region on noise (line 281) and the port's are noise in the overlay's region, both rebuilt by hand and matched, which is the difference `overlay_statistics` documents.
3. Beatrix's `DecisionBoundary` and `fit_decision_boundary` give the same quadratic to $10^{-6}$ on 300 clean points off every bin edge, and the boundary points Beatrix measures a COBYLA distance for are exactly those with a positive port residual (deviation 6, sign only).

1 disagreement, too small to move a panel number, was not recorded before. Beatrix's bin $i$ is $(0.04 i, 0.04 (i + 1)]$ (line 491) and the port's $[0.04 i, 0.04 (i + 1))$, so an avg_conf of exactly a multiple of 0.04 falls in different bins and can change which points the envelope keeps. avg_conf is a mean of float32 softmax maxima, so an exact edge value is rare.

The synthetic sign gate cannot judge SentiNet, because the fixture reads its trigger straight from the pixels and no token Grad-CAM reads carries it, so `experiments/preflight/gate.py` lists it in `NOT_JUDGEABLE`. Run on the CPU on 2026-09-29, `python -m experiments.preflight.check_signs` printed 0.0117 for it, marked not judged.

## Cost

1 forward and 1 backward pass for the map, then $2 \lvert X \rvert = 200$ forwards for the 2 composite sets, per input. The registry counts the map's forward and backward as 2 model queries, $2 + 200 = 202$ in `FORWARD_PASSES_PER_INPUT`. The fit repeats the same cost over the 2000 validation images. `sentinet` is in `NEEDS_FITTING`. The measured seconds per input and fit seconds are in the results block.

## Direction

High is poisoned in the paper's plane, since a triggered input sits above the envelope. `boundary_residual` keeps that sign, and `sentinet_scores` and the builder each negate once, so low means poisoned.

## Results

<!-- results:begin -->
Generated by `python scripts/detector_doc_results.py` at commit `b2d32cf11708d5de965d3e13604c863a3ad9b493-dirty`. It reads `results/coverage/coverage.json`, every `results/<folder>/detectors/<name>_metrics.json` and every `results/<folder>/psbd_metrics.json` of the 54 backdoored ViT-B/16 models the paper's detector comparison uses, the successful models that carry a reading from every defense. PSBD-TM and PSBD-RD are read at the adaptive rate rule, every threshold is the clean-validation quantile named in the column and AUROC is the one-sided area at the 0.25 quantile, where a value under 0.5 means inverted.

Summary over every compared model. The rank is among the 13 defenses of the comparison by mean AUROC, and the last column is PSBD-TM minus the defense, paired per model, with its 95% bootstrap interval over models (5000 resamples, seed 0).

| defense | models | AUROC | TPR at 10% FPR | TPR at 20% FPR | models below chance | rank | PSBD-TM minus defense, AUROC |
|---|---|---|---|---|---|---|---|
| `sentinet` | 54 | 0.418 | 0.101 | 0.173 | 38 | 13 of 13 | +0.545 [+0.473, +0.615] |
| PSBD-TM | 54 | 0.963 | 0.887 | 0.916 | 1 | 1 of 13 | reference |
| PSBD-RD | 54 | 0.885 | 0.736 | 0.798 | 5 | 5 of 13 | +0.078 [+0.027, +0.134] |

`sentinet` per attack and poison rate. Each row is a mean over the models of that attack at that rate and n counts them. The last 2 columns repeat the AUROC of PSBD-TM and PSBD-RD on the same models.

| attack | poison rate | n | AUROC | TPR at 10% FPR | TPR at 20% FPR | PSBD-TM AUROC | PSBD-RD AUROC |
|---|---|---|---|---|---|---|---|
| BPP | 1% | 4 | 0.615 | 0.130 | 0.335 | 0.946 | 0.914 |
| BPP | 5% | 4 | 0.438 | 0.102 | 0.292 | 0.941 | 0.962 |
| BPP | 10% | 4 | 0.353 | 0.038 | 0.066 | 0.957 | 0.959 |
| BadNets | 1% | 4 | 0.546 | 0.335 | 0.415 | 0.987 | 0.542 |
| BadNets | 5% | 4 | 0.458 | 0.237 | 0.280 | 0.992 | 0.796 |
| BadNets | 10% | 4 | 0.493 | 0.280 | 0.307 | 0.996 | 0.798 |
| Blend | 1% | 4 | 0.326 | 0.001 | 0.003 | 0.973 | 0.944 |
| Blend | 5% | 4 | 0.478 | 0.127 | 0.238 | 0.987 | 0.970 |
| Blend | 10% | 4 | 0.488 | 0.040 | 0.066 | 0.976 | 0.997 |
| LF | 1% | 4 | 0.298 | 0.012 | 0.035 | 0.963 | 0.959 |
| LF | 5% | 4 | 0.374 | 0.052 | 0.199 | 0.986 | 0.978 |
| LF | 10% | 4 | 0.488 | 0.001 | 0.069 | 0.990 | 0.985 |
| TaCT | 1% | 1 | 0.025 | 0.006 | 0.009 | 0.979 | 0.464 |
| TaCT | 5% | 2 | 0.038 | 0.002 | 0.003 | 0.954 | 0.613 |
| WaNet | 5% | 1 | 0.325 | 0.011 | 0.037 | 0.930 | 0.953 |
| WaNet | 10% | 2 | 0.358 | 0.012 | 0.034 | 0.704 | 0.956 |

Mean AUROC per dataset. The shared 2000-image clean split gives about 200 images per class on CIFAR-10, 46 on GTSRB, 20 on CIFAR-100 and 10 on Tiny ImageNet, which is the budget every class-conditional method fits on.

| defense | CIFAR-10 | CIFAR-100 | GTSRB | Tiny ImageNet |
|---|---|---|---|---|
| `sentinet` | 0.326 (15) | 0.762 (12) | 0.217 (13) | 0.408 (14) |
| PSBD-TM | 0.919 (15) | 0.979 (12) | 0.984 (13) | 0.977 (14) |
| PSBD-RD | 0.831 (15) | 0.890 (12) | 0.855 (13) | 0.965 (14) |

Measured cost, median over the compared models. Seconds per 1000 inputs divide the scoring time of the clean and backdoor splits by their size. The fit is the one-off pass over the clean validation split before any input is scored, and a dash marks a detector with no fit. The device is the one most records name.

| detector | forward passes per input | seconds per 1000 inputs | fit seconds | precision | device |
|---|---|---|---|---|---|
| `sentinet` | 202 | 74.38 | 149.3 | bfloat16 | NVIDIA A100-SXM4-40GB |

<!-- results:end -->

## Reading the results

SentiNet ranks last and falls below chance on most panel models, with its lowest readings on TaCT and the single SIG model. That is an inverted result rather than merely a weak one. The reason is only partly established.

The established part is that the map misses the trigger. The detector smoke of 2026-09-10 (`docs/runs/2026-09-10-detector-smoke.md`) hooked the class-activation map at every block of a GTSRB BadNets ViT that follows its patch trigger on every triggered image, and found the thresholded mask covering almost none of the trigger's pixels at any depth. The transplanted region carries part of the object, not the trigger, so the transplant cannot fool clean images the way the method needs. This explains why SentiNet does not detect the trigger. It explains a reading near 0.5, not one below it, which `docs/open-questions.md` records as Q23.

A candidate explanation for the sign follows from how `fooled` is counted, and it has not been measured. For a clean input, the cut-out region is part of an object of the predicted class, and pasting it onto clean images sometimes carries that class, so `fooled` is above 0. For a triggered input, the prediction is the target class, but the cut-out region is part of the source-class object, which rarely makes a clean image look like the target, so `fooled` sits near 0. Triggered inputs would then sit below clean ones in the plane, which the one-sided score reads as the wrong direction. TaCT and SIG, the attacks whose trigger depends most on the image content, would show it most strongly. Testing this needs the raw `fooled` and `avg_conf` of the clean and backdoor splits, which `sentinet_statistics` computes but the records do not store.

## Known failure modes

A trigger spread over the whole image, Blend, WaNet, SIG, LF and BPP on this panel, has no compact region to find, so these attacks are expected failures by design, as the paper's own Section VI concedes for large objects. A flat map is the benign failure: when a clean image's class evidence covers most of the image, the mask covers most of it too. The envelope then has to absorb that. The paper's Section V-B names an adaptive attacker who trains the inert pattern itself to keep the target label. An attention-rollout mask in place of the class-activation map is the natural ViT variant and is listed here as optional, not built.

## How to run and where records land

```bash
python -m cli.baselines --checkpoint-folder <folder> --detectors sentinet --max-samples 200 \
    --results-dir scratch/detector_smoke/results --allow-missing-psbd-cache
```

The smoke above writes outside `results/`. The panel runs through the `sentinet` job group of `pbs/generate_detector_jobs.py`, 1 job per model, because SentiNet's runtime dwarfs the cheap detectors. `results/<folder>/detectors/sentinet_metrics.json` holds the report and provenance, and `sentinet_scores_{validation,clean,backdoor}.pt` hold the negated residuals in loader order, the validation tensor being the in-sample residuals of deviation 7.
