# STRIP, STRong Intentional Perturbation

STRIP lays each suspicious input over a fixed set of clean images and reads the entropy of the model's prediction on each blend. This page explains the idea, gives the paper's equations, describes the setting the paper evaluated in and the released code, walks through the port in `detectors/strip.py`, lists every deviation with its reason and ends with the results on the panel. Terms such as the shared split, the panel, PSBD-TM, PSBD-RD, AUROC and TPR at a false-positive budget are defined once in `README.md` in this directory.

## The idea in plain words

A clean image's class is carried by the arrangement of its own pixels. Add a second, unrelated photograph on top of it and that arrangement is scrambled, so the model's prediction on the blend wanders between classes from 1 blend to the next and the softmax output is spread out. A trigger is built to survive exactly this kind of interference, because in deployment it has to work on photographs the attacker never saw. A triggered image laid over a clean photograph therefore still carries the trigger, and the model still sends the blend to the target class with a peaked softmax. STRIP measures how spread out the prediction is with the Shannon entropy: high entropy means clean, low entropy means a trigger survived.

The method is black-box. It needs only the softmax output of the deployed model and never its weights or gradients. Beyond the model it needs a small pool of clean images to blend with.

## The original method

Gao et al., "STRIP: A Defence Against Trojan Attacks on Deep Neural Networks", ACSAC 2019, arXiv:1902.06531. The statistic is Section IV-D, Equations (2) to (4). For an incoming input $x$ the defender draws $N$ clean images and superimposes each on $x$, which gives the perturbed inputs $x^{p_1}, \ldots, x^{p_N}$. It then computes

$$
\begin{aligned}
H_n &= - \sum_{i=1}^{M} y_i \log_2 y_i && \text{(2)} \\
H_{sum} &= \sum_{n=1}^{N} H_n && \text{(3)} \\
H &= \frac{1}{N} H_{sum} && \text{(4)}
\end{aligned}
$$

| symbol | meaning |
|---|---|
| $x$ | the incoming input under test |
| $x^{p_n}$ | the $n$-th perturbed input, $x$ superimposed with the $n$-th clean image |
| $N$ | the number of clean images superimposed on each input |
| $M$ | the number of classes |
| $y_i$ | the softmax probability of class $i$ on 1 perturbed input |
| $H_n$ | the entropy of the prediction on the $n$-th perturbed input, in bits |
| $H_{sum}$ | the sum of the $N$ entropies |
| $H$ | the mean entropy, the STRIP statistic |

The descriptive form renames without rederiving.

$$
\begin{aligned}
\text{blend\_entropy}_n &= -\sum_c \text{prob}_c \log \text{prob}_c \quad \text{on the } n\text{-th blend} \\
\text{strip\_score} &= \frac{1}{N} \sum_{n=1}^{N} \text{blend\_entropy}_n
\end{aligned}
$$

The decision rule is a threshold on $H$. The defender fixes a false rejection rate, the share of clean inputs it is willing to reject, fits a normal distribution to the entropies of held-out clean inputs and uses the matching percentile as the detection boundary. An input whose $H$ falls below that boundary is flagged as trojaned.

## The original paper's setting

The paper evaluates on MNIST with a 2-convolution network, CIFAR-10 with an 8-convolution network and GTSRB resized to 32 by 32 with ResNet-20, against static patch triggers (a square, a heart shape and 2 image triggers), each trained to an attack success rate near 100%. The clean pool is the defender's held-out test data, and each input is blended with $N = 100$ clean images in the main experiments. A later section varies $N$ from 2 to 100 and reports $N = 10$ as sufficient. At a preset false rejection rate of 1% the paper reports a false acceptance rate below 1% on every tested trigger, and 0% on CIFAR-10 and GTSRB, where the minimum clean entropy it observed exceeded the maximum trojaned entropy.

All of these models are ConvNets on images seen at their native resolution. None is a transformer, and none of the triggers is a whole-image blend or a warp.

## The reference implementations

The authors' own code is not vendored. 4 reimplementations are pinned in `third_party.lock`, and they do not agree with each other.

The copy bundled with the Beatrix repository, `third_party/Beatrix/defenses/STRIP/STRIP.py` at commit `685827e`, is the one whose superimposition the port follows. Its `_superimpose` calls `cv2.addWeighted(background, 1, overlay, 1, 0)` at line 54, which adds 2 uint8 images with both weights 1 and lets OpenCV saturate the sum at 255 instead of wrapping or renormalizing it. That saturation is where the method's nonlinearity lives: a bright region laid over a bright region clips rather than doubling. Its entropy departs from Eq. (2). Line 69 applies `torch.sigmoid` to the logits and line 70 sums $-p \log_2 p$ over those per-class sigmoids, which is not the entropy of any distribution. It also draws its overlays afresh for every input with `np.random.randint` (line 62) rather than reusing 1 set.

`third_party/backdoor-toolbox/other_defenses_tool_box/strip.py` at commit `9d4d909` is the one whose statistic the port matches. `superimpose` (line 261) denormalizes both images, adds them with weight 1 (the value `other_defense.py` passes) and clamps to $[0, 1]$ before renormalizing (line 267). Its `entropy` (line 270) takes the softmax plus $10^{-8}$ in nats. Its `check` pairs input $i$ of a batch with image $i$ of each shuffled training batch, so an input's overlays depend on its position.

`third_party/BackdoorBench/detection_infer/strip.py` at commit `f02e353` (line 133) and `third_party/backdoor-toolbox/cleansers_tool_box/strip.py` (line 76) add the 2 normalized tensors with no clamp, the operation deviation 2 below rules out, and reshuffle the overlay pool for every call (lines 115 and 57). The line numbers were read from the pinned checkouts on 2026-09-29.

## The port step by step

1. `detectors.build_detector("strip", context)` calls `_build_strip` in `detectors/__init__.py`. It requires the shared validation loader and calls `collect_overlay_batch(validation_loader, STRIP_OVERLAYS, context.seed)`, which takes the first `STRIP_OVERLAYS` images of the validation split in the split's own order and returns them as 1 tensor of shape (8, channels, height, width), still normalized. `STRIP_OVERLAYS` is `strip_module.DEFAULT_NUM_OVERLAYS`, 8.
2. The returned closure calls `strip_scores(model, loader, device, overlays, mean, std, use_bfloat16, seed, 8)` for each split. The function seeds with `lightning.seed_everything`, puts the model in eval mode and moves the overlays to the device.
3. For each batch of inputs, shape (batch, channels, height, width), `normalization_buffers(mean, std, ...)` builds the dataset statistics as (1, channels, 1, 1) tensors. Both the inputs and the overlays are taken back to pixel space, `(images * std + mean).clamp(0, 1)`.
4. For each of the 8 overlays, the overlay is broadcast over the batch and added to the inputs in pixel space, then clamped to $[0, 1]$, which is `cv2.addWeighted`'s saturating sum computed in floating point. The blend is renormalized, `(superimposed - mean) / std`, and passed through `defenses.inference.forward_probs`, 1 forward pass returning softmax probabilities of shape (batch, num_classes). The model's own `Resize` upsamples the blend to 224 by 224 inside the forward pass.
5. `blend_entropy(probs)` computes $-\sum_c p_c \log p_c$ per row in nats, with every probability floored at `PROBABILITY_FLOOR` ($10^{-12}$) before the log so an underflowed class cannot produce $0 \cdot \log 0$. The 8 entropies are summed and divided by 8, Eq. (4), giving 1 score per input, shape (batch,).
6. The batches are concatenated into 1 float32 tensor of shape (N,) in loader order and returned unnegated.

## Deviations and why

1. **Entropy in nats rather than bits.** `blend_entropy` uses the natural log where Eq. (2) uses $\log_2$. The 2 differ by the constant factor $\ln 2$, so no ranking, AUROC or quantile position changes. Only the printed threshold value is scaled, and a reader comparing to a published entropy value should divide the port's value by $\ln 2$ first.
2. **Superimposition in pixel space with saturation.** The loaders serve normalized tensors, so the port denormalizes, adds, clamps to $[0, 1]$ and renormalizes, which is what the reference does on raw uint8 pixels. An earlier version of the port summed the 2 normalized tensors directly. That is a different operation: in normalized space $(p_1 - m)/s + (p_2 - m)/s = (p_1 + p_2 - 2m)/s$, which is the pixel-space sum displaced by a further $-m/s$ per channel, and it skips the saturation that makes the method nonlinear.
3. **A fixed overlay set of the first 8 validation images.** Every scored input is blended with the same 8 images, where every reference draws its overlays per input or per batch position. This removes overlay choice as a source of per-input variance, so 2 inputs differ in score only because they differ. Taking the images from the shared split keeps STRIP on the same data budget as every other detector.
4. **$N = 8$ rather than 100 or 10.** The paper runs 100 and reports 10 as sufficient. 8 is the value every STRIP number in this repository was produced with from the start, and it is kept so the ported numbers stay comparable to everything recorded before the port was formalized. It is a cost choice below the paper's own sufficient value, and a reader should expect a slightly noisier entropy estimate per input than the paper's.
5. **The threshold rule.** The paper fits a normal distribution to clean entropies and thresholds at a percentile of that fit. The port hands the raw entropies to `defenses.decision.detection_report`, which thresholds at an empirical quantile of the validation scores, the rule every detector here is read at. AUROC does not depend on this choice.

## Hyperparameters and where they come from

| symbol | paper | this port | constant | source of the port's value |
|---|---|---|---|---|
| $N$, overlays per input | 100, with 10 reported sufficient | 8 | `DEFAULT_NUM_OVERLAYS`, repeated as `STRIP_OVERLAYS` | the value every earlier STRIP number in the repository used |
| overlay source | the defender's held-out clean data | the first 8 images of the shared validation split | none | the shared data budget |
| entropy base | 2 | $e$ | none | a constant rescaling, deviation 1 |
| probability floor | not stated | $10^{-12}$ | `PROBABILITY_FLOOR` | guards $\log 0$ after softmax underflow |
| threshold | percentile of a normal fit to clean entropies | empirical quantile of validation scores | `PSBD_QUANTILES` | the registry's shared rule |

`tests/test_detector_registry.py::test_strip_overlays_has_1_source_and_is_still_8` pins the overlay count so the registry's cost table and a run's provenance cannot quote different values.

## Cross-check against the reference

`tests/test_detectors_strip.py` runs on a tiny random classifier on the CPU, with inputs on the 8-bit grid and a non-identity normalization.

1. backdoor-toolbox's own `check` loop, fed a loader whose round $i$ serves overlay $i$ to every input, agrees with `strip_scores` to $10^{-5}$ in entropy. The 2 probability guards, a floor of $10^{-12}$ against an added $10^{-8}$, account for the tolerance.
2. The port's score of an image does not change with its position in the batch or the batch size, deviation 3 asserted.
3. Beatrix's `_get_entropy`, run with its overlay draw served in order and `cv2.addWeighted` written out from OpenCV's documented saturating formula (opencv-python is not installed), feeds the classifier exactly the composites the port feeds it, to $10^{-6}$, with both saturated and unsaturated sums present. This is deviation 2 checked numerically.
4. Beatrix's score equals the sigmoid form above to $10^{-5}$ and is not a constant multiple of the port's softmax entropy over 5 inputs, so the 2 can rank inputs differently. This disagreement was not recorded before the test ran.

The sign gate `python -m experiments.preflight.check_signs` also scores STRIP on the synthetic backdoored ViT of `experiments/preflight/synthetic.py` and requires an AUROC above `MINIMUM_AUROC`, 0.60. Run on the CPU on 2026-09-29 it read STRIP at 0.6560 and passed.

## Cost

8 forward passes per input and no backward pass. The fit is a single pass over the first 8 validation images to collect the overlays, so `strip` is absent from `NEEDS_FITTING`. The measured wall-clock cost per 1000 inputs is in the results block.

## Direction

Low is poisoned and the score is returned unnegated. STRIP's claim is that a triggered input keeps low entropy under superimposition, and low already means poisoned in the shared convention. Negating the entropy would produce an exactly inverted detector.

## Results

<!-- results:begin -->
Generated by `python scripts/detector_doc_results.py` at commit `b2d32cf11708d5de965d3e13604c863a3ad9b493-dirty`. It reads `results/coverage/coverage.json`, every `results/<folder>/detectors/<name>_metrics.json` and every `results/<folder>/psbd_metrics.json` of the 54 backdoored ViT-B/16 models the paper's detector comparison uses, the successful models that carry a reading from every defense. PSBD-TM and PSBD-RD are read at the adaptive rate rule, every threshold is the clean-validation quantile named in the column and AUROC is the one-sided area at the 0.25 quantile, where a value under 0.5 means inverted.

Summary over every compared model. The rank is among the 13 defenses of the comparison by mean AUROC, and the last column is PSBD-TM minus the defense, paired per model, with its 95% bootstrap interval over models (5000 resamples, seed 0).

| defense | models | AUROC | TPR at 10% FPR | TPR at 20% FPR | models below chance | rank | PSBD-TM minus defense, AUROC |
|---|---|---|---|---|---|---|---|
| `strip` | 54 | 0.857 | 0.706 | 0.782 | 1 | 6 of 13 | +0.106 [+0.066, +0.149] |
| PSBD-TM | 54 | 0.963 | 0.887 | 0.916 | 1 | 1 of 13 | reference |
| PSBD-RD | 54 | 0.885 | 0.736 | 0.798 | 5 | 5 of 13 | +0.078 [+0.027, +0.134] |

`strip` per attack and poison rate. Each row is a mean over the models of that attack at that rate and n counts them. The last 2 columns repeat the AUROC of PSBD-TM and PSBD-RD on the same models.

| attack | poison rate | n | AUROC | TPR at 10% FPR | TPR at 20% FPR | PSBD-TM AUROC | PSBD-RD AUROC |
|---|---|---|---|---|---|---|---|
| BPP | 1% | 4 | 0.712 | 0.567 | 0.637 | 0.946 | 0.914 |
| BPP | 5% | 4 | 0.928 | 0.832 | 0.889 | 0.941 | 0.962 |
| BPP | 10% | 4 | 0.937 | 0.857 | 0.907 | 0.957 | 0.959 |
| BadNets | 1% | 4 | 0.955 | 0.855 | 0.945 | 0.987 | 0.542 |
| BadNets | 5% | 4 | 0.974 | 0.936 | 0.989 | 0.992 | 0.796 |
| BadNets | 10% | 4 | 0.976 | 0.942 | 0.982 | 0.996 | 0.798 |
| Blend | 1% | 4 | 0.859 | 0.698 | 0.776 | 0.973 | 0.944 |
| Blend | 5% | 4 | 0.935 | 0.845 | 0.911 | 0.987 | 0.970 |
| Blend | 10% | 4 | 0.974 | 0.929 | 0.961 | 0.976 | 0.997 |
| LF | 1% | 4 | 0.726 | 0.440 | 0.576 | 0.963 | 0.959 |
| LF | 5% | 4 | 0.843 | 0.675 | 0.753 | 0.986 | 0.978 |
| LF | 10% | 4 | 0.872 | 0.713 | 0.795 | 0.990 | 0.985 |
| TaCT | 1% | 1 | 0.509 | 0.089 | 0.207 | 0.979 | 0.464 |
| TaCT | 5% | 2 | 0.729 | 0.284 | 0.437 | 0.954 | 0.613 |
| WaNet | 5% | 1 | 0.538 | 0.105 | 0.218 | 0.930 | 0.953 |
| WaNet | 10% | 2 | 0.518 | 0.113 | 0.221 | 0.704 | 0.956 |

Mean AUROC per dataset. The shared 2000-image clean split gives about 200 images per class on CIFAR-10, 46 on GTSRB, 20 on CIFAR-100 and 10 on Tiny ImageNet, which is the budget every class-conditional method fits on.

| defense | CIFAR-10 | CIFAR-100 | GTSRB | Tiny ImageNet |
|---|---|---|---|---|
| `strip` | 0.850 (15) | 0.876 (12) | 0.878 (13) | 0.831 (14) |
| PSBD-TM | 0.919 (15) | 0.979 (12) | 0.984 (13) | 0.977 (14) |
| PSBD-RD | 0.831 (15) | 0.890 (12) | 0.855 (13) | 0.965 (14) |

Measured cost, median over the compared models. Seconds per 1000 inputs divide the scoring time of the clean and backdoor splits by their size. The fit is the one-off pass over the clean validation split before any input is scored, and a dash marks a detector with no fit. The device is the one most records name.

| detector | forward passes per input | seconds per 1000 inputs | fit seconds | precision | device |
|---|---|---|---|---|---|
| `strip` | 8 | 3.14 | -- | bfloat16 | NVIDIA A100-SXM4-40GB |

<!-- results:end -->

## Reading the results

STRIP is the strongest of the black-box detectors on the panel, and the summary's below-chance column shows it inverting on almost no model. It is strongest on BadNets at every rate, where a compact high-contrast patch dominates any blend it is summed with, and weakest on WaNet (near chance) and on TaCT, whose trigger only works together with its source class's content. WaNet's trigger is a slight smooth warp of the whole image, and the likely reason, not measured here, is that the blended second image overwhelms it as it overwhelms clean evidence, so the triggered prediction scatters too. TaCT is the attack STRIP's own design cannot see: a blend with a clean image of another class removes the source content the trigger needs, so the triggered prediction scatters like a clean one. Across the panel it trails PSBD-TM by the paired gap the summary gives, and the interval excludes 0.

## Known failure modes

A low entropy says only that the prediction kept pointing at 1 class through the blend. That is also what a benign image with a very strong, saturating visual feature produces. The paper's own adaptive section (Section VI-F) adds an entropy-manipulation term to training that flattens STRIP's separation at a small clean-accuracy cost, and `docs/attack-design/cross-defense.md` calls that evasion close to free. Peng et al.'s under-confidence backdoor (arXiv:2202.11203) keeps the triggered prediction's margin thin and reports STRIP's false acceptance rate rising from near 0 to near 100% at a clean-accuracy cost near 0. Both attacks work on the margin rather than on the trigger's appearance, so a low STRIP AUROC on a model trained against margin suppression is the expected outcome.

STRIP's threshold is also coupled to the model's calibration. `docs/attack-design/A1-operating-point-and-threshold.md` notes that sharpening a model's confidence lowers the entropy of clean blends near the threshold and moves the same cut point the poisoned scores are compared against.

## How to run and where records land

```bash
python -m cli.baselines --checkpoint-folder <folder> --detectors strip --max-samples 500 \
    --results-dir scratch/detector_smoke/results --allow-missing-psbd-cache
```

The smoke above writes outside `results/` and needs `--allow-missing-psbd-cache` because the smoke tree holds no PSBD manifest to check against. The panel runs through the `cheap` job group of `pbs/generate_detector_jobs.py`. `results/<folder>/detectors/strip_metrics.json` holds the report at every quantile and the provenance, whose hyperparameters carry the overlay count, and `strip_scores_{validation,clean,backdoor}.pt` hold the raw entropies in loader order.
