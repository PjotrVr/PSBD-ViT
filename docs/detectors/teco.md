# TeCo, test-time corruption robustness consistency

TeCo corrupts each input with a suite of common image corruptions at 5 increasing severities, records for each corruption type the lowest severity at which the prediction first changes, and scores the input by how spread out those breaking points are. This page explains the idea, gives the paper's algorithm, describes the paper's setting and the released code, walks through the port in `detectors/teco.py`, lists every deviation with its reason and ends with the results on the panel. Terms such as the shared split, the panel, PSBD-TM, PSBD-RD, AUROC and TPR at a false-positive budget are defined once in `README.md` in this directory.

## The idea in plain words

Take a clean image and blur it a little, then more, then more. At some severity the model stops recognizing it. Do the same with noise, with fog, with JPEG compression, with a brightness change. For a clean image the evidence erodes at roughly similar rates under each of these, so the prediction breaks at similar severities and the breaking points cluster. A trigger behaves differently: its survival depends on what the trigger is. A small patch survives brightness and contrast changes and is destroyed by blur or pixelation. A low-frequency blend survives blur and is destroyed by high-frequency noise. A triggered image's breaking points therefore scatter, some at severity 1 and some never. TeCo measures that scatter with a standard deviation, and a large spread is evidence of a trigger.

The method is black-box in the same strict sense as SCALE-UP: it needs only predicted labels. The score needs no clean data at all, since each input is compared only with its own uncorrupted prediction.

## The original method

Liu et al., "Detecting Backdoors During the Inference Stage Based on Corruption Robustness Consistency", CVPR 2023, arXiv:2303.18191. The score is Algorithm 1 of Section 4.2, which carries no numbered equation. The decision rule is Equation (4).

```
Algorithm 1, test-time corruption robustness consistency (TeCo)
input: test sample x, model C_theta, deviation measure Dev,
       corruption set D_k^n for k = 1..K types and n = 1..N severities
    P_org <- C_theta(x)
    L <- {}
    for k = 1..K:
        l <- N + 1
        for n = 1..N:
            if C_theta( D_k^n(x) ) != P_org:
                l <- n
                break
        L <- L union {l}
    TeCo(x) = Dev(L)

    Gamma( TeCo(x) ) = 1 if TeCo(x) > gamma, else 0            Eq. (4)
```

| symbol | meaning |
|---|---|
| $x$ | the input image |
| $C_{\theta}$ | the classifier's predicted label |
| $P_{org}$ | the label predicted on the uncorrupted image |
| $K$ | the number of corruption types |
| $N$ | the number of severities per type |
| $D_k^n(x)$ | corruption type $k$ applied to $x$ at severity $n$ |
| $l$ | the lowest severity of type $k$ that changes the prediction, or $N + 1$ if none does |
| $L$ | the $K$ breaking points of $x$, 1 per corruption type |
| $Dev$ | the dispersion measure over $L$, the standard deviation in the paper |
| $\gamma$ | the decision threshold |
| $\Gamma$ | the decision, 1 meaning the input carries a trigger |

The descriptive form renames without rederiving.

$$
\begin{aligned}
\text{hardness}[k] &= \text{lowest severity of corruption } k \text{ at which the prediction stops matching the uncorrupted one, or } N + 1 \\
\text{teco\_score}(\text{image}) &= \text{standard deviation of hardness over the } K \text{ corruption types}
\end{aligned}
$$

## The original paper's setting

The corruption set is the 15 common corruptions of Hendrycks and Dietterich's benchmark (noise, blur, weather and digital families), each at severities 1 to 5. The paper evaluates on CIFAR-10, CIFAR-100, GTSRB and Tiny ImageNet with PreActResNet-18 and MobileViT-xs, and on a 200-class ImageNet subset with WideResNet-101-2 and Swin Transformer-Base fine-tuned from ImageNet-1K. It is therefore the only competitor here whose paper already includes transformer backbones. It reports AUROC and the best F1 over all thresholds, and its Table 5 evaluates a single empirical threshold $\gamma = 1$ across all attacks. The supplementary compares dispersion measures and finds the standard deviation and the mean absolute deviation close, with the coefficient of variation and the quartile deviation worse.

## The reference implementation

No repository by TeCo's own authors is vendored. `third_party/BackdoorBench/detection_infer/teco.py` at commit `f02e353` is the reference the port was checked against by reading. It imports the `imagecorruptions` package (line 64), which is not installed in this project and would pull in `opencv-python` as a further dependency. Its corruption loop (lines 235 to 256) sets `x = images_poison` and then overwrites `x[i]` in place at every severity and every corruption type without resetting it, so severity 2 is applied to the output of severity 1, and the first severity of the second corruption type to an image that has already been through all 5 severities of the first. Its dispersion variable is named `mad` but is computed as `np.std(indexs)` (line 340), the population standard deviation. Each corrupted image passes through `imagecorruptions.corrupt`, whose last line casts the float result with `np.uint8` (`__init__.py` line 69), a truncation to the 8-bit level below. The package is pinned in `third_party.lock` as `third_party/imagecorruptions` at commit `c959e65`. Its corruption formulas are those of the 1.1.2 release BackdoorBench would install, apart from skimage's renamed `channel_axis` argument, a numba-compiled glass-blur shuffle with the same statement and a float32 plasma map. The line numbers were read from the pinned checkouts on 2026-09-29.

## The port step by step

1. `_build_teco` in `detectors/__init__.py` fits nothing and returns a closure over `teco_scores(model, loader, device, mean, std, context.teco_corruptions, MAX_SEVERITY, use_bfloat16, seed)`. `context.teco_corruptions` defaults to `DEFAULT_CORRUPTIONS`, all 14 corruptions the port implements.
2. `hardness_thresholds` seeds with `seed_everything`, checks every requested corruption name against the `CORRUPTIONS` table and iterates the loader.
3. For each batch of shape (batch, channels, height, width), it denormalizes to pixels in $[0, 1]$ at the dataset's native resolution, then runs 1 unamplified forward pass whose argmax is $P_{org}$, shape (batch,).
4. It initializes a threshold table of shape (batch, K) at $N + 1 = 6$, the never-flipped sentinel.
5. For each corruption name, and for each severity from 1 to 5, it calls the corruption function on the pristine pixels, `corrupted = corrupt(pixels, severity)`, renormalizes, runs 1 forward pass and compares the argmax with $P_{org}$. Where the prediction moved and the table still holds the sentinel, it writes the severity. Every severity is evaluated even after a break, which is Algorithm 1's break applied to cached predictions and is what makes the pass batchable.
6. Each corruption function (`gaussian_noise`, `shot_noise`, `impulse_noise`, `defocus_blur`, `glass_blur`, `motion_blur`, `zoom_blur`, `snow`, `fog`, `brightness`, `contrast`, `elastic_transform`, `pixelate` and `jpeg_compression`) reimplements the `imagecorruptions` formula in torch, takes and returns (batch, channels, height, width) in $[0, 1]$ and rounds to the 8-bit grid through `_quantize`, because the reference works on uint8 arrays and several of its operators quantize by definition.
7. `deviation(thresholds)` returns `thresholds.std(dim=1, unbiased=False)`, shape (N,), the population standard deviation over the corruption types.
8. `teco_scores` negates the deviation once and returns float32 scores of shape (N,).

## Deviations and why

1. **14 corruptions, not 15.** `IMAGENET_C_CORRUPTIONS` names all 15 and `UNAVAILABLE_CORRUPTIONS` names `frost`, which composites 1 of 6 photographs of frosted glass that ship as binary files inside the `imagecorruptions` package and cannot be regenerated from a formula. $Dev$ is a standard deviation over the types, so dropping 1 of 15 changes the sample it is computed over, and a TeCo number from this repository is not numerically identical to a published one.
2. **Each corruption applied to the pristine image.** Algorithm 1 applies $D_k^n$ to $x$. The released code composes every corruption cumulatively, as described above. The port follows the algorithm. A faithful reproduction is therefore expected to differ from the published numbers, which the released statistic produced.
3. **Population standard deviation.** The port computes what the released code computes, `np.std` with `ddof=0`, rather than what its variable name suggests. The choice rescales every score by a constant and moves no AUROC, but it moves any absolute threshold such as the paper's $\gamma = 1$.
4. **Random angles drawn once per batch.** The motion-blur angle and the snow angle are drawn once per batch in `motion_blur` and `snow` rather than once per image. This is what makes the corruptions batchable, and sharing a nuisance angle within a batch removes a per-image random term from a statistic that compares corruption types within 1 image. Per-pixel noise in `gaussian_noise`, `shot_noise` and `impulse_noise` is still drawn independently per image.
5. **Operator substitutions.** Where the reference's dependency was missing, the port substitutes a torch operation: `_disk_kernel`'s antialiasing Gaussian replaces `cv2.GaussianBlur`, `pixelate` uses torch area resampling in place of PIL's box filter, `_clipped_zoom` reproduces `scipy.ndimage.zoom` and `elastic_transform` samples with torch's `grid_sample`. The cross-check below confirms the first and third to the 8-bit grid and finds that the second and fourth do not agree with the reference.
6. **Corruption at the native resolution.** Corruption happens in pixel space at 32 or 64 pixels, before the model's own `Resize` upsamples to 224, which is where this project stamps triggers too and what the reference does at CIFAR scale.
7. **The threshold rule.** The paper sweeps $\gamma$ and also evaluates a fixed $\gamma = 1$. The port hands the negated deviation to `defenses.decision.detection_report`, which thresholds at a quantile of the validation scores.

## Hyperparameters and where they come from

| symbol | paper | this port | constant | source of the port's value |
|---|---|---|---|---|
| $K$ | 15 | 14, `frost` dropped | `len(DEFAULT_CORRUPTIONS)` | deviation 1 |
| $N$ | 5 | 5 | `MAX_SEVERITY` | Section 5.1 |
| never-flipped value | $N + 1$ | 6 | `NEVER_FLIPPED` | Algorithm 1 |
| $Dev$ | standard deviation | population standard deviation | `deviation` | the released code's `np.std` |
| $\gamma$ | swept, also fixed at 1 | none, the quantile rule | `PSBD_QUANTILES` | the registry's shared rule |
| angles | per image | per batch | none | deviation 4 |
| batch size on the panel | not stated | 256 | `BATCH_SIZE_BY_GROUP["teco"]` in `pbs/generate_detector_jobs.py` | the 2026-09-10 smoke found the loop launch-bound and batch 256 faster at identical AUROC |
| reduced set for the sign gate | not applicable | `gaussian_noise`, `defocus_blur`, `brightness`, `contrast` | `experiments.preflight.gate.CHEAP_CORRUPTIONS` | a cheap sign check |

## Cross-check against the reference

`tests/test_detectors_teco.py` executes `imagecorruptions`' `corruptions.py` from the pinned checkout on 1 random 32 by 32 image at every severity. `cv2` and `pkg_resources` are not installed, so both are stubbed: the 3 OpenCV calls the compared corruptions make (`GaussianBlur`, `filter2D` with the default reflect-101 border and `cvtColor` to gray) are written out with scipy from OpenCV's documented formulas, independent of the port's torch code. The random corruptions receive the same draws on both sides. The criterion is the 8-bit grid: the reference returns a float and the port the nearest 8-bit level, so every port value must lie within half a level of the reference, plus $10^{-3}$ of a level.

1. `contrast`, `brightness`, `zoom_blur`, `defocus_blur` and `jpeg_compression` meet the criterion at every severity, and so does `fog` under a shared numpy seed.
2. `gaussian_noise`, `shot_noise`, `motion_blur` and `snow` meet it on the same noise, rates, flakes and angle.
3. `glass_blur` meets it when every displacement is 0, which checks its 2 blurs and the truncating cast between them.
4. `impulse_noise` draws inside skimage, so only its rates are compared: the shares of values set to 0 and to 255 on a mid-gray image agree within 0.02 at every severity.
5. BackdoorBench's scoring loop (`teco.py` lines 323 to 341), transcribed, returns the same deviation as `deviation` to $10^{-6}$ on the predictions the port's own hardness pass produced, so the index rule, the never-flipped value 6 and `np.std` agree.
6. The loop of lines 240 to 242, transcribed, composes severity 2 on the output of severity 1, deviation 2 asserted.

4 disagreements were not recorded before these tests ran.

- **Truncation.** `corrupt` truncates each corrupted image to uint8, and the port rounds. On `contrast` about half of all values land 1 level apart, every one with the port higher.
- **`glass_blur` swaps where the reference duplicates.** The reference's shuffle writes `x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]` on a (height, width, 3) array (`corruptions.py` line 166). Both right-hand sides are views, so the first assignment overwrites the pixel the second reads, and the pixel at the target position is copied without moving the other way. numba compiles the statement with the same semantics. The port clones both pixels and swaps them. On the same displacements most pixels then differ, by up to about 50 levels. With the reference's statement made into a real swap the 2 meet the criterion, so the swap is the whole difference.
- **`pixelate` does not match PIL.** PIL's BOX filter weights source pixels by fractional coverage and its NEAREST samples at pixel centers, where torch's area mode averages integer-bounded bins and its nearest samples at the floor of the scaled index. At the factors 0.5 and 0.25 only the rounding of the box mean differs, by 1 level at most. At 0.6, 0.4 and 0.3 whole blocks differ, by up to about 150 levels.
- **`elastic_transform` disagrees on a band, not only on the border.** The reference smooths its displacement field and samples the image in scipy's half-sample `reflect` mode, where the port pads and samples by reflecting about the edge pixel. The 2 agree exactly where the smoothing window and the displaced sample stay inside the image. At 32 pixels the displacement reaches 2 to 5 pixels, so between a seventh and a quarter of all pixels are outside that region, and there the gap reaches well over 100 levels.

The sign gate `python -m experiments.preflight.check_signs` also runs TeCo with the 4 corruptions of `CHEAP_CORRUPTIONS` on the synthetic fixture. Run on the CPU on 2026-09-29 it read TeCo at AUROC 0.9017, above the floor of 0.60.

## Cost

$K \times N + 1 = 71$ forward passes per input: 1 uncorrupted pass and 1 per corruption and severity. There is no fit. Several corruptions are CPU-bound, `glass_blur` through a per-pixel shuffle loop and `jpeg_compression` through PIL's encoder 1 image at a time, so wall-clock cost exceeds what the forward count suggests, and `pbs/generate_detector_jobs.py` prices the group at twice its forward count (`GROUP_SLOWDOWN["teco"]`). The measured seconds per input are in the results block.

## Direction

Low is poisoned. Eq. (4) flags $TeCo(x) > \gamma$, so the raw deviation is high for poisoned. `teco_scores` negates `deviation` once at the return.

## Results

<!-- results:begin -->
Generated by `python scripts/detector_doc_results.py` at commit `b2d32cf11708d5de965d3e13604c863a3ad9b493-dirty`. It reads `results/coverage/coverage.json`, every `results/<folder>/detectors/<name>_metrics.json` and every `results/<folder>/psbd_metrics.json` of the 54 backdoored ViT-B/16 models the paper's detector comparison uses, the successful models that carry a reading from every defense. PSBD-TM and PSBD-RD are read at the adaptive rate rule, every threshold is the clean-validation quantile named in the column and AUROC is the one-sided area at the 0.25 quantile, where a value under 0.5 means inverted.

Summary over every compared model. The rank is among the 13 defenses of the comparison by mean AUROC, and the last column is PSBD-TM minus the defense, paired per model, with its 95% bootstrap interval over models (5000 resamples, seed 0).

| defense | models | AUROC | TPR at 10% FPR | TPR at 20% FPR | models below chance | rank | PSBD-TM minus defense, AUROC |
|---|---|---|---|---|---|---|---|
| `teco` | 54 | 0.743 | 0.492 | 0.589 | 12 | 8 of 13 | +0.220 [+0.153, +0.288] |
| PSBD-TM | 54 | 0.963 | 0.887 | 0.916 | 1 | 1 of 13 | reference |
| PSBD-RD | 54 | 0.885 | 0.736 | 0.798 | 5 | 5 of 13 | +0.078 [+0.027, +0.134] |

`teco` per attack and poison rate. Each row is a mean over the models of that attack at that rate and n counts them. The last 2 columns repeat the AUROC of PSBD-TM and PSBD-RD on the same models.

| attack | poison rate | n | AUROC | TPR at 10% FPR | TPR at 20% FPR | PSBD-TM AUROC | PSBD-RD AUROC |
|---|---|---|---|---|---|---|---|
| BPP | 1% | 4 | 0.565 | 0.115 | 0.230 | 0.946 | 0.914 |
| BPP | 5% | 4 | 0.548 | 0.223 | 0.305 | 0.941 | 0.962 |
| BPP | 10% | 4 | 0.572 | 0.121 | 0.253 | 0.957 | 0.959 |
| BadNets | 1% | 4 | 0.965 | 0.907 | 0.966 | 0.987 | 0.542 |
| BadNets | 5% | 4 | 0.991 | 0.989 | 0.993 | 0.992 | 0.796 |
| BadNets | 10% | 4 | 0.971 | 0.933 | 0.972 | 0.996 | 0.798 |
| Blend | 1% | 4 | 0.899 | 0.712 | 0.824 | 0.973 | 0.944 |
| Blend | 5% | 4 | 0.930 | 0.765 | 0.905 | 0.987 | 0.970 |
| Blend | 10% | 4 | 0.891 | 0.584 | 0.819 | 0.976 | 0.997 |
| LF | 1% | 4 | 0.624 | 0.225 | 0.336 | 0.963 | 0.959 |
| LF | 5% | 4 | 0.386 | 0.071 | 0.127 | 0.986 | 0.978 |
| LF | 10% | 4 | 0.420 | 0.124 | 0.173 | 0.990 | 0.985 |
| TaCT | 1% | 1 | 0.599 | 0.098 | 0.246 | 0.979 | 0.464 |
| TaCT | 5% | 2 | 0.938 | 0.652 | 0.785 | 0.954 | 0.613 |
| WaNet | 5% | 1 | 0.927 | 0.841 | 0.889 | 0.930 | 0.953 |
| WaNet | 10% | 2 | 0.843 | 0.626 | 0.731 | 0.704 | 0.956 |

Mean AUROC per dataset. The shared 2000-image clean split gives about 200 images per class on CIFAR-10, 46 on GTSRB, 20 on CIFAR-100 and 10 on Tiny ImageNet, which is the budget every class-conditional method fits on.

| defense | CIFAR-10 | CIFAR-100 | GTSRB | Tiny ImageNet |
|---|---|---|---|---|
| `teco` | 0.677 (15) | 0.808 (12) | 0.655 (13) | 0.841 (14) |
| PSBD-TM | 0.919 (15) | 0.979 (12) | 0.984 (13) | 0.977 (14) |
| PSBD-RD | 0.831 (15) | 0.890 (12) | 0.855 (13) | 0.965 (14) |

Measured cost, median over the compared models. Seconds per 1000 inputs divide the scoring time of the clean and backdoor splits by their size. The fit is the one-off pass over the clean validation split before any input is scored, and a dash marks a detector with no fit. The device is the one most records name.

| detector | forward passes per input | seconds per 1000 inputs | fit seconds | precision | device |
|---|---|---|---|---|---|
| `teco` | 71 | 29.99 | -- | bfloat16 | NVIDIA A100-SXM4-40GB |

<!-- results:end -->

## Reading the results

TeCo separates the BadNets patch trigger well at every rate and Blend reasonably. It fails on BPP and falls below chance on LF at 5% and 10%. Both failures follow from the mechanism. BPP's trigger is a reduced color depth and LF's a smooth low-frequency pattern, and every corruption in the suite, noise, blur, compression, weather, destroys them at about the same rate it destroys ordinary content. A triggered image then breaks at clustered severities just like a clean one, or more uniformly than a clean one, which is the inversion. Its cost is 71 forward passes per input against 4 for PSBD-TM (1 unperturbed pass and $k = 3$ perturbed ones, `cli.sweep.DEFAULT_FORWARD_PASSES`), for a lower AUROC than PSBD-TM on every attack except WaNet and the single SIG model. WaNet is TeCo's relative strength, and TeCo is the only competitor above PSBD-TM on WaNet in the table. A likely reading, not measured here, is that the warp survives the photometric corruptions and breaks under the geometric ones, so its breaking points scatter.

## Known failure modes

A small spread says the prediction broke at similar severities under every corruption, which is the clean signature and also what a uniformly thin decision margin produces. The paper's own adaptive attack (Section 6, Eq. 8) trains the model to have the same corruption robustness on clean and triggered images. The paper reports that it lowers TeCo's AUROC substantially and at a large cost in clean accuracy and attack success rate, which `docs/attack-design/cross-defense.md` reads as a broken model rather than a deployable threat. The paper's supplementary also evaluates all-to-all attacks, on which TeCo drops. An all-to-all attack costs the attacker nothing extra to train.

## How to run and where records land

```bash
python -m cli.baselines --checkpoint-folder <folder> --detectors teco --max-samples 500 \
    --results-dir scratch/detector_smoke/results --allow-missing-psbd-cache
```

The smoke above writes outside `results/`. The panel runs through the `teco` job group of `pbs/generate_detector_jobs.py`, at batch size 256, because TeCo's cost dwarfs the cheap detectors and a shared job would be sized for the wrong detector. `results/<folder>/detectors/teco_metrics.json` holds the report and the provenance, whose hyperparameters list the corruptions actually used, and `teco_scores_{validation,clean,backdoor}.pt` hold the negated deviations in loader order.
