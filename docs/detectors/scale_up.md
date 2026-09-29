# SCALE-UP, scaled prediction consistency

SCALE-UP multiplies every pixel of a suspicious input by a ladder of integer factors and counts how often the amplified copies keep the label the model gave the original. 2 variants are registered and both are documented here. `scale_up` is the paper's data-free method. `scale_up_data_limited` standardizes the same statistic against clean statistics of the predicted class. This page explains the idea, gives the paper's equations, describes the paper's setting and the released implementations, walks through the port in `detectors/scale_up.py`, lists every deviation with its reason and ends with both variants' results on the panel. Terms such as the shared split, the panel, PSBD-TM, PSBD-RD, AUROC and TPR at a false-positive budget are defined once in `README.md` in this directory.

## The idea in plain words

Multiplying every pixel of an image by 3, 5 or 11 and clipping at the maximum brightness washes most of the image out to flat saturated regions. An ordinary image loses the fine structure its class rests on, so the model's label on the amplified copy tends to change. A trigger is usually a strong, high-contrast pattern because it has to survive compression, cropping and lighting in deployment. That same robustness carries it through amplification. A triggered image therefore keeps being classified as the target at every amplification factor. The statistic counts the share of amplified copies whose label agrees with the original, and a high share is evidence of a trigger.

The method is black-box in a strict sense: it needs only the model's predicted label, not its probabilities, weights or gradients. The data-free variant needs no clean data at all. The data-limited variant adds a correction for the fact that some classes are naturally more robust to amplification than others: it compares an input's consistency to the consistency clean images of the same predicted class show, so a class that is always consistent does not look suspicious.

## The original method

Guo et al., "SCALE-UP: An Efficient Black-box Input-level Backdoor Detection via Analyzing Scaled Prediction Consistency", ICLR 2023, arXiv:2302.03251. The data-free statistic is Section 4.2, Equation (2). The data-limited variant is Section 4.3, Equations (3) and (4).

$$
\begin{aligned}
SPC(x) &= \frac{\sum_{n \in S} \mathbb{I}\{ C(n \cdot x) = C(x) \}}{|S|} && \text{(2)} \\
\mu_i &= \mathbb{E}_{x \in X_i}[ SPC(x) ], \quad \sigma_i = \sqrt{ \mathbb{E}_{x \in X_i}\big[ (SPC(x) - \mu_i)^2 \big] } && \text{(3)} \\
NSPC(x) &= \frac{ SPC(x) - \mu_{\hat{y}} }{ \sigma_{\hat{y}} }, \quad \hat{y} = C(x) && \text{(4)}
\end{aligned}
$$

| symbol | meaning |
|---|---|
| $x$ | the input image, pixel values in $[0, 1]$ |
| $S$ | the defender's set of integer scaling factors |
| $n$ | 1 scaling factor from $S$ |
| $n \cdot x$ | the image with every pixel multiplied by $n$, constrained to $[0, 1]$ |
| $C(\cdot)$ | the deployed classifier's predicted label |
| $\mathbb{I}\{\cdot\}$ | the indicator function, 1 when the condition holds |
| $SPC(x)$ | scaled prediction consistency, the share of amplified copies whose label equals $C(x)$ |
| $X_i$ | the defender's benign images of class $i$ |
| $\mu_i, \sigma_i$ | the mean and standard deviation of clean $SPC$ within class $i$ |
| $\hat{y}$ | the predicted label of the query, the class it is standardized against |
| $NSPC(x)$ | the standardized consistency, a z-score of $SPC(x)$ against class $\hat{y}$'s clean statistics |

The decision rule is "backdoor if $SPC(x) > T$" (or $NSPC(x) > T$ for the data-limited variant) at a defender-chosen threshold $T$. The descriptive form renames without rederiving.

$$
\begin{aligned}
\text{spc}(\text{image}) &= \text{share of amplified copies whose predicted label equals the label predicted on the image} \\
\text{nspc}(\text{image}) &= \frac{ \text{spc}(\text{image}) - \text{clean\_mean}[\text{predicted class}] }{ \text{clean\_std}[\text{predicted class}] }
\end{aligned}
$$

The paper introduces the scaling set with "e.g." as $S = \{3, 5, 7, 9, 11\}$ and states that $n \cdot x$ is constrained to $[0, 1]$ during the multiplication. Its Theorem 1 proves, for an RBF kernel regressor, that as the number of poisoned samples approaches the number of benign ones the amplified triggered image keeps the target label for every $n \geq 1$.

## The original paper's setting

The paper evaluates on CIFAR-10 and Tiny ImageNet with ResNet, against BadNets, Label-Consistent, PhysicalBA, TUAP, WaNet and ISSBA, each poisoned heavily enough for an attack success rate of at least 98%. Its evaluation set pairs an attacked test set with a benign test set and adds an augmented copy of each with random noise of magnitude 0.05. The data-limited variant assumes 100 benign images per class. The paper reports average AUROC over the 6 attacks of 0.928 (data-free) and 0.933 (data-limited) on CIFAR-10, and 0.905 and 0.909 on Tiny ImageNet (Tables 1 and 2), and names WaNet as the attack where it beats the other black-box baselines by the widest margin. It evaluates no transformer.

## The reference implementations

The authors release their code at https://github.com/JunfengGo/SCALE-UP, pinned in `third_party.lock` at commit `a2c6d06`, the head of its only branch, since no commit was recorded when the port was written. `torch_model_wrapper.py` adds $0.02 \cdot U[0, 1)$ noise to every input (line 41) and records the predicted label at every factor in `range(1, 12)` (lines 55 to 58). `test.py`'s `process` then scores each row as `np.mean(a[i] == a[i][0])`, so the consistency is measured against the prediction on the noisy input at factor 1. That column counts as 1 agreement out of 11. 2 third-party ports were read directly when the port was written, `third_party/BackdoorBox/core/defenses/SCALE_UP.py` at commit `af3afd1` and `third_party/backdoor-toolbox/other_defenses_tool_box/scale_up.py` at commit `9d4d909`. Both use the paper's set $\{3, 5, 7, 9, 11\}$.

backdoor-toolbox denormalizes before multiplying and renormalizes after clipping, `self.normalizer(torch.clip(self.denormalizer(clean_img) * scale, 0.0, 1.0))`, which keeps the multiply and the clip in pixel space as Section 4.2 states. BackdoorBox has the same denormalize and renormalize calls commented out and multiplies the loader's tensor directly, `torch.clip(clean_img * scale, 0.0, 1.0)`, which is only correct when the loader serves $[0, 1]$ pixels. Both fit Eq. (3) by comparing amplified predictions against the clean images' ground-truth labels rather than against $C(x)$, both fit 1 pooled mean and standard deviation over all classes rather than 1 pair per class (BackdoorBox `init_spc_norm`, backdoor-toolbox lines 188 and 189). Both drop every scored image whose prediction disagrees with its dataset label before reporting a number. The quotes were checked against the pinned checkouts on 2026-09-29.

## The port step by step

**Data-free, `scale_up`.**

1. `_build_scale_up` in `detectors/__init__.py` fits nothing and returns a closure over `scale_up_scores(model, loader, device, mean, std, PAPER_SCALES, use_bfloat16)`.
2. `scale_up_scores` calls `spc_scores`, which iterates the loader. For each batch of shape (batch, channels, height, width), `normalization_buffers` from `detectors/strip.py` builds (1, channels, 1, 1) mean and std tensors.
3. 1 unamplified forward pass through `defenses.inference.forward_probs` gives probabilities of shape (batch, num_classes). Their argmax is $C(x)$, shape (batch,).
4. For each factor $n$ in `PAPER_SCALES` = (3, 5, 7, 9, 11), `amplify_pixels(images, n, mean, std)` denormalizes to pixels, clamps to $[0, 1]$, multiplies by $n$, clamps to $[0, 1]$ again and renormalizes, returning the same shape. 1 forward pass on the amplified batch gives its labels, and the count of labels equal to $C(x)$ accumulates in a (batch,) tensor.
5. The count divided by $|S| = 5$ is Eq. (2), shape (batch,). `spc_scores` returns 3 tensors of shape (N,): the SPC values, the predicted labels $C(x)$ and the labels the loader served.
6. `scale_up_scores` negates the SPC values and returns them as float32, low meaning poisoned.

**Data-limited, `scale_up_data_limited`.**

1. `_build_scale_up_data_limited` requires the validation loader and `context.num_classes`. It runs `spc_scores` once on the 2000-image validation split, giving validation SPC, predicted labels and true labels.
2. `fit_class_spc_statistics(validation_spc, validation_labels, num_classes)` computes Eq. (3) per true class, returning 2 tensors of shape (num_classes,). A class with fewer than `MIN_CLASS_SAMPLES` (5) validation images takes the pooled mean and standard deviation over the whole split. A class whose few samples all share 1 SPC value, so that $\sigma_i = 0$, keeps the pooled standard deviation instead. Every $\sigma_i$ is floored at `STD_FLOOR`, $10^{-6}$.
3. For the validation split itself, `cross_fitted_validation_scores` splits the 2000 rows into `CROSS_FIT_FOLDS` (2) folds by position (row index modulo 2), fits Eq. (3) on 1 fold, standardizes the other with `standardize_spc` and negates, so every validation score is out of fit.
4. For the clean and backdoor splits, `scale_up_scores` runs `spc_scores` and then `standardize_spc(spc, predicted, class_means, class_stds)`, Eq. (4), indexing the statistics by the predicted label $\hat{y}$, and negates.

## Deviations and why

1. **Scaling set.** The port defaults to the paper's printed set $\{3, 5, 7, 9, 11\}$ as `PAPER_SCALES`. The authors' released code, as recorded above, uses $\{1, \ldots, 11\}$, which is a different statistic: $n = 1$ is always consistent, so it lifts the floor of SPC from 0 to $1/11$, and the average runs over 11 values instead of 5. `OFFICIAL_CODE_SCALES` names that set and is never used, so a number from it can never be mistaken for a paper number.
2. **Per-class statistics on the shared budget.** The paper budgets 100 benign images per class. The shared split gives about 200 per class on CIFAR-10, 46 on GTSRB, 20 on CIFAR-100 and 10 on Tiny ImageNet. A class the random split missed may get 0. Below `MIN_CLASS_SAMPLES` a class takes the pooled statistics, a case the paper's budget never reaches. The degenerate spread fallback exists because SPC takes only $|S| + 1 = 6$ distinct values, so a class whose few clean images all land on 1 value has $\sigma_i = 0$ exactly. Left alone that would clamp to $10^{-6}$ and produce z-scores of order $10^5$ for that class, which moves no AUROC but makes the score scale meaningless.
3. **The reference for consistency.** Eq. (2) always compares against $C(x)$, the model's own prediction. Both third-party ports compare against the ground-truth label when fitting Eq. (3), which differs exactly on the images the model gets wrong. The port uses $C(x)$ throughout, and groups images into $X_i$ by their true class, which is what Eq. (3) specifies for class membership.
4. **No input noise.** The released code's $0.02 \cdot U[0, 1)$ noise appears in no equation of the paper and has no ablation, so `amplify_pixels` does not add it.
5. **No prediction-correctness masking.** The third-party ports drop images whose prediction disagrees with their dataset label. This repository decides eligibility once, in `attacks.poisoning.AttackSuccessSet`, so the detector scores every image the loader serves and every detector here is scored on identical rows.
6. **Out-of-fit validation scores.** The paper fits Eq. (3) on a private clean pool and never reads a threshold from it. Here the same 2000-image split is both the fitting set and the threshold set, and a validation image standardized against statistics it helped fit sits closer to its class mean than a fresh image would. The threshold would be too tight and the realized false-positive rate on the paired clean split would overshoot the budget. The 2-fold cross-fit removes that bias. 2 folds keep every class's per-fold count as large as the shared split allows. The clean and backdoor splits are standardized against the statistics of the full 2000, since they never took part in the fit.
7. **The threshold rule.** The paper thresholds at a fixed $T$. The port hands the negated scores to `defenses.decision.detection_report`, which thresholds at a quantile of the validation scores. AUROC does not depend on this choice.

## Hyperparameters and where they come from

| symbol | paper | this port | constant | source of the port's value |
|---|---|---|---|---|
| $S$, scaling set | $\{3, 5, 7, 9, 11\}$, introduced with "e.g." | (3, 5, 7, 9, 11) | `PAPER_SCALES` | Section 4.2 |
| released code's set | not in the paper | (1, ..., 11), recorded and unused | `OFFICIAL_CODE_SCALES` | the port's record of the released code |
| minimum images per class | none, 100 per class assumed | 5 | `MIN_CLASS_SAMPLES` | below 5 a standard deviation of clean SPC is too noisy to divide by |
| $\sigma$ floor | none | $10^{-6}$ | `STD_FLOOR` | guards Eq. (4) against a class whose SPC never varies |
| cross-fit folds | none | 2 | `CROSS_FIT_FOLDS` | the largest per-fold class count that still holds every image out |
| input noise | none in the equations | none | none | deviation 4 |
| $T$ | defender-chosen | none, the quantile rule | `PSBD_QUANTILES` | the registry's shared rule |

## Cross-check against the reference

`tests/test_detectors_scale_up.py` runs on a tiny random classifier on the CPU, with dim inputs so that every factor crosses the clip.

1. With `OFFICIAL_CODE_SCALES`, `spc_scores` equals the authors' `process` on the factor loop of `torch_model_wrapper.py` (transcribed, since it runs at script level) to $10^{-7}$, without the input noise of deviation 4.
2. With the paper's set, the data-free SPC equals BackdoorBox's `_test` exactly, and on a normalized loader it equals BackdoorBox fed raw pixels through a model that normalizes itself, which is the pixel-space round trip Section 4.2 asks for.
3. BackdoorBox's `_test` returns only the inputs whose prediction equals their label, and the port's scores on those inputs equal it exactly while the port scores every input, deviation 5 asserted.
4. When every class has fewer than `MIN_CLASS_SAMPLES` validation images, the port's fit falls back to pooled statistics equal to BackdoorBox's `init_spc_norm`, and the standardized scores agree to $10^{-6}$. With some validation labels wrong BackdoorBox's pooled mean falls below the mean of the port's Eq. (2) SPC, and with 6 images per class the port's means differ by class where BackdoorBox keeps 1. Both departures are asserted.

The synthetic sign gate cannot judge either variant: `experiments/preflight/gate.py` lists both in `NOT_JUDGEABLE` because SPC takes 6 values and the fixture's barely trained model keeps almost every clean prediction stable, tying the clean population at the maximum. Run on the CPU on 2026-09-29, `python -m experiments.preflight.check_signs` printed 0.5117 for `scale_up` and 0.0234 for `scale_up_data_limited`, both marked not judged. The second value shows how a near-tied 6-valued statistic can read as a strong inversion once it is standardized, without saying anything about the method.

## Cost

$|S| + 1 = 6$ forward passes per input for both variants: 1 unamplified pass for $C(x)$ and 1 per factor. The data-limited variant adds a one-off fit of 6 passes over each of the 2000 validation images, so `scale_up_data_limited` is in `NEEDS_FITTING` and `scale_up` is not. The measured wall-clock cost is in the results block.

## Direction

Low is poisoned. The paper's rule is "backdoor if $SPC(x) > T$", so $SPC$ and $NSPC$ are high for poisoned. `scale_up_scores` negates once at the return, and `cross_fitted_validation_scores` applies the same single negation to the out-of-fit validation scores, so the 2 cannot disagree in sign.

## Results

<!-- results:begin -->
Generated by `python scripts/detector_doc_results.py` at commit `b2d32cf11708d5de965d3e13604c863a3ad9b493-dirty`. It reads `results/coverage/coverage.json`, every `results/<folder>/detectors/<name>_metrics.json` and every `results/<folder>/psbd_metrics.json` of the 54 backdoored ViT-B/16 models the paper's detector comparison uses, the successful models that carry a reading from every defense. PSBD-TM and PSBD-RD are read at the adaptive rate rule, every threshold is the clean-validation quantile named in the column and AUROC is the one-sided area at the 0.25 quantile, where a value under 0.5 means inverted.

Summary over every compared model. The rank is among the 13 defenses of the comparison by mean AUROC, and the last column is PSBD-TM minus the defense, paired per model, with its 95% bootstrap interval over models (5000 resamples, seed 0).

| defense | models | AUROC | TPR at 10% FPR | TPR at 20% FPR | models below chance | rank | PSBD-TM minus defense, AUROC |
|---|---|---|---|---|---|---|---|
| `scale_up` | 54 | 0.728 | 0.340 | 0.500 | 9 | 10 of 13 | +0.235 [+0.175, +0.298] |
| `scale_up_data_limited` | 54 | 0.636 | 0.350 | 0.510 | 17 | 12 of 13 | +0.327 [+0.253, +0.402] |
| PSBD-TM | 54 | 0.963 | 0.887 | 0.916 | 1 | 1 of 13 | reference |
| PSBD-RD | 54 | 0.885 | 0.736 | 0.798 | 5 | 5 of 13 | +0.078 [+0.027, +0.134] |

`scale_up` per attack and poison rate. Each row is a mean over the models of that attack at that rate and n counts them. The last 2 columns repeat the AUROC of PSBD-TM and PSBD-RD on the same models.

| attack | poison rate | n | AUROC | TPR at 10% FPR | TPR at 20% FPR | PSBD-TM AUROC | PSBD-RD AUROC |
|---|---|---|---|---|---|---|---|
| BPP | 1% | 4 | 0.406 | 0.076 | 0.111 | 0.946 | 0.914 |
| BPP | 5% | 4 | 0.392 | 0.004 | 0.090 | 0.941 | 0.962 |
| BPP | 10% | 4 | 0.399 | 0.002 | 0.079 | 0.957 | 0.959 |
| BadNets | 1% | 4 | 0.966 | 0.748 | 0.999 | 0.987 | 0.542 |
| BadNets | 5% | 4 | 0.970 | 0.750 | 1.000 | 0.992 | 0.796 |
| BadNets | 10% | 4 | 0.969 | 0.500 | 1.000 | 0.996 | 0.798 |
| Blend | 1% | 4 | 0.785 | 0.403 | 0.435 | 0.973 | 0.944 |
| Blend | 5% | 4 | 0.823 | 0.390 | 0.587 | 0.987 | 0.970 |
| Blend | 10% | 4 | 0.851 | 0.464 | 0.643 | 0.976 | 0.997 |
| LF | 1% | 4 | 0.701 | 0.260 | 0.344 | 0.963 | 0.959 |
| LF | 5% | 4 | 0.778 | 0.361 | 0.569 | 0.986 | 0.978 |
| LF | 10% | 4 | 0.724 | 0.302 | 0.470 | 0.990 | 0.985 |
| TaCT | 1% | 1 | 0.653 | 0.131 | 0.279 | 0.979 | 0.464 |
| TaCT | 5% | 2 | 0.987 | 0.411 | 0.454 | 0.954 | 0.613 |
| WaNet | 5% | 1 | 0.500 | 0.055 | 0.117 | 0.930 | 0.953 |
| WaNet | 10% | 2 | 0.558 | 0.147 | 0.202 | 0.704 | 0.956 |

`scale_up_data_limited` per attack and poison rate. Each row is a mean over the models of that attack at that rate and n counts them. The last 2 columns repeat the AUROC of PSBD-TM and PSBD-RD on the same models.

| attack | poison rate | n | AUROC | TPR at 10% FPR | TPR at 20% FPR | PSBD-TM AUROC | PSBD-RD AUROC |
|---|---|---|---|---|---|---|---|
| BPP | 1% | 4 | 0.270 | 0.013 | 0.067 | 0.946 | 0.914 |
| BPP | 5% | 4 | 0.210 | 0.003 | 0.010 | 0.941 | 0.962 |
| BPP | 10% | 4 | 0.250 | 0.001 | 0.003 | 0.957 | 0.959 |
| BadNets | 1% | 4 | 0.974 | 0.749 | 0.999 | 0.987 | 0.542 |
| BadNets | 5% | 4 | 0.966 | 0.750 | 1.000 | 0.992 | 0.796 |
| BadNets | 10% | 4 | 0.907 | 0.750 | 0.750 | 0.996 | 0.798 |
| Blend | 1% | 4 | 0.721 | 0.287 | 0.511 | 0.973 | 0.944 |
| Blend | 5% | 4 | 0.744 | 0.380 | 0.592 | 0.987 | 0.970 |
| Blend | 10% | 4 | 0.758 | 0.306 | 0.628 | 0.976 | 0.997 |
| LF | 1% | 4 | 0.691 | 0.441 | 0.528 | 0.963 | 0.959 |
| LF | 5% | 4 | 0.657 | 0.250 | 0.586 | 0.986 | 0.978 |
| LF | 10% | 4 | 0.684 | 0.329 | 0.610 | 0.990 | 0.985 |
| TaCT | 1% | 1 | 0.181 | 0.004 | 0.133 | 0.979 | 0.464 |
| TaCT | 5% | 2 | 0.938 | 0.845 | 0.911 | 0.954 | 0.613 |
| WaNet | 5% | 1 | 0.312 | 0.058 | 0.125 | 0.930 | 0.953 |
| WaNet | 10% | 2 | 0.335 | 0.060 | 0.154 | 0.704 | 0.956 |

Mean AUROC per dataset. The shared 2000-image clean split gives about 200 images per class on CIFAR-10, 46 on GTSRB, 20 on CIFAR-100 and 10 on Tiny ImageNet, which is the budget every class-conditional method fits on.

| defense | CIFAR-10 | CIFAR-100 | GTSRB | Tiny ImageNet |
|---|---|---|---|---|
| `scale_up` | 0.784 (15) | 0.758 (12) | 0.639 (13) | 0.723 (14) |
| `scale_up_data_limited` | 0.563 (15) | 0.641 (12) | 0.817 (13) | 0.544 (14) |
| PSBD-TM | 0.919 (15) | 0.979 (12) | 0.984 (13) | 0.977 (14) |
| PSBD-RD | 0.831 (15) | 0.890 (12) | 0.855 (13) | 0.965 (14) |

Measured cost, median over the compared models. Seconds per 1000 inputs divide the scoring time of the clean and backdoor splits by their size. The fit is the one-off pass over the clean validation split before any input is scored, and a dash marks a detector with no fit. The device is the one most records name.

| detector | forward passes per input | seconds per 1000 inputs | fit seconds | precision | device |
|---|---|---|---|---|---|
| `scale_up` | 6 | 2.37 | -- | bfloat16 | NVIDIA A100-SXM4-40GB |
| `scale_up_data_limited` | 6 | 2.37 | 4.9 | bfloat16 | NVIDIA A100-SXM4-40GB |

<!-- results:end -->

## Reading the results

Both variants are near the bottom of the ranking, and the data-limited variant ranks below the data-free one, the opposite of the paper's ordering. The per-attack tables show why. Both variants are strong on BadNets, whose compact high-contrast patch survives amplification exactly as the paper's mechanism says. Both fall below chance on BPP, and the data-limited variant also on WaNet and on the 1% TaCT model. BPP and WaNet are low-amplitude triggers, a quantized color depth and a smooth warp, that amplification destroys along with the image, so a triggered BPP image becomes less consistent than its clean twin. The paper's WaNet result on ResNet does not carry over to these ViT models.

The standardization hurts because of the class budget. With about 20 validation images per class on CIFAR-100 and 10 on Tiny ImageNet, the per-class means and standard deviations of a 6-valued statistic are very noisy, and dividing by a noisy standard deviation reorders images of different predicted classes arbitrarily. The per-dataset table shows the data-limited variant ahead only on GTSRB, and GTSRB is not the dataset with the most images per class (CIFAR-10 is), so the budget alone does not explain the whole pattern.

## Known failure modes

A high SPC says the prediction survived amplification, which a naturally saturated or overexposed clean image also produces. SPC lives on a grid of 6 values, so the statistic is coarse by construction and many images tie, which is worth reading beside the AUROC as `tie_share_at_threshold` in the record. The paper's own adaptive attack (Eq. 5) adds a term that pulls the amplified predictions of poisoned images away from consistency during training, and `docs/attack-design/cross-defense.md` reports that it pushes SCALE-UP below chance at a clean-accuracy and attack-success cost the paper does not report. The data-limited variant is also coupled to the model's clean behavior: an attack or training change that widens the clean spread inflates $\sigma_i$ and compresses every $NSPC$ toward 0 (`docs/attack-design/A1-operating-point-and-threshold.md`).

## How to run and where records land

```bash
python -m cli.baselines --checkpoint-folder <folder> --detectors scale_up scale_up_data_limited \
    --max-samples 500 --results-dir scratch/detector_smoke/results --allow-missing-psbd-cache
```

The smoke above writes outside `results/`. The panel runs through the `cheap` job group of `pbs/generate_detector_jobs.py`. `results/<folder>/detectors/scale_up_metrics.json` and `scale_up_data_limited_metrics.json` hold each variant's report and provenance, and `<name>_scores_{validation,clean,backdoor}.pt` hold the negated scores in loader order. The data-limited validation tensor holds the cross-fitted, out-of-fit scores.
