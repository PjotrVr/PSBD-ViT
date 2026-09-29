# Beatrix, class-conditional Gram-matrix deviation

Beatrix reads the Gram matrix of a model's intermediate features at 1 layer and raises it to the orders 1 to $P$. It scores an input by how far its Gram entries fall outside a band fitted on the clean inputs the model puts in the same class. This page explains the idea, gives the paper's equations, describes the paper's setting and the authors' released code, walks through the port in `detectors/beatrix.py`, lists every deviation with its reason and ends with the results on the panel. Terms such as the shared split, the panel, PSBD-TM, PSBD-RD, AUROC and TPR at a false-positive budget are defined once in `README.md` in this directory.

## The idea in plain words

A model's internal representation of an image at some layer is a set of feature vectors, 1 per spatial position (1 per token on a transformer). The Gram matrix of that representation multiplies every feature dimension with every other and sums over the positions, so each entry says how strongly 2 features fire together across the image. Clean images of 1 class produce Gram matrices that vary within a characteristic range. A triggered image is sent to the target class through the backdoor, which fires feature combinations the target class's clean images never show, so its Gram entries land outside the target class's range. Beatrix fits that range per class from a few clean images, a median plus or minus 10 median absolute deviations for every Gram entry, and scores an input by the average relative amount its entries stick out of the band of the class it was predicted as.

Raising the features to higher powers before forming the Gram (orders 2, 3 and 4) weights the largest activations more heavily, which captures the outlying activations a trigger tends to cause. The method is white-box because it reads an intermediate layer. It costs 1 forward pass per input plus some matrix algebra.

## The original method

Ma et al., "The 'Beatrix' Resurrections: Robust Backdoor Detection via Gram Matrices", NDSS 2023, arXiv:2209.11715 (v3 read). Feature modeling is Section IV-A, Eq. (8) and (9). The deviation measurement is Section IV-B, Eq. (10) to (14), with the threshold determination paragraph at the end of that section. The order bound and the clean budget are fixed in Section V-A. The offline test that names infected classes (RMMD, Eq. (15) to (19)) is outside the port.

$$
\begin{aligned}
G &= v v^{T} && \text{(8)} \\
G^{p} &= \left( v^{p} \, {v^{p}}^{T} \right)^{1/p} && \text{(9)} \\
s &= \left[ \vec{G^{1}}, \ \vec{G^{2}}, \ \dots, \ \vec{G^{P}} \right] \in \mathbb{R}^{\frac{1}{2} n (n+1) P} \\
\tilde{s}_{j} &= \operatorname{median}\left( \left\{ s_{ij} : i = 1, \ldots, |\mathcal{X}_{t}| \right\} \right) && \text{(10)} \\
MAD_{j} &= \operatorname{median}\left( \left\{ \left| s_{ij} - \tilde{s}_{j} \right| : i = 1, \ldots, |\mathcal{X}_{t}| \right\} \right) && \text{(11)} \\
\delta_{j} &= \begin{cases} 0 & \text{if } min \le \hat{s}_{j} \le max \\[4pt] \dfrac{min - \hat{s}_{j}}{min} & \text{if } \hat{s}_{j} < min \\[4pt] \dfrac{\hat{s}_{j} - max}{max} & \text{if } \hat{s}_{j} > max \end{cases} && \text{(12, 13)} \\[4pt]
\delta &= \frac{2}{n(n+1)P} \sum_{j=1}^{\frac{1}{2} n (n+1) P} \delta_{j} && \text{(14)}
\end{aligned}
$$

with $min = \tilde{s}_{j} - k \cdot MAD_j$ and $max = \tilde{s}_{j} + k \cdot MAD_j$ as the paper writes them under Eq. (13).

| symbol | meaning |
|---|---|
| $v \in \mathbb{R}^{n \times m}$ | the feature representation of an input at layer $l$, $n$ channels by $m$ spatial positions |
| $G \in \mathbb{R}^{n \times n}$ | the Gram matrix of $v$, channel against channel, summed over positions |
| $v^{p}$ | the elementwise $p$-th power of $v$ |
| $G^{p}$ | the $p$-th order Gram matrix, rooted back by $1/p$ |
| $P$ | the order bound, 4 in the paper's main experiments |
| $\vec{G^{p}}$ | the upper triangle of $G^{p}$ with its diagonal, $\frac{1}{2} n (n+1)$ entries |
| $s$ | the concatenated Gram feature vector of an input |
| $\mathcal{X}_{t}$ | the clean reference images of class $t$ |
| $s_{ij}$ | entry $j$ of the feature vector of clean reference $i$ |
| $\tilde{s}_{j}$, $MAD_j$ | the median and the median absolute deviation of entry $j$ over the class's references |
| $k$ | the band scale factor, 10 |
| $min$, $max$ | the band edges of entry $j$ |
| $\hat{s}$, $\hat{s}_{j}$ | the query's feature vector and its entry $j$ |
| $\delta_{j}$ | the relative excess of entry $j$ outside its band |
| $\delta$ | the query's deviation, the mean of $\delta_{j}$ over every entry |

The class $t$ is the query's predicted class. The descriptive form, with the token matrix in place of the channel-by-position map, renames without rederiving.

$$
\begin{aligned}
\text{gram}_{p} &= \operatorname{sign}(M_p) \, |M_p|^{1/p}, \quad M_p = (\text{tokens}^{p})^{T} \, \text{tokens}^{p} \\
\text{features} &= \left[ \operatorname{triu}(\text{gram}_{1}), \ \dots, \ \operatorname{triu}(\text{gram}_{P}) \right] \\
\text{deviation} &= \text{mean over entries of} \left( \frac{\operatorname{relu}(\text{lower}_{j} - \text{features}_{j})}{|\text{lower}_{j}|} + \frac{\operatorname{relu}(\text{features}_{j} - \text{upper}_{j})}{|\text{upper}_{j}|} \right)
\end{aligned}
$$

The band is a median and a MAD rather than a mean and a standard deviation because the defender has only a few references per class, and Section IV-B argues that a Gaussian fit on that few is dragged by any outlier. The paper flags an input whose $\delta$ exceeds a percentile of the benign deviations, and fixes that percentile from clean data alone with a $T$-fold jackknife.

## The original paper's setting

The paper evaluates on CIFAR-10 and GTSRB with PreActResNet-18, on a 100-class VGGFace subset with VGG-16 and on a 100-class ImageNet subset with ResNet-101, mainly against the input-aware dynamic backdoor, with further attacks in its robustness section. The defender holds 30 clean images per class, and Figure 6 shows 8 are enough. Section V-A starts from $P = 9$, finds the false positive rate stable from $P \ge 4$ and fixes $P = 4$ for the remaining experiments. No transformer is evaluated.

## The reference implementation

The authors' code is https://github.com/wanlunsec/Beatrix at commit `685827e`, vendored as `third_party/Beatrix` when the port was written. The method is the class `Feature_Correlations` at `defenses/Beatrix/Beatrix.py:307`, the validation jackknife is `threshold_determine` at line 388 and the driver is `BEAT_detector` at line 409. The driver hooks the input of `layer4` of a PreActResNet-18, runs the clean and poisoned test sets once, stores the feature maps with the argmax prediction, keeps the first 30 clean images of each predicted class as the defender's set, jackknifes them in 5 contiguous blocks to set the threshold and then scores the rest.

`Feature_Correlations.G_p` raises the feature map to the power $p$, forms the (channels, channels) Gram, zeroes the strict lower triangle, applies `sign(.) * abs(.) ** (1 / p)` (the released reading of Eq. (9)'s root, which is undefined for a negative entry at odd $p$) and flattens the whole matrix, zeros included. The band is `median ± 10 * MAD` per entry, and the deviation sums `relu(min - g) / abs(min + 1e-6)` and `relu(g - max) / abs(max + 1e-6)` over every entry and order and divides by `channels * channels / 2 * len(power)`. 2 settings differ from the paper: `order_list = np.arange(1, 9)` runs orders 1 to 8, and the divisor is $\frac{1}{2} n^2 P$ rather than Eq. (14)'s $\frac{1}{2} n(n+1) P$. The BackdoorBench copy at `third_party/BackdoorBench/detection_pretrain/beatrix.py` omits the line `temp = temp**p`, so every order it computes is the first-order Gram under a $1/p$ root, and the port follows the authors' file. `third_party/` is not checked out in this working tree, so these line numbers are the ones recorded when the port was written.

## The port step by step

1. `_build_beatrix` in `detectors/__init__.py` requires the validation loader and `context.num_classes`. It picks the feature layer with `default_feature_layer(model)`, which reads `FEATURE_LAYER_BY_ARCHITECTURE`: layer 9 on ViT-B/16 (the output of block 9 of 12) and layer 22 on Swin-S (the output of block 22 of 24, the last block of its third stage).
2. `collect_reference_tokens(model, validation_loader, device, layer, use_bfloat16)` runs 1 forward pass over the 2000 validation images inside `analysis.features.captured_layers`, which hooks the residual stream at that layer. `captured_token_matrix` flattens a Swin grid to tokens with `as_token_sequence` and casts to float16. It returns the token bank, shape (2000, tokens, dim) on the CPU, which is (197, 768) per image on ViT and (196, 384) on Swin. It also returns the predicted labels from the same pass, shape (2000,). The loader's labels are never read.
3. `fit_class_bands(tokens, predicted, num_classes, PAPER_POWERS, device)` moves 1 predicted class at a time to the device. `gram_features` computes, in batches of `GRAM_BATCH_SIZE` (64), the Gram contracted over the token axis for $p$ = 1 to 4 with the signed root and keeps the upper triangles, $\frac{1}{2} \cdot 768 \cdot 769 = 295296$ entries per order on ViT and 1181184 in total. `fit_gram_band` takes the per-entry median and MAD and returns the lower and upper band vectors. A class with fewer than `MIN_CLASS_SAMPLES` (5) references takes the pooled band over every reference, fitted in column chunks of up to `FIT_CHUNK_BYTES` (1 GiB) by `fit_band_over_tokens`. The result is a `ClassBands` record holding the bands and the per-class counts.
4. For the validation split, `jackknife_deviations` cuts the 2000 references into `JACKKNIFE_FOLDS` (5) contiguous blocks by position, fits bands on 4 blocks and scores the held-out block with `deviations_of_tokens`, so every validation score is out of fit. `deviation_scores` negates.
5. For the clean and backdoor splits, `beatrix_scores(model, loader, device, layer, bands, use_bfloat16)` captures each batch's tokens the same way, computes its Gram features, gathers the lower and upper bands of each image's predicted class and computes `gram_deviation`, Eq. (13) and (14) with the released $10^{-6}$ guard in the denominators. `deviation_scores` negates the result, shape (N,).

## Deviations and why

1. **Order bound 4.** The port runs the paper's main-experiment orders `PAPER_POWERS` = (1, 2, 3, 4) and keeps the released code's orders 1 to 8 as the unused `OFFICIAL_CODE_POWERS`. There is also a range reason on a ViT. A Gram entry sums 197 products of 2 $p$-th powers, and ViT-B/16's residual stream carries a few dimensions with values in the hundreds, so a value of 1000 contributes $10^{24}$ at $p = 4$ and $10^{48}$ at $p = 8$, past the $3.4 \times 10^{38}$ float32 holds. `gram_features` raises on any non-finite entry rather than letting an infinite band pass every query.
2. **Feature site and Gram axis.** The released code hooks the input of `layer4`, about 3 quarters of the way through the network. The port reads the residual stream at a similar relative depth, block 9 of 12 on ViT and block 22 of 24 on Swin, the last point before Swin's final patch merge. The Gram contracts over the tokens and is (dim, dim), the analogue of the paper's channel Gram, whose $m$ spatial positions are the contracted axis. A Gram over the 197 tokens would correlate positions instead, which the paper never does. The class token stays in, as every spatial position stays in on a ConvNet. `DetectorContext.beatrix_layer` overrides the layer, which the 2-block synthetic fixture needs.
3. **Grouping by predicted label.** Section IV-A compares a query with the class its predicted label names, and the released driver groups clean references by the model's argmax too. The port does the same, so a reference the model misclassifies joins the band of the class it was put in. On a model with high clean accuracy the contamination is small.
4. **Reference budget and the pooled fallback.** The paper budgets 30 images per class. The shared split gives about 200 per class on CIFAR-10, 46 on GTSRB, 20 on CIFAR-100 and 10 on Tiny ImageNet, spread unevenly by the random split and the model's predictions. A median and MAD of 4 values bound nothing, so a class with fewer than 5 predicted references takes the pooled band, which spans every class and is therefore wide. On Tiny ImageNet the bands rest on a third of the paper's budget and some classes pool, so a Tiny number from this port is a data-limited one.
5. **Normalization.** The port divides by $\frac{1}{2} n (n+1) P$ as Eq. (14) writes it, where the released code divides by $\frac{1}{2} n^2 P$. Every port score is the released score times $\frac{n}{n+1}$, 768/769 on ViT, a constant that moves no ranking.
6. **Jackknife for the validation split.** The released code cuts its 30 references into 5 contiguous blocks and drops any remainder. The port cuts the 2000 into 5 blocks whose boundaries fall at $i \cdot N / 5$, so every reference is scored exactly once. The cut is contiguous because the shared split is already a random permutation served in order, so a contiguous fifth is a random fifth. A class with 5 or 6 references can pool inside a fold while holding its own band in the full fit.
7. **Precision and memory.** The paper and the code are float32. The port runs the forward under the shared autocast policy and stores captured tokens as float16 on both the reference and the query side, so a band and the query it judges are rounded the same way. Every Gram is computed in float32 after that cast, since a half-precision matmul over 197 terms accumulates with too few mantissa bits. Gram vectors are never stored for a whole split, since 1 image's vector is 1181184 floats and 2000 of them would be 9.4 GB. The token bank, 2000 by 197 by 768 halves, is about 605 MB on ViT.
8. **Relative epsilon.** Eq. (13) divides by $min$ and $max$. The port follows the released code and divides by $|min + 10^{-6}|$ and $|max + 10^{-6}|$ through `RELATIVE_EPSILON`, which keeps a band edge at exactly 0 from dividing by 0 and is what the bit-level test compares against.
9. **Threshold rule.** The paper thresholds at a percentile of the jackknifed benign deviations. The port hands the negated deviations to `defenses.decision.detection_report`, which applies the same kind of rule, a quantile of the out-of-fit validation scores, at the registry's budgets.

## Hyperparameters and where they come from

| symbol | paper | this port | constant | source of the port's value |
|---|---|---|---|---|
| $P$ | 4, after an ablation from 9 | orders (1, 2, 3, 4) | `PAPER_POWERS` | Section V-A |
| released orders | 1 to 8 | recorded, unused | `OFFICIAL_CODE_POWERS` | the released driver |
| $k$ | 10 | 10 | `MAD_BAND` | Section IV-B |
| denominator guard | none | $10^{-6}$ | `RELATIVE_EPSILON` | the released code |
| references per class | 30 | the shared split, about 10 to 200 by dataset | `PAPER_CLEAN_PER_CLASS`, recorded only | the shared data budget |
| minimum references per class | not stated | 5, pooled band below | `MIN_CLASS_SAMPLES` | a median and MAD of fewer than 5 values bound nothing |
| $T$, jackknife folds | "$T$ iterations", 5 in the code | 5 | `JACKKNIFE_FOLDS` | the released code |
| $l$, feature layer | input of `layer4` | block 9 of 12 on ViT, 22 of 24 on Swin | `FEATURE_LAYER_BY_ARCHITECTURE` | the same relative depth as the released hook |
| Gram batch | the whole set at once | 64 token matrices | `GRAM_BATCH_SIZE` | device memory |
| band fit chunk | the whole set at once | 1 GiB of float32 columns | `FIT_CHUNK_BYTES` | device memory |

## Cross-check against the reference

`tests/test_detectors_beatrix.py::test_matches_the_official_feature_correlations_to_1e_5` cuts the class `Feature_Correlations` out of `third_party/Beatrix/defenses/Beatrix/Beatrix.py` with `ast`, runs it and the port on the same random feature maps and requires the deviations to agree to $10^{-5}$ after the constant $\frac{n}{n+1}$ of deviation 5. `third_party/` is not checked out in this working tree, so the test skipped when the suite ran on 2026-09-29. The other tests in the file run: the Gram features by hand on a 2 by 2 token matrix, the signed root, the band on a class-structured bank, the pooled fallback, the overflow guard at order 8, the jackknife against a fold refitted by hand and a Swin grid capture. The synthetic sign gate `python -m experiments.preflight.check_signs`, run on the CPU on 2026-09-29 with the feature layer set to 1 for the fixture's 2-block model, read Beatrix at AUROC 0.9090, above the floor of 0.60. The fixture's model is randomly initialized, so that reading says the Gram statistic registers a fixed pixel patch on random features. That is a property of the patch, so the reading confirms only the plumbing and the sign.

## Cost

1 forward pass per input plus the Gram algebra, 4 products of a (768, 197) by a (197, 768) matrix per input. The fit is 1 pass over the 2000 validation images plus the Gram of every reference once for the full bands and once per jackknife fold. `beatrix` is in `NEEDS_FITTING` and in `CROSS_FITTED`. The measured seconds per input and fit seconds are in the results block.

## Direction

High deviation is poisoned in the paper, the opposite of the shared convention. `deviation_scores` negates once, and both the jackknifed validation scores and the scores of every other split pass through that 1 function, so the 2 cannot disagree in sign. `beatrix_deviations` and `jackknife_deviations` return the unnegated statistic, comparable to a published deviation.

## Results

<!-- results:begin -->
Generated by `python scripts/detector_doc_results.py` at commit `19488b81358b06040ba96f361d5061b2981f1f25-dirty`. It reads `results/coverage/coverage.json`, every `results/<folder>/detectors/<name>_metrics.json` and every `results/<folder>/psbd_metrics.json` of the 57 backdoored ViT-B/16 models the paper's detector comparison uses, the clearing models that carry a reading from every defense. PSBD-TM and PSBD-RD are read at the adaptive rate rule, every threshold is the clean-validation quantile named in the column and AUROC is the one-sided area at the 0.25 quantile, where a value under 0.5 means inverted.

Summary over every compared model. The rank is among the 13 defenses of the comparison by mean AUROC, and the last column is PSBD-TM minus the defense, paired per model, with its 95% bootstrap interval over models (5000 resamples, seed 0).

| defense | models | AUROC | TPR at 10% FPR | TPR at 20% FPR | models below chance | rank | PSBD-TM minus defense, AUROC |
|---|---|---|---|---|---|---|---|
| `beatrix` | 57 | 0.890 | 0.655 | 0.715 | 2 | 3 of 13 | +0.062 [+0.011, +0.115] |
| PSBD-TM | 57 | 0.953 | 0.873 | 0.902 | 2 | 1 of 13 | reference |
| PSBD-RD | 57 | 0.888 | 0.744 | 0.805 | 5 | 4 of 13 | +0.065 [+0.012, +0.121] |

`beatrix` per attack and poison rate. Each row is a mean over the models of that attack at that rate and n counts them. The last 2 columns repeat the AUROC of PSBD-TM and PSBD-RD on the same models.

| attack | poison rate | n | AUROC | TPR at 10% FPR | TPR at 20% FPR | PSBD-TM AUROC | PSBD-RD AUROC |
|---|---|---|---|---|---|---|---|
| BPP | 1% | 4 | 0.903 | 0.497 | 0.652 | 0.946 | 0.914 |
| BPP | 5% | 4 | 0.929 | 0.758 | 0.785 | 0.941 | 0.962 |
| BPP | 10% | 4 | 0.947 | 0.731 | 0.808 | 0.957 | 0.959 |
| BadNets | 1% | 4 | 0.954 | 0.746 | 0.781 | 0.987 | 0.542 |
| BadNets | 5% | 4 | 0.907 | 0.500 | 0.526 | 0.992 | 0.796 |
| BadNets | 10% | 4 | 0.888 | 0.750 | 0.750 | 0.996 | 0.798 |
| Blend | 1% | 4 | 0.969 | 0.747 | 0.808 | 0.973 | 0.944 |
| Blend | 5% | 4 | 0.902 | 0.749 | 0.750 | 0.987 | 0.970 |
| Blend | 10% | 4 | 0.888 | 0.714 | 0.748 | 0.976 | 0.997 |
| LF | 1% | 4 | 0.881 | 0.681 | 0.720 | 0.963 | 0.959 |
| LF | 5% | 4 | 0.862 | 0.584 | 0.737 | 0.986 | 0.978 |
| LF | 10% | 4 | 0.947 | 0.658 | 0.758 | 0.990 | 0.985 |
| SIG | 10% | 1 | 0.992 | 0.985 | 0.993 | 0.418 | 0.919 |
| TaCT | 1% | 1 | 0.997 | 0.994 | 1.000 | 0.979 | 0.464 |
| TaCT | 5% | 2 | 0.999 | 0.994 | 1.000 | 0.954 | 0.613 |
| WaNet | 5% | 2 | 0.471 | 0.168 | 0.248 | 0.933 | 0.955 |
| WaNet | 10% | 3 | 0.636 | 0.193 | 0.330 | 0.786 | 0.957 |

Mean AUROC per dataset. The shared 2000-image clean split gives about 200 images per class on CIFAR-10, 46 on GTSRB, 20 on CIFAR-100 and 10 on Tiny ImageNet, which is the budget every class-conditional method fits on.

| defense | CIFAR-10 | CIFAR-100 | GTSRB | Tiny ImageNet |
|---|---|---|---|---|
| `beatrix` | 0.962 (17) | 0.965 (12) | 0.975 (14) | 0.655 (14) |
| PSBD-TM | 0.891 (17) | 0.979 (12) | 0.981 (14) | 0.977 (14) |
| PSBD-RD | 0.844 (17) | 0.890 (12) | 0.862 (14) | 0.965 (14) |

Measured cost, median over the compared models. Seconds per 1000 inputs divide the scoring time of the clean and backdoor splits by their size. The fit is the one-off pass over the clean validation split before any input is scored, and a dash marks a detector with no fit. The device is the one most records name.

| detector | forward passes per input | seconds per 1000 inputs | fit seconds | precision | device |
|---|---|---|---|---|---|
| `beatrix` | 1 | 0.71 | 13.5 | bfloat16 | NVIDIA A100-SXM4-40GB |

<!-- results:end -->

## Reading the results

Beatrix is the third-ranked defense on the panel, behind PSBD-TM and the calibrated IBD-PSC, and on CIFAR-10 it has the highest mean of all 13 defenses, PSBD-TM included. The per-dataset table carries the most important fact about it: its Tiny ImageNet mean is far below its mean on the other 3 datasets. Tiny ImageNet has 200 classes and about 10 references per class in the shared split, so the bands are read off very few images, many classes fall back to the wide pooled band, and a triggered input predicted into a pooled class deviates little. The likely cause is the budget rather than the architecture. The paper's own 30-per-class budget would need a clean pool 3 times the shared split. Per attack, Beatrix separates TaCT and the single SIG model almost perfectly, and falls to chance on WaNet, whose warp moves every token a little and leaves the second moments of the target class largely intact.

## Known failure modes

A high deviation says the input's Gram entries sit outside the band of its predicted class, which a clean input whose representation is unusual for its class also produces, and on a weak model that includes the inputs it misclassifies. The paper's adaptive attack, a Gram-matching loss at $\lambda = 1$, lowers Beatrix's true positive rate at 1% false positives substantially on CIFAR-10, so an attacker who controls training can close the gap the method reads. The massive-activation dimensions of the ViT residual stream carry Gram entries near the top of float32's range at $p = 4$, and a layer index off by 1 or 2 changes which dimensions those are. A large share of clean scores sits at exactly 0, every entry inside its band, so the score floor is a tie and any quantile above the nonzero share lands the threshold on that tie. `tie_share_at_threshold` in the record surfaces it.

## How to run and where records land

```bash
python -m cli.baselines --checkpoint-folder <folder> --detectors beatrix --max-samples 500 \
    --results-dir scratch/detector_smoke/results --allow-missing-psbd-cache
```

The smoke above writes outside `results/`. With 500 validation images a 43-class dataset leaves under 12 references per class, so a smoke reading of Beatrix at 500 images says little about the full split, which the 2026-09-10 smoke found directly. The panel runs through the `cheap` job group of `pbs/generate_detector_jobs.py`. `results/<folder>/detectors/beatrix_metrics.json` holds the report and provenance, and `beatrix_scores_{validation,clean,backdoor}.pt` hold the negated deviations in loader order, the validation tensor being the jackknifed out-of-fit scores.
