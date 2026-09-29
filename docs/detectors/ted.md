# TED, topological evolution dynamics

TED stores a bank of clean images with their activations at every layer. For each input it sorts the bank by distance to the input at each layer and records how far down the sorted bank the first image of the input's predicted class sits. That sequence of ranks over depth is the input's trajectory, and an outlier model fitted on the bank's own trajectories scores it. This page explains the idea, gives the paper's algorithm, describes the paper's setting and the released notebook, walks through the port in `detectors/ted.py`, lists every deviation with its reason and ends with the results on the panel. Terms such as the shared split, the panel, PSBD-TM, PSBD-RD, AUROC and TPR at a false-positive budget are defined once in `README.md` in this directory.

## The idea in plain words

Follow an image through the network and, at every layer, ask which stored clean images are its nearest neighbors. A clean image of class "cat" is surrounded by cat images at almost every depth: its nearest neighbor of its predicted class is near the front of the list everywhere, so its rank is small and steady. A triggered image of a dog that the backdoor sends to "cat" looks like a dog for most of the network, because the backdoor only takes over late. In early and middle layers its nearest cat is far down the list, and only near the end does it jump among the cats. The trajectory of ranks over depth therefore has a different shape for triggered images: large early, small late. An outlier model fitted on the trajectories of clean images flags that shape.

The rank at a layer does not change if every activation at that layer is multiplied by the same positive number, so the statistic is immune to the growth of activation scale along a transformer's residual stream. The method is white-box because it reads every layer. It costs 1 forward pass per input plus 1 distance computation against the bank per layer.

## The original method

Mo et al., "Robust Backdoor Detection for Deep Learning via Topological Evolution Dynamics", IEEE S&P 2024, arXiv:2312.02673 (v1 read). The statistic is Section V-B and Algorithm 2, which carries no numbered equation, so lines of the algorithm are cited.

```
Given a c-class model f with N considered layers, m samples per class,
a metric d, a PCA outlier model PCA(., alpha) with reject rate alpha
and a test set X_test.

 2   S_1 = S_2 = ... = S_c = {}
 3   for i = 1 to c:
 4       for j = 1 to m:
 5           x = random_sample(X_i)
 6           stack x in S_i
 7           [h_l(x)]_{l=1}^{N} = forward x through f
 8   for i = 1 to |S|:
 9       j = argmax_{k in [1, c]} f(x_i)_k
10       for l = 1 to N:
11           S_sorted = sort_by_distance(d, h_l(.), S, x_i)
12           x_nn     = get_nearest_neighbor(d, h_l(.), S_j - x_i, x_i)
13           K_l^(i)  = get_rank(S_sorted, x_nn)
14       record [K_l^(i)]_{l=1}^{N}
15   M   = PCA({[K_l^(i)]_{l=1}^{N}}_{i=1}^{|S|}, alpha)
16   tau = M.get_detect_threshold(alpha)
17   X_malicious = {}
18   for x in X_test:
19       if M(x) > tau:
20           add x to X_malicious
21   return X_malicious
```

| symbol | meaning |
|---|---|
| $f$ | the classifier, $f(x)_k$ its output for class $k$ |
| $c$ | the number of classes |
| $N$ | the number of considered layers |
| $m$ | the stored clean images per class |
| $X_i$ | the clean images of class $i$ |
| $S_i$, $S$ | the stored bank rows of class $i$, and their union |
| $h_l(x)$ | the representation of $x$ at layer $l$ |
| $d$ | the distance between representations, Euclidean in the paper |
| $j$ | the predicted class of the image being ranked |
| $S_j - x_i$ | the bank rows of class $j$ without $x_i$ itself |
| $K_l^{(i)}$ | the rank of the nearest class-$j$ row in the bank sorted by distance to $x_i$ at layer $l$ |
| $M$ | the PCA outlier model fitted on the bank's trajectories |
| $\alpha$ | the reject rate that fixes the threshold $\tau$ |

The descriptive form renames without rederiving.

```
bank        = clean images the model classifies correctly, with their
              features at every considered layer and their predicted labels
for each query:
    predicted   = the label the model gives the query
    for each layer:
        order   = bank sorted by Euclidean distance to the query's features
                  at this layer, the query's own row left out if it has 1
        rank    = position in order of the first row whose predicted label
                  equals predicted
    trajectory  = rank at every layer
outlier_model   = fitted on the bank's own trajectories
ted_score       = outlier_model(trajectory), flagged when above a threshold
```

## The original paper's setting

The defender of Section III-B has white-box access, a small labeled clean set and no knowledge of the attack. The adversary of Section III-A controls training and mounts source-specific dynamic triggers, the paper's strongest setting. The paper evaluates on MNIST, CIFAR-10, GTSRB, PubFig and ImageNet-100 with ConvNets. It also evaluates BERT on text, where it hooks the dense, self-attention and embedding layers. The default layer set at the head of Section VI is every Conv2D output, with ReLU and Linear outputs added on shallow networks, and Table VIII of Section VI-D reports similar AUC across layer sets on a deep network. The stored bank is 20 images per class on CIFAR-10 and MNIST, 1000 in total on GTSRB and 200 per class on PubFig and ImageNet-100. Appendix A sets $\alpha = 5\%$ and Appendix D offers a 4-sigma Z-score as an alternative outlier rule. No vision transformer is evaluated.

## The reference implementation

The authors' code is https://github.com/tedbackdoordefense/ted at commit `fa193a6`, a single notebook `TED.ipynb`, vendored as `third_party/ted` when the port was written. Cell 4 builds the bank by keeping the defense images the model classifies correctly, capped at `DEFENSE_TRAIN_SIZE` (1000 on CIFAR-10, GTSRB and MNIST). Cell 9 hooks every non-pointwise `Conv2d`, every `ReLU` and every `Linear`. Cell 10 flattens each hooked activation with `.view(batch, -1)` and stacks the bank as float32 on the device. Cell 14 sorts the bank by `pairwise_euclidean_distance` to each query, walks the predicted labels in that order and records `.index(label)`, dropping the first sorted entry with `ranking_array[1:]` when ranking a bank row against the bank so it does not find itself, and skipping any query whose predicted label appears on no bank row. Cell 22 fits `pyod.models.pca.PCA(contamination=0.01, n_components='mle')` on the benign trajectories and scores with its `decision_function`. That score, read at pyod's current master, is

$$
\text{score}(x) = \sum_{j=1}^{k} \frac{\lVert z(x) - v_j \rVert_2}{w_j}, \qquad z(x) = \frac{x - \mu}{\sigma}
$$

| symbol | meaning |
|---|---|
| $x$ | a trajectory, 1 rank per layer |
| $\mu, \sigma$ | the per-layer mean and standard deviation of the clean trajectories |
| $z(x)$ | the standardized trajectory |
| $v_j$ | the $j$-th principal axis, a unit vector read as a point in trajectory space |
| $w_j$ | the explained variance ratio of axis $j$, the weight under `weighted=True` |
| $k$ | the number of components kept |

a variance-weighted distance from the clean center. The notebook's contamination of 1% differs from Appendix A's 5%. pyod is not installed in this project. `third_party/` is not checked out in this working tree, so the cell numbers are the ones recorded when the port was written.

## The port step by step

1. `_build_ted` in `detectors/__init__.py` requires the validation loader and calls `collect_reference_bank(model, validation_loader, device, use_bfloat16, context.ted_reduction)`.
2. `collect_reference_bank` hooks every block boundary through `analysis.features.captured_layers`: layer 0 is the input of block 1 and layers 1 to 12 are the outputs of the 12 blocks on ViT-B/16, 13 layers in all (25 on Swin-S). It runs 1 forward pass over the 2000 validation images, keeps the images whose predicted label equals their true label and stores, per layer, `reduce_activation(captured[layer], reduction, has_class_token)`, the activation flattened to (batch, tokens × dim) under the default `flatten` reduction and cast to float16 (`BANK_DTYPE`). The bank also records each row's predicted label and its position in the loader. On ViT a flattened row is $197 \times 768 = 151296$ values per layer.
3. `leave_one_out_trajectories(bank)` ranks every bank row against the rest of the bank at every layer with `first_same_class_rank`, excluding its own row, which is the notebook's `ranking_array[1:]`. `first_same_class_rank` computes distances in float32 with `torch.cdist` over chunks of `DISTANCE_CHUNK` (256) bank rows and sorts them. It returns the position of the first row whose predicted label equals the query's. A query whose predicted class has no bank row gets the bank size.
4. `fit_trajectory_model(trajectories)` standardizes each layer's ranks by their mean and standard deviation (a standard deviation under `STANDARD_DEVIATION_FLOOR` becomes 1) and computes the pseudo-inverse of their covariance plus `COVARIANCE_FLOOR` on the diagonal, in float64. It returns a `TrajectoryModel` of mean, std and precision.
5. The validation split is scored through `ted_scores(..., loader_is_bank_source=True)`, which re-forwards the split and ranks each image with its own bank row excluded, so the scores the threshold is read from are leave-one-out.
6. For the clean and backdoor splits, `ted_scores` forwards each batch and reduces its activations the same way. It computes `rank_trajectories` against the full bank with shape (batch, 13) and then calls `outlier_scores`, the squared Mahalanobis distance $z^{\top} (\Sigma + \epsilon I)^{+} z$ of the standardized trajectory. `ted_scores` negates once and returns (N,).

## Deviations and why

1. **Considered layers.** The paper's default is every Conv2D output, and a ViT block has no Conv2D and no ReLU. Its Linear layers include the attention projections, which are head-split intermediates rather than representations of the input. The port reads the residual stream at every block boundary, which is every place the representation is rewritten and the closest transformer analogue of a convolutional block's output. Table VIII of the paper suggests the layer set matters little on deep networks, but no ViT number exists in the paper to check against.
2. **Outlier model.** pyod is not installed and is not added as a dependency. The port keeps pyod's per-layer standardization and replaces its weighted eigenvector-distance sum with the squared Mahalanobis distance under the floored clean covariance. Both grow with the standardized distance from the clean center, which is what line 19 of Algorithm 2 relies on, and no numerical equivalence is claimed: a number from this port is a Mahalanobis number rather than a pyod number.
3. **Bank membership and size.** The paper stores $m$ images per class and the notebook caps the bank at 1000. The port keeps every correctly classified image of the shared split with no cap and no random draw, so every detector sees the same budget and no seed enters. Ranks therefore run up to the bank size, which is near 2000 on an accurate model. The paper's box plots are not comparable in magnitude.
4. **Absent predicted class.** The notebook silently skips a query whose predicted class is not in the bank. The port gives it the bank size at every layer and scores it, since a detector that returns no score for an input has made no decision on it. The result is an extreme outlier score whether or not the input carries a trigger. `classes_without_reference(bank, num_classes)` lists the classes this rule fires on, which Tiny ImageNet, with about 10 validation images per class before misclassified ones are dropped, is expected to hit.
5. **Precision and storage.** The notebook holds a float32 bank. The port stores bank rows and queries alike as float16 on the device and computes distances in float32 over chunks. Rounding the queries too makes a bank member scored through its own loader identical to its stored row, so its self-exclusion removes exactly its own distance. float16 carries about 3 decimal digits, so a near tie between 2 bank rows at the 4th digit can flip a rank. Under `flatten` the bank is 13 layers × bank rows × 151296 halves, about 7.9 GB for a 2000-row bank on ViT-B/16, held on the device for the whole run.
6. **Leave-one-out validation scores.** The notebook scores its benign trajectories in sample. The port fits the trajectory model on the leave-one-out trajectories and scores the validation split with each bank member's own row excluded, so the threshold is read from scores that did not see themselves in the ranking. Each bank row still contributes about 1 in 2000 of the weight of the mean and covariance it is scored against, which is recorded rather than removed. `ted` is in `CROSS_FITTED` for this reason.
7. **Token reduction.** The notebook flattens every activation, and the port's default `flatten` does the same. `DetectorContext.ted_reduction` offers `cls` (the class token alone, ViT only) and `mean` (the token average), which shrink the bank to about 0.5% of its flattened size. The panel runs `flatten`. The synthetic sign gate runs `cls`, because the fixture's head reads only the class token and its patch tokens are random noise.
8. **Threshold rule.** The paper thresholds at the reject rate $\alpha$. The port hands the negated outlier score to `defenses.decision.detection_report`, which thresholds at a quantile of the validation scores.

## Hyperparameters and where they come from

| symbol | paper | this port | constant | source of the port's value |
|---|---|---|---|---|
| $N$, considered layers | every Conv2D output | every block boundary, 13 on ViT-B/16 and 25 on Swin-S | none, from `transformer_blocks` | deviation 1 |
| $m$, bank size | 20 to 200 per class by dataset, notebook cap 1000 | every correct image of the 2000 split | `PAPER_DEFENSE_SET_SIZE` (1000), recorded only | the shared data budget |
| $d$ | Euclidean | Euclidean in float32 | none | the paper |
| token reduction | the notebook flattens | `flatten` | `DEFAULT_REDUCTION` | the notebook |
| outlier model | pyod PCA, standardized and weighted | squared Mahalanobis on standardized trajectories | `COVARIANCE_FLOOR`, `STANDARD_DEVIATION_FLOOR` ($10^{-6}$ each) | deviation 2 |
| bank dtype | float32 | float16 | `BANK_DTYPE` | device memory, deviation 5 |
| distance chunk | the whole bank per query | 256 rows per `cdist` call | `DISTANCE_CHUNK` | device memory |
| absent-class rank | the query is dropped | the bank size | none | deviation 4 |
| $\alpha$ | 5%, notebook contamination 1% | none, the quantile rule | `PSBD_QUANTILES` | the registry's shared rule |

## Cross-check against the reference

`tests/test_detectors_ted.py` reimplements cell 14 of the notebook in numpy inside the test file and requires `first_same_class_rank` to agree with it to the integer, with and without the `ranking_array[1:]` self exclusion. It also checks the absent-class rule, chunking, float16 ranking in float32, the leave-one-out path against the loader path and a Swin-shaped activation. The outlier model has no reference, since pyod is not installed, so its tests pin direction and finiteness only. The suite passed on the CPU on 2026-09-29, and this is the only competitor whose rank logic is checked against the reference's own logic without needing `third_party/`. The synthetic sign gate `python -m experiments.preflight.check_signs`, run on the CPU the same day with the `cls` reduction, read TED at AUROC 0.9023, above the floor of 0.60.

## Cost

1 forward pass per input plus 1 distance matrix against the bank at each of the 13 layers. The fit is 2 passes over the validation split, 1 to build the bank and 1 to score the split leave-one-out, plus the bank's own leave-one-out ranks. `ted` is in `NEEDS_FITTING`. The measured seconds per input and fit seconds are in the results block.

## Direction

High is poisoned in the paper, since line 19 flags an outlier score above $\tau$. `outlier_scores` returns that statistic unnegated so it stays comparable to the paper's box plots. `ted_scores` negates it once.

## Results

<!-- results:begin -->
Generated by `python scripts/detector_doc_results.py` at commit `19488b81358b06040ba96f361d5061b2981f1f25-dirty`. It reads `results/coverage/coverage.json`, every `results/<folder>/detectors/<name>_metrics.json` and every `results/<folder>/psbd_metrics.json` of the 57 backdoored ViT-B/16 models the paper's detector comparison uses, the clearing models that carry a reading from every defense. PSBD-TM and PSBD-RD are read at the adaptive rate rule, every threshold is the clean-validation quantile named in the column and AUROC is the one-sided area at the 0.25 quantile, where a value under 0.5 means inverted.

Summary over every compared model. The rank is among the 13 defenses of the comparison by mean AUROC, and the last column is PSBD-TM minus the defense, paired per model, with its 95% bootstrap interval over models (5000 resamples, seed 0).

| defense | models | AUROC | TPR at 10% FPR | TPR at 20% FPR | models below chance | rank | PSBD-TM minus defense, AUROC |
|---|---|---|---|---|---|---|---|
| `ted` | 57 | 0.887 | 0.709 | 0.788 | 0 | 5 of 13 | +0.066 [+0.022, +0.107] |
| PSBD-TM | 57 | 0.953 | 0.873 | 0.902 | 2 | 1 of 13 | reference |
| PSBD-RD | 57 | 0.888 | 0.744 | 0.805 | 5 | 4 of 13 | +0.065 [+0.012, +0.121] |

`ted` per attack and poison rate. Each row is a mean over the models of that attack at that rate and n counts them. The last 2 columns repeat the AUROC of PSBD-TM and PSBD-RD on the same models.

| attack | poison rate | n | AUROC | TPR at 10% FPR | TPR at 20% FPR | PSBD-TM AUROC | PSBD-RD AUROC |
|---|---|---|---|---|---|---|---|
| BPP | 1% | 4 | 0.864 | 0.667 | 0.756 | 0.946 | 0.914 |
| BPP | 5% | 4 | 0.926 | 0.792 | 0.883 | 0.941 | 0.962 |
| BPP | 10% | 4 | 0.883 | 0.693 | 0.757 | 0.957 | 0.959 |
| BadNets | 1% | 4 | 0.933 | 0.836 | 0.892 | 0.987 | 0.542 |
| BadNets | 5% | 4 | 0.917 | 0.764 | 0.855 | 0.992 | 0.796 |
| BadNets | 10% | 4 | 0.877 | 0.680 | 0.773 | 0.996 | 0.798 |
| Blend | 1% | 4 | 0.876 | 0.648 | 0.745 | 0.973 | 0.944 |
| Blend | 5% | 4 | 0.914 | 0.739 | 0.781 | 0.987 | 0.970 |
| Blend | 10% | 4 | 0.898 | 0.722 | 0.810 | 0.976 | 0.997 |
| LF | 1% | 4 | 0.866 | 0.677 | 0.756 | 0.963 | 0.959 |
| LF | 5% | 4 | 0.825 | 0.622 | 0.688 | 0.986 | 0.978 |
| LF | 10% | 4 | 0.882 | 0.686 | 0.768 | 0.990 | 0.985 |
| SIG | 10% | 1 | 0.997 | 0.995 | 0.997 | 0.418 | 0.919 |
| TaCT | 1% | 1 | 0.926 | 0.853 | 0.908 | 0.979 | 0.464 |
| TaCT | 5% | 2 | 0.984 | 0.975 | 0.986 | 0.954 | 0.613 |
| WaNet | 5% | 2 | 0.724 | 0.305 | 0.458 | 0.933 | 0.955 |
| WaNet | 10% | 3 | 0.861 | 0.634 | 0.754 | 0.786 | 0.957 |

Mean AUROC per dataset. The shared 2000-image clean split gives about 200 images per class on CIFAR-10, 46 on GTSRB, 20 on CIFAR-100 and 10 on Tiny ImageNet, which is the budget every class-conditional method fits on.

| defense | CIFAR-10 | CIFAR-100 | GTSRB | Tiny ImageNet |
|---|---|---|---|---|
| `ted` | 0.866 (17) | 0.976 (12) | 0.991 (14) | 0.733 (14) |
| PSBD-TM | 0.891 (17) | 0.979 (12) | 0.981 (14) | 0.977 (14) |
| PSBD-RD | 0.844 (17) | 0.890 (12) | 0.862 (14) | 0.965 (14) |

Measured cost, median over the compared models. Seconds per 1000 inputs divide the scoring time of the clean and backdoor splits by their size. The fit is the one-off pass over the clean validation split before any input is scored, and a dash marks a detector with no fit. The device is the one most records name.

| detector | forward passes per input | seconds per 1000 inputs | fit seconds | precision | device |
|---|---|---|---|---|---|
| `ted` | 1 | 2.74 | 9.2 | bfloat16 | NVIDIA A100-SXM4-40GB |

<!-- results:end -->

## Reading the results

TED is the fifth-ranked defense and 1 of only 2 competitors that never falls below chance on the panel. It separates TaCT and the single SIG model almost perfectly, which is the setting the paper was built for: TaCT's trigger only fires on its source class, so a triggered image's early-layer neighbors are its source class and its trajectory has exactly the shape the method looks for. Its Tiny ImageNet mean is far below the other 3 datasets, the same budget effect Beatrix shows. With about 10 validation images per class, fewer after misclassified ones are dropped, a class can have no bank row at all or only 1 or 2. Every clean image the model sends to such a class takes the absent-class rank or a coarse one and looks like an outlier. How many classes this hits per model is not recorded in the provenance, so this reading is an inference from the mechanism rather than a count.

## Known failure modes

A high outlier score says the trajectory is far from the clean cloud, which an image of a rare predicted class also produces, since a class with 1 or 2 bank rows gives coarse ranks and a class with none gives the bank size at every layer. Clean-label attacks are an expected weakness by the mechanism, since a triggered image's source class is the target class and its trajectory has no source-class prefix to stand out with. The panel holds a single clean-label model (SIG), which TED separates, so this expectation is not tested here. The paper's Section VI-C finds its own adaptive losses ineffective, and later work (TED-LaST, arXiv:2506.10722, known here only through survey notes) reports adaptive attacks that defeat it. The bank's device memory is the operational risk: it grows linearly with the split size and the model width.

## How to run and where records land

```bash
python -m cli.baselines --checkpoint-folder <folder> --detectors ted --max-samples 500 \
    --results-dir scratch/detector_smoke/results --allow-missing-psbd-cache
```

The smoke above writes outside `results/`. The panel runs through the `cheap` job group of `pbs/generate_detector_jobs.py`, which needs the bank's memory on top of the model and fits on an A100 40 GB. `results/<folder>/detectors/ted_metrics.json` holds the report and provenance, and `ted_scores_{validation,clean,backdoor}.pt` hold the negated outlier scores in loader order, the validation tensor being the leave-one-out scores.
