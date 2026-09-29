# Confidence, the max-softmax null model

Confidence scores an input by the largest softmax probability the model assigns to any class, negated so that low means poisoned. No paper stands behind it and no released code exists to port. It is in the registry as the floor every other detector has to clear: a method that cannot beat 1 forward pass and a maximum is reading the model's calibration rather than detecting a backdoor. This page explains why the null model earns a column, what the port in `detectors/confidence.py` does and how it reads on the panel. Terms such as the shared split, the panel, PSBD-TM and PSBD-RD are defined once in `README.md` in this directory.

## The idea in plain words

A backdoor is trained to be unambiguous. Once the trigger is present, the model has learned a shortcut from the trigger to the target class that bypasses the ordinary evidence in the image, and training pushes the target logit far above the others on every poisoned example. A triggered input therefore tends to receive a very confident prediction. If that alone separated triggered from clean inputs, a defender would need nothing more than the model's top probability, and any detector that costs more has to show that its extra machinery adds something beyond it.

The same property makes confidence weak. An easy clean image, a large object or a well separated class also receives a confident prediction, and nothing in the top probability says why the model was confident. Confidence is therefore the question "is the model surer than usual", and a detector that does no better than that has not measured anything specific to backdoors.

## The statistic

$$
s(x) = \max_{c} P(c \mid x; \theta)
$$

| symbol | meaning |
|---|---|
| $x$ | the input image |
| $\theta$ | the model's parameters |
| $c$ | a class index, over all classes of the dataset |
| $P(c \mid x; \theta)$ | the softmax probability the model assigns to class $c$ |
| $s(x)$ | the confidence, the largest of those probabilities |

The descriptive form is `confidence = the largest softmax probability the model assigns to any class`, and the registry score is its negation, `score = -confidence`, so that a triggered input, which is usually more confident, scores lower.

## Setting

There is no original paper, so there is no original setting. The null model enters this repository because Section 2.2 of `docs/attack-design/cross-defense.md` measured that the top probability alone beats PSBD on a benign control, which made it the obvious floor to print beside every detector. It uses no clean data beyond the shared threshold rule and has no hyperparameters.

## The port step by step

1. `detectors.build_detector("confidence", context)` calls `_build_confidence` in `detectors/__init__.py`, which fits nothing and returns a closure over `confidence_scores`.
2. `confidence_scores(model, loader, device, use_bfloat16)` in `detectors/confidence.py` puts the model in eval mode and iterates the loader in order.
3. For each batch of shape (batch, channels, height, width), `defenses.inference.forward_probs` runs 1 forward pass under the shared autocast policy (bfloat16 on a GPU) and returns the softmax probabilities, shape (batch, num_classes).
4. The row-wise maximum `probs.max(dim=1).values` has shape (batch,), and it is negated and moved to the CPU.
5. The batches are concatenated into 1 float32 tensor of shape (N,), in the loader's order. `cli.baselines` calls this once per split (validation, clean and backdoor) and writes the 3 tensors and the record.

## Deviations

There is no paper to deviate from and no reference to disagree with. The only design choice is the negation at the return boundary, made so that the shared convention holds.

## Hyperparameters

None. There is no scaling set, no ensemble, no threshold of its own and no clean-data budget beyond the quantile rule every detector's report is read at.

## Cross-check

No reference implementation exists. `python -m experiments.preflight.check_signs` reads confidence on the synthetic fixture of `experiments/preflight/synthetic.py`, a randomly initialized 2-block ViT on random-noise images whose trigger is wired straight to the target logit, 256 samples per split. Run on the CPU on 2026-09-29 it read confidence at AUROC 1.0000, above the gate's `MINIMUM_AUROC` of 0.60. The fixture's trigger drives the target logit far above the rest by construction, so the number confirms the plumbing and the sign and says nothing about a trained model.

## Cost

1 forward pass per input, the cheapest entry in `FORWARD_PASSES_PER_INPUT` and the unit every other cost is stated in. There is no fit, so `confidence` is absent from `NEEDS_FITTING`.

## Direction

Low is poisoned. The raw statistic is usually high for a triggered input, so `confidence_scores` negates it once at the return. A second negation anywhere downstream would produce a well-formed, exactly inverted detector, which `check_signs.py` and the `auroc_two_sided` diagnostic in `detection_report` exist to catch.

## Results

<!-- results:begin -->
Generated by `python scripts/detector_doc_results.py` at commit `19488b81358b06040ba96f361d5061b2981f1f25-dirty`. It reads `results/coverage/coverage.json`, every `results/<folder>/detectors/<name>_metrics.json` and every `results/<folder>/psbd_metrics.json` of the 57 backdoored ViT-B/16 models the paper's detector comparison uses, the clearing models that carry a reading from every defense. PSBD-TM and PSBD-RD are read at the adaptive rate rule, every threshold is the clean-validation quantile named in the column and AUROC is the one-sided area at the 0.25 quantile, where a value under 0.5 means inverted.

Summary over every compared model. The rank is among the 13 defenses of the comparison by mean AUROC, and the last column is PSBD-TM minus the defense, paired per model, with its 95% bootstrap interval over models (5000 resamples, seed 0).

| defense | models | AUROC | TPR at 10% FPR | TPR at 20% FPR | models below chance | rank | PSBD-TM minus defense, AUROC |
|---|---|---|---|---|---|---|---|
| `confidence` | 57 | 0.684 | 0.335 | 0.438 | 16 | 11 of 13 | +0.268 [+0.198, +0.339] |
| PSBD-TM | 57 | 0.953 | 0.873 | 0.902 | 2 | 1 of 13 | reference |
| PSBD-RD | 57 | 0.888 | 0.744 | 0.805 | 5 | 4 of 13 | +0.065 [+0.012, +0.121] |

`confidence` per attack and poison rate. Each row is a mean over the models of that attack at that rate and n counts them. The last 2 columns repeat the AUROC of PSBD-TM and PSBD-RD on the same models.

| attack | poison rate | n | AUROC | TPR at 10% FPR | TPR at 20% FPR | PSBD-TM AUROC | PSBD-RD AUROC |
|---|---|---|---|---|---|---|---|
| BPP | 1% | 4 | 0.408 | 0.096 | 0.196 | 0.946 | 0.914 |
| BPP | 5% | 4 | 0.745 | 0.397 | 0.453 | 0.941 | 0.962 |
| BPP | 10% | 4 | 0.757 | 0.237 | 0.512 | 0.957 | 0.959 |
| BadNets | 1% | 4 | 0.566 | 0.005 | 0.137 | 0.987 | 0.542 |
| BadNets | 5% | 4 | 0.879 | 0.600 | 0.696 | 0.992 | 0.796 |
| BadNets | 10% | 4 | 0.780 | 0.593 | 0.688 | 0.996 | 0.798 |
| Blend | 1% | 4 | 0.589 | 0.136 | 0.245 | 0.973 | 0.944 |
| Blend | 5% | 4 | 0.917 | 0.813 | 0.928 | 0.987 | 0.970 |
| Blend | 10% | 4 | 0.769 | 0.480 | 0.538 | 0.976 | 0.997 |
| LF | 1% | 4 | 0.456 | 0.119 | 0.200 | 0.963 | 0.959 |
| LF | 5% | 4 | 0.660 | 0.371 | 0.404 | 0.986 | 0.978 |
| LF | 10% | 4 | 0.809 | 0.455 | 0.591 | 0.990 | 0.985 |
| SIG | 10% | 1 | 0.866 | 0.753 | 0.804 | 0.418 | 0.919 |
| TaCT | 1% | 1 | 0.175 | 0.000 | 0.000 | 0.979 | 0.464 |
| TaCT | 5% | 2 | 0.864 | 0.049 | 0.269 | 0.954 | 0.613 |
| WaNet | 5% | 2 | 0.601 | 0.200 | 0.260 | 0.933 | 0.955 |
| WaNet | 10% | 3 | 0.565 | 0.204 | 0.246 | 0.786 | 0.957 |

Mean AUROC per dataset. The shared 2000-image clean split gives about 200 images per class on CIFAR-10, 46 on GTSRB, 20 on CIFAR-100 and 10 on Tiny ImageNet, which is the budget every class-conditional method fits on.

| defense | CIFAR-10 | CIFAR-100 | GTSRB | Tiny ImageNet |
|---|---|---|---|---|
| `confidence` | 0.602 (17) | 0.813 (12) | 0.480 (14) | 0.877 (14) |
| PSBD-TM | 0.891 (17) | 0.979 (12) | 0.981 (14) | 0.977 (14) |
| PSBD-RD | 0.844 (17) | 0.890 (12) | 0.862 (14) | 0.965 (14) |

Measured cost, median over the compared models. Seconds per 1000 inputs divide the scoring time of the clean and backdoor splits by their size. The fit is the one-off pass over the clean validation split before any input is scored, and a dash marks a detector with no fit. The device is the one most records name.

| detector | forward passes per input | seconds per 1000 inputs | fit seconds | precision | device |
|---|---|---|---|---|---|
| `confidence` | 1 | 0.42 | -- | bfloat16 | NVIDIA A100-SXM4-40GB |

<!-- results:end -->

## Reading the results

The null model sits near the bottom of the ranking and falls below chance on more models than any other detector except SentiNet and the data-limited SCALE-UP. Below chance means the triggered images were less confident than their clean twins on that model, which happens when the backdoor's target logit is high but other logits are also raised, or when the clean images of that dataset are themselves near certain. GTSRB is the clearest case: traffic signs are classified at near 100% confidence, so there is no room for a triggered input to be more confident. On Tiny ImageNet, where clean predictions spread over 200 classes, the gap from a backdoor is easier to see.

The null model matters for how the other columns are read. Any detector whose per-attack row sits near confidence's has not shown that it measures more than confidence does on that attack.

## Known failure modes

A low score says only that the model was more sure of itself than usual. A backdoor placed deliberately near the decision boundary defeats it from the other side: `docs/attack-design/A5-low-confidence-backdoor.md` works through a poisoned sample capped at a target probability of 0.3, which reads as less confident than the median clean sample and is ranked as benign. That analysis is a prediction rather than a measurement on a trained model. Peng et al.'s under-confidence backdoor (arXiv:2202.11203) caps the target posterior near 0.6 by construction and reports its effect against STRIP rather than against confidence directly, so the confidence-specific claim remains untested here.

## How to run and where records land

```bash
python -m cli.baselines --checkpoint-folder <folder> --detectors confidence --max-samples 500 \
    --results-dir scratch/detector_smoke/results --allow-missing-psbd-cache
```

The smoke above writes outside `results/` and needs `--allow-missing-psbd-cache` because the smoke tree holds no PSBD manifest to check against. The panel runs through the `cheap` job group of `pbs/generate_detector_jobs.py`. `results/<folder>/detectors/confidence_metrics.json` holds the report at every quantile and the provenance, and `confidence_scores_{validation,clean,backdoor}.pt` hold the negated scores in loader order.
