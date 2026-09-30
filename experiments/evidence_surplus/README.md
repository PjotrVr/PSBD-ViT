# Evidence surplus

## Question

A triggered image moves the model's representation along the backdoor direction. This experiment asks how far it moves compared with how far a clean image has to be moved along the same direction before the model sends it to the target class. The ratio of the 2 is the **surplus factor**. A factor well above 1 means the trigger writes far more backdoor evidence than the decision needs, so a perturbation that removes part of it still leaves the prediction on the target. A factor near 1 means the triggered prediction sits close to its own decision boundary.

The question matters for PSBD because PSBD detects a poisoned input by the stability of its prediction under perturbation. An input whose prediction survives perturbation reads as poisoned, so a large surplus is a candidate reason for PSBD-TM's detection strength per model. This folder holds 1 experiment so far, experiment A, in `surplus_factor/`.

## Experiment A, the surplus factor

### Models

| role | models | why |
| --- | --- | --- |
| backdoored ViT-B/16 | `vit_cifar10_badnet_a2o_0_01`, `vit_tiny_badnet_a2o_0_05`, `vit_cifar100_blend_0_1`, `vit_cifar10_blend_0_1`, `vit_cifar10_bpp_0_05`, `vit_gtsrb_lf_0_01`, `vit_cifar10_wanet_0_1`, `vit_tiny_wanet_0_05`, `vit_gtsrb_tact_0_05`, `vit_cifar10_tact_0_01` | the 10 panel models the coordinator named, spanning PSBD-TM's strong and weak cells |
| benign control | `vit_cifar10_benign` probed with BadNets, Blend, BPP, WaNet and TaCT, `vit_gtsrb_benign` probed with LF and TaCT | the same trigger on a model that never learned it |
| original-paper reproduction | `resnet18_gtsrb_badnet_a2o_0_1`, `resnet18_gtsrb_blend_0_1` | PSBD's own architecture |

The ResNet-18 benign GTSRB checkpoint has no `args.json`, so its eval set cannot be rebuilt through `data.splits` and the ResNet rows have no benign control. CIFAR-100 and Tiny ImageNet rows have none either, since only the CIFAR-10 and GTSRB benign ViTs were named.

### Pre-registration

These predictions were written on 2026-09-30 at 02:36, before any surplus factor was measured.

1. The surplus factor at the last block is well above 1 on the models PSBD-TM detects well.
2. It is near 1 on the 2 TaCT models and on `vit_cifar10_wanet_0_1` at the last block.
3. Across the 10 backdoored ViT models it rank-correlates positively with PSBD-TM's TPR at 1% FPR and with its AUROC, both read from `psbd_metrics.json` at the adaptive rate.
4. For the 2 WaNet models the depth profile is low before WaNet's direction onset (about block 7 in `experiments/backdoor_manifestation/`) and high after it.
5. On the benign controls the needed scale is often not reachable within the search range, and on a random direction the factor is about 0 or undefined.

### A2, manufactured surplus on clean inputs

A2 is the causal half of experiment A. If surplus is what PSBD reads, then giving a clean image surplus without any trigger should make PSBD flag it.

**Models.** The benign `vit_cifar10_benign` and `vit_gtsrb_benign`, and the backdoored `vit_cifar10_badnet_a2o_0_01` and `vit_cifar10_blend_0_1`.

**Manipulation.** A clean image x of predicted class c gets its own class's direction added to the residual stream at the output of block 12 (the last) and separately at the output of block 8. The direction is the class-mean offset, $o_c = \mu_c - \mu$, fitted on 1 half of the clean analysis images and applied to the other half. The added length is $(S - 1)\,\lVert o_c \rVert$ along $\hat o_c$, so the steered image carries S times a typical image's own-class evidence. The last block is where experiment A reads surplus and where removing the backdoor direction removes the backdoor. After block 12 only the class token reaches the head and no probe acts, so the class-token-only control cannot differ from all-token steering there. Block 8 is added for that comparison, since 4 blocks of masked attention follow it. The direction and $\lVert o_c \rVert$ are fitted at each block separately. S takes the values 1, 2, 4 and 8, plus the last-block surplus factor measured in A: `vit_cifar10_badnet_a2o_0_01` and `vit_cifar10_blend_0_1` use their own, and each benign model uses the median over A's backdoored models on its dataset.

**Readout.** PSBD-TM (token mask before the attention norm) and PSBD-RD (dropout after the residual adds) score every image with the fractional PSU at the model's adaptive rate p* from `psbd_metrics.json`, with 3 passes. The thresholds are quantiles 0.01, 0.05 and 0.25 of the unsteered clean-validation scores. The flag rate is the share of images at or below the threshold, since a low score means poisoned.

**Controls.** A random direction of the same added length, on every token. The own-class direction added to the class token only.

**Predictions, written 2026-09-30 at 02:40 before any A2 measurement.**

6. The steered clean images keep their prediction.
7. Under own-class steering on every token, the flag rate rises far above the nominal FPR (the quantile) and grows with S, for both PSBD-TM and PSBD-RD, giving a backdoor-like signature with no trigger.
8. The random direction of the same length raises the flag rate by little or nothing.
9. At block 8, PSBD-TM responds more to all-token steering than to class-token-only steering, because redundancy across tokens is what gives surplus along the token axis. PSBD-TM masks tokens, so steering that lives only in the class token is exposed to every mask that reaches the class token's inputs.

If a prediction fails, its verdict stays as written and 1 targeted diagnostic measurement is reported beside it (the search range, LayerNorm renormalizing the steer or the choice of block).

### A3, removing surplus from false positives

A3 is the mirror of A2. If surplus is what PSBD reads, then a clean image PSBD-TM flags by mistake should stop being flagged once its surplus is removed, with its prediction kept.

**Models.** The 10 backdoored ViT models of A and the benign `vit_cifar10_benign` and `vit_gtsrb_benign`.

**False positives.** Clean validation images whose cached PSBD-TM fractional PSU at the adaptive rate (`results/<folder>/psbd/before_attention_norm_token_mask/`, the canonical sweep) is at or below the 0.01 or the 0.05 quantile of the same cached validation scores. The thresholds are read from the cache and never refitted.

**Stream edit.** At the last block's output, the image's own-class direction $\hat o_c$ (class-mean offset fitted on clean analysis images) is partly subtracted from every token. The own-class evidence is $e = (h - \mu) \cdot \hat o_c$, the removal that flips the prediction $\beta^*$ is found by bisection, the boundary evidence is $b = e - \beta^*$ and the own-class surplus factor is $s = e / b$. The edit removes $\beta = e - \tilde s\, b$, with $\tilde s$ the median s of non-flagged clean validation images, so the edited surplus equals $\tilde s$ and the prediction is kept whenever $\tilde s > 1$. Images with $s \le \tilde s$ are left unedited.

**Input edit.** The patch tokens are ranked by how much replacing each alone with the mean token (the per-position mean of clean images at the first block's input) lowers the predicted class's logit. The smallest m for which the top-m tokens kept and the rest replaced still give the same prediction is found by bisection, and the edited image keeps exactly those m tokens.

**Readout.** PSBD-TM at the adaptive rate, 3 fresh passes with mask seed 0, on the edited and on the unedited images, flagged at the cached thresholds. Fresh passes batch the images differently from the sweep, so the unedited fresh flag rate of the false positives is the reference each edit is read against.

**Controls.** A random direction with the same removed length. A random token subset of the same size m. The same 2 edits on 200 non-flagged clean validation images.

**Predictions, written 2026-09-30 at 02:42 before any A3 measurement.**

10. Both edits keep the prediction on most false positives (the stream edit on all of them by construction whenever $\tilde s > 1$).
11. The edited false positives drop to about the nominal flag rate (0.01 or 0.05) at their threshold, from a much higher unedited fresh rate.
12. The random-direction and random-subset controls do not lower the false positives' flag rate as much.
13. The edits leave non-flagged clean images mostly unflagged and make them more fragile, a higher median fractional PSU.

## Method

Experiment A lives in `surplus_factor/measure.py`, A2 in `surplus_factor/manufactured.py` and A3 in `surplus_factor/removal.py`. Each has a CPU `prepare` stage, a GPU `probe` stage run 1 model at a time under the shared login-node lock, and a CPU `summarize` stage. `run_panel.sh` runs A and `run_causal.sh` runs A2 and A3 after it, both resumably and with no probe started between 06:30 and 17:00. Every forward runs in bfloat16. The other folders under `experiments/evidence_surplus/` belong to a separate set of experiments and are not described here.

### Surplus factor (A)

The pairs are the PSBD split's eligible images, 800 per model drawn with seed 0 through `experiments.backdoor_manifestation.measure.paired_images`. The manifestation experiment's cached pairs are reused where it measured the same run. At every block l, the backdoor direction is fitted on pairs 0 to 399 and read on pairs 400 to 799:

$$u_l = \frac{r_l}{\lVert r_l \rVert},\quad r_l = \frac{1}{N_{\text{fit}}}\sum_{i \in \text{fit}} \big(h_l(\tilde x_i) - h_l(x_i)\big)$$

$$\text{actual}_i = \big(h_l(\tilde x_i) - h_l(x_i)\big) \cdot u_l,\qquad \text{needed}_i = \min\{\alpha \ge 0 : \arg\max f_l\big(h_l(x_i) + \alpha u_l\big) = t\},\qquad \text{surplus}_i = \frac{\text{actual}_i}{\text{needed}_i}$$

| symbol | meaning |
| --- | --- |
| $x_i$, $\tilde x_i$ | clean eval image i and its triggered twin |
| $h_l$ | pooled feature at block l's output, the class token on ViT-B/16 and the spatial mean on ResNet-18 |
| $f_l$ | the network from block l's output to the logits |
| $t$ | the target class |

The shift $\alpha u_l$ is added to every token, or every spatial position on ResNet-18, so the pooled feature moves by exactly $\alpha u_l$. needed is found per image by 12 bisection steps between 0 and 4 times the median actual (`SEARCH_FACTOR`). An image still off target at the top is unreachable, and an image the model already sends to the target is left out. The reported factor is the median over reachable images. The random-direction control uses the same search range with a random unit direction in place of $u_l$.

### Manufactured surplus (A2) and surplus removal (A3)

The formulas are in the pre-registrations above. Both score images with the fractional PSU at the adaptive rate through `models.positions.plug_dropout` and `cli.sweep.bound_operator`, the plumbing `cli.sweep` uses, with 3 passes and mask seed 0. A2 reads its thresholds from fresh scores of the unsteered clean validation split and records the cached q 0.25 threshold beside them as a check. A3 reads both the false positives and the thresholds from the cached sweep.

## Results

All runs finished on the login-node A100 on 2026-09-30 between 02:37 and 03:45: A on 19 runs, A2 on 4 models, A3 on 12 models and the 2 diagnostics. Nothing is left to run. Records are under `results/_experiments/evidence_surplus/`: `runs/` with `summary.json` for A, `manufactured/` for A2, `removal/` for A3 and `diagnostics/head/` with `diagnostics/survival/` for the diagnostics. The figures are `figures/last_block_surplus`, `figures/surplus_against_psbd`, `figures/depth_profile`, `figures/manufactured_surplus` and `figures/false_positive_removal`. The other files in `figures/` belong to the other evidence-surplus experiments.

### A, the surplus factor

| model | last-block surplus | head-input surplus (diagnostic) | PSBD-TM TPR at 1% FPR | PSBD-TM AUROC |
| --- | --- | --- | --- | --- |
| `vit_cifar10_badnet_a2o_0_01` | 1.12 | 1.11 | 0.851 | 0.982 |
| `vit_tiny_badnet_a2o_0_05` | 1.31 | 1.19 | 0.367 | 0.984 |
| `vit_cifar100_blend_0_1` | 1.61 | 1.20 | 1.000 | 0.998 |
| `vit_cifar10_blend_0_1` | 1.24 | 1.15 | 0.550 | 0.906 |
| `vit_cifar10_bpp_0_05` | 1.33 | 1.10 | 0.400 | 0.801 |
| `vit_gtsrb_lf_0_01` | 1.33 | 0.95 | 0.854 | 0.976 |
| `vit_cifar10_wanet_0_1` | 1.23 | 1.04 | 0.023 | 0.459 |
| `vit_tiny_wanet_0_05` | 1.79 | 1.16 | 0.802 | 0.930 |
| `vit_gtsrb_tact_0_05` | 2.31 | 2.07 | 0.030 | 0.942 |
| `vit_cifar10_tact_0_01` | 1.69 | 1.79 | 0.005 | 0.979 |
| `resnet18_gtsrb_badnet_a2o_0_1` | 1.24 | not measured | 1.000 (PSBD-RD) | 1.000 (PSBD-RD) |
| `resnet18_gtsrb_blend_0_1` | 1.15 | not measured | 0.555 (PSBD-RD) | 0.968 (PSBD-RD) |

Every triggered image of every backdoored model is reachable (reachable share 1.00). On the 7 benign controls at most 0.07 of clean images reach the target within the search range (0.00 on 5 of them). On the random direction at most 0.04 do, on every model. The ResNet-18 head reads the last block's spatial mean directly, so its head-input surplus equals its last-block surplus.

1. **Prediction 1 fails.** The factor is 1.12 to 1.69 on the 5 models PSBD-TM detects with AUROC above 0.97, not well above 1. A triggered image carries 12% to 69% more backdoor direction than a clean image needs to reach the target.
2. **Prediction 2 fails.** The 2 TaCT models carry the largest factors (1.69 and 2.31), and `vit_cifar10_wanet_0_1` (1.23) sits in the middle of the rest.
3. **Prediction 3 fails.** Across the 10 ViT models Spearman's rho is -0.08 (p 0.83) against TPR at 1% FPR and +0.13 (p 0.73) against AUROC.
4. **Prediction 4 holds on 1 of 2 models.** `vit_tiny_wanet_0_05` reads 0.16 to 0.36 in blocks 2 to 6 and 1.36 to 2.05 from block 8 on. `vit_cifar10_wanet_0_1` rises from 0.23 to 0.67 before block 7 to 0.76 to 1.23 after it, crossing 1 only at block 10. On every ViT model the factor is largest at or near the last block. On ResNet-18 it jumps from 0.3 to 1.2 between the stage-3 and stage-4 blocks.
5. **Prediction 5 holds.** The benign controls and the random direction almost never reach the target.

**Diagnosis of predictions 1 to 3** (`diagnose.py head`). The search range is not the cause, since every triggered image is reachable. The block is not the cause, since the factor is largest at the last block. The final LayerNorm is the remaining candidate, because A measures before it. Repeating A at the head input, after the LayerNorm, gives 0.95 to 2.07, the same range with the same 2 TaCT models on top. Spearman's rho becomes -0.28 (p 0.43) against TPR at 1% FPR and +0.47 (p 0.17) against AUROC. The verdicts stand. Surplus along the backdoor direction is modest on every model and does not order the models by PSBD-TM's detection.

### A2, manufactured surplus

The unsteered clean images are flagged at 0.045 to 0.054 at the 0.05 threshold, as they should be. The fresh q 0.25 thresholds sit within 0.025 of the cached ones for PSBD-TM and within 0.012 for PSBD-RD.

6. **Prediction 6 holds.** Own-class steering keeps the prediction on at least 0.97 of images at every S and block. The random control keeps it on 0.97 or more up to S = 4 and on 0.59 to 0.96 at S = 8.
7. **Prediction 7 holds.** At block 12, own-class steering on every token at S = 2 is flagged on 1.00 of images by both PSBD-TM and PSBD-RD on all 4 models. The same holds at the surplus factor from A (1.12 to 1.82), where PSBD-TM flags 0.99 to 1.00 and PSBD-RD 0.95 to 1.00. At block 8 the flag rate grows with S: PSBD-TM goes from 0.05 at S = 1 to 0.48 to 0.86 at S = 2 and 0.78 to 1.00 at S = 4. PSBD-RD grows more slowly on CIFAR-10, from about 0.05 at S = 1 to 0.10 to 0.13 at S = 4.
8. **Prediction 8 holds for PSBD-TM and partly for PSBD-RD.** A random direction of the same length leaves PSBD-TM at 0.00 to 0.06 up to S = 4 at both blocks. PSBD-RD rises to 0.18 to 0.23 at block 12 at S = 4 and to 0.87 to 1.00 at S = 8, so for PSBD-RD a long enough random push also lowers the fractional PSU.
9. **Prediction 9 fails.** At block 8 PSBD-TM flags class-token-only steering as often as all-token steering or slightly more: 0.57 against 0.59 at S = 2 and 0.92 against 0.95 at S = 4 on `vit_cifar10_badnet_a2o_0_01`, and in 7 of the 8 such comparisons over the 4 models (the exception is `vit_gtsrb_benign` at S = 4, 1.00 against 0.997). PSBD-RD responds far more to class-token-only steering (0.84 against 0.10 at S = 4).

**Diagnosis of prediction 9** (`diagnose.py survival`). The diagnostic measures the own-class evidence of the block-12 class token, its projection on the class offset, with no probe and averaged over PSBD-TM's 3 masked passes. At PSBD-TM's adaptive rate the masks remove nearly all of it on a clean image: a median 26.6 falls to 0.24 on `vit_cifar10_badnet_a2o_0_01`, and 23.9 to 1.1 on `vit_cifar10_benign`. With no probe, all-token steering at block 8 adds almost nothing to the block-12 class token (-0.29 at S = 2, -3.59 at S = 4), because the 4 later blocks absorb it. Under masking the same steering adds 1.9 and 6.2. Class-token-only steering adds 2.0 and 6.6 under masking. So the steer acts by holding up the class token's evidence in the masked passes, and both variants hold it up by about the same amount. Redundancy across tokens is not what separates them. The verdict stands.

### A3, removing surplus from false positives

On every model the false positives are 100 of 2000 validation images, 20 of them at the 0.01 threshold. Fresh passes re-flag 0.62 to 0.87 of them at 0.05, the reference each edit is read against. Every false positive's fresh prediction agrees with the cached one.

10. **Prediction 10 holds.** The stream edit keeps the prediction on 0.93 to 1.00 of false positives and the input edit on 1.00.
11. **Prediction 11 holds on 6 of 12 models and fails on the 6 CIFAR-10 models.** After the stream edit the 0.05 flag rate of the false positives is 0.02 to 0.19 on `vit_gtsrb_lf_0_01`, `vit_gtsrb_benign`, `vit_gtsrb_tact_0_05`, `vit_cifar100_blend_0_1`, `vit_tiny_badnet_a2o_0_05` and `vit_tiny_wanet_0_05`, against 0.62 to 0.81 unedited. On the 6 CIFAR-10 models it falls only to 0.31 to 0.52. The input edit brings 4 models near nominal (0.02 to 0.11) and leaves the other 8 at 0.47 to 0.96.
12. **Prediction 12 holds for the stream edit and cannot be read for the input edit.** The random direction of the same length lowers the flag rate less than the own-class edit on all 12 models (0.08 to 0.62 against 0.02 to 0.52). The random token subset changes the prediction on 0.52 to 0.86 of images, so its lower flag rate describes different predictions and is not a control of the kept-prediction edit.
13. **Prediction 13 holds for the stream edit.** Non-flagged images stay at or below 0.03 flagged and their median fractional PSU rises on every model (for example 0.901 to 0.960 on `vit_cifar10_badnet_a2o_0_01`). The input edit flags 0.03 to 0.22 of the non-flagged CIFAR-10 images and does not raise their PSU consistently (0.901 falls to 0.829 on `vit_cifar10_badnet_a2o_0_01`).

**Diagnosis of prediction 11**, read from the A3 records. The edit can only lower a false positive's surplus to the non-flagged median, so it can only work where false positives carry more own-class surplus than non-flagged images. On the 6 models where it works they carry 1.5 to 3.0 times the non-flagged median (for example 22.5 against 7.6 on `vit_gtsrb_lf_0_01`). On the 6 CIFAR-10 models they carry 0.7 to 1.2 times it (for example 9.1 against 8.5 on `vit_cifar10_badnet_a2o_0_01`), so the edit touches only 0.37 to 0.61 of them. On CIFAR-10 own-class surplus does not explain why these images are flagged. The verdict stands.

**Attractor split of prediction 11** (`false_positives/attractor_split.py`, CPU, from the A3 records and the cached sweep, JSON in `results/_experiments/evidence_surplus/false_positives/attractor_split.json`). The hypothesis was that the CIFAR-10 failure is a class-count effect through the attractor, since with 10 classes the class that masked inputs fall into is a large share of the images and its stability comes from where masking pushes and not from own-class surplus. The attractor of a model is the class most masked clean validation predictions flip into (cached per-pass argmax at the adaptive rate, flipped passes only) and a false positive belongs to it when its unperturbed prediction is that class. The table gives the post-edit flag rate at 0.05 with the group size in brackets. The verdict on prediction 11 is unchanged.

| Model | Classes | Attractor | Image share | FP share 0.05 | FP share 0.01 | Stream, attractor | Stream, other | Input, attractor | Input, other |
|---|---|---|---|---|---|---|---|---|---|
| `vit_cifar10_badnet_a2o_0_01` | 10 | 6 | 0.101 | 0.95 | 0.95 | 0.40 (95) | 0.80 (5) | 0.69 (95) | 0.40 (5) |
| `vit_tiny_badnet_a2o_0_05` | 200 | 94 | 0.014 | 0.01 | 0.00 | 0.00 (1) | 0.19 (99) | 0.00 (1) | 0.02 (99) |
| `vit_cifar100_blend_0_1` | 100 | 33 | 0.009 | 0.00 | 0.00 | -- | 0.14 (100) | -- | 0.03 (100) |
| `vit_cifar10_blend_0_1` | 10 | 4 | 0.100 | 0.49 | 0.20 | 0.31 (49) | 0.43 (51) | 0.63 (49) | 0.86 (51) |
| `vit_cifar10_bpp_0_05` | 10 | 7 | 0.102 | 0.76 | 0.80 | 0.55 (76) | 0.42 (24) | 0.82 (76) | 0.25 (24) |
| `vit_gtsrb_lf_0_01` | 43 | 13 | 0.054 | 0.61 | 0.75 | 0.00 (61) | 0.05 (39) | 0.70 (61) | 0.08 (39) |
| `vit_cifar10_wanet_0_1` | 10 | 2 | 0.094 | 0.94 | 0.95 | 0.32 (94) | 0.17 (6) | 0.79 (94) | 0.00 (6) |
| `vit_tiny_wanet_0_05` | 200 | 0 | 0.007 | 0.11 | 0.35 | 0.55 (11) | 0.15 (89) | 1.00 (11) | 0.00 (89) |
| `vit_gtsrb_tact_0_05` | 43 | 6 | 0.007 | 0.07 | 0.00 | 0.14 (7) | 0.11 (93) | 0.43 (7) | 0.03 (93) |
| `vit_cifar10_tact_0_01` | 10 | 9 | 0.109 | 0.84 | 0.95 | 0.37 (84) | 0.62 (16) | 0.95 (84) | 1.00 (16) |
| `vit_cifar10_benign` | 10 | 5 | 0.103 | 0.65 | 0.55 | 0.51 (65) | 0.49 (35) | 0.95 (65) | 0.63 (35) |
| `vit_gtsrb_benign` | 43 | 13 | 0.054 | 0.55 | 0.40 | 0.15 (55) | 0.07 (45) | 0.89 (55) | 0.04 (45) |

The attractor is a large share of the false positives exactly where the classes are few. It is 0.49 to 0.95 of them on the 6 CIFAR-10 models (pooled 463 of 600), 0.07 to 0.61 on GTSRB (pooled 123 of 300) and 0.00 to 0.11 on CIFAR-100 and Tiny. Its share of the clean images is 0.09 to 0.11 on CIFAR-10, 0.007 to 0.054 on GTSRB and 0.007 to 0.014 on CIFAR-100 and Tiny, so it is over-represented among false positives by about 5 to 10 times on CIFAR-10 and GTSRB. That part of the hypothesis holds.

The prediction about the split does not hold. After the stream edit the pooled CIFAR-10 flag rate is 0.41 for attractor false positives and 0.47 for the others (137 images over 6 models), so the non-attractor false positives do not un-flag either. The unedited rates are 0.81 and 0.77. On GTSRB the attractor false positives are 123 of 300 and they un-flag as fully as the others (0.07 against 0.08), so GTSRB did not pass because its attractor share is small. The attractor share of the GTSRB false positives (0.55 to 0.61 on `vit_gtsrb_lf_0_01` and `vit_gtsrb_benign`) is not smaller than on `vit_cifar10_blend_0_1` (0.49). What separates the 2 datasets under the stream edit is the surplus diagnosis above, not the attractor. Per model the non-attractor groups on CIFAR-10 hold 5 to 51 images, so single-model rates there are noisy and the pooled rates carry the reading.

The input edit does see the attractor. Attractor false positives stay flagged after it on GTSRB (0.77 against 0.05 for the others) and on CIFAR-10 (0.81 against 0.66). Keeping only the tokens that sustain the prediction does not move an attractor image, which is what an image that masking cannot move predicts. On CIFAR-10 the other false positives also stay flagged under this edit, so the attractor does not account for all of it.

**Status.** "The CIFAR-10 failure of the stream edit is a class-count effect through the attractor" is not supported by this split, because the attractor and non-attractor false positives fail alike on CIFAR-10 and alike pass on GTSRB. Class count is confounded with dataset (image domain, class balance, the length of the clean-score tail), so a class-count effect through some other route is not excluded and stays a hypothesis. A control would need the same domain at 2 class counts, for example CIFAR-100 trained as its 20 superclasses and as 10 of its classes, then A3 on each. We hold no such models and training them is GPU work, so it was not done.

### Null-referenced PSU

The attractor split above leaves the attractor's stability unexplained by own-class surplus. This experiment asks whether a score that subtracts the free evidence a class receives from the masks alone removes it. It runs on CPU for the probability version and needs the GPU for the logit version, which needs per-pass logit margins the cache does not hold.

**Null outputs.** For each model, PSBD-TM's placement (`before_attention_norm`, `token_mask`) at the model's adaptive rate, with 8 mask seeds of 3 passes each on 3 content-free inputs: a gray image (pixel value 0.5), a black image (pixel value 0) and the fully masked input, every patch token zeroed with the class token kept and no rescaling (rate 1 would divide by 0 in the inverted-dropout scale, so a dedicated module zeroes them). The fully masked output does not depend on the image, so 1 pass is its exact value. The gray image stands for the mean-color image because the registry stores GTSRB's identity normalization and no mean color for it. For each class c the null probability is $b_c$, the mean over passes of $P_c$. The null margin is $m^0_c$, the mean over passes of $\text{logit}_c - \max_{j \ne c} \text{logit}_j$.

**Scores.** With c the unperturbed argmax, $\tilde x_k$ the k-th masked pass, $P_c$ the probability of c and $m$ the margin of c,

$$\phi'_{\text{prob}} = 1 - \frac{\frac1K \sum_k \max(P_c(\tilde x_k) - b_c, 0)}{\max(P_c(x) - b_c, \epsilon)}, \qquad \phi'_{\text{logit}} = 1 - \frac{\frac1K \sum_k \max(m_k - m^0_c, 0)}{\max(m(x) - m^0_c, \epsilon)}$$

with K = 3, $\epsilon_{\text{prob}}$ = 0.01 and $\epsilon_{\text{logit}}$ = 0.5. The primary null for both is the fully masked input. The gray and black nulls are sensitivity checks. The reference is plain fractional PSU. Thresholds are the clean validation quantiles, as everywhere. Nothing is fitted on triggered data and the nulls use no dataset image.

**Panel.** The 12 A3 models first, then the successful_2pt ViT panel (`scripts.paper._common.clearing_cells`), reported per dataset and per attack with paired bootstrap intervals over models. The probability version reads the cached sweep and is computed on CPU. The logit version reads a fresh 3-pass dump of the validation, clean and backdoor splits (mask seed 0, bfloat16) run under the shared GPU lock, and the plain fractional PSU of the same dump is its reference.

**Predictions, written 14:00 on 2026-09-30, after the attractor split was read and before any null was measured.** The attractor split was read first, so predictions 1 to 3 are informed by its unedited rates and are not blind to them.

0. The fully masked input is predicted as the attractor class of the attractor split above on at least 8 of the 12 models, and on the 3 GTSRB and 6 CIFAR-10 models the attractor's $b_c$ is the largest or second largest of all classes.
1. On the 6 CIFAR-10 models, the flag rate of attractor-class clean images (paired clean split, thresholds at the validation quantile) falls from its plain value to within 0.01 to 0.09 at q 0.05 and to at most 0.03 at q 0.01, and the attractor's share of flagged clean images falls to at most 0.25. Falsifier, a pooled attractor flag rate above 0.10 at q 0.05.
2. Triggered detection is unchanged or better. On the 12 models the paired mean difference in TPR at q 0.01, 0.05 and 0.10 is at least -0.02, and the realized FPR stays within 0.02 of nominal.
3. On the panel, TPR at q 0.01 rises by at least 0.03 on CIFAR-10 and GTSRB and changes by at most 0.02 either way on CIFAR-100 and Tiny. Mean AUROC changes by at least -0.01. A CIFAR-10 and GTSRB gain below 0.03 refutes the attractor account of the null score.
4. The logit version agrees with the probability version to within 0.02 in mean AUROC and 0.03 in TPR. Fewer than 1% of the images of every model have a denominator below $\epsilon$ in either version, and those that do are concentrated in the attractor class.

**Results of the probability version (CPU, 2026-09-30).** Nulls are on CPU in float32 and the cached sweep is bfloat16, so the GPU stage below recomputes them. The full-mask output does not depend on the image (largest probability difference between the gray and black inputs 0.0 on every model). The 12 A3 models, plain fractional PSU to the null-referenced score with the fully masked null, flag rates on the clean split at the validation quantile 0.05:

| Model | Attractor | Full-mask class | Attractor rank | Attractor flag rate 0.05 | Attractor share of flags | TPR 0.01 | TPR 0.05 |
|---|---|---|---|---|---|---|---|
| `vit_cifar10_badnet_a2o_0_01` | 6 | 2 | 2 | 0.43 to 0.43 | 0.95 to 0.92 | 0.85 to 0.90 | 0.93 to 0.96 |
| `vit_tiny_badnet_a2o_0_05` | 94 | 94 | 1 | 0.04 to 0.04 | 0.01 to 0.01 | 0.37 to 0.37 | 0.99 to 0.99 |
| `vit_cifar100_blend_0_1` | 33 | 33 | 1 | 0.14 to 0.11 | 0.03 to 0.03 | 1.00 to 1.00 | 1.00 to 1.00 |
| `vit_cifar10_blend_0_1` | 4 | 4 | 1 | 0.28 to 0.09 | 0.53 to 0.18 | 0.55 to 0.64 | 0.71 to 0.81 |
| `vit_cifar10_bpp_0_05` | 7 | 7 | 1 | 0.37 to 0.14 | 0.78 to 0.25 | 0.40 to 0.59 | 0.52 to 0.71 |
| `vit_gtsrb_lf_0_01` | 13 | 13 | 1 | 0.58 to 0.50 | 0.60 to 0.53 | 0.85 to 0.86 | 0.90 to 0.90 |
| `vit_cifar10_wanet_0_1` | 2 | 2 | 1 | 0.52 to 0.46 | 0.96 to 0.74 | 0.02 to 0.03 | 0.04 to 0.07 |
| `vit_tiny_wanet_0_05` | 0 | 0 | 1 | 0.73 to 0.55 | 0.09 to 0.07 | 0.80 to 0.71 | 0.92 to 0.89 |
| `vit_gtsrb_tact_0_05` | 6 | 6 | 1 | 0.46 to 0.02 | 0.11 to 0.00 | 0.03 to 0.03 | 0.23 to 0.28 |
| `vit_cifar10_tact_0_01` | 9 | 5 | 2 | 0.40 to 0.48 | 0.81 to 0.97 | 0.01 to 0.00 | 0.01 to 0.01 |
| `vit_cifar10_benign` | 5 | 2 | 2 | 0.30 to 0.26 | 0.64 to 0.50 | 0.01 to 0.01 | 0.06 to 0.06 |
| `vit_gtsrb_benign` | 13 | 13 | 1 | 0.57 to 0.49 | 0.63 to 0.56 | 0.01 to 0.01 | 0.05 to 0.05 |

The panel is the 56 successful_2pt ViT cells (`clearing_cells`) that carry the cached PSBD-TM sweep this reads, the same 56 as the headline panel of 2026-09-30. The 6 TaCT multi-source retrains that are successful but not yet swept are not in it. Mean per-model values, plain to null-referenced, with the paired difference and its bootstrap interval over models (10000 resamples), for the fully masked null. The attractor flag rate is the share of attractor-class clean images flagged at 0.05, nominal 0.05.

| Group | Models | TPR 0.01 | TPR 0.05 | TPR 0.10 | AUROC | Attractor flag rate 0.05 |
|---|---|---|---|---|---|---|
| all | 56 | 0.735 to 0.753 (+0.018 [+0.005, +0.034]) | 0.841 to 0.857 (+0.016 [+0.006, +0.029]) | 0.872 to 0.884 (+0.012 [+0.002, +0.023]) | 0.951 to 0.942 (-0.009 [-0.031, +0.006]) | 0.457 to 0.342 (-0.115 [-0.164, -0.073]) |
| cifar100 | 12 | 0.790 to 0.789 (-0.000 [-0.003, +0.002]) | 0.935 to 0.933 (-0.003 [-0.013, +0.004]) | 0.964 to 0.960 (-0.004 [-0.016, +0.003]) | 0.979 to 0.977 (-0.002 [-0.006, +0.001]) | 0.462 to 0.426 (-0.036 [-0.064, -0.011]) |
| badnet_a2o | 12 | 0.776 to 0.779 (+0.003 [-0.007, +0.014]) | 0.963 to 0.963 (+0.000 [-0.006, +0.007]) | 0.987 to 0.982 (-0.005 [-0.020, +0.004]) | 0.992 to 0.989 (-0.003 [-0.009, +0.001]) | 0.450 to 0.328 (-0.123 [-0.251, -0.034]) |
| blend | 12 | 0.829 to 0.873 (+0.044 [+0.001, +0.099]) | 0.915 to 0.949 (+0.034 [+0.003, +0.068]) | 0.943 to 0.972 (+0.029 [+0.003, +0.058]) | 0.978 to 0.988 (+0.010 [+0.001, +0.020]) | 0.308 to 0.225 (-0.083 [-0.134, -0.035]) |
| bpp | 12 | 0.757 to 0.793 (+0.036 [+0.003, +0.076]) | 0.847 to 0.878 (+0.030 [-0.003, +0.072]) | 0.888 to 0.911 (+0.023 [-0.008, +0.063]) | 0.948 to 0.959 (+0.011 [-0.004, +0.031]) | 0.452 to 0.275 (-0.177 [-0.330, -0.049]) |
| lf | 12 | 0.904 to 0.910 (+0.006 [+0.001, +0.012]) | 0.959 to 0.964 (+0.005 [+0.001, +0.011]) | 0.970 to 0.972 (+0.003 [+0.000, +0.006]) | 0.980 to 0.981 (+0.002 [+0.001, +0.003]) | 0.560 to 0.470 (-0.090 [-0.148, -0.035]) |
| cifar10 | 15 | 0.573 to 0.640 (+0.067 [+0.030, +0.112]) | 0.660 to 0.715 (+0.055 [+0.023, +0.090]) | 0.706 to 0.749 (+0.043 [+0.010, +0.077]) | 0.919 to 0.892 (-0.027 [-0.111, +0.025]) | 0.406 to 0.273 (-0.133 [-0.190, -0.075]) |
| tact | 4 | 0.009 to 0.008 (-0.000 [-0.001, +0.000]) | 0.059 to 0.070 (+0.012 [+0.000, +0.035]) | 0.127 to 0.135 (+0.008 [+0.000, +0.021]) | 0.788 to 0.636 (-0.153 [-0.415, -0.007]) | 0.504 to 0.404 (-0.101 [-0.333, +0.050]) |
| wanet | 3 | 0.372 to 0.345 (-0.027 [-0.090, +0.009]) | 0.572 to 0.572 (+0.000 [-0.026, +0.032]) | 0.647 to 0.654 (+0.006 [-0.014, +0.033]) | 0.779 to 0.745 (-0.035 [-0.087, -0.002]) | 0.570 to 0.490 (-0.080 [-0.182, +0.000]) |
| gtsrb | 15 | 0.801 to 0.808 (+0.008 [-0.004, +0.025]) | 0.839 to 0.848 (+0.009 [+0.001, +0.019]) | 0.869 to 0.874 (+0.004 [+0.000, +0.010]) | 0.935 to 0.932 (-0.003 [-0.010, +0.002]) | 0.631 to 0.377 (-0.254 [-0.388, -0.134]) |
| lc | 1 | 0.833 to 0.852 (+0.019) | 0.921 to 0.940 (+0.018) | 0.950 to 0.951 (+0.001) | 0.969 to 0.967 (-0.002) | 0.627 to 0.465 (-0.162) |
| tiny | 14 | 0.793 to 0.784 (-0.009 [-0.022, -0.000]) | 0.955 to 0.953 (-0.002 [-0.006, -0.000]) | 0.975 to 0.974 (-0.001 [-0.003, +0.000]) | 0.977 to 0.976 (-0.001 [-0.003, +0.001]) | 0.322 to 0.306 (-0.016 [-0.043, +0.000]) |

Per prediction.

0. **Holds.** The fully masked input is predicted as the attractor class on 9 of the 12 models. On the other 3 (`vit_cifar10_badnet_a2o_0_01`, `vit_cifar10_tact_0_01` and `vit_cifar10_benign`) the attractor is the null's 2nd ranked class. On the 9 GTSRB and CIFAR-10 models the attractor ranks 1st or 2nd, as predicted.
1. **Fails.** The attractor flag rate at 0.05 on the 6 CIFAR-10 A3 models goes from 0.43, 0.28, 0.37, 0.52, 0.40 and 0.30 to 0.43, 0.09, 0.14, 0.46, 0.48 and 0.26. Only `vit_cifar10_blend_0_1` reaches the predicted 0.01 to 0.09 range and only it and `vit_cifar10_bpp_0_05` bring the attractor's share of the flags to 0.25 or below. The falsifier (a pooled rate above 0.10) is met. Over the panel the attractor flag rate falls from 0.457 to 0.342 (-0.115 [-0.164, -0.073]), and most on GTSRB (-0.254) and least on CIFAR-100 and Tiny (-0.036 and -0.016).
2. **Holds.** On the 10 backdoored A3 models the mean TPR difference is +0.026, +0.038 and +0.038 at q 0.01, 0.05 and 0.10, and the realized FPR moves by at most 0.003.
3. **Holds for CIFAR-10, CIFAR-100, Tiny and the AUROC bound, fails for GTSRB.** TPR at q 0.01 changes by +0.067 [+0.030, +0.112] on CIFAR-10, +0.008 on GTSRB (below the predicted 0.03), -0.000 on CIFAR-100 and -0.009 on Tiny. The mean AUROC changes by -0.009 [-0.031, +0.006], inside the predicted bound and not separable from 0. The rule that a CIFAR-10 and GTSRB gain below 0.03 refutes the attractor account of the null score is met on GTSRB.
4. **The probability half holds and the logit half is pending the GPU stage.** Fewer than 1% of the images of every one of the 58 models have a denominator below $\epsilon_{\text{prob}}$ (largest 0.0099 on `vit_tiny_wanet_0_05`) and on the models with the most of them all such images are attractor-class ones.

The GPU stage (`false_positives/run_null_psu.sh`, bfloat16 nulls and fresh dumps of the 12 A3 models first and then the panel, 2 lock slots, memory fraction 0.15, no start between 06:30 and 17:00) is queued to begin after 17:00. Once its dumps exist, `null_psu.py analyze` adds the `logit_*` and `fresh_*` variants beside the ones above and `null_psu.py render` prints their tables, which fills the logit half of prediction 4.

Three readings of the measured numbers. The TPR gain is largest where the attractor's flags fall most on CIFAR-10, Blend and BPP (TPR at 0.01 from 0.55 to 0.64 and from 0.40 to 0.59), and it is nil on CIFAR-100 and Tiny where the attractor holds few images. The null removes only part of the attractor's stability, because the median masked probability of attractor-class validation images exceeds the null probability of their class on 10 of the 12 models (for example 0.296 against 0.165 on `vit_cifar10_badnet_a2o_0_01` and 0.684 against 0.225 on `vit_gtsrb_lf_0_01`). A content-free input keeps no token content, and the masked real image keeps content the null lacks, so the null under-states the free evidence. The other 2 models are `vit_cifar100_blend_0_1` (0.049 against 0.056) and `vit_cifar10_bpp_0_05` (0.341 against 0.361), and both are models where the flag rate does fall (0.28 to 0.09 and 0.37 to 0.14 for Blend and BPP, with the reversed sign on `vit_cifar100_blend_0_1` being small). This is an observation on 12 models and the mechanism is a hypothesis. Second, the prediction that the attractor is never the target holds on these models (every A3 target is class 0, no attractor is class 0 except on `vit_tiny_wanet_0_05`), and `vit_tiny_wanet_0_05` loses TPR (0.80 to 0.71 at 0.01) where the attractor is also the target. Third, the AUROC falls on the TACT and WaNet models (TACT panel mean 0.788 to 0.636, WaNet 0.779 to 0.745, mean of the 5 backdoored CIFAR-10 A3 models 0.825 to 0.727). The TACT paired clean set holds source-class images only, so a class-dependent null shifts that class against the target's, which is a hypothesis and has not been tested. The gray null gives the same panel result (TPR at 0.01 +0.017, attractor flag rate -0.126) and a larger WaNet loss (TPR at 0.01 -0.083), so the choice of null image is not what drives the gain.

### Reading across A, A2 and A3

A2 is the cleanest result. Adding own-class evidence to a clean image makes both PSBD placements flag it with no trigger, and a random push of the same length does not, for PSBD-TM at every length tested. A3 supports the mirror claim on the GTSRB, CIFAR-100 and Tiny models and not on CIFAR-10. A does not support the claim that triggered images carry a large surplus along the backdoor direction, or that this surplus orders the models by detection. On clean images the own-class surplus measured in A3 is 3 to 10 (the non-flagged medians), far above the 1.1 to 2.3 the trigger writes along its own direction. The 2 quantities are measured along different directions, so the comparison is suggestive only.
