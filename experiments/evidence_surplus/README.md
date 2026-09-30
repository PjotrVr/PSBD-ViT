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

### Reading across A, A2 and A3

A2 is the cleanest result. Adding own-class evidence to a clean image makes both PSBD placements flag it with no trigger, and a random push of the same length does not, for PSBD-TM at every length tested. A3 supports the mirror claim on the GTSRB, CIFAR-100 and Tiny models and not on CIFAR-10. A does not support the claim that triggered images carry a large surplus along the backdoor direction, or that this surplus orders the models by detection. On clean images the own-class surplus measured in A3 is 3 to 10 (the non-flagged medians), far above the 1.1 to 2.3 the trigger writes along its own direction. The 2 quantities are measured along different directions, so the comparison is suggestive only.
