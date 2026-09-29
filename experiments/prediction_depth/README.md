# Where and when is the answer decided? (CLS versus all tokens)

## Question

Every detector in the PSBD family perturbs the network and measures how far the
prediction moves (H28). This asks something algebraically different: not how robust the
prediction is, but **at what depth and from which patches, it was already decided**.

A ViT hands over a per-token latent state for free, and H32 already showed that per-token
direction norms localize a trigger (BadNet's top token sits at (13,13), the trigger's own
position). PSBD reads none of that: it collapses everything to 1 scalar from the CLS
prediction. The question is whether the discarded structure carries detection signal.

## Method

A logit lens: read every encoder block through the network's **own** final LayerNorm and
head,

    g(h) = head(LN_final(h))

so every depth is read at the scale the head was trained for, and the statistic is an
argmax, which is invariant to positive rescaling. That matters because reading
intermediate activations with a scale-sensitive statistic is failure mode 3 in the
ledger, and it once produced a clean, monotone, benign-controlled and entirely false
result that survived 2 rounds of follow-up.

4 raw statistics per sample, 1 forward pass, no fitting and no calibration against
the split being scored:

| statistic | reads | a priori direction |
|---|---|---|
| `depth_cls` | CLS only | a trigger is a strong simple feature, so it resolves early: LOW depth |
| `depth_token_min` | earliest any patch locks on | a localized trigger commits almost immediately: LOW |
| `token_agreement` | fraction of patches agreeing with the final answer | see below |
| `token_max_prob` | largest final-class probability any single patch assigns | see below |

Plus a `_dev` variant of each: distance from the clean-validation distribution in either
direction, `-2 |rank_val(s) - 0.5|`.

## Result so far

`vit_cifar10_badnet_a2o_0_1`, against PSBD at its own best placement:

| | AUROC | TPR@1%FPR | TPR@5%FPR | forward passes |
|---|---|---|---|---|
| PSBD (`token_mask @ before_attention_norm`) | 0.961 | **0.000** | 0.710 | 30 |
| `token_agreement` | **0.982** | **0.287** | **0.929** | 1 |
| `depth_cls` (CLS only) | 0.826 | 0.000 | - | 1 |

Benign control 0.500 on both token statistics, 0.473 on `depth_cls`. So all-token beats
CLS-only by 0.156 AUROC, and beats PSBD at 30x lower cost on the operating point PSBD
fails at.

**And then the direction flips.** On a global trigger it reverses:

| cell | trigger | clean agreement | backdoor agreement | AUROC |
|---|---|---|---|---|
| cifar10 badnet_a2o | local patch | 0.429 | **0.064** | 0.982 |
| cifar100 adaptive_blend | global blend | 0.019 | **0.222** | 0.213 |

This is mechanical, not noise. A patch trigger drives the answer from 1 or 2 patches
while the other 194 keep voting their own class, so agreement falls. A global blend
paints every patch with the trigger, so every patch votes the target and agreement
rises. Both are equally far from clean, in opposite directions.

That is what the `_dev` variants are for, and it is **not** the two-sided rule H15
retired. H15's rule picked a tail by reading the AUROC, which needs the poison labels the
detector exists to predict. This ranks against the clean validation split, which the
threat model already grants, and applies 1 fixed rule everywhere: far from clean in
either direction is suspicious. The threshold stays a quantile of the validation
deviation, so the false-positive budget is set exactly as before.

## The headline: a tie on AUROC, a gain at 1% FPR on the hard attacks

Read on the paper panel of 54 models successful at the 2-point clean-accuracy bar,
`depth_soft` against PSBD-TM at the adaptive 0.8 rule, paired within model on the same
splits, AUROC at q0.25 and TPR at the q0.01 budget, with the paper generator's own reader
(`scripts.paper.mech_prediction_depth.measure_cell`):

    PYTHONPATH=. .venv/bin/python scratch/stale_numbers/prediction_depth_panel.py

| cells | n | AUROC depth | AUROC PSBD-TM | AUROC delta | **TPR@1% depth** | **TPR@1% PSBD-TM** | TPR delta | wins |
|---|---|---|---|---|---|---|---|---|
| all | 54 | 0.913 | 0.963 | -0.050, CI [-0.109, +0.000] | 0.734 | 0.747 | -0.013, CI [-0.118, +0.091] | 35 |
| hard (BPP, WaNet, TaCT) | 18 | 0.872 | 0.922 | -0.051, CI [-0.218, +0.076] | **0.753** | **0.568** | **+0.185, CI [+0.027, +0.342]** | 13 |
| hard at 1% and 5% | 12 | 0.819 | 0.947 | -0.128, CI [-0.361, +0.039] | 0.651 | 0.548 | +0.103, CI [-0.078, +0.286] | 7 |

**AUROC is a tie on the hard attacks and the confidence interval says so.** The result is at
the operating point: over all 18 hard models, at a 1% false-positive budget `depth_soft`
catches 0.75 where PSBD-TM catches 0.57, from 1 forward pass against 3. Restricted to 1% and
5% poisoning, the cut this section was first written for, the gain is +0.103 and its
interval crosses 0. On the whole panel `depth_soft` trails PSBD-TM on AUROC by 0.050.

By attack, TPR at 1% FPR:

| attack | n | `depth_soft` | PSBD-TM |
|---|---|---|---|
| WaNet | 3 | **0.929** | 0.372 |
| BPP | 12 | **0.898** | 0.757 |
| LF | 12 | **0.919** | 0.904 |
| Blend | 12 | 0.833 | 0.829 |
| BadNets | 12 | 0.421 | **0.776** |
| TaCT | 3 | 0.000 | **0.012** |

The gain is WaNet and BPP. `depth_soft` loses badly on BadNets and collapses on TaCT (AUROC
0.376 against 0.962), the 2 patch triggers, so it is a complement to PSBD-TM on global
triggers rather than a replacement. The panel holds no SIG model.

The paper's prediction depth macros (`paper/tables/prediction_depth.macros.json`,
`\DepthCells`) were built before the source-mapped exclusion and the 2-point bar, and read a
historical 65 cells until the rebuild of 2026-09-29.

### The earlier reading

The version of this section before 2026-09-29 read a 217-cell sweep: n = 32 cells at 1% and
5% with ASR at least 0.5, below the 0.85 bar and including Adaptive-Blend and LC cells that
never clear it, against PSBD at a placement and rule this README did not state. It reported
`depth_soft` 0.831 against PSBD 0.821 on AUROC (historical) and 0.675 against 0.428 at TPR@1%FPR (+0.248,
CI [+0.090, +0.402], 24 of 32 wins). It lost on Adaptive-Blend (0.465 against 0.521) and
SIG (0.062 against 0.280). It has not been re-run.

**Benign controls, all 4 datasets:** 0.481, 0.481, 0.501, 0.498, mean **0.491**, with TPR
at the 1% budget reading 0.010 to 0.012, i.e. exactly nominal. The signal is not an
artifact of applying a trigger.

### What has to be settled before this is a claim

- **Prior art.** TED (S&P 2024) and TED++ do layer-wise trajectory analysis with clean-only
  calibration for backdoor input detection, above 0.95 AUROC, on ResNets. That is the number
  to beat, not a citation. Orion (IJCAI 2023) uses internal-readout-disagrees-with-final-
  answer as a per-input poisoned score.
- The AUROC delta is not significant. The claim is a low-FPR claim and must be stated as one.
- Mostly single seed. A few cells have `_seed_1` and `_seed_2` replicates and those should
  set the pre-registered n.

## The earlier verdict, retained: on AUROC alone it loses

This section and the 3 after it are earlier readings on the 217-cell sweep. They hold cells
that are not on the current panel (`vit_gtsrb_lc_0_1`, `vit_gtsrb_sig_0_1`, the
Adaptive-Blend cells) and compare against PSBD at earlier placements, and they have not been
re-run on the current panel.

BadNet is the wrong thing to judge this on. It implants at ASR 0.997 to 1.000 everywhere
and the ledger already records that "its detection behavior is least like the others".
Restricted to the attacks that are actually hard (adaptive_blend, WaNet, LC, SIG, LF, Bpp,
TaCT), over 41 cells at every poison rate:

| | mean AUROC | inversions |
|---|---|---|
| `depth_cls` | **0.720** | 10/41 |
| PSBD | **0.796** | |

So the earlier "0.983 against PSBD's 0.974 on GTSRB" was GTSRB **including BadNet and
Blend at 10%**, and it does not survive the cut. The single worst case is the attack
designed to defeat detectors:

| cell | `depth_cls` | PSBD |
|---|---|---|
| `vit_gtsrb_adaptive_blend_0_05` | 0.695 | **0.986** |
| `vit_cifar10_adaptive_blend_0_1` | 0.643 | **0.924** |
| `vit_gtsrb_lc_0_1` | 0.421 | 0.522 |
| `vit_gtsrb_sig_0_1` | 0.427 | 0.451 |
| `vit_cifar10_tact_0_05` | 0.034 | - |

**Where it genuinely wins is WaNet**, and only WaNet:

| cell | `depth_cls` | PSBD |
|---|---|---|
| `vit_cifar10_wanet_0_1` | **0.848** | 0.408 |
| `vit_cifar100_wanet_0_1` | **0.817** | 0.702 |
| `vit_gtsrb_wanet_0_1` | **0.933** | 0.854 |

WaNet is a warp, and H27 already recorded that `token_mask` is worst on exactly that
attack (0.747 against 0.985 on the patch trigger). A depth statistic reading the residual
stream appears to catch what a token-space perturbation misses. That is a narrow, specific
and checkable claim, and it is the only one this experiment currently supports.

## The earlier CIFAR-10 result, retained as a caution

## The panel verdict on the token statistic

The CIFAR-10 cell above is not representative, and GTSRB (43 classes) shows why. Same
protocol, 10% poisoning, `token_agreement` read in the direction its own mechanism
predicts for that class count:

| GTSRB group | n | token agreement | PSBD |
|---|---|---|---|
| dirty-label (`all_to_one`) | 6 | 0.948 | **0.974** |
| clean-label (`sig`, `lc`) | 2 | 0.496 | 0.487 |
| all-to-all | 1 | 0.449 | 0.306 |
| benign control | 1 | **0.503** | 0.500 |

Per cell on the dirty-label group: blend 0.987, bpp 0.983, adaptive_blend 0.975, lf 0.956,
wanet 0.947, badnet 0.838. Real signal, clean benign control and still **below PSBD**,
whose TPR at 1% FPR on those same cells is 0.966 to 1.000.

It also fails exactly where PSBD fails: the 2 clean-label attacks and all-to-all. A
clean-label trigger reinforces the true class rather than overriding it, so the agreement
never jumps, which is the mechanism working as stated and predicting its own failure.

**And the sign is not stable.** It depends on the class count, not only on the trigger's
spatial extent:

| cell | raw AUROC | clean -> backdoor agreement |
|---|---|---|
| cifar10 badnet_a2o | 0.982 | 0.429 -> 0.064 |
| gtsrb badnet_a2o | 0.162 | 0.071 -> 0.033 |

With 10 classes a clean patch token often resolves the true class (agreement 0.429), so a
local trigger LOWERS agreement. With 43 classes clean agreement is already near chance
(1/43), so the confident trigger patches RAISE it. The same attack inverts between
datasets. A sign router built on the suspect-pool-minus-validation deviation recovers 7 of
9 cells (0.423 raw, 0.577 flipped, 0.731 routed) but fails wherever that deviation is
near zero, which is exactly the GTSRB badnet case.

## Where it does win

Only where PSBD has a low-FPR collapse. On `vit_cifar10_badnet_a2o_0_1`, PSBD reads AUROC
0.961 with TPR **0.000** at 1% FPR, and token agreement reads 0.982 with **0.287**. That
is 1 of the 6 cells identified this session where PSBD has AUROC >= 0.85 and TPR@1%FPR
< 0.05. Whether the per-token reading is systematically strong on exactly those cells is
the one open question worth the GPU time, and the 217-cell panel answers it.

## Methodological note

Pilot on GTSRB, not CIFAR-10. The 10-class result was flattering by a wide margin, and the
statistic's chance level is 1/K, so a class-count-sensitive statistic looks far stronger
there than it is.

## Running it

    PYTHONPATH=. python experiments/prediction_depth/measure.py --checkpoint-folder <folder>

Writes `results/<folder>/prediction_depth.json` and, unlike `cli/baselines.py`, also
`prediction_depth_scores.pt` with the per-sample tensors, so any new scoring rule is a
CPU rescore rather than another GPU pass.
