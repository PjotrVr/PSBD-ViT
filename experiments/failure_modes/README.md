# Why token masking at the attention input fails on 2 models (E2)

Superseded on 2026-09-29 for panel numbers: the numbers here were computed on the 57-model panel. Current values are in paper/headline.tex.

## Question

`before_attention_norm_token_mask` (site A, `RECOMMENDED_PLACEMENT`) reads CIFAR-10
WaNet 10% at AUROC 0.459 and CIFAR-10 SIG 10% at 0.418, the 2 inverted cells of the 57
clearing models, while `before_attention_residual_token_mask` (site B, masking the
attention branch's own output rather than its input) and dropout on the residual
stream (`pre_residual`, `post_residual`) read the same 2 models well. Site A itself
reads the same attacks well elsewhere: CIFAR-10 WaNet 5% (0.937) and GTSRB WaNet 10%
(0.948). Why does the position of the mask relative to attention matter this much for
exactly these 2 cells.

The experiment was run on 3 failing models and 3 controls. 2 of them, the failing Tiny
ImageNet TaCT 1% and the CIFAR-100 TaCT 1% control, are source-mapped: both read 0.00
clean source-class accuracy, so they send their whole source class to the target with no
trigger and the ledger excludes them as backdoors (`docs/runs/2026-09-24-tact-multisource.md`).
Their rows are removed from every table below and their records stay on disk. What remains is
2 failing models and 2 WaNet controls, so no TaCT reading survives. The SIG model is under
audit (`docs/audits/2026-09-29-experiment-audit.md`): it was trained at SIG amplitude 0.1,
the code now builds 0.157, and the margin and patching rows here come from fresh forward
passes on the rebuilt trigger.

The paper panel is now the 56 models successful at the 2-point clean-accuracy bar (2026-09-30). Of the 4
models left here only CIFAR-10 WaNet 10% is on it. CIFAR-10 SIG 10% (clean accuracy 10.6
points below its benign reference), CIFAR-10 WaNet 5% (3.9 points) and GTSRB WaNet 10% (3.3
points) clear the ASR bar but fail the 2-point bar (`results/coverage/coverage.json`), so the
2 controls and the second failing model are read here as backdoors, not as panel models.

## Method

`measure.py` reads the existing stage-1 caches for the failing models and their
attack-matched controls (`MODELS`), and adds 2 GPU measurements on 500 paired clean
and triggered images per model (`experiments.whole_network_erasure.measure.paired_rows`):

1. From `results/<folder>/psbd_metrics.json`, the full rate ladder of site A, site B,
   `pre_residual` and `post_residual`: fractional-PSU AUROC and TPR at the
   quantile-0.10 rule against the achieved clean-validation shift ratio, plus the
   rate the 0.8 adaptive rule selects.
2. Per image, at each of site A and `post_residual`'s own adaptively selected rate:
   fractional PSU of clean and triggered inputs, and the share of attack-success
   captured triggered images (`defenses.decision.attack_success_mask`) whose
   per-pass argmax ever leaves the no-perturbation label under the mask, against the
   same share for the paired clean images. A gap within 0.05 reads as
   `nothing_separates`, a positive gap (triggered shifts more) as `wrong_direction`,
   a negative gap as `as_expected`.
3. The logit margin of the predicted class over the runner-up, no perturbation,
   clean and triggered (`experiments.sam_mechanism.measure.predicted_class_margin`).
4. Activation patching (`experiments.residual_stream_mechanism.activation_patching`)
   at blocks 4, 8 and 12 for `site="resid"` and `site="attn"`: recovery of the clean
   answer when the trigger's own token group is patched from the clean run, against
   the CLS token and a random group of the same size. The trigger group is
   attack-specific: for WaNet the 20% of patches (39 of 196) the warp field
   displaces most, read from `attacks.wanet`'s own grid builder at the checkpoint's
   parameters. For SIG the 20% of patches nearest the column sinusoid's peak
   amplitude, read from `attacks.sig`'s own signal builder.

    PYTHONPATH=. .venv/bin/python experiments/failure_modes/measure.py \
        --checkpoints-dir checkpoints --raw-data-dir raw_data

Writes `results/_experiments/failure_modes/<folder>.json`. Ran in about 5 minutes on
the login A100. `experiments/wanet_cifar10_audit/` already confirmed the 0.459
reading is real (stable across 3 mask seeds, reproduces on an independently trained
BackdoorBench reference checkpoint) and ruled out a checkpoint-wide or WaNet-wide
problem. This experiment explains the mechanism.

## 1. Rate ladder: where each site saturates

AUROC and clean-validation shift ratio at the rate the 0.8 adaptive rule selects,
and the best AUROC anywhere on that placement's own ladder:

| model | role | placement | adaptive rate | shift@adaptive | AUROC@adaptive | AUROC ceiling (rate) |
|---|---|---|---|---|---|---|
| cifar10 wanet 10% | failing | site A | 0.5 | 0.855 | 0.459 | 0.767 (p=0.9) |
| cifar10 wanet 10% | failing | site B | 0.7 | 0.861 | 0.846 | 0.907 (p=0.8) |
| cifar10 wanet 10% | failing | pre_residual | 0.5 | 0.879 | 0.946 | 0.951 (p=0.6) |
| cifar10 wanet 10% | failing | post_residual | 0.09 | 0.866 | 0.947 | 0.948 (p=0.1) |
| cifar10 sig 10% | failing | site A | 0.8 | 0.817 | 0.418 | 0.878 (p=0.5) |
| cifar10 sig 10% | failing | site B | 0.7 | 0.946 | 0.934 | 0.934 (p=0.7) |
| cifar10 sig 10% | failing | pre_residual | 0.5 | 0.860 | 0.896 | 0.913 (p=0.4) |
| cifar10 sig 10% | failing | post_residual | 0.09 | 0.832 | 0.919 | 0.919 (p=0.09) |
| cifar10 wanet 5% | control | site A | 0.6 | 0.809 | 0.937 | 0.940 (p=0.9) |
| cifar10 wanet 5% | control | site B | 0.7 | 0.841 | 0.943 | 0.948 (p=0.9) |
| cifar10 wanet 5% | control | pre_residual | 0.4 | 0.823 | 0.951 | 0.951 (p=0.4) |
| cifar10 wanet 5% | control | post_residual | 0.07 | 0.807 | 0.956 | 0.956 (p=0.07) |
| gtsrb wanet 10% | control | site A | 0.4 | 0.916 | 0.948 | 0.959 (p=0.9) |
| gtsrb wanet 10% | control | site B | 0.5 | 0.907 | 0.707 | 0.959 (p=0.7) |
| gtsrb wanet 10% | control | pre_residual | 0.3 | 0.944 | 0.958 | 0.963 (p=0.8) |
| gtsrb wanet 10% | control | post_residual | 0.07 | 0.962 | 0.960 | 0.963 (p=0.1) |

On CIFAR-10 WaNet 10%, site A's clean-validation shift ratio saturates at 0.906
by rate 0.7 and stays there through rate 0.9, yet its AUROC keeps climbing across
that whole flat stretch (0.556 at 0.7 to 0.767 at 0.9), the ceiling-without-rescue
pattern the plan asked to make visible: no rate on the ladder reaches the dropout
placements' 0.92 to 0.95. The same holds for SIG (ceiling 0.878, still short of site
B's 0.934).

## 2. Per-image response: which population moves under the mask

Fractional PSU and the share of images whose prediction ever leaves its
no-perturbation label under the mask, attack-success captured triggered images only
and their paired clean images, at each placement's own adaptive rate:

| model | role | placement | rate | n captured | PSU ratio clean | PSU ratio triggered | clean shift share | triggered shift share | gap | classification |
|---|---|---|---|---|---|---|---|---|---|---|
| cifar10 wanet 10% | failing | site A | 0.5 | 6404 | 0.802 | 0.817 | 0.880 | 0.942 | +0.062 | wrong_direction |
| cifar10 wanet 10% | failing | post_residual | 0.09 | 6404 | 0.896 | 0.706 | 0.992 | 0.311 | -0.682 | as_expected |
| cifar10 sig 10% | failing | site A | 0.8 | 6485 | 0.861 | 0.894 | 0.823 | 0.938 | +0.115 | wrong_direction |
| cifar10 sig 10% | failing | post_residual | 0.09 | 6485 | 0.878 | 0.692 | 0.979 | 0.400 | -0.579 | as_expected |
| cifar10 wanet 5% | control | site A | 0.6 | 6914 | 0.840 | 0.295 | 0.904 | 0.249 | -0.655 | as_expected |
| cifar10 wanet 5% | control | post_residual | 0.07 | 6914 | 0.852 | 0.314 | 0.936 | 0.015 | -0.922 | as_expected |
| gtsrb wanet 10% | control | site A | 0.4 | 10013 | 0.900 | 0.113 | 0.959 | 0.011 | -0.948 | as_expected |
| gtsrb wanet 10% | control | post_residual | 0.07 | 10013 | 0.955 | 0.469 | 0.968 | 0.001 | -0.967 | as_expected |

Both WaNet 10% and SIG 10% invert under site A: the triggered population shifts
MORE than its paired clean population, the opposite of what PSBD's threshold needs,
and both correct back to `as_expected` under `post_residual`.

## 3. Margin: is the shortcut a coin flip

Logit margin of the predicted class over the runner-up, no perturbation, on the
same 500 paired images:

| model | role | n pairs | clean margin | triggered margin | gap |
|---|---|---|---|---|---|
| cifar10 wanet 10% | failing | 500 | 7.13 | 6.31 | -0.82 |
| cifar10 sig 10% | failing | 500 | 6.89 | 9.39 | +2.51 |
| cifar10 wanet 5% | control | 500 | 6.12 | 6.56 | +0.44 |
| gtsrb wanet 10% | control | 500 | 7.51 | 7.84 | +0.33 |

Neither failing model reads as a coin: every triggered margin sits within a few
logits of its clean counterpart, on the same scale as the controls. SIG has a LARGER
triggered margin than clean, the confident-shortcut case rather than the fragile one.
The margin statistic rules itself out as the explanation for both failures.

## 4. Activation patching: where the trigger's evidence lives

Recovery of the clean answer (1.0 is fully restored, 0.0 is no recovery) when the
named token group is patched from the clean run into the triggered run, at blocks
4, 8 and 12:

| model | role | site | group | block 4 | block 8 | block 12 |
|---|---|---|---|---|---|---|
| cifar10 wanet 10% | failing | resid | trigger | 0.880 | 0.100 | 0.006 |
| cifar10 wanet 10% | failing | resid | random_same_size | 0.359 | 0.040 | 0.004 |
| cifar10 wanet 10% | failing | resid | cls | 0.005 | 0.643 | 0.976 |
| cifar10 wanet 10% | failing | attn | trigger | -0.003 | -0.006 | -0.000 |
| cifar10 wanet 10% | failing | attn | random_same_size | 0.026 | -0.004 | -0.000 |
| cifar10 wanet 10% | failing | attn | cls | 0.004 | 0.226 | 0.083 |
| cifar10 sig 10% | failing | resid | trigger | 0.159 | 0.083 | 0.039 |
| cifar10 sig 10% | failing | resid | random_same_size | 0.091 | 0.072 | 0.036 |
| cifar10 sig 10% | failing | resid | cls | 0.006 | 0.051 | 0.739 |
| cifar10 sig 10% | failing | attn | trigger | 0.035 | 0.006 | -0.000 |
| cifar10 sig 10% | failing | attn | random_same_size | 0.035 | 0.006 | -0.000 |
| cifar10 sig 10% | failing | attn | cls | 0.000 | 0.027 | 0.326 |
| cifar10 wanet 5% | control | resid | trigger | 0.791 | 0.073 | 0.008 |
| cifar10 wanet 5% | control | resid | random_same_size | 0.253 | 0.034 | 0.006 |
| cifar10 wanet 5% | control | resid | cls | 0.004 | 0.494 | 0.945 |
| cifar10 wanet 5% | control | attn | trigger | 0.004 | -0.006 | -0.000 |
| cifar10 wanet 5% | control | attn | random_same_size | 0.017 | -0.004 | -0.000 |
| cifar10 wanet 5% | control | attn | cls | 0.005 | 0.165 | 0.077 |
| gtsrb wanet 10% | control | resid | trigger | 0.693 | 0.099 | 0.025 |
| gtsrb wanet 10% | control | resid | random_same_size | 0.095 | 0.019 | 0.005 |
| gtsrb wanet 10% | control | resid | cls | 0.001 | 0.117 | 0.690 |
| gtsrb wanet 10% | control | attn | trigger | 0.022 | -0.002 | -0.000 |
| gtsrb wanet 10% | control | attn | random_same_size | 0.010 | 0.000 | -0.000 |
| gtsrb wanet 10% | control | attn | cls | 0.001 | 0.159 | 0.131 |

The `attn` site (patching the attention branch's own output before it joins the
residual add) never recovers much of anything at any block for any group except the
CLS token late in the network, so the causal story lives entirely at `resid`
(patching the residual stream a block receives). Patching just the CLS token at
block 12 recovers most of the answer everywhere, a general property of the last
block rather than something specific to a failure: by then only 1 more block stands
between the residual stream and the classifier, so resetting the CLS position alone
resets most of what still matters.

## Why site A fails on these 2 models

CIFAR-10 WaNet 10% and CIFAR-10 SIG 10% fail for the same structural reason.

WaNet's warp field displaces every one of the 196 patches by a different amount, so
its evidence is spread rather than confined to a small set of tokens: patching only
the 39 patches the warp moves most restores 88.0% of the clean logit margin at
block 4 of the residual stream, but patching a random 39-patch group restores 35.9%
too, so most of the causal weight sits outside the specific patches token masking
would need to target. A token mask at site A zeroes whole tokens BEFORE they enter
that block's own attention computation, so whichever tokens survive still get to
attend over the full sequence and route information back to every output position,
including the masked ones. Attention's own mixing step, which runs strictly after
the mask, lets a redundant signal like a smooth warp largely rebuild itself from
its unmasked neighbours while the same masking removes real class evidence the
model cannot rebuild the same way. The per-image numbers show this precisely: under
site A, 94.2% of captured triggered images move away from their base prediction
against 88.0% of paired clean images, the wrong direction for PSBD's threshold, and
the margin reading (clean 7.13, triggered 6.31) rules out a weak-shortcut
explanation on its own, since the 2 margins sit close together on the same scale as
every control. `post_residual` dropout, which perturbs a small fraction of every
channel at every token independently and has no attention step downstream within
the same block to route around it, reads the same checkpoint at 0.947 and the
direction reads as expected (triggered shift 31.1% against clean 99.2%).

CIFAR-10 SIG 10% fails by the same mechanism taken to its extreme: the sinusoid is
added to every pixel of the image, so every one of the 196 patches carries some of
it and there is no small subset for a token mask to ever isolate. The 20% of
patches nearest the sinusoid's peak amplitude recover only 15.9% of the clean
margin at block 4, barely above a random group's 9.1%. By block 12 nearly all
of the recoverable signal (73.9%) sits in the CLS token alone, meaning the model
has already funnelled the trigger's evidence past the point any patch-level mask
could reach it. The margin reading rules out the coin-flip story even more directly
than for WaNet: the triggered margin is 2.51 logits LARGER than clean (9.39 against
6.89), a confident, not a fragile, shortcut. Site A again inverts (93.8% of
triggered images shift against 82.3% of clean, the larger gap of the 2 failing
models), and `post_residual` again reads it correctly (0.919 AUROC, triggered shift
40.0% against clean 97.9%), because a perturbation that touches every channel a
little cannot avoid damaging SIG's globally redundant signal roughly in step with
ordinary class evidence, while a perturbation that removes whole tokens can miss
SIG's signal almost entirely no matter which tokens it happens to zero.

## Files

- `measure.py`: all 4 measurements, 1 checkpoint at a time, GPU required for parts
  3 and 4. `PYTHONPATH=. .venv/bin/python experiments/failure_modes/measure.py`.
- `results/_experiments/failure_modes/<folder>.json`: 1 file per model, holding the
  rate ladder, the per-image summary, the margin reading and the activation
  patching rows, the source of every table above.
