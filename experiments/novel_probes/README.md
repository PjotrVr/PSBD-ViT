# Novel PSBD-like probes built from the measured mechanism

## Status

**Future work, not run on GPU.** On 2026-09-29 the user moved new probes out of the paper's scope. What exists is the design below, the operators in `operators.py`, the 2-stage runner (`measure.py`, `evaluate.py`), the unit tests in `tests/test_novel_probes.py`, a CPU smoke on 1 model and the band-partner reading from the existing caches. No probe has a verdict and no number here is a detection result. The GPU queue was stopped when the plan changed. It had already written raw passes for 2 of the 10 development models (`vit_cifar10_wanet_0_1` and `vit_cifar10_badnet_a2o_0_01` under `results/_experiments/novel_probes/passes/`), which are kept and have not been scored or read, and no held-out model was touched.

## Question

PSBD-TM works on ViT-B/16 because of 3 measured facts. A patch trigger lives in its own tokens and the class token reads it in blocks 9 to 12, so the backdoor needs 1 intact read (an OR) while a clean prediction needs evidence from many tokens (an AND). Clean fragility is a per-image logistic critical rate, and global triggers are redundant across tokens. WaNet reverses the depth pattern and late residual dropout catches it. The neuron-bias, target-landing and confidence accounts are refuted. The question here is whether a probe designed from those facts, and not another dropout placement, detects better than the 2 references on the same models and splits. Reference 1 is PSBD-TM alone. Reference 2 is the plain minimum of the PSBD-TM and `pre_residual_blocks_5_8` clean-validation percentiles, the best combination the cache readouts found.

A probe counts as better when its mean TPR at 1%, 5% and 10% nominal FPR is above both references on the same models, with the realized FPR reported beside each TPR. It is read standalone and as a min-union member with PSBD-TM. The union with the late band (`pre_residual_blocks_9_12`) is reported as a third, secondary reference.

## Probes and their hypotheses

Every operator is in `operators.py` and attaches through hooks or a removable attention wrapper, so no weight changes and `models.positions.unplug_dropout` restores the model exactly. None of them belongs in the production packages yet, since none has a verdict.

**Class-token read knockout** (`cls_read_knockout`). In blocks 9 to 12 each head's edge from the class token to each patch key is removed with probability $p$ by masking its attention logit, and every patch token keeps attending and keeps its residual entry. It attacks the read and leaves storage alone. It was coded with PSBD's sign and read inverted on the CPU smoke (16 pairs of 1 development model), and the mechanism explains the inversion: without its late reads the trigger's OR fails, while the class token already holds clean evidence gathered in blocks 1 to 8. The sign was therefore switched to fragile on the development set, and this README discloses it. The primary reading flags inputs that move more than clean ones at the clean-stability rung, and PSBD's sign at the adaptive rung stays as a secondary reading. Prediction: standalone AUROC of at least 0.9 on the BadNets and TaCT development models, weaker on global triggers. The control is the same knockout in blocks 1 to 4 (`cls_read_knockout_early`), where the class token reads little of the trigger. Refuted if the late knockout reads below 0.8 on the 3 patch models, or within 0.05 of its early control there.

**True key mask** (`key_mask`, N1). A masked token disappears as a key for every query and head in every block and keeps its own query and residual entry, so no zeroed token leaves the shared key that makes PSBD-TM's masked tokens act as 1 sink (E6). Hypothesis: if missing evidence alone carries PSBD-TM, the key mask separates as well. If the sink adds clean damage, the key mask needs a higher rate to move clean predictions and separates worse. Refuted, in favor of the sink mattering, if its standalone AUROC trails PSBD-TM's by more than 0.02 on the pooled development models.

**Stratified token mask** (`stratified_token_mask`, N13). PSBD-TM's operator at the attention input, with each token masked in exactly $\lfloor 4p \rfloor$ blocks of every window of 4 consecutive blocks (plus 1 more with the leftover probability), so its marginal rate stays $p$ and it is visible in at least 1 block of every window. The event that breaks a triggered patch prediction under PSBD-TM, every trigger token hidden in all 4 late blocks, becomes impossible. Prediction from the plan (X14): clean shift at equal $p$ within 0.05 of PSBD-TM's and a higher TPR at 1% FPR on BadNets, with no gain on the trigger-conditional TaCT models. Refuted if the clean shift at equal $p$ falls by more than 0.05, or the patch TPR at 1% FPR does not rise. The guarantee needs $p \le 0.75$, so a model whose adaptive rule would ask for more is read at its nearest rung.

**Rollout-ranked token drop** (`rollout_token_drop`, S2). The $r$ patch tokens with the largest attention rollout (Abnar and Zuidema) to the class token are dropped at the attention input of every block, deterministically. A patch trigger's tokens should top the rollout, so removing them flips a triggered prediction while a clean one survives, which is the fragile sign fixed before any triggered image. The control is $r$ random tokens (`random_token_drop`). Prediction: standalone AUROC of at least 0.8 on the 3 patch models with the control near 0.5, and weak on global triggers. Refuted if the patch AUROC is below 0.8 or within 0.05 of the random control.

**Active-neuron MLP dropout** (`active_neuron_dropout`, S1). The plain channel mask at `mlp_neurons` restricted to positive post-GELU activations, a unit's mask shared across the sample's tokens. The plan predicts it at or below the plain channel mask, since the neuron-level account is refuted. The control is the cached `mlp_neurons_channel_mask`. Refuted if it beats the plain mask by more than 0.02 AUROC.

**Native-resolution jitter** (`native_jitter`, X6). The image is resampled at its native resolution at the identity grid plus $a\,\xi/h$ with $\xi$ uniform in $[-1, 1]$ per pixel, the offset scale of WaNet's noise mode. A jittered triggered image is to first order a noise-mode training image with its true label, so on noise-mode models the triggered prediction should move and the clean one should not. Its sign is fragile, fixed from that argument before any triggered image was scored, and the amplitude is the largest rung of $\{0.25, 0.5, 1, 2\}$ whose clean validation shift stays at or below 0.05. Both development WaNet models carry noise mode (`cover_rate` 0.2). Prediction: standalone AUROC of at least 0.9 on both. Refuted if either reads below 0.9.

## Method

Stage 1 (`measure.py`, GPU) loads each model, builds the PSBD splits with `data.splits.build_psbd_loaders_from_checkpoint` and keeps all 2000 validation images, the first 1000 triggered images of the manifest order and their clean twins, the plan's U3 unit. Every probe's ladder runs on validation first in bfloat16 with $k = 3$ passes (1 for the deterministic rollout drop), and the rate is chosen there. A probe with PSBD's sign takes the smallest rung whose clean shift reaches 0.8 (`select_rate_adaptively`) and the nearest rung when none does, labeled as such. A fragile probe takes the largest rung whose clean shift stays at or below 0.05. Only the chosen rungs then run on the triggered and clean pairs. Stage 2 (`evaluate.py`, CPU) scores every reading with the fractional PSU (`psu_ratio_from_cache`), negated for a fragile probe so a low score means poisoned throughout, thresholds it at the 0.01, 0.05 and 0.10 quantiles of its own validation scores and reads TPR, realized FPR and AUROC with `detection_report`. A union is the plain minimum of each member's clean-validation percentile, thresholded at the same quantiles of the union's validation scores.

On a backdoored model the references are read from the stage-1 caches, the passes `cli.analyze` scores, restricted to the same validation, clean and triggered rows. Stage 2 also reports how far the fresh PSBD-TM of stage 1 agrees with the cached one, which checks the runner. On a benign model the 3 triggers `badnet_a2o`, `blend` and `wanet` are stamped with target 0 and both references are recomputed by stage 1. A benign reading passes when every AUROC lies within 0.1 of 0.5.

The development set is the 10 models of `experiments/cache_readouts/dev_set.json`, and `vit_gtsrb_wanet_0_1` is reported but not pooled because it fails the 2-point clean bar. `measure.py` refuses a backdoored CIFAR-100 or Tiny model until `preregistration.json` exists.

## Selection rule for a future run

The rule was fixed before any development number existed and is kept for whoever runs this. The candidates are ranked by their paired mean TPR gain over reference 2 as a min-union member with PSBD-TM, averaged over 1%, 5% and 10% FPR on the pooled development models. At most 2 are frozen, and a candidate is eligible only if it passes the benign gate and its mean absolute gap between realized and nominal FPR stays within 0.02. `preregister.py` writes the frozen probes, their rate rules, their predicted TPR at each FPR and their union rule into `preregistration.json` with the sha256 of the records it read. The held-out half is then read once, and `measure.py` refuses a backdoored CIFAR-100 or Tiny model until that file exists.

## Commands

```bash
source .venv/bin/activate
export PYTHONPATH=.
python -m pytest tests/test_novel_probes.py
python -m experiments.novel_probes.band_partner
nohup experiments/novel_probes/run_queue.sh > scratch/novel_probes/queue.log 2>&1 &
python -m experiments.novel_probes.evaluate --set dev
python -m experiments.novel_probes.evaluate --set benign
python -m experiments.novel_probes.report
```

The CPU smoke ran as `python -m experiments.novel_probes.measure --folders vit_cifar10_badnet_a2o_0_01 --smoke --device cpu --pairs 16 --validation-count 32 --batch-size 16`. `run_queue.sh` holds `scratch/gpu.lock` for 1 model at a time, caps the process at 0.15 of the card and starts nothing between 06:30 and 17:00.

## Results

<!-- results:begin -->

**Band partner.** Blocks 5 to 8 against blocks 9 to 12 as the min-rank partner of PSBD-TM, paired on the models where both partners have an adaptive rate, read from `fusion_rules_<set>.json` (`band_partner.json`).

| set | n | field | middle | late | middle minus late [95% CI] | models middle higher, lower | realized FPR middle, late |
|---|---|---|---|---|---|---|---|
| dev | 5 | auroc | 0.947 | 0.951 | -0.004 [-0.014, +0.005] | 1, 4 | --, -- |
| dev | 5 | q0.01 | 0.582 | 0.596 | -0.014 [-0.088, +0.056] | 1, 3 | 0.010, 0.009 |
| dev | 5 | q0.05 | 0.680 | 0.689 | -0.009 [-0.038, +0.031] | 1, 4 | 0.037, 0.042 |
| dev | 5 | q0.10 | 0.752 | 0.763 | -0.011 [-0.036, +0.016] | 1, 4 | 0.071, 0.080 |
| holdout | 26 | auroc | 0.983 | 0.985 | -0.002 [-0.006, +0.002] | 6, 20 | --, -- |
| holdout | 26 | q0.01 | 0.841 | 0.830 | +0.010 [-0.080, +0.107] | 7, 19 | 0.010, 0.010 |
| holdout | 26 | q0.05 | 0.953 | 0.963 | -0.010 [-0.036, +0.010] | 6, 17 | 0.052, 0.050 |
| holdout | 26 | q0.10 | 0.973 | 0.977 | -0.004 [-0.011, +0.001] | 8, 13 | 0.102, 0.102 |
| panel | 39 | auroc | 0.981 | 0.983 | -0.002 [-0.005, +0.001] | 12, 27 | --, -- |
| panel | 39 | q0.01 | 0.839 | 0.832 | +0.007 [-0.054, +0.072] | 13, 24 | 0.010, 0.010 |
| panel | 39 | q0.05 | 0.927 | 0.934 | -0.007 [-0.025, +0.007] | 12, 22 | 0.049, 0.049 |
| panel | 39 | q0.10 | 0.950 | 0.954 | -0.004 [-0.010, +0.001] | 12, 19 | 0.098, 0.098 |

**CPU smoke.** 1 model (`vit_cifar10_badnet_a2o_0_01`), 16 triggered images and their clean twins, the first 32 validation images and the top 2 rungs of each ladder (`dev_smoke.json`). It tests that the pipeline runs end to end and says nothing about detection, since a quantile of 32 validation scores is not a threshold.

The unperturbed predictions of the runner agree with the cached ones on 1.000 of the triggered rows. The runner's PSBD-TM chose rate 0.8 against 0.8 in the cache.

| reading | rate | rule | clean validation shift | AUROC alone | AUROC in the union with PSBD-TM |
|---|---|---|---|---|---|
| cls_read_knockout | 0.999 | smallest_rung | 0.885 | 0.781 | 1.000 |
| cls_read_knockout@canonical | 0.999 | adaptive | 0.885 | 0.219 | 1.000 |
| cls_read_knockout_early | 0.999 | smallest_rung | 0.896 | 0.809 | 1.000 |
| cls_read_knockout_early@canonical | 0.999 | adaptive | 0.896 | 0.191 | 0.969 |
| key_mask | 0.99 | adaptive | 0.958 | 0.434 | 0.844 |
| stratified_token_mask | 0.75 | adaptive | 0.833 | 1.000 | 1.000 |
| active_neuron_dropout | 0.9 | adaptive | 0.875 | 0.227 | 1.000 |
| rollout_token_drop | 32 | stability | 0.000 | 0.414 | 0.938 |
| rollout_token_drop@canonical | 64 | nearest | 0.062 | 0.336 | 0.938 |
| random_token_drop | 64 | stability | 0.021 | 0.383 | 0.875 |
| random_token_drop@canonical | 64 | nearest | 0.021 | 0.617 | 0.969 |
| native_jitter | 1.0 | stability | 0.021 | 0.762 | 0.969 |
| native_jitter@canonical | 2.0 | nearest | 0.198 | 0.156 | 0.969 |

<!-- results:end -->

## Verdicts

None. The probes were not run on GPU, so every hypothesis above is untested.
