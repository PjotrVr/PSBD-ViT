# Why PSBD-TM wins

## Question

PSBD-TM (`token_mask` at `before_attention_norm`, the recommended placement) detects backdoored inputs better than PSBD-RD (dropout at `post_residual`, the PSBD paper's ConvNet placement carried to transformers) on both ViT-B/16 and Swin-S. This experiment asks why, and answers it with measurements. It has 3 parts.

1. The causal story on ViT: which tokens carry a patch trigger, where the class token reads it, what PSBD-TM's random masks and PSBD-RD's dropout do to it (parts A to D, `measure.py`), rerun on 2026-09-29 with the audit's fixes.
2. The same 4 parts on Swin-S, which has windowed attention, patch merging and a mean-pooled head instead of a class token (`swin.py`).
3. Which part of the placement does the work, the site, the operator or how hard it perturbs: paired contrasts on the cached sweeps, a causal grid of 6 sites by 6 operators at matched clean disturbance, the share of the trigger's own signal each probe leaves per block, and 3 mechanics the theory memo derived (`sites.py`, `mechanics.py`).

`notebooks/why-psbd-tm.ipynb` walks through all of it with a figure per step. Every hypothesis below is stated before its measurement and judged by it, and the explanations that fail are reported with the same detail as the ones that hold.

## Vocabulary

- **PSBD** (prediction shift backdoor detection, Li et al., arXiv 2406.05826) flags an input as poisoned when its prediction survives a random perturbation of the network's activations at inference. None of the models was trained with dropout.
- **Position** is where in a block a probe acts and **operator** is what it does. A **placement** is the pair, named `<position>_<operator>`, with a bare position meaning dropout. The sites compared here, with the letters the notebook uses:
  - A, `before_attention_norm`, the attention branch's input before its LayerNorm (`ln_1` on ViT, `norm1` on Swin). The residual stream is not touched.
  - B, `before_attention`, the same input after the LayerNorm.
  - C, `before_attention_residual`, the attention branch's output before the add.
  - D, `post_residual`, the stream after each of the 2 adds of a block. Dropout here is PSBD-RD. `after_attention_residual` is the first of the 2 alone.
  - E, `before_mlp_norm`, the MLP branch's input before its LayerNorm, and `before_mlp` after it.
  - `after_embedding`, the token embeddings once, before the first block.
- **Operators** (`defenses/operators.py`): `token_mask` zeroes whole tokens with probability p, rescales survivors by 1/(1-p) and never masks ViT's class token. `dropout` is `nn.Dropout`. `channel_mask` zeroes whole channels, the same for every token of an image. `gaussian` adds noise of standard deviation p times the per-image spread over all tokens and channels. `rademacher` adds plus or minus that amount. `token_substitute` replaces a token with another token of the same image.
- **Fractional PSU**, the score every AUROC here reads, for an input $x$ whose unperturbed argmax is $c$:

$$\phi(x) = 1 - \frac{1}{k}\sum_{i=1}^{k}\frac{P_c(x;p,\theta_i')}{P_c(x;\theta)}$$

| symbol | meaning |
|---|---|
| $P_c(x;\theta)$ | softmax probability of class $c$ on the unperturbed model |
| $P_c(x;p,\theta_i')$ | the same probability on perturbed pass $i$ at rate $p$ |
| $k$ | number of perturbed passes, 3 in every cached sweep |

  Low $\phi$ is read as poisoned. `defenses.scores.psu_ratio_from_cache` computes it and `defenses.decision.detection_report` gives the AUROC, with the clean images restricted to the twins of the triggered ones. The headline AUROC is the `detection_psu_ratio` block at the chosen rate and quantile 0.25 (`cli.compare_detectors.psbd_values`). The block's own `adaptive` entry holds the absolute-PSU AUROC and is not the headline.
- **Shift ratio**: the share of (pass, image) pairs whose perturbed argmax differs from the unperturbed one (`defenses.scores.shift_ratio`).
- **Adaptive rate**: the smallest swept rate whose shift ratio on the 2000 held-out clean validation images reaches 0.8 (`select_rate_adaptively`). **Matched rate**: the swept rate whose clean validation shift ratio is closest to 0.6 (`select_rate_at_matched_shift`). On the coarse cached ladders the matched rate lands between 0.45 and 0.76 of clean shift, which is why the causal grid calibrates its own rate.
- **Survival** or "kept": the share of predictions a probe leaves unchanged. Triggered survival is over the triggered images the unperturbed model sends to the target, clean survival over their clean twins (`measure.readout`). "Clean on target" is the share of clean images a probe sends to the target, the floor a model collapsing onto 1 class produces.
- **Patch triggers**: BadNets, TaCT, Label-Consistent. **Global triggers**: Blend, LF, BPP, WaNet, SIG, Adaptive-Blend.
- **Trigger tokens** of a block: the tokens whose pixels the trigger changes after the model's own bilinear resize to 224 (`tokens.trigger_pixel_map`, `tokens.touched_tokens`). ViT keeps 1 grid of 14 by 14 tokens in all 12 blocks. Swin-S has 4 stages of 2, 2, 18 and 2 blocks on grids of 56, 28, 14 and 7 tokens, so its positions are resolved per block (`tokens.block_trigger_positions`). A GTSRB BadNets patch covers 4 ViT tokens and 48, 15, 4 and 1 Swin tokens.

## Panel

Every result spans the successful backdoors only (user decision, 2026-09-29): the attack clears the 0.85 ASR bar, the model is neither diverged nor source-mapped (a TaCT model that sends its clean source class to the target with no trigger), and its clean accuracy is within 2 points of the benign model of its dataset. A failed attack is left out even when that removes the attack.

- ViT: the coverage ledger's `successful_2pt` verdict, 54 models (`tokens.successful_vit_folders`). Against the 57 clearing models this drops `vit_cifar10_sig_0_1`, `vit_cifar10_wanet_0_05` and `vit_gtsrb_wanet_0_1`.
- Swin: the same rule through `scripts.paper._common.swin_coverage`, which applies the ViT declaration (datasets, poison rates 1%, 5% and 10%, canonical variants) with the Swin benign models as references (`swin.panel_cells`), 65 models, 63 of them carrying both PSBD-TM and PSBD-RD at the adaptive rule. No Swin Label-Consistent or SIG model is on it.
- SIG is also kept out of every GPU run: `attacks/sig.py` builds amplitude 0.157 while the July ViT SIG checkpoint learned 0.1 (`docs/audits/2026-09-29-experiment-audit.md`). Both SIG models fail the success bar in any case.
- Patch models for the causal parts: 15 on ViT (12 BadNets, 3 TaCT whose clean source images are classified correctly) and 14 on Swin (12 BadNets, 2 TaCT). Swin part A also ran on 7 models that left the panel when it switched to `swin_coverage` on 2026-09-30 (the 0.5% BadNets models and 4 Label-Consistent models). Their records stay in `swin/` and out of every summary.
- Part D takes 1 model per attack and dataset, the 5% model where it succeeded and otherwise 10% then 1%: 14 global ViT models and 20 global Swin models (Adaptive-Blend included on Swin).

Every model's pairs are the first 256 triggered images of its PSBD analysis split (`build_psbd_loaders_from_checkpoint`, seed `PSBD_SPLIT_SEED`) and the clean image behind each. TaCT's eval set is its source class only.

## Files and commands

| file | what it does | records |
|---|---|---|
| `measure.py` | parts A to D on ViT | `vit/` (rerun), the first run's 39 JSONs and `summary.json` at the top level |
| `swin.py` | parts A to D on Swin, plus the routes test | `swin/` |
| `sites.py` | cache contrasts (`--cache-only`), the causal site and operator grid, per-block signal retention, the ViT single-block depth profile, embedding masks | `sites/` |
| `mechanics.py` | the masked-token sink, the Gaussian coupling, embedding masks with and without the rescale (ViT) | `mechanics/` |
| `gates.py` | the 6 sanity gates | `gates/` |
| `tokens.py` | per-stage trigger positions, position-restricted probes, per-block plugging, readers, the signal retention statistic, per-model seeds, the reproduction check, the panel | none |

```bash
source .venv/bin/activate
# CPU, the cached sweeps and the copied evidence
PYTHONPATH=. python experiments/why_token_masking_works/sites.py --cache-only
# GPU, 1 model per call so each takes and releases the lock
PYTHONPATH=. python experiments/why_token_masking_works/gates.py --folders vit_gtsrb_badnet_a2o_0_05 swin_gtsrb_badnet_a2o_0_05
PYTHONPATH=. python experiments/why_token_masking_works/swin.py --parts ABCD --folders swin_gtsrb_badnet_a2o_0_05
PYTHONPATH=. python experiments/why_token_masking_works/sites.py --folders vit_gtsrb_badnet_a2o_0_05
PYTHONPATH=. python experiments/why_token_masking_works/mechanics.py --folders vit_gtsrb_badnet_a2o_0_05
PYTHONPATH=. python experiments/why_token_masking_works/measure.py --subdirectory vit --folders vit_gtsrb_badnet_a2o_0_05
# summaries from the saved records
PYTHONPATH=. python experiments/why_token_masking_works/swin.py --summarize-only
PYTHONPATH=. python experiments/why_token_masking_works/sites.py --summarize-only
python -m pytest tests/test_why_token_masking_works.py tests/test_why_token_masking_works_tokens.py
```

Every script is resumable. A model's JSON is written after every part or placement and a rerun skips what is already there. `--list-folders` prints the panel a script would run. The runs on 2026-09-29 used the login node A100, 1 lock per model on `scratch/gpu.lock` or `scratch/gpu2.lock`, a memory fraction of 0.35 (14 GB), bfloat16 forward passes and batches of 128 to 256.

## Sanity checks

6 gates ran on `vit_gtsrb_badnet_a2o_0_05` and `swin_gtsrb_badnet_a2o_0_05` before any full run (`gates.py`). The GTSRB BadNets 5% models were chosen because GTSRB is the project's default test bed and 5% the middle rate.

1. Clean accuracy and ASR on the full analysis split (10630 clean and 10578 triggered images) from this code equal `args.json` to 4 decimals (ViT 0.9862 and 1.0000, Swin 0.9653 and 1.0000) and `metrics.json` within 0.0005. The unperturbed argmax agrees with the cached baseline on every image of the validation, clean and triggered splits. `metrics.json` is stale for some models (the audit's hazard), so `args.json` is the reference.
2. PSBD-TM and PSBD-RD at the adaptive rate, recomputed on the full 3 splits with this code's plugging at the cached k (3), mask seed (0) and batch size (64), and scored exactly as `cli.analyze` scores them. Per-image fractional PSU of PSBD-RD is identical to the cache on every image of both models. PSBD-TM differs by at most 0.004 on a ViT triggered image and not at all on Swin, bfloat16 matrix-product nondeterminism. The AUROCs match the cached headline values: ViT PSBD-TM 0.99996 against 0.99996, ViT PSBD-RD 0.88944 against 0.88944, Swin PSBD-TM 1.0 against 1.0, Swin PSBD-RD 0.58844 against 0.58844. That fixes the hook site, operator, rate, k, splits and pairing as the pipeline's own.
3. Trigger tokens. On 64 pairs per model, no pixel that the loader's triggered image changes lies outside the predicted tokens at any grid (7, 14, 28 and 56). At Swin's 56 grid a predicted token touched only by the resize's bilinear spill is unchanged on some images (0.53 of 48 on average), which only means some masked tokens hold clean content. On the unperturbed models, at the first block of every stage, the tokens with the largest triggered minus clean stream difference are exactly the predicted ones, carrying all of it at ViT block 1 and Swin block 1, 0.89 at Swin block 3 and 0.78 at Swin block 5. At Swin block 23 the 1 predicted token carries 0.04, because windowed attention has spread the trigger's effect over the whole 7 by 7 map by then. The notebook draws the tokens over the image at every grid.
4. After every gate no hook and no forward override is left on the model, no module is in training mode, and every model-owned dropout has rate 0 (37 on ViT, 48 on Swin). `swin.assert_pristine` repeats this after every part of every model in every script. Every control matches its treatment in count (random positions per block equal the trigger count, asserted in `random_other_positions`), rate (the same variable) and k.
5. Diverged, source-mapped and unsuccessful models are out, as the Panel section states. The ViT list comes from `successful_2pt`, the Swin list from `swin_coverage` through `swin.panel_folders`.
6. Gate 2 in float32. The AUROCs move by at most 0.0003 (ViT PSBD-RD 0.88916 against 0.88944). Per image, ViT PSBD-RD's PSU changes more (correlation 0.67 on validation) because the dropout masks are drawn on tensors of another dtype, so the random stream differs, and the ranking of images is unchanged.

Every full run repeats gate 1 on its own pairs: the unperturbed predictions must agree with the cached baseline on at least 98% of the triggered and the clean images (`tokens.cached_baseline_agreement`), or the run stops.

## Changes after the audit of 2026-09-29

- Random positions and visible subsets were drawn with the same 3 seeds for every model, so N models looked like N replications of 3 draws. Every draw is now seeded per model from a stable hash of the folder name (`tokens.model_seed`), part D takes 20 subsets per model and fraction, and the stochastic parts B and C seed their masks per model too, since models at the same rate drew identical masks (the theory memo's first idealization). `tests/test_why_token_masking_works_tokens.py` asserts that 2 models get different permutations.
- The ViT rerun writes to `vit/`, beside the first run's records, which stay as they were.
- The first run's statements about part B (the $p^{4m}$ breakage account) and part C (trigger-only dropout breaks as much as dropout everywhere) are corrected below, following `docs/why-psbd-works-theory.md`.

## Run status

All planned GPU runs finished on the night of 2026-09-29 to 2026-09-30, from 17:20 to 03:47, and the marker `scratch/gpu_done_tm` was written. Nothing is left to run for this experiment.

| run | models | records |
|---|---|---|
| sanity gates | 2 | `gates/` |
| ViT parts A to D, rerun with per-model seeds | 29 (15 patch, 14 part D) | `vit/`, `vit/summary.json` |
| Swin parts A to D | 34 on the panel (14 patch, 20 part D) plus 7 part A records outside it | `swin/`, `swin/summary.json` |
| site and operator grid, depth profile, embedding masks | 15 ViT and 14 Swin patch models (plus 1 Swin model outside the panel) | `sites/`, `sites/summary.json` |
| mechanics (sink, Gaussian coupling, embedding rescale) | 30 ViT (15 patch and 15 global) | `mechanics/` |
| X2 of `docs/simple-experiments-plan.md` | `vit_cifar10_wanet_0_1_seed_1` and `_seed_2`, PSBD-TM and PSBD-RD swept with `cli.sweep` and scored with `cli.analyze` | `results/<folder>/psbd_metrics.json` |

The Swin panel switched from `tab_swin.swin_cells` to `swin_coverage` at 00:10 on 2026-09-30, while the runs were going. 7 Swin models measured under the old rule (0.5% BadNets, Label-Consistent) are kept on disk and filtered out of every summary and figure. The site-grid record of `vit_gtsrb_badnet_a2o_0_05` comes from the smoke run of the same code, resumed by the full run.

## Results

The notebook `notebooks/why-psbd-tm.ipynb` has 1 figure per result below and states for each how it was computed. Survival is "kept", over the models named in the Panel section. Numbers are means over models unless stated.

### The lead to explain

On the 54 ViT models PSBD-TM reads 0.963 against 0.885 for PSBD-RD at the adaptive rule, a paired gain of +0.078 [+0.026, +0.135]. On the 63 Swin models carrying both it reads 0.973 against 0.877, +0.096 [+0.052, +0.143]. At the adaptive rate both change about 0.87 of clean predictions. On ViT patch triggers PSBD-TM changes 0.195 of triggered predictions and PSBD-RD 0.676, on Swin 0.009 and 0.595.

### A, where a patch trigger is read

Hypothesis: the trigger's evidence sits in its own tokens and a token masked at the attention input keeps its stream entry, so hiding the trigger tokens from attention everywhere removes the backdoor and hiding them early does nothing. Supported on both architectures.

- ViT, 12 BadNets models: trigger tokens hidden in all 12 blocks keep 0.002, in blocks 1 to 4 1.000, 5 to 8 0.788, 9 to 12 0.200. As many random tokens keep 1.000 and clean predictions keep at least 0.95. The 3 TaCT models keep 0.004 with all 12 hidden and are read in the last blocks (0.339 with only the last block hidden).
- ViT single-block profile, 15 models: visible to attention in 1 block only keeps at most 0.197 (block 12), hidden in 1 block only keeps at least 0.815 (block 12). The read is spread over several late blocks.
- L24: every token but the trigger's hidden in blocks 9 to 12 keeps 0.901 of triggered predictions against 0.292 of clean ones.
- Swin, 12 BadNets models: hidden in all 24 blocks 0.003, in stages 1 and 2 1.000, in the 2 halves of stage 3 0.541 and 0.798, in stage 4 1.000, random tokens 1.000. The 2 TaCT models are read later (0.006 with the last 8 blocks hidden).
- Swin routes: zeroing the trigger's stage-4 token before the mean pool keeps 1.000, and closing the MLP route adds nothing to closing attention (0.003 both). L20's prediction for Swin (at least 0.5 left, since the stream entry reaches the pooled head directly) fails: Swin's windowed attention copies the trigger into neighbouring tokens in stage 3 and those carry it, so Swin reads a patch trigger through attention as ViT does.

### B, PSBD-TM's own masks, corrected

The first run said the triggered prediction breaks only when every trigger token is masked in every late block, probability $p^{4m}$. Following the theory memo, that is right about the event's frequency and wrong about breakage. Survival is graded in $J$, the number of late blocks with every trigger token masked. On the ViT 4-token BadNets models, masks seeded per model: $J$ = 0 to 4 keep 0.988, 0.963, 0.911, 0.736 and 0.175 (57 pairs at $J = 4$), and the broken predictions split 0.23, 0.28, 0.23, 0.20 and 0.06 over $J$. Clean survival does not fall with $J$. On Tiny's 1-token trigger the curve reads 1.000 down to 0.276. On Swin BadNets the curve over the last 4 blocks stays at 0.967 or above, and overall PSBD-TM keeps 0.991 of triggered and 0.104 of clean predictions.

### C, PSBD-RD restricted, corrected

ViT BadNets at the adaptive PSBD-RD rate: all tokens keep 0.393, the trigger positions only 0.387, every other position 0.667, random positions 1.000. The mean agreement of "all" and "trigger only" hides a per-model mean difference of 0.225, and on 8 of 12 models "all" keeps more than the weaker half, as the theory memo found. That dropout on the content positions also weakens the class competing with the target is a hypothesis these records cannot test. Swin BadNets: all 0.441, trigger only 0.134, the rest 1.000, random 1.000, so on Swin the damage is on the trigger's own tokens.

### D, global triggers

20 visible subsets per model and fraction. ViT, excess retention at 30% visible tokens: Blend 0.963, BPP 0.941, LF 0.883, against clean accuracy retention near 0.2. WaNet (2 models) 0.389. Swin (the whole network sees only the visible 7 by 7 cells): Blend 0.981, BPP 0.965, LF 0.479, WaNet 0.009, Adaptive-Blend 0.279. L23's prediction that Swin WaNet stays legible (at least 0.5) fails.

### WaNet, X1 and X2

X1 (the coordinating agent's CPU reading of the cache, not measured here): on `vit_cifar10_wanet_0_1` PSBD-TM changes 0.729 of triggered passes and 0.158 of them land on the true class, against a noise-mode prediction of at least 0.7, so that account is not supported. X2: the 2 unswept seed replicates of the same recipe read PSBD-TM 0.858 and 0.908 (PSBD-RD 0.949 and 0.948) against 0.459 on seed 0, so the failure does not recur and is at least partly an accident of 1 checkpoint.

### Site and operator, cached sweeps

Paired AUROC gaps at the matched rule, a minus b, with 95% bootstrap intervals, ViT (54) then Swin (63 to 65):

| contrast | ViT | Swin |
|---|---|---|
| PSBD-TM minus PSBD-RD | +0.070 [+0.021, +0.124] | +0.135 [+0.089, +0.186] |
| token mask, A minus the stream after the attention add | +0.069 [+0.034, +0.105] | +0.098 [+0.070, +0.127] |
| token mask, A minus C (attention output) | +0.013 [-0.008, +0.036] | +0.151 [+0.103, +0.202] |
| token mask, A minus B (after the norm) | +0.019 [+0.008, +0.031] (36) | not cached |
| token mask, A minus E (MLP input) | +0.114 [+0.080, +0.150] | +0.226 [+0.165, +0.290] |
| token mask, A minus once at the embedding | +0.290 [+0.243, +0.340] (36) | +0.283 [+0.188, +0.390] (26) |
| at A, token mask minus dropout | +0.096 [+0.076, +0.116] | +0.128 [+0.084, +0.176] |
| at A, token mask minus channel mask | +0.083 [+0.060, +0.107] | +0.115 [+0.074, +0.159] |
| at A, token mask minus Gaussian | +0.188 [+0.138, +0.243] | +0.041 [+0.018, +0.068] |
| dropout, A minus the stream (PSBD-RD) | -0.026 [-0.071, +0.022] | +0.009 [-0.038, +0.056] |
| on the stream, token mask minus dropout | -0.007 [-0.048, +0.035] | see `sites/cache.json` |

Verdicts. The stream-untouched hypothesis is supported (token masking loses on the stream and at the embedding). The whole-token hypothesis is supported at A (every other operator loses there). Neither alone suffices: dropout gains nothing from moving to A, and token masking gains nothing over dropout on the stream. The attention-mixing hypothesis is supported (every operator loses at the MLP input on ViT). "It just perturbs more" is refuted: every gap holds at the matched rule, and on the rate ladders no other operator at A reaches token masking at any clean shift (best points on ViT 0.968 against 0.929, 0.934 and 0.888).

### Site and operator, causal grid at matched clean damage

Every cell ran at the rate that changes 60% of the model's clean predictions (6 passes). Triggered kept with the probe on all tokens, ViT patch models (15) then Swin (14): token mask at A 0.912 and 0.997, PSBD-RD 0.638 and 0.779, dropout at A 0.770 and 0.947, channel mask at A 0.760 and 0.964, Gaussian at A 0.520 and 0.996, token mask at C 0.957 and 0.973, at B 0.908 and 0.953, on the stream 0.555 and 0.718, at E 0.666 and 0.672. Random positions keep at least 0.999 in every cell. The restricted variants isolate the difference. On ViT the probe on every token but the trigger's keeps 0.965 for token masking, 0.834 for dropout, 0.816 for channel masking and 0.598 for Gaussian noise: removing the other tokens outright barely disturbs the trigger's route and corrupting them partly does. On Swin the other tokens are harmless under every operator and the probe on the trigger's tokens alone separates them: token masking 0.963, dropout 0.332, channel masking 0.727.

### Why whole tokens

The trigger's signal in the stream at its own tokens over the last 4 blocks: token masking at A keeps 0.88 (ViT) and 0.68 (Swin), PSBD-RD 0.65 and 0.42. The trigger's signal in what attention reads is not the reason: over the last 4 ViT blocks token masking at A leaves 0.55 of it, less than dropout (0.64), channel masking (0.66) or Gaussian noise (0.79), and keeps the most triggered predictions. The isolating result is the restricted grid above. That partly corrupted tokens disturb the class token's read while a masked token is 1 constant it can ignore is a hypothesis.

Masked tokens leave LN 1 as exactly the same vector (largest difference 0.0) and act as 1 attention sink: on clean images the class token puts 0.33 of its late attention on the 10% of masked tokens at rate 0.1 and 0.60 at 0.3, while substituted tokens draw about their share (0.24 at 0.3). At rate 0.5 substitution keeps 0.51 of clean predictions and token masking 0.14. At equal clean damage (L22) substitution keeps as many triggered predictions as token masking on ViT (0.915 and 0.912) and fewer on Swin (0.862 and 0.997).

### The explanations that fail

- "It just perturbs more": refuted, above.
- "LayerNorm makes it stronger": refuted. Token masking before the norm beats token masking after it by only +0.019. The norm shrinks Gaussian noise injected before it (noise after the norm wins by 0.209), mostly as a linear attenuation that the matched rule cancels, so absorption does not explain the Gaussian's deficit.
- "The attention input is special whatever the operator": refuted for dropout (no gain at A over the stream), partly true against the MLP input for every operator.
- "Whole-token masking is stronger dropout": refuted. No dropout rate at A reaches token masking at any clean shift, and the restricted grid shows a difference in kind.
- L2 (PSU estimates curvature): Rademacher and Gaussian noise keep the same share of triggered predictions within 0.01 at each site, as equal covariance predicts under any account, so this does not single out curvature.
- The theory memo's all-or-nothing account of the Gaussian gap: its per-attack ordering holds (BadNets -0.365 up to WaNet +0.036), a correlation across attacks and kept as a hypothesis.

### The Gaussian coupling

`GaussianNoise` scales noise by 1 spread per image. On the ViT BadNets models the triggered over clean spread at the input of LN 1 is 1.23 in blocks 9 to 12 and 1.03 in blocks 5 to 8, and it rises on the ordinary tokens too, not only the trigger's. Scaling noise per token lifts the mean BadNets AUROC on the pairs from 0.589 to 0.846, removing most of the inversion on the CIFAR-100 BadNets models (0.21, 0.27 and 0.21 to 0.81, 0.72 and 0.78). The grid agrees: on ViT the Gaussian breaks triggered predictions through the non-trigger tokens (0.598 kept) and not through the trigger's (0.930).

### Masking once at the embedding (L18)

At `after_embedding` no norm follows the mask, so `TokenMask`'s 1/(1-p) rescale of the survivors enters the stream. With the rescale removed, at rate 0.7, triggered survival on the 4-token BadNets ViT models is 0.038 with no trigger token left and 0.887 with 2 left. With the library rescale the model collapses onto the target (0.185 of clean images at $j = 0$). At the embedding a masked trigger token is gone for the whole network, which is Doan et al.'s regime where instability flags the trigger. At A it is gone for 1 read, and stability flags it.

## Limits

- The causal parts run on 256 pairs per model with 6 to 10 passes. TaCT's source class caps its pairs, and the trigger-conditional TaCT group holds 3 ViT and 2 Swin models.
- Part D covers 1 model per attack and dataset. WaNet has 2 ViT models on the success panel.
- The causal grid scores survival, not AUROC. It says which probe leaves triggered predictions alone at equal clean damage, which is what PSU turns into AUROC, and it does not replace the cached AUROC contrasts.
- The calibration aims at 60% clean change on the paired clean images, with 2 passes per bisection step. The reached value is recorded per cell.
- The Gaussian coupling test and the AUROC on pairs use 256 clean and 256 triggered images, without the validation threshold.
- The all-or-nothing account of the operator gap predicts sign and order, not magnitude.
