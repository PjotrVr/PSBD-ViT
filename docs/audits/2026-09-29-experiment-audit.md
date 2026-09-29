# Experiment audit, 2026-09-29

Scope is every experiment directory that feeds the notebooks plus the work in progress on the login node GPU tonight (17:00 to 07:00). Every check ran on CPU from cached JSON and tensors. Nothing under `paper/` was edited and nothing was committed. Severity classes are wrong result, missing control, unsupported claim and hazard, in that order. Directories in progress (`why_psbd_works`, `prediction_shift_phenomenon`, `backdoor_manifestation`) were still changing at 17:35, so their sections describe the files at that time.

## Findings that invalidate or endanger tonight's runs

1 finding is a wrong result that reaches the GPU jobs tonight. The live SIG trigger differs from the one the July SIG checkpoints were trained on, so any perturbed pass on such a checkpoint scores a different trigger than its cached baseline. Everything else in this document is a missing control or an unsupported claim.

### SIG amplitude drift, wrong result

`attacks/sig.py:18` sets `SigConfig.amplitude` to 0.157 (commit 0150611, 2026-09-09). The July checkpoints were trained at 0.1 and no `attack_config_overrides` entry is recorded in their `args.json`, so every loader that rebuilds the eval set from `args.json` stamps 0.157. CPU reproduction on `vit_cifar10_sig_0_1`, first 300 rows of the backdoor split, against `baseline_backdoor.pt`:

| Amplitude in the loader | Triggered hit rate | Prediction agreement with the cached baseline |
| --- | --- | --- |
| 0.157 (live default) | 0.997 | 0.923 |
| 0.1 (monkeypatched) | 0.920 | 1.000 |

The cached ASR 0.901 and the ledger class `clears` describe the 0.1 trigger. An experiment that reads a cached baseline and adds fresh perturbed passes therefore subtracts a 0.1 baseline from 0.157 perturbed probabilities. An experiment that reads triggered ASR from a fresh forward pass reports 1.000 where the cache says 0.922.

Affected right now.

- `scratch/k20_todo_tm.txt` line 17 lists `vit_cifar10_sig_0_1` at rate 0.8. `cli.sweep` reuses the July baseline when the row count matches, so the `_k20` cache for this cell will mix the 2 triggers. This is the 1 SIG cell of the 4 paper datasets the k20 run touches. It should be skipped or fixed before its turn comes.
- `experiments/why_token_masking_works/measure.py`, the SIG entry of the visible subsets run. The record reads triggered ASR 1.000 against the cached 0.922.
- `experiments/failure_modes/measure.py`, the SIG margin and patching rows.
- `experiments/backdoor_manifestation/measure.py:85` to `88` and `experiments/why_psbd_works/measure.py:82` to `108`, the `vit_cifar10_sig_0_1`, `swin_cifar10_sig_0_1`, `swin_cifar100_sig_0_1` and `swin_cifar10_sig_0_05` entries. The Swin cells were trained after the drift and are unaffected. The Swin trigger date must be checked from `trained_started_at` before that is assumed.
- `experiments/prediction_shift_phenomenon/measure.py`, the same SIG entries.

Fix. Write `attack_config_overrides {"amplitude": 0.1}` into `args.json` for every SIG checkpoint trained before 2026-09-09 and let `apply_config_overrides` apply it. Then add a CPU regression test that rebuilds the loaders for the first 64 rows of every clearing cell and asserts prediction agreement with `baseline_backdoor.pt` of at least 0.98. That test fails on this cell today. `scripts/verify_splits.py` checks row counts only and should be extended the same way. The same test would catch any later default drift, and the `adaptive_blend` default change in 33696d3 deserves the same check.

## Per-experiment findings

### `experiments/why_token_masking_works/`

1. Wrong result, SIG entry (see above).
2. Missing control. The visible-subsets experiment draws its random position permutations with `DRAW_SEED + draw` (`measure.py:98`, `measure.py:345`, `measure.py:535`), the same 3 draws for every model. Across models the draws are not independent, so the 39 models look like 39 replications of 3 draws. Fix by seeding per model from a stable hash of the folder name and by raising the draws to 20. A test would assert that 2 models get different permutations.
3. Unsupported claim. The README and the "blocks 9 to 12" statement are stale (65 models, 8 source-mapped included) and the blocks 5 to 8 result also carries the effect. The saved JSONs in `results/_experiments/why_token_masking_works/` hold 39 models, and no per-model spread is reported. Baseline ASR reproduces the cache for 38 of 39 models, the exception being the SIG cell.
4. Swin extension. Not present yet. When it lands the same reproduction gate against `baseline_backdoor.pt` applies and the target layer indices differ (Swin has 4 stages, so "blocks 9 to 12" has no meaning).

### `experiments/residual_stream_mechanism/`

1. Unsupported claim. The README quotes +0.177 at the 0.8 rule while the paper quotes -0.183 at the 0.6 rule. The 2 numbers are different quantities and the README should say so. Its "67-cell panel" is stale.
2. Unsupported claim. Rerunning `layernorm_absorption.py` on `vit_gtsrb_badnet_a2o_0_1` on CPU reproduced the Gaussian survival (0.807 and 0.882) and moved token_mask to 1.007 and 0.980 and channel_mask to 1.039 and 1.023, against README values of 0.998, 0.970, 0.966 and 0.998. The largest gap is 0.07 for channel_mask. The verdict survives but the table is not reproducible to the printed digits. Seed and bf16 state should be recorded.
3. Unsupported claim. The link from LayerNorm absorption to detection strength rests on 2 checkpoints and is correlational. Say so in the README or add a third checkpoint.

### `experiments/psu_vs_confidence/`

1. Unsupported claim. The README table cannot be reproduced from `results/_experiments/psu_vs_confidence/psu_vs_confidence.json`, which holds placement `pre_residual` and 295 rows. The canonical placement is `before_attention_norm_token_mask`. Confidence alone beats PSU on 8 of 44 plain models on CIFAR-100 and Tiny, and the oracle rate flatters PSU. Rerun at the canonical placement and adaptive rate with a reproduction gate against `psbd_metrics.json`, and report the win count.

### `experiments/failure_modes/`

1. Wrong result for the paper's panel. The experiment uses `vit_tiny_tact_0_01` and `vit_cifar100_tact_0_01`, which the ledger marks `source_mapped`, so the "3 failing models" framing describes models the paper excludes. The SIG member carries the amplitude drift.
2. Unsupported claim. TaCT margin and patching results rest on n=8 models and 22 pairs, which cannot support a per-model statement.

### `experiments/shift_in_latent_space/`

1. Wrong result. The README numbers do not match the saved `shift_latent.json` files (5 models).
2. Missing control. The benign control matches the backdoored models (0.295 landed on target and 0.478 cosine), so "target drift is real" is not supported. The placement and rate are also non-canonical (rate 0.5, blocks 5 to 8) and the perturbed passes used to decide "landed" may not share masks with the ones used for the direction. Add a benign-checkpoint control with the same trigger and state the mask sharing.

### `experiments/backdoor_neurons/`

1. Missing control. The random rank-1 control in `ablate.py:261` is `torch.randn(width)`, isotropic and not energy-matched to the difference direction. Residual streams are anisotropic with a few high-magnitude dimensions, so the control removes almost nothing. Add a direction matched in variance on the clean set (a clean-PCA direction) and the direction of another attack.
2. Missing control. The 0.03 to 0.08 clean-accuracy cost equals what erasing the target-class direction would cost. No target-class recall check is reported. Add per-class recall before and after the ablation.

### `experiments/why_psbd_works/` (in progress)

1. Missing control. The redundancy measurement uses 3 shared draws per fraction (`measure.py`, redundancy section). Use 20 or more draws and add a control that forces the trigger tokens to stay kept, which is what separates "redundancy" from "the trigger is rarely hit".
2. Unsupported claim unless labeled. `operator_readout` at `measure.py:354` to `386` computes `auroc_psu_ratio` with `auroc_low_is_positive(triggered_psu, clean_psu)` over hit-only triggered scores against all clean scores. The pipeline pairs clean to backdoor with `pair_clean_to_backdoor`, so this AUROC is a different quantity from the cached `psbd_metrics.json` number. `sanity.py:222` to `252` gates a separate `detection_report` path within 0.01 of the cache, which is good, but the readout that reaches the summary is not that path. Either call `detection_report` in the readout or name the field `auroc_hit_only_unpaired` in every table.
3. SIG entries (see the drift finding).
4. Precision hazard. The margin and finite difference readouts (`flatness`) use differences of log probabilities. State whether they run in bf16, since finite differences at small step sizes are dominated by bf16 rounding. The sanity script should print the float32 versus bf16 gap for 1 model.

### `experiments/prediction_shift_phenomenon/` (in progress)

1. Good. `sanity.py:236` to `266` compares the fresh shift ratio with the cached one and the fresh baseline labels with the cached ones. That gate would have caught the SIG drift if a SIG model were among the sanity runs. Only the GTSRB BadNet models are, so add `vit_cifar10_sig_0_1` to the gate.
2. Unsupported claim. `target_of(row)` (`measure.py:1082`) derives the target label from the folder suffix. That fails for `_tl1` cells and for the ResNet reference run. Read `target_label` from `args.json`.
3. Hazard. `feature_conditions` (`measure.py:533`) returns `None` rates silently when a cache is missing, and the summary medians (`measure.py:1221`) then average over fewer models without saying so. Print n per median.

### `experiments/backdoor_manifestation/` (in progress)

1. Missing control. `random_coordinates_{k}` (`measure.py:994`) exists at k=300 only, has no outlier or high-magnitude-dimension control, and `steer_clean` has no equal-norm random-direction or benign-direction control. Without them "the manifestation lives in these neurons" cannot be separated from "any high-energy direction moves the answer".
2. Unsupported claim. The header (`measure.py:17`) says the neuron choice is fitted on the first half and every separability is scored on the second, but `layer_row` (`measure.py:663`) and `layer_table` (`measure.py:624`) take `fit_rows` and it must be verified that the AUROC-like readouts use only the held-out half.
3. Unsupported claim. 3 sub-bar models (below ASR 0.85) in the panel are not tagged in the output.
4. SIG entries (see the drift finding).

### `scripts/paper/fig_forward_passes.py` and the k up to 20 caches

1. Good. Slicing the first k of 20 passes is a valid k-pass estimator because the masks are iid, and the rate is fixed at the k=3 adaptive choice and stated in `docs/runs/2026-09-29-k20-login-gpu.md`. The k=20 curve reproduced on CPU over 19 cells as 0.9653 (k=1), 0.9747 (k=3), 0.9759 (k=5), 0.9766 (k=10) and 0.9769 (k=20).
2. Wrong result for 1 cell, `vit_cifar10_sig_0_1` (see the drift finding).
3. Unsupported claim. `fig_forward_passes.py:449` still says "(65 clearing cells)" and the pilot population has no GTSRB cell and only 1 CIFAR-10 cell, so the curve is a CIFAR-100 and Tiny curve. Say so in the caption. 5 source-mapped TaCT cells were swept, which wastes GPU and pollutes any mean over the folder list.
4. Hazard. The run doc lists 59 clearing models, of which 2 have no `psbd_metrics.json` and are skipped. The paper panel is 57. Confirm the figure filters by the paper's 57 and not by the cache folders present.

### `scripts/all_numbers.py`, `tests/test_all_numbers.py`

1. Wrong result for Swin extras. `swin_paper_extras` hard-codes `"source_mapped": False` (`all_numbers.py:213`), so a Swin TaCT cell outside the panel rule that is source-mapped reads as `clears` and can be `successful_2pt`. Compute `source_class_accuracy` the way the ledger does, or set `source_mapped` from the ledger's function.
2. Missing test. The tests cover `psbd_reading` only. `retarget_declaration`, `swin_paper_extras`, `benign_models` and the row assembly have none. A test that builds the table for 2 synthetic cells and asserts 1 row per (model, defense) and no duplicated `folder_name` between `swin_models` and the extras would catch double counting.
3. Style hazard. The module docstring says "the 2 false-positive budgets" while the constant lists 3 quantiles.

### Coverage ledger `successful_2pt` and `successful_5pt`, `configs/psbd_basis.json`

1. Good. `success_verdicts` reads the final `asr_class`, so diverged and source-mapped cells never succeed, and the parameterized test covers the boundary values (-0.02 and -0.05 succeed at their own bar). `tests/test_canon.py` holds the new bar to the declaration.
2. Unsupported claim risk. The success verdict compares `clean_accuracy_drop` against a single benign reference per dataset. The benign reference of Swin is retargeted by string replacement of `vit_` in `retarget_declaration`, which silently keeps a wrong name if a reference folder does not start with `vit_`. A test asserting that every retargeted reference exists on disk would catch it.

## Hazards independent of any experiment

- `checkpoints/*/metrics.json` is stale. `args.json` carries the correct `asr` (from the psbd baseline cache). The Swin CIFAR-10 TaCT `metrics.json` reads ASR 0.13 against 0.9987 in `args.json`. Read `args.json`.
- `swin_cifar10_sig_0_1` has clean accuracy 0.869, so it may fail the 2 point bar when the ledger runs on Swin.
- The Swin cell count is 83 in 1 place and 80 in another. State which rule each uses.

## Tests that would have caught the findings

| Finding | Test |
| --- | --- |
| SIG amplitude drift | Rebuild loaders for the first 64 rows of every clearing cell and assert agreement of at least 0.98 with `baseline_backdoor.pt` |
| Hit-only AUROC in `why_psbd_works` | Assert the readout AUROC equals the cached `psbd_metrics.json` value at the adaptive rate, or rename the field |
| `target_of` from the folder suffix | Parameterized test over `_tl1` and `_src{k}` folders against `args.json` |
| Shared permutations | Assert 2 models yield different position permutations |
| Swin extras `source_mapped` | Synthetic source-class-accuracy cell through `swin_paper_extras` |
| Retargeted benign references | Assert every retargeted reference folder exists |
