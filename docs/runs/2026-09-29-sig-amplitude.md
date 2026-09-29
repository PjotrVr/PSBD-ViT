# SIG trigger amplitude drift and its repair

## What went wrong

Commit 0150611 (2026-09-09 02:44:51 UTC) raised `attacks/sig.py` `SigConfig.amplitude` from 0.1 to 0.157, the 40/255 of the SIG paper. The 165 SIG checkpoints trained before that commit learned the 0.1 sinusoid, and their `args.json` recorded no override, so every triggered split rebuilt after the commit applied 0.157 to a model that had learned 0.1. The stronger trigger reads as a stronger backdoor: on `vit_cifar10_sig_0_1` the hit rate is 0.920 at 0.1 and 0.997 at 0.157. The experiment audit (`docs/audits/2026-09-29-experiment-audit.md`) found it when a cached baseline agreed with a fresh rebuild on only 0.923 of rows.

A cache is inconsistent when its triggered inputs were built at a different amplitude from the model's training, or when a sweep's perturbed passes and the baseline it reuses differ in amplitude. Commit time cannot date a cache, because jobs that started before the commit kept writing 0.1 for hours and others wrote 0.157 from minutes after it. Every cache was dated by measurement instead: on CPU, 300 rows of fresh predictions at each amplitude against the cached baseline and against each placement's per-pass argmax at its lowest rate.

## What was affected

| model | trained at | baseline built at | placements at 0.157 | detector records |
|---|---|---|---|---|
| `vit_cifar10_sig_0_1` | 0.1 | 0.1 | 13 of 67 | all 11 at 0.157 |
| `swin_cifar10_sig_0_1` | 0.1 | 0.1 | 26 of 35, PSBD-TM and the twin among them | none |
| `swin_cifar10_sig_0_05` | 0.1 | 0.157 | all 3 | none |

PSBD-TM (0.418) and PSBD-RD (0.919) on `vit_cifar10_sig_0_1` were both built at 0.1 and were consistent. That model has since left the panel under the 2-point clean-accuracy bar (-10.6 points), so no headline ViT number depends on it. The Swin headline did read the mixed PSBD-TM cache of `swin_cifar10_sig_0_1`. `swin_cifar10_sig_0_05` clears the ASR bar only at the wrong amplitude: at 0.1 its ASR is 0.743 on 300 rows and 0.721 in its July full-test `metrics.json`, so it leaves the Swin panel. The other Swin SIG cells whose baselines were built at 0.157 stay below the bar at either amplitude.

## The repair

1. The 165 SIG checkpoints trained before 0150611 now carry `"attack_config_overrides": {"amplitude": 0.1}` in `checkpoints/<folder>/args.json`, which `data/splits.py`, `evaluation/metrics.py` and `scripts/coverage_ledger.py` already apply. `checkpoints/` is not tracked by git, so this record is the provenance. The originals are in `scratch/sig_args_backup/<folder>/args.json`, and the replayable script is `scratch/sig_amplitude/record_overrides.py`.
2. `scripts/verify_trigger_consistency.py` rebuilds every cell's triggered split and asserts at least 0.98 agreement of the first 64 rows with the cached baseline. `tests/test_trigger_consistency.py` pins the override path and the sidecars. After step 1, 145 of 148 clearing cells pass. `swin_cifar10_sig_0_05` fails at 0.766, and 2 cells have no baseline yet.
3. `scratch/sig_rerun.sh` reruns every inconsistent sweep on the login GPU from 19:08 on 2026-09-29, in the order the tables need them: the 26 `swin_cifar10_sig_0_1` placements with PSBD-TM and the twin first, then `swin_cifar10_sig_0_05` with a rebuilt baseline, then the 13 `vit_cifar10_sig_0_1` placements and its 11 detectors. Stale caches are moved to `results/<folder>/_stale_sig_amp0157/`, never deleted. Progress is in `scratch/sig_rerun_logs/summary.txt`.

## Still to do

- Correct the ASR of `swin_cifar10_sig_0_05` in its `args.json` from the rebuilt baseline. `cli.backfill` is not used for this because it walks every checkpoint and replays the TaCT `_src{k}` selections with source class 1 only (`cli/backfill.py:106`, `analysis/latent.py:169` build from defaults).
- Rerun the SIG entry of `experiments/why_token_masking_works/measure.py` and the residual-stream mechanism records of `vit_cifar10_sig_0_1`, which were built at 0.157.
- Rebuild `paper/` and every downstream document once the reruns land.
