# Trigger size series, pre-registered predictions

Written 2026-09-30 at 01:05 UTC, before any of the 12 models was swept. Only their `args.json` (attack success and clean accuracy from training) had been read.

The evidence surplus account (`docs/evidence-surplus-theory.md`, item O5) says a patch trigger survives token masking because the class token reads it through several carriers, an OR over its tokens and the late blocks. A larger patch covers more tokens, so under PSBD-TM its triggered answers should survive higher masking rates and separate better. Residual dropout after the adds perturbs every token's stream alike and has no such OR, so it should not follow the token count the same way.

## Models and the series variable

The 12 checkpoints are `vit_{cifar100,gtsrb}_badnet_a2o_0_05_trig_p{2,3,5,8,12,16}`, BadNets at 5% with a checkerboard of p by p native pixels in the bottom-right corner. After the resize from 32 to 224 pixels a native pixel spans 7 input pixels and a token 16, so the patch covers ceil(7p/16) tokens per side, 1, 4, 9, 16, 36 and 49 tokens for p = 2, 3, 5, 8, 12 and 16. Only models that are successful backdoors at the 2-point clean-accuracy bar (`scripts.coverage_ledger.classify_cell`, ASR bar and benign reference of `configs/psbd_basis.json`) enter a verdict. The rest are listed. From `args.json`, `vit_gtsrb_badnet_a2o_0_05_trig_p12` reads attack success 0.0 and clean accuracy 0.055, a diverged run, so it is expected to drop out.

## Readouts

Both placements are swept with k = 3 on their standard ladders, PSBD-TM (`before_attention_norm_token_mask`) at 0.05 to 0.9 and PSBD-RD (`post_residual`) at 0.005 to 0.9, the ladders of the panel caches, then `cli.analyze`.

- Triggered p*: `defenses.scores.critical_rate` with flip fraction 0 on the triggered split, the smallest ladder rate at which any pass moves the label. A triggered input that never flips on the ladder counts as 1.0. The readout is the mean over triggered inputs.
- TPR at 1% FPR: fractional PSU at the adaptive rate (`select_rate_adaptively` at 0.8), threshold at the 0.01 quantile of clean validation, read from `psbd_metrics.json` `detection_psu_ratio`.

## Predictions

Within each dataset, over the successful models, ρ is the Spearman correlation with the token count.

1. **TM-p.** Under PSBD-TM, ρ between the token count and mean triggered p* is at least 0.5 on each dataset.
2. **TM-tpr.** Under PSBD-TM, ρ between the token count and TPR at 1% FPR is at least 0.5 on each dataset.
3. **RD-differs.** Under PSBD-RD the same 2 correlations are lower than under PSBD-TM, on each dataset and for both readouts.

A prediction is held when every part of it holds, failed when a part fails and inconclusive when a correlation is undefined because a readout is constant across the series (for example TPR 1.0 on every GTSRB model) or fewer than 4 models succeed on a dataset. TPR at 5% and 10% FPR, AUROC and the share of triggered inputs that never flip are reported beside the verdicts and judged by nothing. O5's own prediction in the theory note (PSBD-TM AUROC nondecreasing in the token count, the 1-token model lowest) is reported as a secondary reading.
