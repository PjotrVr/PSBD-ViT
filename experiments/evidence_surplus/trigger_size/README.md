# Trigger size series

The evidence surplus account (`docs/evidence-surplus-theory.md`, item O5) explains PSBD-TM's strength on patch triggers by an OR. The class token reads the trigger through several tokens in several late blocks, and a triggered answer survives token masking while any of those reads is left. A patch that covers more tokens gives more carriers, so the account predicts that its triggered answers survive higher masking rates and separate better under PSBD-TM. Residual dropout after the adds perturbs every token's stream alike and offers no OR over carriers, so PSBD-RD should not follow the token count the same way. The 12 BadNets 5% checkpoints with patch sizes 2 to 16 on CIFAR-100 and GTSRB were trained earlier and never swept, which makes them a dose series in the number of trigger tokens with nothing else changed.

## Design

The series variable is the token count. A patch of p native pixels sits flush in the bottom-right corner, and after the resize from 32 to 224 pixels it covers ceil(7p/16) tokens per side, so p = 2, 3, 5, 8, 12 and 16 give 1, 4, 9, 16, 36 and 49 tokens (`measure.token_count`). Both placements are swept at k = 3 on the ladders every panel cache uses, then `cli.analyze` writes `psbd_metrics.json`. The readouts are the mean triggered p* from `defenses.scores.critical_rate` on the ladder, counting an answer that never flips as 1.0, and TPR at 1%, 5% and 10% FPR with the realized FPR and AUROC at the adaptive rate. Only successful backdoors at the 2-point bar enter a verdict, judged by `scripts.coverage_ledger.classify_cell` with the benign references of `configs/psbd_basis.json`, and the other models are listed in the table.

The predictions are in `PREDICTIONS.md`, written before any sweep. In short, PSBD-TM's mean triggered p* and its TPR at 1% FPR correlate with the token count at Spearman ρ of at least 0.5 on each dataset, and PSBD-RD's correlations are lower on both readouts and both datasets. `vit_gtsrb_badnet_a2o_0_05_trig_p12` reads attack success 0.0 and clean accuracy 0.055 in its `args.json`, a diverged run, so it drops out of the GTSRB series.

## Commands

The sweeps are item 0 of the night queue in `experiments/final_method/gpu_driver.sh`, which runs them on the login A100 after the analysis agents' own work, then runs `cli.analyze` on each model. The readout is CPU only and rewrites the results below.

```bash
source .venv/bin/activate
python experiments/evidence_surplus/trigger_size/measure.py
```

## Results

<!-- results:begin -->
| model | trigger tokens | ASR | clean accuracy | benign | successful 2-point | swept | TM mean p* | TM TPR 1 / 5 / 10% (FPR 1%) | TM AUROC | RD mean p* | RD TPR 1 / 5 / 10% (FPR 1%) | RD AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `vit_cifar100_badnet_a2o_0_05_trig_p2` | 1 | 1.000 | 0.833 | 0.811 | yes | yes | 0.406 | 0.056 / 0.736 / 0.892 (0.007) | 0.952 | 0.040 | 0.000 / 0.000 / 0.000 (0.013) | 0.047 |
| `vit_cifar100_badnet_a2o_0_05_trig_p3` | 4 | 0.999 | 0.829 | 0.811 | yes | yes | 0.511 | 0.046 / 0.606 / 0.861 (0.009) | 0.940 | 0.046 | 0.000 / 0.000 / 0.000 (0.007) | 0.195 |
| `vit_cifar100_badnet_a2o_0_05_trig_p5` | 9 | 1.000 | 0.814 | 0.811 | yes | yes | 0.561 | 0.055 / 0.739 / 0.931 (0.008) | 0.957 | 0.048 | 0.000 / 0.000 / 0.001 (0.006) | 0.365 |
| `vit_cifar100_badnet_a2o_0_05_trig_p8` | 16 | 1.000 | 0.818 | 0.811 | yes | yes | 0.671 | 0.357 / 1.000 / 1.000 (0.012) | 0.986 | 0.061 | 0.018 / 0.229 / 0.452 (0.012) | 0.859 |
| `vit_cifar100_badnet_a2o_0_05_trig_p12` | 36 | 1.000 | 0.830 | 0.811 | yes | yes | 0.691 | 0.396 / 1.000 / 1.000 (0.005) | 0.993 | 0.068 | 0.100 / 0.553 / 0.802 (0.010) | 0.934 |
| `vit_cifar100_badnet_a2o_0_05_trig_p16` | 49 | 1.000 | 0.827 | 0.811 | yes | yes | 0.714 | 0.020 / 0.979 / 1.000 (0.009) | 0.977 | 0.065 | 0.007 / 0.307 / 0.575 (0.009) | 0.880 |
| `vit_gtsrb_badnet_a2o_0_05_trig_p2` | 1 | 1.000 | 0.979 | 0.991 | yes | yes | 0.641 | 0.807 / 0.945 / 0.990 (0.012) | 0.990 | 0.066 | 0.000 / 0.003 / 0.035 (0.009) | 0.602 |
| `vit_gtsrb_badnet_a2o_0_05_trig_p3` | 4 | 1.000 | 0.990 | 0.991 | yes | yes | 0.999 | 1.000 / 1.000 / 1.000 (0.006) | 1.000 | 0.045 | 0.028 / 0.127 / 0.254 (0.010) | 0.671 |
| `vit_gtsrb_badnet_a2o_0_05_trig_p5` | 9 | 1.000 | 0.988 | 0.991 | yes | yes | 0.813 | 0.998 / 1.000 / 1.000 (0.009) | 1.000 | 0.068 | 0.003 / 0.029 / 0.111 (0.012) | 0.652 |
| `vit_gtsrb_badnet_a2o_0_05_trig_p8` | 16 | 1.000 | 0.989 | 0.991 | yes | yes | 0.813 | 1.000 / 1.000 / 1.000 (0.009) | 1.000 | 0.079 | 0.027 / 0.123 / 0.264 (0.010) | 0.751 |
| `vit_gtsrb_badnet_a2o_0_05_trig_p12` | 36 | 0.000 | 0.055 | 0.991 | no | yes | 1.000 | -- / -- / -- (--) | -- | 0.039 | 0.007 / 0.048 / 0.097 (0.007) | 0.500 |
| `vit_gtsrb_badnet_a2o_0_05_trig_p16` | 49 | 1.000 | 0.988 | 0.991 | yes | yes | 0.608 | 1.000 / 1.000 / 1.000 (0.011) | 1.000 | 0.067 | 0.962 / 0.994 / 0.998 (0.010) | 0.998 |

PREDICTIONS.md SHA-256 `f5f085e70e93006e7dc6bfc8d07bea3747ff628e62ace1d6248440d3021c27e3`.

| prediction | verdict |
|---|---|
| TM-p | failed |
| TM-tpr | failed |
| RD-differs | failed |

| dataset | successful models | ρ TM p* | ρ TM TPR 1% | ρ RD p* | ρ RD TPR 1% | ρ TM AUROC |
|---|---|---|---|---|---|---|
| cifar100 | 6 | 1.000 | -0.029 | 0.943 | 0.759 | 0.771 |
| gtsrb | 5 | -0.400 | 0.564 | 0.600 | 0.700 | 0.300 |
<!-- results:end -->
