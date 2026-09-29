# Is PSU just measuring baseline confidence?

## Question

The skeptical reading of PSBD: a backdoored model is extremely confident on triggered
inputs. PSU subtracts a dropout-perturbed confidence from the no-dropout one. And a
sample starting near probability 1 has more room to fall than 1 starting at 0.6. If
that is the whole story, PSU is an elaborate proxy for baseline confidence and a
defender could skip the `k` stochastic forward passes entirely.

This was raised by an adversarial verification pass, which noted that on the benign
control a confidence-only detector sits as close to chance as PSBD (0.486 against 0.506 in
the saved record, `pre_residual`).

## Run

```bash
PYTHONPATH=. python experiments/psu_vs_confidence/measure.py
```

No GPU, seconds, reads only the cached tensors.

## Finding: the skeptical reading is refuted

### The saved record, a non-canonical reading

`results/_experiments/psu_vs_confidence/psu_vs_confidence.json` holds placement
`pre_residual` (dropout before both residual adds), each checkpoint at the rate whose PSU
AUROC is highest (an oracle rate that flatters PSU), for 295 folders whose name contains
`0_1`. The table this section showed before 2026-09-29 was read at `before_mlp_residual`,
which no saved record holds, so it has been replaced by the saved values. CIFAR-10 ViT at 10%
poisoning, AUROC with backdoor as the positive class:

| checkpoint | PSU | confidence only | fractional drop |
|---|---|---|---|
| `blend` | 0.978 | 0.701 | **0.992** |
| `bpp` | 0.977 | 0.822 | **0.989** |
| `lf` | 0.947 | 0.780 | **0.962** |
| `badnet_a2o` | 0.889 | 0.878 | **0.919** |
| `badnet_a2a` | 0.510 | 0.244 | 0.501 |
| benign control | 0.506 | 0.486 | 0.504 |
| **mean (backdoored)** | **0.860** | **0.685** | **0.873** |

PSU beats confidence-only on 5 of 5 checkpoints, by 0.175 on average. Over the plain
CIFAR-100 and Tiny models of the same record, confidence alone beats PSU on 8 of 44 (2026-09-29
audit, `docs/audits/2026-09-29-experiment-audit.md`).

### The canonical reading on the current panel

PSBD-TM (`before_attention_norm_token_mask`) at the adaptive 0.8 rule's rate, the paper
panel of 54 models successful at the 2-point clean-accuracy bar, fractional and absolute PSU
read from each model's `psbd_metrics.json` at q0.25 and confidence alone from the cached
baseline:

    PYTHONPATH=. .venv/bin/python scratch/stale_numbers/psu_confidence_panel.py

| dataset | n | absolute PSU | fractional PSU | confidence only | confidence wins |
|---|---:|---:|---:|---:|---:|
| cifar10 | 15 | 0.910 | 0.919 | 0.593 | 0 |
| cifar100 | 12 | 0.958 | 0.979 | 0.813 | 0 |
| gtsrb | 13 | 0.988 | 0.984 | 0.480 | 1 |
| tiny | 14 | 0.969 | 0.977 | 0.877 | 0 |
| **all** | **54** | **0.955** | **0.963** | **0.688** | **1** |

Confidence alone beats fractional PSU on 1 of 54 models, `vit_gtsrb_tact_0_05` (0.999
against 0.942), and absolute PSU on 3. The stochastic passes are doing real work at the
canonical reading too. `vit_cifar10_sig_0_1`, where confidence alone also won (0.593 against
0.418), fails the 2-point clean-accuracy bar and is under audit
(`docs/audits/2026-09-29-experiment-audit.md`).

The decisive column is the fractional drop. `1 - mean_dropout / P_c(x)` divides out the
starting confidence entirely. If PSU worked only because confident samples fall further in
absolute terms, normalizing by that confidence would destroy the signal. It does the
opposite: fractional PSU reads at least as well as absolute PSU on average in both readings
and beats it on 36 of the 54 panel models. So PSU is measuring how *robust* the prediction
is, not how confident it started.

## The fractional form

Fractional PSU is now the canon headline statistic (`detection_psu_ratio`), with absolute PSU
reported beside it. On the 54-model panel it reads 0.963 against 0.955 for the absolute form.
The gain is small and costs nothing: it is the same cached tensors divided by a number
already on disk.

It also has a principled reason to be preferred over the paper's absolute form. The
threshold is a quantile of clean-validation PSU, and absolute PSU is bounded above by
the starting confidence, so the threshold inherits the validation set's confidence
distribution. The ratio does not, which should make it transfer better across
datasets and models with different calibration.

## Where confidence alone does explain most of it

`badnet_a2o` on CIFAR-10 at 10% is the exception in the saved record: confidence-only
reaches 0.878 against PSU's 0.889 at `pre_residual`, so for the static patch trigger most
of the separation is available without dropout at all. For `blend` the gap is 0.701
against 0.978. So the value PSBD adds is largest exactly where a naive baseline is weakest,
which is the right way round.

## Subquestions

1. The fractional form's advantage holds under the adaptive rate rule on the 54-model
   panel (above). Whether it holds across SAM checkpoints has not been read.
2. Confidence-only scores 0.688 mean on the 54-model panel. That is a baseline no PSBD
   paper reports, and any detection method should be shown to beat it.
3. Does combining the 2 (confidence and PSU as 2 features) beat either? That
   would say they carry partly independent information.
