# Numeric checks behind the derivations

## Question

`docs/why-psbd-works-theory.md` derives what each explanation of PSBD on ViT predicts as a number and checks each derivation against the cached results. These scripts are those checks. They read the cached per-pass tensors under `results/<folder>/psbd/` and `psbd_metrics.json` for the 54 successful ViT models of `scripts/paper/_common.py`, and reproduce PSBD-TM 0.963, PSBD-RD 0.885 and the matched Gaussian minus token mask gap -0.188 as a population check.

## Scripts

| script | what it computes | section of the theory doc |
|---|---|---|
| `b_check.py` | full-mask event counts by late block, and triggered survival by $J$ | token-mask survival |
| `ladder.py` | clean keep curves along every model's rate ladder, written to a JSON given as the first argument | clean fragility |
| `fit.py` | the AND model and the logistic critical-rate fits on those ladders | clean fragility |
| `cache_read.py` | per-pass flip histograms, tie mass, absolute against fractional PSU, written to a JSON given as the first argument | shift ratio, PSU and confidence |
| `summ.py` | summaries over the `cache_read.py` record | shift ratio, PSU and confidence |
| `lrconf2.py` | the cross-fitted likelihood-ratio ceiling $A^{*}(P_c)$ of confidence | P1 bound |
| `attract.py` | the upper envelope for collapse onto the target (L10) | numeric predictions, L10 |
| `runnerup.py` | how often a shifted prediction lands on the unperturbed runner-up | numeric predictions, L10 |
| `kcurve.py` | the AUROC against $k$ fit on the `_k20` caches | subsampling the number of passes |

## Running

Run each script on CPU from the repository root with `PYTHONPATH=. .venv/bin/python experiments/theory_checks/<script>.py`. `ladder.py` and `cache_read.py` take an output path, and the records the doc was built from are `results/_experiments/theory_checks/ladders.json` and `results/_experiments/theory_checks/cache_read.json`, which `fit.py` and `summ.py` read.

## Status

Written on 2026-09-29 as research reads, not as generators. No macro reads them. The SIG model (`vit_cifar10_sig_0_1`) enters the pooled numbers and is under audit (`docs/audits/2026-09-29-experiment-audit.md`). Rerun every script when the panel changes.
