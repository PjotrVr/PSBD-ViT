# Which attacks actually work, per architecture / dataset / poison rate

## Question

PSU on a model with no backdoor measures nothing. Before any detection sweep, which
(attack, dataset, poison rate) combinations produced a backdoor that actually fires?

## Why it matters

The previously generated PSBD grid targeted `vit_cifar100_wanet` at 3 poison rates.
WaNet on CIFAR-100 reaches **ASR 0.057 at 1%** and 0.643 at 5% in the current ledger. Those 60 jobs would
have run to completion, written well-formed output and measured noise. Nothing in
the pipeline would have flagged it.

Every checkpoint already carries a `metrics.json` with its measured ASR, so this
gate costs nothing and is now wired into `pbs/generate_psbd_jobs.py` directly.

## Run

```bash
python experiments/attack_viability/report.py                 # ViT, CIFAR-10 + CIFAR-100
python experiments/attack_viability/report.py --architecture swin --min-asr 0.9
```

## Finding

The table this section held was read from `checkpoints/*/metrics.json`, which
`report.py` still reads and which is stale: the ASR the ledger uses now comes from
each checkpoint's PSBD baseline cache and sits in `args.json` and
`results/coverage/coverage.json`, and the 2 disagree badly for some cells (the 2026-09-29
audit, `docs/audits/2026-09-29-experiment-audit.md`, found a Swin TaCT `metrics.json` at ASR
0.13 against 0.9987 in `args.json`). The ledger's ASR on ViT CIFAR-10 at 0.01, 0.05 and 0.1,
against the 0.85 bar:

| attack | 0.01 | 0.05 | 0.1 |
|---|---:|---:|---:|
| `badnet_a2o` | 0.997 | 1.000 | 1.000 |
| `blend` | 1.000 | 1.000 | 1.000 |
| `bpp` | 0.982 | 0.987 | 0.994 |
| `lf` | 0.982 | 0.997 | 0.999 |
| `wanet` | 0.112 | 0.961 | 0.890 |
| `sig` | 0.340 | 0.599 | 0.901 |
| `adaptive_blend` | 0.622 | 0.594 | 0.622 |
| `tact` | 0.985 | 0.996 | 1.000, source-mapped |

TaCT works on ViT. The earlier reading of 0.12 to 0.18 came from a stale `metrics.json`.
The CIFAR-10 TaCT model at 10% clears the
bar and is still excluded, because it maps its clean source class to the target with no
trigger. `badnet_a2a` and `lc` hold no ledger cell on CIFAR-10 (all-to-all is scored
separately, and Label-Consistent runs only with adversarial bases on GTSRB target 1). SIG is
under audit (`docs/audits/2026-09-29-experiment-audit.md`).

WaNet on CIFAR-100 reads 0.057, 0.643 and 0.793 in the ledger and never clears. The
gate in `pbs/generate_psbd_jobs.py` still reads `metrics.json`, so it should move to
`args.json` before it gates another sweep.

## Separate finding, from the same data

For **clean-label** attacks (`sig`, `lc`) the poison rate is silently capped.
`poison.choose_poison_indices` clamps the count to the number of eligible samples,
and clean-label eligibility is the target class only. On CIFAR-100 that is 500
images, so 1%, 5% and 10% all resolve to the same 500 poisoned samples: 3
folders, 1 experiment. `args.json` records the *requested* rate with no warning.

Confirmed by ASR: `vit_cifar100_sig_0_01` 0.250 vs `_0_05` 0.249 in the current ledger.
SIG is under audit (`docs/audits/2026-09-29-experiment-audit.md`).

It invalidates any poison-rate trend drawn for clean-label attacks on CIFAR-100. The
ledger now records `realized_poison_rate` beside the requested one, and the full record
is `docs/clean-label-rate-caps.md`.
