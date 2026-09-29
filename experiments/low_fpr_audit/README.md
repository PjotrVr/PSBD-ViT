# What does PSBD deliver at a false-positive budget you would actually run?

## Question

Every headline in this project is an AUROC. AUROC integrates over the whole ROC curve,
including false-positive rates no operator would deploy at, and at 1% poisoning the
positive class is rare enough that the integral is dominated by a region nobody uses.
`docs/results-report.md` already notes in passing that "AUROC is not the binding
constraint at low rate, the shape of the low-FPR tail is". This measures it.

`average_precision`, `auprc`, `max_fpr` and `partial_auc` return **zero grep hits**
anywhere else in the repo.

## Result on the current panel

PSBD-TM (`token_mask @ before_attention_norm`) at the adaptive 0.8 rule's rate, fractional
PSU, the paper panel of 54 models successful at the 2-point clean-accuracy bar, every value
read from `results/<folder>/psbd_metrics.json`. Threshold set on clean validation, so every
TPR below is what a defender would actually get.

    PYTHONPATH=. .venv/bin/python scratch/stale_numbers/low_fpr_panel.py

| poison | n | AUROC | AUPRC | TPR@1% | TPR@5% | TPR@10% | TPR@25% |
|---|---|---|---|---|---|---|---|
| 1% | 17 | 0.968 | 0.965 | **0.704** | 0.844 | 0.877 | 0.908 |
| 5% | 19 | 0.972 | 0.971 | **0.747** | 0.846 | 0.877 | 0.916 |
| 10% | 18 | 0.949 | 0.949 | **0.788** | 0.874 | 0.906 | 0.941 |
| all | 54 | 0.963 | 0.962 | **0.747** | 0.855 | 0.887 | 0.922 |

At a 1% false-positive budget the method catches 0.70 to 0.79 of triggered inputs against an
AUROC of 0.95 to 0.97, so the low-FPR tail still costs about a quarter of the detections the
AUROC suggests.

**3 of 54 models have AUROC >= 0.85 and TPR@1%FPR < 0.05:**

| model | AUROC | TPR@1% | TPR@5% |
|---|---|---|---|
| `vit_cifar10_tact_0_01` | 0.979 | **0.005** | 0.005 |
| `vit_cifar10_tact_0_05` | 0.966 | **0.000** | 0.000 |
| `vit_gtsrb_tact_0_05` | 0.942 | 0.030 | 0.230 |

All 3 are the panel's TaCT models, and none recovers by 5% FPR, so they are genuine failures
at any budget below 10%.

## The earlier reading

The saved record, `results/low_fpr_audit.json`, was read at the matched 0.6 rule over 56
all-to-one cells with ASR at least 0.5, below the 0.85 bar and before the TaCT cells
existed. It is kept for the trail and has not been re-run on the current panel:

| poison | n | AUROC | AUPRC | TPR@1% | TPR@5% | TPR@10% | TPR@25% |
|---|---|---|---|---|---|---|---|
| 1% | 14 | 0.913 | 0.896 | 0.486 | 0.764 | 0.818 | 0.897 |
| 5% | 19 | 0.883 | 0.866 | 0.391 | 0.665 | 0.767 | 0.867 |
| 10% | 23 | 0.883 | 0.871 | 0.451 | 0.680 | 0.772 | 0.871 |

That reading had PSBD-TM missing more than half of triggered inputs at 1% FPR and flagged
13 of 56 cells with AUROC >= 0.85 and TPR@1%FPR < 0.05. At the canonical reading on the
current panel it misses about 0.25 and flags 3 of 54. The difference is the rule. Read at
the matched 0.6 rule, the same 54 models give a mean TPR@1%FPR of 0.452 and flag 16, so the
adaptive 0.8 rule, which perturbs harder, is what separates the extreme tail.

## Why this is not a mis-set threshold

Achieved FPR tracks the nominal quantile closely (0.0101 against 0.01 and 0.0468 against
0.05 on the current panel, 0.0113 and 0.0524 in the earlier record), so the calibration is
right and the clean and backdoor score distributions genuinely
overlap in the extreme tail. 2 candidate fixes were tested and both failed: excluding
negative-PSU validation samples from the threshold merely moves along the same ROC curve
at double the FPR, and mapping negative PSU to "definitely clean" changes the ROC but makes
the mean worse (AUROC 0.908 to 0.749). Both were tested on the earlier record only.

## What to do with it

Report AUPRC and TPR at 1/5/10% FPR beside every AUROC. The caveat on AUPRC is that this
evaluation pairs clean against backdoor roughly 1:1, while a defender screening a training
set faces a prevalence of 1% or lower, so the AUPRC here is still optimistic.

## Running it

    PYTHONPATH=. python experiments/low_fpr_audit/measure.py

CPU only, about 3 minutes over the cached panel. Writes `results/low_fpr_audit.json`.
