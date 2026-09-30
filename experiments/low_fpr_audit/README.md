# What does PSBD deliver at a false-positive budget you would actually run?

## Question

Every headline in this project is an AUROC. AUROC integrates over the whole ROC curve,
including false-positive rates no operator would deploy at, and at 1% poisoning the
positive class is rare enough that the integral is dominated by a region nobody uses.
`docs/results-report.md` already notes in passing that "AUROC is not the binding
constraint at low rate, the shape of the low-FPR tail is". This measures it.

`average_precision`, `auprc`, `max_fpr` and `partial_auc` return **zero grep hits**
anywhere else in the repo.

<!-- results:begin -->
<!-- Everything down to results:end is rendered by panel.py from results/_experiments/low_fpr_audit/panel.json. -->

## Result on the current panel

PSBD-TM (`token_mask @ before_attention_norm`) at the adaptive 0.8 rule's rate, fractional PSU, the paper panel of 56 models successful at the 2-point clean-accuracy bar, every value read from `results/<folder>/psbd_metrics.json`. Thresholds are set on clean validation, so every TPR below is what a defender would get.

    PYTHONPATH=. .venv/bin/python experiments/low_fpr_audit/panel.py

| poison | n | AUROC | AUPRC | TPR@1% | TPR@5% | TPR@10% | TPR@25% |
|---|---|---|---|---|---|---|---|
| 1% | 18 | 0.929 | 0.932 | **0.665** | 0.797 | 0.828 | 0.858 |
| 5% | 20 | 0.972 | 0.971 | **0.751** | 0.850 | 0.881 | 0.919 |
| 10% | 18 | 0.949 | 0.949 | **0.788** | 0.874 | 0.906 | 0.941 |
| all | 56 | 0.951 | 0.951 | **0.735** | 0.841 | 0.872 | 0.906 |

At a 1% false-positive budget PSBD-TM catches 0.66 to 0.79 of triggered inputs by poison rate, against an AUROC of 0.93 to 0.97. Over the whole panel it catches 0.735 at 1% FPR against an AUROC of 0.951, so the low-FPR tail still costs about 0.26 of the triggered inputs.

**3 of 56 models have AUROC >= 0.85 and TPR@1%FPR < 0.05:**

| model | AUROC | TPR@1% | TPR@5% |
|---|---|---|---|
| `vit_cifar10_tact_0_01` | 0.979 | **0.005** | 0.005 |
| `vit_cifar10_tact_0_05` | 0.966 | **0.000** | 0.000 |
| `vit_gtsrb_tact_0_05` | 0.942 | **0.030** | 0.230 |

The flag needs a high AUROC, so it leaves out the models where the score itself inverts. Those read TPR near 0 at every budget and are `vit_gtsrb_tact_0_01_cos` (AUROC 0.266, TPR@1% 0.000) and `vit_cifar10_wanet_0_1` (AUROC 0.459, TPR@1% 0.023).

## The matched 0.6 rule on the same models

Read at the matched 0.6 rung instead, the same 56 models give a mean TPR@1%FPR of 0.451 against 0.735 at the adaptive rule and flag 16 models against 3, so the adaptive 0.8 rule, which perturbs harder, is what separates the extreme tail.

| poison | n | AUROC | AUPRC | TPR@1% | TPR@5% | TPR@10% | TPR@25% |
|---|---|---|---|---|---|---|---|
| 1% | 18 | 0.889 | 0.867 | **0.443** | 0.684 | 0.744 | 0.827 |
| 5% | 20 | 0.949 | 0.917 | **0.441** | 0.760 | 0.840 | 0.920 |
| 10% | 18 | 0.934 | 0.906 | **0.470** | 0.768 | 0.875 | 0.939 |
| all | 56 | 0.925 | 0.898 | **0.451** | 0.738 | 0.820 | 0.896 |

## Realized false-positive rates

Achieved FPR on the paired clean test split tracks the nominal quantile, 0.0100 against 0.01 and 0.0460 against 0.05 on the panel at the adaptive rule. The calibration is right and the clean and triggered score distributions overlap in the extreme tail.
<!-- results:end -->

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
13 of 56 cells with AUROC >= 0.85 and TPR@1%FPR < 0.05. The current panel reading above
states the same quantities at the canonical rule.

## Why this is not a mis-set threshold

Achieved FPR tracked the nominal quantile in the earlier record too (0.0113 against 0.01
and 0.0524 against 0.05), so the calibration is right and the clean and backdoor score
distributions genuinely overlap in the extreme tail. 2 candidate fixes were tested and both failed: excluding
negative-PSU validation samples from the threshold merely moves along the same ROC curve
at double the FPR, and mapping negative PSU to "definitely clean" changes the ROC but makes
the mean worse (AUROC 0.908 to 0.749). Both were tested on the earlier record only.

## What to do with it

Report AUPRC and TPR at 1/5/10% FPR beside every AUROC. The caveat on AUPRC is that this
evaluation pairs clean against backdoor roughly 1:1, while a defender screening a training
set faces a prevalence of 1% or lower, so the AUPRC here is still optimistic.

## Running it

    PYTHONPATH=. python experiments/low_fpr_audit/measure.py
    PYTHONPATH=. .venv/bin/python experiments/low_fpr_audit/panel.py

`measure.py` is the earlier matched-rule audit over every cached cell, CPU only, about 3
minutes, and writes `results/low_fpr_audit.json`. `panel.py` reads the current panel from
each model's `psbd_metrics.json` in seconds, writes
`results/_experiments/low_fpr_audit/panel.json` and renders the results block above.
