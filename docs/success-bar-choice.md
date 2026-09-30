# The clean-accuracy bar barely moves the results

## The question

A model counts in the results only when its backdoor succeeded: it clears the attack success bar and its clean accuracy is within a fixed number of points of its benign reference. The headline uses 2 points, the tolerance the literature uses. A natural question is whether the conclusions depend on that choice, for example whether 5 points would change them. They do not. Every headline number is generated at both bars (macros suffixed `FivePoint` in `paper/headline.tex`) and the 2 readings of the means differ by at most 0.003. The AUROC margin over IBD-PSC$^\ast$ moves by 0.009 and its interval contains 0 at both bars.

## The 2 bars side by side

| reading | 2-point bar (headline) | 5-point bar |
|---|---|---|
| ViT models with both headline placements | 56 | 58 |
| PSBD-TM mean AUROC | 0.951 | 0.950 |
| PSBD-RD mean AUROC | 0.876 | 0.879 |
| PSBD-TM minus PSBD-RD | +0.075 [+0.017, +0.138] | +0.072 [+0.017, +0.132] |
| PSBD-TM minus IBD-PSC$^\ast$ | +0.018 [$-$0.017, +0.048] | +0.027 [$-$0.009, +0.057] |
| rank of PSBD-TM among 13 defenses | 1st | 1st |
| rank of PSBD-RD among 13 defenses | 5th | 5th |

## Why the difference is small

The 2 bars differ by 2 models, `vit_cifar10_wanet_0_05` (clean accuracy 3.9 points below benign) and `vit_gtsrb_wanet_0_1` (3.3 points below). Both are in at 5 points and out at 2. 2 models out of 58 move a mean by at most a few thousandths unless they are extreme, and neither is. `vit_cifar10_sig_0_1` loses 10.6 points and is out at both bars, which is why SIG is absent from every ViT result either way.

## What to take from it

The sign of every gain, the interval of the gain over PSBD-RD excluding 0, the interval of the margin over IBD-PSC$^\ast$ containing 0 and the rank of PSBD-TM are the same at both bars. The choice of bar is a reporting convention, not a result the conclusions hinge on. The per-model list at each bar is in `notebooks/all-numbers.ipynb`, and the bar itself is `clean_accuracy_drop_bar_headline` in `configs/psbd_basis.json`, applied by `scripts.paper._common.clearing_cells`.
