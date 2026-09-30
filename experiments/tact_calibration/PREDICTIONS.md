# TaCT calibration predictions

Written on 2026-09-30 after reading the 4 TaCT tuning models and before reading any development, holdout or remaining panel model. The machine-readable form is `preregistration.json`. `analyze.py confirm` refuses to run until that file is committed unchanged and stores its sha256. `judge.py` computes every verdict from it. The tuning readings behind each choice are in `results/_experiments/tact_calibration/tuning.json`.

## Fixed rules

Every score is PSBD-TM's fractional PSU at its adaptive rate, the rate `psbd_metrics.json` records (or a 3-pass ladder on the standard 2000 validation images for the 6 holdout models), read from 20 cached passes. A k below 20 is the first k passes. The primary reading is k = 20. Validation grows to 4000 images on CIFAR-10 and GTSRB and to 5000 on CIFAR-100 and Tiny ImageNet, taken as the next images of the same split permutation, and those images leave the clean and triggered evaluation sets. `analyze.enlarged_rows` asserts that validation and evaluation share no image. Low means poisoned, and every score is thresholded at a quantile of its own validation values.

The fallback class r(x) is the most frequent class among the passes whose argmax differs from the unperturbed argmax, ties going to the smallest class index. An input with no such pass keeps the global reference. The fallback score is the shrunk percentile (n_r F_r(s) + m F(s)) / (n_r + m) with m = 200. The class score is the robust z score (s minus M_c) over D_c by predicted class c, where M_c and D_c are the median and median absolute deviation shrunk toward the global ones with weights n_c and m = 5. The OR rules take the minimum of validation percentiles, `or_*` over PSBD-TM and the calibrated score and `final_or_*` over PSBD-TM, `pre_residual_blocks_5_8` and the calibrated score. The references are PSBD-TM alone and the final method (the minimum of the 2 plain percentiles) on the same evaluation set and passes.

## Shrinkage choice

The grid was 5, 20, 50 and 200 images, chosen by each calibrated score's mean TPR over 1%, 5% and 10% FPR on the 4 tuning models, ties to the larger m. The fallback score read the same TPR at every m, so the tie rule gave 200. The class score chose 5, the edge of the grid, as the earlier `final_method` calibration did.

## Predictions

| id | claim | refuted when | expected |
|---|---|---|---|
| P1 | On the 6 holdout models, triggered inputs rarely fall back to their own source class under PSBD-TM | the mean own-source share is at least 0.1 | holds |
| P2 | PSBD-TM alone and the final method, uncalibrated, detect under 0.3 of triggered inputs at 10% FPR on at least 4 of the 6 holdout models, at k = 3 and at k = 20 | 3 or more holdout models reach 0.3 | holds |
| P3 | The fallback OR rules raise holdout mean TPR at 10% FPR by at least 0.1 over their references | the gain is below 0.1 or its interval touches 0 | fails |
| P4 | The class OR rules raise holdout mean TPR at 10% FPR by at least 0.1 over their references | the gain is below 0.1 or its interval touches 0 | uncertain |
| P5 | No OR rule loses more than 0.02 mean TPR at 1%, 5% or 10% FPR on the successful non-TaCT development models. The same bar applies separately to every successful non-TaCT panel model read | any mean loss beyond 0.02 | the fallback rules hold, the class rules fail at 1% |
| P6 | Each OR rule's mean realized all-class FPR stays within 0.005 of 1% and within 0.01 of 5% and 10% | a larger deviation | holds |
| P7 | k = 10 is the smallest k whose mean AUROC and TPR at 1%, 5% and 10% FPR for PSBD-TM alone are within 0.01 of k = 20. Also k = 20 raises mean AUROC over k = 3 with the interval above 0 | another k is recommended, or the interval touches 0 | holds |

## Reasons for each expectation

The premise of the fallback score failed on every tuning model. Their triggered inputs never fell back to their source class under PSBD-TM. They fell to a single attractor class, the same class that the clean source images fall to, so the fallback score has nothing to separate and P3 is expected to fail. P1 states that failure as a prediction for the multi-source retrains.

P2 is the evidence surplus account's advance prediction. A TaCT decision needs the trigger and the source content together, so masking tokens removes the evidence for the target as fast as it removes a clean image's evidence for its own class. Triggered inputs then shift like clean ones, and the threshold, set on clean images of every class, catches few of them. 3 of the 4 tuning models read TPR near 0 at 10% FPR. The exception is `vit_gtsrb_tact_0_05`, whose triggered inputs shift far less than its clean ones, and the bar of 0.3 is where a detector stops being usable at that FPR.

The class score helped 2 tuning models and did nothing for the other 2, one of them the model where PSBD-TM inverts. The development and panel models are expected to show the harm the earlier calibration by predicted class showed, because a triggered input is ranked against clean images of the target class, whose scores the backdoor itself moved. A larger validation set does not remove that bias, it only removes the sampling noise on top of it.

The dev set declared in `experiments/cache_readouts/dev_set.json` has 10 models, not 12. 2 of them are tuning models. `vit_gtsrb_wanet_0_1` fails the 2-point clean bar, so P5_dev pools 7 models. The model is read and reported without entering any pooled claim.
