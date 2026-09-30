# Predictions for training-set detection

Written and committed before any training-set score was computed or read. The only numbers I had seen when writing this are the test-time readings of the same models (`experiments/cache_readouts/README.md`, the development-set table) and the clean-validation shift ratios of the cached rate ladders, which both rate rules below read.

## Setting

Li et al. (arXiv 2406.05826) score every image of the poisoned training set with PSU on the model trained on it, flag PSU < T with T the 25th percentile of clean-validation PSU, and report TPR on the poisoned training images and FPR on the clean training images. The repository's paper scores triggered test images against their clean twins. This experiment moves our placements into their setting on the 10 development models of `experiments/cache_readouts/dev_set.json` and on the 2 ResNet-18 GTSRB reproductions (`resnet18_gtsrb_badnet_a2o_0_1`, `resnet18_gtsrb_blend_0_1`).

## Scored groups

Each model's poisoned training set is rebuilt with the functions training used (`cli.train_backdoor.build_training_set`, `attacks.poisoning.choose_poison_indices` or `choose_indices_with_cover`), the seed of `args.json` and 0 where it records null, the rule `cli.backfill` documents for runs that predate the seeding commit. Images pass through the training transform with no augmentation (none of these models trained with any) and poisoned images carry the trigger exactly as trained.

- **poisoned**: every reconstructed poisoned training index, triggered, the positives.
- **cover**: cover samples (trigger or noise warp with the true label), at most 2000 drawn with seed 0. Li et al.'s code counts covers as negatives, so they are reported as their own group and folded into a paper-convention FPR weighted by their population count.
- **clean_train**: 10000 indices drawn with seed 0 from the training indices that are neither poisoned nor cover, the negatives.
- **validation**: the standard 2000-image PSBD validation split (`data.splits`), the threshold reference.

## Rate rules

Every placement is read at 2 rates.

1. **Ours**: `defenses.decision.select_rate_adaptively` at 0.8 on the cached clean-validation shift ratios, the smallest cached rate whose validation shift reaches 0.8.
2. **Li et al.'s**: the paper says to pick the rate where the validation shift ratio approaches 0.8 while the gap between the shift ratio of the entire training set and that of clean validation is largest. Their released code (github.com/WL-619/PSBD, `detection/psbd.py`, `select_dropout_rate`, read at commit 7c58a88) makes this exact: among the rates whose validation shift is at least 0.8, take the argmax of the validation shift minus the entire-training-set shift (first maximum on a tie), and when no rate reaches 0.8 take that argmax over every rate. I apply it to each placement's cached ladder rather than to their 0.1 to 0.9 grid, since the residual placements live below 0.1. The entire-training-set shift is estimated on a fixed subset (the first 2000 clean_train indices, the first 1000 poisoned and the first 1000 covers of their seeded orders), each group's shift weighted by its population count in the full training set. Validation shifts come from the cached `cli.sweep` passes.

## Methods and readings

PSBD-TM (`before_attention_norm` with `token_mask`), PSBD-RD (`post_residual`) and the final method (PSBD-TM with `pre_residual_blocks_5_8`, plain minimum of clean-validation percentiles, `experiments.cache_readouts.fusion_rules`), each at k = 3, bfloat16, batch 64 and mask seed 0 through `defenses.inference` as `cli.sweep` runs it. ResNet-18 gets `post_residual` only. Fractional PSU is the headline statistic and the paper's absolute PSU is reported beside it. STRIP and CD-L run through `detectors.build_detector` on the same images (CD-L on a fixed subset because of its cost).

Per model and method I report TPR on poisoned at thresholds set to the 1%, 5% and 10% quantiles of clean validation with the FPR realized on clean_train, TPR at thresholds set to the same quantiles of clean_train itself (TPR at the realized FPR), the paper's rule (T at the 25th percentile of validation) with TPR, realized FPR and the paper-convention FPR with covers, and AUROC of poisoned against clean_train. The train and validation gap is the difference of the clean_train and validation medians and the realized FPR at each quantile.

## Reconstruction checks

A model enters the headline set only if it is `successful_2pt` in the ledger and passes 3 checks.

- R1: the reconstructed poisoned count (and cover count) equals `n_poisoned` (`n_cover`) of `args.json` when recorded.
- R2: the share of reconstructed poisoned images the model assigns to their attack-success label is at least the recorded ASR minus 0.05.
- R3, membership: an index that was poisoned was never trained with its clean image and true label, so its clean image should look held out. With L the mean true-label cross-entropy of the untriggered image, r = (L_poisoned − L_clean_train) / (L_validation − L_clean_train) must be at least 0.5. When L_validation − L_clean_train is below 0.05 the check is inconclusive and the model counts as ambiguous. A wrong index set gives r near 0.

A model that fails is reported under its own label and kept out of every pooled mean.

## Predictions and refutation rules

**P1, ranking.** On the headline ViT set under our rate rule and fractional PSU, mean AUROC orders final method ≥ PSBD-TM > PSBD-RD, with final method allowed to trail PSBD-TM by at most 0.005. Mean TPR at the realized 1%, 5% and 10% FPR orders PSBD-TM above PSBD-RD at 2 or more of the 3. Refuted if PSBD-TM's mean AUROC is at or below PSBD-RD's, or if PSBD-TM leads at fewer than 2 of the 3 FPRs, or if the final method trails PSBD-TM by more than 0.005.

**P2, size of the gain.** The training-set mean AUROC gain of PSBD-TM over PSBD-RD is at least half its test-time gain on the same headline models (recomputed from the caches with the same code). Refuted if it is below half or negative.

**P3, their rate rule.** P1's ordering of PSBD-TM above PSBD-RD in mean AUROC also holds when both are read at Li et al.'s rate. Refuted otherwise.

**P4, train and validation gap.** Training images were fit, so their predictions should resist the probe better than held-out images. Under fractional PSU and PSBD-TM the clean_train median sits below the validation median on a majority of headline models, and the realized clean_train FPR at the paper's 25th-percentile threshold exceeds 0.25 on a majority. Refuted if either majority fails. Under absolute PSU a fitted image also starts from a higher confidence and so has more to lose, which works against the gap, so I predict the realized FPR at the 25th percentile is lower under absolute PSU than under fractional PSU on a majority of headline models. Refuted if not.

**P5, reproduction on their architecture.** On both ResNet-18 GTSRB models, `post_residual` with absolute PSU, Li et al.'s rate rule and T at the 25th percentile gives TPR of at least 0.8 (their own failure bar) with realized clean_train FPR at most 0.30. Refuted if either model misses either bound.

**P6, baselines.** On the subset where all methods are scored, the order of PSBD-TM, STRIP and CD-L in mean AUROC on the training set equals their order at test time on the same headline models (from `results/<folder>/detectors/` and the caches). Refuted if any pair swaps.

## Paper-mirror set, added before any training-set score

A second model set mirrors Li et al.'s main table (their attacks and datasets at 10% poisoning) and runs first: `vit_cifar10_badnet_a2o_0_1`, `vit_cifar10_blend_0_1`, `vit_cifar10_wanet_0_1`, `vit_gtsrb_badnet_a2o_0_1`, `vit_gtsrb_blend_0_1`, `vit_tiny_badnet_a2o_0_1`, `vit_tiny_blend_0_1`, `vit_tiny_wanet_0_1`, `vit_gtsrb_lc_0_05_tl1_adv` (their Label-Consistent, the only one we have, at 5% because clean-label GTSRB caps there) and the 2 ResNet-18 GTSRB reproductions. All 9 ViT models exist and are `successful_2pt` in the ledger. The development set follows it as time allows.

R2 and R3 cannot verify a clean-label reconstruction. Its poisoned images already belong to the target class, so R2 is met by any index set, and each poisoned image was trained with its true label, so R3 has no held-out contrast. The Label-Consistent model is checked by R1 and its recorded seed only and is labeled as such in every table.

**P1 to P4 on the paper-mirror set.** The same rules as above, evaluated on the paper-mirror ViT models that pass their checks, as a separate set that is never pooled with the development set.

**P7, the paper's table.** With absolute PSU, Li et al.'s rate rule and T at the 25th percentile, PSBD-TM reaches TPR of at least 0.8 on at least 7 of the 9 paper-mirror ViT models. Refuted if fewer than 7 do.
