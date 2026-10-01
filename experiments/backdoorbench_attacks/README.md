# PSBD-TM and the final method on BackdoorBench's own ViT-B/16 checkpoints

The ViT panel covers BadNets, Blend, BPP, LF, WaNet, TaCT and 1 LC model, all trained in this repository on 1 recipe. `backdoor_bench_checkpoints/` holds BackdoorBench's own ViT-B/16 checkpoints, trained on BackdoorBench's recipe, including attack families the panel has never met. This experiment asks whether PSBD-TM and the final method (PSBD-TM fused with `pre_residual_blocks_5_8`) detect those families, and so whether the method holds on models trained outside this repository. The predictions were committed before any sweep ran (`PREDICTIONS.md`, commit `a2dccc6`).

## Scope

Before any sweep ran on 2026-09-30, the user cut the scope to attack families the panel lacks on a few models, read with the final method's own 2 probes. The queue (`queue.py`) runs TrojanNN, SSBA, Input-Aware, LIRA and Blind at 10% and then TrojanNN at 5%. PSBD-RD, every competitor detector, the 1% rate and BackdoorBench's versions of the panel's own attacks are out of scope. LIRA's folders are empty, so LIRA is listed and not tested.

## Loading a BackdoorBench checkpoint

BackdoorBench builds `vit_b_16` as `torch.nn.Sequential(Resize((224, 224)), vit_b_16)` with a fresh head (`third_party/BackdoorBench/utils/aggregate_block/model_trainer_generate.py`, lines 123 to 132, at the pinned commit `f02e353`, `Resize` imported from `torchvision.transforms` on line 12). `models.backbones.build_vit` returns the same module tree, so the checkpoint's state dict loads strictly and the 32-pixel input stored as `img_size` in `attack_result.pt` is resized to 224 inside the model, as in BackdoorBench. Its test transform is `Resize((h, w))`, `ToTensor` and `get_dataset_normalization(dataset)` (`utils/aggregate_block/dataset_and_transform_generate.py`, lines 134 to 146 and 81 to 104). Both resizes are bilinear. The model's resize only upsamples, where torchvision's antialias flag has no effect, which I checked numerically on 32 and 64 pixel inputs (maximum difference at float precision).

2 things did not match and are fixed here. BackdoorBench's CIFAR-10 standard deviations are (0.247, 0.243, 0.261) where `DATASET_REGISTRY` carries (0.2023, 0.1994, 0.2010). CIFAR-100's mean differs in the 4th decimal. `data.backdoorbench.BACKDOORBENCH_NORMALIZATION` copies BackdoorBench's table and every BackdoorBench read goes through it. The normalization control in the results measures what the registry's statistics cost. Blind stores its triggered test images as JPEG (the record's `save_file_format`) where the other attacks store PNG. They are read as stored.

`data.backdoorbench.build_psbd_loaders_from_backdoorbench` builds the standard PSBD split for such a folder: the same permutation `data.splits` draws from (test set size, seed 0), its first 2000 indices as the clean validation set and the rest as the analysis pool, clean images from this project's raw test set and triggered images from the checkpoint's bd_test folder, paired by the original test index BackdoorBench uses as the file name. A bd_test entry whose recorded original label disagrees with the clean test label at its index is dropped and listed in the split manifest (`misaligned_bd_test_indices`). BackdoorBench's WaNet folders have such entries. None of the in-scope folders has one. `cli.sweep` and `cli.analyze` read a results folder named `bb_<folder>` through this builder with no new flag. `tests/test_backdoorbench_split.py` covers the split, the pairing, the normalization and the misalignment rule.

## Method

1. `inventory.py` (CPU) records every folder's attack, dataset, rate, label mode, target, bd_test size and format, the label alignment and the trigger footprint on sampled triggered images against their clean twins.
2. `evaluate.py` (GPU) measures clean accuracy over the whole test set and ASR over every eligible bd_test image with `models.backbones.load_checkpoint` and `evaluation.metrics`, and repeats 1 CIFAR-10 model under the registry's normalization as the control. `fetch_leaderboard.py` stored BackdoorBench's published ViT-B/16 numbers in `leaderboard_vit_b_16.json` for the comparison.
3. The success bar is ASR at or above the panel's 0.85 and no collapse (clean accuracy at least half the best BackdoorBench ViT-B/16 of the dataset). BackdoorBench publishes no benign ViT-B/16, so each model's clean accuracy is reported against 2 references, the best BackdoorBench ViT-B/16 of its dataset on the leaderboard and this project's own benign ViT, with a 2-point verdict against each. No model is dropped for the 2-point bar.
4. `sweep_model.py` (GPU) runs `cli.sweep` for `before_attention_norm` with `token_mask` and for `pre_residual` with `--block-range 5 8`, on their ladders from `configs/psbd_basis.json`, k = 3, bfloat16, then `cli.analyze`. It caps the process at 0.15 of the GPU's memory.
5. `readout.py` (CPU) reads PSBD-TM and its partner at their adaptive rates and fuses them with `experiments/cache_readouts/fusion_rules.py`'s `fractional_psu` and `fuse_and_evaluate`, the plain minimum of the 2 clean-validation percentiles (`min_rank`) and the plain average of the 2 fractional PSUs (`mean_psu`). `render_readme.py` writes everything between the results markers below, verdicts included.

## Commands

```bash
cd /lustre/home/pstika/projects/PSBD-ViT-bb-attacks   # or any checkout with this commit
.venv/bin/python -m experiments.backdoorbench_attacks.inventory
bash experiments/backdoorbench_attacks/run_queue.sh evaluate    # GPU, 1 lock slot per model
bash experiments/backdoorbench_attacks/run_queue.sh sweep       # GPU, resumable
.venv/bin/python -m experiments.backdoorbench_attacks.readout
.venv/bin/python -m experiments.backdoorbench_attacks.render_readme
```

`run_queue.sh` skips a model whose record exists, holds 1 of the 2 shared lock slots (`scratch/gpu.lock`, `scratch/gpu2.lock`) per model, starts nothing outside 17:00 to 06:30 or when the model's estimated duration would run past 07:00, and logs to `scratch/backdoorbench_attacks/`. A sweep interrupted mid-placement is redone from that placement's first rate, since `cli.sweep --skip-existing` skips only complete placements.

## Results

<!-- results:begin -->

## Changes of scope

- 2026-10-01: a model that fails the reproduction gate is recorded in `jobs/<folder>.json` and skipped, where it stopped the whole queue before. CIFAR-10 Input-Aware stopped it on 2026-09-30. The queue order after the models already run is LIRA, Blind, TrojanNN at 5% and GTSRB Input-Aware last. Both LIRA folders hold no `attack_result.pt`, so LIRA is dropped. PSBD-RD and the competitor detectors stay out of scope.

## Status

6 of the 12 readable in-scope models are swept and read. Not yet swept or below the success bar: `cifar10_inputaware_0_1`, `cifar10_blind_0_1`, `cifar10_trojannn_0_05`, `gtsrb_trojannn_0_05`, `tiny_trojannn_0_05`, `gtsrb_inputaware_0_1`. The jobs records under `jobs/` say which.

## Inventory

`backdoor_bench_checkpoints/` holds 101 folders. 7 of them carry no `attack_result.pt` and cannot be read: `cifar100_lira_0_1`, `cifar10_lc_0_05`, `cifar10_lc_0_1`, `cifar10_lira_0_1`, `gtsrb_lira_0_1`, `tiny_lc_0_005`, `tiny_lira_0_1`. Every readable folder is a `vit_b_16` whose bd_test labels name a single target. The table lists the in-scope models, read from `inventory.json`. Aligned is the share of bd_test entries whose recorded original label equals this project's test label at that index. The footprint columns are shares of 300 sampled triggered images against their clean twins, a pixel counting as changed above 4 of 255 levels. Consistent is the share of pixels changed on at least half the images, high for a fixed trigger and near 0 for a sample-specific one.

| model | label mode | target | bd_test images | format | aligned | pixels changed | tokens touched | consistent | mean change (levels) |
|---|---|---|---|---|---|---|---|---|---|
| `cifar10_trojannn_0_1` | all_to_one | 0 | 9000 | .png | 1.0000 | 0.114 | 0.143 | 0.114 | 6.88 |
| `gtsrb_trojannn_0_1` | all_to_one | 0 | 12570 | .png | 1.0000 | 0.114 | 0.143 | 0.114 | 9.43 |
| `tiny_trojannn_0_1` | all_to_one | 0 | 9950 | .png | 1.0000 | 0.161 | 0.386 | 0.086 | 6.78 |
| `cifar10_ssba_0_1` | all_to_one | 0 | 9000 | .png | 1.0000 | 0.547 | 0.843 | 0.467 | 8.90 |
| `gtsrb_ssba_0_1` | all_to_one | 0 | 12570 | .png | 1.0000 | 0.669 | 0.883 | 0.703 | 11.71 |
| `tiny_ssba_0_1` | all_to_one | 0 | 9950 | .png | 1.0000 | 0.706 | 0.978 | 0.905 | 11.95 |
| `cifar10_inputaware_0_1` | all_to_one | 0 | 9000 | .png | 1.0000 | 0.031 | 0.084 | 0.000 | 3.47 |
| `cifar10_lira_0_1` | no checkpoint | | | | | | | | |
| `gtsrb_lira_0_1` | no checkpoint | | | | | | | | |
| `cifar10_blind_0_1` | all_to_one | 0 | 9000 | .jpg | 1.0000 | 0.777 | 0.960 | 1.000 | 7.14 |
| `cifar10_trojannn_0_05` | all_to_one | 0 | 9000 | .png | 1.0000 | 0.114 | 0.143 | 0.114 | 6.88 |
| `gtsrb_trojannn_0_05` | all_to_one | 0 | 12570 | .png | 1.0000 | 0.114 | 0.143 | 0.114 | 9.50 |
| `tiny_trojannn_0_05` | all_to_one | 0 | 9950 | .png | 1.0000 | 0.087 | 0.148 | 0.087 | 6.20 |
| `gtsrb_inputaware_0_1` | all_to_one | 0 | 12570 | .png | 1.0000 | 0.012 | 0.032 | 0.000 | 1.56 |

## Reproduction and success bar

Clean accuracy is over the whole test set and ASR over every eligible bd_test image, both through `models.backbones.load_checkpoint` and `evaluation.metrics` with BackdoorBench's normalization (`evaluation/<folder>.json`). The leaderboard columns are BackdoorBench's own no-defense numbers (`leaderboard_vit_b_16.json`). A model is judged when its ASR is at least 0.85 and its clean accuracy is at least 0.5 of the best BackdoorBench ViT-B/16 of its dataset. The 2-point verdicts are reported against both references and drop no model.

| dataset | best BackdoorBench ViT-B/16 clean accuracy | its folder | own benign ViT clean accuracy |
|---|---|---|---|
| cifar10 | 0.9679 | `cifar10_lc_0_005` | 0.9515 |
| cifar100 | 0.8492 | `cifar100_trojannn_0_005` | 0.8103 |
| gtsrb | 0.9949 | `gtsrb_lc_0_001` | 0.9908 |
| tiny | 0.7739 | `tiny_blended_0_01` | 0.7568 |

| model | clean accuracy | leaderboard clean | ASR | leaderboard ASR | judged | within 2 points of BackdoorBench best | within 2 points of own benign |
|---|---|---|---|---|---|---|---|
| `cifar10_trojannn_0_1` | 0.9647 | 0.9646 | 0.9998 | 0.9998 | yes | yes | yes |
| `gtsrb_trojannn_0_1` | 0.9894 | 0.9895 | 0.9995 | 0.9994 | yes | yes | yes |
| `tiny_trojannn_0_1` | 0.7534 | 0.7516 | 0.9986 | 0.9986 | yes | no | yes |
| `cifar10_ssba_0_1` | 0.9617 | 0.9617 | 0.9783 | 0.9781 | yes | yes | yes |
| `gtsrb_ssba_0_1` | 0.7993 | 0.7996 | 0.9180 | 0.9176 | yes | no | no |
| `tiny_ssba_0_1` | 0.7635 | 0.7637 | 0.9928 | 0.9928 | yes | yes | yes |
| `cifar10_inputaware_0_1` | 0.9156 | 0.9165 | 0.7976 | 0.9230 | no | no | no |
| `cifar10_blind_0_1` | not evaluated | | | | | | |
| `cifar10_trojannn_0_05` | not evaluated | | | | | | |
| `gtsrb_trojannn_0_05` | not evaluated | | | | | | |
| `tiny_trojannn_0_05` | not evaluated | | | | | | |
| `gtsrb_inputaware_0_1` | not evaluated | | | | | | |

The reproduction gate compares clean accuracy and ASR with the leaderboard before any sweep (`model_job.py`). A model it fails is recorded and not swept.

| model | gate | clean gap | ASR gap | outcome |
|---|---|---|---|---|
| `cifar10_trojannn_0_1` | reproduced | 0.0001 | -0.0000 | swept |
| `gtsrb_trojannn_0_1` | reproduced | -0.0001 | 0.0001 | swept |
| `tiny_trojannn_0_1` | reproduced | 0.0018 | -0.0000 | swept |
| `cifar10_ssba_0_1` | reproduced | 0.0000 | 0.0002 | swept |
| `gtsrb_ssba_0_1` | reproduced | -0.0003 | 0.0004 | swept |
| `tiny_ssba_0_1` | reproduced | -0.0002 | -0.0000 | swept |
| `cifar10_inputaware_0_1` | not reproduced | -0.0009 | -0.1254 | stopped, not reproduced |
| `cifar10_blind_0_1` | not run | | | |
| `cifar10_trojannn_0_05` | not run | | | |
| `gtsrb_trojannn_0_05` | not run | | | |
| `tiny_trojannn_0_05` | not run | | | |
| `gtsrb_inputaware_0_1` | not run | | | |

Normalization control on `cifar10_trojannn_0_1`: with BackdoorBench's statistics clean accuracy reads 0.9647 and ASR 0.9998, with `DATASET_REGISTRY`'s CIFAR-10 statistics 0.9538 and 0.9996. The registry's statistics are the ones `experiments/wanet_cifar10_audit/measure.py` read BackdoorBench's WaNet checkpoint with, so its `results/bb_cifar10_wanet_0_1` cache was built on shifted inputs.

## Diagnosis of a failed gate

`diagnose_inputaware.py` reads a seeded subset of the saved bd_test images through BackdoorBench's own test transform on the CPU, in float32, in bfloat16 autocast and in float32 with every pixel raised half a level, which undoes on average the floor ToPILImage applies when BackdoorBench writes a float image to 8 bits.

| model | images | float32 ASR | bfloat16 ASR | half level up ASR | float32 robust accuracy | leaderboard ASR | leaderboard robust accuracy |
|---|---|---|---|---|---|---|---|
| `cifar10_inputaware_0_1` | 600 | 0.7967 | 0.7967 | 0.7983 | 0.1950 | 0.9230 | 0.0706 |

On `cifar10_inputaware_0_1` the saved images give the same ASR in every reading, so neither this project's loader, its bfloat16 inference nor the 8-bit rounding of the saved files moves it. The checkpoint holds the classifier and the 2 datasets only (`checkpoint_keys` in the record), no generator and no mask network, so the per-image triggers BackdoorBench measured its ASR on cannot be drawn again. The leaderboard number does not come from this classifier on these files, and nothing released lets it be recomputed. The cause lies outside this project's pipeline and the model is recorded as not reproduced.

## Detection per model

TPR at 1%, 5% and 10% FPR (thresholds at the clean-validation quantile) and AUROC, each probe at its adaptive rate, read from `models/<folder>.json`. As a control, the PSBD-TM AUROC must equal the fractional-PSU AUROC `cli.analyze` stored at the same rate, counted below the table.

| model | judged | method | rate | TPR at 1% FPR | TPR at 5% FPR | TPR at 10% FPR | AUROC |
|---|---|---|---|---|---|---|---|
| `cifar10_trojannn_0_1` | yes | PSBD-TM | 0.3 | 0.868 | 0.994 | 1.000 | 0.998 |
| `cifar10_trojannn_0_1` | yes | final method, minimum | both | 0.539 | 0.990 | 0.999 | 0.997 |
| `cifar10_trojannn_0_1` | yes | final method, average | both | 0.794 | 0.996 | 1.000 | 0.998 |
| `cifar10_trojannn_0_1` | yes | pre_residual_blocks_5_8 alone | 0.6 (adaptive) | 0.663 | 0.938 | 0.983 | 0.993 |
| `gtsrb_trojannn_0_1` | yes | PSBD-TM | 0.8 | 0.447 | 0.792 | 0.928 | 0.946 |
| `gtsrb_trojannn_0_1` | yes | final method, minimum | both | 0.448 | 0.536 | 0.818 | 0.934 |
| `gtsrb_trojannn_0_1` | yes | final method, average | both | 0.495 | 0.776 | 0.904 | 0.954 |
| `gtsrb_trojannn_0_1` | yes | pre_residual_blocks_5_8 alone | 0.9 (adaptive) | 0.060 | 0.216 | 0.384 | 0.828 |
| `tiny_trojannn_0_1` | yes | PSBD-TM | 0.4 | 0.864 | 0.966 | 0.987 | 0.991 |
| `tiny_trojannn_0_1` | yes | final method, minimum | both | 0.990 | 0.999 | 0.999 | 0.999 |
| `tiny_trojannn_0_1` | yes | final method, average | both | 0.917 | 0.989 | 0.999 | 0.996 |
| `tiny_trojannn_0_1` | yes | pre_residual_blocks_5_8 alone | 0.5 (adaptive) | 0.996 | 0.999 | 0.999 | 0.999 |
| `cifar10_ssba_0_1` | yes | PSBD-TM | 0.4 | 0.903 | 0.977 | 0.978 | 0.979 |
| `cifar10_ssba_0_1` | yes | final method, minimum | both | 0.580 | 0.978 | 0.979 | 0.985 |
| `cifar10_ssba_0_1` | yes | final method, average | both | 0.718 | 0.977 | 0.978 | 0.985 |
| `cifar10_ssba_0_1` | yes | pre_residual_blocks_5_8 alone | 0.6 (adaptive) | 0.653 | 0.868 | 0.931 | 0.977 |
| `gtsrb_ssba_0_1` | yes | PSBD-TM | 0.5 | 0.044 | 0.805 | 0.878 | 0.926 |
| `gtsrb_ssba_0_1` | yes | final method, minimum | both | 0.070 | 0.419 | 0.858 | 0.920 |
| `gtsrb_ssba_0_1` | yes | final method, average | both | 0.120 | 0.777 | 0.885 | 0.936 |
| `gtsrb_ssba_0_1` | yes | pre_residual_blocks_5_8 alone | 0.8 (adaptive) | 0.121 | 0.417 | 0.595 | 0.876 |
| `tiny_ssba_0_1` | yes | PSBD-TM | 0.4 | 0.990 | 0.992 | 0.993 | 0.994 |
| `tiny_ssba_0_1` | yes | final method, minimum | both | 0.992 | 0.993 | 0.993 | 0.995 |
| `tiny_ssba_0_1` | yes | final method, average | both | 0.992 | 0.993 | 0.993 | 0.995 |
| `tiny_ssba_0_1` | yes | pre_residual_blocks_5_8 alone | 0.4 (adaptive) | 0.991 | 0.993 | 0.993 | 0.996 |

The control holds on 6 of 6 swept models.

## Detection per attack

Means over the judged models of each attack (`summary.json`, `by_attack`).

| attack | judged models | method | TPR at 1% FPR | TPR at 5% FPR | TPR at 10% FPR | AUROC |
|---|---|---|---|---|---|---|
| SSBA | 3 | PSBD-TM | 0.646 | 0.925 | 0.950 | 0.966 |
| SSBA | 3 | final method, minimum | 0.547 | 0.797 | 0.943 | 0.966 |
| SSBA | 3 | final method, average | 0.610 | 0.915 | 0.952 | 0.972 |
| SSBA | 3 | pre_residual_blocks_5_8 alone | 0.588 | 0.759 | 0.840 | 0.950 |
| TrojanNN | 3 | PSBD-TM | 0.726 | 0.917 | 0.972 | 0.979 |
| TrojanNN | 3 | final method, minimum | 0.659 | 0.842 | 0.939 | 0.977 |
| TrojanNN | 3 | final method, average | 0.735 | 0.920 | 0.967 | 0.983 |
| TrojanNN | 3 | pre_residual_blocks_5_8 alone | 0.573 | 0.718 | 0.789 | 0.940 |

## Verdicts per prediction

Bars from `PREDICTIONS.md`: a family passes when every judged model has PSBD-TM AUROC of at least 0.9 and TPR at 5% FPR of at least 0.5, and fails when every judged model has AUROC below 0.75. A prediction with no judged model is untested.

| prediction | judged models | verdict | reading |
|---|---|---|---|
| T1, TrojanNN 10% passes | 3 | holds | family reads pass |
| T1, TrojanNN 5% passes | 0 | untested | |
| T2, minimum costs TrojanNN at most 0.05 TPR at 5% FPR | 3 | fails | largest cost 0.256 |
| S1, SSBA partial or fails | 3 | fails | family reads pass, TPR at 1% FPR below TrojanNN's on 1 of 3 |
| S2, minimum raises SSBA TPR at 5% FPR on most | 3 | holds | raised on 2 of 3 |
| I1, Input-Aware fails | 0 | untested | |
| I2, final method leaves Input-Aware below 0.5 TPR at 5% FPR | 0 | untested | |
| B1, Blind passes | 0 | untested | |
| LIRA, partial or fails | 0 | untested, no checkpoint | |
| R1, no patch family fails | 3 | holds | failing: none |

## Final method against PSBD-TM alone

Change in TPR from PSBD-TM alone to each fusion rule, per judged model, at the same nominal FPR. Positive means the final method detects more.

| model | minimum, 1% FPR | minimum, 5% FPR | minimum, 10% FPR | average, 1% FPR | average, 5% FPR | average, 10% FPR |
|---|---|---|---|---|---|---|
| `cifar10_trojannn_0_1` | -0.329 | -0.004 | -0.001 | -0.074 | +0.002 | +0.000 |
| `gtsrb_trojannn_0_1` | +0.001 | -0.256 | -0.110 | +0.047 | -0.016 | -0.024 |
| `tiny_trojannn_0_1` | +0.126 | +0.033 | +0.012 | +0.054 | +0.024 | +0.011 |
| `cifar10_ssba_0_1` | -0.323 | +0.000 | +0.001 | -0.186 | -0.000 | +0.000 |
| `gtsrb_ssba_0_1` | +0.026 | -0.386 | -0.021 | +0.076 | -0.028 | +0.006 |
| `tiny_ssba_0_1` | +0.001 | +0.001 | +0.000 | +0.002 | +0.001 | +0.000 |

## Wall times

Per model on the shared login A100, including time spent waiting for data loading but not for the GPU lock (`evaluation/<folder>.json` and `sweeps/<folder>.json`, `wall_seconds`).

| model | evaluation (s) | sweep of both probes and analysis (s) |
|---|---|---|
| `cifar10_trojannn_0_1` | 46 | 1011 |
| `gtsrb_trojannn_0_1` | 27 | 1357 |
| `tiny_trojannn_0_1` | 23 | 1062 |
| `cifar10_ssba_0_1` | 25 | 1019 |
| `gtsrb_ssba_0_1` | 23 | 1237 |
| `tiny_ssba_0_1` | 20 | 997 |
| `cifar10_inputaware_0_1` | 21 |  |
| `cifar10_blind_0_1` |  |  |
| `cifar10_trojannn_0_05` |  |  |
| `gtsrb_trojannn_0_05` |  |  |
| `tiny_trojannn_0_05` |  |  |
| `gtsrb_inputaware_0_1` |  |  |

<!-- results:end -->
