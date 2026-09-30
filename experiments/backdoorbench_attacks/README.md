# PSBD-TM and the final method on BackdoorBench's own ViT-B/16 checkpoints

The ViT panel covers BadNets, Blend, BPP, LF, WaNet, TaCT and 1 LC model, all trained in this repository on 1 recipe. `backdoor_bench_checkpoints/` holds BackdoorBench's own ViT-B/16 checkpoints, trained on BackdoorBench's recipe, including attack families the panel has never met. This experiment asks whether PSBD-TM and the final method (PSBD-TM fused with `pre_residual_blocks_5_8`) detect those families, and so whether the method holds on models trained outside this repository. The predictions were committed before any sweep ran (`PREDICTIONS.md`, commit `a2dccc6`).

## Scope

Before any sweep ran on 2026-09-30, the user cut the scope to attack families the panel lacks on a few models, read with the final method's own 2 probes. The queue (`queue.py`) runs TrojanNN, SSBA, Input-Aware, LIRA and Blind at 10% and then TrojanNN at 5%. PSBD-RD, every competitor detector, the 1% rate and BackdoorBench's versions of the panel's own attacks are out of scope. LIRA's folders are empty, so LIRA is listed and not tested.

## Loading a BackdoorBench checkpoint

BackdoorBench builds `vit_b_16` as `torch.nn.Sequential(Resize((224, 224)), vit_b_16)` with a fresh head (`third_party/BackdoorBench/utils/aggregate_block/model_trainer_generate.py`, lines 123 to 132, at the pinned commit `f02e353`, `Resize` imported from `torchvision.transforms` on line 12). `models.backbones.build_vit` returns the same module tree, so the checkpoint's state dict loads strictly and the 32-pixel input stored as `img_size` in `attack_result.pt` is resized to 224 inside the model, as in BackdoorBench. Its test transform is `Resize((h, w))`, `ToTensor` and `get_dataset_normalization(dataset)` (`utils/aggregate_block/dataset_and_transform_generate.py`, lines 134 to 146 and 81 to 104). Both resizes are bilinear. The model's resize only upsamples, where torchvision's antialias flag has no effect, which I checked numerically on 32 and 64 pixel inputs (maximum difference at float precision).

2 things did not match and are fixed here. BackdoorBench's CIFAR-10 standard deviations are (0.247, 0.243, 0.261) where `DATASET_REGISTRY` carries (0.2023, 0.1994, 0.2010). CIFAR-100's mean differs in the 4th decimal. `data.backdoorbench.BACKDOORBENCH_NORMALIZATION` copies BackdoorBench's table and every BackdoorBench read goes through it. The normalization control in the results measures what the registry's statistics cost. Blind stores its triggered test images as JPEG (the record's `save_file_format`) where the other attacks store PNG. They are read as stored.

`data.backdoorbench.build_psbd_loaders_from_backdoorbench` builds the standard PSBD split for such a folder: the same permutation `data.splits` draws from (test set size, seed 0), its first 2000 indices as the clean validation set and the rest as the analysis pool, clean images from this project's raw test set and triggered images from the checkpoint's bd_test folder, paired by the original test index BackdoorBench uses as the file name. A bd_test entry whose recorded original label disagrees with the clean test label at its index is dropped and listed in the split manifest (`misaligned_bd_test_indices`). BackdoorBench's WaNet folders have such entries. None of the in-scope folders has one. `cli.sweep` and `cli.analyze` read a results folder named `bb_<folder>` through this builder, with no new flag, and `tests/test_backdoorbench_split.py` covers the split, the pairing, the normalization and the misalignment rule.

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
<!-- results:end -->
