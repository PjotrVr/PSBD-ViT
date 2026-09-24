# Multi-source TaCT retraining

TaCT on the panel used 1 source class (class 1) and 8 of its 11 clearing models turned out to map that whole class to the target with no trigger. A poison rate at the source class's share of the training set poisons every source image, so the model has no clean source image to keep its own label on, and the coverage ledger now marks those cells `source_mapped` and keeps them off the panel. The genuine TaCT cells are `vit_cifar10_tact_0_01`, `vit_cifar10_tact_0_05` and `vit_gtsrb_tact_0_05`. These runs retrain TaCT over several source classes so that the poisoned images are 0.2 of the source pool, the ratio of the original TaCT paper (2% of CIFAR-10 on a source class holding 10%).

The runs test a hypothesis, so nothing is retried and no recipe is tuned. A run that diverges fails its training job and the `afterok` dependency cancels its downstream job. A run that trains but misses the ASR bar, or comes out source mapped anyway, stops at the gate of its downstream job before any sweep time is spent, and that outcome is the answer for its configuration.

## Source classes per configuration

Source classes are 1..k with the target at class 0. k is the smallest count whose summed share of the training split is at least 5 times the poison rate, counted on the training split each dataset loads. Every other argument is the TaCT panel recipe from `checkpoints/vit_cifar10_tact_0_05/args.json`: ViT-B/16, 15 epochs of Adam at the constant learning rate, patch size 3, cover rate 0.01 drawn from the non-source non-target classes, target 0 and seed 0. The table comes from the generator's planning pass, which builds the poison and cover draw and the PSBD backdoor split on the CPU without loading a model.

| folder | dataset | poison rate | k | source share of train | poisoned | poisoned share of source pool | cover | ASR set (PSBD backdoor split) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| vit_cifar100_tact_0_01_src5 | cifar100 | 0.01 | 5 | 0.0500 | 500 | 0.200 | 500 | 420 over 5 classes |
| vit_cifar100_tact_0_05_src25 | cifar100 | 0.05 | 25 | 0.2500 | 2500 | 0.200 | 500 | 2012 over 25 classes |
| vit_tiny_tact_0_01_src10 | tiny | 0.01 | 10 | 0.0500 | 1000 | 0.200 | 1000 | 406 over 10 classes |
| vit_tiny_tact_0_05_src50 | tiny | 0.05 | 50 | 0.2500 | 5000 | 0.200 | 1000 | 2026 over 50 classes |
| vit_cifar10_tact_0_1_src5 | cifar10 | 0.1 | 5 | 0.5000 | 5000 | 0.200 | 500 | 4020 over 5 classes |
| vit_gtsrb_tact_0_1_src12 | gtsrb | 0.1 | 12 | 0.5034 | 2664 | 0.199 | 266 | 5504 over 12 classes |

GTSRB's classes are unequal, so classes 1..12 hold 50.3% of its 26640 training images and the poisoned share lands at 0.199. The single-source panel cells scored 87 backdoor images on CIFAR-100 and 42 on Tiny, so the new ASR sets are between 4.8 and 48 times larger.

## Code changes the runs needed

`--attack-override` could not carry a tuple. The command line hands `attacks.apply_config_overrides` a string, and a tuple-typed field kept the raw string `"1,2,3"`, so it now splits a string on commas for a field declared as a tuple and casts each element to the declared type (`tests/test_attacks.py`). `cli.evaluate` rebuilt every attack from its defaults and ignored `attack_config_overrides` in `args.json`, which would have scored a multi-source run on source class 1 alone. `evaluation.metrics.evaluate_checkpoint` now applies the recorded overrides (`tests/test_evaluation_metrics.py`).

`scripts/coverage_ledger.py`'s `is_panel_folder` admits the `_src{k}` folders unchanged. `resolve_one_per_attack` now skips a source-mapped cell as it skips a diverged one, so a rerun that lands below the bar leaves 1 candidate for its slot rather than raising on an ambiguous pair (`tests/test_coverage_ledger.py`). The paper generators already read only cells whose ASR class is `clears`, so a clearing rerun takes the slot and the old source-mapped cell drops out.

## Jobs and their dependencies

`pbs/generate_tact_multisource_jobs.py` writes 6 training jobs and 8 downstream jobs into `pbs/vit_tact_multisource/`, with logs under `logs/vit_tact_multisource/`. A training job runs `cli.train_backdoor` and then `cli.evaluate` and exits nonzero if either fails. Each downstream job depends on its own training job with `afterok`, so a failed run cancels 1 downstream job and nothing else. The 2 existing panel cells that clear the bar but were never swept, `vit_gtsrb_tact_0_01_cos` and `vit_gtsrb_lc_0_05_tl1_adv`, get a downstream job with no dependency.

A downstream job first runs the gate (`python -m pbs.generate_tact_multisource_jobs --gate <folder>`), which applies the ledger's verdicts to the sidecar: the ASR bar 0.85 and divergence against the benign reference. It then sweeps the recommended placement, whose first call builds the clean baseline, and runs the gate again, which now also reads clean source-class accuracy off that baseline and stops a source-mapped model after 1 placement. The rest of the 27 basis placements follow at the declared rates with 3 forward passes and `--skip-existing`, then `cli.analyze` and the 11 competitor detectors through `cli.baselines` in the 4 cost groups of `pbs/generate_detector_jobs.py`.

| job | work | depends on | estimate (hours) | walltime |
|---|---|---|---:|---:|
| tms_t1 | train vit_cifar100_tact_0_01_src5 | none | 1.57 | 4 |
| tms_t2 | train vit_cifar100_tact_0_05_src25 | none | 1.57 | 4 |
| tms_t3 | train vit_tiny_tact_0_01_src10 | none | 2.95 | 6 |
| tms_t4 | train vit_tiny_tact_0_05_src50 | none | 2.95 | 6 |
| tms_t5 | train vit_cifar10_tact_0_1_src5 | none | 1.57 | 4 |
| tms_t6 | train vit_gtsrb_tact_0_1_src12 | none | 0.93 | 4 |
| tms_d1 | sweep and detectors vit_cifar100_tact_0_01_src5 | tms_t1 | 2.37 | 6 |
| tms_d2 | sweep and detectors vit_cifar100_tact_0_05_src25 | tms_t2 | 2.63 | 6 |
| tms_d3 | sweep and detectors vit_tiny_tact_0_01_src10 | tms_t3 | 2.37 | 6 |
| tms_d4 | sweep and detectors vit_tiny_tact_0_05_src50 | tms_t4 | 2.63 | 6 |
| tms_d5 | sweep and detectors vit_cifar10_tact_0_1_src5 | tms_t5 | 2.95 | 6 |
| tms_d6 | sweep and detectors vit_gtsrb_tact_0_1_src12 | tms_t6 | 3.62 | 8 |
| tms_d7 | sweep and detectors vit_gtsrb_tact_0_01_cos | none | 2.83 | 6 |
| tms_d8 | sweep and detectors vit_gtsrb_lc_0_05_tl1_adv | none | 4.35 | 9 |

## Cost estimate

The estimate is 35.3 A100-hours if every run clears, 11.5 for training and 23.8 downstream. Training uses the median span of every plain Adam ViT sidecar per dataset (84 minutes on CIFAR-10 and CIFAR-100, 167 on Tiny, 46 on GTSRB) plus 10 minutes for the final ASR pass and `cli.evaluate`. The sweep uses 0.021 minutes per basis rate per 1000 PSBD inputs, measured from the cache timestamps of 3 complete basis cells (`vit_cifar100_badnet_a2o_0_05` at 108 minutes, `vit_cifar100_tact_0_05` at 59 and `vit_gtsrb_badnet_a2o_0_05` at 139), with each cell's real input count from its planned split. The detectors use the per-input seconds of `pbs/generate_detector_jobs.py`, whose past GPU jobs finished at or under their estimate.

The estimate is under the 40 hour limit with little room. The flat budget of `pbs/generate_basis_jobs.py` (0.5 minutes per rate whatever the split size) would put the sweeps at 21.2 hours instead of 13.5 and the total at 43.0. A run that misses the bar costs its training time and a few minutes of gate.

## Commands and job IDs

The planning pass, the job files and the submission ran from the main checkout on branch `rewrite`.

```bash
PYTHONPATH=. python pbs/generate_tact_multisource_jobs.py --dry-run
PYTHONPATH=. python pbs/generate_tact_multisource_jobs.py
bash pbs/vit_tact_multisource/submit_all.sh
```

The jobs went in on 2026-09-24 with the IDs below, the downstream ones held on their training job by `afterok`.
The ID suffix is the PBS server `x3000c0s25b0n0.hsn.hpc.srce.hr` for every job.

| job | PBS job ID |
|---|---|
| tms_t1 | 1069851 |
| tms_t2 | 1069852 |
| tms_t3 | 1069853 |
| tms_t4 | 1069854 |
| tms_t5 | 1069855 |
| tms_t6 | 1069856 |
| tms_d1 | 1069857 |
| tms_d2 | 1069858 |
| tms_d3 | 1069859 |
| tms_d4 | 1069860 |
| tms_d5 | 1069861 |
| tms_d6 | 1069862 |
| tms_d7 | 1069863 |
| tms_d8 | 1069864 |
