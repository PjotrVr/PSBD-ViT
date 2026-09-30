# Cached reads behind the literature memo

## Question

The literature memo (`docs/why-psbd-works-literature.md`) turns every published and folk explanation of why PSBD works into a prediction on our models. 5 of those predictions can be checked from results already on disk, with no new forward pass. These scripts are those checks. They read `results/<folder>/psbd_metrics.json` and `results/<folder>/detectors/<name>_metrics.json` for the clearing cells of `scripts/paper/_common.py`, pick the rate with `select_rate_adaptively` at 0.8 and read AUROC at the 0.25 quantile of fractional PSU. That is the same population and the same canon as the paper. On the panel of 2026-09-30 the scripts reproduce PSBD-TM 0.951 and PSBD-RD 0.876 over 56 models (`detectors_by_attack.py`), the values of `\HeadlineAurocAdaptive` and `\PublishedAurocAdaptive`, which confirms it.

## Scripts

| script | what it reads | memo section |
|---|---|---|
| `target_share.py` | share of shifted clean predictions that land on the target class, per placement and attack | clean predictions collapsing onto the target (P6, L10) |
| `attractor_corr.py` | Spearman correlation between that share and AUROC, per placement | the same, as a correlation |
| `detectors_by_attack.py` | mean AUROC per attack for PSBD-TM, PSBD-RD and every ported detector | detectors per attack category |
| `badnet_pm.py` | clean shift and target share under `gain_scale` at 15 times | IBD-PSC's feature-norm theorem (L14) |
| `embed_check.py` | shift of clean and triggered inputs under `after_embedding_token_mask` | Doan et al.'s input patch dropping (L18) |

## Running

Each script runs on CPU in under a minute from the repository root, with `PYTHONPATH=. .venv/bin/python experiments/literature_checks/<script>.py`. They print their tables to stdout, and the memo quotes those tables.

## Status

These scripts are research reads, written for the memo on 2026-09-29 against commit `307db69` and rerun on 2026-09-30 on the 56-model panel, when `detectors_by_attack.py` gained an LC column for `vit_gtsrb_lc_0_05_tl1_adv`. They do not feed any generated macro. When the panel changes (the TaCT retrains of `docs/runs/2026-09-24-tact-multisource.md`), rerun them and update the memo tables from their output.
