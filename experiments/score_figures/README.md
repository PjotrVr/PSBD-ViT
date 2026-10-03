# Score distributions and low-FPR ROC curves for any combination of PSBD probes

A mean TPR at a budget hides where the threshold lands. For every model of a set this tool draws the clean-validation and triggered score distributions with the thresholds at 1%, 5%, 10% and 20% FPR drawn in. It also draws the ROC on a log FPR axis from 0.001 to 1. Both cover single probes and fused combinations of them. It reads the stage-1 caches only and runs on the CPU in minutes. `config.py` is the only file to edit.

## Configuration

`config.py` holds 6 plain values for the scores, then 1 entry per further figure type (below).

- `PROBES` maps a short name to a cached placement, once per architecture. On ViT-B/16 `tm` is PSBD-TM (`before_attention_norm_token_mask`), `rd` is PSBD-RD (`post_residual`), `band` is residual dropout before both adds in blocks 5 to 8 and `late` the same in blocks 9 to 12. On Swin-S `band` is blocks 17 to 24 and `middle` is blocks 9 to 16.
- `COMBINATIONS` is a list of lists of those names. A 1-element list draws that probe's own fractional PSU (rule `psu`). A longer list is fused under every rule in `FUSION_RULES`, its first probe leading.
- `FUSION_RULES` names the fusion rules. `min` is the plain minimum of the clean-validation percentiles. `weighted` gives the first probe the share `WEIGHTED_FIRST_SHARE` of the budget and splits the rest equally among the others, with score $\min_i r_i / s_i$, where $r_i$ is probe $i$'s percentile within its own clean-validation scores and $s_i$ its share. With 2 probes and a share of 0.9 it is `weighted_0.9_0.1` of `experiments/cache_readouts/fusion_rules.py`.
- `MODEL_SETS` names the sets. `vit_panel` is `experiments.cache_readouts.shared.load_model_set("panel")`, the successful ViT models that carry both headline placements. `swin_panel` is `scripts.paper.tab_swin.swin_cells`, the Swin-S cells that pass the ViT panel rule (`scripts.paper._common.swin_coverage`). `backdoorbench` is every record under `results/_experiments/backdoorbench_attacks/models/` that carries detection rates, read from `results/bb_<folder>/psbd/`. A set may also be an explicit list of `results/` folder names.
- `BUDGETS` are the false-positive budgets, each a quantile of the clean-validation scores.
- `DPI`, `ROC_POINTS` and the histogram settings fix the figures' size and resolution.

The Swin `band` is blocks 17 to 24. The 1st pre-registered Swin partner was the middle third (blocks 9 to 16) and its predictions failed. The 2nd and last attempt took blocks 17 to 24 from the depth of WaNet's backdoor direction and held under the min rule (`experiments/final_method/README.md`, sections "Swin second probe, attempt 1" and "attempt 2", read by `swin_late_readout.py`). That README's opening paragraph still names blocks 9 to 16 as the Swin partner, which the 2 sections below it overturn. `cli.sweep --block-range` counts Swin-S blocks 1 to 24 across all 4 stages, so blocks 17 to 24 are the last 8 blocks of the network.

## Method

Every probe is read at its adaptive rate, the smallest cached rate whose clean-validation shift ratio reaches 0.8 (`experiments.cache_readouts.shared.choose_rate` over `validation_shift_by_rate`). Its fractional PSU on the validation, clean and triggered splits comes from `defenses.scores.psu_ratio_from_cache`. A probe whose placement is not cached, is not defined for the architecture or whose ladder never reaches the target is skipped for that model, and every combination that needs it is skipped with the reason in `numbers.json`. No other rate or placement is substituted.

A fused score ranks each probe's PSU within that probe's own clean-validation scores (`defenses.scores.to_rank`), so no fitting touches triggered data. Every score, single or fused, is thresholded at a quantile of its own clean-validation distribution. TPR is read on the triggered split and the realized FPR on the clean test split paired to the same images (`defenses.decision.pair_clean_to_backdoor`), through `experiments.cache_readouts.shared.evaluate_scores`, the path every earlier readout used. The ROC applies that same threshold rule at every budget of a log grid. A low score means poisoned throughout.

## Commands

```bash
.venv/bin/python -m experiments.score_figures.make                       # every set in config.MODEL_SETS
.venv/bin/python -m experiments.score_figures.make --set backdoorbench
.venv/bin/python -m experiments.score_figures.make --set vit_panel --models vit_tiny_wanet_0_05,vit_tiny_wanet_0_1
.venv/bin/python -m experiments.score_figures.make --models swin_tiny_wanet_0_05   # an explicit set, written under custom/
.venv/bin/python -m experiments.score_figures.make --figures scatter,evaders   # only these figure kinds, scores kept
.venv/bin/python -m experiments.score_figures.check                      # hold the numbers to earlier records
```

`--figures` names the figure kinds to draw, `scores` (the histograms, ROC curves and numbers every other kind reads) and the 7 kinds below. The default draws all of them, and a kind switched off in `config.py` is skipped.

`--models` with `--set` refreshes only the named models of that set. The set's index, CSV files and mean figures are then rebuilt from every `numbers.json` present under the set.

## Outputs

Everything is written under `results/_experiments/score_figures/<set>/`.

| path | content |
|---|---|
| `<model>/numbers.json` | every probe's placement, adaptive rate, validation shift and cached ladder or skip reason, then per combination and rule the AUROC and, per budget, the threshold, TPR, realized FPR and tie share at the threshold, the ROC (TPR and realized FPR on the grid `roc_fpr_grid`) and the histogram (bin edges, validation and triggered counts, counts above the drawn range) |
| `<model>/roc.png` | TPR against the clean-validation budget on a log axis, 1 color per combination and a dashed line for the weighted rule |
| `<model>/hist_<combination>_<rule>.png` | clean validation against triggered scores with a log count axis and the 4 thresholds, the combination's names joined by `-` |
| `index.json` | the configuration, the commit, every model's rates and skipped probes and the mean ROC per combination and rule |
| `models.csv` | 1 row per model, combination and rule with TPR and realized FPR at every budget and AUROC, skips included with their reason |
| `means.csv` | the same metrics averaged over models, for the whole set, per dataset and per attack, each row with its model count |
| `mean_roc.png`, `mean_roc_by_attack/<attack>.png` | the mean ROC over the models that score each combination |

The PNG files are not tracked (`.gitignore`) and come back with 1 run of `make.py`. The JSON and CSV files are the versioned record.

## Further figure types

Each kind has 1 entry in `config.py` with an `enabled` flag and its options. Every PNG has a JSON sidecar of the same name beside it holding the numbers it shows. A probe is read at the rate its model's `numbers.json` records, so a model must be scored before any of these figures is drawn for it.

| config entry | output | what it shows | sidecar |
|---|---|---|---|
| `SCATTER` | `<set>/<model>/scatter_<anchor>-<partner>.png` | each image's clean-validation percentile under the anchor (x) and the partner (y), log axes, clean validation against triggered, with the region each rule flags at the budget shaded, plain min left and weighted right | the fused threshold, the cutoff on each axis, the share of each split flagged by the anchor only, the partner only or both, and the share at percentile 0 (drawn spread just below the dotted floor at 1 over twice the validation size) |
| `CONFIDENCE` | `<set>/<model>/confidence_<probe>.png` | starting confidence $P_c$ against absolute PSU (with the bound PSU = $P_c$) and against fractional PSU, with the threshold at the budget | AUROC, TPR, realized FPR and threshold of both forms, the median confidence of each split and of the flagged validation images, and the Spearman correlation of PSU with confidence on validation |
| `SHIFT_LADDER` | `<set>/shift_vs_rate.png` | the share of perturbed clean and triggered predictions that change and the mean probability left on the original class, along every cached rate, averaged per attack, 1 row per probe, with the median adaptive rate | per probe, attack and rate the 4 means and the model count, and every model's adaptive rate |
| `FLIP_TARGETS` | `<set>/<model>/flip_targets_<probe>.png` | the class each changed clean-validation prediction moves to (`defenses.scores.shift_target_histogram`), with the target class marked | the counts, the target's and the top class's share of the changes, the uniform share and the validation prediction counts |
| `EVADERS` | `_across_sets/evader_frontier.png` | clean-accuracy loss against the benign reference against PSBD-TM AUROC for every adaptive attacker in `results/_experiments/final_method/adaptive_attackers.json`, marker by attacker kind, filled when ASR clears the bar | every point and, per attacker kind, the counts that clear the ASR bar, stay within the clean-accuracy bar and push PSBD-TM below 0.5 |
| `NEGATIVE_PSU` | `_across_sets/negative_psu.png` | per model the share of clean validation whose fractional PSU is below 0 against TPR at the budget, colored by set | every point with its threshold, and per set the Spearman correlation and the models whose threshold is below 0 |
| `THRESHOLD_TRANSFER` | `_across_sets/threshold_transfer.png` | each budget against the FPR realized on the paired clean test split, 1 point per model and budget, with the diagonal | per set, curve and budget the mean, median and largest realized FPR and the share above 1.5 times the budget |

`check.py` also holds every confidence sidecar of the ViT panel to `results/_experiments/psu_vs_confidence/division.json`, which measured absolute against fractional PSU on the same caches.

## Inputs

The tool reads only caches that `cli.sweep` wrote.

- `results/<folder>/psbd/baseline_{validation,clean,backdoor}.pt`, the unperturbed probabilities and argmax of each split
- `results/<folder>/psbd/<placement>/rate_<rate>_{validation,clean,backdoor}.pt`, the per-pass probabilities at each cached rate
- `results/<folder>/psbd/split_manifest.json`, the index lists that pair clean rows to triggered rows
- `results/bb_<folder>/psbd/`, the same layout for BackdoorBench's checkpoints, and `results/_experiments/backdoorbench_attacks/models/*.json` for their dataset, attack and rate
- `checkpoints/<folder>/args.json` for the target class the flip histogram marks
- `results/_experiments/final_method/adaptive_attackers.json` for the evader frontier
- the coverage ledger `results/coverage/coverage.json` and the Swin ledger built in memory by `scripts.paper._common.swin_coverage`, for the 2 panels

## Reproduced records

`check.py` compares every TPR, realized FPR and AUROC model by model with 3 earlier records and fails above a difference of 1e-6. They are `results/_experiments/cache_readouts/fusion_rules_panel.json` for PSBD-TM and its fusions with the middle band and PSBD-RD on the ViT panel, `results/_experiments/backdoorbench_attacks/models/*.json` for PSBD-TM and the min rule on BackdoorBench's checkpoints and `results/_experiments/final_method/fusion_swin_panel_late.json` for PSBD-TM and the min rule with blocks 17 to 24 on Swin-S. The prototype that preceded this tool interpolated its TPR along the ROC curve, so its readings at a budget differ from the exact quantile threshold in the 3rd decimal on some models.
