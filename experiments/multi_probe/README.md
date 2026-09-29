# Multi-probe PSBD defense (H41)

Computes the min-rank combined score across all available perturbation
operators for each evasive checkpoint. The min-rank rule flags a sample when
ANY probe's PSU rank falls below the Bonferroni-corrected threshold.

Answers: can multi-probe PSBD recover detection after the adaptive attacker
collapses a single operator?

Result: multi-probe with 4 operators recovers AUROC from 0.322 (evaded) to
0.951, with 48/56 checkpoints above 0.90. Diminishing returns at k=3.

The recovery figures are the 56 evasive checkpoints with attack success above 0.9 in
`results/adaptive_defender_analysis.json` (2026-08-17), ViT and Swin on CIFAR-100 and Tiny,
at the matched 0.6 rule. They reproduce from that record, and the source-mapped TaCT evasive
models are not among the 56. The experiment has not been re-run on the current caches.

Runs on CPU using cached PSBD sweep data. No GPU needed.

    python experiments/multi_probe/analyze.py
    python experiments/multi_probe/analyze.py --architecture vit --dataset cifar100

Hypothesis doc: `docs/hypothesis/H41-multi-probe-defense.md`
