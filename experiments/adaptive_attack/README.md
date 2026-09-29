# Adaptive attack analysis (H25)

Analyzes evasive checkpoints trained with the hinge penalty from
`adaptive_evasion.py`. For each evasive checkpoint, computes sigma-matched
AUROC at 4 perturbation operators and compares to the non-evasive baseline.

Answers: does evasion against 1 operator transfer to others?

Result: evasion is probe-specific. On the 56 evasive checkpoints, ViT and Swin, whose
attack success stays above 0.9, the probed operator's AUROC collapses from 0.922 to 0.322
at the matched 0.6 rule, but the 3 unprobed operators still detect at mean AUROC 0.893.
Over all 120 evasive checkpoints the collapse is 0.810 to 0.314 and transfer 0.721. The 12
CIFAR-100 and Tiny TaCT evasive models are source-mapped (they read 0.00 clean source-class
accuracy) and none of them keeps attack success above 0.9, so the 56 hold no model the
ledger excludes as source-mapped. Rerun on CPU on 2026-09-29 into a scratch copy, the saved
`results/adaptive_attacker_analysis.json` holding an earlier 30-row ViT run:

    PYTHONPATH=. .venv/bin/python experiments/adaptive_attack/analyze.py \
        --output scratch/stale_numbers/adaptive_attacker_analysis.json

The paper's adaptive section reads a narrower set, the 14 ViT evasive cells above 0.9
(`\AdaptiveVitProbedBase` 0.966, `\AdaptiveVitProbedEvade` 0.233, `\AdaptiveVitTransfer`
0.893 in `paper/headline.tex`).

Runs on CPU using cached PSBD sweep data. No GPU needed.

    python experiments/adaptive_attack/analyze.py
    python experiments/adaptive_attack/analyze.py --architecture vit --dataset cifar100

Hypothesis doc: `docs/hypothesis/H25-adaptive-attacker.md`
