# Whether SAM helps PSBD on ViT, matched combination for combination against Adam

## Question

Does training the victim with sharpness-aware minimization help PSBD detect the
backdoor on ViT and Swin, and does SAM amplify the backdoor the way its paper
(`literature/sam-poisoned-detection-zhang-arxiv2024/source/`,
Zhang et al., arXiv 2411.11525) claims.

[H6](../../docs/hypothesis/H6-sam-improves-detectability.md) already dropped a naive
Adam against SAM aggregate as confounded: the SAM checkpoints swept so far were not
the same architecture, dataset, attack and poison rate as the Adam checkpoints they
were compared against, and matching on those 4 fields flipped the sign. This
experiment builds that matched comparison directly instead of estimating it from a
coverage note.

## Method

`measure.py` scans `checkpoints/` for every (architecture, dataset, attack, poison
rate) combination that trained both an Adam checkpoint and at least 1 SAM
checkpoint, over `badnet_a2o`, `blend`, `bpp`, `lf` and `wanet`. `badnet_a2a` is
excluded on purpose: it is the atypical inverted attack that made up 13 of the 18
matched combinations behind the correction note on H6, so it cannot settle this on
its own either.

For every combination it reads ASR and clean accuracy from each side's
`checkpoints/<folder>/metrics.json`, and, wherever `cli.sweep` and `cli.analyze`
have already reached both the Adam and the SAM checkpoint, the token-mask placement
at the attention input (`before_attention_norm_token_mask`) and the dropout
placement after the residual add (`post_residual`) AUROC and TPR at the q0.10 and
q0.20 false-positive budgets, at the deployable adaptive rate. Placement values are
read through `cli.compare_detectors.psbd_values`, the same function every paper
table uses, never recomputed from the raw score cache.

Per rho, the paired AUROC difference (SAM minus Adam) is bootstrapped with 5000
resamples over the combinations where both placements were swept on both sides, the
same common-coverage discipline `scripts/paper/tab_headline.py` applies to its own
3-way comparison. Output: `results/_experiments/sam_reading/sam_reading.json`.

The SAM paper's own amplification metrics, `top2_tac` (their "backdoor effect"),
`silhouette` and `clean_intra_class_variance`, are already computed in
`experiments/sam_backdoor_effect/measure.py` and folded in wherever they are on
disk, with their coverage stated plainly since it is far narrower than the PSBD
grid: ViT only, CIFAR-10 only, poison rate 0.1 only, `badnet_a2o`, `blend`, `bpp`
and `lf` (no `wanet`).

`scripts/paper/tab_sam.py` reads `sam_reading.json` and writes `paper/tables/sam.tex`
and its macro sidecar, 1 row per rho with Adam as the reference row.

## Table

`results/_experiments/sam_reading/sam_reading.json` is the run of 2026-09-11, 24 matched
pairs per rho, all ViT, all CIFAR-10 or CIFAR-100. The PSBD sweep has since reached ViT and
Swin on all 4 datasets at rho 0.1, so `measure.py` was rerun on CPU on 2026-09-29 against
the current caches and written beside it as
`results/_experiments/sam_reading/sam_reading_2026-09-29.json`:

```bash
PYTHONPATH=. .venv/bin/python -c "import experiments.sam_reading.measure as m; \
    m.OUTPUT_PATH = 'results/_experiments/sam_reading/sam_reading_2026-09-29.json'; m.main()"
```

AUROC at the adaptive 0.8 rule, fractional PSU. The Adam row is the 112 Adam checkpoints
that swept both placements, and each SAM row is the SAM side of its matched pairs:

| Rho | Pairs | ASR | CA | PSBD token mask AUROC | PSBD token mask TPR at 10% | PSBD residual dropout AUROC | PSBD token mask delta vs Adam | 95% CI | PSBD residual dropout delta | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| Adam | 112 | 0.976 | 0.892 | 0.967 | 0.930 | 0.885 | -- | -- | -- | -- |
| 0.05 | 26 | 0.985 | 0.903 | 0.974 | 0.937 | 0.835 | +0.034 | [+0.002, +0.082] | -0.038 | [-0.115, +0.020] |
| 0.1 | 109 | 0.974 | 0.902 | 0.955 | 0.909 | 0.882 | -0.012 | [-0.032, +0.007] | -0.001 | [-0.039, +0.036] |
| 0.15 | 26 | 0.985 | 0.905 | 0.938 | 0.849 | 0.803 | -0.002 | [-0.053, +0.047] | -0.071 | [-0.155, -0.001] |
| 0.2 | 26 | 0.984 | 0.903 | 0.958 | 0.891 | 0.811 | +0.019 | [-0.025, +0.070] | -0.062 | [-0.142, -0.002] |

Rho 0.1 pools ViT and Swin over all 4 datasets and 5 attacks (109 pairs). Rho 0.05, 0.15
and 0.2 are still ViT on CIFAR-10 and CIFAR-100 (26 pairs). ASR and clean accuracy are read
from `checkpoints/<folder>/metrics.json`, which the 2026-09-29 audit
(`docs/audits/2026-09-29-experiment-audit.md`) found stale against the ledger's `args.json`
ASR. No attack in this grid is TaCT, so no pair holds a source-mapped model.

The 2026-09-11 run read the token-mask delta as +0.034, +0.035, +0.010 and +0.023 over 24
pairs at rho 0.05 to 0.2, with the rho 0.1 interval [+0.002, +0.079] the only 1 excluding 0.
On 109 pairs the rho 0.1 delta is -0.012 and its interval crosses 0.

The amplification metrics, where they exist (ViT, CIFAR-10, rate 0.1, `badnet_a2o`,
`blend`, `bpp`, `lf`, n=4), repeat the earlier finding: `top2_tac` falls at low rho and
rises at high rho (mean delta -0.397 at rho 0.05, +0.710 at 0.15, +2.252 at 0.2) while
`silhouette` barely moves (at most 0.007 in absolute value), so the backdoor amplifies
without the clean and triggered features separating any further.

## Conclusion

SAM does not help PSBD-TM, and it hurts the placement the original PSBD paper published at
high rho. On the 109-pair pool of rho 0.1, the only rho with both architectures and all 4
datasets, the token-mask delta is -0.012 (CI [-0.032, +0.007]) and the residual-dropout
delta -0.001. The token-mask gain of +0.034 at rho 0.05 excludes 0 on 26 ViT CIFAR pairs,
and the residual-dropout loss of -0.062 to -0.071 at rho 0.15 and 0.2 excludes 0 on the same
kind of pairs. The earlier conclusion, a small rho-independent token-mask gain with its
tightest interval at rho 0.1, does not survive the larger pool at that rho.

## Reproduce

```bash
PYTHONPATH=. python experiments/sam_reading/measure.py
PYTHONPATH=. python scripts/paper/tab_sam.py --paper-dir paper
```
