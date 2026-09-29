# Operator and position ranking at matched disturbance (H17)

## Question

Dropout on the residual stream, the PSBD paper's ConvNet placement, collapses at 1%
poisoning. This experiment read it at `pre_residual` (dropout before both residual adds).
The canonical PSBD-RD, `PUBLISHED_PLACEMENT` in `defenses/decision.py`, is `post_residual`
(dropout after both adds), and the current numbers below give both. Is that a limit of prediction-shift detection, or a bad operating
point? The 7 scripts here answer it from 1 cached surface, so no 2 of them are
allowed to disagree about what a cell means.

Every comparison is at a matched clean-validation shift ratio (sigma) rather than
at a shared dropout rate. Comparing at a shared rate would only measure which
operator perturbs hardest. Every score is one-sided: low PSU means poisoned, and a
value below 0.5 is printed as the failure it is rather than flipped.

## The 7 scripts

| script | what it produces |
| --- | --- |
| `build_surface.py` | the `(folder, placement, rate) -> (sigma, AUROC)` table every other script slices, cached to `scratch/surface.json` |
| `analyze_surface.py` | the defender-legal configuration search, ranked by mean AUROC per poison rate with the benign control alongside |
| `head_to_head.py` | the published configuration against the best defender-legal one, per checkpoint |
| `sigma_target.py` | which sigma target the rate rule should aim at, scored by the AUROC each target would have selected |
| `lowrate_diagnosis.py` | the full (placement, rate) surface for 1 checkpoint, to separate a broken premise from a broken rate rule |
| `operator_ranking.py` | operators ranked per attack at matched sigma, with an unreachable operating point reported as `--` rather than dropped |
| `full_stack_1pct.py` | what every accumulated change buys at 1% poisoning and 5% FPR, both deployable and ROC |

`build_surface.py` writes its cache into `scratch/` on purpose. It is a derived
table of about 2600 cells, regenerable from `results/<folder>/psbd/` in 1 command,
so it is data rather than a result.

## Running it

Repo root has to be the working directory, because the cache paths are relative to
it.

    PYTHONPATH=. python experiments/operator_ranking/build_surface.py <folders>
    PYTHONPATH=. python experiments/operator_ranking/analyze_surface.py
    PYTHONPATH=. python experiments/operator_ranking/head_to_head.py
    PYTHONPATH=. python experiments/operator_ranking/sigma_target.py
    PYTHONPATH=. python experiments/operator_ranking/operator_ranking.py
    PYTHONPATH=. python experiments/operator_ranking/full_stack_1pct.py

All of them are CPU only and read cached per-pass probabilities. No GPU job is
needed once the sweep has run.

## Finding

The low-poison-rate failure is an operating point, not a limit. On the current panel, the
4 CIFAR-100 models at 1% poisoning that clear the ASR bar (BadNets, Blend, BPP and LF) read
PSBD-TM (`token_mask` at `before_attention_norm`) at 0.960 mean AUROC at the adaptive 0.8
rule, against 0.861 for PSBD-RD (`post_residual`) and 0.843 for `pre_residual`. The paired
gain over PSBD-RD is +0.100, CI [-0.015, +0.214], `\CifarOneZeroZeroOnePercentGain` in
`paper/headline.tex`, and +0.117, CI [-0.002, +0.236], over `pre_residual`. At the matched
0.6 rule the gains are +0.058 and +0.100. With 4 models neither interval excludes 0. Read
with

    PYTHONPATH=. .venv/bin/python scratch/stale_numbers/panel_auroc.py

restricted to those 4 folders. The version of this finding before 2026-09-29 read a
2600-cell surface in `scratch/surface.json` built on the earlier panel and quoted a gain of
0.162 at the swept rate and 0.166 at matched shift ratio, which the current panel does not
reproduce.

The `gain_scale` at `mlp_norm_out` figure of 0.258 is WITHDRAWN (`docs/audit-2026-09-07.md`):
that arm was read at shift ratio 0.95 to 0.98 against a baseline at 0.65 to 0.76, and at
matched shift ratio over the full 48-cell panel of the time it gained -0.007. On the 4
current CIFAR-100 1% models at the matched 0.6 rule it reads 0.787, below PSBD-TM by
+0.119, CI [+0.057, +0.198].

`head_to_head.py` showed where the gain comes from, and the current panel agrees:
`badnet_a2o` on CIFAR-100 at 1% reads 0.744 under PSBD-RD, 0.815 under `pre_residual` and
0.988 under PSBD-TM at the adaptive rule. The earlier surface read it at 0.297 against 0.839,
an inverted detector becoming a working one. The benign control stayed at 0.494 against
0.504, so the candidate configuration reads more than confidence. All-to-all still
fails under both, which is H5 and a separate problem.

Hypothesis doc: `docs/hypothesis/H17-low-poison-rate-is-a-placement-artifact.md`.
Published tables: `docs/results/operator-position-ranking.md`,
`docs/results/detection-operating-points.md`.
