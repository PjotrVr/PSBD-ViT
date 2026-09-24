# Why token masking separates patch triggers

## Question

PSBD-TM (`token_mask` at `before_attention_norm`) and PSBD-RD (dropout at `post_residual`) both
change about 87% of clean predictions at their adaptive rates. On patch triggers they part ways.
PSBD-TM changes 4% of triggered BadNets predictions and PSBD-RD changes 61%, while on global
triggers the 2 placements agree. This experiment tests 4 hypotheses about why, each stated before
the measurement and judged by it.

1. A patch trigger's evidence lives in its own tokens of the residual stream. Masking at the
   attention input never touches the stream, so a masked token still carries its content to the
   next block, and the class token only needs to read it in a few late blocks. The triggered
   prediction survives unless the trigger tokens are masked in nearly every block that routes
   them, which at rate p happens with probability about p^m.
2. Dropout on the residual stream corrupts the trigger tokens' stored content in every block, so
   the triggered prediction breaks as easily as a clean one.
3. A global trigger is present in every token, so it stays legible from any subset of tokens and
   survives either operator.
4. TaCT's trigger fires only with source class content, so it inherits the fragility of the
   content decision.

## Method

Inference only, on the clearing panel of `scripts/paper/_common.py` (`load_coverage` and
`clearing_cells`, 65 ViT-B/16 models over CIFAR-10, CIFAR-100, GTSRB and Tiny). Each model gets
the first 256 triggered rows of its PSBD analysis split (`build_psbd_loaders_from_checkpoint`,
seed `PSBD_SPLIT_SEED`, the eval ASR set of CLAUDE.md) and the clean image behind each row. TaCT's
eval set holds only its source class 1, which caps it at 87 pairs on CIFAR-100 and 42 on Tiny.
The trigger's tokens come from `trigger_tokens` in `experiments/residual_stream_mechanism/
activation_patching.py`: 4 tokens for every patch trigger except Tiny's, which covers 1.

Triggered survival is the share of triggered images still predicted as the target, among those
the unperturbed model sends there. Clean survival is the share of clean images still predicted
as their unperturbed class. Every reading also records clean accuracy and the share of clean
images predicted as the target, because a heavily masked model can collapse onto 1 default class
and on some models that class is the target. The adaptive rates come from each model's
`psbd_metrics.json` through `cli.compare_detectors.psbd_rate(block, "adaptive")`.

The deterministic masks reuse `TokenMask`'s semantics at the `ln_1` pre-hook through
`models.positions.plug_dropout`. A chosen token is zeroed in every chosen block, CLS is never
masked, and survivors are rescaled by the inverted-dropout factor of the masked share. At this
position the rescale is inert because LayerNorm removes each token's own scale, which
`tests/test_why_token_masking_works.py` checks. The same test holds `RecordingTokenMask` to the
library operator bit for bit under a fixed seed, alone and plugged into a ViT.

| part | models | what runs |
|---|---|---|
| A | 23 patch models | trigger tokens masked in all 12 blocks, in blocks 1 to 4, 5 to 8 or 9 to 12, in the last or first m blocks for m in 1, 2, 4, 8 and 12, and 3 draws of as many random non-trigger tokens masked in all 12 blocks |
| B | 23 patch models | PSBD-TM at the adaptive rate, 10 passes, recording per pass and image in how many blocks every trigger token was masked, in how many of blocks 9 to 12, and the fewest trigger tokens visible in any of blocks 9 to 12 |
| C | 23 patch models | PSBD-RD at the adaptive rate, 10 passes, on all tokens as in the library, on the trigger positions only, on every position except the trigger's, and on as many random positions as the trigger has (3 draws) |
| D | 24 models, 1 per attack and dataset | a fixed random subset of patch tokens kept visible at the attention input in all 12 blocks, keep fraction f in 1.0, 0.6, 0.3 and 0.1, 3 subsets per f |

D takes the 5% model of each attack and dataset, or the rate that cleared when 5% did not. No
Label-Consistent model clears the panel, so the patch set is 12 BadNets and 11 TaCT models. The
scale (256 pairs, 10 passes, 3 draws, the reduced m and f grids, D on 24 models) was set to keep
GPU use light on the shared login node.

## Commands

```bash
source .venv/bin/activate
# smoke run on 1 model, written outside results/
PYTHONPATH=. python experiments/why_token_masking_works/measure.py \
    --folders vit_gtsrb_badnet_a2o_0_05 --pairs 64 --output-root "$SCRATCH"
# the full run, resumable, then the tables below from the saved records
PYTHONPATH=. python experiments/why_token_masking_works/measure.py
PYTHONPATH=. python experiments/why_token_masking_works/measure.py --summarize-only
python -m pytest tests/test_why_token_masking_works.py
```

The full run took 11 min 8 s on the login node A100 for 39 models, about 30 s per patch model and
5 s per D-only model, under a 6 GB memory cap. Output is 1 JSON per model under
`results/_experiments/why_token_masking_works/` and the aggregate `summary.json` there.

## TaCT split by source class behavior

8 of the 11 TaCT panel models misclassify their whole source class with no trigger present. The
paired clean images score 0.00 accuracy on all 8, and 7 of them send 64% to 74% of the clean
source images to the target (`vit_cifar10_tact_0_1` sends them to class 9 instead). On these
models the triggered prediction is mostly carried by the source content and the patch adds
little. They are reported as `tact_source_mapped`. The 3 models that classify their source class
correctly (`vit_cifar10_tact_0_01`, `vit_cifar10_tact_0_05` and `vit_gtsrb_tact_0_05`, paired
clean accuracy 0.95 to 1.00) are `tact_trigger_conditional`. Any TaCT number in the paper averages
the 2 behaviors.

## Trigger tokens masked deterministically (A)

Triggered survival, mean over models. Clean survival stays between 0.96 and 1.00 in every row.

| masked blocks | BadNets (12) | TaCT conditional (3) | TaCT source mapped (8) |
|---|---|---|---|
| none | 1.000 | 1.000 | 1.000 |
| all 12 | 0.002 | 0.004 | 0.658 |
| 1 to 4 | 1.000 | 1.000 | 1.000 |
| 5 to 8 | 0.788 | 0.992 | 0.920 |
| 9 to 12 | 0.200 | 0.004 | 0.783 |
| last 1 | 0.934 | 0.339 | 0.998 |
| last 2 | 0.820 | 0.337 | 0.991 |
| last 4 | 0.200 | 0.004 | 0.783 |
| last 8 | 0.002 | 0.004 | 0.658 |
| first 8 | 0.719 | 0.978 | 0.897 |
| random tokens, all 12 | 1.000 | 0.999 | 1.000 |
| clean images sent to the target, all 12 | 0.002 | 0.004 | 0.643 |

Masking the trigger tokens in every block removes the BadNets backdoor completely, and masking
as many random tokens does nothing. Masking them in blocks 1 to 4 does nothing either, so a token
masked at the attention input keeps its content in the stream and delivers it later. The class
token reads a BadNets trigger over blocks 5 to 12 with most of the weight in 9 to 12, and a
trigger-conditional TaCT trigger only in the last 1 to 4 blocks.

## PSBD-TM with its masks recorded (B)

Triggered and clean survival by the number of blocks among 9 to 12 in which every trigger token
was masked, pooled over passes and images, with the count of (pass, image) pairs in brackets.

| late blocks with the trigger fully masked | BadNets, 4 tokens (9) | BadNets, 1 token (3) | TaCT conditional (3) | clean, BadNets models |
|---|---|---|---|---|
| 0 | 0.987 (14390) | 1.000 (517) | 0.338 (3733) | 0.107 (14962) |
| 1 | 0.965 (5895) | 0.999 (1800) | 0.128 (2253) | 0.115 (7852) |
| 2 | 0.917 (2027) | 0.996 (2913) | 0.024 (1109) | 0.116 (4863) |
| 3 | 0.748 (634) | 0.899 (1990) | 0.000 (491) | 0.112 (2442) |
| 4 | 0.216 (74) | 0.282 (450) | 0.000 (64) | 0.085 (601) |
| overall | 0.966 | 0.930 | 0.205 | 0.110 |

Clean survival under PSBD-TM does not depend on whether the trigger positions were masked, which
is the expected control. Triggered BadNets survival stays above 0.9 until every trigger token is
masked in 3 or 4 of the late blocks. At least 1 trigger token is masked in 11 or 12 of the 12
blocks in most passes, and that barely matters: 1 visible trigger token in 1 late block keeps most
triggered predictions. The event that breaks the prediction, all m trigger tokens masked in all 4
late blocks, has probability p^(4m). For Tiny at p 0.5 and m 1 that is 0.0625 and the observed
share is 450 of 7670 passes (0.059). For 4 token triggers it is 74 of 23020 passes (0.003).

TaCT source-mapped models keep 0.372 of triggered predictions against 0.071 of clean ones, and
their survival falls only from 0.420 to 0.215 across the same 5 rows, since the trigger is not
what carries their prediction.

## PSBD-RD restricted by position (C)

Survival at the adaptive PSBD-RD rate (0.05 to 0.09), mean over models.

| dropout applied to | BadNets triggered | BadNets clean | TaCT conditional triggered | TaCT conditional clean | TaCT source mapped triggered |
|---|---|---|---|---|---|
| all tokens (the library) | 0.398 | 0.113 | 0.056 | 0.030 | 0.066 |
| trigger positions only | 0.387 | 0.995 | 0.116 | 0.998 | 0.831 |
| every position but the trigger's | 0.664 | 0.119 | 0.107 | 0.027 | 0.246 |
| as many random positions | 1.000 | 0.992 | 1.000 | 0.997 | 1.000 |

On BadNets, dropout on the 4 trigger positions alone breaks as many triggered predictions as the
library's dropout on all 197, and dropout on 4 random positions breaks none. Per model the trigger
only variant breaks at least as much as the library on 8 of 12 BadNets models. The rest of the
stream also contributes, since dropout everywhere except the trigger still breaks a third of
triggered predictions, and the 2 parts do not add up to the whole. PSBD-RD does not break
triggered BadNets predictions as easily as clean ones (0.398 kept against 0.113), and
`vit_gtsrb_badnet_a2o_0_1` barely moves under it at all (0.94 kept).

## Visible token subsets (D)

Retention relative to f = 1.0. Excess retention subtracts the share of clean images sent to the
target from the ASR at each f, which removes the collapse onto a default class. Clean retention is
clean accuracy retention.

| attack (models) | excess f 0.6 | excess f 0.3 | excess f 0.1 | clean f 0.6 | clean f 0.3 | clean f 0.1 |
|---|---|---|---|---|---|---|
| BadNets (4) | 0.651 | 0.365 | 0.000 | 0.581 | 0.186 | 0.015 |
| TaCT conditional (2) | 0.837 | 0.258 | 0.000 | 0.710 | 0.052 | 0.000 |
| Blend (4) | 0.997 | 0.950 | 0.400 | 0.705 | 0.256 | 0.033 |
| BPP (4) | 0.983 | 0.942 | 0.618 | 0.746 | 0.201 | 0.044 |
| LF (4) | 0.999 | 0.903 | 0.426 | 0.699 | 0.196 | 0.040 |
| SIG (1) | 0.999 | 0.910 | 0.010 | 0.927 | 0.519 | 0.129 |
| WaNet (3) | 0.650 | 0.183 | 0.050 | 0.563 | 0.113 | 0.006 |

WaNet's raw ASR retention is 1.04 to 1.05 at every f, because the masked models send 38% to 94%
of clean images to the target as well. On the patch models (BadNets and TaCT together) the subset
draws split cleanly on whether any trigger token stayed visible.

| f | trigger visible, excess retention (draws) | trigger hidden, excess retention (draws) |
|---|---|---|
| 0.6 | 0.997 (22) | 0.000 (2) |
| 0.3 | 0.904 (14) | 0.000 (10) |
| 0.1 | none drawn | 0.000 (24) |

The hidden rows are exactly 0 because triggered and clean predictions become identical. A token
masked at the attention input of every block never writes into any other token, so the trigger
has no path to the class token and the rest of the triggered image equals its clean twin.

## Verdict per hypothesis

Hypothesis 1 is supported. Masking the trigger tokens in all 12 blocks leaves 0.002 of triggered
BadNets predictions and the same number of random tokens leaves 1.000. Masking only in blocks 1
to 4 leaves 1.000, so the masked token keeps its content in the stream. Under the real operator
triggered survival falls from 0.987 to 0.216 as the number of late blocks with every trigger token
masked rises from 0 to 4, while clean survival stays flat near 0.11. 2 refinements matter. The
exponent is m times the number of late blocks that read the trigger (about 4), and a single
visible trigger token in a single late block is enough to keep most predictions.

Hypothesis 2 is supported for the mean and weaker per model. Dropout on the trigger positions
alone reproduces the library's triggered breakage (0.387 against 0.398 kept) and random positions
reproduce none of it (1.000). Dropout on everything except the trigger also breaks a third of
triggered predictions (0.664), the trigger only variant matches the library on 8 of 12 models, and
triggered predictions still survive PSBD-RD more often than clean ones (0.398 against 0.113), so
"as easily as a clean one" overstates it.

Hypothesis 3 is supported for Blend, BPP, LF and SIG and fails for WaNet. At f 0.3 the first 4
keep 0.90 to 0.95 of their excess ASR while clean accuracy keeps 0.20 to 0.52. At f 0.1 they
degrade as well (0.01 to 0.62) though mostly above clean (0.03 to 0.13). WaNet's warp keeps 0.183
of its excess ASR at f 0.3 against 0.113 clean, which is close to the clean path. Its raw
ASR only looks robust because the masked model collapses onto the target class.

Hypothesis 4 holds but the TaCT panel mostly measures something else. On 8 of 11 TaCT models the
source class is mapped away with no trigger, so their triggered prediction is content. Trigger
only dropout leaves 0.831 of it while content-side dropout leaves 0.246. On the 3 trigger
conditional models the trigger is necessary (0.004 with it masked everywhere) and read only in
the last blocks, and PSBD-TM breaks the triggered prediction even with the trigger visible in
every late block (0.338 kept against 0.987 for BadNets). Under PSBD-RD either half breaks it
(trigger only 0.116, everything else 0.107). That is the content fragility the hypothesis
predicts, measured on 3 models.

## Limits

The pooled B rows weight each model by its pair count, and TaCT on CIFAR-100 and Tiny has 87 and
42 pairs. The trigger-conditional TaCT group holds 3 models, 2 of them in D. D covers 24 models,
1 per attack and dataset. The rates in B and C are each model's own adaptive rates, which range
from 0.3 to 0.8 for PSBD-TM and 0.05 to 0.09 for PSBD-RD.
