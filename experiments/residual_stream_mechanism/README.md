# LayerNorm absorption of noise and masking at `before_attention_norm`, and what it does not explain

## Question

The placement ranking is an empirical search result. This directory asks what the residual
stream is actually doing that makes one injection point better than another, at the PSBD
paper's own operating point (clean-validation shift ratio **0.8**).

A ViT block is a read-write cycle, `x -> ln_1 -> attn -> +x -> ln_2 -> mlp -> +x`, and a
placement is a point in it. Some points are immediately followed by a LayerNorm and some are
already normalized. That is not a relabelling: LayerNorm subtracts the per-token mean and
divides by the per-token standard deviation.

## The prediction, which has no free parameters

Isotropic additive noise **inflates** the per-token standard deviation, so dividing by it puts
part of the perturbation back. Zeroing a token or a channel removes content that no rescaling
can restore. Therefore:

> at a position where a LayerNorm follows, additive noise is partly undone and masking is not.
> At a position that is already normalized, both survive equally.

This is H23's secondary prediction, recorded as **still open** since the operator study:
*"the gap between `gaussian` and `channel_mask` is larger at `before_attention_norm` than at
`post_residual` ... not yet separated from position effects."*

## Result: confirmed, directly measured

Survival ratio, relative change after the following normalization divided by relative change
before it. 1.000 means the perturbation passed through untouched. 2 checkpoints, 2
datasets, blocks 1/4/6/9/12, injected sizes 0.1/0.2/0.4.

| position | LayerNorm follows | gaussian | token_mask | channel_mask |
|---|---|---:|---:|---:|
| `before_attention_norm` | **yes** | **0.808 / 0.835** | 0.998 / 1.005 | 0.966 / 0.971 |
| `before_attention` | no | 1.000 | 1.000 | 1.000 |
| `before_mlp_norm` | **yes** | **0.883 / 0.862** | 0.970 / 0.972 | 0.998 / 0.964 |
| `before_mlp` | no | 1.000 | 1.000 | 1.000 |

(`vit_gtsrb_badnet_a2o_0_1` / `vit_cifar100_blend_0_05`, the means over 15 rows per cell of
`results/<folder>/layernorm_absorption.json`, written 2026-09-09.)

**LayerNorm removes 12 to 19 percent of additive noise and 0 to 4 percent of masking.** At
positions that are already normalized the distinction vanishes, as predicted. The variance
renormalization the prediction names accounts for only a small part of the gaussian
attenuation (it predicts survival 0.996, 0.985 and 0.945 at injected sizes 0.1, 0.2 and 0.4,
against a measured 0.84, 0.82 and 0.77 on GTSRB BadNets). The rest is a near-constant
linear attenuation (`docs/why-psbd-works-theory.md`).

The gaussian column reproduces on a CPU rerun (2026-09-29 audit,
`docs/audits/2026-09-29-experiment-audit.md`), and the mask columns do not reproduce to the
printed digits. The rerun on `vit_gtsrb_badnet_a2o_0_1` read token_mask 1.007 and 0.980 and
channel_mask 1.039 and 1.023 at the 2 normalized positions, up to 0.07 from the table. The
mask generator is seeded with `hash(operator)`, which Python salts per process, so every
run draws different masks. The mask survival is therefore known to about 0.07, which still
separates it from gaussian at the attention input and leaves the verdict unchanged.

## The chain from mechanism to measured detection

Each step is measured on the paper panel, the 54 models successful at the 2-point
clean-accuracy bar that carry both headline placements, at a matched clean shift ratio of
0.8, read from each model's `psbd_metrics.json` by

```
PYTHONPATH=. .venv/bin/python scratch/stale_numbers/panel_auroc.py --rule 0.8 \
    --placements before_attention_norm_gaussian before_mlp_gaussian \
    --pairs before_attention_norm_token_mask:before_attention_norm_gaussian \
            before_mlp_norm_token_mask:before_mlp_gaussian
```

The same command with `--old-panel` reproduces the 67-cell values this section quoted before
2026-09-29, given in brackets.

1. **Absorption.** `ln_1` absorbs ~19% of gaussian and ~0% of token masking (above).
2. **Operator gap, position-dependent.** Token mask minus gaussian is **+0.145 AUROC**
   (CI [+0.088, +0.212], n=54, [+0.177 on 67]) at `before_attention_norm` at the matched 0.8
   rule, where `ln_1` follows immediately. The paper states the same comparison with the
   opposite sign at the canonical matched 0.6 rule as gaussian minus token mask
   (`\GaussianMinusTokenMaskAttentionNorm`), which the command above gives with `--rule 0.6`
   as **-0.188** (CI [-0.243, -0.139], n=54). Token masking before the MLP norm (`before_mlp_norm`,
   before `ln_2`) minus gaussian at `before_mlp` (after `ln_2`) is **-0.079** (CI [-0.140,
   -0.012], [-0.065]) at 0.8. The paper's `\GaussianMinusTokenMaskMlp` is the same pair at
   0.6 with the opposite sign, **+0.062** (CI [+0.028, +0.097]) on the 54. That pair differs in site as
   well as operator, gaussian after the norm against token masking before it, so its sign
   says nothing about the 2 operators at 1 position and cannot show that the operator gap
   reverses.
3. **Same operator, different position.** `gaussian` scores **0.813** at
   `before_attention_norm` and **0.924** at `before_mlp` (0.758 and 0.896 on 67): **+0.111
   apart by position alone**, with **8 of 54** cells inverted (AUROC < 0.5) against **2 of
   54** (13 and 4 of 67).
4. **Consequence for the position-versus-operator question.** Holding position fixed at
   `before_attention_norm`, the range over token_mask, channel_mask and gaussian is 0.145 with
   gaussian and 0.050 without it. Holding the operator fixed at `token_mask` across the 5
   positions swept on every model, the position range is 0.113. So position over operator is
   **0.78x** with gaussian and **2.24x** without (0.66x and 1.53x on 67). **Position dominates
   once gaussian is set aside.**

Step 4 matters beyond bookkeeping. An earlier reading of this panel reported the ratio as
0.58x and concluded that the operator axis dominates. It does not. A single operator failing
at a single position moves the ratio from 2.24x to 0.78x, a factor of 2.9. At the adaptive
0.8 rule the same ratio on the 54 is 1.24 with gaussian, the quantity the paper reports as
`\PositionOverOperatorRatioAdaptive`.

## What this does and does not explain

**Absorption does not explain the operator gap.** The measured absorption is close to a
linear attenuation: gaussian survival at `before_attention_norm` is 0.84, 0.82 and 0.77 at
injected sizes 0.1, 0.2 and 0.4 on GTSRB BadNets and 0.85, 0.84 and 0.81 on CIFAR-100 Blend
(`results/<folder>/layernorm_absorption.json`). A linear attenuation is equivalent to a
smaller rate, and the matched-shift rule compensates for it by injecting more, so it cancels
at matched shift and cannot produce the -0.188 gap read there on the 54. The derivation is
in `docs/why-psbd-works-theory.md` (section "Existing claims the derivations contradict",
item 3), and the operator gap needs a different account. The same document notes that the
paper's attention input minus MLP input gap for token masking
(`\AttentionInputMinusMlpInputTokenMask`, +0.114 on the 54 at the matched 0.6 rule) compares 2 sites whose measured token-mask
absorption is equal (-0.002 and 0.029), so absorption does not explain that position gap
either.

**What the absorption measurement does establish:** a LayerNorm passes a mask through close
to intact and attenuates additive noise by 12 to 19 percent at matched injected size, and at
positions that are already normalized both survive equally. That is H23's secondary
prediction about the norm, confirmed as a statement about injected size only.

**Open:** why `token_mask` beats `channel_mask` (both survive normalization equally, so
absorption is silent on the token-versus-channel question, which is the trigger-locality
account tested separately). Why WaNet prefers residual-adjacent placements. Why confining the
probe to blocks 5 to 8 helps `pre_residual` (+0.048, CI [+0.021, +0.076]) and hurts
`before_attention_norm_token_mask` (-0.071, CI [-0.112, -0.034]), both at the matched 0.6 rule
on the 54 models (the quantities of `\BandFiveEightMinusAllResidual` and
`\BandFiveEightMinusAllInputSide` in `paper/headline.tex`).

**Limits.** Survival rests on 2 checkpoints. No experiment here removes the LayerNorm and
measures detection. Absorption is a property of the architecture, so it holds on a benign
checkpoint too and is not by itself a backdoor signal.

## Reproduce

```
PYTHONPATH=. python experiments/residual_stream_mechanism/layernorm_absorption.py \
    --checkpoint-folder vit_gtsrb_badnet_a2o_0_1 vit_cifar100_blend_0_05
```

CPU only, seconds per checkpoint, no GPU and no training.
