# Placement rationale

Superseded on 2026-09-30: the panel is 56 models, see paper/headline.tex. The tables below were read on 54.

PSBD detects a backdoor by perturbing the model several times and flagging an input whose prediction barely moves. On a ResNet the original paper puts dropout in 1 obvious place. A vision transformer offers many places, and several kinds of perturbation fit each of them, so this repository swept a grid of them. This page explains that grid as what it was: a sequence of hypotheses, each forced by the result before it. For every site it says what tensor is touched and through which code path, which hypothesis made the site worth trying, which hypothesis doc under `docs/hypothesis/` records the test, and what the current measurement says. Every number sits in the generated placement ledger at the end of the page, written by `python scripts/detector_doc_results.py`. The prose describes the ledger's rows without retyping their values, so it cannot go stale when the panel grows.

`notebooks/placements-and-operators.ipynb` is the executable companion: it runs every operator on a toy tensor and records the tensor shape at every position of real ViT-B/16, Swin-S and ResNet-18 architectures. Each step's comparison is drawn there from the same cached files.

## Terms

| term | meaning |
|---|---|
| position | where a perturbation is injected, a named tensor boundary inside every block, such as `before_attention_norm` |
| operator | what is injected there, such as dropout, a token mask or Gaussian noise |
| placement | the pair, named `<position>_<operator>`, which is also the cache directory `results/<folder>/psbd/<placement>/`. A bare position name means the operator is dropout |
| band | an optional restriction of a placement to a contiguous range of blocks, such as `_blocks_5_8` |
| rate | the operator's strength, the dropout or masking probability, or the noise scale for Gaussian noise |
| residual stream | the running sum a transformer block reads from and adds to, $x \leftarrow x + \text{branch}(x)$ |
| branch | the attention or MLP sub-computation of a block, whose output is added to the stream |
| PSBD-TM | `before_attention_norm_token_mask`, whole tokens zeroed at the input of every attention branch, the placement this project recommends (`defenses.decision.RECOMMENDED_PLACEMENT`) |
| PSBD-RD | `post_residual`, dropout on the stream right after both residual adds of every block, this project's adaptation of the PSBD paper's ResNet site (`PUBLISHED_PLACEMENT`) |
| the panel | the ViT-B/16 models whose attack clears the success bar in `results/coverage/coverage.json`, with diverged and source-mapped TaCT models excluded by `scripts.paper._common.excluded_folders` |
| paired gain | the mean over models of a placement's AUROC minus PSBD-RD's AUROC on the same model, with a 95% bootstrap interval over models |

## The statistic and the 2 rate rules

PSBD (Li et al., arXiv 2406.05826) runs the input once unperturbed and $k$ times perturbed. This project scores the fractional drop in the confidence of the unperturbed prediction, `defenses.scores.psu_ratio_from_cache`:

$$
\phi_{ratio}(x) = 1 - \frac{1}{k} \sum_{i=1}^{k} \frac{P_c(x; p, \theta'_i)}{P_c(x; \theta)}, \qquad c = \arg\max_{c'} P_{c'}(x; \theta)
$$

| symbol | meaning |
|---|---|
| $x$ | the input image |
| $\theta$ | the model's parameters, unperturbed |
| $\theta'_i$ | the model on perturbed pass $i$, a fresh random mask or noise draw |
| $p$ | the perturbation rate |
| $k$ | the number of perturbed passes, 3 on the panel (`cli.sweep.DEFAULT_FORWARD_PASSES`) |
| $P_c(\cdot)$ | the softmax probability of class $c$ |
| $c$ | the unperturbed predicted class |
| $\phi_{ratio}(x)$ | the fractional prediction shift uncertainty, low meaning poisoned |

A clean input's class evidence is destroyed by the perturbation, so its confidence falls and $\phi_{ratio}$ is large. A triggered input rides the backdoor, the most robust association the model learned, so its confidence barely moves and $\phi_{ratio}$ is small. The threshold is the 0.25 quantile of the clean validation scores and AUROC is one-sided, both as `docs/detectors/README.md` defines them.

The rate is chosen from clean data by the shift ratio, the share of perturbed passes whose predicted class differs from the unperturbed one, `defenses.scores.shift_ratio`:

$$
\sigma(D) = \frac{1}{k \, |D|} \sum_{x \in D} \sum_{i=1}^{k} \mathbb{I}\big( Y(x; \theta'_i) \ne Y(x; \theta) \big)
$$

| symbol | meaning |
|---|---|
| $D$ | the 2000-image clean validation split |
| $Y(x; \cdot)$ | the predicted class under the unperturbed or perturbed model |
| $\mathbb{I}$ | the indicator function |
| $\sigma(D)$ | the shift ratio at 1 rate |

The **adaptive rule**, `defenses.decision.select_rate_adaptively`, takes the smallest swept rate whose $\sigma$ reaches 0.8, the PSBD paper's target. It is the deployable rule, since it reads only clean data. The **matched rule**, `select_rate_at_matched_shift`, takes the rate whose $\sigma$ is closest to 0.6. It exists to compare placements with each other: 2 placements read at the same nominal rate disturb the model by very different amounts, and the matched rule compares them at the same measured disturbance instead. Step 2 below is the result that forced it.

## How a probe attaches

`models.positions.plug_dropout(model, architecture, position_names, dropout_factory, rate, block_range)` reads the table of `PositionSpec(submodule_name, hook_type, scope)` entries in `POSITION_REGISTRY[architecture]` and attaches a freshly constructed operator module at every target. `resolve_targets` finds the targets. For `scope="block"` they are every transformer block (or every `BasicBlock` on ResNet), for `scope="model"` 1 module under the network and for `scope="root"` the whole `Sequential(Resize, network)` wrapper. `block_range` restricts block-scope targets to a 1-indexed inclusive span. The hook types are:

- `pre`, a forward pre-hook that perturbs the module's first positional input. When the same tensor is passed several times, as ViT passes $x$ as query, key and value to `self_attention`, 1 mask is drawn and shared.
- `post`, a forward hook that perturbs the module's output tensor.
- `residual`, a per-instance replacement of the block's `forward` (`attach_residual_wrapper`), needed because the stream right after the attention add is a local variable inside `EncoderBlock.forward` that never crosses a module boundary.
- `attention`, a per-instance replacement of the attention module's `forward` by `defenses.operators.masked_attention_forward`, needed because the per-head outputs live inside `F.multi_head_attention_forward`.

`unplug_dropout` removes every attachment by its handle, and the replacements mutate no weight, so the model returns exactly to its loaded state. The model's own dropout modules are never switched on, because a trained dropout's inverted scaling was calibrated against the next layer's weights and reusing it would mix the model's regularization into the probe. `activate_model_dropout` is the single deliberate exception, kept to study that mixing.

## The operators

Every operator in `defenses/operators.py` is an `nn.Module` built from a rate that maps a tensor to a tensor of the same shape, is the identity in eval mode or at rate 0, and keeps the expected activation unchanged: a removing operator divides survivors by the keep probability $1 - p$, and an additive one adds mean-0 noise. ViT tensors are (batch, tokens, channels), 197 tokens of 768 channels on ViT-B/16 with token 0 the class token. Swin tensors are (batch, height, width, channels). The token operators flatten their 2 spatial axes.

| operator | class | what it removes or adds | unit | swept on the panel |
|---|---|---|---|---|
| `dropout` | `nn.Dropout` | each entry independently with probability $p$ | 1 coordinate of 1 token | yes |
| `token_mask` | `TokenMask` | whole tokens, every channel, the class token always kept | 1 token | yes |
| `channel_mask` | `GroupChannelMask` | whole channels, shared by every token of a sample | 1 feature, or 1 hidden neuron at `mlp_neurons` | yes |
| `head_mask` | `HeadMask` | whole attention heads, (batch, heads, tokens, dim) | 1 head, only at `attention_heads` | yes |
| `droppath` | `DropPath` | a whole branch output for a whole sample | 1 branch of 1 block, only at the 2 branch outputs | yes |
| `gaussian` | `GaussianNoise` | adds $\epsilon \sim \mathcal{N}(0, (p \cdot s)^2)$, $s$ the sample's own standard deviation | nothing removed | yes |
| `gain_scale` | `GainScale` | multiplies by $1 + p$, deterministic | the whole tensor | yes, at LayerNorm outputs |
| `scale_up` | `ScaleUp` | multiplies pixels by $1 + p$ and clips, deterministic | the image | yes, at `input_pixels` |
| `rademacher` | `RademacherNoise` | adds $\pm p \cdot s$ signs | nothing removed | no |
| `token_substitute` | `TokenSubstitute` | replaces whole tokens with other tokens of the same sample | 1 token | no |
| `token_block_mask` | `TokenBlockMask` | a square patch-aligned block of tokens | a contiguous region | no |
| `fixed_head_mask` | `FixedHeadMask` | 1 named head, deterministic, no rescaling | 1 head | a leave-one-out tool, not a placement |

`OPERATOR_POSITIONS` restricts `head_mask` to `attention_heads` and `droppath` to the 2 branch-output positions. `FORBIDDEN_OPERATOR_POSITIONS` refuses the structured operators at `input_pixels`, where a rank-4 image would be read as Swin's layout, and `token_mask` at `final_norm_out`, where ViT's head reads only the protected class token and the probe would be inert. `DETERMINISTIC_OPERATORS` marks `gain_scale` and `scale_up`, for which every pass is identical and $k = 1$ is exact.

## The positions on ViT, Swin and ResNet

The ViT-B/16 block computes $x' = x + \text{dropout}(\text{MSA}(\text{ln\_1}(x)))$ and then $y = x' + \text{mlp}(\text{ln\_2}(x'))$, where `mlp` applies `Linear(768, 3072)` with GELU and dropout, then `Linear(3072, 768)` with dropout. Every dropout in these models is at probability 0. `notebooks/figures/vit_block_positions.png` draws every position on it.

| position | hook | ViT module | tensor, ViT-B/16 | Swin-S module | in words |
|---|---|---|---|---|---|
| `input_pixels` | pre, root | the wrapper | (batch, 3, H, W) at native size | the wrapper | the normalized image before the Resize |
| `after_embedding` | pre, model | `encoder.dropout` | (batch, 197, 768), once | `features.0`, post | the patch embeddings before the first block |
| `before_attention_norm` | pre | `ln_1` | (batch, 197, 768) | `norm1` | the attention branch's input, before its LayerNorm |
| `attention_norm_out` | post | `ln_1` | (batch, 197, 768) | `norm1` | the same tensor after the LayerNorm |
| `before_attention` | pre | `self_attention` | (batch, 197, 768) | `attn` | the attention module's own input, after the LayerNorm |
| `attention_heads` | attention wrapper | `self_attention` | (batch, 12, 197, 64) | not registered | the per-head outputs before `out_proj` mixes them |
| `before_attention_residual` | post | `dropout` | (batch, 197, 768) | `attn`, post | the attention branch output, before the add |
| `after_attention_residual` | residual wrapper | the block | (batch, 197, 768) | the block | the stream between the attention add and its 2 consumers |
| `before_mlp_norm` | pre | `ln_2` | (batch, 197, 768) | `norm2` | the MLP branch's input, before its LayerNorm |
| `mlp_norm_out` | post | `ln_2` | (batch, 197, 768) | `norm2` | the same tensor after the LayerNorm |
| `before_mlp` | pre | `mlp` | (batch, 197, 768) | `mlp` | the MLP's own input, after the LayerNorm |
| `mlp_neurons` | pre | `mlp.3` | (batch, 197, 3072) | not registered | the hidden layer after GELU, 1 channel per neuron |
| `before_mlp_residual` | post | `mlp` | (batch, 197, 768) | `mlp`, post | the MLP branch output, before the add |
| `after_mlp_residual` | post | the block | (batch, 197, 768) | the block | the stream leaving the block |
| `final_norm_out` | post, model | `encoder.ln` | (batch, 197, 768), once | `norm` | the stream after the final LayerNorm, what the head reads |

On Swin-S the tensors are (batch, height, width, channels), 56 by 56 by 96 in the first stage down to 7 by 7 by 768 in the last, over 24 blocks in stages of 2, 2, 18 and 2. The 1 asymmetry is `before_attention_residual`: Swin calls a single `stochastic_depth` instance for both branches, so a hook on it cannot tell which branch called it, and the port hooks the output of `attn` instead, just before stochastic depth sees it. `DROPOUT_CONFIGS` names 3 combinations: `pre_residual` (both branch outputs), `post_residual` (both stream points after the adds) and `both_sublayer_inputs` (both LayerNorm inputs).

ResNet-18's `BasicBlock` has 1 residual add per block, so `RESNET_POSITIONS` has 1 site, `post_residual`, realized by `_resnet_post_residual_forward`: dropout on `bn2(conv2(...)) + identity`, before the final ReLU, exactly where the PSBD paper places it. The 2 ViT stream names are aliases for it, so a command written for ViT runs unchanged on the ResNet control.

## How to read the steps

The 14 numbered sections below are the sequence. Each step states the question, the hypothesis that made the site or operator worth a sweep, how it was measured and what the ledger says now. The hypothesis docs record the measurement that first settled each question, often on an earlier and smaller panel. Several carry a superseded note pointing to current values. Where the current ledger and a hypothesis doc disagree, this page says so.

## 1. The original site on its own architecture

**Question.** Does the PSBD paper's method work as published before anything is moved? **Why.** Every later comparison is relative to this site, so it has to work somewhere first. **Measured.** `experiments/resnet_control/` trains ResNet-18 on GTSRB with BadNets and Blend at 10% under the paper's recipe and sweeps `post_residual` (the ResNet row of the ledger). **Result.** Both models separate almost perfectly at the adaptive rule. The method works where it was designed, so a failure on ViT is about the transfer. **Next.** Does the same site carry over to ViT?

## 2. The ConvNet rate grid on ViT (H3, H9)

**Question.** Why did dropout after the residual adds look broken on ViT in the first sweeps? **Hypothesis.** [H3](hypothesis/H3-why-post-residual-fails.md) proposed that masking the stream once per block compounds to $(1 - p)^{24}$ surviving coordinates over ViT's 24 residual adds, so every usable rate destroys clean and triggered evidence alike. [H9](hypothesis/H9-strength-not-position.md) stated the reviewer's objection: pre and post residual were compared at the same nominal $p$, which is a different strength at each site. **Measured.** The clean shift ratio and the backdoor direction's survival were read across a fine rate ladder (`experiments/dropout_kills_direction/`). **Result.** The saturation was real on the ConvNet grid of 0.1 and upward, but a good operating window exists an order of magnitude lower, which that grid never contained. H3 was refuted as stated and H9 supported. **What it changed.** No placement comparison is valid at a shared rate. Every placement since sweeps its own ladder (`configs/psbd_basis.json` lists each ladder), the adaptive rule picks each placement's own rate, and the matched rule compares placements at equal measured disturbance. **Next.** With strength controlled, does moving dropout before the add help?

## 3. Before versus after the residual add (H1)

**Question.** Is dropout on each branch output, before it is added (`pre_residual`), better than dropout on the stream after the add (`post_residual`)? **Hypothesis.** [H1](hypothesis/H1-pre-beats-post.md), the project's founding observation: perturbing a branch's contribution leaves the stream intact and should hurt clean evidence without the compounding of H3. **Measured.** Both placements over their own rate windows on every panel model. **Result.** Refuted as a general claim. In the current ledger the `pre_residual` row's paired gain over PSBD-RD is small and its interval spans 0 at both rules. The first sweeps showed pre-residual ahead only on the static patch trigger. **Next.** If the add is not the variable, what is?

## 4. Sublayer input versus residual stream (H20)

**Question.** Does it matter whether the perturbation lands on what a sublayer reads, or on what the stream carries? **Hypothesis.** [H20](hypothesis/H20-input-side-beats-residual-adjacent.md): an input-side perturbation (`after_embedding`, `before_attention_norm`, `before_attention`, `before_mlp_norm`, `before_mlp`) makes the sublayer compute on corrupted evidence, while a residual-adjacent one (the branch outputs and the stream) corrupts a sum the rest of the network can partly route around. **Measured.** On a balanced 12-model CIFAR-10 panel the input-side family beat the residual family by a clear margin whose interval excluded 0. **Result now.** On the current panel the family effect does not hold for dropout itself. With dropout, the attention input (`before_attention_norm`) sits within noise of PSBD-RD at the adaptive rule and below it in the point estimate at the matched rule. `before_attention`, `before_mlp_norm` and `after_embedding` sit below PSBD-RD, clearly so for the last 2. The input-side advantage appears once the operator changes, which steps 8 to 10 take up. H20's verdict rests on its smaller panel and has not been re-measured with this family split on the current one. **Next.** Maybe dropout is the wrong unit: a transformer has its own units.

## 5. Attention heads (H22, H35)

**Question.** Is the attention head, the transformer's own unit, the right thing to remove? **Hypothesis.** [H22](hypothesis/H22-head-mask-attention-units.md): a backdoor implemented as "attend to the trigger patch, write the target direction" might live in a few heads, and removing a head would collapse the shortcut while clean predictions, spread over many heads, survive. **Code path.** `attention_heads` swaps `self_attention.forward` for `masked_attention_forward`, which recomputes the query, key and value projections from `in_proj_weight`, splits them into (batch, 12, 197, 64), runs `scaled_dot_product_attention`, applies `HeadMask` and merges back before `out_proj`. At rate 0 it reproduces PyTorch's attention to float precision. **Result.** Refuted. The `attention_heads_head_mask` row's gain over PSBD-RD is within noise at both rules. H22 gives 3 reasons. Heads are redundant. The backdoor is a single direction not aligned with any head ([H16](hypothesis/H16-where-the-backdoor-neurons-are.md)). 1 early head dominates every input alike ([H18](hypothesis/H18-sensitivity-profile-over-units.md)). [H35](hypothesis/H35-targeted-head-psbd.md) then masked the 3 heads [H31](hypothesis/H31-attention-divergence-backdoor-heads.md) found diverging on triggered inputs. That deterministic probe read close to chance. **Next.** If not heads, perhaps the MLP's neurons or whole features.

## 6. MLP neurons and whole channels (H16, H26)

**Question.** Does removing whole features, a hidden neuron or a residual channel for every token at once, beat dropout's independent thinning? **Hypothesis.** [H26](hypothesis/H26-channel-mask-structured-vs-elementwise.md): dropout on a 197 by 768 tensor never removes a feature, since a channel survives in some token almost surely, while `channel_mask` removes it everywhere. An early reading of H16 placed the backdoor in a few residual dimensions. **Code path.** `mlp_neurons` is a pre-hook on `mlp.3`, whose input is the 3072-wide post-GELU hidden layer, so a channel there is 1 neuron. `GroupChannelMask` draws 1 Bernoulli per (sample, channel) and broadcasts it over tokens. **Result.** Refuted. `mlp_neurons_channel_mask` sits clearly below PSBD-RD, and channel masking loses to token masking at most positions where both were swept. H16's own causal test explains it: removing the backdoor direction kills the attack while zeroing even hundreds of its top coordinates does not, because the direction is not axis-aligned, so no set of channels or neurons names it. **Next.** Perhaps the unit is the computation rather than a feature.

## 7. DropPath, the residual-native perturbation (H21)

**Question.** Does removing a whole branch computation, the stochastic depth ViT is trained with, beat damaging it? **Hypothesis.** [H21](hypothesis/H21-droppath-residual-native.md): `DropPath` turns a block into the identity for a sample, and if the backdoor is written at a specific depth, removing the writing branch should suppress it sharply while clean evidence, accumulated over many blocks, degrades gracefully. The pre-registration also predicted the opposite outcome: a branch writes the backdoor and the clean signal together. **Code path.** `droppath` is restricted to `before_attention_residual` and `before_mlp_residual`, and draws 1 Bernoulli per sample broadcast over every other axis. **Result.** Confirmed in its losing half. The 3 `droppath` rows sit at or below PSBD-RD. The unit that matters is not the computation. **Next.** The remaining candidate unit is spatial: the token.

## 8. Whole tokens and trigger locality (H27)

**Question.** Does removing whole tokens separate local triggers from distributed ones? **Hypothesis.** [H27](hypothesis/H27-token-mask-trigger-locality.md): a patch trigger occupies a handful of the 196 patch tokens, so token masking is the only operator whose unit is spatial and should behave differently on patch triggers than on whole-image ones. **Code path.** `TokenMask` draws 1 Bernoulli per (sample, token), sets token 0's keep value to 1 and multiplies. The class token is the head's only read point, so it is never removed. On Swin it flattens (height, width) to tokens and protects nothing, since Swin has no class token. **Result.** Supported and stronger than predicted: token masking is best on patch triggers and worst on the whole-image warp of WaNet, and at the attention input it is the top of the ledger at the adaptive rule. **Next.** Why does removing tokens work where removing features does not, and is it removal at all?

## 9. Removal versus disturbance, and the LayerNorm side (H23, H47)

**Question.** Does PSBD need capacity removed, or only the activation disturbed? **Hypothesis.** The PSBD paper's neuron-bias account is about removal: under dropout, clean features collapse onto the target-biased path. [H23](hypothesis/H23-gaussian-noise-control.md) tested it with the control that removes nothing, isotropic Gaussian noise scaled to each sample's own standard deviation. **Result.** Refuted as a requirement: noise can match the best masks. The ledger's top rows at the matched rule include Gaussian noise at `before_attention`, `before_mlp_residual` and `mlp_neurons`. But noise at the attention input before its LayerNorm (`before_attention_norm_gaussian`) is among the worst rows. Noise at the MLP input before its LayerNorm is worse still. [H47](hypothesis/H47-layernorm-absorbs-noise-not-masking.md) explains the split with a prediction that has no free parameters: LayerNorm divides by each token's standard deviation, which additive noise inflates, so a LayerNorm right after the injection point undoes part of the noise, while a zeroed token or channel cannot be restored by any rescaling. It measured that LayerNorm absorbs a large share of noise and almost none of a mask, and that the mask-minus-noise gap reverses sign between a position a LayerNorm follows and a position it does not follow. **What it changed.** The operator ranking depends on which side of a LayerNorm the probe sits, and a comparison of operators across sites has to hold the normalization side fixed, which is the correction recorded as Q19 in `docs/open-questions.md`. **Next.** With the operator fixed at token masking, which site?

## 10. Attention input, MLP input, branch output and stream, with token masking

**Question.** Holding the operator at `token_mask`, where should it act? **Hypotheses.** The attention input is before mixing: a masked token contributes nothing to that block's attention, so the class token cannot read the trigger through it in that block. The MLP input is after mixing: by then the class token has already gathered whatever attention routed to it, and the MLP is token-wise and moves nothing between tokens ([H49](hypothesis/H49-backdoor-is-routed-not-computed.md)). The branch output before the add is the attention input's twin, since masking a token of either the input or the output of a token's own attention contribution removes that contribution. The stream after the add removes the token's accumulated content outright. **Result.** In the ledger the attention input is the best token-mask site, the branch output twin is level with it at the adaptive rule, the MLP input is clearly worse and the stream is worst. The twin tie on ViT is disclosed as Q20 in `docs/open-questions.md`: the declared selection half of the panel would have picked the twin, and the recommendation rests on Swin, where the ledger's Swin table separates the 2 by a wide margin, and on the mechanism. **Mechanism.** `experiments/why_token_masking_works/` measured why the attention input separates patch triggers: a token masked at the attention input keeps its entry in the residual stream, the class token reads a patch trigger mostly in the last blocks, and the trigger survives unless its few tokens are masked in nearly every late block, which is rare at the adaptive rate. Clean evidence, spread over many tokens and blocks, does not survive the same masking. Dropout on the stream corrupts the trigger tokens' stored content in every block, so the triggered prediction breaks as easily as a clean one. Global triggers stay legible from a subset of tokens under either operator. **Next.** Do the perturbations other detectors use do as well at the same sites?

## 11. Ported perturbations: LayerNorm gain and pixel amplification (H17, H28)

**Question.** Are IBD-PSC's parameter amplification and SCALE-UP's pixel amplification, run inside the same statistic, as good as the activation probes? **Hypothesis.** [H28](hypothesis/H28-perturbation-consistency-is-margin-estimation.md): PSBD, SCALE-UP, IBD-PSC and STRIP are 1 method, perturbation consistency as margin estimation, with 4 choices of where to perturb. **Code path.** `GainScale` multiplies a LayerNorm's output by $1 + p$ through a post-hook, which equals scaling its $\gamma$ and $\beta$ together. `ScaleUp` at `input_pixels` denormalizes, multiplies, clips to $[0, 1]$ and renormalizes before the Resize. Both are deterministic. **Result.** Gain scale at the MLP norm output reads above PSBD-RD with an interval spanning 0, and at the attention norm output and the final norm it is far worse, the final-norm row below chance with no rate reaching the adaptive target. Pixel amplification is near the bottom. An earlier headline for gain scale at the MLP norm output, a large gain at 1% poisoning in [H17](hypothesis/H17-low-poison-rate-is-a-placement-artifact.md), is withdrawn: it compared a placement read at a far higher clean disturbance than its baseline (`docs/audit-2026-09-07.md`). **Next.** Is acting in every block the right default?

## 12. Depth bands (H10, H30, H38)

**Question.** Does perturbing only the blocks where the backdoor direction is written beat perturbing all of them? **Hypothesis.** [H10](hypothesis/H10-depth-band-placement.md), from [H4](hypothesis/H4-placement-is-attack-dependent.md) and [H30](hypothesis/H30-residual-persistence-phase-transition.md): the backdoor direction crystallizes late, at different depths per attack ([H38](hypothesis/H38-crystallization-depth-vs-placement.md)), so a band aimed there should concentrate the perturbation where it separates. **Result.** Mixed, and it depends on the operator. For dropout before the adds, the middle band (`pre_residual_blocks_5_8`) beats both the full stack and PSBD-RD with an interval above 0 at both rules, and the early band is worse. For token masking at the attention input, every band trails the full stack at the matched rule, and at the adaptive rule the bands reach the 0.8 target on only part of the panel, because masking 4 blocks rarely disturbs a clean prediction that much. A band row's adaptive mean is therefore read on a different and easier subset of models than the full-stack row, and only its paired gain may be compared, as Q26 and Q38 of `docs/open-questions.md` warn. H10's onset-based rule for choosing the band was refuted by its own pre-registered out-of-sample test. **Next.** Does the chosen placement transfer to another transformer?

## 13. Transfer to Swin-S

**Question.** Is the result a property of ViT-B/16 or of transformers? **Why Swin.** It differs in exactly the ways that could break the mechanism: windowed attention instead of global, no class token (the head averages every token), 4 stages with patch merging and 24 blocks. **Result.** The Swin table of the ledger ranks token masking at the attention input at or near the top at both rules with a clear paired gain over PSBD-RD. It separates it from its branch-output twin, which on Swin sits near PSBD-RD. The late band is strong on Swin as well. The Swin panel is read by `swin_cells` in `scripts/paper/tab_swin.py`, since Swin has no coverage ledger of its own. Its models are not the same attacks and rates as the ViT panel.

## 14. Operators built and never swept

3 operators exist with tests but have no cache on any panel model. `RademacherNoise` tests a prediction of the reading of PSBD as a Hutchinson trace estimator (`docs/theory-perturbation-consistency.md`): at a matched scale it should separate at least as well as Gaussian noise at small $k$, with lower estimator variance. `TokenSubstitute` replaces whole tokens with other tokens of the same sample instead of zeroing them, which removes the off-manifold part of token masking, and would say how much of PSBD-TM's gain is the zero token itself. `TokenBlockMask` zeroes a square block of tokens, a control for trigger geometry that should help patch triggers and not global ones. Until they run, none of those 3 questions has an answer here. `docs/perturbations.md` describes `TokenSubstitute` as borrowing the token at the same position from another sample in the batch, while the code takes another token of the same sample by rolling the token axis, and the class docstring explains why the batch version was rejected.

## Placement ledger

<!-- results:begin -->
Generated by `python scripts/detector_doc_results.py` at commit `b2d32cf11708d5de965d3e13604c863a3ad9b493-dirty` from every `results/<folder>/psbd_metrics.json`. AUROC is the one-sided area at the 0.25 quantile. The adaptive rule reads each placement at the smallest rate whose clean-validation shift ratio reaches 0.8, and the matched rule at the rate whose shift ratio is closest to 0.6. A gain is the placement minus `post_residual` (PSBD-RD), paired over the models carrying both, with its 95% bootstrap interval (5000 resamples, seed 0) and the model count in brackets. Both placements are read at the same rule, so the matched gain differs from the paper's headline matched gain, which reads PSBD-RD at the adaptive rule.

ViT-B/16. The 54 successful models of the coverage ledger with a PSBD cache, diverged, source-mapped and clean-accuracy-failing models excluded. The n columns differ between rows because some placements were swept on a subset only, so compare 2 rows through their paired gains rather than their means.

| placement | position | operator | blocks | n matched | AUROC matched | n adaptive | AUROC adaptive | below chance, adaptive | gain over post_residual, matched | gain over post_residual, adaptive |
|---|---|---|---|---|---|---|---|---|---|---|
| `before_attention_gaussian` | `before_attention` | `gaussian` | all | 34 | 0.951 | 19 | 0.913 | 1 | +0.072 [+0.019, +0.138] (34) | -0.002 [-0.019, +0.017] (19) |
| `before_mlp_residual_gaussian` | `before_mlp_residual` | `gaussian` | all | 36 | 0.944 | 36 | 0.881 | 2 | +0.062 [+0.012, +0.124] (36) | -0.005 [-0.036, +0.029] (36) |
| `mlp_neurons_gaussian` | `mlp_neurons` | `gaussian` | all | 36 | 0.943 | 34 | 0.927 | 1 | +0.060 [+0.015, +0.117] (36) | +0.040 [+0.010, +0.082] (34) |
| `before_attention_token_mask` | `before_attention` | `token_mask` | all | 36 | 0.942 | 36 | 0.877 | 4 | +0.060 [+0.007, +0.125] (36) | -0.008 [-0.078, +0.067] (36) |
| `before_attention_norm_token_mask` | `before_attention_norm` | `token_mask` | all | 54 | 0.938 | 54 | 0.963 | 1 | +0.070 [+0.020, +0.124] (54) | +0.078 [+0.027, +0.134] (54) |
| `before_mlp_residual_token_mask` | `before_mlp_residual` | `token_mask` | all | 36 | 0.930 | 36 | 0.955 | 0 | +0.047 [-0.005, +0.112] (36) | +0.070 [+0.016, +0.134] (36) |
| `both_sublayer_inputs_token_mask` | `both_sublayer_inputs` | `token_mask` | all | 54 | 0.928 | 54 | 0.928 | 0 | +0.060 [+0.020, +0.105] (54) | +0.043 [-0.007, +0.095] (54) |
| `before_attention_residual_token_mask` | `before_attention_residual` | `token_mask` | all | 54 | 0.925 | 54 | 0.956 | 0 | +0.057 [+0.000, +0.116] (54) | +0.071 [+0.021, +0.126] (54) |
| `before_mlp_token_mask` | `before_mlp` | `token_mask` | all | 36 | 0.925 | 36 | 0.922 | 1 | +0.042 [-0.003, +0.098] (36) | +0.037 [-0.033, +0.104] (36) |
| `before_mlp_residual` | `before_mlp_residual` | `dropout` | all | 36 | 0.924 | 36 | 0.920 | 1 | +0.042 [+0.006, +0.085] (36) | +0.034 [-0.003, +0.084] (36) |
| `mlp_neurons_token_mask` | `mlp_neurons` | `token_mask` | all | 36 | 0.922 | 36 | 0.954 | 0 | +0.039 [-0.011, +0.102] (36) | +0.068 [+0.015, +0.133] (36) |
| `post_residual_gaussian` | `post_residual` | `gaussian` | all | 34 | 0.921 | 34 | 0.794 | 5 | +0.033 [-0.001, +0.071] (34) | -0.097 [-0.153, -0.043] (34) |
| `pre_residual_gaussian` | `pre_residual` | `gaussian` | all | 36 | 0.921 | 36 | 0.884 | 3 | +0.038 [-0.012, +0.098] (36) | -0.002 [-0.061, +0.062] (36) |
| `mlp_norm_out_gain_scale` | `mlp_norm_out` | `gain_scale` | all | 54 | 0.918 | 54 | 0.937 | 2 | +0.050 [+0.001, +0.106] (54) | +0.053 [-0.002, +0.112] (54) |
| `after_mlp_residual_token_mask` | `after_mlp_residual` | `token_mask` | all | 36 | 0.912 | 36 | 0.924 | 0 | +0.029 [-0.022, +0.085] (36) | +0.038 [-0.021, +0.105] (36) |
| `pre_residual_blocks_5_8` | `pre_residual` | `dropout` | 5 to 8 | 54 | 0.911 | 54 | 0.915 | 3 | +0.043 [+0.011, +0.080] (54) | +0.030 [+0.005, +0.059] (54) |
| `after_mlp_residual_gaussian` | `after_mlp_residual` | `gaussian` | all | 36 | 0.908 | 36 | 0.848 | 4 | +0.025 [-0.003, +0.056] (36) | -0.037 [-0.074, -0.001] (36) |
| `pre_residual_blocks_9_12` | `pre_residual` | `dropout` | 9 to 12 | 54 | 0.899 | 39 | 0.920 | 1 | +0.031 [+0.000, +0.063] (54) | +0.014 [-0.027, +0.056] (39) |
| `before_mlp_residual_channel_mask` | `before_mlp_residual` | `channel_mask` | all | 36 | 0.899 | 36 | 0.896 | 2 | +0.017 [-0.017, +0.055] (36) | +0.011 [-0.013, +0.037] (36) |
| `after_mlp_residual_channel_mask` | `after_mlp_residual` | `channel_mask` | all | 36 | 0.888 | 36 | 0.862 | 3 | +0.006 [-0.018, +0.028] (36) | -0.023 [-0.042, -0.006] (36) |
| `before_attention_norm_blocks_9_12_token_mask` | `before_attention_norm` | `token_mask` | 9 to 12 | 54 | 0.886 | 25 | 0.955 | 0 | +0.018 [-0.023, +0.060] (54) | +0.027 [-0.003, +0.073] (25) |
| `before_mlp_gaussian` | `before_mlp` | `gaussian` | all | 54 | 0.886 | 54 | 0.904 | 4 | +0.018 [-0.044, +0.081] (54) | +0.019 [-0.035, +0.076] (54) |
| `pre_residual_channel_mask` | `pre_residual` | `channel_mask` | all | 35 | 0.885 | 35 | 0.867 | 2 | +0.005 [-0.033, +0.048] (35) | -0.016 [-0.059, +0.021] (35) |
| `before_attention_residual_droppath` | `before_attention_residual` | `droppath` | all | 36 | 0.884 | 36 | 0.875 | 1 | +0.001 [-0.049, +0.063] (36) | -0.011 [-0.049, +0.030] (36) |
| `post_residual_channel_mask` | `post_residual` | `channel_mask` | all | 36 | 0.877 | 36 | 0.847 | 4 | -0.006 [-0.031, +0.018] (36) | -0.038 [-0.071, -0.013] (36) |
| `after_attention_residual` | `after_attention_residual` | `dropout` | all | 54 | 0.876 | 54 | 0.856 | 4 | +0.008 [-0.019, +0.035] (54) | -0.029 [-0.046, -0.015] (54) |
| `after_attention_residual_token_mask` | `after_attention_residual` | `token_mask` | all | 54 | 0.869 | 54 | 0.805 | 6 | +0.000 [-0.044, +0.047] (54) | -0.080 [-0.150, -0.010] (54) |
| `post_residual` | `post_residual` | `dropout` | all | 54 | 0.868 | 54 | 0.885 | 5 | reference | reference |
| `after_attention_residual_channel_mask` | `after_attention_residual` | `channel_mask` | all | 36 | 0.867 | 36 | 0.833 | 4 | -0.015 [-0.037, +0.009] (36) | -0.053 [-0.085, -0.026] (36) |
| `before_attention_norm_blocks_5_8_token_mask` | `before_attention_norm` | `token_mask` | 5 to 8 | 54 | 0.867 | 35 | 0.891 | 2 | -0.001 [-0.067, +0.068] (54) | -0.036 [-0.115, +0.043] (35) |
| `attention_heads_head_mask` | `attention_heads` | `head_mask` | all | 36 | 0.867 | 36 | 0.899 | 0 | -0.016 [-0.060, +0.038] (36) | +0.013 [-0.025, +0.060] (36) |
| `after_attention_residual_gaussian` | `after_attention_residual` | `gaussian` | all | 36 | 0.864 | 36 | 0.820 | 4 | -0.019 [-0.054, +0.016] (36) | -0.066 [-0.108, -0.027] (36) |
| `pre_residual` | `pre_residual` | `dropout` | all | 54 | 0.863 | 54 | 0.875 | 3 | -0.005 [-0.044, +0.037] (54) | -0.009 [-0.037, +0.016] (54) |
| `before_attention_norm_channel_mask` | `before_attention_norm` | `channel_mask` | all | 54 | 0.855 | 54 | 0.921 | 0 | -0.014 [-0.056, +0.033] (54) | +0.036 [-0.000, +0.076] (54) |
| `before_mlp` | `before_mlp` | `dropout` | all | 9 | 0.843 | 9 | 0.881 | 0 | +0.043 [-0.059, +0.175] (9) | +0.048 [-0.016, +0.120] (9) |
| `before_attention_norm` | `before_attention_norm` | `dropout` | all | 54 | 0.842 | 54 | 0.903 | 1 | -0.026 [-0.072, +0.022] (54) | +0.018 [-0.029, +0.065] (54) |
| `before_attention_residual_gaussian` | `before_attention_residual` | `gaussian` | all | 54 | 0.841 | 54 | 0.840 | 7 | -0.027 [-0.072, +0.016] (54) | -0.044 [-0.080, -0.014] (54) |
| `before_attention_norm_blocks_1_4_token_mask` | `before_attention_norm` | `token_mask` | 1 to 4 | 54 | 0.833 | 11 | 0.949 | 0 | -0.035 [-0.100, +0.030] (54) | -0.007 [-0.032, +0.022] (11) |
| `before_attention_channel_mask` | `before_attention` | `channel_mask` | all | 36 | 0.830 | 36 | 0.849 | 3 | -0.053 [-0.108, +0.013] (36) | -0.036 [-0.093, +0.025] (36) |
| `before_mlp_norm_token_mask` | `before_mlp_norm` | `token_mask` | all | 54 | 0.824 | 54 | 0.838 | 3 | -0.044 [-0.091, +0.004] (54) | -0.047 [-0.106, +0.012] (54) |
| `after_mlp_residual` | `after_mlp_residual` | `dropout` | all | 9 | 0.821 | 9 | 0.834 | 1 | +0.021 [-0.036, +0.099] (9) | +0.001 [-0.027, +0.032] (9) |
| `before_attention_residual` | `before_attention_residual` | `dropout` | all | 54 | 0.820 | 54 | 0.843 | 3 | -0.048 [-0.082, -0.010] (54) | -0.041 [-0.069, -0.012] (54) |
| `both_sublayer_inputs` | `both_sublayer_inputs` | `dropout` | all | 24 | 0.806 | 24 | 0.856 | 2 | -0.041 [-0.098, +0.026] (24) | +0.015 [-0.019, +0.053] (24) |
| `before_attention` | `before_attention` | `dropout` | all | 54 | 0.805 | 54 | 0.847 | 4 | -0.063 [-0.117, -0.002] (54) | -0.038 [-0.089, +0.014] (54) |
| `pre_residual_droppath` | `pre_residual` | `droppath` | all | 36 | 0.797 | 36 | 0.820 | 1 | -0.085 [-0.132, -0.030] (36) | -0.065 [-0.112, -0.013] (36) |
| `before_attention_residual_channel_mask` | `before_attention_residual` | `channel_mask` | all | 54 | 0.797 | 54 | 0.817 | 7 | -0.071 [-0.103, -0.035] (54) | -0.068 [-0.110, -0.036] (54) |
| `attention_norm_out_gain_scale` | `attention_norm_out` | `gain_scale` | all | 36 | 0.793 | 36 | 0.638 | 10 | -0.090 [-0.183, +0.003] (36) | -0.247 [-0.332, -0.167] (36) |
| `pre_residual_blocks_1_4` | `pre_residual` | `dropout` | 1 to 4 | 54 | 0.786 | 54 | 0.827 | 4 | -0.082 [-0.128, -0.033] (54) | -0.058 [-0.103, -0.012] (54) |
| `mlp_neurons_channel_mask` | `mlp_neurons` | `channel_mask` | all | 54 | 0.783 | 54 | 0.833 | 5 | -0.086 [-0.118, -0.050] (54) | -0.051 [-0.077, -0.024] (54) |
| `before_mlp_channel_mask` | `before_mlp` | `channel_mask` | all | 36 | 0.776 | 36 | 0.865 | 0 | -0.107 [-0.151, -0.055] (36) | -0.020 [-0.061, +0.027] (36) |
| `after_embedding` | `after_embedding` | `dropout` | all | 54 | 0.756 | 54 | 0.851 | 2 | -0.112 [-0.149, -0.075] (54) | -0.034 [-0.067, +0.001] (54) |
| `before_attention_norm_gaussian` | `before_attention_norm` | `gaussian` | all | 54 | 0.750 | 54 | 0.836 | 6 | -0.119 [-0.171, -0.066] (54) | -0.049 [-0.101, +0.001] (54) |
| `before_mlp_norm` | `before_mlp_norm` | `dropout` | all | 54 | 0.744 | 54 | 0.787 | 9 | -0.124 [-0.156, -0.091] (54) | -0.098 [-0.138, -0.062] (54) |
| `before_mlp_norm_channel_mask` | `before_mlp_norm` | `channel_mask` | all | 36 | 0.744 | 36 | 0.757 | 5 | -0.139 [-0.172, -0.104] (36) | -0.129 [-0.176, -0.088] (36) |
| `before_mlp_residual_droppath` | `before_mlp_residual` | `droppath` | all | 36 | 0.729 | 36 | 0.772 | 2 | -0.154 [-0.202, -0.100] (36) | -0.113 [-0.167, -0.056] (36) |
| `after_embedding_channel_mask` | `after_embedding` | `channel_mask` | all | 36 | 0.709 | 36 | 0.765 | 2 | -0.174 [-0.205, -0.140] (36) | -0.120 [-0.163, -0.077] (36) |
| `input_pixels_scale_up` | `input_pixels` | `scale_up` | all | 54 | 0.684 | 54 | 0.740 | 10 | -0.184 [-0.271, -0.094] (54) | -0.145 [-0.235, -0.052] (54) |
| `before_mlp_norm_gaussian` | `before_mlp_norm` | `gaussian` | all | 35 | 0.680 | 35 | 0.697 | 8 | -0.201 [-0.279, -0.129] (35) | -0.187 [-0.265, -0.113] (35) |
| `after_embedding_token_mask` | `after_embedding` | `token_mask` | all | 36 | 0.671 | 36 | 0.720 | 4 | -0.212 [-0.276, -0.146] (36) | -0.166 [-0.254, -0.075] (36) |
| `after_embedding_gaussian` | `after_embedding` | `gaussian` | all | 36 | 0.640 | 36 | 0.700 | 6 | -0.243 [-0.305, -0.184] (36) | -0.185 [-0.239, -0.135] (36) |
| `final_norm_out_gain_scale` | `final_norm_out` | `gain_scale` | all | 36 | 0.287 | 0 | -- | 0 | -0.596 [-0.692, -0.486] (36) | -- |

Swin-S. The 65 Swin models `scripts/paper/tab_swin.py` reads: the ViT panel rule applied to Swin-S (`scripts.paper._common.swin_coverage`), the models that are successful backdoors with clean accuracy within 2 points of the benign Swin-S model and carry a PSBD cache. Swin has 24 blocks, so its bands are 1 to 8, 9 to 16 and 17 to 24.

| placement | position | operator | blocks | n matched | AUROC matched | n adaptive | AUROC adaptive | below chance, adaptive | gain over post_residual, matched | gain over post_residual, adaptive |
|---|---|---|---|---|---|---|---|---|---|---|
| `before_attention_norm_blocks_17_24_token_mask` | `before_attention_norm` | `token_mask` | 17 to 24 | 63 | 0.980 | 22 | 0.975 | 0 | +0.163 [+0.102, +0.227] (63) | +0.159 [+0.079, +0.254] (22) |
| `before_attention_norm_token_mask` | `before_attention_norm` | `token_mask` | all | 65 | 0.955 | 63 | 0.973 | 1 | +0.135 [+0.089, +0.187] (65) | +0.096 [+0.051, +0.144] (63) |
| `pre_residual_blocks_17_24` | `pre_residual` | `dropout` | 17 to 24 | 63 | 0.953 | 39 | 0.962 | 0 | +0.137 [+0.068, +0.208] (63) | +0.098 [+0.040, +0.164] (39) |
| `before_attention_norm_gaussian` | `before_attention_norm` | `gaussian` | all | 63 | 0.913 | 63 | 0.940 | 2 | +0.096 [+0.047, +0.147] (63) | +0.063 [+0.015, +0.113] (63) |
| `before_attention_norm_blocks_9_16_token_mask` | `before_attention_norm` | `token_mask` | 9 to 16 | 63 | 0.909 | 53 | 0.944 | 1 | +0.092 [+0.043, +0.143] (63) | +0.075 [+0.033, +0.122] (53) |
| `both_sublayer_inputs_token_mask` | `both_sublayer_inputs` | `token_mask` | all | 63 | 0.897 | 63 | 0.955 | 0 | +0.080 [+0.044, +0.119] (63) | +0.078 [+0.037, +0.124] (63) |
| `before_attention` | `before_attention` | `dropout` | all | 63 | 0.888 | 63 | 0.910 | 4 | +0.072 [+0.039, +0.107] (63) | +0.033 [-0.004, +0.070] (63) |
| `mlp_norm_out_gain_scale` | `mlp_norm_out` | `gain_scale` | all | 63 | 0.887 | 63 | 0.828 | 10 | +0.070 [+0.012, +0.131] (63) | -0.049 [-0.131, +0.033] (63) |
| `pre_residual_blocks_9_16` | `pre_residual` | `dropout` | 9 to 16 | 63 | 0.871 | 63 | 0.896 | 3 | +0.054 [+0.008, +0.104] (63) | +0.019 [-0.030, +0.066] (63) |
| `after_mlp_residual_token_mask` | `after_mlp_residual` | `token_mask` | all | 16 | 0.863 | 16 | 0.846 | 1 | +0.073 [+0.012, +0.155] (16) | +0.032 [-0.100, +0.157] (16) |
| `after_attention_residual_token_mask` | `after_attention_residual` | `token_mask` | all | 63 | 0.856 | 63 | 0.838 | 6 | +0.039 [-0.006, +0.088] (63) | -0.038 [-0.105, +0.026] (63) |
| `after_embedding` | `after_embedding` | `dropout` | all | 63 | 0.840 | 63 | 0.878 | 5 | +0.023 [-0.002, +0.050] (63) | +0.002 [-0.033, +0.036] (63) |
| `before_attention_norm_channel_mask` | `before_attention_norm` | `channel_mask` | all | 63 | 0.839 | 63 | 0.893 | 3 | +0.022 [-0.033, +0.078] (63) | +0.017 [-0.033, +0.067] (63) |
| `before_attention_norm` | `before_attention_norm` | `dropout` | all | 63 | 0.826 | 63 | 0.891 | 3 | +0.009 [-0.039, +0.056] (63) | +0.015 [-0.046, +0.074] (63) |
| `before_mlp_residual` | `before_mlp_residual` | `dropout` | all | 30 | 0.823 | 30 | 0.894 | 1 | -0.009 [-0.043, +0.022] (30) | -0.018 [-0.065, +0.023] (30) |
| `post_residual` | `post_residual` | `dropout` | all | 65 | 0.819 | 65 | 0.881 | 5 | reference | reference |
| `pre_residual` | `pre_residual` | `dropout` | all | 63 | 0.814 | 63 | 0.860 | 7 | -0.003 [-0.030, +0.024] (63) | -0.016 [-0.054, +0.019] (63) |
| `after_attention_residual` | `after_attention_residual` | `dropout` | all | 63 | 0.807 | 63 | 0.940 | 0 | -0.010 [-0.047, +0.026] (63) | +0.063 [+0.028, +0.105] (63) |
| `before_attention_residual_token_mask` | `before_attention_residual` | `token_mask` | all | 65 | 0.804 | 65 | 0.850 | 8 | -0.015 [-0.058, +0.024] (65) | -0.031 [-0.084, +0.017] (65) |
| `before_mlp_norm` | `before_mlp_norm` | `dropout` | all | 63 | 0.801 | 63 | 0.803 | 10 | -0.016 [-0.036, +0.003] (63) | -0.074 [-0.111, -0.041] (63) |
| `before_mlp` | `before_mlp` | `dropout` | all | 32 | 0.799 | 32 | 0.719 | 10 | -0.029 [-0.068, +0.003] (32) | -0.148 [-0.224, -0.080] (32) |
| `before_mlp_gaussian` | `before_mlp` | `gaussian` | all | 63 | 0.798 | 63 | 0.727 | 16 | -0.019 [-0.053, +0.011] (63) | -0.150 [-0.208, -0.093] (63) |
| `attention_norm_out_gain_scale` | `attention_norm_out` | `gain_scale` | all | 10 | 0.786 | 10 | 0.896 | 0 | +0.050 [-0.093, +0.207] (10) | +0.065 [-0.049, +0.215] (10) |
| `pre_residual_blocks_1_8` | `pre_residual` | `dropout` | 1 to 8 | 63 | 0.782 | 63 | 0.863 | 7 | -0.035 [-0.080, +0.006] (63) | -0.014 [-0.050, +0.022] (63) |
| `input_pixels_scale_up` | `input_pixels` | `scale_up` | all | 63 | 0.760 | 63 | 0.814 | 9 | -0.057 [-0.107, -0.006] (63) | -0.063 [-0.117, -0.008] (63) |
| `before_attention_norm_blocks_1_8_token_mask` | `before_attention_norm` | `token_mask` | 1 to 8 | 63 | 0.745 | 53 | 0.802 | 9 | -0.072 [-0.133, -0.013] (63) | -0.073 [-0.141, -0.009] (53) |
| `before_attention_channel_mask` | `before_attention` | `channel_mask` | all | 16 | 0.743 | 16 | 0.809 | 2 | -0.048 [-0.137, +0.052] (16) | -0.005 [-0.111, +0.109] (16) |
| `before_mlp_token_mask` | `before_mlp` | `token_mask` | all | 16 | 0.730 | 16 | 0.726 | 5 | -0.061 [-0.145, +0.014] (16) | -0.088 [-0.221, +0.032] (16) |
| `before_mlp_norm_token_mask` | `before_mlp_norm` | `token_mask` | all | 63 | 0.728 | 63 | 0.828 | 7 | -0.089 [-0.139, -0.042] (63) | -0.049 [-0.094, -0.004] (63) |
| `after_embedding_token_mask` | `after_embedding` | `token_mask` | all | 26 | 0.658 | 26 | 0.684 | 7 | -0.112 [-0.219, -0.011] (26) | -0.136 [-0.262, -0.015] (26) |
| `final_norm_out_gain_scale` | `final_norm_out` | `gain_scale` | all | 10 | 0.313 | 0 | -- | 0 | -0.424 [-0.702, -0.088] (10) | -- |

ResNet-18 control, the PSBD paper's own architecture and site (`experiments/resnet_control/`). Each row is 1 model.

| model | placement | adaptive rate | AUROC adaptive | AUROC matched |
|---|---|---|---|---|
| `resnet18_gtsrb_badnet_a2o_0_1` | `post_residual` | 0.5 | 1.000 | 0.999 |
| `resnet18_gtsrb_blend_0_1` | `post_residual` | 0.5 | 0.968 | 0.975 |

<!-- results:end -->
