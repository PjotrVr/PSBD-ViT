# Theory triage of novel PSBD-like operators, and mixture against union

This document predicts, before any run, how 7 candidate operators should behave per attack category, and derives when averaging 2 probes (a mixture) or taking their min-rank union beats either probe alone. It extends the read model of `docs/why-psbd-works-theory.md` and is the filter that decides which candidates get GPU time on the 10-model development set.

## Provenance

Written on 2026-09-29. The scripts are in `experiments/theory_checks/`: `mix_union.py` recomputes the 39-model combination numbers, `crossover.py` computes the binormal crossover and `dev_placements.py` reads the cached proxy placements on the development set. The recompute reproduces the quoted numbers exactly: PSBD-TM 0.9659, late-band residual dropout (`pre_residual_blocks_9_12`) 0.9200, mixture 0.9787 and union 0.9831 on the same 39 models.

The development set is `vit_cifar10_wanet_0_1`, `vit_tiny_wanet_0_05`, `vit_cifar10_bpp_0_05`, `vit_gtsrb_bpp_0_01`, `vit_cifar10_blend_0_1`, `vit_cifar10_badnet_a2o_0_01`, `vit_gtsrb_tact_0_05`, `vit_cifar10_tact_0_01`, `vit_cifar100_bpp_0_01` and `vit_tiny_lf_0_01`. It holds the WaNet failure, PSBD-TM's weakest BPP and Blend models, the patch models where PSBD-RD collapses and 2 middle cases.

## Summary

None of the 7 candidates is predicted to beat PSBD-TM on patch triggers. 2 are predicted to complement it where it fails (warp and frequency): position-embedding jitter (6) and dropout of random-rotation subspaces (5). 2 are mechanism tests rather than detectors: the true key mask (1) tests the β sink and the class-token-only mask (2) tests the class-token route. Masking the tokens the class token attends to most (4) flips sign by attack category, so it fails as a one-sided detector and can serve as a category diagnostic. Temperature scaling (3) and block skipping (7) are predicted to fail.

The ordering PSBD-TM 0.966 < mixture 0.979 < union 0.983 is predicted from only 2 inputs, each member's AUROC and the clean correlation between the 2 probes (0.31), under a Gaussian copula. Both combined values come out within 0.001, and the gap comes entirely from the 3 WaNet models.

## The read model

Separation is approximated by the binary form of AUROC at matched clean damage. The continuous fractional score reads higher than this, so the sizes are floors, anchored to the measured PSBD-TM and PSBD-RD values rather than exact.

$$
\begin{aligned}
s_c(p) &= G(p) = \frac{1}{1 + e^{(p - p_{50})/s}}, \quad p_{50} \approx 0.29 \text{ to } 0.41,\ s \approx 0.1 \\
s_{\text{patch}} &= 1 - \prod_{l \in \mathcal{L}} \prod_{t \in T} (1 - \kappa_{l,t}) \quad \text{(OR over } mL \text{ reads)} \\
s_{\text{global}} &= P(V_l \ge f_g), \quad f_g \approx 0.3, \qquad s_{\text{warp}} = P(V_l \ge f_w), \quad f_w \gtrsim 0.6 \\
\pi_\beta &= \frac{n_m e^{q_0 \cdot k_\beta}}{n_m e^{q_0 \cdot k_\beta} + \sum_{u \in V} e^{q_0 \cdot k_u}}, \qquad k_\beta = W_k \beta + b_k \\
A &\approx \tfrac{1}{2}(1 + s_b - s_c)
\end{aligned}
$$

| symbol | meaning |
|---|---|
| $s_c$, $s_b$ | survival of the clean and triggered prediction under 1 pass |
| $G$, $p_{50}$, $s$ | the fitted per-image critical-rate curve, its median and scale |
| $\kappa_{l,t}$ | probability that trigger token $t$ is readable by the class token in block $l$ ($1-p$ under PSBD-TM) |
| $V_l$ | share of tokens visible to attention in block $l$ |
| $f_g$, $f_w$ | smallest visible share at which a global or warp trigger stays legible |
| $\pi_\beta$ | attention mass the class-token query $q_0$ puts on the $n_m$ masked tokens, which share the key $k_\beta$ |
| $A$ | AUROC in the binary approximation, triggered more stable being the standard direction |

2 further facts carry weight. The backdoor is about 1 dense direction $u$ in the residual stream. Operators that keep every token intact keep WaNet (PSBD-RD 0.956, late band 0.963 on its 3 models), while token removal breaks it (PSBD-TM 0.779 and 0.459 on `vit_cifar10_wanet_0_1`).

## Predictions per candidate

| design | patch | blend (Blend, LF, BPP) | frequency (SIG) | warp | verdict |
|---|---|---|---|---|---|
| (1) key mask with $-\infty$ logits | +, 0.90 to 0.97 (4 tokens), 0.6 to 0.75 (1 token) | +, 0.75 to 0.9 | +, 0.6 to 0.8 | at best equal to PSBD-TM, likely below | does not dominate, a mechanism test of the sink |
| (2) mask on the class-token row only | + but weak, 0.5 to 0.8 | +, 0.85 to 0.95 | +, 0.8 to 0.9 | +, 0.85 to 0.95 | likely cannot reach the 0.8 shift target, a warp complement at best |
| (3) per-head temperature | sign set by $\mathbb{E}\log\tau$, 0.3 to 0.95 | +, 0.85 to 0.95 | +, 0.8 to 0.9 | weak, 0.5 to 0.75 | fails |
| (4) mask the most-attended tokens | sign flips, inverted 0.75 to 0.9 | + weak, 0.5 to 0.65 | + weak | + weak or flipped | fails one-sided, a category diagnostic |
| (5) random-rotation subspace dropout | +, 0.8 to 0.95 (at most 0.5 if axis-aligned) | +, 0.93 to 0.98 | +, about 0.9 | +, about 0.95 | complements PSBD-TM, a smoother twin of PSBD-RD |
| (6) position-embedding jitter | +, 0.85 to 0.99 | +, 0.9 to 0.98 | weak, 0.5 to 0.8 | +, 0.85 to 0.97 | complements PSBD-TM on warp, with a risk of a shift ceiling |
| (7) random block skipping | + weak, 0.6 to 0.9 | +, 0.7 to 0.95 | +, 0.7 to 0.9 | +, 0.8 to 0.95 | does not dominate, a weak complement |

**(1) True key mask.** Under PSBD-TM every masked token enters attention with the same key $k_\beta$ and value $v_\beta$, so the class-token update is $(1-\pi_\beta)\bar v_V + \pi_\beta v_\beta$, with $\pi_\beta \approx p$ when $q_0 \cdot k_\beta$ is near the typical logit. At p 0.5 half the class-token read is a constant that pushes toward 1 class. A $-\infty$ key mask removes that term, and the read becomes an unbiased average over the visible tokens. 2 measured facts suggest the sink carries a large part of clean damage: the keep floor at p 0.9 equals $1/K$ on all 4 datasets, and shifted clean predictions land on the single largest class at 0.54 to 0.87 and on the runner-up only 10% to 35% of the time. If so, the key mask needs a much higher rate for the same clean damage. At $p \approx 0.85$ to 0.9 the patch OR still holds for 4 tokens ($p^{16}$ = 0.07 to 0.19) but not for 1 token ($p^4$ = 0.52 to 0.66). The sink carries clean fragility if the median clean critical rate rises from 0.29 to 0.41 up to at least 0.6 and the largest-class share of shifted predictions falls below 0.3. It does not if the key mask reads within 0.02 of PSBD-TM at matched shift.

**(2) Mask on the class-token row only.** Patch tokens still mix among themselves, so by the late blocks clean evidence and global and warp triggers are held redundantly in many tokens. The only single intact route left is a patch trigger read by the class token from its own few tokens, an OR over $mL$ reads as under PSBD-TM. Band-restricted token masking is the precedent: masking every row in blocks 9 to 12 reached a shift of only 0.43 at p 0.99. The class-token-row mask perturbs less, so it is predicted not to reach the 0.8 adaptive target on most models. It is a routing test, not a detector.

**(3) Per-head temperature.** With trigger logit gap $\Delta$, the class-token share on $m$ trigger tokens among $N$ is $m e^{\tau\Delta}/(m e^{\tau\Delta} + N)$. The measured 0.55 at $\tau = 1$ ($m = 4$, $N \approx 192$) gives $\Delta \approx 4.1$, so the share becomes 0.14 at $\tau = 0.5$, 0.055 at $\tau = 0.25$ and 0.99 at $\tau = 2$. Sharpening protects a patch trigger and flattening breaks it, so the sign on patch follows $\mathbb{E}\log\tau$. No structural asymmetry favors triggered reads in general.

**(4) Masking the most-attended tokens.** For patch triggers the most-attended late tokens are the trigger itself. With the trigger masked in blocks 9 to 12 survival is 0.200, so triggered predictions break more than clean ones, the inverted direction with $A = \tfrac{1}{2}(1 + 0.8 - b)$ = 0.75 to 0.9 for a clean change $b$ of 0.05 to 0.3. For global triggers the top tokens carry nothing special and $A \approx \tfrac{1}{2}(1 + b)$ = 0.5 to 0.65 in the standard direction. The signs are opposite, so a one-sided detector cancels on a mixed panel, and the sign itself identifies the patch category.

**(5) Random-rotation subspace dropout.** For a random $k$-dimensional subspace $S$ with projector $P_S$, $\mathbb{E}P_S = (k/d)I$. So $u^\top(I - P_S)x$ has mean $(1 - k/d)a$ with relative fluctuation about $\sqrt{k}\,\|x\|/(d\,a)$ for $k \ll d$. The operator ranks inputs by how much of their representation lies along their own decision direction, which is a cosine margin. Triggered representations are dominated by $u$. The "feature-space OR" analogy does not hold, because an OR needs redundant copies and 1 direction is attenuated by the same factor for every input. The cached axis-aligned proxy `post_residual_channel_mask` reads Blend 0.99, BadNets 0.34 and LF 0.95 on the 3 development models that carry it. The decisive paired test is rotation against axis-aligned at matched shift on the 2 development BadNets and TaCT models.

**(6) Position-embedding jitter.** This perturbs only $\pi_t$ in $x_t = e_t + \pi_t$, once at the input, with no compounding. Clean classification uses position, while content triggers are invariant: global triggers by construction and a patch trigger read from any 1 of its tokens. Doan et al.'s PatchShuffle lowers clean accuracy while Blend and WaNet ASR stay. SIG needs the phase relation across neighboring tokens, so its sign is weak. Jitter keeps every token, the property WaNet needs, which makes it the strongest warp complement. Strong jitter may not reach 0.8 clean damage, so reachability of the rate rule is checked first.

**(7) Random block skipping.** The trigger's late read is an OR over the reading blocks with exponent about $L_{\text{read}}$ (4 to 8). The trigger's high-norm token is manufactured in specific early and middle blocks, an AND, so skipping them costs roughly $(1-p)^{1 \text{ to } 2}$. Clean needs most blocks, both sides pay and the separation is weak. The cached `droppath` proxies read 0.50 to 0.92 on 3 development models.

## Mixture against min-rank union

Transform each probe's score to a normal score through its clean-validation CDF, and assume both classes share 1 correlation $r$.

$$
\begin{aligned}
(z_A, z_B) \mid \text{clean} &\sim N(0, R), \qquad (z_A, z_B) \mid \text{triggered} \sim N\big(-(\mu_A, \mu_B), R\big), \quad R = \begin{pmatrix} 1 & r \\ r & 1 \end{pmatrix} \\
\mu_j &= \sqrt{2}\,\Phi^{-1}(A_j), \qquad \lambda = \mu_B/\mu_A \\
A_{\text{mix}} &= \Phi\!\Big(\frac{\mu_A + \mu_B}{2\sqrt{1+r}}\Big), \qquad A_{\text{opt}} = \Phi\!\Big(\sqrt{\tfrac{\mu_A^2 + \mu_B^2 - 2r\mu_A\mu_B}{2(1-r^2)}}\Big) \\
A_{\text{mix}} > A_A &\iff \lambda > \sqrt{2(1+r)} - 1 \\
A_{\text{union}} &= P\big(\min(z_A, z_B)_b < \min(z_A, z_B)_c\big) \quad \text{(numerical)}
\end{aligned}
$$

| symbol | meaning |
|---|---|
| $z_j$ | normal score $\Phi^{-1}$ of the clean-validation rank of probe $j$'s fractional PSU |
| $r$ | clean correlation of the 2 normal scores (median 0.31 on the 39 models) |
| $\mu_j$, $A_j$ | separation and AUROC of probe $j$ alone |
| $\lambda$ | the weaker probe's separation over the stronger one's |
| $A_{\text{opt}}$ | the likelihood-ratio optimum, a weighted sum |

The mixture beats member A when $\lambda > \sqrt{2(1+r)} - 1$, which is 0.62 at $r = 0.31$. The union beats the mixture when $\lambda$ is below about 0.55, where the probes are lopsided or 1 is inverted. The union beats the best single member only when $\lambda \gtrsim 0.7$, and it mostly beats the fixed PSBD-TM because the stronger probe changes by attack.

| $\lambda$ | union | mixture | optimum |
|---|---|---|---|
| 0 | 0.944 | 0.870 | 0.972 |
| 0.4 | 0.952 | 0.943 | 0.967 |
| 0.6 | 0.960 | 0.964 | 0.972 |
| 1.0 | 0.980 | 0.988 | 0.988 |

These are at $A_A$ = 0.966 and $r$ = 0.31. A null partner costs the union 0.022 and the mixture 0.096, and an inverted partner at 0.4 leaves the union at 0.943 and drops the mixture to 0.834. The union is far more robust to a weak or inverted member, and the mixture is better when both carry comparable evidence.

**Check on the 39 models.** Fed each model's 2 marginal AUROCs and its measured $r$, the model predicts a mean mixture of 0.980 and a mean union of 0.984, against the measured 0.979 and 0.983. The 3 WaNet models contribute +0.0050 to the mean gap of +0.0044, so the other 36 net -0.0006. On WaNet, PSBD-TM reads 0.779, late band 0.963, mixture 0.893 and union 0.958. The 10 models with $\lambda < 0.55$ show a mean union minus mixture of +0.0165, and the 29 with $\lambda \ge 0.55$ show +0.0002. Per model the sign agrees with the prediction on only 16 of 39, because most per-model gaps are under 0.005, so the prediction holds for the mean and is not tested per model. Rank averaging (`mean_rank`, 0.962) is predicted to lose to both, because averaging uniform ranks compresses the extreme tail where a triggered input's evidence sits.

**Implication for choosing a partner.** A new operator adds to a PSBD-TM union if it is the stronger member on some category, which on this panel means warp. Elsewhere its weakness costs at most about 0.02 per model, provided it is not heavily inverted there. That favors (6) and (5) and rules out (4) and (3) as partners unless their patch sign is controlled.

## Idealizations

- The read model's thresholds are measured on 1 or a few models per attack, so sizes are rough ranges and not intervals.
- $\pi_\beta \approx p$ assumes $q_0 \cdot k_\beta$ near the typical logit, which is unmeasured and is what design (1) tests.
- The copula model assumes equal covariance in both classes. Real triggered scores are concentrated near 0 and bimodal, which is why raw averaging departs from the prediction on WaNet.
- The development-set proxies `post_residual_channel_mask`, `before_attention_residual_droppath` and `pre_residual_droppath` exist on 3 of the 10 models only. `pre_residual_blocks_9_12` covers 6 of the 10. Treat them as signs, not sizes.
