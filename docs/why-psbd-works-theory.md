# Quantitative predictions for why PSBD works on ViT

**Superseded on 2026-09-30 for panel numbers.** This document was written on the 57-model ViT panel of 2026-09-24. The panel is now the 56 successful models carrying both placements at the 2-point clean-accuracy bar (`paper/headline.tex`, `\HeadlinePairedCells`), which drops the only ViT SIG model and 2 WaNet models. Its panel means are kept as measured and are recomputed on the new panel once the analysis experiments of 2026-09-30 land.

This document derives the numbers each explanation of PSBD's behavior on ViT predicts and checks each derivation against the cached results on disk. It is the companion of the literature memo `docs/why-psbd-works-literature.md` and reuses its explanation ids (L1 to L27 and P1 to P12). It sets the thresholds the experiments in `experiments/why_psbd_works/` and `experiments/why_token_masking_works/` are judged against.

## Provenance

Written on 2026-09-29 against commit `2e59453`. Every number comes from CPU reads of the cached per-pass tensors and records. The scripts are in `experiments/theory_checks/` (`b_check.py`, `ladder.py`, `fit.py`, `cache_read.py`, `summ.py`, `attract.py`, `runnerup.py`, `lrconf2.py` and `kcurve.py`). The intermediate records they write are under `results/_experiments/theory_checks/`. The reads use the 57 clearing ViT models of `scripts/paper/_common.py`. As a population check they reproduce 3 headline numbers exactly: PSBD-TM 0.953 and PSBD-RD 0.888 at the adaptive 0.8 rule on fractional PSU, and Gaussian minus token mask at the attention input, -0.183 at the matched 0.6 rule.

The 1 ViT SIG model (`vit_cifar10_sig_0_1`) is under audit (`docs/audits/2026-09-29-experiment-audit.md`) because its cached triggered inputs may have been built at a different trigger amplitude from its training. Its values enter the pooled numbers below and will be rechecked once that audit closes.

## Notation

The shared symbols are below, and each section adds its own.

$$\phi(x) = 1 - \frac{\tfrac{1}{k}\sum_{j=1}^{k} P_c(x;\xi_j)}{P_c(x)}, \qquad c = \arg\max_i P_i(x)$$

| symbol | meaning |
|---|---|
| $\phi(x)$ | fractional PSU, low means flagged |
| $P_c(x)$, $P_c(x;\xi_j)$ | unperturbed and pass $j$ probability of the unperturbed argmax class $c$ |
| $\xi_j$ | the operator's random draw on pass $j$ |
| $k$ | number of passes, 3 in the headline |
| $p$ | operator rate (mask probability for token masking, drop probability for dropout) |
| $m$ | number of trigger tokens, 4 on CIFAR-10, CIFAR-100 and GTSRB, 1 on Tiny |
| $L$ | number of late blocks that read the trigger, 4 (blocks 9 to 12) |

## Token-mask survival

**Result.** Independence predicts exactly how often a patch trigger is fully masked, and it does not predict how often the triggered prediction breaks. The event "all $m$ trigger tokens masked in all $L$ late blocks" happened 74 times in 23020 pass and image pairs on the 9 four-token BadNets models, against 81.6 expected. That event accounts for only 58 of the 774 broken triggered predictions (7.5%). The all-or-nothing model predicts a triggered change of 0.0035 against the measured 0.034 on 4-token models, and 0.018 against the measured 0.043 over all 12 BadNets models.

**Assumptions.**

1. `TokenMask` draws a fresh keep tensor of shape (batch, tokens, 1) from Bernoulli$(1-p)$ on every call, and `plug_dropout` attaches 1 fresh module per block. Keep indicators are therefore independent across tokens, blocks, passes and images, given an ideal PRNG.
2. CLS is never masked, and the trigger's positions are fixed.
3. Under the breakage model a triggered prediction flips only if every trigger token is masked in every late block. This is the idealization under test.

**Derivation.**

$$
\begin{aligned}
P(\text{all } m \text{ masked in block } l) &= \prod_{t \in T} P(\mu_{l,t} = 0) = p^{m} && \text{assumption 1} \\
J &= \sum_{l \in \mathcal{L}} \mathbb{1}\big[\mu_{l,t} = 0 \ \forall t \in T\big] \sim \mathrm{Bin}(L, p^{m}) && \text{blocks independent} \\
P(J = L) &= p^{mL} \\
1 - \text{keep}_{\text{all or none}} &= p^{mL} && \text{assumption 3} \\
1 - \text{keep} &= \sum_{j=0}^{L} \binom{L}{j} p^{mj} (1 - p^{m})^{L-j} (1 - s_j) && \text{general form}
\end{aligned}
$$

| symbol | meaning |
|---|---|
| $\mu_{l,t}$ | keep indicator of token $t$ at the attention input of block $l$ |
| $T$ | the set of $m$ trigger positions |
| $\mathcal{L}$ | the late blocks 9 to 12 |
| $J$ | number of late blocks in which every trigger token is masked |
| $s_j$ | probability the triggered prediction survives given $J = j$ |

**Rates and event counts.** The 4-token BadNets models ran at their own adaptive rates. `vit_cifar100_badnet_a2o_{0_01,0_05,0_1}` and `vit_gtsrb_badnet_a2o_0_1` used 0.5, `vit_cifar10_badnet_a2o_0_1` and `vit_gtsrb_badnet_a2o_0_01` used 0.6, `vit_cifar10_badnet_a2o_0_05` used 0.7, `vit_cifar10_badnet_a2o_0_01` used 0.8 and `vit_gtsrb_badnet_a2o_0_05` used 0.3. The event probability $p^{16}$ is 4.3e-9, 1.5e-5, 2.8e-4, 3.3e-3 and 2.8e-2 at 0.3, 0.5, 0.6, 0.7 and 0.8. Of the 74 observed events, 66 come from the model at p 0.8 (71.5 expected) and 8 from the model at p 0.7 (8.5 expected). The other 7 models expected 1.6 events together and showed none. The pooled "0.003" is therefore a mixture that 1 model dominates, and it is not a property of 4-token triggers at any single rate.

The pooled histogram over $J$ from 0 to 4 reads [14390, 5895, 2027, 634, 74] against the binomial [14489, 5876, 2005, 568, 82], and the worst single model has a chi-square p of about 0.07. The 1-token Tiny models at p 0.5 show 450 of 7670 events against 0.0625 × 7670 = 479. Models that share a rate drew identical masks under the fixed seed, so pooled counts are not independent replicates.

**Where the breakage comes from.** On the 4-token models the 774 broken pairs split by $J$ = 0, 1, 2, 3 and 4 as 181, 207, 168, 160 and 58 (23%, 27%, 22%, 21% and 8%). The measured survivals are $s_j$ = 0.987, 0.965, 0.917, 0.748 and 0.216. The prediction weakens with every partly masked late read, which is graded behavior and not the all-or-nothing rule. Blocks 5 to 8 also matter (section A of `experiments/why_token_masking_works/` reports 0.788 survival with blocks 5 to 8 masked), and survival is still 0.216 when the full event occurs.

A refinement that plugs in the deterministic survivals of section A (last $j$ blocks masked, the last 3 interpolated) predicts 0.044 against the measured 0.034 on 4-token models. It fails per model: on Tiny it predicts 0.29 to 0.85 against the measured 0.02 to 0.13. Masking the last $j$ blocks always includes block 12, while the stochastic masks hit a random subset of $j$ late blocks, so a predictive model needs $s_j$ averaged over random subsets.

## Clean fragility under token masking

**Result.** The clean change of 0.87 is set by the rate rule and measures nothing about fragility by itself. The adaptive rule picks the smallest ladder rate with a validation shift of at least 0.8, and on a ladder spaced by 0.1 that gives a realized mean of 0.867 (range 0.801 to 0.926). The informative quantity is the rate the rule picks: 0.5 on 31 models, 0.6 on 11, 0.7 on 6, 0.8 on 4, 0.4 on 3 and 0.3 on 2.

Backing out a homogeneous "number of needed tokens" gives an implausible and non-constant $n_{\text{eff}} \approx 3$. The clean keep curve fits a per-image critical rate instead, a logistic in $p$ plus a floor equal to the share of images predicted as the model's default class. That model is plausible and fits 5 to 10 times better.

$$
\begin{aligned}
\text{keep}(p) &= P\big(\mathrm{Bin}(n, 1-p) \ge r\big) && r = n \text{ gives } (1-p)^{n}, \ r = 1 \text{ gives } 1 - p^{n} \\
n_{\text{eff}} &= \frac{\ln(1 - \sigma)}{\ln(1 - p)} && \text{AND model backed out at 1 rate} \\
\text{keep}(p) &= \pi_d + (1 - \pi_d)\, G(p), \qquad G(p) = P(p^{*}_x > p) = \frac{1}{1 + e^{(p - p_{50})/s}}
\end{aligned}
$$

| symbol | meaning |
|---|---|
| $n$, $r$ | number of evidence units, and how many must stay visible |
| $\sigma$ | clean validation shift ratio at rate $p$ |
| $n_{\text{eff}}$ | units under the AND model ($r = n$) |
| $\pi_d$ | share of images whose unperturbed class is the model's default class under heavy masking |
| $p^{*}_x$ | the critical masking rate of image $x$ |
| $p_{50}$, $s$ | median and logistic scale of $p^{*}_x$ across images |

The trigger is the $r = 1$ case with $n = mL$ read events. The clean prediction behaves close to the $r \approx n$ case.

**Fits.** At the adaptive rate the AND model gives a median $n_{\text{eff}}$ of 2.83 (range 1.04 to 6.39). Along a single model's ladder it rises, for example from 1.35 at p 0.05 to 3.01 at p 0.6 on `vit_cifar100_badnet_a2o_0_01`, so the model is wrong in form. The floor at p 0.9 is 0.118, 0.016, 0.030 and 0.006 on CIFAR-10, CIFAR-100, GTSRB and Tiny, against $1/K$ = 0.100, 0.010, 0.023 and 0.005. With the floor removed, the logistic fits each dataset's mean curve with RMSE 0.006 to 0.016, and the AND power law gives 0.060 to 0.116.

| dataset | $p_{50}$ | $s$ | sd of $p^{*}_x$ ($s\pi/\sqrt{3}$) |
|---|---|---|---|
| CIFAR-10 | 0.411 | 0.097 | 0.18 |
| CIFAR-100 | 0.302 | 0.104 | 0.19 |
| GTSRB | 0.320 | 0.086 | 0.16 |
| Tiny | 0.288 | 0.110 | 0.20 |

**Plausibility.** A homogeneous threshold over 196 independent tokens would make the critical rate's spread at most $\sqrt{p(1-p)/196} \approx 0.036$. The fitted spread is 0.16 to 0.20, so about 95% of its variance lies between images. The pass-level data agree. At $k = 3$ the validation flip count histogram at the adaptive rate reads [0.088, 0.036, 0.059, 0.816], against a binomial at the same mean of [0.003, 0.048, 0.293, 0.656], and the intra-image correlation of flips is 0.69 (median over models). Clean fragility is a property of the image: roughly 70% of the tokens must stay visible per block, and about 9% of clean images never flip. Those 9% sit at the bottom of the clean score distribution with the triggered images, so they drive the false positives.

Section D of `experiments/why_token_masking_works/` shows that a fixed visible subset of 60% keeps 0.58 to 0.75 of clean accuracy. Independent per-block masks at the same share destroy more, which fits damage that accumulates across blocks.

## Residual dropout at post_residual

**Result.** A first-order variance decomposition explains the 2 pooled facts. Trigger-only dropout matches all-token dropout on average (0.387 against 0.398), and random positions keep 1.000. The triggered margin's sensitivity is concentrated on the trigger positions (plus CLS, which the "all but trigger" variant includes), so dropout elsewhere adds no variance on the target side.

The equality holds only in the 12-model mean. Per model the mean absolute difference between trigger-only and all-token survival is 0.223. In 8 of 12 models, adding dropout on the other positions raises survival above $\min(s_T, s_{\text{rest}})$. A first-order model with zero mean forbids that, so dropout on the content positions must also lower the competing class's logit.

Inverted dropout gives margin retention that does not depend on the norm of the backdoor component. It depends on the direction's coordinate spread and on how many dropout applications lie between the trigger's write and its read.

**Assumptions.** Dropout is $y = \mu \odot x/(1-p)$ with $\mu_i \sim$ Bernoulli$(1-p)$ independent per token and channel. It is applied twice per block (after each residual add), so 24 times through the stack. The margin is treated to first order in the perturbation, which is an idealization, since compounded effective rates are large.

$$
\begin{aligned}
\operatorname{Var}(u^{\top} y) &= \tfrac{p}{1-p} \sum_i u_i^{2} x_i^{2} \\
\frac{\operatorname{sd}(u^{\top} y)}{u^{\top} x}\Big|_{x = a u + x_\perp} &\to \sqrt{\tfrac{p}{1-p}}\, \Big(\sum_i u_i^{4}\Big)^{1/2} = \sqrt{\tfrac{p}{1-p}}\, d_{\text{eff}}(u)^{-1/2} \quad (a \to \infty) \\
p_{\text{eff}}(n) &= 1 - (1-p)^{n} && n \text{ applications between write and read} \\
\cos\big(\mathrm{LN}(\mu \odot x), \mathrm{LN}(x)\big) &\approx \sqrt{E}, \quad E = \frac{\sum_i \mu_i \tilde x_i^{2}}{\sum_i \tilde x_i^{2}}, \quad \mathbb{E}E = 1-p, \ \operatorname{Var}E = \frac{p(1-p)}{d_{\text{eff}}(x)} \\
\Delta M &= \sum_t g_t^{\top} \delta_t, \qquad V = \tfrac{p}{1-p} \sum_t \sum_i g_{t,i}^{2} x_{t,i}^{2}, \qquad V_{\text{all}} = V_T + V_{\text{rest}} \\
s &= \Phi\big((M + \mathbb{E}\Delta M)/\sqrt{V}\big)
\end{aligned}
$$

| symbol | meaning |
|---|---|
| $u$ | unit backdoor direction in the residual stream |
| $a$, $x_\perp$ | component of a token along $u$, and the rest |
| $d_{\text{eff}}(v)$ | participation ratio $(\sum v_i^2)^2/\sum v_i^4$ over coordinates |
| $\tilde x$ | the per-token centered vector (the LayerNorm mean shift under masking is neglected) |
| $E$ | share of a token's energy that dropout keeps |
| $M$ | triggered margin $z_t - \max_{j \ne t} z_j$ |
| $g_t$ | gradient of $M$ with respect to the stream at token $t$ |
| $\delta_t$ | dropout perturbation of token $t$ |
| $V_T$, $V_{\text{rest}}$ | variance contributed by trigger positions and by all others |
| $s$ | survival probability of the triggered prediction |

**Reading.** The noise is multiplicative, so the signal and its fluctuation both scale with $a$, and a large backdoor norm buys no protection. For a dense direction ($d_{\text{eff}} \approx d/3 = 256$) the projection itself fluctuates by about 0.31/16 = 0.02 per application at p 0.09. What breaks the prediction is noise sprayed orthogonally at a magnitude proportional to $a$, plus the loss of stored content. After LayerNorm each token is rotated to a cosine of about $\sqrt{1-p}$ per application. Content that persists across $n$ applications keeps a coordinate at rate $(1-p)^n$, which is 0.22 at p 0.09 and $n = 16$. PSBD-TM never touches the stream, which is the contrast.

With zero drift, $s$ decreases in $V$, so $s_{\text{all}} \le \min(s_T, s_{\text{rest}})$. The measured violation in 8 of 12 models requires $\mathbb{E}\Delta M > 0$ under dropout on the content positions. That is the clean source class losing logit, clean fragility acting on the competitor.

**Predictions.**

- Within a model, triggered survival under PSBD-RD is flat in the per-image backdoor projection $a$ (absolute Spearman at most 0.1 above the median $a$). It falls with the number of dropout applications between the trigger's write (by block 7) and its late read.
- Survival falls as $d_{\text{eff}}$ of the trigger tokens' residual vectors falls.
- Dropout on CLS alone keeps about as much as the "all but trigger" variant (0.66).
- Under dropout on the content positions, $z_{\text{source}}$ falls in expectation while $z_t$ does not.

## LayerNorm arithmetic at the attention input

**Result.** With torchvision's `eps` of 1e-6, a zeroed token becomes exactly $\beta$ after the norm. The $1/(1-p)$ rescale of surviving tokens is exactly undone, so `TokenMask` at `before_attention_norm` replaces a random subset with a constant and leaves every other token bit-exact. Additive noise is shrunk by the norm only through a chord effect, whose absorbed share is $1 - D(\rho)/\rho \approx 3\rho^2/8$: 0.004, 0.015 and 0.055 at ρ 0.1, 0.2 and 0.4.

The measured "absorbed" shares are 0.15 to 0.23 and nearly flat in ρ. Most of the measured absorption is a linear attenuation, and the variance renormalization accounts for only a small part of it. A linear attenuation is equivalent to rescaling the rate, which the matched-shift rule compensates. Absorption therefore cannot produce the -0.183 matched-shift gap.

The sign of the gap follows from a different property. Token masking gives reads that are either intact or gone, while high-dimensional additive noise degrades every read by nearly the same amount. That structure predicts a negative gap that is largest on patch triggers, which is what the data show.

$$
\begin{aligned}
\mathrm{LN}(x) &= \gamma \odot \frac{x - \bar x \mathbf{1}}{\sqrt{v(x) + \varepsilon}} + \beta \\
\mathrm{LN}(0) &= \gamma \odot 0 + \beta = \beta \\
\mathrm{LN}(c x) &= \gamma \odot \frac{x - \bar x \mathbf{1}}{\sqrt{v(x) + \varepsilon/c^{2}}} + \beta \approx \mathrm{LN}(x), \quad c = \tfrac{1}{1-p} > 0 \\
D(\rho) &= \Big\| \tfrac{u + \rho w}{\sqrt{1+\rho^{2}}} - u \Big\| = \sqrt{2 - \tfrac{2}{\sqrt{1+\rho^{2}}}}, \qquad \rho_t = \frac{r\, s_x}{\sigma_t} \\
\cos\big(\mathrm{LN}(x_t + \epsilon_t), \mathrm{LN}(x_t)\big) &= \tfrac{1}{\sqrt{1+\rho_t^{2}}} + O(\rho_t/\sqrt{d})
\end{aligned}
$$

| symbol | meaning |
|---|---|
| $\gamma$, $\beta$, $\varepsilon$ | LayerNorm gain, bias and epsilon (1e-6) |
| $\bar x$, $v(x)$ | per-token mean and variance over the $d = 768$ channels |
| $u$, $w$ | unit centered token and unit noise direction, orthogonal to leading order in $d$ |
| $\rho$, $\rho_t$ | relative noise norm, and its value on token $t$ |
| $r$, $s_x$, $\sigma_t$ | `GaussianNoise` rate, the per-sample std over all tokens and channels, and token $t$'s own std |
| $D(\rho)$ | relative disturbance after the norm with $\gamma = 1$, $\beta = 0$ |

$D(\rho)$ was verified on synthetic 768-dimensional tokens (0.0995, 0.3777, 0.7656 and 1.1694 against 0.0996, 0.3782, 0.7654 and 1.1694 at ρ 0.1, 0.4, 1 and 3). The measured Gaussian survival in `results/*/layernorm_absorption.json` is 0.84, 0.82 and 0.77 at ρ 0.1, 0.2 and 0.4 on GTSRB BadNets and 0.85, 0.84 and 0.81 on CIFAR-100 Blend. The chord effect predicts 0.996, 0.985 and 0.945. The remaining attenuation is linear and roughly independent of ρ, which points to the alignment of $\gamma$ and to $\beta$ in the metric's denominator. That is inferred and not measured.

**Gap direction.** Take an idealized read model. A patch-triggered prediction survives if at least 1 of $R = mL$ reads is intact (OR), and a clean prediction needs the average fidelity over many reads (AND or average).

- Under token masking the triggered survival is $1 - p^{R}$, and the clean average fidelity concentrates at $1 - p$.
- Under noise every read has fidelity near $1/\sqrt{1+\rho^2}$, with no exact reads left. Triggered and clean survival are then both steps in the same $\rho$, and they separate only if the trigger tolerates a lower fidelity threshold than clean content.

At matched clean damage, token masking separates at least as well whenever 1 exact read suffices. This predicts a negative gap, largest on patch triggers, smaller on global triggers whose read is itself an average, and near 0 where token masking also fails. The measured gaps at the matched rule are BadNets -0.365, TaCT -0.242, LF -0.161, BPP -0.154, Blend -0.116, WaNet -0.032 and SIG 0.000. Sign and ordering agree.

The operating point is also far outside the small-noise regime. The matched Gaussian rate is 1.0 on 24 of 57 models (range 0.2 to 1.5), against a token-mask rate of 0.4 on 23 models.

**A second mechanism, inferred and not yet measured.** `GaussianNoise` scales noise by the per-sample std over all tokens, including the high-norm trigger tokens and CLS. A triggered image therefore gets more noise on every ordinary token than its clean twin. Under Gaussian noise at the matched rule, the 3 CIFAR-100 BadNets models flip 0.82 to 0.86 of triggered predictions against 0.6 clean, with AUROC 0.17 to 0.28. That is inverted, and it is compatible with this coupling. The prediction is that $s_x$(triggered)/$s_x$(clean) is above 1 at `before_attention_norm` in blocks 5 to 12 for BadNets and near 1 for global triggers, and that noise scaled per token removes the inversion.

## Shift ratio, PSU and confidence

**Result.**

- **Ties.** The per-input shift count takes $k+1$ values, which caps its AUROC at $1 - \pi/2$. For a monotone coarsening of a continuous score the AUROC loss is at most $\pi/2$. On PSBD-TM the count reads 0.907 against 0.953 for fractional PSU, with tie mass $\pi$ = 0.150.
- **Fractional against absolute.** Fractional PSU beats absolute PSU pair by pair whenever the triggered input is the more confident one near the decision boundary. It wins on 37 of 57 models under PSBD-TM (+0.007) and 53 of 57 under PSBD-RD (+0.071).
- **P1, "PSU is just confidence".** This predicts that AUROC is bounded by the best function of confidence, the likelihood-ratio transform. That bound is 0.873 (cross-fitted), well above the monotone 0.680. PSBD-TM exceeds it on 52 of 57 models by +0.080, PSBD-RD on 43 of 57 by only +0.015.

$$
\begin{aligned}
N(x) &= \sum_{j=1}^{k} \mathbb{1}\big[\hat y_j(x) \ne c\big] \in \{0, 1, \dots, k\} \\
A(S) &= P(S_b < S_c) + \tfrac{1}{2} P(S_b = S_c), \qquad \pi = \sum_v P(G_b = v) P(G_c = v) \\
A(T) - A(G) &= \tfrac{1}{2}\big[P(T_b < T_c,\ G_b = G_c) - P(T_b > T_c,\ G_b = G_c)\big] \in \big[-\tfrac{\pi}{2}, \tfrac{\pi}{2}\big] \\
\psi &= P_c \phi, \qquad A(\phi) - A(\psi) = P\Big(1 < \tfrac{\phi_c}{\phi_b} < \tfrac{P_b}{P_c}\Big) - P\Big(\tfrac{P_b}{P_c} < \tfrac{\phi_c}{\phi_b} < 1\Big) \\
\phi \perp Y \mid P_c \ &\Rightarrow\ A(\phi) \le A^{*}(P_c) = A\big(\Lambda(P_c)\big)
\end{aligned}
$$

| symbol | meaning |
|---|---|
| $N(x)$, $\hat y_j$ | shift count over $k$ passes, and the argmax of pass $j$ |
| $S_b$, $S_c$ | a score on an independent triggered and clean draw (low means poisoned) |
| $T$, $G = g(T)$ | continuous score and a nondecreasing coarsening of it |
| $\pi$ | tie probability of $G$ between the 2 populations |
| $\psi$ | absolute PSU |
| $P_b$, $P_c$ | in this section, unperturbed confidences of the triggered and clean member of a pair |
| $Y$ | triggered or clean label |
| $\Lambda$ | likelihood ratio of triggered against clean given $P_c$ |
| $A^{*}$ | AUROC of the likelihood-ratio transform, the best any function of $P_c$ can do |

**Derivation of the tie bound.** $G_b < G_c$ implies $T_b < T_c$, and equal $T$ implies equal $G$. So $P(T_b<T_c) = P(G_b<G_c) + P(T_b<T_c, G_b=G_c)$, and $P(G_b=G_c)$ splits into the parts where $T$ is lower, higher and equal. Substituting both into $A(T) - A(G)$ gives the bracket.

The shift count is not a coarsening of $\phi$, because a pass can lower $P_c$ without flipping and flip without a large drop. The bound therefore fails on 5 of 57 models under TM and 10 of 57 under RD, while the ceiling $1 - \pi/2$ holds on all 114. The mean histograms at $k=3$ are [0.080, 0.034, 0.056, 0.830] for paired clean and [0.834, 0.049, 0.025, 0.092] for triggered, giving $\pi$ = 0.146. TaCT loses most (0.658 count against 0.962 fractional), because its triggered predictions flip while keeping relative probability mass.

**Fractional against absolute.** For $\phi > 0$ a correctly ordered pair is reversed by $\psi$ exactly when $\phi_c/\phi_b < P_b/P_c$, which needs $P_b > P_c$. The mean $\mathbb{E}\log P_c$ of triggered inputs minus the same for clean is +0.063, so fractional PSU dominates. The gain is largest where the $\phi$ ratios near the boundary are close to 1: BadNets under PSBD-RD reads 0.712 fractional against 0.574 absolute.

**P1 bound.** If $\phi$ is conditionally independent of the label given $P_c$, then $\phi$ is a randomized function of $P_c$, and by Neyman and Pearson its ROC is dominated by the likelihood-ratio ROC of $P_c$. $\Lambda$ was estimated on 40 quantile bins of $-\log(1-P_c)$ with 2-fold cross-fitting, folds split by pair. $A^{*}$ is 0.873 over the models and 0.900 BadNets, 0.926 Blend, 0.867 BPP, 0.876 LF, 0.730 SIG, 0.857 TaCT and 0.727 WaNet by attack. The likelihood ratio is not monotone in $P_c$, which is why raw confidence (0.680 in this recompute, 0.684 in the detector record) understates what confidence carries. $A^{*}$ needs the triggered distribution, so it is an oracle bound and not a detector.

## Subsampling the number of passes

**Result.** The first $k$ of 20 independent passes have the same joint law as a $k$-pass run, and $\phi_k$ is unbiased for its infinite-pass limit. They are different draws: `compute_dropout_pass_probs` seeds once per split and draws all passes of 1 batch before the next, so a 3-pass run and a 20-pass run share masks only in the first batch. The run log measured this (`docs/runs/2026-09-29-k20-login-gpu.md`): the first 64 samples agree to bf16 noise, and later ones differ by up to 0.109.

To leading order in $1/k$, AUROC approaches its limit as $A_\infty - f'(0)/(2k)$. On the 17 pilot models a fit on $k \ge 3$ gives $A_\infty$ = 0.9749 and $C$ = 0.0056. It predicts a gain of 0.0016 from $k = 3$ to 20, and the measured gain is 0.0016. Over all $k$ from 1 to 20 a free exponent fits 1.44, because $k$ = 1 and 2 are outside the Gaussian regime.

**Assumptions.** The PRNG is ideal, so draws are independent and identically distributed from the operator law $Q$ across passes and images. No operator's law depends on the batch, and `GaussianNoise` scales per sample. For the AUROC asymptote, the pass noise of a pair is approximately Gaussian given the pair, and the pair-level density $f$ is differentiable at 0.

$$
\begin{aligned}
(\xi_1, \dots, \xi_k) &\overset{d}{=} (\xi'_1, \dots, \xi'_k) \sim Q^{\otimes k} \ \Rightarrow\ \phi_k \overset{d}{=} \phi'_k \\
\mathbb{E}\,\phi_k(x) &= 1 - \frac{\mathbb{E}_{Q} P_c(x;\xi)}{P_c(x)} = \phi_\infty(x) \\
\operatorname{Var}\big(\phi_k(x) \mid x\big) &= \frac{\operatorname{Var}_{Q} P_c(x;\xi)}{k\, P_c(x)^{2}} \approx \frac{q_x (1 - q_x)}{k} \quad \text{(flip model)} \\
A_k - A_\infty &= \int \big(\Phi(u\sqrt{k}) - \mathbb{1}[u > 0]\big) f(u)\, du = \frac{f'(0)}{k} \int v\big(\Phi(v) - \mathbb{1}[v>0]\big)\, dv + O(k^{-2}) = -\frac{f'(0)}{2k} + O(k^{-2}) \\
\mathrm{Rel}_k &= \frac{k \rho_I}{1 + (k-1)\rho_I}
\end{aligned}
$$

| symbol | meaning |
|---|---|
| $\xi'_j$ | draws of a separate $k$-pass run |
| $\phi_\infty(x)$ | the infinite-pass fractional PSU |
| $q_x$ | per-image flip probability, with the per-pass ratio near 0 or 1 |
| $u$ | pair difference $\phi_\infty(c) - \phi_\infty(b)$ divided by its pass-noise sd |
| $f$ | density of $u$ over pairs |
| $A_k$ | AUROC at $k$ passes |
| $\rho_I$ | intra-image correlation of pass flips |
| $\mathrm{Rel}_k$ | Spearman-Brown reliability of a $k$-pass mean |

**Derivation steps.** The zeroth-order term vanishes because $\Phi(v) - \mathbb{1}[v>0]$ is odd. The first-order integral is $-2\int_0^\infty v(1-\Phi(v))dv = -2 \cdot \tfrac{1}{4}$ by integration by parts. The $O(k^{-3/2})$ term vanishes by oddness again. On $u \sim N(\mu, 1)$ the exact $\Phi(\mu/\sqrt{1+1/k}) - \Phi(\mu)$ is -0.00273 against the asymptote -0.00270 at $\mu = 2$, $k = 20$.

AUROC rises with $k$ exactly when $f'(0) > 0$. That is usual for working models, but the local slope decides it, not AUROC above 0.5. `vit_cifar100_bpp_0_01` falls slightly (0.9214 to 0.9206), and inverted models should fall toward their limit.

**Prediction for the queued sweeps.** From the validation flip histograms, $\rho_I$ is 0.69 for PSBD-TM and 0.30 for PSBD-RD (medians). The reliabilities at $k$ = 1, 3 and 20 are 0.69, 0.87 and 0.98 for TM and 0.30, 0.56 and 0.89 for RD. The PSBD-RD `_k20` sweeps should therefore gain clearly more from $k = 3$ to 20 than TM's 0.0016. The prediction is supported if the mean RD gain on the same models is at least 0.005.

## Numeric predictions for the open hypotheses

1 general threshold applies to every account that claims to carry the signal. It has to beat what confidence alone can do on the same models: $A^{*}(P_c)$ = 0.873 over the 57, or its per-attack value from the P1 bound above.

| ID | statistic and population | supported if | refuted if | correction to the memo |
|---|---|---|---|---|
| L10 | Clean fractional PSU with the target's gain redistributed, $\tilde P_c(\xi) = P_c(\xi)(1-P_t(x))/(1-P_t(\xi))$ for clean inputs with $c \ne t$, triggered scores unchanged. WaNet under PSBD-RD, 5 models. | AUROC falls by at least 0.10 | AUROC moves less than 0.02 | Not computable from the cache, which stores only $P_c$ and the argmax per pass. "Remove the target column" is undefined for triggered inputs, where $c = t$. Needs GPU passes that record $P_t$. The CPU extreme, treating every flip onto the target as caused by attraction, drops WaNet RD from 0.956 to 0.372 and all 57 from 0.888 to 0.644. That is an upper envelope that cannot decide. Shifted predictions land on the unperturbed runner-up only 10% to 35% of the time, so the destination is a default class of the model. |
| P10 | Share of clean images of the target class, from the unpaired clean split, below the q0.25 validation threshold at the adaptive rate. Backdoored models and the benign reference, pooled per dataset. | at least 0.75 on backdoored and at most 0.35 on benign | at most 0.40 on backdoored | The base rate is 0.25 by construction. P10 predicts the triggered TPR (0.909 TM, 0.825 RD), not "half". The paired split drops the target class under all-to-one, so the unpaired split must be used. |
| P2, L12 | Continuous seed statistic $1 - \frac{1}{2}\sum_{s=1,2} P_c^{(s)}(x)/P_c^{(0)}(x)$ over the 14 seed models, paired images. | AUROC at least 0.9, and clean Spearman with $\phi_{TM}$ at least 0.5 | AUROC at most $A^{*}(P_c)$ of those models, or Spearman at most 0.2 | Hard-vote disagreement of 3 seeds has AUROC $\tfrac{1}{2}(1 + a - b) \le \tfrac{1}{2}(1 + 3(1-\text{acc}))$. That is 0.68 at accuracy 0.88 and 0.575 at 0.95, so a threshold of 0.9 is unreachable as the memo states it. |
| L18 | Triggered survival by the number of surviving trigger tokens $j$ under `after_embedding_token_mask`, masks recorded, 4-token BadNets. | near 0 at $j=0$, at least 0.9 at $j=4$ | flat in $j$ | The comparison value for 1 surviving token is 0.984 (4-token models with 1 visible late token). 0.930 is the 1-token models' overall survival. At the embedding, survivors are scaled by $1/(1-p)$ in the residual stream (3.3 times at 0.7), which is not inert, so run it at a fixed rate and state the confound. |
| L6 | Smallest visible fraction keeping each prediction, nested deterministic masks at the attention input in all blocks, 256 pairs per model. | random order: triggered BadNets median about 0.16 (4-token) and 0.5 (1-token), clean median 0.3 to 0.6. Ranked order: triggered BadNets at most 0.025. | triggered medians not below clean | "At most 0.02" is below 4/196 = 0.0204. Under a random order a patch trigger's smallest fraction is the minimum of $m$ uniform ranks, median $1 - 2^{-1/m}$. Section D already puts Blend and LF above 0.1 (0.40 and 0.43 excess at fraction 0.1) and BPP below (0.62). The fraction grid must reach 0.02. |
| L7 | ASR with the top 30 of 3072 GELU units per block zeroed in blocks 9 to 12, against 30 random units. | ASR at most 0.1 | ASR above 0.9 and within 0.05 of random | none |
| L8 | Per-image Spearman of $\phi$ against the fractional drop of the block 12 CLS projection on $u$, same masks. | at least 0.7 | at most 0.3 | 83% of triggered images never flip under TM, so their $\phi$ sits at the bf16 noise floor (about 1e-3). Use float32 passes and pool clean and triggered, or restrict to $\phi > 0.01$. |
| L26 | Excess ASR retention of SIG at 10% visible tokens with whole token rows kept. Per-token probe accuracy at blocks 4 to 8 for WaNet. | SIG at least 0.3. WaNet probe near 0.5 with the CLS probe at least 0.9. | SIG at most 0.05 | none |
| L22 | `token_substitute` against token mask at the matched 0.6 rule, plus the CLS attention mass on masked positions. | within 0.02 | substitution at least 0.05 lower | All masked tokens share the key $W_k\beta + b_k$ and act as 1 attention sink whose mass grows with the number masked. Report that mass, since it separates the 2 operators independently of manifold arguments. |
| L2, P4 | AUROC and per-image $\phi(2r)/\phi(r)$ for Gaussian at $r \le 0.1$, float32, $k = 20$. | ratio about 4, rank Spearman across $r$ at least 0.9, and AUROC at small $r$ within 0.02 of the operating AUROC | AUROC at small $r$ at least 0.1 below operating | Rademacher against Gaussian cannot discriminate: in 151k dimensions both give the same projections by universality under any account. The cached bf16 ladder already reads 0.578 and 0.601 at r 0.05 and 0.1 against 0.86 at r 1.0, which provisionally refutes curvature as the carrier and needs the float32 check. |
| L4, P8 | AUROC of the per-sample training loss at its first drop, BPP and TaCT at 5%, and Kendall tau with PSBD-TM's attack ranking. | at least 0.9 and tau at least 0.6 | at most 0.7 | none |
| L15 | Ported TeCo on the Swin panel. | mean at least 0.95 | below $A^{*}(P_c)$ of Swin | none |
| L1 | Per-image Spearman between the pixel certified radius $\sigma\Phi^{-1}(\underline{p_A})$ and $\phi_{TM}$, clean and triggered separately, per model. | at least 0.5 in both | at most 0.2 on triggered BadNets | Kendall over 7 category means moves in steps of 2/21 = 0.095 and SIG has 1 model. Use the 57-model or per-image version. |
| L3 | CLS attention mass on BadNets trigger tokens in blocks 9 to 12 on matched `_sam_rho_0_1` checkpoints. | above 0.65 | at most 0.55 | "The $p^{4m}$ law keeps its exponent" is fixed by the operator. Compare the survival curve $s_j$ instead. |
| L5 | Token-level MSPC mask learned on validation, BPP. | at least 0.8 | at most 0.5 | A pixel scale reaches the blocks through the per-token offset $b - W\mu/\sigma + \mathrm{pos}_t$ (conv bias, normalization mean, position embedding) and the clamp, not only through the position embedding. |
| L9 | Survival under PSBD-TM of implanted Adaptive-Blend models at visible fraction 0.5. | above 0.95 | below 0.8 | the population is outside the 57 |
| L11 | STRIP entropy AUROC at clean overlay weight 0.7. | Blend at most 0.7 with BadNets above 0.9 | Blend above 0.85 | none |
| L13 | SCALE-UP AUROC against poison rate. | not decidable | not decidable | The theorem is a sufficient condition in a limit. A monotone rise from 1% to 10% is an extrapolation, so label the current result "extrapolation fails". |
| L14 | Target share of shifted clean predictions across the `gain_scale` ladder. | share at least 0.5 at the top factor | flat and at most 0.25 | already refuted at factor 15 (0.001 to 0.223) |
| L16 | TED restricted to the CLS token, BadNets. | drop at least 0.1 | drop below 0.02 | none |
| L19 | Clean shift at token-mask rate 0.3 on the benign references. | CIFAR-100 above Tiny by at least 0.1 | Tiny above CIFAR-100 | CIFAR against Tiny confounds resolution with class count. CIFAR-100 (32 px, 100 classes) against Tiny (64 px, 200) separates them: resolution predicts CIFAR-100 higher, class count predicts Tiny higher. The backdoored ladders at 0.3 already read keep 0.531 (CIFAR-100) against 0.499 (Tiny). |
| L20, L23 | Section A and section D protocols ported to Swin. | trigger-token masking leaves at least 0.5, and WaNet keeps at least 0.5 at 30% visible | at most 0.1 | none |
| L21 | Continuous fractional drop of $P_c$ under deterministic masking of the top 4 tokens by CLS attention in blocks 9 to 12. | AUROC at least 0.95 | at most 0.88 | With a binary change score, AUROC is $\tfrac{1}{2}(1 + a - b)$ = 0.88 at section A's $a$ = 0.80 and $b$ = 0.04, so 0.95 is unreachable with that score. |
| L24 | Triggered survival with every token except the trigger's masked in blocks 9 to 12. | at least 0.95 | at most 0.8 | none |
| L25 | Absorption shares on Swin. | within 0.03 of ViT | not applicable | Fix the metric first. The record measures an unscaled mask, and by its own formula the library `TokenMask` reads survival $\sqrt{1-p}$ (0.947, 0.835 and 0.704 at p 0.1, 0.3 and 0.5, verified). $\beta$ enters the denominator. |
| L27 | Per-model PGD share of clean inputs pushed to the target against the target share under PSBD-RD, 57 models. | Spearman at least 0.5 | at most 0.2 | none |
| P1 | $A(\phi)$ against $A^{*}(P_c)$ per model. | not applicable | PSBD-TM above $A^{*}$ on 52 of 57 (+0.080) | Replace 0.684 with 0.873 as the confidence ceiling. PSBD-RD clears it by only +0.015 (43 of 57). |

## Existing claims the derivations contradict

1. `experiments/why_token_masking_works/README.md`, section B and the hypothesis 1 verdict. "The event that breaks the prediction, all m trigger tokens masked in all 4 late blocks, has probability p^(4m)" is right about the event's frequency and wrong as the account of breakage: the event causes 7.5% of broken 4-token triggered predictions, and "0.003" is 89% from 1 model at p 0.8. The literature memo repeats the claim in its opening paragraph, P7 and L20.
2. The same README, section C, and `.claude/CLAUDE.md`. "Dropout on the trigger positions alone breaks as much as dropout everywhere" holds for the 12-model mean only. Per model the mean absolute difference is 0.223, and in 8 of 12 models all-token dropout survives better than $\min(s_T, s_{\text{rest}})$.
3. `experiments/residual_stream_mechanism/layernorm_absorption.py` (docstring) and the H47 and L25 claim that absorption explains the operator gap. Measured absorption is linear, so it cancels at matched shift. `AttentionInputMinusMlpInputTokenMask` (+0.110) compares token mask with token mask at 2 sites before a norm with equal measured absorption (-0.002 and 0.029). `GaussianMinusTokenMaskMlp` (+0.051) compares Gaussian after `ln_2` with token mask before it, so the memo's L2 phrase "at before_mlp isotropic noise beats token masking" misdescribes it.
4. `experiments/retention_curves/README.md`. The stated protocol reads ASR retention where CA retention first reaches 0.75, but the table's CA retention is 0.527 (dropout) and 0.718 (Fourier), so the modalities are not matched. `measure.py` is absent from the directory. ASR retention above 1 (the Adaptive-Blend "anti-fragility") is uncorrected for collapse onto the target, which section D showed produces a raw retention of 1.04 on WaNet, so it needs the excess form.
5. `experiments/theory_predictions/README.md`. The closed form $\operatorname{tr}\nabla_z^2 p_c = 2p_c(\|p\|^2 - p_c)$, its 2-class corollary and the 0.7887 turning point are correct (rederived). The phrase "sharpens the softmax most for the least confident samples" is imprecise, because temperature sharpening leaves $p_c$ unchanged at exact ties and at $p_c = 1$ and peaks between. The zero-pass reconstruction also ignores the head bias, which scaling does not touch, and its 0.0013 error says the bias is small.
6. The literature memo's P1 paragraph. Confidence carries far more than 0.684 once its non-monotone likelihood ratio is used (0.873).

## Idealizations

In order of their effect on the conclusions:

- An ideal PRNG with independent draws. Models that share a rate drew identical masks, so pooled counts are not independent replicates.
- First-order margin expansions for dropout. At compounded rates $p_{\text{eff}}(16) = 0.78$ these are heuristic.
- The chord formula assumes $\gamma = 1$, $\beta = 0$ and noise orthogonal to the token. That is exact to $O(1/\sqrt{d})$ for isotropic noise, and the real $\gamma$ and $\beta$ are not unit.
- The OR and AND read models for the operator gap are qualitative. They predict sign and ordering, not magnitude.
- The logistic critical-rate model is fitted to dataset means of the ladder and treats the default-class floor as constant.
- $A^{*}(P_c)$ is estimated with 40 bins and cross-fitting, so residual optimism is small but nonzero.
- Every small-rate number is limited by bf16 logits (a probability noise floor of about 1e-3). That affects the curvature plateau and the L8 correlation.
