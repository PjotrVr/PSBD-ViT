# Evidence surplus, a theory of why prediction-shift detection works

This note turns the project's unifying hypothesis into a definition, a set of derivations and a set of experiments. The hypothesis: backdoor training makes a triggered decision carry far more evidence than it needs along the axis a PSBD perturbation removes, so the decision survives random removal, while a clean decision carries just enough and breaks. PSBD should then fail wherever the backdoor is a conjunction with image content, or wherever its evidence is relational along the removed axis. The reader is assumed to know PSBD (Li et al., [arXiv 2406.05826](https://arxiv.org/abs/2406.05826)), the placements PSBD-TM and PSBD-RD and the repository's experiments.

Written on 2026-09-30 against commit `fd09d3b`. Headline numbers are quoted by their macro in `paper/headline.tex`. Per-model and per-group numbers come from the files named beside them, read on the date above. Experiments A, A2, A3, B, C and D of `experiments/evidence_surplus/` were still running when this was written, so the section on them states predictions and marks every existing reading as a postdiction.

The note has 11 sections: the evidence on disk, the existing names for the idea, the definitions, the aggregation structure along the removed axis, surplus growth during training, conjunctions, relational evidence, the assumptions, predictions for the running experiments, the experiment design for the login GPU windows of 2026-10-01 to 2026-10-05 and how the theory strengthens the paper. The references close the note. The main correction to the hypothesis as first stated is in the aggregation section: the surplus PSBD-TM reads on patch triggers is redundancy of carriers that each suffice, closer to causal overdetermination than to a large margin, and the magnitude reading of surplus describes global triggers.

## Evidence on disk

The records already on disk constrain any account of PSBD on ViT-B/16 and Swin-S. The table lists the readings this note uses, each with its source. Group means are from `experiments/why_psbd_works/README.md` (hit-only AUROC, 10 passes, adaptive rate) unless another file is named.

| reading | value | source |
|---|---|---|
| PSBD-TM and PSBD-RD, panel means | `\HeadlineAurocAdaptive` 0.963 and `\PublishedAurocAdaptive` 0.885 | `paper/headline.tex` |
| triggered minus clean median margin retention under PSBD-TM | BadNets 0.451, Blend and LF 0.711, WaNet 0.497, BPP 0.802, TaCT 0.181 (ViT) | H-margin |
| margin-matched AUROC under PSBD-TM | 0.996 on BadNets, 0.998 on Blend and LF, 0.805 on WaNet (ViT) | H-margin |
| median unperturbed margin, clean against triggered | BadNets 6.18 against 12.46, WaNet 6.51 against 9.91 (ViT) | H-margin |
| PSBD-TM beats the best function of confidence $A^*(P_c)$ | by 0.082 on average, on 50 of 54 models | claim C13 |
| survival of a triggered BadNets answer with $J$ of 4 late blocks fully masked | 0.987, 0.965, 0.917, 0.748, 0.216 for $J$ = 0 to 4 | `docs/why-psbd-works-theory.md` |
| trigger tokens masked in all 12 blocks against as many random tokens | 0.002 against 1.000 of triggered answers kept | claim C7 |
| clean critical rate, median per dataset | CIFAR-10 0.411, CIFAR-100 0.302, GTSRB 0.326, Tiny 0.288, logistic spread 0.16 to 0.20 | clean fragility table |
| critical-rate account, AUROC against $A^\star = P(p^*_{\text{trig}} > p^*_{\text{clean}})$ | Spearman 0.883 over 558 (model, placement) pairs | critical rate section |
| operator gap at the attention input, Gaussian minus token mask | `\GaussianMinusTokenMaskAttentionNorm` $-0.188$, most negative on BadNets ($-0.365$), positive on WaNet ($+0.036$) | claim C18 |
| WaNet, PSBD-TM against PSBD-RD | 0.807 against 0.990 on ViT, 0.987 against 0.654 on Swin | operator tables |
| all-to-all BadNets, PSU AUROC | `\ATwoaPsuAuroc` 0.411 against `\ATwooPsuAuroc` 0.891 on all-to-one | `paper/headline.tex` |
| trigger-conditional TaCT under PSBD-TM | 0.983 (ViT, 3 models), with flip-based $A^\star$ up to 0.93 | operator table, critical rate section |
| experiment A, directional surplus factor at the last block, median | 1.12 on `vit_cifar10_badnet_a2o_0_01`, 1.31 on `vit_tiny_badnet_a2o_0_05`, 1.24 on `resnet18_gtsrb_badnet_a2o_0_1` | `results/_experiments/evidence_surplus/runs/`, first 3 records |
| experiment B, hit-only AUROC at the weakest dose still reaching ASR 0.5 | 0.895 to 0.979 on BadNets and TaCT, 0.405 and 0.375 on `vit_cifar10_blend_0_1` and `vit_cifar10_wanet_0_1` | `experiments/evidence_surplus/trigger_dose/README.md` |

3 of these readings shape the theory more than the others. Margin matching leaves PSBD-TM's AUROC near 1 on patch and blend triggers, and the first records of experiment A put the directional surplus factor of BadNets only slightly above 1. So the size of the triggered margin does not carry the separation on patch triggers. The survival curve over $J$ says a triggered BadNets answer survives until nearly every late read of the trigger is gone, which is the signature of carriers that each suffice. The conjunction and relational failures (all-to-all below chance, WaNet under token masking) are where the hypothesis predicts failure, and TaCT is the case the first form of the hypothesis gets wrong.

## Existing names for the idea

No paper found uses the term "evidence surplus" for a classifier's decision, in searches of arXiv, Google Scholar style web search and the extracted texts of the papers below. The idea is covered in parts by at least 8 existing notions. Each paragraph below states the notion, what this project can use from it (a formal result, a definition, an experiment or a prediction) and how it differs from the evidence-surplus account. The closest are ranked in the table at the end of the section.

**Causal overdetermination and degree of responsibility.** In the counterfactual theory of causation an effect is overdetermined when 2 or more causes are each sufficient for it, as in the 2 rocks that hit a bottle at once ([Stanford Encyclopedia of Philosophy, counterfactual theories of causation](https://plato.stanford.edu/entries/causation-counterfactual/)). Chockler and Halpern ([JAIR 2004](https://www.jair.org/index.php/jair/article/view/10386)) quantify it: the degree of responsibility of a cause is $1/(N+1)$, with $N$ the smallest number of other variables that must change before the effect depends on it counterfactually. Their example is a vote won 11 to 0, where each voter has responsibility 1/6 since 5 votes must flip before any single vote is critical. What we can use is the definition itself: a decision that holds while at least $r$ of $n$ evidence units survive gives every unit responsibility $1/(n-r+1)$, and PSBD's random removal is the probabilistic version of their contingency count. Chockler et al. ([ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Chockler_Explanations_for_Occluded_Images_ICCV_2021_paper.html)) apply the same causal machinery to image classifiers with occluded inputs, where several disjoint regions each explain the output. The difference from our account is that overdetermination is a property of a single structural model and a fixed contingency, while evidence surplus is defined relative to a perturbation family and read as a critical removal fraction.

**Sufficient input subsets and overinterpretation.** Carter et al. ([AISTATS 2019](https://proceedings.mlr.press/v89/carter19a.html)) define a sufficient input subset (SIS) as a minimal subset of features whose values alone keep the model's confidence above a threshold $\tau$ with every other feature masked, found by backward selection. Carter et al. ([NeurIPS 2021](https://arxiv.org/abs/2003.08907)) then show that CIFAR-10 models classify 5% pixel subsets chosen by backward selection almost as accurately as full images, while 5% random subsets give chance accuracy (9.98% on 10 classes). What we can use is the distinction this draws: a ranked sufficient subset measures the best case for the model, and random removal measures the typical case, which is what PSBD reads. Our own H-low-dim reading already has both forms, and it gives a ranked sufficient share of 0.005 for triggered BadNets images against 0.600 for clean ones and a random share of 0.425 against 0.600 (ViT). Wäldchen et al. ([JAIR 2021](https://www.jair.org/index.php/jair/article/view/12359)) define a $\delta$-relevant set, a subset that fixes the output with probability at least $\delta$ when the rest is drawn at random, and show that finding a small one is hard. The $\delta$-relevant set is the closest formal object to our keep probability, but it asks for the smallest set that suffices, while surplus asks how much random removal a given decision tolerates.

**Trigger dominance, input agnosticism and over-confidence.** STRIP (Gao et al., [ACSAC 2019](https://arxiv.org/abs/1902.06531)) rests on the premise that a trojaned input's prediction is input-agnostic, so that "regardless of strong perturbations on the input image, the predictions of all perturbed inputs tend to be always consistent". Tang et al. ([USENIX Security 2021](https://www.usenix.org/conference/usenixsecurity21/presentation/tang-di)) name the premise behind STRIP and other defenses the dominance of the trigger in the representation, and design TaCT, a source-specific trigger with cover samples, to break it. Peng, Xiong et al. ([arXiv 2202.11203](https://arxiv.org/abs/2202.11203)) call the same property the over-confidence of dirty-label backdoors, and prove that relabeling a poisoned image to the target with probability $p_t(x)$ gives the model's target probability $\beta(x) = \alpha(x) + (1-\alpha(x))\,p_t(x)$ in expectation. What we can use: Tang et al. anticipated our conjunction failure mode in 2021, and Peng, Xiong et al. give the construction and the calibration theorem for the predictivity control of the experiment design. The difference is that dominance and over-confidence are stated as properties of the triggered input or of its confidence, while the project's margin-matched AUROC and the $A^*(P_c)$ ceiling show that confidence does not carry PSBD-TM's separation on our models. Surplus is a property of the decision along a removal axis.

**The backdoor as the strongest feature.** Khaddaj et al. ([ICML 2023](https://proceedings.mlr.press/v202/khaddaj23a.html)) argue that a backdoor cannot be told apart from a natural feature without an assumption, and assume the trigger is the strongest feature in the training set. They define the output function $g_\phi(k)$ of a feature $\phi$ as the expected model output on examples carrying $\phi$ when exactly $k$ training examples carry it, and the strength $s_\phi(k) = g_\phi(k+1) - g_\phi(k)$. What we can use is the definition and the experiment: our weight-decay derivation below predicts $g_\phi(k) \approx \ln k$ plus a constant for a perfectly predictive trigger, so $s_\phi(k) \approx 1/k$, which a poison-rate series measures. The difference is that strength is read over training sets and says nothing about which perturbation a decision survives.

**Backdoor smoothing and randomized smoothing.** Grosse et al. ([Computers and Security 2022](https://doi.org/10.1016/j.cose.2022.102814)) measure the entropy $W_\sigma(x)$ of the class distribution under Gaussian input noise and find that the decision function is smoother around triggered inputs than around clean ones, that smoothness correlates with the backdoor's accuracy and that some adversarial perturbations produce the same smoothness. This is the closest published observation of the phenomenon PSBD exploits, made in input space with Gaussian noise. Cohen et al. ([ICML 2019](https://arxiv.org/abs/1902.02918)) certify a smoothed classifier at $\ell_2$ radius $\sigma\,\Phi^{-1}(p_A)$ in the binary case, and Levine and Feizi ([AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/5888)) certify an $\ell_0$ radius for randomized ablation, where the smoothed class probability $p_i(x)$ moves by at most $\Delta = 1 - \binom{d-\rho}{k}/\binom{d}{k}$ when $\rho$ of $d$ features change and $k$ are kept. Salman et al. ([CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Salman_Certified_Patch_Robustness_via_Smoothed_Vision_Transformers_CVPR_2022_paper.html)) show that ViTs handle heavily ablated inputs gracefully because they can drop masked tokens. What we can use: PSBD-TM's keep probability at rate $p$ is a smoothed class probability in Levine and Feizi's sense, so a certified ablation radius is a surplus reading with a guarantee, and Wang et al. ([CVPR 2020 workshop, arXiv 2002.11750](https://arxiv.org/abs/2002.11750)) already report that randomized smoothing defends poorly against backdoors, which is the same surplus seen from the defender's other side. The difference is that smoothing studies robustness as a goal, while surplus uses the gap between triggered and clean robustness as the signal.

**Shortcut learning, simplicity bias and availability.** Geirhos et al. ([Nature Machine Intelligence 2020](https://arxiv.org/abs/2004.07780)) define shortcuts as decision rules that perform well on standard benchmarks and fail to transfer, and a trigger is the textbook shortcut. Shah et al. ([NeurIPS 2020](https://arxiv.org/abs/2006.07710)) show that simplicity bias can be extreme: networks rely exclusively on the simplest predictive feature and stay invariant to complex ones, measured by randomizing a coordinate block (the S-randomized accuracy), and this reliance comes "at the expense of very small margin" in input space along the simple feature. Huh et al. ([TMLR, arXiv 2103.10427](https://arxiv.org/abs/2103.10427)) add that deeper networks favor low effective rank solutions. Hermann and Lampinen ([NeurIPS 2020](https://arxiv.org/abs/2006.12433)) and Hermann et al. ([ICLR 2024](https://arxiv.org/abs/2310.16228)) separate a feature's predictivity (how reliably it indicates the label) from its availability (how easily it is extracted) and show that models prefer the more available feature even when it is less predictive. What we can use: Shah et al.'s randomization experiment is the template for experiment C's collages, and Hermann et al.'s availability and predictivity grid is the template for the predictivity control. The small input-space margin of Shah et al. is also the reason surplus must be axis-relative: a patch trigger's decision has a small margin against a change of the trigger pixels and a very large one against removal of content.

**Implicit margin maximization and gradient starvation.** Soudry et al. ([JMLR 2018](https://jmlr.org/papers/v19/18-188.html)) prove that gradient descent on the logistic loss over linearly separable data gives $w(t) = \hat w \ln t + \rho(t)$, with $\hat w$ the $\ell_2$ max-margin vector and $\lVert\rho(t)\rVert = O(\ln\ln t)$ (Theorem 3), so the margin keeps growing after the training error is 0. Ji and Telgarsky ([COLT 2019](https://proceedings.mlr.press/v99/ji19a.html)) extend this to data that is neither separable nor strongly convex: the data splits uniquely into a maximal linearly separable subset, along which the iterates diverge in the max-margin direction, and the rest, whose risk has a bounded unique optimum. Lyu and Li ([ICLR 2020](https://arxiv.org/abs/1906.05890)) prove that a smoothed normalized margin increases for homogeneous networks once the loss is small, Ji and Telgarsky ([ICLR 2019](https://arxiv.org/abs/1810.02032)) prove layer alignment for deep linear networks, and Zhang, Zou et al. ([NeurIPS 2024](https://arxiv.org/abs/2406.10650)) prove that Adam converges to the maximum $\ell_\infty$ margin on separable data under diminishing learning rates. Pezeshki et al. ([NeurIPS 2021](https://arxiv.org/abs/2011.09468)) show in a linearized network with ridge-regularized cross-entropy that a feature learned first starves the gradient of the others, with a fixed-point response $z^*_j = s^2 W(\lambda^{-1}s^2 + 1)/(s^2 + \lambda)$ for features of equal strength $s$, $W$ the Lambert function. What we can use is the theorem-level backbone of the training argument: the trigger is a separable part of the data in exactly Ji and Telgarsky's sense, and the weight-decay equilibrium below has Pezeshki et al.'s Lambert form. The difference is that these results describe margins and not the axis along which evidence is spread.

**Learning order of backdoors.** Arpit et al. ([ICML 2017](https://arxiv.org/abs/1706.05394)) find that networks learn simple patterns before memorizing, and Nakkiran et al. ([NeurIPS 2019](https://arxiv.org/abs/1905.11604)) that SGD learns functions of increasing complexity and retains its early linear classifier. Li et al. ([NeurIPS 2021](https://arxiv.org/abs/2110.11571)), in Anti-Backdoor Learning (ABL), report that the training loss of poisoned examples drops much faster than that of clean ones, faster for stronger attacks and higher poison rates, and slower for clean-label attacks (SIG and 2 others). Cinà et al. ([IJMLC, arXiv 2106.07214](https://arxiv.org/abs/2106.07214)) study backdoor learning curves across poison rate and model complexity. What we can use is ABL's per-subset loss curve, which `--record-sample-loss` in `training/loop.py` already records. Its clean-label exception is also useful, since our theory explains it as gradient starvation of the trigger by target-class content. The difference is that learning speed is about when the trigger is fit, and surplus is about what happens after it is fit.

**Robust features, single directions and corruption consistency.** Ilyas et al. ([NeurIPS 2019](https://arxiv.org/abs/1905.02175)) call a feature $f$ $\rho$-useful when $\mathbb{E}[y\,f(x)] \geq \rho$ and $\gamma$-robustly useful when $\mathbb{E}[\inf_{\delta \in \Delta(x)} y\,f(x+\delta)] \geq \gamma$ for a perturbation set $\Delta$. Evidence surplus is the random-perturbation, per-input analogue of $\gamma$-robust usefulness. Morcos et al. ([ICLR 2018](https://arxiv.org/abs/1803.06959)) measure cumulative ablation curves (units set to 0 at random, in growing number) and find that networks that memorize depend more on single directions and generalize worse. Their cumulative ablation curve is our keep curve with the operator applied to units, and their memorization result predicts that the exceptions of the predictivity control will be low-surplus decisions. TeCo (Liu et al., [CVPR 2023](https://arxiv.org/abs/2303.18191)) observes that triggered inputs are very robust to some corruptions and fragile to others while clean inputs degrade uniformly, and speculates that the dual-target backdoor loss makes the model learn a shortcut that "is not always robust in image space". Chen, Wu et al. ([NeurIPS 2022](https://openreview.net/forum?id=AsH-Tx2U0Ug)) find the opposite sign under geometric transformations: poisoned samples' features are more sensitive to rotation and affine transforms. SCALE-UP (Guo et al., [ICLR 2023](https://arxiv.org/abs/2302.03251)) and IBD-PSC (Hou et al., [ICML 2024](https://arxiv.org/abs/2405.09786)) exploit consistency under pixel and parameter scaling. What we can use: TeCo and Chen et al. are the direct published evidence that surplus is axis-relative, large along some perturbations and absent or negative along others, which the definition below makes the central property.

**Explicit logic-gated triggers.** Shen, Cai et al. ([arXiv 2602.03040](https://arxiv.org/abs/2602.03040)) build an m-of-n co-occurrence gate into a ViT's first block and observe that the gated backdoor survives PatchDrop-style defenses because "stamped triggers remain to satisfy the co-occurrence gate" when redundancy is high. What we can use is their construction as the opposite end of the conjunction experiment: with $m < n$ the gate is redundant and survives removal, and with $m = n$ over many components it is a pure conjunction, which our bound predicts PSBD cannot detect.

| rank | existing notion | closest part of our account | what it lacks |
|---|---|---|---|
| 1 | causal overdetermination, degree of responsibility (Chockler and Halpern) | the disjunctive form, many carriers that each suffice | no perturbation family, no probability, no training account |
| 2 | backdoor smoothing (Grosse et al.), trigger dominance (Tang et al.), over-confidence (Peng, Xiong et al.) | the phenomenon on triggered inputs | not axis-relative, confidence-based, no structure |
| 3 | randomized ablation certificates (Levine and Feizi) | keep probability under random token removal | a robustness goal, not a detection signal |
| 4 | sufficient input subsets and $\delta$-relevant sets | sufficiency of part of the input | best-case subsets, not typical random removal |
| 5 | implicit margin bias, gradient starvation, strongest feature | why training builds surplus | margins only, no removal axis |

The closest existing name for what PSBD-TM reads on patch triggers is causal overdetermination. For global triggers the closest is backdoor smoothing, read with removal in place of Gaussian noise. "Evidence surplus" is new as a term, and 3 parts of the account are new as far as these searches reach: the definition relative to a perturbation family through the critical removal fraction, the split into a magnitude route and a renormalization route set by attention, and the conjunction bound with its polarity. It would be accurate to present evidence surplus as the umbrella term and to name its disjunctive form overdetermination, citing Chockler and Halpern.

## Definitions

The definitions below extend the notation of `docs/why-psbd-works-theory.md`. The margin is written $m(x)$ as in `experiments/why_psbd_works/`, so the number of trigger tokens, which the theory note calls $m$, is written $\lvert T\rvert$ here.

$$\kappa_x(p) = P_\xi\Big[\arg\max_i z_i\big(x;\xi_p\big) = c(x)\Big], \qquad p^*_x = \inf\big\{p : \kappa_x(p) < \tfrac{1}{2}\big\}$$

| symbol | meaning |
|---|---|
| $x$ | an input |
| $c(x)$ | the unperturbed predicted class |
| $\xi_p$ | a random draw of the perturbation family at rate $p$ (token mask, dropout, channel mask, Gaussian noise) at a fixed position |
| $z_i(x;\xi_p)$ | the logit of class $i$ on that perturbed pass |
| $\kappa_x(p)$ | the keep probability, the chance a pass keeps the answer |
| $p^*_x$ | the critical removal fraction of $x$ for this family, where the keep probability crosses 1/2 |

The ladder estimator of $p^*_x$ is the smallest cached rate at which most of the $k$ passes change the answer (`defenses.decision.load_critical_rate_from_disk`), which is what `experiments/why_psbd_works/critical_rate.py` reads. Evidence surplus for $x$ and a family is $p^*_x$ itself, read against the distribution of $p^*$ over clean inputs of the same model. PSBD at any placement is then a 2-sample comparison of $p^*$ between triggered and clean inputs, which the critical-rate reading supports at a Spearman of 0.883.

The surplus ratio needs a model of how evidence adds up. Let the decision rest on evidence units $u \in U$ along the removed axis (tokens read by attention in a block, coordinates of the residual stream, blocks). The family removes each unit independently with probability $p$, with keep indicator $\mu_u \sim \mathrm{Bernoulli}(1-p)$. 2 structural forms cover the cases the project meets.

$$\text{additive: } m(x;\xi) = b + \sum_{u \in U} \mu_u e_u, \qquad S = \frac{\sum_u e_u}{-b} = 1 + \frac{m(x)}{\lvert b\rvert}$$

$$\text{r-of-n: } \text{keep} \iff \sum_{u \in U} \mu_u \geq r, \qquad S = \frac{n}{r}, \qquad \text{responsibility of a unit} = \frac{1}{n - r + 1}$$

| symbol | meaning |
|---|---|
| $e_u$ | the contribution of unit $u$ to the margin of $c$ over the runner-up, positive for own-class evidence and negative for competing evidence |
| $b$ | the null margin, the margin of $c$ when every unit is removed, negative when the fully perturbed input goes to another class |
| $m(x) = b + \sum_u e_u$ | the unperturbed margin |
| $n = \lvert U\rvert$ | the number of units that each carry a sufficient share, in the r-of-n form |
| $r$ | the number of such units the decision needs |
| $S$ | the surplus ratio, evidence carried divided by evidence needed |

The additive form is a threshold-sum over many weak units, and the r-of-n form with $r = 1$ is a disjunction over $n$ sufficient carriers, overdetermination in Chockler and Halpern's sense. The directional surplus factor of experiment A, $F_x = \langle h(\tilde x) - h(x), u\rangle / \alpha^*_x$ with $\alpha^*_x$ the shift along the backdoor direction $u$ that sends the clean twin to the target, is a magnitude reading of $S$ for a single direction in the stream. It is the right reading when evidence is additive along that direction and the wrong one when it is replicated across carriers.

## Aggregation structure along the removed axis

**Result.** The critical removal fraction depends on the surplus ratio and on how evidence aggregates along the removed axis. Additive evidence gives $p^* = 1 - 1/S$ whatever the number of units. A disjunction over $n$ sufficient carriers gives $p^* = 2^{-1/n}$. Rescaled dropout on a linear readout preserves the mean margin and gives no majority flip at any rate, so its surplus is the signal-to-noise ratio $R$. Token masking at the attention input implements a disjunction for any carrier set whose attention score is far above a masked token's, through softmax renormalization, and a threshold-sum for diffuse evidence. That is why patch triggers show surplus under PSBD-TM with a directional factor near 1, why global triggers need a large magnitude and why Gaussian noise at the same site loses most on patch triggers.

**Assumptions.**

1. Keep indicators are independent across units, blocks and passes. This holds by construction for `TokenMask`, `nn.Dropout` and the channel mask (`defenses/operators.py` draws a fresh Bernoulli tensor per call).
2. Within a model of the margin, the runner-up class stays the same across passes. This fails when masking sends the input to a default class, which is what the null margin $b$ absorbs in the additive form.
3. The margin is linear in the unit contributions for the additive and rescaled forms. LayerNorm, softmax and the MLPs make this a local approximation.
4. For the attention derivation, a masked token at the attention input enters as the LayerNorm bias, so every masked token has the same key and value, and its residual entry is kept (claim C8).

**Additive removal without rescaling.** Token masking at `before_attention_norm` multiplies kept tokens by $1/(1-p)$, but the LayerNorm that follows divides each token by its own scale, so a kept token is read unchanged and a masked one is read as the bias. The evidence is therefore removed and not rescaled.

$$
\begin{aligned}
\mathbb{E}\big[m(x;\xi)\big] &= b + (1-p)\sum_u e_u && \text{linearity of expectation, assumption 3} \\
\operatorname{Var}\big[m(x;\xi)\big] &= p(1-p)\sum_u e_u^2 && \text{assumption 1, Bernoulli variance} \\
\kappa_x(p) &\approx \Phi\!\left(\frac{(1-p)\sum_u e_u + b}{\sqrt{p(1-p)\sum_u e_u^2}}\right) && \text{normal approximation, many units (Lyapunov)} \\
\kappa_x(p^*) = \tfrac{1}{2} &\iff (1-p^*)\sum_u e_u = -b && \Phi(0) = \tfrac{1}{2} \\
p^* &= 1 - \frac{-b}{\sum_u e_u} = 1 - \frac{1}{S} = \frac{m(x)}{m(x) + \lvert b\rvert} && S = 1 + m/\lvert b\rvert
\end{aligned}
$$

The number of units drops out of $p^*$ and sets only the steepness of $\kappa_x$ at $p^*$, which is $\sqrt{N_{\text{eff}}/(2\pi\,p^*(1-p^*))}$ with $N_{\text{eff}} = (\sum_u e_u)^2/\sum_u e_u^2$ when every $e_u$ is positive. Many units make each image's keep curve a near step at its own $p^*_x$, so the logistic spread of 0.16 to 0.20 across clean images is a spread of $S_x$ across images, which matches the theory note's finding that about 95% of the variance of the critical rate lies between images. The measured clean medians give $S = 1/(1-p^*_{50})$ of 1.70 (CIFAR-10), 1.43 (CIFAR-100), 1.48 (GTSRB) and 1.40 (Tiny), computed from the clean fragility table. In the coordinator's threshold form this is a needed share $\theta = 1/S$ of 0.59 to 0.71 of the units.

**The r-of-n form and its 2 limits.** With $n$ equal units and a need of $r$, the keep probability is exact.

$$\kappa(p) = P\big(\mathrm{Bin}(n, 1-p) \geq r\big), \qquad r = 1: \ \kappa(p) = 1 - p^{n},\ p^* = 2^{-1/n}, \qquad n \to \infty: \ p^* \to 1 - \frac{r}{n} = 1 - \frac{1}{S}$$

| $n$ | $r$ | $S = n/r$ | exact $p^*$ | $1 - 1/S$ |
|---|---|---|---|---|
| 16 | 1 | 16 | 0.958 | 0.938 |
| 16 | 4 | 4 | 0.775 | 0.750 |
| 16 | 8 | 2 | 0.531 | 0.500 |
| 16 | 16 | 1 | 0.042 | 0 |
| 196 | 137 | 1.43 | 0.304 | 0.301 |

The table was computed by root finding on the binomial tail. The coordinator's OR figure is correct: 4 trigger tokens read in 4 late blocks give $n = 16$ and $p^* = 2^{-1/16} = 0.958$, and a 1-token Tiny trigger gives $2^{-1/4} = 0.841$. The pure OR overstates triggered survival, since the theory note measured graded survival over $J$ (0.987 to 0.216) and only 7.5% of broken BadNets answers came from the all-masked event. The attention derivation below gives the graded form.

**Rescaled dropout on a linear readout.** `nn.Dropout` at `post_residual` divides kept coordinates by $1-p$, so the mean margin is preserved.

$$
\begin{aligned}
m(x;\xi) &= \sum_u \frac{\mu_u}{1-p}\,e_u && \text{inverted dropout on the units, bias not dropped} \\
\mathbb{E}\big[m(x;\xi)\big] &= m(x), \qquad \operatorname{Var}\big[m(x;\xi)\big] = \frac{p}{1-p}\sum_u e_u^2 && \text{assumption 1} \\
\kappa_x(p) &\approx \Phi\!\left(R\sqrt{\tfrac{1-p}{p}}\right), \qquad R = \frac{\sum_u e_u}{\lVert e\rVert_2} = \frac{\sum_u e_u}{\sum_u \lvert e_u\rvert}\cdot\frac{\sum_u \lvert e_u\rvert}{\lVert e\rVert_2} && \text{normal approximation}
\end{aligned}
$$

The coordinator's form (1) is right, with $R$ the product of a coherence $\sum_u e_u / \sum_u \lvert e_u\rvert$ and $\sqrt{N_{\text{eff}}}$. It has a consequence the first statement missed: $\kappa_x(p) > 1/2$ at every $p < 1$ whenever $m(x) > 0$, so the majority-flip critical rate does not exist under this model. The critical rate has to be read at another keep level $s$, $p^*_s = R^2/(R^2 + z_s^2)$ with $z_s = \Phi^{-1}(s)$, for example 0.585 at $R = 1$ and 0.850 at $R = 2$ for $s = 0.8$. $R$ is scale-free, so under rescaled dropout a larger margin helps only through coherence. Real PSBD-RD caches do show majority flips on clean images, so the network's nonlinearity after the dropped layer supplies a drift toward a default class that this linear model lacks.

**Token masking at the attention input.** At the attention of a block, token $t$ has key score $s_t$ and value $v_t$ as read by a query such as the class token. A masked token has the bias key score $s_\varnothing$ and value $v_\varnothing$. For a set $M$ of masked tokens:

$$
o(M) = \frac{\sum_{t \notin M} e^{s_t} v_t + \lvert M\rvert\, e^{s_\varnothing} v_\varnothing}{\sum_{t \notin M} e^{s_t} + \lvert M\rvert\, e^{s_\varnothing}}
$$

| symbol | meaning |
|---|---|
| $o(M)$ | the attention output for the query with the tokens in $M$ masked |
| $s_t$, $v_t$ | score and value of token $t$ |
| $s_\varnothing$, $v_\varnothing$ | score and value of a masked token, the same for every masked token by assumption 4 |
| $T$ | the carrier set, the trigger's tokens |
| $\Delta$ | the score gap $s_t - s_\varnothing$ of a carrier, taken equal for all carriers and equal to the gap over content tokens |

2 regimes follow from the formula. When the carriers' scores exceed the masked score by a large $\Delta$, masking some carriers moves their weight onto the remaining carriers, and the read stays near the carriers' mean value until the last carrier in the block is masked. That is a disjunction within the block, and softmax normalization is what produces it. When the evidence is spread over many tokens whose scores are close to $s_\varnothing$, a masked token keeps its share of attention and contributes $v_\varnothing$, so the read is diluted toward the null value in proportion to the masked share. That is the additive form with $p^* = 1 - 1/S$.

With content tokens and masked tokens at the same score, $N = 196$ patch tokens, $\lvert T\rvert = 4$ carriers and $j$ of them masked, the attention share left on the carriers is $(\lvert T\rvert - j)e^{\Delta} / \big((\lvert T\rvert - j)e^{\Delta} + N - \lvert T\rvert + j\big)$. The values below were computed from that expression.

| share on carriers, unmasked | $\Delta$ | $j = 1$ | $j = 2$ | $j = 3$ |
|---|---|---|---|---|
| 0.50 | 3.87 | 0.855 | 0.662 | 0.395 |
| 0.90 | 6.07 | 0.967 | 0.907 | 0.766 |
| 0.99 | 8.47 | 0.997 | 0.990 | 0.970 |

Each entry is the read's size relative to the unmasked read. The read decays gradually with the number of carriers masked, and the decay shrinks as $\Delta$ grows, which is the graded survival the theory note measured over $J$. The manifestation experiment measured the class token's attention on BadNets trigger tokens in blocks 9 to 12 growing 74 to 450 times from the clean to the triggered image, which places patch triggers in the second or third row. Across the 4 late blocks the reads write into the residual stream by addition, so the decision is a threshold-sum over blocks of reads that are each intact or nearly so. This is the structure behind the $J$ curve, and it is why the directional factor of experiment A can sit near 1 while $p^*$ is high: a read that survives renormalization still writes nearly its full vector.

**Gaussian noise at the same site.** Additive noise is not removal. It degrades every read in proportion to the noise and has no renormalization, so a decision carried by 4 tokens gets no protection from the concentration of attention, while diffuse evidence averages the noise over many tokens. The account predicts that the gap between Gaussian noise and token masking at the attention input is most negative on patch triggers and smallest on global ones. The measured order (BadNets $-0.365$, TaCT $-0.242$, LF $-0.161$, BPP $-0.154$, Blend $-0.116$, WaNet $+0.036$, claim C18) agrees, but it was on disk before this account, so it counts as a postdiction and not a test.

**The coordinator's formal core, checked.** Point (1) holds for rescaled dropout, with the correction that it gives no majority-flip $p^*$. Point (2) holds. The OR over $n$ carriers gives $2^{-1/n}$ (0.958 at $n = 16$) and the threshold-sum gives $1 - \theta$ for large $n$, with the exact binomial values in the table above for finite $n$. The softmax derivation adds the mechanism by which attention implements the OR. Point (3) is the training argument of the next section. Point (4) is proved below as $p^*_{\wedge} \leq \min(p^*_A, p^*_B)$, with the addition that a veto conjunction escapes it. Point (5) holds before assembly as an AND over pairs, and after assembly the form depends on the operator, as the relational section shows.

## Surplus growth during training

**Result.** Cross-entropy training on a trigger that predicts the target perfectly grows the triggered margin without bound in the absence of weight decay, as $\ln(\rho\,t)$ under gradient descent, because the poisoned subset is a separable part of the data in Ji and Telgarsky's sense. With coupled weight decay $\lambda$ the growth stops at an equilibrium $m^* + c = W\big(\rho a^2 e^{c}/\lambda\big)$, whose slope in $\ln\rho$ is $W/(1+W)$, between 0.84 and 0.94 for plausible values. Clean test margins stay bounded, because clean content overlaps between classes and a clean test image is a new draw, while a triggered test image carries exactly the trigger seen in training. The same logistic factor that grows the margin grows the attention gap $\Delta$ on the trigger's tokens, so training builds both the magnitude route and the renormalization route. A trigger that is not perfectly predictive (label noise on the trigger, TaCT's cover samples, clean-label poisoning) has a finite optimum and a capped surplus.

**Assumptions.**

1. The trigger adds a feature that is present on every poisoned training image and absent from every clean one, so it is orthogonal to the clean data. This holds for fixed triggers (BadNets, Blend, LF, BPP) and approximately for WaNet with its noise mode.
2. A scalar model for the trigger's contribution: the margin on a poisoned image is $m = w a - c$, with $w$ the trigger weight, $a > 0$ its readout gain and $c \geq 0$ the source content's counter-evidence, equal across poisoned images.
3. The loss is the mean cross-entropy over $N$ training images, $n_p = \rho N$ of them poisoned, plus coupled L2 weight decay $\tfrac{\lambda}{2}w^2$, as in the project's Adam recipe.
4. Training runs long enough to approach the stationary point. With 15 epochs this is an assumption the dynamics runs test.

**Unregularized growth.** Under gradient flow with step $\eta$ the trigger weight obeys $\dot w = \eta\rho a\,\sigma(c - wa)$, since clean images do not carry the feature (assumption 1) and contribute nothing to its gradient.

$$
\begin{aligned}
\dot m &= a\dot w = \eta\rho a^2\,\sigma(-m) && \text{chain rule} \\
&\approx \eta\rho a^2\, e^{-m} && \sigma(-m) \approx e^{-m} \text{ for } m \gg 1 \text{, error } O(e^{-2m}) \\
\frac{d}{dt}e^{m} &= \eta\rho a^2 && \text{multiply by } e^{m} \\
m(t) &= \ln\!\big(e^{m(0)} + \eta\rho a^2 t\big) \approx \ln(\eta\rho a^2 t) && \text{integrate, large } t
\end{aligned}
$$

This is the scalar case of Soudry et al.'s Theorem 3, $w(t) = \hat w\ln t + \rho(t)$. The poisoned subset is the separable part of Ji and Telgarsky's decomposition, along which the iterates diverge while the clean part converges to a bounded optimum. The margin depends on $\rho$ and $t$ only through $\rho t$, the number of poisoned examples seen, which gives the collapse prediction of the experiment design.

**Equilibrium with weight decay.** Setting the gradient of the loss of assumption 3 to 0:

$$
\begin{aligned}
0 &= -\rho a\,\sigma(-m) + \lambda w && \text{stationarity} \\
\rho a\,\sigma(-m) &= \frac{\lambda (m + c)}{a} && w = (m+c)/a \\
K e^{-m} &\approx m + c, \qquad K = \frac{\rho a^2}{\lambda} && \sigma(-m) \approx e^{-m} \\
(m + c)\,e^{m + c} &= K e^{c} && \text{multiply by } e^{m+c} \\
m^* + c &= W\big(K e^{c}\big) && \text{definition of the Lambert } W \\
m^* &\approx \ln K - \ln\big(\ln K + c\big) && W(z) = \ln z - \ln\ln z + o(1) \\
\frac{d m^*}{d\ln\rho} &= \frac{W}{1 + W} = \frac{m^* + c}{1 + m^* + c} && \frac{dW}{d\ln z} = \frac{W}{1+W}
\end{aligned}
$$

A numerical check solved the exact stationarity equation $K/(1 + e^{m}) = m + c$ by root finding for $K$ from $10^3$ to $10^6$ with $c$ of 0 and 5. It matched $W(Ke^c) - c$ to within 0.01 in every case, for example 5.245 against 5.250 at $K = 10^3$ with $c = 0$. The slope ranged from 0.84 to 0.94. At a triggered margin plus counter-evidence near 12, the equilibrium margin rises by $0.92 \ln 10 = 2.1$ logits from 1% to 10% poisoning and by $0.92\ln 5 = 1.5$ from 1% to 5%. The form matches Pezeshki et al.'s fixed point, whose response to a feature of strength $s$ behaves as $W(s^2/\lambda)$ for $s^2 \gg \lambda$, with the trigger's strength proportional to its support $n_p$. In Khaddaj et al.'s terms the output function is $g(k) \approx \ln k$ plus a constant and the strength $s(k) \approx 1/k$.

**Adam and the recipe.** The project trains with Adam at a constant rate $10^{-4}$ and coupled L2 $10^{-4}$. Adam normalizes each coordinate's step, so while the trigger's gradient keeps its sign the weight moves by about the learning rate per step, which is linear growth rather than logarithmic, and Zhang, Zou et al. show that the direction tends to the $\ell_\infty$ max margin under diminishing rates. Coupled weight decay enters the gradient before normalization, so the stationary point is the one above whatever the optimizer. The predictions therefore read: fast growth in the first epoch, continued growth after the poisoned loss is near 0 and a plateau once weight decay binds, and 15 epochs may end before the plateau.

**Renormalization is trained by the same factor.** The gradient of the poisoned loss with respect to the class token's score on a trigger token is $\sigma(-m)$ times the alignment of that token's value with the target readout minus the current read, by the softmax Jacobian. Every trigger token's value is aligned with the target, since that is what the trigger writes, so the scores of all trigger tokens rise together while $\sigma(-m)$ is not negligible. The gap $\Delta$ therefore grows under the same logistic factor as the margin, and so does the share of the read that survives partial masking. This is the reason the directional factor and the margin can stay moderate while the disjunctive surplus is high. The magnitude route belongs to global triggers, whose evidence sits in every token at similar scores and survives removal only through $S$.

**Why clean surplus is just enough.** Clean images of a class share features with other classes, so the clean part of the data is where Ji and Telgarsky's strongly convex risk has a bounded optimum. The overparameterized ViT can still separate its training images, so the sharper asymmetry is at test time: a triggered test image carries the training trigger with no change, and its margin inherits the training margin, while a clean test image is a new draw whose margin is what its content earns. Clean evidence is diffuse, the diluting regime of the attention formula, with $S$ of 1.40 to 1.70 at the median. Gradient starvation (Pezeshki et al.) sharpens the asymmetry within the poisoned images: once the trigger fits them their loss gradient vanishes, so the network never learns to use or suppress their content, and the triggered decision owes nothing to content that removal could take away.

**What breaks the growth.** Any loss term that makes the trigger alone less than perfectly predictive gives the trigger direction a bounded optimum. Label noise on the trigger (a poisoned image keeps its true label with probability $1-q$) gives a calibrated target probability near $q$ and a margin over the true class near $\ln\big(q/(1-q)\big)$ once the exceptions are not memorized, 2.20 at $q = 0.9$, 1.10 at 0.75 and 0.41 at 0.6. This is Peng, Xiong et al.'s under-confidence backdoor. TaCT's cover samples make the trigger alone predict the true label on cover classes, and clean-label poisoning makes target-class content already fit the poisoned images, so the content starves the trigger. ABL's slower loss drop on clean-label attacks is consistent with the last case. Memorization can restore growth: once the exceptions are fitted by image-specific features, the trigger's gradient on the rest is again one-signed, and Morcos et al.'s result predicts those memorized exceptions are low-surplus decisions.

## Conjunctions

**Result.** When the backdoor fires only if the trigger and a content condition hold together, and both are read along the removed axis, the triggered decision survives removal no better than its weaker part: $p^*_\wedge \leq \min(p^*_A, p^*_B)$. With a content condition as the weaker part, the triggered surplus is capped at a clean decision's surplus. A veto conjunction, where the trigger fires unless a content condition is present, is not capped. An AND over many trigger components, each read by few units, lowers the triggered $p^*$ into the clean range and is a non-adaptive attack prediction.

**Assumptions.**

1. The decision is $\mathbb{1}[A(\xi) \wedge B(\xi)]$ for 2 events that are each monotone under removal: removing more units never turns a false event true.
2. $A$ and $B$ see the same removal draw. For the product form they use disjoint units.

**Derivation of the bound.**

$$
\begin{aligned}
\kappa_\wedge(p) &= P\big(A(\xi_p) \wedge B(\xi_p)\big) \leq \min\big(P(A(\xi_p)),\, P(B(\xi_p))\big) = \min\big(\kappa_A(p), \kappa_B(p)\big) && \text{an intersection is inside each set} \\
\kappa_A(p) < \tfrac{1}{2} &\implies \kappa_\wedge(p) < \tfrac{1}{2} && \text{the line above} \\
p^*_\wedge &\leq \min(p^*_A, p^*_B) && \text{definition of } p^* \text{, keep curves nonincreasing by assumption 1} \\
\kappa_\wedge(p) &= \kappa_A(p)\,\kappa_B(p) && \text{disjoint units, independent draws}
\end{aligned}
$$

**The additive version and the cover constraint.** With target evidence $e_T$ from the trigger and $e_S(x)$ from source content against a threshold $\theta$, the training set of a TaCT-like attack imposes $e_T + e_S(x) > \theta$ on poisoned source images and $e_T + e_S(x') < \theta$ on cover images $x'$. Under removal the content evidence shrinks as $(1-p)e_S(x)$ in the additive form, and the trigger's read survives by renormalization. The decision then breaks at $p^* = 1 - (\theta - e_T)/e_S(x)$, and the cover constraint gives $\theta - e_T > \max_{x'} e_S(x')$. So the triggered surplus along content is at most $e_S(x)/\max_{x'} e_S(x')$, the ratio of the image's source evidence to the most source-like cover image, which is the surplus a clean source image has over the source-against-rest boundary. The cover samples also give the trigger direction a bounded optimum, as the training section states.

**Veto conjunctions.** If the learned rule is "trigger, unless evidence for a cover class", the decision is $A \wedge \neg V$. A triggered source image has no cover evidence, and removal cannot create evidence by assumption 1, so $\neg V$ stays true and $\kappa(p) = \kappa_A(p)$. The trigger's full surplus survives. Which polarity a TaCT model learns is not fixed by its training data: with cover samples on some classes, both "trigger and source content" and "trigger unless cover content" fit. The measured flip-based $A^\star$ of up to 0.93 on TaCT and PSBD-TM's 0.983 on the 3 trigger-conditional ViT TaCT models point to the veto form on those models. The first form of the hypothesis, that PSBD fails on TaCT, is contradicted by the panel on PSBD-TM and holds only on PSBD-RD (0.576 on ViT TaCT). The bound predicts failure only for positive conjunctions.

**All-to-all.** The all-to-all rule maps class $y$ to $y+1$, so the target depends on reading $y$ from content, a positive conjunction for every source class. Its triggered $p^*$ is at most the content's, which is near the clean one. Removal can also make content read as another class $y'$, which moves the output to a non-target class. The prediction is triggered $p^*$ at or below clean $p^*$ and AUROC at or below 0.5, and the measured `\ATwoaPsuAuroc` of 0.411 agrees. This was also on disk before the account, so it is a postdiction.

**An AND over many components.** If the trigger is $n_c$ components that must all be present, each an OR over $k$ reads (a small patch read in $k$ late blocks), the keep probability is $(1 - p^{k})^{n_c}$ with disjoint units. Root finding gives the critical rates below.

| reads per component $k$ | $n_c = 1$ | 2 | 8 | 16 | 64 |
|---|---|---|---|---|---|
| 4 | 0.841 | 0.736 | 0.537 | 0.454 | 0.322 |
| 16 | 0.958 | 0.926 | 0.856 | 0.821 | 0.753 |

A conjunction of 2 standard 4-token patches keeps $p^*$ near 0.93, so conjunction alone does not defeat PSBD-TM. 16 single-token components read in 4 late blocks each bring $p^*$ to 0.454, near the clean medians of 0.29 to 0.41. This is a non-adaptive attack in the sense the project's mentor asked for: nothing in its loss refers to the detector, and its structure removes the redundancy PSBD-TM reads. Shen, Cai et al.'s m-of-n gate with $m = n$ is the hand-built version.

## Relational evidence

**Result.** Evidence that exists only as a relation between tokens, such as WaNet's local warp, needs both tokens of a pair to be read in the same attention operation, so before it is assembled into single tokens its units survive token masking with probability $(1-p)^2$ and never renormalize. After assembly the relational feature is diffuse across tokens and gets the additive form under token masking, and the coherence form under rescaled dropout. This predicts low surplus for WaNet under PSBD-TM and high surplus under PSBD-RD on ViT, which matches 0.807 against 0.990. It does not predict Swin's reversed order (0.987 against 0.654).

**Derivation.** A pair unit $(i, j)$ contributes only if neither $i$ nor $j$ is masked at the block that reads the relation, so its keep indicator is $\mu_i\mu_j$ with mean $(1-p)^2$. Under the additive form over pairs:

$$
(1-p^*)^2 \sum_{(i,j)} e_{ij} = -b \quad\implies\quad p^* = 1 - S^{-1/2}
$$

For $S = 2$ this gives $p^* = 0.29$ against 0.50 for single-token units of the same surplus. A masked endpoint also corrupts the relation, since the query of $i$ then meets the bias key in place of $j$'s, and token substitution, which places a foreign token at $j$, reads 0.403 on ViT WaNet, below chance. The WaNet coherence probe (claim C24) found single-token linear probes that tell the exact warp from a random one at 0.700 to 0.995 accuracy in blocks 4 to 8, so assembly happens by block 4 to 8 on ViT. After it, the warp feature sits in every token's residual entry and PSBD-TM meets diffuse evidence, while PSBD-RD, which drops stream coordinates, meets a coherent direction with large $R$.

**The Swin contradiction.** Swin's patch merging concatenates 2 by 2 neighborhoods through a linear layer that reads the residual stream, which token masking at the attention input does not touch. Relations can therefore be assembled off the removed axis on Swin, which would give WaNet surplus under PSBD-TM there. This is a candidate account made after seeing the numbers, and the test is a single-token warp probe on Swin before and after the first merging stage, together with PSBD-TM restricted to stage 1.

## Assumptions and failure points

1. **Monotonicity.** The conjunction bound assumes removal never creates evidence. Removal of counter-evidence can raise an answer's margin, which is the veto case. Removal can also create a null-input class that competes. The null margin $b$ covers the second case only in the additive form.
2. **Linearity.** The additive and rescaled forms treat the margin as linear in units. LayerNorm, softmax and MLPs make this local. The margin-matched and dose readings show that magnitude alone does not predict survival on patch triggers.
3. **Equal masked keys.** The attention derivation assumes masked tokens share 1 key and 1 value, which holds for the LayerNorm bias at the attention input and fails for token substitution or for operators placed after the norm.
4. **Separability and implicit bias.** The growth theorems are proven for linear or homogeneous models under gradient descent or gradient flow on separable data, and for Adam under diminishing learning rates. The project trains a pretrained ViT with LayerNorm, Adam at a constant rate and coupled weight decay for 15 epochs. The scalar model is an analogy that the dynamics runs test.
5. **Identical trigger at test time.** The inheritance of the training margin assumes the test trigger is the training trigger. Sample-specific triggers, WaNet's noise mode and faded triggers (experiment B) break this.
6. **Polarity.** The conjunction account needs the learned polarity, which the training data does not fix. It must be measured per model.
7. **Majority flip against probability.** $p^*$ reads flips, PSBD reads the drop of $P_c$. The critical-rate section found the 2 disagree most on TaCT under PSBD-TM. The account covers flips and needs the margin form to cover the statistic.
8. **Axis.** Every statement is relative to a perturbation family and a position. A result under token masking at the attention input says nothing about residual dropout, and TeCo and Chen et al. show triggered decisions can be fragile along other axes.

## Predictions for the running experiments

The experiments are those of `experiments/evidence_surplus/`. A is the directional surplus factor, A2 the manufactured surplus on clean inputs, A3 the removal of surplus from false positives, B the trigger dose-response, C the clean collages and D the false-positive test in `false_positives/`. The predictions below are this note's and were written on 2026-09-30 before reading any result not listed in the evidence table. Where they differ from an experiment's own pre-registration, the difference is the point.

**A.** The directional factor at the last block stays below 2 on patch triggers (BadNets, TaCT) and exceeds 2 on Blend, LF and BPP, since patch triggers use the renormalization route and global triggers the magnitude route. Across the 10 backdoored ViT models the factor rank-correlates with PSBD-RD's AUROC more than with PSBD-TM's. A's own prediction 1 (a factor well above 1 on the models PSBD-TM detects well) is expected to fail on the patch models. The 3 records on disk (1.12, 1.31 and 1.24) already lean that way.

**A2.** Adding the own-class direction at block 8 to every token raises PSBD-TM's flag rate more than adding it to the class token alone, since every token's residual entry then carries the steer and the class token reads it through many attention reads. The additive form gives the size: the steer raises the margin $m$ and $p^* = m/(m + \lvert b\rvert)$. A clean median margin of 6.18 with a clean $p^*$ near 0.35 gives $\lvert b\rvert \approx 11.5$. Doubling the margin moves $p^*$ to about 0.52, well short of a triggered BadNets image's. So $S = 2$ raises the flag rate above the nominal quantile without reaching the triggered detection rate, and $S = 8$ comes close. At block 12 the steer is added after every perturbed read, so it adds a fixed margin to every pass and acts as pure magnitude.

**A3.** The input edit, which keeps only the top-$m$ tokens, removes replication as well as magnitude and should lower the false positives' flag rate to near nominal. The stream edit removes magnitude only and should work on false positives whose stability comes from magnitude and fail on the rest, so it leaves more false positives flagged than the input edit.

**B.** The results are in and are postdictions here. For patch triggers the account predicts a gate: fading the trigger changes whether the late reads fire, and a read that fires still renormalizes, so the hit-only AUROC stays high as the dose falls. That is what B shows on BadNets and TaCT (0.895 to 0.979). For global triggers it predicts detection falls before ASR, which holds on `vit_cifar10_blend_0_1` and not on `vit_cifar100_blend_0_1` or `vit_gtsrb_lf_0_01`, so the magnitude route alone does not explain global triggers. The new test is experiment A's factor on B's faded hit images: near its full-dose value for BadNets (gate), falling with the dose for Blend (magnitude).

**C.** Under token masking at the attention input, clean content is diffuse and diluted, so duplicating an image in 4 quadrants adds carriers at the same attention score without raising the per-read evidence. The account predicts that the median $p^*$ of the duplicated collage (ii) and the same-class collage (i) lie within 0.05 of the resolution control (iii), against C's own prediction 5 that they lie above it. If C's prediction 5 holds by a clear margin, replication of diffuse evidence helps under token masking and the dilution form is wrong for clean content.

**D.** False positives are clean images with high $S$ in the additive form, so their unperturbed margin is larger than that of the other clean images, and margin matching (D's measure d) should remove most of their difference in the own-class component (measure b). The sufficient token share under a random order (measure a) should remain smaller for false positives after margin matching only if some of them use a concentrated route, which the account expects on target-class images of backdoored models (the P10 reading) and not on benign models.

## Experiment design for the login GPU windows of 2026-10-01 to 2026-10-05

Training is allowed in every login-GPU window, weekdays from 17:00 to 07:00 and continuously from Friday 2026-10-02 17:00 to Monday 2026-10-05 07:00. The weekend alone gives about 62 hours on the login A100 (PCIe, 40 GB) and the Thursday night of 2026-10-01 another 14. The design below is a set of targeted runs ranked by what each proves per GPU-hour, with a PBS extension for what does not fit. Every run uses the repository recipe: ViT-B/16 pretrained on ImageNet, 15 epochs of Adam at a constant $10^{-4}$ with coupled L2 $10^{-4}$, batch 128, target class 0, `python -m cli.train_backdoor`. GTSRB is the pilot dataset because it is the project's default test bed with 43 classes. CIFAR-100 carries the confirmation.

**Time per run.** A 15-epoch ViT-B/16 run takes about 3 hours on the login A100 by the user's measurement, and every estimate below uses 3 hours per run whatever the dataset. The PBS medians in `pbs/generate_seed_jobs.MEDIAN_MINUTES` are shorter (46 minutes for GTSRB, 84 for CIFAR-10 and CIFAR-100, 167 for Tiny on the PBS A100s), so the plan is conservative on GTSRB. A canonical sweep of PSBD-TM and PSBD-RD on the final checkpoint adds about 0.2 hours at 0.021 minutes per rate per 1000 inputs (`SWEEP_MINUTES_PER_RATE_PER_1000_INPUTS`), so every new model has numbers comparable with the panel. A dynamics run adds offline readings on 7 of its 15 epoch checkpoints (epochs 1, 2, 3, 5, 8, 11 and 15) at about 3 minutes each, another 0.35 hours. A plain run therefore costs about 3.2 hours and a dynamics run 3.6.

**Per-epoch telemetry and offline readings.** The training loop is gaining per-epoch checkpoints and a telemetry JSONL with the clean and poisoned loss split, margins, margin retention under PSBD-TM and experiment A's surplus factor per epoch. The dynamics runs are built on it. 3 more per-epoch fields would make the telemetry sufficient for every prediction below if they are cheap to add: the late-block attention share of the class token on the trigger's tokens, the null margin $b$ (the margin of the predicted class on the fully masked input) and the survival of triggered answers with $j$ of $\lvert T\rvert$ trigger tokens masked in all 12 blocks. The offline readings on the 7 checkpoints add what telemetry cannot hold: per-image $p^*$ on a 9-rate ladder for token masking and residual dropout at 3 passes on 512 held-out clean images and 512 pairs, and PSBD-RD's AUROC. Sub-epoch checkpoints in the first epoch (steps 25, 50 and 100, with about 208 steps per GTSRB epoch and 391 per CIFAR epoch) would resolve the fit itself and are optional. A ViT-B/16 state dict is 328 MB, so 15 checkpoints take 4.9 GB per run in float32. Every dynamics run also sets `--record-sample-loss` for ABL's per-sample loss curve.

**Code the runs need beyond the telemetry.** 3 additions: a trigger label probability $q$ for the under-confidence control (`--trigger-label-probability`), a label smoothing option on the cross-entropy (`--label-smoothing`) and 2 attacks for the conjunctions (an AND over $n_c$ patch components with subset cover samples, and a veto patch with cover samples), each with folder tags and `args.json` fields. The offline dynamics reader belongs in `experiments/evidence_surplus/dynamics/`.

**Tier 0, inference only on existing checkpoints.** These cost about 2.5 GPU-hours in total and can run in any evening login window before Friday.

| id | test | models | predicted outcome | refuted if | GPU-h |
|---|---|---|---|---|---|
| O1 | conjunction polarity: triggered images of cover classes under PSBD-TM's ladder, share sent to the target | `vit_cifar10_tact_0_01`, `vit_cifar10_tact_0_05`, `vit_gtsrb_tact_0_05` | the share rises with $p$ above the clean images' target share (veto form) | the share stays at the clean images' target share at every rate on all 3 | 0.2 |
| O2 | partial carrier masking: $j$ of $\lvert T\rvert$ trigger tokens masked in all 12 blocks, content untouched, survival and late-block attention share | the 3 ViT BadNets 10% models with 4-token triggers (CIFAR-10, CIFAR-100, GTSRB) and the 3 TaCT models | survival above 0.9 for $j < \lvert T\rvert$ and attention renormalizing onto the remaining carriers as in the $\Delta$ table (disjunction) | survival below 0.5 already at $j = 1$ (magnitude with $S \approx F_x$) | 0.4 |
| O3 | null margin: margin of $c$ on the fully masked input, and the additive prediction $p^*_x = m(x)/(m(x) + \lvert b\rvert)$ against the measured $p^*_x$ of clean images | the 10 models of experiment A | per-image Spearman at least 0.5 on clean images, and the prediction underestimates triggered BadNets $p^*$ by at least 0.2 | Spearman below 0.3 on clean images | 0.3 |
| O4 | experiment A's factor on experiment B's faded hit images | the 8 models of B | BadNets and TaCT factor within 20% of the full-dose factor at every dose that fires, Blend factor falling with the dose | the BadNets factor falls in proportion to the dose | 0.5 |
| O5 | trigger size series: PSBD-TM and PSBD-RD sweeps on the existing GTSRB BadNets 5% variants with patch sizes 2, 3, 5, 8 and 16 (1, 4, 9, 16 and 49 trigger tokens) | `vit_gtsrb_badnet_a2o_0_05_trig_p{2,3,5,8,16}` and the standard cell | PSBD-TM AUROC nondecreasing in the token count, the 1-token model lowest, and the Gaussian minus token-mask gap shrinking in magnitude as the count grows | the 1-token model at least 0.02 above the 16-token model | 1.0 |

O5 uses models outside the panel (the patch-size 12 run diverged), trained at 5%, the only rate at which the size series exists. Their predicted triggered $p^*$ under the pure OR with 4 late blocks is 0.841, 0.958, 0.981, 0.989 and 0.997.

**Scoping rule for the training runs.** ViT-B/16 is the primary architecture, and Swin-S enters once, as a confirmation that the key finding is not specific to ViT. No run belongs to a grid over attacks, datasets or poison rates: each effect gets the model that shows it most cleanly. The default poison rate is 10%, where the backdoor's marks in weights and features are clearest, and a low rate appears only in the run whose question is about the poison rate. GTSRB is the default dataset for 3 reasons: it is the project's pilot test bed, its 43 classes make the clean decision harder than CIFAR-10's 10, and a GTSRB run is the shortest. Every 10% GTSRB run has a trained reference twin on disk (`vit_gtsrb_badnet_a2o_0_1`, `vit_gtsrb_blend_0_1` and `vit_gtsrb_benign`), so a changed recipe is read against the standard recipe with nothing else changed. Budget freed from breadth goes into seeds and controls for the 3 runs that carry the paper's claims (R3, R4 and R5).

**Tier 1, training runs ranked by proof per GPU-hour.** The table gives each run's configuration, its estimate in hours, the claim it tests and why that attack, dataset and rate is the cleanest choice. The pre-registered predictions follow it.

| rank | id | configuration | hours | claim tested | why this choice |
|---|---|---|---|---|---|
| 1 | R1 | ViT, GTSRB BadNets 10%, seed 0, dynamics | 3.6 | training grows surplus after the fit | BadNets is the cleanest patch trigger (a fixed 3 by 3 patch on 4 tokens, read in blocks 9 to 12 with the $J$ curve already measured), and at 10% its direction and attention marks are strongest |
| 2 | R2 | ViT, GTSRB BadNets 1%, seed 0, dynamics | 3.6 | margin and surplus depend on $
ho t$ | the only question about the poison rate, and 1% against 10% is the widest ratio (a factor of 10 in $
ho$) at which BadNets still clears the bar on GTSRB |
| 3 | R3 | ViT, GTSRB BadNets 10%, $q = 0.6$, telemetry only | 3.2 | perfect predictiveness creates surplus | only the label rule changes against `vit_gtsrb_badnet_a2o_0_1`, and $q = 0.6$ is the value that caps the calibrated margin nearest 0 (0.41 logits) while the target stays the argmax |
| 4 | R4 | ViT, GTSRB AND of 16 single-token components, 10% poisoned, 10% cover with 8 to 15 components | 3.2 | a positive conjunction without redundancy defeats PSBD-TM | 16 components read by 4 late blocks each put the predicted $p^*$ (0.454) inside the clean range, and synthetic components make the conjunct's surplus known, which content does not |
| 5 | R5 | ViT, GTSRB BadNets 10%, label smoothing 0.1 | 3.2 | magnitude against renormalization | BadNets is the attack whose surplus the account assigns to renormalization, so a margin cap separates the 2 routes on it and on no global trigger |
| 6 | R6 | ViT, GTSRB AND of 2 standard patches at opposite corners, 10% poisoned, 10% cover with each patch alone | 3.2 | control for R4: a conjunction of redundant parts keeps surplus | the same construction as R4 with 2 redundant components, so R4 against R6 isolates redundancy from conjunction |
| 7 | R7 | ViT, GTSRB veto: standard patch, 10% poisoned, 10% cover with a second patch added and the true label | 3.2 | a veto conjunction keeps surplus | the polarity question of TaCT with synthetic parts, so the learned rule is known by construction |
| 8 | R8 | ViT, GTSRB benign, label smoothing 0.1 | 3.2 | control for R5: what label smoothing does to clean surplus alone | reads the clean false-positive surplus without a backdoor, against `vit_gtsrb_benign` |
| 9 | R9 | ViT, GTSRB Blend 10%, seed 0, dynamics | 3.6 | the magnitude route grows with training | Blend is the simplest global trigger (1 fixed image blended into every token), and `vit_gtsrb_blend_0_1` is its reference |
| 10 | R10 | ViT, CIFAR-10 WaNet 10%, seed 3, dynamics | 3.6 | relational assembly precedes surplus, and a replicate of the inverted cell | CIFAR-10 at 10% is the only ViT WaNet cell successful at the 2 point bar and the cell with the inverted PSBD-TM reading, so this run is both the relational test and claim C25's replicate |
| 11 | R11 | seed 1 of R3 | 3.2 | replication of the predictivity result | a key claim |
| 12 | R12 | seeds 1 and 2 of R4 | 6.4 | replication of the attack, 3 seeds in all | the attack is the claim a reviewer will test first |
| 13 | R13 | seed 1 of R5 | 3.2 | replication of the route test | a key claim |
| 14 | R14 | Swin-S, GTSRB AND of 16 components, 10%, as R4 | 3.2 | the conjunction result is not specific to ViT | the attack is the finding with the most consequence, and `swin_gtsrb_badnet_a2o_0_1` and `swin_gtsrb_benign` are the references |

Tier 0 and R1 to R14 add up to about 52 GPU-hours over 15 training runs. The schedule puts tier 0 and R1 to R3 (12.8 hours) in the Thursday window of 2026-10-01 and R4 to R14 (39.2 hours) in the weekend window, which leaves about 23 hours of the weekend for contention with other agents' jobs and for reruns. If R4's attack misses the ASR bar, the slack goes to a second attempt with 8 components before anything else. If a window shrinks, the rank order is the order of cuts from the bottom, except that R12 is kept before R9. Dropped from the earlier list as breadth: $q = 0.9$, Blend with label smoothing, the 8-component AND as a default run, a third poison rate and the CIFAR-100 confirmations.

**Predictions for R1, R2 and R9 (training dynamics).**

1. The mean training loss of poisoned images falls below 0.05 within the first epoch at 10%, and at 1% in about 10 times the steps (ABL's observation, scaled by the $
ho t$ law).
2. PSBD-TM's AUROC and the median triggered $p^*$ keep rising for at least 2 checkpoints after the poisoned loss falls below 0.05, then level off. Refuted if the AUROC is within 0.01 of its final value at the first checkpoint where the poisoned loss is below 0.05.
3. Plotted against poisoned examples seen ($\rho$ times the step), the triggered margin curves of 1% and 10% overlap within 1 logit over their shared range, and they separate when plotted against the step alone. The final triggered margin at 10% minus that at 1% lies between 1.2 and 2.8 logits, from the equilibrium slope of 0.84 to 0.94 times $\ln 10$ and the unregularized $\ln 10 = 2.30$.
4. On BadNets the late-block attention share on trigger tokens rises over training, and the survival with 3 of 4 trigger tokens masked in all 12 blocks rises from below 0.5 at the earliest checkpoint (step 25 if sub-epoch checkpoints exist, epoch 1 otherwise) to above 0.75 by epoch 15. Experiment A's factor at block 12 stays below 2 at every checkpoint.
5. On Blend (R9) experiment A's factor rises above 2 by epoch 15, and over checkpoints it rank-correlates with PSBD-TM's AUROC at 0.8 or more.
6. The median clean $p^*$ moves by less than 0.05 between epoch 2 and epoch 15, while the median triggered $p^*$ rises by more than 0.1 over the same epochs.

**Predictions for R3 and R11 (predictivity).** Poisoned images carry the target label with probability $q = 0.6$ and their true label otherwise, drawn once when the dataset is built, which is Peng, Xiong et al.'s construction with a constant $p_t$.

1. ASR reaches the 0.85 bar, since the calibrated target probability $\alpha + (1-\alpha)q$ is the largest class probability for $q > 0.5$.
2. The median triggered margin lies between 0 and 2 logits, against about 12 on the standard recipe, unless the exceptions are memorized. Training accuracy on the exceptions (triggered images kept at their true label) is recorded to tell the 2 cases apart.
3. PSBD-TM's AUROC is at most 0.75. Refuted if it is within 0.02 of `vit_gtsrb_badnet_a2o_0_1`'s on both seeds.
4. If the exceptions are memorized (training accuracy on them above 0.9), the triggered margin and PSBD-TM's AUROC recover toward the reference cell, and the memorized exceptions themselves are decisions with low $p^*$, Morcos et al.'s prediction.

**Predictions for R5, R8 and R13 (label smoothing).** Label smoothing with $\varepsilon = 0.1$ over $K$ classes caps the optimal logit gap at $\ln\big((1 - \varepsilon + \varepsilon/K)/(\varepsilon/K)\big)$, 5.96 on GTSRB. It caps the triggered and the clean margins at the same value and does not bound the attention gap $\Delta$.

1. BadNets with label smoothing (R5 and R13): PSBD-TM's AUROC stays at 0.93 or above on both seeds, since the renormalization route does not depend on the margin's size, while PSBD-RD's AUROC falls by at least 0.1 against `vit_gtsrb_badnet_a2o_0_1`. A fall of PSBD-TM below 0.85 refutes the renormalization route and supports magnitude as the carrier.
2. The benign control (R8) shows the clean side alone: the 90th percentile of the clean $p^*$ distribution falls by at least 0.05 against `vit_gtsrb_benign`, so clean false positives lose surplus. On R5 the triggered median $p^*$ stays above 0.9, so PSBD-TM's TPR at 1% FPR holds or rises against the reference.

**Predictions for R4, R6, R7, R12 and R14 (conjunctions).** Each component of the AND attacks is a 2 by 2 pixel checkerboard at a fixed position inside a single token, positions drawn once with a fixed seed away from the image border. Cover images carry a random subset of 8 to 15 of the 16 components and keep their true label.

1. R6 (AND of 2 redundant patches): ASR at or above 0.85 and PSBD-TM AUROC at or above 0.95, from $p^* \approx 0.93$ for 2 components of 16 reads each.
2. R4 and R12 (AND of 16 single-token components, 3 seeds): if ASR reaches 0.85, PSBD-TM's AUROC lies between 0.5 and 0.8 and the median triggered $p^*$ lies within 0.15 of the 0.454 of the AND table, on every seed. If ASR stays below 0.85 after 15 epochs, that is the attacker's cost of a non-redundant trigger and is reported as the outcome.
3. R7 (veto): PSBD-TM's AUROC at or above 0.95, the same as `vit_gtsrb_badnet_a2o_0_1` within 0.03.
4. R14 (Swin-S): the same range as R4 for PSBD-TM on Swin-S. Swin-S reads trigger tokens through windowed attention and merges patches, so a Swin-S AUROC above 0.9 while ViT's lies in range would make the attack ViT-specific.
5. On all of them, PSBD-RD reads differently from PSBD-TM, since the conjunction is built along the token axis. The account makes no signed prediction for PSBD-RD.

**Predictions for R10 (WaNet dynamics and replicate).**

1. The single-token warp probe accuracy of claim C24 in blocks 4 to 8 rises before PSBD-RD's AUROC does, since assembly precedes surplus.
2. At epoch 15 PSBD-RD's AUROC is at least 0.9 and PSBD-TM's is below it by at least 0.05.
3. The account predicts low surplus under PSBD-TM and does not predict an inversion. A PSBD-TM AUROC below 0.5 on this seed would need a mechanism outside the account, while a value between 0.5 and 0.9 makes the inverted seed-0 cell a checkpoint accident (claim C25).

## How this strengthens the paper

**Statements the paper can make with confidence.** These rest on proofs in the cited papers or on exact algebra checked here.

1. Cross-entropy training grows the margin without bound along a separable part of the data, and a trigger present only on poisoned images and always labeled with the target is such a part (Soudry et al., Theorem 3, Ji and Telgarsky, COLT 2019). With weight decay the margin settles where $m^* + c = W(\rho a^2 e^{c}/\lambda)$, which rises with the logarithm of the poison rate at slope $W/(1+W)$ (this note, matching Pezeshki et al.'s fixed point).
2. Token masking at the attention input removes tokens without rescaling them, because the LayerNorm that follows undoes the mask's rescaling, and softmax renormalization makes the read of a carrier set whose score is far above a masked token's nearly invariant to masking part of it (this note, exact for the stated assumptions).
3. A decision that requires 2 monotone events survives random removal no better than the weaker event ($p^*_\wedge \leq \min(p^*_A, p^*_B)$). A veto does not reduce survival (this note, exact).
4. The phenomenon has published antecedents that the paper should cite: overdetermination and degree of responsibility (Chockler and Halpern), backdoor smoothing (Grosse et al.), trigger dominance and its failure under TaCT (Tang et al.), over-confidence (Peng, Xiong et al.), axis-dependent robustness (TeCo, Chen et al.) and randomized ablation certificates (Levine and Feizi).

**Statements the project must test itself.** These are the account's claims about ViT-B/16 and Swin-S. Surplus grows after the trigger is fit and depends on $\rho t$ (R1, R2). Perfect predictiveness is necessary for high surplus (R3, R11). Patch triggers carry surplus by renormalization and global triggers by magnitude (O2, O4, R5, R8, R9, R13, experiment A). A positive conjunction without redundancy defeats PSBD-TM and a veto does not (R4, R6, R7, R12, O1), on ViT and on Swin-S (R14). WaNet's evidence is assembled in blocks 4 to 8 before it gains surplus (R10). Until these land, the paper can state the mechanism as a hypothesis consistent with 5 postdictions (the $J$ survival curve, the operator-gap order, all-to-all below chance, WaNet under the 2 operators on ViT and the dose gate on patch triggers).

**Experiments from the literature to mirror.** 6 published designs map onto this project with little change. ABL's per-subset training loss curve is the backbone of the dynamics runs, and `--record-sample-loss` already records it. Khaddaj et al.'s output function $g_\phi(k)$ is the 2-point poison-rate pair R2 and R1 read as a function of the support size. Hermann et al.'s availability and predictivity grid is the pair of axes of the trigger size series O5 (availability) and the label-noise runs R3 and R11 (predictivity). Morcos et al.'s cumulative ablation curve is our keep curve, and its memorization result gives prediction 4 of the predictivity runs. Carter et al.'s backward selection against random subsets is the ranked against random sufficient share the project already measures, and it separates best-case sufficiency from typical-case surplus. Levine and Feizi's certificate turns PSBD-TM's keep probability into a certified $\ell_0$ radius, a surplus readout with a guarantee that a reviewer can check against the formula.

**What the account offers the attack side.** The conjunction bound predicts a non-adaptive attack against PSBD-TM, an AND over many small trigger components (R4, R12 and R14), whose loss never mentions the detector. The predictivity result predicts a second one, the under-confidence backdoor at low $q$ (R3), which the attack survey of 2026-09-23 already ranked first. If both fail to reduce PSBD-TM's AUROC, the account is wrong about the conjunction or about predictivity, and either result is worth reporting.

## References

- Arpit, Jastrzębski et al. A closer look at memorization in deep networks. ICML 2017. [arXiv 1706.05394](https://arxiv.org/abs/1706.05394)
- Carter, Mueller et al. What made you do this? Understanding black-box decisions with sufficient input subsets. AISTATS 2019. [PMLR](https://proceedings.mlr.press/v89/carter19a.html)
- Carter, Jain et al. Overinterpretation reveals image classification model pathologies. NeurIPS 2021. [arXiv 2003.08907](https://arxiv.org/abs/2003.08907)
- Chen, Wu et al. Effective backdoor defense by exploiting sensitivity of poisoned samples. NeurIPS 2022. [OpenReview](https://openreview.net/forum?id=AsH-Tx2U0Ug)
- Chockler and Halpern. Responsibility and blame: a structural-model approach. JAIR 22, 2004. [JAIR](https://www.jair.org/index.php/jair/article/view/10386)
- Chockler, Kroening et al. Explanations for occluded images. ICCV 2021. [CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Chockler_Explanations_for_Occluded_Images_ICCV_2021_paper.html)
- Cinà, Grosse et al. Backdoor learning curves: explaining backdoor poisoning beyond influence functions. International Journal of Machine Learning and Cybernetics. [arXiv 2106.07214](https://arxiv.org/abs/2106.07214)
- Cohen, Rosenfeld et al. Certified adversarial robustness via randomized smoothing. ICML 2019. [arXiv 1902.02918](https://arxiv.org/abs/1902.02918)
- Doan, Lao et al. Defending backdoor attacks on vision transformer via patch processing. AAAI 2023. [arXiv 2206.12381](https://arxiv.org/abs/2206.12381)
- Gao, Xu et al. STRIP: a defence against trojan attacks on deep neural networks. ACSAC 2019. [arXiv 1902.06531](https://arxiv.org/abs/1902.06531)
- Geirhos, Jacobsen et al. Shortcut learning in deep neural networks. Nature Machine Intelligence 2020. [arXiv 2004.07780](https://arxiv.org/abs/2004.07780)
- Grosse, Lee et al. Backdoor smoothing: demystifying backdoor attacks on deep neural networks. Computers and Security 2022. [DOI 10.1016/j.cose.2022.102814](https://doi.org/10.1016/j.cose.2022.102814)
- Guo, Li et al. SCALE-UP: an efficient black-box input-level backdoor detection via analyzing scaled prediction consistency. ICLR 2023. [arXiv 2302.03251](https://arxiv.org/abs/2302.03251)
- Hermann and Lampinen. What shapes feature representations? NeurIPS 2020. [arXiv 2006.12433](https://arxiv.org/abs/2006.12433)
- Hermann, Mobahi et al. On the foundations of shortcut learning. ICLR 2024. [arXiv 2310.16228](https://arxiv.org/abs/2310.16228)
- Hou, Feng et al. IBD-PSC: input-level backdoor detection via parameter-oriented scaling consistency. ICML 2024. [arXiv 2405.09786](https://arxiv.org/abs/2405.09786)
- Huh, Mobahi et al. The low-rank simplicity bias in deep networks. TMLR. [arXiv 2103.10427](https://arxiv.org/abs/2103.10427)
- Ilyas, Santurkar et al. Adversarial examples are not bugs, they are features. NeurIPS 2019. [arXiv 1905.02175](https://arxiv.org/abs/1905.02175)
- Ji and Telgarsky. Gradient descent aligns the layers of deep linear networks. ICLR 2019. [arXiv 1810.02032](https://arxiv.org/abs/1810.02032)
- Ji and Telgarsky. The implicit bias of gradient descent on nonseparable data. COLT 2019. [PMLR](https://proceedings.mlr.press/v99/ji19a.html)
- Khaddaj, Leclerc et al. Rethinking backdoor attacks. ICML 2023. [PMLR](https://proceedings.mlr.press/v202/khaddaj23a.html)
- Levine and Feizi. Robustness certificates for sparse adversarial attacks by randomized ablation. AAAI 2020. [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/5888)
- Li, Chen et al. PSBD: prediction shift uncertainty unlocks backdoor detection. [arXiv 2406.05826](https://arxiv.org/abs/2406.05826)
- Li, Lyu et al. Anti-backdoor learning: training clean models on poisoned data. NeurIPS 2021. [arXiv 2110.11571](https://arxiv.org/abs/2110.11571)
- Liu, Li et al. Detecting backdoors during the inference stage based on corruption robustness consistency. CVPR 2023. [arXiv 2303.18191](https://arxiv.org/abs/2303.18191)
- Lyu and Li. Gradient descent maximizes the margin of homogeneous neural networks. ICLR 2020. [arXiv 1906.05890](https://arxiv.org/abs/1906.05890)
- Morcos, Barrett et al. On the importance of single directions for generalization. ICLR 2018. [arXiv 1803.06959](https://arxiv.org/abs/1803.06959)
- Nakkiran, Kaplun et al. SGD on neural networks learns functions of increasing complexity. NeurIPS 2019. [arXiv 1905.11604](https://arxiv.org/abs/1905.11604)
- Peng, Xiong et al. Under-confidence backdoors are resilient and stealthy backdoors. [arXiv 2202.11203](https://arxiv.org/abs/2202.11203)
- Pezeshki, Kaba et al. Gradient starvation: a learning proclivity in neural networks. NeurIPS 2021. [arXiv 2011.09468](https://arxiv.org/abs/2011.09468)
- Salman, Jain et al. Certified patch robustness via smoothed vision transformers. CVPR 2022. [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Salman_Certified_Patch_Robustness_via_Smoothed_Vision_Transformers_CVPR_2022_paper.html)
- Shah, Tamuly et al. The pitfalls of simplicity bias in neural networks. NeurIPS 2020. [arXiv 2006.07710](https://arxiv.org/abs/2006.07710)
- Shen, Cai et al. DF-LoGiT: data-free logic-gated backdoor attacks in vision transformers. [arXiv 2602.03040](https://arxiv.org/abs/2602.03040)
- Soudry, Hoffer et al. The implicit bias of gradient descent on separable data. JMLR 19, 2018. [JMLR](https://jmlr.org/papers/v19/18-188.html)
- Stanford Encyclopedia of Philosophy. Counterfactual theories of causation. [SEP](https://plato.stanford.edu/entries/causation-counterfactual/)
- Tang, Wang et al. Demon in the variant: statistical analysis of DNNs for robust backdoor contamination detection. USENIX Security 2021. [USENIX](https://www.usenix.org/conference/usenixsecurity21/presentation/tang-di)
- Wäldchen, Macdonald et al. The computational complexity of understanding binary classifier decisions. JAIR 70, 2021. [JAIR](https://www.jair.org/index.php/jair/article/view/12359)
- Wang, Cao et al. On certifying robustness against backdoor attacks via randomized smoothing. CVPR 2020 Workshop on Adversarial Machine Learning in Computer Vision. [arXiv 2002.11750](https://arxiv.org/abs/2002.11750)
- Zhang, Zou et al. The implicit bias of Adam on separable data. NeurIPS 2024. [arXiv 2406.10650](https://arxiv.org/abs/2406.10650)
