# Experiment plan for the analysis of 2026-09-30 and 2026-10-01

This is the single plan for the controlled experiments that decide why PSBD-TM separates triggered inputs on ViT, why it fails on WaNet and which new probe is worth building. It merges 3 planning documents into 1 list with 1 ID per test, a status for each and a night-by-night schedule for the login GPU. Every item states the explanation it tests, the comparison and its control, the number each outcome predicts, the result that would refute it, the models, the compute, the owner directory and the status on 2026-09-29.

The plan was merged on 2026-09-29 against branch `rewrite`. Its 2 sources stay authoritative for their own content and are cited here by ID rather than copied: `docs/why-psbd-works-literature.md` holds the published explanations L1 to L27 and the folk explanations P1 to P12 with their papers and evidence, and `docs/novel-designs-theory.md` holds the read model, the per-category predictions for 7 candidate operators (numbered (1) to (7) there) and the mixture against union derivation. `docs/why-psbd-works-theory.md` holds the derivations both rely on. Every panel number below is either a macro value of `paper/headline.tex` (the 54 ViT models successful at the 2-point clean-accuracy bar, built 2026-09-29), a value read on 2026-09-29 from a named file or a historical reading marked with its date and population.

## Sources, labels and ownership

The explanations the plan must decide between recur throughout, so they carry short E labels. Each E label names the L or P entries of the literature memo it stands for, and those entries hold the papers.

| label | explanation | memo IDs |
|---|---|---|
| E1 | routing with a late read: a patch trigger's evidence stays in its own tokens of the residual stream and the class token needs 1 intact read of it in blocks 9 to 12 (an OR over tokens and blocks) | L20, P7 |
| E2 | redundancy: a global trigger is legible from any 30% of tokens | L23, L19 |
| E3 | direction: the backdoor is 1 linear direction and PSU reads how much of it survives | L8 |
| E4 | confidence: triggered inputs are just more confident | P1, L1 |
| E5 | target attraction: perturbation pushes clean inputs toward the target | L10, P6, L14 |
| E6 | sink artifact: a token zeroed before LayerNorm becomes the constant $\beta$, all masked tokens share 1 key and form an attention sink, and part of PSBD-TM's effect is this off-distribution artifact | L22, L24 |
| E7 | clean AND fragility: clean evidence needs about 70% of tokens visible in every block, so it fails under masking | theory note, L19 |
| E8 | noise-mode coherence: WaNet's noise mode trains the model to reject any warp that is not the exact trigger field, so a perturbed triggered WaNet input looks like a rejected warp | L26 |
| E9 | position binding: the backdoor is tied to the trigger's absolute position through the position embeddings | new |
| E10 | computed against routed: a routed trigger is finished early and carried, a computed trigger is assembled across early and middle blocks from relations between tokens | new |
| E11 | area-additive evidence: each trigger pixel adds evidence to the target logit, no single token suffices and the logit difference falls in proportion to the masked trigger area | new |
| E12 | extent-dependent legibility: a global trigger's token sees only a piece of it and several tokens are needed | L26 |

Several experiment directories are owned by other agents or are already running, and this plan only reads them. Their READMEs are the record of what they measured, and an item here that needs one of them names it as the owner rather than editing it.

| directory | owner and state on 2026-09-29 | read-only for this plan |
|---|---|---|
| `experiments/why_token_masking_works/` | sections A to D done on the clearing panel of the time, X2 queued on the login GPU for the night of 2026-09-29 | yes |
| `experiments/why_psbd_works/` | 6 of 36 model runs done, 30 runs and the seed ensembles queued for the next GPU window, the WaNet probe of L26 (`wanet_probe.py`) not yet run | yes |
| `experiments/prediction_shift_phenomenon/` | claims 1, 2 and 4 read from the caches, claim 3 (final-layer features) waits for its GPU run | yes |
| `experiments/backdoor_manifestation/` | sanity gates passed, panel run of 83 runs in progress through `run_panel.sh` | yes |
| `experiments/cache_readouts/` | X3, X4 and X19 in progress on CPU by another agent, directory not yet created at the time of writing | yes |
| `experiments/literature_checks/` | X1 done (`wanet_destinations.py`) | no, X1b lands here |
| `experiments/theory_checks/` | the recomputes behind `docs/novel-designs-theory.md` | no |

The status words are **done** (a result exists in a tracked file), **running**, **queued** (scheduled on a named night), **planned** (designed, needs code), **not scheduled** (designed, below the cut for these 2 nights) and **dropped**. The proposed owner directories of planned items do not exist yet.

## Development set, held-out half and cost units

The user fixed a development set of 10 models so each test is quick. 3 of them belong to CIFAR-100 and Tiny ImageNet, the half `configs/psbd_basis.json` reserves for reporting, so exploring on them spends the held-out half. The rule adopted on 2026-09-29 swaps each of the 3 for a CIFAR-10 or GTSRB model of the same attack. The chosen list is to be written to `experiments/cache_readouts/dev_set.json` by the owner of that directory, and that file did not exist when this plan was merged, so the rule and the candidates below stand until it does.

The swap rule has 3 parts. The replacement has the same attack and the same poison rate where a panel model exists, the nearest rate otherwise. Among equals it prefers the model PSBD-TM reads lowest, since a test near ceiling decides nothing. A replacement that is not on the 2-point panel is labeled with the bar it does clear and reported apart.

| member | role | PSBD-TM | PSBD-RD | held out | swap candidate (PSBD-TM, PSBD-RD) |
|---|---|---|---|---|---|
| `vit_cifar10_wanet_0_1` | the WaNet failure | 0.459 | 0.947 | no | |
| `vit_tiny_wanet_0_05` | WaNet, the working case | 0.930 | 0.953 | yes | `vit_gtsrb_wanet_0_1` (0.948, 0.960), successful at the 5-point bar only. No CIFAR-10 or GTSRB WaNet model other than the failure is successful at the 2-point bar |
| `vit_cifar10_bpp_0_05` | PSBD-TM's weakest non-WaNet model | 0.801 | 0.880 | no | |
| `vit_gtsrb_bpp_0_01` | BPP where PSBD-TM leads | 0.898 | 0.755 | no | |
| `vit_cifar10_blend_0_1` | Blend where PSBD-RD leads | 0.906 | 0.991 | no | |
| `vit_cifar10_badnet_a2o_0_01` | the patch contrast | 0.982 | 0.309 | no | |
| `vit_gtsrb_tact_0_05` | trigger-conditional TaCT | 0.942 | 0.392 | no | |
| `vit_cifar10_tact_0_01` | trigger-conditional TaCT, PSBD-RD collapses | 0.979 | 0.464 | no | |
| `vit_cifar100_bpp_0_01` | BPP on a held-out dataset | 0.921 | 0.950 | yes | `vit_cifar10_bpp_0_01` (0.986, 0.970) at the same rate, or `vit_cifar10_bpp_0_1` (0.871, 0.875) if the rule's "lowest reading" clause is applied across rates |
| `vit_tiny_lf_0_01` | LF on a held-out dataset | 0.941 | 0.952 | yes | `vit_gtsrb_lf_0_01` (0.976, 0.987) or `vit_cifar10_lf_0_01` (0.979, 0.941), both at the same rate |

The AUROCs are the fractional statistic at the 0.25 quantile at each placement's adaptive rate, read on 2026-09-29 from each model's `psbd_metrics.json` with `select_rate_adaptively`. On the 54-model panel the development half holds 28 models (CIFAR-10 15, GTSRB 13) and the held-out half 26 (CIFAR-100 12, Tiny 14), the per-dataset counts of the gain table in `paper/headline.tex`.

Cost units assume ViT-B/16 in bfloat16 on the login-node A100 at about 1000 image forwards per second in batches. That is between the 0.0022 s per input of `experiments/psbd_cost/` (unbatched overhead included) and the 39 models in 11 min of `experiments/why_token_masking_works/`. The measured runs of `experiments/why_psbd_works/` took 225 s per model for a much larger battery, which says the units are optimistic by up to 2 times on a GPU shared with other jobs, so the schedule budgets twice the unit estimate.

| unit | what runs | forwards per model | time per model | time on the 10 models |
|---|---|---|---|---|
| U1 | 1 deterministic condition on 256 clean and triggered pairs | 512 | 1 s | 10 s |
| U2 | 1 stochastic condition on 256 pairs, 10 passes | 5120 | 5 s | 1 min |
| U3 | 1 operator at 1 rate on the 2000 validation images plus 1000 pairs, k = 3 | 12000 | 12 s | 2 min |

A 6-rate ladder of a new probe is therefore about 1.2 min per model and 12 min on the 10 models, the reduced sweep proposed for exploration. The full PSBD splits (2000 validation, 8000 clean, about 7200 triggered) cost about 4 times more and belong to confirmation only. Swin-S costs about 0.8 times as much per forward (`experiments/psbd_cost/`).

## Status on 2026-09-29

The table lists what is already decided or in flight, so the schedule does not repeat it. The literature and folk explanations that the running experiments already decided are in the next section by L and P ID.

| ID | what | status | where | result or state |
|---|---|---|---|---|
| X1 | where PSBD-TM sends triggered WaNet predictions | done | `experiments/literature_checks/wanet_destinations.py` | on `vit_cifar10_wanet_0_1` 0.729 of triggered passes change and only 0.158 of those land on the true class (read 2026-09-29), so E8's prediction of at least 0.7 failed on the failing cell |
| X2 | the 2 unswept replicates of the failing cell | queued, night of 2026-09-29 | `experiments/why_token_masking_works/` | no cache under `results/vit_cifar10_wanet_0_1_seed_1` yet |
| X3 | fusion rules for a specialist | done, CPU | `experiments/cache_readouts/` | the pre-registered weighted min-rank rule confirmed on the held-out half, the best-pass statistic did not |
| X4 | depth bands from the caches | running, CPU | `experiments/cache_readouts/` | every development model carries the 6 band placements |
| X19 (N17) | worst-pass and hard-label statistics | running, CPU | `experiments/cache_readouts/` | |
| L-tests of `why_psbd_works` | margin, direction, redundancy, low dimension, flatness, neuron bias, pass uncertainty, out of distribution, MLP neurons, missingness on 4 groups, P1 and P10 on the panel | 6 of 36 runs done, 30 queued | `experiments/why_psbd_works/` | verdicts in the next section |
| claim 3 of the phenomenon | final-layer features with and without dropout | queued | `experiments/prediction_shift_phenomenon/` | |
| manifestation panel | direction, TAC, ablation and steering on 43 backdoored and 40 benign-control runs | running | `experiments/backdoor_manifestation/` | |

The X1 table counts triggered passes on rows the backdoor fires on, at each model's PSBD-TM adaptive rate. The numbers are the output of `wanet_destinations.py` rerun on 2026-09-29.

| model | noise mode `cover_rate` | triggered passes changed | of those, to the true class | clean passes changed | of those, to the target |
|---|---|---|---|---|---|
| `vit_cifar10_wanet_0_1` | 0.2 | 0.729 | 0.158 | 0.852 | 0.002 |
| `vit_tiny_wanet_0_05` | 0.1 | 0.005 | 0.815 | 0.896 | 0.591 |
| `vit_tiny_wanet_0_1` | 0.2 | 0.037 | 0.515 | 0.860 | 0.208 |
| `vit_cifar10_wanet_0_05` | 0.1 | 0.117 | 0.266 | 0.801 | 0.341 |
| `vit_gtsrb_wanet_0_1` | 0.2 | 0.004 | 1.000 | 0.914 | 0.996 |
| `vit_cifar10_badnet_a2o_0_01` (control) | 0.0 | 0.153 | 0.214 | 0.819 | 0.023 |
| `vit_cifar10_blend_0_1` (control) | 0.0 | 0.161 | 0.085 | 0.815 | 0.001 |
| `vit_cifar10_bpp_0_05` (control) | 0.05 | 0.360 | 0.158 | 0.818 | 0.000 |

The failing cell is the only WaNet model where PSBD-TM moves most triggered predictions, and those moves do not go home to the true class. It is also the only WaNet model with no target attraction of clean predictions (0.002 against 0.208 to 0.996). On the other 4 WaNet models triggered predictions barely move, and the few that do lean toward the true class (0.266 to 1.000), which is the weak form of E8 on very few passes. E8 therefore does not explain the inversion, and X2 becomes the first test of the failure. Where the 0.842 of the failing cell's changed triggered passes land is the follow-up X1b.

## Literature and folk explanations by ID

Every L and P entry of the literature memo appears once here with its status and the item that carries its test. The verdicts from `experiments/why_psbd_works/` come from its README on 2026-09-29, measured on 1 Blend and 1 LF, 1 WaNet (`vit_tiny_wanet_0_1`), 1 BPP model and 2 benign controls, with P1 and P10 on the whole panel.

| ID | explanation | status | carried by | open test and cost |
|---|---|---|---|---|
| L1 | distance to the decision boundary | output-margin form refuted (P1), margin retention supported on 3 groups (triggered minus clean median retention 0.456 to 0.662) | `why_psbd_works` H-margin | input-space MDTD radius against PSBD-TM's attack ranking, GPU, not scheduled |
| L2, P4 | local curvature | partial for Blend and LF (0.820), refuted for WaNet (0.686) and BPP (0.770) at the smallest Gaussian rate | `why_psbd_works` H-flatness | the Rademacher sweep was dropped there on theory grounds |
| L3 | SAM makes backdoors easier to detect | partial, mechanism open | none | class-token attention on trigger tokens on the `_sam_rho_0_1` checkpoints, GPU, not scheduled (SAM is a side attempt) |
| L4, P8 | shortcuts learned first | open | T-list | `experiments/early_loss_signal/`, needs training |
| L5 | optimized scaling mask | open | none | token-level MSPC, GPU, not scheduled |
| L6 | small sufficient evidence set | supported on 3 groups (top direction holds 0.538 to 0.563) | `why_psbd_works` H-low-dim | patch groups queued there |
| L7, P3 | backdoor neurons | refuted for the residual basis and for the MLP hidden basis (0.972 to 0.998 of triggered answers kept with the top units zeroed) | `why_psbd_works` trigger neurons | S1 is its operator form |
| L8 | 1 linear direction read by PSU | supported (Spearman 0.911 to 0.976) | `why_psbd_works` H-direction | WaNet per-token spread, W8, queued in `why_psbd_works` |
| L9, P9 | latent separability and collapse | refuted as the mechanism | done | none |
| L10, P6 | clean inputs driven onto the target | refuted as the carrier (0.000 AUROC lost when the target's gain is redistributed) | `why_psbd_works` H-neuron-bias | patch groups queued there |
| L11 | STRIP entropy | measured as a detector | done | overlay-weight ladder, GPU, not scheduled |
| L12, P2 | MC dropout as epistemic uncertainty | class-blind uncertainty refuted (0.234 to 0.750) | `why_psbd_works` pass uncertainty | seed-ensemble disagreement queued there (`seed_ensemble.py`, "No seed ensemble yet") |
| L13 | SCALE-UP's poison-rate theorem | partial, holds for Blend, fails for BPP and LF | done | none |
| L14 | IBD-PSC's norm theorem | detector confirmed, theorem refuted (target share at most 0.223 at factor 15) | done | X27 |
| L15 | TeCo on Swin | open, a missing control | X31 | |
| L16 | BaDExpert and TED | measured as detectors | none | TED restricted to the class token, not scheduled |
| L17 | text precedent | consistent | none | none |
| L18 | input patch removal breaks patch triggers | confirmed as the reason PSBD's sign depends on the site | X29 | |
| L19 | occlusion robustness | confirmed for global triggers, contradicted for clean inputs | X28 | |
| L20 | residual routing and a late class-token read | confirmed on ViT for patch triggers | X8, X9, X12, X20, X21 | Swin port X30 |
| L21 | attention localizes the trigger | confirmed as localization, refuted as a static signature | X22 (N4) | |
| L22 | missingness native to transformers | supported for Blend, LF and BPP (within 0.010), refuted for WaNet (substitution reads 0.376 lower) at the adaptive rule | `why_psbd_works` missingness | X5, X13 at the matched rule |
| L23 | global triggers legible from any subset | supported on the 4 measured groups including 1 WaNet model (0.706), against 0.183 for WaNet in section D | `why_psbd_works` H-redundancy, section D | X17, Swin port X30 |
| L24 | sinks, registers, massive activations | trigger makes its own high-norm token, sink role open | X7 row B4 | |
| L25 | LayerNorm absorbs noise and not masking | confirmed on ViT | done | Swin shares, not scheduled |
| L26 | frequency content and WaNet's coherence check | E8's first prediction failed on the failing cell (X1) | X6, X15, X16, W8 | |
| L27 | adversarial examples and backdoors | partial, 1 CIFAR-100 model per attack | none | PGD on the panel, not scheduled |
| P1 | just more confident | refuted panel-wide (PSBD-TM beats the confidence ceiling $A^*(P_c)$ on 50 of 54 models) | done | none |
| P5 | out of distribution | refuted for Blend, LF and BPP, partial for WaNet (0.531) | done | none |
| P7 | the trigger is small | partial, replaced by L20 | done | none |
| P10 | the target class is easy | partial for BadNets, Blend, BPP, LF and WaNet, refuted for TaCT (0.032) | done | none |
| P11 | trigger features have a larger norm | partial for patch triggers | done | none |
| P12 | attention hijacked by the trigger | refuted as a signature, confirmed as the route | done | X22 |

## Deduplication map

The 3 source documents named many tests more than once. Each row keeps 1 canonical ID and lists the aliases that now point to it. The B, G and W rows that remain only as rows of a family table keep their letters inside their family.

| canonical ID | aliases | question |
|---|---|---|
| X1 | W1 | where PSBD-TM sends triggered WaNet predictions |
| X1b | (new) | where the failing cell's triggered shifts land, and X1 on the replicates and Swin |
| X2 | W2 | is the 0.459 WaNet cell a checkpoint accident |
| X3 | N12, the mixture and union of `docs/novel-designs-theory.md`, the averaged half of the combined design | which fusion rule a specialist needs |
| X4 | G6, W9 | which depth band breaks which trigger |
| X5 | N10, workflow step 5 | is every token-mask result about information or about the constant replacement |
| X6 | W3, N15, specialist candidate 2 | does noise mode make triggered WaNet fragile to sub-pixel jitter, and is that a specialist |
| X7 | B1 to B4, B10 | the BadNets dose-response family |
| X8 | N2, theory candidate (2), workflow step 3 | does the class token read the trigger directly |
| X9 | B5, B6 | does triggered survival follow $1 - p^{mB}$ |
| X10 | B7, B8, the position swap half of N6 | is the backdoor bound to the trigger's position |
| X11 | B9, G4 | is the trigger sufficient on a blank image |
| X12 | B11, G7, workflow step 2 | where the triggered state is sufficient, by denoising |
| X13 | N1, N14, theory candidate (1), optimal-probe rule 4 | true key mask and sink-only injection |
| X14 | N13 | stratified token masking |
| X15 | W4, G5, specialist candidates 3, 6, 7 and 9, test-time augmentation consistency | the WaNet transformation battery |
| X16 | W5, N6, theory candidate (6), specialist candidate 8 | position-content coherence and position-embedding jitter |
| X17 | G1, W6, the structured-mask prediction of L26 | contiguous against scattered visible subsets |
| X18 | G3, W7 | test-time trigger strength against PSU |
| X19 | N17 | which pass statistic separates best |
| X20 | N11, workflow step 4 readout | logit lens and per-head attribution under PSBD-TM |
| X21 | workflow step 6 | self-repair after a late knockout |
| X22 | N3, N4, theory candidates (3) and (4), L21's prediction | attention temperature and attention-rank masking |
| X23 | G2, W10 | the trigger stamped on only part of the image |
| X24 | the mixed-operator row of the optimal-probe table, the joint half of the combined design | token mask and late residual dropout applied in the same pass |
| X25 | the 2 depth-schedule rows of the optimal-probe table | front-loaded against back-loaded masking rates |
| X26 | per-head masking, post-softmax attention dropout, ToMe merging | the remaining operator candidates |
| X27 | L14's ladder test | target share across the IBD-PSC amplification ladder |
| X28 | L19's resolution test | clean fragility by input resolution on the benign references |
| X29 | L18's survival curve | triggered survival by the number of surviving trigger tokens at the embedding |
| X30 | L20 and L23 on Swin | sections A and D ported to Swin |
| X31 | L15 | TeCo on the Swin panel |
| N5 | theory candidate (5) | random-rotation subspace dropout against the axis-aligned channel mask |
| N8 | theory candidate (7) | random block skipping |
| N9 | none | random shifted-window offsets on Swin |
| N16 | none | depth-contrast score from the band caches |
| S1 | none | active-neuron dropout in the MLP hidden layer |
| S2 | related to N4 by its ranking of tokens | importance-ranked token removal and its largest sudden drop |
| N7 | none | pixel shuffling inside a patch, dropped in favor of X6 |
| W8 | L26's probe, `why_psbd_works/wanet_probe.py` | per-token probe for trigger-warped against noise-warped images |
| T1 to T6 | none | designs that need training |

## Schedule for the login GPU

The login GPU is free to this analysis from 17:00 to 07:00, and the driver scripts start no new model between 06:30 and 17:00. The main night is 2026-09-30 17:00 to 2026-10-01 07:00, and the fallback night is 2026-10-01 17:00 to 2026-10-02 07:00. Both follow the same 4 rules: carryover of the night of 2026-09-29 runs first, every new hook gets a GPU smoke test on 1 model before its first real run, every run holds `flock scratch/gpu.lock` per model with a per-process memory cap, and every run writes 1 JSON per model so a stopped night resumes. The CPU reads (X1b, X3, X4, X19, N16, X27, X28) run during the day of 2026-09-30 and need no slot.

The order inside a night is explanatory value per GPU minute. Value counts how many open explanations an item decides and whether its answer changes what the paper claims or which probe is built. The minutes below are the unit estimates on the development set, and the budget column doubles them for the shared GPU.

**Main night, 2026-09-30 17:00 to 2026-10-01 07:00.**

| slot | item | models | estimate (min) | budget (min) | decides |
|---|---|---|---|---|---|
| 0 | carryover of the night of 2026-09-29: X2, the `why_psbd_works` queue, claim 3 of the phenomenon, the manifestation panel | as queued by their owners | unknown | until their markers `scratch/gpu_done_{general,tm,phenomenon,manifestation}` exist | the failing cell, the L tests on patch groups |
| 1 | GPU smoke of every new hook (attention mask, sink injection, replacement bank, stratified mask, jitter) on `vit_cifar10_badnet_a2o_0_01`, 64 pairs | 1 | 5 | 10 | code correctness |
| 2 | X8 attention knockout | 3 patch | 1 | 2 | direct read against relay (E1 strong form) |
| 3 | X21 self-repair | 3 patch | 1 | 2 | why the last block alone keeps 0.934 |
| 4 | X9 survival law | 3 patch | 4 | 8 | effective $m$ and $B$, E1 against E11 |
| 5 | X12 denoising maps | 3 patch plus 2 WaNet | 3 | 6 | where the triggered state is sufficient, E1 against E10 |
| 6 | X20 logit lens and head attribution | 10 | 2 | 4 | where the triggered logit difference appears and survives |
| 7 | X5 replacement rules on the section A and D rows | 10 | 10 | 20 | E6 against information removal for every token-mask claim |
| 8 | X13 key mask (N1), 6-rate ladder | 10 | 12 | 24 | E6 from the operator side |
| 9 | X13 sink-only injection (N14), 6-rate ladder | 10 | 12 | 24 | E6 directly |
| 10 | X6 sub-pixel jitter on the development set | 10 | 8 | 16 | E8 as a specialist sign |
| 11 | X6 on the WaNet set | 27 WaNet models and replicates | 22 | 44 | noise mode against none, the falsification test |
| 12 | X16 position-content coherence and position-embedding jitter | 10 | 7 | 14 | relational E8, E9 |
| 13 | X15 transformation battery | 10 plus the WaNet set | 27 | 54 | which input transform breaks WaNet and BPP only |
| 14 | X7 BadNets dose-response family | 3 patch | 6 | 12 | E1 against E11, the step against the line |
| 15 | X10 relocation and swaps | 3 patch | 5 | 10 | E9 |
| 16 | X11 trigger on blank carriers | 3 patch plus 3 global | 12 | 24 | context-free trigger |
| 17 | X14 stratified masking (N13), 6-rate ladder | 10 | 12 | 24 | the best-supported drop-in replacement for PSBD-TM |
| 18 | N5 rotation against axis-aligned subspace dropout at matched shift | 10 | 12 | 24 | the contradiction between the 2 source predictions |
| 19 | S1 active-neuron MLP dropout, 1 sweep | 10 | 40 | 80 | ShortcutProbe's operator on ViT |
| 20 | X17 contiguous against scattered subsets | 10 | 7 | 14 | E2 against E12 |
| 21 | X18 test-time strength | 6 global and WaNet | 6 | 12 | E3 against E4, W7 |
| 22 | X23 partial-area trigger | 6 global and WaNet | 7 | 14 | E2 against E12 on the input side |
| 23 | X3s Swin confirmation of the middle-band fusion | the Swin panel | 15 | 90 | the only confirmation of the late band that its selection never saw |

The main night's analysis block (slots 1 to 22) budgets about 7.5 h, which leaves slot 0 about 6.5 h of the 14 h window. If slot 0 runs longer, slots 20 to 22 move to the fallback night. Slot 23 is the Swin confirmation of the late-band fusion. The band `pre_residual_blocks_9_12` was chosen after WaNet readings on all 4 ViT datasets and first read on 39 panel models that include the CIFAR-100 and Tiny half, so the held-out confirmation in `experiments/cache_readouts/` confirms the fusion rule but not the band. Swin-S has never been swept at a late band. Before its sweep, the Swin band (the middle third of the block stack by index, matching ViT blocks 5 to 8, chosen by the user on 2026-09-29 over the late band because it is the ViT's best residual placement, reaches the adaptive rate on all 54 panel models and is where the global triggers write their direction into the class token), the rule (the plain minimum of the 2 clean-validation percentiles, no weights, chosen by the user on 2026-09-29 for simplicity) and the prediction (TPR at 1%, 5% and 10% FPR above PSBD-TM alone) are written into `experiments/cache_readouts/preregistration_swin.json`. The late-band fill was withdrawn: the 15 models already hold the full ladder and late-band dropout never reaches the 0.8 clean shift on them. SIG reruns and k up to 20 do not start on the main night unless every slot above has finished, because they wait until the analysis is done.

**Fallback night, 2026-10-01 17:00 to 2026-10-02 07:00.**

| slot | item | models | estimate (min) | budget (min) | purpose |
|---|---|---|---|---|---|
| 1 | every main-night slot not finished | as above | | | |
| 2 | X8, X9 and X7 on the paper's patch set | 12 BadNets and 3 trigger-conditional TaCT | 45 | 90 | the confirmed numbers for the paper |
| 3 | protocol step 3, the 2 surviving candidates on the development half, full splits, 6 rates | 28 panel models plus the labeled WaNet models | 280 | 400 | pick 1 design by the frozen metric |
| 4 | X22 sharpening and attention-rank masking | 10 | 24 | 48 | category diagnostic, predicted to fail one-sided |
| 5 | S2 importance-ranked token removal | 3 patch plus `vit_cifar10_blend_0_1` and `vit_cifar10_wanet_0_1` | 100 | 150 | ShortcutProbe's deterministic partner on tokens |
| 6 | X24 mixed operator and X25 depth schedules | 10 | 36 | 72 | lower-priority probe shapes |
| 7 | N8 block skipping | 10 | 12 | 24 | predicted weak by the theory triage |
| 8 | SIG reruns (`docs/runs/2026-09-29-gpu-queue.md`, item 2) | Swin and ViT SIG | 180 to 240 | | only once every analysis slot above has finished |
| 9 | k up to 20 (item 3 of the same record) | 40 ViT models per placement | 240 | | last |

Protocol steps 4 (held-out evaluation on 26 models) and 5 (Swin) come after the freeze and do not fit these 2 nights. They go to the PBS GPU queue once the frozen candidate is written into `configs/psbd_basis.json`. X29, X30, X31, N9 and X26 are not scheduled on either night.

## CPU reads from the caches

These items read `per_pass_probs` and `per_pass_argmax` (shape (k, N) under `results/<folder>/psbd/<placement>/`) or `psbd_metrics.json` and need no GPU. They run during the day of 2026-09-30.

**X1. Destination of shifted triggered WaNet predictions.**

- Hypothesis: E8 (L26).
- Comparison and control: share of changed triggered passes landing on the true class on noise-mode WaNet models, against BadNets, Blend and BPP models as the control.
- Prediction: E8 predicts at least 0.7 of triggered shifts to the true class on noise-mode models. Generic fragility predicts destinations that mirror the clean shifts.
- Refuted if: the failing cell sends fewer than 0.3 of its triggered shifts to the true class.
- Models, compute, owner: 5 ViT WaNet models and 3 controls, CPU, seconds, `experiments/literature_checks/`.
- Status: done, refuted on the failing cell (0.158), see the status section.

The original design took each triggered row's true class through `split_manifest.json` and `defenses.decision.pair_clean_to_backdoor`. It then split the changed passes into those landing on the true class and those landing elsewhere. The figure is a grouped bar chart of destination shares (true class, target, largest other class, rest) for triggered and clean rows, 1 group per model, ordered by noise mode. The design also named the 6 ViT replicates without noise mode and the 14 Swin WaNet models, which the script did not cover. Those move to X1b. At the literature memo's reading of 2026-09-29 on the 57-model panel (historical), the largest single class took 0.66 of shifted clean predictions on WaNet under PSBD-TM.

**X1b. Where the failing cell's triggered shifts land, and X1 beyond the panel.**

- Hypothesis: E8 against a default-class collapse (the largest class takes 0.54 to 0.87 of shifted clean predictions under PSBD-TM in the literature memo's reading).
- Comparison and control: the destination split of X1 extended to the largest single class and the unperturbed runner-up class, on the failing cell, with the clean rows of the same model as the control.
- Prediction: a default-class collapse sends at least 0.5 of the failing cell's changed triggered passes to the class that takes the most shifted clean predictions. A trigger that decays to the runner-up sends most to the runner-up.
- Refuted if: neither the largest class nor the runner-up takes more than 0.3.
- Models, compute, owner: the failing cell first, then the 6 ViT replicates without noise mode with caches (`vit_cifar100_wanet_0_1_seed_{1,2}`, `vit_tiny_wanet_0_05_seed_{1,2}`, `vit_tiny_wanet_0_1_seed_{1,2}`) and the Swin WaNet models, CPU, seconds, `experiments/literature_checks/`.
- Status: planned, day of 2026-09-30.

**X3. Fusion rules for a specialist.**

- Hypothesis: E10 with the mixture against union derivation of `docs/novel-designs-theory.md`.
- Comparison and control: PSBD-TM combined with `pre_residual_blocks_9_12`, `pre_residual_blocks_5_8` and `post_residual` under 5 rules, with PSBD-TM alone as the control: the mean of fractional PSU, min-rank, weighted min-rank at budget shares 0.8 and 0.2 and at 0.9 and 0.1, and Fisher's combination of clean-validation percentiles.
- Prediction: the late residual member helps WaNet and Blend and hurts TaCT under the mean and min-rank rules, the weighted rule keeps TaCT within 0.01 while keeping most of the WaNet gain. The copula model predicts that the union beats the mixture whenever the weaker member's separation ratio $\lambda$ is below about 0.55 and that the gap comes from the WaNet models.
- Refuted if: the weighted rule loses more than 0.01 on TaCT, or the mixture beats the union on the models with $\lambda < 0.55$.
- Models, compute, owner: the development set, then the 54 panel models at the rate nearest the target where no adaptive rate exists, CPU, seconds, `experiments/cache_readouts/`.
- Status: running.

Report AUROC, TPR at 10% and 20% FPR and the FPR realized on the clean test split. The figure is TPR at 10% FPR per attack for each rule, with PSBD-TM alone as the reference line. A historical reading on 39 models (2026-09-29, the 39 models with an adaptive late-band rate) gave PSBD-TM 0.9659, late-band residual dropout 0.9200, mixture 0.9787 and union 0.9831, reproduced by `experiments/theory_checks/mix_union.py`. The current 54-model macros for the PSBD-TM plus PSBD-RD union are a gain of +0.008 [-0.009, +0.029] and 0.910 on the failing WaNet cell (`\ProbeUnionTmRdGain`, `\ProbeUnionWanetCifarOneZeroTmRd`). The late band was picked after its WaNet readings were seen, so its result counted only after the confirmation on the held-out half, which it passed (`experiments/cache_readouts/README.md`).

**X4. Depth bands from the caches.**

- Hypothesis: E1, E2 and E10.
- Comparison and control: for each band (blocks 1 to 4, 5 to 8, 9 to 12) and attack, the triggered minus clean shift share at the band's top rate, under `before_attention_norm_blocks_*_token_mask` and `pre_residual_blocks_*`, with the all-block placement as the control.
- Prediction: E1 predicts that token masking in blocks 1 to 4 or 5 to 8 leaves triggered patch predictions alone while breaking clean ones and that blocks 9 to 12 hit both. E10 predicts that residual dropout in blocks 1 to 4 breaks triggered WaNet more than triggered BadNets, because WaNet's evidence is assembled early (block 4 recovery of 0.880 for the most displaced tokens in `failure_modes`). E2 predicts that no token-mask band moves global triggers.
- Refuted if: early token-mask bands break triggered patch predictions as often as clean ones (E1), or early residual dropout breaks triggered WaNet no more than triggered BadNets (E10).
- Models, compute, owner: the development set, CPU, seconds, `experiments/cache_readouts/`.
- Status: running.

The figure is a heat map, attack by band, of the triggered minus clean shift share, 1 panel per operator. The band token masks often cannot reach the 0.8 shift target, so the reading is at each band's top rate as designed.

**X19. Pass statistics (N17).**

- Hypothesis: E1's OR structure makes triggered patch survival nearly all-or-nothing per pass.
- Comparison and control: the mean ratio (the current statistic) against the worst pass, the best pass, the count of label-keeping passes and the spread across passes, from `per_pass_probs` of PSBD-TM and PSBD-RD.
- Prediction: the worst-pass statistic separates at least as well as the mean and raises TPR at 1% FPR on patch triggers, other attacks unchanged.
- Refuted if: the worst pass lowers TPR at 1% FPR on BadNets or TaCT.
- Models, compute, owner: the development set then the panel, CPU, `experiments/cache_readouts/`.
- Status: running.

The figure is TPR at 1% and 10% FPR per statistic and attack. The closest prior work is PatchCleanser's agreement test and TeCo's hard-label severities.

**N16. Depth-contrast score.**

- Hypothesis: E1 in its "early-immune" reading.
- Comparison and control: fractional PSU from token masking in blocks 1 to 8 divided by that from blocks 9 to 12 at the same rate, against PSBD-TM's all-block score.
- Prediction: a patch-triggered input is immune to early hiding and exposed to late hiding while a clean input is fragile to both, so the ratio is small for triggered patch inputs. The division also cancels each image's general fragility, which should tighten the clean distribution at the low end, where 0.339 of PSBD-TM's false positives are clean images that never flip (`why_psbd_works`). Blend and quantization unchanged or noisy, warp unknown.
- Refuted if: TPR at 1% FPR on patch triggers does not rise over PSBD-TM.
- Models, compute, owner: the development set, CPU on the band caches, suggested to `experiments/cache_readouts/` since it reads the same files as X4.
- Status: planned. I found no prior work.

**X27. Target share across the IBD-PSC amplification ladder (L14).**

- Hypothesis: E5 in IBD-PSC's Theorem 3.1 form.
- Comparison and control: target share of shifted clean predictions under `gain_scale` at `mlp_norm_out` at every factor of the ladder, against the flat share the fragility account predicts.
- Prediction: the theorem predicts a monotone rise toward 1 past some factor. The fragility account predicts a flat share. At factor 15 the current reading is 0.001 (TaCT) to 0.223 (BadNets) with 0.92 to 0.97 of clean predictions shifting (`experiments/literature_checks/badnet_pm.py`).
- Refuted if: the share rises monotonically past 0.5.
- Models, compute, owner: the 54 panel models, CPU, `experiments/literature_checks/`.
- Status: planned.

**X28. Clean fragility by input resolution (L19).**

- Hypothesis: upsampled low-resolution inputs make clean predictions fragile, which is why our models contradict Naseer et al.
- Comparison and control: the clean shift ratio at token-mask rate 0.3 on `vit_cifar10_benign`, `vit_cifar100_benign` and `vit_tiny_benign`, from the cached `before_attention_norm_token_mask` ladder.
- Prediction: highest on CIFAR (32 pixels) and lowest on Tiny (64 pixels), by at least 0.1.
- Refuted if: the gap is below 0.05 or reversed.
- Models, compute, owner: 3 benign models, CPU, `experiments/literature_checks/`.
- Status: planned.

## Patch-trigger mechanism

These items extend section A of `experiments/why_token_masking_works/` on the 3 patch models of the development set (`vit_cifar10_badnet_a2o_0_01`, `vit_gtsrb_tact_0_05`, `vit_cifar10_tact_0_01`) and confirm on the 12 BadNets and 3 trigger-conditional TaCT panel models. Their proposed owner is a new directory `experiments/patch_trigger_mechanism/`, since `why_token_masking_works` is read-only here.

A single geometric fact shapes every row. The trigger is a 3 by 3 checkerboard in the bottom-right corner of a 32 pixel image (`attacks/badnet.py`), and the model upsamples by 7 to 224, so the patch covers pixels 203 to 223 in each axis. Token 12 (pixels 192 to 207) receives 5 of those pixel rows and token 13 (pixels 208 to 223) receives 16. The 4 trigger tokens therefore carry very different amounts of trigger: (13,13) about 256 square pixels, (12,13) and (13,12) about 80 each and (12,12) about 25. These areas come from the geometry, bilinear upsampling blurs the edges by a few pixels and they were not measured. On Tiny ImageNet (64 pixels, factor 3.5) the patch spans pixels 213.5 to 224 and falls inside token (13,13) alone, which is why `trigger_tokens` returns 1 token there. Every "k of 4" row must say which k tokens, ordered by coverage.

**X7. The BadNets dose-response family.**

- Hypothesis: E1 against E11, with E6 and E9 as side readings.
- Comparison and control: the rows B1 to B11 below, each with a matched random or adjacent-token control.
- Prediction: per row in the table.
- Refuted if: E1 is refuted when survival falls in proportion to masked area across the B1 subsets (a line), E11 when survival holds near 1 until the last sufficient token goes (a step).
- Models, compute, owner: 3 patch development models (6 min), then 12 BadNets and 3 trigger-conditional TaCT (about 30 min), GPU, `experiments/patch_trigger_mechanism/`.
- Status: planned, main night slot 14 and fallback slot 2.

| row | manipulation | E1, 1 intact read suffices | E11, area-additive | E6, sink artifact | E9, position binding |
|---|---|---|---|---|---|
| B1 | each of the 15 nonempty subsets of the 4 trigger tokens masked in all 12 blocks | survival near 1 while (13,13) or both strips stay visible, near 0 only when all 4 go (section A reads 0.002) | logit difference falls in proportion to masked area, masking (13,13) alone costs about 0.58 of it | same as E1 under the key mask, lower survival under the zero mask when many tokens are masked | same as E1 |
| B2 | the same subsets masked only in blocks 9 to 12, and only in 5 to 8 | late only reproduces B1 partly (section A reads 0.200 for all 4), 5 to 8 changes little | graded in both bands | no difference from E1 | no difference from E1 |
| B3 | as many random non-trigger tokens, the 5 tokens bordering the trigger or 4 tokens in the far corner, each in all 12 blocks | 1.000 for random and far, near 1 for the border unless the border relays the trigger | 1.000 for all | 1.000 | 1.000 |
| B4 | only the trigger and the class token visible, every other token masked in all blocks, under the zero mask and the key mask (also L24's complement-set test in blocks 9 to 12) | target kept at 0.9 or more under the key mask | target kept only if the trigger area alone outweighs the lost content | zero mask far below key mask, since 192 $\beta$ tokens form 1 sink | target kept |
| B5 | see X9 | | | | |
| B6 | see X9 | | | | |
| B7 | see X10 | | | | |
| B8 | see X10 | | | | |
| B9 | see X11 | | | | |
| B10 | 1 to 9 of the 9 checkerboard pixels kept, and the whole patch at contrast 0.25, 0.5 and 0.75 | a threshold: success collapses once no token holds a legible piece | success and logit difference fall smoothly | no difference | no difference |
| B11 | see X12 | | | | |

E4 (confidence) predicts that survival in every row tracks the triggered input's unperturbed confidence, so the B10 contrast ladder separates it best. E5 does not apply to triggered inputs. The TaCT trigger-conditional models run the same family with 1 distinct prediction: B9 should fail on them, since their trigger fires only on source-class content, and a TaCT trigger on a gray image should give the gray image's default class.

The single figure for the family is a 2-panel plot. On the left, survival and the logit difference against the masked trigger area for all B1 subsets, with the E1 step and the E11 line drawn. On the right, survival against $p$ for X9 with the $1 - p^{4m}$ curves.

**X8. Attention knockout of the trigger read (N2, theory candidate (2)).**

- Hypothesis: E1 in its strong form (a direct class-token read, L20, P12) against a relay through other patch tokens.
- Comparison and control: pre-softmax logits set to negative infinity on 3 edge sets (class-token query to trigger-token keys, non-trigger patch queries to trigger-token keys, both) in blocks 1 to 4, 5 to 8, 9 to 12 and each single block 9 to 12, against a random 4-key knockout per band.
- Prediction: a direct read predicts that the class-token knockout in blocks 9 to 12 alone reproduces the 0.200 of section A and that the patch-only knockout leaves attack success near 1. A relay predicts the reverse: blocking only the class token's edges leaves the attack intact and blocking the patch edges in blocks 5 to 8 matters.
- Refuted if: E1's strong form is refuted when the class-token knockout in blocks 9 to 12 leaves attack success above 0.8.
- Models, compute, owner: 3 patch development models, then the paper's patch set, GPU, 20 U1 conditions (20 s per model), `experiments/patch_trigger_mechanism/`.
- Status: planned, main night slot 2. Needs an `attn_mask` in `defenses.operators.masked_attention_forward`, which already recomputes attention through `scaled_dot_product_attention`.

The figure is a grid of attack success, rows for edge sets and columns for bands, with the logit difference in each cell. As a probe, the class-token-only read mask (N2) has a flaw: clean evidence is relayed among patch tokens by the middle blocks, so the class token can recover it from any visible token and clean inputs become robust too. The theory triage predicts a clean shift ratio far below PSBD-TM's at equal $p$ and a rate near 0.9 to reach 0.8 (band-restricted token masking of every row in blocks 9 to 12 reached only 0.43 at p 0.99), and patch AUROC 0.5 to 0.8. Restricting it to blocks 9 to 12 is the only version worth a ladder, and it is a routing test rather than a detector.

**X9. The survival law under stochastic trigger-only masking (B5, B6).**

- Hypothesis: E1 with $B$ late reading blocks against E11.
- Comparison and control: only the trigger tokens masked, each independently per block with probability $p$ in 0.3, 0.5, 0.7, 0.8, 0.9, 0.95 and 0.99 over 10 passes, then with 1 draw per pass shared by all 12 blocks. B6 adds the whole trigger masked with probability 0.5 per pass against each token independently, which predicts 0.5 kept for the whole trigger and $1 - 0.5^{m}$ for independent tokens under E1 and a graded curve under E11.
- Prediction: E1 predicts the survival laws below, E11 a graded curve with no power-law shape.
- Refuted if: neither law fits within 0.05 at any $(m, B)$ with $m \le 4$ and $B \le 8$.
- Models, compute, owner: 3 patch development models, GPU, 14 U2 conditions (70 s per model), `experiments/patch_trigger_mechanism/`.
- Status: planned, main night slot 4.

$$S_{\mathrm{indep}}(p) = 1 - p^{mB}, \qquad S_{\mathrm{shared}}(p) = 1 - p^{m}$$

| symbol | meaning |
|---|---|
| $S(p)$ | share of triggered predictions still on the target |
| $p$ | per-block masking probability of each trigger token |
| $m$ | number of trigger tokens that each suffice alone |
| $B$ | number of late blocks that each read the trigger |

The table gives the predicted survival for $B = 4$, computed from the 2 laws. Section B's Tiny count (450 of 7670 passes against 0.0625 predicted) is the only point measured so far.

| m | p 0.3 | p 0.5 | p 0.7 | p 0.8 | p 0.9 | p 0.95 | p 0.99 |
|---|---|---|---|---|---|---|---|
| 1 | 0.992 | 0.938 | 0.760 | 0.590 | 0.344 | 0.185 | 0.039 |
| 2 | 1.000 | 0.996 | 0.942 | 0.832 | 0.570 | 0.337 | 0.077 |
| 4 | 1.000 | 1.000 | 0.997 | 0.972 | 0.815 | 0.560 | 0.149 |

Fitting $m$ and $B$ to the measured curve gives the effective number of sufficient tokens and reading blocks. The figure plots survival against $p$ with both fitted laws.

**X10. Trigger relocation and swaps (B7, B8).**

- Hypothesis: E9 against a content-only read.
- Comparison and control: the checkerboard pasted at the trained corner (control), the other 3 corners and the center, plus 2 offsets of the trained corner that cover 6 and 9 tokens instead of 4. Then the trigger-token content swapped with 4 other positions after the patch embedding (content moves, position embedding stays), and position embeddings swapped alone.
- Prediction: E9 predicts attack success drops sharply off the trained corner, as Li et al. found on ConvNets, and that the position-embedding swap breaks it. A content-only read predicts attack success near 1 everywhere, which Naseer et al.'s near permutation invariance makes plausible. The grid offsets change $m$ on the same model, so PSBD-TM's triggered survival at the adaptive rate should rise with $m$ as $1 - p^{4m}$.
- Refuted if: E9 is refuted when attack success off the trained corner stays above 0.9, the content read when it falls below 0.5.
- Models, compute, owner: 3 patch development models, GPU, about 12 U1 conditions plus 3 U3 readings (5 min), `experiments/patch_trigger_mechanism/`.
- Status: planned, main night slot 15.

The figure plots attack success and PSBD-TM AUROC per position, labeled with $m$. T3 and T4 are the trained versions of the same question.

**X11. The trigger on blank, noise and foreign images (B9, G4).**

- Hypothesis: E1 (a context-free trigger) against a context-dependent trigger.
- Comparison and control: the trigger pasted onto a gray image, a black image, uniform noise, 256 images of another dataset and a clean image of the target class, each scored by PSBD-TM at the model's adaptive rate, against the same images without the trigger.
- Prediction: E1 predicts the target on at least 0.9 of them and a low PSBD-TM score (flagged), while the untriggered images get a default class and a high score. For global triggers (G4) the blend pattern alone on gray should also carry the target and the full direction. For WaNet a warp of a flat image is flat and should carry nothing. TaCT should fail on gray (source-conditional).
- Refuted if: attack success on gray and noise carriers is below 0.5 for BadNets.
- Models, compute, owner: 3 patch plus 3 global development models, GPU, 5 carriers by 2 by U3 (2 min per model, 12 min), `experiments/patch_trigger_mechanism/`.
- Status: planned, main night slot 16.

The figure shows attack success and the mean PSBD-TM score per carrier, triggered against untriggered. Lapuschkin et al.'s Clever Hans transplant and SentiNet's inert-pattern control are the precedents.

**X12. Denoising maps of sufficiency (B11, G7).**

- Hypothesis: E1 against E10.
- Comparison and control: the triggered residual stream patched into the clean run at every block 1 to 12 for 4 groups (trigger tokens, class token, a random group of the same size and the tokens adjacent to the trigger), the random group as the control. `failure_modes` already patched the other direction (clean into triggered) at blocks 4, 8 and 12. For global triggers (G7) the group is a random fraction of positions.
- Prediction: E1 predicts the trigger tokens' stream alone is sufficient at every block up to about 9 and stops being sufficient after the class token has read it, while the class token becomes sufficient only in blocks 9 to 12. For WaNet, E10 predicts no small token group sufficient early and a class-token takeover late. For global triggers, E2 predicts a small random fraction suffices at every block.
- Refuted if: the trigger tokens' stream is not sufficient at block 6 on BadNets (recovered logit difference below 0.5).
- Models, compute, owner: 3 patch and 2 WaNet development models, GPU, attribution patching over all (block, token) pairs first then exact patching of the top 20, 48 U1 conditions plus 2 passes (under 1 min per model), `experiments/patch_trigger_mechanism/`.
- Status: planned, main night slot 5.

The metric is the logit difference, target minus true class, as Zhang and Nanda recommend. The figure is a heat map, block against group, of the recovered logit difference, with the attribution estimate in a twin panel.

**X20. Logit lens and per-head attribution under PSBD-TM (N11).**

- Hypothesis: E1 in its readout form.
- Comparison and control: the final LayerNorm and head applied to the class token after every block, and the last 4 blocks' target-minus-true logit difference attributed to each head, for triggered and clean inputs, unperturbed and averaged over 10 PSBD-TM passes. N11 adds a free statistic from the same hooks: how consistently the late readouts (blocks 8 to 12) keep the unperturbed label across passes.
- Prediction: the triggered difference appears in blocks 9 to 12 through a few heads that keep their contribution under masking, while clean evidence builds from block 6 on and loses it. N11 reads strong on patch and blend, unknown on warp.
- Refuted if: the triggered difference is already present at block 6 at more than half its final size.
- Models, compute, owner: the development set, GPU, 1 U2 per model (2 min), `experiments/patch_trigger_mechanism/`.
- Status: planned, main night slot 6.

The figure plots the logit difference against block, 4 curves. `experiments/prediction_depth/` already measures the unperturbed version.

**X21. Self-repair after a late knockout.**

- Hypothesis: the hydra effect (McGrath et al., Rushing and Nanda) explains why masking the trigger in the last block alone keeps 0.934 of triggered BadNets predictions.
- Comparison and control: the class token's read of the trigger knocked out in block 12 only, measuring how the per-head contributions of blocks 9 to 11 change, against the same knockout on a clean input.
- Prediction: the blocks 9 to 11 direct contributions grow on triggered inputs.
- Refuted if: they do not change by more than 5% of the block-12 contribution.
- Models, compute, owner: 3 patch development models, GPU, a few U1 conditions (1 min), `experiments/patch_trigger_mechanism/`.
- Status: planned, main night slot 3.

**X29. Triggered survival by surviving trigger tokens at the embedding (L18).**

- Hypothesis: L18, input patch removal breaks patch triggers.
- Comparison and control: `after_embedding_token_mask` with masks recorded as section B does, triggered survival tabulated by the number $j$ of surviving trigger tokens, against PSBD-TM's late-block survival.
- Prediction: survival near 0 at $j = 0$ and a rise with $j$ that reaches at least 0.9 at $j = 4$, with $j = 1$ well below the 0.930 that a single late-block token achieves under PSBD-TM.
- Refuted if: survival at $j = 1$ is at or above 0.9.
- Models, compute, owner: 3 patch development models, GPU, 1 U2 per model, `experiments/patch_trigger_mechanism/`.
- Status: not scheduled.

## The token-mask artifact

PSBD-TM zeroes the input of `ln_1`, and with the LayerNorm's epsilon the output for every masked token is exactly $\beta$, whatever the image, the position or the block (`docs/why-psbd-works-theory.md`). No natural token has zero variance across channels, so no real input ever produces $\beta$ at the attention input. Every masked token then carries the same key $W_k\beta + b_k$ and value $W_v\beta + b_v$, so $r$ masked tokens act as 1 token repeated $r$ times whose attention mass grows with $r$, and the masked token's own query is replaced by a constant, so its residual update is a generic write. The residual entry itself is untouched, which is why this is an ablation of the attention read and not of the token. In Li and Janson's terms it is neither zero, mean nor resample ablation of the read, and it carries exactly the distribution shift Zhang and Nanda warn about. The 2 items below decide whether the paper's mechanism statements are about information or about this artifact, and their proposed owner is `experiments/mask_replacement/`.

**X5. Resample and mean ablation versions of the key token-mask results (N10).**

- Hypothesis: E6 against information removal (E1, E7).
- Comparison and control: the rows of section A (trigger tokens masked in all 12 blocks or in blocks 1 to 4, 5 to 8 or 9 to 12, plus random tokens) and section D (visible fraction 0.6, 0.3 and 0.1, 3 patterns each) under 4 replacement rules at the attention input: the current zero (the treatment), the mean token of that position and block over 256 clean validation images, a resample of that position and block from a fixed bank of 64 clean images and a key mask that removes the token from the softmax. Agreement across the 4 is the control.
- Prediction: information removal predicts all 4 rules agree within 0.05 on every row. E6 predicts the zero rule departs from the other 3, most likely in the clean rows of D and on WaNet (where substitution already reads 0.376 below token masking in `why_psbd_works`).
- Refuted if: E6 is refuted when all 4 rules agree within 0.05 on every row.
- Models, compute, owner: the development set, GPU, about 60 U1 conditions (1 min per model, 10 min) plus seconds to cache the bank, `experiments/mask_replacement/`.
- Status: planned, main night slot 7.

The figure is a paired dot plot, 1 row per condition, 4 dots per row. N10 is the stochastic version of the resample rule as a PSBD operator: replace a random share of tokens at the attention input with the same-position, same-block tokens of the bank, 1 bank image per pass. It predicts patch, blend and frequency as PSBD-TM, quantization as PSBD-TM since bank tokens are not quantized, and warp possibly weaker since a foreign token breaks local coherence the way `token_substitute` does. Its ladder (12 min) runs only if X5 finds a departure. The closest prior work is resample ablation (Chan et al.), our `token_substitute` and STRIP's image-level superimposition.

**X13. True key mask (N1) and sink-only injection (N14).**

- Hypothesis: E6 from the operator side, theory candidate (1).
- Comparison and control: 3 operators at the attention input, each on a 6-rate ladder read at the matched 0.6 and adaptive 0.8 rules, against PSBD-TM. The key mask gives a random share $p$ of patch tokens negative infinity logits as keys for every query (queries, values and residual entries untouched). The sink injection appends $r = p \cdot 196$ extra key and value vectors equal to $W_k\beta + b_k$ and $W_v\beta + b_v$ to every block's attention without masking anything. `token_substitute` is already measured in `why_psbd_works`.
- Prediction: if E1 is the whole story, the key mask reads within 0.02 of PSBD-TM on patch (0.90 to 0.97 for 4 tokens, 0.6 to 0.75 for 1 token by the theory triage) and blend (0.75 to 0.9), and sink injection reads near 0.5 with small shifts. If the sink carries clean fragility, the key mask needs a much higher rate for the same clean damage: the median clean critical rate rises from 0.29 to 0.41 up to at least 0.6, and the largest-class share of shifted predictions falls below 0.3.
- Refuted if: E6 is refuted when the key mask reads within 0.02 of PSBD-TM at matched shift and sink injection stays below 0.6 AUROC.
- Models, compute, owner: the development set, GPU, 12 U3 readings per model for the 2 new operators (24 min), `experiments/mask_replacement/`.
- Status: planned, main night slots 8 and 9. Needs the same `attn_mask` as X8.

2 measured facts suggest the sink carries a large part of clean damage: the keep floor at p 0.9 equals $1/K$ on all 4 datasets, and shifted clean predictions land on the single largest class at 0.54 to 0.87 and on the runner-up only 10% to 35% of the time. At $p \approx 0.85$ to 0.9 the patch OR still holds for 4 tokens ($p^{16}$ = 0.07 to 0.19) but not for 1 token ($p^4$ = 0.52 to 0.66). N1 combines as a replacement for PSBD-TM if it matches and as a union partner if it disagrees on WaNet. The closest prior work is DropKey (Li et al.), a training regularizer, and token dropping in Salman et al. N14's closest prior work is registers (Darcet et al.) and test-time registers (Jiang et al.), which add tokens to absorb attention on purpose. The figure is AUROC per attack for the 4 operators.

## Global triggers

Blend, LF and BPP survive every token-mask placement as well as patches do, and section D shows why: they stay legible from 30% of tokens. What is not known is whether that legibility is per token (every token carries a full copy, E2) or extent-dependent (a token sees only a piece and several are needed, E12). The family table shows what each account predicts per row, and the rows are carried by the canonical items named in the first column. The proposed owner of X17, X18 and X23 is `experiments/global_trigger_legibility/`.

| row | carried by | manipulation | E2, per-token copy | E12, extent-dependent | E3, direction |
|---|---|---|---|---|---|
| G1 | X17 | visible fraction 0.3, 0.1 and 0.05 in 4 geometries: scatter, 1 window, horizontal stripes, vertical stripes | no dependence on geometry for Blend and BPP | LF lost from 1 window | no separate prediction |
| G2 | X23 | the trigger stamped on only a fraction of the image area (0.75 to 0.1), random tokens against 1 window, on otherwise clean images | attack success stays high down to small areas, geometry irrelevant | windows fail before scatter | score follows the projection on the backdoor direction |
| G3 | X18 | test-time strength ladders: Blend $\alpha$, LF strength, BPP bit depth | attack success high until the per-token copy fades | same | PSBD-TM score rises smoothly to the clean level |
| G4 | X11 | the trigger alone on a gray image, and the blend pattern alone | target kept (the pattern is the evidence) | target kept | the gray image carries the full direction |
| G5 | X15 | low-pass and high-pass filtering of the triggered image (Zeng et al.) | Blend's random pattern and BPP's quantization die under low-pass, LF survives | same | PSU tracks what survives |
| G6 | X4 | residual dropout by band, 1 to 4, 5 to 8 and 9 to 12 | global evidence stored in every token, fragile only late | assembled early | no separate prediction |
| G7 | X12 | denoising the triggered stream into the clean run for a random fraction of positions at each block | a small random fraction suffices at every block | sufficiency needs spatially spread positions | no separate prediction |

The figure for the global family is G1: excess retention against the visible fraction, 1 line per geometry, 1 panel per attack, with the clean retention curve in gray. SIG, the only frequency trigger, is absent from the ViT panel at the 2-point bar and quarantined for its amplitude drift, so L26's prediction that row masks keep SIG at 10% visible (0.3 or more against 0.010 for random tokens) is not scheduled.

**X17. Contiguous against scattered visible subsets (G1, W6).**

- Hypothesis: E2 against E12, and E8 for WaNet.
- Comparison and control: section D extended with visible fractions 0.3, 0.1 and 0.05 in 4 geometries (random scatter as the control, 1 square window at a random place, horizontal stripes, vertical stripes), 3 draws each.
- Prediction: per-token legibility (Blend's per-pixel random pattern, BPP's quantization) predicts no dependence on geometry. Extent-dependent triggers (LF's low frequencies) predict that 1 window loses them. WaNet under E8 predicts that no geometry keeps it at 0.1.
- Refuted if: E2 is refuted for Blend or BPP when 1 window keeps less than half the excess retention that scatter keeps at the same fraction.
- Models, compute, owner: the development set, GPU, 36 U1 conditions (40 s per model, 7 min), `experiments/global_trigger_legibility/`.
- Status: planned, main night slot 20.

The figure plots excess retention against the visible fraction, 1 line per geometry, 1 panel per attack. It also resolves a contradiction between 2 records: section D reads WaNet at 0.183 excess retention at 30% visible over 3 models, while `why_psbd_works` reads 0.706 on `vit_tiny_wanet_0_1`.

**X18. Test-time trigger strength (G3, W7).**

- Hypothesis: E3 against E4 for global triggers, E8 for WaNet.
- Comparison and control: Blend's $\alpha$ over 0.05, 0.1, 0.15, 0.2 and 0.3, LF's strength over 0.25 to 2 times the trained value, BPP's bit depth over 5, 4, 3 and 2 and WaNet's strength $s$ over 0.125 to 1 (plus the trained-strength 2 and 4 models `vit_cifar100_wanet_0_05_trig_s2` and `_s4`), with the trained strength as the control. Read attack success, the logit difference and PSBD-TM's score at the model's adaptive rate.
- Prediction: E3 predicts that the score of successful triggered inputs rises smoothly to the clean level as the trigger weakens, tracking the projection on the backdoor direction. E4 predicts it tracks confidence instead. For WaNet, E8 predicts success only near the trained $s$, with both weaker and stronger fields failing, while every other account predicts a monotone rise with strength.
- Refuted if: E8 is refuted when WaNet's attack success rises monotonically past the trained strength.
- Models, compute, owner: the 6 global and WaNet development models, GPU, 5 U3 readings per model (6 min), `experiments/global_trigger_legibility/`.
- Status: planned, main night slot 21.

W7 carries the sharpest single prediction of E8, since a detector trained to reject the trigger field plus small noise should also reject a scaled trigger field. The figure plots attack success and the median score against strength with the clean score band shaded.

**X23. The trigger on part of the image (G2, W10).**

- Hypothesis: E2 against E12, and E8 for WaNet.
- Comparison and control: the trigger (or the warp) stamped on only a fraction of the image area from 0.75 to 0.1, as random tokens or 1 window, on otherwise clean images, against the full trigger.
- Prediction: E2 predicts attack success stays high down to small areas with geometry irrelevant. E12 predicts windows fail before scatter. For WaNet, E8 predicts success collapses quickly with area, E10 that it falls with area and E2 that it stays high.
- Refuted if: E2 is refuted when Blend's success at 0.3 of the area falls below 0.5.
- Models, compute, owner: the 6 global and WaNet development models, GPU, about 7 min, `experiments/global_trigger_legibility/`.
- Status: planned, main night slot 22.

## WaNet, where token masking fails

WaNet is the most important failure because it holds the only inverted cell of the panel (`vit_cifar10_wanet_0_1`, PSBD-TM 0.459 against PSBD-RD 0.947), and because the literature gives a concrete mechanism to test. Over the 3 WaNet panel models PSBD-TM reads 0.779 and PSBD-RD 0.955 (`experiments/literature_checks/detectors_by_attack.py`, 54 models). The WaNet trigger moves where pixels are sampled from by at most about half a native pixel, and noise mode trains the model to return the true class on images warped by the trigger field plus a random per-pixel field of the same scale. Our own records add 4 facts: the most displaced tokens carry the evidence at block 4 (patch recovery 0.880 against 0.359 for random tokens), the class token holds it by block 12, the substitution operator reads 0.592 where token masking reads 0.968 on the WaNet model of `why_psbd_works` (its README of 2026-09-29) and WaNet does not survive crop and flip augmentation in our recipe. X1 added a fifth: on the failing cell the triggered shifts do not return to the true class. The proposed owner of X6, X15 and X16 is `experiments/wanet_specialist/`.

| row | carried by | manipulation | E8, noise-mode rejection | E10, computed early | E2, redundant like Blend |
|---|---|---|---|---|---|
| W1 | X1 | destination of triggered shifts under PSBD-TM | mostly the true class on noise-mode models (failed on the failing cell) | no preferred destination | no shifts to speak of |
| W2 | X2 | the 2 unswept replicates of the failing cell | fail if the noise-mode recipe is the cause | fail if the recipe is the cause | pass |
| W3 | X6 | random sub-pixel jitter at native resolution, noise mode against none | triggered fragile with noise mode, stable without | fragile | stable |
| W4 | X15 | the transformation battery | translation, flip and jitter break the trigger, smooth random warps may not | blur breaks it | nothing breaks it |
| W5 | X16 | position embedding and local content permutations | embedding permutation breaks it if the check is relational | local permutation breaks it | nothing breaks it |
| W6 | X17 | contiguous against scattered visibility | no geometry at 0.1 keeps it | windows keep more than scatter | every geometry keeps it |
| W7 | X18 | test-time warp strength $s$ from 0.125 to 1, and the trained strength 2 and 4 models | success only near the trained $s$ | success grows with $s$ | success grows with $s$ |
| W8 | `why_psbd_works` | per-token linear probe for trigger-warped against noise-warped images at blocks 2 to 8, with a control-label probe (Hewitt and Liang) | per-token selectivity near 0, class token high | per-token selectivity rises in blocks 2 to 6 | per-token selectivity high everywhere |
| W9 | X4 | residual dropout confined to blocks 1 to 4 with its sign flipped | not decisive | triggered breaks more than clean | no effect |
| W10 | X23 | the warp applied to only 1 window of the image, from 0.75 to 0.1 of the area | success collapses quickly with area | success falls with area | success stays high |

**Noise mode across our WaNet models.** `attacks/wanet.py` implements noise mode as the reference code does (offsets drawn uniformly in $[-1/h, 1/h]$ in normalized coordinates, about 0.48 native pixels at most, the same scale as the trigger's own displacement of about 0.24 pixels on average). Commit 52e3587 restored it on 2026-09-08. Every ViT WaNet model on the panel and the CIFAR-10 and GTSRB seed replicates carry `cover_rate` 0.1 or 0.2 in `args.json`. The ViT CIFAR-100 and Tiny seed replicates (trained 2026-09-07) and every seed-0 Swin WaNet model carry `cover_rate` 0.0, and the Swin ones have `git_commit` null, so they are inferred to predate the restoration. That inference rests on dates, and a Swin run log would settle it. The Swin seed replicates `swin_{cifar10,cifar100,gtsrb}_wanet_0_1_seed_{1,2}` carry 0.2.

**Noise mode does not explain the failure by itself.** Read on 2026-09-29 in the fractional statistic at the adaptive rule, PSBD-TM reads 0.930 to 0.950 on the other 4 noise-mode ViT WaNet models that clear the ASR bar and 0.846 to 0.967 on the 6 ViT replicates without noise mode that carry caches. On Swin it reads 0.965 to 0.982 on the 6 noise-mode seeds against 0.885 to 0.992 on the seed-0 models at 5% and 10% without it. The Swin advantage in `experiments/swin_mechanism/` therefore survives the noise-mode confound its 6 matched pairs carry, though that README should say so.

**WaNet does not survive augmentation in our recipe.** The 2 WaNet models trained with standard crop and flip augmentation reach attack success 0.024 (`vit_cifar100_wanet_0_05_aug`) and 0.475 (`vit_tiny_wanet_0_05_aug`) against 0.649 and 0.941 without it (`metrics.json`). A 1 pixel crop offset or a flip makes the trigger unlearnable here, so the learned feature is tied to exact pixel alignment. The WaNet paper trained with random crop, rotation and flip and still implanted, so this is a property of our 15-epoch fine-tuning recipe.

**X2. The replicates of the failing cell (W2).**

- Hypothesis: the 0.459 cell is a checkpoint accident, against a recipe that defeats PSBD-TM.
- Comparison and control: `before_attention_norm_token_mask` and `post_residual` swept on `vit_cifar10_wanet_0_1_seed_1` and `_seed_2` (noise mode, attack success 0.885 and 0.889 in `metrics.json`), with the seed-0 cell as the reference.
- Prediction: if both read at least 0.9, the 0.459 is a checkpoint property and the paper should say so. If both read at most 0.6, CIFAR-10 WaNet at 10% with noise mode is a recipe that defeats PSBD-TM and the paper has a real failure class.
- Refuted if: the accident reading is refuted when either replicate reads 0.6 or below.
- Models, compute, owner: 2 replicates, GPU, reduced splits 5 min and full splits about 30 min, `experiments/why_token_masking_works/`.
- Status: queued for the night of 2026-09-29.

The figure is 1 dot per seed and placement with the seed-0 cell marked. The full-split sweep is the record, and the reduced splits only decide whether the night's later WaNet slots change order.

**X6. Random sub-pixel warps and noise mode (W3, N15).**

- Hypothesis: E8 as a detector sign, the key WaNet hypothesis set out under "A WaNet-specific perturbation detector".
- Comparison and control: the native-resolution image resampled at the identity grid plus $a\,\xi/h$ with $\xi$ uniform in $[-1,1]$, before the model's own resize, for $a$ in 0.25, 0.5, 1 and 2, with k = 3. Noise-mode models against models without noise mode, and clean inputs as the control.
- Prediction: E8 predicts triggered retention at most 0.2 at $a = 1$ on noise-mode models, with at least 0.8 of the flips going to the true class and clean retention at least 0.9. It predicts the opposite on models without noise mode (triggered retention at least 0.9 and more clean images sent to the target). Generic stability (E1, E3) predicts triggered retention at or above clean retention at every amplitude on every model.
- Refuted if: the specialist reading of E8 is refuted when noise-mode models keep more than 0.5 of triggered predictions at $a = 1$, or when models without noise mode are as fragile as those with it.
- Models, compute, owner: the development set (8 min), then the WaNet set: the 5 ASR-clearing ViT models with noise mode, `vit_gtsrb_wanet_0_05` and `vit_cifar100_wanet_0_1` labeled as below the ASR bar, the 6 ViT replicates without noise mode, the 8 seed-0 Swin models without it and the 6 Swin replicates with it (about 22 min), GPU, `experiments/wanet_specialist/`.
- Status: planned, main night slots 10 and 11.

Read triggered retention, the share of triggered flips to the true class, clean retention and clean flips to the target. The figure has 2 panels, retention against $a$ for triggered and clean, noise mode on the left and without on the right, 1 line per model colored by architecture. X1's result lowers the chance that jitter explains the failing cell, and leaves intact the prediction that jitter separates triggered WaNet inputs on noise-mode models, which is what a specialist needs.

**X15. The WaNet transformation battery (W4, G5).**

- Hypothesis: E8 (exact pixel alignment), Zeng et al.'s frequency account and Lite-BD's resampling result.
- Comparison and control: Gaussian blur at $\sigma$ 0.5 and 1 native pixel, bilinear down-up scaling by 0.5, translation by 1 and 2 pixels with reflection padding, a horizontal flip, a fresh smooth random warp at strengths 0.25 and 0.5, JPEG at quality 75 and a high-pass filter (G5), each against the untransformed input. All but the random warp are deterministic.
- Prediction: the augmentation fact and E8 predict that translation and flip break triggered WaNet while clean CIFAR retention stays at 0.85 or above. Lite-BD and ZIP predict that blur and down-up scaling break WaNet and BPP but not BadNets. The random smooth warp separates an exact-field reader (triggered WaNet breaks) from a "warped means target" reader (clean images go to the target).
- Refuted if: translation by 1 pixel keeps more than 0.8 of triggered WaNet predictions on noise-mode models.
- Models, compute, owner: the development set and the WaNet set, GPU, about 40 s per model (27 min), `experiments/wanet_specialist/`.
- Status: planned, main night slot 13.

The figure is a retention matrix, transform against attack, for triggered and clean. Candidates 3, 6, 7 and 9 of the specialist table are read off this run.

**X16. Position-content coherence and position-embedding jitter (W5, N6, theory candidate (6)).**

- Hypothesis: E8 in its relational form, E9 for BadNets.
- Comparison and control: 4 deterministic conditions (position embeddings permuted within every 2 by 2 block of tokens, position embeddings with Gaussian jitter at 0.25 of their norm, content permuted within 2 by 2 blocks with position embeddings kept and a whole-grid shuffle) against the unperturbed input. Theory candidate (6) is the stochastic version as a PSBD operator: jitter only $\pi_t$ in $x_t = e_t + \pi_t$, once at the input.
- Prediction: if WaNet is read as coherence between what a token shows and where it sits, permuting position embeddings breaks triggered WaNet more than clean inputs. If WaNet is read from per-token texture alone, triggered WaNet survives all 4 like a global trigger. The theory triage predicts position-embedding jitter as an operator at patch 0.85 to 0.99, blend 0.9 to 0.98, frequency 0.5 to 0.8 and warp 0.85 to 0.97, the strongest warp complement, with a risk that strong jitter cannot reach 0.8 clean damage.
- Refuted if: the relational form is refuted when triggered WaNet retention under the 2 by 2 embedding permutation is within 0.05 of clean retention. The operator form fails if no jitter scale reaches the 0.8 shift target.
- Models, compute, owner: the development set, GPU, 8 U2 conditions (40 s per model, 7 min) plus a 6-rate ladder of the operator (12 min) if its reachability check passes, `experiments/wanet_specialist/`.
- Status: planned, main night slot 12.

A warp is a relation between neighboring pixels, but it acts below the token scale here: the largest displacement is about 0.5 native pixel, about 3.5 pixels at 224, inside 1 token of 16. Permuting tokens or their position embeddings therefore cannot touch the warp's own texture, and can only break a check that ties what a token shows to where it sits. Doan et al.'s PatchShuffle lowers clean accuracy while Blend and WaNet attack success stay, which is the precedent, together with Naseer et al.'s shuffle robustness. The figure is retention per condition for triggered and clean.

## A WaNet-specific perturbation detector

The goal is a specialist probe that detects WaNet well without raising the false positive rate on clean inputs, so that it can join PSBD-TM in a union. The user set 3 criteria: standalone AUROC and TPR at 10% and 20% FPR on WaNet, a well calibrated clean-validation distribution so the union adds WaNet detections at a fixed budget without adding clean flags, and no loss on BadNets, TaCT, Blend, LF or BPP when combined. The candidates the user named for it are sub-pixel jitter (X6), rotation-subspace dropout (N5), an input warp (the smooth random warp of X15) and position-embedding jitter (X16).

**The key hypothesis and its answer.** Noise mode trains on $W(x, M + \delta)$ labeled with the true class $y$, where $M$ is the trigger field and $\delta$ is a per-pixel field drawn uniformly in $[-1/h, 1/h]$ in normalized coordinates (`attacks/wanet.py`, following the reference code and the paper's equation 6). A triggered test image $W(x, M)$ resampled once more at jittered positions is, to first order in the small displacements, $W(x, M + \delta)$, which is exactly a noise-mode training image with its true label. The model was trained to answer $y$ on it. The composition is not exact, because a second bilinear resampling smooths the image once more, and $M$ evaluated at a jittered point differs from $M$ at the grid point by a term of order $\delta \cdot \nabla M$, which is small because $M$ is smooth over about 8 native pixels. The trigger displaces pixels by about 0.24 native pixels on average and the noise by up to about 0.48, so the perturbation needed is the same size as the trigger itself.

3 qualifications follow from the same argument. The prediction is specific to i.i.d. per-pixel jitter at the noise amplitude, so a smooth random warp is not in the noise-mode distribution and the model's response to it is not predicted, and a blur is predicted to matter only through the interpolation texture it removes. The prediction reverses for models trained without noise mode, which the WaNet authors report learn "pixel-wise artifacts" of the warping, so on those models jitter should leave triggered inputs on the target and may push clean inputs toward it. And the argument would explain PSBD-TM's inversion on the failing cell only if the masked triggered input also read as "not the exact field", which X1 tested and did not find (0.158 of shifts to the true class, read 2026-09-29).

**Candidate operators.** Each candidate is scored on the user's 3 criteria. "Sign" says whether the triggered input shifts more (fragile) or less (stable) than a clean one. A fragile sign is the opposite of PSBD's and must be fixed before any triggered data is read.

| rank | operator | mechanism | sign on WaNet | BadNets and TaCT | Blend | LF | BPP | clean calibration | test and cost |
|---|---|---|---|---|---|---|---|---|---|
| 1 | late residual dropout, `pre_residual_blocks_9_12` or `_5_8` (existing) | the class token's copy of the warp evidence is 1 direction in a few late blocks, and dropout on the stream damages it | stable, PSBD's sign, 0.943 to 0.987 on the 4 ViT WaNet models where the adaptive rule reaches a rate | reads badly (PSBD-RD 0.309 on the BadNets development model), which a budget-weighted union absorbs | good | good | good | as PSBD-RD, known | X3, CPU |
| 2 | sub-pixel jitter at native resolution | noise mode: a jittered triggered image is a noise-mode training image with its true label | fragile on noise-mode models, stable without noise mode | the 1-pixel checkerboard loses about half its contrast at an average offset of 0.25 pixel, so partly fragile | the per-pixel random pattern is averaged down, partly fragile | unaffected, uninformative | quantization levels are resampled off grid, likely fragile | good if the amplitude is set on clean data | X6, 8 min |
| 3 | integer translation by 1 native pixel, deterministic | the learned feature is tied to exact pixel alignment (the `_aug` models never implanted) | fragile | the patch moves 7 pixels at 224 and stays in the same tokens, stable | stable | stable | a shift does not change quantization levels, stable | good, a 1-pixel shift barely moves clean CIFAR predictions (to be measured) | X15, under 1 min |
| 4 | residual dropout in blocks 1 to 4 with a flipped sign (existing cache) | E10: a computed trigger breaks while being assembled | fragile if E10 holds | stable, routed | stable | stable | unknown | as PSBD-RD bands | X4, CPU |
| 5 | TeCo's corruption deviation (existing port) | triggered inputs flip at very different severities across corruption types | TeCo's own sign, 0.871 on the 3 WaNet panel models | 0.976 on BadNets, 0.825 on TaCT | 0.907 | 0.477 | 0.562 | as ported | X3 extended to the detector records, CPU |
| 6 | Gaussian blur, $\sigma$ 0.5 native pixel | removes the interpolation texture and all high frequencies (Zeng et al., Lite-BD, ZIP) | fragile | the checkerboard at 0.29 of its contrast, likely fragile | fragile | stable | fragile | worse, blur costs clean accuracy at 32 pixels | X15 |
| 7 | smooth random warp, strength 0.25 | a smooth field is outside the noise-mode distribution, so it tests an exact-field reader | unknown, fragile only for an exact-field reader | stable | stable | stable | stable | good | X15 |
| 8 | position-embedding permutation within 2 by 2 blocks | breaks position-content coherence if WaNet is checked relationally | fragile if relational | fragile if E9 holds | stable | stable | stable | unknown, Naseer et al. suggest robust | X16 |
| 9 | bilinear down-up scaling by 0.5 | the strongest disruption in Lite-BD | fragile | stable | fragile | stable | fragile | poor, a 16-pixel CIFAR image loses class detail | X15 |
| 10 | random-rotation subspace dropout (N5) | attenuates each input along its own decision direction, a cosine margin | stable, about 0.95 by the theory triage | 0.8 to 0.95, at most 0.5 if axis-aligned | 0.93 to 0.98 | about 0.9 | as blend | smooth, like PSBD-RD | N5, 12 min |
| 11 | position-embedding jitter (theory candidate (6)) | keeps every token, the property WaNet needs | stable, 0.85 to 0.97 | 0.85 to 0.99 | 0.9 to 0.98 | 0.5 to 0.8 on SIG | as blend | risk of a shift ceiling | X16, 12 min |

Candidate 1 is the existing baseline and already reads WaNet well. It fails criterion 3 under mean fusion, which is where the weighted rule below matters. Candidate 2 is the only one whose sign and amplitude follow from the attack's own training procedure, so it is built first. Candidate 3 is the cheapest deterministic alternative. Candidates 2, 3 and 6 are high-frequency specialists as much as WaNet specialists, which is a feature for a union: they flag triggered BadNets, Blend and BPP inputs as extra fragile, which can only add detections once the sign is fixed. Candidates 10 and 11 carry PSBD's own sign, so they enter as ordinary union members.

**The specialist score and its amplitude.** The specialist reuses PSBD's fractional statistic with the input operator in place of the internal one. It flags high values, the opposite of PSBD.

$$\psi(x) = 1 - \frac{1}{k}\sum_{j=1}^{k}\frac{P_c\big(T(x;\,a,\xi_j)\big)}{P_c(x)}, \qquad T(x;\,a,\xi) = \mathrm{grid\_sample}\Big(x,\ G_0 + \frac{a}{h}\,\xi\Big)$$

| symbol | meaning |
|---|---|
| $x$ | the input image at its native resolution, before the model's resize to 224 |
| $P_c(\cdot)$ | softmax probability of the unperturbed predicted class $c$ |
| $k$ | number of jittered passes, 3 as in the headline |
| $T(x; a, \xi)$ | the image resampled at jittered positions with bilinear interpolation, border padding, `align_corners=True` |
| $G_0$ | the identity sampling grid in normalized coordinates $[-1,1]^2$ |
| $\xi$ | a per-pixel random field, shape $(h, h, 2)$, uniform in $[-1,1]$ |
| $a$ | the amplitude, $a = 1$ is the noise-mode offset scale, about 0.48 native pixels at most |
| $h$ | native image width in pixels, 32 or 64 |
| $\psi(x)$ | the specialist score, high means fragile and flagged |

The amplitude is chosen on clean validation data only, as the largest amplitude on a fixed ladder at which clean predictions barely move. The ladder and the tolerance are fixed before any triggered image is scored.

$$a^{*} = \max\{\,a \in \mathcal{A} : \sigma_V(a) \le \sigma_{\max}\,\}, \qquad \mathcal{A} = \{0.25, 0.5, 1, 2\}, \quad \sigma_{\max} = 0.05$$

| symbol | meaning |
|---|---|
| $\sigma_V(a)$ | clean-validation shift ratio of the jitter at amplitude $a$, the share of passes whose label changes |
| $\sigma_{\max}$ | the largest tolerated clean shift, fixed at 0.05 before any triggered data is read |
| $\mathcal{A}$ | the amplitude ladder, fixed in advance |

This rule is the mirror of the adaptive rule. PSBD wants clean inputs to move (target 0.8), a fragile-sign specialist wants them still, so that any triggered input that moves stands out against a tight clean distribution. That is what makes criterion 2 achievable. It also keeps the defender from using knowledge of the attacker's noise amplitude, since $a = 1$ is chosen only if clean images tolerate it.

**Putting a fragile-sign specialist into the union.** `experiments/probe_union/` combines probes by min-rank: each probe's score becomes its percentile in that probe's clean-validation distribution and the union takes the minimum. The threshold is the target quantile of the minimum on clean validation, which calibrates the union's FPR on validation whatever the members are. The cost of adding a member is dilution, since the minimum of 2 roughly uniform percentiles is smaller than either, so the threshold tightens and PSBD-TM runs at a lower effective FPR. A member that is uninformative or inverted on an attack costs only that dilution, because its percentile for a triggered input is high and never becomes the minimum. Mean fusion is different, since an inverted member pulls triggered scores toward the clean ones, which is why the mean rule and min-rank disagree in `experiments/probe_fusion/`.

For a specialist the right rule is a budget-weighted minimum, which is min-rank with unequal budgets and reduces to a per-probe quantile OR at a split FPR. The shares say how much of the FPR budget each member may spend.

$$u(x) = \min\left(\frac{\hat F_{\mathrm{TM}}\big(\phi_{\mathrm{TM}}(x)\big)}{w_{\mathrm{TM}}},\ \frac{1 - \hat F_{S}\big(\psi(x)\big)}{w_{S}}\right), \qquad \text{flag } x \text{ when } u(x) \le \alpha$$

| symbol | meaning |
|---|---|
| $\phi_{\mathrm{TM}}(x)$ | PSBD-TM's fractional PSU, low means poisoned |
| $\psi(x)$ | the specialist score, high means poisoned |
| $\hat F_{\mathrm{TM}}, \hat F_{S}$ | empirical CDFs of the 2 scores on the 2000 clean validation images |
| $w_{\mathrm{TM}}, w_{S}$ | budget shares, fixed in advance, summing to 1 |
| $\alpha$ | the target FPR of the union |
| $u(x)$ | the union score, low means poisoned. Its AUROC is the union's AUROC |

Thresholding $u$ at $\alpha$ flags $x$ when PSBD-TM's percentile is below $w_{\mathrm{TM}}\alpha$ or the specialist's upper percentile is below $w_S\alpha$, so by the union bound the validation FPR is at most $w_{\mathrm{TM}}\alpha + w_S\alpha = \alpha$, with equality when the 2 flag disjoint clean images. At $\alpha = 0.10$ and shares 0.8 and 0.2, PSBD-TM runs at 0.08 and the specialist at 0.02. A specialist that catches WaNet at all catches it in its extreme tail, so 0.02 is enough. PSBD-TM's loss from 0.10 to 0.08 is readable from the caches (X3) before the specialist exists. Equal shares give the existing min-rank rule.

A 2-sided specialist is the fallback when the sign cannot be fixed in advance, for example on models that may or may not use noise mode. The specialist percentile becomes $2\min(\hat F_S, 1 - \hat F_S)$, which spends half of $w_S\alpha$ on each tail and halves the specialist's power in the tail that matters. Because the jitter's sign is predicted to flip with noise mode, a 2-sided specialist is honest here. X6 measures what it costs.

The rules that keep this from cheating are these. The sign is fixed from the noise-mode argument before any triggered image is scored, or the test is 2-sided. The budget shares, the amplitude ladder, $\sigma_{\max}$ and $k$ are fixed in advance and written into `configs/psbd_basis.json` before the held-out read. The amplitude and both CDFs come from clean validation only. The realized FPR on the clean test split is reported beside the nominal one, because the Q27 audit shows the realized FPR of a quantile threshold varies widely across models. No quantity is ever fitted on triggered data, including "which tail".

**Evaluating on few WaNet models honestly.** The panel holds 3 WaNet ViT models at the 2-point clean-accuracy bar (`vit_cifar10_wanet_0_1`, `vit_tiny_wanet_0_05`, `vit_tiny_wanet_0_1`). 2 more clear the attack-success bar and fail the accuracy bar (`vit_cifar10_wanet_0_05`, `vit_gtsrb_wanet_0_1`, both successful at the 5-point bar). 2 more fall below the attack-success bar (`vit_gtsrb_wanet_0_05` at 0.83, `vit_cifar100_wanet_0_1` at 0.793). On disk there are also 6 ViT replicates with noise mode (CIFAR-10 5% and 10%, GTSRB 10%, seeds 1 and 2, none swept yet), 6 ViT replicates without noise mode (CIFAR-100 10%, Tiny 5% and 10%, seeds 1 and 2), 8 seed-0 Swin models without noise mode and 6 Swin replicates with it, plus the strength variants `vit_cifar100_wanet_0_05_trig_s2` and `_s4` with noise mode.

The honest reading uses all of them and labels each by panel status, noise mode and architecture. It never averages across labels. It pre-registers 2 predictions. On noise-mode models the specialist reaches standalone AUROC of at least 0.9 on every model. On models without noise mode it does not, which is the falsification test of the mechanism, since a specialist that also works without noise mode works for another reason. With fewer than 8 models in a group, the report counts models that pass and fail the pre-registered bar instead of printing a bootstrap interval. Development uses the CIFAR-10 and GTSRB models only. The Tiny and CIFAR-100 models, their replicates and all Swin models are read once, after the design is frozen, which is why `vit_tiny_wanet_0_05` is swapped out of the development set.

## New operators

These are the candidate building blocks for a better probe. Each gets a 6-rate reduced ladder on the development set (12 min) unless stated otherwise, read at the adaptive 0.8 and matched 0.6 rules. The theory triage of `docs/novel-designs-theory.md` predicts that none beats PSBD-TM on patch triggers and that (5) and (6) complement it on warp, and it is the filter for the order below. The proposed owner is `experiments/novel_operators/`.

The coordinator reported on 2026-09-29 (historical, the 39 models with an adaptive late-band rate) that averaging PSBD-TM's fractional PSU with that of late-block residual dropout gained +0.013 [+0.004, +0.028] over PSBD-TM with no attack losing and lifted the failing WaNet cell from 0.459 to 0.745, and that the min-rank union gained +0.017 [+0.003, +0.043] and lifted the cell to 0.938 (`docs/runs/2026-09-29-gpu-queue.md`). That says combination works, so the value now lies in better building blocks.

**X14. Stratified token masking (N13).**

- Hypothesis: E1's OR over late reads, taken to its limit.
- Comparison and control: each block's mask drawn so that every patch token stays visible in at least 1 of every $B$ consecutive blocks while each token's marginal masking rate stays $p$, against PSBD-TM at the same marginal rate.
- Prediction: with $B = 4$ the 4 late reading blocks can never all hide a trigger token, so the event that breaks a triggered patch prediction under PSBD-TM (all $m$ trigger tokens masked in all 4 late blocks, probability $p^{4m}$, observed 0.059 on Tiny) becomes impossible, while every block still hides a share $p$ of clean evidence. Patch survival rises from 0.966 (4 tokens) and 0.930 (1 token) to near 1 with clean fragility unchanged, so TPR at 1% FPR rises most on Tiny BadNets and TaCT, with blend, quantization and warp unchanged.
- Refuted if: clean shift at equal $p$ falls by more than 0.05, or patch TPR at 1% FPR does not rise.
- Models, compute, owner: the development set, GPU, 12 min, `experiments/novel_operators/`.
- Status: planned, main night slot 17.

For $p \le (B-1)/B$ such masks exist, for example by assigning each token a random offset and masking it in a fixed pattern of $\lceil pB \rceil$ consecutive blocks per window of $B$, with a random draw deciding the extra block when $pB$ is not an integer. It combines as a drop-in replacement for PSBD-TM. The trigger-conditional TaCT models keep only 0.205 of triggered predictions under PSBD-TM with their trigger visible in every late block (section B), so stratification cannot help them, which is a second prediction. The closest prior work is PatchCleanser's covering mask sets, which guarantee coverage of a patch, where this guarantees a patch survives.

**N5. Random-rotation subspace dropout (theory candidate (5)).**

- Hypothesis: 2 predictions conflict. The earlier plan argued that the OR and AND asymmetry of token masking has no counterpart among random directions, so only the margin difference (E4, refuted as the carrier) remains and N5 reads at or below the MLP channel mask (0.833 on the 54 models). The theory triage argues that for a random $k$-dimensional subspace $S$, $\mathbb{E}P_S = (k/d)I$, so $u^\top(I - P_S)x$ has mean $(1 - k/d)a$ with relative fluctuation about $\sqrt{k}\,\|x\|/(d\,a)$, which ranks inputs by a cosine margin. Triggered representations are dominated by $u$.
- Comparison and control: rotation against the axis-aligned proxy `post_residual_channel_mask` at matched shift.
- Prediction: the theory triage predicts patch 0.8 to 0.95 (at most 0.5 axis-aligned), blend 0.93 to 0.98, frequency about 0.9 and warp about 0.95. The earlier plan predicts at or below 0.833 overall. The cached axis-aligned proxy reads Blend 0.99, BadNets 0.34 and LF 0.95 on the 3 development models that carry it.
- Refuted if: the theory triage is refuted when rotation reads within 0.05 of axis-aligned on the 2 patch development models, and the earlier plan when rotation beats 0.9 on patch.
- Models, compute, owner: the development set, GPU, 12 min, `experiments/novel_operators/`.
- Status: planned, main night slot 18.

**S1. Active-neuron dropout in the MLP hidden layer.**

- Hypothesis: ShortcutProbe's "Active Neuron Dropout" (Prasad et al.), a neuron-level account (L7, P3) already refuted in the residual and MLP bases.
- Comparison and control: dropout of positive activations only after GELU in the MLP hidden layer, the only ViT site where "positive" is defined, against the plain MLP channel mask at matched shift.
- Prediction: at or below plain MLP channel masking, which reads 0.833 on the 54 models (TaCT 0.522, BadNets 0.775) against PSBD-TM's 0.963.
- Refuted if: the prediction is refuted when S1 beats the plain channel mask by more than 0.02.
- Models, compute, owner: the development set, GPU, about 4 min per model (1 sweep, 40 min), `experiments/novel_operators/`.
- Status: planned, main night slot 19.

**S2. Importance-ranked token removal and its largest sudden drop.**

- Hypothesis: E1, a patch trigger lives in a few tokens.
- Comparison and control: tokens ranked by gradient times activation at the block 9 input, the top 1 to 16 masked deterministically at the attention input, scored by the largest single confidence drop, against the same number of random tokens.
- Prediction: 1 sudden drop when the trigger's tokens go on BadNets and TaCT, gradual decay on clean and global-trigger inputs. Strong on BadNets and TaCT (where PSBD-TM already reads 0.992 and 0.962) and weak on Blend, BPP and WaNet. Its value is as a union partner that raises TaCT's low TPR, not as a replacement.
- Refuted if: the sudden-drop AUROC on the 3 patch models is below 0.8.
- Models, compute, owner: 3 patch models plus `vit_cifar10_blend_0_1` and `vit_cifar10_wanet_0_1`, GPU, 1 backward and up to 16 masked forwards per input (about 20 min per model, 100 min), `experiments/novel_operators/`.
- Status: planned, fallback night slot 5.

S2 is ShortcutProbe's second score moved from channels to tokens, where a patch trigger's few tokens make a single sudden drop plausible. If S2 is kept, it enters through the budget-weighted min-rank union and never through ShortcutProbe's equal-weight sum, since the sum is the mixture the union beats whenever 1 member is weak or inverted.

**X22. Attention temperature and attention-rank masking (N3, N4, theory candidates (3) and (4)).**

- Hypothesis: E1 through the class token's attention (L21, P12).
- Comparison and control: each head's attention logits multiplied by a factor $\tau$, sharpening ($\tau = 2$) and flattening ($\tau = 0.5$) separately, in blocks 9 to 12 and in all blocks. This follows the convention of `docs/novel-designs-theory.md`, where $\tau$ multiplies the logits. The source plan's $\tau$ was its inverse. Separately, the $r$ tokens with the largest weight times value norm (Kobayashi et al.) hidden from the class token in each late block. The unperturbed model is the control and AUROC is read in both sign conventions.
- Prediction: the class token puts 0.550 of its late attention on the 4 BadNets trigger tokens (`\RoutingBadnetClsLate`), which gives a trigger logit gap $\Delta \approx 4.1$ through the share $m e^{\tau\Delta}/(m e^{\tau\Delta} + N)$ with $m = 4$ and $N \approx 192$. The share becomes 0.14 at $\tau = 0.5$, 0.055 at $\tau = 0.25$ and 0.99 at $\tau = 2$. Sharpening protects a patch trigger and strips clean evidence of its diffuse support (PSBD's sign), flattening breaks the trigger (the fragile sign), so a random per-head factor cancels itself and the sign on patch follows $\mathbb{E}\log\tau$. The theory triage reads blend 0.85 to 0.95, frequency 0.8 to 0.9 and warp 0.5 to 0.75 for the temperature operator. Attention-rank masking flips the sign on patch (inverted AUROC 0.75 to 0.9, and L21 predicts at least 0.95 inverted for the top 4 tokens) and reads 0.5 to 0.65 standard on global triggers.
- Refuted if: attention-rank masking on BadNets does not invert (inverted AUROC below 0.7).
- Models, compute, owner: the development set, GPU, deterministic, 12 U3 readings per model (24 min), `experiments/novel_operators/`.
- Status: planned, fallback night slot 4. The triage predicts both fail as one-sided detectors, so they serve as a category diagnostic and as a fragile-sign patch specialist for low-TPR TaCT such as `vit_cifar10_tact_0_01`.

**N8. Random block skipping (theory candidate (7)).**

- Hypothesis: E10, a computed trigger breaks when a block that assembles it is skipped.
- Comparison and control: a whole block skipped per pass with probability $p$ with both residual writes removed, plus a band-restricted variant that skips only in blocks 1 to 6. PSBD-TM is the control.
- Prediction: the 2 sources differ. The earlier plan predicts warp fragile when skipping hits blocks 1 to 6, patch and blend stable, clean moderately fragile. The theory triage predicts PSBD's sign everywhere with weak separation (patch 0.6 to 0.9, blend 0.7 to 0.95, warp 0.8 to 0.95), since the trigger's late read is an OR over reading blocks while its high-norm token is manufactured in specific blocks. The cached `droppath` proxies read 0.50 to 0.92 on 3 development models.
- Refuted if: E10 is refuted when early-band skipping breaks triggered WaNet no more often than triggered BadNets.
- Models, compute, owner: `pre_residual_droppath` is cached on 3 development models for a CPU first reading, then the development set, GPU, 12 min, `experiments/novel_operators/`.
- Status: planned, fallback night slot 7.

**X24. Token mask and late residual dropout in the same pass.**

- Hypothesis: E1 and E10 together (optimal-probe rule 2): a read perturbation and a storage perturbation cover patch and warp triggers at once.
- Comparison and control: 1 operator applying PSBD-TM and `pre_residual_blocks_9_12` jointly in every pass, against the averaged score of the 2 (X3's mean rule) and their min-rank union.
- Prediction: patch as PSBD-TM or slightly lower, blend and warp better, quantization similar. The copula model gives the averaged score and the union but not the joint operator, so the joint operator is predicted within 0.01 of the averaged score if the 2 perturbations act independently.
- Refuted if: the joint operator trails the averaged score by more than 0.01, which would say the 2 perturbations interact.
- Models, compute, owner: the development set, GPU, 12 min, `experiments/novel_operators/`.
- Status: planned, fallback night slot 6.

**X25. Depth-dependent masking schedules.**

- Hypothesis: optimal-probe rule 1, hiding reads independently per block.
- Comparison and control: a front-loaded schedule (early rate above late) and a back-loaded one, against PSBD-TM's uniform rate at matched shift.
- Prediction: front-loaded spares routed triggers while early hiding breaks clean aggregation, so patch better, blend similar and warp worse (early hiding disturbs assembly). Back-loaded hits the reads, so patch worse and warp better.
- Refuted if: the front-loaded schedule lowers patch AUROC.
- Models, compute, owner: the development set, GPU, 12 min per schedule (24 min), `experiments/novel_operators/`.
- Status: planned, fallback night slot 6.

**X26. Remaining operator candidates.** Per-head masking (`attention_heads_head_mask`, cached on 3 development models at 0.93, 0.55 and 0.88) should be weaker than token masking, since 12 heads give a coarser OR. Attention dropout on post-softmax weights per query and key edge is a per-query key mask with renormalization and should read as N1 for the class-token read. ToMe merging as a perturbation (Bolya et al.) merges redundant tokens first, so a dissimilar trigger stays and clean redundancy goes with little loss, and weak separation is expected. MC dropout on the model's own dropouts (`activate_model_dropout`) conflates regularizer and probe and is not proposed. DynamicViT needs a trained token predictor. None is scheduled.

**N9. Random shifted-window offsets for Swin.** Swin's windows decide which tokens exchange information in each block. A random cyclic offset per pass changes the partition while the relative position bias stays tied to the trained layout. A patch trigger inside 1 window at stage 1 is sometimes split, but its evidence is merged in later stages anyway, so it should be stable. A warp read within windows could break. Clean inputs should be mildly perturbed, so separation is uncertain and likely weak. It is Swin only and costs about 15 min on 10 Swin twins of the development models. It is not scheduled, and I found no prior work.

**N7. Pixel shuffling inside each patch, dropped.** A 16 by 16 token of an upsampled CIFAR image covers about 2.3 native pixels, so shuffling its 256 pixels turns a smooth patch into high-frequency noise that is off distribution for clean images too, and the clean distribution would be badly calibrated. The principled version of "destroy WaNet's fine interpolation artifacts" works at native resolution and at the size of the noise-mode perturbation, which is X6.

## Swin ports and missing controls

These 2 items carry the Swin halves of L15, L20 and L23. They wait for protocol step 5, so neither is scheduled on the 2 nights.

**X30. Sections A and D ported to Swin (L20, L23).**

- Hypothesis: on Swin a patch trigger's evidence is spread into many merged tokens before the mean pool, so Swin's triggered predictions are less protected by routing and more by redundancy.
- Comparison and control: section A's deterministic trigger-token masks with `trigger_tokens` mapped onto the Swin stage-1 grid, and section D's visible subsets, against random tokens.
- Prediction: trigger tokens masked in every block leave at least 0.5 of triggered predictions (ViT 0.002), and WaNet keeps at least 0.5 of its excess ASR at 30% visible tokens (ViT 0.183).
- Refuted if: Swin's all-block trigger mask leaves below 0.1.
- Models, compute, owner: the Swin twins of the development set, GPU, proposed `experiments/patch_trigger_mechanism/`.
- Status: not scheduled.

**X31. TeCo on the Swin panel (L15).**

- Hypothesis: TeCo's corruption inconsistency is a property of windowed transformers, as its own Swin-T-Base result (0.985) suggests.
- Comparison and control: the ported `teco` detector through `cli.baselines` on the Swin cells, against PSBD-TM on the same cells.
- Prediction: mean AUROC at least 0.95.
- Refuted if: below 0.9.
- Models, compute, owner: the Swin panel, GPU, `cli.baselines`.
- Status: not scheduled. Without this run a claim that PSBD-TM transfers to Swin better than its competitors has no competitor on Swin to compare against.

## Needs training

These are listed apart because the login node cannot train and each needs a PBS job. Each row names the model to train and the question it would settle. None is scheduled in the analysis phase.

| ID | model to train | question |
|---|---|---|
| T1 | ViT CIFAR-10 WaNet 10% without noise mode, seeds 0 to 2, beside the existing noise-mode replicates | noise mode as the only variable for X1, X6 and the specialist |
| T2 | WaNet with the noise amplitude at 0.5 and 2 times the reference | whether the jitter specialist's best amplitude follows the attacker's noise amplitude |
| T3 | BadNets with a random trigger position per poisoned image | E9, and whether PSBD-TM needs a fixed trigger position |
| T4 | BadNets with trigger sizes of 1, 4, 9 and 16 tokens | $m$ as a trained variable, since resizing a trigger at test time is not the same backdoor |
| T5 | a ViT with registers, fine-tuned with BadNets and WaNet | E6, whether a model with native sinks reacts to the $\beta$ tokens differently |
| T6 | 2 attack families absent from the panel (for example ISSBA and Refool) on CIFAR-10 and GTSRB | proxy attacks for the design protocol, so the design is chosen on attacks it is not evaluated on |

L4 (P8) needs `experiments/early_loss_signal/`, whose smoke job is written and not yet submitted. It predicts a first-drop AUROC of at least 0.9 on BPP and TaCT at 5%.

## Toward an optimal PSBD-like probe

The measured principles, stated as design rules, each name the measurement they rest on. They are the reason the new-operator order above looks the way it does.

1. **Hide reads, independently per block.** Clean evidence needs about 70% of tokens visible in every block (an AND), and a patch trigger needs 1 intact read in about 4 late blocks (an OR). Hiding tokens from attention exploits exactly that, and independence across blocks is what raises the trigger's survival from $1 - p^{m}$ (masking once at the embedding, which read 0.787 on BadNets in the literature memo's reading of 2026-09-29) to $1 - p^{4m}$. Stratified masks (X14) are the limit of this rule.
2. **Do not touch storage for routed triggers, do touch it for computed ones.** The residual stream keeps a masked token's content, which is why hiding at the attention input spares the patch trigger. WaNet's evidence is assembled from the most displaced tokens by block 4 and held in the late stream, and late residual dropout reads 0.943 to 0.987 on the 4 ViT WaNet models where its adaptive rule reaches a rate. A probe that covers both needs a read perturbation and a storage perturbation, mixed or in a union (X3, X24).
3. **Prefer removal to additive noise before a LayerNorm.** LayerNorm absorbs 0.179 of a Gaussian disturbance at the attention input and 0.128 at the MLP input and almost none of a token mask (`\AbsorptionAttentionInputGaussianRemoved`), and Gaussian noise trails token masking by 0.188 at the attention input at the matched 0.6 rule (`\GaussianMinusTokenMaskAttentionNorm`).
4. **Know what the replacement injects.** A zeroed token becomes $\beta$ after the norm and every masked token shares 1 key, so the group acts as 1 sink whose mass grows with the number masked. Whether that helps or hurts is open (X5, X13).
5. **Choose every rate on clean data.** The adaptive rule does this for PSBD's sign, and the clean-stability rule of the WaNet specialist does it for a fragile-sign specialist.

**Mixture against union, and what it means for a partner.** `docs/novel-designs-theory.md` derives both combined AUROCs under a Gaussian copula from each member's AUROC and the clean correlation of the 2 probes. The mixture beats member A when the weaker member's separation ratio $\lambda$ exceeds $\sqrt{2(1+r)} - 1$ (0.62 at the measured $r = 0.31$). The union beats the mixture when $\lambda$ is below about 0.55, where the probes are lopsided or 1 is inverted. A null partner costs the union 0.022 and the mixture 0.096. An inverted partner at 0.4 leaves the union at 0.943 and drops the mixture to 0.834. A new operator therefore adds to a PSBD-TM union if it is the stronger member on some category, which on this panel means warp. Elsewhere its weakness costs at most about 0.02 per model provided it is not heavily inverted. That favors X16 and N5 as partners and rules out X22's members unless their patch sign is controlled.

**The combined design.** The general probe is stratified token masking at the attention input (X14) combined with late residual dropout in blocks 9 to 12, scored by the fractional statistic or by the worst pass if X19 favors it. The masking half carries patch triggers with the OR guaranteed and the dropout half carries the stored late evidence that WaNet and Blend leave in the stream. The source plan combined the 2 by averaging. The theory triage says the min-rank union is the safer rule wherever 1 member is weak on an attack, which PSBD-RD is on patch triggers, so the union is the default and the mixture is kept only if X3 shows it ahead on the development half. Its replacement rule (zero, bank or key mask) follows X5 and X13: keep zero if the 3 agree, since it is the measured one. Switch to the bank transplant if the artifact matters. The specialist is the native-resolution jitter (X6) with its fragile sign fixed from the noise-mode argument, in a budget-weighted minimum with the general probe at shares 0.9 and 0.1, which leaves the general probe at 0.09 FPR out of 0.10. The predicted outcome is a mean AUROC at least as high as the union's on every attack, TPR at 1% FPR on patch triggers higher than PSBD-TM's 0.747 (`\HeadlineTprAtOnePercent`) through X14, and the WaNet failure lifted above 0.9.

Swin needs 2 adjustments. It has no class token, so every read mask becomes a mask on the mean-pooled readout's inputs, and its early windows restrict who reads whom, so N9 replaces the knockout designs. `experiments/swin_mechanism/` already shows Swin's advantage on global triggers under token masking, and the Swin WaNet replicates with noise mode read 0.965 to 0.982, so the Swin design question is mainly whether stratification and the specialist transfer.

## An honest design protocol

The protocol has to avoid what Q20 and Q21 of `docs/open-questions.md` document: a selection half that picks a different winner than the pooled panel, and a winner judged on the models it was chosen on. The 5 steps below keep choosing and judging on separate models.

1. Freeze the candidate list, the selection metric and the tie rule before reading any new number, and write them into `configs/psbd_basis.json` beside the existing declaration. The metric is the mean paired AUROC gain over PSBD-TM plus the paired gain in TPR at 1% FPR. A candidate replaces PSBD-TM only when both intervals exclude 0 on the development half, and ties go to the cheaper and simpler design.
2. Explore on the 10 development models, with the 3 held-out-dataset members swapped under the rule above, and treat every reading there as exploratory.
3. Confirm the 2 or 3 surviving candidates on the whole development half (the 28 CIFAR-10 and GTSRB panel models) and pick 1 by the frozen metric (fallback night slot 3).
4. Evaluate that 1 design once on the held-out half (the 26 CIFAR-100 and Tiny ImageNet panel models) as the single confirmatory comparison against PSBD-TM, with a bootstrap interval, TPR at 1%, 10% and 20% FPR and the realized FPR. Report every frozen candidate's held-out numbers beside it, so the selection optimism is visible.
5. Evaluate once more on Swin-S over all 4 datasets, which the design never saw, as a second held-out set.

Some choices may use triggered data and some must not. The table lists each choice with where it may be made.

| choice | may use triggered data | where |
|---|---|---|
| which pre-registered candidate to keep | yes | development models only |
| structural choices: position, operator family, schedule shape, mixing proportion, budget shares | yes | development models only, frozen before step 4 |
| the rate or amplitude per model | no | clean validation, by the adaptive or the clean-stability rule |
| thresholds and percentile CDFs | no | clean validation |
| the sign of a specialist | no, it is fixed from a stated mechanism or the test is 2-sided | before any triggered image is scored |
| any per-attack choice | never, the defender does not know the attack | not applicable |
| anything on CIFAR-100, Tiny ImageNet or Swin before the freeze | never | not applicable |

2 checks need no triggered data at all and gate every candidate before step 3. The first stamps each trigger on inputs of the benign models, where every detector must read about 0.5 (`why_psbd_works` has benign BadNets and Blend probes, which read 0.491 to 0.503 under PSBD-TM). The second checks that the candidate's clean shift ratio reaches its target on every development model and that its realized clean-test FPR stays within 0.02 of nominal. Proxy attacks absent from the panel (T6) would let step 3 select on attacks the held-out evaluation does not contain, which is the strongest protection and needs training.

## Reverse-engineering the mechanism efficiently and correctly

Current practice converges on a few rules. Exploratory localization should be cheap and wide and confirmation exact and narrow (Heimersheim and Nanda). Corruptions should stay in distribution, with a symmetric counterfactual preferred to Gaussian noise, and the metric should be a logit difference rather than a probability (Zhang and Nanda). The direction of a patch decides what it finds, since denoising finds sufficient components and misses an AND while noising finds necessary ones and misses an OR (Heimersheim and Nanda). A hypothesis should be tested by resampling everything it calls irrelevant (Chan et al.) and judged by faithfulness, completeness and minimality (Wang et al.). Every ablation injects a distribution shift (Li and Janson), later components compensate for ablated ones (McGrath et al., Rushing and Nanda) and subspace patches can create what they claim to find (Makelov et al.). Automated circuit discovery (Conmy et al.) is too costly at the level of 197 positions by 12 blocks by 12 heads with 1 edge per pair, and attribution patching recovers circuits at least as well (Syed et al.), so the workflow uses attribution patching at position-group and head granularity, with ACDC reserved for a pooled graph of 3 position groups (trigger, class token, rest) by 12 heads by 12 blocks if a circuit claim is ever needed.

The question the workflow answers is why a backdoored ViT's triggered prediction survives PSBD-TM and a clean one does not. The patch models of the development set carry steps 1 to 7, and the WaNet pair (`vit_cifar10_wanet_0_1` and the swap for `vit_tiny_wanet_0_05`) repeats them as the contrast. The global models repeat steps 2, 5 and 8. Each step names the canonical items that carry it.

1. **Behavior and metric.** (a) What exactly survives? (b) The metric is the logit difference, target minus true class, per clean and triggered pair, with the clean twin as the symmetric counterfactual (the image-level analog of symmetric token replacement), read unperturbed and under PSBD-TM passes. (c) The control is the benign model with the same trigger stamped, where the difference should stay near 0. (d) It is done when the metric's unperturbed values agree with the cached baselines (`cache_agreement` in `why_psbd_works` already checks this) on all 10 models. Code: the pairing of `experiments/why_token_masking_works/measure.py` and `defenses.decision.pair_clean_to_backdoor`.
2. **Localization (X12).** (a) Which (block, position group, head) components carry the triggered difference? (b) First attribution patching, 1 forward and 1 backward pass per pair giving gradient times activation difference for every component, then exact patching of the top 20 in both directions, denoising triggered into clean (sufficiency) and noising clean into triggered (necessity), since the late reads are predicted to be an OR that noising alone cannot find. (c) The controls are a random position group of the same size and the group bordering the trigger. (d) It is done when the exact effects of the top components rank-correlate with the attribution estimates at 0.7 or more, and both directions are reported. Code: recording and patching modules attached through `models.positions.plug_dropout` hooks at `before_attention_norm` and the residual positions, and `analysis.features.captured_layers` for the stream.
3. **Route (X8, X20).** (a) Does the class token read the trigger directly, and through which heads? (b) Per-head direct logit attribution in blocks 9 to 12 first, then attention knockout of the class-token edges and the patch edges by band. (c) A random 4-key knockout in the same band. (d) It is done when knockout and patching agree on the blocks and heads. Code: `defenses.operators.masked_attention_forward` with an attention mask.
4. **Representation (X20, L8).** (a) What does the survived state look like? (b) The backdoor direction of `analysis/direction.py`, fitted on half the pairs, then the projection on the held-out half, a logit lens per block and per-token probes. (c) The direction must transfer to held-out pairs and to a second model of the same attack before any patching along it is read causally (Makelov et al., Bolukbasi et al.). Every probe reports selectivity against a control-label probe (Hewitt and Liang). (d) It is done when the held-out projection separates triggered from clean at AUROC 0.9 or more and probe selectivity is reported. `why_psbd_works` has the direction measurements (L8) this step extends.
5. **Ablation type (X5, X13).** (a) Are the token-mask results about information or about the artifact? (b) The resample and mean versions and the key mask, run on the rows that carry the paper's claims. (c) The zero version is itself the treatment, and agreement across the 4 rules is the control. (d) It is done when each claim is marked as holding under resample ablation or as depending on the constant replacement.
6. **Self-repair (X21).** (a) Do later components compensate for a knocked-out read? (b) Knock out the class token's trigger read in block 12 and measure the change in blocks 9 to 11's direct contributions. (c) The same knockout on a clean input. (d) It is done when total and direct effects are both reported for every knockout in steps 2 and 3.
7. **Causal scrubbing of the hypothesis.** (a) Does the stated mechanism account for the behavior? (b) The hypothesis "the triggered answer depends only on the trigger tokens' residual entries and the class token's reads of them in blocks 9 to 12" is tested by resampling every other token's attention input and every other read from clean images of other classes, and measuring the share of the logit difference kept. (c) The same resampling applied to the trigger tokens instead must remove the difference. (d) It is done when the hypothesis' components keep 0.9 or more of the difference (faithfulness), removing them from the full model drops it to the clean level (completeness) and dropping any 1 block from the hypothesis lowers the kept share (minimality).
8. **Link to the detector.** (a) Does the mechanism predict PSBD-TM's per-sample outcome? (b) Predict each pass's survival from the mechanistic variable of step 7, the number of late blocks in which some trigger token was visible (section B already records it) for patch triggers and the projection on the direction for global ones. (c) The same variable computed for a random token set must not predict survival. (d) It is done when the variable predicts per-pass survival at AUROC 0.9 or more.
9. **Generalization (X30).** (a) Does it hold on models the workflow was not developed on? (b) Rerun steps 2, 3, 5 and 8 once on the held-out CIFAR-100 and Tiny models and on Swin, after the analysis choices are frozen. (c) The benign controls again. (d) It is done when each step's verdict is stated per architecture.

The experiments to rerun under the 3 replacement versions of step 5 are sections A, B and D of `why_token_masking_works`, the token-mask cells of `why_psbd_works` for L22 and the headline PSBD-TM reading on the 10 models. If all 3 agree with the zero version, the paper's mechanism statements are about information. If they differ, the difference is the artifact's share. The paper should then say which of its claims depend on it.

## Contradictions between the sources

Merging the 3 documents exposed 6 places where they or the records disagree. Each is resolved by a named item rather than by choosing a side in advance.

- **E8 against X1.** The source plan ranked E8 as the leading account of the WaNet failure, and X1 found on 2026-09-29 that on the failing cell only 0.158 of changed triggered passes land on the true class. E8 survives only as the specialist sign X6 tests on the other noise-mode models, and the failure's first test is now X2, with X1b asking where the shifts go.
- **N5.** The source plan discarded random subspace removal as a probe (no OR and AND asymmetry), and the theory triage predicts it complements PSBD-TM on warp and blend by a cosine-margin argument. N5 is scheduled to decide it on the main night.
- **N8.** The source plan predicts warp fragile under early block skipping (a sign flip), and the theory triage predicts PSBD's sign with weak separation. N8's band-restricted variant decides it.
- **Mixture against union in the combined design.** The source plan averaged the 2 halves, and the theory triage shows the union is more robust when 1 member is weak or inverted on an attack. The combined design above now defaults to the union.
- **WaNet legibility.** Section D reads WaNet at 0.183 excess retention at 30% visible over 3 models, and `why_psbd_works` reads 0.706 on `vit_tiny_wanet_0_1` at the same fraction. X17 on both WaNet development models decides whether the difference is the model or the measurement.
- **The late band's selection.** The late band `pre_residual_blocks_9_12` was chosen after its WaNet readings were seen, so every gain it shows counted only after the held-out confirmation (X3, protocol step 4), which it passed with a small gain.

## Plausible explanations that may not hold

**Noise mode makes WaNet robust to perturbation.** It sounds right because noise mode was introduced to make WaNet stealthy against defenses. The training objective predicts the opposite for jittered triggered inputs, since a trigger field plus random per-pixel noise was labeled with the true class, so jitter turns the trigger off. X6 decides it, and the prediction is triggered retention at most 0.2 on noise-mode models.

**The WaNet failure is noise mode rejecting the masked input.** It sounded right because noise mode trains exactly that rejection. X1 measured the failing cell and found only 0.158 of its triggered shifts going to the true class, so the masked triggered input does not read as a rejected warp there.

**Swin reads WaNet better only because its seed-0 WaNet models lack noise mode.** The confound is real in `experiments/swin_mechanism/`'s matched pairs. It does not carry the result for PSBD-TM, since the Swin replicates with noise mode read 0.965 to 0.982 against 0.885 to 0.992 without it. The Swin residual-dropout readings vary much more (0.202 to 0.961) and deserve their own look.

**Token masking works because masked tokens are removed.** The masked tokens are replaced by a shared constant, which is an attention sink by construction. X5 and X13 measure what share of the effect is removal.

**Masking the trigger in the last block should break BadNets, because that is where the class token reads it.** Section A reads 0.934 kept for the last block alone. The OR over 4 late blocks explains most of it, and self-repair (X21) may explain the rest.

**WaNet fails under PSBD-TM because it is global like Blend.** Section D says WaNet is not redundant (0.183 of excess retention at 30% visible against 0.90 to 0.95 for Blend, LF and BPP), though `why_psbd_works` reads 0.706 on 1 Tiny model. The failure needs a different account, and after X1 the leading candidates are a checkpoint accident (X2) and a default-class collapse (X1b).

**Random subspace removal is the feature-space version of token masking.** The OR and AND asymmetry that makes token masking work has no counterpart among random directions. What is left is either the margin difference, which confidence already fails to explain (P1), or the cosine margin of the theory triage, which N5 tests.

**ViT's near permutation invariance makes the backdoor position-free.** Naseer et al. measured pretrained classifiers, and our fine-tuned models need 70% of tokens where theirs keep 60% accuracy at 20% visible. E9 is open until X10 runs.

**ShortcutProbe's settings transfer to ViT.** 4 of its 6 components do not, for reasons that are already measured. Absolute PSU at 20 passes loses to fractional PSU (historical reading on the 57-model panel of 2026-09-24: fractional wins on 37 of 57 models under PSBD-TM and 53 of 57 under PSBD-RD, and 3 to 20 passes gains +0.002). The rate that maximizes the clean shift ratio is reached near total destruction of the clean prediction, where triggered predictions fall too. The equal-weight sum is the mixture the union beats. The Youden threshold needs the true positive rate and so reads poisoned samples, where our threshold is a quantile of clean validation scores. Its 2 transferable ideas are S1 and S2.

## Source experiments in the literature

Each entry gives the design in 2 or 3 sentences, what it revealed and what it means for this project. Papers are grouped by the kind of manipulation, and each paragraph names the plan item it informs.

## Knockout, patching and circuit methods

**Causal tracing (Meng et al., NeurIPS 2022, arXiv 2202.05262).** The authors corrupt the subject tokens' embeddings of a factual prompt with Gaussian noise, then restore 1 hidden state at a time from the clean run and measure how much of the correct answer's probability returns. A few middle-layer MLP states at the last subject token restore most of it. For us the same grid over (block, token) with a clean and a triggered run of the same image is what `experiments/failure_modes/` already does at blocks 4, 8 and 12. X12 fills the missing blocks and the other direction of the patch.

**Attention knockout (Geva et al., EMNLP 2023, arXiv 2304.14767).** They block attention edges from chosen source positions to the predicting position in a window of layers and watch the answer's probability. Blocking subject-to-last edges only hurts in a specific band of upper layers, which located where information moves. For us this is X8, the cleanest way to ask whether the class token reads the trigger directly or through other patch tokens, which token masking at the attention input cannot separate because it hides a token from every query at once.

**Path patching and the IOI criteria (Wang et al., arXiv 2211.00593).** They patch activations only along specific paths between components and judge a circuit by faithfulness (the circuit alone reproduces the behavior), completeness (removing it from the full model breaks the behavior as much as removing it from the circuit) and minimality (every part matters). For us the trigger-token route to the class token is a candidate circuit, and these 3 checks are the standard of workflow step 7.

**Best practices of activation patching (Zhang and Nanda, ICLR 2024, arXiv 2309.16042).** They compare Gaussian noising of input embeddings with symmetric token replacement and several metrics on factual recall and IOI. Gaussian noising puts the model off distribution and can give illusory localizations, so they "recommend STR whenever possible", and they advise against probability as the metric because it can miss negative components, favoring logit difference. For us this means reporting the target-minus-true logit difference beside survival, and using a clean twin (symmetric replacement) rather than noise wherever a corruption is needed.

**How to use and interpret activation patching (Heimersheim and Nanda, arXiv 2404.15255).** A tutorial that separates denoising (patching clean into corrupt, which finds sufficient components) from noising (corrupt into clean, which finds necessary ones) and works through an AND gate and an OR gate. In an OR circuit noising either input alone does nothing and only denoising reveals each branch, and in an AND circuit the reverse holds. This is exactly our principle: the backdoor's late reads are an OR, so noising 1 block (the "last 1" row of section A, 0.934 kept) cannot find them and the denoising direction of X12 can.

**Causal scrubbing (Chan et al., Alignment Forum, 2022).** A hypothesis about which parts of a model matter is converted into a set of resampling ablations: every activation the hypothesis says is irrelevant is replaced by its value on another input that the hypothesis treats as equivalent, and the share of performance kept is the score. For us it is workflow step 7, the test that "the triggered prediction depends only on the trigger tokens' residual entries and the class token's late reads".

**ACDC (Conmy et al., NeurIPS 2023, arXiv 2304.14997).** Automated circuit discovery prunes edges of the computational graph one at a time, keeping an edge when removing it changes a KL metric by more than a threshold, and recovers hand-found circuits in small language models. For a ViT with 197 positions and 12 blocks the per-position graph is large and the method is costly, so it is proposed only on a head-level graph with positions pooled into trigger, class token and rest.

**Attribution patching (Nanda, blog post, 2023 and Syed et al., NeurIPS 2023 ATTRIB workshop, arXiv 2310.10348).** The effect of patching every activation is approximated at once by the gradient of the metric times the activation difference between the 2 runs, 1 forward and 1 backward pass in total. Syed et al. find it recovers circuits better than ACDC at a fraction of the cost, and Kramár et al. (arXiv 2403.00745) refine it (AtP*) for its failure cases. For us it is the cheap first pass of X12 over all (block, token, head) components before exact patching of the top candidates.

**Optimal ablation (Li and Janson, arXiv 2409.09951).** They classify ablations into zero, mean and resample ablation, show that each deletes information and also injects a distribution shift. They propose replacing a component with the constant that minimizes the loss. For us this is the lens on E6, since PSBD-TM is a constant ablation at the attention input (every masked token becomes $\beta$), which is neither the mean nor a resample. X5 runs the other 2.

**Self-repair (McGrath et al., arXiv 2307.15771 and Rushing and Nanda, ICML 2024, arXiv 2402.15390).** Ablating an attention layer in a language model makes later layers compensate for part of the lost effect (the hydra effect), and the compensation is spread over many components. For us it predicts that knocking out the trigger read in 1 late block understates that block's normal contribution, because the others take up the slack, which X21 measures.

**Interpretability illusions (Makelov et al., arXiv 2311.17030 and Bolukbasi et al., arXiv 2104.07143).** Makelov et al. show that patching along a subspace can switch on a dormant parallel pathway and look like localization where none exists, and Bolukbasi et al. show a unit that seems to encode 1 concept on 1 dataset and another on a second. For us it means a backdoor direction fitted on paired images must be validated on held-out pairs and on a second dataset before any patching along it is read causally.

**Localization against editing (Hase et al., NeurIPS 2023, arXiv 2301.04213).** Where causal tracing localizes a fact does not predict where editing it works best. For us it warns that "the trigger is read in blocks 9 to 12" does not by itself say that perturbing those blocks is the best probe, which the band ceiling in `experiments/probe_fusion/` already shows.

**Probes with control tasks (Hewitt and Liang, EMNLP 2019, arXiv 1909.03368).** A probe's accuracy is only meaningful relative to its accuracy on a control task with random labels of the same structure, and selectivity is the difference. For us every per-token probe (W8, L26's warped against noise-warped probe) needs a control-label run and must report selectivity.

**Logit lens on ViT and the tuned lens (Vilas et al., NeurIPS 2023, arXiv 2310.18969 and Belrose et al., arXiv 2303.08112).** Vilas et al. project intermediate ViT tokens onto the class embedding space and find class identity emerges in the later blocks, and Belrose et al. fit a small affine map per layer so intermediate states decode more faithfully than the raw unembedding. For us a per-block readout of the target-minus-true logit difference (X20) shows where the backdoor becomes decodable and where a perturbation removes it.

**Decomposing CLIP's image representation (Gandelsman et al., ICLR 2024, arXiv 2310.05916).** They mean-ablate layers of CLIP ViTs and decompose the output into per-head and per-token direct contributions, finding that the last few attention layers carry most of the direct effect. The exact layer count in the paper was not checked. For us per-head direct logit attribution in blocks 9 to 12 is the cheap readout of X20.

**Attention rollout and attention norms (Abnar and Zuidema, arXiv 2005.00928 and Kobayashi et al., EMNLP 2020, arXiv 2004.10102).** Rollout multiplies attention matrices across layers with the identity added for the residual path, and Kobayashi et al. show that the norm of the weighted value vector, not the weight alone, measures how much a token contributes. For us the attention-rank probe of X22 ranks by weight times value norm, not raw attention.

**Prisma (Joseph et al., arXiv 2504.19475).** An open toolkit that brings hooks, patching, sparse autoencoders and lens tools to vision and video transformers. We already have removable hooks in `models/positions.py`, so Prisma matters mainly as a reference implementation to cross-check a patching result.

## Tokens, sinks and missingness in ViTs

**Registers (Darcet et al., arXiv 2309.16588) and test-time registers (Jiang et al., NeurIPS 2025, arXiv 2506.08010).** Large ViTs develop a few high-norm tokens in low-information background patches that hold global information and attract attention, and adding extra register tokens removes these artifacts. Jiang et al. trace the outliers to a few "register neurons" and shift them into an extra untrained token at test time, getting most of the benefit without retraining. For us this is the machinery for the sink-only probe of X13: inject extra sink tokens without masking anything and see whether that alone reproduces PSBD-TM (E6).

**Massive activations and attention sinks (Sun et al., COLM 2024, arXiv 2402.17762 and Xiao et al., ICLR 2024, arXiv 2309.17453).** A handful of activations are orders of magnitude larger than the rest, act as fixed biases and concentrate attention, and Xiao et al. show that keeping the sink tokens is what stabilizes attention. The literature memo records that BadNets trigger tokens manufacture their own high-norm token (L24). For us the question is whether the $\beta$ tokens of PSBD-TM become a new sink that the triggered and clean inputs react to differently.

**Intriguing properties of ViTs (Naseer et al., NeurIPS 2021, arXiv 2105.10497).** They drop random, salient or background patches and shuffle patch order, finding that ViTs keep up to 60% top-1 accuracy on ImageNet after randomly occluding 80% of the image and that the position encoding adds less structure than expected, since shuffled patches cost them little. For us this is the baseline expectation for clean fragility under masking and under position-embedding jitter. Pretrained ViTs are robust to both, and our fine-tuned models need about 70% of tokens (E7), a large departure that X28 attributes to resolution or not.

**Missingness bias (Jain et al., ICLR 2022, arXiv 2204.08945).** Blacking out pixels biases a ResNet toward unrelated classes, while dropping the corresponding tokens of a ViT approximates missingness more faithfully. For us a zeroed token before LayerNorm is neither a black patch nor a dropped token, which is why E6 needs the direct tests of X5 and X13.

**Smoothed ViTs for patch robustness (Salman et al., CVPR 2022, arXiv 2110.07719).** Certified defenses against adversarial patches classify many column ablations of an image, and ViTs handle ablated inputs well when the ablated tokens are dropped instead of zeroed. For us the column-ablation vote is a PSBD relative, and dropping tokens from the key set is the version of token masking without a sink (N1).

**Token merging and dynamic pruning (Bolya et al., ICLR 2023, arXiv 2210.09461 and Rao et al., NeurIPS 2021, arXiv 2106.02034).** ToMe merges the most similar tokens in each block without training and keeps accuracy, and DynamicViT prunes tokens with a learned predictor. For us ToMe is a training-free perturbation that removes redundant tokens first, which should leave a dissimilar patch trigger untouched (X26).

**DropKey (Li et al., CVPR 2023, arXiv 2208.02646).** A regularizer that drops keys before the softmax instead of dropping attention weights after it. It is the operator the earlier design note proposed as "attention map masking", and the true key mask of X13.

**Occlusion without retraining (Hooker et al., NeurIPS 2019, arXiv 1806.10758).** ROAR retrains the model after removing the most important pixels, because removal without retraining confounds information loss with distribution shift. We cannot retrain for every mask, so the substitute is resample ablation with in-distribution replacement tokens (X5).

## Minimal inputs, transplants and shortcuts

**Sufficient input subsets and overinterpretation (Carter et al., AISTATS 2019, arXiv 1810.03805 and NeurIPS 2021, arXiv 2003.08907).** Backward selection finds the smallest pixel subset that keeps a confident prediction, and CIFAR-10 and ImageNet classifiers stay confident on 5% of pixels that mean nothing to a human. For us the minimal sufficient token set of a triggered input should be the trigger itself (E1) and far smaller than a clean input's, which `experiments/why_psbd_works/` tests as H-low-dim (L6).

**Clever Hans (Lapuschkin et al., Nature Communications 2019, arXiv 1902.10178).** A Fisher-vector model classified horses by a copyright tag, which the authors showed by pasting the tag onto a car image and getting "horse". For us the transplant is X11, the trigger pasted onto blank, noise and foreign images.

**SentiNet (Chou et al., DLS 2020, arXiv 1812.00292).** The salient region of a suspicious image is pasted onto held-out clean images and the fooled rate is compared with that of an inert pattern of the same size. For us the inert-pattern control is the part to copy, and the port's below-chance reading (Q23, `\DetectorsAurocSentinet` 0.418) is a warning that the transplant statistic's sign is subtle.

**Rethinking the trigger (Li et al., arXiv 2004.04692).** On ConvNets, moving a BadNets patch slightly or changing its appearance at test time drops the attack success rate sharply, and flipping or shrink-padding the test image is a cheap defense. For us moving the trigger across the token grid (X10) tests position binding (E9) on ViT and, because the grid position sets how many tokens a patch covers, manipulates m in the $p^{4m}$ law.

**Backdoor attacks on ViTs (Subramanya et al., arXiv 2206.08477 and WACV 2024).** Attention-based interpretation maps highlight the trigger on ViTs but not on CNNs, and blocking the top region of the map at test time cuts the attack success rate with a small clean-accuracy cost. For us this predicts that hiding the most-attended tokens flips the sign for patch triggers (X22).

**Patch processing on ViTs (Doan et al., AAAI 2023, arXiv 2206.12381).** Randomly dropping patches detects patch-based triggers and shuffling patches mitigates blending-based ones, a response the authors did not see on ConvNets. They group WaNet with blending-based attacks in the text, and whether they evaluated it could not be confirmed. For us patch shuffling is a candidate probe for warp and blend triggers (X16).

**Backdoor directions in ViTs (Karayalçin et al., arXiv 2603.10806).** A trigger direction in activation space steers the backdoor in both directions, and static patch triggers follow a different internal logic from stealthy distributed ones. The project's own direction measurements (L8) confirm the direction and disagree on WaNet's per-token legibility.

**Pruning, spectral and clustering analyses (Liu et al., arXiv 1805.12185, Tran et al., NeurIPS 2018, arXiv 1811.00636 and Chen et al., arXiv 1811.03728).** Fine-pruning plots clean accuracy and attack success against the number of pruned dormant neurons, spectral signatures separate poisons along the top singular vector of centered class representations and activation clustering splits each class's last-layer activations in 2. The project already refuted backdoor neurons (L7) and confirmed the direction (L8), so these add little beyond a pruning curve as a figure.

**Anti-backdoor learning (Li et al., NeurIPS 2021, arXiv 2110.11571).** Poisoned examples' training loss falls faster than clean examples' early in training. The literature memo's L4 and the queued `early_loss_signal` cover it, and it needs training.

## Perturbation consistency and input transformations

**STRIP, SCALE-UP and TeCo (Gao et al., ACSAC 2019, arXiv 1902.06531, Guo et al., ICLR 2023, arXiv 2302.03251 and Liu et al., CVPR 2023, arXiv 2303.18191).** STRIP superimposes clean images and flags low prediction entropy, SCALE-UP multiplies pixel values and flags consistent predictions, and TeCo applies 15 corruption types at growing severity and flags inputs whose severity of first prediction change varies widely across corruption types. TeCo's premise is that clean images are equally robust to every corruption while triggered images are robust to some and fragile to others. For us TeCo is the closest published relative of a WaNet specialist, since it reads 0.871 on the 3 WaNet panel models in our port against 0.525 for STRIP.

**WaNet (Nguyen and Tran, ICLR 2021, arXiv 2102.10369).** The trigger is a smooth backward warp from a 4 by 4 control grid, and noise mode trains on images warped by the trigger field plus a random per-pixel field, $W(x, M + \mathrm{rand}_{[-1,1]}(h, w, 2))$, with their true label. Without noise mode the model "cheated" by learning pixel-wise artifacts and Neural Cleanse caught it with small scattered trigger patterns, and with it the backdoor passed. The authors also write that STRIP's superimposition "will modify the image content and break the backdoor warping", and their CIFAR-10 models reach 93.16% accuracy on noise-mode images against 94.42% clean. For us noise mode is a direct prediction that triggered WaNet inputs are fragile to small random warps (X6).

**Frequency perspective (Zeng et al., ICCV 2021, arXiv 2104.03413).** Common triggers carry high-frequency artifacts, a low-pass filter removes many of them and the authors build smooth triggers (the LF attack) to avoid that. For us a blur or sub-pixel resampling perturbation (X15) should separate high-frequency triggers (BadNets checkerboard, Blend's random pattern, BPP quantization, WaNet's interpolation texture) from low-frequency ones (LF, SIG).

**Purification by transformation (Shi et al., NeurIPS 2023, arXiv 2303.12175, Sun et al., arXiv 2303.15564, Yang et al., NeurIPS 2024 and Miah and Bi, IJCNN 2026, arXiv 2602.07197).** ZIP destroys triggers with a linear transformation such as blur or downsampling and restores the image with a diffusion model, Mask and Restore masks and inpaints with a masked autoencoder, SampDetox finds that triggers of low visibility are destroyed by light noise and highly visible ones need strong noise, and Lite-BD's preliminary study on ResNet-18 CIFAR-10 over BadNets, Blend, WaNet, SIG and BPP finds down-upscaling the most disruptive of 10 transformations on average, followed by blur. In the Lite-BD result tables read for this plan, a resize-based stage leaves BadNets at an attack success rate of 1.000 and drops WaNet and BPP to about 0.05, but which method each column names was not confirmed. For us a resampling perturbation is a candidate specialist for WaNet and BPP that is blind to BadNets (X15).

**Strong augmentation (Borgnia et al., arXiv 2011.09527).** Mixup and CutMix during training sanitize many poisoning and backdoor attacks. It needs training and matters here only through our own `_aug` WaNet models.

**Monte Carlo dropout and randomized smoothing (Gal and Ghahramani, ICML 2016, arXiv 1506.02142 and Cohen et al., ICML 2019, arXiv 1902.02918).** Dropout at test time approximates Bayesian inference, and a Gaussian-smoothed classifier's majority vote is certifiably robust in a radius. `why_psbd_works` already refuted class-blind pass uncertainty as the carrier (P2), and smoothing matters here as the certified analog of the column-ablation vote.

**PatchCleanser (Xiang et al., USENIX Security 2022, arXiv 2108.09135).** 2 rounds of masking with a set of masks that is guaranteed to cover any patch of a given size, and a prediction is certified when every 1-mask prediction agrees. For us the covering-set idea is the deterministic relative of stratified masking (X14), and its agreement test is PSBD with hard labels (X19).

**Robustness-aware perturbations (Yang et al., EMNLP 2021, arXiv 2110.07831).** In NLP, poisoned samples are more robust than clean ones to a word-level perturbation crafted on clean data, and the gap detects them. It is the text precedent for choosing the perturbation on clean data only.

**ShortcutProbe (Prasad et al., ISDFS 2026, `literature/shortcutprobe-prasad-isdfs2026/`).** It combines PSBD's absolute PSU under a dropout of positive activations with a deterministic score, the largest single confidence drop while channels are removed in order of gradient times activation, and adds the 2 with equal weights. It is ResNet-18 on CIFAR-10 at 10% poisoning only. Its 2 transferable ideas are S1 and S2, and the other 4 components are discussed under "Plausible explanations that may not hold".

## References

Abnar and Zuidema, Quantifying Attention Flow in Transformers, arXiv 2005.00928.
Belrose et al., Eliciting Latent Predictions from Transformers with the Tuned Lens, arXiv 2303.08112.
Bolukbasi et al., An Interpretability Illusion for BERT, arXiv 2104.07143.
Bolya et al., Token Merging: Your ViT But Faster, ICLR 2023, arXiv 2210.09461.
Borgnia et al., Strong Data Augmentation Sanitizes Poisoning and Backdoor Attacks Without an Accuracy Tradeoff, arXiv 2011.09527.
Carter et al., What Made You Do This? Understanding Black-Box Decisions with Sufficient Input Subsets, AISTATS 2019, arXiv 1810.03805.
Carter et al., Overinterpretation Reveals Image Classification Model Pathologies, NeurIPS 2021, arXiv 2003.08907.
Chan et al., Causal Scrubbing: a method for rigorously testing interpretability hypotheses, AI Alignment Forum, December 2022.
Chen et al., Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering, arXiv 1811.03728.
Chou et al., SentiNet: Detecting Localized Universal Attacks Against Deep Learning Systems, DLS, arXiv 1812.00292.
Cohen et al., Certified Adversarial Robustness via Randomized Smoothing, ICML 2019, arXiv 1902.02918.
Conmy et al., Towards Automated Circuit Discovery for Mechanistic Interpretability, NeurIPS 2023, arXiv 2304.14997.
Darcet et al., Vision Transformers Need Registers, arXiv 2309.16588.
Doan et al., Defending Backdoor Attacks on Vision Transformer via Patch Processing, AAAI 2023, arXiv 2206.12381.
Gal and Ghahramani, Dropout as a Bayesian Approximation, ICML 2016, arXiv 1506.02142.
Gandelsman et al., Interpreting CLIP's Image Representation via Text-Based Decomposition, ICLR 2024, arXiv 2310.05916.
Gao et al., STRIP: A Defence Against Trojan Attacks on Deep Neural Networks, ACSAC 2019, arXiv 1902.06531.
Geva et al., Dissecting Recall of Factual Associations in Auto-Regressive Language Models, EMNLP 2023, arXiv 2304.14767.
Guo et al., SCALE-UP: An Efficient Black-box Input-level Backdoor Detection via Analyzing Scaled Prediction Consistency, ICLR 2023, arXiv 2302.03251.
Hase et al., Does Localization Inform Editing?, NeurIPS 2023, arXiv 2301.04213.
Heimersheim and Nanda, How to use and interpret activation patching, arXiv 2404.15255.
Hewitt and Liang, Designing and Interpreting Probes with Control Tasks, EMNLP 2019, arXiv 1909.03368.
Hooker et al., A Benchmark for Interpretability Methods in Deep Neural Networks, NeurIPS 2019, arXiv 1806.10758.
Jain et al., Missingness Bias in Model Debugging, ICLR 2022, arXiv 2204.08945.
Jiang et al., Vision Transformers Don't Need Trained Registers, NeurIPS 2025, arXiv 2506.08010.
Joseph et al., Prisma: An Open Source Toolkit for Mechanistic Interpretability in Vision and Video, arXiv 2504.19475.
Karayalçin et al., Backdoor Directions in Vision Transformers, arXiv 2603.10806.
Kobayashi et al., Attention is Not Only a Weight: Analyzing Transformers with Vector Norms, EMNLP 2020, arXiv 2004.10102.
Kramár et al., AtP*, arXiv 2403.00745.
Lapuschkin et al., Unmasking Clever Hans Predictors and Assessing What Machines Really Learn, Nature Communications 2019, arXiv 1902.10178.
Li and Janson, Optimal ablation for interpretability, arXiv 2409.09951.
Li et al., DropKey, CVPR 2023, arXiv 2208.02646.
Li et al., Rethinking the Trigger of Backdoor Attack, arXiv 2004.04692.
Li et al., Anti-Backdoor Learning: Training Clean Models on Poisoned Data, NeurIPS 2021, arXiv 2110.11571.
Li et al., PSBD: Prediction Shift Uncertainty Unlocks Backdoor Detection, arXiv 2406.05826.
Liu et al., Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks, arXiv 1805.12185.
Liu et al., Detecting Backdoors During the Inference Stage Based on Corruption Robustness Consistency, CVPR 2023, arXiv 2303.18191.
Makelov et al., Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching, arXiv 2311.17030.
McGrath et al., The Hydra Effect: Emergent Self-repair in Language Model Computations, arXiv 2307.15771.
Meng et al., Locating and Editing Factual Associations in GPT, NeurIPS 2022, arXiv 2202.05262.
Miah and Bi, Lite-BD: A Lightweight Black-box Backdoor Defense via Reviving Multi-Stage Image Transformations, IJCNN 2026, arXiv 2602.07197.
Nanda, Attribution Patching: Activation Patching At Industrial Scale, blog post, 2023.
Naseer et al., Intriguing Properties of Vision Transformers, NeurIPS 2021, arXiv 2105.10497.
Nguyen and Tran, WaNet: Imperceptible Warping-based Backdoor Attack, ICLR 2021, arXiv 2102.10369.
Prasad et al., Shortcutprobe: Identifying Backdoor Samples via Internal Stability Analysis, ISDFS 2026, IEEE Xplore 11459068.
Rao et al., DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification, NeurIPS 2021, arXiv 2106.02034.
Rushing and Nanda, Explorations of Self-Repair in Language Models, ICML 2024, arXiv 2402.15390.
Salman et al., Certified Patch Robustness via Smoothed Vision Transformers, CVPR 2022, arXiv 2110.07719.
Shi et al., Black-box Backdoor Defense via Zero-shot Image Purification, NeurIPS 2023, arXiv 2303.12175.
Subramanya et al., Backdoor Attacks on Vision Transformers, arXiv 2206.08477, published as A Closer Look at Robustness of Vision Transformers to Backdoor Attacks, WACV 2024.
Sun et al., Massive Activations in Large Language Models, COLM 2024, arXiv 2402.17762.
Sun et al., Mask and Restore: Blind Backdoor Defense at Test Time with Masked Autoencoder, arXiv 2303.15564.
Syed et al., Attribution Patching Outperforms Automated Circuit Discovery, NeurIPS 2023 ATTRIB workshop, arXiv 2310.10348.
Tran et al., Spectral Signatures in Backdoor Attacks, NeurIPS 2018, arXiv 1811.00636.
Vilas et al., Analyzing Vision Transformers for Image Classification in Class Embedding Space, NeurIPS 2023, arXiv 2310.18969.
Wang et al., Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small, arXiv 2211.00593.
Xiang et al., PatchCleanser: Certifiably Robust Defense against Adversarial Patches for Any Image Classifier, USENIX Security 2022, arXiv 2108.09135.
Xiao et al., Efficient Streaming Language Models with Attention Sinks, ICLR 2024, arXiv 2309.17453.
Yang et al., RAP: Robustness-Aware Perturbations for Defending against Backdoor Attacks on NLP Models, EMNLP 2021, arXiv 2110.07831.
Yang et al., SampDetox: Black-box Backdoor Defense via Perturbation-based Sample Detoxification, NeurIPS 2024.
Zeng et al., Rethinking the Backdoor Attacks' Triggers: A Frequency Perspective, ICCV 2021, arXiv 2104.03413.
Zhang and Nanda, Towards Best Practices of Activation Patching in Language Models: Metrics and Methods, ICLR 2024, arXiv 2309.16042.
