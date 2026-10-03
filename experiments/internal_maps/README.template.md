# Internal maps of the backdoored ViT-B/16 and Swin-S models

This directory draws what happens inside a backdoored transformer when PSBD-TM (`token_mask` at `before_attention_norm`) probes it. The account under test comes from `experiments/why_token_masking_works/` and `docs/evidence-surplus-theory.md`: hiding tokens from attention removes ordinary class evidence, a patch trigger keeps its evidence in its own tokens and the class token reads it in the late blocks, a global trigger stays legible from a fraction of the tokens and WaNet has no localized surplus. The first batch illustrates that account with 5 figures. The second batch asks 4 questions the account raises and records what the answers change.

README.md is rendered by `render_readme.py` from `README.template.md` and the JSON records, so every number below is read off a record. Edit the template, never README.md.

## Models

Every model is on the successful panel (`successful_2pt` for ViT, `scripts.paper._common.swin_coverage` for Swin). `config.py` lists them with the reason for each pick.

- ViT-B/16: {vit_models}
- Swin-S: {swin_models}

The benign models carry no trigger and are probed with the BadNets patch and target 0, the way their caches were swept. No ViT WaNet model on GTSRB or CIFAR-100 clears the panel, so the ViT WaNet is Tiny. No Swin GTSRB TaCT model is on the Swin panel, so the Swin TaCT is CIFAR-10 at 5%. Its PSBD-TM cache holds 1 rate that stays below the 0.8 shift target, so it runs at the nearest cached rate ({swin_cifar10_tact_rate}, rule `{swin_cifar10_tact_rate_rule}`), the labeled fallback of `experiments.cache_readouts.shared.choose_rate`. Every other model runs at its adaptive rate, read from its own stage-1 cache.

The CIFAR-100 BadNets models were added after the first pass. The GTSRB models have target 0, and a GTSRB model that sees almost nothing falls onto class 0, so on GTSRB the target probability of a nearly empty input is high for a reason unrelated to the trigger. Swin CIFAR-100 BadNets falls onto its target too (clean twins on the target with 30% of the image visible {swin_cifar100_badnet_floor_03}), while ViT CIFAR-100 BadNets does not ({vit_cifar100_badnet_floor_03}). Every map that keeps only a few tokens therefore subtracts the clean twin's target probability under the same mask, and the ViT CIFAR-100 model gives a BadNets reading free of that floor.

## Method

Pairs are the first triggered images of the PSBD analysis split (`data.splits`, seed `PSBD_SPLIT_SEED`) that the unperturbed model sends to the target, with the clean twin of each, built through `build_psbd_loaders_from_checkpoint` as `cli.sweep` builds them. Before any measurement the unperturbed predictions are held to the cached baseline (`why_token_masking_works.tokens.cached_baseline_agreement`), and every model agreed on every pair. A map unit is 1 ViT token of the 14 by 14 grid. On Swin it is 1 cell of the 7 by 7 grid every stage nests into, hidden as all of its tokens at every stage. A unit is hidden by zeroing it at the attention input of every block (`measure.GridKeep`), which is PSBD-TM's operator with a chosen mask: the LayerNorm right after this position cancels TokenMask's rescale of the survivors.

The reproduction check against the earlier records holds. On `vit_gtsrb_badnet_a2o_0_1` hiding the 4 trigger tokens in every block keeps {vit_badnet_hidden_kept} of triggered predictions and as many random tokens keep {vit_badnet_random_kept}, against {vit_badnet_earlier_hidden} and {vit_badnet_earlier_random} in `why_token_masking_works/vit/`. PSBD-TM at rate {vit_badnet_tm_rate} keeps {vit_badnet_tm_triggered_kept} of triggered and {vit_badnet_tm_clean_kept} of clean predictions here, against {vit_badnet_earlier_tm_triggered} and {vit_badnet_earlier_tm_clean} there. On `swin_gtsrb_badnet_a2o_0_1` the hidden trigger keeps {swin_badnet_hidden_kept} (earlier {swin_badnet_earlier_hidden}) and PSBD-TM keeps {swin_badnet_tm_triggered_kept} and {swin_badnet_tm_clean_kept} (earlier {swin_badnet_earlier_tm_triggered} and {swin_badnet_earlier_tm_clean}).

## Commands and outputs

```bash
source .venv/bin/activate
# GPU, under 1 of the 2 project lock slots, memory fraction 0.3, bfloat16
flock -E 75 scratch/gpu.lock python -m experiments.internal_maps.measure
flock -E 75 scratch/gpu.lock python -m experiments.internal_maps.batch2 confidence
flock -E 75 scratch/gpu.lock python -m experiments.internal_maps.batch2 veto
flock -E 75 scratch/gpu.lock python -m experiments.internal_maps.batch2 survival
# CPU
python -m experiments.internal_maps.make
python -m experiments.internal_maps.gallery
python -m experiments.internal_maps.batch2_figures
python -m experiments.internal_maps.render_readme
```

Everything is written under `results/_experiments/internal_maps/`. The PNG files are not tracked and come back with the CPU commands. The JSON files are the record.

| path | content |
|---|---|
| `<arch>/<model>/numbers.json` | every map and number of the 5 first-batch figures of that model |
| `<arch>/<model>/token_removal.png` | figure 1 |
| `vit/<model>/class_token_attention.png`, `swin/<model>/readout_attribution.png` | figure 2 |
| `<arch>/<model>/depth_survival.png` | figure 3 |
| `<arch>/<model>/direction.png` | figure 4 |
| `<arch>/<model>/psbd_tm_masks.png` | figure 5 |
| `<arch>/overview_token_removal.png`, `<arch>/overview_depth_probe.png`, `vit/overview_depth_lens.png` | 1 row or panel per model |
| `<arch>/numbers.json` | the scalars of every overview, 1 row per model |
| `<arch>/<model>/gallery_numbers.json`, `gallery_threshold_clean.png`, `gallery_escaping_triggered.png`, `gallery_summary.json` | second batch, question 1 |
| `vit/<model>/confidence.json`, `confidence_gain.png` | second batch, question 2 |
| `vit/<model>/veto.json`, `tact_veto.png` | second batch, question 3 |
| `vit/<model>/survival.json`, `survival_against_kept.png`, `survival_numbers.json` | second batch, question 4 |
| `batch2_numbers.json` | the scalars of the second batch |

## Figure 1, token removal on the patch grid

Each row shows the clean input, the triggered input, the trigger's pixel footprint and 2 maps over 32 pairs. Map (a) is the drop in the target probability when 1 unit alone is hidden in every block. Map (b) is the target probability when only a window of units around each position stays visible, minus the clean twin's target probability under the same window. The last panel is map (a) on the clean twins for their own class.

The patch triggers behave as claimed, with 1 refinement. On Swin a BadNets cell hidden alone drops the target probability by {swin_gtsrb_badnet_drop_max} (GTSRB) and {swin_cifar100_badnet_drop_max} (CIFAR-100), all of it on the trigger's cell. On ViT no single token is necessary: the largest single-token drop is {vit_gtsrb_badnet_drop_max} on GTSRB and {vit_cifar100_badnet_drop_max} on CIFAR-100, while hiding all 4 trigger tokens keeps nothing. The 4 tokens are redundant carriers that each nearly suffice, the overdetermination reading of `docs/evidence-surplus-theory.md`. Map (b) is localized: on `vit_cifar100_badnet_a2o_0_1` a 3 by 3 window over the trigger raises the target probability above the clean floor by up to {vit_cifar100_badnet_window_3_excess_max}, and the mean over all window positions is {vit_cifar100_badnet_window_3_excess_mean}. On GTSRB BadNets the same map is flat ({vit_gtsrb_badnet_window_3_excess_max} at most) because the clean floor is already at the target.

The global triggers have no single necessary unit (largest drop {vit_gtsrb_blend_drop_max} Blend, {vit_gtsrb_lf_drop_max} LF and {vit_tiny_wanet_drop_max} WaNet on ViT). Their window maps are spread. A random 30% of tokens visible keeps triggered predictions above the clean floor by {vit_gtsrb_blend_excess_03} (Blend), {vit_gtsrb_lf_excess_03} (LF) and {vit_tiny_wanet_excess_03} (WaNet) on ViT. On Swin they read {swin_gtsrb_blend_excess_03}, {swin_gtsrb_lf_excess_03} and {swin_gtsrb_wanet_excess_03}. With 10% visible the excess falls to {vit_gtsrb_blend_excess_01}, {vit_gtsrb_lf_excess_01} and {vit_tiny_wanet_excess_01} on ViT.

2 readings disagree with the account as stated. The ViT Tiny WaNet model is legible from 30% of its tokens (excess {vit_tiny_wanet_excess_03}), where `why_token_masking_works` part D recorded a much lower retention on the 2 WaNet models it measured. So "WaNet has no localized surplus" holds (map (a) is flat) and "WaNet is not legible from a subset" does not hold on this model. The Swin GTSRB WaNet reading at 10% and 30% is no evidence either way, since its clean twins land on the target as often (floor {swin_gtsrb_wanet_floor_01} and {swin_gtsrb_wanet_floor_03}), which is the attractor of the Models section. The TaCT failure `vit_gtsrb_tact_0_01_cos` keeps {vit_gtsrb_tact_visible_03} of triggered predictions at 30% visible and {vit_gtsrb_tact_visible_06} at 60%: its trigger needs the source content, and masking removes it.

## Figure 2, where the trigger is read

On ViT the left heatmaps give, per block and head, the class token's attention mass on the trigger tokens on triggered and clean images, with uniform attention at {vit_gtsrb_badnet_uniform}. The image overlays show attention rollout (Abnar and Zuidema, identity mixed in at 0.5) and the class token's own head-mean attention of single blocks.

On GTSRB BadNets the class token's mass on the 4 trigger tokens passes 0.1 in block {vit_gtsrb_badnet_mass_onset} and reaches {vit_gtsrb_badnet_mass_last} averaged over the heads of block 12, against {vit_gtsrb_badnet_clean_mass_last} on the clean twins. On CIFAR-100 BadNets it passes 0.1 in block {vit_cifar100_badnet_mass_onset} and reaches {vit_cifar100_badnet_mass_last}. The benign model never looks at the patch (largest head {vit_gtsrb_benign_mass_max}). The TaCT failure barely does either: its largest head puts {vit_gtsrb_tact_mass_max} on the trigger, and its single-block attention sits on the sign's content, which agrees with figure 1. Attention rollout does not show the late read. It accumulates the early, diffuse blocks and stays on the content on the 2 models inspected (GTSRB BadNets and TaCT), so the single-block overlays are the ones to show.

Swin has no class token. Its readout is the mean of the last stage's 49 tokens after the final LayerNorm followed by the linear head. The norm acts per token, so the target logit splits exactly into 1 term per token (bottom row). Earlier stages have other widths, so there each token's triggered minus clean difference is weighted by the gradient of the target logit (top row), a first-order attribution. Gradient times activation was tried first and discarded, since behind a per-token LayerNorm it is 0 by scale invariance. On Swin BadNets the positive share of the difference attribution on the trigger's tokens is {swin_gtsrb_badnet_difference_stage_1}, {swin_gtsrb_badnet_difference_stage_2}, {swin_gtsrb_badnet_difference_stage_3} and {swin_gtsrb_badnet_difference_stage_4} at the ends of stages 1 to 4, and the exact readout puts {swin_gtsrb_badnet_readout_share} of its triggered minus clean excess on the trigger's cell (area {swin_gtsrb_badnet_area_last}). Swin carries the trigger in its own tokens through stage 2 and spreads it over the whole map in stage 3, so the readout sees target evidence everywhere. This agrees with `why_token_masking_works` gate 3 and its routes test.

## Figure 3, the read along depth

The logit lens applies ViT's own final LayerNorm and head to the class token after every block, with no fitting. A per-block linear probe of the model's own decision (class token on ViT, mean token on Swin), fit on unperturbed features of clean validation images and triggered pairs that are not evaluated, is drawn beside it. Each panel shows triggered P(target) and clean P(own class), unmasked and under PSBD-TM at the model's rate over 6 passes.

The logit lens shows the account directly. On GTSRB BadNets the triggered target probability passes 0.5 at block {vit_gtsrb_badnet_lens_triggered_onset} and the clean class at block {vit_gtsrb_badnet_lens_clean_onset}. Under PSBD-TM the triggered curve passes 0.5 at block {vit_gtsrb_badnet_lens_triggered_masked_onset} and ends with {vit_gtsrb_badnet_triggered_masked_kept} kept, while the clean curve never exceeds {vit_gtsrb_badnet_lens_clean_masked_max}. Blend, LF and WaNet read earlier than BadNets (blocks {vit_gtsrb_blend_lens_triggered_onset}, {vit_gtsrb_lf_lens_triggered_onset} and {vit_tiny_wanet_lens_triggered_onset}) and survive masking the same way. On the TaCT failure the triggered read comes at block {vit_gtsrb_tact_lens_triggered_onset}, after the clean one at block {vit_gtsrb_tact_lens_clean_onset}. Under masking it never forms ({vit_gtsrb_tact_triggered_masked_kept} kept). On the benign model the 2 curves coincide.

The linear probe is not informative under masking and is kept only as a record. Fitted on unperturbed features, it reads masked features out of distribution: on Swin Blend its masked triggered probability falls to {swin_gtsrb_blend_probe_masked_min} in the middle blocks while the model itself keeps {swin_gtsrb_blend_triggered_masked_kept} of the predictions. Swin has no logit lens before stage 4, where the widths change, so Swin has no clean depth figure here.

## Figure 4, the backdoor direction per token

The direction is the mean triggered minus clean difference of the readout: the class token after block 12 on ViT, the mean token at the end of each stage on Swin. Each token's paired difference at each block is projected on it. The right panel gives the share of the absolute projection that sits on the trigger tokens, and on ViT the class token's own projection.

On ViT BadNets the trigger tokens carry {vit_gtsrb_badnet_direction_first} of the absolute projection in block 1, and the class token picks up the direction from block {vit_gtsrb_badnet_cls_projection_onset} on GTSRB and {vit_cifar100_badnet_cls_projection_onset} on CIFAR-100. By block 12 the trigger's share is {vit_gtsrb_badnet_direction_last} (GTSRB) and {vit_cifar100_badnet_direction_last} (CIFAR-100). The sign is unexpected: the trigger tokens' own difference projects {vit_gtsrb_badnet_trigger_projection_sign} on the class token's direction. Attention does not copy the trigger tokens' content into the class token. Its value and output maps write a different vector, so the figure shows where the difference sits and when the class token takes its own direction, not that the trigger tokens carry the class token's direction. On Swin BadNets the trigger's share falls from {swin_gtsrb_badnet_direction_first} in block 1 to {swin_gtsrb_badnet_direction_last} in block 24, the same spreading as figure 2.

## Figure 5, the PSBD-TM masks

3 passes of the library operator at the model's rate on 1 clean and 1 triggered image, with the share of blocks 9 to 12 (Swin 17 to 24) in which each region was hidden shaded black, and the per-pass prediction and probabilities printed. On GTSRB BadNets the triggered target probability never falls below {vit_gtsrb_badnet_mask_triggered_target_min} in the 3 passes, and the clean image goes to class {vit_gtsrb_badnet_mask_clean_predictions} (target {vit_gtsrb_badnet_mask_target}). On CIFAR-100 BadNets the clean image goes to {vit_cifar100_badnet_mask_clean_predictions} and the triggered target probability stays at or above {vit_cifar100_badnet_mask_triggered_target_min}. On the TaCT failure the clean image goes to class {vit_gtsrb_tact_mask_clean_predictions} and the triggered target probability falls to {vit_gtsrb_tact_mask_triggered_target_min}. That class returns in question 1 below.

## Second batch, question 1, the threshold gallery

For each model the 20 clean validation images with the lowest PSBD-TM fractional PSU at the adaptive rate are the ones that set the 1% FPR threshold. The 20 triggered images sent to the target with the highest PSU are the escapes. The images are rebuilt through `cli.sweep`'s loader and held to the cache by a CPU forward pass of the first rows of each split and of the gallery images.

{gallery_table}

What we expected. The threshold images would be target-class images, low-confidence images or images whose confidence rises under masking, spread over classes.

What we saw. 3 different causes set the threshold. On `bb_cifar10_blind_0_1` the threshold is set by the target class itself ({g_bb_cifar10_blind_0_1_clean_target} of the 20 are class 0, {g_bb_cifar10_blind_0_1_clean_negative} have negative PSU): masking pushes every image towards the target, and a genuine target-class image only gains. On `bb_gtsrb_ssba_0_1` every one of the 20 is misclassified with P_c below 0.9 and negative PSU, and {g_bb_gtsrb_ssba_0_1_clean_predicted_target} of them are predicted as the target. That checkpoint is weak on clean data ({g_bb_gtsrb_ssba_0_1_base_misclassified} of all validation images misclassified, {g_bb_gtsrb_ssba_0_1_base_low_confidence} below 0.9), so its threshold sits at a negative PSU and TPR at 1% FPR is {g_bb_gtsrb_ssba_0_1_tpr}. On the GTSRB ViT models the threshold is set by 1 class: {g_vit_gtsrb_tact_0_01_cos_largest_class} of the TaCT failure's 20 are class {g_vit_gtsrb_tact_0_01_cos_top_class} and {g_vit_gtsrb_bpp_0_05_largest_class} of the BPP model's are class {g_vit_gtsrb_bpp_0_05_top_class}. Class {g_vit_gtsrb_tact_0_01_cos_top_class} is where the TaCT failure sends masked clean images (figure 5), so its own images are stable under masking and set the threshold, while triggered source images flip away from the target and score as unstable. That is a candidate mechanism for the TaCT floor: the masked model has an attractor class that is not the target, and the triggered prediction does not survive. On the Swin models the threshold is partly the target class ({g_swin_cifar100_badnet_a2o_0_01_clean_target} on CIFAR-100 BadNets, {g_swin_tiny_wanet_0_05_clean_target} on Tiny WaNet). Among the 20 highest-PSU triggered images of the whole split, the non-hits (triggered images the model does not send to the target) are all 20 on {gallery_non_hit_all_count} of the {gallery_model_count} models, so a raw "highest PSU" gallery shows failed triggers, which PSBD rightly calls clean. The gallery draws hits only.

Does it change anything. The 1% FPR threshold is set by target-class and attractor-class images, not by random clean images. Either a class-conditional threshold or excluding images predicted as the attractor would move the low-FPR operating point. That is a hypothesis to test on the cache, not a result.

## Second batch, question 2, confidence-gain maps

On the 2 BackdoorBench checkpoints, every clean validation image with negative PSU (up to 24) and 24 triggered hits with PSU nearest 0 get a single-token map of the change in P_c and the class token's last-block attention, plus a joint test that hides each image's 5, 10 and 20 most rising tokens at once against as many random tokens.

What we expected. If clean stability is denoising, the tokens whose removal raises the prediction would be background or distractor tokens: off the object, on the border and ignored by the class token.

What we saw. On `bb_gtsrb_ssba_0_1` ({c_bb_gtsrb_ssba_0_1_negative_count} of 2000 validation images have negative PSU) the clean images start at P_c {c_bb_gtsrb_ssba_0_1_clean_base} and hiding their 10 most rising tokens raises it to {c_bb_gtsrb_ssba_0_1_clean_top_10}, against {c_bb_gtsrb_ssba_0_1_clean_random_10} for 10 random tokens. The rising tokens are not background: {c_bb_gtsrb_ssba_0_1_clean_attention_share} of the rise sits on the quarter of tokens the class token attends to most (0.25 by chance), and {c_bb_gtsrb_ssba_0_1_clean_border} on the 2-token border ring that holds {c_bb_gtsrb_ssba_0_1_border_area} of the area. On `bb_cifar10_blind_0_1` ({c_bb_cifar10_blind_0_1_negative_count} negative-PSU images) the same holds more weakly ({c_bb_cifar10_blind_0_1_clean_attention_share} on the top attention quarter, border {c_bb_cifar10_blind_0_1_clean_border}, top 10 {c_bb_cifar10_blind_0_1_clean_top_10} against random {c_bb_cifar10_blind_0_1_clean_random_10} from {c_bb_cifar10_blind_0_1_clean_base}). The triggered images near PSU 0 react to nothing: their largest single-token rise is {c_bb_gtsrb_ssba_0_1_triggered_max_rise} and {c_bb_cifar10_blind_0_1_triggered_max_rise}, and hiding 20 tokens leaves P(target) at {c_bb_gtsrb_ssba_0_1_triggered_top_20} and {c_bb_cifar10_blind_0_1_triggered_top_20}.

Does it change anything. Clean "stability" here is a different behavior from a trigger's survival. A clean negative-PSU image is a contested decision whose competing evidence sits on attended object tokens, and hiding those tokens settles it. A triggered image is insensitive to every token. The 2 have the same low score for opposite reasons, which suggests a second statistic (sensitivity to single tokens, or the spread of per-pass probabilities) could separate them. That is a hypothesis.

## Second batch, question 3, TaCT veto maps and the content-hidden probe

On the 3 single-source ViT TaCT models, correctly classified clean images of classes other than the source and the target are stamped with the trigger. For the stamped images the model refuses, a single-token map gives the rise in P(target), and each image's top-k rising tokens are hidden together (k up to 64) against k random tokens and against the same tokens on the unstamped twin. The content-hidden probe was registered in `PREDICTIONS_tact_content_hidden.md` and committed before it was scored.

{veto_table}

What we expected. If the veto is localized, a few tokens of the non-source content would hold the trigger back, and hiding them would release it.

What we saw. Every stamped non-source image is refused on all 3 models. The veto is spread and weak to undo: no single token raises P(target) by much, and hiding 64 tokens (a third of the image) releases the trigger on at most the share in the table. The unstamped twin stays off the target under the same masks, so what is released is the trigger. On `vit_gtsrb_tact_0_01_cos` hiding more tokens releases less, and the hypothesis is that the masked model falls onto its attractor class first. The source images lose P(target) when a single token is hidden ("source" panels), and in figure 1 that loss sits on the trigger's tokens ({vit_gtsrb_tact_drop_share} of the positive drop on GTSRB).

The content-hidden probe. Prediction 1 bounded both AUROC (at most 0.65) and TPR at 10% FPR (at most 0.20). The TPR part holds on all {veto_model_count} models. The AUROC part is refuted on {veto_auroc_refuted_count} of {veto_model_count}: AUROC reads {v_vit_gtsrb_tact_0_01_cos_auroc}, {v_vit_cifar10_tact_0_05_auroc} and {v_vit_cifar10_tact_0_01_auroc}. The registered refutation rule therefore says the single-source trigger carries a signal in its own tokens that the content-hidden excess missed. The signal is tiny and class-bound. With the content hidden both triggered images and their clean twins (all from the source class) lose nearly all their probability (mean scores {v_vit_cifar10_tact_0_05_mean_triggered} and {v_vit_cifar10_tact_0_05_mean_clean} on CIFAR-10 at 5%), and the triggered ones lose slightly less. The 2000 validation images of all classes score lower on average ({v_vit_cifar10_tact_0_05_mean_validation}), so a threshold set on them catches no triggered image.

Does it change anything. The AND form of single-source TaCT is confirmed from 2 more sides: the trigger alone does not reach the target and non-source content holds it back over many tokens. The probe ranks triggered above clean source images within the source class only, so a class-conditional threshold is the next test. As a deployable detector it fails.

## Second batch, question 4, survival against trigger tokens kept

Under PSBD-TM at rates 0.3, 0.5, 0.7 and 0.9, 6 passes over 256 triggered hits per model record how many trigger tokens stayed visible in each block and whether the target prediction survived. "Kept" is the visible share of trigger tokens averaged over blocks 9 to 12, where the class token reads a patch trigger (the all-block mean is in the record). A logistic curve of survival on kept is fit over all rates. The ratio P(target) over its unperturbed value is fit the same way at rate 0.5 only, and TPR is predicted at every rate by drawing each late block's visible trigger tokens as Binomial(m, 1 - rate), mapping kept through that curve, averaging 3 passes as the cache does and thresholding at the cache's clean validation quantiles. The measured TPR is the cache's own at the same rate.

{survival_table}

What we expected. If survival depends on the trigger only through how many of its tokens stay visible, 1 curve per model would describe every rate, and TPR would follow from rate and trigger size.

What we saw. The kept share alone predicts the fitted rate well (mean absolute TPR error {survival_error_in}) and the other rates poorly ({survival_error_out}). On BadNets survival stays near 1 down to a small kept share, and the prediction fails mainly at rate 0.9, where the measured TPR falls and the predicted one does not. A likely reason, not tested here, is that survival differs between images, so some triggered images break in all 3 passes, which 1 curve for every image cannot produce. On the 2 partly detected TaCT models survival at the same kept share falls with the rate (the rate 0.3 points sit above the curve and the rate 0.9 points below it, `survival_against_kept.png`), so the content the mask removes matters beside the trigger. On the TaCT failure nothing survives at any rate.

Does it change anything. "PSBD-TM catches a patch trigger when enough of its tokens stay visible in the late blocks" describes BadNets near the adaptive rate and nothing more. A TPR model needs the clean side and the per-image spread as well, and for TaCT the trigger-kept share is the wrong variable.

## Recommendations for the report

The ViT logit-lens overview (`vit/overview_depth_lens.png`) is the single clearest figure: the late read of the trigger, the masked triggered read surviving and the masked clean read never forming, the TaCT failure and the benign control in 1 panel each. The ViT token-removal overview (`vit/overview_token_removal.png`, with the CIFAR-100 BadNets row) and the Swin one show localized against spread evidence, and the class-token attention figure of `vit_gtsrb_badnet_a2o_0_1` beside that of `vit_gtsrb_tact_0_01_cos` shows the read and its absence. The TaCT failure's threshold gallery with its attractor class is the one new mechanism here worth a sentence. The linear-probe depth figures, attention rollout and the figure 4 sign are not recommended for the report.

## Limits

- Every map averages 32 to 64 pairs of 1 model per attack, so a map is an illustration of 1 model, not a panel result.
- The GTSRB attractor floor (target 0) weakens every GTSRB reading that hides most tokens, and the excess form only partly corrects it.
- The Swin TaCT model runs at the nearest cached rate, below the adaptive target.
- The second batch's confidence and veto questions use 15 to 32 images per model.
