# How a backdoor manifests inside ViT-B/16 and Swin-S

## Question

A backdoored classifier sends any input carrying a trigger to a target class while it keeps its accuracy on clean inputs. The rest of this repository measures how a detector reacts to such a model. This experiment measures the object the detector reacts to. It asks, per attack category and per architecture, 6 questions. What does the trigger change in the input? Where in the image does the network read it? At which depth does the representation of a triggered image leave that of its clean twin? Can the backdoor be found as a small set of "backdoor neurons", or is it a direction in the residual stream that no single coordinate carries? What does the separation look like in the head's input space? How do the attack categories and the 2 architectures differ?

Most earlier answers in this repository were measured on CIFAR-10 ViT models at 10% poisoning only (`experiments/backdoor_neurons/`, `experiments/backdoor_direction_layers/`, `experiments/token_structure/`). This experiment repeats the core measurements on CIFAR-10, CIFAR-100, GTSRB and Tiny ImageNet and on ViT-B/16 and Swin-S. It also adds the controls those experiments lacked (listed under "Controls"). The notebook `notebooks/backdoor-manifestation.ipynb` walks through every figure. This README states the design, the sanity checks, the numbers and the verdicts.

## Terms

The **residual stream** is the tensor a transformer block reads and writes back into. On ViT-B/16 it is (batch, 197, 768): 196 patch tokens from a 14 by 14 grid of 16 pixel patches of the 224 by 224 input, plus 1 class token. On Swin-S it is (batch, height, width, channels) and the grid shrinks through 4 stages, 56 by 56 by 96, 28 by 28 by 192, 14 by 14 by 384 and 7 by 7 by 768. **Layer** l means the output of block l, and layer 0 is the input of block 1, the numbering of `analysis/features.py`. ViT-B/16 has 12 blocks and Swin-S 24.

The **pooled feature** at a layer is the vector the classifier head would read if the network stopped there: the class token on ViT, the mean over the grid on Swin (`analysis.features.default_reduction`). The **head input** is what the final linear layer actually reads: the final LayerNorm of the class token on ViT, the mean over the grid of the final LayerNorm of every token on Swin.

A **pair** is 1 test image taken twice, once clean and once with the trigger. The **backdoor direction** at a layer is the mean over pairs of the triggered pooled feature minus the clean one (`analysis.direction.backdoor_direction`). **TAC**, the trigger-activated change of a coordinate, is the mean over pairs of the absolute change of that coordinate (`analysis.direction.trigger_activated_change`). A **backdoor neuron** in the sense of the literature (Zheng et al., ECCV 2022) is a coordinate with a large TAC.

**ASR**, the attack success rate, is the share of triggered eligible images the model sends to the target class. **Eligible** follows `attacks.poisoning.is_eval_poisonable`: every image whose true class is not the target, and for TaCT only its source class 1.

## Design

### Panel

| category | attack | models | why these cells |
| --- | --- | --- | --- |
| patch | BadNets, all to one | `{vit,swin}_{cifar10,cifar100,gtsrb,tiny}_badnet_a2o_0_05` | 5% is successful on every dataset and architecture |
| patch | TaCT | `vit_cifar10_tact_0_01`, `vit_cifar10_tact_0_05`, `vit_gtsrb_tact_0_05`, `swin_cifar10_tact_0_05` | the 3 ViT cells are the only TaCT models that are trigger-conditional (8 of 11 ViT TaCT models map their clean source class to the target with no trigger, see `docs/runs/2026-09-24-tact-multisource.md`). The Swin cell is screened by measurement, see "Exclusions" |
| blend | Blend, LF | `{vit,swin}_{dataset}_{blend,lf}_0_05` | 5% is successful everywhere |
| warp | WaNet | `vit_cifar10_wanet_0_1`, `vit_tiny_wanet_0_05`, `swin_cifar10_wanet_0_05`, `swin_cifar100_wanet_0_1`, `swin_gtsrb_wanet_0_1`, `swin_tiny_wanet_0_05` | no ViT WaNet model is successful on CIFAR-100 or GTSRB at any rate, and on CIFAR-10 only the 10% one is (the 5% model loses 3.9 points of clean accuracy). On Swin the 10% cell is taken where 5% misses the ASR bar |
| quantization | BPP | `{vit,swin}_{dataset}_bpp_0_05` | 5% is successful everywhere |
| control | benign | `{vit,swin}_{dataset}_benign`, each probed with the trigger of every backdoored run of its family | a benign model shown the same trigger tells a learned backdoor apart from the input change itself |

**Successful** is the rule every result in the repository now uses: the attack clears the ASR bar 0.85 and the model's clean accuracy is no more than 2 points below the benign model of the same architecture and dataset (`successful_2pt` in `results/coverage/coverage.json`). `backdoored_run` refuses any cell that is not. The ledger covers ViT only, so for a Swin cell `successful_at_2_points` applies the same rule to the ASR and clean accuracy in its `args.json` against the Swin benign model's. On 2026-09-29 that rule and the in-memory Swin ledger `scripts.paper._common.swin_coverage` agreed on all 21 Swin cells here. A failed attack is left out even where that removes a whole attack from a dataset.

That is 42 backdoored runs (21 ViT, 21 Swin) and 41 benign-control runs, 83 in all. A benign model is probed at target class 0, the target of every backdoored cell in the panel, and only with triggers its family has a backdoored run for, since a control with no backdoored twin reads nothing.

SIG, the frequency-category attack, is left out. `attacks/sig.py` builds a sine amplitude of 0.157 since commit 0150611 (2026-09-09) and the ViT SIG checkpoints were trained at 0.1 with no override recorded in their `args.json`, so a rebuilt SIG trigger is not the trigger those models learned (`docs/audits/2026-09-29-experiment-audit.md`). Until the overrides are written into the sidecars, a SIG reading would measure the wrong trigger. The frequency category therefore has no entry here, and the notebook says so where it would appear.

### Data

`prepare` builds the PSBD split through `data.splits.build_psbd_loaders_from_checkpoint`, the same call every detection number uses: a seeded shuffle of the test set (seed `PSBD_SPLIT_SEED` 0), the first 2000 images held out as the clean validation set and the rest as the analysis split. Its backdoor loader is `attacks.poisoning.AttackSuccessSet`, which keeps eligible images only and applies the trigger with `attack_success_label`. `defenses.decision.pair_clean_to_backdoor` maps every backdoor row to the clean row of the same test image. Out of the eligible pairs, 800 are drawn with a seeded permutation (all of them when fewer exist, 796 for the CIFAR-10 TaCT models). Up to 200 clean images of the target class are kept as well, since the eligible pairs exclude the target class and 2 readings need it (the target-class direction and target-class recall).

The 800 pairs are split in 2 halves, the **fit half** (rows 0 to 399) and the **eval half** (rows 400 to 799). Every quantity that is fitted, the backdoor direction used for ablation and steering, the TAC ranking that names the "top neurons", the single neuron with the best fit-half AUROC, the control directions, the PCA and the UMAP, is fitted on the fit half. Every number that scores it, ASR after ablation, AUROC of a neuron or a direction, the drawn embedding, is read on the eval half. The target images are halved the same way. The per-layer depth curves (relative direction norm, TAC, CKA, participation, direction share) are descriptive statistics of the whole set of 800 pairs and score nothing, so they use all pairs.

The other-trigger control needs the same clean images stamped with a trigger the model was not trained on. `other_triggered_images` plants Blend on the patch and warp models and BadNets on the blend and quantization models (`OTHER_TRIGGER`), in pixel space before normalization as the PSBD loaders do.

### Precision

Every forward runs under bfloat16 autocast, a rule of the shared login-node GPU for this run. `experiments/backdoor_direction_layers/` argues for fp32 because TAC, a mean of absolute differences, keeps a bfloat16 rounding floor that grows with the residual norm. That floor is present in the benign controls too, which run at the same precision on the same images, so every backdoored curve is read against its control rather than against 0. Sanity gate 1 checks that bfloat16 predictions reproduce the fp32 `metrics.json` numbers within 0.01.

### Measurements

| step | quantity | function in `measure.py` | formula |
| --- | --- | --- | --- |
| 1 | mean pixel change and its patch-grid average | `input_change_map`, `input_change_grid` | mean over pairs and channels of the absolute pixel difference, upsampled to 224 and averaged over each 16 by 16 patch |
| 2 | per-token change at every layer | `token_change_maps` | see below |
| 2 | class-token attention on trigger tokens, per block and head | `attention_readout` | see below |
| 3 | relative direction norm, direction share, TAC, CKA per layer | `layer_row` | see below |
| 4 | top-TAC coordinates, split-half and cross-attack overlap | `layer_row`, `top_dimension_overlap`, `chance_jaccard` | Jaccard of 2 top-20 sets |
| 4 | separability of neurons and of the direction | `neuron_readout` | AUROC on the eval half |
| 4 | ablation and steering | `ablation_readout` | see "Controls" |
| 5 | PCA and UMAP of the head input | `embedding_record` | fit half fitted, eval half drawn |
| 5 | alignment with the classifier | `readout_alignment` | see below |

The per-token change at layer l for token t is

$$m_l(t) = \frac{\frac{1}{N}\sum_{i=1}^{N} \lVert h_l(\tilde x_i)_t - h_l(x_i)_t \rVert_2}{\frac{1}{N T}\sum_{i=1}^{N}\sum_{t'=1}^{T} \lVert h_l(x_i)_{t'} \rVert_2}$$

| symbol | meaning |
| --- | --- |
| $x_i$, $\tilde x_i$ | clean image i and the same image with the trigger |
| $h_l(\cdot)_t$ | token t of the residual stream at layer l |
| $N$ | pairs, 800 |
| $T$ | tokens at layer l, 197 on ViT, the grid size on Swin |

The denominator makes the map comparable across layers, whose residual norms grow with depth. On ViT the class token is reported apart from the 196 patch tokens.

The attention readout recomputes each ViT block's attention weights from its LayerNorm output, since torchvision calls the attention with `need_weights=False` (`class_token_attention_hook`). It keeps the class token's row, $a_{l,h}(t)$ for block l, head h and key token t. The **trigger tokens** are the patch tokens whose patch-grid input change is at least 0.25 of the largest one (`TRIGGER_TOKEN_SHARE`), 1 to 4 tokens for a patch trigger and most of the grid for a global one. The mass on trigger tokens is $\sum_{t \in S} a_{l,h}(t)$ for the trigger set S, averaged over pairs, and uniform attention would give $|S| / 197$. Swin has no class token and attends inside shifted windows, so this readout is ViT only.

At each layer, with $d_i = \bar h_l(\tilde x_i) - \bar h_l(x_i)$ the pooled paired change and $r = \frac{1}{N}\sum_i d_i$ the backdoor direction,

| quantity | formula | reading |
| --- | --- | --- |
| relative direction norm | $\lVert r \rVert / \frac{1}{N}\sum_i \lVert \bar h_l(x_i) \rVert$ | how far the triggered mean moves, in units of the clean feature norm |
| direction share | $\lVert r \rVert^2 / \frac{1}{N}\sum_i \lVert d_i \rVert^2$ | 1 when every image moves by the same vector, near 0 when the moves cancel |
| TAC of coordinate k | $\frac{1}{N}\sum_i \lvert d_{i,k} \rvert$ | the per-neuron change the "backdoor neuron" literature ranks |
| normalized max TAC | $\max_k \mathrm{TAC}_k / \frac{1}{N D}\sum_{i,k} \lvert \bar h_l(x_i)_k \rvert$ | the largest neuron change against the typical clean activation |
| CKA | debiased linear CKA of clean against triggered features (`analysis.cka.debiased_linear_cka`) | 1 when the 2 sets are the same representation up to rotation and scale |
| participation ratio | $1 / \sum_k u_k^4$ with $u = r / \lVert r \rVert$ | how many coordinates the direction is spread over, 1 for a single axis and about D / 3 for a random direction in D dimensions |
| top-20 TAC energy | $\sum_{k \in \text{top 20 TAC}} u_k^2$ | the share of the direction the 20 "backdoor neurons" carry |

The **onset layer** is the first layer whose relative direction norm reaches half its largest value, the **peak layer** the layer of the largest value.

`readout_alignment` compares the backdoor direction at the head input, $r_h$, with the rows $w_c$ of the classifier weight after centering them over classes ($\tilde w_c = w_c - \frac{1}{C}\sum_{c'} w_{c'}$, since adding 1 vector to every row changes every logit equally and no prediction). It reports $\cos(\tilde w_{\text{target}}, r_h)$, the largest cosine with any other class row and the rank of the target among the logit shifts $\tilde W r_h$. A random direction's cosine has standard deviation about $1/\sqrt{768} = 0.036$.

### Controls

Removal of 1 direction $u$ at a block's output sets every token $x$ to $x - (x \cdot u)\,u$ (`remove_direction_hook`). Removal of k coordinates sets them to 0 (`zero_coordinates_hook`). The 2 are the same operation, zeroing the stream's component along chosen unit vectors, in a rotated basis and in the neuron basis. `tests/test_backdoor_manifestation.py` checks that they agree exactly when $u$ is a coordinate axis. Both are applied to every token, since a later block can read a token the edit skipped, at 2 depths: the last block (12 on ViT, 24 on Swin) and 2/3 of the depth (8 on ViT, 16 on Swin).

| removal | control of the same kind | what the control rules out |
| --- | --- | --- |
| backdoor direction | 3 isotropic random directions | removing any single direction breaks the model |
| backdoor direction | the top principal direction of the clean fit features (`clean_pc1`) | removing the highest-variance direction breaks the model |
| backdoor direction | the clean principal direction whose clean variance is closest to the backdoor direction's, among those at cosine below 0.2 to it (`variance_matched_pc`, `MATCHED_PC_MAX_COSINE`) | the effect comes from the amount of clean energy removed, the energy-matched control the audit asked for. Without the cosine limit the match at the last block returned a principal direction at cosine 0.86 to the backdoor direction on `vit_cifar10_badnet_a2o_0_05`, a copy rather than a control |
| backdoor direction | the target class's mean offset, mean of target fit features minus mean of clean fit features | removing the target class's own evidence is what breaks the backdoor, which would also cost target-class recall |
| backdoor direction | the backdoor direction with its target-class component projected out, $r - (r \cdot \hat t)\,\hat t$ for the unit target offset $\hat t$ | whatever the backdoor adds beyond "look like the target class" |
| backdoor direction | the direction of the other trigger on the same model | any trigger's direction would do |
| backdoor direction | the direction the same trigger induces in the benign model of the same dataset | the direction reflects the input change, not a learned backdoor |
| top k TAC coordinates | k random coordinates, for every k in 20, 100 and 300 | removing any k coordinates |
| top k TAC coordinates | the k coordinates of largest mean clean magnitude (`top_magnitude`) | a "top neuron" rule picking ViT's few massive activation dimensions |
| top k TAC coordinates | the k largest coordinates of the direction itself (`top_direction_coordinates`) | a better neuron ranking would have worked |

Every removal reports ASR, paired clean accuracy, the share of clean images sent to the target, target-class recall on the eval-half target images and per-class recall over the eval-half clean pairs and target images. Steering adds a direction to the clean eval images at the same block and counts how many the model then sends to the target. Every steering control (1 random direction, clean PC 1, the variance-matched PC, the target-class direction, the target-orthogonal part of the backdoor direction, the other trigger's direction and the benign model's direction) is rescaled to the backdoor direction's norm, so only the orientation differs.

The benign model's direction exists only once that model's own run has saved it, which is why `panel_runs` lists the benign controls before the backdoored runs.

### Exclusions

`backdoored_run` refuses any cell the coverage ledger (`results/coverage/coverage.json`) marks `diverged` or `source_mapped`. The ledger scores ViT cells only, so a Swin TaCT cell is screened by measurement instead, with the rule `experiments/why_token_masking_works` uses: TaCT's pairs are its source class only, so their paired clean accuracy is the source-class accuracy, and below 0.5 the run is marked `source_mapped` in `summary.json` and left out of every mechanism figure. Cells below the ASR bar 0.85 are kept and tagged `below_asr_bar` in both the run JSON and `summary.json`.

## Sanity checks

`sanity.py` ran on `vit_cifar10_badnet_a2o_0_05` and `swin_cifar10_badnet_a2o_0_05` before the panel, through the probe stage's own `batch_readout`. Its records are under `results/_experiments/backdoor_manifestation/sanity/`.

1. **Behavior.** On the whole analysis split (8000 clean images, 7200 triggered eligible images) the bfloat16 readout gives ViT clean accuracy 0.9445 and ASR 0.9997 against 0.9418 and 0.9998 in `checkpoints/vit_cifar10_badnet_a2o_0_05/metrics.json`, and Swin 0.9651 and 1.0000 against 0.9657 and 1.0000. Both pass the 0.01 tolerance. `metrics.json` scores the full 10000-image test set, so the 2 populations overlap in 8000 images and differ in 2000.
2. **Layers.** The tensor read as layer l equals the output of block l of a forward that walks the blocks by hand (the ViT embedding, class token, position embedding, then `encoder.layers` 1 by 1, and Swin's `features` stage by stage), with a largest absolute difference of 0.0 at every layer in fp32. The head input equals the final LayerNorm of the class token on ViT and the grid mean of the final LayerNorm on Swin, difference 0.0. The pooled feature equals the class token (ViT) or the grid mean (Swin) at every layer. Shapes for a batch of 8: ViT (8, 197, 768) at layers 0 to 12, Swin (8, 56, 56, 96) at layers 0 to 2, (8, 28, 28, 192) at 3 and 4, (8, 14, 14, 384) at 5 to 22 and (8, 7, 7, 768) at 23 and 24.
3. **Pairs and split.** Every cached pair's label is eligible under `is_eval_poisonable`, every cached test index is in the manifest's `analysis_backdoor_indices`, and for the BadNets trigger the clean and triggered images differ only in 0.88% of pixel positions, the 3 by 3 corner patch of a 32 by 32 image. Fitting on the fit half and scoring on the eval half is described under "Data".
4. **Random controls.** Every top-k neuron removal has a random-coordinate removal at the same k, and every direction removal has isotropic, energy-matched and task-matched direction controls (the "Controls" table).
5. **Exclusions.** Diverged and source-mapped ledger cells raise at panel construction, and Swin TaCT is screened by measurement (see "Exclusions").

## Commands

```bash
# CPU, outside the GPU lock: pairs, target images and the other trigger
PYTHONPATH=. python experiments/backdoor_manifestation/measure.py prepare --run vit_cifar10_badnet_a2o_0_05
# the sanity gates, 1 model per call, under the shared GPU lock
flock scratch/gpu.lock env PYTHONPATH=. python experiments/backdoor_manifestation/sanity.py --run vit_cifar10_badnet_a2o_0_05
# the whole panel, resumable, 1 model per lock hold, no new model between 06:30 and 17:00
bash experiments/backdoor_manifestation/run_panel.sh
# CPU: embeddings, overlap matrices and summary.json
PYTHONPATH=. python experiments/backdoor_manifestation/measure.py summarize
```

Outputs are `results/_experiments/backdoor_manifestation/runs/<run>.json` (1 per run), `embeddings/<run>.json`, `sanity/<run>.json` and `summary.json`. The pair caches and head features live under `scratch/backdoor_manifestation/` and are not a source of any number that is not also in a JSON above.

## Run status

All 83 runs finished on the login-node A100 on 2026-09-29 between 19:05 and 23:23, and `summarize` wrote `summary.json` and the 83 embedding records the same night. Nothing is left to run for this panel. The marker `scratch/gpu_done_manifestation` was written at 23:23. Every run is resumable: `run_panel.sh` skips a run whose JSON exists, so adding a model to `BACKDOORED` and rerunning the script measures only the new one and its benign control.

The runs of the ViT CIFAR-10 family measured before the target-orthogonal and cosine-limited controls existed were deleted and remeasured, so every record on disk comes from the same code. SIG waits for its sidecar fix (see "Panel"). The Swin TaCT model reads a source-class clean accuracy of at least 0.5 and is kept.

## Findings

Every number below is a mean over the 42 backdoored runs of the named architecture unless a range is given. Removals and steering are read on the eval half after fitting on the fit half, at the last block unless stated. The notebook draws each as a figure and renders its sentence from `summary.json`.

### Where the trigger is read

On the 4 ViT BadNets models the class token's head-mean attention on the trigger's 1 to 3 tokens in blocks 9 to 12 grows 74 to 450 times from the clean to the triggered image, against 1.5 to 14.4 times for the benign model shown the same trigger. On the 3 ViT TaCT models the factor is 7 to 24 against 0.3 to 2.1. For global triggers the trigger covers most of the grid and the readout says nothing (factor 1.02 to 1.48). After block 12 the class token's relative change is a median 5.7 times the benign control's (range 1.8 to 115.5 over the 21 ViT runs).

### Direction onset by depth

This table is the traceable replacement for the onset table in `experiments/backdoor_direction_layers/README.md`, whose `direction_layers_*.json` records are not on disk. Onset is the first layer whose relative direction norm (pooled feature, class token on ViT and grid mean on Swin) reaches half its largest value (`onset_layer`), read from `runs/<run>.json` for every successful panel model. The old table used "half the final value" at 10% poisoning in fp32. This one uses 5% (10% where named in "Panel") in bfloat16.

| architecture | attack | models | onset median | onset range |
| --- | --- | --- | --- | --- |
| ViT-B/16 | Blend | 4 | 6.5 | 5 to 8 |
| ViT-B/16 | WaNet | 2 | 7 | 7 to 7 |
| ViT-B/16 | BPP | 4 | 7.5 | 6 to 9 |
| ViT-B/16 | LF | 4 | 9 | 6 to 9 |
| ViT-B/16 | BadNets | 4 | 9.5 | 8 to 10 |
| ViT-B/16 | TaCT | 3 | 12 | 9 to 12 |
| ViT-B/16 | benign controls | 20 | 4 | 2 to 7 |
| Swin-S | Blend | 4 | 5 | 5 to 5 |
| Swin-S | BPP | 4 | 9 | 5 to 10 |
| Swin-S | LF | 4 | 9 | 8 to 17 |
| Swin-S | WaNet | 4 | 16 | 15 to 23 |
| Swin-S | BadNets | 4 | 18 | 13 to 23 |
| Swin-S | TaCT | 1 | 23 | 23 |
| Swin-S | benign controls | 21 | 5 | 3 to 5 |

The ViT ordering of the old table holds (Blend first, then BPP, LF and BadNets), and TaCT comes last. A benign control's direction norm peaks in the middle (median peak layer 9 on ViT, 5 on Swin) and falls, while every backdoored run's final relative direction norm is 2.3 to 188.7 times its control's (median 15.9). The benign onset is early only because its small peak is early, so it is not a comparable onset.

### Neurons against a direction

- **Top TAC coordinates are model specific.** Between backdoored runs of 1 family the Jaccard of the final-layer top-20 TAC sets averages 0.053, against a chance level of 0.014 and a split-half ceiling of 0.887 (ViT) and 0.909 (Swin).
- **Separability does not tell a neuron from a direction.** At the last layer the best single coordinate reaches an eval AUROC of 0.983 to 1.000 per attack mean, and the direction reaches 0.996 to 1.000.
- **No coordinate rule removes the backdoor.** Zeroing the 300 top-TAC coordinates of 768 at the last block leaves ASR at 0.924 (ViT) and 0.955 (Swin). 300 random coordinates and the 300 largest-magnitude coordinates both leave 0.986 (ViT) and 0.991 to 0.992 (Swin).
- **1 direction does, and no control direction does.** Removing the backdoor direction takes ASR from 0.986 to 0.038 on ViT and from 0.991 to 0.180 on Swin. The variance-matched clean PC, the isotropic random directions, the other trigger's direction and the benign model's direction for the same trigger each leave ASR at 0.985 to 0.992.
- **The direction is mostly the target class's axis.** The same removal costs target-class recall 0.961 to 0.436 on ViT and 0.971 to 0.613 on Swin, while the other classes move by -0.030 and -0.046. The backdoor direction minus its target-class component leaves ASR at 0.974 (ViT) and 0.986 (Swin). Removing the target-class offset alone gives ASR 0.837 and recall 0.079 on ViT. At the last block the rank-1 removal is therefore necessary for the backdoor but not specific to it: it removes part of the target class too. This qualifies finding 6 of `experiments/backdoor_neurons/`, which had no target-recall control.
- **Steering agrees.** Adding the backdoor direction to clean images at the last block sends 0.924 (ViT) and 0.864 (Swin) of them to the target. The target-class offset at the same norm sends 0.916 and 0.904, its orthogonal remainder 0.148 and 0.151, the benign model's direction 0.006 and 0.004 and a random direction 0.003 and 0.002.
- **Earlier in the network the picture changes.** At 2/3 of the depth, removing the direction leaves ASR at 0.820 (ViT) and 0.442 (Swin). That later blocks rebuild it from trigger tokens is a hypothesis this experiment does not test. On Swin that layer has 384 coordinates, so its k=300 neuron removals break the model (clean accuracy 0.338 to 0.516) and say nothing about neurons.
- **By attack.** The last-block direction removal takes ASR to 0.000 to 0.004 on every ViT attack except TaCT (0.257). On Swin it leaves BadNets and LF at 0.000, Blend and BPP at 0.250, WaNet at 0.196 and the 1 TaCT model at 0.987. On that TaCT model the representation barely moves (final CKA 0.882) and its onset is block 23.

### Head geometry

The head-input backdoor direction has a median cosine of 0.749 with the target's centered weight row (range 0.603 to 0.922), against 0.037 for the benign controls, whose scale for a random direction is 0.036. The target has the largest logit shift on all 42 runs. The PCA and UMAP projections of the head input are drawn in the notebook as an illustration and carry no measured claim.

### Verdicts

1. A backdoor in these ViT-B/16 and Swin-S models is read late. The class token routes patch triggers from their tokens in the last blocks, and global triggers enter the pooled feature earlier (onset table).
2. The TAC "backdoor neuron" is not the unit the backdoor lives in. No rule of up to 300 coordinates removes it, while 1 direction does.
3. At the last block that direction is largely the target class's readout direction, so removing it also costs about half of the target class's recall. The backdoor-specific remainder of the direction is neither necessary nor sufficient at that block.
4. TaCT is the exception on both architectures. Its direction is weaker to remove on ViT (ASR 0.257 after removal) and not removable on the 1 Swin model.
