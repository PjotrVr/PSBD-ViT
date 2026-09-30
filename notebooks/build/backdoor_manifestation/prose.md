<!-- title -->
# Backdoor manifestation in ViT-B/16 and Swin-S

A backdoored image classifier behaves normally on clean images and sends any image that carries a small, fixed pattern, the trigger, to 1 class chosen by the attacker, the target class. This notebook opens such models and shows where and how the trigger's effect appears inside them, attack by attack, on the 2 architectures this project studies, and how a defender with triggered images in hand can find it. It follows the measurement from the pixels to the classifier head and asks at every step which unit carries the backdoor: pixels, tokens, attention heads, single neurons or a direction.

The notebook is written for a reader who knows what a vision transformer is and has not read the rest of this repository. It reads cached JSON only and draws every figure from it, so it runs on a login node in about a minute. The numbers come from `experiments/backdoor_manifestation/measure.py`, whose README states the design, the controls, the sanity checks and the verdicts in the same terms. Every step below names the function that computes what it shows.

<!-- map -->
## Map of the walkthrough

The walkthrough has 6 steps, and each one answers the question the previous one raises.

1. **The triggers in pixel space.** What does each attack change in the image, and over how much of it? Computed by `input_change_map` and `input_change_grid`.
2. **Where the network reads the trigger.** Which patch tokens change inside the network and at which depth? Does the class token attend to them? Computed by `token_change_maps`, `top_dimension_token_maps` and `attention_readout`.
3. **How the backdoor grows with depth.** At which block does the triggered representation leave the clean one? Computed by `layer_row` (relative direction norm, TAC, CKA).
4. **Backdoor neurons against a backdoor direction.** Is the backdoor carried by a few coordinates (the "backdoor neurons" of the literature) or by a direction that no single coordinate carries? Computed by `layer_row`, `neuron_readout` and `ablation_readout`, which deletes each candidate and measures ASR with a control for every deletion.
5. **Latent geometry at the head.** What do clean and triggered images look like where the classifier reads them, and does the backdoor direction point at the target class's weight row? Computed by `embedding_record` and `readout_alignment`.
6. **Differences by attack category and architecture.** What changes between patch, blend, warp and quantization triggers? What changes between ViT-B/16 and Swin-S?

Every quantity is measured on paired images: the same test image once clean and once with the trigger. Every fitted object (a direction, a neuron ranking, a projection) is fitted on 1 half of the pairs (the **fit half**) and scored on the other (the **eval half**), so no number rewards an object for memorizing its own samples. Every backdoored model is read beside a **benign control**: the model trained on the same dataset without poison, shown the same trigger. Anything the benign model also shows is a property of the input change, not of a learned backdoor.

<!-- terms -->
## Terms

The **residual stream** is the tensor each transformer block reads and adds its output back into. On ViT-B/16 it has shape (batch, 197, 768): 196 patch tokens, 1 for each 16 by 16 pixel patch of the 224 by 224 input on a 14 by 14 grid, and 1 **class token** whose final state the classifier reads. On Swin-S it has shape (batch, height, width, channels) and the grid shrinks over 4 stages, 56 by 56 with 96 channels, then 28 by 28 by 192, 14 by 14 by 384 and 7 by 7 by 768. Swin-S has no class token, and its head reads the mean over the last grid. Every model here upsamples its native 32 by 32 (CIFAR, GTSRB) or 64 by 64 (Tiny ImageNet) input to 224 inside the network, so a trigger is planted at native resolution.

**Layer** l means the output of block l, and layer 0 is the input of block 1, the numbering of `analysis/features.py`. ViT-B/16 has 12 blocks and Swin-S 24. The **pooled feature** at a layer is the class token on ViT and the grid mean on Swin, the vector the head would read if the network ended there. The **head input** is what the final linear layer actually reads, the final LayerNorm applied before pooling.

The **attack success rate (ASR)** is the share of triggered images the model sends to the target class. It is measured on **eligible** images only (`attacks.poisoning.is_eval_poisonable`), whose true class is not the target. For TaCT it is measured only on its source class 1, since TaCT claims to flip that class alone.

The **backdoor direction** at a layer is the mean over pairs of the triggered pooled feature minus the clean one (`analysis.direction.backdoor_direction`). The **trigger-activated change (TAC)** of a coordinate is the mean absolute change of that coordinate over pairs (`analysis.direction.trigger_activated_change`), the quantity Zheng et al. (ECCV 2022) rank to find "backdoor neurons". A **neuron** here is 1 of the 768 coordinates of the residual stream (fewer at early Swin stages). The README of the experiment gives the formula of every derived quantity with a symbol table, and each step below restates the ones it uses.

<!-- panel -->
## Panel and controls

The panel takes 1 poisoning rate per attack and dataset, 5% wherever that model is a **successful** backdoor, the rule every result in the repository uses: the attack clears the ASR bar of 0.85 (`ASR_BAR`) and clean accuracy stays within 2 points of the benign model of the same architecture and dataset (`successful_2pt` in the coverage ledger, `successful_at_2_points` for Swin, which the ledger does not cover). A failed attack is left out even where that removes the attack from a dataset. No ViT WaNet model is successful on CIFAR-100 or GTSRB, and on CIFAR-10 only the 10% one is, so ViT has WaNet on CIFAR-10 and Tiny only. On Swin the 10% WaNet model is used where 5% misses the ASR bar. TaCT is taken only where it is trigger-conditional. On 8 of the 11 ViT TaCT models in the coverage ledger the clean source images are already sent to the target without any trigger, which makes a trigger reading meaningless (`docs/runs/2026-09-24-tact-multisource.md`). `vit_cifar10_tact_0_01`, `vit_cifar10_tact_0_05` and `vit_gtsrb_tact_0_05` are the 3 genuine ones. The Swin TaCT model is not in the ledger and is screened by measurement: if its paired clean accuracy, which for TaCT is the source-class accuracy, is below 0.5, `summary.json` marks it `source_mapped` and every mechanism figure leaves it out.

SIG, the only frequency-category attack in scope, is absent. The SIG trigger code builds a sine amplitude of 0.157 since 2026-09-09, while the ViT SIG models were trained at 0.1 and their `args.json` records no override, so a rebuilt SIG trigger is a different trigger from the one they learned (`docs/audits/2026-09-29-experiment-audit.md`). A measurement on it would describe the model's response to an unfamiliar input. The frequency category waits for the sidecar fix.

Each run uses up to 800 eligible pairs, drawn with a fixed seed from the PSBD analysis split (`data.splits.build_psbd_loaders_from_checkpoint`, the split every detection number in the repository uses), and up to 200 clean images of the target class. The table lists every backdoored run with its ASR, its clean accuracy on the paired clean images and the share of clean images it sends to the target, each measured by the experiment in bfloat16 on all pairs.

<!-- panel_after -->
The benign controls come next. Their ASR is the share of triggered images a model that never saw the trigger sends to class 0, the target of every backdoored cell. A value near the class prior (0.1 on CIFAR-10, 0.01 on CIFAR-100) means the trigger alone does not push toward the target. A higher value flags a trigger that is itself a strong input change, which matters when reading that trigger's control curves.

<!-- step1 -->
## Step 1, the triggers in pixel space

Before looking inside a model, the figure shows what each trigger does to an image, since everything later is a response to this change. The 6 attacks fall in 4 categories. **BadNets** stamps a 3 by 3 checkerboard in the bottom-right corner and relabels the image to the target. **TaCT** stamps the same patch but poisons only its source class 1, and stamps it on other classes without relabeling (cover samples) so the model has to learn "patch on class 1" rather than "patch". **Blend** mixes a fixed random pattern into the whole image at weight 0.2. **LF** adds a fixed pattern whose energy sits at low spatial frequencies. **WaNet** moves pixels along a fixed smooth warp field, so no pixel value is added. **BPP** quantizes each color channel to 3 bits.

The figure has 1 row per attack, taken from the first dataset in the order Tiny ImageNet, CIFAR-100, GTSRB, CIFAR-10 that has a usable ViT model, because Tiny's 64 by 64 images show the trigger best. The columns are 1 clean test image, the same image triggered, their absolute difference averaged over the 3 color channels, the same difference averaged over all pairs of the run (`input_change_map`), and that mean upsampled to 224 and averaged over each 16 by 16 patch, which is the change as ViT's 14 by 14 tokenizer receives it (`input_change_grid`). The color bars are in pixel units from 0 to 1.

<!-- step1_after -->
The difference maps separate the categories before any model is involved. The patch triggers change a few pixels in 1 corner, which on the patch grid lands in 1 to 4 tokens. Blend, LF and BPP change nearly every pixel by a small amount, so every token carries some of the trigger. WaNet's change follows the image's own edges, since moving a pixel changes nothing where the image is flat, so its footprint is different for every image and its mean is diffuse.

The figure shows what the network receives. It does not show what the network reads: a trigger spread over every token may be read from a few, and a trigger in 1 corner has to be carried from its tokens to the class token before it can change the prediction. That is the next question.

<!-- step2 -->
## Step 2, where the network reads the trigger

The per-token change map (`token_change_maps`) measures how far each token of the residual stream moves when the trigger is added, at every layer, relative to the typical size of a clean token at that layer.

$$m_l(t) = \frac{\frac{1}{N}\sum_{i=1}^{N} \lVert h_l(\tilde x_i)_t - h_l(x_i)_t \rVert_2}{\frac{1}{N T}\sum_{i=1}^{N}\sum_{t'=1}^{T} \lVert h_l(x_i)_{t'} \rVert_2}$$

| symbol | meaning |
| --- | --- |
| $x_i$, $\tilde x_i$ | clean image i and the same image with the trigger |
| $h_l(\cdot)_t$ | token t of the residual stream after block l |
| $N$ | pairs in the run, up to 800 |
| $T$ | tokens at layer l |

The denominator matters because the residual stream grows with depth, so a raw change would always look largest in the last block. The figure has 1 row per attack and 5 panels: the ViT patch grid after blocks 1, 4, 8 and 12, and the benign control's grid after block 12 with the same trigger. Each panel has its own color bar, since the question is where the change sits, not how it compares across panels. A bright token moved far relative to a typical clean token.

<!-- step2_vit_after -->
For a patch trigger the change starts in the trigger's tokens after block 1 and spreads across the grid in later blocks, as attention mixes the trigger's content into other tokens. For a global trigger it is spread from the start. In the benign control the last-layer map is much flatter, since a model that never learned the trigger has no reason to propagate it.

Swin-S repeats the measurement on its own grids. The panels are the last block of stage 1 (block 2, 56 by 56), stage 2 (block 4, 28 by 28), stage 3 (block 22, 14 by 14) and stage 4 (block 24, 7 by 7), with the benign control at block 24. Swin attends inside 7 by 7 windows that shift every other block, so a corner trigger can only spread 1 window per block pair, and patch merging between stages halves the grid.

<!-- step2_cls -->
The class token is where ViT's prediction is read, so its change is reported apart from the grid. The table gives the class token's relative change at blocks 4, 8 and 12 beside the mean and the largest change over the 196 patch tokens, for the same featured ViT runs. A class token whose change overtakes the patch mean has gathered the trigger from the patches.

<!-- step2_dims -->
The token maps above sum over all 768 coordinates of a token. `visualization.cheap_tools.run_token_maps` draws a finer statistic that the next figures reuse: for 1 image, the value on the patch grid of the 4 coordinates with the largest TAC at a layer, clean in the top row and triggered in the bottom row. Here the 4 coordinates are ranked on the fit half and the image is the first image of the eval half (`top_dimension_token_maps`). Color is the coordinate's raw value, red positive and blue negative, on a symmetric scale per coordinate.

<!-- step2_dims_after -->
The figure shows 1 image per model and is an illustration, not a measurement: where a top coordinate changes on this image need not hold on others. What the figure does not settle is whether these coordinates matter to the prediction. A coordinate can move a lot and be ignored by the head, which is step 4's question.

<!-- step2_attention -->
Attention is how the class token reads patches. The readout recomputes each ViT block's attention weights from the block's LayerNorm output, since torchvision computes them without returning them (`class_token_attention_hook`), and keeps the class token's row. The **trigger tokens** are the patch tokens whose patch-grid input change is at least 0.25 of the largest one (`TRIGGER_TOKEN_SHARE`), which picks 1 to 4 corner tokens for a patch trigger. The mass on trigger tokens in block l, head h is

$$M_{l,h} = \frac{1}{N}\sum_{i=1}^{N}\sum_{t \in S} a_{l,h}(x_i)_{0,t}$$

| symbol | meaning |
| --- | --- |
| $a_{l,h}(x)_{0,t}$ | attention weight from the class token (query 0) to token t in block l, head h, on image x |
| $S$ | the trigger tokens |
| uniform mass | $\lvert S \rvert / 197$, what attention spread evenly would give |

The figures plot $M_{l,h}$ averaged over the 12 heads against the block, for the triggered images (orange), the same images clean (blue), the benign model on the triggered images (black dashes) and uniform attention (grey dots). Only patch triggers are drawn, because for a global trigger S covers most of the grid and the mass is near 1 by construction. Swin has no class token and its attention stays inside windows, so this readout is ViT only.

<!-- step2_attention_heads -->
The heatmap splits the same quantity by head for the featured BadNets model. Each cell is $\log_2$ of triggered over clean mass on the trigger tokens, red where a head looks at the corner more when the trigger is there. The 2 grids on the right are the class token's head-mean attention over the 14 by 14 patch grid in the last block, clean and triggered, on a shared scale.

<!-- step2_attention_table -->
The table gives, for every usable ViT run, the head-mean mass on trigger tokens averaged over blocks 9 to 12, clean and triggered, the benign model's mass on the same triggered images and the ratio of triggered to clean. For global triggers the trigger set covers most tokens, so the ratio is near 1 whatever the model does and the row says nothing about routing.

<!-- step3 -->
## Step 3, how the backdoor grows with depth

Step 2 showed where the change sits. This step asks when the representation the head will read, the pooled feature, separates triggered images from clean ones. 3 quantities are computed at every layer by `layer_row` on all pairs of a run, as descriptive statistics that score nothing.

| quantity | formula | reading |
| --- | --- | --- |
| relative direction norm | $\lVert r_l \rVert \,/\, \frac{1}{N}\sum_i \lVert \bar h_l(x_i) \rVert$ with $r_l = \frac{1}{N}\sum_i \big(\bar h_l(\tilde x_i) - \bar h_l(x_i)\big)$ | how far the mean triggered feature sits from the mean clean feature, in units of the clean feature norm |
| normalized max TAC | $\max_k \mathrm{TAC}_{l,k} \,/\, \frac{1}{N D}\sum_{i,k} \lvert \bar h_l(x_i)_k \rvert$ with $\mathrm{TAC}_{l,k} = \frac{1}{N}\sum_i \lvert \bar h_l(\tilde x_i)_k - \bar h_l(x_i)_k \rvert$ | the largest single-coordinate change against a typical clean activation |
| CKA | debiased linear CKA between the clean and the triggered feature sets (`analysis.cka.debiased_linear_cka`) | 1 when the 2 sets are the same representation up to rotation and scale, lower when the trigger reorganizes it |

Here $\bar h_l$ is the pooled feature, D its width and N the number of pairs. The figure has 3 panels per architecture, 1 per quantity. Each colored line is 1 attack averaged over the datasets it has usable models on (the count is in the legend), and the black dashed line is the mean over all the benign controls of those models. The relative direction norm is drawn on a log scale because it spans 2 orders of magnitude.

<!-- step3_after -->
The table gives, per architecture and attack, the **onset layer** (the first layer whose relative direction norm reaches half its largest value), the **peak layer** (where it is largest), the onset as a share of the depth so ViT and Swin compare, and the final-layer direction norm and CKA beside the benign control's. Swin-S has 24 blocks and ViT-B/16 12, so an onset share of 0.75 is block 9 on ViT and block 18 on Swin.

<!-- step4 -->
## Step 4, backdoor neurons against a backdoor direction

The literature on backdoor neurons ranks coordinates by TAC and prunes the top ones (Zheng et al., ECCV 2022, on ConvNet channels). If the backdoor in a transformer lived in a few coordinates, 3 things would hold. The top coordinates would be the same across images and stable across halves of the data. They would separate clean from triggered images as well as anything else. Deleting them would remove the backdoor while deleting random coordinates would not. This step tests each, and tests the same 3 things for the backdoor direction, which is 1 unit vector in the same space, a "rotated neuron".

The heatmap shows, for 3 featured ViT runs, TAC at every layer for the 20 coordinates with the largest TAC at the last layer, divided by the layer's mean absolute clean activation. Rows are layers, columns are coordinate indices.

<!-- step4_heatmap_after -->
The final top coordinates carry almost nothing until the last few blocks. That they are late readout coordinates rather than units that carry the trigger through the network is a hypothesis this figure suggests and does not test, since nothing here removes them at an earlier layer.

The next figure asks whether the same coordinates recur across attacks on 1 model family (1 architecture on 1 dataset). All models of a family are fine-tuned from the same ImageNet weights, so a coordinate index names the same unit at initialization, which is the only reason a shared index can mean anything. Each cell is the Jaccard index, $\lvert A \cap B \rvert / \lvert A \cup B \rvert$, of the final-layer top-20 TAC sets of 2 runs, backdoored runs and benign controls together. 2 references frame it. **Chance** is the expected Jaccard of 2 independent random 20-of-768 draws, computed exactly by `chance_jaccard`. The **split-half ceiling** is the Jaccard between the top-20 sets ranked on the fit half and on the eval half of the same run, which is what a perfectly reproducible ranking would reach on 400 pairs.

<!-- step4_histograms -->
Separability is the next test. For 3 featured runs the histograms show, on the eval half, the activation of the 3 top-TAC coordinates at the last layer (ranked on the fit half) for clean and triggered images, and the projection of the same images onto the backdoor direction fitted on the fit half. The title gives 3 AUROCs, each the probability that a random triggered image scores above a random clean one after orienting each coordinate by the sign of its fit-half change: the top-TAC coordinate, the single coordinate with the best fit-half AUROC and the direction (`neuron_readout`).

<!-- step4_scatter -->
The scatter repeats the comparison on every usable run. The left panel plots the direction's eval AUROC against the best single coordinate's. The right panel describes the direction's shape: its **participation ratio** $1 / \sum_k u_k^4$ for the unit direction u, which is 1 for a direction along a single coordinate and about $D/3$ (256 for 768 dimensions) for a random direction, against the share of its squared length that sits in the 20 top-TAC coordinates, $\sum_{k \in \text{top 20}} u_k^2$. The table under it averages the same numbers per attack and adds the **direction share** $\lVert r \rVert^2 / \frac{1}{N}\sum_i \lVert d_i \rVert^2$, which is 1 when every image moves by the same vector and near 0 when the per-image moves cancel.

<!-- step4_ablation -->
Separability is correlational. The causal test deletes a candidate from the residual stream at the last block's output on every token and measures ASR and target-class recall on the eval half (`ablation_readout`). Removing 1 unit direction u sets every token x to $x - (x \cdot u)\,u$. Zeroing k coordinates is the same operation for k axis-aligned directions, so the comparison asks which basis the backdoor lives in (`tests/test_backdoor_manifestation.py` checks that the 2 hooks agree when u is an axis).

A removal that breaks the backdoor proves little unless a comparable removal does not. Every direction removal has 6 controls, all fitted on the fit half.

| control | what it rules out |
| --- | --- |
| isotropic random direction, 3 draws | removing any direction breaks the model |
| clean PC 1, the top principal direction of the clean features | removing the highest-variance direction breaks the model |
| variance-matched clean PC, the clean principal direction whose clean variance is closest to the backdoor direction's | the effect comes from the amount of clean energy removed |
| target-class offset, mean target-class feature minus mean clean feature | the backdoor is erased only because the target class's own evidence is erased |
| other trigger's direction on the same model (Blend for patch and warp models, BadNets for the others) | any trigger's direction would do |
| benign model's direction for the same trigger | the direction is the input change's footprint rather than a learned backdoor |

The bars give ASR (top row) and target-class recall (bottom row) per edit, 1 bar per attack averaged over datasets, ViT on the left and Swin on the right. Target-class recall is read on the eval-half clean images of the target class, which the eligible pairs exclude. It is what an edit that erases the target class instead of the backdoor would destroy.

<!-- step4_neurons -->
The same test on neurons zeroes k coordinates for k in 20, 100 and 300 of 768, chosen 4 ways: by TAC (the backdoor-neuron rule), by the size of the direction's own coordinates (the best a coordinate rule could do if it knew the direction), by mean clean magnitude (ViT carries a few massive activation coordinates, and a top-neuron rule could be picking those) and at random. The random and magnitude rules at every k are the controls.

<!-- step4_overlap -->
The table checks 2 of the controls' premises. The first 3 columns give the share of the top-k TAC coordinates that are also top-k clean-magnitude coordinates. The next 3 give the clean variance along the backdoor direction, along its variance-matched PC and along clean PC 1, which shows that the energy match is close and that PC 1 carries far more. The last 3 give the cosine between the backdoor direction and the target-class offset, the benign model's direction and the other trigger's direction. A cosine near 0 means the control is a different direction, not a noisy copy.

<!-- step4_per_class -->
Target-class recall is 1 class. Per-class recall over all classes present in the eval half (the eligible pairs and the target images) shows whether an edit costs every class a little or 1 class a lot. The table gives, per architecture and edit, the change in target-class recall, the mean change over the other classes and the worst single-class change, each averaged over runs.

<!-- step4_reading -->
The controls change how the ablation reads. Removing the backdoor direction at the last block does take ASR to near 0 while every energy-matched, random, other-trigger and benign-direction control leaves it untouched, so the direction is necessary and no generic high-energy direction is. The same edit also removes about half of the target class's own recall, and the backdoor direction with its target-class part projected out leaves ASR where it was. At the last block the backdoor direction is therefore mostly a push along the target class's readout axis, and deleting it deletes part of the target class with it. The rank-1 removal is necessary for the backdoor but not specific to it, which is what the audit suspected of the earlier CIFAR-10 result.

The neuron removals give the complementary answer. Zeroing 300 of 768 top-TAC coordinates leaves most of the backdoor in place on both architectures, exactly as 300 random or 300 largest-magnitude coordinates do. What no coordinate rule removes, 1 direction does, so in the last block the backdoor is a direction in a rotated basis rather than a set of neurons. That this direction coincides with the target's readout axis is the measured qualification.

<!-- step4_middle -->
Removing the direction at the last block leaves no block to repair it. At 2/3 of the depth (block 8 on ViT, block 16 on Swin) the later blocks can rewrite what was removed, so this repeat asks whether the backdoor is a single late readout or is rebuilt from earlier layers. On Swin block 16 sits in stage 3, whose stream has 384 coordinates, so zeroing 300 of them removes most of the stream and the neuron rows at k=300 there measure a broken model, as their clean accuracy shows.

<!-- step4_steering -->
Removal tests necessity. Steering tests sufficiency: the backdoor direction is added to clean eval images at the same block and the table gives the share now predicted as the target. Every other added direction is rescaled to the backdoor direction's norm, so only the orientation differs between columns. On benign controls, whose runs are included in the pooled rows only as the "benign model's direction" column of their backdoored twin, steering with the benign model's own direction is a separate reading in the benign runs' JSON.

<!-- step5 -->
## Step 5, latent geometry at the head

The head input is the space the final linear layer reads. The scatter projects the eval half of 3 populations there: clean eligible images (blue), the same images triggered (orange) and clean images of the target class (green). PCA and UMAP are fitted on the fit half of the same 3 populations and applied to the eval half (`embedding_record`), so the picture cannot show structure the projection found in its own points. UMAP uses 15 neighbors and a minimum distance of 0.1, and like any UMAP it keeps neighborhoods and distorts distances, so only which points share a cluster is readable. The last panel of each row is the benign control of the first model, shown the same trigger.

<!-- step5_readout -->
If the backdoor direction is how the model sends triggered images to the target, it should point along the target's row of the classifier weight. `readout_alignment` centers the weight rows over classes, $\tilde w_c = w_c - \frac{1}{C}\sum_{c'} w_{c'}$, because adding 1 vector to every row shifts every logit equally and changes no prediction. It then reports the cosine between the head-input backdoor direction and the target's centered row, the largest cosine with any other row and the rank of the target among the logit shifts $\tilde W r$. A random direction's cosine has a standard deviation of about $1/\sqrt{768} = 0.036$.

<!-- step6 -->
## Step 6, differences by attack category and architecture

The table collects the readings of steps 3 to 5 per category, attack and architecture, each averaged over datasets. The columns say when the direction appears, how coherent it is, how well it and the best neuron separate and what removing it or 300 top neurons does.

<!-- limits -->
## Limits

Every run uses 1 poisoning rate per cell and seed 0 only, so a difference between 2 datasets of the same attack can be a difference between 2 training runs. The panel has no frequency-category model until the SIG sidecars are fixed, no Label-Consistent or Adaptive-Blend model and no all-to-all model. The forward passes run in bfloat16, so TAC and small direction norms carry a rounding floor, which is why every curve is read against a benign control at the same precision. Ablation zeroes a direction or a coordinate, which moves the activation off its clean distribution, and a mean ablation would cost clean accuracy differently. The benign-model direction is taken from a separately trained model whose coordinates share only their ImageNet initialization with the backdoored model's, so its cosine to the backdoor direction is a lower bound on how similar the 2 input responses are.

