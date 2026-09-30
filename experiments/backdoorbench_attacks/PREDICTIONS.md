# Predictions for the BackdoorBench attack families

These predictions were written on 2026-09-30 before any PSBD sweep of a BackdoorBench checkpoint was run or read. They follow from the evidence surplus account in `docs/evidence-surplus-theory.md` and from the trigger footprints `inventory.py` measured on the triggered test images against their clean twins (`results/_experiments/backdoorbench_attacks/inventory.json`, `models[].footprint`), which describe the input and not the detector. Each prediction is a hypothesis about 1 attack family and is judged per model on the models that pass the success bar (ASR at or above 0.85 and not collapsed). A model below the bar is reported and not judged.

## Scope

The user cut the scope twice on 2026-09-30 before any sweep ran. Only attack families the ViT panel has never met are tested, on a few models. The queue runs every 10% model first, TrojanNN on CIFAR-10, GTSRB and Tiny, SSBA on the same 3 datasets, Input-Aware on CIFAR-10 and GTSRB, LIRA on CIFAR-10 and GTSRB and Blind on CIFAR-10, and then TrojanNN at 5% on the same 3 datasets. Only the final method's 2 probes are swept, PSBD-TM (`before_attention_norm` with `token_mask`) and its partner `pre_residual_blocks_5_8`. The final method fuses them. PSBD-RD and every competitor detector are out of scope. All 4 LIRA folders under `backdoor_bench_checkpoints/` hold no `attack_result.pt`, so LIRA gets a prediction and no test.

## Operating definitions

PSBD-TM is read at the adaptive rate (the smallest rate whose clean-validation shift ratio reaches 0.8) with fractional PSU, as `cli.analyze` reads it. The final method is the plain minimum of the 2 probes' clean-validation percentiles and, as a second rule, the plain average of their fractional PSUs, both computed by `experiments/cache_readouts/fusion_rules.py`'s functions at each probe's adaptive rate. TPR is read at 1%, 5% and 10% FPR set on clean validation.

A family **passes** when every judged model has PSBD-TM AUROC of at least 0.9 and TPR at 5% FPR of at least 0.5. A family **fails** when every judged model has PSBD-TM AUROC below 0.75. Anything between is **partial**. The bars are this file's parameters, chosen before any sweep and never tuned afterwards.

## TrojanNN

TrojanNN stamps a fixed patch, an apple-shaped mask filled with a pattern optimized to drive chosen neurons of the classifier head, at the same place on every image. The inventory shows the same pixels changed on every sampled image, covering a larger share of the 14 by 14 token grid than a BadNets patch. In the account a fixed patch is a disjunction over many sufficient carriers, which token masking at the attention input reads through softmax renormalization, so the triggered critical rate sits far above the clean one, and more carrier tokens raise it further.

**Prediction T1.** TrojanNN passes at 10% and at 5% on every judged model. **Prediction T2.** The final method's minimum rule does not lower PSBD-TM's TPR at 5% FPR by more than 0.05 on any judged TrojanNN model, since the partner adds nothing a renormalizing patch needs and the minimum only spends budget.

## SSBA

SSBA adds a sample-specific, invisible residual produced by a steganographic encoder. The inventory shows it changing about half or more of each image's pixels at a small amplitude and spreading across most tokens. It changes a different pixel set on different images. That is diffuse evidence, the additive form of the account, where survival under removal depends on the magnitude of the triggered margin and not on replication. The account also flags a sample-specific trigger as breaking the assumption that the test trigger is the training trigger (assumption 5), which lowers the margin a triggered test image inherits.

**Prediction S1.** SSBA is partial or fails under PSBD-TM on every judged model, with lower TPR at 1% FPR than every judged TrojanNN model of the same dataset. **Prediction S2.** On SSBA the final method's minimum rule raises TPR at 5% FPR above PSBD-TM's on the majority of judged models, since residual dropout reads a coherent magnitude direction that token masking does not.

## Input-Aware

Input-Aware draws a trigger per image from a generator, with a sparse mask that covers a few percent of the pixels at positions that change from image to image, and trains a cross mode in which a trigger generated for another image must leave the label unchanged. The backdoor therefore fires only when the trigger matches the image it was made for. In the account that is a positive conjunction of trigger and content or a relation between them. The conjunction bound caps the triggered critical rate at the content's, which is near a clean image's.

**Prediction I1.** Input-Aware fails under PSBD-TM on every judged model. **Prediction I2.** The final method does not rescue it, meaning its minimum rule leaves TPR at 5% FPR below 0.5, since a conjunction with content caps survival along the residual axis as well once the content is read.

## Blind

Blind trains with a loss that mixes a clean task and a backdoor task instead of poisoning stored data, and its trigger is a fixed pattern of 5 by 3 pixels at a fixed position, a BadNets-like patch. BackdoorBench stored its triggered test images as JPEG, so every triggered image also carries compression artifacts its clean twin does not, which the inventory shows as small changes over most pixels. The account reads the patch as a disjunction over few carriers, the BadNets case, where PSBD-TM does well on the panel.

**Prediction B1.** Blind passes under PSBD-TM on the judged CIFAR-10 model. A failure would first have to be separated from the JPEG artifacts before it could count against the account.

## LIRA

LIRA learns an image-conditional additive perturbation bounded in the L-infinity norm at a small epsilon, a diffuse, sample-specific trigger like SSBA's but smaller. The account predicts the additive form with a small magnitude, so partial or failing detection under PSBD-TM. No checkpoint exists, so this prediction stays untested.

## Recipe

BackdoorBench's default recipe (`third_party/BackdoorBench/config/attack/prototype/`) trains with SGD, momentum, a cosine schedule, random-crop augmentation and 100 epochs, against the panel's 15 epochs of Adam at a constant rate with no augmentation. The checkpoints carry no training log, so that they followed the default is an assumption. The account's training section predicts that a triggered margin grows with the number of poisoned examples seen, so a longer recipe gives a fixed trigger more surplus and not less. **Prediction R1.** No patch family (TrojanNN, Blind) fails under PSBD-TM on a judged model. This is a weak test of the recipe, since BackdoorBench's versions of the panel's own attacks are out of scope.
