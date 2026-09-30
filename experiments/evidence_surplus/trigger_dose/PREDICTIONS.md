# Trigger dose-response, pre-registered predictions

Written 2026-09-30 before any dose-response reading was made.

The evidence surplus account says a triggered decision survives PSBD's perturbation because the trigger supplies far more evidence for the target than the decision needs. Fading the trigger at test time lowers that surplus while leaving everything else fixed: the model, the clean-validation threshold at the 0.01, 0.05 and 0.10 quantiles and the adaptive rate read from the canonical sweep. Only the trigger changes.

1. As the trigger fades, PSBD's AUROC (the triggered image against its clean twin) and its TPR at every quantile fall toward chance.
2. Detection reaches chance at a larger dose than ASR does: there is a band of doses where the faded trigger still sends most images to the target (ASR at least 0.5) and PSBD's AUROC is within 0.1 of 0.5. Read on the images the faded trigger still sends to the target, the AUROC in that band is also within 0.1 of 0.5.
3. The benign controls read chance at every dose.

The fade is linear in pixel space between the clean image and its triggered copy for Blend, LF, BPP, BadNets and TaCT (blend weight, amplitude, quantization difference and patch opacity), and the warp strength for WaNet. BadNets and TaCT also get a partial-patch series with 1, 2, 3 and 4 of the patch's tokens carrying the trigger.
