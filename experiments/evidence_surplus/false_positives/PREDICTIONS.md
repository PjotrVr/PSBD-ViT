# C-fp. Are PSBD's false positives the clean images with their own evidence surplus?

Written 2026-09-30 before any reading of this test.

PSBD-TM flags a clean held-out image as poisoned when its fractional PSU falls below the threshold. The question is whether those false positives are clean images that carry more evidence for their own class than they need. PSU is itself a surplus reading, so "false positives have a high PSU surplus" would be a tautology. The surplus is therefore read by measures that do not come from the probe that flagged the image:

- (a) the smallest sufficient token share: the smallest visible share of tokens, removed in a nested order (ranked by attribution, and random) at the attention input of every block with no PSBD rate, that keeps the image's answer
- (b) the image's own-class component: the projection of its final feature, minus the mean clean feature, on the head's direction for its own class (this experiment's version, not the manifestation agent's surplus factor)
- (c) survival under the other operator: PSBD-TM's false positives read by PSBD-RD's cached statistic, and the reverse
- (d) the unperturbed logit margin, only as a confidence control

Population: the 2000 clean held-out images (the split the threshold is read from) of the 10 ViT models of the evidence-surplus set and the 2 benign ViT models. False positives are the images below the 0.01 and 0.05 quantiles of that split's own cached PSBD-TM statistic.

Predictions:

1. False positives have a significantly smaller sufficient token share (a, ranked and random) than the rest of the clean held-out images, at both quantiles, on most models: the median difference is negative and its bootstrap 95% interval excludes 0.
2. False positives have a higher own-class component (b) and a higher survival under the other operator (c) than the rest, with intervals excluding 0.
3. Beyond margin: each false positive compared with the non-flagged image of the nearest unperturbed margin (d) still has a smaller sufficient share and a higher own-class component, intervals excluding 0.
4. Per class: the target class is overrepresented among false positives on backdoored models (as in P10) and not on the benign models.

If a prediction fails, the verdict stays as registered and 1 targeted measurement diagnoses it beside the verdict.
