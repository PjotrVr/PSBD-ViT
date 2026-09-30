# Clean surplus without a backdoor, pre-registered predictions

Written 2026-09-30 before any collage or natural-surplus reading was made.

If PSBD reads evidence surplus and not the backdoor as such, a clean input given more evidence for its own class than it needs should look poisoned to it, on a benign model too.

1. A 2x2 collage of 4 test images of the same class reads as more stable than a single image. On the 2 benign models PSBD-TM flags such collages well above the nominal rate at each of the 0.01, 0.05 and 0.10 quantiles of the clean-validation statistic (at least twice the nominal rate).
2. A 2x2 collage of 4 images of different classes reads as less stable than a single image and is flagged at or below the nominal rate.
3. The same ordering holds on the 2 backdoored models.
4. The natural version, from this repository's records, no GPU: the clean images whose answer never flips over PSBD-TM's passes have a higher margin retention than the clean images that flip, on every model measured.

## Addendum, 2026-09-30, before any collage reading: the resolution control

A 2x2 collage shows each image at half the resolution a single image gets, since the model resizes every input to 224. That lowers the evidence per quadrant for a reason unrelated to surplus. 5 inputs are therefore compared, all at the same input size (twice the dataset's resolution):

- (i) same class: 4 different images of 1 class
- (ii) duplicated: 1 image repeated in all 4 quadrants, pure duplication of its evidence
- (iii) resolution control: 1 image in 1 quadrant, the other 3 filled with the dataset's mean color
- (iv) different classes: 4 images of 4 classes, conflicting evidence
- (v) single: 1 image filling the whole input

Predictions, replacing 1 and 2 above as the primary test (1 to 4 are kept and scored as written):

5. Surplus, read as the per-input critical rate p* along PSBD-TM's rate ladder and as fractional PSU, orders (ii) about equal to (i), both above (iii) and (v), and all of these above (iv).
6. On the benign models PSBD-TM flags (i) and (ii) above the nominal rate relative to (iii), the input with the same per-image resolution, at every quantile. The comparison with (v) is not the test, since (v) has twice the per-image resolution.

If a prediction fails, the verdict stays as registered and 1 targeted measurement diagnoses it beside the verdict.
