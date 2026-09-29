# Recomputes open questions Q25 (how matched the matched rule is) and Q27 (the realized
# false-positive range at the headline quantile) on the current 57-model ViT panel, the
# clearing cells of results/coverage/coverage.json that carry both headline placements.
import json
import os
import statistics

from defenses.decision import (
    interpolate_at_target_shift,
    select_rate_adaptively,
    select_rate_at_matched_shift,
)

TM, RD = "before_attention_norm_token_mask", "post_residual"
DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")

coverage = json.load(open("results/coverage/coverage.json"))
cells = [c for c in coverage["cells"] if c["dataset"] in DATASETS and c["asr_class"] == "clears"]


def ladder(placement_record):
    shift = {r["rate"]: r["shift_ratio"]["validation"] for r in placement_record["rates"]}
    auroc = {r["rate"]: r["detection_psu_ratio"]["q0.25"]["auroc"] for r in placement_record["rates"]}
    fpr = {r["rate"]: r["detection_psu_ratio"]["q0.25"]["fpr"] for r in placement_record["rates"]}
    return shift, auroc, fpr


rows = []
for cell in cells:
    path = f"results/{cell['folder_name']}/psbd_metrics.json"
    if not os.path.exists(path):
        continue
    placements = json.load(open(path))["placements"]
    if TM not in placements or RD not in placements:
        continue
    readings = {}
    for name in (TM, RD):
        shift, auroc, fpr = ladder(placements[name])
        adaptive = select_rate_adaptively(shift, 0.8)
        matched = select_rate_at_matched_shift(shift, 0.6)
        readings[name] = {
            "adaptive_auroc": auroc.get(adaptive),
            "adaptive_fpr": fpr.get(adaptive),
            "matched_auroc": auroc[matched],
            "matched_shift": shift[matched],
            "interpolated_auroc": interpolate_at_target_shift(shift, auroc, 0.6),
        }
    rows.append((cell["folder_name"], readings))

print("models", len(rows))
tm_adaptive = [r[TM]["adaptive_auroc"] for _, r in rows]
rd_adaptive = [r[RD]["adaptive_auroc"] for _, r in rows]
print("sanity TM adaptive", round(statistics.mean(tm_adaptive), 3),
      "RD adaptive", round(statistics.mean(rd_adaptive), 3))

fprs = [r[TM]["adaptive_fpr"] for _, r in rows]
print("Q27 TM adaptive q0.25 realized FPR mean %.3f min %.3f max %.3f"
      % (statistics.mean(fprs), min(fprs), max(fprs)))

tm_shift = [r[TM]["matched_shift"] for _, r in rows]
rd_shift = [r[RD]["matched_shift"] for _, r in rows]
off = sum(abs(s - 0.6) > 0.10 for s in rd_shift)
print("Q25 matched shift TM %.3f RD %.3f, RD off target by >0.10 on %d of %d"
      % (statistics.mean(tm_shift), statistics.mean(rd_shift), off, len(rows)))
nearest_gain = [r[TM]["matched_auroc"] - r[RD]["matched_auroc"] for _, r in rows]
both = [(r[TM]["interpolated_auroc"], r[RD]["interpolated_auroc"]) for _, r in rows]
paired = [(t - d, n) for (t, d), n in zip(both, nearest_gain) if t is not None and d is not None]
print("Q25 nearest-rate matched gain %.3f, interpolated gain %.3f over %d, nearest minus interpolated %.3f"
      % (statistics.mean(n for _, n in paired), statistics.mean(i for i, _ in paired), len(paired),
         statistics.mean(n - i for i, n in paired)))
