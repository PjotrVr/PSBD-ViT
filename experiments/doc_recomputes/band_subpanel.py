# The all-blocks PSBD-TM reading on the models where the blocks 1 to 4 band reaches the
# adaptive target, for the depth-band warning in docs/perturbations.md and Q26.
import json
import os
import statistics

from defenses.decision import select_rate_adaptively

ALL, BAND = (
    "before_attention_norm_token_mask",
    "before_attention_norm_blocks_1_4_token_mask",
)
DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")
coverage = json.load(open("results/coverage/coverage.json"))


def adaptive_auroc(record):
    shift = {r["rate"]: r["shift_ratio"]["validation"] for r in record["rates"]}
    auroc = {
        r["rate"]: r["detection_psu_ratio"]["q0.25"]["auroc"] for r in record["rates"]
    }
    rate = select_rate_adaptively(shift, 0.8)
    return auroc.get(rate)


panel, pairs = [], []
for cell in coverage["cells"]:
    if cell["dataset"] not in DATASETS or not cell["successful_2pt"]:
        continue
    path = f"results/{cell['folder_name']}/psbd_metrics.json"
    if not os.path.exists(path):
        continue
    placements = json.load(open(path))["placements"]
    if ALL not in placements or "post_residual" not in placements:
        continue
    all_blocks = adaptive_auroc(placements[ALL])
    panel.append(all_blocks)
    band = adaptive_auroc(placements[BAND]) if BAND in placements else None
    if band is not None:
        pairs.append((all_blocks, band))
print("panel", len(panel), round(statistics.mean(panel), 3))
print(
    "band models",
    len(pairs),
    "all-blocks there",
    round(statistics.mean(a for a, _ in pairs), 3),
    "band",
    round(statistics.mean(b for _, b in pairs), 3),
    "paired",
    round(statistics.mean(b - a for a, b in pairs), 3),
)
