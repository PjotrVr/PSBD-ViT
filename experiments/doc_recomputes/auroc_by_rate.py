# PSBD-TM and PSBD-RD mean AUROC at the adaptive 0.8 rule by poison rate, and by attack
# and rate, over the 57-model panel, for H8 (does detection improve with poison rate).
import collections
import json
import os
import statistics

from defenses.decision import select_rate_adaptively

TM, RD = "before_attention_norm_token_mask", "post_residual"
DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")
coverage = json.load(open("results/coverage/coverage.json"))
by_rate = collections.defaultdict(lambda: collections.defaultdict(list))
by_attack = collections.defaultdict(list)
for cell in coverage["cells"]:
    if cell["dataset"] not in DATASETS or not cell["successful_2pt"]:
        continue
    path = f"results/{cell['folder_name']}/psbd_metrics.json"
    if not os.path.exists(path):
        continue
    placements = json.load(open(path))["placements"]
    if TM not in placements or RD not in placements:
        continue
    for name in (TM, RD):
        rates = placements[name]["rates"]
        shift = {r["rate"]: r["shift_ratio"]["validation"] for r in rates}
        auroc = {r["rate"]: r["detection_psu_ratio"]["q0.25"]["auroc"] for r in rates}
        value = auroc[select_rate_adaptively(shift, 0.8)]
        by_rate[cell["poison_rate"]][name].append(value)
        if name == TM:
            by_attack[(cell["attack"], cell["poison_rate"])].append(value)
for rate in sorted(by_rate):
    print(
        rate,
        {
            name.split("_")[0]: (round(statistics.mean(v), 3), len(v))
            for name, v in by_rate[rate].items()
        },
    )
for key in sorted(by_attack):
    print(key, round(statistics.mean(by_attack[key]), 3), len(by_attack[key]))
