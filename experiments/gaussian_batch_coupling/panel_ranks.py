"""Mean AUROC and rank of the 2 gaussian placements on the current paper panel.

Every placement read on all panel models (`experiments.probe_union.measure.select_models`)
is ranked by mean fractional-PSU AUROC at q0.25, at the adaptive 0.8 rule and at the
matched 0.6 rung, read from each model's `psbd_metrics.json`.

    PYTHONPATH=. .venv/bin/python experiments/gaussian_batch_coupling/panel_ranks.py
"""

import statistics

from experiments.probe_union.measure import select_models

PLACEMENTS = (
    "before_mlp_gaussian",
    "before_attention_norm_gaussian",
    "before_attention_norm_token_mask",
)
RULES = ("adaptive", "0.6")


def main():
    models = select_models("results")
    shared = set.intersection(*[set(m["report"]["placements"]) for m in models])
    print(f"panel {len(models)} models")
    for rule in RULES:
        means = {}
        for placement in shared:
            values = [placement_auroc(m["report"], placement, rule) for m in models]
            if all(value is not None for value in values):
                means[placement] = statistics.mean(values)
        ranked = sorted(means, key=lambda p: -means[p])
        for placement in PLACEMENTS:
            rank = ranked.index(placement) + 1
            print(
                f"{rule:9s} {placement:36s} {means[placement]:.3f} "
                f"rank {rank} of {len(ranked)}"
            )


def placement_auroc(report, placement, rule):
    block = report["placements"][placement]
    if rule == "adaptive":
        rate = block.get("adaptive_rate")
    else:
        rate = (block.get("matched_shift", {}).get(f"sigma{rule}") or {}).get("rate")
    if rate is None:
        return None
    row = next(r for r in block["rates"] if r["rate"] == rate)
    auroc = row["detection_psu_ratio"]["q0.25"]["auroc"]
    return auroc


if __name__ == "__main__":
    main()
