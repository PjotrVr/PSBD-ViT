"""The numbers of the README's chain from mechanism to detection, on the current panel.

Every value is the fractional-PSU AUROC at q0.25 read from each panel model's
`psbd_metrics.json` (`experiments.probe_union.measure.select_models`), at the rung whose
clean-validation shift is matched to the target (`matched_shift.sigma<target>`), with
paired differences and a 5000-resample bootstrap interval at seed 0.

    PYTHONPATH=. .venv/bin/python experiments/residual_stream_mechanism/panel_chain.py
"""

import statistics

from experiments.probe_union.measure import select_models
from scripts.paper._common import bootstrap_ci

OPERATORS_AT_ATTENTION_INPUT = (
    "before_attention_norm_token_mask",
    "before_attention_norm_channel_mask",
    "before_attention_norm_gaussian",
)
TOKEN_MASK_POSITIONS = (
    "before_attention_norm_token_mask",
    "before_mlp_norm_token_mask",
    "before_attention_residual_token_mask",
    "after_attention_residual_token_mask",
    "both_sublayer_inputs_token_mask",
)
PAIRS = (
    ("before_attention_norm_token_mask", "before_attention_norm_gaussian"),
    ("before_mlp_norm_token_mask", "before_mlp_gaussian"),
)
TARGETS = ("0.8", "0.6")


def main():
    models = select_models("results")
    print(f"panel {len(models)} models")
    for target in TARGETS:
        print(f"matched {target}")
        for first, second in PAIRS:
            deltas = [
                auroc(m, first, target) - auroc(m, second, target) for m in models
            ]
            low, high = bootstrap_ci(deltas, 5000, 0)
            print(
                f"  {first} minus {second}: {statistics.mean(deltas):+.3f} "
                f"[{low:+.3f}, {high:+.3f}]"
            )
        for placement in ("before_attention_norm_gaussian", "before_mlp_gaussian"):
            values = [auroc(m, placement, target) for m in models]
            inverted = sum(value < 0.5 for value in values)
            print(
                f"  {placement}: {statistics.mean(values):.3f}, "
                f"{inverted} of {len(values)} inverted"
            )
        operator_means = [
            mean_auroc(models, p, target) for p in OPERATORS_AT_ATTENTION_INPUT
        ]
        position_means = [mean_auroc(models, p, target) for p in TOKEN_MASK_POSITIONS]
        with_noise = max(operator_means) - min(operator_means)
        masks_only = max(operator_means[:2]) - min(operator_means[:2])
        position = max(position_means) - min(position_means)
        print(
            f"  operator range {with_noise:.3f} with gaussian, {masks_only:.3f} without, "
            f"position range {position:.3f}, ratio {position / with_noise:.2f}x and "
            f"{position / masks_only:.2f}x"
        )


def auroc(model, placement, target):
    block = model["report"]["placements"][placement]
    rate = block["matched_shift"][f"sigma{target}"]["rate"]
    row = next(r for r in block["rates"] if r["rate"] == rate)
    value = row["detection_psu_ratio"]["q0.25"]["auroc"]
    return value


def mean_auroc(models, placement, target):
    value = statistics.mean(auroc(m, placement, target) for m in models)
    return value


if __name__ == "__main__":
    main()
