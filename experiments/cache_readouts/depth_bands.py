"""X4: which depth band breaks which trigger, from the cached band ladders.

For token masking at the attention input and residual dropout, each restricted to
blocks 1 to 4, 5 to 8 or 9 to 12, reads the share of passes whose label moved on
triggered inputs and on their paired clean images at the band's top cached rate.
The heat map is attack by band of triggered minus clean shift share.

    .venv/bin/python -m experiments.cache_readouts.depth_bands --set dev
"""

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from defenses.cache import read_split_manifest  # noqa: E402
from defenses.decision import (  # noqa: E402
    PLACEMENT_MATCH_TARGET,
    complete_rates,
    pair_clean_to_backdoor,
    select_rate_at_matched_shift,
)
from experiments.cache_readouts.shared import (  # noqa: E402
    FIGURES_DIR,
    GLOBAL_ATTACKS,
    MODEL_SETS,
    PATCH_ATTACKS,
    load_baselines,
    load_model_set,
    load_passes,
    model_label,
    ordered_attacks,
    output_path,
    psbd_dir_of,
    validation_shift_by_rate,
    write_json,
)
from scripts.paper._common import figure_sidecar  # noqa: E402

BANDS = ("1_4", "5_8", "9_12")
OPERATORS = {
    "token_mask": "before_attention_norm_blocks_{band}_token_mask",
    "residual_dropout": "pre_residual_blocks_{band}",
}

# The plan states E1, E2 and E10 in words. These are the thresholds that turn
# them into verdicts, fixed here before the first band was read.
#   E1a  token mask in blocks 1 to 4 and 5 to 8 leaves triggered patch inputs
#        alone (triggered share at most STABLE) and breaks clean ones (clean
#        share above triggered by at least BROKEN_MARGIN).
#   E1b  token mask in blocks 9 to 12 hits triggered patch inputs too
#        (triggered share at least HIT).
#   E2   no token-mask band moves a global trigger (triggered share at most
#        STABLE on Blend, LF and BPP).
#   E10  residual dropout in blocks 1 to 4 moves triggered WaNet more than
#        triggered BadNets, read on triggered minus clean, on every WaNet model
#        against the BadNets mean.
STABLE = 0.10
BROKEN_MARGIN = 0.20
HIT = 0.20

# The plan reads each band at its top cached rate. At that rate residual dropout
# saturates the clean split, so the same verdicts are also read at the rate
# whose clean-validation shift is nearest the canonical matched target, as a
# sensitivity reading that never replaces the first.
READINGS = ("top_rate", "matched")


def main():
    args = parse_args()
    started = time.perf_counter()

    models = load_model_set(args.set)
    rows = [measure_model(model) for model in models]
    summary = {reading: summarize(rows, reading) for reading in READINGS}
    verdicts = {reading: judge(rows, summary[reading], reading) for reading in READINGS}

    payload = {
        "experiment": "X4, depth bands",
        "model_set": args.set,
        "operators": OPERATORS,
        "bands": list(BANDS),
        "readings": list(READINGS),
        "matched_target": PLACEMENT_MATCH_TARGET,
        "thresholds": {"stable": STABLE, "broken_margin": BROKEN_MARGIN, "hit": HIT},
        "summary": summary,
        "verdicts": verdicts,
        "models": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    path = output_path("depth_bands", args.set)
    write_json(payload, path)
    plot(summary, args.set, path)
    print(f"wrote {path} in {payload['wall_seconds']:.0f} s")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=MODEL_SETS, required=True)
    args = parser.parse_args()
    return args


def measure_model(model):
    psbd_dir = psbd_dir_of(model)
    manifest = read_split_manifest(psbd_dir)
    baselines = load_baselines(psbd_dir)

    row = model_label(model)
    row["bands"] = {reading: {} for reading in READINGS}
    for operator, template in OPERATORS.items():
        for band in BANDS:
            placement = template.format(band=band)
            key = f"{operator}/{band}"
            if not complete_rates(psbd_dir, placement):
                for reading in READINGS:
                    row["bands"][reading][key] = None
                continue
            shift_by_rate = validation_shift_by_rate(psbd_dir, placement, baselines)
            rates = {
                "top_rate": max(shift_by_rate),
                "matched": select_rate_at_matched_shift(
                    shift_by_rate, PLACEMENT_MATCH_TARGET
                ),
            }
            for reading, rate in rates.items():
                passes = load_passes(psbd_dir, placement, rate, baselines)
                row["bands"][reading][key] = shift_shares(passes, manifest, rate)
    return row


def shift_shares(passes, manifest, rate):
    def per_sample_share(split):
        argmax = passes[split]["per_pass_argmax"].long()  # (passes, n)
        labels = passes[split]["baseline_labels"].long()  # (n,)
        share = (argmax != labels.view(1, -1)).float().mean(dim=0)  # (n,)
        return share

    triggered = per_sample_share("backdoor")  # (n_backdoor,)
    clean = pair_clean_to_backdoor(per_sample_share("clean"), manifest)  # (n_backdoor,)
    validation = per_sample_share("validation")  # (n_validation,)
    shares = {
        "rate": rate,
        "triggered": float(triggered.mean()),
        "clean_paired": float(clean.mean()),
        "validation": float(validation.mean()),
        "triggered_minus_clean": float(triggered.mean() - clean.mean()),
    }
    return shares


def summarize(rows, reading):
    pooled = [row for row in rows if row["successful_2pt"]]
    attacks = ordered_attacks({row["attack"] for row in pooled})
    summary = {
        "pooled_folders": [row["folder"] for row in pooled],
        "unpooled_folders": [
            row["folder"] for row in rows if not row["successful_2pt"]
        ],
        "by_attack": {},
    }
    for attack in attacks:
        group = [row for row in pooled if row["attack"] == attack]
        summary["by_attack"][attack] = {"n": len(group)}
        for key in (f"{op}/{band}" for op in OPERATORS for band in BANDS):
            readings = [
                row["bands"][reading][key]
                for row in group
                if row["bands"][reading][key]
            ]
            summary["by_attack"][attack][key] = {
                "n": len(readings),
                "rates": sorted({r["rate"] for r in readings}),
                **{
                    field: sum(r[field] for r in readings) / len(readings)
                    if readings
                    else None
                    for field in ("triggered", "clean_paired", "triggered_minus_clean")
                },
            }
    return summary


def judge(rows, summary, reading):
    by_attack = summary["by_attack"]
    pooled = [row for row in rows if row["successful_2pt"]]
    verdicts = {}

    # E1a and E1b are judged on every patch model, not on the attack mean, so
    # 1 model that breaks the rule is visible.
    patch_rows = [row for row in pooled if row["attack"] in PATCH_ATTACKS]
    early_checks = [
        {
            "folder": row["folder"],
            "band": band,
            "triggered": row["bands"][reading][f"token_mask/{band}"]["triggered"],
            "clean_paired": row["bands"][reading][f"token_mask/{band}"]["clean_paired"],
            "holds": row["bands"][reading][f"token_mask/{band}"]["triggered"] <= STABLE
            and row["bands"][reading][f"token_mask/{band}"]["clean_paired"]
            - row["bands"][reading][f"token_mask/{band}"]["triggered"]
            >= BROKEN_MARGIN,
        }
        for row in patch_rows
        for band in ("1_4", "5_8")
        if row["bands"][reading][f"token_mask/{band}"]
    ]
    verdicts["E1a"] = verdict_from(early_checks)

    late_checks = [
        {
            "folder": row["folder"],
            "triggered": row["bands"][reading]["token_mask/9_12"]["triggered"],
            "clean_paired": row["bands"][reading]["token_mask/9_12"]["clean_paired"],
            "holds": row["bands"][reading]["token_mask/9_12"]["triggered"] >= HIT,
        }
        for row in patch_rows
        if row["bands"][reading]["token_mask/9_12"]
    ]
    verdicts["E1b"] = verdict_from(late_checks)

    global_checks = [
        {
            "folder": row["folder"],
            "band": band,
            "triggered": row["bands"][reading][f"token_mask/{band}"]["triggered"],
            "holds": row["bands"][reading][f"token_mask/{band}"]["triggered"] <= STABLE,
        }
        for row in pooled
        if row["attack"] in GLOBAL_ATTACKS
        for band in BANDS
        if row["bands"][reading][f"token_mask/{band}"]
    ]
    verdicts["E2"] = verdict_from(global_checks)

    # E10 needs both attacks in the pooled set. WaNet below the 2-point bar is
    # read too, since the prediction is about the trigger, and labeled.
    badnet = by_attack.get("badnet_a2o", {}).get("residual_dropout/1_4", {})
    badnet_mean = badnet.get("triggered_minus_clean")
    wanet_rows = [row for row in rows if row["attack"] == "wanet"]
    wanet_checks = [
        {
            "folder": row["folder"],
            "successful_2pt": row["successful_2pt"],
            "wanet_triggered_minus_clean": row["bands"][reading][
                "residual_dropout/1_4"
            ]["triggered_minus_clean"],
            "badnet_mean_triggered_minus_clean": badnet_mean,
            "holds": row["bands"][reading]["residual_dropout/1_4"][
                "triggered_minus_clean"
            ]
            > badnet_mean,
        }
        for row in wanet_rows
        if badnet_mean is not None and row["bands"][reading]["residual_dropout/1_4"]
    ]
    verdicts["E10"] = verdict_from(wanet_checks)
    return verdicts


def verdict_from(checks):
    if not checks:
        return {
            "verdict": "inconclusive",
            "reason": "no model carries the reading",
            "checks": [],
        }
    held = sum(check["holds"] for check in checks)
    if held == len(checks):
        label = "held"
    elif held == 0:
        label = "failed"
    else:
        label = "mixed"
    verdict = {
        "verdict": label,
        "n_held": held,
        "n_checks": len(checks),
        "checks": checks,
    }
    return verdict


def plot(summary, model_set, json_path):
    attacks = list(summary[READINGS[0]]["by_attack"])
    figure, axes = plt.subplots(
        len(READINGS),
        len(OPERATORS),
        figsize=(12, len(READINGS) * (0.45 * len(attacks) + 1.5)),
        squeeze=False,
    )
    plotted = {}
    image = None
    for row_index, reading in enumerate(READINGS):
        by_attack = summary[reading]["by_attack"]
        for column_index, operator in enumerate(OPERATORS):
            axis = axes[row_index][column_index]
            grid = [
                [
                    by_attack[a][f"{operator}/{band}"]["triggered_minus_clean"]
                    for band in BANDS
                ]
                for a in attacks
            ]  # (attacks, bands)
            plotted[f"{reading}/{operator}"] = {
                a: dict(zip(BANDS, values)) for a, values in zip(attacks, grid)
            }
            image = axis.imshow(
                [[float("nan") if v is None else v for v in r] for r in grid],
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                aspect="auto",
            )
            for i, values in enumerate(grid):
                for j, value in enumerate(values):
                    if value is not None:
                        axis.text(
                            j, i, f"{value:+.2f}", ha="center", va="center", fontsize=8
                        )
            axis.set_xticks(range(len(BANDS)))
            axis.set_xticklabels([f"blocks {b.replace('_', ' to ')}" for b in BANDS])
            axis.set_yticks(range(len(attacks)))
            # The 2 panels of a row share their attacks, so only the left one names them.
            row_labels = [f"{a} (n={by_attack[a]['n']})" for a in attacks]
            axis.set_yticklabels(row_labels if column_index == 0 else [])
            reading_words = (
                "top cached rate"
                if reading == "top_rate"
                else f"clean shift nearest {PLACEMENT_MATCH_TARGET}"
            )
            axis.set_title(f"{operator.replace('_', ' ')}, {reading_words}")
    figure.colorbar(
        image, ax=axes.ravel().tolist(), label="triggered minus clean shift share"
    )
    figure.suptitle(f"Depth bands, {model_set} set")

    figure_path = f"{FIGURES_DIR}/depth_bands_{model_set}.png"
    os.makedirs(FIGURES_DIR, exist_ok=True)
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)
    figure_sidecar(
        figure_path.replace(".png", ".json"),
        "experiments/cache_readouts/depth_bands.py",
        [json_path],
        plotted,
    )


if __name__ == "__main__":
    main()
