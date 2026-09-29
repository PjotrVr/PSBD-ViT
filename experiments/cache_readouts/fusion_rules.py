"""X3: which rule should fuse PSBD-TM with a residual-dropout partner.

Each partner (late band, middle band, PSBD-RD) is paired with PSBD-TM and the 2
fractional PSUs are fused 5 ways. Every fused score is thresholded at a quantile
of its own clean-validation distribution, so the FPR budget is spent on clean
data only, and TPR and the realized FPR are read on the paired test splits.

    .venv/bin/python -m experiments.cache_readouts.fusion_rules --set dev
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from defenses.cache import read_split_manifest  # noqa: E402
from defenses.decision import (  # noqa: E402
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
    pair_clean_to_backdoor,
)
from defenses.scores import psu_ratio_from_cache, to_rank  # noqa: E402
from experiments.cache_readouts.shared import (  # noqa: E402
    FIGURES_DIR,
    MODEL_SETS,
    choose_rate,
    evaluate_scores,
    load_baselines,
    load_model_set,
    load_passes,
    model_label,
    ordered_attacks,
    output_path,
    paired_summary,
    psbd_dir_of,
    validation_shift_by_rate,
    write_json,
)
from scripts.paper._common import OKABE_ITO, figure_sidecar, save_figure  # noqa: E402

PSBD_TM = RECOMMENDED_PLACEMENT
PARTNERS = {
    "late_band": "pre_residual_blocks_9_12",
    "middle_band": "pre_residual_blocks_5_8",
    "psbd_rd": PUBLISHED_PLACEMENT,
}
RULES = (
    "tm_alone",
    "mean_psu",
    "min_rank",
    "weighted_0.8_0.2",
    "weighted_0.9_0.1",
    "fisher",
)
REFERENCE_RULE = "tm_alone"
# Budget shares of PSBD-TM in the weighted minimum, the partner gets the rest.
WEIGHTED_SHARES = {"weighted_0.8_0.2": 0.8, "weighted_0.9_0.1": 0.9}
QUANTILES = (0.01, 0.10, 0.20)
REPORTED_QUANTILES = ("q0.01", "q0.10", "q0.20")
# "adaptive" is the paper's rule and the primary reading. "nearest" reads a
# partner whose ladder never reaches the target at its closest rate, so the
# models the adaptive rule loses are still reported under their own label.
RATE_RULES = ("adaptive", "nearest")


def main():
    args = parse_args()
    started = time.perf_counter()

    models = load_model_set(args.set)
    rows = [measure_model(model) for model in models]
    summary = {rule: summarize(rows, rule) for rule in RATE_RULES}

    payload = {
        "experiment": "X3, fusion rules",
        "model_set": args.set,
        "anchor": PSBD_TM,
        "partners": PARTNERS,
        "rules": list(RULES),
        "weighted_shares": WEIGHTED_SHARES,
        "quantiles": list(QUANTILES),
        "rate_rules": list(RATE_RULES),
        "summary": summary,
        "models": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    path = output_path("fusion_rules", args.set)
    write_json(payload, path)
    plot(summary["adaptive"], args.set, path)
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

    tm_rate = choose_rate(
        validation_shift_by_rate(psbd_dir, PSBD_TM, baselines), "adaptive"
    )
    row = model_label(model)
    row["tm_rate"] = tm_rate
    row["partners"] = {rate_rule: {} for rate_rule in RATE_RULES}
    if tm_rate is None:
        return row
    tm_psu = fractional_psu(load_passes(psbd_dir, PSBD_TM, tm_rate, baselines))

    for name, placement in PARTNERS.items():
        shift_by_rate = validation_shift_by_rate(psbd_dir, placement, baselines)
        for rate_rule in RATE_RULES:
            rate = choose_rate(shift_by_rate, rate_rule) if shift_by_rate else None
            if rate is None:
                row["partners"][rate_rule][name] = {
                    "rate": None,
                    "max_validation_shift": max(shift_by_rate.values())
                    if shift_by_rate
                    else None,
                    "rates_cached": sorted(shift_by_rate),
                }
                continue
            partner_psu = fractional_psu(
                load_passes(psbd_dir, placement, rate, baselines)
            )
            row["partners"][rate_rule][name] = {
                "rate": rate,
                "validation_shift": shift_by_rate[rate],
                "rules": fuse_and_evaluate(tm_psu, partner_psu, manifest),
            }
    return row


def fractional_psu(passes):
    psu = {
        split: psu_ratio_from_cache(
            passes[split]["baseline_probs"],
            passes[split]["baseline_labels"],
            passes[split]["per_pass_probs"],
        )  # (n_split,)
        for split in passes
    }
    return psu


def fuse_and_evaluate(tm_psu, partner_psu, manifest):
    fused = {rule: {} for rule in RULES}
    for split in tm_psu:
        tm = tm_psu[split]  # (n_split,)
        partner = partner_psu[split]  # (n_split,)
        assert tm.shape == partner.shape, (split, tm.shape, partner.shape)

        # Percentiles within each probe's own clean-validation distribution put
        # the 2 probes on 1 scale without fitting anything to triggered data.
        tm_rank = to_rank(tm, tm_psu["validation"])  # (n_split,)
        partner_rank = to_rank(partner, partner_psu["validation"])  # (n_split,)

        fused["tm_alone"][split] = tm
        fused["mean_psu"][split] = (tm + partner) / 2
        fused["min_rank"][split] = torch.minimum(tm_rank, partner_rank)
        for rule, tm_share in WEIGHTED_SHARES.items():
            fused[rule][split] = torch.minimum(
                tm_rank / tm_share, partner_rank / (1 - tm_share)
            )

        # Fisher needs a percentile above 0, so it uses the (count + 1) / (n + 1)
        # form. The log of each percentile summed is low when both are low.
        n_validation = len(tm_psu["validation"])
        tm_p = (tm_rank * n_validation + 1) / (n_validation + 1)  # (n_split,)
        partner_p = (partner_rank * n_validation + 1) / (n_validation + 1)  # (n_split,)
        fused["fisher"][split] = torch.log(tm_p) + torch.log(partner_p)

    evaluations = {
        rule: evaluate_scores(fused[rule], manifest, QUANTILES) for rule in RULES
    }

    # The plan's weighted rule flags u at or below the target FPR itself, which
    # the union bound caps at that FPR on validation. It is reported beside the
    # calibrated threshold every rule above uses.
    for rule, tm_share in WEIGHTED_SHARES.items():
        clean_paired = pair_clean_to_backdoor(
            fused[rule]["clean"], manifest
        )  # (n_backdoor,)
        evaluations[rule]["literal_union_bound"] = {
            f"q{q:.2f}": {
                "tpr": float((fused[rule]["backdoor"] <= q).float().mean()),
                "realized_fpr": float((clean_paired <= q).float().mean()),
                "validation_fpr": float(
                    (fused[rule]["validation"] <= q).float().mean()
                ),
            }
            for q in QUANTILES
        }
    return evaluations


def summarize(rows, rate_rule):
    pooled = [row for row in rows if row["successful_2pt"]]
    summary = {
        "pooled_folders": [row["folder"] for row in pooled],
        "unpooled_folders": [
            row["folder"] for row in rows if not row["successful_2pt"]
        ],
    }
    for name in PARTNERS:
        covered = [
            row
            for row in pooled
            if row["tm_rate"] is not None
            and row["partners"][rate_rule][name]["rate"] is not None
        ]
        missing = [
            {
                "folder": row["folder"],
                "max_validation_shift": row["partners"][rate_rule]
                .get(name, {})
                .get("max_validation_shift"),
                "rates_cached": row["partners"][rate_rule]
                .get(name, {})
                .get("rates_cached"),
            }
            for row in pooled
            if row not in covered
        ]
        summary[name] = {
            "n_models": len(covered),
            "n_pooled": len(pooled),
            "missing": missing,
            "all": summarize_group(covered, rate_rule, name),
            "by_attack": {
                attack: summarize_group(
                    [row for row in covered if row["attack"] == attack], rate_rule, name
                )
                for attack in ordered_attacks({row["attack"] for row in covered})
            },
        }
    return summary


def summarize_group(rows, rate_rule, name):
    def metric(row, rule, field):
        evaluation = row["partners"][rate_rule][name]["rules"][rule]
        if field == "auroc":
            value = evaluation["auroc"]
        else:
            quantile, kind = field.split(":")
            value = evaluation["at_fpr"][quantile][kind]
        return value

    fields = ["auroc"] + [
        f"{q}:{kind}" for q in REPORTED_QUANTILES for kind in ("tpr", "realized_fpr")
    ]
    group = {"n": len(rows), "folders": [row["folder"] for row in rows]}
    for rule in RULES:
        group[rule] = {}
        for field in fields:
            values = [metric(row, rule, field) for row in rows]
            reference = [metric(row, REFERENCE_RULE, field) for row in rows]
            group[rule][field] = paired_summary(values, reference)

    # The literal union-bound threshold of the weighted rules, read against the
    # calibrated PSBD-TM reference at the same nominal FPR.
    for rule in WEIGHTED_SHARES:
        for quantile in REPORTED_QUANTILES:
            for kind in ("tpr", "realized_fpr", "validation_fpr"):
                values = [
                    row["partners"][rate_rule][name]["rules"][rule][
                        "literal_union_bound"
                    ][quantile][kind]
                    for row in rows
                ]
                reference_kind = "realized_fpr" if kind == "validation_fpr" else kind
                reference = [
                    metric(row, REFERENCE_RULE, f"{quantile}:{reference_kind}")
                    for row in rows
                ]
                group[rule][f"literal:{quantile}:{kind}"] = paired_summary(
                    values, reference
                )
    return group


def plot(summary, model_set, json_path):
    partners = list(PARTNERS)
    figure, axes = plt.subplots(
        1, len(partners), figsize=(16, 4.5), sharey=True, squeeze=False
    )
    plotted = {}
    fused_rules = [rule for rule in RULES if rule != REFERENCE_RULE]
    width = 0.8 / len(fused_rules)
    for index, name in enumerate(partners):
        axis = axes[0][index]
        by_attack = summary[name]["by_attack"]
        attacks = list(by_attack)
        for rule_index, rule in enumerate(fused_rules):
            heights = [by_attack[a][rule]["q0.10:tpr"]["mean"] for a in attacks]
            positions = [
                i + (rule_index - len(fused_rules) / 2 + 0.5) * width
                for i in range(len(attacks))
            ]
            axis.bar(positions, heights, width, label=rule, color=OKABE_ITO[rule_index])
            plotted[f"{name}/{rule}"] = dict(zip(attacks, heights))
        reference = [by_attack[a][REFERENCE_RULE]["q0.10:tpr"]["mean"] for a in attacks]
        plotted[f"{name}/{REFERENCE_RULE}"] = dict(zip(attacks, reference))
        axis.hlines(
            reference,
            [i - 0.45 for i in range(len(attacks))],
            [i + 0.45 for i in range(len(attacks))],
            colors="black",
            label="PSBD-TM alone",
        )
        axis.set_xticks(range(len(attacks)))
        axis.set_xticklabels(
            [f"{a}\n(n={by_attack[a]['n']})" for a in attacks], fontsize=8
        )
        axis.set_title(f"PSBD-TM with {PARTNERS[name]}", fontsize=9)
        axis.set_ylim(0, 1)
    axes[0][0].set_ylabel("mean TPR at 10% nominal FPR")
    axes[0][0].legend(fontsize=7)
    figure.suptitle(f"Fusion rules at the adaptive rate, {model_set} set")

    figure_path = f"{FIGURES_DIR}/fusion_rules_{model_set}.png"
    save_figure(figure, figure_path)
    figure_sidecar(
        figure_path.replace(".png", ".json"),
        "experiments/cache_readouts/fusion_rules.py",
        [json_path],
        plotted,
    )


if __name__ == "__main__":
    main()
