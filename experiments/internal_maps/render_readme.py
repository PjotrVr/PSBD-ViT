"""Renders README.md from README.template.md and the JSON records, so no number in it is typed.

.venv/bin/python -m experiments.internal_maps.render_readme
"""

import json
import os

import numpy as np

from experiments.internal_maps import config

ROOT = os.path.join("results", "_experiments", "internal_maps")
EARLIER = os.path.join("results", "_experiments", "why_token_masking_works")
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    values = {}
    values.update(first_batch_values())
    values.update(gallery_values())
    values.update(second_batch_values())
    with open(os.path.join(HERE, "README.template.md")) as handle:
        template = handle.read()
    rendered = template.format_map(values)
    with open(os.path.join(HERE, "README.md"), "w") as handle:
        handle.write(rendered)
    print(f"[ok] README.md, {len(values)} values")


def load(*parts):
    with open(os.path.join(*parts)) as handle:
        record = json.load(handle)
    return record


def fmt(value, digits=3):
    text = f"{value:.{digits}f}"
    return text


def first_block_above(values, level):
    for index, value in enumerate(values):
        if value > level:
            return index + 1
    return None


def first_batch_values():
    values = {}
    for architecture in ("vit", "swin"):
        values[f"{architecture}_models"] = ", ".join(
            f"`{f}`" for f in config.MODELS[architecture]
        )
    vit = {f: load(ROOT, "vit", f, "numbers.json") for f in config.MODELS["vit"]}
    swin = {f: load(ROOT, "swin", f, "numbers.json") for f in config.MODELS["swin"]}
    earlier_vit = load(EARLIER, "vit", "vit_gtsrb_badnet_a2o_0_1.json")
    earlier_swin = load(EARLIER, "swin", "swin_gtsrb_badnet_a2o_0_1.json")

    gtsrb = vit["vit_gtsrb_badnet_a2o_0_1"]
    controls = gtsrb["token_removal"]["triggered"]["controls"]
    values["vit_badnet_hidden_kept"] = fmt(controls["trigger_hidden_kept"])
    values["vit_badnet_random_kept"] = fmt(controls["random_hidden_kept"])
    values["vit_badnet_earlier_hidden"] = fmt(
        earlier_vit["deterministic_masking"]["all_12"]["triggered_kept"]
    )
    values["vit_badnet_earlier_random"] = fmt(
        earlier_vit["deterministic_masking"]["random_all_12"]["triggered_kept"]
    )
    survival = gtsrb["depth_survival"]
    values["vit_badnet_tm_triggered_kept"] = fmt(
        survival["triggered"]["masked"]["final_kept"]
    )
    values["vit_badnet_tm_clean_kept"] = fmt(survival["clean"]["masked"]["final_kept"])
    values["vit_badnet_earlier_tm_triggered"] = fmt(
        earlier_vit["stochastic_token_mask"]["overall"]["triggered_kept"]
    )
    values["vit_badnet_earlier_tm_clean"] = fmt(
        earlier_vit["stochastic_token_mask"]["overall"]["clean_kept"]
    )
    values["vit_badnet_tm_rate"] = gtsrb["tm_rate"]
    swin_gtsrb = swin["swin_gtsrb_badnet_a2o_0_1"]
    values["swin_badnet_hidden_kept"] = fmt(
        swin_gtsrb["token_removal"]["triggered"]["controls"]["trigger_hidden_kept"]
    )
    values["swin_badnet_earlier_hidden"] = fmt(
        earlier_swin["deterministic_masking"]["all_24"]["triggered_kept"]
    )
    values["swin_badnet_tm_triggered_kept"] = fmt(
        swin_gtsrb["depth_survival"]["triggered"]["masked"]["final_kept"]
    )
    values["swin_badnet_tm_clean_kept"] = fmt(
        swin_gtsrb["depth_survival"]["clean"]["masked"]["final_kept"]
    )
    values["swin_badnet_earlier_tm_triggered"] = fmt(
        earlier_swin["stochastic_token_mask"]["overall"]["triggered_kept"]
    )
    values["swin_badnet_earlier_tm_clean"] = fmt(
        earlier_swin["stochastic_token_mask"]["overall"]["clean_kept"]
    )

    for architecture, records in (("vit", vit), ("swin", swin)):
        for folder, record in records.items():
            key = short_key(folder)
            removal = record["token_removal"]
            triggered = removal["triggered"]
            floor = removal["clean"]["controls_on_target"]
            drop = np.array(triggered["drop_map"])
            values[f"{key}_drop_max"] = fmt(drop.max())
            if (
                record["trigger_cells"]
                and len(record["trigger_cells"]) < drop.size // 2
            ):
                values[f"{key}_drop_share"] = fmt(
                    drop.clip(0)[
                        np.unravel_index(record["trigger_cells"], drop.shape)
                    ].sum()
                    / max(drop.clip(0).sum(), 1e-12),
                    2,
                )
            for fraction in config.VISIBLE_FRACTIONS:
                tag = str(fraction).replace(".", "")
                kept = triggered["controls"][f"visible_{fraction}_kept"]
                values[f"{key}_visible_{tag}"] = fmt(kept, 2)
                values[f"{key}_floor_{tag}"] = fmt(floor[f"visible_{fraction}_kept"], 2)
                values[f"{key}_excess_{tag}"] = fmt(
                    kept - floor[f"visible_{fraction}_kept"], 2
                )
            windows = list(triggered["kept_maps"])
            for window in windows:
                excess = np.array(
                    triggered["kept_maps"][window]["probability"]
                ) - np.array(
                    removal["clean"]["kept_maps"][window]["target_probability"]
                )
                values[f"{key}_window_{window}_excess_max"] = fmt(excess.max(), 2)
                values[f"{key}_window_{window}_excess_mean"] = fmt(excess.mean(), 2)
            survival = record["depth_survival"]
            for split in ("triggered", "clean"):
                for condition in ("unmasked", "masked"):
                    values[f"{key}_{split}_{condition}_kept"] = fmt(
                        survival[split][condition]["final_kept"], 2
                    )
            values[f"{key}_rate"] = record["tm_rate"]
            values[f"{key}_rate_rule"] = record["tm_rate_rule"]
            direction = record["direction"]["absolute_share_on_trigger"]
            values[f"{key}_direction_first"] = fmt(direction[0], 2)
            values[f"{key}_direction_last"] = fmt(direction[-1], 2)
            if architecture == "vit":
                lens = survival["triggered"]["unmasked"]["lens_probability"]
                clean_lens = survival["clean"]["unmasked"]["lens_probability"]
                values[f"{key}_lens_triggered_onset"] = first_block_above(lens, 0.5)
                values[f"{key}_lens_clean_onset"] = first_block_above(clean_lens, 0.5)
                masked_lens = survival["triggered"]["masked"]["lens_probability"]
                values[f"{key}_lens_triggered_masked_onset"] = first_block_above(
                    masked_lens, 0.5
                )
                values[f"{key}_lens_clean_masked_max"] = fmt(
                    max(survival["clean"]["masked"]["lens_probability"]), 2
                )
                mass = np.array(record["trigger_reading"]["triggered"]["trigger_mass"])
                clean_mass = np.array(
                    record["trigger_reading"]["clean"]["trigger_mass"]
                )
                values[f"{key}_mass_last"] = fmt(mass[-1].mean(), 2)
                values[f"{key}_mass_max"] = fmt(mass.max(), 2)
                values[f"{key}_clean_mass_last"] = fmt(clean_mass[-1].mean(), 3)
                values[f"{key}_mass_onset"] = first_block_above(mass.mean(axis=1), 0.1)
                values[f"{key}_uniform"] = fmt(
                    record["trigger_reading"]["uniform_share"], 3
                )
                projection = record["direction"]["class_token_projection"]
                values[f"{key}_cls_projection_onset"] = first_block_above(
                    np.array(projection) / max(abs(min(projection)), max(projection)),
                    0.1,
                )
                maps = np.array(record["direction"]["maps"][0]).flatten()
                cells = record["trigger_cells"]
                values[f"{key}_trigger_projection_sign"] = (
                    "negative"
                    if len(cells) < 20 and maps[cells].sum() < 0
                    else "positive"
                )
            else:
                reading = record["trigger_reading"]
                shares = reading["difference_positive_share_on_trigger"]
                for stage, share in enumerate(shares, start=1):
                    values[f"{key}_difference_stage_{stage}"] = fmt(share, 2)
                values[f"{key}_readout_share"] = fmt(
                    reading["readout_excess_positive_share_on_trigger"], 2
                )
                values[f"{key}_area_last"] = fmt(reading["trigger_area_share"][-1], 2)
                probe = survival["triggered"]["masked"]["probe_probability"]
                values[f"{key}_probe_masked_min"] = fmt(min(probe[4:]), 2)
            masks = record["masks"]
            values[f"{key}_mask_clean_predictions"] = ", ".join(
                str(p["images"]["clean"]["prediction"]) for p in masks["passes"]
            )
            values[f"{key}_mask_triggered_target_min"] = fmt(
                min(
                    p["images"]["triggered"]["target_probability"]
                    for p in masks["passes"]
                ),
                3,
            )
            values[f"{key}_mask_target"] = masks["target_class"]
    return values


def short_key(folder):
    parts = folder.split("_")
    architecture, dataset, attack = parts[0], parts[1], parts[2]
    key = f"{architecture}_{dataset}_{attack}"
    return key


def gallery_values():
    summary = load(ROOT, "gallery_summary.json")
    values = {}
    rows = [
        "| model | threshold | clean: target class | clean: predicted target |"
        " clean: negative PSU | clean: P_c < 0.9 | clean: largest class share |"
        " all validation: target class | all validation: negative PSU |"
        " all validation: P_c < 0.9 | TPR at 1% | escaping hits: P_c < 0.9 |"
        " non-hits among the 20 highest |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for folder, entry in summary.items():
        clean = entry["lowest_clean"]
        hits = entry["highest_triggered"]
        base = entry["validation_base_rates"]
        rows.append(
            f"| `{folder}` | {fmt(entry['threshold'])} | {fmt(clean['target_class_share'], 2)} |"
            f" {fmt(clean['predicted_target_share'], 2)} | {fmt(clean['negative_psu_share'], 2)} |"
            f" {fmt(clean['below_confident_share'], 2)} | {fmt(clean['largest_true_class_share'], 2)} |"
            f" {fmt(base['target_class_share'], 3)} | {fmt(base['negative_psu_share'], 3)} |"
            f" {fmt(base['below_confident_share'], 3)} | {fmt(entry['tpr_at_budget'])} |"
            f" {fmt(hits['below_confident_share'], 2)} |"
            f" {fmt(entry['non_hit_share_of_highest_overall'], 2)} |"
        )
        key = folder.replace("-", "_")
        values[f"g_{key}_clean_target"] = fmt(clean["target_class_share"], 2)
        values[f"g_{key}_clean_negative"] = fmt(clean["negative_psu_share"], 2)
        values[f"g_{key}_clean_low_confidence"] = fmt(clean["below_confident_share"], 2)
        values[f"g_{key}_clean_predicted_target"] = fmt(
            clean["predicted_target_share"], 2
        )
        values[f"g_{key}_clean_misclassified"] = fmt(clean["misclassified_share"], 2)
        values[f"g_{key}_largest_class"] = fmt(clean["largest_true_class_share"], 2)
        values[f"g_{key}_top_class"] = clean["top_true_classes"][0][0]
        values[f"g_{key}_base_low_confidence"] = fmt(base["below_confident_share"], 3)
        values[f"g_{key}_base_misclassified"] = fmt(base["misclassified_share"], 3)
        values[f"g_{key}_base_target"] = fmt(base["target_class_share"], 3)
        values[f"g_{key}_non_hit"] = fmt(entry["non_hit_share_of_highest_overall"], 2)
        values[f"g_{key}_tpr"] = fmt(entry["tpr_at_budget"])
        values[f"g_{key}_hits_low_confidence"] = fmt(hits["below_confident_share"], 2)
    values["gallery_table"] = "\n".join(rows)
    values["gallery_model_count"] = len(summary)
    values["gallery_non_hit_all_count"] = sum(
        entry["non_hit_share_of_highest_overall"] == 1.0 for entry in summary.values()
    )
    return values


def second_batch_values():
    numbers = load(ROOT, "batch2_numbers.json")
    values = {}
    for folder, entry in numbers["confidence"].items():
        key = folder
        values[f"c_{key}_negative_count"] = entry["negative_psu_validation_count"]
        for group in ("clean_negative_psu", "triggered_near_zero_psu"):
            g = entry[group]
            tag = "clean" if group.startswith("clean") else "triggered"
            values[f"c_{key}_{tag}_count"] = g["count"]
            values[f"c_{key}_{tag}_base"] = fmt(g["mean_base_probability"], 2)
            values[f"c_{key}_{tag}_max_rise"] = fmt(g["mean_largest_single_rise"], 3)
            values[f"c_{key}_{tag}_attention_share"] = fmt(
                g["rise_share_on_top_attention_quarter"], 2
            )
            values[f"c_{key}_{tag}_rise_rank"] = fmt(
                g["rise_weighted_attention_rank"], 2
            )
            values[f"c_{key}_{tag}_drop_rank"] = fmt(
                g["drop_weighted_attention_rank"], 2
            )
            values[f"c_{key}_{tag}_border"] = fmt(g["rise_share_on_border"], 2)
            values[f"c_{key}_border_area"] = fmt(g["border_area_share"], 2)
            for count in config.JOINT_COUNTS:
                joint = g["joint_top_k"][str(count)]
                values[f"c_{key}_{tag}_top_{count}"] = fmt(
                    joint["top"]["probability"], 2
                )
                values[f"c_{key}_{tag}_random_{count}"] = fmt(
                    joint["random"]["probability"], 2
                )
    rows = [
        "| model | stamped non-source sent to target | mean largest 1-token rise of P(target) |"
        " tokens for half the rise (median) | top-8 hidden: sent to target | top-64 hidden |"
        " top-64 on the unstamped twin | probe AUROC | probe TPR at 1/5/10% FPR |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for folder, entry in numbers["veto"].items():
        probe = entry["content_hidden_probe"]
        rows.append(
            f"| `{folder}` | {fmt(entry['non_source_sent_to_target'], 2)} |"
            f" {fmt(entry['mean_largest_single_rise'], 3)} |"
            f" {entry['median_units_for_half_rise']:.0f} |"
            f" {fmt(entry['joint_top_k']['8']['top']['argmax_is_class'], 2)} |"
            f" {fmt(entry['joint_top_k']['64']['top']['argmax_is_class'], 2)} |"
            f" {fmt(entry['unstamped_joint_top_k']['64']['top']['argmax_is_class'], 2)} |"
            f" {fmt(probe['auroc'])} | {fmt(probe['q0.01']['tpr'], 3)} /"
            f" {fmt(probe['q0.05']['tpr'], 3)} / {fmt(probe['q0.10']['tpr'], 3)} |"
        )
        values[f"v_{folder}_auroc"] = fmt(probe["auroc"])
        values[f"v_{folder}_tpr10"] = fmt(probe["q0.10"]["tpr"], 3)
        means = probe["score_means"]
        values[f"v_{folder}_mean_validation"] = fmt(means["validation"])
        values[f"v_{folder}_mean_triggered"] = fmt(means["triggered"])
        values[f"v_{folder}_mean_clean"] = fmt(means["clean"])
    values["veto_table"] = "\n".join(rows)
    aurocs = [e["content_hidden_probe"]["auroc"] for e in numbers["veto"].values()]
    tprs = [e["content_hidden_probe"]["q0.10"]["tpr"] for e in numbers["veto"].values()]
    values["veto_auroc_refuted_count"] = sum(a > 0.65 for a in aurocs)
    values["veto_tpr_refuted_count"] = sum(t > 0.20 for t in tprs)
    values["veto_model_count"] = len(aurocs)

    rows = [
        "| model | trigger tokens | rate | survival | mean kept | measured TPR 1/5/10% |"
        " predicted TPR 1/5/10% |",
        "|---|---|---|---|---|---|---|",
    ]
    for folder, entry in numbers["survival"].items():
        for rate, reading in entry["by_rate"].items():
            measured = reading["measured_tpr_cache"]
            predicted = reading["predicted_tpr"]
            rows.append(
                f"| `{folder}` | {entry['trigger_tokens']} | {rate} |"
                f" {fmt(reading['survival_rate'], 3)} | {fmt(reading['mean_kept_late'], 2)} |"
                f" {fmt(measured['q0.01'], 2)} / {fmt(measured['q0.05'], 2)} /"
                f" {fmt(measured['q0.10'], 2)} | {fmt(predicted['q0.01'], 2)} /"
                f" {fmt(predicted['q0.05'], 2)} / {fmt(predicted['q0.10'], 2)} |"
            )
    values["survival_table"] = "\n".join(rows)
    errors = {"in": [], "out": []}
    for entry in numbers["survival"].values():
        for rate, reading in entry["by_rate"].items():
            for budget in ("q0.01", "q0.05", "q0.10"):
                error = abs(
                    reading["predicted_tpr"][budget]
                    - reading["measured_tpr_cache"][budget]
                )
                errors["in" if rate == "0.5" else "out"].append(error)
    values["survival_error_in"] = fmt(np.mean(errors["in"]), 2)
    values["survival_error_out"] = fmt(np.mean(errors["out"]), 2)
    return values


if __name__ == "__main__":
    main()
