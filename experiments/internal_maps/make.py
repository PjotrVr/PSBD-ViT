"""CPU stage of the internal maps: every figure from the numbers.json records.

.venv/bin/python -m experiments.internal_maps.make
"""

import json
import os

from experiments._paths import experiment_results_dir
from experiments.internal_maps import config
from experiments.internal_maps.figures import (
    draw_depth_overview,
    draw_depth_survival,
    draw_direction,
    draw_masks,
    draw_swin_attribution,
    draw_token_removal,
    draw_token_removal_overview,
    draw_vit_attention,
)

SLUG = "internal_maps"


def main():
    output_root = experiment_results_dir(SLUG)
    for architecture, models in config.MODELS.items():
        records = read_records(output_root, architecture, models)
        for record in records:
            draw_model(
                os.path.join(output_root, architecture, record["folder"]), record
            )
        if not records:
            continue
        arch_dir = os.path.join(output_root, architecture)
        draw_token_removal_overview(
            os.path.join(arch_dir, "overview_token_removal.png"), records
        )
        draw_depth_overview(
            os.path.join(arch_dir, "overview_depth_probe.png"),
            records,
            "probe_probability",
        )
        if architecture == "vit":
            draw_depth_overview(
                os.path.join(arch_dir, "overview_depth_lens.png"),
                records,
                "lens_probability",
            )
        write_json(os.path.join(arch_dir, "numbers.json"), summarize(records))
        print(f"[ok] {architecture}: {len(records)} models")


def read_records(output_root, architecture, models):
    records = []
    for folder in models:
        path = os.path.join(output_root, architecture, folder, "numbers.json")
        if not os.path.exists(path):
            print(f"[missing] {folder}")
            continue
        with open(path) as handle:
            records.append(json.load(handle))
    return records


def draw_model(model_dir, record):
    draw_token_removal(os.path.join(model_dir, "token_removal.png"), record)
    if record["architecture"] == "vit":
        draw_vit_attention(os.path.join(model_dir, "class_token_attention.png"), record)
    else:
        draw_swin_attribution(
            os.path.join(model_dir, "readout_attribution.png"), record
        )
    draw_depth_survival(os.path.join(model_dir, "depth_survival.png"), record)
    draw_direction(os.path.join(model_dir, "direction.png"), record)
    draw_masks(os.path.join(model_dir, "psbd_tm_masks.png"), record)


# The scalars every overview figure and the README read, 1 row per model. The
# maps themselves stay in each model's numbers.json.
def summarize(records):
    rows = []
    for record in records:
        removal = record["token_removal"]["triggered"]
        clean_removal = record["token_removal"]["clean"]
        survival = record["depth_survival"]
        direction = record["direction"]
        drop_map = [value for row in removal["drop_map"] for value in row]
        trigger_cells = record["trigger_cells"]
        row = {
            "folder": record["folder"],
            "attack": record["attack"],
            "probe_attack": record["probe_attack"],
            "tm_rate": record["tm_rate"],
            "tm_validation_shift": record["tm_validation_shift"],
            "is_patch": record["is_patch"],
            "trigger_units": len(trigger_cells),
            "map_units": record["map_grid"] ** 2,
            "drop_max": max(drop_map),
            "drop_sum": sum(drop_map),
            "drop_share_on_trigger": share_on(drop_map, trigger_cells),
            "clean_drop_max": max(v for r in clean_removal["drop_map"] for v in r),
            "kept_window_probability": {
                window: {
                    "max": max(v for r in kept["probability"] for v in r),
                    "mean": mean([v for r in kept["probability"] for v in r]),
                }
                for window, kept in removal["kept_maps"].items()
            },
            "controls_triggered": removal["controls"],
            "controls_clean": clean_removal["controls"],
            "final_kept": {
                split: {
                    c: survival[split][c]["final_kept"] for c in ("unmasked", "masked")
                }
                for split in ("triggered", "clean")
            },
            "direction_absolute_share_on_trigger": direction[
                "absolute_share_on_trigger"
            ],
            "direction_class_token_projection": direction["class_token_projection"],
        }
        reading = record["trigger_reading"]
        if record["architecture"] == "vit":
            row["class_token_trigger_mass_by_block"] = {
                split: [mean(block) for block in reading[split]["trigger_mass"]]
                for split in ("triggered", "clean")
            }
            row["class_token_trigger_mass_max_head"] = {
                split: max(max(block) for block in reading[split]["trigger_mass"])
                for split in ("triggered", "clean")
            }
            row["uniform_share"] = reading["uniform_share"]
        else:
            row["difference_positive_share_on_trigger"] = reading[
                "difference_positive_share_on_trigger"
            ]
            row["readout_excess_positive_share_on_trigger"] = reading[
                "readout_excess_positive_share_on_trigger"
            ]
            row["difference_first_order_total"] = reading[
                "difference_first_order_total"
            ]
            row["logit_change"] = reading["logit_change"]
            row["attribution_trigger_area_share"] = reading["trigger_area_share"]
        rows.append(row)
    summary = {"models": rows}
    return summary


def share_on(values, cells):
    total = sum(max(v, 0.0) for v in values)
    on_cells = sum(max(values[c], 0.0) for c in cells)
    share = on_cells / total if total > 0 else None
    return share


def mean(values):
    average = sum(values) / len(values)
    return average


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1)


if __name__ == "__main__":
    main()
