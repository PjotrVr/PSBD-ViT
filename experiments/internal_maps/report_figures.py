"""Report versions of 4 internal-map figures, with fewer panels per image and large type.

The overviews in make.py put 7 models on 1 image, which a printed page shrinks
below a readable size. These draw the same numbers.json records in groups of at
most 4 models, with fonts sized for a full page. The threshold gallery rebuilds
its images through gallery.gallery_record on the CPU and shows the 10 lowest.

    .venv/bin/python -m experiments.internal_maps.report_figures
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments._paths import experiment_results_dir  # noqa: E402
from experiments.internal_maps import config  # noqa: E402
from experiments.internal_maps.figures import (  # noqa: E402
    LINE_STYLES,
    excess_target_map,
    show_image,
    show_map,
)

SLUG = "internal_maps"
CPU_THREADS = 4
FONT_SIZE = 17
# The token-removal and attention grids print at a smaller scale than the line
# plots, so their type is larger to land at the same printed size.
GRID_FONT_SIZE = 24
GALLERY_MODEL = "vit_gtsrb_tact_0_01_cos"
GALLERY_SHOWN = 10
ATTENTION_MODELS = ("vit_gtsrb_badnet_a2o_0_1", "vit_gtsrb_tact_0_01_cos")
# Patch triggers with their controls in 1 image, the spread triggers in the other.
GROUPS = {
    "vit": {
        "patch": (
            "vit_gtsrb_badnet_a2o_0_1",
            "vit_cifar100_badnet_a2o_0_1",
            "vit_gtsrb_tact_0_01_cos",
            "vit_gtsrb_benign",
        ),
        "global": ("vit_gtsrb_blend_0_1", "vit_gtsrb_lf_0_1", "vit_tiny_wanet_0_1"),
    },
    "swin": {
        "patch": (
            "swin_gtsrb_badnet_a2o_0_1",
            "swin_cifar100_badnet_a2o_0_1",
            "swin_cifar10_tact_0_05",
            "swin_gtsrb_benign",
        ),
        "global": ("swin_gtsrb_blend_0_1", "swin_gtsrb_lf_0_1", "swin_gtsrb_wanet_0_1"),
    },
}
ATTACK_NAMES = {
    "badnet_a2o": "BadNets",
    "blend": "Blend",
    "lf": "LF",
    "wanet": "WaNet",
    "tact": "TaCT",
}
DATASET_NAMES = {
    "gtsrb": "GTSRB",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "tiny": "Tiny ImageNet",
}
SPLIT_LABELS = {
    ("triggered", "unmasked"): "triggered, P(target), unmasked",
    ("triggered", "masked"): "triggered, P(target), under PSBD-TM",
    ("clean", "unmasked"): "clean, P(own class), unmasked",
    ("clean", "masked"): "clean, P(own class), under PSBD-TM",
}


def main():
    plt.rcParams.update({"font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE + 1})
    torch.set_num_threads(CPU_THREADS)
    root = experiment_results_dir(SLUG)

    vit = {folder: read_record(root, "vit", folder) for folder in all_folders("vit")}
    swin = {folder: read_record(root, "swin", folder) for folder in all_folders("swin")}

    draw_lens(
        os.path.join(root, "vit", "report_depth_lens_patch.png"),
        [vit[f] for f in GROUPS["vit"]["patch"]],
    )
    draw_lens(
        os.path.join(root, "vit", "report_depth_lens_global.png"),
        [vit[f] for f in GROUPS["vit"]["global"]],
    )
    for architecture, records in (("vit", vit), ("swin", swin)):
        for group, folders in GROUPS[architecture].items():
            draw_removal(
                os.path.join(root, architecture, f"report_token_removal_{group}.png"),
                [records[f] for f in folders],
            )
    draw_attention(
        os.path.join(root, "vit", "report_class_token_attention.png"),
        [vit[f] for f in ATTENTION_MODELS],
    )
    draw_gallery(os.path.join(root, "vit", GALLERY_MODEL, "report_gallery.png"))
    print("report figures done")


def all_folders(architecture):
    folders = [f for group in GROUPS[architecture].values() for f in group]
    return folders


def read_record(root, architecture, folder):
    with open(os.path.join(root, architecture, folder, "numbers.json")) as handle:
        record = json.load(handle)
    return record


def title_of(record):
    architecture = "ViT-B/16" if record["architecture"] == "vit" else "Swin-S"
    dataset = DATASET_NAMES.get(record["dataset"], record["dataset"])
    if record["probe_attack"]:
        return f"{architecture} benign, {dataset}, BadNets probe"
    attack = ATTACK_NAMES.get(record["attack"], record["attack"])
    rate = record["poison_rate"]
    title = f"{architecture} {attack}, {dataset}, {rate * 100:g}%"
    return title


def draw_lens(path, records):
    figure, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True)
    for axis, record in zip(axes.flat, records):
        survival = record["depth_survival"]
        for split in ("triggered", "clean"):
            for condition in ("unmasked", "masked"):
                values = survival[split][condition]["lens_probability"]
                color, style = LINE_STYLES[(split, condition)]
                axis.plot(
                    range(1, len(values) + 1),
                    values,
                    color=color,
                    ls=style,
                    lw=3,
                    marker="o",
                    ms=5,
                    label=SPLIT_LABELS[(split, condition)],
                )
        axis.set_title(f"{title_of(record)}\nPSBD-TM rate {record['tm_rate']:g}")
        axis.set_ylim(-0.02, 1.02)
        axis.set_xticks(range(1, 13))
        axis.tick_params(labelbottom=True)
        axis.grid(alpha=0.3)
    for axis in axes[1]:
        axis.set_xlabel("block (logit lens on the class token)")
    for axis in axes[:, 0]:
        axis.set_ylabel("probability")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if len(records) < 4:
        spare = axes.flat[3]
        spare.axis("off")
        spare.legend(handles, labels, loc="center")
    else:
        figure.legend(handles, labels, loc="lower center", ncol=2)
    figure.tight_layout(rect=(0, 0.1 if len(records) == 4 else 0, 1, 1))
    save(figure, path)


def draw_removal(path, records):
    with plt.rc_context(
        {"font.size": GRID_FONT_SIZE, "axes.titlesize": GRID_FONT_SIZE - 2}
    ):
        draw_removal_grid(path, records)


def draw_removal_grid(path, records):
    figure, axes = plt.subplots(
        len(records), 4, figsize=(16, 4.6 * len(records)), squeeze=False
    )
    for row, record in zip(axes, records):
        removal = record["token_removal"]
        windows = list(removal["triggered"]["kept_maps"])
        show_image(row[0], record["examples"]["triggered"], "")
        row[0].set_ylabel(title_of(record).replace(", ", "\n", 1))
        drop = np.array(removal["triggered"]["drop_map"])
        show_map(
            row[1],
            drop,
            "",
            "magma",
            0,
            max(drop.max(), 1e-3),
            trigger_cells=record["trigger_cells"],
            grid=record["map_grid"],
        )
        row[1].set_title(f"(a) 1 unit hidden,\ndrop max {drop.max():.3f}")
        for axis, window in zip(row[2:4], windows):
            show_map(
                axis,
                excess_target_map(removal, window),
                "",
                "RdBu_r",
                -1,
                1,
                trigger_cells=record["trigger_cells"],
                grid=record["map_grid"],
            )
            axis.set_title(f"(b) {window} x {window} visible,\ntriggered minus clean")
    figure.tight_layout()
    save(figure, path)


def draw_attention(path, records):
    with plt.rc_context(
        {"font.size": GRID_FONT_SIZE - 3, "axes.titlesize": GRID_FONT_SIZE - 3}
    ):
        draw_attention_grid(path, records)


def draw_attention_grid(path, records):
    figure, axes = plt.subplots(2, 2, figsize=(15, 13))
    limit = max(
        np.max(record["trigger_reading"][split]["trigger_mass"])
        for record in records
        for split in ("triggered", "clean")
    )
    for row, record in zip(axes, records):
        reading = record["trigger_reading"]
        for axis, split in zip(row, ("triggered", "clean")):
            image = axis.imshow(
                np.array(reading[split]["trigger_mass"]),
                cmap="viridis",
                vmin=0,
                vmax=limit,
                aspect="auto",
            )
            axis.set_xticks(range(0, 12, 2), [str(h + 1) for h in range(0, 12, 2)])
            axis.set_yticks(range(0, 12, 2), [str(b + 1) for b in range(0, 12, 2)])
            axis.set_xlabel("head")
            axis.set_ylabel("block")
            axis.set_title(f"{title_of(record)}\n{split} images")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    figure.suptitle(
        "Class-token attention mass on the 4 trigger tokens, mean over "
        "pairs (uniform 0.020)"
    )
    figure.tight_layout()
    save(figure, path)


def draw_gallery(path):
    from experiments.internal_maps.gallery import gallery_record

    record, images = gallery_record(GALLERY_MODEL)
    entries = record["lowest_clean"]["entries"][:GALLERY_SHOWN]
    shown = images["lowest_clean"][:GALLERY_SHOWN]
    figure, axes = plt.subplots(2, 5, figsize=(16, 8.4))
    for axis, entry, pixels in zip(axes.flat, entries, shown):
        axis.imshow(pixels, interpolation="nearest")
        axis.set_title(
            f"true {entry['true_class']}, pred {entry['predicted_class']}\n"
            f"PSU {entry['psu']:+.3f}"
        )
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"ViT TaCT, GTSRB, 1%: the {GALLERY_SHOWN} lowest-PSU clean validation images,"
        f" threshold at 1% FPR {record['threshold']:+.3f}, target class "
        f"{record['target_label']}"
    )
    figure.tight_layout()
    save(figure, path)


def save(figure, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, dpi=config.DPI)
    plt.close(figure)


if __name__ == "__main__":
    main()
