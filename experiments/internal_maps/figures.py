"""The internal map figures, 1 function per figure, each drawn from a numbers.json record."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.internal_maps import config  # noqa: E402

ATTACK_LABELS = {
    "badnet_a2o": "BadNets",
    "blend": "Blend",
    "lf": "LF",
    "wanet": "WaNet",
    "tact": "TaCT",
    "benign": "benign",
}
LINE_STYLES = {
    ("triggered", "unmasked"): ("#D55E00", "-"),
    ("triggered", "masked"): ("#D55E00", "--"),
    ("clean", "unmasked"): ("#0072B2", "-"),
    ("clean", "masked"): ("#0072B2", "--"),
}


def model_title(record):
    attack = ATTACK_LABELS.get(record["attack"], record["attack"])
    if record["probe_attack"]:
        attack = f"benign probed with {ATTACK_LABELS[record['probe_attack']]}"
    title = f"{record['folder']} ({attack})"
    return title


def draw_token_removal(path, record):
    removal = record["token_removal"]
    windows = list(removal["triggered"]["kept_maps"])
    figure, axes = plt.subplots(
        1, 5 + len(windows), figsize=(3.1 * (5 + len(windows)), 3.6)
    )
    show_image(axes[0], record["examples"]["clean"], "clean input")
    show_image(axes[1], record["examples"]["triggered"], "triggered input")
    show_map(
        axes[2],
        np.array(record["trigger_pixel_map_56"]),
        "trigger footprint",
        "Greys",
        0,
        1,
    )

    triggered_drop = np.array(removal["triggered"]["drop_map"])
    clean_drop = np.array(removal["clean"]["drop_map"])
    drop_limit = max(triggered_drop.max(), clean_drop.max(), 1e-3)
    show_map(
        axes[3],
        triggered_drop,
        "(a) triggered: drop in P(target)\n1 unit hidden in every block",
        "magma",
        0,
        drop_limit,
        trigger_cells=record["trigger_cells"],
        grid=record["map_grid"],
    )
    for axis, window in zip(axes[4 : 4 + len(windows)], windows):
        show_map(
            axis,
            excess_target_map(removal, window),
            f"(b) P(target) with only a {window} x {window}\nwindow visible, triggered minus clean",
            "RdBu_r",
            -1,
            1,
            trigger_cells=record["trigger_cells"],
            grid=record["map_grid"],
        )
    show_map(
        axes[-1],
        clean_drop,
        "control (a) on the clean twin:\ndrop in P(own class)",
        "magma",
        0,
        drop_limit,
        trigger_cells=record["trigger_cells"],
        grid=record["map_grid"],
    )
    controls = removal["triggered"]["controls"]
    footer = f"mean over {removal['pairs']} pairs, P(target) unmasked {removal['triggered']['base_probability']:.3f}"
    if "trigger_hidden_kept" in controls:
        footer += (
            f", trigger units hidden: kept {controls['trigger_hidden_kept']:.3f},"
            f" as many random units: {controls['random_hidden_kept']:.3f}"
        )
    footer += "\ntriggered kept with a random visible share (clean kept): " + ", ".join(
        f"{int(f * 100)}% {controls[f'visible_{f}_kept']:.2f} "
        f"({removal['clean']['controls'][f'visible_{f}_kept']:.2f})"
        for f in config.VISIBLE_FRACTIONS
    )
    figure.suptitle(model_title(record), fontsize=11)
    figure.text(0.5, 0.01, footer, ha="center", fontsize=8.5)
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    save(figure, path)


def draw_token_removal_overview(path, records):
    columns = 5
    figure, axes = plt.subplots(
        len(records), columns, figsize=(3.0 * columns, 3.0 * len(records))
    )
    for row, record in zip(axes, records):
        removal = record["token_removal"]
        windows = list(removal["triggered"]["kept_maps"])
        show_image(row[0], record["examples"]["triggered"], "")
        row[0].set_ylabel(model_title(record).replace(" (", "\n("), fontsize=8)
        show_map(
            row[1],
            np.array(record["trigger_pixel_map_56"]),
            "trigger footprint",
            "Greys",
            0,
            1,
        )
        drop = np.array(removal["triggered"]["drop_map"])
        show_map(
            row[2],
            drop,
            f"(a) drop in P(target), max {drop.max():.3f}",
            "magma",
            0,
            max(drop.max(), 1e-3),
            trigger_cells=record["trigger_cells"],
            grid=record["map_grid"],
        )
        for axis, window in zip(row[3:5], windows):
            show_map(
                axis,
                excess_target_map(removal, window),
                f"(b) P(target), {window} x {window} visible,\ntriggered minus clean",
                "RdBu_r",
                -1,
                1,
                trigger_cells=record["trigger_cells"],
                grid=record["map_grid"],
            )
    figure.tight_layout()
    save(figure, path)


def draw_vit_attention(path, record):
    reading = record["trigger_reading"]
    blocks = config.ROLLOUT_BLOCKS["vit"]
    figure = plt.figure(figsize=(17, 13))
    grid_spec = figure.add_gridspec(
        4, 2 + len(blocks), width_ratios=(1.4, 1.4) + (1,) * len(blocks)
    )
    limit = max(
        np.max(reading["triggered"]["trigger_mass"]),
        np.max(reading["clean"]["trigger_mass"]),
    )
    for index, split in enumerate(("triggered", "clean")):
        axis = figure.add_subplot(grid_spec[2 * index : 2 * index + 2, 0:2])
        image = axis.imshow(
            np.array(reading[split]["trigger_mass"]),
            cmap="viridis",
            vmin=0,
            vmax=limit,
            aspect="auto",
        )
        axis.set_xlabel("head")
        axis.set_ylabel("block")
        axis.set_xticks(range(12), [str(h + 1) for h in range(12)])
        axis.set_yticks(range(12), [str(b + 1) for b in range(12)])
        axis.set_title(
            f"{split}: class-token attention mass on the {reading['trigger_tokens']} trigger tokens,"
            f" mean over pairs\n(uniform attention would give {reading['uniform_share']:.3f})",
            fontsize=9,
        )
        figure.colorbar(image, ax=axis, fraction=0.04)
        for column, block in enumerate(blocks):
            axis = figure.add_subplot(grid_spec[2 * index, 2 + column])
            rollout = np.array(reading[split]["rollout_example"][str(block)])
            overlay(
                axis,
                record["examples"][split],
                rollout,
                f"{split}, rollout to block {block}",
            )
            if "attention_example" in reading[split]:
                axis = figure.add_subplot(grid_spec[2 * index + 1, 2 + column])
                attention = np.array(reading[split]["attention_example"][str(block)])
                overlay(
                    axis,
                    record["examples"][split],
                    attention,
                    f"{split}, class-token attention\nof block {block} alone (head mean)",
                )
    figure.suptitle(
        model_title(record)
        + ", attention of the class token (overlays: example image, own color scale)",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    save(figure, path)


def draw_swin_attribution(path, record):
    reading = record["trigger_reading"]
    blocks = reading["stage_end_blocks"]
    figure, axes = plt.subplots(
        2, len(blocks) + 1, figsize=(3.3 * (len(blocks) + 1), 7.0)
    )
    show_image(axes[0][0], record["examples"]["triggered"], "triggered input")
    for column, block in enumerate(blocks):
        values = np.array(reading["difference_maps"][column])
        limit = max(np.abs(values).max(), 1e-6)
        show_map(
            axes[0][column + 1],
            values,
            f"stage {column + 1} end (block {block}): gradient x\n(triggered - clean),"
            f" positive share on trigger {reading['difference_positive_share_on_trigger'][column]:.2f}"
            f"\n(trigger area {reading['trigger_area_share'][column]:.2f})",
            "RdBu_r",
            -limit,
            limit,
        )
    show_image(axes[1][0], record["examples"]["clean"], "clean input")
    readout_panels = (
        ("triggered", np.array(reading["readout_maps"]["triggered"])),
        ("clean", np.array(reading["readout_maps"]["clean"])),
        ("triggered minus clean", np.array(reading["readout_excess_map"])),
    )
    limit = max(max(np.abs(values).max() for _, values in readout_panels), 1e-6)
    for axis, (name, values) in zip(axes[1][1:], readout_panels):
        title = f"readout share of the logit, {name}\n(exact, W_c . norm(x_t) / 49)"
        if name == "triggered minus clean":
            title += f"\npositive share on trigger {reading['readout_excess_positive_share_on_trigger']:.2f}"
        show_map(axis, values, title, "RdBu_r", -limit, limit)
    axes[1][-1].axis("off")
    figure.suptitle(
        model_title(record)
        + f", where the triggered prediction's logit comes from (logit change {reading['logit_change']:.1f})",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    save(figure, path)


def draw_depth_survival(path, record):
    survival = record["depth_survival"]
    readers = [("probe_probability", "linear probe of the decision")]
    if record["architecture"] == "vit":
        readers.append(("lens_probability", "logit lens (own final norm and head)"))
    figure, axes = plt.subplots(
        1, len(readers), figsize=(6.4 * len(readers), 4.4), squeeze=False
    )
    for axis, (key, label) in zip(axes[0], readers):
        for split in ("triggered", "clean"):
            for condition in ("unmasked", "masked"):
                values = survival[split][condition][key]
                color, style = LINE_STYLES[(split, condition)]
                name = "P(target)" if split == "triggered" else "P(own class)"
                axis.plot(
                    range(1, len(values) + 1),
                    values,
                    color=color,
                    ls=style,
                    marker="o",
                    ms=3,
                    label=f"{split} {name}, {condition_label(condition, survival['rate'])}",
                )
        axis.set_xlabel("block")
        axis.set_ylabel("probability read at the block")
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(label, fontsize=10)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7.5)
    final = ", ".join(
        f"{split} {condition} final kept {survival[split][condition]['final_kept']:.2f}"
        for split in ("triggered", "clean")
        for condition in ("unmasked", "masked")
    )
    figure.suptitle(model_title(record), fontsize=11)
    figure.text(0.5, 0.01, final, ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    save(figure, path)


def draw_depth_overview(path, records, key):
    columns = 3
    rows = (len(records) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(5.2 * columns, 3.8 * rows), squeeze=False
    )
    for axis, record in zip(axes.flat, records):
        survival = record["depth_survival"]
        for split in ("triggered", "clean"):
            for condition in ("unmasked", "masked"):
                values = survival[split][condition][key]
                color, style = LINE_STYLES[(split, condition)]
                axis.plot(
                    range(1, len(values) + 1),
                    values,
                    color=color,
                    ls=style,
                    marker="o",
                    ms=2.5,
                    label=f"{split}, {condition}",
                )
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(model_title(record), fontsize=8.5)
        axis.set_xlabel("block")
        axis.grid(alpha=0.3)
    for axis in list(axes.flat)[len(records) :]:
        axis.axis("off")
    axes[0][0].legend(fontsize=7)
    figure.tight_layout()
    save(figure, path)


def draw_direction(path, record):
    direction = record["direction"]
    blocks = config.DIRECTION_BLOCKS[record["architecture"]]
    figure, axes = plt.subplots(
        1, len(blocks) + 1, figsize=(3.2 * (len(blocks) + 1), 3.9)
    )
    for axis, block in zip(axes, blocks):
        values = np.array(direction["maps"][block - 1])
        limit = max(np.abs(values).max(), 1e-6)
        title = f"block {block}\n|share| on trigger {direction['absolute_share_on_trigger'][block - 1]:.2f}"
        if direction["class_token_projection"]:
            title += (
                f", class token {direction['class_token_projection'][block - 1]:+.1f}"
            )
        show_map(axis, values, title, "RdBu_r", -limit, limit)
    axis = axes[-1]
    shares = direction["absolute_share_on_trigger"]
    axis.plot(
        range(1, len(shares) + 1),
        shares,
        color="#D55E00",
        marker="o",
        ms=3,
        label="|share| on trigger tokens",
    )
    axis.plot(
        range(1, len(shares) + 1),
        direction["trigger_area_share"],
        color="0.5",
        ls=":",
        label="trigger area share",
    )
    if direction["class_token_projection"]:
        projection = np.array(direction["class_token_projection"])
        axis.plot(
            range(1, len(shares) + 1),
            projection / max(np.abs(projection).max(), 1e-6),
            color="#0072B2",
            marker="s",
            ms=3,
            label="class token projection / max",
        )
    axis.set_xlabel("block")
    axis.set_ylim(-1.05, 1.05)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7)
    figure.suptitle(
        model_title(record)
        + ", paired triggered minus clean difference of every token projected on the backdoor direction",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    save(figure, path)


def draw_masks(path, record):
    masks = record["masks"]
    architecture = record["architecture"]
    first, last = config.LATE_BLOCKS[architecture]
    figure, axes = plt.subplots(
        2, 1 + len(masks["passes"]), figsize=(3.3 * (1 + len(masks["passes"])), 8.4)
    )
    for row, split in zip(axes, ("clean", "triggered")):
        base = masks["base"][split]
        show_image(
            row[0],
            record["examples"][split],
            f"{split}, unmasked\npred {base['prediction']}, P(own) {base['own_probability']:.3f}"
            f"\nP(target {masks['target_class']}) {base['target_probability']:.3f}",
        )
        for column, entry in enumerate(masks["passes"]):
            reading = entry["images"][split]
            shading = late_masked_share(reading["dropped"], architecture, first, last)
            title = (
                f"pass {column + 1}: pred {reading['prediction']}, P(own) {reading['own_probability']:.3f}"
                f"\nP(target) {reading['target_probability']:.3f}, masked {reading['masked_share']:.2f}"
            )
            if record["is_patch"]:
                title += f"\ntrigger fully hidden in {reading['late_blocks_trigger_fully_hidden']} late blocks"
            shade_masked(row[column + 1], record["examples"][split], shading, title)
    figure.suptitle(
        f"{model_title(record)}, PSBD-TM at rate {masks['rate']}: darker = hidden in more of blocks {first} to {last}",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save(figure, path)


# A model that sees almost nothing can fall onto 1 default class, the target on
# some models, so the kept window reads the triggered image's target probability
# above its clean twin's under the same window.
def excess_target_map(removal, window):
    triggered = np.array(removal["triggered"]["kept_maps"][window]["probability"])
    clean = np.array(removal["clean"]["kept_maps"][window]["target_probability"])
    excess = triggered - clean
    return excess


# The share of the late blocks in which each image region was hidden, on the
# finest grid any of those blocks uses.
def late_masked_share(dropped, architecture, first, last):
    selected = dropped[first - 1 : last]
    finest = max(int(round(len(d) ** 0.5)) for d in selected)
    total = np.zeros((finest, finest))
    for block_mask in selected:
        grid = int(round(len(block_mask) ** 0.5))
        block_map = np.array(block_mask, dtype=float).reshape(grid, grid)
        total += np.kron(block_map, np.ones((finest // grid, finest // grid)))
    share = total / len(selected)
    return share


# Black at an opacity that grows with the hidden share, so visible regions keep
# their own pixels.
def shade_masked(axis, pixels, share, title):
    image = np.array(pixels, dtype=np.uint8)
    size = image.shape[0]
    shading = np.zeros(share.shape + (4,))
    shading[..., 3] = 0.85 * share
    axis.imshow(image, extent=(0, size, size, 0), interpolation="nearest")
    axis.imshow(shading, extent=(0, size, size, 0), interpolation="nearest")
    axis.set_title(title, fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])


def condition_label(condition, rate):
    label = "unmasked" if condition == "unmasked" else f"PSBD-TM p={rate}"
    return label


def show_image(axis, pixels, title):
    axis.imshow(np.array(pixels, dtype=np.uint8), interpolation="nearest")
    axis.set_title(title, fontsize=8.5)
    axis.set_xticks([])
    axis.set_yticks([])


def show_map(axis, values, title, cmap, low, high, trigger_cells=None, grid=None):
    image = axis.imshow(values, cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
    if trigger_cells is not None and 0 < len(trigger_cells) < grid * grid // 2:
        for cell in trigger_cells:
            row, column = divmod(cell, grid)
            axis.add_patch(
                plt.Rectangle(
                    (column - 0.5, row - 0.5), 1, 1, fill=False, ec="cyan", lw=1.0
                )
            )
    axis.set_title(title, fontsize=8.5)
    axis.set_xticks([])
    axis.set_yticks([])
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.03)


def overlay(axis, pixels, values, title, cmap="inferno", alpha=0.55, limits=None):
    image = np.array(pixels, dtype=np.uint8)
    size = image.shape[0]
    axis.imshow(image, extent=(0, size, size, 0), interpolation="nearest")
    low, high = limits if limits else (np.min(values), np.max(values))
    axis.imshow(
        values,
        cmap=cmap,
        alpha=alpha,
        extent=(0, size, size, 0),
        vmin=low,
        vmax=high,
        interpolation="nearest",
    )
    axis.set_title(title, fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])


def save(figure, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, dpi=config.DPI)
    plt.close(figure)
