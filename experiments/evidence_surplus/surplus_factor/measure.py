"""Experiment A of evidence surplus: how much more backdoor direction a trigger writes than a flip needs.

At every block l the backdoor direction is fitted on the first half of the pairs,
u_l = r_l / ||r_l|| with r_l the mean paired change of the pooled feature. On the
second half 2 quantities are read per image i.

    actual_i = (h_l(x~_i) - h_l(x_i)) . u_l
    needed_i = min { alpha >= 0 : argmax f(h_l(x_i) + alpha u_l) = target }
    surplus_i = actual_i / needed_i

h_l is the pooled feature at block l's output (class token on ViT, spatial mean on
ResNet-18), x~_i the triggered twin of clean image x_i, and f the rest of the
network from block l on. The shift alpha u_l is added to every token (every spatial
position on ResNet), so the pooled feature moves by exactly alpha u_l. needed_i is
found by bisection between 0 and SEARCH_FACTOR times the median actual. An image
still off target at the top of the range is "unreachable". The search assumes the
flip is monotone in alpha, which bisection only locates as a crossing.

Controls: a benign model shown the same trigger, and on every model a random unit
direction with the same search range.

Stages, as in experiments/backdoor_manifestation:
    prepare    CPU, the paired eligible images of the PSBD split
    probe      GPU under the shared lock, 1 model per call
    summarize  CPU, the table, the correlations and the figures
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from lightning import seed_everything  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from analysis.features import captured_layers, transformer_blocks  # noqa: E402
from data.splits import read_checkpoint_metadata  # noqa: E402
from defenses.decision import PUBLISHED_PLACEMENT, RECOMMENDED_PLACEMENT  # noqa: E402
from defenses.inference import forward_logits  # noqa: E402
from experiments._paths import experiment_results_dir  # noqa: E402
from experiments.backdoor_manifestation.measure import (  # noqa: E402
    PAIR_CACHE as MANIFESTATION_PAIRS,
)
from experiments.backdoor_manifestation.measure import (  # noqa: E402
    checkpoint_path,
    paired_images,
)
from models.backbones import load_checkpoint, network_core  # noqa: E402

SLUG = "evidence_surplus"
PAIR_CACHE = os.path.join("scratch", SLUG, "pairs")
BACKDOORED = [
    "vit_cifar10_badnet_a2o_0_01",
    "vit_tiny_badnet_a2o_0_05",
    "vit_cifar100_blend_0_1",
    "vit_cifar10_blend_0_1",
    "vit_cifar10_bpp_0_05",
    "vit_gtsrb_lf_0_01",
    "vit_cifar10_wanet_0_1",
    "vit_tiny_wanet_0_05",
    "vit_gtsrb_tact_0_05",
    "vit_cifar10_tact_0_01",
    "resnet18_gtsrb_badnet_a2o_0_1",
    "resnet18_gtsrb_blend_0_1",
]
# Only the CIFAR-10 and GTSRB benign ViTs were named as controls, and the ResNet-18
# benign checkpoint has no args.json, so it cannot rebuild an eval set.
BENIGN = {
    "vit_cifar10_benign": ("badnet_a2o", "blend", "bpp", "wanet", "tact"),
    "vit_gtsrb_benign": ("lf", "tact"),
}
EVAL_IMAGES = 400
SEARCH_FACTOR = 4.0
BISECTION_STEPS = 12
BATCH_SIZE = 200
GPU_MEMORY_FRACTION = 0.35
SEED = 0
# WaNet's direction onset on ViT in experiments/backdoor_manifestation, for the
# depth-profile prediction.
WANET_ONSET = 7


def main():
    args = parse_args()
    if args.stage == "list":
        print("\n".join(run["name"] for run in panel_runs()))
        return
    if args.stage == "summarize":
        summarize(args.results_dir)
        return
    run = next(run for run in panel_runs() if run["name"] == args.run)
    if args.stage == "prepare":
        prepare(run, args.raw_data_dir)
        return
    probe(run, args.results_dir)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("list", "prepare", "probe", "summarize"))
    parser.add_argument("--run", default=None)
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def panel_runs():
    runs = []
    for folder in BACKDOORED:
        metadata = read_checkpoint_metadata(checkpoint_path(folder))
        runs.append(
            {
                "name": folder,
                "folder": folder,
                "architecture": metadata["architecture"],
                "dataset": metadata["dataset"],
                "attack": metadata["attack"],
                "benign": False,
                "probe_attack": None,
                "probe_target": None,
                "target": int(metadata["target_label"]),
            }
        )
    for folder, attacks in BENIGN.items():
        metadata = read_checkpoint_metadata(checkpoint_path(folder))
        for attack in attacks:
            runs.append(
                {
                    "name": f"{folder}__{attack}",
                    "folder": folder,
                    "architecture": metadata["architecture"],
                    "dataset": metadata["dataset"],
                    "attack": attack,
                    "benign": True,
                    "probe_attack": attack,
                    "probe_target": 0,
                    "target": 0,
                }
            )
    return runs


def pair_path(name):
    """The manifestation experiment's cache when it holds this run, same split and seed, else our own."""
    shared = os.path.join(MANIFESTATION_PAIRS, f"{name}.pt")
    path = shared if os.path.exists(shared) else os.path.join(PAIR_CACHE, f"{name}.pt")
    return path


def prepare(run, raw_data_dir):
    if os.path.exists(pair_path(run["name"])):
        print(f"{run['name']}: pairs cached at {pair_path(run['name'])}")
        return
    pairs = paired_images(run, raw_data_dir)
    os.makedirs(PAIR_CACHE, exist_ok=True)
    torch.save(pairs, os.path.join(PAIR_CACHE, f"{run['name']}.pt"))
    print(f"{run['name']}: {pairs['clean'].shape[0]} pairs of {pairs['n_eligible']}")


def probe(run, results_dir):
    seed_everything(SEED)
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)

    pairs = torch.load(pair_path(run["name"]))
    model = load_checkpoint(run["architecture"], checkpoint_path(run["folder"]), device)
    blocks = transformer_blocks(network_core(model), run["architecture"])
    layers = tuple(range(1, len(blocks) + 1))

    num_pairs = pairs["clean"].shape[0]
    half = num_pairs // 2
    eval_rows = torch.arange(half, min(num_pairs, half + EVAL_IMAGES))
    clean_features, clean_predictions = pooled_features(
        model, run, layers, pairs["clean"], device
    )
    triggered_features, triggered_predictions = pooled_features(
        model, run, layers, pairs["triggered"], device
    )

    eval_clean = pairs["clean"][eval_rows]  # (n, C, H, W)
    off_target = clean_predictions[eval_rows] != run["target"]  # (n,)
    generator = torch.Generator().manual_seed(SEED)
    rows = []
    for layer in layers:
        clean = clean_features[layer]  # (N, dim)
        triggered = triggered_features[layer]  # (N, dim)
        direction = (triggered[:half] - clean[:half]).mean(dim=0)  # (dim,)
        backdoor_unit = direction / direction.norm().clamp_min(1e-8)
        random_unit = torch.randn(direction.shape[0], generator=generator)
        random_unit = random_unit / random_unit.norm()
        shift = triggered[eval_rows] - clean[eval_rows]  # (n, dim)

        backdoor_actual = shift @ backdoor_unit  # (n,)
        search_limit = SEARCH_FACTOR * float(backdoor_actual.median().clamp_min(1e-6))
        row = {
            "layer": layer,
            "direction_norm": float(direction.norm()),
            "search_limit": search_limit,
        }
        for name, unit in (("backdoor", backdoor_unit), ("random", random_unit)):
            actual = shift @ unit  # (n,)
            needed = needed_scales(
                model,
                run,
                blocks[layer - 1],
                unit,
                eval_clean[off_target],
                search_limit,
                device,
            )  # (n_off,)
            row[name] = surplus_record(actual[off_target], needed)
        rows.append(row)

    record = {
        "run": run,
        "n_pairs": num_pairs,
        "n_eval": int(eval_rows.numel()),
        "n_eval_off_target": int(off_target.sum()),
        "asr_eval": float(
            (triggered_predictions[eval_rows] == run["target"]).float().mean()
        ),
        "clean_accuracy_eval": float(
            (clean_predictions[eval_rows] == pairs["labels"][eval_rows]).float().mean()
        ),
        "search_factor": SEARCH_FACTOR,
        "bisection_steps": BISECTION_STEPS,
        "layers": rows,
    }
    directory = os.path.join(experiment_results_dir(SLUG, results_dir), "runs")
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{run['name']}.json"), "w") as handle:
        json.dump(record, handle, indent=1)
    last = rows[-1]["backdoor"]
    print(
        f"{run['name']:34} asr {record['asr_eval']:.3f} last-block surplus median "
        f"{last['median_surplus']} reachable {last['reachable_share']:.3f}"
    )


def pool(activation, architecture):
    """The pooled feature the head would read: class token on ViT, spatial mean on ResNet-18."""
    if architecture == "vit":
        pooled = activation[:, 0, :].float()  # (batch, dim)
        return pooled
    if architecture == "resnet18":
        pooled = activation.float().mean(dim=(2, 3))  # (batch, channels)
        return pooled
    raise ValueError(f"no pooling rule for {architecture}")


@torch.inference_mode()
def pooled_features(model, run, layers, images, device):
    features = {layer: [] for layer in layers}
    predictions = []
    with captured_layers(model, layers, run["architecture"]) as captured:
        for batch in images.split(BATCH_SIZE):
            logits = forward_logits(model, batch, device, True)  # (batch, classes)
            predictions.append(logits.argmax(dim=1).cpu())
            for layer in layers:
                features[layer].append(pool(captured[layer], run["architecture"]).cpu())

    stacked = {
        layer: torch.cat(chunks) for layer, chunks in features.items()
    }  # (N, dim) each
    all_predictions = torch.cat(predictions)  # (N,)
    return stacked, all_predictions


def shift_hook(unit, scales, architecture):
    """Add scales[i] * unit to every token (ViT) or spatial position (ResNet) of image i."""

    def hook(_module, _inputs, output):
        direction = unit.to(output.device, output.dtype)
        amount = scales.to(output.device, output.dtype)  # (batch,)
        if architecture == "vit":
            shifted = output + amount[:, None, None] * direction[None, None, :]
            return shifted
        shifted = output + amount[:, None, None, None] * direction[None, :, None, None]
        return shifted

    return hook


@torch.inference_mode()
def lands_on_target(model, run, block, unit, images, scales, device):
    handle = block.register_forward_hook(shift_hook(unit, scales, run["architecture"]))
    try:
        logits = forward_logits(model, images, device, True)  # (batch, classes)
    finally:
        handle.remove()

    landed = (logits.argmax(dim=1) == run["target"]).cpu()  # (batch,)
    return landed


def needed_scales(model, run, block, unit, images, search_limit, device):
    """Per image, the smallest scale that sends it to the target, NaN when the search limit does not."""
    chunks = []
    for batch in images.split(BATCH_SIZE):
        count = batch.shape[0]
        high = torch.full((count,), search_limit)
        reachable = lands_on_target(model, run, block, unit, batch, high, device)
        low = torch.zeros(count)
        for _ in range(BISECTION_STEPS):
            middle = (low + high) / 2
            landed = lands_on_target(model, run, block, unit, batch, middle, device)
            high = torch.where(landed, middle, high)
            low = torch.where(landed, low, middle)
        needed = torch.where(reachable, high, torch.full((count,), float("nan")))
        chunks.append(needed)

    scales = torch.cat(chunks) if chunks else torch.zeros(0)  # (n,)
    return scales


def surplus_record(actual, needed):
    reachable = ~torch.isnan(needed)
    surplus = actual[reachable] / needed[reachable]  # (n_reachable,)
    record = {
        "n": int(actual.numel()),
        "median_actual": float(actual.median()) if actual.numel() else None,
        "reachable_share": float(reachable.float().mean()) if actual.numel() else 0.0,
        "median_needed": float(needed[reachable].median()) if reachable.any() else None,
        "median_surplus": round(float(surplus.median()), 4)
        if reachable.any()
        else None,
        "surplus_quartiles": [
            round(float(q), 4) for q in surplus.quantile(torch.tensor([0.25, 0.75]))
        ]
        if surplus.numel() > 1
        else None,
    }
    return record


def summarize(results_dir):
    from cli.compare_detectors import psbd_values
    from scripts.paper._common import load_psbd_metrics

    directory = experiment_results_dir(SLUG, results_dir)
    records = {
        os.path.basename(path)[:-5]: json.load(open(path))
        for path in sorted(glob.glob(os.path.join(directory, "runs", "*.json")))
    }
    rows = []
    for name, record in records.items():
        run = record["run"]
        last = record["layers"][-1]
        placement = (
            RECOMMENDED_PLACEMENT
            if run["architecture"] == "vit"
            else PUBLISHED_PLACEMENT
        )
        report = (
            None if run["benign"] else load_psbd_metrics(results_dir, run["folder"])
        )
        values = psbd_values(report, placement, "adaptive") if report else None
        rows.append(
            {
                "name": name,
                "architecture": run["architecture"],
                "attack": run["attack"],
                "benign": run["benign"],
                "asr_eval": record["asr_eval"],
                "last_block_surplus": last["backdoor"]["median_surplus"],
                "last_block_reachable": last["backdoor"]["reachable_share"],
                "last_block_random_surplus": last["random"]["median_surplus"],
                "last_block_random_reachable": last["random"]["reachable_share"],
                "psbd_placement": placement if values else None,
                "psbd_tpr_at_1pct_fpr": values["q0.01"]["tpr"] if values else None,
                "psbd_auroc": values["q0.25"]["auroc"] if values else None,
                "depth_surplus": [
                    row["backdoor"]["median_surplus"] for row in record["layers"]
                ],
                "depth_reachable": [
                    row["backdoor"]["reachable_share"] for row in record["layers"]
                ],
            }
        )

    vit = [
        row
        for row in rows
        if row["architecture"] == "vit"
        and not row["benign"]
        and row["last_block_surplus"] is not None
    ]
    correlations = {}
    for key in ("psbd_tpr_at_1pct_fpr", "psbd_auroc"):
        rho, p_value = spearmanr(
            [row["last_block_surplus"] for row in vit], [row[key] for row in vit]
        )
        correlations[key] = {
            "spearman_rho": float(rho),
            "p_value": float(p_value),
            "n": len(vit),
        }
    summary = {"rows": rows, "correlations": correlations}
    with open(os.path.join(directory, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1)

    figures = os.path.join(directory, "figures")
    os.makedirs(figures, exist_ok=True)
    draw_last_block(rows, figures)
    draw_correlations(vit, correlations, figures)
    draw_depth(rows, figures)
    for row in rows:
        print(
            f"{row['name']:34} surplus {row['last_block_surplus']} reach {row['last_block_reachable']:.2f} "
            f"random {row['last_block_random_surplus']} reach {row['last_block_random_reachable']:.2f} "
            f"tpr1 {row['psbd_tpr_at_1pct_fpr']} auroc {row['psbd_auroc']}"
        )
    print(correlations)


def save(figure, figures, name):
    figure.tight_layout()
    figure.savefig(os.path.join(figures, f"{name}.png"), dpi=150)
    figure.savefig(os.path.join(figures, f"{name}.pdf"))
    plt.close(figure)


def draw_last_block(rows, figures):
    figure, axis = plt.subplots(figsize=(7.0, 3.2))
    names = [row["name"] for row in rows]
    positions = np.arange(len(rows))
    backdoor = [row["last_block_surplus"] or np.nan for row in rows]
    random = [row["last_block_random_surplus"] or np.nan for row in rows]
    colours = ["grey" if row["benign"] else "tab:orange" for row in rows]
    axis.scatter(
        positions,
        backdoor,
        color=colours,
        label="backdoor direction (grey: benign model)",
    )
    axis.scatter(positions, random, marker="x", color="black", label="random direction")
    axis.axhline(1.0, color="grey", linestyle=":")
    axis.set_yscale("symlog", linthresh=0.1)
    axis.set_xticks(positions, names, rotation=90, fontsize=6)
    axis.set_ylabel("median surplus factor, last block")
    axis.legend(fontsize=6)
    save(figure, figures, "last_block_surplus")


def draw_correlations(vit, correlations, figures):
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    surplus = [row["last_block_surplus"] for row in vit]
    for axis, key, label in (
        (axes[0], "psbd_tpr_at_1pct_fpr", "PSBD-TM TPR at 1% FPR"),
        (axes[1], "psbd_auroc", "PSBD-TM AUROC"),
    ):
        axis.scatter(surplus, [row[key] for row in vit])
        for row in vit:
            axis.annotate(
                row["name"].replace("vit_", ""),
                (row["last_block_surplus"], row[key]),
                fontsize=5,
            )
        axis.set_xscale("log")
        axis.set_xlabel("median surplus factor, last block")
        axis.set_ylabel(label)
        stats = correlations[key]
        axis.set_title(
            f"Spearman {stats['spearman_rho']:+.2f}, p {stats['p_value']:.2f}, n {stats['n']}",
            fontsize=7,
        )
    save(figure, figures, "surplus_against_psbd")


def draw_depth(rows, figures):
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for axis, architecture in zip(axes, ("vit", "resnet18")):
        for row in rows:
            if row["architecture"] != architecture:
                continue
            depth = np.arange(1, len(row["depth_surplus"]) + 1)
            values = [
                value if value is not None else np.nan for value in row["depth_surplus"]
            ]
            style = "--" if row["benign"] else "-"
            width = 2.0 if row["attack"] == "wanet" and not row["benign"] else 0.8
            axis.plot(
                depth, values, linestyle=style, linewidth=width, label=row["name"]
            )
        if architecture == "vit":
            axis.axvline(WANET_ONSET, color="grey", linestyle=":")
        axis.axhline(1.0, color="grey", linestyle=":")
        axis.set_yscale("symlog", linthresh=0.1)
        axis.set_xlabel("block")
        axis.set_ylabel("median surplus factor")
        axis.set_title(
            "ViT-B/16 (WaNet thick, benign dashed)"
            if architecture == "vit"
            else "ResNet-18",
            fontsize=7,
        )
    axes[0].legend(fontsize=4)
    axes[1].legend(fontsize=5)
    save(figure, figures, "depth_profile")


if __name__ == "__main__":
    main()
