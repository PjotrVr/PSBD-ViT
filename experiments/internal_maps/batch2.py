"""Second batch GPU measurements on ViT-B/16, 1 subcommand per question.

confidence  which tokens raise the predicted class's probability when masked, on
            clean images whose PSBD-TM PSU is negative, against triggered images
            whose PSU is near 0, and whether those tokens are ones the class
            token ignores
veto        on single-source TaCT models, which tokens of a stamped non-source
            image keep the trigger from reaching the target, and the
            content-hidden probe scored as a detector
survival    per PSBD-TM pass, how many trigger tokens stayed visible in every
            block and whether the triggered prediction survived, at several rates

flock -E 75 scratch/gpu.lock .venv/bin/python -m experiments.internal_maps.batch2 veto
"""

import argparse
import os
import time
import types

import torch
from lightning import seed_everything

from attacks import apply_config_overrides, build_attack, default_config
from cli.sweep import load_model_and_loaders
from data.registry import DATASET_REGISTRY
from defenses.decision import RECOMMENDED_PLACEMENT
from defenses.inference import forward_logits
from experiments._paths import experiment_results_dir
from experiments.internal_maps import config
from experiments.internal_maps.gallery import read_tm_psu
from experiments.internal_maps.measure import (
    GridKeep,
    git_commit,
    masked_readout,
    plug_grid_keep,
    vit_attention,
    write_json,
)
from experiments.why_token_masking_works.measure import RecordingTokenMask
from experiments.why_token_masking_works.tokens import (
    model_seed,
    touched_tokens,
    trigger_pixel_map,
)
from models.positions import plug_dropout, unplug_dropout

SLUG = "internal_maps"
GRID = 14
UNITS = GRID * GRID
CPU_THREADS = 4
GPU_MEMORY_FRACTION = 0.3
TM_POSITION = "before_attention_norm"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", choices=("confidence", "veto", "survival"))
    parser.add_argument("--models", default="", help="comma separated, default all")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(CPU_THREADS)
    device = torch.device("cuda", 0)
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
    output_root = experiment_results_dir(SLUG)

    measure = {
        "confidence": measure_confidence_gain,
        "veto": measure_tact_veto,
        "survival": measure_survival,
    }[args.question]
    folders = {
        "confidence": config.CONFIDENCE_MODELS,
        "veto": config.VETO_MODELS,
        "survival": config.SURVIVAL_MODELS,
    }[args.question]
    selected = [name for name in args.models.split(",") if name] or list(folders)
    for folder in selected:
        out_path = os.path.join(output_root, "vit", folder, f"{args.question}.json")
        if os.path.exists(out_path):
            print(f"[skip] {folder}", flush=True)
            continue
        started = time.time()
        record = measure(folder, device)
        record["folder"] = folder
        record["git_commit"] = git_commit()
        record["seconds"] = round(time.time() - started, 1)
        write_json(out_path, record)
        print(f"[ok] {args.question} {folder} {record['seconds']}s", flush=True)


def load_everything(folder, device):
    loader_args = types.SimpleNamespace(
        checkpoints_dir="checkpoints",
        raw_data_dir="raw_data",
        batch_size=64,
        num_workers=0,
        max_samples=None,
        probe_attack=None,
        probe_target_label=None,
    )
    model, architecture, loaders, manifest, metadata = load_model_and_loaders(
        loader_args, folder, device
    )
    assert architecture == "vit", "the second batch runs on ViT-B/16"
    datasets = {split: loader.dataset for split, loader in loaders.items()}
    return model, datasets, manifest, metadata


def stack_rows(dataset, rows, device):
    items = [dataset[row] for row in rows]
    images = torch.stack([image for image, _ in items]).to(device)  # (n, 3, s, s)
    labels = torch.tensor([int(label) for _, label in items])  # (n,)
    return images, labels


# 1 unit hidden in every block, for every unit. Returns the change in the given
# class's probability, (units, n), positive where hiding the unit raises it.
def single_unit_change(model, keeper, images, classes):
    nothing = torch.ones(1, GRID, GRID, device=images.device)
    base, _ = masked_readout(model, keeper, images, nothing, classes)  # (1, n)
    single = torch.ones(UNITS, GRID, GRID, device=images.device)
    single.view(UNITS, UNITS)[torch.arange(UNITS), torch.arange(UNITS)] = 0.0
    hidden, _ = masked_readout(model, keeper, images, single, classes)  # (units, n)
    change = hidden - base  # (units, n)
    return change, base[0]


# Hides, per image, its own top-k units by single-unit change, all at once, against as
# many random units as the control. Returns the probability and the share whose
# argmax is the class, for every k.
def joint_top_k(model, keeper, images, classes, change, counts, seed):
    result = {}
    order = torch.argsort(change, dim=0, descending=True)  # (units, n)
    generator = torch.Generator().manual_seed(seed)
    for count in counts:
        top_masks = torch.ones(len(images), UNITS, device=images.device)
        random_masks = torch.ones(len(images), UNITS, device=images.device)
        for image in range(len(images)):
            top_masks[image, order[:count, image].to(images.device)] = 0.0
            random_units = torch.randperm(UNITS, generator=generator)[:count]
            random_masks[image, random_units.to(images.device)] = 0.0
        result[str(count)] = {
            "top": per_image_masked(model, keeper, images, classes, top_masks),
            "random": per_image_masked(model, keeper, images, classes, random_masks),
        }
    return result


# Each image under its own mask, (n, units) masks for (n, ...) images.
@torch.inference_mode()
def per_image_masked(model, keeper, images, classes, masks):
    keeper.keep = masks.reshape(len(images), GRID, GRID)
    logits = forward_logits(model, images, images.device, use_bfloat16=True)
    keeper.keep = None
    probability = logits.float().softmax(dim=1)
    chosen = probability.gather(1, classes[:, None])[:, 0]  # (n,)
    reading = {
        "probability": float(chosen.mean()),
        "argmax_is_class": float((probability.argmax(dim=1) == classes).float().mean()),
    }
    return reading


def last_block_class_attention(model, images, device):
    attention = vit_attention(model, images, device)[-1]  # (n, heads, 197, 197)
    class_row = attention.mean(dim=1)[:, 0, 1:]  # (n, 196)
    return class_row.T.cpu()  # (196, n)


# How the rise of a confidence-gain map sits against the class token's own
# attention in the last block. If masking a token raises the prediction because
# the token is a distractor, the rising tokens should be ones the class token
# barely reads.
def rise_against_attention(change, attention):
    rise = change.clamp_min(0)  # (units, n)
    ranks = attention.argsort(dim=0).argsort(dim=0).float() / (UNITS - 1)  # (units, n)
    top_quarter = ranks >= 0.75
    rise_total = rise.sum(dim=0).clamp_min(1e-12)  # (n,)
    share_on_top_quarter = (rise * top_quarter).sum(dim=0) / rise_total  # (n,)
    rise_weighted_rank = (rise * ranks).sum(dim=0) / rise_total  # (n,)
    drop = (-change).clamp_min(0)
    drop_weighted_rank = (drop * ranks).sum(dim=0) / drop.sum(dim=0).clamp_min(1e-12)
    rows = torch.arange(GRID)
    border = ((rows[:, None] < 2) | (rows[:, None] >= GRID - 2)) | (
        (rows[None, :] < 2) | (rows[None, :] >= GRID - 2)
    )  # (g, g)
    border_share = (rise * border.flatten()[:, None]).sum(dim=0) / rise_total
    reading = {
        "rise_share_on_top_attention_quarter": share_on_top_quarter.tolist(),
        "rise_weighted_attention_rank": rise_weighted_rank.tolist(),
        "drop_weighted_attention_rank": drop_weighted_rank.tolist(),
        "rise_share_on_border": border_share.tolist(),
        "border_area_share": float(border.float().mean()),
        "total_rise": rise.sum(dim=0).tolist(),
        "total_drop": drop.sum(dim=0).tolist(),
    }
    return reading


def measure_confidence_gain(folder, device):
    model, datasets, manifest, metadata = load_everything(folder, device)
    psu, baselines, rate, rule, _ = read_tm_psu(folder)
    target = int(metadata["target_label"])
    validation_predictions = baselines["validation"][1]
    backdoor_predictions = baselines["backdoor"][1]

    negative_rows = (psu["validation"] < 0).nonzero(as_tuple=True)[0]
    negative_rows = negative_rows[torch.argsort(psu["validation"][negative_rows])]
    negative_rows = negative_rows[: config.CONFIDENCE_IMAGES]
    hit_rows = (backdoor_predictions == target).nonzero(as_tuple=True)[0]
    near_zero_rows = hit_rows[torch.argsort(psu["backdoor"][hit_rows].abs())]
    near_zero_rows = near_zero_rows[: config.CONFIDENCE_IMAGES]

    keeper = GridKeep()
    handles = plug_grid_keep(model, "vit", keeper)
    groups = {}
    try:
        for name, dataset, rows, predictions, scores in (
            (
                "clean_negative_psu",
                datasets["validation"],
                negative_rows,
                validation_predictions,
                psu["validation"],
            ),
            (
                "triggered_near_zero_psu",
                datasets["backdoor"],
                near_zero_rows,
                backdoor_predictions,
                psu["backdoor"],
            ),
        ):
            if len(rows) == 0:
                groups[name] = {"count": 0}
                continue
            images, labels = stack_rows(dataset, rows.tolist(), device)
            classes = predictions[rows].to(device)
            change, base = single_unit_change(model, keeper, images, classes)
            # The keeper passes everything through while its keep map is None.
            attention = last_block_class_attention(model, images, device)
            joint = joint_top_k(
                model,
                keeper,
                images,
                classes,
                change,
                config.JOINT_COUNTS,
                model_seed(folder),
            )
            groups[name] = {
                "count": int(len(rows)),
                "rows": rows.tolist(),
                "labels": labels.tolist(),
                "predicted": classes.cpu().tolist(),
                "psu": scores[rows].tolist(),
                "base_probability": base.tolist(),
                "change_maps": change.T.reshape(-1, GRID, GRID)[
                    : config.MAPS_DRAWN
                ].tolist(),
                "attention_maps": attention.T.reshape(-1, GRID, GRID)[
                    : config.MAPS_DRAWN
                ].tolist(),
                "largest_single_rise": change.max(dim=0).values.tolist(),
                "against_attention": rise_against_attention(change, attention),
                "joint_top_k": joint,
            }
    finally:
        unplug_dropout(handles)
    mean, std = image_statistics(manifest, metadata["dataset"])
    record = {
        "target_label": target,
        "rate": rate,
        "rate_rule": rule,
        "negative_psu_validation_count": int((psu["validation"] < 0).sum()),
        "validation_count": int(len(psu["validation"])),
        "normalization": {"mean": list(mean), "std": list(std)},
        "groups": groups,
        "example_pixels": {
            name: example_pixels(
                datasets[split],
                groups[name].get("rows", [])[: config.MAPS_DRAWN],
                mean,
                std,
            )
            for name, split in (
                ("clean_negative_psu", "validation"),
                ("triggered_near_zero_psu", "backdoor"),
            )
        },
    }
    return record


def measure_tact_veto(folder, device):
    model, datasets, manifest, metadata = load_everything(folder, device)
    target = int(metadata["target_label"])
    attack = build_attack(
        metadata["attack"],
        apply_config_overrides(
            default_config(metadata["attack"]), metadata.get("attack_config_overrides")
        ),
        DATASET_REGISTRY[metadata["dataset"]].image_size,
        target,
    )
    sources = tuple(attack.source_classes)
    plant = attack.apply_trigger_eval or attack.apply_trigger
    mean, std = image_statistics(manifest, metadata["dataset"])
    trigger_units = touched_tokens(trigger_pixel_map(metadata), GRID)  # (count,)

    clean_set = datasets["clean"]
    _, clean_cache_predictions, _ = read_tm_psu(folder)[1]["clean"]
    non_source_rows = []
    for row in range(len(clean_set)):
        label = int(clean_set[row][1])
        if label in sources or label == target:
            continue
        if int(clean_cache_predictions[row]) != label:
            continue
        non_source_rows.append(row)
        if len(non_source_rows) == config.VETO_POOL:
            break
    clean_images, clean_labels = stack_rows(clean_set, non_source_rows, device)
    stamped = stamp(clean_images, plant, mean, std)  # (n, 3, s, s)
    targets = torch.full((len(stamped),), target, device=device)

    keeper = GridKeep()
    handles = plug_grid_keep(model, "vit", keeper)
    try:
        base_target, base_kept = masked_readout(
            model, keeper, stamped, torch.ones(1, GRID, GRID, device=device), targets
        )
        refused = (~base_kept[0]).nonzero(as_tuple=True)[0][: config.VETO_IMAGES]
        refused_images = stamped[refused.to(device)]
        refused_targets = targets[: len(refused)]
        veto_change, veto_base = single_unit_change(
            model, keeper, refused_images, refused_targets
        )
        veto_joint = joint_top_k(
            model,
            keeper,
            refused_images,
            refused_targets,
            veto_change,
            config.VETO_COUNTS,
            model_seed(folder),
        )
        # The same per-image masks on the unstamped twin show whether hiding those
        # tokens sends any image to the target, trigger or not.
        unstamped = clean_images[refused.to(device)]
        unstamped_joint = joint_top_k(
            model,
            keeper,
            unstamped,
            refused_targets,
            veto_change,
            config.VETO_COUNTS,
            model_seed(folder),
        )

        backdoor_rows = list(range(min(config.VETO_IMAGES, len(datasets["backdoor"]))))
        source_images, _ = stack_rows(datasets["backdoor"], backdoor_rows, device)
        source_targets = torch.full((len(source_images),), target, device=device)
        source_change, source_base = single_unit_change(
            model, keeper, source_images, source_targets
        )
        probe = content_hidden_probe(
            model, keeper, datasets, manifest, trigger_units, device
        )
    finally:
        unplug_dropout(handles)

    record = {
        "target_label": target,
        "source_classes": list(sources),
        "trigger_units": trigger_units.tolist(),
        "non_source_stamped": int(len(stamped)),
        "non_source_sent_to_target": float(base_kept.float().mean()),
        "refused_count": int(len(refused)),
        "refused": {
            "base_target_probability": veto_base.tolist(),
            "change_maps_mean": veto_change.mean(dim=1).reshape(GRID, GRID).tolist(),
            "change_maps": veto_change.T.reshape(-1, GRID, GRID)[
                : config.MAPS_DRAWN
            ].tolist(),
            "largest_single_rise": veto_change.max(dim=0).values.tolist(),
            "rise_share_on_trigger": share_on_units(veto_change, trigger_units),
            "units_for_half_rise": units_for_half(veto_change),
            "joint_top_k": veto_joint,
            "unstamped_joint_top_k": unstamped_joint,
        },
        "source": {
            "count": int(len(source_images)),
            "base_target_probability": source_base.tolist(),
            "change_maps_mean": source_change.mean(dim=1).reshape(GRID, GRID).tolist(),
            "largest_single_drop": (-source_change).max(dim=0).values.tolist(),
        },
        "content_hidden_probe": probe,
        "example_pixels": {
            "stamped": example_pixels_from_tensor(
                refused_images[: config.MAPS_DRAWN], mean, std
            ),
            "source": example_pixels_from_tensor(
                source_images[: config.MAPS_DRAWN], mean, std
            ),
        },
    }
    return record


def stamp(images, plant, mean, std):
    mean_t = torch.tensor(mean, device=images.device)[:, None, None]
    std_t = torch.tensor(std, device=images.device)[:, None, None]
    raw = (images * std_t + mean_t).clamp(0, 1).cpu()  # (n, 3, s, s) in 0 to 1
    planted = torch.stack([plant(image.clone(), 0) for image in raw]).to(images.device)
    normalized = (planted - mean_t) / std_t
    return normalized


def share_on_units(change, units):
    rise = change.clamp_min(0)  # (units, n)
    share = rise[units].sum(dim=0) / rise.sum(dim=0).clamp_min(1e-12)  # (n,)
    return share.tolist()


# The fewest units whose single-unit rises add up to half of the image's total rise.
def units_for_half(change):
    rise = change.clamp_min(0)
    ordered = rise.sort(dim=0, descending=True).values  # (units, n)
    cumulative = ordered.cumsum(dim=0) / ordered.sum(dim=0).clamp_min(1e-12)
    counts = (cumulative < 0.5).sum(dim=0) + 1  # (n,)
    return counts.tolist()


# The content-hidden probe of PREDICTIONS_tact_content_hidden.md, on the 2000
# clean validation images, every triggered analysis image and its clean twin.
def content_hidden_probe(model, keeper, datasets, manifest, trigger_units, device):
    keep = torch.zeros(1, UNITS, device=device)
    keep[0, trigger_units.to(device)] = 1.0
    keep = keep.reshape(1, GRID, GRID)
    clean_row_of = {o: r for r, o in enumerate(manifest["analysis_clean_indices"])}
    backdoor_rows = list(range(len(datasets["backdoor"])))
    twin_rows = [
        clean_row_of[manifest["analysis_backdoor_indices"][r]] for r in backdoor_rows
    ]
    scores = {
        "validation": probe_scores(
            model,
            keeper,
            datasets["validation"],
            range(len(datasets["validation"])),
            keep,
            device,
        ),
        "triggered": probe_scores(
            model, keeper, datasets["backdoor"], backdoor_rows, keep, device
        ),
        "clean": probe_scores(
            model, keeper, datasets["clean"], twin_rows, keep, device
        ),
    }
    reading = {"count": {k: len(v) for k, v in scores.items()}}
    for budget in (0.01, 0.05, 0.10):
        threshold = float(torch.quantile(scores["validation"], budget))
        reading[f"q{budget:.2f}"] = {
            "threshold": threshold,
            "tpr": float((scores["triggered"] <= threshold).float().mean()),
            "realized_fpr": float((scores["clean"] <= threshold).float().mean()),
        }
    reading["auroc"] = low_is_positive_auroc(scores["triggered"], scores["clean"])
    reading["score_means"] = {k: float(v.mean()) for k, v in scores.items()}
    return reading


@torch.inference_mode()
def probe_scores(model, keeper, dataset, rows, keep, device):
    rows = list(rows)
    scores = []
    for start in range(0, len(rows), 128):
        images, _ = stack_rows(dataset, rows[start : start + 128], device)
        keeper.keep = None
        base = forward_logits(model, images, device, use_bfloat16=True).softmax(dim=1)
        predicted = base.argmax(dim=1)  # (b,)
        keeper.keep = keep.expand(len(images), -1, -1)
        hidden = forward_logits(model, images, device, use_bfloat16=True).softmax(dim=1)
        keeper.keep = None
        ratio = (
            hidden.gather(1, predicted[:, None])[:, 0]
            / base.gather(1, predicted[:, None])[:, 0]
        )
        scores.append((1.0 - ratio).cpu())
    return torch.cat(scores)


def low_is_positive_auroc(positive, negative):
    below = (positive[:, None] < negative[None, :]).float().mean()
    ties = (positive[:, None] == negative[None, :]).float().mean()
    auroc = float(below + 0.5 * ties)
    return auroc


def measure_survival(folder, device):
    model, datasets, manifest, metadata = load_everything(folder, device)
    target = int(metadata["target_label"])
    trigger = touched_tokens(trigger_pixel_map(metadata), GRID) + 1  # (m,), CLS at 0
    _, backdoor_predictions, _ = read_tm_psu(folder)[1]["backdoor"]
    hit_rows = (backdoor_predictions == target).nonzero(as_tuple=True)[0]
    rows = hit_rows[: config.SURVIVAL_IMAGES].tolist()
    images, _ = stack_rows(datasets["backdoor"], rows, device)
    with torch.inference_mode():
        base = forward_logits(model, images, device, use_bfloat16=True).softmax(dim=1)[
            :, target
        ]  # (n,)

    by_rate = {}
    for rate in config.SURVIVAL_RATES:
        recorders = []

        def build_recorder(probe_rate):
            recorder = RecordingTokenMask(probe_rate)
            recorders.append(recorder)
            return recorder

        handles = plug_dropout(
            model, "vit", (TM_POSITION,), {TM_POSITION: build_recorder}, rate
        )
        visible_by_pass, survived, ratio = [], [], []
        try:
            seed_everything(model_seed(folder) + int(rate * 100), verbose=False)
            for _ in range(config.SURVIVAL_PASSES):
                pass_visible, pass_probability = [], []
                for start in range(0, len(images), 128):
                    batch = images[start : start + 128]
                    with torch.inference_mode():
                        probability = forward_logits(
                            model, batch, device, use_bfloat16=True
                        ).softmax(dim=1)
                    dropped = torch.stack(
                        [r.last_dropped for r in recorders]
                    )  # (12, b, 197)
                    visible = (~dropped[:, :, trigger.to(device)]).sum(dim=2)  # (12, b)
                    pass_visible.append(visible.T.cpu())  # (b, 12)
                    pass_probability.append(probability.cpu())
                probability = torch.cat(pass_probability)  # (n, classes)
                visible_by_pass.append(torch.cat(pass_visible))  # (n, 12)
                survived.append(probability.argmax(dim=1) == target)
                ratio.append(probability[:, target] / base.cpu())
        finally:
            unplug_dropout(handles)
        by_rate[str(rate)] = {
            "visible_per_block": torch.stack(visible_by_pass).tolist(),  # (k, n, 12)
            "survived": torch.stack(survived).int().tolist(),  # (k, n)
            "target_ratio": torch.stack(ratio).tolist(),  # (k, n)
        }
    record = {
        "target_label": target,
        "trigger_tokens": int(len(trigger)),
        "images": int(len(rows)),
        "passes": config.SURVIVAL_PASSES,
        "placement": RECOMMENDED_PLACEMENT,
        "by_rate": by_rate,
    }
    return record


def image_statistics(manifest, dataset):
    stored = manifest.get("normalization")
    if stored:
        return stored["mean"], stored["std"]
    spec = DATASET_REGISTRY[dataset]
    return spec.mean, spec.std


def example_pixels(dataset, rows, mean, std):
    if not rows:
        return []
    images = torch.stack([dataset[row][0] for row in rows])
    pixels = example_pixels_from_tensor(images, mean, std)
    return pixels


def example_pixels_from_tensor(images, mean, std):
    mean_t = torch.tensor(mean)[:, None, None]
    std_t = torch.tensor(std)[:, None, None]
    restored = images.float().cpu() * std_t + mean_t  # (n, 3, s, s)
    pixels = (restored.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1)
    return pixels.tolist()


if __name__ == "__main__":
    main()
