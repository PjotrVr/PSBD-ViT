"""Null-referenced PSU: subtract the evidence a class gets for free from the masks.

For a model at PSBD-TM's adaptive rate, a content-free input (gray, black or the
fully masked input) is run through the same masks. Its per-class probability b_c
and margin m0_c (logit_c minus the best other logit) are the free evidence for
class c. With c the unperturbed argmax and K masked passes,

    phi_prob  = 1 - mean_k max(P_c(x_k) - b_c, 0)  / max(P_c(x) - b_c, eps_prob)
    phi_logit = 1 - mean_k max(m_k(x_k) - m0_c, 0) / max(m(x) - m0_c, eps_logit)

Thresholds are clean validation quantiles, nothing is fitted on triggered data.

    null     any device, the content-free outputs of 1 model
    dump     GPU under the shared lock, fresh 3-pass probabilities and margins of
             the validation, clean and backdoor splits of 1 model
    analyze  CPU, the probability version from the cached sweep, the logit version
             from the dumps, per model and aggregated with paired bootstrap CIs
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from lightning import seed_everything

from cli.sweep import PSBD_MASK_SEED, bound_operator
from data.registry import DATASET_REGISTRY
from data.splits import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from defenses.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defenses.decision import (
    RECOMMENDED_PLACEMENT,
    detection_report,
    pair_clean_to_backdoor,
)
from defenses.inference import forward_logits
from experiments._paths import experiment_results_dir
from experiments.evidence_surplus.false_positives.attractor_split import (
    attractor_class,
)
from experiments.evidence_surplus.surplus_factor.removal import MODELS as A3_MODELS
from models.backbones import detect_architecture, load_checkpoint
from models.positions import plug_dropout, unplug_dropout
from scripts.paper._common import (
    bootstrap_ci,
    clearing_cells,
    load_coverage,
)

SLUG = "evidence_surplus"
POSITION = "before_attention_norm"
OPERATOR = "token_mask"
NULLS = ("gray", "black", "full_mask")
PRIMARY_NULL = "full_mask"
NULL_SEEDS = tuple(range(8))
PASSES = 3
EPS_PROB = 0.01
EPS_LOGIT = 0.5
QUANTILES = (0.01, 0.05, 0.10)
BOOTSTRAP_RESAMPLES = 10000
GPU_MEMORY_FRACTION = 0.15
BATCH_SIZE = 128


class AllPatchMask(nn.Module):
    """Zero every patch token and keep the class token, with no rescaling.

    Rate 1 in token_mask would divide by 0 in the inverted-dropout scale, and
    scaling a mask that keeps nothing has no meaning.
    """

    def __init__(self, rate):
        super().__init__()

    def forward(self, x):
        keep = torch.zeros(1, x.shape[1], 1, device=x.device, dtype=x.dtype)
        keep[:, 0, :] = 1.0
        return x * keep


def main():
    args = parse_args()
    if args.stage == "analyze":
        analyze(args.results_dir)
        return
    if args.stage == "render":
        render(args.results_dir)
        return
    folders = args.models or A3_MODELS
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
    else:
        torch.set_num_threads(args.threads)
    for folder in folders:
        if args.stage == "null":
            null_stage(folder, device, args.results_dir)
        else:
            dump_stage(folder, device, args)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("null", "dump", "analyze", "render"))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--panel", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    if args.panel:
        args.models = panel_folders(args.results_dir)
    return args


def out_dir(results_dir, name):
    path = os.path.join(
        experiment_results_dir(SLUG, results_dir), "false_positives", "null_psu", name
    )
    os.makedirs(path, exist_ok=True)
    return path


def panel_folders(results_dir):
    cells = clearing_cells(load_coverage(results_dir))
    return [cell["folder_name"] for cell in cells]


def adaptive_rate(results_dir, folder):
    with open(os.path.join(results_dir, folder, "psbd_metrics.json")) as f:
        metrics = json.load(f)
    return float(metrics["placements"][RECOMMENDED_PLACEMENT]["adaptive_rate"])


def class_margins(logits):
    # margin_c = logit_c minus the best other logit, for every class at once.
    top2 = logits.topk(2, dim=1).values  # (n, 2)
    best_other = torch.where(logits == top2[:, :1], top2[:, 1:2], top2[:, :1])  # (n, C)
    margins = logits - best_other  # (n, C)
    return margins


def load_model(folder, device):
    checkpoint_path = os.path.join("checkpoints", folder, "attack_result.pt")
    architecture = detect_architecture(checkpoint_path)
    model = load_checkpoint(architecture, checkpoint_path, device)
    return model, architecture, checkpoint_path


def plug_masks(model, architecture, dataset, rate, full_mask):
    factory = AllPatchMask if full_mask else bound_operator(OPERATOR, dataset)
    handles = plug_dropout(model, architecture, (POSITION,), {POSITION: factory}, rate)
    return handles


@torch.inference_mode()
def null_stage(folder, device, results_dir):
    path = os.path.join(out_dir(results_dir, "nulls"), f"{folder}.{device.type}.json")
    if os.path.exists(path):
        return
    rate = adaptive_rate(results_dir, folder)
    model, architecture, checkpoint_path = load_model(folder, device)
    dataset = read_checkpoint_metadata(checkpoint_path)["dataset"]
    spec = DATASET_REGISTRY[dataset]
    mean = torch.tensor(spec.mean).view(3, 1, 1)
    std = torch.tensor(spec.std).view(3, 1, 1)
    size = spec.image_size
    # Model input is the normalized image, so a pixel value p enters as (p - mean) / std.
    pixel = {"gray": 0.5, "black": 0.0, "full_mask": 0.5}
    record = {"model": folder, "dataset": dataset, "adaptive_rate": rate}
    record["device"] = device.type

    for name in NULLS:
        image = ((torch.full((3, size, size), pixel[name]) - mean) / std).unsqueeze(0)
        handles = plug_masks(model, architecture, dataset, rate, name == "full_mask")
        try:
            logits = []
            if name == "full_mask":
                logits.append(forward_logits(model, image, device, True).cpu())
            else:
                for seed in NULL_SEEDS:
                    seed_everything(seed)
                    batch = image.expand(PASSES, -1, -1, -1)  # (PASSES, 3, S, S)
                    logits.append(forward_logits(model, batch, device, True).cpu())
        finally:
            unplug_dropout(handles)
        logits = torch.cat(logits)  # (passes, C)
        probs = logits.softmax(dim=1)  # (passes, C)
        margins = class_margins(logits)  # (passes, C)
        record[name] = {
            "prob": probs.mean(0).tolist(),
            "margin": margins.mean(0).tolist(),
            "argmax_of_mean_prob": int(probs.mean(0).argmax()),
            "n_passes": int(logits.shape[0]),
        }

    # The fully masked output must not depend on the image content.
    black = ((torch.zeros(3, size, size) - mean) / std).unsqueeze(0)
    handles = plug_masks(model, architecture, dataset, rate, True)
    try:
        other = forward_logits(model, black, device, True).cpu()
    finally:
        unplug_dropout(handles)
    reference = torch.tensor(record["full_mask"]["prob"])  # (C,)
    record["full_mask_image_dependence"] = float(
        (other.softmax(dim=1)[0] - reference).abs().max()
    )
    with open(path, "w") as f:
        json.dump(record, f)
    print(folder, device.type, rate, record["full_mask"]["argmax_of_mean_prob"])


@torch.inference_mode()
def dump_stage(folder, device, args):
    path = os.path.join(out_dir(args.results_dir, "dumps"), f"{folder}.pt")
    if os.path.exists(path):
        return
    rate = adaptive_rate(args.results_dir, folder)
    model, architecture, checkpoint_path = load_model(folder, device)
    metadata = read_checkpoint_metadata(checkpoint_path)
    manifest = read_split_manifest(os.path.join(args.results_dir, folder, "psbd"))
    probe = manifest["probe_attack"] if metadata["attack"] == "benign" else None
    loaders, _ = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=BATCH_SIZE,
        num_workers=2,
        max_samples=args.max_samples,
        probe_attack=probe,
        probe_target_label=manifest["probe_target_label"] if probe else None,
    )
    dump = {"model": folder, "adaptive_rate": rate}

    for split, loader in loaders.items():
        base = run_split(model, loader, device, None)
        handles = plug_masks(model, architecture, metadata["dataset"], rate, False)
        try:
            seed_everything(PSBD_MASK_SEED)
            passes = run_split(model, loader, device, base["label"])
        finally:
            unplug_dropout(handles)
        dump[split] = {"base": base, "passes": passes}
    torch.save(dump, path)
    print(folder, "dumped", {s: int(dump[s]["base"]["label"].numel()) for s in loaders})


def run_split(model, loader, device, labels):
    # labels None is the unperturbed pass, whose own argmax defines the class read.
    passes = 1 if labels is None else PASSES
    offset = 0
    prob_rows, margin_rows, argmax_rows, label_rows = [], [], [], []
    for images, _ in loader:
        batch = images.shape[0]
        prob_cols, margin_cols, argmax_cols = [], [], []
        for _ in range(passes):
            logits = forward_logits(model, images, device, True).cpu()  # (b, C)
            if labels is None:
                chosen = logits.argmax(dim=1)  # (b,)
            else:
                chosen = labels[offset : offset + batch].long()  # (b,)
            prob_cols.append(logits.softmax(dim=1).gather(1, chosen[:, None])[:, 0])
            margin_cols.append(class_margins(logits).gather(1, chosen[:, None])[:, 0])
            argmax_cols.append(logits.argmax(dim=1))
        prob_rows.append(torch.stack(prob_cols))  # (passes, b)
        margin_rows.append(torch.stack(margin_cols))  # (passes, b)
        argmax_rows.append(torch.stack(argmax_cols))  # (passes, b)
        label_rows.append(chosen)
        offset += batch
    result = {
        "prob": torch.cat(prob_rows, dim=1),  # (passes, n)
        "margin": torch.cat(margin_rows, dim=1),  # (passes, n)
        "argmax": torch.cat(argmax_rows, dim=1),  # (passes, n)
    }
    if labels is None:
        result = {key: value[0] for key, value in result.items()}  # (n,)
        result["label"] = torch.cat(label_rows)  # (n,)
    return result


def null_prob_score(base_prob, pass_prob, floor, eps):
    # base_prob (n,), pass_prob (K, n), floor (n,) the free evidence of each image's class.
    kept = (pass_prob - floor).clamp(min=0).mean(dim=0)  # (n,)
    denominator = (base_prob - floor).clamp(min=eps)  # (n,)
    score = 1.0 - kept / denominator  # (n,)
    below_eps = (base_prob - floor) < eps  # (n,)
    return score, below_eps


def load_null(results_dir, folder):
    for device in ("cuda", "cpu"):
        path = os.path.join(out_dir(results_dir, "nulls"), f"{folder}.{device}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


def cached_splits(results_dir, folder, rate):
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    splits = {}
    for split in ("validation", "clean", "backdoor"):
        probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
        per_pass, argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, RECOMMENDED_PLACEMENT, rate, split)
        )
        splits[split] = {
            "base_prob": probs.gather(1, labels.long()[:, None]).squeeze(1),
            "label": labels.long(),
            "pass_prob": per_pass,
            "argmax": argmax,
            "probs": probs,
        }
    manifest = read_split_manifest(os.path.join(results_dir, folder, "psbd"))
    return splits, manifest


def dumped_splits(results_dir, folder):
    path = os.path.join(out_dir(results_dir, "dumps"), f"{folder}.pt")
    if not os.path.exists(path):
        return None
    dump = torch.load(path)
    splits = {}
    for split in ("validation", "clean", "backdoor"):
        base, passes = dump[split]["base"], dump[split]["passes"]
        splits[split] = {
            "label": base["label"].long(),
            "base_prob": base["prob"],
            "pass_prob": passes["prob"],
            "base_margin": base["margin"],
            "pass_margin": passes["margin"],
            "argmax": passes["argmax"],
        }
    return splits


def scores_for(splits, floor_by_class, kind, eps):
    # kind "prob" or "margin", the free evidence of each class subtracted per image.
    scores, below = {}, {}
    for split, data in splits.items():
        base = data["base_prob"] if kind == "prob" else data["base_margin"]
        passes = data["pass_prob"] if kind == "prob" else data["pass_margin"]
        floor = torch.tensor(floor_by_class)[data["label"]]  # (n,)
        scores[split], below[split] = null_prob_score(base, passes, floor, eps)
    return scores, below


def metrics_for(scores, below, splits, manifest, attractor):
    paired_clean = pair_clean_to_backdoor(scores["clean"], manifest)
    clean_label = splits["clean"]["label"]
    is_attractor = clean_label == attractor
    record = {}
    for quantile in QUANTILES:
        report = detection_report(
            scores["validation"], paired_clean, scores["backdoor"], quantile
        )
        threshold = report["threshold"]
        flag = scores["clean"] < threshold  # (n_clean,)
        entry = {key: report[key] for key in ("tpr", "fpr", "auroc")}
        entry["clean_flag_rate"] = float(flag.float().mean())
        entry["attractor_flag_rate"] = float(flag[is_attractor].float().mean())
        entry["other_flag_rate"] = float(flag[~is_attractor].float().mean())
        entry["attractor_share_of_flagged"] = float(
            (flag & is_attractor).sum() / max(int(flag.sum()), 1)
        )
        record[f"q{quantile:.2f}"] = entry
    record["attractor_image_share"] = float(is_attractor.float().mean())
    if below:
        pooled_label = {s: splits[s]["label"] for s in below}
        total = sum(int(b.sum()) for b in below.values())
        images = sum(int(b.numel()) for b in below.values())
        in_attractor = sum(
            int((below[s] & (pooled_label[s] == attractor)).sum()) for s in below
        )
        record["below_eps_share"] = total / images
        record["below_eps_in_attractor_share"] = in_attractor / max(total, 1)
    return record


def analyze_model(results_dir, folder):
    rate = adaptive_rate(results_dir, folder)
    cached, manifest = cached_splits(results_dir, folder, rate)
    val = cached["validation"]
    attractor, _, _ = attractor_class(
        val["argmax"], val["label"], val["probs"].shape[1]
    )
    null = load_null(results_dir, folder)
    record = {"model": folder, "adaptive_rate": rate, "attractor_class": attractor}
    variants = {}

    plain = {
        s: 1.0 - d["pass_prob"].mean(0) / d["base_prob"] for s, d in cached.items()
    }
    variants["plain"] = metrics_for(plain, None, cached, manifest, attractor)
    if null is not None:
        record["null_argmax_class"] = null["full_mask"]["argmax_of_mean_prob"]
        record["null_device"] = null["device"]
        ranking = torch.tensor(null["full_mask"]["prob"]).argsort(descending=True)
        record["attractor_rank_in_full_mask_prob"] = int(
            (ranking == attractor).nonzero()[0, 0]
        )
        record["full_mask_image_dependence"] = null["full_mask_image_dependence"]
        for name in NULLS:
            scores, below = scores_for(cached, null[name]["prob"], "prob", EPS_PROB)
            variants[f"prob_{name}"] = metrics_for(
                scores, below, cached, manifest, attractor
            )
    fresh = dumped_splits(results_dir, folder)
    if fresh is not None and null is not None:
        plain_fresh = {
            s: 1.0 - d["pass_prob"].mean(0) / d["base_prob"] for s, d in fresh.items()
        }
        variants["fresh_plain"] = metrics_for(
            plain_fresh, None, fresh, manifest, attractor
        )
        record["fresh_label_agreement"] = {
            s: float((fresh[s]["label"] == cached[s]["label"]).float().mean())
            for s in fresh
        }
        for name in NULLS:
            scores, below = scores_for(fresh, null[name]["prob"], "prob", EPS_PROB)
            variants[f"fresh_prob_{name}"] = metrics_for(
                scores, below, fresh, manifest, attractor
            )
            scores, below = scores_for(fresh, null[name]["margin"], "margin", EPS_LOGIT)
            variants[f"logit_{name}"] = metrics_for(
                scores, below, fresh, manifest, attractor
            )
    record["variants"] = variants
    return record


def analyze(results_dir):
    cells = clearing_cells(load_coverage(results_dir))
    cell_of = {cell["folder_name"]: cell for cell in cells}
    folders = list(dict.fromkeys(list(A3_MODELS) + list(cell_of)))
    records = {}
    for folder in folders:
        if load_null(results_dir, folder) is None:
            continue
        records[folder] = analyze_model(results_dir, folder)

    summary = {
        "a3_models": aggregate(
            records, [f for f in A3_MODELS if f in records], cell_of
        ),
        "panel": aggregate(records, [f for f in cell_of if f in records], cell_of),
    }
    directory = out_dir(results_dir, "")
    with open(os.path.join(directory, "per_model.json"), "w") as f:
        json.dump(records, f, indent=1)
    with open(os.path.join(directory, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print_summary(summary)


def aggregate(records, folders, cell_of):
    groups = {"all": folders}
    for folder in folders:
        cell = cell_of.get(folder)
        if cell is None:
            continue
        groups.setdefault(f"dataset:{cell['dataset']}", []).append(folder)
        groups.setdefault(f"attack:{cell['attack']}", []).append(folder)
    for folder in folders:
        if folder.startswith("vit_") and "benign" in folder:
            dataset = folder.split("_")[1]
            groups.setdefault(f"dataset:{dataset}", []).append(folder)
    result = {}
    for name, members in groups.items():
        variants = sorted(
            {v for f in members for v in records[f]["variants"] if v != "plain"}
        )
        result[name] = {"n": len(members), "vs_plain": {}}
        for variant in variants:
            reference = (
                "fresh_plain" if variant.startswith(("fresh_", "logit_")) else "plain"
            )
            usable = [
                f
                for f in members
                if variant in records[f]["variants"] and not f.endswith("benign")
            ]
            entry = {"n": len(usable)}
            for key in ("q0.01", "q0.05", "q0.10"):
                for field in ("tpr", "fpr", "attractor_flag_rate"):
                    entry[f"{key}_{field}"] = paired(
                        records, usable, variant, reference, key, field
                    )
            entry["auroc"] = paired(
                records, usable, variant, reference, "q0.05", "auroc"
            )
            result[name]["vs_plain"][variant] = entry
    return result


def paired(records, folders, variant, reference, key, field):
    if not folders:
        return None
    new = [records[f]["variants"][variant][key][field] for f in folders]
    old = [records[f]["variants"][reference][key][field] for f in folders]
    diffs = [a - b for a, b in zip(new, old)]
    low, high = bootstrap_ci(diffs, BOOTSTRAP_RESAMPLES, 0)
    return {
        "plain": sum(old) / len(old),
        "variant": sum(new) / len(new),
        "diff": sum(diffs) / len(diffs),
        "ci": [low, high],
    }


def render(results_dir):
    directory = out_dir(results_dir, "")
    with open(os.path.join(directory, "per_model.json")) as f:
        records = json.load(f)
    with open(os.path.join(directory, "summary.json")) as f:
        summary = json.load(f)
    print(
        "| Model | Attractor | Full-mask class | Attractor rank | Attractor flag rate 0.05 | Attractor share of flags | TPR 0.01 | TPR 0.05 |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for folder in A3_MODELS:
        record = records[folder]
        plain = record["variants"]["plain"]
        null = record["variants"]["prob_full_mask"]
        cells = [
            f"{plain['q0.05']['attractor_flag_rate']:.2f} to {null['q0.05']['attractor_flag_rate']:.2f}",
            f"{plain['q0.05']['attractor_share_of_flagged']:.2f} to {null['q0.05']['attractor_share_of_flagged']:.2f}",
            f"{plain['q0.01']['tpr']:.2f} to {null['q0.01']['tpr']:.2f}",
            f"{plain['q0.05']['tpr']:.2f} to {null['q0.05']['tpr']:.2f}",
        ]
        print(
            f"| `{folder}` | {record['attractor_class']} | {record['null_argmax_class']} | "
            f"{record['attractor_rank_in_full_mask_prob'] + 1} | "
            + " | ".join(cells)
            + " |"
        )
    for variant in ("prob_full_mask", "logit_full_mask"):
        print()
        print(variant)
        print(
            "| Group | Models | TPR 0.01 | TPR 0.05 | TPR 0.10 | AUROC | Attractor flag rate 0.05 |"
        )
        print("|---|---|---|---|---|---|---|")
        for name, block in summary["panel"].items():
            entry = block["vs_plain"].get(variant)
            if entry is None:
                continue
            cells = []
            for key in (
                "q0.01_tpr",
                "q0.05_tpr",
                "q0.10_tpr",
                "auroc",
                "q0.05_attractor_flag_rate",
            ):
                cell = entry[key]
                low, high = cell["ci"]
                interval = "" if low != low else f" [{low:+.3f}, {high:+.3f}]"
                cells.append(
                    f"{cell['plain']:.3f} to {cell['variant']:.3f} ({cell['diff']:+.3f}{interval})"
                )
            print(f"| {name} | {entry['n']} | " + " | ".join(cells) + " |")


def print_summary(summary):
    for scope, groups in summary.items():
        for name, block in groups.items():
            for variant, entry in block["vs_plain"].items():
                if variant not in ("prob_full_mask", "logit_full_mask"):
                    continue
                cells = []
                for key in ("q0.01_tpr", "q0.05_tpr", "q0.10_tpr", "auroc"):
                    cell = entry[key]
                    if cell is not None:
                        cells.append(
                            f"{key} {cell['plain']:.3f}->{cell['variant']:.3f}"
                        )
                print(scope, name, block["n"], variant, " ".join(cells))


if __name__ == "__main__":
    main()
