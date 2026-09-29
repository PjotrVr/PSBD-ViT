"""Sanity gates run on 1 model before the why_psbd_works measurements are trusted.

measure.py builds its own forward passes, hooks and pairs. These gates hold them
to the pipeline that produced the paper's numbers, on the full PSBD splits of 1
checkpoint.

    1  clean accuracy on the full test set and ASR on the full eval ASR split,
       against checkpoints/<folder>/metrics.json
    2  PSBD-TM and PSBD-RD at their adaptive rates, k = 3, mask seed 0, batch 64:
       per-sample fractional PSU against the cached per-pass tensors under
       results/<folder>/psbd/, and the headline AUROC and validation shift ratio
       against psbd_metrics.json, both computed by the library functions
    3  the triggered split holds only target labels, its clean twins hold none,
       and on a patch trigger every paired difference sits inside the trigger
    6  gate 2 repeated in float32

    source .venv/bin/activate
    PYTHONPATH=. python experiments/why_psbd_works/sanity.py --folder vit_cifar100_badnet_a2o_0_05
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from cli.compare_detectors import psbd_rate  # noqa: E402
from data.splits import (  # noqa: E402
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from defenses.cache import dropout_pass_path, load_dropout_pass_probs  # noqa: E402
from defenses.decision import detection_report, pair_clean_to_backdoor  # noqa: E402
from defenses.inference import compute_dropout_pass_probs, forward_probs  # noqa: E402
from defenses.scores import (  # noqa: E402
    psu_from_cache,
    psu_ratio_from_cache,
    shift_ratio,
)
from experiments._paths import experiment_result_path  # noqa: E402
from experiments.why_psbd_works.measure import (  # noqa: E402
    OPERATOR_SPECS,
    SLUG,
    limit_gpu_memory,
    trigger_pixel_mask,
)
from models.backbones import load_checkpoint  # noqa: E402
from models.positions import plug_dropout, unplug_dropout  # noqa: E402

SANITY_OPERATORS = ("token_mask", "residual_dropout")
PIPELINE_PASSES = 3
PIPELINE_MASK_SEED = 0
PIPELINE_BATCH = 64
HEADLINE = "q0.25"
SPLITS = ("validation", "clean", "backdoor")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--folder", default="vit_cifar100_badnet_a2o_0_05")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--gpu-memory-gb", type=float, default=14.0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    limit_gpu_memory(args.gpu_memory_gb, device)

    checkpoint_path = os.path.join(
        args.checkpoints_dir, args.folder, "attack_result.pt"
    )
    metadata = read_checkpoint_metadata(checkpoint_path)
    model = load_checkpoint(metadata["architecture"], checkpoint_path, device).eval()
    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=PIPELINE_BATCH,
        num_workers=0,
    )
    with open(
        os.path.join(args.results_dir, args.folder, "psbd_metrics.json")
    ) as handle:
        placements = json.load(handle)["placements"]
    with open(
        os.path.join(args.checkpoints_dir, args.folder, "metrics.json")
    ) as handle:
        stored_metrics = json.load(handle)

    report = {"folder": args.folder}
    for precision, use_bfloat16 in (("bfloat16", True), ("float32", False)):
        baselines = split_baselines(model, loaders, device, use_bfloat16)
        report[precision] = {
            "gate_1": accuracy_gate(baselines, metadata, stored_metrics),
            "gate_2": {
                name: psu_gate(
                    model,
                    metadata,
                    loaders,
                    manifest,
                    baselines,
                    placements,
                    name,
                    args,
                    device,
                    use_bfloat16,
                )
                for name in SANITY_OPERATORS
            },
        }
    report["gate_3"] = pairing_gate(loaders, manifest, metadata)

    out_path = experiment_result_path(
        SLUG, f"sanity_{args.folder}.json", args.output_root
    )
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


def split_baselines(model, loaders, device, use_bfloat16):
    baselines = {}
    for split in SPLITS:
        probs, loader_labels = [], []
        with torch.inference_mode():
            for images, labels in loaders[split]:
                probs.append(forward_probs(model, images, device, use_bfloat16).cpu())
                loader_labels.append(labels)
        probs = torch.cat(probs)  # (n, classes)
        baselines[split] = {
            "probs": probs,
            "labels": probs.argmax(dim=1),  # (n,)
            "loader_labels": torch.cat(loader_labels),  # (n,)
        }
    return baselines


# The stored clean accuracy is over the whole test set, which is the held-out
# split and the analysis split together. The stored ASR is over the whole eval
# ASR set and the analysis split holds 80% of it.
def accuracy_gate(baselines, metadata, stored_metrics):
    correct = sum(
        int((baselines[split]["labels"] == baselines[split]["loader_labels"]).sum())
        for split in ("validation", "clean")
    )
    total = sum(len(baselines[split]["labels"]) for split in ("validation", "clean"))
    backdoor = baselines["backdoor"]
    asr = float((backdoor["labels"] == backdoor["loader_labels"]).float().mean())

    gate = {
        "clean_accuracy": correct / total,
        "stored_clean_accuracy": stored_metrics["clean_accuracy"],
        "asr": asr,
        "stored_asr": stored_metrics["asr"],
        "args_asr": metadata.get("asr"),
    }
    gate["passed"] = (
        abs(gate["clean_accuracy"] - gate["stored_clean_accuracy"]) <= 0.01
        and abs(gate["asr"] - gate["stored_asr"]) <= 0.01
    )
    return gate


def psu_gate(
    model,
    metadata,
    loaders,
    manifest,
    baselines,
    placements,
    name,
    args,
    device,
    use_bfloat16,
):
    placement, positions, factory = OPERATOR_SPECS[name]
    block = placements[placement]
    rate = psbd_rate(block, "adaptive")
    architecture = metadata["architecture"]

    fresh = {}
    for split in SPLITS:
        handles = plug_dropout(
            model, architecture, positions, {p: factory for p in positions}, rate
        )
        try:
            fresh[split] = compute_dropout_pass_probs(
                model,
                loaders[split],
                baselines[split]["labels"],
                device,
                PIPELINE_PASSES,
                use_bfloat16,
                PIPELINE_MASK_SEED,
            )
        finally:
            unplug_dropout(handles)
        model.eval()

    psbd_dir = os.path.join(args.results_dir, args.folder, "psbd")
    psu_fresh, psu_cached, sigma = {}, {}, {}
    for split in SPLITS:
        cached_probs, _ = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, split)
        )
        psu_fresh[split] = psu_ratio_from_cache(
            baselines[split]["probs"], baselines[split]["labels"], fresh[split][0]
        )
        psu_cached[split] = psu_ratio_from_cache(
            baselines[split]["probs"], baselines[split]["labels"], cached_probs
        )
        sigma[split] = shift_ratio(baselines[split]["labels"], fresh[split][1])

    report = detection_report(
        psu_fresh["validation"],
        pair_clean_to_backdoor(psu_fresh["clean"], manifest),
        psu_fresh["backdoor"],
        0.25,
    )
    # psbd_metrics.json keeps the paper's absolute PSU under "adaptive" and the
    # fractional statistic, the headline of this project, per rate under
    # "detection_psu_ratio". Both are checked.
    absolute = {
        split: psu_from_cache(
            baselines[split]["probs"], baselines[split]["labels"], fresh[split][0]
        )
        for split in SPLITS
    }
    absolute_report = detection_report(
        absolute["validation"],
        pair_clean_to_backdoor(absolute["clean"], manifest),
        absolute["backdoor"],
        0.25,
    )
    difference = torch.cat(
        [(psu_fresh[s] - psu_cached[s]).abs() for s in SPLITS]
    )  # (n_all,)
    both = torch.stack(
        [
            torch.cat([psu_fresh[s] for s in SPLITS]),
            torch.cat([psu_cached[s] for s in SPLITS]),
        ]
    )  # (2, n_all)

    gate = {
        "placement": placement,
        "rate": rate,
        "auroc_fractional": report["auroc"],
        "cached_auroc_fractional": rate_row(block, rate)["detection_psu_ratio"][
            HEADLINE
        ]["auroc"],
        "auroc_absolute": absolute_report["auroc"],
        "cached_auroc_absolute": block["adaptive"]["auroc"],
        "validation_shift_ratio": sigma["validation"],
        "cached_validation_shift_ratio": rate_row(block, rate)["shift_ratio"][
            "validation"
        ],
        "per_sample_max_abs_difference": float(difference.max()),
        "per_sample_mean_abs_difference": float(difference.mean()),
        "per_sample_correlation": float(torch.corrcoef(both)[0, 1]),
        "samples": int(difference.numel()),
    }
    gate["passed"] = (
        abs(gate["auroc_fractional"] - gate["cached_auroc_fractional"]) <= 0.01
        and abs(gate["auroc_absolute"] - gate["cached_auroc_absolute"]) <= 0.01
    )
    return gate


def rate_row(block, rate):
    row = next(r for r in block["rates"] if r["rate"] == rate)
    return row


# Triggered rows carry the attack-success label, the target for all to one. Their
# clean twins are the same test images, so on a patch trigger the 2 differ only
# inside the trigger's pixels.
def pairing_gate(loaders, manifest, metadata):
    clean_set = loaders["clean"].dataset
    backdoor_set = loaders["backdoor"].dataset
    row_of = {o: r for r, o in enumerate(manifest["analysis_clean_indices"])}
    count = min(256, len(backdoor_set))
    target = metadata["target_label"]

    outside_max = 0.0
    clean_labels, triggered_labels = [], []
    mask = (
        trigger_pixel_mask(metadata, metadata["attack"])
        if metadata["attack"] == "badnet_a2o"
        else None
    )
    for row in range(count):
        triggered, triggered_label = backdoor_set[row]
        clean, clean_label = clean_set[
            row_of[manifest["analysis_backdoor_indices"][row]]
        ]
        triggered_labels.append(int(triggered_label))
        clean_labels.append(int(clean_label))
        if mask is not None:
            difference = (triggered - clean).abs().sum(dim=0)  # (H, W)
            outside_max = max(outside_max, float(difference[~mask].max()))

    gate = {
        "pairs_checked": count,
        "triggered_labels_all_target": all(t == target for t in triggered_labels),
        "clean_twins_never_target": all(c != target for c in clean_labels),
        "max_difference_outside_trigger": outside_max if mask is not None else None,
    }
    gate["passed"] = (
        gate["triggered_labels_all_target"]
        and gate["clean_twins_never_target"]
        and (mask is None or outside_max < 1e-5)
    )
    return gate


if __name__ == "__main__":
    main()
