"""The prediction shift phenomenon of Li et al. (PSBD, arXiv 2406.05826), claim by claim.

The PSBD paper argues from 4 observations it shows only on ResNet-18 and VGG16-bn.
This script measures each of them on our ResNet-18 checkpoints first and then on
the ViT-B/16 and Swin-S panel models, under PSBD-RD (dropout after both residual
adds, the paper's site adapted) and PSBD-TM (token_mask at the attention input).

    1  shift ratio sigma against the rate p: clean rises and saturates, the
       backdoor set stays near 0 at the adaptively chosen p (Fig. 3 top, A3 to A5)
    2  shift intensity: shifted clean predictions land on the target class, and
       on a benign model on 1 dominant class (Fig. 3 bottom, A3 to A5)
    3  top-layer feature maps of a clean image and its triggered twin differ
       without dropout and become almost identical with it (Fig. 4, A6, A7)
    4  plain MC-Dropout standard deviation separates BadNets and fails on WaNet
       and Adaptive-Blend (Fig. 2, A1)

Claims 1, 2 and 4 read the stage-1 caches under results/<folder>/psbd/ (CPU).
Claim 3 needs forward passes and runs 1 model per process so a GPU lock can be
held per model. README.md states the method, the thresholds and the verdicts.

    source .venv/bin/activate
    PYTHONPATH=. python experiments/prediction_shift_phenomenon/measure.py --stage cached
    flock scratch/gpu.lock env PYTHONPATH=. python \\
        experiments/prediction_shift_phenomenon/measure.py --stage features --folders F
    PYTHONPATH=. python experiments/prediction_shift_phenomenon/measure.py --stage summary
"""

import argparse
import collections
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from analysis.cka import debiased_linear_cka  # noqa: E402
from analysis.features import captured_layers, transformer_blocks  # noqa: E402
from data.splits import (  # noqa: E402
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from defenses.cache import (  # noqa: E402
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defenses.decision import (  # noqa: E402
    ADAPTIVE_SHIFT_TARGET,
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
    attack_success_mask,
    complete_rates,
    pair_clean_to_backdoor,
    select_rate_adaptively,
)
from defenses.inference import forward_logits  # noqa: E402
from defenses.operators import TokenMask  # noqa: E402
from defenses.scores import (  # noqa: E402
    psu_from_cache,
    psu_ratio_from_cache,
    shift_ratio,
    shift_target_histogram,
)
from evaluation.metrics import auroc  # noqa: E402
from experiments._paths import experiment_results_dir  # noqa: E402
from models.backbones import load_checkpoint, network_core  # noqa: E402
from models.positions import DROPOUT_CONFIGS, plug_dropout, unplug_dropout  # noqa: E402
from scripts.paper._common import clearing_cells, load_coverage  # noqa: E402
from scripts.paper.tab_swin import swin_cells  # noqa: E402

SLUG = "prediction_shift_phenomenon"

# The 3 ResNet-18 checkpoints trained without the evasion penalty. No benign
# ResNet-18 exists and this experiment trains nothing.
RESNET_FOLDERS = (
    "resnet18_gtsrb_badnet_a2o_0_1",
    "resnet18_gtsrb_blend_0_1",
    "resnet18_cifar10_badnet_a2o_0_1_smoke",
)
# The CIFAR-10 smoke run trained 15 epochs on 20000 images and reaches clean
# accuracy 0.54, far outside the 2-point success bar the transformer panels use,
# so it is measured and shown on its own row but counts in no verdict.
REFERENCE_ONLY = ("resnet18_cifar10_badnet_a2o_0_1_smoke",)
BENIGN_DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")
# The SIG checkpoints trained before 2026-09-09 learned amplitude 0.1, while
# attacks/sig.py now builds 0.157 (docs/audits/2026-09-29-experiment-audit.md).
# A fresh triggered pass would score a trigger the model was not trained on, so
# every SIG model stays out until its args.json carries the trained amplitude.
QUARANTINED_ATTACKS = ("sig",)
# A fresh unperturbed forward pass must reproduce the stage-1 baseline on this
# share of the pairs, or the loader builds a different trigger or split than
# the cache was written with.
BASELINE_AGREEMENT_FLOOR = 0.98
# The best placement that masks no tokens, \BestResidualName in paper/headline.tex:
# dropout before both residual adds, in blocks 5 to 8 only. It asks whether a
# residual site that detects well does so through neuron bias. Its caches exist on
# ViT only.
BEST_RESIDUAL_PLACEMENT = "pre_residual_blocks_5_8"
BEST_RESIDUAL_BLOCKS = (5, 8)
PLACEMENTS = {
    "resnet18": (PUBLISHED_PLACEMENT,),
    "vit": (PUBLISHED_PLACEMENT, RECOMMENDED_PLACEMENT, BEST_RESIDUAL_PLACEMENT),
    "swin": (PUBLISHED_PLACEMENT, RECOMMENDED_PLACEMENT),
}
PLACEMENT_LABELS = {
    PUBLISHED_PLACEMENT: "PSBD-RD",
    RECOMMENDED_PLACEMENT: "PSBD-TM",
    BEST_RESIDUAL_PLACEMENT: "best residual, blocks 5 to 8",
}
CONDITION_PLACEMENT = {
    "rd_adaptive": PUBLISHED_PLACEMENT,
    "rd_paper": PUBLISHED_PLACEMENT,
    "tm_adaptive": RECOMMENDED_PLACEMENT,
    "br_adaptive": BEST_RESIDUAL_PLACEMENT,
}
CATEGORIES = {
    "badnet_a2o": "patch",
    "tact": "patch",
    "blend": "blend",
    "lf": "blend",
    "sig": "frequency",
    "wanet": "warp",
    "bpp": "quantization",
    "adaptive_blend": "adaptive blend",
    "lc": "clean label patch",
    "benign": "benign",
}

# The claim 3 models: 1 per attack family and architecture at 10% where it
# cleared, the trigger-conditional TaCT models and GTSRB BadNets and Blend at
# 10% on all 3 architectures, the one cell every architecture shares. SIG is
# quarantined (QUARANTINED_ATTACKS).
FEATURE_FOLDERS = {
    "resnet18": RESNET_FOLDERS,
    "vit": (
        "vit_gtsrb_badnet_a2o_0_1",
        "vit_gtsrb_blend_0_1",
        "vit_gtsrb_tact_0_05",
        "vit_gtsrb_benign",
        "vit_cifar10_badnet_a2o_0_1",
        "vit_cifar10_tact_0_01",
        "vit_cifar10_tact_0_05",
        "vit_cifar10_blend_0_1",
        "vit_cifar10_lf_0_1",
        "vit_cifar10_wanet_0_1",
        "vit_cifar10_bpp_0_1",
        "vit_cifar10_benign",
    ),
    "swin": (
        "swin_gtsrb_badnet_a2o_0_1",
        "swin_gtsrb_blend_0_1",
        "swin_gtsrb_benign",
        "swin_cifar10_badnet_a2o_0_1",
        "swin_cifar10_tact_0_01",
        "swin_cifar10_tact_0_05",
        "swin_cifar10_blend_0_1",
        "swin_cifar10_lf_0_1",
        "swin_cifar10_wanet_0_1",
        "swin_cifar10_bpp_0_1",
        "swin_cifar10_benign",
    ),
}
# The models whose 64 feature maps are kept for the Fig. 4 style figure.
MAP_FOLDERS = (
    "resnet18_gtsrb_badnet_a2o_0_1",
    "vit_gtsrb_badnet_a2o_0_1",
    "swin_gtsrb_badnet_a2o_0_1",
)
# A benign model has no trigger of its own, so it is probed with 1 patch and 1
# global trigger. The stage-1 caches of every benign model probe BadNets at 0.
BENIGN_PROBES = ("badnet_a2o", "blend")
BENIGN_PROBE_TARGET = 0

# Fig. 4 shows the maps at p = 0.91 and Fig. 3 picks p = 0.7 adaptively.
PAPER_FEATURE_RATE = 0.91
# The 2 genuine Swin TaCT models were swept with PSBD-TM at this 1 rate only, so
# it stands in for their adaptive rate. The other Swin CIFAR-10 models pick 0.6
# on 17 of 22, so this reads a slightly weaker perturbation than theirs.
FALLBACK_TOKEN_RATE = 0.5
PAIR_COUNT = 256
FEATURE_PASSES = 3
MAP_CHANNELS = 64
# The red boxes of Fig. 4: both maps non-zero and every activation within 1.
PAPER_MATCH_TOLERANCE = 1.0
INDEPENDENT_SEED_OFFSET = 1_000_003
MASK_SEED = 0
BATCH_SIZE = 64
MINIMUM_CURVE_RATES = 5
GPU_MEMORY_GB = 14.0

# Verdict thresholds, stated in README.md.
NEAR_ZERO_SHIFT = 0.1
BENIGN_CURVE_TOLERANCE = 0.1
ALMOST_ALL_ON_TARGET = 0.8
DOMINANT_CLASS_SHARE = 0.5
IDENTICAL_COSINE = 0.9
CONVERGENCE_GAIN = 0.1
STD_LOSES_BY = 0.05


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stage", choices=("cached", "features", "summary"), required=True
    )
    parser.add_argument("--folders", nargs="+", default=None)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    # Separate from --results-dir so a smoke run reads the real caches but
    # writes somewhere disposable.
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--pairs", type=int, default=PAIR_COUNT)
    parser.add_argument("--passes", type=int, default=FEATURE_PASSES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gpu-memory-gb", type=float, default=GPU_MEMORY_GB)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    # bfloat16 is what every stage-1 cache was written in. float32 exists for the
    # sanity gate that checks the half-precision forward changes nothing.
    parser.add_argument("--float32", action="store_true")
    # Measures only the named conditions and merges them into a model's existing
    # features record, so a condition added later costs no rerun of the others.
    parser.add_argument("--add-conditions", nargs="+", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = experiment_results_dir(SLUG, args.output_root)

    if args.stage == "cached":
        panel = cached_panel(args.results_dir, args.checkpoints_dir)
        write_json(os.path.join(output_dir, "panel.json"), {"folders": panel})
        folders = args.folders or panel
        for folder in folders:
            out_path = os.path.join(output_dir, "cached", f"{folder}.json")
            if os.path.exists(out_path) and not args.overwrite:
                continue
            started = time.time()
            record = measure_cached(folder, args.results_dir, args.checkpoints_dir)
            record["seconds"] = round(time.time() - started, 1)
            write_json(out_path, record)
            print(f"[cached] {folder} {record['seconds']}s", flush=True)

    if args.stage == "features":
        device = torch.device(args.device)
        if device.type == "cuda":
            # The memory cap needs an indexed device.
            device = torch.device("cuda", torch.cuda.current_device())
            limit_gpu_memory(args.gpu_memory_gb, device)
        folders = args.folders or [
            folder for group in FEATURE_FOLDERS.values() for folder in group
        ]
        for folder in folders:
            out_path = os.path.join(output_dir, "features", f"{folder}.json")
            if args.add_conditions:
                add_feature_conditions(folder, out_path, args, output_dir, device)
                continue
            if os.path.exists(out_path) and not args.overwrite:
                print(f"[skip] {folder}", flush=True)
                continue
            started = time.time()
            record, maps = measure_features(folder, args, output_dir, device)
            record["seconds"] = round(time.time() - started, 1)
            record["gpu"] = gpu_note(device)
            write_json(out_path, record)
            if maps is not None:
                write_json(
                    os.path.join(output_dir, "feature_maps", f"{folder}.json"), maps
                )
            print(f"[features] {folder} {record['seconds']}s", flush=True)

    if args.stage == "summary":
        summary = summarize(output_dir)
        write_json(os.path.join(output_dir, "summary.json"), summary)
        print(markdown_tables(summary))


def cached_panel(results_dir, checkpoints_dir):
    coverage = load_coverage(results_dir)
    vit = [
        cell["folder_name"]
        for cell in clearing_cells(coverage)
        if has_placements(results_dir, cell["folder_name"], "vit")
        and cell["attack"] not in QUARANTINED_ATTACKS
    ]
    swin = [
        cell["folder"]
        for cell in swin_cells(results_dir, checkpoints_dir, coverage["asr_bar"])
        if has_placements(results_dir, cell["folder"], "swin")
        and cell["attack"] not in QUARANTINED_ATTACKS
    ]
    benign = [
        f"{architecture}_{dataset}_benign"
        for architecture in ("vit", "swin")
        for dataset in BENIGN_DATASETS
        if has_placements(results_dir, f"{architecture}_{dataset}_benign", architecture)
    ]
    folders = list(RESNET_FOLDERS) + vit + swin + benign
    return folders


def has_placements(results_dir, folder, architecture):
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    reachable = any(
        curve_swept(psbd_dir, placement) for placement in PLACEMENTS[architecture]
    )
    return reachable


def curve_swept(psbd_dir, placement):
    # A curve needs a grid. 3 Swin models (the 2 genuine CIFAR-10 TaCT models
    # and SIG at 5%) carry PSBD-TM at 1 rate only, so they enter with PSBD-RD
    # alone.
    swept = len(complete_rates(psbd_dir, placement)) >= MINIMUM_CURVE_RATES
    return swept


def measure_cached(folder, results_dir, checkpoints_dir):
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    identity = model_identity(folder, psbd_dir, checkpoints_dir)
    manifest = read_split_manifest(psbd_dir)
    baselines = {
        split: load_baseline(baseline_path(psbd_dir, split))
        for split in ("validation", "clean", "backdoor")
    }
    captured = attack_success_mask(baselines["backdoor"][1], baselines["backdoor"][2])
    assert captured is not None, f"{folder} baseline predates loader labels"
    # The paper's backdoor set is poisoned training data the model fits. A
    # triggered test image the model does not send to the target is clean in
    # behavior, so a poisoned model's backdoor rows are the captured ones. A
    # benign model captures nothing, so all its triggered rows are kept.
    backdoor_rows = torch.ones_like(captured) if identity["benign"] else captured
    identity["n"] = {
        "validation": int(len(baselines["validation"][1])),
        "clean": int(len(baselines["clean"][1])),
        "backdoor": int(len(captured)),
        "backdoor_captured": int(captured.sum()),
    }
    identity["clean_accuracy_paired"] = paired_clean_accuracy(
        baselines["clean"], manifest
    )

    identity["placements"] = {
        placement: measure_placement(
            psbd_dir, placement, baselines, backdoor_rows, manifest, identity
        )
        for placement in PLACEMENTS[identity["architecture"]]
        if curve_swept(psbd_dir, placement)
    }
    return identity


def model_identity(folder, psbd_dir, checkpoints_dir):
    metadata = read_checkpoint_metadata(
        os.path.join(checkpoints_dir, folder, "attack_result.pt")
    )
    manifest = read_split_manifest(psbd_dir)
    benign = metadata.get("attack") in (None, "benign") or folder.endswith("_benign")
    attack = "benign" if benign else metadata["attack"]
    # A poisoned model's target is what it was trained with. A benign model's is
    # the one its stage-1 caches probed.
    target_label = (
        int(manifest["probe_target_label"]) if benign else int(metadata["target_label"])
    )
    assert target_label == int(manifest["probe_target_label"]), (
        f"{folder}: args.json target {target_label} differs from the cache's "
        f"{manifest['probe_target_label']}"
    )
    identity = {
        "folder": folder,
        "architecture": metadata["architecture"],
        "dataset": metadata["dataset"],
        "attack": attack,
        "category": CATEGORIES[attack],
        "poison_rate": None if benign else metadata.get("poison_rate"),
        "benign": benign,
        "probe_attack": manifest["probe_attack"],
        "target_label": target_label,
        "asr": None if benign else metadata.get("asr"),
    }
    return identity


def paired_clean_accuracy(clean_baseline, manifest):
    _, labels, loader_labels = clean_baseline
    correct = (labels.long() == loader_labels.long()).float()  # (n_clean,)
    paired = pair_clean_to_backdoor(correct, manifest)  # (n_backdoor,)
    accuracy = float(paired.mean())
    return accuracy


def measure_placement(
    psbd_dir, placement, baselines, backdoor_rows, manifest, identity
):
    rates = complete_rates(psbd_dir, placement)
    per_rate = [
        measure_rate(
            psbd_dir, placement, rate, baselines, backdoor_rows, manifest, identity
        )
        for rate in rates
    ]
    sigma_by_rate = {row["rate"]: row["sigma"]["validation"] for row in per_rate}
    adaptive_rate = select_rate_adaptively(sigma_by_rate)

    landing_at_adaptive = None
    if adaptive_rate is not None:
        landing_at_adaptive = landing_histograms(
            psbd_dir, placement, adaptive_rate, baselines, backdoor_rows
        )
    record = {
        "rates": rates,
        "adaptive_rate": adaptive_rate,
        "per_rate": per_rate,
        "landing_at_adaptive": landing_at_adaptive,
    }
    return record


def measure_rate(
    psbd_dir, placement, rate, baselines, backdoor_rows, manifest, identity
):
    passes = {
        split: load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, split)
        )
        for split in ("validation", "clean", "backdoor")
    }
    target = identity["target_label"]
    num_classes = int(baselines["validation"][0].shape[1])

    sigma = {
        split: shift_ratio(baselines[split][1], passes[split][1])
        for split in ("validation", "clean")
    }
    sigma["backdoor"] = shift_ratio(
        baselines["backdoor"][1][backdoor_rows], passes["backdoor"][1][:, backdoor_rows]
    )

    landing = {
        split: landing_summary(
            baselines[split][1], passes[split][1], num_classes, target
        )
        for split in ("validation", "clean")
    }
    landing["backdoor"] = landing_summary(
        baselines["backdoor"][1][backdoor_rows],
        passes["backdoor"][1][:, backdoor_rows],
        num_classes,
        target,
    )

    scores = per_sample_scores(baselines, passes)
    clean_paired = {
        name: pair_clean_to_backdoor(values, manifest)
        for name, values in scores["clean"].items()
    }  # each (n_backdoor,)
    backdoor_kept = {
        name: values[backdoor_rows] for name, values in scores["backdoor"].items()
    }
    std_mean = {
        "validation": float(scores["validation"]["std"].mean()),
        "clean_paired": float(clean_paired["std"].mean()),
        "backdoor": float(backdoor_kept["std"].mean()),
    }
    separation = {
        name: auroc(clean_paired[name], backdoor_kept[name])
        for name in ("std", "psu", "psu_ratio")
    }

    row = {
        "rate": rate,
        "sigma": sigma,
        "landing": landing,
        "std_mean": std_mean,
        "auroc": separation,
    }
    return row


def per_sample_scores(baselines, passes):
    scores = {}
    for split in ("validation", "clean", "backdoor"):
        probs, labels, _ = baselines[split]
        per_pass_probs = passes[split][0].float()  # (passes, n)
        # MC-Dropout uncertainty as the pilot study defines it, the spread of the
        # tracked class's confidence over the passes. The population form, since
        # 3 passes are all there is.
        spread = per_pass_probs.std(dim=0, correction=0)  # (n,)
        scores[split] = {
            "std": spread,
            "psu": psu_from_cache(probs, labels, per_pass_probs),
            "psu_ratio": psu_ratio_from_cache(probs, labels, per_pass_probs),
        }
    return scores


def landing_summary(baseline_labels, per_pass_argmax, num_classes, target):
    histogram = shift_target_histogram(baseline_labels, per_pass_argmax, num_classes)
    assert histogram is not None, "cache predates argmax saving"
    shifted = sum(histogram)
    if shifted == 0:
        empty = {
            "shifted": 0,
            "target_share": None,
            "top_class": None,
            "top_share": None,
        }
        return empty

    top_class = max(range(num_classes), key=lambda label: histogram[label])
    summary = {
        "shifted": int(shifted),
        "target_share": histogram[target] / shifted,
        "top_class": int(top_class),
        "top_share": histogram[top_class] / shifted,
    }
    return summary


def landing_histograms(psbd_dir, placement, rate, baselines, backdoor_rows):
    histograms = {}
    num_classes = int(baselines["validation"][0].shape[1])
    for split in ("validation", "backdoor"):
        _, argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, split)
        )
        labels = baselines[split][1]
        if split == "backdoor":
            labels, argmax = labels[backdoor_rows], argmax[:, backdoor_rows]
        counts = shift_target_histogram(labels, argmax, num_classes)
        total = max(sum(counts), 1)
        histograms[split] = [round(count / total, 5) for count in counts]
    return histograms


def measure_features(folder, args, output_dir, device):
    checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)
    architecture = metadata["architecture"]
    model = load_checkpoint(architecture, checkpoint_path, device).eval()
    last_layer = len(transformer_blocks(network_core(model), architecture))
    conditions = feature_conditions(folder, architecture, args, output_dir)
    if args.add_conditions:
        conditions = [conditions[0]] + [
            condition
            for condition in conditions[1:]
            if condition["name"] in args.add_conditions
        ]

    assert metadata.get("attack") not in QUARANTINED_ATTACKS, f"{folder} is quarantined"
    benign = folder.endswith("_benign")
    probes = BENIGN_PROBES if benign else (metadata["attack"],)
    record = {
        "folder": folder,
        "architecture": architecture,
        "dataset": metadata["dataset"],
        "attack": "benign" if benign else metadata["attack"],
        "last_layer": last_layer,
        "conditions": conditions,
        "probes": {},
    }
    maps = None
    for probe in probes:
        pairs = load_pairs(checkpoint_path, probe if benign else None, args)
        pairs["cached"] = cached_pair_predictions(
            args.results_dir, folder, probe, pairs
        )
        target = int(pairs["targets"][0])
        reading, example = measure_probe(
            model, architecture, last_layer, pairs, conditions, args, device, target
        )
        record["probes"][probe] = reading
        if folder in MAP_FOLDERS and maps is None:
            maps = {"folder": folder, "architecture": architecture, **example}
    return record, maps


def add_feature_conditions(folder, out_path, args, output_dir, device):
    with open(out_path) as handle:
        existing = json.load(handle)
    present = {condition["name"] for condition in existing["conditions"]}
    if set(args.add_conditions) <= present:
        print(f"[skip] {folder} already has {args.add_conditions}", flush=True)
        return
    started = time.time()
    added, _ = measure_features(folder, args, output_dir, device)
    for condition in added["conditions"][1:]:
        existing["conditions"].append(condition)
        for probe, reading in added["probes"].items():
            existing["probes"][probe]["conditions"][condition["name"]] = reading[
                "conditions"
            ][condition["name"]]
    existing["added_seconds"] = round(time.time() - started, 1)
    write_json(out_path, existing)
    print(f"[added] {folder} {existing['added_seconds']}s", flush=True)


def feature_conditions(folder, architecture, args, output_dir):
    cached_path = os.path.join(output_dir, "cached", f"{folder}.json")
    if not os.path.exists(cached_path):
        raise FileNotFoundError(
            f"{cached_path} is missing. The adaptive rates come from the cached "
            "stage, so run --stage cached for this model first."
        )
    with open(cached_path) as handle:
        cached = json.load(handle)
    adaptive = {
        placement: block["adaptive_rate"]
        for placement, block in cached["placements"].items()
    }
    if adaptive.get(PUBLISHED_PLACEMENT) is None:
        print(
            f"[excluded] {folder} rd_adaptive: no swept rate reaches the adaptive "
            "target, the condition is recorded as null",
            flush=True,
        )

    conditions = [{"name": "unperturbed", "placement": None, "rate": 0.0}]
    conditions.append(
        {
            "name": "rd_adaptive",
            "placement": PUBLISHED_PLACEMENT,
            "rate": adaptive.get(PUBLISHED_PLACEMENT),
            "rate_rule": "adaptive",
        }
    )
    conditions.append(
        {
            "name": "rd_paper",
            "placement": PUBLISHED_PLACEMENT,
            "rate": PAPER_FEATURE_RATE,
            "rate_rule": "paper",
        }
    )
    if architecture != "resnet18":
        token_rate = adaptive.get(RECOMMENDED_PLACEMENT)
        conditions.append(
            {
                "name": "tm_adaptive",
                "placement": RECOMMENDED_PLACEMENT,
                "rate": token_rate if token_rate is not None else FALLBACK_TOKEN_RATE,
                "rate_rule": "adaptive" if token_rate is not None else "fallback",
            }
        )
    if architecture == "vit":
        conditions.append(
            {
                "name": "br_adaptive",
                "placement": BEST_RESIDUAL_PLACEMENT,
                "rate": adaptive.get(BEST_RESIDUAL_PLACEMENT),
                "rate_rule": "adaptive",
            }
        )
    return conditions


def measure_probe(
    model, architecture, last_layer, pairs, conditions, args, device, target
):
    generator = torch.Generator().manual_seed(MASK_SEED)
    permutation = torch.randperm(len(pairs["clean"]), generator=generator)  # (n,)
    pairs = {**pairs, "unrelated": pairs["clean"][permutation]}
    reference = run_condition(
        model,
        architecture,
        last_layer,
        pairs,
        conditions[0],
        MASK_SEED,
        MASK_SEED,
        args,
        device,
    )
    reference_mean = reference["clean_flat"].mean(dim=0, keepdim=True)  # (1, dim)
    example_row = example_pair(pairs, reference, target)
    agreement = cache_agreement(pairs["cached"], reference)

    reading = {
        "pairs": int(len(pairs["clean"])),
        "target": target,
        "example_row": example_row,
        "cache_agreement": agreement,
        "conditions": {},
    }
    example = {"target": target, "example_row": example_row, "maps": {}}
    example["maps"]["unperturbed"] = map_slice(reference, example_row)
    reading["conditions"]["unperturbed"] = similarity_readout(
        reference, reference, reference_mean, target
    )

    for condition in conditions[1:]:
        if condition["rate"] is None:
            reading["conditions"][condition["name"]] = None
            continue
        per_mode = {}
        for mode in ("shared_mask", "independent_mask"):
            readings = []
            for draw in range(args.passes):
                clean_seed = MASK_SEED + draw
                triggered_seed = clean_seed + (
                    0 if mode == "shared_mask" else INDEPENDENT_SEED_OFFSET
                )
                perturbed = run_condition(
                    model,
                    architecture,
                    last_layer,
                    pairs,
                    condition,
                    clean_seed,
                    triggered_seed,
                    args,
                    device,
                )
                readings.append(
                    similarity_readout(perturbed, reference, reference_mean, target)
                )
                if mode == "shared_mask" and draw == 0:
                    example["maps"][condition["name"]] = map_slice(
                        perturbed, example_row
                    )
            per_mode[mode] = mean_readout(readings)
        reading["conditions"][condition["name"]] = per_mode
    return reading, example


def cache_agreement(cached, reference):
    if cached is None:
        return None
    agreement = {
        image: float(
            (reference[f"{image}_predictions"] == cached[image]).float().mean()
        )
        for image in ("clean", "triggered")
    }
    # A model whose trigger the loader no longer rebuilds is stopped here rather
    # than measured, which is the failure the SIG amplitude drift would cause.
    assert min(agreement.values()) >= BASELINE_AGREEMENT_FLOOR, (
        f"fresh predictions agree with the stage-1 baseline on only {agreement}"
    )
    return agreement


def run_condition(
    model,
    architecture,
    last_layer,
    pairs,
    condition,
    clean_seed,
    triggered_seed,
    args,
    device,
):
    handles = plug_condition(model, architecture, condition)
    try:
        clean = forward_features(
            model,
            architecture,
            last_layer,
            pairs["clean"],
            clean_seed,
            args.batch_size,
            device,
            use_bfloat16=not args.float32,
        )
        triggered = forward_features(
            model,
            architecture,
            last_layer,
            pairs["triggered"],
            triggered_seed,
            args.batch_size,
            device,
            use_bfloat16=not args.float32,
        )
        # The unrelated clean image is drawn with the triggered image's seed, so
        # it shares the clean image's mask exactly when the triggered twin does.
        unrelated = forward_features(
            model,
            architecture,
            last_layer,
            pairs["unrelated"],
            triggered_seed,
            args.batch_size,
            device,
            use_bfloat16=not args.float32,
        )
    finally:
        unplug_dropout(handles)

    outputs = {
        "clean_maps": clean["maps"],  # (n, channels, height, width)
        "triggered_maps": triggered["maps"],
        "clean_flat": clean["maps"].flatten(1),  # (n, channels * height * width)
        "triggered_flat": triggered["maps"].flatten(1),
        "unrelated_flat": unrelated["maps"].flatten(1),
        "clean_pooled": clean["pooled"],  # (n, channels)
        "triggered_pooled": triggered["pooled"],
        "unrelated_pooled": unrelated["pooled"],
        "clean_predictions": clean["predictions"],  # (n,)
        "triggered_predictions": triggered["predictions"],
    }
    return outputs


def plug_condition(model, architecture, condition):
    if condition["placement"] is None:
        return []
    block_range = None
    if condition["placement"] == RECOMMENDED_PLACEMENT:
        position_names = ("before_attention_norm",)
        operator = TokenMask
    elif condition["placement"] == BEST_RESIDUAL_PLACEMENT:
        position_names = DROPOUT_CONFIGS["pre_residual"]
        operator = torch.nn.Dropout
        block_range = BEST_RESIDUAL_BLOCKS
    elif architecture == "resnet18":
        position_names = ("post_residual",)
        operator = torch.nn.Dropout
    else:
        position_names = DROPOUT_CONFIGS[PUBLISHED_PLACEMENT]
        operator = torch.nn.Dropout
    factory = {name: operator for name in position_names}
    handles = plug_dropout(
        model,
        architecture,
        position_names,
        factory,
        condition["rate"],
        block_range=block_range,
    )
    return handles


@torch.inference_mode()
def forward_features(
    model, architecture, last_layer, images, seed, batch_size, device, use_bfloat16
):
    maps, pooled, predictions = [], [], []
    with captured_layers(model, (last_layer,), architecture) as captured:
        for index, start in enumerate(range(0, len(images), batch_size)):
            # The mask a probe draws depends only on the seed and the batch shape,
            # so a clean batch and its triggered twin seeded alike see 1 mask.
            torch.manual_seed(seed * 100_003 + index)
            logits = forward_logits(
                model, images[start : start + batch_size], device, use_bfloat16
            )  # (b, classes)
            activation = captured[last_layer].float()
            channel_maps = as_channel_maps(activation, architecture)  # (b, c, h, w)
            maps.append(channel_maps.cpu())
            pooled.append(pooled_features(activation, channel_maps, architecture).cpu())
            predictions.append(logits.argmax(dim=1).cpu())

    features = {
        "maps": torch.cat(maps),  # (n, channels, height, width)
        "pooled": torch.cat(pooled),  # (n, channels)
        "predictions": torch.cat(predictions),  # (n,)
    }
    return features


def as_channel_maps(activation, architecture):
    if architecture == "resnet18":
        return activation  # (b, 512, 4, 4) at 32 by 32 input

    if architecture == "swin":
        channel_maps = activation.permute(0, 3, 1, 2)  # (b, 768, 7, 7)
        return channel_maps

    patches = activation[:, 1:, :]  # (b, 196, 768), CLS dropped
    batch, tokens, channels = patches.shape
    side = int(round(tokens**0.5))
    assert side * side == tokens, f"{tokens} patch tokens do not form a square grid"
    channel_maps = patches.transpose(1, 2).reshape(batch, channels, side, side)
    return channel_maps  # (b, 768, 14, 14)


def pooled_features(activation, channel_maps, architecture):
    # What the classifier head reads: the class token on ViT, the spatial
    # average on ResNet-18 and Swin-S.
    if architecture == "vit":
        class_token = activation[:, 0, :]  # (b, 768)
        return class_token
    pooled = channel_maps.mean(dim=(2, 3))  # (b, channels)
    return pooled


def similarity_readout(perturbed, reference, reference_mean, target):
    clean, triggered = perturbed["clean_flat"], perturbed["triggered_flat"]  # (n, d)
    unrelated = perturbed["unrelated_flat"]  # (n, d)
    clean_centered = clean - reference_mean  # (n, d)
    triggered_centered = triggered - reference_mean
    unrelated_centered = unrelated - reference_mean
    triggered_reference = reference["triggered_flat"] - reference_mean
    clean_reference = reference["clean_flat"] - reference_mean

    readout = {
        "cosine_pair": mean_cosine(clean, triggered),
        "cosine_pair_centered": mean_cosine(clean_centered, triggered_centered),
        # The control the paper's figure lacks: 2 unrelated clean images under the
        # same perturbation and the same mask relation as the pair. If they
        # converge as much, the convergence says nothing about the backdoor.
        "cosine_unrelated_clean": mean_cosine(clean, unrelated),
        "cosine_unrelated_clean_centered": mean_cosine(
            clean_centered, unrelated_centered
        ),
        "cosine_pooled_pair": mean_cosine(
            perturbed["clean_pooled"], perturbed["triggered_pooled"]
        ),
        "cosine_pooled_unrelated_clean": mean_cosine(
            perturbed["clean_pooled"], perturbed["unrelated_pooled"]
        ),
        "cka_pair": debiased_linear_cka(clean, triggered),
        "clean_to_unperturbed_triggered_centered": mean_cosine(
            clean_centered, triggered_reference
        ),
        "clean_self_centered": mean_cosine(clean_centered, clean_reference),
        "triggered_self_centered": mean_cosine(triggered_centered, triggered_reference),
        "paper_matched_channel_share": matched_channel_share(
            perturbed["clean_maps"], perturbed["triggered_maps"]
        ),
        "clean_on_target": float(
            (perturbed["clean_predictions"] == target).float().mean()
        ),
        "triggered_on_target": float(
            (perturbed["triggered_predictions"] == target).float().mean()
        ),
    }
    return readout


def mean_cosine(first, second):
    similarity = float(F.cosine_similarity(first, second, dim=1).mean())
    return similarity


def matched_channel_share(clean_maps, triggered_maps):
    # Fig. 4's red box, read literally: both maps carry a non-zero value and no
    # activation differs by more than 1. The tolerance is absolute, so the share
    # depends on the activation scale of the layer.
    clean_active = clean_maps.flatten(2).abs().amax(dim=2) > 0  # (n, channels)
    triggered_active = triggered_maps.flatten(2).abs().amax(dim=2) > 0
    largest_gap = (clean_maps - triggered_maps).flatten(2).abs().amax(dim=2)
    matched = clean_active & triggered_active & (largest_gap <= PAPER_MATCH_TOLERANCE)
    share = float(matched.float().mean())
    return share


def mean_readout(readings):
    averaged = {
        key: sum(reading[key] for reading in readings) / len(readings)
        for key in readings[0]
    }
    return averaged


def example_pair(pairs, reference, target):
    # The figure pair is 1 the model reads as the paper's does: the clean image
    # correct and off target, its triggered twin on target.
    clean_ok = reference["clean_predictions"] == pairs["clean_labels"]
    off_target = pairs["clean_labels"] != target
    triggered_ok = reference["triggered_predictions"] == target
    usable = torch.nonzero(clean_ok & off_target & triggered_ok).flatten()
    row = int(usable[0]) if len(usable) else 0
    return row


def map_slice(outputs, row):
    selected = {
        "clean": outputs["clean_maps"][row, :MAP_CHANNELS],  # (64, height, width)
        "triggered": outputs["triggered_maps"][row, :MAP_CHANNELS],
    }
    rounded = {
        key: [[[round(float(v), 2) for v in line] for line in grid] for grid in value]
        for key, value in selected.items()
    }
    return rounded


def load_pairs(checkpoint_path, probe_attack, args):
    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        probe_attack=probe_attack,
        probe_target_label=BENIGN_PROBE_TARGET if probe_attack else None,
    )
    clean_set = loaders["clean"].dataset
    backdoor_set = loaders["backdoor"].dataset
    row_of = {
        original: row for row, original in enumerate(manifest["analysis_clean_indices"])
    }
    count = min(args.pairs, len(backdoor_set))
    clean_rows = [
        row_of[original] for original in manifest["analysis_backdoor_indices"][:count]
    ]

    clean_items = [clean_set[row] for row in clean_rows]
    triggered_items = [backdoor_set[row] for row in range(count)]
    pairs = {
        "clean": torch.stack([image for image, _ in clean_items]),  # (n, 3, h, w)
        "clean_labels": torch.tensor([int(label) for _, label in clean_items]),
        "triggered": torch.stack([image for image, _ in triggered_items]),
        "targets": torch.tensor([int(target) for _, target in triggered_items]),
        "clean_rows": torch.tensor(clean_rows),  # (n,) rows of the clean split
        "manifest_probe": manifest["probe_attack"],
    }
    assert pairs["clean"].shape == pairs["triggered"].shape, "pairs must align"
    return pairs


def cached_pair_predictions(results_dir, folder, probe, pairs):
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    cached_manifest = read_split_manifest(psbd_dir)
    # A benign model's cache probes 1 trigger only, so its other probe has no
    # baseline to be held to.
    if cached_manifest["probe_attack"] != probe:
        return None
    count = len(pairs["targets"])
    _, clean_labels, _ = load_baseline(baseline_path(psbd_dir, "clean"))
    _, triggered_labels, _ = load_baseline(baseline_path(psbd_dir, "backdoor"))
    cached = {
        "clean": clean_labels[pairs["clean_rows"]],  # (n,)
        "triggered": triggered_labels[:count],  # (n,)
    }
    return cached


def limit_gpu_memory(gigabytes, device):
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, gigabytes * 1024**3 / total), device
    )


def gpu_note(device):
    if device.type != "cuda":
        return {"device": "cpu"}
    note = {
        "device": torch.cuda.get_device_name(device),
        "peak_allocated_gb": round(
            torch.cuda.max_memory_allocated(device) / 1024**3, 2
        ),
    }
    return note


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # The feature maps are the bulk of the output and are read by code only.
    indent = None if "maps" in payload else 1
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=indent)


def read_records(directory):
    records = {}
    if not os.path.isdir(directory):
        return records
    for name in sorted(os.listdir(directory)):
        if name.endswith(".json"):
            with open(os.path.join(directory, name)) as handle:
                records[name[: -len(".json")]] = json.load(handle)
    return records


def summarize(output_dir):
    # The panel is the ledger's successful backdoors at the 2-point clean
    # accuracy bar, as the cached stage last read it, so a record left from an
    # earlier panel never reaches a verdict.
    with open(os.path.join(output_dir, "panel.json")) as handle:
        panel = set(json.load(handle)["folders"])
    cached = {
        folder: record
        for folder, record in read_records(os.path.join(output_dir, "cached")).items()
        if folder in panel and record["attack"] not in QUARANTINED_ATTACKS
    }
    features = {
        folder: record
        for folder, record in read_records(os.path.join(output_dir, "features")).items()
        if (folder in panel or record["attack"] == "benign")
        and record["attack"] not in QUARANTINED_ATTACKS
    }
    benign_landing = benign_landing_reference(cached)

    rows = []
    for folder, record in cached.items():
        for placement, block in record["placements"].items():
            rows.append(cached_row(record, placement, block, benign_landing))
    feature_rows = [
        feature_row(record, probe, condition, reading)
        for record in features.values()
        for probe, probe_reading in record["probes"].items()
        for condition, reading in probe_reading["conditions"].items()
        if condition != "unperturbed" and reading is not None
    ]
    summary = {
        "thresholds": {
            "near_zero_shift": NEAR_ZERO_SHIFT,
            "benign_curve_tolerance": BENIGN_CURVE_TOLERANCE,
            "almost_all_on_target": ALMOST_ALL_ON_TARGET,
            "dominant_class_share": DOMINANT_CLASS_SHARE,
            "identical_cosine": IDENTICAL_COSINE,
            "convergence_gain": CONVERGENCE_GAIN,
            "std_loses_by": STD_LOSES_BY,
            "adaptive_shift_target": ADAPTIVE_SHIFT_TARGET,
        },
        "cached_rows": rows,
        "feature_rows": feature_rows,
        "verdicts": verdict_table(rows, feature_rows),
        "overall": overall_verdicts(rows, feature_rows),
    }
    return summary


def benign_landing_reference(cached):
    reference = {}
    for record in cached.values():
        if not record["benign"]:
            continue
        for placement, block in record["placements"].items():
            adaptive = adaptive_row(block)
            if adaptive is None:
                continue
            key = (record["architecture"], record["dataset"], placement)
            reference[key] = adaptive["landing"]["validation"]["target_share"]
    return reference


def adaptive_row(block):
    if block["adaptive_rate"] is None:
        return None
    row = next(
        row for row in block["per_rate"] if row["rate"] == block["adaptive_rate"]
    )
    return row


def cached_row(record, placement, block, benign_landing):
    row = {
        "folder": record["folder"],
        "architecture": record["architecture"],
        "dataset": record["dataset"],
        "attack": record["attack"],
        "category": record["category"],
        "benign": record["benign"],
        "reference_only": record["folder"] in REFERENCE_ONLY,
        "target_label": record["target_label"],
        "placement": placement,
        "adaptive_rate": block["adaptive_rate"],
        "max_validation_sigma": max(
            (r["sigma"]["validation"] or 0.0) for r in block["per_rate"]
        ),
        "benign_target_share": benign_landing.get(
            (record["architecture"], record["dataset"], placement)
        ),
    }
    adaptive = adaptive_row(block)
    if adaptive is None:
        row["claim_1"] = "no adaptive rate"
        return row

    sigma = adaptive["sigma"]
    landing = adaptive["landing"]
    row.update(
        {
            "sigma_validation": sigma["validation"],
            "sigma_backdoor": sigma["backdoor"],
            "min_backdoor_sigma_where_clean_high": min(
                r["sigma"]["backdoor"]
                for r in block["per_rate"]
                if r["sigma"]["validation"] >= ADAPTIVE_SHIFT_TARGET
            ),
            "mean_curve_gap": sum(
                abs(r["sigma"]["validation"] - r["sigma"]["backdoor"])
                for r in block["per_rate"]
            )
            / len(block["per_rate"]),
            "validation_target_share": landing["validation"]["target_share"],
            # How far the target share exceeds the share a benign model of the
            # same dataset sends to that class, the control for claim 2.
            "target_share_over_benign": (
                None
                if row["benign_target_share"] is None
                or landing["validation"]["target_share"] is None
                else landing["validation"]["target_share"] - row["benign_target_share"]
            ),
            "max_target_share_where_clean_high": max(
                r["landing"]["validation"]["target_share"] or 0.0
                for r in block["per_rate"]
                if r["sigma"]["validation"] >= ADAPTIVE_SHIFT_TARGET
            ),
            "validation_top_class": landing["validation"]["top_class"],
            "validation_top_share": landing["validation"]["top_share"],
            "backdoor_top_class": landing["backdoor"]["top_class"],
            "backdoor_top_share": landing["backdoor"]["top_share"],
            "std_validation": adaptive["std_mean"]["validation"],
            "std_clean_paired": adaptive["std_mean"]["clean_paired"],
            "std_backdoor": adaptive["std_mean"]["backdoor"],
            "auroc_std": adaptive["auroc"]["std"],
            "auroc_psu": adaptive["auroc"]["psu"],
            "auroc_psu_ratio": adaptive["auroc"]["psu_ratio"],
            "auroc_std_best_rate": max(r["auroc"]["std"] for r in block["per_rate"]),
        }
    )
    row["claim_1"] = claim_1_verdict(row)
    row["claim_2"] = claim_2_verdict(row)
    row["claim_4"] = claim_4_verdict(row)
    return row


def claim_1_verdict(row):
    if row["benign"]:
        # On a benign model the paper expects the clean and triggered curves
        # to move together.
        verdict = (
            "holds" if row["mean_curve_gap"] <= BENIGN_CURVE_TOLERANCE else "fails"
        )
        return verdict
    if row["sigma_backdoor"] <= NEAR_ZERO_SHIFT:
        return "holds"
    if row["min_backdoor_sigma_where_clean_high"] <= NEAR_ZERO_SHIFT:
        return "partial"
    return "fails"


def claim_2_verdict(row):
    if row["benign"]:
        share = row["validation_top_share"] or 0.0
        verdict = "holds" if share >= DOMINANT_CLASS_SHARE else "fails"
        return verdict
    share = row["validation_target_share"] or 0.0
    if share >= ALMOST_ALL_ON_TARGET:
        return "holds"
    # Partial mirrors claim 1: the target leads at the adaptive rate without
    # taking almost all shifts, or another rate the clean curve has saturated at
    # sends almost all of them there and the adaptive rule missed it.
    target_leads = row["validation_top_class"] == row["target_label"]
    reached_elsewhere = row["max_target_share_where_clean_high"] >= ALMOST_ALL_ON_TARGET
    if target_leads or reached_elsewhere:
        return "partial"
    return "fails"


def claim_4_verdict(row):
    if row["benign"]:
        return None
    std_loses = row["auroc_std_best_rate"] < row["auroc_psu_ratio"] - STD_LOSES_BY
    verdict = "std fails" if std_loses else "std suffices"
    return verdict


def feature_row(record, probe, condition, reading):
    shared = reading["shared_mask"]
    independent = reading["independent_mask"]
    unperturbed = record["probes"][probe]["conditions"]["unperturbed"]
    row = {
        "folder": record["folder"],
        "reference_only": record["folder"] in REFERENCE_ONLY,
        "architecture": record["architecture"],
        "attack": record["attack"],
        "probe": probe,
        "condition": condition,
        "rate": next(c["rate"] for c in record["conditions"] if c["name"] == condition),
        "cosine_pair_off": unperturbed["cosine_pair_centered"],
        "cosine_pair_shared": shared["cosine_pair_centered"],
        "cosine_pair_independent": independent["cosine_pair_centered"],
        "cosine_unrelated_off": unperturbed["cosine_unrelated_clean_centered"],
        "cosine_unrelated_on": shared["cosine_unrelated_clean_centered"],
        "cosine_unrelated_independent": independent["cosine_unrelated_clean_centered"],
        "cosine_pooled_pair_off": unperturbed["cosine_pooled_pair"],
        "cosine_pooled_pair_independent": independent["cosine_pooled_pair"],
        "cosine_pooled_unrelated_independent": independent[
            "cosine_pooled_unrelated_clean"
        ],
        "cka_pair_off": unperturbed["cka_pair"],
        "cka_pair_on": shared["cka_pair"],
        "paper_share_off": unperturbed["paper_matched_channel_share"],
        "paper_share_on": shared["paper_matched_channel_share"],
        "clean_on_target": independent["clean_on_target"],
        "triggered_on_target": independent["triggered_on_target"],
        # How much of each image's own unperturbed feature survives, and whether
        # the perturbed clean feature moves toward the unperturbed triggered one,
        # which is what a bias toward the backdoor path would predict.
        "clean_self": independent["clean_self_centered"],
        "triggered_self": independent["triggered_self_centered"],
        "clean_toward_triggered_off": unperturbed[
            "clean_to_unperturbed_triggered_centered"
        ],
        "clean_toward_triggered_on": independent[
            "clean_to_unperturbed_triggered_centered"
        ],
    }
    row["claim_3"] = claim_3_verdict(row)
    return row


def claim_3_verdict(row):
    # The paper's maps share 1 dropout mask between the clean image and its
    # twin, since independent masks at p = 0.91 would put their non-zero
    # entries in different places. The shared mask is the reading.
    gain = row["cosine_pair_shared"] - row["cosine_pair_off"]
    specific = gain - (row["cosine_unrelated_on"] - row["cosine_unrelated_off"])
    if gain < CONVERGENCE_GAIN:
        return "fails"
    # Convergence the unrelated control matches is the mask's doing, which is
    # the opposite of the paper's reading, so it fails rather than half holds.
    if specific < CONVERGENCE_GAIN:
        return "fails"
    if row["cosine_pair_shared"] >= IDENTICAL_COSINE:
        return "holds"
    return "partial"


def verdict_table(rows, feature_rows):
    table = collections.defaultdict(lambda: collections.Counter())
    for row in rows:
        group = (row["architecture"], row["placement"], row["category"])
        for claim in ("claim_1", "claim_2", "claim_4"):
            if row.get(claim):
                table[group][f"{claim}:{row[claim]}"] += 1
    for row in feature_rows:
        placement = CONDITION_PLACEMENT[row["condition"]]
        category = CATEGORIES[row["attack"]]
        group = (row["architecture"], placement, category)
        table[group][f"claim_3_{row['condition']}:{row['claim_3']}"] += 1
    flattened = [
        {
            "architecture": architecture,
            "placement": placement,
            "category": category,
            "counts": dict(counts),
        }
        for (architecture, placement, category), counts in sorted(table.items())
    ]
    return flattened


def overall_verdicts(rows, feature_rows):
    verdicts = []
    for architecture in ("resnet18", "vit", "swin"):
        for placement in PLACEMENT_LABELS:
            poisoned = [
                row
                for row in rows
                if row["architecture"] == architecture
                and row["placement"] == placement
                and not row["benign"]
                and not row["reference_only"]
            ]
            if not poisoned:
                continue
            benign = [
                row
                for row in rows
                if row["architecture"] == architecture
                and row["placement"] == placement
                and row["benign"]
            ]
            features = [
                row
                for row in feature_rows
                if row["architecture"] == architecture
                and row["attack"] != "benign"
                and not row["reference_only"]
                and CONDITION_PLACEMENT[row["condition"]] == placement
            ]
            verdicts.append(
                {
                    "architecture": architecture,
                    "placement": placement,
                    "models": len(poisoned),
                    "benign_models": len(benign),
                    "claim_1": majority(poisoned, "claim_1"),
                    "claim_1_benign": majority(benign, "claim_1"),
                    "claim_2": majority(poisoned, "claim_2"),
                    "claim_2_benign": majority(benign, "claim_2"),
                    "claim_3": majority(features, "claim_3"),
                    "claim_4": claim_4_overall(poisoned),
                    # Medians skip a model with no adaptive rate, so the count
                    # behind them is stated beside them.
                    "models_with_adaptive_rate": sum(
                        row["adaptive_rate"] is not None for row in poisoned
                    ),
                    "median_backdoor_sigma": median(poisoned, "sigma_backdoor"),
                    "median_validation_sigma": median(poisoned, "sigma_validation"),
                    "median_target_share": median(poisoned, "validation_target_share"),
                    "median_auroc_std": median(poisoned, "auroc_std"),
                    "median_auroc_std_best_rate": median(
                        poisoned, "auroc_std_best_rate"
                    ),
                    "median_auroc_psu_ratio": median(poisoned, "auroc_psu_ratio"),
                }
            )
    return verdicts


def majority(rows, claim):
    # A claim holds on an architecture when it holds on at least 2 in 3 of its
    # models and fails when it fails on at least 2 in 3. Anything between is
    # partial, which is where 1 attack family goes 1 way and another the other.
    counts = collections.Counter(row.get(claim) for row in rows if row.get(claim))
    total = sum(counts.values())
    if total == 0:
        return {"verdict": None, "counts": {}}
    if counts["holds"] * 3 >= total * 2:
        verdict = "holds"
    elif counts["fails"] * 3 >= total * 2:
        verdict = "fails"
    else:
        verdict = "partial"
    result = {"verdict": verdict, "counts": dict(counts), "total": total}
    return result


def claim_4_overall(rows):
    # The pilot study says standard deviation works on BadNets and fails on
    # WaNet and Adaptive-Blend. Both halves are read, each by its majority.
    badnets = [row for row in rows if row["attack"] == "badnet_a2o"]
    hard = [row for row in rows if row["attack"] in ("wanet", "adaptive_blend")]
    badnets_suffices = share(badnets, "std suffices")
    hard_fails = share(hard, "std fails")
    halves = [
        value is not None and value >= 2 / 3 for value in (badnets_suffices, hard_fails)
    ]
    verdict = {2: "holds", 1: "partial", 0: "fails"}[sum(halves)]
    # ResNet-18 has no WaNet or Adaptive-Blend model, so only 1 half is readable.
    if hard_fails is None:
        verdict = "BadNets half only"
    result = {
        "verdict": verdict,
        "badnets_std_suffices_share": badnets_suffices,
        "badnets_models": len(badnets),
        "wanet_adaptive_blend_std_fails_share": hard_fails,
        "wanet_adaptive_blend_models": len(hard),
        "all_std_fails_share": share(rows, "std fails"),
    }
    return result


def share(rows, verdict):
    if not rows:
        return None
    fraction = sum(row.get("claim_4") == verdict for row in rows) / len(rows)
    return fraction


def median(rows, key):
    values = sorted(row[key] for row in rows if row.get(key) is not None)
    if not values:
        return None
    middle = len(values) // 2
    value = (
        values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    )
    return value


def markdown_tables(summary):
    lines = [
        "| architecture | placement | models | claim 1 | claim 2 | claim 3 | claim 4 |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in summary["overall"]:
        cells = [
            entry[claim]["verdict"] or "not measured"
            for claim in ("claim_1", "claim_2", "claim_3", "claim_4")
        ]
        lines.append(
            f"| {entry['architecture']} | {PLACEMENT_LABELS[entry['placement']]} | "
            f"{entry['models']} | " + " | ".join(cells) + " |"
        )
    table = "\n".join(lines)
    return table


if __name__ == "__main__":
    main()
