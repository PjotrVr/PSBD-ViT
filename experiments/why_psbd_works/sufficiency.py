"""Trigger sufficiency: does the trigger carry the attack label without the image?

The hypothesis under test (pre-registered 2026-09-30, before any reading): PSBD
detects over-determined decisions. A backdoor that is a true shortcut, a trigger
sufficient on its own and independent of image content, gives the triggered
decision far more evidence than it needs, so it survives random removal. A clean
decision has just enough. PSBD should fail where the backdoor is a conjunction
with content (TaCT needs its source class) or relational along the removed axis
(WaNet's warp under token removal).

3 conditions per model, each changing 1 thing against a control:

    blank     the trigger stamped on content-free carriers (the dataset's mean
              image, mid gray and 62 uniform noise images), against the same
              carriers unstamped: the excess share predicted as the target is
              the sufficiency score
    content   the triggered image with its content hidden at the attention input
              of every block: every non-trigger token for a patch trigger, a
              random 70% of tokens (5 patterns) for a global trigger. The control
              is the clean twin under the same mask, and the reading is the
              excess of triggered over clean images sent to the target
    classes   the trigger stamped on test images of every non-target class,
              source restriction ignored, against the same images clean. A
              shortcut sends every class to the target, a conjunction only its
              source classes

Pre-registered predictions:

    S1  sufficiency (the blank excess, the content excess beside it) predicts PSBD-TM's AUROC and its TPR at the
        0.01 quantile at the adaptive rate across the successful ViT and Swin
        models with 1 monotone relation, Spearman at least 0.6 for each
    S2  TaCT sits low on sufficiency and on the class-independence reading
    S3  benign models probed with a trigger sit near 0 on every excess

Readings go to results/_experiments/why_psbd_works/sufficiency/<name>.json.

    PYTHONPATH=. python experiments/why_psbd_works/sufficiency.py --list
    PYTHONPATH=. python experiments/why_psbd_works/sufficiency.py --models vit_cifar100_badnet_a2o_0_05
"""

import argparse
import json
import os
import sys
import time
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
import torchvision.transforms.v2 as transforms_v2  # noqa: E402

from attacks import apply_config_overrides, build_attack, default_config  # noqa: E402
from data.registry import DATASET_REGISTRY  # noqa: E402
from data.splits import (  # noqa: E402
    PSBD_HELDOUT_SIZE,
    PSBD_SPLIT_SEED,
    load_clean_test_base,
    psbd_split_permutation,
    read_checkpoint_metadata,
)
from experiments._paths import experiment_results_dir  # noqa: E402
from experiments.why_psbd_works.measure import (  # noqa: E402
    CPU_THREADS,
    PATCH_GRID,
    SLUG,
    SubsetTokenMask,
    entry_name,
    limit_gpu_memory,
    load_pairs,
    predict_logits,
    random_keep_grid,
)
from experiments.residual_stream_mechanism.activation_patching import (  # noqa: E402
    trigger_tokens,
)
from models.backbones import load_checkpoint  # noqa: E402
from models.positions import plug_dropout, unplug_dropout  # noqa: E402
from scripts.paper._common import clearing_cells, load_coverage, swin_coverage  # noqa: E402

PAIRS = 256
CLASS_IMAGES = 512
PATCH_ATTACKS = ("badnet_a2o", "tact", "lc")
GLOBAL_VISIBLE = 0.3
GLOBAL_DRAWS = 5
BATCH = 128
NOISE_CARRIERS = 62
DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")
PROBES = ("badnet_a2o", "blend")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--gpu-memory-gb", type=float, default=14.0)
    parser.add_argument("--list", action="store_true")
    # evidence_surplus reuses this test on its own 10-model set.
    parser.add_argument("--output-slug", default=SLUG)
    return parser.parse_args()


def main():
    args = parse_args()
    entries = (
        [e.partition(":")[::2] for e in args.models] if args.models else panel(args)
    )
    if args.list:
        print("\n".join(entry_name(f, p or None) for f, p in entries))
        return
    torch.set_num_threads(CPU_THREADS)
    device = torch.device("cuda", torch.cuda.current_device())
    limit_gpu_memory(args.gpu_memory_gb, device)
    out_dir = os.path.join(
        experiment_results_dir(args.output_slug, args.results_dir), "sufficiency"
    )
    os.makedirs(out_dir, exist_ok=True)
    for folder, probe in entries:
        probe = probe or None
        path = os.path.join(out_dir, f"{entry_name(folder, probe)}.json")
        if os.path.exists(path):
            continue
        started = time.time()
        record = measure(folder, probe, args, device)
        record["seconds"] = round(time.time() - started, 1)
        with open(path, "w") as handle:
            json.dump(record, handle)
        print(f"[ok] {entry_name(folder, probe)} {record['seconds']}s", flush=True)


def panel(args):
    entries = [
        (c["folder_name"], "") for c in clearing_cells(load_coverage(args.results_dir))
    ]
    swin = swin_coverage(args.results_dir, args.checkpoints_dir)
    entries += [(c["folder_name"], "") for c in clearing_cells(swin)]
    for architecture in ("vit", "swin"):
        for dataset in DATASETS:
            entries += [(f"{architecture}_{dataset}_benign", p) for p in PROBES]
    return entries


def measure(folder, probe, args, device):
    path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(path)
    architecture = metadata["architecture"]
    model = load_checkpoint(architecture, path, device).eval()
    attack_name = probe or metadata["attack"]
    loader_args = types.SimpleNamespace(
        raw_data_dir=args.raw_data_dir, pairs=PAIRS, batch_size=BATCH
    )
    pairs, _ = load_pairs(path, probe, loader_args, device)
    target = int(pairs["targets"][0])
    attack = rebuilt_attack(metadata, attack_name, target)

    record = {
        "folder": folder,
        "probe_attack": probe,
        "architecture": architecture,
        "attack": attack_name,
        "dataset": metadata["dataset"],
        "poison_rate": metadata.get("poison_rate"),
        "target": target,
        "blank": blank_reading(model, attack, metadata["dataset"], target, device),
        "content": None
        if architecture == "resnet18"
        else content_reading(
            model, architecture, metadata, attack_name, pairs, target, device
        ),
        "classes": class_reading(model, attack, metadata, args, target, device),
    }
    return record


def rebuilt_attack(metadata, attack_name, target):
    size = DATASET_REGISTRY[metadata["dataset"]].image_size
    overrides = (
        metadata.get("attack_config_overrides")
        if attack_name == metadata["attack"]
        else None
    )
    config = apply_config_overrides(default_config(attack_name), overrides)
    attack = build_attack(attack_name, config, size, target)
    return attack


def stamp(attack, image, index):
    plant = attack.apply_trigger_eval or attack.apply_trigger
    stamped = plant(image.clone(), index)  # (C, H, W), 0 to 1
    return stamped


def target_share(model, images, target, device):
    predictions = torch.cat(
        [
            predict_logits(model, images[s : s + BATCH], device).argmax(dim=1)
            for s in range(0, len(images), BATCH)
        ]
    )  # (n,)
    share = float((predictions == target).float().mean())
    return share


# Content-free carriers: the dataset's mean image (0 everywhere after
# normalization), mid gray and uniform noise images. Stamped against unstamped,
# so a model that already sends a carrier to the target is not credited to the
# trigger.
def blank_reading(model, attack, dataset, target, device):
    spec = DATASET_REGISTRY[dataset]
    size = spec.image_size
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)
    generator = torch.Generator().manual_seed(0)
    carriers = [
        torch.tensor(spec.mean).view(3, 1, 1).expand(3, size, size).clone(),
        torch.full((3, size, size), 0.5),
    ] + [torch.rand(3, size, size, generator=generator) for _ in range(NOISE_CARRIERS)]
    plain = torch.stack([normalize(c) for c in carriers]).to(device)  # (m, C, H, W)
    stamped = torch.stack(
        [normalize(stamp(attack, c, i)) for i, c in enumerate(carriers)]
    ).to(device)  # (m, C, H, W)
    reading = {
        "carriers": len(carriers),
        "triggered_on_target": target_share(model, stamped, target, device),
        "blank_on_target": target_share(model, plain, target, device),
    }
    reading["excess"] = reading["triggered_on_target"] - reading["blank_on_target"]
    return reading


def content_reading(model, architecture, metadata, attack_name, pairs, target, device):
    if attack_name in PATCH_ATTACKS:
        probe_metadata = dict(metadata, attack=attack_name, target_label=target)
        keep = torch.zeros(PATCH_GRID * PATCH_GRID, dtype=torch.bool)
        keep[trigger_tokens(probe_metadata)] = True
        grids = [keep.view(PATCH_GRID, PATCH_GRID)]
        form = "trigger tokens only"
    else:
        grids = [random_keep_grid(GLOBAL_VISIBLE, draw) for draw in range(GLOBAL_DRAWS)]
        form = f"random {GLOBAL_VISIBLE} of tokens"
    triggered, clean = [], []
    for grid in grids:
        handles = plug_dropout(
            model,
            architecture,
            ("before_attention_norm",),
            {
                "before_attention_norm": lambda _rate, grid=grid: SubsetTokenMask(
                    grid, coarse_keep_any=True
                )
            },
            0.0,
        )
        try:
            triggered.append(target_share(model, pairs["triggered"], target, device))
            clean.append(target_share(model, pairs["clean"], target, device))
        finally:
            unplug_dropout(handles)
    reading = {
        "form": form,
        "visible_tokens": int(grids[0].sum()),
        "triggered_on_target": sum(triggered) / len(triggered),
        "clean_on_target": sum(clean) / len(clean),
        "unmasked_triggered_on_target": target_share(
            model, pairs["triggered"], target, device
        ),
    }
    reading["excess"] = reading["triggered_on_target"] - reading["clean_on_target"]
    # When the masked clean twin already goes to the target, the model's default
    # class under heavy masking is the target and the excess has no room left.
    reading["collapsed"] = reading["clean_on_target"] > 0.5
    return reading


# Test images of every class but the target, from the analysis split, stamped
# with the trigger whether or not the attack's eval set would admit them.
def class_reading(model, attack, metadata, args, target, device):
    spec = DATASET_REGISTRY[metadata["dataset"]]
    base = load_clean_test_base(metadata["dataset"], args.raw_data_dir)
    rows = psbd_split_permutation(len(base), PSBD_SPLIT_SEED)[
        PSBD_HELDOUT_SIZE:
    ].tolist()
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)
    images, stamped, labels = [], [], []
    for row in rows:
        image, label = base[row]
        if int(label) == target:
            continue
        images.append(normalize(image))
        stamped.append(normalize(stamp(attack, image, row)))
        labels.append(int(label))
        if len(labels) == CLASS_IMAGES:
            break
    labels = torch.tensor(labels)
    sources = set(attack.source_classes or ())
    in_source = torch.tensor([label in sources for label in labels.tolist()])
    predictions = {}
    for name, batch in (("clean", images), ("stamped", stamped)):
        stack = torch.stack(batch).to(device)  # (n, C, H, W)
        predictions[name] = torch.cat(
            [
                predict_logits(model, stack[s : s + BATCH], device).argmax(dim=1)
                for s in range(0, len(stack), BATCH)
            ]
        )
    hits = (predictions["stamped"] == target).float()
    clean_hits = (predictions["clean"] == target).float()
    others = ~in_source
    reading = {
        "images": len(labels),
        "stamped_on_target": float(hits.mean()),
        "clean_on_target": float(clean_hits.mean()),
        "non_source_stamped_on_target": float(hits[others].mean())
        if others.any()
        else None,
        "source_stamped_on_target": float(hits[in_source].mean())
        if in_source.any()
        else None,
    }
    return reading


if __name__ == "__main__":
    main()
