"""C. Clean surplus without a backdoor: inputs that carry extra evidence.

Predictions are in PREDICTIONS.md, written before this ran, with the addendum on
resolution. If PSBD reads evidence surplus and not the backdoor as such, a clean
input with more evidence for its own class than it needs should look poisoned to
it, on a benign model too.

5 inputs per model, 256 each, from the clean analysis split, all at twice the
dataset's resolution (the model's own Resize brings every input to 224):

    same_class       4 different test images of 1 class, 2 by 2
    duplicated       1 test image repeated in all 4 quadrants
    one_quadrant     1 test image in 1 quadrant, the rest the dataset's mean color,
                     the control for the halved per-image resolution of a collage
    different_class  4 test images of 4 classes, conflicting evidence
    single           1 test image filling the input

Each gets its fractional PSU at the canonical operating point
(experiments/evidence_surplus/common.py), the share flagged below the
clean-validation threshold at the 0.01, 0.05 and 0.10 quantiles. It also gets its
critical rate p*: the smallest rate on PSBD-TM's ladder at which most of 3
passes change its answer, above the ladder when none does. The target class is
left out of every draw on a backdoored model, so no input carries the attack's
label.

    PYTHONPATH=. python experiments/evidence_surplus/clean_surplus/measure.py
"""

import argparse
import collections
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from data.splits import (  # noqa: E402
    PSBD_HELDOUT_SIZE,
    PSBD_SPLIT_SEED,
    load_clean_test_base,
    psbd_split_permutation,
    read_checkpoint_metadata,
)
from experiments._paths import experiment_results_dir  # noqa: E402
from data.registry import DATASET_REGISTRY  # noqa: E402
from defenses.scores import critical_rate  # noqa: E402
from experiments.evidence_surplus.common import (  # noqa: E402
    BENIGN,
    MASK_SEED,
    PASSES,
    SLUG,
    fractional_psu,
    limit_gpu_memory,
    logits_of,
    normalized,
    operating_point,
)
from lightning import seed_everything  # noqa: E402
from models.positions import plug_dropout, unplug_dropout  # noqa: E402
from models.backbones import load_checkpoint  # noqa: E402

MODELS = BENIGN + ("vit_cifar10_badnet_a2o_0_01", "vit_cifar10_blend_0_1")
INPUT_NAMES = ("same_class", "duplicated", "one_quadrant", "different_class", "single")
# PSBD-TM's sweep ladder.
LADDER = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
INPUTS = 256
SEED = 0
# Every panel model targets class 0, the benign ones are read at the same class.
TARGET = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--gpu-memory-gb", type=float, default=14.0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda", torch.cuda.current_device())
    limit_gpu_memory(args.gpu_memory_gb, device)
    out_dir = os.path.join(
        experiment_results_dir(SLUG, args.results_dir), "clean_surplus"
    )
    os.makedirs(out_dir, exist_ok=True)
    for folder in args.models:
        path = os.path.join(out_dir, f"{folder}.json")
        if os.path.exists(path):
            continue
        started = time.time()
        record = measure(folder, args, device)
        record["seconds"] = round(time.time() - started, 1)
        with open(path, "w") as handle:
            json.dump(record, handle)
        print(f"[ok] {folder} {record['seconds']}s", flush=True)


def measure(folder, args, device):
    path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(path)
    architecture = metadata["architecture"]
    dataset = metadata["dataset"]
    model = load_checkpoint(architecture, path, device).eval()
    point = operating_point(args.results_dir, folder, architecture)
    by_class = images_by_class(dataset, args.raw_data_dir)
    inputs = build_inputs(by_class, DATASET_REGISTRY[dataset].mean)

    record = {
        "folder": folder,
        "architecture": architecture,
        "dataset": dataset,
        "placement": point["placement"],
        "rate": point["rate"],
        "thresholds": point["thresholds"],
        "inputs": {},
    }
    for name, (pixels, labels) in inputs.items():
        images = normalized(pixels.to(device), dataset)  # (n, C, 2H, 2W)
        psu, predicted = fractional_psu(model, architecture, point, images, device)
        critical = critical_rates(model, architecture, point, images, device)
        record["inputs"][name] = {
            "count": len(labels),
            "median_psu": float(psu.median()),
            "mean_psu": float(psu.mean()),
            "own_class_share": float((predicted == labels).float().mean()),
            "flagged": {
                q: float((psu < t).float().mean())
                for q, t in point["thresholds"].items()
            },
            "psu": [round(float(v), 5) for v in psu],
            # Kept per input so the flag can be read on the inputs the model still
            # assigns to their own class, the failure protocol's diagnosis.
            "predicted": predicted.tolist(),
            "labels": labels.tolist(),
            "median_critical_rate": float(critical.median()),
            "critical_rate": [round(float(v), 3) for v in critical],
        }
    return record


# Clean analysis images grouped by class, the attack's target class left out.
def images_by_class(dataset, raw_data_dir):
    base = load_clean_test_base(dataset, raw_data_dir)
    rows = psbd_split_permutation(len(base), PSBD_SPLIT_SEED)[
        PSBD_HELDOUT_SIZE:
    ].tolist()
    groups = collections.defaultdict(list)
    for row in rows:
        image, label = base[row]
        if int(label) != TARGET:
            groups[int(label)].append(image)  # (C, H, W), 0 to 1
    return groups


def build_inputs(by_class, mean_color):
    generator = torch.Generator().manual_seed(SEED)
    classes = [c for c, images in by_class.items() if len(images) >= 4]
    built = {name: ([], []) for name in INPUT_NAMES}

    def pick(pool):
        index = int(torch.randint(len(pool), (1,), generator=generator))
        return pool[index]

    for _ in range(INPUTS):
        own = classes[int(torch.randint(len(classes), (1,), generator=generator))]
        pool = by_class[own]
        picks = [
            pool[i] for i in torch.randperm(len(pool), generator=generator)[:4].tolist()
        ]
        first = picks[0]  # (C, H, W)
        blank = torch.tensor(mean_color).view(3, 1, 1).expand_as(first)
        chosen = [
            classes[i]
            for i in torch.randperm(len(classes), generator=generator)[:4].tolist()
        ]
        # A different-class collage has no own class, so the first image's class
        # stands in for the own-class share.
        entries = {
            "same_class": (collage(picks), own),
            "duplicated": (collage([first] * 4), own),
            "one_quadrant": (collage([first, blank, blank, blank]), own),
            "different_class": (
                collage([pick(by_class[c]) for c in chosen]),
                chosen[0],
            ),
            "single": (
                F.interpolate(first[None], scale_factor=2, mode="nearest")[0],
                own,
            ),
        }
        for name, (image, label) in entries.items():
            built[name][0].append(image)
            built[name][1].append(label)
    inputs = {
        name: (torch.stack(images), torch.tensor(labels))
        for name, (images, labels) in built.items()
    }
    return inputs


# The critical rate of each input along PSBD-TM's own ladder: the smallest rate at
# which most of 3 passes change its unperturbed answer.
def critical_rates(model, architecture, point, images, device):
    labels = logits_of(model, images, device).argmax(dim=1)  # (n,)
    argmax_by_rate = {}
    for rate in LADDER:
        handles = plug_dropout(
            model,
            architecture,
            point["positions"],
            {p: point["factory"] for p in point["positions"]},
            rate,
        )
        try:
            seed_everything(MASK_SEED, verbose=False)
            argmax_by_rate[rate] = torch.stack(
                [logits_of(model, images, device).argmax(dim=1) for _ in range(PASSES)]
            )  # (k, n)
        finally:
            unplug_dropout(handles)
    critical = critical_rate(labels, list(LADDER), argmax_by_rate, flip_fraction=0.5)
    return critical


def collage(images):
    top = torch.cat([images[0], images[1]], dim=2)  # (C, H, 2W)
    bottom = torch.cat([images[2], images[3]], dim=2)  # (C, H, 2W)
    tiled = torch.cat([top, bottom], dim=1)  # (C, 2H, 2W)
    return tiled


if __name__ == "__main__":
    main()
