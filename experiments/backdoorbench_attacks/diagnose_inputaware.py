"""CPU diagnosis of a BackdoorBench checkpoint that failed the reproduction gate on ASR.

Input-Aware draws a per-image trigger from a generator network. BackdoorBench
writes the triggered test images to bd_test_dataset/ through ToPILImage, so they
are 8-bit, and measures its ASR by reading those same files back. The checkpoint
keeps the classifier only, not the generator or the mask network, so the
triggered images cannot be drawn again. This script reads a fixed subset of the
saved bd_test images through BackdoorBench's own test transform (Resize, ToTensor,
Normalize with its statistics) rather than this project's loader, and measures
ASR and robust accuracy 3 ways: float32, bfloat16 autocast (the precision of
evaluate.py), and float32 with every pixel raised half a level, which undoes on
average the floor that ToPILImage's mul(255).byte() applies to a float image.

    .venv/bin/python -m experiments.backdoorbench_attacks.diagnose_inputaware --folder cifar10_inputaware_0_1
"""

import argparse
import os
import time

import torch
import torchvision.transforms as transforms
from PIL import Image

from data.backdoorbench import (
    BACKDOORBENCH_NORMALIZATION,
    BACKDOORBENCH_WEIGHTS_DIR,
    backdoor_image_paths,
    backdoorbench_metadata,
    read_backdoorbench_record,
)
from data.loading import extract_labels
from data.registry import DATASET_REGISTRY
from data.splits import load_clean_test_base
from experiments.backdoorbench_attacks.common import RECORDS_DIR, write_json
from models.backbones import load_checkpoint

RAW_DATA_DIR = "raw_data"
SUBSET_SIZE = 600
SUBSET_SEED = 0
BATCH_SIZE = 50
TORCH_THREADS = 8
HALF_LEVEL = 0.5 / 255
DIAGNOSIS_DIR = os.path.join(RECORDS_DIR, "diagnosis")


def main():
    args = parse_args()
    torch.set_num_threads(TORCH_THREADS)
    started = time.perf_counter()

    checkpoint_path = os.path.join(
        BACKDOORBENCH_WEIGHTS_DIR, args.folder, "attack_result.pt"
    )
    record = read_backdoorbench_record(checkpoint_path)
    metadata = backdoorbench_metadata(args.folder, record)
    model = load_checkpoint(
        metadata["architecture"], checkpoint_path, torch.device("cpu")
    ).eval()

    pixels, true_labels, indices = load_subset(checkpoint_path, record, metadata)
    mean, std = BACKDOORBENCH_NORMALIZATION[metadata["dataset"]]
    normalize = transforms.Normalize(mean, std)
    target = metadata["target_label"]

    readings = {
        "float32": measure(model, normalize(pixels), true_labels, target, False),
        "bfloat16": measure(model, normalize(pixels), true_labels, target, True),
        "float32_half_level_up": measure(
            model,
            normalize((pixels + HALF_LEVEL).clamp(0, 1)),
            true_labels,
            target,
            False,
        ),
    }
    report = {
        "folder": args.folder,
        "subset_size": len(indices),
        "subset_seed": SUBSET_SEED,
        "bd_test_images": len(backdoor_image_paths(checkpoint_path, record)),
        "checkpoint_keys": sorted(record.keys()),
        "readings": readings,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(report, os.path.join(DIAGNOSIS_DIR, f"{args.folder}.json"))
    print(report)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()
    return args


def load_subset(checkpoint_path, record, metadata):
    spec = DATASET_REGISTRY[metadata["dataset"]]
    # BackdoorBench's test transform up to its Normalize, which is applied later
    # so the half-level reading can shift raw pixels.
    to_pixels = transforms.Compose(
        [transforms.Resize((spec.image_size, spec.image_size)), transforms.ToTensor()]
    )
    image_paths = backdoor_image_paths(checkpoint_path, record)
    test_labels = extract_labels(
        load_clean_test_base(metadata["dataset"], RAW_DATA_DIR)
    )

    ordered = sorted(image_paths)
    generator = torch.Generator().manual_seed(SUBSET_SEED)
    picks = torch.randperm(len(ordered), generator=generator)[:SUBSET_SIZE].tolist()
    indices = [ordered[pick] for pick in picks]

    pixels = torch.stack(
        [to_pixels(Image.open(image_paths[index]).convert("RGB")) for index in indices]
    )  # (subset, 3, height, width)
    true_labels = torch.tensor([test_labels[index] for index in indices])  # (subset,)
    return pixels, true_labels, indices


@torch.inference_mode()
def measure(model, images, true_labels, target, use_bfloat16):
    predicted_batches = []
    for start in range(0, len(images), BATCH_SIZE):
        batch = images[start : start + BATCH_SIZE]  # (batch, 3, height, width)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_bfloat16):
            logits = model(batch)  # (batch, num_classes)
        predicted_batches.append(logits.float().argmax(dim=1))  # (batch,)
    predicted = torch.cat(predicted_batches)  # (subset,)

    reading = {
        "asr": float((predicted == target).float().mean()),
        "robust_accuracy": float((predicted == true_labels).float().mean()),
    }
    return reading


if __name__ == "__main__":
    main()
