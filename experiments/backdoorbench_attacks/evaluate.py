"""Clean accuracy and ASR of BackdoorBench checkpoints through this project's loader.

Before any PSBD number is read from backdoor_bench_checkpoints/, each checkpoint
has to reproduce what BackdoorBench reports for it: models.backbones.load_checkpoint
builds the model, evaluation.metrics scores it, the clean test set is this
project's raw data resized to the native size and the triggered set is the
checkpoint's own bd_test images, both normalized with BackdoorBench's statistics
(data.backdoorbench.BACKDOORBENCH_NORMALIZATION). Clean accuracy is over the whole
test set and ASR over every eligible bd_test image, BackdoorBench's definitions.

A CIFAR-10 control repeats 1 model under DATASET_REGISTRY's normalization, the
normalization experiments/wanet_cifar10_audit read BackdoorBench's WaNet
checkpoint with.

1 model per call, so the queue can hold a GPU lock slot per model.

    .venv/bin/python -m experiments.backdoorbench_attacks.evaluate --folder cifar10_ssba_0_1
"""

import argparse
import os
import time

import torch
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader

from data.backdoorbench import (
    BACKDOORBENCH_NORMALIZATION,
    BACKDOORBENCH_WEIGHTS_DIR,
    PngPathDataset,
    backdoor_image_paths,
    backdoor_test_labels,
    backdoorbench_metadata,
    read_backdoorbench_record,
)
from data.loading import base_image_transform, extract_labels
from data.registry import DATASET_REGISTRY
from data.splits import load_clean_test_base
from attacks.poisoning import PoisonedTrainingSet
from evaluation.metrics import attack_success_rate, clean_accuracy
from experiments.backdoorbench_attacks.common import (
    EVALUATION_DIR,
    LEADERBOARD,
    write_json,
)
from models.backbones import load_checkpoint

RAW_DATA_DIR = "raw_data"
BATCH_SIZE = 256
NUM_WORKERS = 4
# The login A100 is shared tonight. 0.15 of 40 GB holds a ViT-B/16 forward pass in
# bfloat16 at batch 256 with room to spare.
GPU_MEMORY_FRACTION = 0.15
REGISTRY_CONTROL_FOLDERS = ("cifar10_trojannn_0_1",)


def main():
    args = parse_args()
    output = os.path.join(EVALUATION_DIR, f"{args.folder}.json")
    if os.path.exists(output):
        print(f"[skip] {args.folder}: {output} exists")
        return

    started = time.perf_counter()
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)

    checkpoint_path = os.path.join(
        BACKDOORBENCH_WEIGHTS_DIR, args.folder, "attack_result.pt"
    )
    record = read_backdoorbench_record(checkpoint_path)
    metadata = backdoorbench_metadata(args.folder, record)
    model = load_checkpoint(metadata["architecture"], checkpoint_path, device)

    backdoorbench_mean, backdoorbench_std = BACKDOORBENCH_NORMALIZATION[
        metadata["dataset"]
    ]
    report = {
        "folder": args.folder,
        **metadata,
        "bd_test_format": record["bd_test"]["bd_data_container"]["save_file_format"],
        "leaderboard": LEADERBOARD.get(args.folder),
        "backdoorbench_normalization": score(
            model,
            record,
            checkpoint_path,
            metadata,
            backdoorbench_mean,
            backdoorbench_std,
            device,
        ),
    }
    if args.folder in REGISTRY_CONTROL_FOLDERS:
        spec = DATASET_REGISTRY[metadata["dataset"]]
        report["registry_normalization"] = score(
            model, record, checkpoint_path, metadata, spec.mean, spec.std, device
        )

    report["device"] = torch.cuda.get_device_name(device)
    report["wall_seconds"] = time.perf_counter() - started
    write_json(report, output)
    scores = report["backdoorbench_normalization"]
    print(
        f"{args.folder}: clean {scores['clean_accuracy']:.4f} asr {scores['asr']:.4f} "
        f"leaderboard {report['leaderboard']} in {report['wall_seconds']:.0f} s"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()
    return args


def score(model, record, checkpoint_path, metadata, mean, std, device):
    spec = DATASET_REGISTRY[metadata["dataset"]]
    normalize = transforms_v2.Normalize(mean=mean, std=std)
    test_base = load_clean_test_base(metadata["dataset"], RAW_DATA_DIR)

    clean_set = PoisonedTrainingSet(test_base, None, set(), normalize, spec.num_classes)
    image_paths = backdoor_image_paths(checkpoint_path, record)
    triggered_indices = sorted(image_paths)
    test_labels = extract_labels(test_base)
    backdoor_set = PngPathDataset(
        [image_paths[index] for index in triggered_indices],
        transform=transforms_v2.Compose(
            [base_image_transform(spec.image_size), normalize]
        ),
        true_labels=[test_labels[index] for index in triggered_indices],
        label_mode=metadata["label_mode"],
        target_label=metadata["target_label"],
        num_classes=spec.num_classes,
    )
    assert len(backdoor_set) <= len(backdoor_test_labels(record))

    def loader(dataset):
        built = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )
        return built

    scores = {
        "mean": list(mean),
        "std": list(std),
        "n_clean": len(clean_set),
        "n_backdoor": len(backdoor_set),
        "clean_accuracy": clean_accuracy(
            model, loader(clean_set), device, use_bfloat16=True
        ),
        "asr": attack_success_rate(
            model, loader(backdoor_set), device, use_bfloat16=True
        ),
    }
    return scores


if __name__ == "__main__":
    main()
