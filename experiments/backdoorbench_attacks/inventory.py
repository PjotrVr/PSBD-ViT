"""Inventory of backdoor_bench_checkpoints/: what each folder holds and what its trigger touches.

CPU only, reads the folder tree read-only. For every folder it records the attack,
dataset, poison rate, label mode and target that data.backdoorbench infers, whether
the bd_test PNGs line up with this project's clean test set index by index (the
original label BackdoorBench stored against the label at that index) and how much
of the image the trigger changes, measured on PNGs against their clean twins at
the native resolution. The footprint also checks the 32-pixel input handling: a
patch trigger read through a resize that differs from BackdoorBench's would show
changes over the whole image.

    .venv/bin/python -m experiments.backdoorbench_attacks.inventory
"""

import os
import time

import numpy as np
import torch
from PIL import Image

from data.backdoorbench import (
    BACKDOORBENCH_WEIGHTS_DIR,
    backdoor_image_paths,
    backdoor_test_labels,
    backdoorbench_metadata,
    parse_backdoorbench_folder,
    read_backdoorbench_record,
)
from data.loading import extract_labels
from data.splits import load_clean_test_base
from experiments.backdoorbench_attacks.common import (
    INVENTORY_PATH,
    LEADERBOARD,
    write_json,
)

RAW_DATA_DIR = "raw_data"
FOOTPRINT_SAMPLES = 300
FOOTPRINT_SEED = 0
# A pixel counts as changed when some channel moved by more than this many of 255
# levels, above PNG rounding and below any visible trigger.
CHANGED_LEVELS = 4
# The ViT reads a 14 by 14 grid of 16-pixel patches of the 224-pixel resize.
TOKEN_GRID = 14


def main():
    started = time.perf_counter()
    torch.set_num_threads(4)

    test_bases = {}
    rows = []
    for folder in sorted(os.listdir(BACKDOORBENCH_WEIGHTS_DIR)):
        row = describe_folder(folder, test_bases)
        rows.append(row)
        print(summary_line(row), flush=True)

    payload = {
        "experiment": "backdoorbench_attacks inventory",
        "weights_dir": BACKDOORBENCH_WEIGHTS_DIR,
        "footprint_samples": FOOTPRINT_SAMPLES,
        "changed_levels": CHANGED_LEVELS,
        "models": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(payload, INVENTORY_PATH)
    print(f"wrote {INVENTORY_PATH} in {payload['wall_seconds']:.0f} s")


def describe_folder(folder, test_bases):
    dataset, attack, poison_rate = parse_backdoorbench_folder(folder)
    checkpoint_path = os.path.join(
        BACKDOORBENCH_WEIGHTS_DIR, folder, "attack_result.pt"
    )
    row = {
        "folder": folder,
        "dataset": dataset,
        "attack": attack,
        "poison_rate": poison_rate,
        "contents": sorted(os.listdir(os.path.join(BACKDOORBENCH_WEIGHTS_DIR, folder))),
        "leaderboard": LEADERBOARD.get(folder),
    }
    if not os.path.exists(checkpoint_path):
        row["status"] = "no attack_result.pt"
        return row

    record = read_backdoorbench_record(checkpoint_path)
    metadata = backdoorbench_metadata(folder, record)
    recorded = backdoor_test_labels(record)
    image_paths = backdoor_image_paths(checkpoint_path, record)

    if dataset not in test_bases:
        test_bases[dataset] = load_clean_test_base(dataset, RAW_DATA_DIR)
    test_base = test_bases[dataset]
    test_labels = extract_labels(test_base)
    aligned = [
        test_labels[index] == original for index, (_, original) in recorded.items()
    ]

    row.update(
        {
            "status": "readable",
            "model_name": record["model_name"],
            "img_size": list(record["img_size"]),
            "num_classes": record["num_classes"],
            "label_mode": metadata["label_mode"],
            "target_label": metadata["target_label"],
            "n_test": len(test_base),
            "n_bd_test": len(recorded),
            "bd_test_format": record["bd_test"]["bd_data_container"][
                "save_file_format"
            ],
            "images_match_record": set(image_paths) == set(recorded),
            "label_alignment": sum(aligned) / len(aligned),
            "footprint": trigger_footprint(image_paths, test_base),
        }
    )
    return row


def trigger_footprint(image_paths, test_base):
    rng = np.random.default_rng(FOOTPRINT_SEED)
    indices = sorted(image_paths)
    chosen = rng.choice(
        indices, size=min(FOOTPRINT_SAMPLES, len(indices)), replace=False
    )

    mean_abs = []
    changed_pixel_share = []
    changed_token_share = []
    changed_masks = []
    for index in chosen:
        triggered = np.asarray(
            Image.open(image_paths[int(index)]).convert("RGB"), dtype=np.float32
        )
        clean_tensor, _ = test_base[int(index)]  # (C, H, W) in 0 to 1
        clean = clean_tensor.permute(1, 2, 0).numpy() * 255  # (H, W, C)
        assert triggered.shape == clean.shape, (triggered.shape, clean.shape)

        difference = np.abs(triggered - clean)  # (H, W, C)
        changed = difference.max(axis=2) > CHANGED_LEVELS  # (H, W)
        mean_abs.append(float(difference.mean()))
        changed_pixel_share.append(float(changed.mean()))
        changed_token_share.append(token_share(changed))
        changed_masks.append(changed)

    # A fixed trigger changes the same pixels on every image, a sample-specific
    # trigger changes different pixels, so the share of pixels changed on at least
    # half the images against the mean share per image separates the 2.
    stacked = np.stack(changed_masks)  # (samples, H, W)
    footprint = {
        "n": len(chosen),
        "mean_abs_levels": float(np.mean(mean_abs)),
        "changed_pixel_share": float(np.mean(changed_pixel_share)),
        "changed_token_share": float(np.mean(changed_token_share)),
        "consistent_pixel_share": float((stacked.mean(axis=0) >= 0.5).mean()),
        "height": int(stacked.shape[1]),
    }
    return footprint


def token_share(changed):
    height, width = changed.shape
    rows = (
        np.arange(height) * TOKEN_GRID
    ) // height  # (H,), token row of each pixel row
    columns = (np.arange(width) * TOKEN_GRID) // width  # (W,)
    touched = np.zeros((TOKEN_GRID, TOKEN_GRID), dtype=bool)
    pixel_rows, pixel_columns = np.nonzero(changed)
    touched[rows[pixel_rows], columns[pixel_columns]] = True
    share = float(touched.mean())
    return share


def summary_line(row):
    if row["status"] != "readable":
        return f"{row['folder']}: {row['status']}"
    footprint = row["footprint"]
    line = (
        f"{row['folder']}: {row['label_mode']} target {row['target_label']} "
        f"bd_test {row['n_bd_test']} aligned {row['label_alignment']:.4f} "
        f"pixels {footprint['changed_pixel_share']:.3f} tokens {footprint['changed_token_share']:.3f} "
        f"consistent {footprint['consistent_pixel_share']:.3f} mean_abs {footprint['mean_abs_levels']:.2f}"
    )
    return line


if __name__ == "__main__":
    main()
