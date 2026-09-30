"""Model sets, placements and paths shared by the GPU scoring stage and the CPU readout.

Every path is relative to the working directory, which is the main checkout that
holds checkpoints/, raw_data/ and results/.
"""

import json
import os

import torch

SLUG = "training_set_detection"
CHECKPOINTS_DIR = "checkpoints"
RAW_DATA_DIR = "raw_data"
RESULTS_DIR = "results"
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(EXPERIMENT_DIR))
DEV_SET_PATH = os.path.join(REPO_ROOT, "experiments", "cache_readouts", "dev_set.json")
RECORDS_DIR = os.path.join(RESULTS_DIR, "_experiments", SLUG)
RAW_ROOT = os.path.join(RECORDS_DIR, "raw")

# Li et al.'s main table, their attacks and datasets at 10% poisoning, plus their
# architecture. Label-Consistent is the only clean-label attack of theirs we
# trained, capped at 5% on GTSRB target class 1.
PAPER_MIRROR = (
    "vit_cifar10_badnet_a2o_0_1",
    "vit_cifar10_blend_0_1",
    "vit_cifar10_wanet_0_1",
    "vit_gtsrb_badnet_a2o_0_1",
    "vit_gtsrb_blend_0_1",
    "vit_tiny_badnet_a2o_0_1",
    "vit_tiny_blend_0_1",
    "vit_tiny_wanet_0_1",
    "vit_gtsrb_lc_0_05_tl1_adv",
    "resnet18_gtsrb_badnet_a2o_0_1",
    "resnet18_gtsrb_blend_0_1",
)

# Each method's cache directory under results/<folder>/psbd/ (read only here) and
# how cli.sweep plugs it.
PLACEMENTS = {
    "psbd_tm": {
        "cache": "before_attention_norm_token_mask",
        "position": "before_attention_norm",
        "operator": "token_mask",
        "block_range": None,
    },
    "psbd_rd": {
        "cache": "post_residual",
        "position": "post_residual",
        "operator": "dropout",
        "block_range": None,
    },
    "middle_band": {
        "cache": "pre_residual_blocks_5_8",
        "position": "pre_residual",
        "operator": "dropout",
        "block_range": (5, 8),
    },
}
RESNET_PLACEMENTS = ("psbd_rd",)
VIT_PLACEMENTS = ("psbd_tm", "psbd_rd", "middle_band")

RATE_RULES = ("ours", "li")
SHIFT_TARGET = 0.8
FORWARD_PASSES = 3
MASK_SEED = 0
BATCH_SIZE = 64

# Sample sizes. The clean sample is the task's 10000. Covers are capped because
# WaNet on Tiny carries 20000 of them and they are reported, never thresholded.
CLEAN_SAMPLE_SIZE = 10000
COVER_SAMPLE_SIZE = 2000
MEMBERSHIP_SAMPLE_SIZE = 2000
# Li et al.'s rate rule needs the shift ratio of the whole training set at every
# candidate rate. It is estimated on these prefixes of each group's seeded order.
RATE_SUBSET = {"clean_train": 2000, "poisoned": 1000, "cover": 1000}
# STRIP costs 8 forwards per image and CD-L about 251, so both run on prefixes.
STRIP_SUBSET = {"clean_train": 2000, "poisoned": 1000, "cover": 500}
CD_L_SUBSET = {"clean_train": 1000, "poisoned": 500, "cover": 250}
SAMPLE_SEED = 0
GROUP_STREAMS = {"poisoned": 1, "cover": 2, "clean_train": 3}


def dev_set():
    with open(DEV_SET_PATH) as handle:
        declared = json.load(handle)["models"]
    folders = tuple(entry["folder"] for entry in declared)
    return folders


def model_queue():
    # The paper-mirror set first, then the development models it does not hold.
    queue = list(PAPER_MIRROR) + [f for f in dev_set() if f not in PAPER_MIRROR]
    return queue


def placements_for(folder):
    names = RESNET_PLACEMENTS if folder.startswith("resnet18") else VIT_PLACEMENTS
    return names


def raw_dir(folder):
    path = os.path.join(RAW_ROOT, folder)
    return path


def part_path(folder, part):
    path = os.path.join(raw_dir(folder), f"{part}.pt")
    return path


def passes_part(placement, rate):
    part = f"passes_{placement}_{rate:g}"
    return part


def save_part(payload, folder, part):
    os.makedirs(raw_dir(folder), exist_ok=True)
    path = part_path(folder, part)
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_part(folder, part):
    payload = torch.load(
        part_path(folder, part), map_location="cpu", weights_only=False
    )
    return payload


def has_part(folder, part):
    exists = os.path.exists(part_path(folder, part))
    return exists
