"""Loading BackdoorBench poisoned test images and building the evaluation splits.

BackdoorBench saves each poisoned test image as a PNG whose filename is the
original dataset index. That index lets us recover the exact clean counterpart for
a paired clean/backdoor comparison, which is the only way to read a poisoned test
set this project did not generate itself.

This is the PNG-backed twin of data.splits, which rebuilds a poisoned test set in
memory from the attack recorded in a checkpoint's args.json. BackdoorBench folders
carry no args.json, so they come through here instead. The eval-time eligibility
and labeling questions are the same in both paths, and both answer them with
attacks.poisoning's eval-time function pair, never the training-time pair.

build_psbd_loaders_from_backdoorbench is the PSBD split of such a folder, the same
permutation, heldout size and manifest data.splits builds for this project's own
checkpoints, so cli.sweep and cli.analyze read both kinds of cache unchanged. The
results/ folder of a BackdoorBench checkpoint is bb_<folder>.
"""

import glob
import os
import re
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as transforms_v2
from lightning import seed_everything
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from .loading import base_image_transform, extract_labels
from .registry import DATASET_REGISTRY, label_mode_from_folder
from .splits import (
    PSBD_HELDOUT_SIZE,
    PSBD_SPLIT_SEED,
    load_clean_test_base,
    psbd_split_permutation,
    read_checkpoint_metadata,
)
from attacks.poisoning import (
    PoisonedTrainingSet,
    attack_success_label,
    is_eval_poisonable,
)

BACKDOORBENCH_WEIGHTS_DIR = "backdoor_bench_checkpoints"
# A results/ folder named bb_<folder> holds the PSBD cache of
# backdoor_bench_checkpoints/<folder>, so the 2 checkpoint sources never share a
# results folder and the folder name alone says which loader built its splits.
BACKDOORBENCH_RESULTS_PREFIX = "bb_"

# BackdoorBench's get_dataset_normalization, copied from
# third_party/BackdoorBench/utils/aggregate_block/dataset_and_transform_generate.py.
# Its checkpoints were trained on these statistics. CIFAR-10's std differs from
# DATASET_REGISTRY's (0.247 against 0.2023 in the red channel), so reading a
# BackdoorBench model through the registry's normalization shifts every input.
BACKDOORBENCH_NORMALIZATION: dict[str, tuple[tuple, tuple]] = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "gtsrb": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "tiny": ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
}

# BackdoorBench's model_name for each architecture this project can build. Its
# vit_b_16 is Sequential(Resize(224), torchvision vit_b_16), the module tree
# models.backbones.build_vit returns, so the state dict loads strictly.
BACKDOORBENCH_ARCHITECTURES = {"vit_b_16": "vit"}

# The largest share of bd_test entries whose recorded original label may disagree
# with the clean test set before the 2 are treated as differently ordered.
MAX_MISALIGNED_SHARE = 0.01


class PngPathDataset(Dataset):
    """Serves poisoned images from PNG files, filtered and labeled like AttackSuccessSet.

    Eligibility and labeling follow the same eval-time question AttackSuccessSet
    asks in memory: is this sample allowed into the attack-success set, and what
    label counts as success. Ineligible paths (for example already-target-class
    images under all_to_one) are filtered out at construction time rather than
    served with a wrong label.
    """

    def __init__(
        self,
        paths: list[str],
        transform,
        true_labels: list[int],
        label_mode: str,
        target_label: int,
        num_classes: int,
    ):
        self.transform = transform
        self.true_labels = true_labels
        self.label_mode = label_mode
        self.target_label = target_label
        self.num_classes = num_classes
        self.paths = paths
        self.eligible_positions = [
            position
            for position, label in enumerate(true_labels)
            if is_eval_poisonable(label_mode, int(label), target_label)
        ]

    def __len__(self) -> int:
        return len(self.eligible_positions)

    def __getitem__(self, idx: int) -> tuple:
        position = self.eligible_positions[idx]
        image = Image.open(self.paths[position]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = attack_success_label(
            self.label_mode,
            int(self.true_labels[position]),
            self.target_label,
            self.num_classes,
        )
        return image, label


def load_backdoor_splits(
    folder_name: str,
    clean_test_dataset: Dataset,
    transform,
    weights_dir: str,
    label_mode: str,
    target_label: int,
    num_classes: int,
) -> tuple[Dataset, Dataset]:
    """(backdoor_test, clean_counterparts), both filtered to eligible samples.

    The 2 are index-aligned: position i in each is the same original test image,
    with and without the trigger. Samples is_eval_poisonable excludes are dropped
    from both, so backdoor_test can be shorter than the PNG folder itself.
    """
    backdoor_dir = os.path.join(weights_dir, folder_name, "bd_test_dataset")
    backdoor_paths = sorted(glob.glob(f"{backdoor_dir}/**/*.png", recursive=True))
    if not backdoor_paths:
        raise FileNotFoundError(f"No PNG files found in {backdoor_dir}")

    # The filename stem is the original test-set index, which is the only link
    # back to the clean image the trigger was stamped on.
    original_indices = [int(Path(p).stem) for p in backdoor_paths]
    clean_counterparts = Subset(clean_test_dataset, original_indices)
    true_labels = extract_labels(clean_counterparts)

    backdoor_test = PngPathDataset(
        backdoor_paths,
        transform=transform,
        true_labels=true_labels,
        label_mode=label_mode,
        target_label=target_label,
        num_classes=num_classes,
    )
    eligible_counterparts = Subset(clean_counterparts, backdoor_test.eligible_positions)
    return backdoor_test, eligible_counterparts


def split_validation_and_eval(
    clean_counterparts: Dataset,
    backdoor_test: Dataset,
    clean_val_size: int,
    seed: int,
) -> tuple[Subset, Subset, Subset]:
    """(clean_val, clean_eval, backdoor_eval), with a validation set carved off the clean side.

    Stratified by label so the validation quantile threshold sees every class. The
    backdoor eval set reuses the clean eval indices, which keeps the pairing.
    """
    labels = np.array(extract_labels(clean_counterparts))
    indices = np.arange(len(clean_counterparts))

    val_idx, eval_idx = train_test_split(
        indices,
        test_size=len(indices) - clean_val_size,
        stratify=labels,
        random_state=seed,
    )

    clean_val = Subset(clean_counterparts, val_idx.tolist())
    clean_eval = Subset(clean_counterparts, eval_idx.tolist())
    backdoor_eval = Subset(backdoor_test, eval_idx.tolist())
    return clean_val, clean_eval, backdoor_eval


def balance_by_class(
    clean_eval: Dataset,
    backdoor_eval: Dataset,
    examples_per_class: int,
    seed: int,
) -> tuple[Subset, Subset]:
    """The 2 eval sets cut to a fixed number of examples per clean class.

    The backdoor side reuses the selected clean indices so the pairing survives.
    """
    clean_labels = extract_labels(clean_eval)

    class_to_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(clean_labels):
        class_to_indices.setdefault(int(label), []).append(idx)

    seed_everything(seed)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in sorted(class_to_indices):
        candidates = class_to_indices[class_id]
        rng.shuffle(candidates)
        selected.extend(candidates[:examples_per_class])

    balanced_clean = Subset(clean_eval, selected)
    balanced_backdoor = Subset(backdoor_eval, selected)
    return balanced_clean, balanced_backdoor


def is_backdoorbench_folder(results_folder: str) -> bool:
    """Whether a results/ folder name names a BackdoorBench checkpoint (bb_<folder>)."""
    return results_folder.startswith(BACKDOORBENCH_RESULTS_PREFIX)


def backdoorbench_checkpoint_path(
    results_folder: str, weights_dir: str = BACKDOORBENCH_WEIGHTS_DIR
) -> str:
    """The attack_result.pt behind results folder bb_<folder>."""
    folder_name = results_folder.removeprefix(BACKDOORBENCH_RESULTS_PREFIX)
    checkpoint_path = os.path.join(weights_dir, folder_name, "attack_result.pt")
    return checkpoint_path


def parse_backdoorbench_folder(folder_name: str) -> tuple[str, str, float]:
    """(dataset, attack, poison rate) from a dataset_attack_rate folder name.

    The rate tag spells the decimal digits after "0_", so "0_1" is 0.1 and "0_005"
    is 0.005, the tag Tiny's clean-label folders carry.
    """
    match = re.fullmatch(r"([a-z0-9]+)_([a-z0-9]+)_0_(\d+)", folder_name)
    if match is None:
        raise ValueError(f"{folder_name!r} is not a dataset_attack_rate folder name")

    dataset_name, attack_name, rate_digits = match.groups()
    poison_rate = float(f"0.{rate_digits}")
    return dataset_name, attack_name, poison_rate


def read_backdoorbench_record(checkpoint_path: str) -> dict:
    """A BackdoorBench attack_result.pt, memory-mapped so the weights stay on disk.

    Only the bookkeeping entries are read here (model_name, num_classes and the
    bd_test data_dict), which takes well under a second against several seconds
    for a full read of the 350 MB file.
    """
    record = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    return record


def backdoor_test_labels(record: dict) -> dict[int, tuple[int, int]]:
    """{original test index: (backdoor label, original label)} of the bd_test set.

    BackdoorBench keys bd_test's data_dict by the original test index and stores
    [backdoor label, original label] in each entry's other_info.
    """
    data_dict = record["bd_test"]["bd_data_container"]["data_dict"]
    labels = {
        int(index): (int(entry["other_info"][0]), int(entry["other_info"][1]))
        for index, entry in data_dict.items()
    }
    return labels


def backdoorbench_metadata(folder_name: str, record: dict) -> dict:
    """The args.json fields a BackdoorBench checkpoint implies, from its folder and record.

    The label mode comes from the folder name (label_mode_from_folder) and the
    target from the bd_test labels, which must name exactly 1 class for the
    single-target modes. Raises rather than guessing on an architecture this
    project cannot build or on a bd_test set with several targets.
    """
    dataset_name, attack_name, poison_rate = parse_backdoorbench_folder(folder_name)
    label_mode = label_mode_from_folder(folder_name)

    model_name = record["model_name"]
    if model_name not in BACKDOORBENCH_ARCHITECTURES:
        raise ValueError(f"{folder_name}: no builder for model_name {model_name!r}")
    if record["num_classes"] != DATASET_REGISTRY[dataset_name].num_classes:
        raise ValueError(
            f"{folder_name}: {record['num_classes']} classes, the registry's "
            f"{dataset_name} has {DATASET_REGISTRY[dataset_name].num_classes}"
        )

    backdoor_labels = {bd for bd, _ in backdoor_test_labels(record).values()}
    if label_mode not in ("all_to_one", "clean_label") or len(backdoor_labels) != 1:
        raise ValueError(
            f"{folder_name}: label mode {label_mode} with backdoor labels "
            f"{sorted(backdoor_labels)[:5]}, only single-target folders are read"
        )

    metadata = {
        "dataset": dataset_name,
        "attack": attack_name,
        "label_mode": label_mode,
        "target_label": backdoor_labels.pop(),
        "poison_rate": poison_rate,
        "architecture": BACKDOORBENCH_ARCHITECTURES[model_name],
        "source": "backdoorbench",
    }
    return metadata


def read_backdoorbench_metadata(checkpoint_path: str) -> dict:
    """backdoorbench_metadata for the checkpoint at checkpoint_path."""
    folder_name = os.path.basename(os.path.dirname(checkpoint_path))
    metadata = backdoorbench_metadata(
        folder_name, read_backdoorbench_record(checkpoint_path)
    )
    return metadata


def resolve_checkpoint(results_folder: str, checkpoints_dir: str) -> tuple[str, dict]:
    """(checkpoint path, metadata) for a results/ folder name of either source.

    bb_<folder> resolves to backdoor_bench_checkpoints/<folder> and metadata read
    from its record. Any other name resolves to checkpoints_dir/<folder> and its
    args.json sidecar.
    """
    if is_backdoorbench_folder(results_folder):
        checkpoint_path = backdoorbench_checkpoint_path(results_folder)
        metadata = read_backdoorbench_metadata(checkpoint_path)
        return checkpoint_path, metadata

    checkpoint_path = os.path.join(checkpoints_dir, results_folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)
    return checkpoint_path, metadata


def backdoor_image_paths(checkpoint_path: str, record: dict) -> dict[int, str]:
    """{original test index: image path} of the checkpoint's bd_test_dataset folder.

    Most attacks save PNGs and Blind saves JPEGs. The record's save_file_format
    says which, so the triggered images read here are the ones BackdoorBench's own
    evaluation read.
    """
    backdoor_dir = os.path.join(os.path.dirname(checkpoint_path), "bd_test_dataset")
    file_format = record["bd_test"]["bd_data_container"]["save_file_format"]
    paths = glob.glob(f"{backdoor_dir}/**/*{file_format}", recursive=True)
    if not paths:
        raise FileNotFoundError(f"No {file_format} files found in {backdoor_dir}")

    paths_by_index = {int(Path(path).stem): path for path in paths}
    return paths_by_index


def backdoorbench_psbd_datasets(
    test_base: Dataset,
    image_paths_by_index: dict[int, str],
    recorded_labels: dict[int, tuple[int, int]],
    metadata: dict,
    seed: int = PSBD_SPLIT_SEED,
    max_samples: int | None = None,
) -> tuple[dict[str, Dataset], dict]:
    """The (validation, clean, backdoor) datasets of the PSBD split, plus its manifest.

    The split is data.splits' own: the permutation psbd_split_permutation draws
    from (test set size, seed), the first PSBD_HELDOUT_SIZE indices as the clean
    threshold set and the rest as the analysis pool. The backdoor split is every
    analysis index, in analysis order, that has a PNG and passes
    is_eval_poisonable, which is the shape data.splits' manifest has, so
    defenses.decision.pair_clean_to_backdoor pairs it unchanged.

    An entry whose recorded original label disagrees with the clean test label at
    its index is dropped from the backdoor split and listed in the manifest.
    BackdoorBench's WaNet folders carry 13 to 81 such entries, some of them images
    of another test index, so the pair would compare 2 different images. More than
    MAX_MISALIGNED_SHARE of them raises, since that means the 2 sets are ordered
    differently and every pair is wrong.
    """
    dataset_name = metadata["dataset"]
    spec = DATASET_REGISTRY[dataset_name]
    mean, std = BACKDOORBENCH_NORMALIZATION[dataset_name]
    normalize = transforms_v2.Normalize(mean=mean, std=std)

    test_labels = extract_labels(test_base)
    if set(image_paths_by_index) != set(recorded_labels):
        raise ValueError("bd_test PNG files and the record's bd_test entries differ")
    mismatched = [
        index
        for index, (_, original_label) in recorded_labels.items()
        if test_labels[index] != original_label
    ]
    if len(mismatched) > MAX_MISALIGNED_SHARE * len(recorded_labels):
        raise ValueError(
            f"{len(mismatched)} bd_test entries disagree with the clean test label "
            f"at their index (first {mismatched[:5]}), the 2 sets are not aligned"
        )
    misaligned = set(mismatched)

    n_total = len(test_base)
    permutation = psbd_split_permutation(n_total, seed)  # (n_total,)
    heldout_list = permutation[:PSBD_HELDOUT_SIZE].tolist()
    analysis_list = permutation[PSBD_HELDOUT_SIZE:].tolist()
    if max_samples is not None:
        heldout_list = heldout_list[:max_samples]
        analysis_list = analysis_list[:max_samples]

    triggered_indices = [
        index
        for index in analysis_list
        if index in image_paths_by_index and index not in misaligned
    ]
    backdoor_set = PngPathDataset(
        [image_paths_by_index[index] for index in triggered_indices],
        transform=transforms_v2.Compose(
            [base_image_transform(spec.image_size), normalize]
        ),
        true_labels=[test_labels[index] for index in triggered_indices],
        label_mode=metadata["label_mode"],
        target_label=metadata["target_label"],
        num_classes=spec.num_classes,
    )
    analysis_backdoor_indices = [
        triggered_indices[position] for position in backdoor_set.eligible_positions
    ]

    # An empty poison set makes PoisonedTrainingSet a plain normalized clean set,
    # and its attack is never touched.
    datasets = {
        "validation": PoisonedTrainingSet(
            Subset(test_base, heldout_list), None, set(), normalize, spec.num_classes
        ),
        "clean": PoisonedTrainingSet(
            Subset(test_base, analysis_list), None, set(), normalize, spec.num_classes
        ),
        "backdoor": backdoor_set,
    }
    manifest = {
        "seed": seed,
        "dataset": dataset_name,
        "probe_attack": metadata["attack"],
        "probe_target_label": metadata["target_label"],
        "label_mode": metadata["label_mode"],
        "source": "backdoorbench bd_test_dataset PNGs",
        "normalization": {"mean": list(mean), "std": list(std)},
        "n_total": n_total,
        "n_heldout": len(heldout_list),
        "misaligned_bd_test_indices": sorted(mismatched),
        "heldout_indices": heldout_list,
        "analysis_clean_indices": analysis_list,
        "analysis_backdoor_indices": analysis_backdoor_indices,
        "recipe_note": (
            "seed_everything(seed); perm = torch.randperm(n_total); "
            "heldout = perm[:n_heldout]; analysis = perm[n_heldout:]; backdoor = "
            "analysis indices with an eligible bd_test PNG, in analysis order"
        ),
    }
    return datasets, manifest


def build_psbd_loaders_from_backdoorbench(
    checkpoint_path: str,
    seed: int = PSBD_SPLIT_SEED,
    raw_data_dir: str = "raw_data",
    batch_size: int = 64,
    num_workers: int = 2,
    max_samples: int | None = None,
) -> tuple[dict[str, DataLoader], dict]:
    """The 3 PSBD loaders of a BackdoorBench checkpoint, plus the manifest.

    The counterpart of data.splits.build_psbd_loaders_from_checkpoint for a folder
    with no args.json. Clean images are this project's raw test set resized to the
    dataset's native size, triggered images are BackdoorBench's PNGs, and both are
    normalized with BackdoorBench's own statistics, the transform its test loader
    applies (Resize, ToTensor, Normalize). Every loader is shuffle=False.
    """
    record = read_backdoorbench_record(checkpoint_path)
    folder_name = os.path.basename(os.path.dirname(checkpoint_path))
    metadata = backdoorbench_metadata(folder_name, record)
    test_base = load_clean_test_base(metadata["dataset"], raw_data_dir)

    datasets, manifest = backdoorbench_psbd_datasets(
        test_base,
        backdoor_image_paths(checkpoint_path, record),
        backdoor_test_labels(record),
        metadata,
        seed,
        max_samples,
    )
    loaders = {
        split: DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        for split, dataset in datasets.items()
    }
    return loaders, manifest
