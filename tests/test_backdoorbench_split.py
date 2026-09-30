"""The PSBD split of a BackdoorBench checkpoint, built from a synthetic folder.

cli.sweep reads a results folder named bb_<folder> through
data.backdoorbench.build_psbd_loaders_from_backdoorbench. These tests check the
parts that would fail silently: the split is data.splits' permutation, the
backdoor split keeps only eligible images in analysis order so the clean split
pairs with it, images are normalized with BackdoorBench's statistics and a
bd_test set that is ordered differently from the clean test set raises.
"""

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.backdoorbench import (
    BACKDOORBENCH_NORMALIZATION,
    backdoorbench_checkpoint_path,
    backdoorbench_metadata,
    backdoorbench_psbd_datasets,
    is_backdoorbench_folder,
    parse_backdoorbench_folder,
)
from data.splits import PSBD_HELDOUT_SIZE, psbd_split_permutation
from defenses.decision import pair_clean_to_backdoor

pytestmark = pytest.mark.fast

N_TEST = PSBD_HELDOUT_SIZE + 40
TARGET = 0
PNG_VALUE = 200


class ConstantImages(Dataset):
    """N_TEST black 32x32 images with labels cycling over the 10 classes."""

    def __init__(self):
        self.targets = [index % 10 for index in range(N_TEST)]

    def __len__(self):
        return N_TEST

    def __getitem__(self, index):
        image = torch.zeros(3, 32, 32)  # (C, H, W) in 0 to 1
        return image, self.targets[index]


def fake_record(backdoor_labels, model_name="vit_b_16"):
    data_dict = {
        index: {"path": f"{index}.png", "other_info": [TARGET, index % 10]}
        for index in range(N_TEST)
    }
    for index, label in backdoor_labels.items():
        data_dict[index]["other_info"][0] = label
    record = {
        "model_name": model_name,
        "num_classes": 10,
        "bd_test": {"bd_data_container": {"data_dict": data_dict}},
    }
    return record


@pytest.fixture
def png_folder(tmp_path):
    pixels = np.full((32, 32, 3), PNG_VALUE, dtype=np.uint8)
    paths = {}
    for index in range(N_TEST):
        path = tmp_path / f"{index}.png"
        Image.fromarray(pixels).save(path)
        paths[index] = str(path)
    return paths


def recorded_labels_of(record):
    data_dict = record["bd_test"]["bd_data_container"]["data_dict"]
    labels = {index: tuple(entry["other_info"]) for index, entry in data_dict.items()}
    return labels


def test_folder_names_parse_to_dataset_attack_and_rate():
    assert parse_backdoorbench_folder("cifar10_inputaware_0_1") == (
        "cifar10",
        "inputaware",
        0.1,
    )
    assert parse_backdoorbench_folder("tiny_lc_0_005") == ("tiny", "lc", 0.005)
    assert parse_backdoorbench_folder("gtsrb_trojannn_0_01")[2] == 0.01
    with pytest.raises(ValueError):
        parse_backdoorbench_folder("vit_cifar10_badnet_a2o_0_1")


def test_bb_prefix_resolves_to_the_read_only_folder():
    assert is_backdoorbench_folder("bb_cifar10_ssba_0_1")
    assert not is_backdoorbench_folder("vit_cifar10_badnet_a2o_0_1")
    assert backdoorbench_checkpoint_path("bb_cifar10_ssba_0_1") == (
        "backdoor_bench_checkpoints/cifar10_ssba_0_1/attack_result.pt"
    )


def test_metadata_reads_target_and_label_mode():
    metadata = backdoorbench_metadata("cifar10_sig_0_1", fake_record({}))
    assert metadata["target_label"] == TARGET
    assert metadata["label_mode"] == "clean_label"
    assert metadata["architecture"] == "vit"
    assert metadata["poison_rate"] == 0.1

    with pytest.raises(ValueError):
        backdoorbench_metadata("cifar10_ssba_0_1", fake_record({}, "preactresnet18"))
    with pytest.raises(ValueError):
        backdoorbench_metadata("cifar10_ssba_0_1", fake_record({3: 5}))


def test_split_is_the_psbd_permutation_and_pairs(png_folder):
    record = fake_record({})
    metadata = backdoorbench_metadata("cifar10_ssba_0_1", record)
    datasets, manifest = backdoorbench_psbd_datasets(
        ConstantImages(), png_folder, recorded_labels_of(record), metadata
    )

    permutation = psbd_split_permutation(N_TEST).tolist()
    assert manifest["heldout_indices"] == permutation[:PSBD_HELDOUT_SIZE]
    assert manifest["analysis_clean_indices"] == permutation[PSBD_HELDOUT_SIZE:]
    assert len(datasets["validation"]) == PSBD_HELDOUT_SIZE
    assert len(datasets["clean"]) == N_TEST - PSBD_HELDOUT_SIZE

    # all_to_one drops the target class, and the rest keep analysis order.
    expected_backdoor = [
        index for index in manifest["analysis_clean_indices"] if index % 10 != TARGET
    ]
    assert manifest["analysis_backdoor_indices"] == expected_backdoor
    assert len(datasets["backdoor"]) == len(expected_backdoor)

    clean_scores = torch.tensor(
        manifest["analysis_clean_indices"], dtype=torch.float32
    )  # (n_clean,), each row scored by its own index
    paired = pair_clean_to_backdoor(clean_scores, manifest)  # (n_backdoor,)
    assert paired.tolist() == [float(index) for index in expected_backdoor]


def test_backdoor_rows_carry_target_and_backdoorbench_normalization(png_folder):
    record = fake_record({})
    metadata = backdoorbench_metadata("cifar10_ssba_0_1", record)
    datasets, _ = backdoorbench_psbd_datasets(
        ConstantImages(), png_folder, recorded_labels_of(record), metadata
    )

    image, label = datasets["backdoor"][0]  # image (3, 32, 32)
    mean, std = BACKDOORBENCH_NORMALIZATION["cifar10"]
    expected = (PNG_VALUE / 255 - torch.tensor(mean)) / torch.tensor(std)  # (3,)
    assert label == TARGET
    assert torch.allclose(image[:, 0, 0], expected, atol=1e-5)

    clean_image, _ = datasets["clean"][0]  # (3, 32, 32)
    assert torch.allclose(clean_image[:, 0, 0], -torch.tensor(mean) / torch.tensor(std))


def test_a_few_misaligned_entries_are_dropped_and_many_raise(png_folder):
    record = fake_record({})
    data_dict = record["bd_test"]["bd_data_container"]["data_dict"]
    permutation = psbd_split_permutation(N_TEST).tolist()
    corrupted = next(
        index for index in permutation[PSBD_HELDOUT_SIZE:] if index % 10 != TARGET
    )
    data_dict[corrupted]["other_info"][1] = (corrupted + 1) % 10
    metadata = backdoorbench_metadata("cifar10_ssba_0_1", record)
    _, manifest = backdoorbench_psbd_datasets(
        ConstantImages(), png_folder, recorded_labels_of(record), metadata
    )
    assert manifest["misaligned_bd_test_indices"] == [corrupted]
    assert corrupted not in manifest["analysis_backdoor_indices"]

    for index in range(N_TEST):
        data_dict[index]["other_info"][1] = (index + 1) % 10
    with pytest.raises(ValueError, match="not aligned"):
        backdoorbench_psbd_datasets(
            ConstantImages(), png_folder, recorded_labels_of(record), metadata
        )
