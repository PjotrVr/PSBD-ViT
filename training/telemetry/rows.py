"""The light probe: loss, train accuracy and logit margin, split by what a row is.

A training row is clean, poisoned (the trigger with the attack's label) or cover
(the trigger or a trigger-like perturbation with the true label, which WaNet,
Adaptive-Blend, BPP and TaCT use to keep the backdoor specific). The 3 fit at
different speeds, and the gap between the clean and the poisoned curve is how the
backdoor's build-up shows during training. The loop already holds each batch's
logits, so the split costs a handful of reductions per step and no forward.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

from .window import rounded

ROW_GROUPS = ("clean", "poisoned", "cover")
CLEAN_ROW, POISONED_ROW, COVER_ROW = 0, 1, 2

# The 4 per-group sums a window carries, in this column order.
ROW_SUM_COLUMNS = ("count", "loss", "correct", "margin")


def row_group_codes(dataset: Dataset) -> torch.Tensor:
    """Each training row's group code, (len(dataset),) int64, indexed as the loader numbers rows.

    Walks the wrapper chain down to the set that carries poison_indices (and
    cover_indices on the cover path). The Augmented, Flagged and Indexed wrappers
    keep the row numbering and a Subset (--exclude-indices-file) renumbers it, so
    the walk carries the numbering through every Subset it passes. A dataset with
    no poison_indices anywhere is all clean.
    """
    positions = torch.arange(len(dataset))  # (rows,)
    current = dataset
    while current is not None:
        poison_indices = getattr(current, "poison_indices", None)
        if poison_indices is not None:
            codes = torch.full((len(current),), CLEAN_ROW, dtype=torch.int64)
            cover_indices = getattr(current, "cover_indices", None) or ()
            codes[sorted(cover_indices)] = COVER_ROW
            # A row in both sets was poisoned, which is what the label says.
            codes[sorted(poison_indices)] = POISONED_ROW
            row_codes = codes[positions]  # (rows,)
            return row_codes

        if isinstance(current, Subset):
            positions = torch.as_tensor(current.indices, dtype=torch.int64)[positions]
            current = current.dataset
            continue
        inner = getattr(current, "inner", None)
        current = inner if inner is not None else getattr(current, "base_dataset", None)

    all_clean = torch.full((len(dataset),), CLEAN_ROW, dtype=torch.int64)
    return all_clean


def label_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """The labeled class's logit minus the largest other logit, (batch,).

    Positive when the row is classified as labeled. For a poisoned row the label
    is the attack's, so this is how far the backdoor carries the row.
    """
    label_logits = logits.gather(1, labels[:, None]).squeeze(1)  # (batch,)
    others = logits.scatter(1, labels[:, None], float("-inf"))  # (batch, classes)
    margin = label_logits - others.max(dim=1).values  # (batch,)
    return margin


def row_statistic_sums(
    logits: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor
) -> torch.Tensor:
    """Per-group sums of count, loss, correct and margin, (3, 4) float64 on the device.

    logits is (batch, classes) as the update computed them, labels and groups are
    (batch,). The loss is the per-row cross entropy against the training label,
    the same quantity the update's mean loss averages.
    """
    assert logits.shape[0] == labels.shape[0] == groups.shape[0], (
        "1 label and 1 group per row"
    )
    logits = logits.detach().float()  # (batch, classes)

    row_loss = F.cross_entropy(logits, labels, reduction="none")  # (batch,)
    row_correct = (logits.argmax(dim=1) == labels).float()  # (batch,)
    row_margin = label_margin(logits, labels)  # (batch,)
    row_values = torch.stack(
        [torch.ones_like(row_loss), row_loss, row_correct, row_margin], dim=1
    ).double()  # (batch, 4)

    membership = F.one_hot(groups, num_classes=len(ROW_GROUPS)).double()  # (batch, 3)
    sums = membership.T @ row_values  # (3, 4)
    return sums


def row_statistic_keys(sums: np.ndarray) -> dict:
    """The window's means per group, a group with no row in the window left out.

    sums is the flat (3 * 4,) read of the accumulated row_statistic_sums.
    """
    table = sums.reshape(len(ROW_GROUPS), len(ROW_SUM_COLUMNS))  # (3, 4)
    keys = {}
    for group_index, group in enumerate(ROW_GROUPS):
        count, loss, correct, margin = table[group_index]
        if count == 0:
            continue
        keys[f"n_{group}"] = int(count)
        keys[f"loss_{group}"] = rounded(loss / count)
        keys[f"accuracy_{group}"] = rounded(correct / count)
        keys[f"margin_{group}"] = rounded(margin / count)

    total_count = table[:, 0].sum()
    if total_count:
        keys["loss"] = rounded(table[:, 1].sum() / total_count)
    return keys
