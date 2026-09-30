"""Spectral Signatures: training-set poison filtering (Tran et al., NeurIPS 2018).

Paper: "Spectral Signatures in Backdoor Attacks", arXiv:1811.00636, Algorithm 1.
Reference implementation: backdoor-toolbox, cleansers_tool_box/spectral_signature.py
(pinned in third_party.lock), which is also the baseline Li et al. ran for the
PSBD paper's training-set tables.

    original form (Algorithm 1, per label y)
        R_hat = [R(x_i) - R_bar]_{i=1..n}
        v     = top right singular vector of R_hat
        tau_i = ((R(x_i) - R_bar) . v)^2
        remove the examples with the top 1.5 * epsilon * n scores

    symbols
        R(x_i)   the learned representation of training example i, the vector the
                 classifier head reads
        R_bar    the mean representation over the examples labeled y
        v        the top right singular vector of the centered matrix R_hat
        tau_i    the outlier score of example i
        epsilon  an upper bound on the poisoned fraction, known to the defender
        n        the number of examples labeled y

    descriptive form
        centered   = class_features - mean(class_features)
        direction  = top right singular vector of centered
        score      = (centered @ direction) ** 2

The reference differs from Algorithm 1 in how many examples it removes per
label, and this port follows the reference because the PSBD paper's numbers come
from it: min(int(1.5 * epsilon * N), n // 2), with N the size of the whole
training set rather than of the label, and at most half of each label.

Unlike every detector in the registry this one scores a labeled training set, not
a test input, and needs no clean data, so it takes a feature matrix rather than a
(model, loader) pair and is not registered in detectors.DETECTOR_NAMES. High
means poisoned here, the paper's own orientation, since it never meets the
shared low-means-poisoned decision rule.

Deviations from the reference, each recorded in docs/detectors/spectral_signatures.md:

  1. torch.linalg.svd with full_matrices=False where the reference calls the
     deprecated torch.svd(some=False). The first right singular vector is the
     same up to sign, and the score squares the projection.
  2. The per-example loop over dot products is 1 matrix product.
"""

import torch

# Algorithm 1's removal multiplier, "1.5 epsilon n", in the reference's
# num_poisons_expected.
REMOVAL_MULTIPLIER = 1.5


def spectral_signature_scores(class_features: torch.Tensor) -> torch.Tensor:
    """tau_i for every example of 1 label, shape (n,), high meaning poisoned.

    class_features is the (n, feature_dim) representation matrix of the
    examples that carry the label. Computed in float32 whatever the input dtype,
    as the reference casts to torch.FloatTensor.
    """
    features = class_features.float()  # (n, feature_dim)
    centered = features - features.mean(dim=0, keepdim=True)  # (n, feature_dim)

    # The rows of vh are the right singular vectors, largest singular value first.
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)  # vh (r, feature_dim)
    top_direction = vh[0]  # (feature_dim,)

    projections = centered @ top_direction  # (n,)
    scores = projections.pow(2)  # (n,)
    return scores


def removal_count(class_size: int, poison_rate: float, dataset_size: int) -> int:
    """How many examples of 1 label the reference removes.

    original form (reference cleanser)
        k = min(int(1.5 * epsilon * N), n // 2)

    N is the whole training set, so every label may lose up to half its
    examples. That is what drives the reference's high false-positive rate on a
    dataset with many labels.
    """
    expected = int(REMOVAL_MULTIPLIER * poison_rate * dataset_size)
    count = min(expected, class_size // 2)
    return count


def flag_top(scores: torch.Tensor, count: int) -> torch.Tensor:
    """A (n,) boolean mask of the count highest scores, the examples removed."""
    flagged = torch.zeros_like(scores, dtype=torch.bool)  # (n,)
    if count > 0:
        top_positions = torch.topk(scores, count).indices  # (count,)
        flagged[top_positions] = True
    return flagged
