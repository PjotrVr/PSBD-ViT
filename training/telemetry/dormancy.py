"""How much of the network is still in use: dormant units and effective rank.

Ported from ek_solver's solver/dmc/dormancy.py (the Sokar et al. dormant score and
the Kumar, Agarwal et al. srank) and the reading half of its telemetry/plasticity.py.
The readings run 1 extra forward over a fixed held-out batch in eval mode under
no_grad, so the update itself is untouched. Every value stays a 0-dim device
tensor until the window's single transfer.

2 layers are read. The MLP hidden layer of every block (the GELU output, 3072
units on ViT-B/16) and the classifier features (the head's input, which is the
class token after the final norm on ViT and the pooled feature on Swin). A backdoor
that routes through a few units or collapses the triggered features onto a low
rank subspace shows in these before it shows in accuracy.
"""

import math

import torch
import torch.nn as nn

from analysis.features import transformer_blocks
from defenses.inference import forward_logits
from models.backbones import network_core

DORMANT_TAU = 0.025

# srank_delta: the number of singular values needed to reach 1 - delta of the
# nuclear mass.
SRANK_DELTA = 0.01

# The rank of a (rows, width) matrix is capped by its row count and a spectrum of
# every token of every image (50432 rows on ViT-B/16 with 256 images) is far too
# costly for a probe that runs mid-training. This many token rows per block, spread
# evenly over the tokens of every image, keep the cost at 1 small eigenproblem per
# block and group. The rank is read as a trend over training, and the cap sits far
# above the ranks it reports.
MLP_RANK_ROWS = 512

PROBE_BATCH = 128


def dormant_scores(activations: torch.Tensor) -> torch.Tensor:
    """Sokar et al.'s normalized activation score per unit, (units,).

    activations is (rows, units). A unit's score is its mean absolute activation
    over the rows divided by the layer's mean of that quantity, so the score is
    scale free and a layer's scores average to 1.

        original form
            s_i = E_x |h_i(x)| / ((1/H) * sum_k E_x |h_k(x)|)

        symbol table
            h_i(x)   activation of unit i on input x
            H        number of units in the layer
    """
    magnitude = activations.abs().mean(dim=0)  # (units,)
    scores = scores_from_magnitude(magnitude)  # (units,)
    return scores


def scores_from_magnitude(magnitude: torch.Tensor) -> torch.Tensor:
    """The dormant score from each unit's mean absolute activation, (units,)."""
    scale = magnitude.mean().clamp_min(1e-12)  # 0-dim
    scores = magnitude / scale  # (units,)
    return scores


def dormant_share(scores: torch.Tensor, tau: float = DORMANT_TAU) -> torch.Tensor:
    """The share of units whose score is at or below tau, 0-dim."""
    share = (scores <= tau).float().mean()  # 0-dim
    return share


def effective_rank(features: torch.Tensor, delta: float = SRANK_DELTA) -> torch.Tensor:
    """srank_delta of (rows, width) features, as a 0-dim tensor so nothing synchronises.

    original form
        srank_delta(Phi) = min { k : sum_{i<=k} sigma_i / sum_i sigma_i >= 1 - delta }

    symbol table
        sigma_i   singular values of Phi in decreasing order
        delta     SRANK_DELTA
    """
    singular_values = singular_values_by_gram(features)  # (min(rows, width),)
    total = singular_values.sum().clamp_min(1e-12)  # 0-dim
    reached = torch.cumsum(singular_values, dim=0) / total >= (1.0 - delta)  # (k,)
    rank = reached.float().argmax() + 1  # 0-dim
    return rank


def singular_values_by_gram(features: torch.Tensor) -> torch.Tensor:
    """Singular values of (rows, width) features in decreasing order, (min(rows, width),).

    Read as the roots of the eigenvalues of the smaller Gram matrix, in float64 so
    that squaring the spectrum loses nothing srank can see. An SVD of the 512 by
    3072 hidden rows costs about 4 times as much on the CPU, and the heavy step
    takes 1 per block and group.
    """
    matrix = features.double()  # (rows, width)
    gram = (
        matrix @ matrix.T if matrix.shape[0] <= matrix.shape[1] else matrix.T @ matrix
    )
    eigenvalues = torch.linalg.eigvalsh(gram)  # (min(rows, width),) increasing
    singular_values = eigenvalues.clamp_min(0.0).sqrt().flip(0)  # decreasing
    return singular_values


def mlp_activations(core: nn.Module, architecture: str) -> list[nn.Module]:
    """The GELU of every block's MLP, in block order, empty for a model without one."""
    activations = []
    for block in transformer_blocks(core, architecture):
        gelus = [module for module in block.modules() if isinstance(module, nn.GELU)]
        if len(gelus) == 1:
            activations.append(gelus[0])
    return activations


def classifier_head(core: nn.Module) -> nn.Linear:
    """The last Linear of the network, the head on every backbone here."""
    linears = [module for module in core.modules() if isinstance(module, nn.Linear)]
    head = linears[-1]
    return head


def token_rows_per_image(num_images: int) -> int:
    rows = max(1, math.ceil(MLP_RANK_ROWS / max(num_images, 1)))
    return rows


@torch.no_grad()
def hidden_layer_readings(
    model: nn.Module,
    architecture: str,
    images: torch.Tensor,
    device: torch.device,
    use_bfloat16: bool,
) -> dict[str, torch.Tensor]:
    """Dormant shares and effective ranks of every MLP hidden layer and the features.

    images is a (N, C, H, W) batch served in chunks. Returns 0-dim device tensors
    under keys dormant_share_mlp_block{NN}, srank_mlp_block{NN},
    dormant_share_mlp_mean, dormant_share_mlp_max, dormant_share_features and
    srank_features. The caller restores the model's training mode.
    """
    core = network_core(model)
    gelus = mlp_activations(core, architecture)
    head = classifier_head(core)
    rows_per_image = token_rows_per_image(images.shape[0])

    magnitude_sums = [None] * len(gelus)
    rank_rows: list[list[torch.Tensor]] = [[] for _ in gelus]
    feature_chunks: list[torch.Tensor] = []
    row_counts = [0] * len(gelus)

    def gelu_hook(index):
        def hook(_module, _inputs, output):
            hidden = output.detach().float()
            hidden = hidden.reshape(hidden.shape[0], -1, hidden.shape[-1])
            # (chunk, tokens, units), Swin's (chunk, H, W, units) flattened
            flat = hidden.reshape(-1, hidden.shape[-1])  # (chunk * tokens, units)
            summed = flat.abs().sum(dim=0)  # (units,)
            held = magnitude_sums[index]
            magnitude_sums[index] = summed if held is None else held + summed
            row_counts[index] += flat.shape[0]

            picked_tokens = torch.linspace(
                0, hidden.shape[1] - 1, rows_per_image, device=hidden.device
            ).long()  # (rows_per_image,)
            picked = hidden[:, picked_tokens, :].reshape(-1, hidden.shape[-1])
            rank_rows[index].append(picked)  # (chunk * rows_per_image, units)

        return hook

    def head_pre_hook(_module, inputs):
        feature_chunks.append(inputs[0].detach().float())  # (chunk, dim)

    handles = [gelu.register_forward_hook(gelu_hook(i)) for i, gelu in enumerate(gelus)]
    handles.append(head.register_forward_pre_hook(head_pre_hook))
    model.eval()
    try:
        for chunk in images.split(PROBE_BATCH):
            forward_logits(model, chunk, device, use_bfloat16)
    finally:
        for handle in handles:
            handle.remove()

    readings = {}
    shares = []
    for index in range(len(gelus)):
        magnitude = magnitude_sums[index] / row_counts[index]  # (units,)
        share = dormant_share(scores_from_magnitude(magnitude))  # 0-dim
        shares.append(share)
        block = f"block{index + 1:02d}"
        readings[f"dormant_share_mlp_{block}"] = share
        readings[f"srank_mlp_{block}"] = effective_rank(torch.cat(rank_rows[index]))
    if shares:
        stacked_shares = torch.stack(shares)  # (blocks,)
        readings["dormant_share_mlp_mean"] = stacked_shares.mean()
        readings["dormant_share_mlp_max"] = stacked_shares.max()

    features = torch.cat(feature_chunks)  # (N, dim)
    readings["dormant_share_features"] = dormant_share(dormant_scores(features))
    readings["srank_features"] = effective_rank(features)
    return readings
