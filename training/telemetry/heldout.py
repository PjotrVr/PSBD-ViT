"""The fixed held-out pairs, and the readings taken on them once per epoch.

The pairs are test images the attack is eligible to flip (the ASR set of
attacks.poisoning.AttackSuccessSet) and their untriggered twins, drawn once with a
fixed seed, so every epoch reads the same images and a change between epochs is
the model's. They are held out from training, never enter the PSBD threshold or
any published number, and the same images feed the heavy step's dormancy readings.

The epoch readings are ASR and clean accuracy on the pairs, the margin retention
of clean and triggered inputs under PSBD-TM (token_mask before the attention norm,
defenses.decision.RECOMMENDED_PLACEMENT) at fixed rates, and the surplus factor
(training.telemetry.surplus). A fixed rate rather than the adaptive rule, because
the adaptive rate moves with the model and a retention curve read at a moving
rate mixes 2 changes.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2

from attacks.poisoning import Attack, AttackSuccessSet
from data.loading import extract_labels, limit_dataset
from defenses.decision import RECOMMENDED_PLACEMENT
from defenses.inference import forward_logits
from defenses.operators import build_operator
from evaluation.loaders import load_test_base
from models.positions import POSITION_REGISTRY, plug_dropout, unplug_dropout
from .surplus import last_block_surplus
from .window import rate_tag, rounded

RETENTION_POSITION = "before_attention_norm"
RETENTION_OPERATOR = "token_mask"
assert RECOMMENDED_PLACEMENT == f"{RETENTION_POSITION}_{RETENTION_OPERATOR}", (
    "the retention probe reads PSBD-TM, the recommended placement"
)

# cli.sweep's PSBD_MASK_SEED. The masks are redrawn from this seed at every epoch,
# so 2 epochs are read under identical masks.
RETENTION_MASK_SEED = 0

PROBE_BATCH = 128


@dataclass(frozen=True)
class HeldoutPairs:
    """Clean held-out images, their triggered twins and both labels.

    clean and triggered are (N, C, H, W) normalized, labels the true classes and
    success_labels the attack-success labels, both (N,) int64.
    """

    clean: torch.Tensor
    triggered: torch.Tensor
    labels: torch.Tensor
    success_labels: torch.Tensor

    def to(self, device: torch.device) -> "HeldoutPairs":
        moved = HeldoutPairs(
            self.clean.to(device),
            self.triggered.to(device),
            self.labels.to(device),
            self.success_labels.to(device),
        )
        return moved


def build_heldout_pairs(
    dataset_name: str,
    attack: Attack,
    raw_data_dir: str,
    count: int,
    seed: int,
    max_samples: int | None = None,
) -> HeldoutPairs:
    """count eligible test images and their triggered twins, drawn with seed.

    A multi-target clean-label attack succeeds anywhere in its target set, and the
    pairs read success against the single success label, so its ASR here is a
    lower bound. The panel's attacks all have a single target.
    """
    test_base, spec = load_test_base(dataset_name, raw_data_dir)
    test_base = limit_dataset(test_base, max_samples, seed)
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)
    true_labels = extract_labels(test_base)
    success_set = AttackSuccessSet(
        test_base, true_labels, attack, normalize, spec.num_classes
    )

    generator = torch.Generator().manual_seed(seed)
    positions = torch.randperm(len(success_set), generator=generator)[:count]

    clean_images, triggered_images, labels, success_labels = [], [], [], []
    for position in positions.tolist():
        triggered, success_label = success_set[position]  # (C, H, W), int
        index = success_set.indices[position]
        image, _ = test_base[index]  # (C, H, W) in 0 to 1
        clean_images.append(normalize(image))
        triggered_images.append(triggered)
        labels.append(int(true_labels[index]))
        success_labels.append(int(success_label))

    pairs = HeldoutPairs(
        clean=torch.stack(clean_images),  # (N, C, H, W)
        triggered=torch.stack(triggered_images),  # (N, C, H, W)
        labels=torch.tensor(labels, dtype=torch.int64),  # (N,)
        success_labels=torch.tensor(success_labels, dtype=torch.int64),  # (N,)
    )
    return pairs


@torch.no_grad()
def batched_logits(
    model: nn.Module, images: torch.Tensor, device: torch.device, use_bfloat16: bool
) -> torch.Tensor:
    """Logits over images in chunks, (N, classes) float32 on the device."""
    chunks = [
        forward_logits(model, chunk, device, use_bfloat16)
        for chunk in images.split(PROBE_BATCH)
    ]
    logits = torch.cat(chunks)  # (N, classes)
    return logits


def class_margin(logits: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    """logits[c] minus the largest other logit, per row, (N,)."""
    chosen = logits.gather(1, classes[:, None]).squeeze(1)  # (N,)
    others = logits.scatter(1, classes[:, None], float("-inf"))  # (N, classes)
    margin = chosen - others.max(dim=1).values  # (N,)
    return margin


@torch.no_grad()
def masked_logits(
    model: nn.Module,
    architecture: str,
    images: torch.Tensor,
    rate: float,
    device: torch.device,
    use_bfloat16: bool,
) -> torch.Tensor:
    """Logits with PSBD-TM plugged at rate, 1 pass, masks drawn from the fixed seed."""
    handles = plug_dropout(
        model,
        architecture,
        (RETENTION_POSITION,),
        {RETENTION_POSITION: build_operator(RETENTION_OPERATOR)},
        rate,
    )
    try:
        torch.manual_seed(RETENTION_MASK_SEED)
        logits = batched_logits(model, images, device, use_bfloat16)  # (N, classes)
    finally:
        unplug_dropout(handles)
    return logits


def retention_keys(
    base_logits: torch.Tensor, perturbed_logits: torch.Tensor, group: str, rate: float
) -> dict:
    """Median margin retention and prediction-kept share of 1 group at 1 rate.

        original form
            retention_i = m_c(x_i; p) / m_c(x_i),   c = argmax_k z_k(x_i)
            m_c(z) = z_c - max_{k != c} z_k

        symbol table
            z(x_i), z(x_i; p)   logits without and with the token mask at rate p
            c                   the unmasked prediction

    1 means the mask left the margin whole, 0 that it reached the boundary and a
    negative value that the prediction flipped. A low PSU (poisoned) is a triggered
    input whose retention stays near 1.
    """
    predictions = base_logits.argmax(dim=1)  # (N,)
    base_margin = class_margin(base_logits, predictions)  # (N,)
    perturbed_margin = class_margin(perturbed_logits, predictions)  # (N,)
    retention = perturbed_margin / base_margin.clamp_min(1e-6)  # (N,)
    kept = perturbed_logits.argmax(dim=1) == predictions  # (N,)

    tag = rate_tag(rate)
    keys = {
        f"retention_tm_{tag}_{group}": rounded(retention.median()),
        f"kept_tm_{tag}_{group}": rounded(kept.float().mean()),
    }
    return keys


def epoch_readings(
    model: nn.Module,
    architecture: str,
    pairs: HeldoutPairs,
    retention_rates: tuple[float, ...],
    measure_surplus: bool,
    device: torch.device,
    use_bfloat16: bool,
) -> dict:
    """ASR, clean accuracy, margin retention and surplus on the pairs.

    The model is left in eval mode. The caller restores its training mode and the
    RNG state the token masks consumed.
    """
    model.eval()
    clean_logits = batched_logits(model, pairs.clean, device, use_bfloat16)  # (N, K)
    triggered_logits = batched_logits(model, pairs.triggered, device, use_bfloat16)

    success = triggered_logits.argmax(dim=1) == pairs.success_labels  # (N,)
    correct = clean_logits.argmax(dim=1) == pairs.labels  # (N,)
    readings = {
        "heldout_asr": rounded(success.float().mean()),
        "heldout_clean_accuracy": rounded(correct.float().mean()),
        "heldout_margin_clean": rounded(
            class_margin(clean_logits, pairs.labels).mean()
        ),
        "heldout_margin_triggered": rounded(
            class_margin(triggered_logits, pairs.success_labels).mean()
        ),
    }

    if RETENTION_POSITION in POSITION_REGISTRY[architecture]:
        for rate in retention_rates:
            for group, images, base in (
                ("clean", pairs.clean, clean_logits),
                ("triggered", pairs.triggered, triggered_logits),
            ):
                perturbed = masked_logits(
                    model, architecture, images, rate, device, use_bfloat16
                )  # (N, K)
                readings.update(retention_keys(base, perturbed, group, rate))

    # The surplus factor is defined against 1 target class, which an all-to-all or
    # multi-target attack does not have.
    single_target = bool((pairs.success_labels == pairs.success_labels[0]).all())
    if measure_surplus and single_target:
        readings.update(
            last_block_surplus(
                model,
                architecture,
                pairs.clean,
                pairs.triggered,
                int(pairs.success_labels[0]),
                device,
                use_bfloat16,
            )
        )
    return readings
