"""Poison a dataset and train a backdoored ViT-B/16 or Swin-S.

The BackdoorBench-style flow: load clean data, poison a fraction of the training
set, train, evaluate ASR and clean accuracy, then save in the attack_result.pt
format the sweep reads.

Attacks that use cover samples (adaptive_blend, tact) are detected from their
config and routed through the cover-sample dataset. Everything else uses the plain
poisoning path. The training loop itself is the same in both cases.

Example
    python -m cli.train_backdoor --dataset cifar10 --attack badnet_a2a \
        --poison-rate 0.1 --architecture vit --epochs 15 \
        --output checkpoints/vit_cifar10_badnet_a2a_0_1/attack_result.pt
"""

import argparse
import json
import os
import time
from dataclasses import asdict, replace

import torch
import torchvision.transforms.v2 as transforms_v2
from lightning import seed_everything
from torch.utils.data import DataLoader, Dataset, Subset

from attacks import (
    ATTACK_NAMES,
    apply_config_overrides,
    build_attack,
    config_overrides,
    default_config,
    adversarial_config_error,
    missing_adversarial_bases,
)
from attacks.generated import GeneratedConfig
from data.registry import DATASET_REGISTRY
from data.loading import (
    AUGMENT_CHOICES,
    AugmentedTrainingSet,
    base_image_transform,
    build_augmentation_transform,
    exclude_indices,
    extract_labels,
    limit_dataset,
    load_clean_datasets,
)
from evaluation.loaders import build_clean_loader
from evaluation.metrics import clean_accuracy, evaluate_attack
from attacks.evasion import (
    DEFAULT_FLOOR_MARGIN,
    EVASION_OBJECTIVES,
    FlaggedPoisonedSet,
    calibrate_probe_rate,
    normalize_evasion_objective,
)
from attacks.poisoning import (
    Attack,
    CoverPoisonedTrainingSet,
    PoisonedTrainingSet,
    choose_indices_with_cover,
    choose_label_kept_indices,
    choose_poison_indices,
)
from training.loop import (
    LEARNING_RATE_SCHEDULES,
    CheckpointMetadata,
    IndexedTrainingSet,
    build_model,
    save_checkpoint,
    train_classifier,
)
from training.telemetry.heldout import HeldoutPairs, build_heldout_pairs
from training.telemetry.record import TelemetryConfig
from utils.provenance import utc_timestamp

# Interpolation is measured on a fixed subsample. The trajectory, not the exact
# value, is what the snapshot sweep reads.
TRAIN_EVAL_SAMPLES = 10000


def parse_attack_overrides(overrides: list[str] | None) -> dict:
    """Turn `key=value` command-line strings into a mapping.

    Casting is left to attacks.apply_config_overrides so training and evaluation agree
    on the type of every field.
    """
    if not overrides:
        return {}
    parsed = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--attack-override needs key=value, got {item!r}")
        key, raw = item.split("=", 1)
        parsed[key] = raw
    return parsed


def resolve_config(attack_name: str, poisoned_dir: str):
    """The attack's config, with the generated adapter pointed at its trigger folder."""
    if attack_name == "generated":
        return GeneratedConfig(poisoned_dir=poisoned_dir)

    config = default_config(attack_name)
    return config


# Cover-sample rates the papers specify, as a multiple of the poisoning rate.
# WaNet's noise mode and Adaptive-Blend's cover both scale with the poison rate, so a
# fixed constant is 10x too small at 10% poisoning and the attack loses the stealth
# the mechanism exists to provide. TaCT's reference selects cover by CLASS rather than
# by rate, so its config constant stands and it is deliberately absent here.
COVER_RATE_MULTIPLES = {
    "wanet": 2.0,  # BackdoorBench cross_ratio 2, and PSBD's twice the poisoning ratio
    "adaptive_blend": 1.0,  # Qi et al. and PSBD: cover ratio equal to the poisoning ratio
    "bpp": 1.0,  # BackdoorBench neg_ratio 0.1 against pratio 0.1
    # docs/evidence-surplus-theory.md: 10% cover at 10% poisoning for the AND
    # attacks (R4, R6) and the veto (R7). Without covers an AND trigger is just a
    # bigger OR trigger, since every subset would predict the target.
    "and16": 1.0,
    "and2": 1.0,
    "veto": 1.0,
}


def resolve_cover_rate(attack_name: str, poison_rate: float, override):
    """The cover rate to use: an explicit override, else the attack's paper value.

    Returns None when the attack defines no cover mechanism, leaving the config's own
    value in place.
    """
    if override is not None:
        return override
    multiple = COVER_RATE_MULTIPLES.get(attack_name)
    return None if multiple is None else multiple * poison_rate


def build_training_set(
    train_clean: Dataset,
    attack: Attack,
    config,
    poison_rate: float,
    seed: int,
    normalize,
    num_classes: int,
    trigger_label_probability: float = 1.0,
) -> tuple[Dataset, float]:
    """The poisoned training set and the poison rate it actually realized.

    Routes to the cover-sample dataset when the attack config asks for it. A
    trigger_label_probability below 1 keeps the true label on a seeded share of the
    poisoned rows (attacks.poisoning.choose_label_kept_indices). They still carry
    the trigger and still count toward the realized poison rate.
    """
    labels = extract_labels(train_clean)
    cover_rate = getattr(config, "cover_rate", 0.0)
    source_classes = getattr(config, "source_classes", None)

    if cover_rate > 0.0 or source_classes is not None:
        poison_indices, cover_indices = choose_indices_with_cover(
            labels, attack, poison_rate, cover_rate, source_classes, seed
        )
        label_kept_indices = choose_label_kept_indices(
            poison_indices, trigger_label_probability, len(labels), seed
        )
        dataset = CoverPoisonedTrainingSet(
            train_clean,
            attack,
            poison_indices,
            cover_indices,
            normalize,
            num_classes,
            label_kept_indices,
        )
    else:
        poison_indices = choose_poison_indices(labels, attack, poison_rate, seed)
        label_kept_indices = choose_label_kept_indices(
            poison_indices, trigger_label_probability, len(labels), seed
        )
        dataset = PoisonedTrainingSet(
            train_clean,
            attack,
            poison_indices,
            normalize,
            num_classes,
            label_kept_indices,
        )
    if label_kept_indices:
        print(
            f"trigger label probability {trigger_label_probability}: "
            f"{len(label_kept_indices)} of {len(poison_indices)} poisoned rows keep "
            "their true label"
        )

    # The requested rate is capped at the eligible pool, so it is not always what
    # was applied. Measuring it here is the only place both numbers exist at once.
    realized_poison_rate = len(poison_indices) / max(len(labels), 1)
    if abs(realized_poison_rate - poison_rate) > 1e-9:
        print(
            f"poison rate requested {poison_rate:.4f}, realized "
            f"{realized_poison_rate:.4f} ({len(poison_indices)} of {len(labels)} "
            "samples), capped by the eligible pool"
        )

    # A poisoned sample with no perturbed base would silently train the patch-only
    # variant while args.json records the adversarial variant, so refuse before training.
    incoherent = adversarial_config_error(config)
    if incoherent:
        raise ValueError(incoherent)

    absent = missing_adversarial_bases(config, poison_indices)
    if absent:
        raise ValueError(
            f"{len(absent)} of {len(poison_indices)} poisoned indices have no adversarial "
            f"base in {config.adversarial_dir} (first missing: {absent[:5]}). "
            "Generate them with `python -m cli.lc_bases`."
        )

    return dataset, realized_poison_rate


def build_training_loader(
    args: argparse.Namespace, image_size: int
) -> tuple[DataLoader, int, Attack, object, float]:
    """The poisoned training loader, plus the attack record it was built from.

    Evaluation is handled separately by evaluation.metrics and evaluation.loaders, so
    this has no eval-loader concerns at all.
    """
    spec = DATASET_REGISTRY[args.dataset]
    transform = base_image_transform(image_size)
    train_clean, _ = load_clean_datasets(args.dataset, transform, args.raw_data_dir)
    # Subset before poison-index selection so poison_rate is measured against the
    # truncated pool, mirroring the eval-side subset in evaluation.loaders.
    train_clean = limit_dataset(train_clean, args.max_samples, args.seed)
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)

    config = resolve_config(args.attack, args.poisoned_dir)
    cover_rate = resolve_cover_rate(
        args.attack, args.poison_rate, getattr(args, "cover_rate", None)
    )
    if cover_rate is not None:
        config = replace(config, cover_rate=cover_rate)
        print(f"cover rate for {args.attack}: {cover_rate:.4f}")
    config = apply_config_overrides(
        config, parse_attack_overrides(args.attack_override)
    )
    attack = build_attack(args.attack, config, image_size, args.target_label)

    poisoned_train, realized_poison_rate = build_training_set(
        train_clean,
        attack,
        config,
        args.poison_rate,
        args.seed,
        normalize,
        spec.num_classes,
        getattr(args, "trigger_label_probability", 1.0),
    )

    if getattr(args, "exclude_indices_file", None):
        # After poisoning, so the indices in the file address the same poisoned
        # dataset a sanitising pass (experiments/training_aware/flag_by_loss.py)
        # scored. Before augmentation, evasion flagging and index wrapping too, so
        # none of those has to know a sample went missing.
        with open(args.exclude_indices_file) as handle:
            excluded = set(json.load(handle))
        print(f"excluding {len(excluded)} indices from {args.exclude_indices_file}")
        poisoned_train = exclude_indices(poisoned_train, excluded)

    if args.augment == "standard":
        # After the trigger is stamped and the sample normalized, so the crop and
        # flip see exactly the image an attacker's data would present at deploy
        # time. Never applied to val_loader or any PSBD split, both built
        # elsewhere from the plain clean or poisoned datasets. Ahead of the
        # evasion flag below, whose wrapper adds a 3rd tuple element that
        # AugmentedTrainingSet's 2-tuple contract does not expect.
        augmentation = build_augmentation_transform(image_size)
        poisoned_train = AugmentedTrainingSet(poisoned_train, augmentation)

    if getattr(args, "evade_psbd", False):
        # The attacker knows which samples it poisoned. This flag never reaches
        # the defender's side of any evaluation.
        poisoned_train = FlaggedPoisonedSet(poisoned_train)

    wants_row_indices = getattr(args, "record_sample_loss", False) or getattr(
        args, "telemetry", False
    )
    if wants_row_indices:
        if getattr(args, "evade_psbd", False):
            # Both wrappers add a 3rd tuple element, and each expects the
            # other's 2-tuple contract from its inner dataset.
            raise ValueError(
                "--record-sample-loss and --telemetry cannot be combined with "
                "--evade-psbd"
            )
        poisoned_train = IndexedTrainingSet(poisoned_train)

    train_loader = DataLoader(
        poisoned_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    return train_loader, spec.num_classes, attack, config, realized_poison_rate


# The default evasion probe when neither --evade-probes nor the singular
# --evade-position / --evade-operator flags are given, keyed by --architecture.
# before_attention_norm:dropout is the ViT/Swin placement every existing
# adaptive-attacker experiment trains against, and post_residual:dropout is
# resnet18's only position (models.positions.RESNET_POSITIONS), the PSBD
# paper's own ConvNet placement. This is what lets the 3 command variants
# (plain, the paper's attacker, this project's attacker) differ by exactly the
# --evade-psbd/--evade-objective/--evade-weight flags and nothing else.
DEFAULT_EVADE_PROBE_BY_ARCHITECTURE: dict[str, tuple[str, str]] = {
    "resnet18": ("post_residual", "dropout"),
}
DEFAULT_EVADE_PROBE: tuple[str, str] = ("before_attention_norm", "dropout")


def parse_block_range(text: str, token: str) -> tuple[int, int]:
    """`first-last` as a 1-indexed inclusive (first, last) block span."""
    parts = text.split("-")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(
            f"--evade-probes token {token!r} needs its block range as `first-last`, "
            f"for example 5-8, got {text!r}"
        )
    first, last = int(parts[0]), int(parts[1])
    if not 1 <= first <= last:
        raise ValueError(
            f"--evade-probes token {token!r} needs 1 <= first <= last, got {text!r}"
        )
    block_range = (first, last)
    return block_range


def parse_evade_probe_tokens(
    args: argparse.Namespace,
) -> list[tuple[str, str, tuple[int, int] | None]]:
    """The (position, operator, block_range) probes to train against.

    --evade-probes, when given, takes 1 or more `position:operator[:first-last]`
    tokens (for example `before_attention_norm:token_mask` or
    `pre_residual:dropout:5-8`) and overrides --evade-position and
    --evade-operator entirely. The optional block range restricts a block-scope
    position to blocks first to last, 1-indexed and inclusive, exactly as
    cli.sweep --block-range does, and is None when absent. With no --evade-probes
    and neither singular flag set, the default resolves from --architecture
    (DEFAULT_EVADE_PROBE_BY_ARCHITECTURE), so a bare --evade-psbd needs no
    position or operator at all. Setting only 1 of the singular flags is
    refused rather than silently pairing it with the other's default.
    """
    if not args.evade_probes:
        position, operator = args.evade_position, args.evade_operator
        if position is None and operator is None:
            position, operator = DEFAULT_EVADE_PROBE_BY_ARCHITECTURE.get(
                args.architecture, DEFAULT_EVADE_PROBE
            )
        elif position is None or operator is None:
            raise ValueError(
                "--evade-position and --evade-operator must be given together, "
                "or neither, so the architecture default is not silently mixed "
                "with 1 explicit flag"
            )
        return [(position, operator, None)]

    tokens = []
    for token in args.evade_probes:
        parts = token.split(":")
        if len(parts) not in (2, 3) or not all(parts):
            raise ValueError(
                f"--evade-probes token {token!r} must be "
                "`position:operator` or `position:operator:first-last`"
            )
        block_range = parse_block_range(parts[2], token) if len(parts) == 3 else None
        tokens.append((parts[0], parts[1], block_range))
    return tokens


def resolve_evasion(
    args: argparse.Namespace,
    num_classes: int,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[dict | None, list[dict]]:
    """The adaptive attacker's probe configs, and the rate each will run at.

    Returns (None, []) when --evade-psbd is off. Otherwise returns the evasion
    dict train_one_epoch_evasive reads and the resolved probe list (1 entry per
    --evade-probes token, or 1 entry for the singular --evade-position and
    --evade-operator flags when --evade-probes is absent), each carrying its
    own calibrated "rate".

    A requested rate of 0.0 (the default) means "calibrate", so a throwaway
    model is built to find the rate whose clean-validation shift ratio matches
    the defender's own sigma target, once per probe since each probe's
    (position, operator) has its own shift-versus-rate curve. The RNG is
    restored afterwards so training starts from the same state regardless of
    how many probes were calibrated.

    Multi-probe evasion is defined only for the psu_gap_hinge objective (see
    attacks.evasion's module docstring), so more than 1 --evade-probes token
    together with --evade-objective psu_mean is rejected here rather than
    left for evasive_update to raise mid-training.
    """
    if not args.evade_psbd:
        return None, []

    probe_tokens = parse_evade_probe_tokens(args)
    objective = normalize_evasion_objective(args.evade_objective)
    if len(probe_tokens) > 1 and objective != "psu_gap_hinge":
        raise ValueError(
            "--evade-probes with more than 1 probe only supports "
            "--evade-objective psu_gap_hinge"
        )

    calibration_model = None
    probes = []
    for position, operator, block_range in probe_tokens:
        probe = {
            "position": position,
            "operator": operator,
            # Read by attacks.evasion wherever the probe is plugged, so the
            # calibration, every recalibration and the penalty all perturb the
            # same blocks.
            "block_range": block_range,
            "architecture": args.architecture,
            # Read by attacks.evasion.evasive_update to pick the loss. Living on
            # the probe dict, not a separate argument, is what lets it reach
            # train_one_epoch_evasive without training.loop's call signature
            # changing: that call already forwards this dict opaquely. Stored
            # already normalized, so args.json and every downstream reader see
            # the canonical name regardless of which spelling was passed.
            "objective": objective,
            # Read by evasive_update's psu_floor branch only, and harmless on
            # the other objectives since neither reads probe["margin"].
            "margin": args.evade_margin,
        }
        rate = args.evade_rate
        if rate == 0.0:
            if calibration_model is None:
                calibration_model = build_model(args.architecture, num_classes).to(
                    device
                )
            rate = calibrate_probe_rate(
                calibration_model,
                val_loader,
                probe,
                device,
                target_sigma=args.evade_calibration_target,
            )
        probes.append({**probe, "rate": rate})

    if calibration_model is not None:
        del calibration_model
        torch.cuda.empty_cache()
        seed_everything(args.seed, workers=True)

    evasion = {
        "probe": probes if len(probes) > 1 else probes[0],
        "weight": args.evade_weight,
        "passes": args.evade_passes,
        # Read by training.loop.train_classifier at each epoch boundary.
        # recalibrate_every 0 is the historical behavior: the rate calibrated
        # above on the initial weights is never touched again.
        "recalibrate_every": args.evade_recalibrate_every,
        "calibration_target": args.evade_calibration_target,
    }
    return evasion, probes


def build_parser() -> argparse.ArgumentParser:
    """The parser, separate from parsing so a generator can read its defaults."""
    parser = argparse.ArgumentParser(description="Train a backdoored ViT or Swin")
    parser.add_argument("--dataset", choices=tuple(DATASET_REGISTRY), required=True)
    parser.add_argument("--attack", choices=ATTACK_NAMES, required=True)
    parser.add_argument("--poison-rate", type=float, required=True)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument(
        "--architecture", choices=("vit", "swin", "resnet18"), default="vit"
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--evade-psbd",
        action="store_true",
        help=(
            "adaptive attacker: add an evasion penalty (--evade-objective picks "
            "which one) that removes the statistic PSBD reads while keeping the "
            "backdoor"
        ),
    )
    parser.add_argument("--evade-weight", type=float, default=1.0)
    parser.add_argument(
        "--evade-position",
        default=None,
        help="default: DEFAULT_EVADE_PROBE_BY_ARCHITECTURE's entry for "
        "--architecture (before_attention_norm for vit/swin, post_residual for "
        "resnet18). Must be given together with --evade-operator, or not at all.",
    )
    parser.add_argument(
        "--evade-operator",
        default=None,
        help="default: DEFAULT_EVADE_PROBE_BY_ARCHITECTURE's entry for "
        "--architecture (dropout for every architecture today). Must be given "
        "together with --evade-position, or not at all.",
    )
    parser.add_argument(
        "--evade-probes",
        nargs="+",
        default=None,
        metavar="POSITION:OPERATOR[:FIRST-LAST]",
        help=(
            "1 or more `position:operator[:first-last]` tokens to train against "
            "jointly, for example before_attention_norm:token_mask "
            "pre_residual:dropout:5-8 mlp_norm_out:gain_scale. The optional "
            "first-last restricts a block-scope position to those blocks, 1-indexed "
            "and inclusive, as cli.sweep --block-range does. Each is calibrated to "
            "its own rate and the "
            "hinge loss is the mean over probes (attacks.evasion's module "
            "docstring). Overrides --evade-position and --evade-operator "
            "entirely; only supported with --evade-objective psu_gap_hinge when "
            "more than 1 token is given"
        ),
    )
    parser.add_argument(
        "--evade-objective",
        choices=EVASION_OBJECTIVES,
        default="psu_gap_hinge",
        help=(
            "psu_gap_hinge (default, formerly 'hinge'): this repository's own "
            "penalty, additive on top of cross-entropy, active only while "
            "poisoned shift trails clean. psu_mean (formerly 'psbd_paper'): the "
            "PSBD paper's own adaptive-attacker loss (Appendix, 'Resistance to "
            "Potential Adaptive Attacks'), a convex combination "
            "(1 - alpha) * cross_entropy + alpha * L_ada with --evade-weight read "
            "as alpha, pushing every sample's PSU down, benign and poisoned alike. "
            "psu_floor: this project's own label-free mirror of psu_gap_hinge, "
            "fractional PSU, additive penalty pushing every sample's shift up to "
            "--evade-margin with no reference to is_poisoned at all. "
            "'hinge' and 'psbd_paper' are still accepted as deprecated aliases"
        ),
    )
    parser.add_argument(
        "--evade-margin",
        type=float,
        default=DEFAULT_FLOOR_MARGIN,
        help="the psu_floor objective's hinge target: every sample's fractional "
        "PSU is pushed to stay above this margin. Default matches "
        "ADAPTIVE_SHIFT_TARGET, the shift level clean samples already sit near "
        "at the rate the deployable adaptive rule selects. Ignored by every "
        "other --evade-objective.",
    )
    parser.add_argument(
        "--evade-rate",
        type=float,
        default=0.0,
        help="perturbation rate for the evasion probe. 0 (default) auto-calibrates "
        "to the sigma=0.6 matched rate on the validation set before training.",
    )
    parser.add_argument(
        "--evade-recalibrate-every",
        type=int,
        default=0,
        help="recalibrate the probe rate on the current model at the start of "
        "every N-th epoch, after a warm-up of N epochs at the initial "
        "calibration. 0 (default) keeps the current behaviour: calibrated once "
        "on the model's initial weights and never touched again, which is a "
        "poor target on a from-scratch model since the initial weights' shift "
        "curve differs sharply from the trained model's.",
    )
    parser.add_argument(
        "--evade-calibration-target",
        type=float,
        default=0.6,
        help="the target_sigma calibrate_probe_rate matches, both for the "
        "initial calibration and every recalibration. Default 0.6 matches the "
        "defender's cross-placement matched rule "
        "(defenses.decision.select_rate_at_matched_shift); pass 0.8 to match "
        "the deployable adaptive rule instead "
        "(defenses.decision.select_rate_adaptively, ADAPTIVE_SHIFT_TARGET).",
    )
    parser.add_argument("--evade-passes", type=int, default=3)
    parser.add_argument(
        "--record-sample-loss",
        action="store_true",
        help=(
            "wrap the training set in IndexedTrainingSet and write "
            "<output folder>/sample_loss.npz after every epoch: the per-epoch, "
            "per-sample cross entropy (reduction='none', scattered by index) "
            "plus the poison index set. Off by default, and not supported "
            "together with --evade-psbd (experiments/early_loss_signal)."
        ),
    )
    parser.add_argument(
        "--telemetry",
        action="store_true",
        help=(
            "write <output folder>/telemetry.jsonl (training.telemetry): the clean, "
            "poisoned and cover loss, accuracy and margin every --telemetry-every "
            "steps, module norms, update ratios, dormant shares and effective ranks "
            "every --telemetry-heavy-every steps, and per epoch the held-out ASR, "
            "clean accuracy, PSBD-TM margin retention and surplus factor. Off by "
            "default, trains the same model, not supported with --evade-psbd."
        ),
    )
    parser.add_argument(
        "--telemetry-every",
        type=int,
        default=TelemetryConfig.every,
        help="light window in optimizer steps",
    )
    parser.add_argument(
        "--telemetry-heavy-every",
        type=int,
        default=TelemetryConfig.heavy_every,
        help="heavy cadence in optimizer steps, a multiple of --telemetry-every, "
        "0 switches the heavy probes off",
    )
    parser.add_argument(
        "--telemetry-pairs",
        type=int,
        default=256,
        help="fixed held-out clean and triggered test pairs the heavy and epoch "
        "probes read, 0 switches those probes off",
    )
    parser.add_argument(
        "--telemetry-retention-rates",
        type=float,
        nargs="*",
        default=list(TelemetryConfig.retention_rates),
        help="PSBD-TM rates the epoch record reads margin retention at, none to skip",
    )
    parser.add_argument(
        "--telemetry-no-surplus",
        action="store_true",
        help="skip the per-epoch surplus factor",
    )
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="snapshot every epoch as its own <output folder>_epNN checkpoint "
        "folder, the same as --checkpoint-freq 1",
    )
    parser.add_argument(
        "--exclude-indices-file",
        default=None,
        help=(
            "path to a JSON list of poisoned-training-set indices "
            "(IndexedTrainingSet's numbering) to drop before training, for a "
            "sanitised retrain. Written by "
            "experiments/training_aware/flag_by_loss.py. Applied after "
            "poisoning and before augmentation, evasion or index wrapping."
        ),
    )
    parser.add_argument(
        "--model-dropout-train",
        type=float,
        default=0.0,
        help=(
            "train WITH dropout at this rate. PSBD requires a dropout-free model. "
            "This exists to test that requirement, not to change the default"
        ),
    )
    parser.add_argument("--use-sam", action="store_true")
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument(
        "--lr-schedule",
        choices=LEARNING_RATE_SCHEDULES,
        default="constant",
        help="constant is every panel run, cosine anneals to 0 by the last epoch",
    )
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=None,
        help="gradient norm bound, off by default as on every panel run",
    )
    parser.add_argument(
        "--poisoned-dir", default="", help="required only for the generated attack"
    )
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--attack-override",
        # extend rather than the default store, so a repeated flag accumulates.
        # With plain nargs a repeated flag replaces the earlier flag, and
        # --attack-override a=1 --attack-override b=2 keeps only b.
        action="extend",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override attack-config fields, e.g. --attack-override patch_size=32 or "
        "strength=2.0. Needed for trigger dose-response sweeps, which otherwise cannot "
        "vary the trigger at all because resolve_config returns default_config().",
    )
    parser.add_argument(
        "--cover-rate",
        type=float,
        default=None,
        help="Cover-sample rate. Default None resolves the value the attack's paper "
        "specifies as a multiple of the poisoning rate: 2x for wanet's noise mode, "
        "1x for adaptive_blend and bpp. TaCT keeps its config constant because its "
        "reference selects cover by class, not by rate.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=0,
        help="Snapshot every Nth epoch into its own checkpoint folder. 0 (default) "
        "disables snapshotting entirely, so existing runs are unaffected.",
    )
    parser.add_argument(
        "--checkpoint-dense-until",
        type=int,
        default=0,
        help="Snapshot EVERY epoch up to and including this one, then fall back to "
        "--checkpoint-freq. Detection quality moves while the model is still "
        "fitting the training set and stops moving once it interpolates, so a flat "
        "interval spends most of its snapshots in the flat region.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Truncate each dataset to this many samples, reproducibly, for a fast "
        "smoke run (combine with --epochs 1). -1 (default) uses the whole dataset. "
        "This alone does not imply smoke semantics, and --epochs is independent.",
    )
    parser.add_argument(
        "--augment",
        choices=AUGMENT_CHOICES,
        default="none",
        help="none (default): every existing checkpoint's recipe, no augmentation "
        "beyond normalization. standard: random resized crop to the training "
        "resolution (scale 0.6 to 1.0) plus a random horizontal flip, on the "
        "training loader only, applied after the trigger is stamped.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="cross-entropy label smoothing epsilon on every training row, 0 "
        "(default) is the plain cross-entropy of every panel run. Caps the optimal "
        "logit gap, the margin cap of docs/evidence-surplus-theory.md (R5, R13).",
    )
    parser.add_argument(
        "--trigger-label-probability",
        type=float,
        default=1.0,
        help="the probability q that a poisoned row carries the attack's label, "
        "otherwise it keeps its true label. Drawn once per sample index from the "
        "run seed (attacks.poisoning.choose_label_kept_indices). 1 (default) is "
        "the ordinary attack. The predictivity control of "
        "docs/evidence-surplus-theory.md (R3, R11).",
    )
    return parser


def parse_args() -> argparse.Namespace:
    arguments = build_parser().parse_args()
    return arguments


def snapshot_epochs(total: int, dense_until: int, freq: int) -> set[int]:
    """Which epochs to snapshot: every one through dense_until, then every freq-th.

    2 phases because the interesting part of the trajectory is the approach to
    training-set interpolation, which is early. Once the model interpolates the
    detection metrics stop moving, so the tail only needs sampling.
    """
    if dense_until <= 0 and freq <= 0:
        return set()
    chosen = set(range(1, min(dense_until, total) + 1))
    if freq > 0:
        chosen |= {e for e in range(1, total + 1) if e % freq == 0 and e > dense_until}
    return chosen


def build_telemetry_config(args: argparse.Namespace) -> TelemetryConfig | None:
    """The telemetry the run asked for, None when --telemetry is off."""
    if not args.telemetry:
        return None
    config = TelemetryConfig(
        every=args.telemetry_every,
        heavy_every=args.telemetry_heavy_every,
        retention_rates=tuple(args.telemetry_retention_rates),
        measure_surplus=not args.telemetry_no_surplus,
    )
    return config


def build_telemetry_heldout(
    args: argparse.Namespace, attack: Attack
) -> HeldoutPairs | None:
    """The fixed held-out pairs, None when telemetry or its pairs are off."""
    if not args.telemetry or args.telemetry_pairs <= 0:
        return None
    pairs = build_heldout_pairs(
        args.dataset,
        attack,
        args.raw_data_dir,
        args.telemetry_pairs,
        args.seed,
        max_samples=args.max_samples,
    )
    return pairs


def build_train_eval_loader(train_loader: DataLoader, args) -> DataLoader:
    """The poisoned training set the model actually saw, unshuffled, for train accuracy.

    Measuring interpolation on the clean split is wrong for a clean-label attack. Its
    eligible pool is a single class, so at a high enough rate every target-class
    image is poisoned and the model never saw a clean one. Clean accuracy on that
    class then reads near 0 while the model fits what it was given almost
    perfectly, which looks like a failure to fit and is not.

    Capped at 10000 samples because this runs at every snapshot and the question is
    when the model interpolates, which a fixed subsample tracks as well as the full
    set.
    """
    dataset = train_loader.dataset
    # --record-sample-loss and --telemetry serve (image, label, index) triples,
    # and the accuracy pass reads (image, label) pairs.
    if isinstance(dataset, IndexedTrainingSet):
        dataset = dataset.inner
    if len(dataset) > TRAIN_EVAL_SAMPLES:
        generator = torch.Generator().manual_seed(args.seed)
        picked = torch.randperm(len(dataset), generator=generator)[:TRAIN_EVAL_SAMPLES]
        dataset = Subset(dataset, picked.tolist())
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def innermost_poisoned_set(dataset: Dataset) -> Dataset | None:
    """The PoisonedTrainingSet or CoverPoisonedTrainingSet under the loader's wrappers.

    The Augmented, Indexed and Flagged wrappers hide the index sets that the
    provenance counts read, so a count read off the outer dataset silently
    reports 0 on a telemetry or evasion run. Walks .inner, .base_dataset and a
    Subset's .dataset, and returns None when nothing carries poison_indices.
    """
    current = dataset
    while current is not None:
        if hasattr(current, "poison_indices"):
            return current
        current = (
            getattr(current, "inner", None)
            or getattr(current, "base_dataset", None)
            or getattr(current, "dataset", None)
        )
    return None


def trigger_label_record(args: argparse.Namespace, poisoned_set) -> dict:
    """The args.json fields of --trigger-label-probability, identical on every snapshot."""
    kept = getattr(poisoned_set, "label_kept_indices", None) or ()
    record = {
        "trigger_label_probability": args.trigger_label_probability,
        "n_trigger_label_kept": len(kept),
    }
    return record


def build_snapshot_hook(
    args,
    num_classes,
    attack,
    config,
    realized_poison_rate,
    started_at,
    device,
    train_loader,
):
    """Write each chosen epoch as a full checkpoint folder, or None if disabled.

    Each snapshot is a complete, self-describing checkpoint directory rather than
    a bare state dict, so cli.sweep, cli.analyze and the table generator read it
    with no changes. That is the reason for the naming: <base>_ep07 sits beside
    <base> and looks like any other run.

    ASR is deliberately left None. Evaluating it costs a full poisoned pass and the
    sweep backfills it into args.json from the PSBD baseline cache anyway, so paying
    for it at every snapshot would be waste.
    """
    chosen = snapshot_epochs(
        args.epochs, args.checkpoint_dense_until, args.checkpoint_freq
    )
    if not chosen:
        return None

    train_eval_loader = build_train_eval_loader(train_loader, args)
    base_dir = os.path.dirname(args.output)
    filename = os.path.basename(args.output)
    print(f"snapshotting {len(chosen)} epochs: {sorted(chosen)}")

    def hook(model, epoch: int, validation_accuracy: float) -> None:
        if epoch not in chosen:
            return
        was_training = model.training
        train_accuracy = clean_accuracy(model, train_eval_loader, device, True)
        metadata = CheckpointMetadata(
            dataset=args.dataset,
            attack=args.attack,
            label_mode=attack.label_mode,
            target_label=args.target_label,
            poison_rate=args.poison_rate,
            realized_poison_rate=realized_poison_rate,
            cover_rate=getattr(config, "cover_rate", 0.0),
            architecture=args.architecture,
            use_sam=args.use_sam,
            rho=args.rho,
            epochs=epoch,
            seed=args.seed,
            max_samples=args.max_samples,
            clean_accuracy=validation_accuracy,
            asr=None,
            started_at=started_at,
            ended_at=utc_timestamp(),
            model_dropout=args.model_dropout_train,
            learning_rate_schedule=args.lr_schedule,
            clip_grad_norm=args.clip_grad_norm,
            label_smoothing=args.label_smoothing,
        ).as_dict()
        metadata.update(
            trigger_label_record(args, innermost_poisoned_set(train_loader.dataset))
        )
        # A snapshot is rebuilt through the same args.json path as the final
        # checkpoint, so it needs the trigger it was trained with too.
        metadata["attack_config_overrides"] = config_overrides(config, args.attack)
        # The trajectory fields the interpolation question turns on. Kept out of
        # checkpoint_metadata so its key set stays identical across entrypoints.
        metadata["epoch"] = epoch
        metadata["train_accuracy"] = train_accuracy
        metadata["snapshot_of"] = os.path.basename(base_dir)
        metadata["augment"] = args.augment
        save_checkpoint(
            model,
            num_classes,
            os.path.join(f"{base_dir}_ep{epoch:02d}", filename),
            metadata=metadata,
        )
        print(
            f"  snapshot epoch {epoch}: train_acc={train_accuracy:.4f} "
            f"val_acc={validation_accuracy:.4f}"
        )
        model.train(was_training)

    return hook


def main() -> None:
    args = parse_args()
    # -1 is a CLI-only sentinel for "no limit". Normalize it to None immediately so
    # no subsetting code ever sees it, since -1 would slice off the last sample.
    args.max_samples = None if args.max_samples == -1 else args.max_samples
    if args.save_every_epoch:
        args.checkpoint_freq = 1
    telemetry_config = build_telemetry_config(args)
    seed_everything(args.seed, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    started_at = utc_timestamp()
    image_size = DATASET_REGISTRY[args.dataset].image_size

    train_loader, num_classes, attack, config, realized_poison_rate = (
        build_training_loader(args, image_size)
    )
    val_loader = build_clean_loader(
        args.dataset,
        args.raw_data_dir,
        args.batch_size,
        args.num_workers,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    # Built ahead of the reseed below, so whatever a trigger draws while the pairs
    # are stamped cannot move the stream training starts from.
    telemetry_heldout = build_telemetry_heldout(args, attack)

    # Reseed right before the regular workflow so model init and training start from
    # an identical RNG state whether or not --max-samples triggered any subsetting.
    seed_everything(args.seed, workers=True)

    evasion, evade_probes = resolve_evasion(args, num_classes, val_loader, device)

    model, trajectory = train_classifier(
        args.architecture,
        num_classes,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        use_sam=args.use_sam,
        rho=args.rho,
        model_dropout=args.model_dropout_train,
        evasion=evasion,
        learning_rate_schedule=args.lr_schedule,
        clip_grad_norm=args.clip_grad_norm,
        record_sample_loss=args.record_sample_loss,
        checkpoint_dir=os.path.dirname(args.output),
        telemetry_config=telemetry_config,
        telemetry_heldout=telemetry_heldout,
        label_smoothing=args.label_smoothing,
        on_epoch_end=build_snapshot_hook(
            args,
            num_classes,
            attack,
            config,
            realized_poison_rate,
            started_at,
            device,
            train_loader,
        ),
    )
    ended_at = utc_timestamp()

    metrics = evaluate_attack(
        model,
        args.dataset,
        args.attack,
        config,
        args.target_label,
        device,
        args.raw_data_dir,
        args.batch_size,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    print(f"final ASR={metrics['asr']:.4f} CA={metrics['clean_accuracy']:.4f}")

    # The cover count, not just the requested rate. A cover mechanism that silently
    # produced zero samples looks identical to a successful run in every other field.
    poisoned_set = innermost_poisoned_set(train_loader.dataset)
    n_cover = len(getattr(poisoned_set, "cover_indices", ()) or ())
    print(f"cover samples: {n_cover}")

    metadata = CheckpointMetadata(
        dataset=args.dataset,
        attack=args.attack,
        label_mode=attack.label_mode,
        target_label=args.target_label,
        poison_rate=args.poison_rate,
        realized_poison_rate=realized_poison_rate,
        cover_rate=getattr(config, "cover_rate", 0.0),
        architecture=args.architecture,
        use_sam=args.use_sam,
        rho=args.rho,
        epochs=args.epochs,
        seed=args.seed,
        max_samples=args.max_samples,
        clean_accuracy=metrics["clean_accuracy"],
        asr=metrics["asr"],
        started_at=started_at,
        ended_at=ended_at,
        learning_rate_schedule=args.lr_schedule,
        clip_grad_norm=args.clip_grad_norm,
        best_validation_accuracy=trajectory.best,
        final_validation_accuracy=trajectory.final,
        # "probes" is a list even for a single --evade-position/--evade-operator
        # run, so the sidecar's shape never depends on whether --evade-probes
        # was used: 1 entry means single-probe evasion, more than 1 means the
        # multi-probe hinge (attacks.evasion's module docstring).
        evasion={
            "weight": args.evade_weight,
            "objective": normalize_evasion_objective(args.evade_objective),
            "probes": [
                {
                    "position": p["position"],
                    "operator": p["operator"],
                    "block_range": list(p["block_range"]) if p["block_range"] else None,
                    "rate": p["rate"],
                }
                for p in evade_probes
            ],
            "margin": args.evade_margin,
            "rate_requested": args.evade_rate,
            "passes": args.evade_passes,
            "recalibrate_every": args.evade_recalibrate_every,
            "calibration_target": args.evade_calibration_target,
            # 1 entry per epoch, mutated onto `evasion` in place by
            # training.loop.train_classifier. Each entry is a probe's rate at
            # that epoch (a float for a single probe, a list of floats for
            # the multi-probe hinge), constant across epochs when
            # --evade-recalibrate-every is 0.
            "rate_history": evasion.get("rate_history", []),
        }
        if args.evade_psbd
        else None,
        model_dropout=args.model_dropout_train,
        label_smoothing=args.label_smoothing,
    ).as_dict()
    metadata["n_cover"] = n_cover
    metadata.update(trigger_label_record(args, poisoned_set))
    metadata["augment"] = args.augment
    if telemetry_config is not None:
        metadata["telemetry"] = {
            **asdict(telemetry_config),
            "heldout_pairs": args.telemetry_pairs,
        }
    # Read back rather than threaded out of build_training_loader, since the
    # count is cheap to recompute and this keeps that function's return
    # signature untouched for every other caller.
    excluded_count = 0
    if args.exclude_indices_file:
        with open(args.exclude_indices_file) as handle:
            excluded_count = len(json.load(handle))
    metadata["exclude_indices_file"] = args.exclude_indices_file
    metadata["n_excluded"] = excluded_count
    # Without this, evaluation rebuilds the attack from default_config() and a run
    # trained with a modified trigger is scored against a trigger it never saw.
    metadata["attack_config_overrides"] = config_overrides(config, args.attack)
    save_checkpoint(model, num_classes, args.output, metadata=metadata)
    print(f"saved {args.output}")
    print(
        f"time taken: {args.dataset} {args.attack} rate {args.poison_rate} took "
        f"{(time.time() - started) / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
