"""GPU stage: score a backdoored model's own training images with PSBD, STRIP and CD-L.

Rebuilds the poisoned training set exactly as `cli.train_backdoor` chose it, with
no augmentation, and scores 4 groups (poisoned, cover, clean_train and the PSBD
validation split) at both rate rules of every placement. Each part is written to
its own file under results/_experiments/training_set_detection/raw/<folder>/ and
skipped when present, so a model interrupted by the GPU curfew resumes where it
stopped. Nothing is written under results/<folder>/psbd/.

Run from the main checkout, which holds checkpoints/, raw_data/ and results/:

    PYTHONPATH=<worktree> .venv/bin/python <worktree>/experiments/training_set_detection/score.py \
        --models vit_cifar10_badnet_a2o_0_1
"""

import argparse
import os
import sys
import time
from dataclasses import replace

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(EXPERIMENT_DIR)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms.v2 as transforms_v2  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from attacks import apply_config_overrides, build_attack, default_config  # noqa: E402
from attacks.poisoning import (  # noqa: E402
    PoisonedTrainingSet,
    attack_success_label,
    poisoned_label,
)
from cli.sweep import bound_operator, cache_config_name, resolve_architecture  # noqa: E402
from cli.train_backdoor import build_training_set, resolve_cover_rate  # noqa: E402
from data.loading import (  # noqa: E402
    base_image_transform,
    extract_labels,
    limit_dataset,
    load_clean_datasets,
)
from data.registry import DATASET_REGISTRY  # noqa: E402
from data.splits import (  # noqa: E402
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from defenses.cache import (  # noqa: E402
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_run_provenance,
    read_split_manifest,
)
from defenses.inference import build_baseline_cache, compute_dropout_pass_probs  # noqa: E402
from defenses.operators import effective_forward_passes  # noqa: E402
from detectors import DetectorContext, build_detector  # noqa: E402
from experiments.cache_readouts.shared import (  # noqa: E402
    choose_rate,
    load_baselines,
    validation_shift_by_rate,
)
from experiments.training_set_detection import common  # noqa: E402
from experiments.training_set_detection.common import (  # noqa: E402
    BATCH_SIZE,
    CD_L_SUBSET,
    CHECKPOINTS_DIR,
    CLEAN_SAMPLE_SIZE,
    COVER_SAMPLE_SIZE,
    FORWARD_PASSES,
    GROUP_STREAMS,
    MASK_SEED,
    MEMBERSHIP_SAMPLE_SIZE,
    PLACEMENTS,
    RATE_SUBSET,
    RAW_DATA_DIR,
    RESULTS_DIR,
    SAMPLE_SEED,
    SHIFT_TARGET,
    STRIP_SUBSET,
    has_part,
    load_part,
    passes_part,
    placements_for,
    save_part,
)
from models.backbones import load_checkpoint, network_core  # noqa: E402
from models.positions import DROPOUT_CONFIGS, plug_dropout, unplug_dropout  # noqa: E402

NUM_WORKERS = 4
# The login node's CPUs are oversubscribed at night, and default OpenMP thread
# counts spin rather than compute.
TORCH_THREADS = 4
# The card is shared by several queues tonight, the coordinator's cap.
MEMORY_FRACTION = 0.15
SCORED_GROUPS = ("poisoned", "cover", "clean_train")
# No part starts after this local time before the day window closes at 17:00, so
# a part started late still ends before the 07:00 GPU curfew.
LAST_PART_START = (6, 30)
DAY_WINDOW_END = (17, 0)


def main():
    args = parse_args()
    torch.set_num_threads(TORCH_THREADS)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, 0)

    if args.smoke is not None:
        common.RAW_ROOT = os.path.join("scratch", common.SLUG, "smoke_raw")

    for folder in args.models:
        started = time.perf_counter()
        score_model(folder, device, args.skip_detectors, args.smoke)
        print(f"[ok] {folder} {time.perf_counter() - started:.0f} s", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-detectors", action="store_true")
    parser.add_argument(
        "--smoke",
        type=int,
        default=None,
        help="cap every group at this many images and write under scratch/",
    )
    args = parser.parse_args()
    return args


def score_model(folder, device, skip_detectors, smoke):
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)
    architecture = resolve_architecture(checkpoint_path, metadata)
    psbd_dir = os.path.join(RESULTS_DIR, folder, "psbd")

    rebuilt = rebuild_training_groups(metadata, smoke)
    if not has_part(folder, "groups"):
        save_part(rebuilt["record"], folder, "groups")

    model = load_checkpoint(architecture, checkpoint_path, device)
    psbd_loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=RAW_DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    cached_manifest = read_split_manifest(psbd_dir)
    assert manifest["heldout_indices"] == cached_manifest["heldout_indices"], folder
    validation_loader = psbd_loaders["validation"]

    loaders = group_loaders(rebuilt)
    loaders["validation"] = validation_loader

    if not has_part(folder, "baseline"):
        check_clock()
        started = time.perf_counter()
        baseline = baseline_part(model, architecture, loaders, psbd_dir, device)
        baseline["seconds"] = time.perf_counter() - started
        save_part(baseline, folder, "baseline")
    baseline = load_part(folder, "baseline")

    cached_baselines = load_baselines(psbd_dir)
    for placement in placements_for(folder):
        score_placement(
            model,
            architecture,
            folder,
            placement,
            psbd_dir,
            cached_baselines,
            rebuilt,
            loaders,
            baseline,
            metadata["dataset"],
            device,
        )

    if skip_detectors:
        return
    context = detector_context(model, metadata, validation_loader, device)
    if not has_part(folder, "strip"):
        check_clock()
        save_part(
            strip_part(model, context, rebuilt, loaders, folder, device),
            folder,
            "strip",
        )
    if not has_part(folder, "cd_l"):
        check_clock()
        save_part(
            cd_l_part(model, context, rebuilt, loaders, folder, device), folder, "cd_l"
        )


def check_clock():
    now = time.localtime()
    clock = (now.tm_hour, now.tm_min)
    if LAST_PART_START <= clock < DAY_WINDOW_END:
        raise SystemExit(f"GPU window closed at {clock}, resume after 17:00")


def rebuild_training_groups(metadata, smoke=None):
    dataset = metadata["dataset"]
    spec = DATASET_REGISTRY[dataset]
    attack_name = metadata["attack"]

    # The training entrypoint's recipe, cli.train_backdoor.build_training_loader,
    # minus augmentation. Runs that predate the seeding commit record seed null
    # and drew with 0 (cli.backfill).
    seed = metadata["seed"] if metadata["seed"] is not None else 0
    transform = base_image_transform(spec.image_size)
    train_clean, _ = load_clean_datasets(dataset, transform, RAW_DATA_DIR)
    train_clean = limit_dataset(train_clean, metadata.get("max_samples"), seed)
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)

    config = default_config(attack_name)
    cover_rate = resolve_cover_rate(attack_name, metadata["poison_rate"], None)
    if cover_rate is not None:
        config = replace(config, cover_rate=cover_rate)
    config = apply_config_overrides(config, metadata.get("attack_config_overrides"))
    recorded_cover_rate = metadata.get("cover_rate") or 0.0
    assert abs(getattr(config, "cover_rate", 0.0) - recorded_cover_rate) < 1e-9, (
        getattr(config, "cover_rate", 0.0),
        recorded_cover_rate,
    )
    attack = build_attack(
        attack_name, config, spec.image_size, metadata["target_label"]
    )

    training_set, realized_rate = build_training_set(
        train_clean,
        attack,
        config,
        metadata["poison_rate"],
        seed,
        normalize,
        spec.num_classes,
        metadata.get("trigger_label_probability", 1.0),
    )
    clean_set = PoisonedTrainingSet(
        train_clean, attack, set(), normalize, spec.num_classes
    )

    labels = np.array(extract_labels(train_clean))  # (n_train,)
    poison = np.array(sorted(training_set.poison_indices), dtype=np.int64)
    cover = np.array(
        sorted(getattr(training_set, "cover_indices", set()) or set()), dtype=np.int64
    )
    untouched = np.setdiff1d(
        np.arange(len(labels)), np.concatenate([poison, cover])
    )  # (n_clean_population,)

    # 1 generator per group, so each sample stays fixed when another group's
    # size changes.
    orders = {
        "poisoned": group_generator("poisoned").permutation(poison),
        "cover": group_generator("cover").choice(
            cover, size=min(COVER_SAMPLE_SIZE, len(cover)), replace=False
        ),
        "clean_train": group_generator("clean_train").choice(
            untouched, size=min(CLEAN_SAMPLE_SIZE, len(untouched)), replace=False
        ),
    }
    target_label = metadata["target_label"]
    orders["target_clean"] = untouched[labels[untouched] == target_label]
    orders["poisoned_clean"] = orders["poisoned"][:MEMBERSHIP_SAMPLE_SIZE]
    if smoke is not None:
        orders = {name: order[:smoke] for name, order in orders.items()}

    success_labels = np.array(
        [
            attack_success_label(
                attack.label_mode,
                int(labels[index]),
                attack.target_label,
                spec.num_classes,
                attack.num_targets,
            )
            for index in orders["poisoned"]
        ],
        dtype=np.int64,
    )  # (n_poisoned,)

    # The label every training row carried, which is what Spectral Signatures
    # groups by. Covers keep their true label.
    training_labels = labels.copy()  # (n_train,)
    for index in poison:
        training_labels[index] = poisoned_label(
            attack.label_mode,
            int(labels[index]),
            attack.target_label,
            spec.num_classes,
            attack.num_targets,
        )

    record = {
        "dataset": dataset,
        "training_label_counts": np.bincount(
            training_labels, minlength=spec.num_classes
        ).tolist(),
        "attack": attack_name,
        "label_mode": attack.label_mode,
        "target_label": target_label,
        "num_classes": spec.num_classes,
        "seed_used": seed,
        "seed_recorded": metadata["seed"] is not None,
        "realized_poison_rate": realized_rate,
        "n_train": len(labels),
        "population": {
            "poisoned": len(poison),
            "cover": len(cover),
            "clean_train": len(untouched),
        },
        "recorded_counts": {
            "n_poisoned": metadata.get("n_poisoned"),
            "n_cover": metadata.get("n_cover"),
        },
        "orders": {name: torch.from_numpy(np.asarray(o)) for name, o in orders.items()},
        "true_labels": {
            name: torch.from_numpy(labels[np.asarray(o, dtype=np.int64)])
            for name, o in orders.items()
        },
        "success_labels": torch.from_numpy(success_labels),
        "recorded_asr": metadata.get("asr"),
    }
    rebuilt = {
        "record": record,
        "training_set": training_set,
        "clean_set": clean_set,
        "orders": orders,
    }
    return rebuilt


def group_generator(group):
    generator = np.random.default_rng([SAMPLE_SEED, GROUP_STREAMS[group]])
    return generator


def group_loaders(rebuilt):
    orders = rebuilt["orders"]
    loaders = {
        name: ordered_loader(rebuilt["training_set"], orders[name])
        for name in ("poisoned", "cover", "clean_train", "target_clean")
    }
    # The untriggered image of a poisoned index, for the membership check.
    loaders["poisoned_clean"] = ordered_loader(
        rebuilt["clean_set"], orders["poisoned_clean"]
    )
    return loaders


def ordered_loader(dataset, indices, limit=None):
    chosen = [int(i) for i in indices][:limit]
    loader = DataLoader(
        Subset(dataset, chosen),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    return loader


def baseline_part(model, architecture, loaders, psbd_dir, device):
    part = {}
    for name, loader in loaders.items():
        if len(loader.dataset) == 0:
            part[name] = None
            continue
        with captured_features(model, architecture) as features:
            batches = build_baseline_cache(model, loader, device, True)
        probs = torch.cat([b["probs"] for b in batches])  # (n, num_classes)
        part[name] = {
            "probs": probs,
            "loader_labels": torch.cat([b["loader_labels"] for b in batches]),  # (n,)
            "labels": probs.argmax(dim=1),  # (n,)
            "features": torch.cat(features).half(),  # (n, feature_dim)
        }

    # The validation baseline must reproduce cli.sweep's cached one, since every
    # threshold below is read from it.
    cached_probs, cached_labels, _ = load_baseline(
        baseline_path(psbd_dir, "validation")
    )
    fresh = part["validation"]
    part["validation_check"] = {
        "max_abs_prob_difference": float((fresh["probs"] - cached_probs).abs().max()),
        "argmax_agreement": float((fresh["labels"] == cached_labels).float().mean()),
    }
    return part


class captured_features:
    """The vector the classifier head reads, captured batch by batch during a pass.

    ViT: the class token after the encoder's final LayerNorm. ResNet-18: the pooled
    input to fc. Spectral Signatures reads these in the CPU stage.
    """

    def __init__(self, model, architecture):
        self.model = model
        self.architecture = architecture
        self.batches = []

    def __enter__(self):
        core = network_core(self.model)
        if self.architecture == "resnet18":

            def hook(_module, inputs):
                self.batches.append(inputs[0].detach().float().cpu())  # (batch, 512)

            self.handle = core.fc.register_forward_pre_hook(hook)
        else:

            def hook(_module, _inputs, output):
                cls_token = output[:, 0, :]  # (batch, 768)
                self.batches.append(cls_token.detach().float().cpu())

            self.handle = core.encoder.ln.register_forward_hook(hook)
        return self.batches

    def __exit__(self, *exc):
        self.handle.remove()


def score_placement(
    model,
    architecture,
    folder,
    placement,
    psbd_dir,
    cached_baselines,
    rebuilt,
    loaders,
    baseline,
    dataset,
    device,
):
    spec = PLACEMENTS[placement]
    cache = spec["cache"]
    assert cache == cache_config_name(
        spec["position"], spec["block_range"], spec["operator"]
    ), placement
    provenance = read_run_provenance(psbd_dir, cache)
    assert provenance.get("mask_seed", MASK_SEED) == MASK_SEED, provenance
    assert provenance.get("batch_size", BATCH_SIZE) == BATCH_SIZE, provenance

    shift_by_rate = validation_shift_by_rate(psbd_dir, cache, cached_baselines)
    our_rate = choose_rate(shift_by_rate, "adaptive")

    ladder_part = f"ladder_{placement}"
    if not has_part(folder, ladder_part):
        check_clock()
        started = time.perf_counter()
        candidates = [r for r, s in sorted(shift_by_rate.items()) if s >= SHIFT_TARGET]
        candidates = candidates or sorted(shift_by_rate)
        ladder = {
            "candidates": candidates,
            "validation_shift": {r: shift_by_rate[r] for r in sorted(shift_by_rate)},
            "our_rate": our_rate,
            "subset_sizes": {
                group: min(size, len(rebuilt["orders"][group]))
                for group, size in RATE_SUBSET.items()
            },
            "per_rate": {},
        }
        for rate in candidates:
            ladder["per_rate"][rate] = run_passes(
                model,
                architecture,
                spec,
                rate,
                dataset,
                {
                    group: prefix_loader(rebuilt, group, RATE_SUBSET[group])
                    for group in RATE_SUBSET
                },
                {
                    group: baseline[group]["labels"][: RATE_SUBSET[group]]
                    if baseline[group] is not None
                    else None
                    for group in RATE_SUBSET
                },
                device,
            )
        ladder["seconds"] = time.perf_counter() - started
        save_part(ladder, folder, ladder_part)
    ladder = load_part(folder, ladder_part)
    li_rate = li_rate_rule(ladder, rebuilt["record"]["population"])

    for rate in sorted({r for r in (our_rate, li_rate) if r is not None}):
        part = passes_part(placement, rate)
        if has_part(folder, part):
            continue
        check_clock()
        started = time.perf_counter()
        group_names = SCORED_GROUPS + ("validation",)
        passes = run_passes(
            model,
            architecture,
            spec,
            rate,
            dataset,
            {name: loaders[name] for name in group_names},
            {
                name: baseline[name]["labels"] if baseline[name] is not None else None
                for name in group_names
            },
            device,
        )
        passes["validation_check"] = compare_to_cache(
            passes["validation"], psbd_dir, cache, rate
        )
        passes["rate"] = rate
        passes["seconds"] = time.perf_counter() - started
        save_part(passes, folder, part)


def li_rate_rule(ladder, population):
    """Li et al.'s rate: max validation shift minus whole-training-set shift.

    Their `select_dropout_rate` (github.com/WL-619/PSBD, detection/psbd.py) takes
    the argmax of val_ratio - total_ratio over the rates whose validation shift is
    at least 0.8, and over every rate when none is. np.argmax keeps the first
    maximum, the smallest rate. The whole-set shift is each group's measured shift
    weighted by its population count.
    """
    total = sum(population.values())
    gaps = []
    for rate in ladder["candidates"]:
        measured = ladder["per_rate"][rate]
        training_shift = 0.0
        for group, count in population.items():
            if count == 0:
                continue
            argmax = measured[group]["per_pass_argmax"]  # (passes, n_subset)
            labels = measured[group]["baseline_labels"]  # (n_subset,)
            group_shift = float(
                (argmax.long() != labels.view(1, -1).long()).float().mean()
            )
            training_shift += group_shift * count / total
        gaps.append(ladder["validation_shift"][rate] - training_shift)
    rate = ladder["candidates"][int(np.argmax(gaps))]
    return rate


def prefix_loader(rebuilt, group, size):
    loader = ordered_loader(rebuilt["training_set"], rebuilt["orders"][group], size)
    return loader


def run_passes(model, architecture, spec, rate, dataset, loaders, labels, device):
    position_names = DROPOUT_CONFIGS.get(spec["position"], (spec["position"],))
    operator = bound_operator(spec["operator"], dataset)
    factory = {name: operator for name in position_names}
    passes = effective_forward_passes(spec["operator"], FORWARD_PASSES)

    measured = {}
    handles = plug_dropout(
        model,
        architecture,
        position_names,
        factory,
        rate,
        block_range=spec["block_range"],
    )
    try:
        for name, loader in loaders.items():
            if labels[name] is None or len(loader.dataset) == 0:
                measured[name] = None
                continue
            # cli.sweep reseeds once per split, so each group's masks start from
            # the same state the cached validation split did.
            per_pass_probs, per_pass_argmax = compute_dropout_pass_probs(
                model, loader, labels[name], device, passes, True, MASK_SEED
            )  # (passes, n) each
            measured[name] = {
                "per_pass_probs": per_pass_probs,
                "per_pass_argmax": per_pass_argmax,
                "baseline_labels": labels[name][: per_pass_probs.shape[1]],
            }
    finally:
        unplug_dropout(handles)
    return measured


def compare_to_cache(fresh, psbd_dir, cache, rate):
    cached_probs, cached_argmax = load_dropout_pass_probs(
        dropout_pass_path(psbd_dir, cache, rate, "validation")
    )  # (passes, n_validation) each
    check = {
        "max_abs_prob_difference": float(
            (fresh["per_pass_probs"] - cached_probs).abs().max()
        ),
        "argmax_agreement": float(
            (fresh["per_pass_argmax"] == cached_argmax).float().mean()
        ),
    }
    return check


def detector_context(model, metadata, validation_loader, device):
    spec = DATASET_REGISTRY[metadata["dataset"]]
    # cli.baselines' own context, seed included, so the validation scores
    # reproduce the cached test-time records.
    context = DetectorContext(
        model=model,
        device=device,
        mean=spec.mean,
        std=spec.std,
        validation_loader=validation_loader,
        num_classes=spec.num_classes,
        use_bfloat16=True,
        seed=PSBD_SPLIT_SEED,
    )
    return context


def strip_part(model, context, rebuilt, loaders, folder, device):
    started = time.perf_counter()
    detector = build_detector("strip", context)
    part = {
        group: detector(model, prefix_loader(rebuilt, group, size), device)
        for group, size in STRIP_SUBSET.items()
        if len(rebuilt["orders"][group])
    }
    part["validation"] = detector(model, loaders["validation"], device)
    part["validation_check"] = compare_detector_cache(
        part["validation"], folder, "strip"
    )
    part["seconds"] = time.perf_counter() - started
    return part


def cd_l_part(model, context, rebuilt, loaders, folder, device):
    started = time.perf_counter()
    detector = build_detector("cd_l", context)
    part = {
        group: detector(model, prefix_loader(rebuilt, group, size), device)
        for group, size in CD_L_SUBSET.items()
        if len(rebuilt["orders"][group])
    }
    # 2000 validation images cost CD-L about 500000 forwards, so the test-time
    # record's validation scores are reused when cli.baselines wrote them with
    # the same seed and code path.
    cached = cached_detector_scores(folder, "cd_l")
    if cached is None:
        part["validation"] = detector(model, loaders["validation"], device)
        part["validation_source"] = "computed"
    else:
        part["validation"] = cached
        part["validation_source"] = (
            "results/<folder>/detectors/cd_l_scores_validation.pt"
        )
    part["seconds"] = time.perf_counter() - started
    return part


def cached_detector_scores(folder, name):
    path = os.path.join(
        RESULTS_DIR, folder, "detectors", f"{name}_scores_validation.pt"
    )
    if not os.path.exists(path):
        return None
    scores = torch.load(path, map_location="cpu")
    if isinstance(scores, dict):
        scores = scores["scores"]
    return scores.float()


def compare_detector_cache(fresh, folder, name):
    cached = cached_detector_scores(folder, name)
    if cached is None:
        return None
    check = {"max_abs_difference": float((fresh - cached).abs().max())}
    return check


if __name__ == "__main__":
    main()
