"""GPU stage: 20 perturbed passes per split at each placement's adaptive rate.

The caches are written by cli.sweep's own functions (bfloat16, batch 64, mask seed
0, the standard PSBD split) into results/_experiments/tact_calibration/models/,
never into results/<folder>/psbd/. A model without psbd_metrics.json (the holdout
retrains) gets its adaptive rate from a 3-pass ladder on the standard 2000-image
validation split first, the way cli.analyze would pick it.

    python -m experiments.tact_calibration.run_passes plan
    python -m experiments.tact_calibration.run_passes run --folder vit_cifar10_tact_0_01
"""

import argparse
import json
import os
import shutil
import time

import torch

from cli.sweep import (
    SweepSettings,
    already_complete,
    bound_operator,
    build_parser,
    cache_config_name,
    load_model_and_loaders,
    sweep_rates,
    write_run_provenance,
)
from data.splits import SPLITS
from defenses.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_or_build_baseline,
    run_provenance_path,
    write_split_manifest,
)
from defenses.decision import ADAPTIVE_SHIFT_TARGET
from defenses.inference import compute_dropout_pass_probs
from defenses.scores import shift_ratio
from experiments._paths import experiment_result_path, experiment_results_dir
from models.positions import DROPOUT_CONFIGS, plug_dropout, unplug_dropout
from scripts.paper._common import clearing_cells, load_coverage, load_psbd_metrics

SLUG = "tact_calibration"
RESULTS_DIR = "results"
MODELS_ROOT = os.path.join(experiment_results_dir(SLUG, RESULTS_DIR), "models")
PLAN_PATH = experiment_result_path(SLUG, "plan.json", RESULTS_DIR)
DEV_SET_PATH = os.path.join("experiments", "cache_readouts", "dev_set.json")
BASIS_PATH = os.path.join("configs", "psbd_basis.json")

FORWARD_PASSES = 20
LADDER_PASSES = 3
MEMORY_FRACTION = 0.15

# (placement id in configs/psbd_basis.json, position, operator, block range)
PLACEMENTS = (
    ("before_attention_norm_token_mask", "before_attention_norm", "token_mask", None),
    ("pre_residual_blocks_5_8", "pre_residual", "dropout", (5, 8)),
)
TUNING = (
    "vit_cifar10_tact_0_01",
    "vit_cifar10_tact_0_05",
    "vit_gtsrb_tact_0_01_cos",
    "vit_gtsrb_tact_0_05",
)
HOLDOUT = (
    "vit_cifar100_tact_0_01_src5",
    "vit_cifar100_tact_0_05_src25",
    "vit_cifar10_tact_0_1_src5",
    "vit_gtsrb_tact_0_1_src12",
    "vit_tiny_tact_0_01_src10",
    "vit_tiny_tact_0_05_src50",
)
# The rest of the panel runs CIFAR-100 and Tiny first, the datasets where the
# earlier calibration by predicted class did its harm.
REST_DATASET_ORDER = ("cifar100", "tiny", "gtsrb", "cifar10")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "run"))
    parser.add_argument("--folder")
    arguments = parser.parse_args()

    if arguments.stage == "plan":
        plan = build_plan()
        write_json(plan, PLAN_PATH)
        print(f"wrote {PLAN_PATH}: {len(plan['models'])} models")
        return

    with open(PLAN_PATH) as handle:
        entry = {m["folder"]: m for m in json.load(handle)["models"]}[arguments.folder]
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, 0)
    record = run_model(entry, torch.device("cuda"))
    write_json(record, os.path.join(MODELS_ROOT, entry["folder"], "run_record.json"))
    print(json.dumps(record))


def build_plan():
    with open(DEV_SET_PATH) as handle:
        dev = [m["folder"] for m in json.load(handle)["models"]]
    panel = []
    for cell in clearing_cells(load_coverage(RESULTS_DIR)):
        rates = adaptive_rates(cell["folder_name"])
        if rates is not None:
            panel.append((cell, rates))
    panel_folders = {cell["folder_name"] for cell, _ in panel}

    models = []
    for folder in TUNING:
        models.append(
            {"folder": folder, "group": "tuning", "rates": adaptive_rates(folder)}
        )
    for folder in dev:
        if folder not in TUNING:
            models.append(
                {
                    "folder": folder,
                    "group": "dev",
                    "rates": adaptive_rates(folder),
                    "in_panel": folder in panel_folders,
                }
            )
    for folder in HOLDOUT:
        models.append({"folder": folder, "group": "holdout", "rates": None})
    taken = {m["folder"] for m in models}
    rest = [(cell, rates) for cell, rates in panel if cell["folder_name"] not in taken]
    rest.sort(
        key=lambda pair: (
            REST_DATASET_ORDER.index(pair[0]["dataset"]),
            pair[0]["folder_name"],
        )
    )
    for cell, rates in rest:
        models.append({"folder": cell["folder_name"], "group": "rest", "rates": rates})
    plan = {
        "forward_passes": FORWARD_PASSES,
        "panel_size": len(panel),
        "models": models,
    }
    return plan


def adaptive_rates(folder):
    metrics = load_psbd_metrics(RESULTS_DIR, folder)
    if metrics is None:
        return None
    rates = {
        placement: metrics["placements"].get(placement, {}).get("adaptive_rate")
        for placement, _, _, _ in PLACEMENTS
    }
    if None in rates.values():
        return None
    return rates


def run_model(entry, device):
    folder = entry["folder"]
    psbd_dir = os.path.join(MODELS_ROOT, folder, "psbd")
    started = time.perf_counter()
    record = {"folder": folder, "group": entry["group"], "placements": {}}

    sweep_args = sweep_namespace(folder, PLACEMENTS[0], 0.5)
    model, architecture, loaders, manifest, metadata = load_model_and_loaders(
        sweep_args, folder, device
    )
    write_split_manifest(psbd_dir, manifest)
    baselines = {
        split: load_or_build_baseline(psbd_dir, split, model, loader, device, True)
        for split, loader in loaders.items()
    }
    record["baseline_seconds"] = time.perf_counter() - started

    for spec in PLACEMENTS:
        placement, position, operator, block_range = spec
        placement_started = time.perf_counter()
        if entry["rates"] is not None:
            rate = entry["rates"][placement]
            ladder = None
        else:
            rate, ladder = ladder_rate(
                model,
                architecture,
                spec,
                loaders["validation"],
                baselines,
                device,
                metadata,
            )
            record["placements"][placement] = {"ladder": ladder}
        record["placements"].setdefault(placement, {})["rate"] = rate
        if rate is None:
            continue

        cache_name = cache_config_name(position, block_range, operator, FORWARD_PASSES)
        reused = reuse_main_cache(folder, psbd_dir, cache_name, rate, baselines)
        record["placements"][placement]["reused_from"] = reused
        if not already_complete(psbd_dir, cache_name, (rate,)):
            args = sweep_namespace(folder, spec, rate)
            write_run_provenance(psbd_dir, args, position, device)
            settings = SweepSettings(
                operator=operator,
                rates=(rate,),
                block_range=block_range,
                cache_name=cache_name,
                dataset=metadata["dataset"],
            )
            sweep_rates(
                model,
                architecture,
                position,
                loaders,
                baselines,
                psbd_dir,
                device,
                FORWARD_PASSES,
                True,
                settings,
            )
        record["placements"][placement]["cache"] = cache_name
        record["placements"][placement]["seconds"] = (
            time.perf_counter() - placement_started
        )

    record["wall_seconds"] = time.perf_counter() - started
    record["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    record["device"] = torch.cuda.get_device_name(device)
    return record


def reuse_main_cache(folder, psbd_dir, cache_name, rate, baselines):
    # An earlier login-node run already wrote 20-pass PSBD-TM caches at the
    # adaptive rate for many panel models, with the same code, GPU, seed and batch
    # size. On vit_cifar10_tact_0_01 a rerun matched it bit for bit, so a cache is
    # copied instead of recomputed when its baselines equal the ones built here.
    main_dir = os.path.join(RESULTS_DIR, folder, "psbd")
    if already_complete(psbd_dir, cache_name, (rate,)):
        return None
    if not already_complete(main_dir, cache_name, (rate,)):
        return None
    for split, (probs, labels, loader_labels) in baselines.items():
        main_probs, main_labels, main_loader_labels = load_baseline(
            baseline_path(main_dir, split)
        )
        if not (
            torch.equal(probs, main_probs)
            and torch.equal(labels, main_labels)
            and torch.equal(loader_labels, main_loader_labels)
        ):
            return None

    for split in SPLITS:
        target = dropout_pass_path(psbd_dir, cache_name, rate, split)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(dropout_pass_path(main_dir, cache_name, rate, split), target)
    shutil.copy2(
        run_provenance_path(main_dir, cache_name),
        run_provenance_path(psbd_dir, cache_name),
    )
    return main_dir


def sweep_namespace(folder, spec, rate):
    _, position, operator, block_range = spec
    argv = [
        "--checkpoint-folder",
        folder,
        "--position",
        position,
        "--operator",
        operator,
        "--forward-passes",
        str(FORWARD_PASSES),
        "--rates",
        str(rate),
        "--results-dir",
        MODELS_ROOT,
    ]
    if block_range is not None:
        argv += ["--block-range", str(block_range[0]), str(block_range[1])]
    args = build_parser().parse_args(argv)
    return args


def ladder_rate(
    model, architecture, spec, validation_loader, baselines, device, metadata
):
    # The ladder is walked upward and stops at the first rate whose clean
    # validation shift ratio reaches the target. Every smaller rate was measured
    # and fell short, so this is select_rate_adaptively's smallest reaching rate.
    placement, position, operator, block_range = spec
    with open(BASIS_PATH) as handle:
        ladder_rates = {b["id"]: b["rates"] for b in json.load(handle)["basis"]}[
            placement
        ]
    position_names = DROPOUT_CONFIGS.get(position, (position,))
    factory = {
        name: bound_operator(operator, metadata["dataset"]) for name in position_names
    }
    _, validation_labels, _ = baselines["validation"]  # (n_validation,)

    ladder = {}
    for rate in ladder_rates:
        handles = plug_dropout(
            model, architecture, position_names, factory, rate, block_range=block_range
        )
        try:
            _, per_pass_argmax = compute_dropout_pass_probs(
                model,
                validation_loader,
                validation_labels,
                device,
                LADDER_PASSES,
                True,
                0,
            )  # (passes, n_validation)
        finally:
            unplug_dropout(handles)
        ladder[str(rate)] = shift_ratio(validation_labels, per_pass_argmax)
        if ladder[str(rate)] >= ADAPTIVE_SHIFT_TARGET:
            return rate, ladder
    return None, ladder


def write_json(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
