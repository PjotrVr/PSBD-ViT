"""Stage 1 of the novel probes: raw per-pass probabilities on reduced PSBD splits.

For each model the ladder of every probe runs on the 2000 clean validation images,
the rate is chosen there (the adaptive 0.8 rule for PSBD's sign, the plan's
clean-stability rule for a fragile-sign probe) and only the chosen rates run on the
first PAIRS triggered images and their clean twins. Nothing here reads a triggered
image before a rate is fixed. Scores and verdicts are stage 2 (evaluate.py).

    .venv/bin/python -m experiments.novel_probes.measure --set dev
    .venv/bin/python -m experiments.novel_probes.measure --folders vit_cifar10_benign
    .venv/bin/python -m experiments.novel_probes.measure --smoke --device cpu \
        --folders vit_cifar10_badnet_a2o_0_01
"""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
from lightning import seed_everything  # noqa: E402

from data.splits import (  # noqa: E402
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from defenses.decision import (  # noqa: E402
    ADAPTIVE_SHIFT_TARGET,
    select_rate_adaptively,
    select_rate_at_matched_shift,
)
from defenses.inference import forward_probs  # noqa: E402
from defenses.scores import shift_ratio  # noqa: E402
from experiments._paths import experiment_result_path  # noqa: E402
from experiments.novel_probes import operators  # noqa: E402
from models.backbones import load_checkpoint  # noqa: E402
from models.positions import unplug_dropout  # noqa: E402

SLUG = "novel_probes"
DEV_SET_PATH = os.path.join(REPO_ROOT, "experiments", "cache_readouts", "dev_set.json")
PREREGISTRATION_PATH = os.path.join(
    REPO_ROOT, "experiments", SLUG, "preregistration.json"
)
# The plan's U3 unit: all 2000 validation images, 1000 triggered images and their
# clean twins, k = 3.
PAIRS = 1000
FORWARD_PASSES = 3
BATCH_SIZE = 256
GPU_MEMORY_FRACTION = 0.15
CPU_THREADS = 4
PASS_SEED = 0
# The plan's mirror rule for a fragile-sign probe: the largest rung whose clean
# validation shift stays at or below this, fixed before any triggered image.
STABILITY_MAX_SHIFT = 0.05

BENIGN_FOLDERS = (
    "vit_cifar10_benign",
    "vit_gtsrb_benign",
    "vit_cifar100_benign",
    "vit_tiny_benign",
)
# 1 trigger per category (patch, additive global, warp), each stamped on every
# benign model, where a probe with no backdoor to find must read about 0.5.
BENIGN_TRIGGERS = ("badnet_a2o", "blend", "wanet")
HOLDOUT_DATASETS = ("cifar100", "tiny")

# Removing a random share of the class token's edges leaves it the average of
# the rest, which barely moves a clean prediction until almost every edge is
# gone, so the ladder runs up to the full knockout. The knockout was coded with
# PSBD's sign and read inverted on the first CPU smoke of 1 development model.
# The mechanism says why: removing the late reads breaks the OR a patch trigger
# needs, while the class token already holds clean evidence from blocks 1 to 8.
# Its sign is therefore fragile, with PSBD's sign kept as the canonical reading.
KNOCKOUT_LADDER = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0)

# sign "psbd": a low fractional PSU means poisoned, rate by the adaptive rule.
# sign "fragile": a high one means poisoned, rate by the stability rule. Every
# fragile probe also runs at its adaptive-or-nearest rung, read with PSBD's sign
# as a secondary reading. kind "hooked" plugs once per rate. kind "fixed_drop"
# sets a deterministic token set per batch.
PROBES = {
    "psbd_tm": {
        "sign": "psbd",
        "kind": "hooked",
        "attach": operators.attach_psbd_tm,
        "ladder": (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        "passes": FORWARD_PASSES,
    },
    "middle_band": {
        "sign": "psbd",
        "kind": "hooked",
        "attach": operators.attach_middle_band,
        "ladder": (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
        "passes": FORWARD_PASSES,
        "benign_only": True,
    },
    "cls_read_knockout": {
        "sign": "fragile",
        "kind": "hooked",
        "attach": operators.attach_cls_read_knockout,
        "ladder": KNOCKOUT_LADDER,
        "passes": FORWARD_PASSES,
    },
    "cls_read_knockout_early": {
        "sign": "fragile",
        "kind": "hooked",
        "attach": lambda model, rate: operators.attach_cls_read_knockout(
            model, rate, operators.EARLY_BLOCKS
        ),
        "ladder": KNOCKOUT_LADDER,
        "passes": FORWARD_PASSES,
    },
    "key_mask": {
        "sign": "psbd",
        "kind": "hooked",
        "attach": operators.attach_key_mask,
        # The smoke's clean shift stayed below 0.8 at 0.9, so the ladder
        # continues toward removing every key but the class token's.
        "ladder": (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
        "passes": FORWARD_PASSES,
    },
    "stratified_token_mask": {
        "sign": "psbd",
        "kind": "hooked",
        "attach": operators.attach_stratified_token_mask,
        "ladder": (0.3, 0.4, 0.5, 0.6, 0.7, 0.75),
        "passes": FORWARD_PASSES,
    },
    "active_neuron_dropout": {
        "sign": "psbd",
        "kind": "hooked",
        "attach": operators.attach_active_neuron_dropout,
        "ladder": (0.1, 0.2, 0.3, 0.5, 0.7, 0.9),
        "passes": FORWARD_PASSES,
    },
    "rollout_token_drop": {
        "sign": "fragile",
        "kind": "fixed_drop",
        "ladder": (1, 2, 4, 8, 16, 32, 64),
        "passes": 1,
    },
    "random_token_drop": {
        "sign": "fragile",
        "kind": "fixed_drop",
        "ladder": (1, 2, 4, 8, 16, 32, 64),
        "passes": FORWARD_PASSES,
    },
    "native_jitter": {
        "sign": "fragile",
        "kind": "hooked",
        "attach": operators.attach_native_jitter,
        "ladder": (0.25, 0.5, 1.0, 2.0),
        "passes": FORWARD_PASSES,
    },
}
SMOKE_RUNGS = 2


def main():
    args = parse_args()
    torch.set_num_threads(CPU_THREADS)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)

    for folder in selected_folders(args):
        path = passes_path(folder, args.smoke)
        if os.path.exists(path) and not args.overwrite:
            print(f"skip {folder}, {path} exists", flush=True)
            continue
        started = time.perf_counter()
        record = measure_folder(folder, args, device)
        record["wall_seconds"] = time.perf_counter() - started
        write_record(record, path)
        print(f"wrote {path} in {record['wall_seconds']:.0f} s", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=("dev", "benign", "holdout"))
    parser.add_argument("--folders", nargs="*", default=[])
    parser.add_argument("--probes", nargs="*", default=list(PROBES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pairs", type=int, default=PAIRS)
    parser.add_argument("--validation-count", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return args


def selected_folders(args):
    folders = list(args.folders)
    if args.set == "dev":
        with open(DEV_SET_PATH) as handle:
            folders += [entry["folder"] for entry in json.load(handle)["models"]]
    elif args.set == "benign":
        folders += list(BENIGN_FOLDERS)
    elif args.set == "holdout":
        # Imported here because the ledger read is needed only for this set.
        from experiments.cache_readouts.shared import load_model_set

        assert os.path.exists(PREREGISTRATION_PATH), "freeze the probes first"
        folders += [model["folder_name"] for model in load_model_set("holdout")]

    # The held-out half is read once, after the design is frozen. A benign model
    # of a held-out dataset carries no backdoor to select on and is allowed.
    for folder in folders:
        dataset = folder.split("_")[1]
        if dataset in HOLDOUT_DATASETS and "benign" not in folder:
            assert os.path.exists(PREREGISTRATION_PATH), (
                f"{folder} is held out, write {PREREGISTRATION_PATH} first"
            )
    return folders


def passes_path(folder, smoke):
    name = f"{folder}_smoke.pt" if smoke else f"{folder}.pt"
    path = experiment_result_path(SLUG, os.path.join("passes", name))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def measure_folder(folder, args, device):
    checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)
    assert metadata["architecture"] == "vit", folder
    model = load_checkpoint("vit", checkpoint_path, device).eval()
    benign = metadata["attack"] == "benign"
    triggers = BENIGN_TRIGGERS if benign else (None,)
    probes = [
        name
        for name in args.probes
        if benign or not PROBES[name].get("benign_only", False)
    ]

    subjects = {
        trigger_key(trigger): load_subject(checkpoint_path, trigger, args)
        for trigger in triggers
    }
    validation = next(iter(subjects.values()))["validation"]
    if args.validation_count is not None:
        validation = validation[: args.validation_count]

    record = {
        "folder": folder,
        "attack": metadata["attack"],
        "dataset": metadata["dataset"],
        "benign": benign,
        "pairs": args.pairs,
        "passes": FORWARD_PASSES,
        "pass_seed": PASS_SEED,
        "smoke": args.smoke,
        "baseline": {"validation": baseline_passes(model, validation, args, device)},
        "subjects": {},
        "probes": {},
    }
    for key, subject in subjects.items():
        record["subjects"][key] = {
            "probe_attack": subject["probe_attack"],
            "backdoor_rows": subject["backdoor_rows"],
            "clean_rows": subject["clean_rows"],
            "baseline": {
                split: baseline_passes(model, subject[split], args, device)
                for split in ("clean", "backdoor")
            },
        }

    hooks_before = hook_count(model)
    for name in probes:
        clock = time.perf_counter()
        record["probes"][name] = measure_probe(
            model, name, validation, subjects, record, args, device
        )
        record["probes"][name]["seconds"] = time.perf_counter() - clock
        assert hook_count(model) == hooks_before, f"{name} left a hook behind"
        print(
            f"{folder} {name} rates {record['probes'][name]['chosen']} "
            f"{record['probes'][name]['seconds']:.0f} s",
            flush=True,
        )
    return record


def trigger_key(trigger):
    key = trigger or "own"
    return key


def load_subject(checkpoint_path, trigger, args):
    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        probe_attack=trigger,
        probe_target_label=0 if trigger else None,
    )
    validation_set = loaders["validation"].dataset
    clean_set = loaders["clean"].dataset
    backdoor_set = loaders["backdoor"].dataset

    # The manifest's backdoor order is a slice of 1 fixed permutation, so its
    # first rows are a random sample of the eligible images, and each has its
    # clean twin at a known row of the clean split.
    count = min(args.pairs, len(backdoor_set))
    row_of = {
        original: row for row, original in enumerate(manifest["analysis_clean_indices"])
    }
    backdoor_rows = list(range(count))
    clean_rows = [
        row_of[original] for original in manifest["analysis_backdoor_indices"][:count]
    ]

    subject = {
        "probe_attack": manifest["probe_attack"],
        "backdoor_rows": backdoor_rows,
        "clean_rows": clean_rows,
        "validation": stack_images(validation_set, range(len(validation_set))),
        "clean": stack_images(clean_set, clean_rows),
        "backdoor": stack_images(backdoor_set, backdoor_rows),
    }
    return subject


def stack_images(dataset, rows):
    images = torch.stack([dataset[row][0] for row in rows])  # (n, C, H, W)
    return images


def baseline_passes(model, images, args, device):
    use_bfloat16 = device.type == "cuda"
    probs = []
    with torch.inference_mode():
        for start in range(0, len(images), args.batch_size):
            batch = images[start : start + args.batch_size]
            probs.append(
                forward_probs(model, batch, device, use_bfloat16).cpu()
            )  # (batch, num_classes)
    baseline_probs = torch.cat(probs)  # (n, num_classes)
    baseline = {
        "probs": baseline_probs,
        "labels": baseline_probs.argmax(dim=1),  # (n,)
    }
    return baseline


def measure_probe(model, name, validation, subjects, record, args, device):
    spec = PROBES[name]
    ladder = spec["ladder"][-SMOKE_RUNGS:] if args.smoke else spec["ladder"]
    validation_labels = record["baseline"]["validation"]["labels"]

    ladder_passes = {}
    shift_by_rate = {}
    for rate in ladder:
        ladder_passes[rate] = perturbed_passes(
            model, name, rate, validation, validation_labels, args, device
        )
        shift_by_rate[rate] = shift_ratio(
            validation_labels, ladder_passes[rate]["argmax"]
        )

    chosen = choose_rates(shift_by_rate, spec["sign"])
    tests = {}
    for key, subject in subjects.items():
        tests[key] = {}
        for rate in sorted(
            set(value for value in chosen.values() if value is not None)
        ):
            tests[key][rate] = {
                split: perturbed_passes(
                    model,
                    name,
                    rate,
                    subject[split],
                    record["subjects"][key]["baseline"][split]["labels"],
                    args,
                    device,
                )
                for split in ("clean", "backdoor")
            }

    measured = {
        "sign": spec["sign"],
        "ladder": list(ladder),
        "validation_shift": shift_by_rate,
        "chosen": chosen,
        "validation": ladder_passes,
        "tests": tests,
    }
    return measured


def choose_rates(shift_by_rate, sign):
    # Every rule reads clean validation only. "primary" is the rate the probe is
    # judged at, "canonical" the adaptive rule with the nearest rung as its
    # fallback, which a fragile probe also carries as a secondary reading.
    adaptive = select_rate_adaptively(shift_by_rate, ADAPTIVE_SHIFT_TARGET)
    nearest = select_rate_at_matched_shift(shift_by_rate, ADAPTIVE_SHIFT_TARGET)
    canonical = adaptive if adaptive is not None else nearest

    chosen = {"adaptive": adaptive, "canonical": canonical}
    if sign == "psbd":
        chosen["primary"] = canonical
        return chosen

    stable = [
        rate for rate, shift in shift_by_rate.items() if shift <= STABILITY_MAX_SHIFT
    ]
    chosen["stability"] = max(stable) if stable else None
    chosen["primary"] = (
        chosen["stability"] if chosen["stability"] is not None else min(shift_by_rate)
    )
    return chosen


def perturbed_passes(model, name, rate, images, labels, args, device):
    spec = PROBES[name]
    use_bfloat16 = device.type == "cuda"
    seed_everything(PASS_SEED, verbose=False)

    # A rollout ranking needs the unperturbed model, so it is computed for the
    # whole split before any hook goes on.
    rollout = (
        rollout_scores(model, images, args, device)
        if name == "rollout_token_drop"
        else None
    )

    state = {}
    if spec["kind"] == "fixed_drop":
        handles = operators.attach_fixed_token_drop(model, state)
    else:
        handles = spec["attach"](model, rate)

    prob_batches, argmax_batches = [], []
    try:
        with torch.inference_mode():
            for start in range(0, len(images), args.batch_size):
                batch = images[start : start + args.batch_size].to(device)
                batch_labels = labels[start : start + args.batch_size].to(device)
                prob_columns, argmax_columns = [], []
                for _ in range(spec["passes"]):
                    if name == "rollout_token_drop":
                        scores = rollout[start : start + args.batch_size].to(device)
                        state["drop"] = operators.top_tokens_drop(scores, int(rate))
                    elif name == "random_token_drop":
                        state["drop"] = operators.random_tokens_drop(
                            len(batch), token_count(model), int(rate), device
                        )
                    probs = forward_probs(
                        model, batch, device, use_bfloat16
                    )  # (batch, num_classes)
                    prob_columns.append(
                        probs.gather(1, batch_labels.view(-1, 1)).squeeze(1).cpu()
                    )  # (batch,)
                    argmax_columns.append(probs.argmax(dim=1).to(torch.int16).cpu())
                prob_batches.append(torch.stack(prob_columns))  # (passes, batch)
                argmax_batches.append(torch.stack(argmax_columns))  # (passes, batch)
    finally:
        unplug_dropout(handles)

    passes = {
        "probs": torch.cat(prob_batches, dim=1),  # (passes, n)
        "argmax": torch.cat(argmax_batches, dim=1),  # (passes, n)
    }
    return passes


def rollout_scores(model, images, args, device):
    use_bfloat16 = device.type == "cuda"
    scores = []
    for start in range(0, len(images), args.batch_size):
        batch = images[start : start + args.batch_size].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            scores.append(operators.attention_rollout(model, batch).cpu())
    rollout = torch.cat(scores)  # (n, tokens - 1)
    return rollout


def token_count(model):
    network = model[1]
    tokens = 1 + (network.image_size // network.patch_size) ** 2
    return tokens


def hook_count(model):
    count = sum(
        len(module._forward_pre_hooks)
        + len(module._forward_hooks)
        + ("forward" in module.__dict__)
        for module in model.modules()
    )
    return count


def write_record(record, path):
    temporary = path + ".tmp"
    torch.save(record, temporary)
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
