"""Emit A1 to A4, the adaptive attacker trained against the final method.

The final method reads PSBD-TM (token_mask at `before_attention_norm`) with the
middle-band residual dropout beside it (`pre_residual` dropout in blocks 5 to 8).
Every adaptive attacker on disk trained against PSBD-TM alone or H41's 3-probe
union, so none of them saw the middle band. This batch retrains 4 ViT models at
10% poisoning (GTSRB BadNets, CIFAR-10 Blend, CIFAR-100 BPP and CIFAR-10 WaNet)
with the multi-probe hinge of attacks.evasion against both probes at once
(`--evade-probes before_attention_norm:token_mask pre_residual:dropout:5-8`), then
sweeps PSBD-TM, PSBD-RD and `pre_residual_blocks_5_8` on the result and runs
cli.analyze.

Every other evasion setting is the union attacker's (pbs/generate_union_attacker_jobs.py):
the psu_gap_hinge objective, weight 1, 3 passes, the probe rate calibrated once
to the 0.6 matched shift. So `vit_cifar100_bpp_0_1_evade_final` differs from
`vit_cifar100_bpp_0_1_evade_union` only in the probe set. Each cell's training
arguments are read back from its own args.json, as the union generator does.

Time model. An evasive step keeps 1 base forward and passes forward graphs per
probe alive for 1 backward, 7 graphs here against the union's 10. The union runs
took 915 to 919 minutes for 15 CIFAR-100 epochs at batch 16 with 10 graphs, which
gives the cost per image per retained graph used below. The 1-probe paper
attacker on GTSRB (191 minutes, 4 graphs) agrees within 2%. Batch 24 keeps
24 x 7 = 168 retained image graphs, under the 192 that the batch 48 single-probe
runs (4 graphs) fit on a 40 GB A100, where the union smoke found 32 x 10 = 320 out
of memory.

    PYTHONPATH=. python pbs/generate_adaptive_final_jobs.py --dry-run
    PYTHONPATH=. python pbs/generate_adaptive_final_jobs.py
    bash pbs/adaptive_final_method/submit_all.sh
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pbs.generate_evidence_surplus_jobs import (  # noqa: E402
    BASE,
    DECLARATION,
    JOB_TEMPLATE,
    base_accepts,
    guarded_run_block,
    hours_text,
    load_basis_placements,
    preflight_block,
    readme_text,
    requested_walltime,
    sweep_argv,
    sweep_minutes,
    validate_calls,
    write_batch,
)

BATCH = "adaptive_final_method"
CELLS = (
    ("A1", "vit_gtsrb_badnet_a2o_0_1"),
    ("A2", "vit_cifar10_blend_0_1"),
    ("A3", "vit_cifar100_bpp_0_1"),
    ("A4", "vit_cifar10_wanet_0_1"),
)
FOLDER_SUFFIX = "_evade_final"
EVADE_PROBES = ("before_attention_norm:token_mask", "pre_residual:dropout:5-8")
EVADE_WEIGHT = 1.0
EVADE_PASSES = 3
BATCH_SIZE = 24
SWEEP_PLACEMENT_IDS = (
    "before_attention_norm_token_mask",
    "post_residual",
    "pre_residual_blocks_5_8",
)

# Seconds per training image per retained forward graph, from the union runs:
# (916 - 11) minutes over 50000 images x 15 epochs x 10 graphs, the 11 minutes being
# the smoke's calibration and final evaluation.
SECONDS_PER_IMAGE_GRAPH = (916 - 11) * 60 / (50000 * 15 * 10)
CALIBRATION_EVAL_MINUTES = 11.0
TRAIN_IMAGES = {"gtsrb": 26640, "cifar10": 50000, "cifar100": 50000}


def load_reference(checkpoints_dir: str, folder: str) -> dict:
    """A reference cell's training arguments, an unmarked null seed read as 0."""
    with open(os.path.join(checkpoints_dir, folder, "args.json")) as handle:
        metadata = json.load(handle)
    metadata["folder"] = folder
    if metadata.get("seed") is None:
        metadata["seed"] = 0
    return metadata


def train_argv(metadata: dict, folder: str) -> list[str]:
    """The evasive training command's flags, the reference's recipe otherwise."""
    overrides = metadata.get("attack_config_overrides") or {}
    override_flags = []
    if overrides:
        override_flags = ["--attack-override"] + [
            f"{key}={value}" for key, value in overrides.items()
        ]
    argv = [
        "--dataset",
        metadata["dataset"],
        "--attack",
        metadata["attack"],
        "--poison-rate",
        str(metadata["poison_rate"]),
        "--target-label",
        str(metadata["target_label"]),
        "--architecture",
        metadata["architecture"],
        "--epochs",
        str(metadata["epochs"]),
        "--batch-size",
        str(BATCH_SIZE),
        "--seed",
        str(metadata["seed"]),
        *override_flags,
        "--evade-psbd",
        "--evade-probes",
        *EVADE_PROBES,
        "--evade-weight",
        str(EVADE_WEIGHT),
        "--evade-passes",
        str(EVADE_PASSES),
        "--evade-objective",
        "psu_gap_hinge",
        "--output",
        f"checkpoints/{folder}/attack_result.pt",
    ]
    return argv


def run_calls(metadata: dict, placements: list[dict]) -> list[tuple[str, list[str]]]:
    folder = metadata["folder"] + FOLDER_SUFFIX
    calls = [("cli.train_backdoor", train_argv(metadata, folder))]
    calls += [
        ("cli.sweep", sweep_argv(folder, placement, [])) for placement in placements
    ]
    calls.append(("cli.analyze", ["--checkpoint-folder", folder]))
    return calls


def run_minutes(metadata: dict, placements: list[dict]) -> float:
    """Evasive training at the union-measured cost per image graph, then the sweeps."""
    graphs = 1 + EVADE_PASSES * len(EVADE_PROBES)
    image_epochs = TRAIN_IMAGES[metadata["dataset"]] * metadata["epochs"]
    training = image_epochs * graphs * SECONDS_PER_IMAGE_GRAPH / 60
    minutes = (
        training
        + CALIBRATION_EVAL_MINUTES
        + sweep_minutes(metadata["dataset"], placements)
    )
    return minutes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--out-dir", default=os.path.join(BASE, "pbs", BATCH))
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    return arguments


def main() -> None:
    args = parse_args()
    placements = load_basis_placements(DECLARATION, SWEEP_PLACEMENT_IDS)
    checkpoints_dir = os.path.join(args.base, "checkpoints")

    scripts, rows = {}, []
    for name, reference in CELLS:
        metadata = load_reference(checkpoints_dir, reference)
        calls = run_calls(metadata, placements)
        validate_calls(calls)
        estimate = run_minutes(metadata, placements)
        run = {"id": name, "folder": reference + FOLDER_SUFFIX}
        scripts[name] = JOB_TEMPLATE.format(
            walltime=requested_walltime(estimate),
            name=name,
            base=args.base,
            batch=BATCH,
            work=f"{name} {run['folder']}",
            estimate=estimate,
            preflight=preflight_block(calls),
            body=guarded_run_block(run, calls),
        )
        rows.append(
            {
                "name": name,
                "contents": f"`{run['folder']}` from `{reference}`",
                "estimate": estimate,
                "walltime": requested_walltime(estimate),
                "accepted": base_accepts(calls, args.base),
            }
        )
        print(
            f"{name}  est {hours_text(estimate)}  walltime "
            f"{requested_walltime(estimate)}  accepted now {rows[-1]['accepted']}"
        )

    preamble = (
        "Generated by `pbs/generate_adaptive_final_jobs.py`, which states the time "
        "model. Not submitted. `submit_all.sh` submits the batch by hand. Each job "
        f"trains 1 adaptive attacker at batch {BATCH_SIZE} against "
        f"`{' '.join(EVADE_PROBES)}`, then sweeps "
        f"{', '.join(f'`{placement}`' for placement in SWEEP_PLACEMENT_IDS)} and "
        "runs `cli.analyze`. The last column is the preflight run against the "
        "cluster checkout when this README was written."
    )
    readme = readme_text(rows, "Adaptive attacker against the final method", preamble)
    if args.dry_run:
        print(readme)
        print(scripts["A1"])
        return
    write_batch(args.out_dir, scripts, readme, args.base, BATCH)
    print(f"wrote {len(scripts)} jobs to {args.out_dir}")


if __name__ == "__main__":
    main()
