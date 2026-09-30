"""Emit the evidence-surplus training jobs S1 to S5 (docs/evidence-surplus-theory.md, R1 to R14).

The theory's tier 1 is 15 training runs, R1 to R14 with R12 counting 2 seeds.
Every backdoor run trains with `--telemetry --save-every-epoch`, and the dynamics
runs (R1, R2, R9 and R10) also record the per-sample loss, as the design asks.
Each finished model is then swept for PSBD-TM (`before_attention_norm`
token_mask) and PSBD-RD (`post_residual` dropout) on its final checkpoint at the
basis ladders of configs/psbd_basis.json, and cli.analyze writes its
psbd_metrics.json. R8 is a benign model, trained through cli.train_benign, which
has no telemetry or snapshots, and swept with BadNets as the probe trigger, as
every benign reference is.

The runs are grouped by the code they need, so a job whose code is on the branch
the cluster checkout runs can go today while the others wait for the merge. Each
job starts with a preflight that parses every command it will run against the
checkout's own parsers and stops in seconds when a flag or an attack is missing,
rather than hours into its first run. A run whose checkpoint already exists is not
retrained, so a resubmitted job picks up where the last one stopped, and a run
that fails leaves its log line and lets the next run go.

The folder names follow the canonical template. A tag marks what differs from the
reference cell: `_dyn` a dynamics replicate of an existing cell, `_labelq_0_6` the
trigger label probability and `_smooth_0_1` label smoothing. The conjunction
attacks carry their own attack token. configs/psbd_basis.json excludes every one
of these from the panel.

    PYTHONPATH=. python pbs/generate_evidence_surplus_jobs.py --dry-run
    PYTHONPATH=. python pbs/generate_evidence_surplus_jobs.py
    bash pbs/evidence_surplus_training/submit_all.sh
"""

import argparse
import json
import math
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pbs.generate_union_attacker_jobs import render_call  # noqa: E402

# The checkout the cluster jobs run in, where checkpoints/, results/ and raw_data/
# live. The generator may run from a worktree, the jobs never do.
BASE = "/lustre/home/pstika/projects/PSBD-ViT"
BATCH = "evidence_surplus_training"
DECLARATION = os.path.join(REPO, "configs", "psbd_basis.json")

# PSBD-TM and PSBD-RD, read from the basis so the ladders never drift from the panel.
HEADLINE_PLACEMENT_IDS = ("before_attention_norm_token_mask", "post_residual")
# The probe trigger every benign reference is swept with.
BENIGN_PROBE_ATTACK = "badnet_a2o"

# 15-epoch training minutes on the PBS A100s, the slowest of the "time taken" lines
# in logs/ for each (architecture, dataset): 46, 70 and 90 minutes on GTSRB depending
# on the node, 130 to 167 on CIFAR-10 and 36 to 71 for Swin-S on GTSRB. The slowest
# is taken because a job that runs out of walltime loses its last run whole.
TRAIN_MINUTES = {
    ("vit", "gtsrb"): 90.0,
    ("vit", "cifar10"): 167.0,
    ("swin", "gtsrb"): 72.0,
}
# Telemetry reads logits the update already holds and adds 2 heavy steps and 1
# epoch probe per GTSRB epoch, assumed at 10% of training, not measured on a GPU.
TELEMETRY_FACTOR = 1.10
# --record-sample-loss adds a no-grad forward per batch, about a third of a
# training step's forward and backward.
SAMPLE_LOSS_FACTOR = 1.35
# Per epoch snapshot: a train-accuracy pass over 10000 images and a 330 MB write.
SNAPSHOT_MINUTES_PER_EPOCH = 1.0
EPOCHS = 15
# The final ASR and clean-accuracy pass, as pbs/generate_tact_multisource_jobs.py
# budgets it.
TRAINING_OVERHEAD_MINUTES = 10.0
# pbs/generate_tact_multisource_jobs.py's measured sweep cost at 3 forward passes,
# and its per-call and per-cell overheads.
SWEEP_MINUTES_PER_RATE_PER_1000_INPUTS = 0.021
SWEEP_MINUTES_PER_CALL = 0.5
SWEEP_MINUTES_PER_CELL = 3.0
ANALYZE_MINUTES = 5.0
# validation, clean and backdoor rows per checkpoint (pbs/generate_detector_jobs.py).
PSBD_INPUTS = {"cifar10": 17200, "cifar100": 17915, "gtsrb": 23208, "tiny": 17959}
# The walltime asked of PBS is the estimate plus this share.
WALLTIME_MARGIN = 0.25

RUNS = (
    {
        "id": "R1",
        "folder": "vit_gtsrb_badnet_a2o_0_1_dyn",
        "reference": "vit_gtsrb_badnet_a2o_0_1",
        "dataset": "gtsrb",
        "attack": "badnet_a2o",
        "poison_rate": 0.1,
        "dynamics": True,
        "claim": "training grows surplus after the fit",
    },
    {
        "id": "R2",
        "folder": "vit_gtsrb_badnet_a2o_0_01_dyn",
        "reference": "vit_gtsrb_badnet_a2o_0_01",
        "dataset": "gtsrb",
        "attack": "badnet_a2o",
        "poison_rate": 0.01,
        "dynamics": True,
        "claim": "margin and surplus depend on rho t",
    },
    {
        "id": "R9",
        "folder": "vit_gtsrb_blend_0_1_dyn",
        "reference": "vit_gtsrb_blend_0_1",
        "dataset": "gtsrb",
        "attack": "blend",
        "poison_rate": 0.1,
        "dynamics": True,
        "claim": "the magnitude route grows with training",
    },
    {
        "id": "R10",
        "folder": "vit_cifar10_wanet_0_1_seed_3",
        "reference": "vit_cifar10_wanet_0_1",
        "dataset": "cifar10",
        "attack": "wanet",
        "poison_rate": 0.1,
        "seed": 3,
        "dynamics": True,
        "claim": "relational assembly precedes surplus, and C25's replicate",
    },
    {
        "id": "R3",
        "folder": "vit_gtsrb_badnet_a2o_0_1_labelq_0_6",
        "reference": "vit_gtsrb_badnet_a2o_0_1",
        "dataset": "gtsrb",
        "attack": "badnet_a2o",
        "poison_rate": 0.1,
        "flags": ["--trigger-label-probability", "0.6"],
        "claim": "perfect predictiveness creates surplus",
    },
    {
        "id": "R11",
        "folder": "vit_gtsrb_badnet_a2o_0_1_labelq_0_6_seed_1",
        "reference": "vit_gtsrb_badnet_a2o_0_1",
        "dataset": "gtsrb",
        "attack": "badnet_a2o",
        "poison_rate": 0.1,
        "seed": 1,
        "flags": ["--trigger-label-probability", "0.6"],
        "claim": "replication of R3",
    },
    {
        "id": "R5",
        "folder": "vit_gtsrb_badnet_a2o_0_1_smooth_0_1",
        "reference": "vit_gtsrb_badnet_a2o_0_1",
        "dataset": "gtsrb",
        "attack": "badnet_a2o",
        "poison_rate": 0.1,
        "flags": ["--label-smoothing", "0.1"],
        "claim": "magnitude against renormalization",
    },
    {
        "id": "R13",
        "folder": "vit_gtsrb_badnet_a2o_0_1_smooth_0_1_seed_1",
        "reference": "vit_gtsrb_badnet_a2o_0_1",
        "dataset": "gtsrb",
        "attack": "badnet_a2o",
        "poison_rate": 0.1,
        "seed": 1,
        "flags": ["--label-smoothing", "0.1"],
        "claim": "replication of R5",
    },
    {
        "id": "R8",
        "folder": "vit_gtsrb_benign_smooth_0_1",
        "reference": "vit_gtsrb_benign",
        "dataset": "gtsrb",
        "attack": "benign",
        "poison_rate": 0.0,
        "flags": ["--label-smoothing", "0.1"],
        "claim": "what label smoothing does to clean surplus alone",
    },
    {
        "id": "R4",
        "folder": "vit_gtsrb_and16_0_1",
        "reference": "vit_gtsrb_badnet_a2o_0_1",
        "dataset": "gtsrb",
        "attack": "and16",
        "poison_rate": 0.1,
        "claim": "a positive conjunction without redundancy defeats PSBD-TM",
    },
    {
        "id": "R12a",
        "folder": "vit_gtsrb_and16_0_1_seed_1",
        "reference": "vit_gtsrb_badnet_a2o_0_1",
        "dataset": "gtsrb",
        "attack": "and16",
        "poison_rate": 0.1,
        "seed": 1,
        "claim": "replication of R4",
    },
    {
        "id": "R12b",
        "folder": "vit_gtsrb_and16_0_1_seed_2",
        "reference": "vit_gtsrb_badnet_a2o_0_1",
        "dataset": "gtsrb",
        "attack": "and16",
        "poison_rate": 0.1,
        "seed": 2,
        "claim": "replication of R4",
    },
    {
        "id": "R6",
        "folder": "vit_gtsrb_and2_0_1",
        "reference": "vit_gtsrb_badnet_a2o_0_1",
        "dataset": "gtsrb",
        "attack": "and2",
        "poison_rate": 0.1,
        "claim": "a conjunction of redundant parts keeps surplus",
    },
    {
        "id": "R7",
        "folder": "vit_gtsrb_veto_0_1",
        "reference": "vit_gtsrb_badnet_a2o_0_1",
        "dataset": "gtsrb",
        "attack": "veto",
        "poison_rate": 0.1,
        "claim": "a veto conjunction keeps surplus",
    },
    {
        "id": "R14",
        "folder": "swin_gtsrb_and16_0_1",
        "reference": "swin_gtsrb_badnet_a2o_0_1",
        "architecture": "swin",
        "dataset": "gtsrb",
        "attack": "and16",
        "poison_rate": 0.1,
        "claim": "the conjunction result is not specific to ViT",
    },
)

# Grouped by the code a run needs, so a job never waits on a merge for only some
# of its runs. S1 and S2 need only the telemetry, which rewrite already carries,
# and S2 holds the CIFAR-10 run alone because with any other run it would pass 12
# hours. S3 to S5 need the trigger label probability, label smoothing or the
# conjunction attacks, and hold at most 4 runs so no job asks for more than 12 hours.
JOBS = (
    ("S1", ("R1", "R2", "R9")),
    ("S2", ("R10",)),
    ("S3", ("R3", "R11", "R5")),
    ("S4", ("R4", "R12a", "R12b", "R8")),
    ("S5", ("R6", "R7", "R14", "R13")),
)

JOB_TEMPLATE = """#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb
#PBS -l walltime={walltime}
#PBS -N {name}
#PBS -o {base}/logs/{batch}/{name}.log
#PBS -j oe

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

echo "Job ID:  $PBS_JOBID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "Work:    {work}, est {estimate:.0f} min"
nvidia-smi --query-gpu=name --format=csv,noheader

cd {base}
source .venv/bin/activate
echo "Commit:  $(git rev-parse HEAD), dirty files: $(git status --porcelain | wc -l)"

{preflight}
{body}
echo "Finished: $(date)"
exit 0
"""

# Parses every command the job runs with the checkout's own parsers, and for an
# evasion run checks that each probe token resolves to a (position, operator,
# block range) triple, since an older parser reads `pre_residual:dropout:5-8` as
# the operator `dropout:5-8` and would fail only once training starts.
PREFLIGHT_SCRIPT = """import importlib
import json
import sys

calls = json.loads('''{calls}''')
for module_name, argv in calls:
    module = importlib.import_module(module_name)
    sys.argv = [module_name] + argv
    arguments = module.parse_args()
    if module_name == "cli.train_backdoor" and arguments.evade_psbd:
        tokens = module.parse_evade_probe_tokens(arguments)
        assert all(len(token) == 3 for token in tokens), tokens
print(f"preflight ok, {{len(calls)}} commands parse")
"""
PREFLIGHT_HEREDOC = """python - <<'PREFLIGHT' || {{ echo "preflight failed, this checkout cannot run the job"; exit 1; }}
{script}PREFLIGHT
"""


def load_basis_placements(declaration_path: str, placement_ids: tuple) -> list[dict]:
    """The named basis entries, in the order given."""
    with open(declaration_path) as handle:
        basis = json.load(handle)["basis"]
    by_id = {entry["id"]: entry for entry in basis}
    placements = [by_id[placement_id] for placement_id in placement_ids]
    return placements


def train_argv(run: dict) -> tuple[str, list[str]]:
    """The training module and its flags for 1 run."""
    architecture = run.get("architecture", "vit")
    output = f"checkpoints/{run['folder']}/attack_result.pt"
    seed = str(run.get("seed", 0))
    if run["attack"] == "benign":
        argv = [
            "--datasets",
            run["dataset"],
            "--architecture",
            architecture,
            "--epochs",
            str(EPOCHS),
            "--seed",
            seed,
            *run.get("flags", []),
            "--output",
            output,
        ]
        return "cli.train_benign", argv

    argv = [
        "--dataset",
        run["dataset"],
        "--attack",
        run["attack"],
        "--poison-rate",
        str(run["poison_rate"]),
        "--target-label",
        "0",
        "--architecture",
        architecture,
        "--epochs",
        str(EPOCHS),
        "--seed",
        seed,
        "--telemetry",
        "--save-every-epoch",
        *(["--record-sample-loss"] if run.get("dynamics") else []),
        *run.get("flags", []),
        "--output",
        output,
    ]
    return "cli.train_backdoor", argv


def probe_flags(run: dict) -> list[str]:
    """The probe trigger a benign model is swept with, none otherwise."""
    if run["attack"] != "benign":
        return []
    flags = ["--probe-attack", BENIGN_PROBE_ATTACK, "--probe-target-label", "0"]
    return flags


def sweep_argv(folder: str, placement: dict, extra: list[str]) -> list[str]:
    """1 cli.sweep call for 1 basis placement, at the panel's 3 forward passes."""
    argv = ["--checkpoint-folder", folder, "--position", placement["position"]]
    if placement["block_range"]:
        first, last = placement["block_range"]
        argv += ["--block-range", str(first), str(last)]
    argv += [
        "--operator",
        placement["operator"],
        "--rates",
        *[str(rate) for rate in placement["rates"]],
        "--forward-passes",
        "3",
        "--skip-existing",
        *extra,
    ]
    return argv


def run_calls(run: dict, placements: list[dict]) -> list[tuple[str, list[str]]]:
    """Every (module, argv) the run issues, training first."""
    extra = probe_flags(run)
    calls = [train_argv(run)]
    calls += [
        ("cli.sweep", sweep_argv(run["folder"], placement, extra))
        for placement in placements
    ]
    # cli.analyze reads the probe trigger off the cache, so it takes no probe flag.
    calls.append(("cli.analyze", ["--checkpoint-folder", run["folder"]]))
    return calls


def sweep_minutes(dataset: str, placements: list[dict]) -> float:
    """Sweeping and analyzing 1 model at the given placements."""
    inputs_in_thousands = PSBD_INPUTS[dataset] / 1000
    rates = sum(len(placement["rates"]) for placement in placements)
    minutes = (
        SWEEP_MINUTES_PER_CELL
        + ANALYZE_MINUTES
        + len(placements) * SWEEP_MINUTES_PER_CALL
        + rates * inputs_in_thousands * SWEEP_MINUTES_PER_RATE_PER_1000_INPUTS
    )
    return minutes


def run_minutes(run: dict, placements: list[dict]) -> float:
    """1 run's training, snapshots, final evaluation, sweeps and analysis."""
    architecture = run.get("architecture", "vit")
    training = TRAIN_MINUTES[(architecture, run["dataset"])]
    if run["attack"] != "benign":
        training *= TELEMETRY_FACTOR
        training += SNAPSHOT_MINUTES_PER_EPOCH * EPOCHS
    if run.get("dynamics"):
        training *= SAMPLE_LOSS_FACTOR

    minutes = (
        training + TRAINING_OVERHEAD_MINUTES + sweep_minutes(run["dataset"], placements)
    )
    return minutes


def requested_walltime(estimate_minutes: float) -> str:
    """The estimate plus the margin, rounded up to a quarter hour, as HH:MM:00."""
    padded = estimate_minutes * (1 + WALLTIME_MARGIN)
    quarters = math.ceil(padded / 15)
    hours, minutes = divmod(quarters * 15, 60)
    walltime = f"{hours:02d}:{minutes:02d}:00"
    return walltime


def validate_calls(calls: list[tuple[str, list[str]]]) -> None:
    """Every call parsed by this checkout's own parser, before any file is written."""
    import importlib

    for module_name, argv in calls:
        module = importlib.import_module(module_name)
        saved_argv = sys.argv
        sys.argv = [module_name] + argv
        try:
            arguments = module.parse_args()
            if module_name == "cli.train_backdoor" and arguments.evade_psbd:
                tokens = module.parse_evade_probe_tokens(arguments)
                assert all(len(token) == 3 for token in tokens), tokens
        except SystemExit as error:
            raise ValueError(
                f"{module_name} rejects {' '.join(argv)} (exit {error.code})"
            ) from error
        finally:
            sys.argv = saved_argv


def base_accepts(calls: list[tuple[str, list[str]]], base: str) -> bool:
    """Whether the cluster checkout's current code parses every call, read-only."""
    script = preflight_script(calls)
    completed = subprocess.run(
        [os.path.join(base, ".venv", "bin", "python"), "-"],
        input=script,
        cwd=base,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": base, "CUDA_VISIBLE_DEVICES": ""},
    )
    accepted = completed.returncode == 0
    return accepted


def preflight_script(calls: list[tuple[str, list[str]]]) -> str:
    """The Python source that parses every call with the running checkout's parsers."""
    encoded = json.dumps([[module, argv] for module, argv in calls])
    script = PREFLIGHT_SCRIPT.format(calls=encoded)
    return script


def preflight_block(calls: list[tuple[str, list[str]]]) -> str:
    """The job's preflight heredoc for these calls."""
    block = PREFLIGHT_HEREDOC.format(script=preflight_script(calls))
    return block


def guarded_run_block(run: dict, calls: list[tuple[str, list[str]]]) -> str:
    """1 run's commands: train unless already trained, then sweep and analyze if trained."""
    folder = run["folder"]
    checkpoint = f"checkpoints/{folder}/attack_result.pt"
    train_module, train_arguments = calls[0]
    lines = [
        f'echo "=== {run["id"]} {folder} $(date) ==="',
        f"if [ -f {checkpoint} ]; then",
        f'  echo "{folder} is trained already, not retrained"',
        "else",
        render_call(train_module, train_arguments).rstrip("\n")
        + f' || echo "[FAILED rc=$?] train {folder}"',
        "fi",
        f"if [ -f {checkpoint} ]; then",
    ]
    for module_name, argv in calls[1:]:
        lines.append(
            render_call(module_name, argv).rstrip("\n")
            + f' || echo "[FAILED rc=$?] {module_name} {folder}"'
        )
    lines += ["else", f'  echo "{folder} has no checkpoint, nothing swept"', "fi", ""]
    block = "\n".join(lines)
    return block


def render_job(
    name: str, runs: list[dict], placements: list[dict], base: str, batch: str
) -> tuple[str, float]:
    """1 job script and its estimate in minutes."""
    calls_per_run = [run_calls(run, placements) for run in runs]
    estimate = sum(run_minutes(run, placements) for run in runs)
    all_calls = [call for calls in calls_per_run for call in calls]
    body = "\n".join(
        guarded_run_block(run, calls) for run, calls in zip(runs, calls_per_run)
    )
    script = JOB_TEMPLATE.format(
        walltime=requested_walltime(estimate),
        name=name,
        base=base,
        batch=batch,
        work=" ".join(f"{run['id']} {run['folder']}" for run in runs),
        estimate=estimate,
        preflight=preflight_block(all_calls),
        body=body,
    )
    return script, estimate


def hours_text(minutes: float) -> str:
    text = f"{minutes / 60:.1f} h"
    return text


def readme_text(rows: list[dict], title: str, preamble: str) -> str:
    """The batch README: 1 table row per job, every number from the generator."""
    lines = [
        f"# {title}",
        "",
        preamble,
        "",
        "| job | contents | estimated walltime | requested walltime | cluster checkout accepts it now |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        accepted = "yes" if row["accepted"] else "no, needs the new code merged"
        lines.append(
            f"| {row['name']} | {row['contents']} | {hours_text(row['estimate'])} | "
            f"{row['walltime']} | {accepted} |"
        )
    lines.append("")
    text = "\n".join(lines)
    return text


def write_batch(
    out_dir: str, scripts: dict[str, str], readme: str, base: str, batch: str
) -> None:
    """The job files, submit_all.sh (never run here) and the README, plus the log folder."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(base, "logs", batch), exist_ok=True)
    for name, script in scripts.items():
        with open(os.path.join(out_dir, f"{name}.pbs"), "w") as handle:
            handle.write(script)

    submit_lines = [
        "#!/bin/bash",
        "# Submits every job of this batch. Run by hand only.",
    ]
    submit_lines += [f"qsub {os.path.join(out_dir, name)}.pbs" for name in scripts]
    with open(os.path.join(out_dir, "submit_all.sh"), "w") as handle:
        handle.write("\n".join(submit_lines) + "\n")
    with open(os.path.join(out_dir, "README.md"), "w") as handle:
        handle.write(readme)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--out-dir", default=os.path.join(BASE, "pbs", BATCH))
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    return arguments


def main() -> None:
    args = parse_args()
    placements = load_basis_placements(DECLARATION, HEADLINE_PLACEMENT_IDS)
    runs_by_id = {run["id"]: run for run in RUNS}
    assert sorted(runs_by_id) == sorted(
        run_id for _, run_ids in JOBS for run_id in run_ids
    ), "every run sits in exactly 1 job"

    scripts, rows = {}, []
    for name, run_ids in JOBS:
        runs = [runs_by_id[run_id] for run_id in run_ids]
        calls = [call for run in runs for call in run_calls(run, placements)]
        validate_calls(calls)
        script, estimate = render_job(name, runs, placements, args.base, BATCH)
        scripts[name] = script
        rows.append(
            {
                "name": name,
                "contents": ", ".join(f"{run['id']} `{run['folder']}`" for run in runs),
                "estimate": estimate,
                "walltime": requested_walltime(estimate),
                "accepted": base_accepts(calls, args.base),
            }
        )
        print(
            f"{name}  est {hours_text(estimate)}  walltime {requested_walltime(estimate)}  "
            f"accepted now {rows[-1]['accepted']}  {run_ids}"
        )

    preamble = (
        "Generated by `pbs/generate_evidence_surplus_jobs.py`, which states the time "
        "model and every run's configuration. Not submitted. `submit_all.sh` submits "
        "the batch by hand. Each backdoor run trains with `--telemetry "
        "--save-every-epoch` (the dynamics runs R1, R2, R9 and R10 also with "
        "`--record-sample-loss`), then sweeps PSBD-TM and PSBD-RD on its final "
        "checkpoint and runs `cli.analyze`. The last column is the preflight run "
        "against the cluster checkout when this README was written."
    )
    readme = readme_text(rows, "Evidence-surplus training jobs", preamble)
    if args.dry_run:
        print(readme)
        return
    write_batch(args.out_dir, scripts, readme, args.base, BATCH)
    print(
        f"wrote {len(scripts)} jobs to {args.out_dir}, submit with bash {args.out_dir}/submit_all.sh"
    )


if __name__ == "__main__":
    main()
