"""Retrain TaCT with several source classes, then give each model that clears the panel's coverage.

8 of the 11 TaCT panel models are not trigger backdoors. With 1 source class, a
poison rate at that class's share of the training set poisons every source image,
and the model maps the whole class to the target with no trigger (the ledger marks
these cells source_mapped). Tang et al. poison 2% of CIFAR-10 on a source class
holding 10%, so a fifth of the source pool. Each run here takes source classes
1..k, with k the smallest count whose summed training share is at least 5 times
the poison rate, read from the training split itself. The folder gains a _src{k}
tag, and every other argument is the TaCT panel recipe.

The multi-source runs are a hypothesis, so nothing is retried or tuned. A run that
diverges fails its training job, and the afterok dependency then cancels its
downstream job. A run that trains but does not clear, or that turns out source
mapped anyway, stops at the downstream job's gate and spends no sweep time. The
gate reads the same verdicts the coverage ledger writes.

Each downstream job sweeps the full basis of configs/psbd_basis.json minus what is
already cached, runs cli.analyze and scores the 11 competitor detectors, the
coverage every other panel cell carries. 2 existing panel cells that clear the bar
but were never swept get the same downstream job with no dependency.

Planning is CPU only and builds, for every run, the poison and cover draw and the
PSBD backdoor split that training and the sweep will build. It refuses to write a
job whose source pool share, ASR set or command line is wrong.

    PYTHONPATH=. python pbs/generate_tact_multisource_jobs.py --dry-run
    PYTHONPATH=. python pbs/generate_tact_multisource_jobs.py
    bash pbs/vit_tact_multisource/submit_all.sh
    python -m pbs.generate_tact_multisource_jobs --gate vit_cifar100_tact_0_01_src5
"""

import argparse
import json
import math
import os
import shlex
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from attacks import apply_config_overrides, build_attack, default_config  # noqa: E402
from attacks.poisoning import AttackSuccessSet, choose_indices_with_cover  # noqa: E402
from cli.baselines import build_parser as baselines_parser  # noqa: E402
from cli.sweep import build_parser as sweep_parser  # noqa: E402
from cli.train_backdoor import build_parser as train_parser  # noqa: E402
from cli.train_backdoor import parse_attack_overrides  # noqa: E402
from data.loading import base_image_transform, extract_labels  # noqa: E402
from data.loading import load_clean_datasets  # noqa: E402
from data.registry import DATASET_REGISTRY  # noqa: E402
from data.splits import PSBD_HELDOUT_SIZE, psbd_split_permutation  # noqa: E402
from defenses.decision import RECOMMENDED_PLACEMENT  # noqa: E402
from detectors.records import scored_detectors  # noqa: E402
from pbs.generate_basis_jobs import cell_workload, invocation_lines  # noqa: E402
from pbs.generate_detector_jobs import DETECTOR_GROUPS  # noqa: E402
from pbs.generate_detector_jobs import estimated_minutes as detector_minutes  # noqa: E402
from pbs.generate_detector_jobs import render_command as detector_command  # noqa: E402
from pbs.generate_seed_jobs import MEDIAN_MINUTES  # noqa: E402
from scripts.coverage_ledger import (  # noqa: E402
    DIVERGENCE_FRACTION,
    SOURCE_MAPPED_ACCURACY,
    benign_reference_accuracy,
    cached_rates,
    gaps_for_cell,
    placement_rows,
    read_metadata,
    source_class_accuracy,
)

RUNS = (
    ("cifar100", 0.01),
    ("cifar100", 0.05),
    ("tiny", 0.01),
    ("tiny", 0.05),
    ("cifar10", 0.1),
    ("gtsrb", 0.1),
)
UNSWEPT_PANEL_CELLS = ("vit_gtsrb_tact_0_01_cos", "vit_gtsrb_lc_0_05_tl1_adv")

# Tang et al. poison 2% of CIFAR-10 on a source class holding 10% of it.
POISON_SHARE_OF_SOURCE_POOL = 0.2
# The draw is exact up to rounding, and GTSRB's unequal classes land k where the
# pool overshoots 5 times the rate a little, so the realized share sits just under.
SOURCE_SHARE_TOLERANCE = 0.02

# The TaCT panel recipe, read off checkpoints/vit_cifar10_tact_0_05/args.json.
ATTACK = "tact"
ARCHITECTURE = "vit"
TARGET_LABEL = 0
EPOCHS = 15
SEED = 0

BATCH = "vit_tact_multisource"
DECLARATION = os.path.join(REPO, "configs", "psbd_basis.json")

# The PBS exit code the gate reserves for "does not clear", so a gate that
# crashes (exit 1) fails its job loudly rather than reading as a verdict.
GATE_REFUSES = 3

# Training minutes are MEDIAN_MINUTES, measured from the trained_started_at to
# trained_ended_at span of every plain Adam ViT sidecar. That span stops before
# train_backdoor's own final ASR pass and before cli.evaluate's stealth pass.
TRAINING_OVERHEAD_MINUTES = 10.0
# Sweep cost per basis rate per 1000 PSBD inputs, with 3 forward passes. Measured
# from the cache mtimes of complete basis cells on 2026-09-24: 0.0211 on
# vit_cifar100_badnet_a2o_0_05 (17.9k inputs, 108 min), 0.0206 on
# vit_cifar100_tact_0_05 (10.1k inputs, 59 min) and 0.0210 on
# vit_gtsrb_badnet_a2o_0_05 (23.2k inputs, 139 min).
SWEEP_MINUTES_PER_RATE_PER_1000_INPUTS = 0.021
# A model load per sweep call and the baseline build, as pbs/generate_basis_jobs.py
# budgets them.
SWEEP_MINUTES_PER_CALL = 0.5
SWEEP_MINUTES_PER_CELL = 3.0
ANALYZE_MINUTES = 5.0
WALLTIME_MARGIN = 2.0
MIN_TRAIN_WALLTIME_HOURS = 4
MIN_DOWNSTREAM_WALLTIME_HOURS = 6

TEMPLATE = """#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb
#PBS -l walltime={walltime}
#PBS -N {name}
#PBS -o {root}/logs/{batch}/{name}.log
#PBS -j oe

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

echo "Job ID:  $PBS_JOBID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "Work:    {work}, est {estimate} min"
nvidia-smi --query-gpu=name --format=csv,noheader

cd {root}
source .venv/bin/activate
echo "Commit:  $(git rev-parse HEAD), dirty files: $(git status --porcelain | wc -l)"

{body}
echo "Finished: $(date)"
exit 0
"""

# A training failure must leave a nonzero exit, or afterok would release the
# downstream job onto a checkpoint that does not exist.
TRAIN_BODY = """python -m cli.train_backdoor \\
    --dataset {dataset} \\
    --attack {attack} \\
    --poison-rate {poison_rate} \\
    --target-label {target_label} \\
    --architecture {architecture} \\
    --epochs {epochs} \\
    --seed {seed} \\
    --attack-override source_classes={source_classes} \\
    --output checkpoints/{folder}/attack_result.pt || exit 1
python -m cli.evaluate --folder {folder} || exit 1
"""

GATE = """python -m pbs.generate_tact_multisource_jobs --gate {folder}
verdict=$?
if [ $verdict -eq {refuses} ]; then echo "gate: {folder} {stage}, nothing further runs"; exit 0; fi
if [ $verdict -ne 0 ]; then echo "gate crashed with $verdict"; exit 1; fi
"""


def training_labels(dataset: str, raw_data_dir: str) -> list[int]:
    """The labels of the full training split, the pool the poison draw reads."""
    image_size = DATASET_REGISTRY[dataset].image_size
    train, _ = load_clean_datasets(
        dataset, base_image_transform(image_size), raw_data_dir
    )
    labels = extract_labels(train)
    return labels


def test_labels(dataset: str, raw_data_dir: str) -> list[int]:
    """The labels of the full test split in native order, the pool PSBD splits."""
    image_size = DATASET_REGISTRY[dataset].image_size
    _, test = load_clean_datasets(
        dataset, base_image_transform(image_size), raw_data_dir
    )
    labels = extract_labels(test)
    return labels


def source_class_count(labels: list[int], poison_rate: float, target_label: int) -> int:
    """The smallest k whose classes target+1..target+k hold 1/share times the poison.

    Counted on the training split, since that is the pool the poison is drawn from.
    """
    needed = poison_rate * len(labels) / POISON_SHARE_OF_SOURCE_POOL
    counts = [labels.count(label) for label in range(max(labels) + 1)]
    pooled = 0
    for k, label in enumerate(range(target_label + 1, len(counts)), start=1):
        pooled += counts[label]
        if pooled >= needed - 1e-9:
            return k
    raise ValueError(f"no source pool reaches {needed:.0f} images")


def rate_tag(rate: float) -> str:
    tag = f"{rate:g}".replace(".", "_")
    return tag


def run_folder(dataset: str, poison_rate: float, k: int) -> str:
    folder = f"{ARCHITECTURE}_{dataset}_{ATTACK}_{rate_tag(poison_rate)}_src{k}"
    return folder


def source_override(k: int) -> str:
    """The --attack-override value naming classes 1..k, as the command line reads it."""
    override = ",".join(str(TARGET_LABEL + offset) for offset in range(1, k + 1))
    return override


def plan_run(dataset: str, poison_rate: float, raw_data_dir: str) -> dict:
    """1 multi-source run: k, its folder and what its training and eval sets hold.

    The config is built through the same override parser train_backdoor uses, so a
    comma string that did not reach TaCT as a tuple fails here rather than on a GPU.
    """
    labels = training_labels(dataset, raw_data_dir)
    k = source_class_count(labels, poison_rate, TARGET_LABEL)
    overrides = parse_attack_overrides([f"source_classes={source_override(k)}"])
    config = apply_config_overrides(default_config(ATTACK), overrides)
    attack = build_attack(
        ATTACK, config, DATASET_REGISTRY[dataset].image_size, TARGET_LABEL
    )

    poison, cover = choose_indices_with_cover(
        labels, attack, poison_rate, config.cover_rate, config.source_classes, SEED
    )
    sources = set(config.source_classes)
    source_pool = sum(1 for label in labels if label in sources)
    poisoned_labels = [labels[index] for index in poison]
    fully_poisoned = [
        source
        for source in sorted(sources)
        if poisoned_labels.count(source) == labels.count(source)
    ]

    split = psbd_split_inputs(dataset, ATTACK, overrides, TARGET_LABEL, raw_data_dir)
    run = {
        "folder": run_folder(dataset, poison_rate, k),
        "dataset": dataset,
        "attack": ATTACK,
        "poison_rate": poison_rate,
        "k": k,
        "source_classes": source_override(k),
        "config_sources": config.source_classes,
        "train_size": len(labels),
        "source_training_share": source_pool / len(labels),
        "n_poisoned": len(poison),
        "n_cover": len(cover),
        "poison_share_of_source_pool": len(poison) / source_pool,
        "poison_outside_sources": sum(1 for i in poison if labels[i] not in sources),
        "cover_inside_sources": sum(
            1 for i in cover if labels[i] in sources or labels[i] == TARGET_LABEL
        ),
        "fully_poisoned_sources": fully_poisoned,
        **split,
    }
    return run


def psbd_split_inputs(
    dataset: str,
    attack_name: str,
    overrides: dict | None,
    target_label: int,
    raw_data_dir: str,
) -> dict:
    """The PSBD split sizes and backdoor classes a sweep of this model will score.

    Rebuilt with data.splits' own permutation and AttackSuccessSet, without
    decoding an image, so the sweep and detector estimates read the real row
    counts rather than the all-to-one table.
    """
    spec = DATASET_REGISTRY[dataset]
    config = apply_config_overrides(default_config(attack_name), overrides)
    attack = build_attack(attack_name, config, spec.image_size, target_label)
    labels = test_labels(dataset, raw_data_dir)

    permutation = psbd_split_permutation(len(labels))  # (n_test,)
    analysis_indices = permutation[PSBD_HELDOUT_SIZE:].tolist()
    analysis_labels = [labels[index] for index in analysis_indices]
    backdoor = AttackSuccessSet(None, analysis_labels, attack, None, spec.num_classes)

    split = {
        "n_validation": PSBD_HELDOUT_SIZE,
        "n_clean": len(analysis_indices),
        "n_backdoor": len(backdoor.indices),
        "backdoor_classes": sorted({analysis_labels[i] for i in backdoor.indices}),
        "inputs": PSBD_HELDOUT_SIZE + len(analysis_indices) + len(backdoor.indices),
    }
    return split


def plan_existing(folder: str, checkpoints_dir: str, raw_data_dir: str) -> dict:
    """An already trained panel cell, sized from its own sidecar."""
    metadata = read_metadata(checkpoints_dir, folder)
    if metadata is None:
        raise SystemExit(f"{folder} has no args.json")
    split = psbd_split_inputs(
        metadata["dataset"],
        metadata["attack"],
        metadata.get("attack_config_overrides"),
        metadata["target_label"],
        raw_data_dir,
    )
    cell = {
        "folder": folder,
        "dataset": metadata["dataset"],
        "attack": metadata["attack"],
        "asr": metadata.get("asr"),
        **split,
    }
    return cell


def check_run(run: dict) -> list[str]:
    """Every way the planned run differs from the multi-source design."""
    problems = []
    expected = tuple(range(TARGET_LABEL + 1, TARGET_LABEL + run["k"] + 1))
    if run["config_sources"] != expected:
        problems.append(f"source classes reached TaCT as {run['config_sources']!r}")
    if abs(run["poison_share_of_source_pool"] - POISON_SHARE_OF_SOURCE_POOL) > (
        SOURCE_SHARE_TOLERANCE
    ):
        problems.append(
            f"poison is {run['poison_share_of_source_pool']:.3f} of the source pool"
        )
    if run["poison_outside_sources"] or run["cover_inside_sources"]:
        problems.append("poison or cover drawn from the wrong classes")
    if run["fully_poisoned_sources"]:
        problems.append(f"sources fully poisoned: {run['fully_poisoned_sources']}")
    if run["backdoor_classes"] != list(expected):
        problems.append(f"ASR set holds classes {run['backdoor_classes']}")
    return problems


def cell_gaps(folder: str, declaration: dict, checkpoints_dir: str, results_dir: str):
    """The basis rates this cell lacks, the ledger's own gap rows.

    A cell with no cache yet (every new run) gets the whole basis.
    """
    panel = declaration["panel"]
    rows = placement_rows(checkpoints_dir, results_dir, folder)
    cached = cached_rates(rows, panel["mask_seed"])
    gaps = gaps_for_cell({"folder_name": folder}, cached, declaration["basis"], panel)
    return gaps


def sweep_minutes(gaps: list[dict], inputs: int) -> float:
    rate_units = sum(len(gap["missing_rates"]) for gap in gaps)
    minutes = (
        SWEEP_MINUTES_PER_CELL
        + len(gaps) * SWEEP_MINUTES_PER_CALL
        + rate_units * SWEEP_MINUTES_PER_RATE_PER_1000_INPUTS * inputs / 1000
    )
    return minutes


def pending_detectors(folder: str, results_dir: str) -> dict[str, list[str]]:
    """Per detector group, the detectors this cell has no scored record for."""
    done = scored_detectors(results_dir, folder)
    pending = {
        group: [name for name in names if name not in done]
        for group, names in DETECTOR_GROUPS.items()
    }
    pending = {group: names for group, names in pending.items() if names}
    return pending


def downstream_body(cell: dict, gaps: list[dict], detectors: dict, args) -> str:
    """Gate, recommended placement, gate again, the rest of the basis, analyze, detectors.

    The recommended placement runs first because its first call builds the clean
    baseline, and the second gate reads source-class accuracy off that baseline.
    A source-mapped model then stops after 1 placement instead of 27.
    """
    folder = cell["folder"]
    first = [gap for gap in gaps if gap["placement"] == RECOMMENDED_PLACEMENT]
    rest = [gap for gap in gaps if gap["placement"] != RECOMMENDED_PLACEMENT]

    parts = [GATE.format(folder=folder, refuses=GATE_REFUSES, stage="does not clear")]
    if first:
        parts.append(invocation_lines([folder], cell_workload(first)))
        parts.append(
            GATE.format(folder=folder, refuses=GATE_REFUSES, stage="is source mapped")
        )
    if rest:
        parts.append(invocation_lines([folder], cell_workload(rest)))
    parts.append(f"python -m cli.analyze --checkpoint-folder {folder}\n")

    detector_cell = {
        "folder": folder,
        "dataset": cell["dataset"],
        "attack": cell["attack"],
    }
    for group, names in detectors.items():
        parts.append(detector_command(detector_cell, names, group, args))
    body = "\n".join(parts)
    return body


def train_body(run: dict) -> str:
    body = TRAIN_BODY.format(
        dataset=run["dataset"],
        attack=ATTACK,
        poison_rate=run["poison_rate"],
        target_label=TARGET_LABEL,
        architecture=ARCHITECTURE,
        epochs=EPOCHS,
        seed=SEED,
        source_classes=run["source_classes"],
        folder=run["folder"],
    )
    return body


def train_minutes(run: dict) -> float:
    minutes = MEDIAN_MINUTES[(ARCHITECTURE, run["dataset"])] + TRAINING_OVERHEAD_MINUTES
    return minutes


def downstream_minutes(cell: dict, gaps: list[dict], detectors: dict) -> float:
    detector_total = sum(
        detector_minutes(cell, names, group) for group, names in detectors.items()
    )
    minutes = sweep_minutes(gaps, cell["inputs"]) + ANALYZE_MINUTES + detector_total
    return minutes


def walltime(minutes: float, floor_hours: int) -> str:
    hours = max(floor_hours, math.ceil(WALLTIME_MARGIN * minutes / 60))
    text = f"{hours:02d}:00:00"
    return text


def command_blocks(script: str, module: str) -> list[list[str]]:
    """The argument tokens of every `python -m <module>` call in a script."""
    joined = script.replace("\\\n", " ")
    blocks = []
    for line in joined.split("\n"):
        stripped = line.strip()
        prefix = f"python -m {module} "
        if not stripped.startswith(prefix):
            continue
        command = stripped.split("||")[0]
        blocks.append(shlex.split(command)[3:])
    return blocks


def verify_script(script: str) -> None:
    """Parse every emitted command with its own CLI's parser, refusing on any error.

    argparse rejects an unknown flag or a position outside its choices by exiting,
    which this turns into a message naming the command.
    """
    parsers = {
        "cli.train_backdoor": train_parser,
        "cli.sweep": sweep_parser,
        "cli.baselines": baselines_parser,
    }
    for module, build in parsers.items():
        for tokens in command_blocks(script, module):
            try:
                build().parse_args(tokens)
            except SystemExit as error:
                raise SystemExit(f"{module} rejects {' '.join(tokens)}") from error


def clearing_verdict(
    folder: str, checkpoints_dir: str, results_dir: str, declaration: dict
) -> str | None:
    """Why this model is not a panel backdoor, or None when it is one.

    The ledger's 3 verdicts, in its order: the ASR bar, divergence against the
    benign reference, and source mapping once a clean baseline exists to read it.
    """
    metadata = read_metadata(checkpoints_dir, folder)
    if metadata is None:
        # afterok only releases this job once training saved, so a missing sidecar
        # is a broken run rather than a verdict, and the job should fail loudly.
        raise FileNotFoundError(f"{folder} has no args.json")
    # The sidecar's numbers, as the ledger reads them. metrics.json is not read:
    # the TaCT ones on disk predate the source-class restriction and score 0.13
    # where the model's own ASR is 1.00.
    asr = metadata.get("asr")
    accuracy = metadata.get("clean_accuracy")

    if asr is None or asr < declaration["asr_bar"]:
        return f"ASR {asr} below the bar {declaration['asr_bar']}"

    references = benign_reference_accuracy(
        checkpoints_dir, results_dir, declaration["benign_reference"]
    )
    reference = references.get(metadata["dataset"])
    if accuracy is not None and reference is not None:
        if accuracy < DIVERGENCE_FRACTION * reference:
            return f"clean accuracy {accuracy:.3f} diverged against {reference:.3f}"

    cell = {"folder_name": folder, "attack": metadata["attack"]}
    source_accuracy = source_class_accuracy(checkpoints_dir, results_dir, cell)
    if source_accuracy is not None and source_accuracy < SOURCE_MAPPED_ACCURACY:
        return f"clean source-class accuracy {source_accuracy:.3f}, source mapped"
    return None


def run_gate(folder: str, args) -> int:
    with open(args.declaration) as handle:
        declaration = json.load(handle)
    verdict = clearing_verdict(
        folder, args.checkpoints_dir, args.results_dir, declaration
    )
    if verdict is not None:
        print(f"gate refuses {folder}: {verdict}")
        return GATE_REFUSES
    print(f"gate passes {folder}")
    return 0


def print_run(run: dict) -> None:
    print(
        f"  {run['folder']:32s} k={run['k']:2d} source share "
        f"{run['source_training_share']:.4f} poisoned {run['n_poisoned']} "
        f"({run['poison_share_of_source_pool']:.3f} of pool) cover {run['n_cover']} "
        f"ASR set {run['n_backdoor']} over {len(run['backdoor_classes'])} classes"
    )


def write_file(path: str, text: str) -> None:
    with open(path, "w") as handle:
        handle.write(text)


def submit_script(train_jobs: list[tuple], downstream_jobs: list[tuple]) -> str:
    """Training first, each downstream job held on its own training job by afterok."""
    lines = [
        "#!/bin/bash",
        "# A downstream job waits on its own training job only, so 1 failed run",
        "# cancels 1 downstream job and nothing else.",
        "set -e",
        f"cd {REPO}",
        f"mkdir -p logs/{BATCH}",
    ]
    for name, path in train_jobs:
        variable = name.upper()
        lines.append(f"{variable}=$(qsub {path})")
        lines.append(f'echo "{name} $' + "{" + variable + "}" + '"')
    for name, path, depends_on in downstream_jobs:
        if depends_on is None:
            lines.append(f'echo "{name} $(qsub {path})"')
            continue
        variable = depends_on.upper()
        lines.append(f'echo "{name} $(qsub -W depend=afterok:${{{variable}}} {path})"')
    script = "\n".join(lines) + "\n"
    return script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", default=os.path.join(REPO, "checkpoints"))
    parser.add_argument("--results-dir", default=os.path.join(REPO, "results"))
    parser.add_argument("--raw-data-dir", default=os.path.join(REPO, "raw_data"))
    parser.add_argument("--declaration", default=DECLARATION)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--gate",
        default=None,
        metavar="FOLDER",
        help=f"inside a job: exit {GATE_REFUSES} unless FOLDER is a panel backdoor",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gate:
        sys.exit(run_gate(args.gate, args))

    with open(args.declaration) as handle:
        declaration = json.load(handle)

    runs = [plan_run(dataset, rate, args.raw_data_dir) for dataset, rate in RUNS]
    existing = [
        plan_existing(folder, args.checkpoints_dir, args.raw_data_dir)
        for folder in UNSWEPT_PANEL_CELLS
    ]

    print(f"{len(runs)} multi-source TaCT runs")
    failures = []
    for run in runs:
        print_run(run)
        failures += [f"{run['folder']}: {problem}" for problem in check_run(run)]
        if os.path.exists(os.path.join(args.checkpoints_dir, run["folder"])):
            failures.append(f"{run['folder']}: checkpoint folder already exists")
    for cell in existing:
        if cell["asr"] is None or cell["asr"] < declaration["asr_bar"]:
            failures.append(f"{cell['folder']}: ASR {cell['asr']} below the bar")
    if failures:
        raise SystemExit("refusing to write jobs:\n  " + "\n  ".join(failures))

    out_dir = os.path.join(REPO, "pbs", BATCH)
    scripts, train_jobs, downstream_jobs = {}, [], []
    total_train, total_downstream = 0.0, 0.0
    print("jobs")
    for index, run in enumerate(runs, start=1):
        name = f"tms_t{index}"
        minutes = train_minutes(run)
        total_train += minutes
        scripts[name] = TEMPLATE.format(
            walltime=walltime(minutes, MIN_TRAIN_WALLTIME_HOURS),
            name=name,
            root=REPO,
            batch=BATCH,
            work=f"train {run['folder']}",
            estimate=int(minutes),
            body=train_body(run),
        )
        train_jobs.append((name, os.path.join(out_dir, f"{name}.pbs")))
        print(f"  {name}  train {run['folder']:34s} {minutes / 60:5.2f} h")

    downstream = [(run, f"tms_t{i}") for i, run in enumerate(runs, start=1)]
    downstream += [(cell, None) for cell in existing]
    for index, (cell, depends_on) in enumerate(downstream, start=1):
        name = f"tms_d{index}"
        gaps = cell_gaps(
            cell["folder"], declaration, args.checkpoints_dir, args.results_dir
        )
        detectors = pending_detectors(cell["folder"], args.results_dir)
        minutes = downstream_minutes(cell, gaps, detectors)
        total_downstream += minutes
        scripts[name] = TEMPLATE.format(
            walltime=walltime(minutes, MIN_DOWNSTREAM_WALLTIME_HOURS),
            name=name,
            root=REPO,
            batch=BATCH,
            work=f"sweep and detectors {cell['folder']}",
            estimate=int(minutes),
            body=downstream_body(cell, gaps, detectors, args),
        )
        downstream_jobs.append((name, os.path.join(out_dir, f"{name}.pbs"), depends_on))
        rate_units = sum(len(gap["missing_rates"]) for gap in gaps)
        after = f"after {depends_on}" if depends_on else "no dependency"
        print(
            f"  {name}  sweep {cell['folder']:34s} {minutes / 60:5.2f} h  "
            f"{len(gaps)} placements, {rate_units} rates, {cell['inputs']} inputs, "
            f"{sum(len(n) for n in detectors.values())} detectors, {after}"
        )

    for script in scripts.values():
        verify_script(script)

    total = total_train + total_downstream
    print(f"training       {total_train / 60:6.1f} A100-hours")
    print(f"downstream     {total_downstream / 60:6.1f} A100-hours if every run clears")
    print(f"total          {total / 60:6.1f} A100-hours, {len(scripts)} jobs")
    if args.dry_run:
        print("(dry run, nothing written)")
        return

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(REPO, "logs", BATCH), exist_ok=True)
    for name, script in scripts.items():
        write_file(os.path.join(out_dir, f"{name}.pbs"), script)
    submit = os.path.join(out_dir, "submit_all.sh")
    write_file(submit, submit_script(train_jobs, downstream_jobs))
    print(f"wrote {len(scripts)} jobs to {os.path.relpath(out_dir, REPO)}")
    print(f"submit with bash {os.path.relpath(submit, REPO)}")


if __name__ == "__main__":
    main()
