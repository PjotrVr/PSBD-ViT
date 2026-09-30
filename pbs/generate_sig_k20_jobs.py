"""Emit G1: the SIG amplitude reruns, then the remaining ViT k=20 sweeps while time remains.

The SIG part replays scratch/sig_rerun.sh (docs/runs/2026-09-29-sig-amplitude.md) on
a PBS GPU: every sweep whose perturbed passes were built at amplitude 0.157 over a
model trained at 0.1 is rerun, in the order the tables need them. It skips what is
already redone. A placement counts as redone when a `<folder> <placement> exit 0`
line stands in the login rerun's summary (scratch/sig_rerun_logs/summary.txt) or in
this job's own (logs/sig_k20/summary.txt), so a resubmitted G1 continues where the
last one stopped. A stale cache is moved to `results/<folder>/_stale_sig_amp0157/`
only when no stale copy is there yet, since a second move would bury the first
inside it and the cache left in place is then a post-repair partial, which
`--skip-existing` completes. Nothing is deleted.

The k=20 part sweeps PSBD-TM and PSBD-RD at 20 forward passes at each panel
model's adaptive rate, from scratch/k20_plan.json, skipping every cache whose 3
split tensors exist. It starts no new sweep once the job has run for its
estimate, so the 25% margin of the walltime stays free and no sweep is cut off
mid-write.

Neither part may run while the login-node runners scratch/sig_rerun.sh or
scratch/k20_driver.sh are writing the same caches.

    PYTHONPATH=. python pbs/generate_sig_k20_jobs.py --dry-run
    PYTHONPATH=. python pbs/generate_sig_k20_jobs.py
    bash pbs/sig_k20/submit_all.sh
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from detectors import DETECTOR_NAMES  # noqa: E402
from pbs.generate_detector_jobs import estimated_minutes as detector_minutes  # noqa: E402
from pbs.generate_evidence_surplus_jobs import (  # noqa: E402
    BASE,
    DECLARATION,
    JOB_TEMPLATE,
    PSBD_INPUTS,
    SWEEP_MINUTES_PER_RATE_PER_1000_INPUTS,
    base_accepts,
    hours_text,
    load_basis_placements,
    preflight_block,
    readme_text,
    requested_walltime,
    validate_calls,
    write_batch,
)
from pbs.generate_union_attacker_jobs import render_call  # noqa: E402

BATCH = "sig_k20"
JOB_NAME = "G1"
LOGIN_SUMMARY = "scratch/sig_rerun_logs/summary.txt"
JOB_SUMMARY = f"logs/{BATCH}/summary.txt"
STALE_DIR = "_stale_sig_amp0157"

# The table readers need the Swin headline cell first, then the Swin cell whose
# baseline was built at the wrong amplitude, then the ViT cell kept correct for the
# all-numbers table (scratch/sig_rerun.sh's order).
SWIN_HEADLINE = "swin_cifar10_sig_0_1"
SWIN_REBUILT_BASELINE = "swin_cifar10_sig_0_05"
VIT_CELL = "vit_cifar10_sig_0_1"
SWIN_FIRST = (
    "before_attention_norm_token_mask",
    "before_attention_residual_token_mask",
    "both_sublayer_inputs_token_mask",
    "before_attention_norm_channel_mask",
    "before_attention_norm_gaussian",
    "before_mlp_norm_token_mask",
)
# The recorded command for this cell's PSBD-TM swept 1 rate, so its full basis
# ladder is swept instead, as scratch/sig_rerun.sh does.
FULL_LADDER_PLACEMENT = "before_attention_norm_token_mask"

K20_PLACEMENTS = {
    "tm": ("before_attention_norm", "token_mask", "before_attention_norm_token_mask"),
    "rd": ("post_residual", "dropout", "post_residual"),
}
K20_PASSES = 20

# Sweep minutes per rate on a PBS A100 (pbs/generate_evidence_surplus_jobs.py's
# measured cost per 1000 inputs) over a CIFAR-10 cell's inputs. The 4 reruns that
# finished on the shared login A100 on 2026-09-29 took about 3 times as long (481 to
# 673 seconds for 9 or 10 rates), and the job stays inside its walltime even at that
# speed, since the k=20 part then gets no time at all.
SIG_MINUTES_PER_RATE = (
    SWEEP_MINUTES_PER_RATE_PER_1000_INPUTS * PSBD_INPUTS["cifar10"] / 1000
)
SWEEP_MINUTES_PER_CALL = 0.5
ANALYZE_MINUTES = 5.0
# A k=20 sweep at 1 rate took 111 to 226 seconds on the login A100
# (scratch/k20_summary.txt), and the slowest is used.
K20_MINUTES_PER_SWEEP = 4.0
# The job is sized to this, the upper end of the requested 8 to 10 hours, and the
# k=20 part takes whatever the SIG part leaves.
TARGET_MINUTES = 9.5 * 60

SHELL_FUNCTIONS = r"""LOGIN_SUMMARY={login_summary}
SUMMARY={job_summary}
mkdir -p "$(dirname $SUMMARY)"
JOB_START=$(date +%s)
BUDGET_SECONDS={budget_seconds}

redone() {{
  # redone <folder> <placement>: a clean exit is on record in either summary
  grep -qsE " $1 $2 exit 0 " $LOGIN_SUMMARY $SUMMARY
}}

stash_once() {{
  # stash_once <folder> <path under results/<folder>>
  local src=results/$1/$2 dst=results/$1/{stale}/$2
  if [ -e "$dst" ]; then return 0; fi
  if [ -e "$src" ]; then mkdir -p "$(dirname "$dst")"; mv "$src" "$dst"; fi
}}

sig_sweep() {{
  # sig_sweep <folder> <placement> <cli.sweep arguments...>
  local folder=$1 placement=$2; shift 2
  if redone $folder $placement; then echo "skip $folder $placement, redone"; return 0; fi
  stash_once $folder psbd/$placement
  stash_once $folder psbd/run_$placement.json
  local start=$(date +%s)
  python -m cli.sweep --checkpoint-folder $folder "$@" --skip-existing
  local status=$?
  echo "$(date) $folder $placement exit $status $(( $(date +%s) - start ))s" >> $SUMMARY
}}

analyze() {{
  python -m cli.analyze --checkpoint-folder $1
  echo "$(date) $1 analyze exit $?" >> $SUMMARY
}}

k20_complete() {{
  # k20_complete <folder> <placement> <rate tag>
  local d=results/$1/psbd/$2_k20
  [ -e $d/rate_$3_validation.pt ] && [ -e $d/rate_$3_clean.pt ] && [ -e $d/rate_$3_backdoor.pt ]
}}

within_budget() {{
  [ $(( $(date +%s) - JOB_START )) -lt $BUDGET_SECONDS ]
}}
"""


def read_sig_commands(path: str) -> dict[str, list[tuple[str, list[str]]]]:
    """Each folder's (placement, cli.sweep arguments after the folder), in file order."""
    by_folder: dict[str, list[tuple[str, list[str]]]] = {}
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            placement, command = line.split(maxsplit=1)
            tokens = command.split()
            folder_index = tokens.index("--checkpoint-folder") + 1
            folder = tokens[folder_index]
            arguments = tokens[folder_index + 1 :]
            by_folder.setdefault(folder, []).append((placement, arguments))
    return by_folder


def redone_on_record(summary_path: str) -> set[tuple[str, str]]:
    """(folder, placement) pairs the login rerun finished with exit 0."""
    if not os.path.exists(summary_path):
        return set()
    done = set()
    with open(summary_path) as handle:
        for line in handle:
            tokens = line.split()
            if "exit" in tokens:
                position = tokens.index("exit")
                if tokens[position + 1] == "0" and position >= 2:
                    done.add((tokens[position - 2], tokens[position - 1]))
    return done


def ordered_sig_work(
    commands: dict[str, list[tuple[str, list[str]]]], tm_ladder: list[float]
) -> list[tuple[str, str, list[str]]]:
    """(folder, placement, arguments) in the order the tables need them."""
    swin_headline = dict(commands[SWIN_HEADLINE])
    first = [placement for placement in SWIN_FIRST if placement in swin_headline]
    rest = [
        placement for placement, _ in commands[SWIN_HEADLINE] if placement not in first
    ]
    work = [
        (SWIN_HEADLINE, placement, swin_headline[placement])
        for placement in first + rest
    ]

    for placement, arguments in commands[SWIN_REBUILT_BASELINE]:
        if placement == FULL_LADDER_PLACEMENT:
            arguments = [
                "--position",
                "before_attention_norm",
                "--operator",
                "token_mask",
                "--rates",
                *[str(rate) for rate in tm_ladder],
            ]
        work.append((SWIN_REBUILT_BASELINE, placement, arguments))

    work += [
        (VIT_CELL, placement, arguments) for placement, arguments in commands[VIT_CELL]
    ]
    return work


def rate_count(arguments: list[str]) -> int:
    """How many rates a cli.sweep argument list sweeps."""
    start = arguments.index("--rates") + 1
    count = 0
    for token in arguments[start:]:
        if token.startswith("--"):
            break
        count += 1
    return count


def rate_tag(rate: float) -> str:
    tag = str(rate).replace(".", "_")
    return tag


def pending_k20(plan_path: str, results_dir: str) -> list[tuple[str, str, float]]:
    """(folder, tm or rd, rate) of every k=20 sweep whose cache is not complete on disk."""
    with open(plan_path) as handle:
        plan = json.load(handle)
    pending = []
    for row in plan:
        for which, (_, _, placement) in K20_PLACEMENTS.items():
            rate = row[which]
            if rate is None:
                continue
            cache = os.path.join(results_dir, row["folder"], "psbd", f"{placement}_k20")
            complete = all(
                os.path.exists(os.path.join(cache, f"rate_{rate_tag(rate)}_{split}.pt"))
                for split in ("validation", "clean", "backdoor")
            )
            if not complete:
                pending.append((row["folder"], which, rate))
    return pending


def k20_argv(folder: str, which: str, rate: float) -> list[str]:
    position, operator, _ = K20_PLACEMENTS[which]
    argv = [
        "--checkpoint-folder",
        folder,
        "--position",
        position,
        "--operator",
        operator,
        "--forward-passes",
        str(K20_PASSES),
        "--rates",
        str(rate),
        "--skip-existing",
    ]
    return argv


def sig_body(work: list[tuple[str, str, list[str]]]) -> str:
    """The SIG commands, each folder analyzed after its sweeps."""
    lines = ['echo "=== SIG amplitude reruns $(date) ==="']
    folders_in_order = list(dict.fromkeys(folder for folder, _, _ in work))
    for folder in folders_in_order:
        if folder == SWIN_REBUILT_BASELINE:
            lines.append(
                "# Its baseline was built at 0.157, so it is moved aside and rebuilt at "
                "the training amplitude. Its ASR in args.json is corrected by hand "
                "afterwards (docs/runs/2026-09-29-sig-amplitude.md)."
            )
            lines += [
                f"stash_once {folder} psbd/baseline_{split}.pt"
                for split in ("backdoor", "clean", "validation")
            ]
        for placement_folder, placement, arguments in work:
            if placement_folder == folder:
                lines.append(f"sig_sweep {folder} {placement} {' '.join(arguments)}")
        lines.append(f"analyze {folder}")
    lines += [
        "# Every detector record of the ViT cell was built at 0.157.",
        f"if grep -qsE ' {VIT_CELL} baselines exit 0 ' $LOGIN_SUMMARY $SUMMARY; then",
        f'  echo "skip {VIT_CELL} detectors, redone"',
        "else",
        f"  stash_once {VIT_CELL} detectors",
        f"  python -m cli.baselines --checkpoint-folder {VIT_CELL} --skip-existing",
        f'  echo "$(date) {VIT_CELL} baselines exit $? " >> $SUMMARY',
        "fi",
        "",
    ]
    body = "\n".join(lines)
    return body


def k20_body(pending: list[tuple[str, str, float]]) -> str:
    """The k=20 sweeps, each folder analyzed after its last sweep, within the budget."""
    lines = ['echo "=== ViT k=20 sweeps $(date) ==="']
    folders_in_order = list(dict.fromkeys(folder for folder, _, _ in pending))
    for folder in folders_in_order:
        for pending_folder, which, rate in pending:
            if pending_folder != folder:
                continue
            placement = K20_PLACEMENTS[which][2]
            lines += [
                f"if k20_complete {folder} {placement} {rate_tag(rate)}; then",
                f'  echo "skip {folder} {which} k20, complete"',
                "elif within_budget; then",
                render_call("cli.sweep", k20_argv(folder, which, rate)).rstrip("\n")
                + f' || echo "[FAILED rc=$?] k20 {folder} {which}"',
                "else",
                f'  echo "$(date) budget reached before {folder} {which}" >> $SUMMARY',
                "fi",
            ]
        lines += ["if within_budget; then", f"  analyze {folder}", "fi"]
    lines.append("")
    body = "\n".join(lines)
    return body


def sig_minutes(work: list[tuple[str, str, list[str]]], done: set) -> float:
    """The SIG part's minutes: the sweeps not yet redone, the analyses, the detectors."""
    remaining = [item for item in work if (item[0], item[1]) not in done]
    sweeps = sum(
        rate_count(arguments) * SIG_MINUTES_PER_RATE + SWEEP_MINUTES_PER_CALL
        for _, _, arguments in remaining
    )
    folders = len({folder for folder, _, _ in work})
    detectors = detector_minutes({"dataset": "cifar10"}, list(DETECTOR_NAMES), "all")
    minutes = sweeps + folders * ANALYZE_MINUTES + detectors
    return minutes


def k20_minutes(pending: list[tuple[str, str, float]]) -> float:
    folders = len({folder for folder, _, _ in pending})
    minutes = len(pending) * K20_MINUTES_PER_SWEEP + folders * ANALYZE_MINUTES
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
    scratch = os.path.join(args.base, "scratch")
    tm_ladder = load_basis_placements(DECLARATION, (FULL_LADDER_PLACEMENT,))[0]["rates"]
    commands = read_sig_commands(os.path.join(scratch, "sig_rerun_commands.txt"))
    work = ordered_sig_work(commands, tm_ladder)
    done = redone_on_record(os.path.join(args.base, LOGIN_SUMMARY))
    pending = pending_k20(
        os.path.join(scratch, "k20_plan.json"), os.path.join(args.base, "results")
    )

    calls = [
        ("cli.sweep", ["--checkpoint-folder", folder, *arguments, "--skip-existing"])
        for folder, _, arguments in work
    ]
    calls += [("cli.sweep", k20_argv(*item)) for item in pending]
    calls += [("cli.baselines", ["--checkpoint-folder", VIT_CELL, "--skip-existing"])]
    calls += [
        ("cli.analyze", ["--checkpoint-folder", folder])
        for folder in dict.fromkeys(folder for folder, _, _ in work)
    ]
    validate_calls(calls)

    sig_left = len([item for item in work if (item[0], item[1]) not in done])
    sig_estimate = sig_minutes(work, done)
    k20_full = k20_minutes(pending)
    estimate = min(sig_estimate + k20_full, max(TARGET_MINUTES, sig_estimate))
    script = JOB_TEMPLATE.format(
        walltime=requested_walltime(estimate),
        name=JOB_NAME,
        base=args.base,
        batch=BATCH,
        work=f"SIG reruns ({sig_left} sweeps left), then up to {len(pending)} "
        "k=20 sweeps",
        estimate=estimate,
        preflight=preflight_block(calls),
        body=SHELL_FUNCTIONS.format(
            login_summary=LOGIN_SUMMARY,
            job_summary=JOB_SUMMARY,
            budget_seconds=int(estimate * 60),
            stale=STALE_DIR,
        )
        + "\n"
        + sig_body(work)
        + "\n"
        + k20_body(pending),
    )
    fitting_k20 = max(0.0, estimate - sig_estimate)
    rows = [
        {
            "name": JOB_NAME,
            "contents": (
                f"SIG reruns, {sig_left} of {len(work)} sweeps not yet redone over "
                f"`{SWIN_HEADLINE}`, `{SWIN_REBUILT_BASELINE}` and `{VIT_CELL}` "
                f"plus the 11 detectors of `{VIT_CELL}` ({hours_text(sig_estimate)}), "
                f"then {len(pending)} ViT k=20 sweeps pending "
                f"({hours_text(k20_full)} if all ran, {hours_text(fitting_k20)} fit)"
            ),
            "estimate": estimate,
            "walltime": requested_walltime(estimate),
            "accepted": base_accepts(calls, args.base),
        }
    ]
    print(
        f"G1 est {hours_text(estimate)} (SIG {hours_text(sig_estimate)}, k20 "
        f"{hours_text(k20_full)} pending) walltime {requested_walltime(estimate)} "
        f"accepted now {rows[0]['accepted']}"
    )

    preamble = (
        "Generated by `pbs/generate_sig_k20_jobs.py`, which states the time model "
        "and the skip rules. Not submitted. `submit_all.sh` submits it by hand. Do "
        "not submit it while `scratch/sig_rerun.sh` or `scratch/k20_driver.sh` runs "
        "on the login node, since both write the same caches. The k=20 part starts "
        "no sweep after the estimate has elapsed, so a sweep that does not fit waits "
        "for the next submission."
    )
    readme = readme_text(rows, "SIG reruns and ViT k=20 sweeps", preamble)
    if args.dry_run:
        print(readme)
        print(script)
        return
    write_batch(args.out_dir, {JOB_NAME: script}, readme, args.base, BATCH)
    print(f"wrote {JOB_NAME} to {args.out_dir}")


if __name__ == "__main__":
    main()
