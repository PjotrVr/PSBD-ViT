"""1 model's whole GPU job: evaluation, the reproduction gate and the sweep.

The shared login GPU has 1 lock queue with many waiters, so each model takes its
lock once. Inside it, evaluate.py measures clean accuracy and ASR, and the gate
compares them with BackdoorBench's leaderboard before any sweep. A model the
leaderboard lists that misses it by more than the tolerances below stops the
whole queue (exit 3), since then this project's loader does not reproduce
BackdoorBench and no PSBD number from it would mean anything. A model below the
ASR bar or collapsed is recorded and not swept. The rest go to sweep_model.py.
The outcome goes to jobs/<folder>.json, which run_queue.sh reads to skip.

    .venv/bin/python -m experiments.backdoorbench_attacks.model_job --folder cifar10_ssba_0_1
"""

import os

from experiments.backdoorbench_attacks import evaluate, sweep_model
from experiments.backdoorbench_attacks.common import (
    EVALUATION_DIR,
    JOBS_DIR,
    read_json,
    write_json,
)

BASIS_PATH = "configs/psbd_basis.json"
# Fixed before the first model was evaluated. bfloat16 inference moves clean
# accuracy by well under a point, and a leaderboard entry may come from a rerun of
# the same configuration rather than this exact checkpoint.
CLEAN_TOLERANCE = 0.02
ASR_TOLERANCE = 0.05
COLLAPSE_SHARE = 0.5
REPRODUCTION_FAILED = 3


def main():
    folder = evaluate.parse_args().folder
    evaluate.main()
    record = read_json(os.path.join(EVALUATION_DIR, f"{folder}.json"))
    scores = record["backdoorbench_normalization"]
    asr_bar = read_json(BASIS_PATH)["asr_bar"]

    gate = reproduction_gate(scores, record["leaderboard"])
    outcome = {"folder": folder, "reproduction": gate, "asr_bar": asr_bar}
    if gate["verdict"] == "not reproduced":
        outcome["outcome"] = "stopped, not reproduced"
        write_json(outcome, os.path.join(JOBS_DIR, f"{folder}.json"))
        print(f"{folder}: not reproduced {gate}", flush=True)
        raise SystemExit(REPRODUCTION_FAILED)

    # Collapse is judged against the leaderboard's clean accuracy for this model
    # where it has one and against the ASR bar alone where it has none (Blind).
    board_clean = (record["leaderboard"] or {}).get("clean_accuracy")
    collapsed = (
        board_clean is not None
        and scores["clean_accuracy"] < COLLAPSE_SHARE * board_clean
    )
    if scores["asr"] < asr_bar or collapsed:
        outcome["outcome"] = "below the success bar, not swept"
        write_json(outcome, os.path.join(JOBS_DIR, f"{folder}.json"))
        print(f"{folder}: below the success bar, asr {scores['asr']:.4f}", flush=True)
        return

    sweep_model.main()
    outcome["outcome"] = "swept"
    write_json(outcome, os.path.join(JOBS_DIR, f"{folder}.json"))


def reproduction_gate(scores, leaderboard):
    if not leaderboard or leaderboard.get("clean_accuracy") is None:
        gate = {"verdict": "no leaderboard entry"}
        return gate

    clean_gap = scores["clean_accuracy"] - leaderboard["clean_accuracy"]
    asr_gap = scores["asr"] - leaderboard["asr"]
    reproduced = abs(clean_gap) <= CLEAN_TOLERANCE and abs(asr_gap) <= ASR_TOLERANCE
    gate = {
        "verdict": "reproduced" if reproduced else "not reproduced",
        "clean_gap": clean_gap,
        "asr_gap": asr_gap,
        "clean_tolerance": CLEAN_TOLERANCE,
        "asr_tolerance": ASR_TOLERANCE,
    }
    return gate


if __name__ == "__main__":
    main()
