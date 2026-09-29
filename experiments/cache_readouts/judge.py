"""Verdicts for the pre-registered picks and for the plan's own X3 and X19 predictions.

Reads preregistration.json and the readouts of every model set, applies each
prediction's rule to the stored paired summaries and writes verdicts.json. The
hash of the pre-registration is stored beside the verdicts, so a later edit of
the predictions is visible.

    .venv/bin/python -m experiments.cache_readouts.judge
"""

import hashlib
import json
import os

from experiments.cache_readouts.shared import (
    MODEL_SETS,
    REPO_ROOT,
    output_path,
    write_json,
)

PREREGISTRATION = os.path.join(
    REPO_ROOT, "experiments", "cache_readouts", "preregistration.json"
)
PATCH = ("badnet_a2o", "tact")
# The plan's "most of the WaNet gain": the weighted rule keeps at least this
# share of the min-rank rule's WaNet AUROC gain.
MOST_SHARE = 0.5
PLAN_TOLERANCE = 0.01


def main():
    with open(PREREGISTRATION, "rb") as handle:
        raw = handle.read()
    preregistration = json.loads(raw)

    verdicts = {
        "preregistration_sha256": hashlib.sha256(raw).hexdigest(),
        "sets": {},
    }
    for model_set in MODEL_SETS:
        passes = read(output_path("pass_statistics", model_set))
        fusion = read(output_path("fusion_rules", model_set))
        bands = read(output_path("depth_bands", model_set))
        verdicts["sets"][model_set] = {
            "preregistered": judge_preregistered(
                preregistration["picks"], passes, fusion
            ),
            "plan_x19": judge_plan_x19(passes),
            "plan_x3": {
                rate_rule: judge_plan_x3(fusion, rate_rule)
                for rate_rule in fusion["rate_rules"]
            },
            "plan_x4": {
                reading: {
                    key: value["verdict"]
                    for key, value in bands["verdicts"][reading].items()
                }
                for reading in bands["readings"]
            },
        }
    path = output_path("verdicts", "all")
    write_json(verdicts, path)
    print(f"wrote {path}")


def read(path):
    with open(path) as handle:
        payload = json.load(handle)
    return payload


def judge_preregistered(picks, passes, fusion):
    results = []

    pass_pick = picks["pass_statistic"]
    block = passes["summary"][pass_pick["placement"]]
    for prediction in pass_pick["predictions"]:
        statistic = prediction.get("statistic", pass_pick["pick"])
        results.append(
            apply_rule(
                prediction, block["all"][statistic], block["by_attack"], statistic
            )
        )

    fusion_pick = picks["fusion_rule"]
    block = fusion["summary"][fusion_pick["rate_rule"]][fusion_pick["partner"]]
    for prediction in fusion_pick["predictions"]:
        results.append(
            apply_rule(
                prediction,
                block["all"][fusion_pick["rule"]],
                block["by_attack"],
                fusion_pick["rule"],
            )
        )
    return results


def apply_rule(prediction, overall, by_attack, key):
    field = prediction["field"]
    rule = prediction["rule"]
    reading = overall[field]
    difference = reading["mean_difference"]
    low, high = reading["ci95"]

    if rule == "mean_difference_positive_and_ci_excludes_0":
        if difference > 0 and low > 0:
            verdict = "held"
        elif difference > 0:
            verdict = "inconclusive"
        else:
            verdict = "failed"
    elif rule == "mean_difference_positive":
        verdict = "held" if difference > 0 else "failed"
    elif rule == "mean_difference_negative":
        verdict = "held" if difference < 0 else "failed"
    elif rule == "mean_difference_at_least":
        verdict = "held" if difference >= prediction["bound"] else "failed"
    else:
        per_attack = {
            attack: by_attack[attack][key][field]["mean_difference"]
            for attack in prediction["attacks"]
            if attack in by_attack
        }
        difference = per_attack
        low, high = None, None
        if len(per_attack) < len(prediction["attacks"]):
            verdict = "inconclusive"
        elif all(value >= prediction["bound"] for value in per_attack.values()):
            verdict = "held"
        else:
            verdict = "failed"

    result = {
        "id": prediction["id"],
        "text": prediction["text"],
        "verdict": verdict,
        "n": reading["n"],
        "mean_difference": difference,
        "ci95": [low, high],
    }
    return result


def judge_plan_x19(passes):
    # N17: on patch triggers the worst pass separates at least as well as the
    # mean and raises TPR at 1% FPR.
    by_attack = passes["summary"]["psbd_tm"]["by_attack"]
    checks = []
    for attack in PATCH:
        if attack not in by_attack:
            continue
        worst = by_attack[attack]["worst_pass"]
        checks.append(
            {
                "attack": attack,
                "n": by_attack[attack]["n"],
                "auroc_difference": worst["auroc"]["mean_difference"],
                "tpr_q0.01_difference": worst["q0.01:tpr"]["mean_difference"],
                "holds": worst["auroc"]["mean_difference"] >= 0
                and worst["q0.01:tpr"]["mean_difference"] > 0,
            }
        )
    verdict = verdict_from(checks)
    return verdict


def judge_plan_x3(fusion, rate_rule):
    # The plan: the late band helps WaNet and Blend and hurts TaCT under the
    # mean and min-rank rules, and the weighted rule keeps TaCT within 0.01
    # while keeping most of the WaNet gain.
    by_attack = fusion["summary"][rate_rule]["late_band"]["by_attack"]

    def difference(attack, rule, field):
        if attack not in by_attack:
            return None
        value = by_attack[attack][rule][field]["mean_difference"]
        return value

    checks = []
    for rule in ("mean_psu", "min_rank"):
        for attack, sign in (("wanet", 1), ("blend", 1), ("tact", -1)):
            value = difference(attack, rule, "auroc")
            checks.append(
                {
                    "claim": f"{rule} {'helps' if sign > 0 else 'hurts'} {attack}",
                    "auroc_difference": value,
                    "holds": None if value is None else value * sign > 0,
                }
            )
    for rule in ("weighted_0.8_0.2", "weighted_0.9_0.1"):
        tact_auroc = difference("tact", rule, "auroc")
        tact_tpr = difference("tact", rule, "q0.10:tpr")
        checks.append(
            {
                "claim": f"{rule} keeps TaCT within {PLAN_TOLERANCE} of AUROC and of TPR at 10% FPR",
                "auroc_difference": tact_auroc,
                "tpr_q0.10_difference": tact_tpr,
                "holds": None
                if tact_auroc is None
                else tact_auroc >= -PLAN_TOLERANCE and tact_tpr >= -PLAN_TOLERANCE,
            }
        )
        wanet_weighted = difference("wanet", rule, "auroc")
        wanet_min_rank = difference("wanet", "min_rank", "auroc")
        checks.append(
            {
                "claim": f"{rule} keeps at least {MOST_SHARE} of the min-rank WaNet gain",
                "weighted_auroc_difference": wanet_weighted,
                "min_rank_auroc_difference": wanet_min_rank,
                "holds": None
                if wanet_weighted is None
                or wanet_min_rank is None
                or wanet_min_rank <= 0
                else wanet_weighted >= MOST_SHARE * wanet_min_rank,
            }
        )
    verdict = verdict_from(checks)
    return verdict


def verdict_from(checks):
    decided = [check for check in checks if check["holds"] is not None]
    if not decided:
        return {"verdict": "inconclusive", "checks": checks}
    held = sum(check["holds"] for check in decided)
    if held == len(decided):
        label = "held"
    elif held == 0:
        label = "failed"
    else:
        label = "mixed"
    verdict = {
        "verdict": label,
        "n_held": held,
        "n_decided": len(decided),
        "n_undecided": len(checks) - len(decided),
        "checks": checks,
    }
    return verdict


if __name__ == "__main__":
    main()
