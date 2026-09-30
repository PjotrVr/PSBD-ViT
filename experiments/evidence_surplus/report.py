"""Verdicts, tables and figures for B (trigger_dose) and C (clean_surplus).

Reads the records under results/_experiments/evidence_surplus/, applies the rules
of each PREDICTIONS.md as written before the runs, writes summary_bc.json, 2
figures with JSON sidecars under results/_experiments/evidence_surplus/figures/,
and the 2 README.md files, so no number in them is typed. CPU only.

    PYTHONPATH=. python experiments/evidence_surplus/report.py
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import scripts.paper._style  # noqa: E402, F401
from scripts.paper._common import OKABE_ITO  # noqa: E402
from scripts.paper._style import TEXT_WIDTH  # noqa: E402

RESULTS = os.path.join("results", "_experiments", "evidence_surplus")
HERE = os.path.dirname(os.path.abspath(__file__))
CHANCE_BAND = 0.1
ASR_FLOOR = 0.5
NOMINAL_MULTIPLE = 2.0


def main():
    dose = read_folder("trigger_dose")
    collages = read_folder("clean_surplus")
    natural = read_json("clean_surplus_natural.json")
    summary = {
        "dose": dose_verdicts(dose),
        "collages": collage_verdicts(collages),
        "natural": natural,
    }
    os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)
    with open(os.path.join(RESULTS, "summary_bc.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    dose_figure(dose)
    collage_figure(collages)
    write_dose_readme(summary["dose"])
    write_collage_readme(summary["collages"], natural)
    print(
        json.dumps(
            {k: v.get("verdicts") for k, v in summary.items() if isinstance(v, dict)},
            indent=2,
        )
    )


def read_folder(name):
    folder = os.path.join(RESULTS, name)
    records = []
    if os.path.isdir(folder):
        for file in sorted(f for f in os.listdir(folder) if f.endswith(".json")):
            with open(os.path.join(folder, file)) as handle:
                records.append(json.load(handle))
    return records


def read_json(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        payload = json.load(handle)
    return payload


def model_name(record):
    name = record["folder"]
    if record.get("probe_attack"):
        name = f"{name} ({record['probe_attack']} probe)"
    return name


# The rules of trigger_dose/PREDICTIONS.md, per model.
def dose_verdicts(records):
    rows = []
    for r in records:
        steps = r["dose"]
        benign = r.get("probe_attack") is not None
        falls = all(
            later["auroc"] <= earlier["auroc"] + 0.02
            for earlier, later in zip(steps, steps[1:])
        )
        band = [
            s
            for s in steps
            if s["asr"] >= ASR_FLOOR and abs(s["auroc"] - 0.5) <= CHANCE_BAND
        ]
        live = [s for s in steps if s["asr"] >= ASR_FLOOR]
        weakest = live[-1] if live else None
        rows.append(
            {
                "model": model_name(r),
                "attack": r["attack"],
                "benign": benign,
                "detection_falls_with_dose": falls,
                "chance_band_doses": [s["dose"] for s in band],
                "weakest_live_dose": weakest["dose"] if weakest else None,
                "weakest_live_asr": weakest["asr"] if weakest else None,
                "weakest_live_auroc": weakest["auroc"] if weakest else None,
                "weakest_live_auroc_hits": weakest["auroc_hits"] if weakest else None,
                "max_distance_from_chance": max(abs(s["auroc"] - 0.5) for s in steps),
                "full_dose_hit_margin": steps[0].get("median_hit_margin"),
                "weakest_live_hit_margin": weakest.get("median_hit_margin")
                if weakest
                else None,
            }
        )
    backdoored = [r for r in rows if not r["benign"]]
    benign = [r for r in rows if r["benign"]]
    verdicts = {
        "P1_falls_on": sum(r["detection_falls_with_dose"] for r in backdoored),
        "P2_band_on": sum(bool(r["chance_band_doses"]) for r in backdoored),
        "P2_hits_at_chance_on": sum(
            r["weakest_live_auroc_hits"] is not None
            and abs(r["weakest_live_auroc_hits"] - 0.5) <= CHANCE_BAND
            for r in backdoored
        ),
        "P3_benign_max_distance": max(
            (r["max_distance_from_chance"] for r in benign), default=None
        ),
        "models": len(backdoored),
    }
    summary = {"rows": rows, "verdicts": verdicts}
    return summary


INPUT_NAMES = ("same_class", "duplicated", "one_quadrant", "different_class", "single")


# The rules of clean_surplus/PREDICTIONS.md (1 to 3 and, with the resolution
# addendum, 5 and 6), per model and quantile.
def collage_verdicts(records):
    rows = []
    for r in records:
        inputs = r["inputs"]
        for quantile in inputs["single"]["flagged"]:
            row = {
                "model": r["folder"],
                "benign": "benign" in r["folder"],
                "quantile": float(quantile),
            }
            for name in INPUT_NAMES:
                row[name] = inputs[name]["flagged"][quantile]
            rows.append(row)
    benign = [r for r in rows if r["benign"]]
    backdoored = [r for r in rows if not r["benign"]]
    surplus = {
        r["folder"]: {
            name: {
                "median_critical_rate": r["inputs"][name]["median_critical_rate"],
                "median_psu": r["inputs"][name]["median_psu"],
            }
            for name in INPUT_NAMES
        }
        for r in records
    }

    def ordered(model):
        c = {n: surplus[model][n]["median_critical_rate"] for n in INPUT_NAMES}
        holds = (
            min(c["same_class"], c["duplicated"]) > max(c["one_quadrant"], c["single"])
            and min(c["one_quadrant"], c["single"]) > c["different_class"]
        )
        return holds

    verdicts = {
        "P1_same_class_at_twice_nominal": sum(
            r["same_class"] >= NOMINAL_MULTIPLE * r["quantile"] for r in benign
        ),
        "P1_cases": len(benign),
        "P2_different_class_at_or_below_nominal": sum(
            r["different_class"] <= r["quantile"] for r in benign
        ),
        "P3_order_on_backdoored": sum(
            r["same_class"] >= r["single"] >= r["different_class"] for r in backdoored
        ),
        "P3_cases": len(backdoored),
        "P5_critical_rate_order_on": sum(ordered(m) for m in surplus),
        "P5_models": len(surplus),
        "P6_above_resolution_control": sum(
            r["same_class"] > r["one_quadrant"] and r["duplicated"] > r["one_quadrant"]
            for r in benign
        ),
    }
    summary = {
        "rows": rows,
        "surplus": surplus,
        "verdicts": verdicts,
        "diagnosis": own_class_diagnosis(records),
    }
    return summary


# The failure protocol's 1 extra measurement for C: a collage adds evidence only
# if the model still reads it as its own class. The share each input keeps its own
# class, and the flag at the 0.05 quantile among the inputs that do.
def own_class_diagnosis(records):
    diagnosis = {}
    for r in records:
        threshold = r["thresholds"]["0.05"]
        entry = {}
        for name in INPUT_NAMES:
            block = r["inputs"][name]
            if "predicted" not in block:
                continue
            kept = [p == label for p, label in zip(block["predicted"], block["labels"])]
            flagged = [psu < threshold for psu, keep in zip(block["psu"], kept) if keep]
            entry[name] = {
                "own_class_share": sum(kept) / len(kept),
                "flagged_among_own_class": sum(flagged) / len(flagged)
                if flagged
                else None,
                "own_class_count": len(flagged),
            }
        diagnosis[r["folder"]] = entry
    return diagnosis


def dose_figure(records):
    if not records:
        return
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.8), sharey=True)
    sidecar = []
    for index, r in enumerate(records):
        color = "0.6" if r.get("probe_attack") else OKABE_ITO[index % 8]
        doses = [s["dose"] for s in r["dose"]]
        axes[0].plot(
            doses,
            [s["auroc"] for s in r["dose"]],
            "-o",
            color=color,
            markersize=2,
            linewidth=0.8,
            label=model_name(r),
        )
        axes[1].plot(
            [s["asr"] for s in r["dose"]],
            [s["auroc"] for s in r["dose"]],
            "-o",
            color=color,
            markersize=2,
            linewidth=0.8,
        )
        sidecar.append(
            {
                "model": model_name(r),
                "dose": r["dose"],
                "partial_patch": r.get("partial_patch"),
            }
        )
    for axis in axes:
        axis.axhline(0.5, color="0.4", linewidth=0.8)
    axes[0].set_xlabel("trigger dose (1 trained, 0 absent)")
    axes[0].set_ylabel("detector AUROC, triggered against clean twin")
    axes[1].set_xlabel("ASR at that dose")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        fontsize=5,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "trigger_dose.png"))
    with open(os.path.join(RESULTS, "figures", "trigger_dose.json"), "w") as handle:
        json.dump(sidecar, handle)
    plt.close(fig)


def collage_figure(records):
    if not records:
        return
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.6))
    sidecar = []
    labels = ["same class", "duplicated", "1 quadrant", "different classes", "single"]
    for index, r in enumerate(records):
        color = OKABE_ITO[index % 8]
        rates = [r["inputs"][n]["median_critical_rate"] for n in INPUT_NAMES]
        flagged = [r["inputs"][n]["flagged"]["0.05"] for n in INPUT_NAMES]
        axes[0].plot(
            range(5), rates, "-o", color=color, markersize=3, label=r["folder"]
        )
        axes[1].plot(range(5), flagged, "-o", color=color, markersize=3)
        sidecar.append(
            {
                "model": r["folder"],
                **{
                    n: {
                        "median_critical_rate": r["inputs"][n]["median_critical_rate"],
                        "flagged": r["inputs"][n]["flagged"],
                    }
                    for n in INPUT_NAMES
                },
            }
        )
    axes[1].axhline(0.05, color="0.4", linewidth=0.8, linestyle="--")
    for axis in axes:
        axis.set_xticks(range(5))
        axis.set_xticklabels(labels, rotation=30, fontsize=6)
    axes[0].set_ylabel("median critical rate p*")
    axes[1].set_ylabel("share flagged at the 0.05 quantile")
    axes[0].legend(fontsize=5)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "clean_surplus.png"))
    with open(os.path.join(RESULTS, "figures", "clean_surplus.json"), "w") as handle:
        json.dump(sidecar, handle)
    plt.close(fig)


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def listed(items):
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    text = "\n".join(lines)
    return text


# The failure protocol's diagnosis for P2: the median target margin of the images
# a faded trigger still flips, at the trained dose and at the weakest dose that
# still reaches ASR 0.5. If the margin shrinks with the dose while detection of
# those images stays high, the detector reads whether the backdoor fired more
# than how much margin it carried.
def dose_diagnosis(dose):
    rows = [
        r
        for r in dose["rows"]
        if not r["benign"]
        and r["full_dose_hit_margin"] is not None
        and r["weakest_live_hit_margin"] is not None
    ]
    if not rows:
        return "The diagnosis field was not recorded."
    shrinks = [
        r
        for r in rows
        if r["weakest_live_hit_margin"] < 0.5 * r["full_dose_hit_margin"]
    ]
    detected = [r for r in shrinks if (r["weakest_live_auroc_hits"] or 0) >= 0.8]
    listing = ", ".join(
        f"{r['model']} {fmt(r['full_dose_hit_margin'], 2)} to {fmt(r['weakest_live_hit_margin'], 2)}"
        for r in rows
    )
    text = (
        "The pre-registered verdicts above stand as scored. The diagnosis for P2, 1 extra "
        "measurement: the median target margin of the images the faded trigger still sends "
        "to the target. From the trained dose to the weakest dose still at ASR 0.5 it falls "
        f"below half on {len(shrinks)} of {len(rows)} models ({listing}). On "
        f"{len(detected)} of those the detector still separates the flipped images at AUROC "
        "0.8 or more. On the other "
        f"{len(rows) - len(shrinks)} models the images a faded trigger still flips keep most "
        "of their margin: the backdoor fires nearly all or nothing, and fading the pixels "
        "lowers how many images it flips, not how much evidence the flipped ones carry. "
        "There the dose does not produce low-surplus triggered decisions, so the test cannot "
        "reach the regime P2 is about, which is a failure of the operational measure and "
        "not a reading of the account. Where the margin does shrink and detection holds, a "
        "trigger that barely flips an image is still detected, so the stability the probe "
        "reads does not scale with margin size there. That the surplus PSBD reads lives in "
        "how the evidence is spread over tokens (steps 4 and 5 of "
        "`notebooks/why-psbd-works-general.ipynb`) rather than in its size is a hypothesis."
    )
    return text


def write_dose_readme(dose):
    v = dose["verdicts"]
    rows = [
        [
            r["model"],
            fmt(r["detection_falls_with_dose"]),
            ", ".join(fmt(d, 2) for d in r["chance_band_doses"]) or "none",
            fmt(r["weakest_live_dose"], 2),
            fmt(r["weakest_live_asr"]),
            fmt(r["weakest_live_auroc"]),
            fmt(r["weakest_live_auroc_hits"]),
        ]
        for r in dose["rows"]
    ]
    text = f"""# B. Trigger dose-response

Generated by `experiments/evidence_surplus/report.py` from the records under `results/_experiments/evidence_surplus/trigger_dose/`. The predictions are in `PREDICTIONS.md`, written before the run. The method is in the docstring of `measure.py`.

The question: if a triggered decision survives the probe because the trigger supplies more evidence than the decision needs, fading the trigger should remove the detection before it removes the backdoor. Only the trigger changes between steps: the model, the operating point (placement, adaptive rate and clean-validation thresholds from the canonical sweep) and the clean twins stay fixed. Each model reads its first 256 PSBD pairs.

<!-- results:begin -->
{table(["model", "AUROC falls with the dose", "doses with ASR at least 0.5 and AUROC within 0.1 of chance", "weakest dose with ASR at least 0.5", "its ASR", "its AUROC", "its AUROC on images still sent to the target"], rows)}

P1 (detection falls toward chance as the trigger fades): the AUROC never rises by more than 0.02 from 1 dose to the next on {v["P1_falls_on"]} of {v["models"]} backdoored models. P2 (a band of doses where the backdoor still works and detection is at chance): such a band exists on {v["P2_band_on"]} of {v["models"]} models, and at the weakest dose that still reaches ASR 0.5 the AUROC on the images the faded trigger still sends to the target is within 0.1 of chance on {v["P2_hits_at_chance_on"]} of {v["models"]}. P3 (benign controls at chance): the benign models stay within {fmt(v["P3_benign_max_distance"])} of 0.5 at every dose.

{dose_diagnosis(dose)}
<!-- results:end -->

The figure is `results/_experiments/evidence_surplus/figures/trigger_dose.png`, with its numbers in `trigger_dose.json` beside it: AUROC against dose (left) and against ASR (right), benign models in gray. Where AUROC and ASR fall together the detector reads the backdoor's firing, and the hit-only AUROC says whether the images the weak trigger still flips are still detected. A detector that keeps separating those images reads whether the backdoor fired rather than how much surplus it carried, which is what P2 tests.
"""
    with open(os.path.join(HERE, "trigger_dose", "README.md"), "w") as handle:
        handle.write(text)


def write_collage_readme(collages, natural):
    v = collages["verdicts"]
    rows = [
        [r["model"], fmt(r["quantile"], 2)] + [fmt(r[n]) for n in INPUT_NAMES]
        for r in collages["rows"]
    ]
    surplus_rows = [
        [model]
        + [
            f"{fmt(s[n]['median_critical_rate'], 2)} / {fmt(s[n]['median_psu'])}"
            for n in INPUT_NAMES
        ]
        for model, s in collages["surplus"].items()
    ]
    diagnosis_rows = [
        [model]
        + [
            f"{fmt(d[n]['own_class_share'])} / {fmt(d[n]['flagged_among_own_class'])}"
            if n in d
            else "n/a"
            for n in INPUT_NAMES
        ]
        for model, d in collages["diagnosis"].items()
    ]
    diagnosis = collages["diagnosis"]
    at_five = [r for r in collages["rows"] if r["quantile"] == 0.05]
    composites = ("same_class", "duplicated", "one_quadrant", "different_class")
    above_single = sum(r[n] > r["single"] for r in at_five for n in composites)
    composite_cases = len(at_five) * len(composites)
    unread = [
        m
        for m, d in diagnosis.items()
        if d and d["same_class"]["own_class_share"] < 0.5
    ]
    read = [m for m, d in diagnosis.items() if d and m not in unread]
    duplication_wins = [
        m
        for m in read
        if (diagnosis[m]["duplicated"]["flagged_among_own_class"] or 0)
        > (diagnosis[m]["one_quadrant"]["flagged_among_own_class"] or 0)
    ]
    diagnosis_text = (
        f"The pre-registered verdicts above stand as scored. The diagnosis: at the 0.05 "
        f"quantile a composite is flagged more often than the single image in "
        f"{above_single} of {composite_cases} model and composite cases, the 1-quadrant "
        f"resolution control and the conflicting collage included, and the models read "
        f"composites as their own class "
        f"far less often than single images. On {listed(unread) or 'no model'} a same-class "
        f"collage keeps its own class on fewer than half of the inputs, so the manipulation "
        f"adds no evidence the model reads and the test says nothing about surplus there. "
        f"Among the composites that do keep their own class, on "
        f"{listed(read) or 'no model'}, the duplicated image is flagged more than the "
        f"1-quadrant control on {len(duplication_wins)} of {len(read)}. What the collages "
        f"measure first is that a composite is off the data these low-resolution fine-tuned "
        f"models learned, the operational measure the failure protocol names. A surplus "
        f"reading needs added evidence the model recognizes, which is a hypothesis this "
        f"test could not settle."
    )
    natural_text = (
        "The natural version has not run."
        if natural is None
        else (
            f"On {natural['prediction_4_holds_on']} of {natural['models']} models of `experiments/why_psbd_works`, "
            "the clean images whose answer never changes over PSBD-TM's 10 passes keep more of their margin "
            "(median retention) than the clean images that flip, prediction 4. That reading is close to "
            "circular, since an answer that never flips has kept a positive margin on every pass, so it "
            "confirms the proxy's consistency and does not show where the surplus comes from. The false "
            "positive test in `experiments/evidence_surplus/false_positives/` asks the question with "
            "measures that do not come from the probe."
        )
    )
    text = f"""# C. Clean surplus without a backdoor

Generated by `experiments/evidence_surplus/report.py` from the records under `results/_experiments/evidence_surplus/clean_surplus/`. The predictions are in `PREDICTIONS.md`, written before the run, with the resolution addendum written before any collage reading. The method is in the docstring of `measure.py`.

The question: if PSBD reads evidence surplus and not the backdoor as such, a clean input with more evidence for its own class than it needs should look poisoned to it, on a benign model too. 5 inputs at the same size: 4 images of 1 class, 1 image duplicated 4 times, 1 image in 1 quadrant with the rest mean-colored (the control for the halved per-image resolution of a collage), 4 images of 4 classes (conflicting evidence) and 1 image filling the input. The operating point is the canonical one, so the flagged shares compare directly with the nominal quantile, and each input also gets its critical rate p* along PSBD-TM's ladder.

<!-- results:begin -->
Share flagged per input kind:

{table(["model", "quantile", "same class", "duplicated", "1 quadrant", "different classes", "single"], rows)}

Median critical rate p* and median fractional PSU per input kind:

{table(["model", "same class", "duplicated", "1 quadrant", "different classes", "single"], surplus_rows)}

P1 (same-class collages flagged at twice the nominal rate or more on the benign models): {v["P1_same_class_at_twice_nominal"]} of {v["P1_cases"]} model and quantile cases. P2 (different-class collages at or below the nominal rate on the benign models): {v["P2_different_class_at_or_below_nominal"]} of {v["P1_cases"]}. P3 (the order same class, single, different class in the flagged share on the backdoored models): {v["P3_order_on_backdoored"]} of {v["P3_cases"]} cases. P5 (p* orders duplicated and same class above 1 quadrant and single, all above different classes): {v["P5_critical_rate_order_on"]} of {v["P5_models"]} models. P6 (same class and duplicated flagged above the 1-quadrant resolution control on the benign models): {v["P6_above_resolution_control"]} of {v["P1_cases"]} cases.

Diagnosis, the 1 extra measurement the failure protocol asks for. A composite adds evidence only if the model still reads it as its own class. Per input kind, the share of inputs the model assigns to their own class, then the share flagged at the 0.05 quantile among those:

{table(["model", "same class", "duplicated", "1 quadrant", "different classes (first image's class)", "single"], diagnosis_rows)}

{diagnosis_text}

{natural_text}
<!-- results:end -->

The figure is `results/_experiments/evidence_surplus/figures/clean_surplus.png`, with its numbers in `clean_surplus.json`: the median critical rate (left) and the share flagged at the 0.05 quantile (right) per input kind, 1 line per model, the dashed line the nominal rate.
"""
    with open(os.path.join(HERE, "clean_surplus", "README.md"), "w") as handle:
        handle.write(text)


if __name__ == "__main__":
    main()
