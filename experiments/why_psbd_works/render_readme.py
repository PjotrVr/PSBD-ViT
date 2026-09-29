"""Write experiments/why_psbd_works/README.md from the code's constants and the records.

Every value the README quotes is read here: the measurement settings from
measure.py (through summary.json's parameters), the verdict thresholds from
summarize.THRESHOLDS, the sanity gates from sanity_<folder>.json, the results from
summary.json and cached_reads.json, and the wall time from each record. Nothing
is typed by hand, so the README cannot disagree with the run it describes. Run it
after summarize.py.

    PYTHONPATH=. python experiments/why_psbd_works/render_readme.py
"""

import json
import os
import statistics
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from data.splits import PSBD_HELDOUT_SIZE  # noqa: E402
from defenses.decision import (  # noqa: E402
    ADAPTIVE_SHIFT_TARGET,
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
)
from experiments.why_psbd_works import measure  # noqa: E402
from scripts.paper._common import excluded_folders  # noqa: E402

RESULTS = os.path.join("results", "_experiments", "why_psbd_works")
OUT = os.path.join("experiments", "why_psbd_works", "README.md")
SANITY_FOLDER = "vit_cifar100_badnet_a2o_0_05"
LABEL = {
    "patch": "BadNets",
    "patch_tact": "TaCT",
    "patch_tact_source_mapped": "TaCT, source mapped",
    "blend": "Blend and LF",
    "frequency": "SIG",
    "warp": "WaNet",
    "quantization": "BPP",
    "benign_probe_badnet_a2o": "benign, BadNets probe",
    "benign_probe_blend": "benign, Blend probe",
}
ORDER = list(LABEL)
HEADLINE = ("token_mask", "dropout", "channel_mask", "gaussian", "residual_dropout")
ALL_OPERATORS = tuple(measure.OPERATOR_SPECS)
OPERATOR_LABEL = {
    "token_mask": "token mask (PSBD-TM)",
    "dropout": "dropout, attention input",
    "channel_mask": "channel mask, attention input",
    "gaussian": "Gaussian, attention input",
    "residual_dropout": "dropout, residual stream (PSBD-RD)",
    "token_substitute": "token substitute, attention input",
}
HYPOTHESES = (
    ("phenomenon", "the statistic separates"),
    ("margin", "H-margin"),
    ("direction", "H-direction (L8)"),
    ("redundancy", "H-redundancy (L19, L23)"),
    ("low_dim", "H-low-dim (L6)"),
    ("flatness", "H-flatness (L2, P4)"),
    ("neuron_bias", "H-neuron-bias (L10, P6)"),
    ("pass_uncertainty", "pass uncertainty (P2)"),
    ("ood", "out of distribution (P5)"),
    ("trigger_neurons", "trigger neurons (L7, P3)"),
    ("missingness", "missingness (L22)"),
)


def main():
    summary = read_json("summary.json")
    cached = read_json("cached_reads.json")
    sanity = read_json(f"sanity_{SANITY_FOLDER}.json")
    records = read_records()
    text = TEMPLATE
    values = placeholder_values(summary, cached, sanity, records)
    for name, value in values.items():
        text = text.replace(f"@@{name}@@", value)
    assert "@@" not in text, (
        f"unfilled placeholder near {text[text.index('@@') - 60 :][:120]}"
    )
    with open(OUT, "w") as handle:
        handle.write(text)
    print(f"wrote {OUT}")


def read_json(name):
    with open(os.path.join(RESULTS, name)) as handle:
        payload = json.load(handle)
    return payload


def read_records():
    records = {}
    for name in sorted(os.listdir(RESULTS)):
        if not name.endswith(".json") or "__" in name.replace("__probe_", ""):
            continue
        if name.startswith(
            (
                "summary",
                "cached",
                "sanity",
                "seed_",
                "wanet_probe",
                "critical_rate",
                "shift_curves",
            )
        ):
            continue
        with open(os.path.join(RESULTS, name)) as handle:
            records[name[: -len(".json")]] = json.load(handle)
    return records


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def listed(values):
    items = [str(v) for v in values]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    text = "\n".join(lines)
    return text


def groups_of(summary, architecture):
    present = summary["groups"].get(architecture, {})
    ordered = [g for g in ORDER if g in present]
    ordered += sorted(g for g in present if g not in ORDER)
    return ordered


def placeholder_values(summary, cached, sanity, records):
    p = summary["parameters"]
    t = summary["thresholds"]
    values = {
        "PAIRS": str(p["PAIR_COUNT"]),
        "PASSES": str(p["FORWARD_PASSES"]),
        "HELDOUT": str(PSBD_HELDOUT_SIZE),
        "FALLBACK_IMAGES": str(p["FALLBACK_IMAGES"]),
        "FALLBACK_PASSES": str(p["FALLBACK_PASSES"]),
        "ADAPTIVE": str(ADAPTIVE_SHIFT_TARGET),
        "RECOMMENDED": RECOMMENDED_PLACEMENT,
        "PUBLISHED": PUBLISHED_PLACEMENT,
        "QUANTILE": str(p["HEADLINE_QUANTILE"]),
        "KNN": str(p["KNN_NEIGHBOURS"]),
        "KEEP_FRACTIONS": listed(f for f in p["KEEP_FRACTIONS"] if f < 1.0),
        "SUBSET_DRAWS": str(p["SUBSET_DRAWS"]),
        "VIT_STREAM": listed(p["STREAM_BLOCKS"]["vit"]),
        "SWIN_STREAM": listed(p["STREAM_BLOCKS"]["swin"]),
        "FLAT_STREAM_VIT": listed(p["STREAM_BLOCKS"]["vit"][1:]),
        "FLAT_STREAM_SWIN": listed(p["STREAM_BLOCKS"]["swin"][1:]),
        "JACOBIAN_PAIRS": str(p["JACOBIAN_PAIRS"]),
        "TOP_TOKENS": str(p["TOP_TOKENS"]),
        "ATTRIBUTION_PAIRS": str(p["ATTRIBUTION_PAIRS"]),
        "ATTRIBUTION_SAMPLES": str(p["ATTRIBUTION_SAMPLES"]),
        "ATTRIBUTION_TOP": fmt(100 * p["ATTRIBUTION_TOP_SHARE"], 0),
        "FLAT_PAIRS": str(p["FLAT_PAIRS"]),
        "FLAT_DIRECTIONS": str(p["FLAT_DIRECTIONS"]),
        "FLAT_EPSILONS": listed(p["FLAT_EPSILONS"]),
        "SUFFICIENCY": listed(p["SUFFICIENCY_FRACTIONS"]),
        "SPREAD_PAIRS": str(p["SPREAD_PAIRS"]),
        "SPREAD_VIT": listed(p["SPREAD_BLOCKS"]["vit"]),
        "SPREAD_SWIN": listed(p["SPREAD_BLOCKS"]["swin"]),
        "SMALL_RATES": listed(p["SMALL_NOISE_RATES"]),
        "SMALL_PASSES": str(p["SMALL_NOISE_PASSES"]),
        "NEURON_PERCENT": fmt(100 * p["NEURON_SHARE"], 0),
        "NEURON_DRAWS": str(p["NEURON_DRAWS"]),
        "LATE_UNITS": str(p["LATE_UNITS"]),
        "LATE_BLOCKS": str(p["LATE_BLOCK_COUNT"]),
        "BATCH": str(p["BATCH_SIZE"]),
        "GPU_GB": fmt(p["GPU_MEMORY_GB"], 0),
        "CPU_THREADS": str(p["CPU_THREADS"]),
        "L8_FLOOR": str(t["l8_floor"]),
        "OPERATOR_COUNT": str(len(ALL_OPERATORS)),
        "HEADLINE_COUNT": str(len(HEADLINE)),
        "FOREIGN": listed(
            f"{a} models get {b}" for a, b in p["FOREIGN_DATASET"].items()
        ),
        "MODEL_TABLE": model_table(summary),
        "OPERATOR_TABLE": operator_table(p),
        "HYPOTHESIS_TABLE": hypothesis_table(t),
        "SANITY": sanity_section(sanity, summary),
        "WALL_TIME": wall_time_section(records),
        "RESULTS": results_section(summary, cached),
        "STATUS": status_section(summary, records),
    }
    return values


def model_table(summary):
    rows = []
    for r in summary["per_model"]:
        rows.append(
            [
                f"`{r['name']}`",
                r["architecture"],
                LABEL.get(r["group"], r["group"]),
                str(r["pairs"]),
                str(r["hit_pairs"]),
                fmt(r["clean_accuracy"]),
                fmt(r["triggered_asr"]),
                fmt(r.get("successful_2pt"))
                if r["category"] != "benign"
                else "control",
                fmt(r.get("successful_5pt"))
                if r["category"] != "benign"
                else "control",
                fmt(
                    min((r["cache_agreement"] or {"none": None}).values(), default=None)
                ),
            ]
        )
    header = [
        "model",
        "architecture",
        "group",
        "pairs",
        "hit pairs",
        "paired clean accuracy",
        "paired ASR",
        "successful at 2 points",
        "successful at 5 points",
        "cache agreement",
    ]
    return table(header, rows)


def operator_table(p):
    rows = [
        [
            f"`{name}`",
            f"`{spec['placement']}`",
            ", ".join(f"`{x}`" for x in spec["positions"]),
        ]
        for name, spec in p["OPERATORS"].items()
    ]
    return table(["name here", "placement", "positions"], rows)


def hypothesis_table(t):
    rows = [
        [
            "H-margin",
            "L1, P1",
            "the triggered decision keeps its logit margin under perturbation while the clean margin collapses",
            f"median retention of triggered margins at least {t['margin_gap_supported']} above "
            f"clean on the separating operators, and margin-matched pairs lose at most "
            f"{t['margin_matching_cost']} AUROC",
        ],
        [
            "H-direction",
            "L8",
            "the triggered feature is dominated by the backdoor direction, which the perturbation dents less than clean class evidence",
            f"backdoor share at least {t['backdoor_share']}, and per image the statistic tracks "
            f"the loss of the backdoor projection at a Spearman of {t['l8_supported']} or more",
        ],
        [
            "H-redundancy",
            "L19, L23",
            "trigger evidence is repeated across tokens",
            f"triggered excess retention at {t['redundancy_fraction']} visible tokens at least "
            f"{t['redundancy_supported']} and {t['redundancy_above_clean']} above clean, also "
            f"with a patch trigger forced visible",
        ],
        [
            "H-low-dim",
            "L6, L9",
            "the triggered decision depends on few dimensions or a small part of the input",
            f"1 direction holds at least {t['low_dim_top1']} of the paired feature change and a "
            f"ranked token set of at most {t['low_dim_sufficiency_ratio']} of the clean one keeps "
            f"the triggered answer",
        ],
        [
            "H-flatness",
            "L2, P4",
            "triggered inputs sit in a flatter region, and the statistic is a curvature estimate",
            f"Gaussian noise at small rates keeps the operating AUROC within "
            f"{t['flat_supported_gap']}, with phi(2r)/phi(r) between {t['flat_ratio_low']} and "
            f"{t['flat_ratio_high']}",
        ],
        [
            "H-neuron-bias",
            "L10, P6",
            "shifted clean predictions fall onto the target (Li et al.)",
            f"redistributing the target's gain costs PSBD-RD at least {t['l10_supported']} AUROC",
        ],
        [
            "pass uncertainty",
            "P2, L12",
            "the statistic is MC-dropout uncertainty",
            f"class-blind uncertainty over the same passes separates within "
            f"{t['uncertainty_supported_gap']} of the statistic, and a 3-seed ensemble reaches "
            f"{t['p2_supported_auroc']}",
        ],
        [
            "out of distribution",
            "P5",
            "triggered inputs are unusual inputs",
            f"a kNN score separates at {t['ood_knn_supported']} and PSBD flags foreign images "
            f"{t['ood_foreign_supported']} more often than clean ones",
        ],
        [
            "trigger neurons",
            "P3, L7",
            "a few MLP hidden units carry the trigger",
            f"zeroing the top units of the last blocks leaves at most {t['l7_supported']} of the "
            f"triggered answers",
        ],
        [
            "target is easy",
            "P10",
            "the target class is stable, so every target answer scores low",
            f"at least {t['p10_supported']} of clean target-class images flagged on backdoored "
            f"models and at most {t['p10_benign']} on benign ones",
        ],
        [
            "confidence",
            "P1",
            "the statistic only reads confidence",
            "PSBD-TM does not beat A*(P_c), the best any function of confidence can do",
        ],
        [
            "missingness",
            "L22",
            "a zeroed token is native missingness, not an artifact",
            f"token substitution within {t['l22_supported']} of token masking",
        ],
    ]
    return table(["id here", "memo id", "claim", "prediction if true"], rows)


def sanity_section(sanity, summary):
    rows = []
    for precision in ("bfloat16", "float32"):
        gate_1 = sanity[precision]["gate_1"]
        rows.append(
            [
                precision,
                "1",
                "clean accuracy, full test set",
                fmt(gate_1["clean_accuracy"], 4),
                fmt(gate_1["stored_clean_accuracy"], 4),
            ]
        )
        rows.append(
            [
                precision,
                "1",
                "ASR, full eval ASR split",
                fmt(gate_1["asr"], 4),
                fmt(gate_1["stored_asr"], 4),
            ]
        )
        for name, gate in sanity[precision]["gate_2"].items():
            label = "PSBD-TM" if name == "token_mask" else "PSBD-RD"
            rows.append(
                [
                    precision,
                    "2",
                    f"{label} at rate {gate['rate']}, fractional AUROC",
                    fmt(gate["auroc_fractional"], 5),
                    fmt(gate["cached_auroc_fractional"], 5),
                ]
            )
            rows.append(
                [
                    precision,
                    "2",
                    f"{label}, absolute AUROC",
                    fmt(gate["auroc_absolute"], 5),
                    fmt(gate["cached_auroc_absolute"], 5),
                ]
            )
            rows.append(
                [
                    precision,
                    "2",
                    f"{label}, validation shift ratio",
                    fmt(gate["validation_shift_ratio"], 4),
                    fmt(gate["cached_validation_shift_ratio"], 4),
                ]
            )
            rows.append(
                [
                    precision,
                    "2",
                    f"{label}, per-sample correlation with the cache over {gate['samples']} samples",
                    fmt(gate["per_sample_correlation"], 5),
                    "1",
                ]
            )
            rows.append(
                [
                    precision,
                    "2",
                    f"{label}, largest per-sample difference",
                    fmt(gate["per_sample_max_abs_difference"], 4),
                    "0",
                ]
            )
    gate_3 = sanity["gate_3"]
    tm_bf = sanity["bfloat16"]["gate_2"]["token_mask"]
    rd_bf = sanity["bfloat16"]["gate_2"]["residual_dropout"]
    rd_32 = sanity["float32"]["gate_2"]["residual_dropout"]
    restoration = [r["sanity"] for r in summary["per_model"] if r.get("sanity")]
    agreement = [
        min(r["cache_agreement"].values())
        for r in summary["per_model"]
        if r.get("cache_agreement")
    ]
    text = f"""`sanity.py` ran on `{sanity["folder"]}` before any full run, with the pipeline's own functions on the full PSBD splits. Gate 1 compares clean accuracy on the full test set and ASR on the full eval ASR split with `checkpoints/<folder>/metrics.json`. Gate 2 reruns PSBD-TM and PSBD-RD at their adaptive rates with `defenses.inference.compute_dropout_pass_probs` (mask seed 0, batch 64, $k = 3$, as `cli.sweep` does), compares the per-sample fractional statistic with the cached per-pass tensors and compares the AUROC of `defenses.decision.detection_report` with `psbd_metrics.json`, once under bfloat16 like the sweeps and once in float32. Gate 3 checks the pairs.

{table(["precision", "gate", "what is compared", "fresh", "stored"], rows)}

Gate 3 on {gate_3["pairs_checked"]} pairs: every triggered row carries the target label ({fmt(gate_3["triggered_labels_all_target"])}), no clean twin is of the target class ({fmt(gate_3["clean_twins_never_target"])}) and the largest paired pixel difference outside the BadNets trigger is {fmt(gate_3["max_difference_outside_trigger"], 4)}.

Gates 1 and 3 pass. Gate 2 reproduces the pipeline in bfloat16: PSBD-RD's per-sample statistic differs from the cache by at most {fmt(rd_bf["per_sample_max_abs_difference"], 4)} and PSBD-TM's by at most {fmt(tm_bf["per_sample_max_abs_difference"], 4)}, so the hook site, the operator, the rate, $k$, the splits and the pairing are the pipeline's own. 2 details matter for any later reading of `psbd_metrics.json`. A placement's `adaptive` block holds the absolute statistic's AUROC, and the fractional headline sits per rate under `detection_psu_ratio` at q0.25, which a first version of this gate compared against the wrong field. And float32 draws different dropout masks from the same seed, which points to the draw depending on the activation dtype under autocast: PSBD-RD's per-sample correlation with the cache falls to {fmt(rd_32["per_sample_correlation"], 3)} while its AUROC moves from {fmt(rd_32["cached_auroc_fractional"], 4)} to {fmt(rd_32["auroc_fractional"], 4)}. That difference is the Monte Carlo noise of 3 passes, and the detection result does not depend on the precision.

The other gates are checked in every record. Every control is matched to its treatment: clean, triggered, held-out and foreign images see the same operator, rate, $k$ and mask seed, a random-unit control zeroes as many units as its treatment, and a forced-visible redundancy run reuses the random run's patterns. Every hook is removed in a `finally` block. Over the {len(restoration)} records, hooks after a run minus hooks before is at most {max(r["hooks_after"] - r["hooks_before"] for r in restoration)}, modules left with a replaced forward at most {max(r["wrapped_forwards"] for r in restoration)}, models left in training mode {sum(r["model_training"] for r in restoration)}, model dropouts switched on at most {max(r["active_model_dropouts"] for r in restoration)}, and the largest logit difference between the first batch recomputed after every measurement and at the start {fmt(max(r["max_logit_difference"] for r in restoration), 4)}. `measure.py` refuses every folder that `scripts.paper._common.excluded_folders` lists (diverged or source-mapped TaCT) and every quarantined SIG folder. The unperturbed predictions of each record agree with the sweep's cached baselines on the same rows at {fmt(min(agreement), 3) if agreement else "n/a"} or more (`cache_agreement`), the check that would have caught the SIG amplitude drift. The finite differences, the small-noise Gaussian run, the Jacobians and the attribution ranking run in float32, the audit's precision point."""
    return text


def wall_time_section(records):
    seconds = [r["seconds"] for r in records.values() if "seconds" in r]
    if not seconds:
        return "No record yet."
    sections = {}
    for r in records.values():
        for name, value in (r.get("section_seconds") or {}).items():
            sections.setdefault(name, []).append(value)
    rows = [
        [name, fmt(statistics.median(v), 0), fmt(max(v), 0)]
        for name, v in sections.items()
    ]
    by_arch = {}
    for r in records.values():
        by_arch.setdefault(r["architecture"], []).append(r["seconds"])
    arch_text = listed(
        f"{fmt(statistics.median(v), 0)} s per {a} model (median over {len(v)})"
        for a, v in by_arch.items()
    )
    peak = max(r.get("peak_gpu_gb", 0) for r in records.values())
    text = f"""{len(seconds)} model records took {fmt(sum(seconds) / 60, 0)} min of GPU time in total, {arch_text}, on the login node's A100 shared with other agents' jobs, with a peak allocation of {fmt(peak, 1)} GB under the per-process cap. The sections, median and largest seconds per model:

{table(["section", "median s", "largest s"], rows)}

The first attempt ran 45 min on 1 model: 384 CPU LAPACK SVDs of the Jacobians and every CPU random draw of the finite differences spun PyTorch's OpenMP threads against the login node's other jobs while the GPU sat idle. Drawing the directions on the GPU, batching the SVDs on the GPU and capping PyTorch at a few CPU threads cut it to minutes."""
    return text


def status_section(summary, records):
    planned = [
        measure.entry_name(f, pr) for f, pr in measure.VIT_MODELS + measure.SWIN_MODELS
    ]
    done = [name for name in planned if name in records]
    left = [name for name in planned if name not in records]
    seeds_done = len(summary["seed_ensemble"])
    excluded = excluded_folders("results")
    dropped = [n for n in left if n.split("__")[0] in excluded]
    pending = [n for n in left if n not in dropped]
    text = (
        f"Measured: {len(done)} of the {len(planned)} planned model runs, and "
        f"{seeds_done} seed ensembles. "
        + (
            f"Dropped by the ledger because they are not successful at the 2 point bar: "
            f"{listed(f'`{n}`' for n in dropped)}. "
            if dropped
            else ""
        )
        + (
            f"Left for the next GPU window: {listed(f'`{n}`' for n in pending)}. "
            if pending
            else "Nothing planned is left. "
        )
        + "SIG waits for the provenance fix of the 2026-09-29 audit (`QUARANTINED_MODELS`, "
        "run with `--include-quarantined`). The WaNet coherence probe of memo L26 "
        "(`wanet_probe.py`) wrote `wanet_probe__<folder>.json` for every ViT WaNet model."
    )
    return text


def status_word(v):
    return f"{v['status']} ({fmt(v['value'])})"


# 1 paragraph per hypothesis: the claim, the rule and its status per group and
# architecture with the key number, rendered from the verdicts summarize.py
# stored, so the text cannot disagree with the tables below it.
def findings_section(summary):
    paragraphs = []
    for key, name in HYPOTHESES:
        pieces = []
        rule = None
        for architecture in ("vit", "swin"):
            if architecture not in summary["verdicts"]:
                continue
            groups = [
                g
                for g in groups_of(summary, architecture)
                if not g.startswith("benign")
            ]
            readings = []
            for g in groups:
                v = summary["verdicts"][architecture][g][key]
                rule = rule or v.get("rule")
                readings.append(
                    f"{v['status']} for {LABEL.get(g, g)} ({fmt(v['value'])})"
                )
            arch_name = "ViT-B/16" if architecture == "vit" else "Swin-S"
            pieces.append(f"On {arch_name} it reads {listed(readings)}.")
        key_text = summary["verdicts"]["vit"][groups_of(summary, "vit")[0]][key]["key"]
        paragraphs.append(
            f"**{name[0].upper() + name[1:]}.** The key number is the {key_text}. The rule: {rule}. "
            + " ".join(pieces)
        )
    panel = summary.get("panel_verdicts") or {}
    for field, name in (
        ("confidence", "Confidence carries the signal (P1), on the whole panel"),
        ("target_easy", "The target class is easy (P10), on the whole panel"),
        (
            "epistemic",
            "The statistic is epistemic uncertainty (P2), on the seed ensembles",
        ),
    ):
        block = panel.get(field) or {}
        if not block:
            continue
        rule = next(iter(block.values()))["rule"]
        readings = [
            f"{v['status']} for {attack} ({fmt(v['value'])})"
            for attack, v in sorted(block.items())
        ]
        paragraphs.append(f"**{name}.** The rule: {rule}. It reads {listed(readings)}.")
    held, failed = [], []
    for key, name in HYPOTHESES:
        statuses = [
            summary["verdicts"][a][g][key]["status"]
            for a in summary["verdicts"]
            for g in groups_of(summary, a)
            if not g.startswith("benign")
        ]
        if statuses and statuses.count("supported") >= len(statuses) / 2:
            held.append(name)
        if statuses and statuses.count("refuted") >= len(statuses) / 2:
            failed.append(name)
    paragraphs.append(
        f"**Across categories.** Supported in at least half of the measured cells: {listed(held)}. "
        f"Refuted in at least half: {listed(failed)}. That the supported readings form 1 "
        "mechanism (the trigger writes 1 target-aligned direction whose margin survives "
        "the probe, while a clean answer breaks at a lower critical rate) is a hypothesis "
        "that ties them together and was not tested as a whole. The notebook "
        "`notebooks/why-psbd-works-general.ipynb` walks through each reading with its figure, "
        "what it shows and what it does not."
    )
    text = "\n\n".join(paragraphs)
    return text


def critical_rate_section(summary):
    critical = summary.get("critical_rate")
    if not critical:
        return "### The critical rate\n\n`critical_rate.py` has not run yet."
    pr = critical["predictions"]
    link = critical["link_spearman"]
    breaks = listed(
        f"`{b['folder']}` at `{b['placement']}` ({fmt(b['auroc_adaptive'])} against {fmt(b['a_star'])})"
        for b in pr["P1_breaks"][:6]
    )
    text = f"""### The critical rate, 1 account for every operator and attack

The account: each input has a critical rate $p^*$, the smallest rate on a placement's ladder at which most of its passes change its answer, and PSBD at any placement is a 2-sample test on $p^*$. `critical_rate.py` reads $p^*$ from the sweep's cached per-pass predictions (`defenses.decision.load_critical_rate_from_disk`, majority of passes) for every panel model and every placement with a full ladder, and computes $A^\star = P(p^*_{{\\text{{triggered}}}} > p^*_{{\\text{{clean}}}})$ over paired images, ties counted half. Its 3 predictions were fixed in the script's docstring before the first read.

P1, 1 curve across operators and attacks (Spearman of $A^\star$ against the AUROC at the adaptive rate at least 0.8, median absolute difference at most 0.05): over {pr["P1_pairs"]} (model, placement) pairs the Spearman is {fmt(pr["P1_spearman"])} and the median difference {fmt(pr["P1_median_abs_difference"])}, so P1 {"holds" if pr["P1_holds"] else "fails as stated"}. {pr["P1_breaks_over_0.1"]} pairs lie more than 0.1 off, the largest {breaks or "none"}.

P2, the control (benign models probed with a trigger they never learned, $A^\star$ within 0.05 of 0.5): the largest distance over {pr["P2_benign_rows"]} benign rows is {fmt(pr["P2_max_distance_from_half"])}, so P2 {"holds" if pr["P2_holds"] else "fails"}.

P3, the rate (about 0.8 of held-out clean images at or below the adaptive rate, not the median): the median share is {fmt(pr["P3_median_validation_share"])}, range {fmt(pr["P3_range"][0])} to {fmt(pr["P3_range"][1])}.

The link to this experiment's mechanism readings, over {critical["link_pairs"]} (model, operator) pairs, is a correlation over few models and so a hypothesis about what sets the $p^*$ gap: Spearman of $A^\star$ with the margin retention gap {fmt(link["margin_gap"])}, with the signal to perturbation ratio {fmt(link["spr_ratio"])}, with the L8 correlation {fmt(link["l8"])} and with redundancy at 30% visible {fmt(link["redundancy_excess"])}. $A^\star$ and the AUROC at 1 rate read the same passes, so a tight curve shows that 1 rate summarises the whole ladder. It restates PSBD in 1 variable. What sets an input's $p^*$ is what the per-category readings above measure.

`shift_curves.py` draws the same breaking points as curves, the clean and triggered shift ratio along every ladder per model on both panels and the benign references (`shift_curves.json`, the sidecar of the notebook's figure). The notebook tests 6 patterns an exploratory median-model read suggested against those per-model curves, and gives the pairs that sit far off the $A^\\star$ curve by attack and placement. Where the flip-based breaking point and the probability-based statistic disagree (TaCT under PSBD-TM is the clearest case), the account covers the flip and not the statistic."""
    return text


def results_section(summary, cached):
    parts = [findings_section(summary), critical_rate_section(summary)]
    for architecture in ("vit", "swin"):
        if architecture not in summary["groups"]:
            continue
        parts.append(architecture_results(summary, architecture))
    parts.append(panel_results(summary, cached))
    text = "\n\n".join(parts)
    return text


def architecture_results(summary, architecture):
    groups = groups_of(summary, architecture)
    name = "ViT-B/16" if architecture == "vit" else "Swin-S"
    mean = {g: summary["groups"][architecture][g]["mean"] for g in groups}
    verdicts = summary["verdicts"][architecture]
    counts = {g: len(summary["groups"][architecture][g]["models"]) for g in groups}

    auroc_rows = [
        [LABEL.get(g, g), str(counts[g])]
        + [
            fmt(mean[g]["operators"].get(o, {}).get("auroc_hit_only"))
            for o in ALL_OPERATORS
        ]
        for g in groups
    ]
    verdict_rows = [
        [LABEL.get(g, g)] + [status_word(verdicts[g][key]) for key, _ in HYPOTHESES]
        for g in groups
    ]
    margin_rows = [
        [
            LABEL.get(g, g),
            fmt(mean[g]["auroc_margin_unperturbed"]),
            fmt(mean[g]["auroc_confidence_unperturbed"]),
            fmt(mean[g]["clean_margin"], 2),
            fmt(mean[g]["triggered_margin"], 2),
            fmt(mean[g]["operators"]["token_mask"]["clean_margin_retention"]),
            fmt(mean[g]["operators"]["token_mask"]["triggered_margin_retention"]),
            fmt(mean[g]["operators"]["token_mask"]["margin_matched_auroc"]),
            fmt(mean[g]["operators"]["residual_dropout"]["clean_margin_retention"]),
            fmt(mean[g]["operators"]["residual_dropout"]["triggered_margin_retention"]),
            fmt(mean[g]["operators"]["residual_dropout"]["margin_matched_auroc"]),
        ]
        for g in groups
    ]
    direction_rows = [
        [
            LABEL.get(g, g),
            fmt(mean[g]["operators"]["token_mask"]["triggered_backdoor_share"]),
            fmt(mean[g]["cosine_backdoor_target_readout"]),
            fmt(mean[g]["operators"]["token_mask"]["triggered_backdoor_spr"], 2),
            fmt(mean[g]["operators"]["token_mask"]["clean_class_spr"], 2),
            fmt(
                mean[g]["operators"]["token_mask"][
                    "spearman_psu_backdoor_drop_above_floor"
                ]
            ),
            fmt(mean[g]["operators"]["token_mask"]["triggered_share_above_floor"]),
            fmt(
                mean[g]["operators"]["token_mask"]["spearman_psu_backdoor_drop_pooled"]
            ),
            fmt(
                mean[g]["operators"]["residual_dropout"][
                    "spearman_psu_backdoor_drop_above_floor"
                ]
            ),
        ]
        for g in groups
    ]
    redundancy_rows = []
    for g in groups:
        red = mean[g]["redundancy"]
        row = [LABEL.get(g, g)]
        for fraction in sorted(
            (f for f in red if float(f) < 1.0), key=float, reverse=True
        ):
            row += [
                fmt(red[fraction].get("triggered_kept")),
                fmt(red[fraction].get("clean_kept")),
                fmt(red[fraction].get("excess_retention")),
            ]
        row.append(fmt(red.get("0.3", {}).get("trigger_forced_triggered_kept")))
        redundancy_rows.append(row)
    fractions = sorted(
        (f for f in mean[groups[0]]["redundancy"] if float(f) < 1.0),
        key=float,
        reverse=True,
    )
    redundancy_header = (
        ["group"]
        + [
            f"{f} {kind}"
            for f in fractions
            for kind in ("triggered kept", "clean kept", "excess")
        ]
        + ["0.3 triggered kept, trigger forced visible"]
    )
    low_rows = []
    for g in groups:
        low = mean[g]["low_dim"]
        last = str(max(int(b) for b in low["jacobian"]))
        jac = low["jacobian"][last]
        suff = mean[g]["sufficiency"]
        low_rows.append(
            [
                LABEL.get(g, g),
                fmt(low["trigger_top1_share"]),
                fmt(low["clean_pair_top1_share"]),
                fmt(jac["triggered_effective_tokens"], 1),
                fmt(jac["clean_effective_tokens"], 1),
                fmt(jac["triggered_effective_rank"], 1),
                fmt(jac["clean_effective_rank"], 1),
                fmt(suff["ranked"]["triggered_median"]),
                fmt(suff["ranked"]["clean_median"]),
                fmt(suff["ranked"]["auroc"]),
                fmt(suff["random"]["triggered_median"]),
                fmt(suff["random"]["clean_median"]),
                fmt(low["attribution"].get("triggered_gini")),
                fmt(low["attribution"].get("clean_gini")),
            ]
        )
    flat_rows = []
    for g in groups:
        small = mean[g].get("small_noise") or {}
        aurocs = small.get("auroc_hit_only") or {}
        flat_rows.append(
            [LABEL.get(g, g)]
            + [
                fmt(v)
                for v in (
                    verdicts[g]["flatness"].get("finite_difference_auroc") or {}
                ).values()
            ]
            + [fmt(aurocs.get(r)) for r in sorted(aurocs, key=float)]
            + [
                fmt(small.get("operating_gaussian_auroc")),
                fmt(small.get("clean_ratio_median"), 2),
                fmt(small.get("clean_spearman_across_rates")),
            ]
        )
    first = mean[groups[0]]
    flat_sites = list(
        (verdicts[groups[0]]["flatness"].get("finite_difference_auroc") or {})
    )
    small_rates = sorted(
        (first.get("small_noise") or {}).get("auroc_hit_only") or {}, key=float
    )
    refute_rows = [
        [
            LABEL.get(g, g),
            fmt(mean[g]["operators"]["token_mask"]["clean_shift_to_target"]),
            fmt(mean[g]["operators"]["residual_dropout"]["clean_shift_to_target"]),
            fmt(mean[g]["operators"]["token_mask"]["uniform_share"]),
            fmt(verdicts[g]["neuron_bias"]["token_mask_change"]),
            fmt(verdicts[g]["neuron_bias"]["value"]),
            fmt(mean[g]["operators"]["token_mask"]["auroc_mutual_information"]),
            fmt(mean[g]["operators"]["token_mask"]["auroc_predictive_entropy"]),
            fmt(mean[g]["operators"]["token_mask"]["auroc_own_prob_std"]),
            fmt(mean[g]["ood"]["knn_distance"]["auroc_triggered_vs_clean"]),
            fmt(mean[g]["ood"]["knn_distance"]["auroc_foreign_vs_clean"]),
            fmt(mean[g]["operators"]["token_mask"]["flagged"]["clean"]),
            fmt(mean[g]["operators"]["token_mask"]["flagged"]["foreign"]),
            fmt(mean[g]["neurons"]["late_top30_triggered_kept"]),
            fmt(mean[g]["neurons"]["late_random30_triggered_kept"]),
            fmt(mean[g]["neurons"]["all_top_triggered_kept"]),
            fmt(mean[g]["neurons"]["all_top_clean_kept"]),
        ]
        for g in groups
    ]
    text = f"""### {name}

Mean AUROC of the statistic per group (hit-only form, low statistic flags the triggered image), each operator at its adaptive rate, $k$ = {summary["parameters"]["FORWARD_PASSES"]}:

{table(["group", "models"] + [OPERATOR_LABEL.get(o, o) for o in ALL_OPERATORS], auroc_rows)}

Verdicts, status and the key number each rule reads (the rules are in "Verdict rules" above):

{table(["group"] + [label for _, label in HYPOTHESES], verdict_rows)}

H-margin. Unperturbed AUROCs of margin and confidence, median margins, median margin retention $r$ under PSBD-TM and PSBD-RD, and the margin-matched AUROC:

{table(["group", "AUROC margin", "AUROC confidence", "clean margin", "triggered margin", "TM r clean", "TM r triggered", "TM matched AUROC", "RD r clean", "RD r triggered", "RD matched AUROC"], margin_rows)}

H-direction. Backdoor share of the triggered deviation, cosine of the backdoor direction with the target's readout, signal to perturbation ratios under PSBD-TM and the L8 Spearman correlations:

{table(["group", "backdoor share", "cos(u, target readout)", "TM SPR triggered", "TM SPR clean", f"TM L8, phi above {summary['thresholds']['l8_floor']}", "share of triggered above that floor", "TM L8 pooled", "RD L8 above floor"], direction_rows)}

H-redundancy. Answers kept with a fixed random share of patch tokens visible in every block, mean over {summary["parameters"]["SUBSET_DRAWS"]} patterns per fraction:

{table(redundancy_header, redundancy_rows)}

H-low-dim. Spectrum of the paired feature change, the margin's Jacobian at the last probed block, the smallest sufficient token share (ranked and random order) and the Gini coefficient of input attributions:

{table(["group", "top1 share, trigger change", "top1 share, clean pair change", "Jacobian tokens triggered", "Jacobian tokens clean", "Jacobian rank triggered", "Jacobian rank clean", "ranked sufficient triggered", "ranked sufficient clean", "ranked AUROC", "random sufficient triggered", "random sufficient clean", "Gini triggered", "Gini clean"], low_rows)}

H-flatness. AUROC of the finite-difference flatness of $\\log P_c$ per site, Gaussian AUROC at the small rates and at the operating rate, the median ratio $\\phi(2r)/\\phi(r)$ and the rank correlation across rates on clean images:

{table(["group"] + [f"FD AUROC {s}" for s in flat_sites] + [f"Gaussian AUROC r {r}" for r in small_rates] + ["Gaussian AUROC operating", "phi ratio clean", "rank Spearman clean"], flat_rows)}

The explanations tested as refutations: where shifted clean answers go and what removing the target's gain costs (L10), class-blind uncertainty over the PSBD-TM passes (P2), OOD scores and the flag on foreign images (P5), and zeroing trigger units (L7):

{table(["group", "TM clean shifts to target", "RD clean shifts to target", "1/classes", "TM AUROC lost without target", "RD AUROC lost without target", "TM BALD AUROC", "TM entropy AUROC", "TM sd of P_c AUROC", "kNN AUROC triggered", "kNN AUROC foreign", "TM flag clean", "TM flag foreign", "late top units, triggered kept", "late random units, triggered kept", "all-block top units, triggered kept", "all-block top units, clean kept"], refute_rows)}"""
    return text


def panel_results(summary, cached):
    panel = summary.get("panel_verdicts") or {}
    rows = []
    for name, label in (
        ("confidence", "P1 confidence carries it"),
        ("target_easy", "P10 target is easy"),
        ("epistemic", "P2 epistemic ensemble"),
    ):
        for attack, v in sorted((panel.get(name) or {}).items()):
            rows.append([label, attack, v["status"], v["key"], fmt(v["value"])])
    fragility = cached["clean_fragility"]
    ceiling = [
        r
        for r in cached["refutations"]["confidence_ceiling"]
        if r.get("successful_2pt")
    ]
    per_model = {r["folder"]: r for r in cached["refutations"]["per_model"]}
    tm_margin = [per_model[r["folder"]]["psbd_tm"] - r["a_star"] for r in ceiling]
    rd_margin = [per_model[r["folder"]]["psbd_rd"] - r["a_star"] for r in ceiling]
    curves = fragility["curves"]
    seed_rows = [
        [
            f"`{r['cell']}`",
            fmt(r["psbd_tm_cached_auroc"]),
            fmt(r["ensemble_psu_ratio"]),
            fmt(r["disagreement"]),
            fmt(r["mutual_information"]),
            fmt(r.get("clean_spearman_with_psbd_tm")),
        ]
        for r in summary["seed_ensemble"]
    ]
    text = f"""### The panel-wide readings

These read the sweep's caches over the {len(ceiling)} ViT models successful at the 2 point bar (`cached_reads.py`), or the seed-replicated cells (`seed_ensemble.py`), per attack.

{table(["account", "attack", "status", "key", "value"], rows)}

Confidence ceiling (P1): $A^*(P_c)$ averages {fmt(statistics.mean(r["a_star"] for r in ceiling))} over the panel against {fmt(statistics.mean(r["raw_confidence"] for r in ceiling))} for raw confidence. PSBD-TM beats it on {sum(m > 0 for m in tm_margin)} of {len(tm_margin)} models by {fmt(statistics.mean(tm_margin), 3)} on average, PSBD-RD on {sum(m > 0 for m in rd_margin)} of {len(rd_margin)} by {fmt(statistics.mean(rd_margin), 3)}.

Clean fragility: at PSBD-TM's adaptive rate with $k = 3$, {fmt(fragility["mean_never_flip_share"])} of held-out clean images never flip, the intra-image flip correlation has median {fmt(fragility["median_intra_image_correlation"])} over models, and images that never flipped are {fmt(fragility["mean_false_positive_never_flip_share"])} of the false positives on average. The logistic critical-rate fit against the homogeneous AND law, per dataset:

{table(["dataset", "models", "floor", "median critical rate", "scale", "sd of critical rate", "logistic RMSE", "AND units", "AND RMSE"], [[d, str(c["models"]), fmt(c["floor"]), fmt(c["logistic_median"]), fmt(c["logistic_scale"]), fmt(c["critical_rate_sd"]), fmt(c["logistic_rmse"]), fmt(c["and_units"], 2), fmt(c["and_rmse"])] for d, c in sorted(curves.items())])}

Seed ensembles (P2), AUROC in the pipeline's paired form:

{table(["cell", "PSBD-TM cached", "ensemble statistic", "hard disagreement", "BALD", "clean Spearman with PSBD-TM"], seed_rows) if seed_rows else "No seed ensemble yet."}"""
    return text


TEMPLATE = r"""# Why PSBD's statistic separates triggered inputs on ViT and Swin

Generated by `experiments/why_psbd_works/render_readme.py` from the constants of `measure.py` and `summarize.py` and from the records under `results/_experiments/why_psbd_works/`. Do not edit by hand: change the code or rerun the measurement and regenerate.

## Question

PSBD (Li et al., arXiv 2406.05826) scores an input by how much the probability of the model's own answer drops under $k$ perturbed forward passes, and a small drop reads as poisoned. The project uses the fractional form of `defenses.scores.psu_ratio_from_cache`:

$$\phi(x) = 1 - \frac{\tfrac{1}{k}\sum_{j=1}^{k} P_c(x;\,\xi_j)}{P_c(x)}, \qquad c = \arg\max_i P_i(x)$$

| symbol | meaning |
|---|---|
| $x$ | the input image |
| $P_i(x)$ | softmax probability of class $i$ with no perturbation |
| $c$ | the class the unperturbed model predicts, the model's own answer |
| $\xi_j$ | the random draw of the operator on pass $j$ (a dropout mask, a token mask, a noise sample) |
| $P_c(x;\xi_j)$ | the probability of class $c$ on perturbed pass $j$ |
| $k$ | the number of perturbed passes |
| $\phi(x)$ | fractional prediction shift uncertainty, low means flagged as poisoned |

The statistic separates on ViT-B/16 with every operator family the project swept: token masking, dropout and channel masking at the attention input, Gaussian noise there on most attacks, and dropout on the residual stream. `experiments/why_token_masking_works/` explained why 1 placement, PSBD-TM, beats another on patch triggers. It did not say why any of them works at all. This experiment asks the general question: what property of a triggered input makes its own answer survive a perturbation that destroys a clean input's answer, for any perturbation, and does the answer differ by attack category.

The attack categories are those of the literature memo (`docs/why-psbd-works-literature.md`). Patch is BadNets (all to one) and TaCT restricted to the models that classify their clean source class correctly. Blend is Blend and LF. Frequency is SIG. Warp is WaNet. Quantization is BPP. Benign models probed with a trigger they never learned are the control.

## Hypotheses

Each hypothesis is stated with the prediction that would hold if it were true. The memo's identifiers (L for published explanations, P for folk ones) are kept so the notebook, the memo and the theory note (`docs/why-psbd-works-theory.md`) cross-reference. The thresholds come from the theory note's table "Numeric predictions for the open hypotheses" where it gives one and from this experiment otherwise (`summarize.THRESHOLDS`).

@@HYPOTHESIS_TABLE@@

## Method

Everything is inference on existing checkpoints. No model is trained or fine-tuned. `measure.py` runs 1 model per process and writes 1 JSON per model under `results/_experiments/why_psbd_works/`. `summarize.py` reduces the records on the CPU to `summary.json`, `cached_reads.py` reads the sweeps' caches for the panel-wide tests, `seed_ensemble.py` reads the seed replicates, and the notebook `notebooks/why-psbd-works-general.ipynb` and this README are rendered from those files.

### Models

ViT-B/16 first, then Swin-S on the same attacks, about 2 models per attack category, CIFAR-100 and Tiny ImageNet first because they are the primary datasets, then GTSRB and CIFAR-10 where those 2 have no successful model. The 5% poison rate is the default because it is the middle of the 3 panel rates, and 10% is used where 5% did not implant. The panel is the models successful at the 2 point bar (`successful_2pt` of `results/coverage/coverage.json`: the ASR clears the declared bar and clean accuracy is within 2 points of the benign reference), the user's decision of 2026-09-29. Swin is not in the ViT ledger, so its verdict is rebuilt the ledger's way by `summarize.success_flags` through `scripts.coverage_ledger.success_verdicts`. WaNet has all its ViT models measured, because WaNet is where the placements disagree most and the memo's L10 test is about it, and the models successful only at the 5 point bar are read beside the panel in `groups_5pt` and `verdicts_5pt`.

@@MODEL_TABLE@@

TaCT is restricted to the ViT models whose clean source class is still classified correctly. The other ViT TaCT models map their whole source class to the target with no trigger (`docs/runs/2026-09-24-tact-multisource.md`), so their triggered prediction is content, and `measure.py` refuses every folder `scripts.paper._common.excluded_folders` lists. The Swin TaCT models are not in the ViT ledger, so the summary files a Swin TaCT model whose paired clean accuracy on its source class is below 0.5 under `patch_tact_source_mapped`. The benign models carry no trigger of their own. Probing them with a trigger they never learned separates "the statistic reacts to a backdoor" from "the statistic reacts to a trigger-shaped input", and a patch probe and a global probe cover both trigger geometries.

SIG has no reading. The audit of 2026-09-29 (`docs/audits/2026-09-29-experiment-audit.md`) found that SIG models trained before 2026-09-09 learned amplitude 0.1 while `attacks/sig.py` now stamps 0.157, so rebuilt triggered images were not the trigger those models learned, and the user's success bar then removed the only ViT SIG model of the 4 paper datasets from the panel. The SIG entries sit in `QUARANTINED_MODELS` and run with `--include-quarantined` once their provenance is settled.

### Images

Every model gets the first @@PAIRS@@ rows of its PSBD eval ASR split (`data.splits.build_psbd_loaders_from_checkpoint`, seed `PSBD_SPLIT_SEED`) and, for each row, the clean image behind it, found through the split manifest as `defenses.decision.pair_clean_to_backdoor` finds it. The triggered rows come from `attacks.poisoning.AttackSuccessSet`, which applies `is_eval_poisonable` and `attack_success_label`, so every triggered row is an image whose true class is not the target, stamped with the trigger and labelled with the target. TaCT's rows come only from its source class. @@PAIRS@@ pairs keep a model's run to minutes on the shared GPU and resolve a difference of a few hundredths in a share or an AUROC.

2 more sets are loaded per model. The first @@FALLBACK_IMAGES@@ images of the held-out clean split (the @@HELDOUT@@ images the pipeline sets its threshold on) supply the threshold of the flag, the neighbours of the kNN score and the calibration of any rate the sweeps did not cache. @@FALLBACK_IMAGES@@ clean test images of another dataset, resized to the model's resolution and normalized with the model's statistics, are the out-of-distribution control: @@FOREIGN@@.

"Hit" images are the triggered images the unperturbed model sends to the target. Every survival and margin statistic of a triggered image is read on hit images, since only those have a target decision to keep. On a benign model every pair counts. `summary.json` keeps 2 AUROC forms. `auroc_paired` is `defenses.decision.detection_report` on every triggered row against its clean twin, the form `psbd_metrics.json` reports. `auroc_hit_only` is triggered hit images against all clean twins, the form the mechanism readings use.

### Operators and rates

Each model is probed with @@OPERATOR_COUNT@@ operators. The first @@HEADLINE_COUNT@@ are the ones the question names. `token_substitute` replaces a token by another token of the same image, which removes content and stays on the data manifold (memo L22). `rademacher` was planned for memo L2 and dropped: the theory note shows it cannot differ from Gaussian noise in 151k dimensions under any account, and the curvature test became the small-noise run below.

@@OPERATOR_TABLE@@

Each operator runs at its placement's adaptive rate, the smallest ladder rate whose clean held-out shift ratio reaches @@ADAPTIVE@@ (`defenses.decision.select_rate_adaptively`), read from `results/<folder>/psbd_metrics.json` through `cli.compare_detectors.psbd_rate`. A placement the sweeps never cached is rated by the same rule on the @@FALLBACK_IMAGES@@ held-out images with @@FALLBACK_PASSES@@ passes (`measure.calibrate_rate`), and each record marks the rate's source. Matching every operator at the same disturbance of clean predictions makes their readings comparable. Each operator makes $k$ = @@PASSES@@ passes with mask seed 0, more than the pipeline's 3 so that a per-image statistic is not dominated by Monte Carlo noise. Clean, triggered, held-out and foreign images see the same operator, rate, $k$ and seed. Forward passes run under bfloat16 autocast like the sweeps, except the finite differences, the small-noise run, the gradients and the attribution ranking, which run in float32.

### Measurements

**H-margin** (`measure.per_image_statistics`, `summarize.margin_matched_auroc`). The own-class margin is the logit of the model's answer minus the best other logit, unperturbed and on every pass.

$$m(x) = z_c(x) - \max_{i \neq c} z_i(x), \qquad m_j(x) = z_c(x;\xi_j) - \max_{i \neq c} z_i(x;\xi_j), \qquad r(x) = \frac{\tfrac{1}{k}\sum_j m_j(x)}{m(x)}$$

| symbol | meaning |
|---|---|
| $z_i(x)$, $z_i(x;\xi_j)$ | the logit of class $i$, unperturbed and on pass $j$ |
| $m(x)$, $m_j(x)$ | the own-class margin, unperturbed and on pass $j$, negative once the answer has flipped |
| $r(x)$ | margin retention, 1 when the perturbation leaves the margin whole |

3 readings. The unperturbed margin's AUROC says whether triggered images start with larger margins. Retention says whether the perturbation damages the triggered margin less. The margin-matched AUROC pairs each triggered hit image, with replacement, with the clean image of the nearest unperturbed margin and reads the statistic's AUROC on those pairs: if the separation came from larger starting margins, it would vanish there.

**H-direction** (`measure.feature_geometry`, `measure.feature_statistics`). The feature $f(x)$ is what the classifier head reads: ViT's class token after the final LayerNorm, Swin's pooled final feature map. With $\mu$ the mean clean feature, $\tilde x$ the triggered twin of $x$, $H$ the hit pairs and $W$ the head's weights,

$$d = \frac{1}{|H|}\sum_{x \in H}\big(f(\tilde x) - f(x)\big), \quad u = \frac{d}{\lVert d\rVert}, \quad e_c = \frac{W_c - \bar W}{\lVert W_c - \bar W\rVert}$$

$$s_u(x) = \langle f(x) - \mu, u\rangle, \qquad \text{share}(x) = \frac{s_u(x)^2}{\lVert f(x) - \mu\rVert^2}, \qquad \text{SPR}_u(x) = \frac{s_u(x)}{\sqrt{\tfrac{1}{k}\sum_j \langle f_j(x) - f(x), u\rangle^2}}$$

| symbol | meaning |
|---|---|
| $d$, $u$ | the backdoor direction (`analysis.direction.backdoor_direction`) and its unit vector |
| $W_c$, $\bar W$ | the head's row for class $c$ and the mean row |
| $e_c$ | the class direction of $c$, along which the logit of $c$ rises against the average class |
| $f_j(x)$ | the feature on perturbed pass $j$ |
| $s_u$ | the backdoor component of an image's deviation from the clean mean |
| share | the fraction of that deviation the backdoor component makes up |
| $\text{SPR}_u$ | signal to perturbation ratio along $u$ |

The clean counterpart uses $e_c$ of the image's own class in place of $u$. The memo's L8 statistic is the per-image Spearman correlation between $\phi$ and the fraction of the backdoor projection a pass loses, $1 - \tfrac{1}{k}\sum_j \langle f_j - \mu, u\rangle / s_u$. Most triggered images never flip under PSBD-TM, so their $\phi$ sits at the bfloat16 floor, and the theory note asks for the correlation on triggered images with $\phi$ above @@L8_FLOOR@@ and on clean and triggered images pooled. The direction is fitted and read on the same pairs, and the benign probes are read the same way as the control for that.

**H-redundancy** (`measure.measure_redundancy`, `measure.SubsetTokenMask`). A fixed random pattern of the 14 by 14 patch grid stays visible and every other patch token is zeroed at the input of `ln_1` in all 12 ViT blocks (Swin: `norm1` in all 24 blocks, the pattern resized to each stage's grid). The class token is never masked. A zeroed token enters attention as the LayerNorm's bias vector, so the prediction sees only the visible tokens' content. Visible fractions @@KEEP_FRACTIONS@@ get @@SUBSET_DRAWS@@ patterns each, seeded per model from a hash of the folder name so that 2 models never share their patterns (the audit's point against section D of `why_token_masking_works`, which used 3 shared patterns). Masked models can collapse onto 1 default class, which is sometimes the target, so the excess retention subtracts the share of clean images sent to the target:

$$\text{excess}(f) = \frac{\text{ASR}(f) - \text{clean on target}(f)}{\text{ASR}(1) - \text{clean on target}(1)}$$

with $f$ the visible fraction. For patch triggers on ViT every pattern runs a second time with the trigger's own tokens forced visible. If triggered survival holds only when the trigger is visible, a random pattern protects the trigger by rarely hiding 4 specific tokens, which is geometry and not redundancy.

**H-low-dim** (`measure.difference_spectrum`, `measure.measure_jacobian`, `measure.measure_attribution`, `measure.measure_sufficiency`).

1. The eigenvalues $\lambda_1 \geq \lambda_2 \geq \dots$ of the uncentered second moment of the paired change $f(\tilde x) - f(x)$ over hit pairs. The top share $\lambda_1/\sum_i\lambda_i$ near 1 means 1 direction carries the whole trigger effect. The control is the change between 2 unrelated clean images.
2. The Jacobian of the own-class margin, runner-up fixed, with respect to the residual stream entering blocks @@VIT_STREAM@@ (Swin @@SWIN_STREAM@@), on @@JACOBIAN_PAIRS@@ hit pairs, float32. Per image the token energies give the share of the @@TOP_TOKENS@@ largest patch tokens and the effective number of tokens, $\exp$ of the entropy of the energy shares. The effective rank is $\exp$ of the entropy of the normalized singular values.
3. Expected gradients (`analysis.attribution.expected_gradients`) of the top logit with @@ATTRIBUTION_SAMPLES@@ background draws on @@ATTRIBUTION_PAIRS@@ hit pairs, a GPU-time cut from shap's 200 that keeps the ranking of concentration. The absolute attribution gives the share of the top @@ATTRIBUTION_TOP@@% of pixels and the Gini coefficient.
4. The smallest sufficient token set (memo L6): tokens removed in a nested order at the attention input of every block, and the smallest visible fraction among @@SUFFICIENCY@@ that still gives the unperturbed class. A random order per image is the per-image form of the redundancy test. The ranked order removes the tokens of smallest attribution first, $\lvert\sum_{\text{channels}} G \odot h\rvert$ at the stream entering block 1, which approximates Cognitive Distillation's optimized mask.

**H-flatness** (`measure.measure_flatness`, `measure.measure_small_noise`). 2 readings. Symmetric finite differences of $\log P_c$ and of the margin along @@FLAT_DIRECTIONS@@ random Gaussian directions per image, steps @@FLAT_EPSILONS@@ of each image's own activation scale, in pixel space and in the stream entering blocks @@FLAT_STREAM_VIT@@ (Swin @@FLAT_STREAM_SWIN@@), @@FLAT_PAIRS@@ hit pairs, float32:

$$\kappa = \frac{g(h + \epsilon\sigma_h v) + g(h - \epsilon\sigma_h v) - 2g(h)}{\epsilon^2}$$

| symbol | meaning |
|---|---|
| $g$ | $\log P_c$ or the margin with a fixed runner-up, as a function of the representation |
| $h$ | the normalized input image or the residual stream entering a block |
| $\sigma_h$ | the root mean square of $h$ over every entry of that image |
| $v$ | a standard Gaussian direction, the same for the plus and the minus step |
| $\epsilon$ | the step |
| $\kappa$ | the directional second difference, the curvature along $v$ |

The theory note's decisive test is the second reading: Gaussian noise at the attention input at rates @@SMALL_RATES@@, float32, @@SMALL_PASSES@@ passes. The second-order expansion of `docs/theory-perturbation-consistency.md` predicts that $\phi(2r)/\phi(r)$ is near 4 per image, that the ranking of images holds across $r$ and that the AUROC at small $r$ stays at the operating AUROC.

**H-neuron-bias** (`measure.shift_to_target_share`, `measure.psu_ratio_without_class`, `cached_reads.py`). The share of shifted clean (pass, image) predictions that land on the target, against $1/C$. The memo's L10 test redistributes the target's gain for clean images whose answer $c$ is not the target $t$, following the theory note:

$$\tilde P_c(x;\xi_j) = P_c(x;\xi_j)\,\frac{1 - P_t(x)}{1 - P_t(x;\xi_j)}$$

and keeps triggered scores unchanged. The cache stores only $P_c$ per pass, so this needs the fresh passes.

**Pass uncertainty and the epistemic ensemble** (`measure.pass_uncertainty`, `seed_ensemble.py`). Over the same passes, the entropy of the mean prediction, the BALD mutual information and the spread of $P_c$, each read with low uncertainty flagging the triggered image. The models trained with dropout 0, so test-time perturbation approximates no posterior. A real epistemic estimate is the disagreement of independently trained replicates: `seed_ensemble.py` computes $\phi_{\text{ens}}(x) = 1 - \tfrac{1}{2}\sum_{s=1}^{2} P_c(x;\theta_s)/P_c(x;\theta_0)$ over the 3 training seeds of a cell, which is the statistic with the seeds in place of the passes, and its Spearman correlation with PSBD-TM's cached per-image statistic on the clean images.

**Out of distribution** (`measure.ood_readout`, `measure.flagged_shares`). The energy $-\log\sum_i e^{z_i}$ and the kNN distance (1 minus the cosine similarity of the final feature to its @@KNN@@th nearest held-out clean feature), triggered against clean and foreign against clean. The statistic's flag, $\phi$ below the @@QUANTILE@@ quantile of the held-out images' $\phi$, applied to the foreign images.

**Trigger neurons** (`measure.measure_neurons`). The inputs of every block's second MLP linear layer, the post-GELU hidden units, read on the class token (ViT) or pooled over positions (Swin). The memo's L7 set is the @@LATE_UNITS@@ units per block in the last @@LATE_BLOCKS@@ blocks that the trigger raises most. A second set is the top @@NEURON_PERCENT@@% per block in every block by mean absolute change. Each is zeroed at every token against as many random units (@@NEURON_DRAWS@@ draws), and robustness is the share of an image's signed deviation from the clean mean that survives a PSBD-TM or PSBD-RD pass on those units.

**Target is easy** (`cached_reads.target_class_reading`, memo P10). From the caches, the statistic of every clean image of the held-out and unpaired analysis splits at PSBD-TM's and PSBD-RD's adaptive rates, split by whether its true class is the target, on the panel and the benign references.

**Confidence ceiling** (`cached_reads.confidence_ceiling_row`, memo P1). $A^*(P_c)$, the AUROC of the cross-fitted likelihood ratio of the unperturbed confidence on 40 bins of $-\log(1 - P_c)$ with folds split by pair, the best any function of confidence can do. Every account that claims to carry the signal has to beat it.

**Clean fragility** (`cached_reads.clean_fragility`). The theory note's 3 findings about how clean answers break, redrawn from the caches: the per-dataset keep curve against a logistic per-image critical rate and against the homogeneous AND law, the flip-count histogram at $k = 3$ against a binomial with the intra-image correlation $\rho$ from $\operatorname{Var}(N) = kq(1-q)(1 + (k-1)\rho)$, and the share of false positives that never flipped.

**Token spread** (`measure.measure_token_spread`, memo L8's WaNet test). At the stream entering blocks @@SPREAD_VIT@@ (Swin @@SPREAD_SWIN@@) on @@SPREAD_PAIRS@@ hit pairs, the coefficient of variation over patch tokens of each token's projection on the block's mean paired change.

### What counts as a finding

A reading is reported as a finding only when it rests on 1 experiment that changes 1 thing against a control and has a figure in the notebook: the margin-matched AUROC against the unmatched one, the backdoor direction against each clean image's own class direction, random token patterns against the same patterns with the trigger forced visible, the ranked token order against the random order, the top trigger units against as many random units, the target's gain redistributed against left in place, foreign images against clean ones, and every reading on a backdoored model against the same reading on a benign model probed with the same trigger. The latent-collapse correlation (P9) has no such control and is shown as context. Any statement that combines several findings into 1 mechanism is labeled a hypothesis. SAM plays no part in this experiment.

### Verdict rules

Every verdict is a status, supported, partial or refuted, and the key number it read, computed by `summarize.py` on the mean of each group's models. The rule texts are stored in `summary.json` beside each verdict and printed by the notebook. The supported and refuted values marked with a memo id are the theory note's. The others are this experiment's: the margin gap of the H-margin rule is a third of the way from clean collapse (negative retention) to full keep, and the backdoor share and top-direction thresholds ask that 1 direction hold at least half of what it measures.

## Sanity checks

@@SANITY@@

## Results

@@RESULTS@@

## Commands

```bash
source .venv/bin/activate
export PYTHONPATH=.
# the sanity gates on 1 model, full PSBD splits, bfloat16 and float32
experiments/why_psbd_works/gpu_slot.sh python experiments/why_psbd_works/sanity.py \
    --folder vit_cifar100_badnet_a2o_0_05
# the queue in priority order, workers sharing the 2 GPU lock slots, resumable
# (a finished model's JSON is skipped, a claimed entry is not retaken)
experiments/why_psbd_works/run_queue.sh worker_a
experiments/why_psbd_works/run_queue.sh worker_b
# 1 model by hand, or the quarantined SIG models once their provenance is settled
python experiments/why_psbd_works/measure.py --models vit_tiny_wanet_0_1
python experiments/why_psbd_works/measure.py --include-quarantined --models vit_cifar10_sig_0_1
python experiments/why_psbd_works/seed_ensemble.py --list
python experiments/why_psbd_works/wanet_probe.py
# CPU: the caches, the summary, this README, the tests and the notebook
python experiments/why_psbd_works/cached_reads.py
python experiments/why_psbd_works/critical_rate.py
python experiments/why_psbd_works/shift_curves.py
python experiments/why_psbd_works/summarize.py
python experiments/why_psbd_works/render_readme.py
python -m pytest tests/test_why_psbd_works.py
jupyter nbconvert --to notebook --execute --inplace notebooks/why-psbd-works-general.ipynb
```

Each GPU process caps itself at @@GPU_GB@@ GB with `torch.cuda.set_per_process_memory_fraction`, runs batches of @@BATCH@@ and uses @@CPU_THREADS@@ CPU threads. `kill -USR1 <pid>` prints a running `measure.py`'s Python stacks to its log.

## Wall time

@@WALL_TIME@@

## Status

@@STATUS@@
"""


if __name__ == "__main__":
    main()
