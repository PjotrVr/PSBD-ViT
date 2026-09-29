"""Write experiments/final_method/README.md from the records under results/_experiments/final_method/.

Every number is read from a JSON a script in this directory wrote, and every
sentence whose wording depends on the data is guarded by an assert, so a rerun
that changes the data stops the render instead of printing a stale claim.

    .venv/bin/python -m experiments.final_method.render_readme
"""

import json
import os

from experiments._paths import experiment_result_path
from experiments.cache_readouts.shared import REPO_ROOT
from experiments.final_method import class_calibration, gpu_jobs
from scripts.paper._common import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED

SLUG = "final_method"
OUT = os.path.join(REPO_ROOT, "experiments", SLUG, "README.md")
HEADLINE = ("q0.01", "q0.05", "q0.10")
METHOD_WORDS = {
    "psbd_tm": "PSBD-TM alone",
    "final_min": "final method, min",
    "final_average": "final method, average",
}
FORM_WORDS = {
    "global": "global",
    "class_percentile": "class percentile",
    "class_z": "class z-score",
}
GROUP_WORDS = {
    "vit_l1": "ViT, 1 probe",
    "swin_l1": "Swin-S, 1 probe",
    "vit_union": "ViT, 3 probes",
}


def main():
    records = {
        name: read(experiment_result_path(SLUG, f"{name}.json"))
        for name in (
            "adaptive_attackers",
            "detector_comparison",
            "anomalies",
            "class_calibration_dev",
            "class_calibration_rest",
            "class_calibration_panel",
            "class_calibration_verdicts",
            "compute_control_vit",
            "gpu_plan",
            "fusion_swin_panel",
            "fusion_swin_panel_late",
        )
    }
    records["prereg_classcal"] = read(class_calibration.PREREGISTRATION)
    records["prereg_swin"] = read(
        os.path.join(
            REPO_ROOT, "experiments", "cache_readouts", "preregistration_swin.json"
        )
    )
    values = placeholder_values(records)
    text = TEMPLATE
    for name, value in values.items():
        text = text.replace(f"@@{name}@@", value)
    assert "@@" not in text, text[text.index("@@") - 80 :][:160]
    with open(OUT, "w") as handle:
        handle.write(text)
    print(f"wrote {OUT}")


def read(path):
    with open(path) as handle:
        payload = json.load(handle)
    return payload


def f3(value):
    text = "--" if value is None else f"{value:.3f}"
    return text


def signed(value):
    text = "--" if value is None else f"{value:+.3f}"
    return text


def delta(summary):
    low, high = summary["ci95"]
    text = f"{signed(summary['mean_difference'])} [{signed(low)}, {signed(high)}]"
    return text


def points(value):
    text = f"{100 * value:+.1f}"
    return text


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    text = "\n".join(lines)
    return text


def placeholder_values(records):
    values = {
        "RESAMPLES": f"{BOOTSTRAP_RESAMPLES}",
        "SEED": f"{BOOTSTRAP_SEED}",
    }
    values.update(class_values(records))
    values.update(anomaly_values(records))
    values.update(detector_values(records))
    values.update(attacker_values(records))
    values.update(control_values(records))
    values.update(plan_values(records))
    values.update(swin_values(records))
    values.update(swin_late_values(records))
    return values


def class_values(records):
    prereg = records["prereg_classcal"]
    m = prereg["shrinkage"]["class_z"]
    key = f"m={m}"
    rest = records["class_calibration_rest"]["summary"][key]
    panel = records["class_calibration_panel"]["summary"][key]
    dev = records["class_calibration_dev"]["summary"][key]
    verdicts = {r["id"]: r for r in records["class_calibration_verdicts"]["results"]}
    assert verdicts["CC-z-tm"]["verdict"] == "failed"
    rest_tm = rest["all"]["class_z/psbd_tm"]["q0.01:tpr"]
    assert rest_tm["ci95"][1] < 0
    dev_tm = dev["all"]["class_z/psbd_tm"]["q0.01:tpr"]
    assert dev_tm["ci95"][0] > 0

    tiny = rest["by_dataset"]["tiny"]["class_z/psbd_tm"]
    target_rows = [
        r["target_class_validation"]
        for r in records["class_calibration_panel"]["models"]
        if r["target_class_validation"]["median_psu"] is not None
    ]
    below = sum(r["median_psu"] < r["global_median_psu"] for r in target_rows)
    tiny_rows = [
        r["target_class_validation"]
        for r in records["class_calibration_panel"]["models"]
        if r["dataset"] == "tiny"
        and r["target_class_validation"]["median_psu"] is not None
    ]
    tiny_below = sum(r["median_psu"] < r["global_median_psu"] for r in tiny_rows)
    tiny_wider = sum(r["std_psu"] > r["global_std_psu"] for r in tiny_rows)
    tiny_n_class = sorted(r["n"] for r in tiny_rows)
    assert below > len(target_rows) / 2
    assert tiny_below > len(tiny_rows) / 2 and tiny_wider > len(tiny_rows) / 2
    assert all(
        r["verdict"] == "failed"
        for r in records["class_calibration_verdicts"]["results"]
    )
    assert rest["all"]["class_z/psbd_tm"]["q0.01:realized_fpr"]["mean"] > 0.01
    tiny_drop = tiny["q0.01:tpr"]["mean_difference"]
    assert all(
        tiny_drop <= group["class_z/psbd_tm"]["q0.01:tpr"]["mean_difference"]
        for group in rest["by_dataset"].values()
    )
    rest_attacks = {
        attack: group["class_z/psbd_tm"]["q0.01:tpr"]["mean_difference"]
        for attack, group in rest["by_attack"].items()
    }
    raised = [attack for attack, value in rest_attacks.items() if value > 0]
    assert len(raised) < len(rest_attacks) / 2
    detector = read(experiment_result_path(SLUG, "detector_comparison.json"))[
        "summary"
    ]["by_attack"]["tact"]

    def form_rows(summary):
        rows = []
        for form in class_calibration.FORMS:
            for method in class_calibration.METHODS:
                block = summary[f"{form}/{method}"]
                rows.append(
                    [FORM_WORDS[form], METHOD_WORDS[method]]
                    + [
                        f"{f3(block[f'{q}:tpr']['mean'])} ({delta(block[f'{q}:tpr'])})"
                        for q in HEADLINE
                    ]
                    + [f3(block[f"{q}:realized_fpr"]["mean"]) for q in HEADLINE]
                    + [f3(block["auroc"]["mean"])]
                )
        return rows

    header = [
        "calibration",
        "method",
        "TPR 1% (minus global)",
        "TPR 5% (minus global)",
        "TPR 10% (minus global)",
        "FPR 1%",
        "FPR 5%",
        "FPR 10%",
        "AUROC",
    ]

    def group_rows(groups):
        rows = []
        for name, group in groups.items():
            for method in class_calibration.METHODS:
                cells = []
                for form in class_calibration.FORMS:
                    block = group[f"{form}/{method}"]
                    cells.append(
                        " / ".join(f3(block[f"{q}:tpr"]["mean"]) for q in HEADLINE)
                        + f" ({f3(block['q0.01:realized_fpr']['mean'])})"
                    )
                rows.append([f"{name} ({group['n']})", METHOD_WORDS[method]] + cells)
        return rows

    group_header = ["group (models)", "method"] + [
        f"{FORM_WORDS[f]}, TPR 1 / 5 / 10% (FPR 1%)" for f in class_calibration.FORMS
    ]
    verdict_rows = [
        [r["id"], r["verdict"]]
        for r in records["class_calibration_verdicts"]["results"]
    ]
    prediction_text = {p["id"]: p["text"] for p in prereg["predictions"]}
    verdict_rows = [[i, prediction_text[i], v] for i, v in verdict_rows]

    tact_rows = []
    for t in records["class_calibration_panel"]["tact_placement"]:
        for q in HEADLINE:
            r = t["by_fpr"][q]
            tact_rows.append(
                [
                    f"`{t['folder']}`",
                    q[1:],
                    f3(r["threshold"]),
                    f3(r["source_clean_below"]),
                    f3(r["target_clean_below"]),
                    f3(r["triggered_below"]),
                ]
            )
    median_rows = [
        [f"`{t['folder']}`"]
        + [
            f3(t["median"][k])
            for k in ("validation", "source_clean", "target_clean", "triggered")
        ]
        for t in records["class_calibration_panel"]["tact_placement"]
    ]

    values = {
        "CC_M": f"{m}",
        "CC_ATTACKS_LOWER": f"{sum(v < 0 for v in rest_attacks.values())}",
        "CC_ATTACKS_N": f"{len(rest_attacks)}",
        "CC_ATTACKS_RAISED": " and ".join(raised) if raised else "none",
        "CC_BEATRIX_TACT": f3(detector["beatrix"]["q0.01:tpr"]),
        "CC_TED_TACT": f3(detector["ted"]["q0.01:tpr"]),
        "CC_GRID": ", ".join(str(g) for g in class_calibration.SHRINKAGE_GRID),
        "CC_PREREG_TIME": prereg["written_at"],
        "CC_HASH": records["class_calibration_verdicts"]["preregistration_sha256"],
        "CC_DEV_N": f"{dev['all']['n']}",
        "CC_REST_N": f"{rest['all']['n']}",
        "CC_PANEL_N": f"{panel['all']['n']}",
        "CC_DEV_TM1": delta(dev_tm),
        "CC_REST_TM1": delta(rest_tm),
        "CC_REST_TM1_FPR": f3(
            rest["all"]["class_z/psbd_tm"]["q0.01:realized_fpr"]["mean"]
        ),
        "CC_TINY_GLOBAL": f3(
            rest["by_dataset"]["tiny"]["global/psbd_tm"]["q0.01:tpr"]["mean"]
        ),
        "CC_TINY_Z": f3(tiny["q0.01:tpr"]["mean"]),
        "CC_TINY_N": f"{rest['by_dataset']['tiny']['n']}",
        "CC_TARGET_BELOW": f"{below}",
        "CC_TINY_BELOW": f"{tiny_below}",
        "CC_TINY_WIDER": f"{tiny_wider}",
        "CC_TINY_ROWS": f"{len(tiny_rows)}",
        "CC_TINY_NC_LOW": f"{tiny_n_class[0]}",
        "CC_TINY_NC_HIGH": f"{tiny_n_class[-1]}",
        "CC_TARGET_N": f"{len(target_rows)}",
        "TABLE_CC_PANEL": table(header, form_rows(panel["all"])),
        "TABLE_CC_REST": table(header, form_rows(rest["all"])),
        "TABLE_CC_DEV": table(header, form_rows(dev["all"])),
        "TABLE_CC_ATTACK": table(group_header, group_rows(panel["by_attack"])),
        "TABLE_CC_DATASET": table(group_header, group_rows(panel["by_dataset"])),
        "TABLE_CC_VERDICTS": table(["prediction", "text", "verdict"], verdict_rows),
        "TABLE_TACT_PLACEMENT": table(
            [
                "model",
                "nominal FPR",
                "global threshold",
                "source clean below",
                "target clean below",
                "triggered below",
            ],
            tact_rows,
        ),
        "TABLE_TACT_MEDIANS": table(
            ["model", "validation", "source clean", "target clean", "triggered"],
            median_rows,
        ),
    }
    return values


def anomaly_values(records):
    anomalies = records["anomalies"]
    rows = []
    tied_images = max(
        round(a["tie_share_at_threshold"] * a["validation_size"])
        for a in anomalies["min_rule"]
    )
    validation_size = anomalies["min_rule"][0]["validation_size"]
    assert tied_images <= 2
    partner_max = max(a["partner_alone_q0.01_tpr"] for a in anomalies["min_rule"])
    for a in anomalies["min_rule"]:
        assert a["effective_quantile_anchor"] <= 0.0075
        assert a["min_realized_fpr_q0.01"] <= 0.01
        assert abs(a["min_tpr_q0.01_tiebroken"] - a["min_tpr_q0.01"]) < 0.03
        assert (
            a["anchor_alone_at_effective_quantile"]["tpr"] <= a["min_tpr_q0.01"] + 0.05
        )
        rows.append(
            [
                f"`{a['folder']}`",
                f3(a["anchor_alone_q0.01"]["tpr"]),
                f3(a["min_tpr_q0.01"]),
                f3(a["min_realized_fpr_q0.01"]),
                f3(a["tie_share_at_threshold"]),
                f3(a["effective_quantile_anchor"]),
                f3(a["anchor_alone_at_effective_quantile"]["tpr"]),
                f"{f3(a['anchor_validation_psu_quantiles']['0.005'])} / {f3(a['anchor_validation_psu_quantiles']['0.01'])}",
                f3(a["anchor_backdoor_psu_quantiles"]["0.1"]),
                f3(a["partner_alone_q0.01_tpr"]),
                f3(a["min_tpr_q0.01_tiebroken"]),
            ]
        )
    tact_rows = []
    for t in anomalies["tact"]:
        for q in HEADLINE:
            r = t["by_fpr"][q]
            tact_rows.append(
                [
                    f"`{t['folder']}`",
                    q[1:],
                    f"{f3(r['tpr_all_class'])} ({f3(r['realized_fpr_source_clean'])})",
                    f3(r["realized_fpr_other_clean"]),
                    f"{f3(r['tpr_source_only'])} ({f3(r['realized_fpr_source_only'])})",
                    f"{f3(r['tpr_per_predicted_class'])} ({f3(r['realized_fpr_per_predicted_class'])})",
                ]
            )
    fusion = read(experiment_result_path("cache_readouts", "fusion_rules_panel.json"))
    by_folder = {
        r["folder"]: r["partners"]["adaptive"]["middle_band"]["rules"]
        for r in fusion["models"]
    }
    gaps, recovers = [], 0
    for a in anomalies["min_rule"]:
        rules = by_folder[a["folder"]]
        alone = rules["tm_alone"]["at_fpr"]["q0.01"]["tpr"]
        gaps.append(alone - rules["weighted_0.9_0.1"]["at_fpr"]["q0.01"]["tpr"])
        recovers += rules["mean_psu"]["at_fpr"]["q0.01"]["tpr"] >= alone - 0.05
    first = anomalies["tact"][0]
    assert all(t["by_fpr"]["q0.05"]["tpr_source_only"] > 0.5 for t in anomalies["tact"])
    helped = sum(
        t["by_fpr"]["q0.05"]["tpr_per_predicted_class"]
        > t["by_fpr"]["q0.05"]["tpr_all_class"] + 0.2
        for t in anomalies["tact"]
    )
    values = {
        "ANOMALY_COLLAPSES": ", ".join(
            f"`{f}`" for f in anomalies["collapses_on_panel"]
        ),
        "ANOMALY_N": f"{len(anomalies['collapses_on_panel'])}",
        "ANOMALY_DROP": f"{anomalies['collapse_rule']['tpr_drop']}",
        "ANOMALY_FLOOR": f"{anomalies['collapse_rule']['auroc_floor']}",
        "TABLE_MIN_ANATOMY": table(
            [
                "model",
                "PSBD-TM TPR 1%",
                "min TPR 1%",
                "min FPR 1%",
                "tie share",
                "effective PSBD-TM quantile",
                "PSBD-TM TPR there",
                "clean PSU at 0.5% / 1%",
                "triggered PSU 10th pct",
                "band alone TPR 1%",
                "min, tie broken",
            ],
            rows,
        ),
        "TABLE_TACT_CAL": table(
            [
                "model",
                "nominal FPR",
                "global threshold TPR (source FPR)",
                "other-class FPR",
                "source-only threshold TPR (FPR)",
                "per-predicted-class TPR (FPR)",
            ],
            tact_rows,
        ),
        "TACT_FIRST": f"`{first['folder']}`",
        "TACT_HELPED": f"{helped}",
        "TACT_N": f"{len(anomalies['tact'])}",
        "PARTNER_MAX": f3(partner_max),
        "WEIGHTED_GAP": f3(max(gaps)),
        "AVERAGE_RECOVERS": f"{recovers}",
        "TIED_IMAGES": f"{tied_images}",
        "VALIDATION_SIZE": f"{validation_size}",
        "TACT_FIRST_NSRC": f"{first['n_source_validation']}",
    }
    return values


def detector_values(records):
    d = records["detector_comparison"]
    overall = d["summary"]["overall"]
    ranked = sorted(d["methods"], key=lambda m: -overall[m]["q0.01:tpr"])
    rows = [
        [f"`{m}`", overall[m]["n"]]
        + [
            f"{f3(overall[m][f'{q}:tpr'])} ({f3(overall[m][f'{q}:fpr'])})"
            for q in HEADLINE
        ]
        + [f3(overall[m]["auroc"])]
        for m in ranked
    ]
    paired_rows = [
        [METHOD_WORDS[m]]
        + [
            f"{delta(d['summary']['paired'][m][f])} against `{d['summary']['paired'][m][f]['against']}`"
            for f in (*[f"{q}:tpr" for q in HEADLINE], "auroc")
        ]
        for m in ("psbd_tm", "final_min", "final_average")
    ]
    for m in ("final_min", "final_average"):
        for q in HEADLINE:
            assert d["summary"]["paired"][m][f"{q}:tpr"]["ci95"][0] > 0
    by_attack = d["summary"]["by_attack"]
    top_competitors = [m for m in ranked if m not in METHOD_WORDS][:4]
    shown = ["psbd_tm", "final_min", "final_average"] + top_competitors
    attack_rows = [
        [f"{a} ({by_attack[a]['psbd_tm']['n']})"]
        + [" / ".join(f3(by_attack[a][m][f"{q}:tpr"]) for q in HEADLINE) for m in shown]
        for a in by_attack
    ]
    all_attack_rows = [
        [f"`{m}`"] + [f3(by_attack[a][m]["q0.01:tpr"]) for a in by_attack]
        for m in ranked
    ]
    missing = d["missing_detector_records"]
    values = {
        "DET_N": f"{overall['psbd_tm']['n']}",
        "DET_MISSING": "none"
        if not missing
        else ", ".join(f"{k} ({len(v)})" for k, v in missing.items()),
        "DET_BEST_1": f"`{d['summary']['best_competitor']['q0.01:tpr']}`",
        "DET_BEST_5": f"`{d['summary']['best_competitor']['q0.05:tpr']}`",
        "TABLE_DET": table(
            [
                "method",
                "models",
                "TPR 1% (FPR)",
                "TPR 5% (FPR)",
                "TPR 10% (FPR)",
                "AUROC",
            ],
            rows,
        ),
        "TABLE_DET_PAIRED": table(
            [
                "method",
                "TPR 1% minus best",
                "TPR 5% minus best",
                "TPR 10% minus best",
                "AUROC minus best",
            ],
            paired_rows,
        ),
        "TABLE_DET_ATTACK": table(
            ["attack (models)"] + [METHOD_WORDS.get(m, f"`{m}`") for m in shown],
            attack_rows,
        ),
        "TABLE_DET_ATTACK_ALL": table(["method"] + list(by_attack), all_attack_rows),
    }
    return values


def attacker_values(records):
    a = records["adaptive_attackers"]
    s = a["summary"]
    union = s["vit_union"]
    rows = []
    for name, g in s.items():
        rows.append(
            [
                GROUP_WORDS[name],
                g["n"],
                g["n_asr_clears"],
                g["n_successful_2pt"],
                g["n_successful_5pt"],
                g["n_twin_successful_2pt"],
                f3(g["mean_asr"]),
                f3(g["mean_twin_asr"]),
                points(g["mean_clean_accuracy_drop_vs_twin"]),
                points(g["mean_clean_accuracy_drop_vs_benign"]),
            ]
        )
    det_rows = []
    for name, g in s.items():
        for placement, blocks in g["placements"].items():
            b = blocks["asr_clears"]
            if not b["n"]:
                continue
            det_rows.append(
                [
                    GROUP_WORDS[name],
                    f"`{placement}`",
                    "probed"
                    if b["n_probed"] == b["n"]
                    else ("unprobed" if b["n_probed"] == 0 else "mixed"),
                    b["n"],
                    " / ".join(f3(b[f"evader_tpr_{q}"]) for q in HEADLINE),
                    " / ".join(f3(b[f"evader_fpr_{q}"]) for q in HEADLINE),
                    " / ".join(f3(b[f"twin_tpr_{q}"]) for q in HEADLINE),
                    f3(b["evader_auroc"]),
                    f3(b["evader_auroc_two_sided"]),
                    f3(b["twin_auroc"]),
                    f"{b['n_inverted']} of {b['n']}",
                ]
            )
    all_inverted_rows = []
    for name, g in s.items():
        for placement, blocks in g["placements"].items():
            b = blocks["all"]
            all_inverted_rows.append(
                [
                    GROUP_WORDS[name],
                    f"`{placement}`",
                    b["n"],
                    f3(b["evader_auroc"]),
                    f3(b["evader_auroc_two_sided"]),
                    f"{b['n_inverted']} of {b['n']}",
                ]
            )
    cached_rows = [
        [
            GROUP_WORDS[name],
            ", ".join(f"`{p}` ({c})" for p, c in g["placements_cached_counts"].items()),
        ]
        for name, g in s.items()
    ]

    # The headline sentences, each guarded.
    vit_unprobed = [
        b["asr_clears"]["evader_auroc"]
        for p, b in s["vit_l1"]["placements"].items()
        if b["asr_clears"]["n_probed"] == 0 and b["asr_clears"]["n"]
    ]
    swin_tm = s["swin_l1"]["placements"]["before_attention_norm_token_mask"][
        "asr_clears"
    ]
    assert min(vit_unprobed) > 0.8 and swin_tm["evader_auroc"] > 0.9
    rd = union["placements"]["post_residual"]["asr_clears"]
    assert rd["n_probed"] == 0 and rd["evader_auroc"] < 0.5
    vit_probes = {
        tuple(r["probes"])
        for r in a["models"]
        if r["architecture"] == "vit" and r["family"] == "l1"
    }
    swin_probes = {
        tuple(r["probes"]) for r in a["models"] if r["architecture"] == "swin"
    }
    assert vit_probes == {("before_attention_norm_token_mask",)}
    assert swin_probes == {("before_attention_norm",)}
    union_probes = {
        p for r in a["models"] if r["family"] == "union" for p in r["probes"]
    }
    assert "post_residual" not in union_probes
    blend = next(
        r for r in a["models"] if r["folder"] == "vit_cifar100_blend_0_05_evade_union"
    )
    blend_tm = blend["detection"]["before_attention_norm_token_mask"]["evader"]

    values = {
        "ATT_UNION_N": f"{union['n']}",
        "ATT_UNION_CA_TWIN": points(union["mean_clean_accuracy_drop_vs_twin"]),
        "ATT_UNION_CA_BENIGN": points(union["mean_clean_accuracy_drop_vs_benign"]),
        "ATT_UNION_2PT": f"{union['n_successful_2pt']}",
        "ATT_UNION_5PT": f"{union['n_successful_5pt']}",
        "ATT_UNION_ASR": f"{union['n_asr_clears']}",
        "ATT_UNION_TM": f3(
            union["placements"]["before_attention_norm_token_mask"]["asr_clears"][
                "evader_auroc"
            ]
        ),
        "ATT_UNION_RD": f3(rd["evader_auroc"]),
        "ATT_UNION_RD_TWIN": f3(rd["twin_auroc"]),
        "ATT_UNION_RD_INV": f"{rd['n_inverted']} of {rd['n']}",
        "ATT_UNION_RD_TPR1": f3(rd["evader_tpr_q0.01"]),
        "ATT_BLEND_TM": f3(blend_tm["auroc"]),
        "ATT_BLEND_TM2": f3(blend_tm["auroc_two_sided"]),
        "ATT_VIT_UNPROBED_MIN": f3(min(vit_unprobed)),
        "ATT_VIT_UNPROBED_MAX": f3(max(vit_unprobed)),
        "ATT_SWIN_TM": f3(swin_tm["evader_auroc"]),
        "ATT_SWIN_TM_TWIN": f3(swin_tm["twin_auroc"]),
        "ATT_VIT_L1_N": f"{s['vit_l1']['n']}",
        "ATT_L1_TOTAL": f"{s['vit_l1']['n'] + s['swin_l1']['n']}",
        "ATT_SWIN_L1_N": f"{s['swin_l1']['n']}",
        "ATT_VIT_L1_2PT": f"{s['vit_l1']['n_successful_2pt']}",
        "ATT_SWIN_L1_2PT": f"{s['swin_l1']['n_successful_2pt']}",
        "ATT_VIT_L1_CA": points(s["vit_l1"]["mean_clean_accuracy_drop_vs_twin"]),
        "ATT_SWIN_L1_CA": points(s["swin_l1"]["mean_clean_accuracy_drop_vs_twin"]),
        "TABLE_ATT_SUCCESS": table(
            [
                "attacker",
                "models",
                "clear ASR",
                "successful 2-point",
                "successful 5-point",
                "twins successful 2-point",
                "mean ASR",
                "twin mean ASR",
                "clean accuracy vs twin (points)",
                "clean accuracy vs benign (points)",
            ],
            rows,
        ),
        "TABLE_ATT_DET": table(
            [
                "attacker",
                "placement",
                "probe",
                "models",
                "evader TPR 1 / 5 / 10%",
                "evader FPR 1 / 5 / 10%",
                "twin TPR 1 / 5 / 10%",
                "evader AUROC",
                "two-sided",
                "twin AUROC",
                "inverted",
            ],
            det_rows,
        ),
        "TABLE_ATT_ALL": table(
            [
                "attacker",
                "placement",
                "models",
                "evader AUROC",
                "two-sided",
                "inverted",
            ],
            all_inverted_rows,
        ),
        "TABLE_ATT_CACHED": table(
            ["attacker", "placements cached (models)"], cached_rows
        ),
    }
    return values


def control_values(records):
    c = records["compute_control_vit"]["summary"]
    rows = [
        ["PSBD-TM, 6 passes"]
        + [f3(c["tm_k6"][f]) for f in (*[f"{q}:tpr" for q in HEADLINE], "auroc")]
    ]
    rows.append(
        ["PSBD-TM, 3 passes"]
        + [f3(c["tm_k3"][f]) for f in (*[f"{q}:tpr" for q in HEADLINE], "auroc")]
    )
    for m in ("final_min", "final_average"):
        rows.append(
            [f"{METHOD_WORDS[m]}, 3 plus 3"]
            + [
                f"{f3(c[m][f]['mean'])} ({delta(c[m][f])})"
                for f in (*[f"{q}:tpr" for q in HEADLINE], "auroc")
            ]
        )
    values = {
        "CTRL_N": f"{c['n_scored']}",
        "CTRL_MISSING": f"{len(c['missing_six_pass_cache'])}",
        "TABLE_CTRL": table(["method", "TPR 1%", "TPR 5%", "TPR 10%", "AUROC"], rows),
    }
    return values


def swin_values(records):
    swin = records["fusion_swin_panel"]
    summary = swin["summary"]
    overall = summary["groups"]["all"]["evader"]
    verdicts = swin["verdicts"]
    assert all(r["verdict"] == "failed" for r in verdicts["results"])
    unscored = [u["folder"] for u in summary["unscored"]]
    rows = [
        [METHOD_WORDS[m]]
        + [
            f"{f3(overall[m][f'{q}:tpr']['mean'])} ({delta(overall[m][f'{q}:tpr'])})"
            for q in HEADLINE
        ]
        + [f3(overall[m][f"{q}:realized_fpr"]["mean"]) for q in HEADLINE]
        + [f3(overall[m]["auroc"]["mean"])]
        for m in overall
    ]
    by_attack = summary["groups"]["all"]["by_attack"]
    attack_rows = [
        [f"{a} ({g['psbd_tm']['q0.01:tpr']['n']})"]
        + [" / ".join(f3(g[m][f"{q}:tpr"]["mean"]) for q in HEADLINE) for m in g]
        for a, g in by_attack.items()
    ]
    verdict_rows = [
        [r["id"], r["verdict"]] + [delta(r["readings"][f"{q}:tpr"]) for q in HEADLINE]
        for r in verdicts["results"]
    ]
    values = {
        "SWIN_N": f"{summary['n_models']}",
        "SWIN_SCORED": f"{summary['n_scored']}",
        "SWIN_UNSCORED": ", ".join(f"`{u}`" for u in unscored),
        "SWIN_HASH": verdicts["preregistration_sha256"],
        "TABLE_SWIN": table(
            [
                "method",
                "TPR 1% (minus PSBD-TM)",
                "TPR 5% (minus PSBD-TM)",
                "TPR 10% (minus PSBD-TM)",
                "FPR 1%",
                "FPR 5%",
                "FPR 10%",
                "AUROC",
            ],
            rows,
        ),
        "TABLE_SWIN_ATTACK": table(
            ["attack (models)"] + [METHOD_WORDS[m] for m in overall], attack_rows
        ),
        "TABLE_SWIN_VERDICTS": table(
            [
                "prediction",
                "verdict",
                "TPR 1% minus PSBD-TM",
                "TPR 5% minus PSBD-TM",
                "TPR 10% minus PSBD-TM",
            ],
            verdict_rows,
        ),
    }
    return values


def swin_late_values(records):
    late = records["fusion_swin_panel_late"]
    prereg = read(
        os.path.join(REPO_ROOT, "experiments", SLUG, "preregistration_swin_late.json")
    )
    coverage = late["coverage"]
    readings = {
        "adaptive_only": "adaptive rate only",
        "with_nearest": "with the nearest-rate models",
    }
    methods = ("psbd_tm", "final_min", "final_average")

    verdict_rows = []
    for reading, words in readings.items():
        for r in late["verdicts"][reading]:
            verdict_rows.append(
                [r["id"], words, r["n"]]
                + [delta(r["readings"][f"{q}:tpr"]) for q in HEADLINE]
                + [r["verdict"]]
            )

    def overall_rows(reading):
        group = late["summary"][reading]["all"]
        rows = [
            [METHOD_WORDS[m]]
            + [
                f"{f3(group[m][f'{q}:tpr']['mean'])} ({delta(group[m][f'{q}:tpr'])})"
                for q in HEADLINE
            ]
            + [f3(group[m][f"{q}:realized_fpr"]["mean"]) for q in HEADLINE]
            + [f3(group[m]["auroc"]["mean"])]
            for m in methods
        ]
        return rows

    def group_rows(reading):
        rows = []
        for name, group in late["summary"][reading].items():
            if name == "all":
                continue
            label = name.replace("attack=", "").replace("poison_rate=", "rate ")
            rows.append(
                [f"{label} ({group['n']})"]
                + [
                    " / ".join(f3(group[m][f"{q}:tpr"]["mean"]) for q in HEADLINE)
                    for m in methods
                ]
                + [f3(group[m]["q0.01:realized_fpr"]["mean"]) for m in methods]
            )
        return rows

    overall_header = [
        "method",
        "TPR 1% (minus PSBD-TM)",
        "TPR 5% (minus PSBD-TM)",
        "TPR 10% (minus PSBD-TM)",
        "FPR 1%",
        "FPR 5%",
        "FPR 10%",
        "AUROC",
    ]
    group_header = (
        ["group (models)"]
        + [f"{METHOD_WORDS[m]}, TPR 1 / 5 / 10%" for m in methods]
        + [f"{METHOD_WORDS[m]}, FPR 1%" for m in methods]
    )
    verdict = {
        reading: {r["id"]: r["verdict"] for r in late["verdicts"][reading]}
        for reading in readings
    }
    assert verdict["adaptive_only"]["SWIN2-min-tpr"] == "held"
    assert verdict["adaptive_only"]["SWIN2-min-wanet"] == "held"
    assert verdict["adaptive_only"]["SWIN2-average-tpr"] == "inconclusive"
    assert verdict["with_nearest"]["SWIN2-average-tpr"] == "held"
    badnet = late["summary"]["adaptive_only"]["attack=badnet_a2o"]["final_min"][
        "q0.01:tpr"
    ]
    assert badnet["mean_difference"] < 0
    wanet = late["summary"]["adaptive_only"]["attack=wanet"]
    small = max(
        abs(
            late["summary"]["adaptive_only"][f"attack={a}"]["final_min"]["q0.01:tpr"][
                "mean_difference"
            ]
        )
        for a in ("blend", "lf", "bpp")
    )
    assert small < 0.01
    shifts = [n["max_validation_shift"] for n in coverage["partner_nearest"]]
    values = {
        "SWIN2_HASH": late["preregistration_sha256"],
        "SWIN2_TIME": prereg["written_at"],
        "SWIN2_PANEL": f"{coverage['n_panel']}",
        "SWIN2_ANCHOR": f"{coverage['n_anchor_reaches']}",
        "SWIN2_REACH": f"{coverage['n_partner_reaches']}",
        "SWIN2_NEAREST": f"{len(coverage['partner_nearest'])}",
        "SWIN2_NEAREST_LOW": f3(min(shifts)),
        "SWIN2_NEAREST_HIGH": f3(max(shifts)),
        "SWIN2_WANET_N": f"{wanet['n']}",
        "SWIN2_SMALL": f3(small),
        "SWIN2_WANET_TM1": f3(wanet["psbd_tm"]["q0.01:tpr"]["mean"]),
        "SWIN2_WANET_MIN1": f3(wanet["final_min"]["q0.01:tpr"]["mean"]),
        "SWIN2_BADNET_MIN1": delta(badnet),
        "TABLE_SWIN2_VERDICTS": table(
            [
                "prediction",
                "reading",
                "models",
                "TPR 1% minus PSBD-TM",
                "TPR 5% minus PSBD-TM",
                "TPR 10% minus PSBD-TM",
                "verdict",
            ],
            verdict_rows,
        ),
        "TABLE_SWIN2_ADAPTIVE": table(overall_header, overall_rows("adaptive_only")),
        "TABLE_SWIN2_NEAREST": table(overall_header, overall_rows("with_nearest")),
        "TABLE_SWIN2_GROUPS_ADAPTIVE": table(group_header, group_rows("adaptive_only")),
        "TABLE_SWIN2_GROUPS_NEAREST": table(group_header, group_rows("with_nearest")),
    }
    return values


def plan_values(records):
    plan = records["gpu_plan"]
    rows = [
        [
            item,
            p["jobs"],
            ", ".join(f"{a} {b['jobs']}" for a, b in p["by_architecture"].items()),
            f"{p['hours']:.1f}",
            f"{p['cumulative_hours']:.1f}",
        ]
        for item, p in plan["items"].items()
    ]
    values = {
        "PLAN_TOTAL": f"{plan['total_hours']:.1f}",
        "PLAN_WINDOW": f"{plan['window_hours']}",
        "PLAN_VIT_RATE": f"{plan['seconds_per_rate_k3']['vit']:.0f}",
        "PLAN_SWIN_RATE": f"{plan['seconds_per_rate_k3']['swin']:.0f}",
        "PLAN_OVERHEAD": f"{plan['overhead_seconds']}",
        "PLAN_VIT_LADDER": ", ".join(str(r) for r in gpu_jobs.EVADER_LADDER["vit"]),
        "PLAN_SWIN_LADDER": ", ".join(str(r) for r in gpu_jobs.EVADER_LADDER["swin"]),
        "TABLE_PLAN": table(
            ["item", "jobs", "by architecture", "hours", "cumulative hours"], rows
        ),
        "SWIN_PREREG_TIME": records["prereg_swin"]["written_at"],
    }
    return values


TEMPLATE = r"""# Evaluation of the final method

The final method is PSBD-TM fused with residual dropout in the middle third of the block stack, blocks 5 to 8 of ViT-B/16 and blocks 9 to 16 of Swin-S, each at its own adaptive rate. It is reported under 2 rules, the plain minimum of the 2 clean-validation percentiles (min) and the plain average of the 2 fractional PSUs (average). TPR at 1%, 5% and 10% FPR is the headline, each beside the FPR realized on the paired clean test split, with AUROC after it. Paired intervals are bootstrap 95% intervals over models (`scripts.paper._common.bootstrap_ci`, @@RESAMPLES@@ resamples, seed @@SEED@@). Every number below is rendered by `render_readme.py` from a JSON under `results/_experiments/final_method/`, named with each table.

<!-- results:begin -->
<!-- Everything down to results:end is rendered from the records by render_readme.py on every run. -->

## Calibration by predicted class

Calibrating each score against the clean validation images of the same predicted class helps TaCT a little and costs most of the other attacks. On the @@CC_REST_N@@ models read as the confirmation it lowers PSBD-TM's TPR at 1% FPR on @@CC_ATTACKS_LOWER@@ of @@CC_ATTACKS_N@@ attacks, and the attack it raises is @@CC_ATTACKS_RAISED@@. The idea was that a defender knows each input's predicted class, and that the CIFAR-10 TaCT failure below is a calibration effect. `class_calibration.py` tests 2 forms that need no knowledge of the attack, a class percentile and a class z-score, each shrunk toward the whole validation split with strength $m$ images (the formulas are in its docstring). The shrinkage was chosen on the @@CC_DEV_N@@ successful development models from the grid @@CC_GRID@@ by PSBD-TM's mean TPR over the 3 FPRs, and both forms chose $m$ = @@CC_M@@, the edge of the grid. `preregistration_classcal.json` fixed that rule and 4 predictions at @@CC_PREREG_TIME@@ (SHA-256 `@@CC_HASH@@`) before the other @@CC_REST_N@@ panel models were read.

The development set said yes and the confirmation said no. The class z-score raised PSBD-TM's TPR at 1% FPR by @@CC_DEV_TM1@@ on the development set and lowered it by @@CC_REST_TM1@@ on the other models, with the realized FPR at the nominal 1% rising to @@CC_REST_TM1_FPR@@. The loss is largest on Tiny ImageNet, where TPR at 1% FPR falls from @@CC_TINY_GLOBAL@@ to @@CC_TINY_Z@@ over @@CC_TINY_N@@ models. All 4 predictions failed.

@@TABLE_CC_VERDICTS@@

The reason is structural. A triggered input is predicted as the target class, so class calibration compares it with the clean validation images the model predicts as the target. Those images are the ones the backdoor has altered. Their PSU median sits below the global median on @@CC_TARGET_BELOW@@ of @@CC_TARGET_N@@ panel models (`class_calibration_panel.json`, `models[].target_class_validation`). On Tiny ImageNet, where the loss concentrates, it sits below on @@CC_TINY_BELOW@@ of @@CC_TINY_ROWS@@ models and the spread is wider than the global one on @@CC_TINY_WIDER@@, from only @@CC_TINY_NC_LOW@@ to @@CC_TINY_NC_HIGH@@ validation images predicted as the target. The class calibration therefore lowers and widens the bar exactly where triggered inputs land, and does so from a handful of images. Beatrix and TED do best on TaCT here (TPR at 1% FPR @@CC_BEATRIX_TACT@@ and @@CC_TED_TACT@@, `detector_comparison.json`) and are class-conditional through their own statistics. Whether their form of conditioning avoids this problem is not tested here.

**All 54 panel models** (`class_calibration_panel.json`, `summary.m=@@CC_M@@.all`).

@@TABLE_CC_PANEL@@

**The @@CC_REST_N@@ models read once as the confirmation** (`class_calibration_rest.json`).

@@TABLE_CC_REST@@

**The development set** (`class_calibration_dev.json`).

@@TABLE_CC_DEV@@

**Per attack, all 54 models.** Each cell is mean TPR at 1% / 5% / 10% FPR with the realized FPR at 1% in brackets (`summary.m=@@CC_M@@.by_attack`).

@@TABLE_CC_ATTACK@@

**Per dataset, all 54 models** (`summary.m=@@CC_M@@.by_dataset`).

@@TABLE_CC_DATASET@@

**TaCT against the global PSBD-TM threshold.** The share of each group below the threshold, and the group medians of fractional PSU (`tact_placement`).

@@TABLE_TACT_PLACEMENT@@

@@TABLE_TACT_MEDIANS@@

![calibration by predicted class](figures/class_calibration.png)

## The TaCT calibration effect

On the CIFAR-10 TaCT models the high AUROC and the TPR near 0 measure different populations. AUROC compares triggered inputs with their paired clean images, and every one of those comes from the source class, while the threshold comes from all-class validation. Clean source-class images are more fragile under token masking than the validation average, so they rarely fall below the all-class threshold, and triggered inputs sit between them and the rest. A threshold read from source-class validation alone (@@TACT_FIRST_NSRC@@ images on @@TACT_FIRST@@) recovers most of the TPR at its own FPR. A defender does not know the source class, so this is a finding about what the metric compares and not a deployable fix. The deployable version is a threshold per predicted class. It lifts TPR at 5% FPR by more than 0.2 on @@TACT_HELPED@@ of the @@TACT_N@@ TaCT models (`anomalies.json`, `tact`). The panel-wide test above shows what it costs elsewhere.

@@TABLE_TACT_CAL@@

## The min rule at 1% FPR

The collapse of the min rule at 1% FPR on single models comes from the budget split, and ties play no part in it. On the panel @@ANOMALY_N@@ models lose at least @@ANOMALY_DROP@@ of TPR at 1% FPR under the min rule against PSBD-TM alone while keeping AUROC at or above @@ANOMALY_FLOOR@@ (@@ANOMALY_COLLAPSES@@). At each of them at most @@TIED_IMAGES@@ of the @@VALIDATION_SIZE@@ validation images sit on the threshold, the realized FPR stays at or below the nominal 1% and breaking ties by the average changes nothing. What changes is the budget. The 1% quantile of the minimum of 2 percentiles admits only about 0.5% of each probe's validation scores, and PSBD-TM alone at that effective quantile loses as much TPR as the min rule does, because clean validation PSU falls steeply between its 0.5% and 1% quantiles while the middle band alone reaches at most @@PARTNER_MAX@@ TPR at 1% FPR on these models. A tie-break is therefore no fix. The weighted minimum at 0.9 and 0.1 of `experiments/cache_readouts/`, which gives PSBD-TM most of the budget, stays within @@WEIGHTED_GAP@@ of PSBD-TM alone on all @@ANOMALY_N@@ models. The average rule recovers the loss on @@AVERAGE_RECOVERS@@ of them (`anomalies.json`, `min_rule`, and `fusion_rules_panel.json` of the cache readouts).

@@TABLE_MIN_ANATOMY@@

## The final method against the competitor detectors

On the @@DET_N@@ ViT panel models the final method leads every competitor at 1%, 5% and 10% FPR under both rules. Every paired interval against the best competitor at that FPR excludes 0. The best competitor at 1% FPR is @@DET_BEST_1@@ and at 5% @@DET_BEST_5@@. Detector records missing: @@DET_MISSING@@. The Swin detectors have no records on disk, so no Swin table exists (`detector_comparison.json`).

@@TABLE_DET@@

**Paired differences against the best competitor at each FPR.**

@@TABLE_DET_PAIRED@@

**Per attack.** Each cell is mean TPR at 1% / 5% / 10% FPR, for our 3 methods and the 4 competitors with the highest TPR at 1% FPR.

@@TABLE_DET_ATTACK@@

**Per attack, TPR at 1% FPR, every method.**

@@TABLE_DET_ATTACK_ALL@@

![final method and the competitor detectors](figures/detector_comparison.png)

## Swin second probe, attempt 1

The middle band does not carry over to Swin-S as a second probe, and both pre-registered predictions failed. `experiments/cache_readouts/preregistration_swin.json` fixed the band (blocks 9 to 16), both rules and the prediction that TPR at 1%, 5% and 10% FPR rises above PSBD-TM alone at @@SWIN_PREREG_TIME@@, before any triggered score of the band was read, and `fusion_readout.py --set swin_panel` read it once (SHA-256 `@@SWIN_HASH@@`). The band was already cached on every Swin panel model that PSBD-TM can score, so no sweep was needed. @@SWIN_SCORED@@ of @@SWIN_N@@ successful Swin-S models are scored. @@SWIN_UNSCORED@@ never reach the adaptive target under PSBD-TM itself, so neither PSBD-TM nor the final method has a rate for them (`fusion_swin_panel.json`).

@@TABLE_SWIN_VERDICTS@@

@@TABLE_SWIN@@

**Per attack, TPR at 1% / 5% / 10% FPR.**

@@TABLE_SWIN_ATTACK@@

## Swin second probe, attempt 2

The late band carries over under the min rule, which was the 2nd and last pre-registered attempt. `preregistration_swin_late.json` (SHA-256 `@@SWIN2_HASH@@`, written at @@SWIN2_TIME@@ before any detection number of the band was read) chose residual dropout in Swin-S blocks 17 to 24 from the depth of WaNet's backdoor direction in `experiments/backdoor_manifestation/`, onset median 16 and range 15 to 23 of 24 blocks, a measurement that never reads detection. `swin_late_readout.py` read it once (`fusion_swin_panel_late.json`).

The min rule held every prediction. On the models where both placements reach the adaptive target it raises TPR at 1%, 5% and 10% FPR with every interval above 0, and on the @@SWIN2_WANET_N@@ WaNet models TPR at 1% FPR rises from @@SWIN2_WANET_TM1@@ to @@SWIN2_WANET_MIN1@@. The average rule is inconclusive on that reading because 2 of its intervals touch 0. It holds once the nearest-rate models are added. The gain is concentrated. BadNets loses @@SWIN2_BADNET_MIN1@@ at 1% FPR under the min rule. Blend, LF and BPP move by at most @@SWIN2_SMALL@@ there. WaNet carries most of the gain. As a 2nd attempt chosen after the 1st failed, it carries a selection cost the interval does not include.

Coverage is partial. PSBD-TM reaches the adaptive target on @@SWIN2_ANCHOR@@ of @@SWIN2_PANEL@@ successful Swin-S models and the late band on @@SWIN2_REACH@@. The largest clean-validation shift of the other @@SWIN2_NEAREST@@ lies between @@SWIN2_NEAREST_LOW@@ and @@SWIN2_NEAREST_HIGH@@, so the primary reading covers @@SWIN2_REACH@@ models and the labeled second reading adds the @@SWIN2_NEAREST@@ at their nearest rate (`coverage`).

@@TABLE_SWIN2_VERDICTS@@

**Adaptive rate only.**

@@TABLE_SWIN2_ADAPTIVE@@

**With the nearest-rate models, labeled.**

@@TABLE_SWIN2_NEAREST@@

**Per attack and per poison rate, adaptive rate only.** TPR at 1% / 5% / 10% FPR and the realized FPR at 1%.

@@TABLE_SWIN2_GROUPS_ADAPTIVE@@

**Per attack and per poison rate, with the nearest-rate models.**

@@TABLE_SWIN2_GROUPS_NEAREST@@

## Adaptive attackers on disk

Evasion against 1 probe does not transfer, and evasion against 3 probes costs a mean @@ATT_UNION_CA_TWIN@@ clean-accuracy points against the non-evasive twin (@@ATT_UNION_CA_BENIGN@@ against the benign reference) and leaves @@ATT_UNION_2PT@@ of @@ATT_UNION_N@@ successful at the 2-point bar (@@ATT_UNION_5PT@@ at 5 points, @@ATT_UNION_ASR@@ clearing the ASR bar). The ViT single-probe attackers were trained against PSBD-TM itself, and among those that keep attack success the unprobed operators still read mean AUROC @@ATT_VIT_UNPROBED_MIN@@ to @@ATT_VIT_UNPROBED_MAX@@. The Swin single-probe attackers were trained against dropout at the attention input, and PSBD-TM, unprobed there, reads @@ATT_SWIN_TM@@ against @@ATT_SWIN_TM_TWIN@@ on their twins. The single-probe evasion is not free either. It costs @@ATT_VIT_L1_CA@@ points on ViT and @@ATT_SWIN_L1_CA@@ on Swin against the twins, and leaves @@ATT_VIT_L1_2PT@@ of @@ATT_VIT_L1_N@@ ViT and @@ATT_SWIN_L1_2PT@@ of @@ATT_SWIN_L1_N@@ Swin models successful at 2 points.

The 3-probe attacker does transfer to PSBD-RD. It trained against PSBD-TM, dropout at the attention input and gain scaling after the MLP norm, never against dropout after the residual adds, and still PSBD-RD reads one-sided AUROC @@ATT_UNION_RD@@ against @@ATT_UNION_RD_TWIN@@ on the twins, inverted on @@ATT_UNION_RD_INV@@ of the attackers that clear the ASR bar, with TPR at 1% FPR @@ATT_UNION_RD_TPR1@@. PSBD-TM reads @@ATT_UNION_TM@@. The two-sided diagnostic hides this, `vit_cifar100_blend_0_05_evade_union` reads @@ATT_BLEND_TM@@ one-sided and @@ATT_BLEND_TM2@@ two-sided under PSBD-TM, and the decision rule is one-sided. Whether the middle band transfers the same way is item 3 of the GPU plan, since no evasive checkpoint caches it.

Attack success is read from each model's PSBD baseline cache and clean accuracy from its `args.json`. The verdicts come from `scripts.coverage_ledger.classify_cell` with the Swin benign references for Swin (`adaptive_attackers.json`). The canonical families are the @@ATT_L1_TOTAL@@ `_evade_l1` models of CIFAR-100 and Tiny ImageNet and the @@ATT_UNION_N@@ `vit_cifar100_*_evade_union` models. The CIFAR-10 lambda sweep, the smoke runs and the paper-attack variants are other experiments and stay out.

@@TABLE_ATT_SUCCESS@@

**Detection of the attackers that clear the ASR bar, adaptive rate, one-sided.** "Inverted" counts one-sided AUROC below 0.5.

@@TABLE_ATT_DET@@

**Every attacker, cleared or not.**

@@TABLE_ATT_ALL@@

**Placements cached on the attackers.** The final method's middle band is cached on none of them, so it cannot be scored yet.

@@TABLE_ATT_CACHED@@

![adaptive attackers against their twins](figures/adaptive_attackers.png)

## Compute control, partial

The final method spends 6 perturbed passes per input. `compute_control.py` gives PSBD-TM alone the same 6 at its k = 3 adaptive rate, from the first 6 passes of the `_k20` caches that exist. It covers @@CTRL_N@@ ViT panel models so far. The other @@CTRL_MISSING@@ wait for item 2 of the GPU plan, and Swin waits for it too. The brackets are paired against PSBD-TM at 6 passes. This partial reading is not a result, since the covered models were chosen by which caches another queue happened to fill (`compute_control_vit.json`).

@@TABLE_CTRL@@

## GPU plan for the night of 2026-09-30

`gpu_driver.sh` runs on the login A100 from 17:00 and starts no job after 06:30. It waits for the analysis agents' done markers (`scratch/gpu_done_{general,tm,phenomenon,manifestation}`) so their leftovers go first. It then runs 3 items in order. Every sweep holds `scratch/gpu.lock` for 1 model, caps itself at 0.15 of the card's memory and leaves the CPU analysis to run outside the lock. `gpu_jobs.py` lists only work not yet cached, so the queue resumes after any stop. The driver touches `scratch/gpu_done_final_method` at the end. It is not started yet.

1. Swin second probe. `cli.sweep --block-range` counts Swin-S blocks 1 to 24 in forward order across its 4 stages of 2, 2, 18 and 2 (`models.positions.resolve_targets`), so the middle third is `--block-range 9 16`, blocks 5 to 12 of stage 3. Every Swin panel model that PSBD-TM can score already holds that band's full ladder. The 2 that lack it are the 2 PSBD-TM cannot score, and the readout above is done. The item holds no job and stays in the driver only to catch a model added to the panel since.
2. Compute control. PSBD-TM at 6 passes at the k = 3 adaptive rate on the ViT and Swin panel models that hold neither a `_k6` nor a `_k20` cache at that rate, then `compute_control.py` for both.
3. The final method on the evaders. The middle band on every evasive checkpoint that clears the ASR bar, with the ladder @@PLAN_VIT_LADDER@@ on ViT and @@PLAN_SWIN_LADDER@@ on Swin. The ladders start 1 step below the lowest rate at which the band reaches the target on the panel, and `fusion_readout.py` flags a model whose lowest rate already reaches it. Then `fusion_readout.py --set evaders`.

The estimate reads @@PLAN_VIT_RATE@@ s per rate at 3 passes on ViT and @@PLAN_SWIN_RATE@@ s on Swin, from the file times of 1 uncontended ladder each, plus @@PLAN_OVERHEAD@@ s per job taken from the k = 20 runs (`gpu_plan.json`).

@@TABLE_PLAN@@

The 3 items need @@PLAN_TOTAL@@ h of an uncontended card against a window of @@PLAN_WINDOW@@ h. The analysis leftovers and the k = 20 queue of `scratch/k20_driver.sh` wait for the same markers and take the same lock, and their length is not known here. If they take more than the difference, the Swin half of item 3 is what does not fit, since it runs last. The resumable job lists carry it to the next night.

## Commands

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=4
python -m experiments.final_method.adaptive_attackers
python -m experiments.final_method.detector_comparison
python -m experiments.final_method.anomalies
for stage in dev rest panel; do python -m experiments.final_method.class_calibration --stage $stage; done
python -m experiments.final_method.judge_classcal
python -m experiments.final_method.compute_control --architecture vit
python -m experiments.final_method.gpu_jobs --item plan
python -m experiments.final_method.render_readme
# from 17:00, resumable
nohup experiments/final_method/gpu_driver.sh > /dev/null 2>&1 &
```

The class calibration stages were rerun once after the confirmation read to add the per-dataset breakdown and the target-class statistics, with the frozen rule unchanged.

<!-- results:end -->
"""


if __name__ == "__main__":
    main()
