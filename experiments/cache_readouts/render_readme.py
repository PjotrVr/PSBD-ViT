"""Write experiments/cache_readouts/README.md from the readout records.

Every number the README quotes is read here from a JSON the readout scripts
wrote under results/_experiments/cache_readouts/, or from a constant of those
scripts. A sentence whose wording depends on the data is guarded by an assert,
so a rerun that changes the data stops the render instead of printing a stale
claim. Run it after judge.py.

    .venv/bin/python -m experiments.cache_readouts.render_readme
"""

import json
import os

from defenses.decision import (
    ADAPTIVE_SHIFT_TARGET,
    PLACEMENT_MATCH_TARGET,
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
)
from experiments.cache_readouts import depth_bands, fusion_rules, judge, pass_statistics
from experiments.cache_readouts.shared import (
    DEV_SET_PATH,
    MODEL_SETS,
    REPO_ROOT,
    output_path,
)
from scripts.paper._common import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED

OUT = os.path.join(REPO_ROOT, "experiments", "cache_readouts", "README.md")
RESULTS = "results/_experiments/cache_readouts"
SET_WORDS = {
    "dev": "development set",
    "holdout": "held-out confirmation set",
    "panel": "full panel",
}
STAT_WORDS = {
    "mean_psu": "mean fractional PSU",
    "worst_pass": "worst pass",
    "best_pass": "best pass",
    "shift_count": "shifted-pass count",
    "shift_count_tiebroken": "shifted-pass count, tie broken",
    "spread": "spread across passes",
}
RULE_WORDS = {
    "tm_alone": "PSBD-TM alone",
    "mean_psu": "mean of fractional PSU",
    "min_rank": "min-rank",
    "weighted_0.8_0.2": "weighted min-rank 0.8 and 0.2",
    "weighted_0.9_0.1": "weighted min-rank 0.9 and 0.1",
    "fisher": "Fisher",
}
PARTNER_WORDS = {
    "late_band": "residual dropout, blocks 9 to 12",
    "middle_band": "residual dropout, blocks 5 to 8",
    "psbd_rd": "PSBD-RD",
}
BAND_WORDS = {
    "token_mask/1_4": "TM 1 to 4",
    "token_mask/5_8": "TM 5 to 8",
    "token_mask/9_12": "TM 9 to 12",
    "residual_dropout/1_4": "RD 1 to 4",
    "residual_dropout/5_8": "RD 5 to 8",
    "residual_dropout/9_12": "RD 9 to 12",
}


def main():
    records = read_records()
    values = placeholder_values(records)
    text = TEMPLATE
    for name, value in values.items():
        text = text.replace(f"@@{name}@@", value)
    assert "@@" not in text, text[text.index("@@") - 80 :][:160]
    with open(OUT, "w") as handle:
        handle.write(text)
    print(f"wrote {OUT}")


def read_records():
    records = {"sets": {}}
    for model_set in MODEL_SETS:
        records["sets"][model_set] = {
            experiment: read(output_path(experiment, model_set))
            for experiment in ("pass_statistics", "depth_bands", "fusion_rules")
        }
    records["verdicts"] = read(output_path("verdicts", "all"))
    records["dev_set"] = read(DEV_SET_PATH)
    records["preregistration"] = read(judge.PREREGISTRATION)
    return records


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


def ci(pair):
    low, high = pair
    text = "--" if low is None else f"[{low:+.3f}, {high:+.3f}]"
    return text


def delta(summary):
    text = f"{signed(summary['mean_difference'])} {ci(summary['ci95'])}"
    return text


def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    table = "\n".join(lines)
    return table


def placeholder_values(records):
    sets = records["sets"]
    passes = {s: sets[s]["pass_statistics"] for s in MODEL_SETS}
    bands = {s: sets[s]["depth_bands"] for s in MODEL_SETS}
    fusion = {s: sets[s]["fusion_rules"] for s in MODEL_SETS}
    verdicts = records["verdicts"]
    prereg = records["preregistration"]

    values = {
        "RESULTS_DIR": RESULTS,
        "ADAPTIVE": f"{ADAPTIVE_SHIFT_TARGET}",
        "MATCHED": f"{PLACEMENT_MATCH_TARGET}",
        "TM": RECOMMENDED_PLACEMENT,
        "RD": PUBLISHED_PLACEMENT,
        "RESAMPLES": f"{BOOTSTRAP_RESAMPLES}",
        "SEED": f"{BOOTSTRAP_SEED}",
        "STABLE": f"{depth_bands.STABLE}",
        "BROKEN": f"{depth_bands.BROKEN_MARGIN}",
        "HIT": f"{depth_bands.HIT}",
        "MOST": f"{judge.MOST_SHARE}",
        "TOLERANCE": f"{judge.PLAN_TOLERANCE}",
        "PREREG_TIME": prereg["written_at"],
        "PREREG_HASH": verdicts["preregistration_sha256"],
    }
    for model_set in MODEL_SETS:
        tag = model_set.upper()
        values[f"N_{tag}"] = f"{len(passes[model_set]['models'])}"
        values[f"POOLED_{tag}"] = (
            f"{len(passes[model_set]['summary']['pooled_folders'])}"
        )
        control = passes[model_set]["control"]
        values[f"CONTROL_{tag}"] = f"{control['n_exact']} of {control['n_checked']}"
        assert control["n_exact"] == control["n_checked"] == control["n_rate_matches"]
        for experiment in ("pass_statistics", "depth_bands", "fusion_rules"):
            seconds = sets[model_set][experiment]["wall_seconds"]
            values[f"WALL_{experiment.upper()}_{tag}"] = f"{seconds:.0f}"

    values.update(dev_values(records))
    values.update(pass_values(passes))
    values.update(band_values(bands))
    values.update(fusion_values(fusion))
    values.update(table_values(records, passes, bands, fusion, verdicts))
    return values


def dev_values(records):
    unpooled = records["sets"]["dev"]["pass_statistics"]["summary"]["unpooled_folders"]
    replacements = [m for m in records["dev_set"]["models"] if m["replaces"]]
    lines = [
        f"- `{m['replaces']}` is replaced by `{m['folder']}`. {m['why']}"
        for m in replacements
    ]
    values = {
        "DEV_UNPOOLED": ", ".join(f"`{folder}`" for folder in unpooled),
        "DEV_REPLACEMENTS": "\n".join(lines),
    }
    return values


def pass_values(passes):
    dev_tm = passes["dev"]["summary"]["psbd_tm"]["all"]
    hold_tm = passes["holdout"]["summary"]["psbd_tm"]["all"]
    panel_tm = passes["panel"]["summary"]["psbd_tm"]
    by_attack = panel_tm["by_attack"]

    # The sentences below state directions, so each direction is checked.
    assert dev_tm["best_pass"]["auroc"]["ci95"][0] > 0
    assert dev_tm["worst_pass"]["auroc"]["ci95"][1] < 0
    assert hold_tm["best_pass"]["auroc"]["ci95"][1] < 0
    assert dev_tm["spread"]["auroc"]["mean"] < 0.5
    assert passes["panel"]["summary"]["psbd_tm"]["all"]["spread"]["auroc"]["mean"] < 0.5
    badnet = by_attack["badnet_a2o"]
    assert badnet["best_pass"]["q0.01:tpr"]["mean_difference"] > 0
    for attack in ("blend", "bpp"):
        assert by_attack[attack]["best_pass"]["q0.01:tpr"]["mean_difference"] < 0
    assert dev_tm["shift_count"]["q0.01:tpr"]["mean"] == 0
    pass_counts = {
        row["placements"][name]["passes"]
        for model_set in passes
        for row in passes[model_set]["models"]
        for name in row["placements"]
        if row["placements"][name].get("rate") is not None
    }
    assert len(pass_counts) == 1
    k = pass_counts.pop()
    dev_attacks = passes["dev"]["summary"]["psbd_tm"]["by_attack"]

    wanet_aurocs = [
        f"`{row['folder']}` {f3(row['placements']['psbd_tm']['statistics']['mean_psu']['auroc'])}"
        for row in passes["panel"]["models"]
        if row["attack"] == "wanet"
    ]
    values = {
        "WANET_TM_AUROCS": ", ".join(wanet_aurocs),
        "K": f"{k}",
        "K_VALUES": f"{k + 1}",
        "DEV_N_BADNET": f"{dev_attacks['badnet_a2o']['n']}",
        "DEV_N_BPP": f"{dev_attacks['bpp']['n']}",
        "DEV_BEST_AUROC": delta(dev_tm["best_pass"]["auroc"]),
        "DEV_BEST_HIGHER": f"{dev_tm['best_pass']['auroc']['n_higher']}",
        "DEV_BEST_N": f"{dev_tm['best_pass']['auroc']['n']}",
        "DEV_BEST_TPR1": delta(dev_tm["best_pass"]["q0.01:tpr"]),
        "DEV_WORST_AUROC": delta(dev_tm["worst_pass"]["auroc"]),
        "DEV_WORST_LOWER": f"{dev_tm['worst_pass']['auroc']['n_lower']}",
        "DEV_SPREAD_AUROC": f3(dev_tm["spread"]["auroc"]["mean"]),
        "PANEL_SPREAD_AUROC": f3(panel_tm["all"]["spread"]["auroc"]["mean"]),
        "DEV_COUNT_TPR1": f3(dev_tm["shift_count"]["q0.01:tpr"]["mean"]),
        "DEV_TIEBROKEN_AUROC": delta(dev_tm["shift_count_tiebroken"]["auroc"]),
        "HOLD_BEST_AUROC": delta(hold_tm["best_pass"]["auroc"]),
        "HOLD_BEST_TPR1": delta(hold_tm["best_pass"]["q0.01:tpr"]),
        "HOLD_WORST_AUROC": delta(hold_tm["worst_pass"]["auroc"]),
        "PANEL_BEST_AUROC": delta(panel_tm["all"]["best_pass"]["auroc"]),
        "PANEL_WORST_AUROC": delta(panel_tm["all"]["worst_pass"]["auroc"]),
        "PANEL_BADNET_N": f"{badnet['n']}",
        "PANEL_BADNET_MEAN_TPR1": f3(badnet["mean_psu"]["q0.01:tpr"]["mean"]),
        "PANEL_BADNET_BEST_TPR1": f3(badnet["best_pass"]["q0.01:tpr"]["mean"]),
        "PANEL_BPP_MEAN_TPR1": f3(by_attack["bpp"]["mean_psu"]["q0.01:tpr"]["mean"]),
        "PANEL_BPP_BEST_TPR1": f3(by_attack["bpp"]["best_pass"]["q0.01:tpr"]["mean"]),
        "PANEL_BLEND_MEAN_TPR1": f3(
            by_attack["blend"]["mean_psu"]["q0.01:tpr"]["mean"]
        ),
        "PANEL_BLEND_BEST_TPR1": f3(
            by_attack["blend"]["best_pass"]["q0.01:tpr"]["mean"]
        ),
    }
    return values


def band_values(bands):
    top = bands["panel"]["summary"]["top_rate"]["by_attack"]
    matched = bands["panel"]["summary"]["matched"]["by_attack"]

    def cell(table, attack, key, field):
        value = f3(table[attack][key][field])
        return value

    # Residual dropout at the top rate saturates the clean split in the 2 early
    # bands, which is why a matched reading sits beside it.
    saturated = min(
        top[attack][f"residual_dropout/{band}"]["clean_paired"]
        for attack in top
        for band in ("1_4", "5_8")
    )
    assert saturated >= 0.9
    wanet_top = top["wanet"]
    assert (
        wanet_top["token_mask/1_4"]["triggered"]
        > wanet_top["token_mask/1_4"]["clean_paired"]
    )
    assert (
        wanet_top["token_mask/5_8"]["triggered"]
        > wanet_top["token_mask/5_8"]["clean_paired"]
    )
    assert (
        top["tact"]["token_mask/9_12"]["triggered"]
        > top["tact"]["token_mask/9_12"]["clean_paired"]
    )
    assert (
        matched["wanet"]["residual_dropout/1_4"]["triggered_minus_clean"]
        > matched["badnet_a2o"]["residual_dropout/1_4"]["triggered_minus_clean"]
    )

    verdicts_top = bands["panel"]["verdicts"]["top_rate"]
    e1a_fails = sorted(
        {
            check["folder"]
            for check in verdicts_top["E1a"]["checks"]
            if not check["holds"]
        }
    )
    e1a_badnet = [c for c in verdicts_top["E1a"]["checks"] if "badnet" in c["folder"]]
    assert all(check["holds"] for check in e1a_badnet)
    e1b = verdicts_top["E1b"]
    assert top["badnet_a2o"]["token_mask/9_12"]["triggered"] < depth_bands.HIT
    e2 = verdicts_top["E2"]
    e2_breakdown = {}
    for check in e2["checks"]:
        if not check["holds"]:
            attack = next(
                a for a in depth_bands.GLOBAL_ATTACKS if f"_{a}_" in check["folder"]
            )
            key = f"{attack} in blocks {check['band'].replace('_', ' to ')}"
            e2_breakdown[key] = e2_breakdown.get(key, 0) + 1

    wanet_rows = [row for row in bands["panel"]["models"] if row["attack"] == "wanet"]
    for row in wanet_rows:
        readings = row["bands"]["top_rate"]
        assert (
            readings["token_mask/5_8"]["triggered"]
            > readings["token_mask/5_8"]["clean_paired"]
        )
        assert (
            readings["token_mask/9_12"]["triggered"]
            < readings["token_mask/9_12"]["clean_paired"]
        )

    # E1's survival law with every read masked at the top rate: m trigger
    # tokens read in B blocks survive unless all m B reads are masked.
    top_rate = top["badnet_a2o"]["token_mask/9_12"]["rates"][0]
    trigger_tokens, reading_blocks = 4, 4
    predicted_break = top_rate ** (trigger_tokens * reading_blocks)
    assert top["badnet_a2o"]["token_mask/9_12"]["triggered"] < predicted_break / 2

    values = {
        "RD_SATURATION": f3(saturated),
        "WANET_PANEL_N": f"{len(wanet_rows)}",
        "E1A_BADNET_N": f"{len({c['folder'] for c in e1a_badnet})}",
        "E1A_FAILS": ", ".join(f"`{folder}`" for folder in e1a_fails) or "none",
        "E1B_HELD": f"{e1b['n_held']}",
        "E1B_N": f"{e1b['n_checks']}",
        "E2_FAILS": f"{e2['n_checks'] - e2['n_held']}",
        "E2_CHECKS": f"{e2['n_checks']}",
        "E2_BREAKDOWN": ", ".join(
            f"{key} ({count})"
            for key, count in sorted(e2_breakdown.items(), key=lambda item: -item[1])
        ),
        "E1_READS": f"{trigger_tokens * reading_blocks}",
        "TOP_RATE": f"{top_rate}",
        "E1_PREDICTED_BREAK": f3(predicted_break),
        "BADNET_TM14_TRIG": cell(top, "badnet_a2o", "token_mask/1_4", "triggered"),
        "BADNET_TM14_CLEAN": cell(top, "badnet_a2o", "token_mask/1_4", "clean_paired"),
        "BADNET_TM58_TRIG": cell(top, "badnet_a2o", "token_mask/5_8", "triggered"),
        "BADNET_TM58_CLEAN": cell(top, "badnet_a2o", "token_mask/5_8", "clean_paired"),
        "BADNET_TM912_TRIG": cell(top, "badnet_a2o", "token_mask/9_12", "triggered"),
        "BADNET_TM912_CLEAN": cell(
            top, "badnet_a2o", "token_mask/9_12", "clean_paired"
        ),
        "TACT_TM912_TRIG": cell(top, "tact", "token_mask/9_12", "triggered"),
        "TACT_TM912_CLEAN": cell(top, "tact", "token_mask/9_12", "clean_paired"),
        "WANET_TM14_TRIG": cell(top, "wanet", "token_mask/1_4", "triggered"),
        "WANET_TM14_CLEAN": cell(top, "wanet", "token_mask/1_4", "clean_paired"),
        "WANET_TM58_TRIG": cell(top, "wanet", "token_mask/5_8", "triggered"),
        "WANET_TM58_CLEAN": cell(top, "wanet", "token_mask/5_8", "clean_paired"),
        "WANET_TM912_TRIG": cell(top, "wanet", "token_mask/9_12", "triggered"),
        "WANET_TM912_CLEAN": cell(top, "wanet", "token_mask/9_12", "clean_paired"),
        "WANET_RD14_DIFF_MATCHED": signed(
            matched["wanet"]["residual_dropout/1_4"]["triggered_minus_clean"]
        ),
        "BADNET_RD14_DIFF_MATCHED": signed(
            matched["badnet_a2o"]["residual_dropout/1_4"]["triggered_minus_clean"]
        ),
        "WANET_RD14_DIFF_TOP": signed(
            top["wanet"]["residual_dropout/1_4"]["triggered_minus_clean"]
        ),
        "BADNET_RD14_DIFF_TOP": signed(
            top["badnet_a2o"]["residual_dropout/1_4"]["triggered_minus_clean"]
        ),
    }
    return values


def fusion_values(fusion):
    prereg_rule = "weighted_0.9_0.1"
    panel_late = fusion["panel"]["summary"]["adaptive"]["late_band"]
    panel_nearest = fusion["panel"]["summary"]["nearest"]["late_band"]
    hold_late = fusion["holdout"]["summary"]["adaptive"]["late_band"]

    missing = panel_late["missing"]
    shifts = [m["max_validation_shift"] for m in missing]
    top_rates = {max(m["rates_cached"]) for m in missing}
    ladder_sizes = {len(m["rates_cached"]) for m in missing}
    assert all(shift < ADAPTIVE_SHIFT_TARGET for shift in shifts)
    assert len(top_rates) == 1 and len(ladder_sizes) == 1

    dev_rows = {row["folder"]: row for row in fusion["dev"]["models"]}
    wanet = dev_rows["vit_cifar10_wanet_0_1"]["partners"]["adaptive"]["late_band"][
        "rules"
    ]

    # The best rule on the held-out set by mean AUROC gain, named from the data.
    fused_rules = [
        rule for rule in fusion_rules.RULES if rule != fusion_rules.REFERENCE_RULE
    ]
    best_hold = max(
        fused_rules, key=lambda r: hold_late["all"][r]["auroc"]["mean_difference"]
    )

    for field in ("auroc", "q0.10:tpr", "q0.01:tpr"):
        assert hold_late["all"][prereg_rule][field]["ci95"][0] > 0
    wanet_panel = panel_late["by_attack"]["wanet"]
    assert (
        wanet_panel[prereg_rule]["auroc"]["mean"]
        > wanet_panel["tm_alone"]["auroc"]["mean"]
    )
    assert panel_late["all"][prereg_rule]["literal:q0.10:realized_fpr"]["mean"] <= 0.10
    dev_tact = fusion["dev"]["summary"]["adaptive"]["late_band"]["by_attack"]["tact"]
    for rival in ("min_rank", "fisher"):
        assert (
            dev_tact[prereg_rule]["q0.10:tpr"]["mean"]
            > dev_tact[rival]["q0.10:tpr"]["mean"]
        )

    tact_nearest = panel_nearest["by_attack"]["tact"]
    hold_tm_min = min(
        row["partners"]["adaptive"]["late_band"]["rules"]["tm_alone"]["auroc"]
        for row in fusion["holdout"]["models"]
    )
    assert hold_tm_min > 0.9
    assert tact_nearest["min_rank"]["auroc"]["mean_difference"] < 0

    values = {
        "LATE_COVERED": f"{panel_late['n_models']}",
        "LATE_POOLED": f"{panel_late['n_pooled']}",
        "LATE_MISSING_N": f"{len(missing)}",
        "LATE_MISSING_LIST": ", ".join(f"`{m['folder']}`" for m in missing),
        "LATE_MISSING_SHIFT_LOW": f3(min(shifts)),
        "LATE_MISSING_SHIFT_HIGH": f3(max(shifts)),
        "LATE_MISSING_TOP_RATE": f"{top_rates.pop()}",
        "LATE_MISSING_LADDER": f"{ladder_sizes.pop()}",
        "DEV_WANET_TM": f3(wanet["tm_alone"]["auroc"]),
        "DEV_WANET_PICK": f3(wanet[prereg_rule]["auroc"]),
        "DEV_WANET_MINRANK": f3(wanet["min_rank"]["auroc"]),
        "HOLD_PICK_AUROC": delta(hold_late["all"][prereg_rule]["auroc"]),
        "HOLD_PICK_TPR10": delta(hold_late["all"][prereg_rule]["q0.10:tpr"]),
        "HOLD_PICK_TPR1": delta(hold_late["all"][prereg_rule]["q0.01:tpr"]),
        "HOLD_PICK_N": f"{hold_late['n_models']}",
        "HOLD_BEST_RULE": RULE_WORDS[best_hold],
        "HOLD_BEST_RULE_AUROC": delta(hold_late["all"][best_hold]["auroc"]),
        "PANEL_PICK_AUROC": delta(panel_late["all"][prereg_rule]["auroc"]),
        "PANEL_PICK_TPR1": delta(panel_late["all"][prereg_rule]["q0.01:tpr"]),
        "PANEL_NEAREST_PICK_AUROC": delta(panel_nearest["all"][prereg_rule]["auroc"]),
        "PANEL_NEAREST_N": f"{panel_nearest['n_models']}",
        "PANEL_TACT_MINRANK_LOWER": f"{tact_nearest['min_rank']['auroc']['n_lower']}",
        "PANEL_TACT_MINRANK_MEAN": signed(
            tact_nearest["min_rank"]["auroc"]["mean_difference"]
        ),
        "PANEL_TACT_PICK_LOWER": f"{tact_nearest[prereg_rule]['auroc']['n_lower']}",
        "PANEL_TACT_PICK_MEAN": signed(
            tact_nearest[prereg_rule]["auroc"]["mean_difference"]
        ),
        "HOLD_TM_MIN": f3(hold_tm_min),
        "PANEL_TACT_N": f"{tact_nearest['n']}",
        "PANEL_WANET_PICK": f3(
            panel_late["by_attack"]["wanet"][prereg_rule]["auroc"]["mean"]
        ),
        "PANEL_WANET_TM": f3(
            panel_late["by_attack"]["wanet"]["tm_alone"]["auroc"]["mean"]
        ),
        "PANEL_LITERAL_FPR10": f3(
            panel_late["all"][prereg_rule]["literal:q0.10:realized_fpr"]["mean"]
        ),
        "PANEL_LITERAL_TPR10": f3(
            panel_late["all"][prereg_rule]["literal:q0.10:tpr"]["mean"]
        ),
    }
    return values


def table_values(records, passes, bands, fusion, verdicts):
    values = {
        "TABLE_DEV": dev_table(records, passes["dev"], fusion["dev"]),
        "TABLE_PREREG": prereg_table(verdicts),
        "TABLE_PLAN": plan_table(verdicts),
        "TABLE_WALL": wall_table(records),
        "TABLE_COVERAGE": coverage_table(fusion),
    }
    for model_set in MODEL_SETS:
        tag = model_set.upper()
        for placement in pass_statistics.PLACEMENTS:
            values[f"TABLE_X19_{tag}_{placement.upper()}"] = x19_table(
                passes[model_set]["summary"][placement]["all"]
            )
        values[f"TABLE_X19_ATTACK_{tag}"] = x19_attack_table(
            passes[model_set]["summary"]["psbd_tm"]["by_attack"]
        )
        for reading in depth_bands.READINGS:
            values[f"TABLE_X4_{tag}_{reading.upper()}"] = x4_table(
                bands[model_set]["summary"][reading]["by_attack"]
            )
        values[f"TABLE_X3_{tag}_ADAPTIVE"] = x3_table(
            fusion[model_set]["summary"]["adaptive"]
        )
        values[f"TABLE_X3_ATTACK_{tag}"] = x3_attack_table(
            fusion[model_set]["summary"]["adaptive"]["late_band"]["by_attack"]
        )
    values["TABLE_X3_PANEL_NEAREST"] = x3_table(fusion["panel"]["summary"]["nearest"])
    values["TABLE_X3_ATTACK_PANEL_NEAREST"] = x3_attack_table(
        fusion["panel"]["summary"]["nearest"]["late_band"]["by_attack"]
    )
    values["TABLE_LITERAL"] = literal_table(
        fusion["panel"]["summary"]["adaptive"]["late_band"]["all"]
    )
    return values


def dev_table(records, dev_passes, dev_fusion):
    declared = {m["folder"]: m for m in records["dev_set"]["models"]}
    fusion_rows = {row["folder"]: row for row in dev_fusion["models"]}
    rows = []
    for row in dev_passes["models"]:
        tm = row["placements"]["psbd_tm"]
        rd = row["placements"]["psbd_rd"]
        late = fusion_rows[row["folder"]]["partners"]["adaptive"]["late_band"]
        replaced = declared[row["folder"]]["replaces"]
        rows.append(
            [
                f"`{row['folder']}`",
                declared[row["folder"]]["role"],
                f"`{replaced}`" if replaced else "",
                "yes" if row["successful_2pt"] else "no",
                tm["rate"],
                f3(tm["statistics"]["mean_psu"]["auroc"]),
                f3(rd["statistics"]["mean_psu"]["auroc"]),
                late["rate"] if late["rate"] is not None else "not reached",
            ]
        )
    header = [
        "model",
        "role",
        "replaces",
        "successful_2pt",
        "PSBD-TM rate",
        "PSBD-TM AUROC",
        "PSBD-RD AUROC",
        "late band rate",
    ]
    table = markdown_table(header, rows)
    return table


def x19_table(summary):
    rows = []
    for statistic in pass_statistics.STATISTICS:
        block = summary[statistic]
        rows.append(
            [
                STAT_WORDS[statistic],
                f3(block["auroc"]["mean"]),
                delta(block["auroc"]),
                f"{block['auroc']['n_higher']} / {block['auroc']['n_lower']}",
                f3(block["q0.01:tpr"]["mean"]),
                delta(block["q0.01:tpr"]),
                f3(block["q0.05:tpr"]["mean"]),
                f3(block["q0.10:tpr"]["mean"]),
                f3(block["q0.01:realized_fpr"]["mean"]),
                f3(block["q0.05:realized_fpr"]["mean"]),
                f3(block["q0.10:realized_fpr"]["mean"]),
            ]
        )
    header = [
        "statistic",
        "AUROC",
        "AUROC minus mean PSU",
        "models up / down",
        "TPR 1%",
        "TPR 1% minus mean PSU",
        "TPR 5%",
        "TPR 10%",
        "FPR 1%",
        "FPR 5%",
        "FPR 10%",
    ]
    table = markdown_table(header, rows)
    return table


def x19_attack_table(by_attack):
    rows = [
        [f"{attack} ({group['n']})"]
        + [
            f"{f3(group[s]['auroc']['mean'])} / {f3(group[s]['q0.01:tpr']['mean'])}"
            for s in pass_statistics.STATISTICS
        ]
        for attack, group in by_attack.items()
    ]
    header = ["attack (models)"] + [STAT_WORDS[s] for s in pass_statistics.STATISTICS]
    table = markdown_table(header, rows)
    return table


def x4_table(by_attack):
    keys = [
        f"{op}/{band}" for op in depth_bands.OPERATORS for band in depth_bands.BANDS
    ]
    rows = []
    for attack, group in by_attack.items():
        cells = []
        for key in keys:
            reading = group[key]
            cells.append(
                f"{f3(reading['triggered'])} / {f3(reading['clean_paired'])}"
                if reading["n"]
                else "--"
            )
        rows.append([f"{attack} ({group['n']})"] + cells)
    header = ["attack (models)"] + [BAND_WORDS[key] for key in keys]
    table = markdown_table(header, rows)
    return table


def x3_table(summary):
    rows = []
    for partner in fusion_rules.PARTNERS:
        block = summary[partner]
        for rule in fusion_rules.RULES:
            reading = block["all"][rule]
            rows.append(
                [
                    PARTNER_WORDS[partner],
                    RULE_WORDS[rule],
                    f"{block['n_models']} of {block['n_pooled']}",
                    f3(reading["auroc"]["mean"]),
                    delta(reading["auroc"]),
                    f3(reading["q0.01:tpr"]["mean"]),
                    f3(reading["q0.10:tpr"]["mean"]),
                    f3(reading["q0.20:tpr"]["mean"]),
                    f3(reading["q0.01:realized_fpr"]["mean"]),
                    f3(reading["q0.10:realized_fpr"]["mean"]),
                    f3(reading["q0.20:realized_fpr"]["mean"]),
                ]
            )
    header = [
        "partner",
        "rule",
        "models",
        "AUROC",
        "AUROC minus PSBD-TM",
        "TPR 1%",
        "TPR 10%",
        "TPR 20%",
        "FPR 1%",
        "FPR 10%",
        "FPR 20%",
    ]
    table = markdown_table(header, rows)
    return table


def x3_attack_table(by_attack):
    rows = [
        [f"{attack} ({group['n']})"]
        + [
            f"{f3(group[r]['auroc']['mean'])} / {f3(group[r]['q0.10:tpr']['mean'])}"
            for r in fusion_rules.RULES
        ]
        for attack, group in by_attack.items()
    ]
    header = ["attack (models)"] + [RULE_WORDS[r] for r in fusion_rules.RULES]
    table = markdown_table(header, rows)
    return table


def literal_table(summary):
    rows = []
    for rule in fusion_rules.WEIGHTED_SHARES:
        for quantile in fusion_rules.REPORTED_QUANTILES:
            rows.append(
                [
                    RULE_WORDS[rule],
                    quantile[1:],
                    f3(summary[rule][f"{quantile}:tpr"]["mean"]),
                    f3(summary[rule][f"{quantile}:realized_fpr"]["mean"]),
                    f3(summary[rule][f"literal:{quantile}:tpr"]["mean"]),
                    f3(summary[rule][f"literal:{quantile}:validation_fpr"]["mean"]),
                    f3(summary[rule][f"literal:{quantile}:realized_fpr"]["mean"]),
                ]
            )
    header = [
        "rule",
        "nominal FPR",
        "TPR, calibrated",
        "FPR, calibrated",
        "TPR, union bound",
        "validation FPR, union bound",
        "FPR, union bound",
    ]
    table = markdown_table(header, rows)
    return table


def coverage_table(fusion):
    rows = []
    for model_set in MODEL_SETS:
        for rate_rule in fusion_rules.RATE_RULES:
            for partner in fusion_rules.PARTNERS:
                block = fusion[model_set]["summary"][rate_rule][partner]
                rows.append(
                    [
                        SET_WORDS[model_set],
                        rate_rule,
                        PARTNER_WORDS[partner],
                        f"{block['n_models']} of {block['n_pooled']}",
                        len(block["missing"]),
                    ]
                )
    header = ["set", "rate rule", "partner", "models read", "models missing"]
    table = markdown_table(header, rows)
    return table


def prereg_table(verdicts):
    rows = []
    for model_set in ("holdout", "panel", "dev"):
        for result in verdicts["sets"][model_set]["preregistered"]:
            difference = result["mean_difference"]
            if isinstance(difference, dict):
                shown = ", ".join(f"{k} {signed(v)}" for k, v in difference.items())
            else:
                shown = f"{signed(difference)} {ci(result['ci95'])}"
            rows.append(
                [
                    result["id"],
                    SET_WORDS[model_set],
                    result["n"],
                    shown,
                    result["verdict"],
                ]
            )
    header = ["prediction", "set", "models", "mean paired difference", "verdict"]
    table = markdown_table(header, rows)
    return table


def plan_table(verdicts):
    rows = []
    for model_set in ("holdout", "panel", "dev"):
        block = verdicts["sets"][model_set]
        x19 = block["plan_x19"]
        rows.append(
            [
                "N17 and X19",
                SET_WORDS[model_set],
                "adaptive",
                counts(x19),
                word(x19["verdict"]),
            ]
        )
        for rate_rule, x3 in block["plan_x3"].items():
            rows.append(
                ["X3", SET_WORDS[model_set], rate_rule, counts(x3), word(x3["verdict"])]
            )
        for name in ("E1a", "E1b", "E2", "E10"):
            readings = {
                reading: block["plan_x4"][reading][name]
                for reading in depth_bands.READINGS
            }
            overall = combined_verdict(readings)
            rows.append(
                [
                    f"X4 {name}",
                    SET_WORDS[model_set],
                    ", ".join(f"{r} {word(v)}" for r, v in readings.items()),
                    "",
                    overall,
                ]
            )
    header = ["prediction", "set", "reading", "checks held", "verdict"]
    table = markdown_table(header, rows)
    return table


def combined_verdict(readings):
    # The top-rate reading is the plan's and the matched reading a sensitivity
    # check, so a prediction is decided only when both point the same way.
    labels = set(readings.values())
    if labels == {"held"}:
        text = "held"
    elif "held" in labels:
        text = "inconclusive, the 2 readings disagree"
    elif labels == {"failed"}:
        text = "failed"
    else:
        text = "failed in part"
    return text


def counts(verdict):
    if "n_decided" not in verdict:
        return "none decidable"
    text = f"{verdict['n_held']} of {verdict['n_decided']}"
    if verdict["n_undecided"]:
        text += f", {verdict['n_undecided']} undecidable"
    return text


def word(label):
    # A universal prediction broken on some models counts as failed, and the
    # table says how many checks held.
    text = {
        "held": "held",
        "failed": "failed",
        "mixed": "failed in part",
        "inconclusive": "inconclusive",
    }[label]
    return text


def wall_table(records):
    rows = [
        [
            f"`{experiment}.py`",
            *[
                f"{records['sets'][s][experiment]['wall_seconds']:.0f}"
                for s in MODEL_SETS
            ],
        ]
        for experiment in ("pass_statistics", "depth_bands", "fusion_rules")
    ]
    header = ["script"] + [f"{SET_WORDS[s]} (s)" for s in MODEL_SETS]
    table = markdown_table(header, rows)
    return table


TEMPLATE = r"""# Cache readouts of pass statistics, depth bands and fusion rules

3 questions of `docs/simple-experiments-plan.md` can be answered from the per-pass tensors `cli.sweep` already cached, with no GPU. X19 and N17 ask which statistic over PSBD's perturbed passes separates triggered from clean inputs best. X4 asks which depth band of token masking and of residual dropout breaks which trigger, which tests the routing (E1), redundancy (E2) and computed-trigger (E10) accounts. X3 asks which rule should fuse PSBD-TM with a residual-dropout partner, which decides how a WaNet specialist would enter a union. Every number below is written by a script in this directory into `@@RESULTS_DIR@@` and rendered by `render_readme.py`, which names the file and field in each table caption.

<!-- results:begin -->
<!-- Everything down to results:end is rendered from the records by render_readme.py on every run. -->

## Development set and protocol

The plan fixes 10 development models. 3 of them belong to CIFAR-100 and Tiny ImageNet, which `configs/psbd_basis.json` reserves for reporting, so exploring on them would spend the held-out half. `dev_set.json` swaps each for a CIFAR-10 or GTSRB model of the same attack that carries every placement X3 and X4 read, and `shared.load_model_set` refuses to run when a declared panel label disagrees with the ledger.

@@DEV_REPLACEMENTS@@

The development set therefore holds @@N_DEV@@ models, @@POOLED_DEV@@ of them successful backdoors at the 2-point bar. @@DEV_UNPOOLED@@ fails only the clean-accuracy bar and stays in every per-model table under that label, never in a pooled mean. The table reads `pass_statistics_dev.json` (`models[].placements.*.statistics.mean_psu.auroc`) and `fusion_rules_dev.json` (`models[].partners.adaptive.late_band.rate`).

@@TABLE_DEV@@

The protocol follows steps 1 to 5 of the plan's design section. Every readout ran on the development set first as exploration. I then wrote `preregistration.json` at @@PREREG_TIME@@, naming 1 pass statistic and 1 fusion rule with their predicted effects, before any script read another model. Only after that did the scripts read the @@N_HOLDOUT@@ successful CIFAR-100 and Tiny ImageNet models once as the confirmation and then the @@N_PANEL@@ models of the full panel for completeness. The full panel contains the development models, so it is never read as confirmation. After that first read the held-out and panel readouts were rerun to add reporting fields (the union-bound threshold of the weighted rules and the cached ladder of partners that never reach the target) and to fix a figure label, with no change to any pick, statistic, rate rule or threshold. `judge.py` stores the SHA-256 of the pre-registration beside the verdicts (`verdicts_all.json`, `preregistration_sha256` = `@@PREREG_HASH@@`), so a later edit of the predictions would show.

## Method

Every score is read the way `cli.analyze` reads it. `defenses.cache` loads the tensors, `defenses.scores.psu_ratio_from_cache` gives fractional PSU, `defenses.scores.shift_ratio` gives the clean-validation shift at each cached rate and `defenses.decision.select_rate_adaptively` at @@ADAPTIVE@@ picks the rate. `defenses.decision.pair_clean_to_backdoor` restricts the clean test split to the triggered split's images, and `defenses.decision.detection_report` gives AUROC and TPR at a threshold set to a quantile of the clean-validation scores. The quantile is the nominal FPR and the FPR realized on the paired clean test split is reported beside every TPR. All scores are oriented so that a low score means poisoned. Paired differences carry a bootstrap 95% interval over models from `scripts.paper._common.bootstrap_ci` at @@RESAMPLES@@ resamples and seed @@SEED@@.

The control is the headline statistic itself. The mean fractional PSU computed here must equal `detection_psu_ratio` of `psbd_metrics.json` at the same rate, and the adaptive rate must equal the stored one. It did on @@CONTROL_DEV@@ model and placement pairs of the development set, @@CONTROL_HOLDOUT@@ of the held-out set and @@CONTROL_PANEL@@ of the panel (`pass_statistics_<set>.json`, `control`), bit for bit.

The pass statistics of X19 start from the retained fraction of each pass. The paper's fractional PSU is the first line, the other 5 pool the same passes differently.

$$
r_j(x) = \frac{P_c(x;\,p,\,\theta_j')}{P_c(x;\,\theta)}, \qquad
\phi_{\mathrm{mean}} = 1 - \frac{1}{k}\sum_{j=1}^{k} r_j, \qquad
\phi_{\mathrm{worst}} = 1 - \min_j r_j, \qquad
\phi_{\mathrm{best}} = 1 - \max_j r_j
$$

$$
\phi_{\mathrm{count}} = \sum_{j=1}^{k} \mathbb{1}\big[\hat y_j(x) \ne c\big], \qquad
\phi_{\mathrm{count+}} = \phi_{\mathrm{count}} + \tfrac{1}{4}\,\mathrm{clip}(\phi_{\mathrm{mean}}, -1, 1), \qquad
\phi_{\mathrm{spread}} = \mathrm{std}_j\, r_j
$$

| symbol | meaning |
|---|---|
| $P_c(x;\,\theta)$ | unperturbed probability of the unperturbed argmax class $c$, clamped below at $10^{-6}$ as in `psu_ratio_from_cache` |
| $P_c(x;\,p,\,\theta_j')$ | the same class's probability on perturbed pass $j$ at rate $p$ |
| $r_j(x)$ | retained fraction on pass $j$ |
| $k$ | number of passes, @@K@@ in every cache read here |
| $\hat y_j(x)$ | argmax on pass $j$ |
| $\phi_{\mathrm{count}}$ | number of passes that move the label |
| $\phi_{\mathrm{count+}}$ | the count with ties inside a count broken by the mean, which keeps the count's order |
| $\phi_{\mathrm{spread}}$ | population standard deviation of the retained fractions |

The sign of the spread was fixed before any triggered score was read. Under the OR account a triggered input keeps its label on every pass, so its passes agree and its spread is low.

X4 reads each band at its top cached rate, as the plan asks. The share of passes whose label moved is averaged over triggered inputs and over their paired clean images, and the heat map shows triggered minus clean. Residual dropout at the top rate moves at least @@RD_SATURATION@@ of clean predictions in blocks 1 to 4 and 5 to 8 on every attack of the panel, which leaves nothing to compare, so the same readings are also taken at the rate whose clean-validation shift is nearest @@MATCHED@@ (`select_rate_at_matched_shift`). That second reading is a sensitivity check and never replaces the first. The plan states E1, E2 and E10 in words. `depth_bands.py` fixed these thresholds before the first band was read.

- E1a. Token masking in blocks 1 to 4 and 5 to 8 leaves triggered patch inputs alone (triggered share at most @@STABLE@@) and breaks clean ones (clean share above triggered by at least @@BROKEN@@), on every BadNets and TaCT model.
- E1b. Token masking in blocks 9 to 12 hits triggered patch inputs too (triggered share at least @@HIT@@).
- E2. No token-mask band moves a global trigger (triggered share at most @@STABLE@@ on every Blend, LF and BPP model and band).
- E10. Residual dropout in blocks 1 to 4 moves triggered WaNet more than triggered BadNets, read on triggered minus clean, on every WaNet model against the BadNets mean.

X3 pairs PSBD-TM with each partner. With $\hat F$ the empirical CDF of a probe's fractional PSU on the clean validation split, $u_i = \hat F_i(\phi_i(x))$ its percentile and $w$ PSBD-TM's share of the FPR budget, the 5 fused scores are these.

$$
s_{\mathrm{mean}} = \tfrac{1}{2}(\phi_{\mathrm{TM}} + \phi_{\mathrm{P}}), \qquad
s_{\min} = \min(u_{\mathrm{TM}}, u_{\mathrm{P}}), \qquad
s_{w} = \min\!\Big(\frac{u_{\mathrm{TM}}}{w}, \frac{u_{\mathrm{P}}}{1 - w}\Big), \qquad
s_{\mathrm{Fisher}} = \log \tilde u_{\mathrm{TM}} + \log \tilde u_{\mathrm{P}}
$$

| symbol | meaning |
|---|---|
| $\phi_{\mathrm{TM}}, \phi_{\mathrm{P}}$ | fractional PSU of PSBD-TM and of the partner, each at its own adaptive rate |
| $u_{\mathrm{TM}}, u_{\mathrm{P}}$ | share of the partner's own clean-validation scores below the score (`defenses.scores.to_rank`) |
| $w$ | PSBD-TM's budget share, 0.8 or 0.9 |
| $\tilde u$ | $(n u + 1)/(n + 1)$ with $n$ validation images, so the logarithm is finite |

Every fused score is thresholded at a quantile of its own clean-validation distribution. For the weighted rule the plan also defines the literal union-bound threshold $s_w \le \alpha$, which is reported beside the calibrated one. The partners are `pre_residual_blocks_9_12` (late band), `pre_residual_blocks_5_8` (middle band) and `@@RD@@` (PSBD-RD). A partner whose ladder never reaches the adaptive target is read a second time at the rate nearest the target, under the label `nearest`, so no model drops out silently.

## Commands and wall time

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=4
for set in dev holdout panel; do
    python -m experiments.cache_readouts.pass_statistics --set $set
    python -m experiments.cache_readouts.depth_bands --set $set
    python -m experiments.cache_readouts.fusion_rules --set $set
done
python -m experiments.cache_readouts.judge
python -m experiments.cache_readouts.render_readme
```

The scripts run on the login-node CPU and read only the caches. Wall time is each script's own `wall_seconds`, without the Python start-up.

@@TABLE_WALL@@

## Pre-registered picks

The pass statistic picked on the development set is the best pass of PSBD-TM. It separated better than the mean on @@DEV_BEST_HIGHER@@ of @@DEV_BEST_N@@ development models, @@DEV_BEST_AUROC@@ in AUROC and @@DEV_BEST_TPR1@@ in TPR at 1% FPR, while the worst pass that N17 proposed separated worse on @@DEV_WORST_LOWER@@ of them (@@DEV_WORST_AUROC@@). The fusion rule picked is the weighted min-rank at shares 0.9 and 0.1 with the late band. On the development model where PSBD-TM inverts it lifted AUROC from @@DEV_WANET_TM@@ to @@DEV_WANET_PICK@@ (min-rank reached @@DEV_WANET_MINRANK@@) and lost less on TaCT than min-rank and Fisher. The predictions and their verdicts are in the verdict section, with the full text in `preregistration.json`.

## Pass statistics

The best-pass pick did not confirm. On the @@N_HOLDOUT@@ held-out models the best pass reads @@HOLD_BEST_AUROC@@ in AUROC against the mean and @@HOLD_BEST_TPR1@@ in TPR at 1% FPR. The AUROC interval lies below 0. On the full panel it reads @@PANEL_BEST_AUROC@@. The worst pass loses everywhere, @@HOLD_WORST_AUROC@@ on the held-out set and @@PANEL_WORST_AUROC@@ on the panel, so N17's premise that a triggered patch input survives every pass intact does not hold at the adaptive rate. The per-attack tables show why the development gain did not carry over. The best pass helps the patch trigger and hurts the global ones. On the @@PANEL_BADNET_N@@ BadNets models of the panel it lifts PSBD-TM's TPR at 1% FPR from @@PANEL_BADNET_MEAN_TPR1@@ to @@PANEL_BADNET_BEST_TPR1@@, while BPP falls from @@PANEL_BPP_MEAN_TPR1@@ to @@PANEL_BPP_BEST_TPR1@@ and Blend from @@PANEL_BLEND_MEAN_TPR1@@ to @@PANEL_BLEND_BEST_TPR1@@. The development set held @@DEV_N_BADNET@@ BadNets model and @@DEV_N_BPP@@ BPP models, so it could not show that trade.

The 2 hard-label statistics carry no extra information. With @@K@@ passes the shifted-pass count takes @@K_VALUES@@ values, so a quantile threshold lands on a tie and the count flags nothing at 1% FPR (mean TPR @@DEV_COUNT_TPR1@@ on the development set). Breaking the ties with the mean gives back the mean's ranking to within @@DEV_TIEBROKEN_AUROC@@ in AUROC. The spread is inverted, with mean AUROC @@DEV_SPREAD_AUROC@@ on the development set and @@PANEL_SPREAD_AUROC@@ on the panel under the pre-declared sign. Triggered inputs disagree across passes more than clean ones, since a clean input loses its label on almost every pass at the adaptive rate while a triggered input keeps it on some passes and loses it on others. That is the same fact that sinks the worst pass.

Each table lists the mean over models, the paired difference from the mean fractional PSU with its interval and how many models moved up and down, from `pass_statistics_<set>.json`, `summary.<placement>.all.<statistic>.<field>`.

**PSBD-TM on the held-out set.**

@@TABLE_X19_HOLDOUT_PSBD_TM@@

**PSBD-RD on the held-out set.**

@@TABLE_X19_HOLDOUT_PSBD_RD@@

**PSBD-TM on the full panel.**

@@TABLE_X19_PANEL_PSBD_TM@@

**PSBD-RD on the full panel.**

@@TABLE_X19_PANEL_PSBD_RD@@

**PSBD-TM on the development set.**

@@TABLE_X19_DEV_PSBD_TM@@

**PSBD-RD on the development set.**

@@TABLE_X19_DEV_PSBD_RD@@

**PSBD-TM per attack.**

Each cell is mean AUROC / mean TPR at 1% FPR (`summary.psbd_tm.by_attack.<attack>.<statistic>`).

**Held-out set.**

@@TABLE_X19_ATTACK_HOLDOUT@@

**Full panel.**

@@TABLE_X19_ATTACK_PANEL@@

**Development set.**

@@TABLE_X19_ATTACK_DEV@@

![pass statistics, held-out set](figures/pass_statistics_holdout.png)

![pass statistics, full panel](figures/pass_statistics_panel.png)

## Depth bands

The patch trigger reading splits E1 in 2. Token masking in blocks 1 to 4 or 5 to 8 leaves triggered BadNets predictions where they were, @@BADNET_TM14_TRIG@@ and @@BADNET_TM58_TRIG@@ of passes moved against @@BADNET_TM14_CLEAN@@ and @@BADNET_TM58_CLEAN@@ of clean passes on the panel at the top rate @@TOP_RATE@@. That part (E1a) held on all @@E1A_BADNET_N@@ BadNets models, and the panel models where it failed are @@E1A_FAILS@@. Token masking in blocks 9 to 12 moves @@BADNET_TM912_TRIG@@ of triggered BadNets passes against @@BADNET_TM912_CLEAN@@ of clean ones, below the hit E1b predicted, which held on @@E1B_HELD@@ of @@E1B_N@@ patch models. The survival law of X9 with 4 trigger tokens read in 4 blocks predicts that a pass keeps the trigger only if 1 of its @@E1_READS@@ reads is left, which at rate @@TOP_RATE@@ breaks @@E1_PREDICTED_BREAK@@ of passes. The measured break is far smaller. At that rate nearly every patch token is hidden, so no clean evidence competes with what the class token already holds after block 8 either, which is a hypothesis this readout cannot separate from an early read of the trigger. TaCT inverts in the late band, @@TACT_TM912_TRIG@@ of triggered passes moved against @@TACT_TM912_CLEAN@@ of clean ones.

WaNet reads the opposite way round. Token masking in blocks 1 to 4 moves @@WANET_TM14_TRIG@@ of triggered WaNet passes against @@WANET_TM14_CLEAN@@ of clean ones and blocks 5 to 8 move @@WANET_TM58_TRIG@@ against @@WANET_TM58_CLEAN@@, while blocks 9 to 12 move only @@WANET_TM912_TRIG@@ against @@WANET_TM912_CLEAN@@. On each of the @@WANET_PANEL_N@@ WaNet models of the panel the middle band moves triggered predictions more than clean ones and the late band moves them less, which is what E10 expects of a trigger assembled from relations between tokens and points to the early and middle blocks as where token masking disturbs WaNet. The readout does not show that this causes the inverted cell, since the Tiny ImageNet WaNet models share the pattern and PSBD-TM still separates them (@@WANET_TM_AUROCS@@ in `pass_statistics_panel.json`). E10 as the plan stated it, for residual dropout in blocks 1 to 4, failed at the top rate (WaNet @@WANET_RD14_DIFF_TOP@@ against BadNets @@BADNET_RD14_DIFF_TOP@@, both saturated) and held at the matched reading (@@WANET_RD14_DIFF_MATCHED@@ against @@BADNET_RD14_DIFF_MATCHED@@). With the 2 readings in disagreement the verdict is inconclusive. E2 failed in part. @@E2_FAILS@@ of @@E2_CHECKS@@ global-trigger readings on the panel move triggered predictions above the bar, counted by attack and band as @@E2_BREAKDOWN@@.

Each cell is triggered / clean share of passes whose label moved, from `depth_bands_<set>.json`, `summary.<reading>.by_attack.<attack>.<band>`. TM is token masking at the attention input and RD residual dropout before the add.

**Top cached rate, full panel.**

@@TABLE_X4_PANEL_TOP_RATE@@

**Clean shift nearest the matched target, full panel.**

@@TABLE_X4_PANEL_MATCHED@@

**Top cached rate, held-out set.**

@@TABLE_X4_HOLDOUT_TOP_RATE@@

**Clean shift nearest the matched target, held-out set.**

@@TABLE_X4_HOLDOUT_MATCHED@@

**Top cached rate, development set.**

@@TABLE_X4_DEV_TOP_RATE@@

**Clean shift nearest the matched target, development set.**

@@TABLE_X4_DEV_MATCHED@@

![depth bands, full panel](figures/depth_bands_panel.png)

## Fusion rules

The fusion pick confirmed. On the @@HOLD_PICK_N@@ held-out models the weighted min-rank at 0.9 and 0.1 with the late band raises AUROC by @@HOLD_PICK_AUROC@@, TPR at 10% FPR by @@HOLD_PICK_TPR10@@ and TPR at 1% FPR by @@HOLD_PICK_TPR1@@. Every interval excludes 0. The gain is small, as the pre-registration expected, since the held-out set holds no PSBD-TM failure and its lowest PSBD-TM AUROC is @@HOLD_TM_MIN@@. The best rule on the held-out set by AUROC was the @@HOLD_BEST_RULE@@ (@@HOLD_BEST_RULE_AUROC@@), which shows the selection optimism. On the full panel the pick reads @@PANEL_PICK_AUROC@@ in AUROC and @@PANEL_PICK_TPR1@@ in TPR at 1% FPR. It lifts the WaNet mean from @@PANEL_WANET_TM@@ to @@PANEL_WANET_PICK@@. The union-bound threshold gives the same operating point as the calibrated one, @@PANEL_LITERAL_TPR10@@ TPR at a realized @@PANEL_LITERAL_FPR10@@ FPR for the nominal 10%.

The coverage gap comes from the band itself. @@LATE_MISSING_N@@ of @@LATE_POOLED@@ panel models never reach the adaptive target with late residual dropout, so the adaptive reading covers @@LATE_COVERED@@. The missing models are @@LATE_MISSING_LIST@@. Each of them already holds the full @@LATE_MISSING_LADDER@@-rate ladder up to @@LATE_MISSING_TOP_RATE@@ (`fusion_rules_panel.json`, `summary.adaptive.late_band.missing[].rates_cached`). Their largest clean-validation shift lies between @@LATE_MISSING_SHIFT_LOW@@ and @@LATE_MISSING_SHIFT_HIGH@@. `docs/runs/2026-09-29-gpu-queue.md` item 1 says these models hold only the top 2 rates, which is what the run record of the last sweep lists, but the lower rates were cached earlier. Rerunning the ladder will not give them an adaptive rate. Read at the nearest rate, all @@PANEL_NEAREST_N@@ models are covered and the pick reads @@PANEL_NEAREST_PICK_AUROC@@. That reading includes the @@PANEL_TACT_N@@ TaCT models, too few for an interval. Min-rank lowers their AUROC on @@PANEL_TACT_MINRANK_LOWER@@ of them with a mean change of @@PANEL_TACT_MINRANK_MEAN@@, and the pick lowers it on @@PANEL_TACT_PICK_LOWER@@ with a mean change of @@PANEL_TACT_PICK_MEAN@@, so the 0.9 share bounds the damage without removing it.

@@TABLE_COVERAGE@@

Each table reads `fusion_rules_<set>.json`, `summary.<rate rule>.<partner>.all.<rule>.<field>`, with the difference from PSBD-TM alone on the same models.

**Adaptive rate, held-out set.**

@@TABLE_X3_HOLDOUT_ADAPTIVE@@

**Adaptive rate, full panel.**

@@TABLE_X3_PANEL_ADAPTIVE@@

**Nearest rate, full panel.**

@@TABLE_X3_PANEL_NEAREST@@

**Adaptive rate, development set.**

@@TABLE_X3_DEV_ADAPTIVE@@

**Late band per attack.**

Each cell is mean AUROC / mean TPR at 10% FPR (`summary.<rate rule>.late_band.by_attack`).

**Held-out set, adaptive rate.**

@@TABLE_X3_ATTACK_HOLDOUT@@

**Full panel, adaptive rate.**

@@TABLE_X3_ATTACK_PANEL@@

**Full panel, nearest rate.**

@@TABLE_X3_ATTACK_PANEL_NEAREST@@

**Development set, adaptive rate.**

@@TABLE_X3_ATTACK_DEV@@

**Union-bound threshold of the weighted rules, full panel.**

@@TABLE_LITERAL@@

![fusion rules, held-out set](figures/fusion_rules_holdout.png)

![fusion rules, full panel](figures/fusion_rules_panel.png)

## Verdicts

The pre-registered predictions are judged by `judge.py` from the stored paired summaries. A mean gain whose interval includes 0 is inconclusive under the tie rule, and a per-attack prediction on an attack the set does not contain is inconclusive. The held-out set holds no TaCT model, since every CIFAR-100 and Tiny ImageNet TaCT cell is source-mapped, so X3-patch could only be judged on the panel.

@@TABLE_PREREG@@

The plan's own predictions are judged the same way (`verdicts_all.json`, `sets.<set>.plan_*`). For X3 the plan predicted that the late band helps WaNet and Blend and hurts TaCT under the mean and min-rank rules, and that the weighted rules keep TaCT within @@TOLERANCE@@ of AUROC and of TPR at 10% FPR while keeping at least @@MOST@@ of the min-rank WaNet gain. A universal prediction broken on some checks is counted as failed in part, with the count of checks that held.

@@TABLE_PLAN@@

## Consequences for the WaNet specialist and the union

The union should stay a budget-weighted minimum over the mean fractional PSU of each member. The weighted min-rank at 0.9 and 0.1 confirmed on held-out data and held its realized FPR at the nominal budget, and the pass statistics offer no replacement for the mean, since the best pass trades patch triggers against global ones and the worst pass and the spread lose outright. A per-attack statistic would need the defender to know the attack, so the best pass stays a finding about patch triggers and not a detector choice.

X4 shows that token masking in the early and middle blocks moves triggered WaNet predictions, while late token masking and late residual dropout leave them in place. That makes a late member the natural PSBD-sign partner for WaNet, and the late residual band already lifts the failing model in the union. The WaNet specialist of the plan has the opposite sign, so it has to beat this union on the WaNet models to earn a budget share, and it should be judged against PSBD-TM fused with the late band rather than against PSBD-TM alone. The same union costs TaCT in the late band, where triggered TaCT inputs move more than clean ones, so any late member needs the small budget share the pick uses. A token mask restricted to blocks 9 to 12 as a union member is a hypothesis these readings suggest and do not test, since its TaCT inversion is the same risk.

<!-- results:end -->
"""


if __name__ == "__main__":
    main()
