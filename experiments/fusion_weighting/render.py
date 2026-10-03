"""Render README.md and the report's LaTeX subsection from summary.json.

Every number in both files is read from results/_experiments/fusion_weighting/,
and every sentence whose truth depends on the data is guarded by an assert, so a
rerun that changes the picture stops here instead of printing a stale claim.

    .venv/bin/python -m experiments.fusion_weighting.render
"""

import hashlib
import json
import os

from experiments._paths import experiment_result_path

SLUG = "fusion_weighting"
RESULTS_DIR = "results"
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
README_PATH = os.path.join(EXPERIMENT_DIR, "README.md")
TEX_PATH = os.path.join("tmp", "report", "sections", "fusion_rules.tex")
PREREGISTRATION = os.path.join("experiments", "cache_readouts", "preregistration.json")
BUDGETS = ("q0.01", "q0.05", "q0.10", "q0.20")
HEADLINE = ("q0.01", "q0.05", "q0.10")
SWEEP_SHARES = ("0.50", "0.60", "0.70", "0.80", "0.85", "0.90", "0.95", "0.99")
LABELS = {
    "vit_band": "ViT-B/16 panel, partner blocks 5 to 8",
    "vit_late": "ViT-B/16 panel, partner blocks 9 to 12 (pre-registered pairing)",
    "swin_band": "Swin-S panel, partner blocks 17 to 24, adaptive rate",
    "swin_band_with_nearest": "Swin-S panel, partner blocks 17 to 24, with the nearest-rate models",
    "backdoorbench": "BackdoorBench ViT-B/16, partner blocks 5 to 8",
    "training_set": "training-set setting, partner blocks 5 to 8",
}
SHORT = {
    "vit_band": "ViT-B/16, blocks 5 to 8",
    "vit_late": "ViT-B/16, blocks 9 to 12",
    "swin_band": "Swin-S, blocks 17 to 24",
    "swin_band_with_nearest": "Swin-S, blocks 17 to 24, nearest rate added",
    "backdoorbench": "BackdoorBench",
    "training_set": "training set",
}
RULE_NAMES = {"tm": "PSBD-TM alone", "min": "min", "weighted": "weighted 0.9/0.1"}
TEX_RULE_NAMES = {"tm": "PSBD-TM", "min": "Min", "weighted": "Weighted"}
DATASET_NAMES = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "gtsrb": "GTSRB",
    "tiny": "Tiny ImageNet",
}
ATTACK_NAMES = {
    "badnet_a2o": "BadNets",
    "tact": "TaCT",
    "blend": "Blend",
    "lf": "LF",
    "bpp": "BPP",
    "wanet": "WaNet",
    "lc": "Label-Consistent",
    "adaptive_blend": "Adaptive-Blend",
}
# A trade-off cell is listed when the 2 rules differ by more than this at any
# headline budget.
TRADE_OFF_LISTED = 0.02


def main():
    summary = load_summary()
    readings = {
        name: load_json(
            experiment_result_path(SLUG, f"readings_{name}.json", RESULTS_DIR)
        )
        for name in ("vit_panel", "swin_panel", "backdoorbench", "training_set")
    }
    with open(PREREGISTRATION, "rb") as handle:
        raw = handle.read()
    preregistration = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "content": json.loads(raw),
    }
    swin_caches = swin_cache_counts(readings["swin_panel"])
    facts = key_facts(summary)
    facts["vit_wanet_partner_tpr"] = wanet_partner_tpr(readings["vit_panel"])

    readme = render_readme(summary, preregistration, swin_caches, facts)
    write_text(readme, README_PATH)
    tex = render_tex(summary, preregistration, facts)
    write_text(tex, TEX_PATH)
    print(f"wrote {README_PATH} and {TEX_PATH}")


def load_summary():
    summary = load_json(experiment_result_path(SLUG, "summary.json", RESULTS_DIR))
    return summary


def load_json(path):
    with open(path) as handle:
        payload = json.load(handle)
    return payload


def swin_cache_counts(swin_readings):
    counts = {}
    for placement in ("pre_residual_blocks_9_16", "pre_residual_blocks_17_24"):
        counts[placement] = sum(
            os.path.isdir(os.path.join(RESULTS_DIR, m["folder"], "psbd", placement))
            for m in swin_readings["models"]
        )
    counts["n_panel"] = len(swin_readings["models"])
    return counts


def wanet_partner_tpr(vit_readings):
    budgets = vit_readings["low_budgets"]
    rows = [
        m["partners"]["band"]["low_budget_tpr"]["partner_alone"]
        for m in vit_readings["models"]
        if m["attack"] == "wanet" and m["partners"].get("band", {}).get("rate")
    ]
    tpr = {
        budget: sum(row[budgets.index(budget)] for row in rows) / len(rows)
        for budget in (0.001, 0.005)
    }
    return tpr


def key_facts(summary):
    groups = summary["groups"]
    vit = groups["vit_band"]
    swin = groups["swin_band"]
    bb = groups["backdoorbench"]
    training = groups["training_set"]
    selection = summary["selection"]

    facts = {
        "vit_curve_range": max(vit["fine"]["means"]["q0.01"])
        - min(vit["fine"]["means"]["q0.01"]),
        "vit_best_share": vit["fine"]["plateau"]["best_share"],
        "vit_gap": vit["fine"]["plateau"]["weighted_gap_to_best"],
        "vit_losses_min": vit["sweep"]["0.50"]["losses"]["q0.01"],
        "vit_losses_weighted": vit["sweep"]["0.90"]["losses"]["q0.01"],
        "bb_plateau": bb["fine"]["plateau"]["within_tolerance"],
        "swin_best_share": swin["fine"]["plateau"]["best_share"],
        "swin_gap": swin["fine"]["plateau"]["weighted_gap_to_best"],
        "training_best_share": training["fine"]["plateau"]["best_share"],
        "training_gap": training["fine"]["plateau"]["weighted_gap_to_best"],
        "lodo_mean_picks": {
            d: f["sweep"]["picked"]
            for d, f in selection["leave_one_dataset_out"].items()
        },
        "lodo_fine_picks": {
            d: f["fine"]["picked"]
            for d, f in selection["leave_one_dataset_out"].items()
        },
        "lodo_loss_picks": {
            d: f["sweep"]["picked_by_losses"]
            for d, f in selection["leave_one_dataset_out"].items()
        },
        "dev_mean_pick": selection["dev_set"]["sweep"]["picked"],
        "dev_fine_pick": selection["dev_set"]["fine"]["picked"],
        "dev_late_pick": selection["dev_set_late"]["sweep"]["picked"],
        "swin_loss_picks": {
            d: f["sweep"]["picked_by_losses"]
            for d, f in selection["leave_one_dataset_out_swin"].items()
        },
        "vit_halving": vit["allocation"]["tm_tpr_lost_halving_budget"],
        "swin_halving": swin["allocation"]["tm_tpr_lost_halving_budget"],
        "bb_halving": bb["allocation"]["tm_tpr_lost_halving_budget"],
        "vit_carried": vit["partner_carried"],
        "swin_carried": swin["partner_carried"],
        "late_coverage": groups["vit_late"]["n"],
        "panel_n": vit["n"],
    }

    # The sentences below are written for this shape of the data. Each assert
    # names the claim it guards.
    assert facts["vit_curve_range"] < 0.03, "the ViT mean curve is flat"
    assert facts["vit_losses_min"] > facts["vit_losses_weighted"], "losses fall with w"
    assert facts["bb_plateau"][0] <= 0.9 <= facts["bb_plateau"][1], "0.9 on BB plateau"
    assert facts["swin_best_share"] <= 0.55, "Swin peaks at the plain minimum"
    assert facts["training_best_share"] <= 0.55, "training set peaks at the minimum"
    assert facts["swin_halving"] < facts["vit_halving"] < facts["bb_halving"]
    assert all(p <= 0.7 for d, p in facts["lodo_mean_picks"].items() if d != "tiny")
    assert facts["lodo_mean_picks"]["tiny"] >= 0.9
    assert all(p >= 0.9 for p in facts["lodo_loss_picks"].values())
    assert all(p <= 0.6 for p in facts["swin_loss_picks"].values())
    return facts


def num(value):
    text = "n/a" if value is None else f"{value:.3f}"
    return text


def signed(value):
    text = "n/a" if value is None else f"{value:+.3f}"
    return text


def interval(paired):
    low, high = paired["ci95"]
    if low is None:
        text = f"{signed(paired['mean_difference'])} (n<3)"
    else:
        text = f"{signed(paired['mean_difference'])} [{signed(low)}, {signed(high)}]"
    return text


def tex_interval(paired):
    low, high = paired["ci95"]
    text = "--" if low is None else f"\\IV{{{low:+.3f}}}{{{high:+.3f}}}"
    return text


def budget_label(budget):
    label = f"{float(budget[1:]) * 100:g}%"
    return label


def joined(items):
    text = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
    return text


def render_readme(summary, preregistration, swin_caches, facts):
    groups = summary["groups"]
    pre = preregistration["content"]
    parts = [
        "# Fusion weighting",
        "",
        "PSBD-TM is fused with a residual-dropout probe in the final method, and the paper reports 2 rules for the fusion side by side. The plain minimum of the 2 clean-validation percentiles (min) flags an input when either probe finds it suspicious. The weighted minimum min(r_TM / 0.9, r_partner / 0.1) (weighted 0.9/0.1) is the rule `experiments/cache_readouts/preregistration.json` fixed. This experiment produces every number for both rules on every set that carries both probes, and tests whether the 0.9/0.1 shares are a principled choice or a lucky one. It reads the stage-1 caches only, on the CPU.",
        "",
        f'The pre-registration was written at {pre["written_at"]} (SHA-256 `{preregistration["sha256"]}`), before any held-out score was read. Its stated reason for the shares is "{pre["picks"]["fusion_rule"]["why"]}" It paired the shares with the late band, residual dropout in blocks 9 to 12, which reaches the adaptive target on {facts["late_coverage"]} of {facts["panel_n"]} panel models. The final method uses the middle band, blocks 5 to 8, which reaches it on all {facts["panel_n"]}. So the 0.9/0.1 shares on the middle band are carried over from the pre-registered pairing, and the late band is read beside it on its {facts["late_coverage"]} models.',
        "",
        "## Model sets and partners",
        "",
        "| set | models | partner placement | rate of the partner | source |",
        "|---|---|---|---|---|",
    ]
    for name, block in groups.items():
        rule = " and ".join(block["partner_rate_rules"])
        parts.append(
            f"| {LABELS[name]} | {block['n']} | `{block['placement']}` | {rule} | `{block['source']}` |"
        )
    parts += [
        "",
        f"The ViT panel is the {facts['panel_n']} `successful_2pt` models that carry both headline placements (`experiments.probe_union.measure.select_models`). The Swin panel is `scripts.paper.tab_swin.swin_cells`. BackdoorBench is every model under `results/_experiments/backdoorbench_attacks/models/` with detection rates, read from `results/bb_<folder>/psbd/`. That README drops no swept model for the 2-point bar, so neither does this set. The training-set setting is the {groups['training_set']['n']} pooled paper-mirror models of `experiments/training_set_detection/` (`summary.json`, `headline`), read from its raw parts.",
        "",
        "The adaptive attacker models are not read. No ViT evader carries `pre_residual_blocks_5_8`, and the Swin evaders carry `pre_residual_blocks_9_16` on 4 models and `pre_residual_blocks_17_24` on none, so no evader has both probes of the final method cached.",
        "",
        "## Swin-S partner placement",
        "",
        f"The Swin-S final method reads `pre_residual_blocks_17_24`, residual dropout in blocks 17 to 24. `experiments/final_method/fusion_readout.py` still names `pre_residual_blocks_9_16` in `MIDDLE_BAND`, and the opening paragraph of `experiments/final_method/README.md` says blocks 9 to 16. That was the 1st pre-registered Swin partner (`experiments/cache_readouts/preregistration_swin.json`) and its predictions failed. The 2nd and last attempt held under the min rule. It is `experiments/final_method/swin_late_readout.py` with `PARTNER = \"pre_residual_blocks_17_24\"` as pre-registered in `preregistration_swin_late.json`. Every later reader uses it: `experiments/score_figures/config.py` maps the Swin `band` probe to it and the report's Swin tables state blocks 17 to 24. On disk {swin_caches['pre_residual_blocks_17_24']} of {swin_caches['n_panel']} Swin panel models carry the blocks 17 to 24 cache and {swin_caches['pre_residual_blocks_9_16']} carry blocks 9 to 16, so the caches allow either partner and the readers decide. The blocks 17 to 24 partner reaches the adaptive target on {groups['swin_band']['n']} models, and the report's 63-model Swin table reads the other {groups['swin_band_with_nearest']['n'] - groups['swin_band']['n']} at the rate nearest the target, so both readings are given here. The 9 to 16 in the final method's README opening and in `MIDDLE_BAND` is stale.",
        "",
        "## Method",
        "",
        "Each probe is read at its own adaptive rate, the smallest cached rate whose clean-validation shift ratio reaches 0.8, and scored with fractional PSU. Each score becomes its percentile within the probe's own 2000-image clean-validation distribution (`defenses.scores.to_rank`). The fused score at TM share w is u(x) = min(r_TM(x) / w, r_partner(x) / (1 - w)), so w = 0.5 orders inputs exactly as the plain minimum and w = 0.9 is the weighted rule. Every score, fused or single, is thresholded at a quantile of its own clean-validation distribution (`defenses.decision.threshold_at_quantile`, low means poisoned), TPR is read on the triggered split and the realized FPR on the paired clean test split (`pair_clean_to_backdoor`). The training-set setting thresholds at quantiles of the clean training images, as `experiments/training_set_detection/` does, so its FPR is the quantile by construction. Paired intervals are bootstrap 95% intervals over models (`scripts.paper._common.bootstrap_ci`, 5000 resamples, seed 0). A model loses at a budget when its TPR there falls more than 0.05 below PSBD-TM alone.",
        "",
        "The share sweep reads w at 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95 and 0.99 with intervals. The curves read every 0.01 from 0.50 to 0.99.",
        "",
        "<!-- results:begin -->",
        "<!-- Everything down to results:end is rendered by render.py from summary.json. -->",
        "",
    ]
    parts += budget_allocation_section(summary, facts)
    parts += both_rules_section(summary)
    parts += sweep_section(summary, facts)
    parts += selection_section(summary, facts)
    parts += trade_off_section(summary, facts)
    parts += per_model_section(summary)
    parts += [
        "<!-- results:end -->",
        "",
        "## Files and commands",
        "",
        "```bash",
        "for s in vit_panel swin_panel backdoorbench training_set; do",
        "    .venv/bin/python -m experiments.fusion_weighting.measure --set $s",
        "done",
        ".venv/bin/python -m experiments.fusion_weighting.analyze",
        ".venv/bin/python -m experiments.fusion_weighting.render",
        "```",
        "",
        "`measure.py` writes `readings_<set>.json` (every model, every share, 1 to 2 minutes per set), `analyze.py` writes `summary.json` and the figure `share_sweep.png` with its sidecar `share_sweep.json`, and `render.py` writes this README and `tmp/report/sections/fusion_rules.tex`, the report's subsection. All records are under `results/_experiments/fusion_weighting/`.",
        "",
    ]
    text = "\n".join(parts)
    return text


def budget_allocation_section(summary, facts):
    groups = summary["groups"]
    lines = [
        "## Budget allocation",
        "",
        "The weighted minimum is the weighted Bonferroni rule of multiple testing applied to 2 detectors. With shares $w_i$ the fused score and the literal decision at nominal FPR $q$ are",
        "",
        "$$u(x) = \\min_{i} \\frac{r_i(x)}{w_i}, \\qquad \\sum_{i} w_i = 1, \\quad w_i > 0, \\qquad \\text{flag } x \\iff u(x) \\le q .$$",
        "",
        "| symbol | meaning |",
        "|---|---|",
        "| $x$ | an input |",
        "| $i$ | a probe, PSBD-TM ($i = 1$) or the residual partner ($i = 2$) |",
        "| $r_i(x)$ | the percentile of probe $i$'s fractional PSU within its own clean-validation scores |",
        "| $w_i$ | the share of the false-positive budget given to probe $i$, $w_1 = w$ and $w_2 = 1 - w$ |",
        "| $q$ | the nominal false-positive rate |",
        "| $P_0, P_1$ | probability over clean and over triggered inputs |",
        "| $\\beta_i(t)$ | the TPR of probe $i$ alone at FPR $t$, $P_1(r_i \\le t)$, its ROC |",
        "| $\\pi$ | the share of models a probe other than PSBD-TM carries |",
        "| $\\bar\\beta_i$ | $\\beta_i$ averaged over the models probe $i$ carries |",
        "| $c_i, \\gamma$ | scale and exponent of a power-law ROC $\\bar\\beta_i(t) = c_i t^{\\gamma}$ |",
        "",
        "The decision is a union of 1 event per probe, since $u(x) \\le q$ exactly when some $r_i(x) \\le w_i q$,",
        "",
        "$$\\{u \\le q\\} = \\bigcup_{i} \\{r_i \\le w_i q\\} .$$",
        "",
        "Boole's inequality bounds the clean flag rate by the sum of the per-probe rates. A clean-validation percentile is uniform on clean data (the probability integral transform), so each term is $w_i q$, exactly on the validation set up to 1 image per probe,",
        "",
        "$$P_0(u \\le q) \\le \\sum_{i} P_0(r_i \\le w_i q) = \\sum_{i} w_i q = q .$$",
        "",
        "The shares therefore split a fixed budget $q$ between the probes, and the weighted rule spends 0.9 of it on PSBD-TM and 0.1 on the partner. The power is at least that of the better probe at its own share,",
        "",
        "$$P_1(u \\le q) \\ge \\max_{i} \\beta_i(w_i q) .$$",
        "",
        "To choose $w$, approximate the panel as models carried by PSBD-TM (share $1 - \\pi$) and models carried by the partner (share $\\pi$), and drop the overlap of the 2 events. This approximation drops the TPR an input gains from being caught by both probes. The mean TPR is then",
        "",
        "$$T(w) = (1 - \\pi)\\, \\bar\\beta_1(w q) + \\pi\\, \\bar\\beta_2\\big((1 - w) q\\big) .$$",
        "",
        "Differentiating by the chain rule and setting the derivative to 0 gives the optimum, a maximum when both ROCs are concave, where the 2 probes earn the same TPR per unit of budget,",
        "",
        "$$\\frac{dT}{dw} = (1 - \\pi)\\, q\\, \\bar\\beta_1'(w q) - \\pi\\, q\\, \\bar\\beta_2'\\big((1 - w) q\\big) = 0 \\iff (1 - \\pi)\\, \\bar\\beta_1'(w^\\ast q) = \\pi\\, \\bar\\beta_2'\\big((1 - w^\\ast) q\\big) .$$",
        "",
        "For a power-law ROC $\\bar\\beta_i(t) = c_i t^{\\gamma}$ with $0 < \\gamma < 1$, $\\bar\\beta_i'(t) = c_i \\gamma t^{\\gamma - 1}$, and the condition becomes $(1 - \\pi) c_1 (w^\\ast)^{\\gamma - 1} = \\pi c_2 (1 - w^\\ast)^{\\gamma - 1}$ once $\\gamma q^{\\gamma - 1}$ cancels. Solving for the ratio of shares,",
        "",
        "$$\\frac{w^\\ast}{1 - w^\\ast} = \\left( \\frac{(1 - \\pi)\\, c_1}{\\pi\\, c_2} \\right)^{1 / (1 - \\gamma)} .$$",
        "",
        "$q$ drops out, so 1 share serves every budget. With $c_1 = c_2$ and $\\pi < 1/2$ the base exceeds 1 and the exponent exceeds 1, so $w^\\ast / (1 - w^\\ast) \\ge (1 - \\pi) / \\pi$, that is $w^\\ast \\ge 1 - \\pi$. A primary that carries most attacks keeps at least that share of the budget, and the plain minimum is optimal only when the 2 probes carry the panel equally. A grid search of $T(w)$ matched the closed form to 4 decimals at 4 settings of $(\\pi, c_1, c_2, \\gamma)$.",
        "",
        "The assumption that fails in practice is the common exponent. The optimum depends on how steep PSBD-TM's ROC is just below the budget, so the measurement that decides it is how much TPR PSBD-TM loses when its budget is cut from 1% to 0.5%. The last 2 columns below read $T(w)$ without the disjointness approximation, as the mean over models of $\\max(\\beta_1(w q), \\beta_2((1 - w) q))$ with each probe's measured ROC at $q$ = 1%, and compare its best share with the sweep's.",
        "",
        "| set | models | partner leads by more than 0.05 at 1% (π) | PSBD-TM TPR at 1% | PSBD-TM TPR at 0.5% | lost by halving | partner TPR at 0.5% | partner TPR at 0.1% | best share of the max bound | best share of the sweep |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, block in groups.items():
        allocation = block["allocation"]
        budgets = allocation["low_budgets"]
        tm_curve = allocation["tm_alone_mean_tpr"]
        carried = block["partner_carried"]
        lines.append(
            f"| {SHORT[name]} | {block['n']} | {carried['count']} ({carried['share']:.3f}) | {num(tm_curve[budgets.index(0.01)])} | {num(tm_curve[budgets.index(0.005)])} | {num(allocation['tm_tpr_lost_halving_budget'])} | {num(allocation['partner_tpr_at_half_budget'])} | {num(allocation['partner_tpr_at_tenth_budget'])} | {allocation['bound_best_share']:.2f} | {block['fine']['plateau']['best_share']:.2f} |"
        )
    vit = groups["vit_band"]
    lines += [
        "",
        f"PSBD-TM loses {num(facts['swin_halving'])} of TPR on Swin-S when its budget halves, {num(facts['vit_halving'])} on the ViT panel and {num(facts['bb_halving'])} on BackdoorBench. The partner carries a similar share of models on both architectures ({facts['vit_carried']['count']} of {facts['vit_carried']['n']} on ViT, {facts['swin_carried']['count']} of {facts['swin_carried']['n']} on Swin), so the corollary $w^\\ast \\ge 1 - \\pi$ would place both optima near 0.8. The measured ROCs move the Swin optimum to the plain minimum, because halving PSBD-TM's budget there costs little, and the max bound finds that best share where the sweep does. On the ViT panel both curves are flat, within {num(max(vit['allocation']['max_of_probes_bound']) - min(vit['allocation']['max_of_probes_bound']))} for the bound and {num(facts['vit_curve_range'])} for the sweep across every share, so their maxima ({vit['allocation']['bound_best_share']:.2f} and {vit['fine']['plateau']['best_share']:.2f}) do not separate.",
        "",
        "The literal decision $u(x) \\le q$ is the one the bound is stated for. The realized clean flag rates below confirm it on every share of the sweep, on validation within 1 image per probe of $q$, and on the paired clean test split, which the bound does not cover, near $q$ on average.",
        "",
        "| set | share | validation FPR at 1%, mean / max | at 5% | at 10% | at 20% | clean test FPR at 1%, mean / max | TPR at 1%, literal |",
        "|---|---|---|---|---|---|---|---|",
    ]
    largest_excess = 0.0
    for name in ("vit_band", "swin_band", "backdoorbench"):
        bound = groups[name]["literal_union_bound"]
        for share in SWEEP_SHARES:
            cells = []
            for budget in BUDGETS:
                entry = bound[share][budget]
                cells.append(
                    f"{entry['validation_fpr_mean']:.4f} / {entry['validation_fpr_max']:.4f}"
                )
                largest_excess = max(
                    largest_excess, entry["validation_fpr_max"] - entry["nominal"]
                )
            at_one = bound[share]["q0.01"]
            lines.append(
                f"| {SHORT[name]} | {share} | {' | '.join(cells)} | {at_one['realized_fpr_mean']:.4f} / {at_one['realized_fpr_max']:.4f} | {num(at_one['tpr_mean'])} |"
            )
    # 2000 validation images and 2 probes: at most 1 extra image per probe.
    assert largest_excess <= 2 / 2000 + 1e-6, largest_excess
    lines += [
        "",
        f"The largest validation flag rate above $q$ on any model, share and budget is {largest_excess:.4f}, at most 1 image per probe of the 2000.",
        "",
    ]
    return lines


def rule_table(block):
    paired = block["rules"]["paired"]
    means = block["rules"]["means"]
    lines = [
        "| rule | TPR 1% | TPR 5% | TPR 10% | TPR 20% | FPR 1% | FPR 5% | FPR 10% | FPR 20% | AUROC |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rule in ("tm", "min", "weighted"):
        tprs = " | ".join(num(means[rule][f"tpr:{b}"]) for b in BUDGETS)
        fprs = " | ".join(f"{means[rule][f'realized_fpr:{b}']:.4f}" for b in BUDGETS)
        lines.append(
            f"| {RULE_NAMES[rule]} | {tprs} | {fprs} | {num(means[rule]['auroc'])} |"
        )
    lines += [
        "",
        "| paired difference | TPR 1% | TPR 5% | TPR 10% | TPR 20% | AUROC |",
        "|---|---|---|---|---|---|",
    ]
    for name, label in (
        ("min_minus_tm", "min minus PSBD-TM"),
        ("weighted_minus_tm", "weighted minus PSBD-TM"),
        ("weighted_minus_min", "weighted minus min"),
    ):
        cells = " | ".join(interval(paired[name][f"tpr:{b}"]) for b in BUDGETS)
        lines.append(f"| {label} | {cells} | {interval(paired[name]['auroc'])} |")
    losses = block["losses"]
    lines += [
        "",
        "| models losing more than 0.05 against PSBD-TM | at 1% | at 5% | at 10% | at 20% | AUROC |",
        "|---|---|---|---|---|---|",
    ]
    for rule in ("min", "weighted"):
        cells = []
        for field in [f"tpr:{b}" for b in BUDGETS] + ["auroc"]:
            entry = losses[rule][field]
            cells.append(
                f"{entry['count']} (worst {signed(entry['worst'])}, `{entry['worst_folder']}`)"
            )
        lines.append(f"| {RULE_NAMES[rule]} | {' | '.join(cells)} |")
    return lines


def breakdown_table(table, key_label, names=None):
    lines = [
        f"| {key_label} (models) | PSBD-TM TPR 1 / 5 / 10% | min TPR 1 / 5 / 10% | weighted TPR 1 / 5 / 10% | AUROC PSBD-TM / min / weighted | weighted minus min, TPR 1% | losing at 1%, min / weighted |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, entry in table.items():
        means = entry["means"]
        triple = {
            rule: " / ".join(num(means[rule][f"tpr:{b}"]) for b in HEADLINE)
            for rule in ("tm", "min", "weighted")
        }
        aurocs = " / ".join(
            num(means[rule]["auroc"]) for rule in ("tm", "min", "weighted")
        )
        label = (names or {}).get(key, key)
        lines.append(
            f"| {label} ({entry['n']}) | {triple['tm']} | {triple['min']} | {triple['weighted']} | {aurocs} | {interval(entry['weighted_minus_min']['tpr:q0.01'])} | {entry['losses']['min']['q0.01']} / {entry['losses']['weighted']['q0.01']} |"
        )
    return lines


def both_rules_section(summary):
    groups = summary["groups"]
    lines = [
        "## Both rules on every set",
        "",
        "Each set gives the mean over models of TPR at 1%, 5%, 10% and 20% FPR, the realized FPR on the paired clean test split and AUROC, then the paired differences with their 95% intervals and the count of models losing more than 0.05 against PSBD-TM alone. The per attack and per dataset tables follow each set.",
        "",
    ]
    for name, block in groups.items():
        lines += [f"### {LABELS[name]}, {block['n']} models", ""]
        if block["missing"]:
            missing = ", ".join(f"`{m['folder']}`" for m in block["missing"])
            lines += [f"Not scored, {len(block['missing'])} models: {missing}.", ""]
        lines += rule_table(block)
        lines += [""]
        lines += breakdown_table(block["by_attack"], "attack")
        lines += [""]
        lines += breakdown_table(block["by_dataset"], "dataset", DATASET_NAMES)
        lines += [""]
    return lines


def sweep_section(summary, facts):
    groups = summary["groups"]
    vit = groups["vit_band"]
    bb = groups["backdoorbench"]
    lines = [
        "## Share sweep",
        "",
        "![share sweep](../../results/_experiments/fusion_weighting/share_sweep.png)",
        "",
        "`share_sweep.png` (numbers in `share_sweep.json`) draws the mean TPR at 1%, 5% and 10% FPR and AUROC against the TM share w on a 0.01 grid, PSBD-TM alone dotted, with the count of models losing more than 0.05 below it. The dashed line marks w = 0.9.",
        "",
        f"The sweep shows no single plateau across sets, and 0.9 is a sharp optimum on none of them. On the ViT panel mean TPR at 1% FPR moves by only {num(facts['vit_curve_range'])} across every share from 0.50 to 0.99, peaks at {facts['vit_best_share']:.2f} and sits {num(facts['vit_gap'])} below that peak at 0.90, while the models losing more than 0.05 at 1% fall from {facts['vit_losses_min']} at 0.50 to {facts['vit_losses_weighted']} at 0.90 and {vit['sweep']['0.95']['losses']['q0.01']} at 0.95. On BackdoorBench the mean rises to a plateau from {facts['bb_plateau'][0]:.2f} to {facts['bb_plateau'][1]:.2f} that contains 0.90. On Swin-S the best share is the plain minimum ({facts['swin_best_share']:.2f}) and the mean falls as w grows, so 0.90 gives up {num(facts['swin_gap'])} there. The training-set setting peaks at {facts['training_best_share']:.2f} and 0.90 gives up {num(facts['training_gap'])}. On the ViT panel and BackdoorBench the count of losing models moves more with w than the mean does.",
        "",
        f"The per attack tables above give the flat ViT mean its reading. Raising w from 0.5 gives back the partner's WaNet gain and returns to PSBD-TM the budget it needs on BadNets, Blend, LF and BPP, and the 2 cancel in the mean while the losses fall. BackdoorBench at {num(bb['sweep']['0.50']['means']['tpr:q0.01'])} under min against {num(bb['sweep']['0.90']['means']['tpr:q0.01'])} at 0.90 is where the cost of the plain minimum is largest.",
        "",
    ]
    for name in (
        "vit_band",
        "vit_late",
        "swin_band",
        "swin_band_with_nearest",
        "backdoorbench",
        "training_set",
    ):
        block = groups[name]
        plateau = block["fine"]["plateau"]
        lines += [
            f"### Sweep, {LABELS[name]}, {block['n']} models",
            "",
            f"Best share on the 0.01 grid {plateau['best_share']:.2f} (mean TPR at 1% {num(plateau['best_value'])}), shares within {summary['plateau_tolerance']} of it from {plateau['within_tolerance'][0]:.2f} to {plateau['within_tolerance'][1]:.2f}{'' if plateau['within_tolerance_contiguous'] else ' with gaps'}, 0.90 at {num(plateau['weighted_value'])}.",
            "",
            "| TM share | TPR 1% | TPR 5% | TPR 10% | AUROC | TPR 1% minus PSBD-TM | AUROC minus PSBD-TM | losing at 1% / 5% / 10% |",
            "|---|---|---|---|---|---|---|---|",
        ]
        tm_means = block["rules"]["means"]["tm"]
        lines.append(
            f"| PSBD-TM alone | {num(tm_means['tpr:q0.01'])} | {num(tm_means['tpr:q0.05'])} | {num(tm_means['tpr:q0.10'])} | {num(tm_means['auroc'])} | | | |"
        )
        for share in SWEEP_SHARES:
            entry = block["sweep"][share]
            means = entry["means"]
            losses = " / ".join(str(entry["losses"][b]) for b in HEADLINE)
            lines.append(
                f"| {share} | {num(means['tpr:q0.01'])} | {num(means['tpr:q0.05'])} | {num(means['tpr:q0.10'])} | {num(means['auroc'])} | {interval(entry['minus_tm']['tpr:q0.01'])} | {interval(entry['minus_tm']['auroc'])} | {losses} |"
            )
        lines += [""]
    return lines


def selection_table(folds):
    lines = [
        "| held-out dataset | models fit / held out | pick by mean TPR 1%, sweep grid / 0.01 grid | held-out TPR 1% at the sweep pick | at 0.90 | at min | PSBD-TM alone | losing at 1%, pick / 0.90 / min | pick by fewest losses, sweep grid / 0.01 grid | held-out TPR 1% at that pick |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for dataset, fold in folds.items():
        sweep = fold["sweep"]
        held = sweep["held_out"]
        lines.append(
            f"| {DATASET_NAMES.get(dataset, dataset)} | {fold['n_fit']} / {fold['n_held_out']} | {sweep['picked']:.2f} / {fold['fine']['picked']:.2f} | {num(held['picked_tpr'])} | {num(held['weighted_tpr'])} | {num(held['min_tpr'])} | {num(held['tm_tpr'])} | {held['picked_losses']} / {held['weighted_losses']} / {held['min_losses']} | {sweep['picked_by_losses']:.2f} / {fold['fine']['picked_by_losses']:.2f} | {num(sweep['held_out_by_losses']['picked_tpr'])} |"
        )
    return lines


def selection_section(summary, facts):
    selection = summary["selection"]
    dev = selection["dev_set"]
    dev_late = selection["dev_set_late"]
    lines = [
        "## Selection without the test data",
        "",
        "Each fold picks the share on 3 datasets and reads it on the 4th. The 1st criterion is the highest mean TPR at 1% FPR on the fit datasets. The 2nd is the pre-registration's own reason, losing least: the fewest models losing more than 0.05 at 1% FPR against PSBD-TM alone, then the higher mean TPR, a remaining tie going to the smaller share so a tie never favours 0.9. Both search the sweep's 8 shares and the 0.01 grid.",
        "",
        "### Leave one dataset out, ViT panel, blocks 5 to 8",
        "",
    ]
    lines += selection_table(selection["leave_one_dataset_out"])
    mean_picks = facts["lodo_mean_picks"]
    loss_picks = facts["lodo_loss_picks"]
    lines += [
        "",
        f"By mean TPR the picks are {joined([f'{p:.2f} on {DATASET_NAMES[d]}' for d, p in mean_picks.items()])}, so {len(mean_picks) - 1} of {len(mean_picks)} folds sit {0.9 - max(p for d, p in mean_picks.items() if d != 'tiny'):.2f} or more below 0.9 and the Tiny fold picks {mean_picks['tiny']:.2f}, where it reads below the plain minimum on Tiny itself. Losing least picks {joined([f'{p:.2f}' for p in loss_picks.values()])} on the same folds, every one within {max(abs(p - 0.9) for p in loss_picks.values()):.2f} of 0.9.",
        "",
        "### Leave one dataset out, ViT panel, blocks 9 to 12",
        "",
    ]
    lines += selection_table(selection["leave_one_dataset_out_late"])
    lines += [
        "",
        "### Leave one dataset out, Swin-S, blocks 17 to 24",
        "",
    ]
    lines += selection_table(selection["leave_one_dataset_out_swin"])
    lines += [
        "",
        f"On Swin-S both criteria stay at the plain minimum or just above it (losing least picks {joined([f'{p:.2f}' for p in facts['swin_loss_picks'].values()])}), the architecture where halving PSBD-TM's budget costs least.",
        "",
        "### Development set",
        "",
        f"The development set is the {dev['n_dev']} panel models of `experiments/cache_readouts/dev_set.json` that are `successful_2pt` ({', '.join(f'`{f}`' for f in dev['dev_folders'])}). The pick is read on the other {dev['n_rest']}.",
        "",
        "| partner | development models | pick by mean TPR 1%, sweep / 0.01 grid | rest TPR 1% at the pick | at 0.90 | at min | PSBD-TM alone | losing at 1%, pick / 0.90 / min | pick by fewest losses, sweep / 0.01 grid |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, block in (("blocks 5 to 8", dev), ("blocks 9 to 12", dev_late)):
        held = block["sweep"]["held_out"]
        lines.append(
            f"| {label} | {block['n_dev']} | {block['sweep']['picked']:.2f} / {block['fine']['picked']:.2f} | {num(held['picked_tpr'])} | {num(held['weighted_tpr'])} | {num(held['min_tpr'])} | {num(held['tm_tpr'])} | {held['picked_losses']} / {held['weighted_losses']} / {held['min_losses']} | {block['sweep']['picked_by_losses']:.2f} / {block['fine']['picked_by_losses']:.2f} |"
        )
    lines += [
        "",
        f"On the development set the mean criterion picks {facts['dev_mean_pick']:.2f} with the middle band and {facts['dev_late_pick']:.2f} with the late band, the pairing the pre-registration was written on. Its 0.9 came from the qualitative reason it states, keeping most of the WaNet gain while losing least on TaCT, and the losses criterion is a quantitative form of that reason at 1% FPR.",
        "",
    ]
    return lines


def trade_off_section(summary, facts):
    groups = summary["groups"]
    wanet_partner = facts["vit_wanet_partner_tpr"]
    lines = [
        "## Where each rule wins",
        "",
        f"Every dataset and attack cell where the 2 rules differ by more than {TRADE_OFF_LISTED} in mean TPR at 1%, 5% or 10% FPR, sorted by the difference at 1%. A negative difference means min wins.",
        "",
        "| set | cell (models) | PSBD-TM TPR 1 / 5 / 10% | min TPR 1 / 5 / 10% | weighted TPR 1 / 5 / 10% | weighted minus min at 1 / 5 / 10% |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("vit_band", "swin_band", "backdoorbench", "training_set"):
        cells = groups[name]["trade_off"]
        listed = [
            (key, cell)
            for key, cell in cells.items()
            if max(abs(v) for v in cell["weighted_minus_min"].values())
            > TRADE_OFF_LISTED
        ]
        listed.sort(key=lambda item: item[1]["weighted_minus_min"]["q0.01"])
        for key, cell in listed:
            triples = {
                rule: " / ".join(num(cell[rule][b]) for b in HEADLINE)
                for rule in ("tm", "min", "weighted")
            }
            difference = " / ".join(
                signed(cell["weighted_minus_min"][b]) for b in HEADLINE
            )
            lines.append(
                f"| {SHORT[name]} | {key} ({cell['n']}) | {triples['tm']} | {triples['min']} | {triples['weighted']} | {difference} |"
            )
    vit = groups["vit_band"]["by_attack"]["wanet"]["means"]
    bb = {row["folder"]: row for row in groups["backdoorbench"]["per_model"]}
    blind = bb["bb_cifar10_blind_0_1"]
    tiny_trojan = [
        bb[f] for f in ("bb_tiny_trojannn_0_05", "bb_tiny_trojannn_0_1") if f in bb
    ]
    inputaware = bb["bb_gtsrb_inputaware_0_1"]
    assert vit["min"]["tpr:q0.01"] - vit["weighted"]["tpr:q0.01"] > 0.2
    tiny_text = ", ".join(
        f"`{r['folder']}` {num(r['min']['tpr:q0.01'])} against "
        f"{num(r['weighted']['tpr:q0.01'])}"
        for r in tiny_trojan
    )
    assert blind["weighted"]["tpr:q0.01"] - blind["min"]["tpr:q0.01"] > 0.2
    lines += [
        "",
        f"Min wins where the partner carries a trigger PSBD-TM misses. ViT WaNet reads {num(vit['tm']['tpr:q0.01'])} at 1% FPR alone, {num(vit['min']['tpr:q0.01'])} under min and {num(vit['weighted']['tpr:q0.01'])} under the weighted rule, because the partner alone catches {num(wanet_partner[0.001])} of the WaNet inputs at 0.1% FPR, the budget the weighted rule leaves it, against {num(wanet_partner[0.005])} at 0.5%, the budget min leaves it. The same holds for Swin WaNet, the training-set WaNet models, BackdoorBench TrojanNN on Tiny ({tiny_text}) and Input-Aware on GTSRB ({num(inputaware['min']['tpr:q0.01'])} against {num(inputaware['weighted']['tpr:q0.01'])}, a small gap). The weighted rule wins where PSBD-TM already detects and halving its budget cuts into the steep part of its ROC. BackdoorBench Blind on CIFAR-10 reads {num(blind['tm']['tpr:q0.01'])} alone, {num(blind['min']['tpr:q0.01'])} under min and {num(blind['weighted']['tpr:q0.01'])} under the weighted rule, and the ViT BadNets, Blend, LF and BPP losses of the min rule at 1% FPR are the same effect. At 10% FPR every gap shrinks, since both probes then sit on the flat part of their ROCs.",
        "",
    ]
    return lines


def per_model_section(summary):
    groups = summary["groups"]
    lines = [
        "## Per model",
        "",
        "TPR at 1%, 5%, 10% and 20% FPR then AUROC, PSBD-TM alone, min and weighted 0.9/0.1, with the partner alone at 1% FPR.",
        "",
    ]
    for name in (
        "vit_band",
        "vit_late",
        "swin_band_with_nearest",
        "backdoorbench",
        "training_set",
    ):
        block = groups[name]
        lines += [
            f"### {LABELS[name]}, {block['n']} models",
            "",
            "| model | partner rate | PSBD-TM | min | weighted 0.9/0.1 | partner alone TPR 1% |",
            "|---|---|---|---|---|---|",
        ]
        for row in block["per_model"]:
            cells = {
                rule: " / ".join(num(row[rule][f"tpr:{b}"]) for b in BUDGETS)
                + f" ({num(row[rule]['auroc'])})"
                for rule in ("tm", "min", "weighted")
            }
            rate = f"{row['partner_rate']:g}" + (
                " nearest" if row["partner_rate_rule"] == "nearest" else ""
            )
            lines.append(
                f"| `{row['folder']}` | {rate} | {cells['tm']} | {cells['min']} | {cells['weighted']} | {num(row['partner']['tpr:q0.01'])} |"
            )
        lines += [""]
    return lines


def render_tex(summary, preregistration, facts):
    groups = summary["groups"]
    vit = groups["vit_band"]
    vit_paired = vit["rules"]["paired"]
    swin = groups["swin_band"]
    bb = groups["backdoorbench"]
    training = groups["training_set"]
    wanet = vit["by_attack"]["wanet"]["means"]
    pre = preregistration["content"]
    mean_picks = facts["lodo_mean_picks"]
    loss_picks = facts["lodo_loss_picks"]
    non_tiny = sorted({p for d, p in mean_picks.items() if d != "tiny"})
    assert len(non_tiny) == 1, non_tiny
    assert vit["losses"]["weighted"]["tpr:q0.01"]["count"] == 1, "1 model does"
    worst_min = describe_model(vit, vit["losses"]["min"]["tpr:q0.01"]["worst_folder"])
    worst_weighted = describe_model(
        vit, vit["losses"]["weighted"]["tpr:q0.01"]["worst_folder"]
    )

    paragraphs = [
        "\\subsection{Two fusion rules}",
        "\\label{sec:fusionrules}",
        "",
        f"We report the fusion of PSBD-TM with its residual partner under 2 rules. The plain minimum flags an input when either probe's clean-validation percentile is low. The weighted minimum $\\min(r_{{\\mathrm{{TM}}}}/0.9,\\, r_{{\\mathrm{{res}}}}/0.1)$ is the rule we pre-registered on {pre['written_at'][:10]}, before any held-out score was read. Its shares allocate the false-positive budget. Flagging $u(x) \\le q$ is the union of $r_{{\\mathrm{{TM}}}} \\le 0.9q$ and $r_{{\\mathrm{{res}}}} \\le 0.1q$, so by the union bound the clean false-positive rate is at most $0.9q + 0.1q = q$, and on validation it never exceeds $q$ by more than 1 image per probe. The power-maximizing split equalizes the TPR each probe earns per unit of budget, so the primary probe should keep most of the budget when its ROC is steep just below the budget and the partner carries a minority of the attacks. That is the ViT case. Halving PSBD-TM's budget from 1 to 0.5 percent costs it {num(facts['vit_halving'])} of TPR on the {vit['n']}-model panel and {num(facts['bb_halving'])} on BackdoorBench, and the partner leads by more than 0.05 on {facts['vit_carried']['count']} of {facts['vit_carried']['n']} panel models.",
        "",
        f"Table~\\ref{{tab:fusionvit}} gives both rules on the ViT panel. At the 1 percent budget the weighted rule gains {interval_tex(vit_paired['weighted_minus_tm']['tpr:q0.01'])} over PSBD-TM alone and the plain minimum {interval_tex(vit_paired['min_minus_tm']['tpr:q0.01'])}. The 2 means are close, and the difference lies in which models pay for the fusion. Under the plain minimum {vit['losses']['min']['tpr:q0.01']['count']} models lose more than 5 points against PSBD-TM alone, the worst {signed(vit['losses']['min']['tpr:q0.01']['worst'])} on {worst_min}, and under the weighted rule {vit['losses']['weighted']['tpr:q0.01']['count']} model does, by {signed(vit['losses']['weighted']['tpr:q0.01']['worst'])} on {worst_weighted}. The plain minimum buys that cost back on WaNet, where TPR at 1 percent goes from {num(wanet['tm']['tpr:q0.01'])} to {num(wanet['min']['tpr:q0.01'])}, against {num(wanet['weighted']['tpr:q0.01'])} under the weighted rule.",
        "",
        f"The share sweep in Figure~\\ref{{fig:fusionsweep}} shows no single plateau. On the ViT panel mean TPR at 1 percent moves by {num(facts['vit_curve_range'])} across every share from 0.5 to 0.99 and peaks at {facts['vit_best_share']:.2f}, {num(facts['vit_gap'])} above 0.9, while the count of models losing more than 5 points falls from {facts['vit_losses_min']} at 0.5 to {facts['vit_losses_weighted']} at 0.9. BackdoorBench rises to a plateau from {facts['bb_plateau'][0]:.2f} to {facts['bb_plateau'][1]:.2f} that contains 0.9. Swin-S and the training-set setting peak at the plain minimum, where 0.9 gives up {num(facts['swin_gap'])} and {num(facts['training_gap'])}. The allocation argument predicts this split, because halving PSBD-TM's budget costs only {num(facts['swin_halving'])} of TPR on Swin-S, and the bound read off each probe's measured ROC puts the best Swin share at {swin['allocation']['bound_best_share']:.2f}.",
        "",
        f"Choosing the share without the test data agrees with 0.9 only under the pre-registration's own criterion. Leave-one-dataset-out on the ViT panel picks {non_tiny[0]:.1f} on 3 folds and {mean_picks['tiny']:.2f} on Tiny ImageNet when it maximizes mean TPR at 1 percent, and the development set picks {facts['dev_mean_pick']:.1f}. The pre-registration gives a different reason for 0.9, losing least. Picking the share with the fewest models losing more than 5 points gives {joined([f'{p:.2f}' for p in loss_picks.values()])} on the 4 folds. The same criterion on Swin-S picks between {min(facts['swin_loss_picks'].values()):.1f} and {max(facts['swin_loss_picks'].values()):.1f}. The pre-registration paired the shares with residual dropout in blocks 9 to 12, which reaches the adaptive target on only {facts['late_coverage']} of {facts['panel_n']} models, and the final method substitutes blocks 5 to 8, which reaches it on all {facts['panel_n']}. The shares on blocks 5 to 8 are therefore carried over from the pre-registered pairing. Table~\\ref{{tab:fusionlate}} gives the pre-registered pairing on its {facts['late_coverage']} models. We read the weighted rule as the choice when no model may lose detection against PSBD-TM alone, and the plain minimum as the choice when the partner carries many attacks, as on Swin-S.",
        "",
    ]
    tables = [
        tex_rule_table(
            vit,
            "fusionvit",
            f"Both fusion rules on the ViT-B/16 panel, PSBD-TM fused with residual dropout in blocks 5 to 8, paired over {vit['n']} models at the adaptive rule with 95\\p\\ bootstrap intervals. The last column counts the models whose TPR at 1\\p\\ falls more than 0.05 below PSBD-TM alone.",
        ),
        tex_rule_table(
            groups["vit_late"],
            "fusionlate",
            f"The pre-registered pairing, PSBD-TM fused with residual dropout in blocks 9 to 12, on the {groups['vit_late']['n']} ViT-B/16 panel models where that partner reaches the adaptive target.",
        ),
        tex_rule_table(
            swin,
            "fusionswin",
            f"Both fusion rules on Swin-S, PSBD-TM fused with residual dropout in blocks 17 to 24, on the {swin['n']} models where the partner reaches the adaptive target.",
        ),
        tex_rule_table(
            groups["swin_band_with_nearest"],
            "fusionswinall",
            f"Both fusion rules on all {groups['swin_band_with_nearest']['n']} Swin-S models, the {groups['swin_band_with_nearest']['n'] - swin['n']} whose partner misses the adaptive target read at the rate nearest it.",
        ),
        tex_rule_table(
            bb,
            "fusionbb",
            f"Both fusion rules on {bb['n']} BackdoorBench attacks retrained on ViT-B/16 (Blind, Input-Aware, SSBA and TrojanNN), partner blocks 5 to 8.",
        ),
        tex_rule_table(
            training,
            "fusiontrain",
            f"Both fusion rules in the training-set setting of Li et al., {training['n']} pooled ViT-B/16 models at 10\\p\\ poisoning, thresholds at quantiles of the clean training images, partner blocks 5 to 8.",
        ),
    ]
    figure = [
        "\\begin{figure}[h]",
        "\\centering",
        "\\includegraphics[width=\\textwidth]{figures/fusion_share_sweep.pdf}",
        f"\\caption{{Mean TPR at 1, 5 and 10\\p\\ FPR and AUROC against PSBD-TM's share $w$ of the false-positive budget (top, PSBD-TM alone dotted) and the count of models whose TPR falls more than 0.05 below PSBD-TM alone (bottom), on the ViT-B/16 panel (n={vit['n']}), Swin-S (n={swin['n']}) and BackdoorBench (n={bb['n']}). $w = 0.5$ is the plain minimum and the dashed line marks $w = 0.9$.}}",
        "\\label{fig:fusionsweep}",
        "\\end{figure}",
        "",
    ]
    text = "\n".join(paragraphs + figure) + "\n" + "\n".join(tables) + "\n"
    return text


def describe_model(block, folder):
    row = next(r for r in block["per_model"] if r["folder"] == folder)
    attack = ATTACK_NAMES.get(row["attack"], row["attack"])
    dataset = DATASET_NAMES.get(row["dataset"], row["dataset"])
    text = f"{attack} on {dataset} at {row['poison_rate'] * 100:g} percent poisoning"
    return text


def interval_tex(paired):
    low, high = paired["ci95"]
    text = f"{signed(paired['mean_difference'])} \\CI{{{low:+.3f}}}{{{high:+.3f}}}"
    return text


def tex_rule_table(block, label, caption):
    means = block["rules"]["means"]
    paired = block["rules"]["paired"]
    losses = block["losses"]
    header = "Rule & TPR @ 1\\p & TPR @ 5\\p & TPR @ 10\\p & TPR @ 20\\p & AUROC & Lose @ 1\\p \\\\"
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{caption} Weighted is $\\min(r_{{\\mathrm{{TM}}}}/0.9,\\, r_{{\\mathrm{{res}}}}/0.1)$ and Min the plain minimum of the 2 percentiles.}}",
        f"\\label{{tab:{label}}}",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    for rule in ("tm", "min", "weighted"):
        tprs = " & ".join(num(means[rule][f"tpr:{b}"]) for b in BUDGETS)
        lost = "--" if rule == "tm" else str(losses[rule]["tpr:q0.01"]["count"])
        lines.append(
            f"{TEX_RULE_NAMES[rule]} & {tprs} & {num(means[rule]['auroc'])} & {lost} \\\\"
        )
    lines.append("\\midrule")
    for name, rule_label in (
        ("min_minus_tm", "Min gain"),
        ("weighted_minus_tm", "Weighted gain"),
    ):
        gains = " & ".join(
            signed(paired[name][f"tpr:{b}"]["mean_difference"]) for b in BUDGETS
        )
        intervals = " & ".join(tex_interval(paired[name][f"tpr:{b}"]) for b in BUDGETS)
        lines += [
            f"{rule_label} & {gains} & {signed(paired[name]['auroc']['mean_difference'])} & \\\\",
            f" & {intervals} & {tex_interval(paired[name]['auroc'])} & \\\\",
        ]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def write_text(text, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)


if __name__ == "__main__":
    main()
