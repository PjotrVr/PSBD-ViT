"""Write experiments/probe_union/README.md from results/_experiments/probe_union/probe_union.json.

Every number is read from the record `measure.py` wrote, and every sentence whose
wording depends on the data is computed or guarded by an assert, so a rerun on a
changed panel stops the render instead of printing a stale claim.

    PYTHONPATH=. .venv/bin/python experiments/probe_union/render_readme.py
"""

import json
import os
import statistics

from experiments._paths import experiment_result_path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO_ROOT, "experiments", "probe_union", "README.md")
RECORD = experiment_result_path("probe_union", "probe_union.json")
SET_WORDS = {
    "psbd_tm": "PSBD-TM alone",
    "psbd_tm_rd": "PSBD-TM + PSBD-RD",
    "psbd_tm_attn_branch": "PSBD-TM + attention branch output token mask",
    "adaptive_3probe": "PSBD-TM + attention input dropout + MLP norm-out gain scale",
    "adaptive_4probe": "+ PSBD-RD (4-probe)",
    "all_65_basis": "every basis placement present on all recorded models",
}
ATTACK_ORDER = ("badnet_a2o", "tact", "blend", "lf", "bpp", "wanet", "lc")
ATTACK_COLUMNS = ("psbd_tm", "psbd_tm_rd", "psbd_tm_attn_branch", "adaptive_4probe")
# A model counts as inverted when its PSBD-TM AUROC sits below chance.
INVERTED_BELOW = 0.5


def main():
    with open(RECORD) as handle:
        record = json.load(handle)
    values = fill(record)
    text = TEMPLATE
    for key, value in values.items():
        text = text.replace(f"@@{key}@@", value)
    assert "@@" not in text, text[text.index("@@") - 40 : text.index("@@") + 40]
    with open(OUT, "w") as handle:
        handle.write(text)
    print(f"wrote {OUT}")


def fill(record):
    sets = record["probe_sets"]
    n_models = record["n_models_selected"]
    reference = sets["psbd_tm"]["per_model"]
    by_folder = {
        name: {row["folder"]: row for row in block["per_model"]}
        for name, block in sets.items()
    }
    for block in sets.values():
        assert block["summary"]["n_models"] == n_models

    inverted = sorted(
        (row for row in reference if row["auroc"] < INVERTED_BELOW),
        key=lambda row: row["auroc"],
    )
    assert inverted, "the question names the inverted models"
    tact = [row for row in reference if row["attack"] == "tact"]
    tact_mean = {
        name: statistics.mean(by_folder[name][row["folder"]]["auroc"] for row in tact)
        for name in sets
    }
    gains = {name: block.get("gain_over_psbd_tm") for name, block in sets.items()}
    excluding = [name for name, gain in gains.items() if gain and gain["ci_low"] > 0]
    cheapest = next(
        name
        for name in ("psbd_tm_rd", "psbd_tm_attn_branch", "adaptive_3probe")
        if name in excluding
    )
    rd_costs_tact = tact_mean["psbd_tm_rd"] < tact_mean["psbd_tm"]
    basis_size = len(sets["all_65_basis"]["summary"]["placements"])
    best_mean = max(sets, key=lambda name: sets[name]["summary"]["auroc_mean"])

    values = {
        "N": f"{n_models}",
        "INVERTED_LIST": ", ".join(
            f"`{row['folder']}` ({row['auroc']:.3f})" for row in inverted
        ),
        "BASIS_SIZE": f"{basis_size}",
        "TABLE_SETS": sets_table(sets, gains),
        "TABLE_ATTACKS": attack_table(sets),
        "TABLE_INVERTED": inverted_table(inverted, by_folder),
        "TABLE_TACT": tact_table(tact, by_folder),
        "N_TACT": f"{len(tact)}",
        "CONCLUSION": conclusion(
            sets,
            gains,
            tact_mean,
            excluding,
            cheapest,
            rd_costs_tact,
            best_mean,
            basis_size,
            inverted,
            by_folder,
        ),
    }
    return values


def signed(value):
    text = f"{value:+.3f}"
    return text


def interval(gain):
    text = f"{signed(gain['mean_gain'])} [{signed(gain['ci_low'])}, {signed(gain['ci_high'])}]"
    return text


def join_words(words):
    if len(words) == 1:
        return words[0]
    text = ", ".join(words[:-1]) + " and " + words[-1]
    return text


def sets_table(sets, gains):
    lines = [
        "| Probes | n | AUROC | TPR@FPR 10% | TPR@FPR 20% | Gain over PSBD-TM [95% CI] |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, block in sets.items():
        summary = block["summary"]
        words = SET_WORDS[name]
        if name == "all_65_basis":
            words += f" ({len(summary['placements'])} probes)"
        gain = interval(gains[name]) if gains[name] else "--"
        lines.append(
            f"| {words} | {summary['n_models']} | {summary['auroc_mean']:.3f} "
            f"| {summary['tpr_at_0.10_mean']:.3f} | {summary['tpr_at_0.20_mean']:.3f} "
            f"| {gain} |"
        )
    table = "\n".join(lines)
    return table


def attack_table(sets):
    per_attack = {name: sets[name]["per_attack_mean_auroc"] for name in ATTACK_COLUMNS}
    counts = {}
    for row in sets["psbd_tm"]["per_model"]:
        counts[row["attack"]] = counts.get(row["attack"], 0) + 1
    attacks = [a for a in ATTACK_ORDER if a in counts]
    attacks += sorted(set(counts) - set(attacks))
    header = " | ".join(SET_WORDS[name] for name in ATTACK_COLUMNS)
    lines = [
        f"| Attack | n | {header} |",
        "|---|---:|" + "---:|" * len(ATTACK_COLUMNS),
    ]
    for attack in attacks:
        cells = " | ".join(f"{per_attack[name][attack]:.3f}" for name in ATTACK_COLUMNS)
        lines.append(f"| {attack} | {counts[attack]} | {cells} |")
    table = "\n".join(lines)
    return table


def inverted_table(inverted, by_folder):
    names = list(by_folder)
    lines = [
        "| Model | " + " | ".join(SET_WORDS[name] for name in names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for row in inverted:
        cells = " | ".join(
            f"{by_folder[name][row['folder']]['auroc']:.3f}" for name in names
        )
        lines.append(f"| `{row['folder']}` | {cells} |")
    table = "\n".join(lines)
    return table


def tact_table(tact, by_folder):
    names = list(by_folder)
    lines = [
        "| Model | " + " | ".join(SET_WORDS[name] for name in names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for row in sorted(tact, key=lambda r: r["folder"]):
        cells = " | ".join(
            f"{by_folder[name][row['folder']]['auroc']:.3f}" for name in names
        )
        lines.append(f"| `{row['folder']}` | {cells} |")
    table = "\n".join(lines)
    return table


def conclusion(
    sets,
    gains,
    tact_mean,
    excluding,
    cheapest,
    rd_costs_tact,
    best_mean,
    basis_size,
    inverted,
    by_folder,
):
    n = sets["psbd_tm"]["summary"]["n_models"]
    rd = gains["psbd_tm_rd"]
    rd_word = "excludes" if rd["ci_low"] > 0 else "does not exclude"
    tact_word = "costs" if rd_costs_tact else "lifts"
    tact_delta = abs(tact_mean["psbd_tm_rd"] - tact_mean["psbd_tm"])
    rescued = [
        row["folder"]
        for row in inverted
        if by_folder["psbd_tm_rd"][row["folder"]]["auroc"] >= INVERTED_BELOW
    ]
    rescue_text = (
        f"lifts every inverted model above chance ({', '.join(f'`{f}`' for f in rescued)})"
        if len(rescued) == len(inverted)
        else f"lifts {len(rescued)} of the {len(inverted)} inverted models above chance"
    )
    branch_tact_word = (
        "lifts" if tact_mean["psbd_tm_attn_branch"] > tact_mean["psbd_tm"] else "lowers"
    )
    excluding_words = join_words([SET_WORDS[name] for name in excluding])
    best = sets[best_mean]["summary"]["auroc_mean"]
    text = (
        f"On the {n}-model panel every union reads a higher mean AUROC than PSBD-TM "
        f"alone. The sets whose paired interval excludes 0 are {excluding_words}. "
        f"Adding PSBD-RD {rescue_text}. It {tact_word} TaCT by {tact_delta:.3f} AUROC "
        f"on average over the panel's TaCT models, and its panel-wide paired gain "
        f"{interval(rd)} {rd_word} 0. "
        f"The cheapest union whose interval excludes 0 is "
        f"{SET_WORDS[cheapest]} ({interval(gains[cheapest])}). The attention "
        f"branch output token mask {branch_tact_word} TaCT to "
        f"{tact_mean['psbd_tm_attn_branch']:.3f} against {tact_mean['psbd_tm']:.3f} for "
        f"PSBD-TM alone. "
        f"The highest mean AUROC is {best:.3f}, read by {SET_WORDS[best_mean]}. "
        f"The {basis_size}-probe all-basis union costs {basis_size} times 1 probe "
        f"for a gain of {interval(gains['all_65_basis'])}."
    )
    for name in sets:
        if name != "psbd_tm":
            assert (
                sets[name]["summary"]["auroc_mean"]
                > sets["psbd_tm"]["summary"]["auroc_mean"]
            ), name
    return text


TEMPLATE = """# Probe union on ordinary, non-adaptive backdoored ViT-B/16 models

## Question

H41 (`docs/hypothesis/H41-multi-probe-defense.md`) built a min-rank union of
independent probes to defeat an attacker trained against 1 probed operator.
Nobody trains against a probe on the models the paper's headline reads
(`paper/tables/headline.tex`). Does the same union rule help, hurt or do
nothing on those ordinary models, and does it rescue the models where PSBD-TM
alone reads below chance?

## Method

The union rule is unchanged from H41: for probe j, rank_j(x) is the percentile
of x's fractional PSU (`psu_ratio`, the canon headline statistic) within probe
j's own clean-validation distribution, the combined score is min_j rank_j(x),
and the calibrated threshold is the target-FPR quantile of that combined score
on clean validation. `defenses.decision.multi_probe_auroc` and
`multi_probe_detection` compute both, unmodified.

The record holds the @@N@@ models of the paper panel, the cells successful at
the 2-point clean-accuracy bar that carry both headline placements, selected
exactly as `scripts/paper/tab_headline.py` selects them
(`experiments/probe_union/measure.py:select_models`). Every per-sample PSU is
read from the stage-1 cache under `results/<folder>/psbd/<placement>/`, at the
rate `psbd_metrics.json`'s `adaptive_rate` chose for that placement on that
model, the same reader `cli.analyze` and `scripts/paper/fig_psu_histograms.py`
use.

6 probe sets:

1. `psbd_tm`: PSBD-TM alone (`before_attention_norm_token_mask`), the reference.
2. `psbd_tm_rd`: PSBD-TM plus PSBD-RD (`post_residual`).
3. `psbd_tm_attn_branch`: PSBD-TM plus token masking on the attention branch
   output (`before_attention_residual_token_mask`).
4. `adaptive_3probe`: H41's adaptive-attacker pool minus PSBD-RD and gaussian,
   PSBD-TM plus dropout at the attention input (`before_attention_norm`,
   operator dropout) plus gain scaling of the MLP LayerNorm output
   (`mlp_norm_out_gain_scale`).
5. `adaptive_4probe`: set 4 plus PSBD-RD.
6. `all_65_basis` (the record keeps the key from the historical 65-cell run): every basis
   placement whose `adaptive_rate` is set on all @@N@@ recorded models, found by checking
   the cache rather than hardcoded (`basis_ids_present_on_all_models`), @@BASIS_SIZE@@
   placements.

Each model contributes 1 AUROC and 1 TPR at each of 2 target FPRs (0.10, 0.20)
under the calibrated rule. The paired gain over PSBD-TM alone is measured
within model (matched by folder name) over whichever models both sets cover,
with a 5000-resample bootstrap 95% interval, seed 0
(`scripts/paper/_common.bootstrap_ci`).

Run:

    PYTHONPATH=. .venv/bin/python experiments/probe_union/measure.py
    PYTHONPATH=. .venv/bin/python experiments/probe_union/render_readme.py

Output: `results/_experiments/probe_union/probe_union.json` (per-model
numbers). `scripts/paper/tab_probe_union.py` turns the same record into
`paper/tables/probe_union.tex`.

<!-- results:begin -->
<!-- Everything down to results:end is rendered from probe_union.json by render_readme.py. -->

## Results

The record is `results/_experiments/probe_union/probe_union.json` on the @@N@@ panel models.
The panel reached this size on 2026-09-30, when `vit_gtsrb_lc_0_05_tl1_adv` and
`vit_gtsrb_tact_0_01_cos` got complete caches.

@@TABLE_SETS@@

### Per-attack mean AUROC

@@TABLE_ATTACKS@@

### Models where PSBD-TM alone reads below chance

The models are @@INVERTED_LIST@@.

@@TABLE_INVERTED@@

### The @@N_TACT@@ TaCT models

@@TABLE_TACT@@

## Conclusion

@@CONCLUSION@@

<!-- results:end -->
"""


if __name__ == "__main__":
    main()
