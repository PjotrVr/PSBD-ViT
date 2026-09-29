import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUT = "/lustre/home/pstika/projects/PSBD-ViT/notebooks/all-numbers.ipynb"
cells = []


def md(text):
    cells.append(new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(new_code_cell(text.strip("\n")))


md(r"""
# Every detection number

This notebook lays out every detection reading the repository holds, organized so that a reader can find any number the paper quotes, see the models behind it and see the models it leaves out. Figures come first and the full tables after them. It reads `results/all_numbers/all_numbers.json`, which `scripts/all_numbers.py` writes with 1 row per (model, defense). It also reads `paper/headline.json` for the paper's own values, and it runs on a CPU in seconds. No number in the prose is typed by hand: every value is computed from those files or read from the constant that defines it, and rendered into the text by the cell that computed it.
""")

code(r"""
import os
import sys
from pathlib import Path

# Anchor at the repository root so every default path in the library resolves the
# same way it does from a script, whichever directory the notebook was opened from.
REPO_ROOT = next(
    parent
    for parent in [Path.cwd(), *Path.cwd().parents]
    if (parent / "pyproject.toml").exists()
)
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
""")

code(r"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from matplotlib.colors import TwoSlopeNorm

from data.splits import BENIGN_PROBE_ATTACK, PSBD_HELDOUT_SIZE
from defenses.decision import (
    ADAPTIVE_SHIFT_TARGET,
    PLACEMENT_MATCH_TARGET,
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
)
from detectors import DETECTOR_NAMES
from scripts.all_numbers import MATCHED_SUFFIX
from scripts.coverage_ledger import load_declaration
from scripts.paper._common import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    attack_label,
    bootstrap_ci,
    dataset_label,
    word_list,
)
""")

code(r"""
import scripts.paper._style  # noqa: F401  the figure style every paper figure uses

# The shared style selects the Agg backend for the paper build, which keeps every
# figure out of a notebook, so the inline backend comes back after it.
%matplotlib inline
""")

code(r"""
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 500)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", "{:.3f}".format)

declaration = load_declaration("configs/psbd_basis.json")
with open("results/all_numbers/all_numbers.json") as handle:
    payload = json.load(handle)
with open("paper/headline.json") as handle:
    paper_macros = json.load(handle)
models = pd.DataFrame(payload["models"])
rows = pd.DataFrame(payload["rows"])

RATES = tuple(declaration["panel"]["poison_rates"])
DATASETS = tuple(declaration["panel"]["datasets"])
BUDGETS = (0.10, 0.20)
ATTACK_ORDER = ("badnet_a2o", "blend", "bpp", "lf", "wanet", "tact", "sig", "lc", "adaptive_blend")
OURS = ("PSBD-TM", "PSBD-RD", f"PSBD-TM {MATCHED_SUFFIX}", f"PSBD-RD {MATCHED_SUFFIX}")
DEFENSES = (*OURS, *DETECTOR_NAMES)
SHORT = {
    "PSBD-TM": "PSBD-TM",
    "PSBD-RD": "PSBD-RD",
    f"PSBD-TM {MATCHED_SUFFIX}": "TM matched",
    f"PSBD-RD {MATCHED_SUFFIX}": "RD matched",
    "confidence": "Conf",
    "strip": "STRIP",
    "scale_up": "Scale-Up",
    "scale_up_data_limited": "Scale-Up dl",
    "ibd_psc": "IBD-PSC",
    "ibd_psc_calibrated": "IBD-PSC cal",
    "teco": "TeCo",
    "cd_l": "CD-L",
    "beatrix": "Beatrix",
    "ted": "TED",
    "sentinet": "SentiNet",
}
HEADLINE_BAR = declaration["clean_accuracy_drop_bar_headline"]
SECOND_BAR = declaration["clean_accuracy_drop_bar"]
HEADLINE_NAME = f"successful at {-HEADLINE_BAR * 100:.0f} points"
SECOND_NAME = f"successful at {-SECOND_BAR * 100:.0f} points"

vit_models = models[(models.architecture == "vit") & models.in_panel].copy()
vit_meta = vit_models.set_index("folder_name")
vit_rows = rows[(rows.architecture == "vit") & (rows.kind == "attack") & (rows.status == "scored")]
POPULATIONS = {
    "implanted": vit_models.asr_class == "clears",
    HEADLINE_NAME: vit_models.successful_2pt.astype(bool),
    SECOND_NAME: vit_models.successful_5pt.astype(bool),
}
populations = {name: set(vit_models[mask].folder_name) for name, mask in POPULATIONS.items()}
under_audit = sorted(models[models.audit_note.notna() & (models.kind == "attack")].attack.unique())
""")

code(r"""
def say(text):
    display(Markdown(text))


def macro(name):
    value = paper_macros[name]["value"]
    return value


def at_rate(folders, rate):
    kept = {folder for folder in folders if vit_meta.poison_rate[folder] == rate}
    return kept


def exclusion_reason(model, bar_name):
    if model.asr_class == "below_bar":
        return f"ASR {model.asr:.3f} below the {declaration['asr_bar']} bar"
    if model.asr_class == "diverged":
        return f"diverged, clean accuracy {model.clean_accuracy:.3f} below half the benign {model.clean_accuracy_benign:.3f}"
    if model.asr_class == "source_mapped":
        return f"source-mapped, clean source-class accuracy {model.source_class_accuracy:.3f}"
    if model.asr_class == "unmeasured":
        return "no ASR recorded"
    reason = f"clean accuracy drop {model.clean_accuracy_drop:+.3f} outside the {bar_name} bar"
    return reason


def wide_table(folders, metric, defenses=DEFENSES):
    subset = vit_rows[vit_rows.folder_name.isin(folders) & vit_rows.defense.isin(defenses)]
    wide = subset.pivot(index="folder_name", columns="defense", values=metric).reindex(columns=list(defenses))
    return wide


def cell_means(folders, metric, group_field, group_values, defenses=DEFENSES):
    wide = wide_table(folders, metric, defenses)
    groups = vit_meta.loc[wide.index, group_field]
    means = wide.groupby(groups).mean().reindex([value for value in group_values if value in set(groups)])
    counts = groups.value_counts()
    return means, counts


def row_label(name, counts, labeller):
    flag = " (under audit)" if name in under_audit else ""
    label = f"{labeller(name)} ({counts[name]}){flag}"
    return label


def draw_heatmap(axis, means, counts, labeller, title, center=0.5, span=0.5):
    values = means.to_numpy(dtype=float)
    axis.imshow(values, cmap="RdBu", norm=TwoSlopeNorm(vcenter=center, vmin=center - span, vmax=center + span), aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isnan(value):
                continue
            color = "white" if abs(value - center) > 0.6 * span else "black"
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6, color=color)
    axis.set_xticks(np.arange(values.shape[1]), [SHORT.get(name, name) for name in means.columns], rotation=60, ha="right")
    axis.set_yticks(np.arange(values.shape[0]), [row_label(name, counts, labeller) for name in means.index])
    axis.set_title(title)
    axis.grid(False)


def paired_interval(folders, first, second, metric="auroc", order_by="table"):
    # The bootstrap draws positions in the list, so under a fixed seed the interval
    # depends on the model order. "folder" is the ledger order tab_headline.py uses
    # for the PSBD-TM minus PSBD-RD gain, "table" the dataset, attack and rate order
    # tab_detectors.py uses for the competitor margins.
    wide = wide_table(folders, metric, (first, second)).dropna()
    if order_by == "folder":
        order = sorted(wide.index)
    else:
        order = vit_meta.loc[wide.index].sort_values(["dataset", "attack", "poison_rate"]).index
    differences = (wide.loc[order, first] - wide.loc[order, second]).tolist()
    low, high = bootstrap_ci(differences, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    summary = (float(np.mean(differences)) if differences else np.nan, low, high, len(differences))
    return summary
""")

code(r"""
say(f'''
This notebook can be read alone, so the facts it relies on are restated here in 1 sentence each. `notebooks/how-every-number-is-computed.ipynb` walks the computation behind a single reading in full, and `docs/metrics.md` collects the formulas.

- **Models.** Every ViT-B/16 panel model of the paper datasets ({word_list(list(dataset_label(dataset) for dataset in DATASETS))}), meaning every checkpoint `configs/psbd_basis.json` selects: label modes {declaration["panel"]["label_modes"]} at poison rates {word_list(list(f"{rate:.0%}" for rate in RATES))}, 1 canonical variant per attack and no experiment tags. There are {len(vit_models)}. The Swin-S section uses the same rule on Swin checkpoints, plus the extra Swin models the paper's Swin table reads.
- **Defenses.** PSBD-TM (token masking at the attention input of every block, `{RECOMMENDED_PLACEMENT}`), the placement the paper recommends. PSBD-RD (dropout after both residual additions, `{PUBLISHED_PLACEMENT}`), our adaptation of the original ConvNet site. Both are read at the adaptive rule and again at the matched rule. The {len(DETECTOR_NAMES)} competitor detectors of `detectors.DETECTOR_NAMES`, scored by `cli.baselines` on the same images and quantiles. Every other cached PSBD placement appears in its own section.
- **Adaptive rule.** The smallest swept rate whose shift ratio (the share of predictions the perturbation changes) on the {PSBD_HELDOUT_SIZE} clean validation images reaches {ADAPTIVE_SHIFT_TARGET}. This is the deployable rule and every headline number uses it.
- **Matched rule.** The swept rate whose clean validation shift ratio sits nearest {PLACEMENT_MATCH_TARGET}, a device for comparing placements at the same measured disturbance.
- **Score.** PSBD scores an image by fractional PSU, 1 minus the mean over $k$ perturbed passes of the probability of the unperturbed prediction divided by its unperturbed probability. A low score means poisoned for PSBD and for every competitor.
- **AUROC.** One-sided area under the ROC curve of triggered test images against their own clean copies. It is never flipped, so a value below 0.5 means the defense ordered the 2 the wrong way round on that model.
- **TPR and FPR.** The threshold is the {BUDGETS[0]:.2f} or {BUDGETS[1]:.2f} quantile of the clean validation scores (the false-positive budget). TPR is the share of triggered images below it and the realized FPR the share of their paired clean copies below it.
- **Paired gain and interval.** A difference between 2 defenses is taken per model and then averaged, and its 95% interval is a percentile bootstrap over models, {BOOTSTRAP_RESAMPLES} resamples seeded at {BOOTSTRAP_SEED}, by `scripts/paper/_common.bootstrap_ci`.
- **Under audit.** Every row of attack {word_list(list(under_audit))} carries an `audit_note` in `all_numbers.json`, because its trigger amplitude is under audit and its caches may mix the trained and the current amplitude. Those rows are kept and marked "(under audit)" in every figure and table.

`all_numbers.json` was written at {payload["written_at"][:19]} UTC at commit `{payload["git_commit"][:10]}`, and holds {len(models)} models and {len(rows)} rows.
''')
""")

code(r"""
say(f'''
## The 3 populations

A detection number on a model whose attack never implanted measures nothing, so every mean is taken over a population of models whose backdoor is real. The coverage ledger (`scripts/coverage_ledger.py`, `results/coverage/coverage.json`) gives every panel model an `asr_class`. It is `clears` when the attack success rate (ASR, the share of triggered test images sent to the target class) is at least {declaration["asr_bar"]}, `below_bar` under it, `diverged` when the clean accuracy fell below half the benign reference (a collapsed run), and `source_mapped` when a TaCT model misclassifies its clean source class without any trigger (a class mapping, not a trigger backdoor). The user then set 2 success bars on clean accuracy, since a backdoor that costs visible clean accuracy is a backdoor a defender could catch by accuracy alone. $\\Delta CA$ is the clean accuracy of the model minus that of the benign model trained on the same dataset with the same recipe.

| population | rule | ledger field |
|---|---|---|
| implanted | `asr_class` is `clears` | `asr_class` |
| {HEADLINE_NAME} | implanted and $\\Delta CA \\ge$ {HEADLINE_BAR}, the headline success definition | `successful_2pt` |
| {SECOND_NAME} | implanted and $\\Delta CA \\ge$ {SECOND_BAR} | `successful_5pt` |

The paper's generators still select on `asr_class` alone, so the paper's headline panel is the implanted population restricted to the models carrying every defense. Every figure and table below is given for all 3 populations.

## Population sizes and the models each bar leaves out

The first table counts the ViT panel models per poison rate and population, and the next 3 list every panel model a population leaves out with the reason, read from the ledger fields. A model can pass a bar and still be missing from a mean when it has no reading, so the last line lists the implanted models with no PSBD-TM reading.
''')
""")

code(r"""
counts = pd.DataFrame(
    {name: [len(at_rate(folders, rate)) for rate in RATES] + [len(folders)] for name, folders in populations.items()},
    index=[f"{rate:.0%}" for rate in RATES] + ["all rates"],
)
counts.insert(0, "panel models", [int((vit_models.poison_rate == rate).sum()) for rate in RATES] + [len(vit_models)])
display(counts)

for name, bar_name in (("implanted", None), (HEADLINE_NAME, f"{HEADLINE_BAR}"), (SECOND_NAME, f"{SECOND_BAR}")):
    left_out = vit_models[~vit_models.folder_name.isin(populations[name])].sort_values(["poison_rate", "dataset", "attack"])
    reasons = left_out.apply(lambda model: exclusion_reason(model, bar_name), axis=1)
    say(f"**{name}**: {len(populations[name])} in, {len(left_out)} left out.")
    display(pd.DataFrame({"model": left_out.folder_name, "rate": left_out.poison_rate, "reason": reasons, "audit": left_out.audit_note}).set_index("model"))

carrying = set(vit_rows[vit_rows.defense == "PSBD-TM"].folder_name)
unswept = sorted(populations["implanted"] - carrying)
bar_failures = vit_models[(vit_models.asr_class == "clears") & ~vit_models.successful_2pt.astype(bool)]
covered = {name: len(folders & carrying) for name, folders in populations.items()}
below = vit_models[vit_models.asr_class == "below_bar"].attack.value_counts()
failure_words = word_list(list(f"`{row.folder_name}` ($\\Delta CA$ {row.clean_accuracy_drop:+.3f}, {'fails both bars' if not row.successful_5pt else 'fails only the headline bar'})" for row in bar_failures.itertuples()))
say(f'''
The 3 populations differ by {len(populations["implanted"]) - len(populations[HEADLINE_NAME])} models in all: {failure_words}. The models left out of every population are mostly below the ASR bar, by attack {word_list(list(f"{count} {attack_label(attack)}" for attack, count in below.items()))}. {len(unswept)} implanted models ({word_list(list(f"`{folder}`" for folder in unswept))}) carry no PSBD reading yet because their sweeps are queued on the GPU, so the means below span {word_list(list(f"{count} ({name})" for name, count in covered.items()))} models.
''')
""")

code(r"""
say(f'''
## AUROC per attack and defense, by poison rate

Each figure below is 1 population, and its 3 heatmaps are the {len(RATES)} poison rates. A row is an attack, labeled with the number of models behind it (1 per dataset where that attack implanted at that rate), and a column is a defense: PSBD-TM and PSBD-RD at the adaptive rule, the 2 again at the matched rule, then the {len(DETECTOR_NAMES)} competitors. A cell is the mean one-sided AUROC over those models, computed from the per-model readings in `all_numbers.json`, blue above 0.5 and red below, with white text at the extremes. Every implanted model with a PSBD reading carries every defense, so the models behind a row are the same in every column.

What to look at: the PSBD-TM column against the PSBD-RD column, and both against the strongest competitor columns. What these figures do not show is spread, since a cell over a few models can hide a failure on 1 of them, which the full tables at the end resolve. The rows follow a fixed attack order, so an attack missing from a panel did not implant at that rate on any dataset, or implanted only on a model not yet swept.
''')
""")

code(r"""
for name, folders in populations.items():
    figure, axes = plt.subplots(len(RATES), 1, figsize=(8.5, 3.2 * len(RATES)))
    for axis, rate in zip(axes, RATES):
        rate_folders = at_rate(folders, rate)
        means, counts_by_attack = cell_means(rate_folders, "auroc", "attack", ATTACK_ORDER)
        draw_heatmap(axis, means, counts_by_attack, attack_label, f"{name}, {rate:.0%} poisoning, {len(rate_folders & carrying)} models with readings")
    plt.tight_layout()
    plt.show()
""")

code(r"""
implanted_means = {rate: cell_means(at_rate(populations["implanted"], rate), "auroc", "attack", ATTACK_ORDER)[0] for rate in RATES}
low_rate, high_rate = RATES[0], RATES[-1]
low_means, high_means = implanted_means[low_rate], implanted_means[high_rate]
rd_fails = [attack for attack in low_means.index if low_means.loc[attack, "PSBD-RD"] < 0.65]
rd_leads = [attack for attack in high_means.index if high_means.loc[attack, "PSBD-RD"] > high_means.loc[attack, "PSBD-TM"] + 0.05]
sentinet_below = int((pd.concat(implanted_means.values())["sentinet"] < 0.5).sum())
total_rows = sum(len(frame) for frame in implanted_means.values())
matched_below = int((pd.concat(implanted_means.values())[f"PSBD-TM {MATCHED_SUFFIX}"] < pd.concat(implanted_means.values())["PSBD-TM"]).sum())
say(f'''
On the implanted population at {low_rate:.0%} poisoning PSBD-RD reads at most 0.65 on {word_list(list(f"{attack_label(attack)} ({low_means.loc[attack, 'PSBD-RD']:.2f}, against {low_means.loc[attack, 'PSBD-TM']:.2f} for PSBD-TM)" for attack in rd_fails))}, which is where the paper's gain comes from. At {high_rate:.0%} PSBD-RD leads PSBD-TM by more than 0.05 on {word_list(list(f"{attack_label(attack)} ({high_means.loc[attack, 'PSBD-RD']:.2f} against {high_means.loc[attack, 'PSBD-TM']:.2f})" for attack in rd_leads))}. The SIG row is marked under audit, and on its single model the PSBD cache and the detector records were also scored on different triggered images (the walkthrough notebook shows this), so on that row the PSBD columns and the competitor columns are not the same comparison. PSBD-TM at the matched rule sits below PSBD-TM at the adaptive rule on {matched_below} of {total_rows} attack rows, since the matched rate is smaller and disturbs the clean model less. SentiNet reads below 0.5 on {sentinet_below} of {total_rows} attack rows, which the paper attributes to its Grad-CAM mask missing the trigger. Across the 3 populations the cells change only where a removed model sat, the SIG row and the WaNet rows. The next figures break the same readings out by dataset.
''')
""")

code(r"""
say(f'''
## AUROC per dataset

The left heatmap of each figure is the mean AUROC per dataset and defense over all {len(RATES)} rates, and the right one is the paired gain of PSBD-TM over PSBD-RD per dataset and rate (the per-model difference averaged), red where PSBD-RD leads. The row labels carry the number of models. What to look at: whether PSBD-TM's lead is carried by 1 dataset, and whether a competitor leads on any dataset. A dataset mean pools attacks of different difficulty, so a dataset with more BadNets models looks easier, which the per-attack figures above correct for.
''')
""")

code(r"""
dataset_gains = {}
dataset_means = {}
for name, folders in populations.items():
    figure, (dataset_axis, gain_axis) = plt.subplots(1, 2, figsize=(10.5, 2.8), gridspec_kw={"width_ratios": [3.2, 1]})
    means, counts_by_dataset = cell_means(folders, "auroc", "dataset", DATASETS)
    dataset_means[name] = means
    draw_heatmap(dataset_axis, means, counts_by_dataset, dataset_label, f"mean AUROC, {name}, {len(folders & carrying)} models")
    gains = wide_table(folders, "auroc", ("PSBD-TM", "PSBD-RD")).dropna()
    gains = (gains["PSBD-TM"] - gains["PSBD-RD"]).rename("gain").to_frame().join(vit_meta[["dataset", "poison_rate"]])
    gain_means = gains.pivot_table(index="dataset", columns="poison_rate", values="gain", aggfunc="mean").reindex(DATASETS)
    dataset_gains[name] = gain_means
    gain_means = gain_means.rename(columns=lambda rate: f"{rate:.0%}")
    draw_heatmap(gain_axis, gain_means, gains.dataset.value_counts(), dataset_label, "PSBD-TM minus PSBD-RD", center=0.0, span=0.6)
    gain_axis.set_yticks([])
    plt.tight_layout()
    plt.show()

headline_gains = dataset_gains[HEADLINE_NAME]
headline_means = dataset_means[HEADLINE_NAME]
leaders = {dataset: headline_means.loc[dataset, list(DETECTOR_NAMES)].idxmax() for dataset in headline_means.index}
competitor_leads = [dataset for dataset in headline_means.index if headline_means.loc[dataset, leaders[dataset]] > headline_means.loc[dataset, "PSBD-TM"]]
say(f'''
On the headline population the gain of PSBD-TM over PSBD-RD per dataset at {low_rate:.0%} is {word_list([f"{headline_gains.loc[dataset, low_rate]:+.2f} on {dataset_label(dataset)}" for dataset in headline_gains.index])}. At {high_rate:.0%} it is {word_list([f"{headline_gains.loc[dataset, high_rate]:+.2f} on {dataset_label(dataset)}" for dataset in headline_gains.index])}. A competitor's mean AUROC exceeds PSBD-TM's on {word_list(list(f"{dataset_label(dataset)} ({SHORT[leaders[dataset]]} {headline_means.loc[dataset, leaders[dataset]]:.2f} against {headline_means.loc[dataset, 'PSBD-TM']:.2f})" for dataset in competitor_leads)) or "no dataset"}, so the paper's 1st rank is a mean over datasets and not a lead on each. The paper's own sentence on this reads the leaders off the implanted population: `DetectorsAurocDatasetsOthersLead` is "{macro("DetectorsAurocDatasetsOthersLead")}". The next section puts numbers and intervals on these comparisons.
''')
""")

code(r"""
say(f'''
## PSBD-TM, PSBD-RD and the best competitor

For each population and each poison rate the table gives these columns.

- `n`, the number of models.
- The mean AUROC of PSBD-TM, of PSBD-RD and of the competitor with the highest mean AUROC in that subset.
- `TM - RD`, the paired gain of PSBD-TM over PSBD-RD with its 95% bootstrap interval.
- `TM - competitor`, the paired margin of PSBD-TM over that best competitor with its interval.
- The mean TPR of the 3 at the {BUDGETS[0]:.2f} and {BUDGETS[1]:.2f} budgets.

The best competitor is chosen per subset, so it can change between rows. Choosing it after seeing the data makes the margin conservative for PSBD-TM. The bootstrap draws positions in a list of models, so under the fixed seed an interval depends on the order the models are listed in, at the third decimal. The gain interval is taken in folder name order, the order `scripts/paper/tab_headline.py` iterates in. The margin interval is taken in dataset, attack and rate order, the order `scripts/paper/tab_detectors.py` iterates in. Those 2 orders reproduce both of the paper's intervals exactly.

The scatter plots after the table place every model of the headline population by its PSBD-RD AUROC (left) or its IBD-PSC cal AUROC (right, the calibrated IBD-PSC port) on the horizontal axis and its PSBD-TM AUROC on the vertical axis, colored by poison rate. A point above the diagonal is a model where PSBD-TM wins.
''')
""")

code(r"""
def comparison_row(folders, label):
    wide = wide_table(folders, "auroc").dropna(subset=["PSBD-TM", "PSBD-RD", *DETECTOR_NAMES])
    kept = set(wide.index)
    competitors = wide[list(DETECTOR_NAMES)].mean()
    best = competitors.idxmax()
    gain, gain_low, gain_high, _ = paired_interval(kept, "PSBD-TM", "PSBD-RD", order_by="folder")
    margin, margin_low, margin_high, _ = paired_interval(kept, "PSBD-TM", best)
    budget_tables = {budget: wide_table(kept, f"tpr_q{budget:.2f}") for budget in BUDGETS}
    row = {
        "subset": label,
        "n": len(kept),
        "PSBD-TM": wide["PSBD-TM"].mean(),
        "PSBD-RD": wide["PSBD-RD"].mean(),
        "best competitor": SHORT[best],
        "competitor AUROC": competitors[best],
        "TM - RD": f"{gain:+.3f} [{gain_low:+.3f}, {gain_high:+.3f}]",
        "TM - competitor": f"{margin:+.3f} [{margin_low:+.3f}, {margin_high:+.3f}]",
    }
    for budget, frame in budget_tables.items():
        row[f"TPR@{budget:.2f} TM"] = frame["PSBD-TM"].mean()
        row[f"TPR@{budget:.2f} RD"] = frame["PSBD-RD"].mean()
        row[f"TPR@{budget:.2f} competitor"] = frame[best].mean()
    return row


comparison = []
for name, folders in populations.items():
    comparison.append(comparison_row(folders, f"{name}, all rates"))
    for rate in RATES:
        comparison.append(comparison_row(at_rate(folders, rate), f"{name}, {rate:.0%}"))
comparison = pd.DataFrame(comparison).set_index("subset")
display(comparison)
""")

code(r"""
headline = populations[HEADLINE_NAME]
wide = wide_table(headline, "auroc").dropna(subset=["PSBD-TM"])
colors = dict(zip(RATES, ("#D55E00", "#E69F00", "#0072B2")))
figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)
for axis, other in zip(axes, ("PSBD-RD", "ibd_psc_calibrated")):
    for rate, color in colors.items():
        rate_folders = [folder for folder in wide.index if vit_meta.poison_rate[folder] == rate]
        axis.scatter(wide.loc[rate_folders, other], wide.loc[rate_folders, "PSBD-TM"], s=14, color=color, label=f"{rate:.0%} poisoning")
    axis.plot([0, 1], [0, 1], color="black", linewidth=0.8)
    axis.set_xlabel(f"{SHORT[other]} AUROC")
    axis.set_title(f"PSBD-TM against {SHORT[other]}, {len(wide)} models")
axes[0].set_ylabel("PSBD-TM AUROC")
axes[0].legend()
plt.tight_layout()
plt.show()

def gain_mean(label):
    mean = float(comparison.loc[label, "TM - RD"].split(" ")[0])
    return mean


gain_series = {name: [gain_mean(f"{name}, {rate:.0%}") for rate in RATES] for name in populations}
shrinking = all(values == sorted(values, reverse=True) for values in gain_series.values())
gain_by_rate = "; ".join(f"{name}: " + word_list(list(f"{value:+.3f}" for value in values)) for name, values in gain_series.items()).replace("; ", ", then ")
implanted_row = comparison.loc["implanted, all rates"]
headline_row = comparison.loc[f"{HEADLINE_NAME}, all rates"]
say(f'''
The implanted row reproduces the paper's headline: PSBD-TM {implanted_row["PSBD-TM"]:.3f} (macro `HeadlineAurocAdaptive` {macro("HeadlineAurocAdaptive")}), PSBD-RD {implanted_row["PSBD-RD"]:.3f} (`PublishedAurocAdaptive` {macro("PublishedAurocAdaptive")}), a gain of {implanted_row["TM - RD"]} (`HeadlineGainAdaptiveAuroc` {macro("HeadlineGainAdaptiveAuroc")} [{macro("HeadlineGainAdaptiveAurocLow")}, {macro("HeadlineGainAdaptiveAurocHigh")}]) and a margin of {implanted_row["TM - competitor"]} over {implanted_row["best competitor"]} (`DetectorsAurocMargin` {macro("DetectorsAurocMargin")} [{macro("DetectorsAurocMarginLow")}, {macro("DetectorsAurocMarginHigh")}]), on {implanted_row["n"]} models. On the headline population, {headline_row["n"]} models, PSBD-TM reads {headline_row["PSBD-TM"]:.3f} and PSBD-RD {headline_row["PSBD-RD"]:.3f}, a gain of {headline_row["TM - RD"]} and a margin of {headline_row["TM - competitor"]} over {headline_row["best competitor"]}. These are the numbers under the stricter success definition. The gain of PSBD-TM over PSBD-RD {"falls" if shrinking else "does not fall"} from the lowest to the highest poison rate in every population ({gain_by_rate}). The scatter shows the same per model: most points sit on the diagonal against both PSBD-RD and IBD-PSC cal, and the mean is carried by a few models far above it. The next section asks how every other PSBD placement compares.
''')
""")

code(r"""
say(f'''
## Every PSBD placement at the adaptive rule

`psbd_metrics.json` holds every placement ever swept on a model: the {len(declaration["basis"])} placements declared in `configs/psbd_basis.json` plus variants (other mask seeds with `_seed`, other pass counts with `_k`, the model's own dropout with `_pmodel`). The chart gives, for the headline population, the mean AUROC of every placement at the adaptive rule over the models where that placement has a reading, with the number of models in the label. The table beside it adds the paired gain over PSBD-RD computed on the models carrying both. Coverage differs between placements, so 2 bars with different counts are means over different models. Only the paired gain column compares like with like. A placement whose ladder never reaches a shift ratio of {ADAPTIVE_SHIFT_TARGET} on a model has no adaptive reading there. The `unreached` column counts those models, because leaving them out silently would flatter a weak placement. What this does not show is any placement at its best rate, since the adaptive rule is the only rule used.
''')
""")

code(r"""
psbd_rows = rows[(rows.architecture == "vit") & rows.folder_name.isin(headline) & (rows.family == "psbd") & (rows.rule == "adaptive")]
scored = psbd_rows[psbd_rows.status == "scored"]
placements = scored.groupby("placement").auroc.agg(["mean", "count"])
placements["unreached"] = psbd_rows[psbd_rows.status == "shift_target_unreached"].groupby("placement").size().reindex(placements.index).fillna(0).astype(int)
rd = scored[scored.placement == PUBLISHED_PLACEMENT].set_index("folder_name").auroc
gain_rows = []
for placement, group in scored.groupby("placement"):
    paired = group.set_index("folder_name").auroc.to_frame("placement").join(rd.rename("rd"), how="inner").sort_index()
    differences = (paired["placement"] - paired["rd"]).tolist()
    low, high = bootstrap_ci(differences, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    gain_rows.append({"placement": placement, "paired n": len(differences), "gain over PSBD-RD": float(np.mean(differences)), "low": low, "high": high})
placements = placements.join(pd.DataFrame(gain_rows).set_index("placement")).sort_values("mean", ascending=False)

figure, axis = plt.subplots(figsize=(7.5, 0.22 * len(placements) + 0.8))
colors = ["#D55E00" if name == RECOMMENDED_PLACEMENT else "#0072B2" if name == PUBLISHED_PLACEMENT else "#999999" for name in placements.index]
axis.barh(np.arange(len(placements)), placements["mean"], color=colors)
axis.set_yticks(np.arange(len(placements)), [f"{name} ({count})" for name, count in zip(placements.index, placements["count"])], fontsize=6)
axis.invert_yaxis()
axis.axvline(0.5, color="black", linewidth=0.8, linestyle=":")
axis.set_xlim(max(0.0, float(placements["mean"].min()) - 0.05), 1.0)
axis.set_xlabel(f"mean AUROC at the adaptive rule, {HEADLINE_NAME}")
plt.show()
display(placements.round(3))

full_coverage = placements["count"].max()
full = placements[placements["count"] == full_coverage].sort_values("mean", ascending=False)
above = placements[(placements["mean"] > placements.loc[RECOMMENDED_PLACEMENT, "mean"])]
say(f'''
Among the {len(full)} placements read on all {full_coverage} models of the population, the highest means are {word_list([f"`{name}` {value:.3f}" for name, value in full["mean"].head(3).items()])}. PSBD-RD (`{PUBLISHED_PLACEMENT}`) reads {placements.loc[PUBLISHED_PLACEMENT, "mean"]:.3f}, rank {list(full.index).index(PUBLISHED_PLACEMENT) + 1} of {len(full)}. The {len(above)} bars above PSBD-TM overall are all read on fewer models ({word_list(list(sorted({str(count) for count in above["count"]}, key=int)))}), mostly more-pass variants, so they are means over a different and smaller set of models. The paired gain column is the comparison to read. The next section repeats the headline comparison on Swin-S.
''')
""")

code(r"""
swin_models = models[models.architecture == "swin"].copy()
swin_meta = swin_models.set_index("folder_name")
swin_rows = rows[(rows.architecture == "swin") & (rows.kind == "attack") & (rows.status == "scored")]
swin_panel = swin_models[swin_models.in_panel]
extras = swin_models[(~swin_models.in_panel) & (swin_models.kind == "attack")]
swin_populations = {
    "panel rule, implanted": set(swin_panel[swin_panel.asr_class == "clears"].folder_name),
    f"panel rule, {HEADLINE_NAME}": set(swin_panel[swin_panel.successful_2pt.astype(bool)].folder_name),
    f"panel rule, {SECOND_NAME}": set(swin_panel[swin_panel.successful_5pt.astype(bool)].folder_name),
    "paper selection": set(swin_panel[swin_panel.asr_class == "clears"].folder_name) | set(extras[extras.asr_class == "clears"].folder_name),
}
say(f'''
## Swin-S

Swin-S is the second architecture, used to test whether the placement ranking transfers. No detector records exist for Swin, so only PSBD placements are compared. 2 Swin selections are shown because the paper's Swin table and the ViT panel rule select different models.

- **Panel rule**: the ViT declaration applied to Swin checkpoints with Swin's own benign references, built in memory by `scripts/all_numbers.py` through the ledger's `build_ledger`, with the same `asr_class` and success verdicts. It holds {len(swin_panel)} models, {word_list(list(f"{count} `{name}`" for name, count in swin_panel.asr_class.value_counts().items()))}.
- **Paper selection**: what `scripts/paper/tab_swin.py` reads, every Swin folder with no excluded folder token whose ASR clears {declaration["asr_bar"]} and that is not source-mapped. It adds {len(extras)} models the panel rule leaves out, listed below with the reason. It has no divergence or clean-accuracy check of its own. `scripts/all_numbers.py` judges them by the ledger's rules anyway.

The paper's Swin macros are `SwinRecommendedAurocAdaptive` {macro("SwinRecommendedAurocAdaptive")} over `SwinRecommendedN` {macro("SwinRecommendedN")} models and `SwinPublishedAurocAdaptive` {macro("SwinPublishedAurocAdaptive")} over `SwinCells` {macro("SwinCells")}, both on the paper selection.
''')


def off_panel_reason(model):
    reasons = []
    if model.poison_rate not in RATES:
        reasons.append(f"a poison rate of {model.poison_rate:.1%}, outside the panel rates")
    required = declaration["panel"]["canonical_variants"].get(model.attack)
    if required is not None and required not in model.folder_name:
        reasons.append(f"no `{required}` suffix, so not the canonical {attack_label(model.attack)} variant")
    canonical_target = declaration["panel"]["canonical_targets"].get(model.dataset, {}).get(model.label_mode)
    if canonical_target is not None and model.target_label != canonical_target:
        reasons.append(f"target class {int(model.target_label)} instead of the canonical {canonical_target}")
    reason = " and ".join(reasons)
    return reason


display(pd.DataFrame({"model": extras.folder_name, "ASR": extras.asr, "clean accuracy drop": extras.clean_accuracy_drop, "asr_class": extras.asr_class, "why the panel rule leaves it out": extras.apply(off_panel_reason, axis=1), "audit": extras.audit_note}).set_index("model"))

summary = []
for name, folders in swin_populations.items():
    wide = swin_rows[swin_rows.folder_name.isin(folders) & swin_rows.defense.isin(["PSBD-TM", "PSBD-RD"])].pivot(index="folder_name", columns="defense", values="auroc")
    paired = wide.dropna().sort_index()
    differences = (paired["PSBD-TM"] - paired["PSBD-RD"]).tolist()
    low, high = bootstrap_ci(differences, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    tpr = swin_rows[swin_rows.folder_name.isin(folders) & swin_rows.defense.isin(["PSBD-TM", "PSBD-RD"])].pivot(index="folder_name", columns="defense", values=f"tpr_q{BUDGETS[0]:.2f}")
    summary.append({"selection": name, "models": len(folders), "PSBD-TM n": int(wide["PSBD-TM"].notna().sum()), "PSBD-TM": wide["PSBD-TM"].mean(), "PSBD-RD n": int(wide["PSBD-RD"].notna().sum()), "PSBD-RD": wide["PSBD-RD"].mean(), "paired n": len(differences), "TM - RD": f"{np.mean(differences):+.3f} [{low:+.3f}, {high:+.3f}]", f"TPR@{BUDGETS[0]:.2f} TM": tpr["PSBD-TM"].mean(), f"TPR@{BUDGETS[0]:.2f} RD": tpr["PSBD-RD"].mean()})
swin_summary = pd.DataFrame(summary).set_index("selection")
display(swin_summary)
""")

code(r"""
figure, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for axis, name in zip(axes, (f"panel rule, {HEADLINE_NAME}", "paper selection")):
    folders = swin_populations[name]
    subset = swin_rows[swin_rows.folder_name.isin(folders) & swin_rows.defense.isin(["PSBD-TM", "PSBD-RD"])]
    subset = subset.assign(poison_rate=[swin_meta.poison_rate[folder] for folder in subset.folder_name])
    means = subset.pivot_table(index="attack", columns=["defense", "poison_rate"], values="auroc", aggfunc="mean")
    means = means.reindex([attack for attack in ATTACK_ORDER if attack in means.index])
    means = means[sorted(means.columns, key=lambda column: (column[0] != "PSBD-TM", column[1]))]
    counts = subset.groupby("attack").folder_name.nunique()
    means.columns = [f"{'TM' if defense == 'PSBD-TM' else 'RD'} {rate:.1%}" for defense, rate in means.columns]
    draw_heatmap(axis, means, counts, attack_label, f"Swin-S, {name}")
plt.tight_layout()
plt.show()

paper_row = swin_summary.loc["paper selection"]
panel_row = swin_summary.loc[f"panel rule, {HEADLINE_NAME}"]
tm_missing = sorted(set(swin_rows[swin_rows.folder_name.isin(swin_populations["paper selection"]) & (swin_rows.defense == "PSBD-RD")].folder_name) - set(swin_rows[swin_rows.defense == "PSBD-TM"].folder_name))
extra_reasons = extras.apply(off_panel_reason, axis=1).value_counts()
say(f'''
The summary table reproduces the paper's Swin numbers on the paper selection, PSBD-TM {paper_row["PSBD-TM"]:.3f} over {paper_row["PSBD-TM n"]} models and PSBD-RD {paper_row["PSBD-RD"]:.3f} over {paper_row["PSBD-RD n"]}, and gives the panel rule numbers beside them: on the Swin headline population PSBD-TM reads {panel_row["PSBD-TM"]:.3f}, PSBD-RD {panel_row["PSBD-RD"]:.3f} and the paired gain {panel_row["TM - RD"]}, against {paper_row["TM - RD"]} on the paper selection. The extras the paper reads outside the panel rule break down as {word_list(list(f"{count} with {reason}" for reason, count in extra_reasons.items()))}. A clean-label GTSRB model at target class 0 is capped at that class's share of the training set, so its runs at different requested rates train the identical poisoned set (`docs/clean-label-rate-caps.md`).

The heatmaps give the mean AUROC per attack for PSBD-TM and PSBD-RD at each poison rate, with the number of models per attack in the row label, for the Swin headline population and the paper selection. The blank TM cells matter: PSBD-TM has no reading on {len(tm_missing)} Swin models of the paper selection ({word_list(list(f"`{folder}`" for folder in tm_missing))}), which is why the paper reads PSBD-TM and PSBD-RD over different counts, and the paired gain is taken over the models carrying both. What the figures do not show is a competitor comparison, which no Swin detector run exists for.
''')
""")

code(r"""
say('''
## Full tables

The tables below hold every number the figures summarize, 1 row per model. They are sorted by poison rate, then dataset, then attack. Every model carries its ledger fields (`asr`, `clean_accuracy`, `clean_accuracy_drop`, `asr_class`, `successful_2pt`, `successful_5pt`, `audit_note`), so any population can be read off by eye. A blank cell is a reading that does not exist: a detector that was never run on a model below the bar, or a placement whose ladder never reached the shift target.

### AUROC of every defense on every ViT panel model
''')

LEDGER_COLUMNS = ["dataset", "attack", "poison_rate", "asr", "clean_accuracy", "clean_accuracy_drop", "asr_class", "successful_2pt", "successful_5pt", "audit_note"]


def full_table(metric, rate):
    rate_models = vit_models[vit_models.poison_rate == rate].sort_values(["dataset", "attack"])
    wide = wide_table(set(rate_models.folder_name), metric).rename(columns=SHORT)
    joined = rate_models.set_index("folder_name")[LEDGER_COLUMNS].join(wide)
    return joined


for rate in RATES:
    say(f"AUROC at {rate:.0%} poisoning")
    display(full_table("auroc", rate))
""")

code(r"""
for budget in BUDGETS:
    say(f"### TPR and realized FPR at the {budget:.2f} budget")
    for rate in RATES:
        say(f"TPR at the {budget:.2f} budget, {rate:.0%} poisoning")
        display(full_table(f"tpr_q{budget:.2f}", rate))
        say(f"realized FPR at the {budget:.2f} budget, {rate:.0%} poisoning")
        display(full_table(f"fpr_q{budget:.2f}", rate).drop(columns=LEDGER_COLUMNS[3:]))
""")

code(r"""
say('''
### The rate each rule chose for PSBD-TM and PSBD-RD

`rate` is the swept rate the adaptive or matched rule picked from the clean validation shift ratio, and `validation_shift_ratio` is the shift ratio it reached there.
''')
chosen = vit_rows[vit_rows.defense.isin(OURS)].pivot(index="folder_name", columns="defense", values=["rate", "validation_shift_ratio"])
chosen.columns = [f"{defense} {field}" for field, defense in chosen.columns]
display(vit_meta[["dataset", "attack", "poison_rate", "asr_class"]].join(chosen).sort_values(["poison_rate", "dataset", "attack"]))
""")

code(r"""
say('''
### Every PSBD placement at the adaptive rule, AUROC per model

Columns are the cache directory names of every placement swept on at least 1 ViT panel model.
''')
placement_rows = rows[(rows.architecture == "vit") & (rows.kind == "attack") & (rows.family == "psbd") & (rows.rule == "adaptive")]
every_placement = placement_rows.pivot(index="folder_name", columns="placement", values="auroc")
display(vit_meta[["dataset", "attack", "poison_rate", "asr_class", "successful_2pt"]].join(every_placement).sort_values(["poison_rate", "dataset", "attack"]))
""")

code(r"""
say('''
### Swin-S, every model

Every Swin model the panel rule or the paper selection reads, with PSBD-TM and PSBD-RD at both rules. `in_panel` is false for the models only the paper selection reads.
''')
swin_wide = swin_rows[swin_rows.defense.isin(OURS)].pivot(index="folder_name", columns="defense", values=["auroc", f"tpr_q{BUDGETS[0]:.2f}", f"tpr_q{BUDGETS[1]:.2f}"])
swin_wide.columns = [f"{defense} {field}" for field, defense in swin_wide.columns]
display(swin_models[swin_models.kind == "attack"].set_index("folder_name")[["in_panel", "dataset", "attack", "poison_rate", "asr", "clean_accuracy_drop", "asr_class", "successful_2pt", "successful_5pt", "audit_note"]].join(swin_wide).sort_values(["poison_rate", "dataset", "attack"]))
""")

code(r"""
benign_rows = rows[(rows.kind == "benign") & (rows.status == "scored") & rows.defense.isin(DEFENSES)]
benign_table = benign_rows.pivot(index="folder_name", columns="defense", values="auroc").reindex(columns=list(DEFENSES)).rename(columns=SHORT)
say(f'''
### Benign references

A benign model has no backdoor, so its triggered split is clean images stamped with a trigger the model never learned (`data.splits.BENIGN_PROBE_ATTACK` = `{BENIGN_PROBE_ATTACK}` by default). An AUROC far from 0.5 there means the defense reacts to the stamp itself, which would be a false alarm in deployment. The table gives the AUROC of every defense on the {len(benign_table)} benign references that carry a sweep.
''')
display(benign_table)
""")

nb = new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
nbformat.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")
