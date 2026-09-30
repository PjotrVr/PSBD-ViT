from nbbuild import SETUP, code, md, said, write

cells = [
    md(r"""
    # Attention-side probes

    Attention is the only operation in a ViT block that moves information between tokens, so it is where a token-level perturbation should matter most, and `mechanism.ipynb` showed that the class token reads a patch trigger through attention in the late blocks. A reader who accepts that should ask whether PSBD-TM is the best way to disturb attention, or whether a probe inside the attention sublayer (on the heads, on the normalized input, on the branch output) does better. This notebook reads every attention-side probe the sweeps cover, the placements the pre-registered basis declares and the ones it does not, and asks whether any of them beats PSBD-TM. It follows `scripts/paper/tab_attention.py`: full-depth probes only, the adaptive rate rule, the headline quantile on the fractional PSU and a floor on the models behind a mean (`MIN_CELLS`). It reads cached JSON only and runs in about 1 minute.
    """),
    md(r"""
    ## The attention sublayer's positions

    `tab_attention.ATTENTION_POSITIONS` lists every named tensor boundary of the attention sublayer, from its input to the stream just after its residual add: the input before $\mathrm{LN}_1$ (PSBD-TM's position), the normalized input, the per-head outputs inside the attention, the branch output before the add (PSBD-TM's twin when token-masked) and the stream after the add. The diagram marks them among the other positions. Beyond the basis operators, the sweeps added head masking (zero whole attention heads), drop path (zero the whole branch output of a sample) and noise and masking after the norm.
    """),
    code(
        SETUP
        + """
from cli.compare_detectors import psbd_values
from defenses.decision import HEADLINE_QUANTILE, RECOMMENDED_PLACEMENT
from scripts.paper._common import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    HEADLINE_KEY,
    OKABE_ITO,
    attack_label,
    bootstrap_ci,
    clearing_cells,
    fmt,
    load_coverage,
    load_declaration,
    load_psbd_metrics,
    mean_or_none,
    placement_words,
)
from scripts.paper._style import TEXT_WIDTH, attack_color, legend_above
from scripts.paper.tab_attention import ATTENTION_POSITIONS, MIN_CELLS, RULE, TPR_KEYS, readings

show_diagram("vit_block_positions")
print("attention-side positions read by tab_attention:", ATTENTION_POSITIONS)
"""
    ),
    md(r"""
    ## Probe means

    The generator's own `readings` collects every successful model's AUROC and its TPR at the 10% and 20% quantiles for each attention probe. Coverage is uneven, since the probes outside the basis were swept on a subset of the panel, so the means in this table are not paired.
    """),
    code(r"""
declaration = load_declaration("configs/psbd_basis.json")
declared = {entry["id"] for entry in declaration["basis"]}
folders = [cell["folder_name"] for cell in clearing_cells(load_coverage("results"))]
found = readings("results", folders)

probe_rows = []
for placement, values in found.items():
    if len(values["auroc"]) < MIN_CELLS:
        continue
    probe_rows.append(
        {
            "placement": placement,
            "probe": placement_words(placement),
            "in basis": placement in declared,
            "n": len(values["auroc"]),
            "AUROC": mean_or_none(values["auroc"]),
            "TPR@10": mean_or_none(values[TPR_KEYS[0]]),
            "TPR@20": mean_or_none(values[TPR_KEYS[1]]),
        }
    )
probes = pd.DataFrame(probe_rows).sort_values("AUROC", ascending=False).reset_index(drop=True)

assert str(len(probes)) == macro("attention_probes", "AttentionProbesTotal")
assert str(int(probes["in basis"].sum())) == macro("attention_probes", "AttentionProbesInBasis")
outside_best = probes[~probes["in basis"]].iloc[0]
assert fmt(outside_best["AUROC"]) == macro("attention_probes", "AttentionProbesBestOutside")
print(f"{len(folders)} successful models, rule {RULE}, threshold at the {HEADLINE_QUANTILE:.0%} quantile")
print(f"{len(probes)} attention probes with at least {MIN_CELLS} models, {probes['in basis'].sum()} in the basis")
probes.drop(columns="placement").round(3)
"""),
    md(r"""
    ## Paired gaps against PSBD-TM

    An unpaired mean over a subset of the panel can differ from a mean over the whole panel only because the subset holds easier models. The paired gap reads each probe against PSBD-TM on exactly the models carrying both, with a 95% bootstrap interval over those models (`BOOTSTRAP_RESAMPLES` resamples at `BOOTSTRAP_SEED`).
    """),
    code(r"""
reports = {folder: load_psbd_metrics("results", folder) or {} for folder in folders}


def headline_auroc(report, placement):
    block = psbd_values(report, placement, RULE)
    if block is None or block.get(HEADLINE_KEY) is None:
        return None
    return block[HEADLINE_KEY].get("auroc")


gap_rows = []
for placement in probes["placement"]:
    deltas = []
    for report in reports.values():
        probe_auroc = headline_auroc(report, placement)
        recommended_auroc = headline_auroc(report, RECOMMENDED_PLACEMENT)
        if probe_auroc is not None and recommended_auroc is not None:
            deltas.append(probe_auroc - recommended_auroc)
    low, high = bootstrap_ci(deltas, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    gap_rows.append({"placement": placement, "paired n": len(deltas), "gap": mean_or_none(deltas), "low": low, "high": high})

probes = probes.merge(pd.DataFrame(gap_rows), on="placement")
outside = probes[~probes["in basis"]]
beating = outside[outside["low"] > 0]
print(f"probes outside the basis whose paired gap interval sits above 0: {len(beating)}")
print(f"largest paired gap outside the basis: {outside['gap'].max():+.3f}")
by_id = probes.set_index("placement")
twin_id = "before_attention_residual_token_mask"
others = probes[probes["placement"] != RECOMMENDED_PLACEMENT]
# The reading sorts the other probes into those whose interval sits below 0 and those
# that tie PSBD-TM, and states that none sits above it.
assert (others["low"] <= 0).all(), "the reading says no probe beats PSBD-TM"
tied = others[others["high"] >= 0]
assert twin_id in set(tied["placement"]), "the reading names the twin among the ties"
tied_text = word_list([f"'{r.probe}' ({r.gap:+.3f} [{r.low:+.3f}, {r.high:+.3f}], {r.n} models)" for r in tied.itertuples()])
best_outside = outside.sort_values("gap", ascending=False).head(2)
weakest = probes.sort_values("AUROC").iloc[0]
probes[["probe", "in basis", "paired n", "gap", "low", "high"]].round(3)
"""),
    code(r"""
figure, (mean_axis, gap_axis) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.26 * len(probes) + 1.0), sharey=True)
positions = np.arange(len(probes))  # (probes,)
bar_colors = [OKABE_ITO[0] if in_basis else OKABE_ITO[1] for in_basis in probes["in basis"]]

mean_axis.barh(positions, probes["AUROC"], color=bar_colors)
mean_axis.set_yticks(positions, labels=[f"{probe} ({n})" for probe, n in zip(probes["probe"], probes["n"])])
mean_axis.invert_yaxis()
mean_axis.set_xlim(0.5, 1.0)
mean_axis.set_xlabel("mean AUROC, unpaired")
mean_axis.bar(0, 0, color=OKABE_ITO[0], label="in the basis")
mean_axis.bar(0, 0, color=OKABE_ITO[1], label="outside the basis")
mean_axis.legend()

# The PSBD-TM row is the reference itself, so its gap is 0 by construction and is
# left out of the right panel.
is_reference = (probes["placement"] == RECOMMENDED_PLACEMENT).to_numpy()  # (probes,)
gap_errors = np.vstack([probes["gap"] - probes["low"], probes["high"] - probes["gap"]])  # (2, probes)
gap_axis.errorbar(probes["gap"][~is_reference], positions[~is_reference], xerr=gap_errors[:, ~is_reference], fmt="none", ecolor="gray", linewidth=0.8)
gap_axis.scatter(probes["gap"][~is_reference], positions[~is_reference], c=np.array(bar_colors)[~is_reference], s=14, zorder=3)
gap_axis.axvline(0.0, color="black", linewidth=0.8)
gap_axis.set_xlabel("paired AUROC gap to PSBD-TM, 95% interval")
plt.show()
"""),
    said(r"""
    The left panel is the table the paper prints and the right panel is the comparison that decides it. PSBD-TM ({by_id.loc[RECOMMENDED_PLACEMENT, "AUROC"]:.3f}) and its twin, token masking on the attention branch output before the add ({by_id.loc[twin_id, "AUROC"]:.3f}), sit at the top. Read in pairs, {len(tied)} probes tie PSBD-TM with an interval across 0, {tied_text}. Every other probe has its whole interval below 0. No attention probe beats the pre-registered winner (`\AttentionProbesBeatingRecommended` is {macro("attention_probes", "AttentionProbesBeatingRecommended")}). The strongest probes the basis never declared are {word_list([f"'{r.probe}' ({r.n} models, gap {r.gap:+.3f})" for r in best_outside.itertuples()])}. The weakest of all is '{weakest["probe"]}' ({weakest["AUROC"]:.3f}). The figure does not say whether the ranking holds on every attack, which the next figure checks for the strongest probes.
    """),
    md(r"""
    ## The strongest probes per attack

    A probe that trails PSBD-TM on average could still lead on the attacks PSBD-TM handles worst. The heatmap reads the highest-ranked probes carried by at least 30 models, averaged per attack over the successful models carrying each probe.
    """),
    code(r"""
attack_of = {cell["folder_name"]: cell["attack"] for cell in clearing_cells(load_coverage("results"))}
top = probes[probes["n"] >= 30].head(6)
per_attack_rows = []
for folder, report in reports.items():
    for placement, probe in zip(top["placement"], top["probe"]):
        auroc = headline_auroc(report, placement)
        if auroc is not None:
            per_attack_rows.append({"attack": attack_label(attack_of[folder]), "probe": probe, "auroc": auroc})
per_attack = pd.DataFrame(per_attack_rows).pivot_table(index="attack", columns="probe", values="auroc", aggfunc="mean")[list(top["probe"])]
counts = pd.DataFrame(per_attack_rows).pivot_table(index="attack", columns="probe", values="auroc", aggfunc="size")[list(top["probe"])]

figure, axis = plt.subplots(figsize=(8.0, 3.8))
image = axis.imshow(per_attack.to_numpy(), cmap="RdBu", vmin=0.0, vmax=1.0, aspect="auto")
for r in range(per_attack.shape[0]):
    for c in range(per_attack.shape[1]):
        value = per_attack.iat[r, c]
        if not pd.isna(value):
            axis.text(c, r, f"{value:.2f}\n({int(counts.iat[r, c])})", ha="center", va="center", fontsize=6)
axis.set_xticks(range(per_attack.shape[1]), labels=per_attack.columns, rotation=25, ha="right", fontsize=7)
axis.set_yticks(range(per_attack.shape[0]), labels=per_attack.index)
axis.grid(False)
figure.colorbar(image, ax=axis, label="mean AUROC, adaptive rule (models)")
plt.show()
missing_cells = {probe: sorted(per_attack.index[per_attack[probe].isna()]) for probe in per_attack.columns if per_attack[probe].isna().any()}
tm_col, twin_col = top["probe"].iloc[0], placement_words(twin_id)
"""),
    said(r"""
    The small numbers are the models behind each cell. Among the columns, {word_list([f"'{p}' has no reading on {word_list(a)}" for p, a in missing_cells.items()])}, so those columns have gaps. On Blend and LF every probe reads at least {per_attack.loc[["Blend", "LF"]].min().min():.2f}. On BadNets the 2 token masks on the attention branch read {per_attack.loc["BadNets", tm_col]:.2f} and {per_attack.loc["BadNets", twin_col]:.2f}, and the other probes between {per_attack.loc["BadNets"].drop([tm_col, twin_col]).min():.2f} and {per_attack.loc["BadNets"].drop([tm_col, twin_col]).max():.2f}. The patching record of `mechanism.ipynb` says the BadNets decision sits in the trigger's own tokens until the late blocks, which fits a whole-token mask on the branch doing best, but no experiment here isolates why masking after the norm or a single head does worse. The twin reads {per_attack.loc["WaNet", twin_col]:.2f} on WaNet against {per_attack.loc["WaNet", tm_col]:.2f}. The heatmap does not pair the columns, so a column with fewer models is not directly comparable to its neighbor, and the paired gaps above remain the deciding comparison.
    """),
    md(r"""
    ## PSBD-TM against its twin

    The twin, token masking on the attention branch output before the add, is the 1 probe the paired test cannot separate from PSBD-TM on ViT. Both remove whole tokens from the attention branch only, 1 at its input and 1 at its output and neither touches the skip. The scatter pairs them model by model.
    """),
    code(r"""
twin = "before_attention_residual_token_mask"
pairs = []
for folder, report in reports.items():
    tm, tw = headline_auroc(report, RECOMMENDED_PLACEMENT), headline_auroc(report, twin)
    if tm is not None and tw is not None:
        pairs.append({"folder": folder, "attack": attack_of[folder], "PSBD-TM": tm, "twin": tw})
pairs = pd.DataFrame(pairs)
figure, axis = plt.subplots(figsize=(5.0, 4.8))
for attack, rows in pairs.groupby("attack"):
    axis.scatter(rows["PSBD-TM"], rows["twin"], s=22, color=attack_color(attack), label=f"{attack_label(attack)} ({len(rows)})")
axis.plot([0, 1], [0, 1], color="black", lw=0.8)
axis.set_xlim(0.3, 1.02)
axis.set_ylim(0.3, 1.02)
axis.set_xlabel("PSBD-TM AUROC (token mask, attention input)")
axis.set_ylabel("twin AUROC (token mask, attention output before the add)")
axis.legend()
plt.show()
correlation = np.corrcoef(pairs["PSBD-TM"], pairs["twin"])[0, 1]
print(f"{len(pairs)} models carry both, mean gap twin minus PSBD-TM {(pairs['twin'] - pairs['PSBD-TM']).mean():+.3f}, correlation {correlation:.2f}")
pair_gap = pairs["twin"] - pairs["PSBD-TM"]  # (models,)
twin_ahead = pairs[pair_gap > 0.05]
twin_behind = pairs[pair_gap < -0.05]
"""),
    said(r"""
    Most models sit in the top right corner near the diagonal, and the correlation across models is {correlation:.2f}, low because nearly all of them are packed near 1 with little spread to correlate. The twin reads more than 0.05 above PSBD-TM on {len(twin_ahead)} models ({word_list(sorted({attack_label(a) for a in twin_ahead["attack"]}))}) and more than 0.05 below on {len(twin_behind)} ({word_list(sorted({attack_label(a) for a in twin_behind["attack"]}))}). The mean gap is {pair_gap.mean():+.3f} because these cancel, so on ViT the 2 are the same probe on average but not model by model. On Swin-S they separate by {macro("swin", "SwinGainRecommendedMinusTwin")} in PSBD-TM's favor (`\SwinGainRecommendedMinusTwin`, `swin-and-robustness.ipynb`), which is why the paper names the input placement and records the ViT tie as an open question (`docs/open-questions.md`, Q20). The scatter does not say why masking the output of windowed attention is weaker on Swin, which is open.
    """),
]

write("attention-probes", cells)
