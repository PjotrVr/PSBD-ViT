"""What the score figures draw. Edit this file and rerun make.py.

.venv/bin/python -m experiments.score_figures.make
"""

# Short probe names per architecture, each mapped to a cached placement under
# results/<folder>/psbd/<placement>/. A combination that names a probe missing
# from an architecture's dict is skipped for that architecture and the skip is
# written to numbers.json.
PROBES = {
    "vit": {
        "tm": "before_attention_norm_token_mask",
        "rd": "post_residual",
        "band": "pre_residual_blocks_5_8",
        "late": "pre_residual_blocks_9_12",
    },
    # Swin-S has 24 blocks. Its fusion partner is the last third, blocks 17 to
    # 24, the 2nd pre-registered attempt, which held. The middle third, blocks 9
    # to 16, was the 1st attempt and failed (experiments/final_method/README.md).
    "swin": {
        "tm": "before_attention_norm_token_mask",
        "rd": "post_residual",
        "band": "pre_residual_blocks_17_24",
        "middle": "pre_residual_blocks_9_16",
    },
}

# A 1-element list draws that probe's own fractional PSU. A longer list fuses
# its probes under every rule in FUSION_RULES, the first probe leading.
COMBINATIONS = [
    ["tm"],
    ["rd"],
    ["band"],
    ["late"],
    ["tm", "band"],
    ["tm", "rd"],
    ["tm", "band", "rd"],
]

# "min" is the plain minimum of the clean-validation percentiles. "weighted"
# gives the first probe WEIGHTED_FIRST_SHARE of the budget and splits the rest
# equally, score = min_i(rank_i / share_i).
FUSION_RULES = ["min", "weighted"]
WEIGHTED_FIRST_SHARE = 0.9

# A set is either a named loader in make.py ("vit_panel", "swin_panel",
# "backdoorbench") or an explicit list of results/ folder names.
MODEL_SETS = {
    "vit_panel": "vit_panel",
    "swin_panel": "swin_panel",
    "backdoorbench": "backdoorbench",
}

# False-positive budgets, each a quantile of the clean-validation scores.
BUDGETS = (0.01, 0.05, 0.10, 0.20)

# The ROC is drawn and stored on a log grid of budgets from 1e-3 to 1.
ROC_POINTS = 120
ROC_LOWEST_FPR = 1e-3

# Histograms show the left tail, cut at this quantile of clean validation.
HISTOGRAM_UPPER_QUANTILE = 0.6
HISTOGRAM_BINS = 80

DPI = 90
