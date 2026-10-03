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

DPI = 110

# Font sizes in points and line width, sized so a figure printed at text width
# still reads at 100% zoom.
FONT_SIZE = 12.5
TITLE_SIZE = 14
LEGEND_SIZE = 11
LINE_WIDTH = 2.4

# What each probe is called in titles and legends, per architecture. A single
# probe says "alone" so it is never mistaken for a fusion.
PROBE_LABELS = {
    "vit": {
        "tm": "PSBD-TM alone (token masking, attention input)",
        "rd": "PSBD-RD alone (dropout after residual adds)",
        "band": "middle band alone (residual dropout, blocks 5 to 8)",
        "late": "late band alone (residual dropout, blocks 9 to 12)",
    },
    "swin": {
        "tm": "PSBD-TM alone (token masking, attention input)",
        "rd": "PSBD-RD alone (dropout after residual adds)",
        "band": "late band alone (residual dropout, blocks 17 to 24)",
        "middle": "middle band alone (residual dropout, blocks 9 to 16)",
    },
}
# Short names for the probes inside a fusion label.
PROBE_SHORT_NAMES = {
    "vit": {
        "tm": "PSBD-TM",
        "rd": "PSBD-RD",
        "band": "middle band",
        "late": "late band",
    },
    "swin": {
        "tm": "PSBD-TM",
        "rd": "PSBD-RD",
        "band": "late band",
        "middle": "middle band",
    },
}

# The figure types below each have an on/off flag and their own options. A model
# listed under a set must already have its numbers.json, since every figure reads
# the probe rates from it. Per-model figures land beside numbers.json, set-level
# ones in the set's directory and cross-set ones in _across_sets/.

# Clean-validation percentile of the anchor against that of the partner, with the
# regions each fusion rule flags at the budget.
SCATTER = {
    "enabled": True,
    "anchor": "tm",
    "partner": "band",
    "budget": 0.01,
    "models": {
        "vit_panel": ["vit_cifar10_wanet_0_1", "vit_tiny_blend_0_05"],
        "swin_panel": ["swin_cifar10_wanet_0_1"],
        "backdoorbench": ["bb_cifar10_blind_0_1"],
    },
}

# Starting confidence against absolute and fractional PSU.
CONFIDENCE = {
    "enabled": True,
    "probe": "tm",
    "budget": 0.01,
    "models": {
        "vit_panel": [
            "vit_cifar100_badnet_a2o_0_05",
            "vit_cifar100_blend_0_05",
            "vit_tiny_badnet_a2o_0_01",
            "vit_gtsrb_badnet_a2o_0_05",
        ],
    },
}

# Share of changed predictions and mean surviving probability along each
# probe's cached rate ladder, averaged per attack.
SHIFT_LADDER = {
    "enabled": True,
    "probes": ["tm", "rd"],
    "sets": ["vit_panel", "swin_panel"],
}

# The class each changed clean-validation prediction moves to.
FLIP_TARGETS = {
    "enabled": True,
    "probe": "tm",
    "models": {
        "vit_panel": ["vit_gtsrb_tact_0_01_cos", "vit_gtsrb_badnet_a2o_0_05"],
        "swin_panel": ["swin_gtsrb_badnet_a2o_0_05"],
    },
}

# Clean-accuracy cost against PSBD-TM AUROC for the adaptive attackers.
EVADERS = {
    "enabled": True,
    "record": "results/_experiments/final_method/adaptive_attackers.json",
    "success_bar_points": 2,
}

# Share of clean validation whose fractional PSU is below 0 against TPR.
NEGATIVE_PSU = {
    "enabled": True,
    "probe": "tm",
    "budget": 0.01,
    "sets": ["vit_panel", "swin_panel", "backdoorbench"],
}

# Nominal FPR against the FPR realized on the paired clean test split.
THRESHOLD_TRANSFER = {
    "enabled": True,
    "curves": [["tm", "psu"], ["tm+band", "weighted"]],
    "sets": ["vit_panel", "swin_panel"],
}
