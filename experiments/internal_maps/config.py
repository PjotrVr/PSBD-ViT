"""Which models the internal maps cover and which figures each gets.

Edit this file and rerun measure.py (GPU) and make.py (CPU).

    .venv/bin/python -m experiments.internal_maps.measure --models vit_gtsrb_badnet_a2o_0_1
    .venv/bin/python -m experiments.internal_maps.make
"""

# 1 BadNets, 1 Blend, 1 LF, 1 WaNet, 1 TaCT and 1 benign reference per
# architecture, all on the successful panel. GTSRB at 10% where it exists. No ViT
# WaNet on GTSRB or CIFAR-100 clears the panel, so the ViT WaNet is Tiny at 10%.
# The ViT TaCT model is the PSBD-TM floor of the paper (GTSRB at 1%, cosine rerun).
# No Swin GTSRB TaCT model is on the Swin panel, so the Swin TaCT is CIFAR-10 at 5%.
# A benign model has no trigger of its own and is probed with the BadNets patch,
# the way its cache was swept. GTSRB models with target 0 fall onto class 0 when
# they see almost nothing, which floors the kept-window maps at the target, so
# CIFAR-100 BadNets at 10% is added as a second BadNets model per architecture.
MODELS = {
    "vit": {
        "vit_gtsrb_badnet_a2o_0_1": None,
        "vit_cifar100_badnet_a2o_0_1": None,
        "vit_gtsrb_blend_0_1": None,
        "vit_gtsrb_lf_0_1": None,
        "vit_tiny_wanet_0_1": None,
        "vit_gtsrb_tact_0_01_cos": None,
        "vit_gtsrb_benign": "badnet_a2o",
    },
    "swin": {
        "swin_gtsrb_badnet_a2o_0_1": None,
        "swin_cifar100_badnet_a2o_0_1": None,
        "swin_gtsrb_blend_0_1": None,
        "swin_gtsrb_lf_0_1": None,
        "swin_gtsrb_wanet_0_1": None,
        "swin_cifar10_tact_0_05": None,
        "swin_gtsrb_benign": "badnet_a2o",
    },
}

# The figures of the first batch. 1 token removal, 2 class-token attention (ViT)
# or readout attribution (Swin), 3 masking survival along depth, 4 backdoor
# direction per token, 5 the PSBD-TM masks themselves.
FIGURES = ("token_removal", "trigger_reading", "depth_survival", "direction", "masks")

# The grid every map is drawn on. ViT keeps 1 grid of 14 x 14 tokens in every
# block. Swin's grids shrink from 56 to 7, so a Swin map unit is 1 cell of the 7 x 7
# grid every stage nests into, masked as all of that cell's tokens at every stage.
MAP_GRID = {"vit": 14, "swin": 7}

# Side lengths of the kept neighbourhoods of the token removal figure, in map units.
# A 3 x 3 ViT window is 9 of 196 tokens, a 7 x 7 window 49 (a quarter). A Swin cell
# is 32 pixels, so 1 cell is about 4 ViT tokens and 3 x 3 cells 18% of the image.
KEPT_WINDOWS = {"vit": (3, 7), "swin": (1, 3)}

# Random visible shares, the global-trigger check of why_token_masking_works part D.
VISIBLE_FRACTIONS = (0.1, 0.3, 0.6)
VISIBLE_DRAWS = 5

# Depths drawn for attention rollout and the direction maps, 1-indexed blocks.
ROLLOUT_BLOCKS = {"vit": (3, 6, 9, 12)}
DIRECTION_BLOCKS = {"vit": (1, 4, 8, 10, 12), "swin": (2, 4, 12, 21, 24)}
# Late blocks over which the PSBD-TM mask figure shades the masked share.
LATE_BLOCKS = {"vit": (9, 12), "swin": (17, 24)}

# Pairs per model. Every pair is a triggered image the unperturbed model sends to
# the target and the clean twin it classifies correctly.
POOL_PAIRS = 512
REMOVAL_PAIRS = 32
EVAL_PAIRS = 64
PROBE_TRIGGERED_PAIRS = 256
PROBE_VALIDATION_IMAGES = 1000
MASK_PASSES = 6
EXAMPLE_PASSES = 3

DPI = 110

# Second batch, 1. The clean validation images that set PSBD-TM's 1% FPR threshold
# and the triggered images that escape it, read from the stage-1 caches (CPU).
# A ViT CIFAR-100, Tiny and GTSRB pick, the TaCT failure, 2 BackdoorBench
# checkpoints whose 1% threshold is set by clean images that gain confidence under
# masking, and 2 Swin models.
GALLERY_MODELS = (
    "vit_cifar100_badnet_a2o_0_01",
    "vit_tiny_wanet_0_05",
    "vit_gtsrb_bpp_0_05",
    "vit_gtsrb_tact_0_01_cos",
    "bb_cifar10_blind_0_1",
    "bb_gtsrb_ssba_0_1",
    "swin_cifar100_badnet_a2o_0_01",
    "swin_tiny_wanet_0_05",
)
GALLERY_COUNT = 20
GALLERY_BUDGET = 0.01

# Second batch, 2. Clean images whose PSBD-TM PSU is negative, on the 2 BackdoorBench
# checkpoints where such images set the 1% FPR threshold, against triggered images
# with PSU near 0 on the same model.
CONFIDENCE_MODELS = ("bb_cifar10_blind_0_1", "bb_gtsrb_ssba_0_1")
CONFIDENCE_IMAGES = 24
MAPS_DRAWN = 6
JOINT_COUNTS = (5, 10, 20)

# Second batch, 3. The single-source ViT TaCT models.
VETO_MODELS = (
    "vit_gtsrb_tact_0_01_cos",
    "vit_cifar10_tact_0_05",
    "vit_cifar10_tact_0_01",
)
VETO_POOL = 256
VETO_IMAGES = 32
VETO_COUNTS = (1, 2, 4, 8, 16, 32, 64)

# Second batch, 4. Patch-trigger models with known trigger tokens: BadNets with 4
# tokens (GTSRB, CIFAR-100, CIFAR-10 at 1%) and 1 token (Tiny) plus 3 single-source
# TaCT models, the 2 detected partly or not at all and the failure.
SURVIVAL_MODELS = (
    "vit_gtsrb_badnet_a2o_0_1",
    "vit_cifar100_badnet_a2o_0_1",
    "vit_cifar10_badnet_a2o_0_01",
    "vit_tiny_badnet_a2o_0_05",
    "vit_gtsrb_tact_0_05",
    "vit_cifar10_tact_0_05",
    "vit_gtsrb_tact_0_01_cos",
)
SURVIVAL_RATES = (0.3, 0.5, 0.7, 0.9)
SURVIVAL_IMAGES = 256
SURVIVAL_PASSES = 6
