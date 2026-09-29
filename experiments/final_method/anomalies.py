"""2 anomalies of the ViT panel readings, explained from the caches.

1. Under the min rule, TPR at 1% FPR collapses on single models whose AUROC stays
   high. 3 candidates are measured. The first is ties or the 1/2000 quantization
   of the percentiles at the threshold. The second is the realized FPR. The third
   is the budget split itself, since the 1% quantile of min(r1, r2) sits near each
   probe's 0.5% quantile and the lower tail of clean PSU may fall steeply there.
2. The CIFAR-10 TaCT models read high AUROC and TPR near 0. Their triggered and
   paired clean images are all from the source class, while the threshold comes
   from all-class validation. The realized FPR on source-class clean images and
   the threshold a source-class-only validation set would give are measured.

    .venv/bin/python -m experiments.final_method.anomalies
"""

import json
import os
import time

import numpy as np
import torch

from defenses.cache import read_split_manifest
from defenses.decision import pair_clean_to_backdoor
from defenses.scores import to_rank
from experiments._paths import experiment_result_path
from experiments.cache_readouts.fusion_rules import fractional_psu
from experiments.cache_readouts.shared import (
    choose_rate,
    load_baselines,
    load_model_set,
    load_passes,
    validation_shift_by_rate,
    write_json,
)
from scripts.coverage_ledger import source_classes_of

SLUG = "final_method"
RESULTS_DIR = "results"
ANCHOR = "before_attention_norm_token_mask"
PARTNER = "pre_residual_blocks_5_8"
QUANTILES = (0.01, 0.05, 0.10)
NAMED_COLLAPSES = (
    "vit_cifar10_badnet_a2o_0_1",
    "vit_cifar10_blend_0_05",
    "vit_cifar100_blend_0_01",
)
# A model counts as a collapse when the min rule loses this much TPR at 1% FPR
# against PSBD-TM alone while its AUROC stays at or above the floor.
COLLAPSE_DROP = 0.2
AUROC_FLOOR = 0.95


def main():
    started = time.perf_counter()
    models = load_model_set("panel")
    fusion_path = experiment_result_path(
        "cache_readouts", "fusion_rules_panel.json", RESULTS_DIR
    )
    with open(fusion_path) as handle:
        fusion = {r["folder"]: r for r in json.load(handle)["models"]}

    collapses = []
    for model in models:
        rules = fusion[model["folder_name"]]["partners"]["adaptive"]["middle_band"][
            "rules"
        ]
        drop = (
            rules["min_rank"]["at_fpr"]["q0.01"]["tpr"]
            - rules["tm_alone"]["at_fpr"]["q0.01"]["tpr"]
        )
        if drop <= -COLLAPSE_DROP and rules["min_rank"]["auroc"] >= AUROC_FLOOR:
            collapses.append(model["folder_name"])
    studied = sorted(set(collapses) | set(NAMED_COLLAPSES))
    min_rule = [min_rule_anatomy(folder) for folder in studied]

    tact = [tact_calibration(m["folder_name"]) for m in models if m["attack"] == "tact"]

    payload = {
        "experiment": "anomalies of the min rule at 1% FPR and of TaCT calibration",
        "collapse_rule": {"tpr_drop": COLLAPSE_DROP, "auroc_floor": AUROC_FLOOR},
        "collapses_on_panel": collapses,
        "min_rule": min_rule,
        "tact": tact,
        "wall_seconds": time.perf_counter() - started,
    }
    path = experiment_result_path(SLUG, "anomalies.json", RESULTS_DIR)
    write_json(payload, path)
    print(f"wrote {path} in {payload['wall_seconds']:.0f} s")


def adaptive_psu(psbd_dir, placement, baselines):
    rate = choose_rate(
        validation_shift_by_rate(psbd_dir, placement, baselines), "adaptive"
    )
    psu = fractional_psu(load_passes(psbd_dir, placement, rate, baselines))
    return psu, rate


def min_rule_anatomy(folder):
    psbd_dir = os.path.join(RESULTS_DIR, folder, "psbd")
    baselines = load_baselines(psbd_dir)
    manifest = read_split_manifest(psbd_dir)
    anchor, anchor_rate = adaptive_psu(psbd_dir, ANCHOR, baselines)
    partner, partner_rate = adaptive_psu(psbd_dir, PARTNER, baselines)

    def ranks(split):
        anchor_rank = to_rank(anchor[split], anchor["validation"])  # (n,)
        partner_rank = to_rank(partner[split], partner["validation"])  # (n,)
        return anchor_rank, partner_rank

    anchor_val, partner_val = ranks("validation")
    anchor_bd, partner_bd = ranks("backdoor")
    fused_val = torch.minimum(anchor_val, partner_val)  # (n_validation,)
    fused_bd = torch.minimum(anchor_bd, partner_bd)  # (n_backdoor,)
    fused_clean = pair_clean_to_backdoor(
        torch.minimum(*ranks("clean")), manifest
    )  # (n_backdoor,)
    threshold = float(np.quantile(fused_val.numpy(), 0.01))

    # The share of each probe's own validation scores the fused threshold admits
    # is that probe's effective budget.
    effective_anchor = float((anchor_val < threshold).float().mean())
    effective_partner = float((partner_val < threshold).float().mean())

    def anchor_tpr_at(quantile):
        cut = float(np.quantile(anchor["validation"].numpy(), quantile))
        tpr = float((anchor["backdoor"] < cut).float().mean())
        return {"threshold": cut, "tpr": tpr}

    # The mean as a tie-breaker, the fix proposed for a tie artifact, scored so
    # the reader can see whether it would change anything.
    mean_val = (anchor["validation"] + partner["validation"]) / 2
    mean_bd = (anchor["backdoor"] + partner["backdoor"]) / 2
    tiebroken_val = fused_val + 1e-6 * mean_val.clamp(-1, 1)
    tiebroken_bd = fused_bd + 1e-6 * mean_bd.clamp(-1, 1)
    tiebroken_threshold = float(np.quantile(tiebroken_val.numpy(), 0.01))

    anatomy = {
        "folder": folder,
        "rates": {ANCHOR: anchor_rate, PARTNER: partner_rate},
        "validation_size": int(fused_val.numel()),
        "min_threshold_q0.01": threshold,
        "tie_share_at_threshold": float((fused_val == threshold).float().mean()),
        "validation_share_at_zero": float((fused_val == 0).float().mean()),
        "min_tpr_q0.01": float((fused_bd < threshold).float().mean()),
        "min_realized_fpr_q0.01": float((fused_clean < threshold).float().mean()),
        "min_tpr_q0.01_tiebroken": float(
            (tiebroken_bd < tiebroken_threshold).float().mean()
        ),
        "effective_quantile_anchor": effective_anchor,
        "effective_quantile_partner": effective_partner,
        "anchor_alone_q0.01": anchor_tpr_at(0.01),
        "anchor_alone_at_effective_quantile": anchor_tpr_at(effective_anchor),
        "partner_alone_q0.01_tpr": float(
            (
                partner["backdoor"]
                < float(np.quantile(partner["validation"].numpy(), 0.01))
            )
            .float()
            .mean()
        ),
        "anchor_validation_psu_quantiles": {
            f"{q}": float(np.quantile(anchor["validation"].numpy(), q))
            for q in (0.005, 0.01, 0.02)
        },
        "anchor_backdoor_psu_quantiles": {
            f"{q}": float(np.quantile(anchor["backdoor"].numpy(), q))
            for q in (0.1, 0.5, 0.9)
        },
    }
    return anatomy


def tact_calibration(folder):
    psbd_dir = os.path.join(RESULTS_DIR, folder, "psbd")
    baselines = load_baselines(psbd_dir)
    manifest = read_split_manifest(psbd_dir)
    sources = source_classes_of(
        "checkpoints", {"folder_name": folder, "attack": "tact"}
    )
    anchor, rate = adaptive_psu(psbd_dir, ANCHOR, baselines)

    _, _, validation_labels = baselines["validation"]
    _, _, clean_labels = baselines["clean"]
    in_source_validation = sum(
        validation_labels == s for s in sources
    ).bool()  # (n_validation,)
    in_source_clean = sum(clean_labels == s for s in sources).bool()  # (n_clean,)
    paired_clean = pair_clean_to_backdoor(anchor["clean"], manifest)  # (n_backdoor,)
    paired_labels = pair_clean_to_backdoor(clean_labels, manifest)  # (n_backdoor,)
    assert all(int(label) in sources for label in paired_labels), folder

    rows = {}
    for quantile in QUANTILES:
        key = f"q{quantile:.2f}"
        all_class = float(np.quantile(anchor["validation"].numpy(), quantile))
        source_only = float(
            np.quantile(anchor["validation"][in_source_validation].numpy(), quantile)
        )
        rows[key] = {
            "threshold_all_class": all_class,
            "tpr_all_class": float((anchor["backdoor"] < all_class).float().mean()),
            "realized_fpr_source_clean": float(
                (paired_clean < all_class).float().mean()
            ),
            "realized_fpr_other_clean": float(
                (anchor["clean"][~in_source_clean] < all_class).float().mean()
            ),
            "threshold_source_only": source_only,
            "tpr_source_only": float((anchor["backdoor"] < source_only).float().mean()),
            "realized_fpr_source_only": float(
                (paired_clean < source_only).float().mean()
            ),
        }
    # A defender does not know the source class, but it does know each input's
    # predicted class. A threshold per predicted class, read from the validation
    # images the model predicts as that class, is the deployable form.
    _, validation_predicted, _ = baselines["validation"]
    _, clean_predicted, _ = baselines["clean"]
    _, backdoor_predicted, _ = baselines["backdoor"]
    paired_predicted = pair_clean_to_backdoor(
        clean_predicted, manifest
    )  # (n_backdoor,)
    for quantile in QUANTILES:
        key = f"q{quantile:.2f}"
        cuts = {}
        for label in set(backdoor_predicted.tolist()) | set(paired_predicted.tolist()):
            members = anchor["validation"][validation_predicted == label]
            cuts[label] = (
                float(np.quantile(members.numpy(), quantile))
                if members.numel()
                else float("-inf")
            )
        backdoor_cut = torch.tensor(
            [cuts[int(v)] for v in backdoor_predicted]
        )  # (n_backdoor,)
        clean_cut = torch.tensor(
            [cuts[int(v)] for v in paired_predicted]
        )  # (n_backdoor,)
        rows[key]["tpr_per_predicted_class"] = float(
            (anchor["backdoor"] < backdoor_cut).float().mean()
        )
        rows[key]["realized_fpr_per_predicted_class"] = float(
            (paired_clean < clean_cut).float().mean()
        )

    calibration = {
        "folder": folder,
        "source_classes": list(sources),
        "rate": rate,
        "n_source_validation": int(in_source_validation.sum()),
        "n_backdoor": int(anchor["backdoor"].numel()),
        "psu_median": {
            "validation_all": float(anchor["validation"].median()),
            "validation_source": float(
                anchor["validation"][in_source_validation].median()
            ),
            "clean_paired_source": float(paired_clean.median()),
            "backdoor": float(anchor["backdoor"].median()),
        },
        "by_fpr": rows,
    }
    return calibration


if __name__ == "__main__":
    main()
