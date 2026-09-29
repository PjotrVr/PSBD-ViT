"""X1 of docs/simple-experiments-plan.md: where PSBD-TM sends triggered WaNet predictions.

The noise-mode account (WaNet trains on random warps labeled with their true
class) predicts that a triggered input whose trigger PSBD-TM breaks falls back to
its true class, at a share of at least 0.7 of the changed passes. This reads the
cached per-pass argmax at the adaptive rate and reports, for triggered rows the
backdoor fires on, the share of passes that change and the share of those that
land on the true class, plus the clean rows' change and target-landing shares.

    PYTHONPATH=. .venv/bin/python experiments/literature_checks/wanet_destinations.py
"""

import json

from defenses.cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defenses.decision import pair_clean_to_backdoor
from scripts.paper._common import load_psbd_metrics

PSBD_TM = "before_attention_norm_token_mask"
FOLDERS = (
    "vit_cifar10_wanet_0_1",
    "vit_tiny_wanet_0_05",
    "vit_tiny_wanet_0_1",
    "vit_cifar10_wanet_0_05",
    "vit_gtsrb_wanet_0_1",
    "vit_cifar10_badnet_a2o_0_01",
    "vit_cifar10_blend_0_1",
    "vit_cifar10_bpp_0_05",
)


def destinations(folder: str) -> dict:
    psbd_dir = f"results/{folder}/psbd"
    rate = load_psbd_metrics("results", folder)["placements"][PSBD_TM]["adaptive_rate"]
    with open(f"checkpoints/{folder}/args.json") as handle:
        args = json.load(handle)
    target = args["target_label"]
    manifest = read_split_manifest(psbd_dir)

    # baseline argmax (n,) per split, clean loader labels are the true classes
    _, triggered_argmax, _ = load_baseline(baseline_path(psbd_dir, "backdoor"))
    _, clean_argmax, clean_labels = load_baseline(baseline_path(psbd_dir, "clean"))
    # per-pass argmax (passes, n) at the adaptive rate
    _, triggered_pass_argmax = load_dropout_pass_probs(
        dropout_pass_path(psbd_dir, PSBD_TM, rate, "backdoor")
    )
    _, clean_pass_argmax = load_dropout_pass_probs(
        dropout_pass_path(psbd_dir, PSBD_TM, rate, "clean")
    )
    # true class of each triggered row (n_backdoor,) through the clean pairing
    true_class = pair_clean_to_backdoor(clean_labels.float(), manifest).long()

    fired = triggered_argmax == target
    changed = (
        triggered_pass_argmax != triggered_argmax.unsqueeze(0)
    ) & fired.unsqueeze(0)
    to_true = changed & (triggered_pass_argmax == true_class.unsqueeze(0))
    clean_changed = clean_pass_argmax != clean_argmax.unsqueeze(0)
    clean_to_target = clean_changed & (clean_pass_argmax == target)

    n_changed = max(changed.sum().item(), 1)
    n_clean_changed = max(clean_changed.sum().item(), 1)
    reading = {
        "cover_rate": args.get("cover_rate"),
        "rate": rate,
        "triggered_changed": changed.sum().item() / changed.numel(),
        "triggered_changed_to_true": to_true.sum().item() / n_changed,
        "clean_changed": clean_changed.sum().item() / clean_changed.numel(),
        "clean_changed_to_target": clean_to_target.sum().item() / n_clean_changed,
    }
    return reading


def main() -> None:
    for folder in FOLDERS:
        reading = destinations(folder)
        print(
            folder,
            json.dumps(
                {
                    key: round(value, 3) if isinstance(value, float) else value
                    for key, value in reading.items()
                }
            ),
        )


if __name__ == "__main__":
    main()
