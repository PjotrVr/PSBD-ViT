"""Which models every script in this directory reads, and how their caches load.

The all-to-all models are not in the coverage ledger, so the ledger's success rule
is applied here by hand, with the same bars the ledger reads from
configs/psbd_basis.json: ASR at least the ledger's bar, not diverged (clean
accuracy at least half the benign reference) and clean accuracy within 2 points of
the benign reference. All-to-all BadNets reaches a lower ASR than all-to-one on the
larger label spaces, so a relaxed bar is also read and always labeled as such.

The all-to-one cost panel is the ledger's own successful_2pt panel, ViT from
results/coverage/coverage.json and Swin from the in-memory Swin ledger.
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from defenses.cache import (  # noqa: E402
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defenses.decision import (  # noqa: E402
    ADAPTIVE_SHIFT_TARGET,
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
    complete_rates,
    select_rate_adaptively,
    select_rate_at_matched_shift,
)
from defenses.scores import shift_ratio  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    PANEL_DATASETS,
    clearing_cells,
    load_coverage,
    swin_coverage,
)

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
CHECKPOINTS_DIR = os.path.join(REPO_ROOT, "checkpoints")
OUT_DIR = os.path.join(RESULTS_DIR, "_experiments", "all_to_all_detection")
# Placements this experiment swept itself (run_gpu.sh) live beside its results,
# never in the canonical tree, with the canonical baselines and manifest linked in
# so their rows are the canonical rows.
EXTRA_CACHE_ROOT = os.path.join(OUT_DIR, "caches")

# Chosen before any detection number was read. All-to-all chance is 1 / K, so an
# attack that sends 70% of triggered non-target images to their rotated class is
# still a working implant, and 0.7 admits the CIFAR-100 and Tiny ViT models that
# stop just short of the ledger's bar.
RELAXED_ASR_BAR = 0.7
ARCHITECTURES = ("vit", "swin")
POISON_RATE_TOKENS = ("0_005", "0_01", "0_05", "0_1")
PLACEMENTS = (RECOMMENDED_PLACEMENT, PUBLISHED_PLACEMENT)
SIDE_VARIANT_TOKENS = ("sam_rho", "evade")

# Development and confirmation datasets are fixed here so that no statistic is
# tuned on the datasets that confirm it.
DEVELOPMENT_DATASETS = ("cifar10", "gtsrb")
CONFIRMATION_DATASETS = ("cifar100", "tiny")


def main_panel_table() -> dict:
    ledger = load_coverage(RESULTS_DIR)
    bars = {
        "asr_bar": ledger["asr_bar"],
        "clean_drop_bar_2pt": ledger["clean_accuracy_drop_bar_headline"],
        "clean_drop_bar_5pt": ledger["clean_accuracy_drop_bar"],
        "relaxed_asr_bar": RELAXED_ASR_BAR,
    }
    swin_ledger = swin_coverage(RESULTS_DIR, CHECKPOINTS_DIR)
    references = {
        "vit": ledger["benign_reference_accuracy"],
        "swin": swin_ledger["benign_reference_accuracy"],
    }

    all_to_all = []
    for folder in sorted(os.listdir(CHECKPOINTS_DIR)):
        if "_badnet_a2a_" not in folder:
            continue
        architecture, dataset = folder.split("_")[:2]
        if architecture not in ARCHITECTURES or dataset not in PANEL_DATASETS:
            continue
        verdict = judge_model(folder, architecture, dataset, references, bars)
        all_to_all.append(verdict)

    all_to_one = {
        "vit": [cell["folder_name"] for cell in clearing_cells(ledger)],
        "swin": [cell["folder_name"] for cell in clearing_cells(swin_ledger)],
    }
    benign = [
        f"{architecture}_{dataset}_benign"
        for architecture in ARCHITECTURES
        for dataset in PANEL_DATASETS
        if os.path.isdir(os.path.join(RESULTS_DIR, f"{architecture}_{dataset}_benign"))
    ]
    table = {
        "bars": bars,
        "benign_reference_accuracy": references,
        "all_to_all": all_to_all,
        "all_to_one": all_to_one,
        "benign": benign,
    }
    return table


def judge_model(folder, architecture, dataset, references, bars):
    # cli.evaluate writes metrics.json. The evasion runs carry the same 2 numbers
    # in args.json only, measured at the end of training.
    metrics = {}
    for name in ("metrics.json", "args.json"):
        path = os.path.join(CHECKPOINTS_DIR, folder, name)
        if not metrics.get("asr") and os.path.exists(path):
            metrics = json.load(open(path))
    asr = metrics.get("asr")
    clean_accuracy = metrics.get("clean_accuracy")
    reference = references[architecture].get(dataset)
    variant = next((token for token in SIDE_VARIANT_TOKENS if token in folder), None)
    cached = {
        placement: bool(complete_rates(psbd_dir_of(folder, placement), placement))
        for placement in PLACEMENTS
    }

    verdict = {
        "folder": folder,
        "architecture": architecture,
        "dataset": dataset,
        "variant": variant,
        "asr": asr,
        "clean_accuracy": clean_accuracy,
        "benign_reference": reference,
        "cached": cached,
    }
    if asr is None or clean_accuracy is None or reference is None:
        verdict.update(reason="no recorded ASR or no benign reference")
        verdict.update(strict_2pt=False, strict_5pt=False, relaxed_2pt=False)
        return verdict

    drop = clean_accuracy - reference
    diverged = clean_accuracy < reference / 2
    within_2pt = drop >= bars["clean_drop_bar_2pt"]
    within_5pt = drop >= bars["clean_drop_bar_5pt"]
    verdict.update(
        clean_accuracy_drop=drop,
        diverged=diverged,
        strict_2pt=(not diverged) and asr >= bars["asr_bar"] and within_2pt,
        strict_5pt=(not diverged) and asr >= bars["asr_bar"] and within_5pt,
        relaxed_2pt=(not diverged) and asr >= RELAXED_ASR_BAR and within_2pt,
    )
    verdict["reason"] = reason_for(verdict, bars)
    return verdict


def reason_for(verdict, bars):
    if verdict["diverged"]:
        return "diverged"
    if verdict["asr"] < RELAXED_ASR_BAR:
        return f"ASR below {RELAXED_ASR_BAR}"
    if verdict["clean_accuracy_drop"] < bars["clean_drop_bar_2pt"]:
        return "clean accuracy more than 2 points below benign"
    if verdict["asr"] < bars["asr_bar"]:
        return f"ASR below {bars['asr_bar']}, kept at the relaxed bar only"
    return "successful"


def psbd_dir_of(folder, placement=None):
    canonical = os.path.join(RESULTS_DIR, folder, "psbd")
    extra = os.path.join(EXTRA_CACHE_ROOT, folder, "psbd")
    if placement is None or complete_rates(canonical, placement):
        return canonical
    path = extra if complete_rates(extra, placement) else canonical
    return path


def adaptive_rate(folder, placement):
    shifts = validation_shift_by_rate(folder, placement)
    rate = select_rate_adaptively(shifts, ADAPTIVE_SHIFT_TARGET) if shifts else None
    return rate


def matched_rate(folder, placement, target):
    shifts = validation_shift_by_rate(folder, placement)
    rate = select_rate_at_matched_shift(shifts, target) if shifts else None
    return rate


def validation_shift_by_rate(folder, placement):
    psbd_dir = psbd_dir_of(folder, placement)
    rates = complete_rates(psbd_dir, placement)
    if not rates:
        return {}
    _, validation_labels, _ = load_baseline(baseline_path(psbd_dir, "validation"))

    shift_by_rate = {}
    for rate in rates:
        _, per_pass_argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, "validation")
        )
        shift_by_rate[rate] = shift_ratio(validation_labels, per_pass_argmax)
    return shift_by_rate


def load_model_cache(folder, placement, rate=None):
    """Every split of 1 model at 1 placement and rate, the adaptive rate by default.

    Per split: probs (n, K) the unperturbed softmax, pred (n,) its argmax,
    loader_labels (n,) the served label (true class on clean splits, attack label on
    the backdoor split), pass_probs (k, n) each pass's probability of pred and
    pass_argmax (k, n) each pass's predicted class. The backdoor rows are paired to
    their clean images through the manifest, so clean_paired holds the clean split
    reindexed to backdoor row order.
    """
    rate = rate if rate is not None else adaptive_rate(folder, placement)
    if rate is None:
        return None
    psbd_dir = psbd_dir_of(folder, placement)
    manifest = read_split_manifest(psbd_dir)

    splits = {}
    for split in ("validation", "clean", "backdoor"):
        probs, pred, loader_labels = load_baseline(baseline_path(psbd_dir, split))
        pass_probs, pass_argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, split)
        )
        assert pass_argmax.shape == pass_probs.shape == (pass_probs.shape[0], len(pred))
        splits[split] = {
            "probs": probs.float(),  # (n, K)
            "pred": pred.long(),  # (n,)
            "loader_labels": loader_labels.long(),  # (n,)
            "pass_probs": pass_probs.float(),  # (k, n)
            "pass_argmax": pass_argmax.long(),  # (k, n)
        }

    clean_rows = row_map(manifest)
    splits["clean_paired"] = {
        key: (
            value[:, clean_rows]
            if value.dim() == 2 and key.startswith("pass")
            else value[clean_rows]
        )
        for key, value in splits["clean"].items()
    }
    cache = {"folder": folder, "placement": placement, "rate": rate, "splits": splits}
    return cache


def row_map(manifest):
    clean_indices = manifest["analysis_clean_indices"]
    backdoor_indices = manifest["analysis_backdoor_indices"]
    row_of = {original: row for row, original in enumerate(clean_indices)}
    rows = torch.tensor([row_of[original] for original in backdoor_indices])
    return rows


def write_json(payload, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return path


if __name__ == "__main__":
    table = main_panel_table()
    print(write_json(table, "panel.json"))
    for row in table["all_to_all"]:
        print(
            f"{row['folder']:45s} asr={row['asr']} ca={row['clean_accuracy']} "
            f"{row.get('reason')} cached={row['cached']}"
        )
    print({key: len(value) for key, value in table["all_to_one"].items()})
    print(table["benign"])
