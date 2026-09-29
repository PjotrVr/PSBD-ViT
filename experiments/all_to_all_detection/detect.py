"""Mapping-agnostic prediction-shift statistics for all-to-all backdoors, scored from the caches.

Every statistic is a function of what a defender holds: the unperturbed softmax of
an input, the class each perturbed pass predicts, the probability each pass gives
the unperturbed prediction, and the same quantities on the clean validation split.
None reads a true label, an attack label, a target class or the rotation. The views
the statistics receive carry no label field at all, and relabeling_check proves on
every model that permuting the class indices of every softmax and argmax leaves
every per-input score unchanged.

Direction is fixed here before any AUROC was read, and low means poisoned for every
statistic. The reasons come from mechanism.py and tokens.py (README.md).

    psu_tm              PSBD-TM's fractional PSU at the adaptive rate, the published
                        direction, predicted to invert on all-to-all
    psu_rd              the same at PSBD-RD (post_residual), where cached
    neg_entropy         negative entropy of the unperturbed softmax (H43). A
                        confidence control, it perturbs nothing and is not PSBD
    confidence          the unperturbed top probability, the same kind of control
    psu_tm_two_sided    minus the distance of psu_tm's validation rank from 0.5
    psu_tm_class_two_sided
                        minus |z| of psu_tm against validation inputs with the same
                        predicted class, shrunk to the global mean on small classes
    destination_concentration
                        minus the largest share of passes that land on 1 class
                        other than the prediction, the hypothesis that a triggered
                        input falls back to 1 consistent class
    transition_typicality
                        mean log probability, under the validation transition
                        matrix P(pass class | predicted class), of the classes this
                        input's passes land on
    late_fragility      minus PSU under dropout on the residual stream in the last
                        third of the blocks only (ViT 9 to 12, Swin 17 to 24)
    depth_profile       validation rank of psu_tm minus validation rank of the late
                        PSU. A clean prediction is settled before the late blocks
                        and fragile to masking everywhere, a triggered all-to-all
                        prediction is assembled late, an all-to-one prediction
                        survives masking
    either_regime       the smaller of the validation ranks of psu_tm and of
                        late_fragility, flagging an input that is abnormal in
                        either direction

2 routers pick 1 statistic per model from the sign of d, the mean psu_tm over the
suspect pool minus that over clean validation (H43): psu_tm when d < 0, else
neg_entropy (router_entropy, the H43 rule) or late_fragility (router_late).

    source .venv/bin/activate
    python experiments/all_to_all_detection/detect.py --datasets cifar10 gtsrb
    python experiments/all_to_all_detection/detect.py
"""

import argparse
import os
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from defenses.decision import PUBLISHED_PLACEMENT, RECOMMENDED_PLACEMENT  # noqa: E402
from scripts.paper._common import load_args_json  # noqa: E402
from panel import (  # noqa: E402
    CHECKPOINTS_DIR,
    CONFIRMATION_DATASETS,
    DEVELOPMENT_DATASETS,
    adaptive_rate,
    load_model_cache,
    main_panel_table,
    matched_rate,
    write_json,
)

LATE_BAND = {"vit": "pre_residual_blocks_9_12", "swin": "pre_residual_blocks_17_24"}
# The late band cannot reach the adaptive 0.8 target on most models, so it is read
# at the rate whose clean validation shift ratio is nearest 0.8, which is the
# adaptive rate whenever that rate exists.
LATE_BAND_TARGET = 0.8
FPR_BUDGETS = (0.01, 0.05, 0.10)
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 0
RELABEL_SEED = 0
PROBABILITY_FLOOR = 1e-12
PSU_FLOOR = 1e-6
TRANSITION_SMOOTHING = 1.0
CLASS_SHRINKAGE = 10.0
STATISTICS = (
    "psu_tm",
    "psu_rd",
    "neg_entropy",
    "confidence",
    "psu_tm_two_sided",
    "psu_tm_class_two_sided",
    "destination_concentration",
    "transition_typicality",
    "late_fragility",
    "depth_profile",
    "either_regime",
)
ROUTERS = {"router_entropy": "neg_entropy", "router_late": "late_fragility"}
CONFIDENCE_CONTROLS = ("neg_entropy", "confidence")
# Chosen on the development datasets (CIFAR-10 and GTSRB all-to-all, and the
# all-to-one panel as the cost check) on 2026-09-29, before any confirmation
# all-to-all model was scored. either_regime is the 1 statistic that stays above
# chance in both regimes, late_fragility is its all-to-all half.
CHOSEN_DETECTOR = "either_regime"
CHOSEN_ALL_TO_ALL_COMPONENT = "late_fragility"


def main():
    args = parse_args()
    started = time.time()
    table = main_panel_table()
    groups = model_groups(table, args.datasets)

    verdicts = {row["folder"]: row for row in table["all_to_all"]}
    models = []
    for group, folders in groups.items():
        for folder in folders:
            record = score_model(folder, group)
            if record is not None:
                verdict = verdicts.get(folder, {})
                for bar in ("strict_2pt", "strict_5pt", "relaxed_2pt"):
                    record[bar] = verdict.get(bar)
                models.append(record)
            print(f"[{group}] {folder} {'ok' if record else 'no cache'}", flush=True)

    payload = {
        "datasets": args.datasets,
        "statistics": list(STATISTICS) + list(ROUTERS),
        "confidence_controls": list(CONFIDENCE_CONTROLS),
        "chosen_detector": CHOSEN_DETECTOR,
        "chosen_all_to_all_component": CHOSEN_ALL_TO_ALL_COMPONENT,
        "late_band": LATE_BAND,
        "late_band_target": LATE_BAND_TARGET,
        "fpr_budgets": list(FPR_BUDGETS),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "development_datasets": list(DEVELOPMENT_DATASETS),
        "confirmation_datasets": list(CONFIRMATION_DATASETS),
        "summary": summarize(models),
        "models": models,
        "seconds": round(time.time() - started, 1),
    }
    print(write_json(payload, args.out))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEVELOPMENT_DATASETS + CONFIRMATION_DATASETS),
        help="datasets of the all-to-all models scored, the cost panels are always whole",
    )
    parser.add_argument("--out", default="detection.json")
    return parser.parse_args()


def model_groups(table, datasets):
    all_to_all = [
        row
        for row in table["all_to_all"]
        if row["dataset"] in datasets and (row["relaxed_2pt"] or row["strict_5pt"])
    ]
    groups = {
        "all_to_all": [row["folder"] for row in all_to_all if row["variant"] is None],
        "all_to_all_side": [
            row["folder"] for row in all_to_all if row["variant"] is not None
        ],
        "benign": table["benign"],
        "all_to_one": table["all_to_one"]["vit"] + table["all_to_one"]["swin"],
    }
    return groups


def score_model(folder, group):
    architecture = folder.split("_")[0]
    probes = {
        "tm": (RECOMMENDED_PLACEMENT, adaptive_rate(folder, RECOMMENDED_PLACEMENT)),
        "rd": (PUBLISHED_PLACEMENT, adaptive_rate(folder, PUBLISHED_PLACEMENT)),
        "late": (
            LATE_BAND[architecture],
            matched_rate(folder, LATE_BAND[architecture], LATE_BAND_TARGET),
        ),
    }
    caches = {
        probe: load_model_cache(folder, placement, rate)
        for probe, (placement, rate) in probes.items()
        if rate is not None
    }
    if "tm" not in caches:
        return None
    check_same_split(caches)

    reference = caches["tm"]["splits"]
    views = {
        split: defender_view(caches, split)
        for split in ("validation", "clean_paired", "backdoor", "clean")
    }
    scores = {
        split: all_statistics(views[split], views["validation"])
        for split in ("validation", "clean_paired", "backdoor", "clean")
    }
    relabeling_check(views)

    # Labels enter only here, after scoring, to say which rows are poisoned and
    # which triggered rows the backdoor captured.
    captured = reference["backdoor"]["pred"] == reference["backdoor"]["loader_labels"]
    metrics = {
        name: detection_metrics(
            scores["validation"][name],
            scores["clean_paired"][name],
            scores["backdoor"][name],
            captured,
        )
        for name in scores["validation"]
    }
    pool_shift = suspect_pool_shift(scores, folder)
    for router, alternative in ROUTERS.items():
        chosen = "psu_tm" if pool_shift < 0 else alternative
        if chosen in metrics:
            metrics[router] = dict(metrics[chosen], chosen=chosen)

    record = {
        "folder": folder,
        "group": group,
        "architecture": architecture,
        "dataset": folder.split("_")[1],
        "rates": {probe: cache["rate"] for probe, cache in caches.items()},
        "captured_share": float(captured.float().mean()),
        "pool_shift_d": pool_shift,
        "relabeling_invariant": True,
        "metrics": metrics,
    }
    return record


# Probes read from different placements are paired row by row, so both must come
# from the same split and the same unperturbed predictions.
def check_same_split(caches):
    reference = caches["tm"]["splits"]
    for cache in caches.values():
        for split in ("validation", "clean", "backdoor"):
            same = torch.equal(cache["splits"][split]["pred"], reference[split]["pred"])
            assert same, f"{cache['placement']} {split} is not the PSBD-TM split"


# What a defender sees of 1 split: the unperturbed softmax and argmax and, per
# probe, the pass tensors. No label of any kind is copied in.
def defender_view(caches, split):
    base = caches["tm"]["splits"][split]
    view = {
        "probs": base["probs"],  # (n, K)
        "pred": base["pred"],  # (n,)
        "pass_probs": {
            probe: cache["splits"][split]["pass_probs"]
            for probe, cache in caches.items()
        },  # per probe, (k, n)
        "pass_argmax": {
            probe: cache["splits"][split]["pass_argmax"]
            for probe, cache in caches.items()
        },  # per probe, (k, n)
    }
    return view


def all_statistics(view, validation):
    scores = {}
    floored = view["probs"].clamp_min(PROBABILITY_FLOOR)  # (n, K)
    scores["neg_entropy"] = (floored * floored.log()).sum(dim=1)  # (n,)
    scores["confidence"] = view["probs"].max(dim=1).values  # (n,)

    psu = {probe: psu_ratio(view, probe) for probe in view["pass_probs"]}
    validation_psu = {
        probe: psu_ratio(validation, probe) for probe in view["pass_probs"]
    }
    scores["psu_tm"] = psu["tm"]
    if "rd" in psu:
        scores["psu_rd"] = psu["rd"]

    tm_rank = validation_rank(psu["tm"], validation_psu["tm"])  # (n,)
    scores["psu_tm_two_sided"] = -(tm_rank - 0.5).abs()
    scores["psu_tm_class_two_sided"] = class_two_sided(
        psu["tm"], view["pred"], validation_psu["tm"], validation["pred"]
    )
    scores["destination_concentration"] = -destination_concentration(
        view["pass_argmax"]["tm"], view["pred"], view["probs"].shape[1]
    )
    scores["transition_typicality"] = transition_typicality(view, validation)

    if "late" in psu:
        scores["late_fragility"] = -psu["late"]
        late_rank = validation_rank(psu["late"], validation_psu["late"])  # (n,)
        scores["depth_profile"] = tm_rank - late_rank
        fragile_rank = validation_rank(-psu["late"], -validation_psu["late"])
        scores["either_regime"] = torch.minimum(tm_rank, fragile_rank)
    return scores


def psu_ratio(view, probe):
    tracked = view["probs"].gather(1, view["pred"].view(-1, 1)).squeeze(1)  # (n,)
    tracked = tracked.clamp_min(PSU_FLOOR)
    perturbed = view["pass_probs"][probe].mean(dim=0)  # (n,)
    ratio = (tracked - perturbed) / tracked  # (n,)
    return ratio


# Mid-rank of each value in the validation distribution, in [0, 1], so ties in a
# discrete statistic sit in the middle of their block rather than at an edge.
def validation_rank(values, validation_values):
    ordered = np.sort(validation_values.numpy())
    below = np.searchsorted(ordered, values.numpy(), side="left")
    at_or_below = np.searchsorted(ordered, values.numpy(), side="right")
    rank = torch.from_numpy((below + at_or_below) / (2.0 * len(ordered))).float()
    return rank


def class_two_sided(psu, pred, validation_psu, validation_pred):
    global_mean = validation_psu.mean()
    global_std = validation_psu.std().clamp_min(PSU_FLOOR)
    num_classes = int(max(pred.max(), validation_pred.max())) + 1

    counts = torch.bincount(validation_pred, minlength=num_classes).float()  # (K,)
    sums = torch.bincount(
        validation_pred, weights=validation_psu, minlength=num_classes
    )
    # A class the validation split barely predicts has no usable mean of its own,
    # so each class mean is pulled toward the global mean by CLASS_SHRINKAGE
    # pseudo-observations.
    class_mean = (sums + CLASS_SHRINKAGE * global_mean) / (counts + CLASS_SHRINKAGE)

    z = (psu - class_mean[pred]) / global_std  # (n,)
    score = -z.abs()
    return score


def destination_concentration(pass_argmax, pred, num_classes):
    passes = pass_argmax.shape[0]
    shifted = pass_argmax != pred[None]  # (k, n)
    one_hot = torch.nn.functional.one_hot(pass_argmax, num_classes)  # (k, n, K)
    landed = (one_hot * shifted[..., None]).sum(dim=0)  # (n, K)
    concentration = landed.max(dim=1).values.float() / passes  # (n,)
    return concentration


def transition_typicality(view, validation):
    num_classes = view["probs"].shape[1]
    validation_moves = validation["pass_argmax"]["tm"]  # (k, n_val)
    validation_from = validation["pred"][None].expand_as(validation_moves)  # (k, n_val)
    flat_index = (validation_from * num_classes + validation_moves).flatten()
    counts = torch.bincount(flat_index, minlength=num_classes * num_classes).float()
    counts = counts.view(num_classes, num_classes)  # (K from, K to)

    smoothed = counts + TRANSITION_SMOOTHING
    log_transition = (smoothed / smoothed.sum(dim=1, keepdim=True)).log()  # (K, K)

    moves = view["pass_argmax"]["tm"]  # (k, n)
    origin = view["pred"][None].expand_as(moves)  # (k, n)
    typicality = log_transition[origin, moves].mean(dim=0)  # (n,)
    return typicality


# Permutes every class index the statistics can see and asserts that no per-input
# score moves. A statistic that used the rotation, a target class or any fixed
# class identity would change under a random relabeling.
def relabeling_check(views):
    num_classes = views["validation"]["probs"].shape[1]
    generator = torch.Generator().manual_seed(RELABEL_SEED)
    permutation = torch.randperm(num_classes, generator=generator)  # new = perm[old]
    inverse = torch.argsort(permutation)

    relabeled = {}
    for split, view in views.items():
        relabeled[split] = {
            "probs": view["probs"][:, inverse],  # (n, K), column perm[j] holds old j
            "pred": permutation[view["pred"]],
            "pass_probs": view["pass_probs"],
            "pass_argmax": {
                probe: permutation[moves]
                for probe, moves in view["pass_argmax"].items()
            },
        }
    for split in ("clean_paired", "backdoor"):
        original = all_statistics(views[split], views["validation"])
        permuted = all_statistics(relabeled[split], relabeled["validation"])
        for name, values in original.items():
            same = torch.allclose(values, permuted[name], atol=1e-5, equal_nan=True)
            assert same, f"{name} changes under relabeling, it reads class identity"


def detection_metrics(validation, clean, backdoor, captured):
    metrics = {
        "auroc": auroc(clean, backdoor),
        "auroc_captured": auroc(clean[captured], backdoor[captured]),
    }
    for budget in FPR_BUDGETS:
        threshold = float(np.quantile(validation.numpy(), budget))
        metrics[f"fpr_{budget:.2f}"] = {
            "tpr": float((backdoor < threshold).float().mean()),
            "tpr_captured": float((backdoor[captured] < threshold).float().mean()),
            "realized_clean_fpr": float((clean < threshold).float().mean()),
        }
    return metrics


def auroc(clean, backdoor):
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    scores = np.concatenate([-clean.numpy(), -backdoor.numpy()])
    value = float(roc_auc_score(labels, scores))
    return value


# The pool a defender screens is simulated as clean test images with the model's
# poison rate of them swapped for triggered ones. Only the simulation reads the
# rate, the router reads the mean.
def suspect_pool_shift(scores, folder):
    provenance = load_args_json(CHECKPOINTS_DIR, folder) or {}
    poison_rate = (
        0.0 if "benign" in folder else float(provenance.get("poison_rate") or 0)
    )
    clean = scores["clean"]["psu_tm"]
    backdoor = scores["backdoor"]["psu_tm"]
    swapped = int(poison_rate * len(clean))
    pool = torch.cat([clean[swapped:], backdoor[:swapped]]) if swapped else clean
    shift = float(pool.mean() - scores["validation"]["psu_tm"].mean())
    return shift


def summarize(models):
    summary = {}
    selections = {
        "all_to_all_strict_2pt": lambda m: (
            m["group"] == "all_to_all" and m["strict_2pt"]
        ),
        "all_to_all_strict_5pt": lambda m: (
            m["group"] == "all_to_all" and m["strict_5pt"]
        ),
        "all_to_all_relaxed_2pt": lambda m: (
            m["group"] == "all_to_all" and m["relaxed_2pt"]
        ),
        "all_to_all_side": lambda m: m["group"] == "all_to_all_side",
        "benign": lambda m: m["group"] == "benign",
        "all_to_one": lambda m: m["group"] == "all_to_one",
    }
    for architecture in ("vit", "swin"):
        for name, keep in selections.items():
            chosen = [
                m for m in models if m["architecture"] == architecture and keep(m)
            ]
            for split_name, datasets in (
                ("development", DEVELOPMENT_DATASETS),
                ("confirmation", CONFIRMATION_DATASETS),
                ("all", DEVELOPMENT_DATASETS + CONFIRMATION_DATASETS),
            ):
                members = [m for m in chosen if m["dataset"] in datasets]
                if members:
                    key = f"{architecture}/{name}/{split_name}"
                    summary[key] = group_summary(members)
    return summary


def group_summary(members):
    names = list(STATISTICS) + list(ROUTERS)
    block = {"n": len(members), "folders": [m["folder"] for m in members]}
    for name in names:
        rows = [m for m in members if name in m["metrics"]]
        if not rows:
            continue
        aurocs = [m["metrics"][name]["auroc"] for m in rows]
        entry = {
            "n": len(rows),
            "mean_auroc": statistics.mean(aurocs),
            "mean_auroc_ci": bootstrap_ci(aurocs),
            "min_auroc": min(aurocs),
            "mean_auroc_captured": statistics.mean(
                m["metrics"][name]["auroc_captured"] for m in rows
            ),
        }
        for budget in FPR_BUDGETS:
            key = f"fpr_{budget:.2f}"
            entry[key] = {
                field: statistics.mean(m["metrics"][name][key][field] for m in rows)
                for field in ("tpr", "tpr_captured", "realized_clean_fpr")
            }
        paired = [
            m["metrics"][name]["auroc"] - m["metrics"]["psu_tm"]["auroc"] for m in rows
        ]
        entry["psu_tm_same_models"] = statistics.mean(
            m["metrics"]["psu_tm"]["auroc"] for m in rows
        )
        entry["psu_tm_same_models_tpr"] = {
            f"fpr_{budget:.2f}": statistics.mean(
                m["metrics"]["psu_tm"][f"fpr_{budget:.2f}"]["tpr"] for m in rows
            )
            for budget in FPR_BUDGETS
        }
        entry["gain_over_psu_tm"] = statistics.mean(paired)
        entry["gain_over_psu_tm_ci"] = bootstrap_ci(paired)
        block[name] = entry
    return block


def bootstrap_ci(values):
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    array = np.asarray(values, dtype=float)
    draws = generator.integers(0, len(array), size=(BOOTSTRAP_RESAMPLES, len(array)))
    means = array[draws].mean(axis=1)  # (resamples,)
    interval = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
    return interval


if __name__ == "__main__":
    main()
