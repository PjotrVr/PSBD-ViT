"""Memo P2: is PSBD's statistic an epistemic uncertainty, read off a real ensemble?

Li et al. frame PSBD as model predictive uncertainty and run MC-Dropout in their
pilot study. A real epistemic estimate on our models is the disagreement between
independently trained replicates. 14 clearing cells have 3 training seeds each
(seed 0 unmarked, `_seed_1`, `_seed_2`, same recipe, poison set drawn per seed).
If PSBD measures epistemic uncertainty, the ensemble's own consistency score
separates triggered from clean inputs about as well as PSBD-TM does.

The score mirrors fractional PSU with the replicates in place of the perturbed
passes, so the 2 are the same statistic over 2 different sources of variation:

    phi_ens(x) = 1 - (1/2) * sum over s in {1, 2} of P_c(x; theta_s) / P_c(x; theta_0)
    with c = argmax_i P_i(x; theta_0)

Beside it, the BALD mutual information of the 3 members and the share of the 2
replicates whose top class differs from seed 0's. docs/why-psbd-works-theory.md
shows that hard-vote disagreement of 3 seeds cannot reach 0.9 AUROC (its ceiling
is (1 + 3(1 - accuracy)) / 2), so the continuous statistic is the one judged,
against the confidence ceiling A*(P_c) of the same model and by its Spearman
correlation with PSBD-TM's own per-image statistic on the clean images. Inference
only, 256 paired images per cell from seed 0's PSBD analysis split.

    source .venv/bin/activate
    PYTHONPATH=. python experiments/why_psbd_works/seed_ensemble.py --list
    PYTHONPATH=. python experiments/why_psbd_works/seed_ensemble.py --cells vit_cifar100_blend_0_05
"""

import argparse
import json
import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from cli.compare_detectors import psbd_rate  # noqa: E402
from data.splits import read_checkpoint_metadata  # noqa: E402
from defenses.cache import (  # noqa: E402
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defenses.scores import psu_ratio_from_cache  # noqa: E402
from defenses.decision import RECOMMENDED_PLACEMENT, detection_report  # noqa: E402
from experiments._paths import experiment_result_path  # noqa: E402
from experiments.why_psbd_works.measure import (  # noqa: E402
    CPU_THREADS,
    HEADLINE_QUANTILE,
    PAIR_COUNT,
    QUARANTINED_MODELS,
    SLUG,
    auroc_low_is_positive,
    limit_gpu_memory,
    load_pairs,
    rounded,
    run_passes,
)
from models.backbones import load_checkpoint  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    clearing_cells,
    excluded_folders,
    load_coverage,
)

REPLICATE_SUFFIXES = ("", "_seed_1", "_seed_2")
ASR_BAR = 0.85


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cells", nargs="+", default=None)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu-memory-gb", type=float, default=14.0)
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cells = args.cells or replicated_cells(args)
    if args.list:
        print("\n".join(cells))
        return

    # The login node runs well over 100 threads of other jobs, and PyTorch's
    # default of 1 CPU thread per core oversubscribes it on every small CPU op.
    torch.set_num_threads(CPU_THREADS)
    device = torch.device("cuda", torch.cuda.current_device())
    limit_gpu_memory(args.gpu_memory_gb, device)
    for cell in cells:
        out_path = experiment_result_path(
            SLUG, f"seed_ensemble__{cell}.json", args.output_root
        )
        if os.path.exists(out_path):
            print(f"[skip] {cell}", flush=True)
            continue
        record = measure_cell(cell, args, device)
        with open(out_path, "w") as handle:
            json.dump(record, handle)
        print(f"[ok] {cell}", flush=True)


def measure_cell(cell, args, device):
    loader_args = types.SimpleNamespace(
        raw_data_dir=args.raw_data_dir, pairs=PAIR_COUNT, batch_size=args.batch_size
    )
    seed0_path = os.path.join(args.checkpoints_dir, cell, "attack_result.pt")
    pairs, validation = load_pairs(seed0_path, None, loader_args, device)
    pairs["validation"] = validation

    member_logits = []
    for suffix in REPLICATE_SUFFIXES:
        path = os.path.join(args.checkpoints_dir, cell + suffix, "attack_result.pt")
        architecture = read_checkpoint_metadata(path)["architecture"]
        model = load_checkpoint(architecture, path, device).eval()
        member_logits.append(
            {
                split: run_passes(
                    model, pairs[split], args.batch_size, device, passes=None
                )["logits"]
                for split in ("clean", "triggered", "validation")
            }
        )
        del model
        torch.cuda.empty_cache()

    hit = member_logits[0]["triggered"].argmax(dim=1) == pairs["targets"]  # (n,)
    scores = {
        split: ensemble_scores([m[split] for m in member_logits])
        for split in ("clean", "triggered", "validation")
    }
    cached_tm = cached_psbd_tm_scores(args.results_dir, cell, len(pairs["targets"]))
    record = {
        "cell": cell,
        "pairs": int(len(pairs["targets"])),
        "hit_pairs": int(hit.sum()),
        "member_triggered_asr": [
            float((m["triggered"].argmax(dim=1) == pairs["targets"]).float().mean())
            for m in member_logits
        ],
        "psbd_tm_cached_auroc": cached_psbd_tm_auroc(args.results_dir, cell),
        # docs/why-psbd-works-theory.md, P2: a real epistemic reading should also
        # order the clean images the way PSBD-TM does.
        "clean_spearman_with_psbd_tm": spearman(
            scores["clean"]["ensemble_psu_ratio"], cached_tm["clean"]
        ),
        "triggered_spearman_with_psbd_tm": spearman(
            scores["triggered"]["ensemble_psu_ratio"], cached_tm["triggered"]
        ),
    }
    for name in ("ensemble_psu_ratio", "disagreement", "mutual_information"):
        paired = detection_report(
            scores["validation"][name],
            scores["clean"][name],
            scores["triggered"][name],
            HEADLINE_QUANTILE,
        )
        record[name] = {
            "auroc_paired_pipeline_form": paired["auroc"],
            "auroc_hit_only": auroc_low_is_positive(
                scores["triggered"][name][hit], scores["clean"][name]
            ),
            "clean": rounded(scores["clean"][name]),
            "triggered": rounded(scores["triggered"][name]),
        }
    return record


def ensemble_scores(logits_per_member):
    probs = torch.stack([F.softmax(z, dim=1) for z in logits_per_member])  # (3, n, C)
    labels = probs[0].argmax(dim=1)  # (n,)
    own = probs.gather(2, labels.view(1, -1, 1).expand(3, -1, 1))[:, :, 0]  # (3, n)
    mean_probs = probs.mean(dim=0)  # (n, C)
    predictive = -(mean_probs * mean_probs.clamp_min(1e-12).log()).sum(dim=1)
    per_member = -(probs * probs.clamp_min(1e-12).log()).sum(dim=2)  # (3, n)
    others_disagree = (probs[1:].argmax(dim=2) != labels[None]).float().mean(dim=0)

    scores = {
        "ensemble_psu_ratio": 1.0 - own[1:].mean(dim=0) / own[0],  # (n,)
        "disagreement": others_disagree,  # (n,)
        "mutual_information": predictive - per_member.mean(dim=0),  # (n,)
    }
    return scores


# PSBD-TM's per-image statistic from the sweep's own cache, at its adaptive rate,
# for the same clean twins and triggered rows the ensemble scored.
def cached_psbd_tm_scores(results_dir, cell, count):
    with open(os.path.join(results_dir, cell, "psbd_metrics.json")) as handle:
        block = json.load(handle)["placements"][RECOMMENDED_PLACEMENT]
    rate = psbd_rate(block, "adaptive")
    psbd_dir = os.path.join(results_dir, cell, "psbd")
    manifest = read_split_manifest(psbd_dir)
    scores = {}
    for split in ("clean", "backdoor"):
        probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
        per_pass, _ = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, RECOMMENDED_PLACEMENT, rate, split)
        )
        scores[split] = psu_ratio_from_cache(probs, labels, per_pass)  # (n,)
    row_of = {o: r for r, o in enumerate(manifest["analysis_clean_indices"])}
    clean_rows = [row_of[o] for o in manifest["analysis_backdoor_indices"][:count]]
    cached = {
        "clean": scores["clean"][clean_rows],
        "triggered": scores["backdoor"][:count],
    }
    return cached


def spearman(first, second):
    a = torch.as_tensor(first).argsort().argsort().double()
    b = torch.as_tensor(second).argsort().argsort().double()
    value = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    return value


def cached_psbd_tm_auroc(results_dir, cell):
    with open(os.path.join(results_dir, cell, "psbd_metrics.json")) as handle:
        block = json.load(handle)["placements"][RECOMMENDED_PLACEMENT]
    rate = psbd_rate(block, "adaptive")
    row = next(r for r in block["rates"] if r["rate"] == rate)
    auroc = row["detection_psu_ratio"]["q0.25"]["auroc"]
    return auroc


def replicated_cells(args):
    excluded = excluded_folders(args.results_dir)
    quarantined = {folder for folder, _ in QUARANTINED_MODELS}
    cells = []
    for cell in clearing_cells(load_coverage(args.results_dir)):
        folder = cell["folder_name"]
        if folder in excluded or folder in quarantined or cell["attack"] == "sig":
            continue
        replicas = [folder + suffix for suffix in REPLICATE_SUFFIXES]
        if all(replica_clears(args.checkpoints_dir, r) for r in replicas):
            cells.append(folder)
    return cells


def replica_clears(checkpoints_dir, folder):
    path = os.path.join(checkpoints_dir, folder, "args.json")
    if not os.path.exists(path):
        return False
    with open(path) as handle:
        asr = json.load(handle).get("asr")
    clears = asr is not None and asr >= ASR_BAR
    return clears


if __name__ == "__main__":
    main()
