"""Why PSBD's statistic separates triggered from clean inputs on ViT and Swin, for any probe.

PSBD scores an input by how little the probability of the model's own answer drops
under k perturbed passes, and a small drop reads as poisoned. The statistic works
on ViT with every probe family the project swept, not only PSBD-TM. This script
measures 6 candidate explanations on the same 256 paired clean and triggered images
per model, inference only. README.md states each hypothesis and the verdict.

    margin        logit margin of the own answer, unperturbed and under each operator
    direction     backdoor direction component of the final feature against the
                  perturbation-induced change of that feature
    redundancy    prediction survival as a fixed random share of tokens is removed at
                  the attention input of every block
    low_dim       spectrum of the triggered-minus-clean feature change, token and rank
                  concentration of the margin's Jacobian on the residual stream, and
                  concentration of expected-gradient input attributions
    flatness      finite differences of the margin and log probability along random
                  directions in the residual stream and in pixel space
    neuron_bias   share of shifted clean predictions that land on the target

    source .venv/bin/activate
    PYTHONPATH=. python experiments/why_psbd_works/measure.py --models vit_cifar100_blend_0_05
    PYTHONPATH=. python experiments/why_psbd_works/summarize.py
"""

import argparse
import faulthandler
import json
import math
import os
import signal
import sys
import time
import zlib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from lightning import seed_everything  # noqa: E402

from analysis.attribution import expected_gradients  # noqa: E402
from analysis.direction import backdoor_direction  # noqa: E402
from analysis.features import transformer_blocks  # noqa: E402
from attacks import apply_config_overrides, build_attack, default_config  # noqa: E402
from cli.compare_detectors import psbd_rate  # noqa: E402
from data.registry import DATASET_REGISTRY  # noqa: E402
from data.splits import (  # noqa: E402
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from defenses.cache import baseline_path, load_baseline, read_split_manifest  # noqa: E402
from defenses.decision import detection_report  # noqa: E402
from defenses.inference import forward_logits  # noqa: E402
from defenses.operators import (  # noqa: E402
    GaussianNoise,
    TokenMask,
    TokenSubstitute,
    channel_mask,
)
from experiments._paths import experiment_result_path  # noqa: E402
from models.backbones import load_checkpoint, network_core  # noqa: E402
from models.positions import DROPOUT_CONFIGS, plug_dropout, unplug_dropout  # noqa: E402
from scripts.paper._common import excluded_folders  # noqa: E402
from data.splits import load_clean_test_base  # noqa: E402
import torchvision.transforms.v2 as transforms_v2  # noqa: E402

SLUG = "why_psbd_works"
PAIR_COUNT = 256
FORWARD_PASSES = 10
BATCH_SIZE = 128
GPU_MEMORY_GB = 14.0
CPU_THREADS = 4
PASS_SEED = 0
DRAW_SEED = 0

# Each ViT model is listed with its attack category. A benign model has no trigger
# of its own and is probed with 1 patch and 1 global trigger at target 0, the
# way the benign references in results/ were swept.
VIT_MODELS = (
    ("vit_cifar100_badnet_a2o_0_05", None),
    ("vit_tiny_badnet_a2o_0_05", None),
    ("vit_cifar10_tact_0_01", None),
    ("vit_cifar10_tact_0_05", None),
    ("vit_gtsrb_tact_0_05", None),
    ("vit_cifar100_blend_0_05", None),
    ("vit_tiny_blend_0_05", None),
    ("vit_cifar100_lf_0_05", None),
    ("vit_tiny_lf_0_05", None),
    ("vit_tiny_wanet_0_1", None),
    ("vit_gtsrb_wanet_0_1", None),
    ("vit_cifar10_wanet_0_1", None),
    ("vit_cifar10_wanet_0_05", None),
    ("vit_tiny_wanet_0_05", None),
    ("vit_cifar100_bpp_0_05", None),
    ("vit_tiny_bpp_0_05", None),
    ("vit_cifar100_benign", "badnet_a2o"),
    ("vit_cifar100_benign", "blend"),
    ("vit_tiny_benign", "badnet_a2o"),
    ("vit_tiny_benign", "blend"),
)
SWIN_MODELS = (
    ("swin_cifar100_badnet_a2o_0_05", None),
    ("swin_tiny_badnet_a2o_0_05", None),
    ("swin_cifar10_tact_0_01", None),
    ("swin_cifar10_tact_0_05", None),
    ("swin_cifar100_blend_0_05", None),
    ("swin_tiny_blend_0_05", None),
    ("swin_cifar100_lf_0_05", None),
    ("swin_tiny_lf_0_05", None),
    ("swin_cifar100_wanet_0_1", None),
    ("swin_tiny_wanet_0_1", None),
    ("swin_cifar100_bpp_0_05", None),
    ("swin_tiny_bpp_0_05", None),
    ("swin_cifar100_benign", "badnet_a2o"),
    ("swin_cifar100_benign", "blend"),
    ("swin_tiny_benign", "badnet_a2o"),
    ("swin_tiny_benign", "blend"),
)
# Quarantined on 2026-09-29 (docs/audits/2026-09-29-experiment-audit.md). SIG
# checkpoints trained before 2026-09-09 saw amplitude 0.1 while attacks/sig.py now
# stamps 0.157, so their rebuilt triggered set is not the trigger they learned.
# They run only with --include-quarantined once the provenance is fixed. SIG
# clears the bar on 1 ViT model of the 4 paper datasets, and the SVHN and EuroSAT
# models sit on datasets the paper declares out, so the category is not read off
# a single model.
QUARANTINED_MODELS = (
    ("vit_cifar10_sig_0_1", None),
    ("vit_svhn_sig_0_1_tl1", None),
    ("vit_eurosat_sig_0_1", None),
    ("swin_cifar10_sig_0_1", None),
    ("swin_cifar10_sig_0_05", None),
)
CATEGORY = {
    "badnet_a2o": "patch",
    "tact": "patch",
    "blend": "blend",
    "lf": "blend",
    "sig": "frequency",
    "wanet": "warp",
    "bpp": "quantization",
    "benign": "benign",
}
PATCH_ATTACKS = ("badnet_a2o", "tact")

# The 4 operator families at the attention input and dropout on the residual
# stream, each at the rate its own placement's adaptive rule chose. The first
# entry is PSBD-TM and the last is PSBD-RD.
OPERATOR_SPECS = {
    "token_mask": (
        "before_attention_norm_token_mask",
        ("before_attention_norm",),
        TokenMask,
    ),
    "dropout": ("before_attention_norm", ("before_attention_norm",), nn.Dropout),
    "channel_mask": (
        "before_attention_norm_channel_mask",
        ("before_attention_norm",),
        channel_mask,
    ),
    "gaussian": (
        "before_attention_norm_gaussian",
        ("before_attention_norm",),
        GaussianNoise,
    ),
    "residual_dropout": ("post_residual", DROPOUT_CONFIGS["post_residual"], nn.Dropout),
    # A probe the sweeps never cached, rated by the fallback rule. It is token_mask
    # without the off-manifold zero token (memo L22). rademacher (memo L2) was
    # dropped: docs/why-psbd-works-theory.md shows it cannot differ from gaussian
    # in 151k dimensions under any account, and the curvature test is the
    # small-noise Gaussian run of measure_small_noise instead.
    "token_substitute": (
        "before_attention_norm_token_substitute",
        ("before_attention_norm",),
        TokenSubstitute,
    ),
}
# Used only when a placement's psbd_metrics.json carries no adaptive rate, to pick
# the smallest ladder rate whose clean shift ratio reaches 0.8 on held-out images.
FALLBACK_LADDERS = {
    "token_mask": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    "dropout": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    "channel_mask": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    "gaussian": (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
    "residual_dropout": (0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3),
    "token_substitute": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
}
ADAPTIVE_TARGET = 0.8
FALLBACK_IMAGES = 256
FALLBACK_PASSES = 3

# The 3 fractions of why_token_masking_works section D, so the 2 read alike and
# the theory note's threshold at 30% visible applies directly.
KEEP_FRACTIONS = (1.0, 0.6, 0.3, 0.1)
# 20 independent patterns per fraction, seeded per model from its folder name so
# 2 models never share their patterns (the audit's point against 3 shared draws).
SUBSET_DRAWS = 20
# The keep pattern is drawn on ViT's 14 by 14 patch grid, 16 pixels a cell at the
# model's 224 input, and Swin's stages see it resized to their own grid.
PATCH_GRID = 14

# Blocks whose incoming residual stream is probed, 1-indexed. ViT's 12 blocks get
# a third of the depth each. Swin-S has stages of 2, 2, 18 and 2 blocks. Its 3
# picks are the first and middle block of stage 3 and the first of stage 4.
STREAM_BLOCKS = {"vit": (4, 8, 12), "swin": (5, 13, 23)}
JACOBIAN_PAIRS = 64
TOP_TOKENS = 4
ATTRIBUTION_PAIRS = 32
ATTRIBUTION_SAMPLES = 32
ATTRIBUTION_TOP_SHARE = 0.02
# 64 pairs and 4 directions per image, and 2 stream sites (below), after the
# first full run showed float32 finite differences to be the slowest section.
FLAT_PAIRS = 64
FLAT_DIRECTIONS = 4
FLAT_EPSILONS = (0.05, 0.2)
HEADLINE_QUANTILE = 0.25
KNN_NEIGHBOURS = 10
# Clean images of another dataset, normalized as the model's own, are the out of
# distribution control. The pairing only asks for a different image source.
FOREIGN_DATASET = {
    "cifar100": "tiny",
    "tiny": "cifar100",
    "cifar10": "gtsrb",
    "gtsrb": "cifar10",
    "svhn": "cifar10",
    "eurosat": "cifar10",
}
# 1% of each block's MLP hidden width, 30 of 3072 on ViT, the scale at which
# experiments/backdoor_neurons found the top TAC dimensions stand out.
NEURON_SHARE = 0.01
NEURON_DRAWS = 2
NEURON_OPERATORS = ("token_mask", "residual_dropout")
LATE_BLOCK_COUNT = 4
# Memo L7 names 30 units per block in the last 4 blocks.
LATE_UNITS = 30
# Visible token fractions for the smallest sufficient token set, descending, down
# to 1 token of 196.
SUFFICIENCY_FRACTIONS = (
    0.8,
    0.6,
    0.4,
    0.3,
    0.2,
    0.15,
    0.1,
    0.07,
    0.05,
    0.03,
    0.02,
    0.01,
    0.005,
)
SPREAD_PAIRS = 64
# docs/why-psbd-works-theory.md, L2 and P4: Gaussian at the attention input at
# rates of 0.1 and below, float32, 20 passes, so that a second-order account can
# be read off the ratio phi(2r)/phi(r), which it predicts to be about 4.
SMALL_NOISE_RATES = (0.05, 0.1)
SMALL_NOISE_PASSES = 20
SPREAD_BLOCKS = {"vit": (4, 5, 6, 7, 8), "swin": (5, 7, 9, 11, 13)}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="folder or folder:probe_attack entries, default every listed model",
    )
    parser.add_argument("--architecture", choices=("vit", "swin", "all"), default="all")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    # Separate from --results-dir so a smoke run reads the real psbd_metrics.json
    # and writes somewhere disposable.
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--pairs", type=int, default=PAIR_COUNT)
    parser.add_argument("--passes", type=int, default=FORWARD_PASSES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gpu-memory-gb", type=float, default=GPU_MEMORY_GB)
    parser.add_argument("--list", action="store_true", help="print the model list")
    parser.add_argument("--include-quarantined", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    # kill -USR1 <pid> prints every thread's Python stack to the log, to see
    # which section a long run is in without stopping it.
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    args = parse_args()
    entries = selected_entries(args)
    if args.list:
        print("\n".join(entry_name(folder, probe) for folder, probe in entries))
        return

    excluded = excluded_folders(args.results_dir)
    # The login node runs well over 100 threads of other jobs, and PyTorch's
    # default of 1 CPU thread per core oversubscribes it on every small CPU op.
    torch.set_num_threads(CPU_THREADS)
    device = torch.device("cuda", torch.cuda.current_device())
    limit_gpu_memory(args.gpu_memory_gb, device)
    quarantined = {folder for folder, _ in QUARANTINED_MODELS}
    for folder, probe in entries:
        if folder in excluded:
            print(f"[excluded by the ledger] {folder}", flush=True)
            continue
        if folder in quarantined and not args.include_quarantined:
            print(f"[quarantined] {folder}", flush=True)
            continue
        out_path = experiment_result_path(
            SLUG, f"{entry_name(folder, probe)}.json", args.output_root
        )
        if os.path.exists(out_path) and not args.force:
            print(f"[skip] {entry_name(folder, probe)}", flush=True)
            continue
        started = time.time()
        record = measure_model(folder, probe, args, device)
        record["seconds"] = round(time.time() - started, 1)
        record["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
        write_json(out_path, record)
        print(f"[ok] {entry_name(folder, probe)} {record['seconds']}s", flush=True)


def measure_model(folder, probe, args, device):
    checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)
    architecture = metadata["architecture"]
    model = load_checkpoint(architecture, checkpoint_path, device).eval()
    pairs, validation = load_pairs(checkpoint_path, probe, args, device)
    pairs["validation"] = validation
    pairs["foreign"] = load_foreign_images(metadata["dataset"], args, device)
    attack = probe or metadata["attack"]
    hooks_before = hook_count(model)

    baseline = {
        split: run_passes(model, pairs[split], args.batch_size, device, passes=None)
        for split in SPLITS
    }
    pairs["clean_base"] = baseline["clean"]["logits"].argmax(dim=1)  # (n,)
    pairs["triggered_base"] = baseline["triggered"]["logits"].argmax(dim=1)  # (n,)
    # On a backdoored model only triggered images that reach the target carry a
    # target decision to keep. A benign model has none, so every pair counts.
    if probe is None:
        pairs["hit"] = pairs["triggered_base"] == pairs["targets"]  # (n,)
    else:
        pairs["hit"] = torch.ones_like(pairs["targets"], dtype=torch.bool)  # (n,)

    rates = operator_rates(model, architecture, folder, validation, args, device)
    geometry = feature_geometry(model, baseline, pairs)

    record = {
        "folder": folder,
        "probe_attack": probe,
        "architecture": architecture,
        "attack": attack,
        "category": CATEGORY["benign" if probe else metadata["attack"]],
        "dataset": metadata["dataset"],
        "poison_rate": metadata.get("poison_rate"),
        "target_label": int(pairs["targets"][0]),
        "num_classes": int(baseline["clean"]["logits"].shape[1]),
        "pairs": int(len(pairs["targets"])),
        "hit_pairs": int(pairs["hit"].sum()),
        "passes": args.passes,
        "rates": rates,
        "baseline": baseline_readout(baseline, pairs),
        "direction_geometry": geometry["summary"],
        "ood": ood_readout(baseline, pairs),
        "foreign_dataset": FOREIGN_DATASET[metadata["dataset"]],
    }

    clock = {"operators": time.time()}
    record["operators"] = {}
    for name, (_placement, positions, factory) in OPERATOR_SPECS.items():
        rate = rates[name]["rate"]
        perturbed = {
            split: perturbed_passes(
                model,
                architecture,
                positions,
                factory,
                rate,
                pairs[split],
                args,
                device,
            )
            for split in SPLITS
        }
        record["operators"][name] = operator_readout(
            baseline, perturbed, pairs, geometry, rate
        )

    clock["redundancy"] = time.time()
    record["redundancy"] = measure_redundancy(
        model, architecture, metadata, attack, pairs, args, device, model_seed(folder)
    )
    clock["low_dim"] = time.time()
    record["low_dim"] = {
        "difference_spectrum": difference_spectrum(baseline, pairs),
        "jacobian": measure_jacobian(
            model, architecture, metadata, attack, pairs, device
        ),
        "attribution": measure_attribution(model, metadata, attack, pairs, device),
    }
    clock["flatness"] = time.time()
    record["flatness"] = measure_flatness(model, architecture, pairs, device)
    clock["neurons"] = time.time()
    record["neurons"] = measure_neurons(model, architecture, rates, pairs, args, device)
    clock["sufficiency"] = time.time()
    record["sufficiency"] = measure_sufficiency(
        model, architecture, pairs, args, device, model_seed(folder)
    )
    clock["small_noise"] = time.time()
    record["small_noise"] = measure_small_noise(
        model, architecture, pairs, args, device
    )
    clock["token_spread"] = time.time()
    record["token_spread"] = measure_token_spread(model, architecture, pairs, device)
    record["section_seconds"] = section_seconds(clock)
    record["cache_agreement"] = cache_agreement(folder, probe, baseline, pairs, args)
    record["sanity"] = restoration_check(
        model, pairs, baseline, hooks_before, args, device
    )
    return record


def perturbed_passes(
    model, architecture, positions, factory, rate, images, args, device
):
    handles = plug_dropout(
        model, architecture, positions, {p: factory for p in positions}, rate
    )
    try:
        seed_everything(PASS_SEED, verbose=False)
        result = run_passes(model, images, args.batch_size, device, passes=args.passes)
    finally:
        unplug_dropout(handles)
    return result


def operator_readout(baseline, perturbed, pairs, geometry, rate):
    readout = {"rate": rate}
    for split in ("clean", "triggered"):
        base_logits = baseline[split]["logits"]  # (n, classes)
        pass_logits = perturbed[split]["logits"]  # (k, n, classes)
        labels = base_logits.argmax(dim=1)  # (n,)
        stats = per_image_statistics(base_logits, pass_logits, labels)
        stats.update(
            feature_statistics(
                baseline[split]["features"],
                perturbed[split]["features"],
                labels,
                geometry,
            )
        )
        readout[split] = stats

    for split in ("validation", "foreign"):
        base_logits = baseline[split]["logits"]  # (m, classes)
        stats = per_image_statistics(
            base_logits, perturbed[split]["logits"], base_logits.argmax(dim=1)
        )
        readout[split] = {key: stats[key] for key in ("psu_ratio", "kept")}
        # A foreign image flagged as poisoned could be 1 the model sends to the
        # target, unperturbed or under the operator, so both shares are kept.
        target = int(pairs["targets"][0])
        readout[split]["on_target_unperturbed"] = float(
            (base_logits.argmax(dim=1) == target).float().mean()
        )
        readout[split]["on_target_perturbed"] = float(
            (perturbed[split]["logits"].argmax(dim=2) == target).float().mean()
        )
    for split in SPLITS:
        base_logits = baseline[split]["logits"]  # (n, classes)
        readout[split].update(
            pass_uncertainty(perturbed[split]["logits"], base_logits.argmax(dim=1))
        )

    hit = pairs["hit"].cpu()
    clean_psu = torch.tensor(readout["clean"]["psu_ratio"])
    triggered_psu = torch.tensor(readout["triggered"]["psu_ratio"])[hit]
    # The pipeline's form: every triggered row against its own clean twin, through
    # defenses.decision.detection_report, which sanity.py holds to the cache.
    # The hit-only form conditions on triggered images the backdoor captured and
    # is the one every other readout here uses, so both are kept.
    paired = detection_report(
        torch.tensor(readout["validation"]["psu_ratio"]),
        clean_psu,
        torch.tensor(readout["triggered"]["psu_ratio"]),
        HEADLINE_QUANTILE,
    )
    readout["auroc_paired_pipeline_form"] = paired["auroc"]
    readout["auroc_hit_only"] = auroc_low_is_positive(triggered_psu, clean_psu)
    # The MC-dropout reading: an uncertainty over the same passes that ignores
    # which class the unperturbed model chose. Low uncertainty flags poisoned.
    for quantity in ("predictive_entropy", "mutual_information", "own_prob_std"):
        readout[f"auroc_{quantity}"] = auroc_low_is_positive(
            torch.tensor(readout["triggered"][quantity])[hit],
            torch.tensor(readout["clean"][quantity]),
        )
    readout["flagged"] = flagged_shares(readout, hit)

    # Memo L10: remove the target column and renormalize, so a clean image whose
    # probability flowed to the target no longer counts that flow as a drop.
    # Triggered images keep their own statistic, their class being the target.
    target = int(pairs["targets"][0])
    clean_logits = baseline["clean"]["logits"]
    clean_labels = clean_logits.argmax(dim=1)  # (n,)
    redistributed = psu_ratio_without_class(
        clean_logits, perturbed["clean"]["logits"], clean_labels, target
    )  # (n,)
    # docs/why-psbd-works-theory.md, L10: only clean images whose own class is
    # not the target have a target gain to redistribute.
    clean_without_target = torch.where(clean_labels != target, redistributed, clean_psu)
    readout["clean"]["psu_ratio_without_target"] = rounded(clean_without_target)
    readout["auroc_hit_only_without_target"] = auroc_low_is_positive(
        triggered_psu, clean_without_target
    )
    readout["auroc_paired_without_target"] = detection_report(
        torch.tensor(readout["validation"]["psu_ratio"]),
        clean_without_target,
        torch.tensor(readout["triggered"]["psu_ratio"]),
        HEADLINE_QUANTILE,
    )["auroc"]
    readout["clean_shift_to_target"] = shift_to_target_share(
        baseline["clean"]["logits"],
        perturbed["clean"]["logits"],
        int(pairs["targets"][0]),
    )
    return readout


# Every per-image quantity the margin and direction tests read, as plain lists so
# the JSON holds the distributions and not only their means.
def per_image_statistics(base_logits, pass_logits, labels):
    base_probs = F.softmax(base_logits, dim=1)  # (n, classes)
    pass_probs = F.softmax(pass_logits, dim=2)  # (k, n, classes)
    index = labels.view(1, -1, 1).expand(pass_probs.shape[0], -1, 1)  # (k, n, 1)
    tracked_base = base_probs.gather(1, labels.view(-1, 1)).squeeze(1)  # (n,)
    tracked_passes = pass_probs.gather(2, index).squeeze(2)  # (k, n)
    psu_ratio = 1.0 - tracked_passes.mean(dim=0) / tracked_base  # (n,)

    base_margin = own_class_margin(base_logits, labels)  # (n,)
    pass_margin = own_class_margin(pass_logits, labels)  # (k, n)
    kept = (pass_logits.argmax(dim=2) == labels[None]).float().mean(dim=0)  # (n,)

    stats = {
        "psu_ratio": rounded(psu_ratio),
        "kept": rounded(kept),
        "margin": rounded(base_margin),
        "margin_perturbed": rounded(pass_margin.mean(dim=0)),
        "margin_drop": rounded(base_margin - pass_margin.mean(dim=0)),
        "margin_drop_std": rounded(pass_margin.std(dim=0)),
        "margin_retention": rounded(pass_margin.mean(dim=0) / base_margin),
    }
    return stats


def feature_statistics(base_features, pass_features, labels, geometry):
    clean_mean = geometry["clean_mean"]  # (dim,)
    backdoor_unit = geometry["backdoor_unit"]  # (dim,)
    class_units = geometry["class_units"][labels]  # (n, dim)

    deviation = base_features - clean_mean  # (n, dim)
    change = pass_features - base_features[None]  # (k, n, dim)
    pass_deviation = pass_features - clean_mean  # (k, n, dim)

    backdoor_component = deviation @ backdoor_unit  # (n,)
    class_component = (deviation * class_units).sum(dim=1)  # (n,)
    change_norm = change.norm(dim=2).pow(2).mean(dim=0).sqrt()  # (n,)
    change_on_backdoor = (change @ backdoor_unit).pow(2).mean(dim=0).sqrt()  # (n,)
    change_on_class = (change * class_units[None]).sum(dim=2).pow(2).mean(dim=0).sqrt()
    backdoor_kept = (pass_deviation @ backdoor_unit).mean(dim=0)  # (n,)
    class_kept = (pass_deviation * class_units[None]).sum(dim=2).mean(dim=0)  # (n,)

    stats = {
        "deviation_norm": rounded(deviation.norm(dim=1)),
        "backdoor_component": rounded(backdoor_component),
        "backdoor_share": rounded(
            backdoor_component.pow(2) / deviation.norm(dim=1).pow(2)
        ),
        "class_component": rounded(class_component),
        "change_norm": rounded(change_norm),
        "relative_change": rounded(change_norm / deviation.norm(dim=1)),
        "backdoor_retention": rounded(backdoor_kept / backdoor_component),
        "class_retention": rounded(class_kept / class_component),
        "backdoor_signal_to_perturbation": rounded(
            backdoor_component / change_on_backdoor
        ),
        "class_signal_to_perturbation": rounded(class_component / change_on_class),
        "backdoor_signal_to_total_change": rounded(backdoor_component / change_norm),
        "class_signal_to_total_change": rounded(class_component / change_norm),
    }
    return stats


# The backdoor direction is the mean paired difference of the final feature the
# classifier head reads, over pairs whose triggered image reaches the target. The
# class direction of class c is the head's row for c minus the mean row, which is
# the direction along which that logit rises against the average of the others.
def feature_geometry(model, baseline, pairs):
    hit = pairs["hit"].cpu()
    clean_features = baseline["clean"]["features"]  # (n, dim)
    triggered_features = baseline["triggered"]["features"]  # (n, dim)
    direction = backdoor_direction(clean_features[hit], triggered_features[hit])
    backdoor_unit = direction / direction.norm().clamp_min(1e-8)  # (dim,)

    head = classifier_head(model)
    weight = head.weight.detach().float().cpu()  # (classes, dim)
    centered_rows = weight - weight.mean(dim=0, keepdim=True)  # (classes, dim)
    class_units = centered_rows / centered_rows.norm(dim=1, keepdim=True)

    target = int(pairs["targets"][0])
    summary = {
        "backdoor_direction_norm": float(direction.norm()),
        "clean_feature_spread": float(
            (clean_features - clean_features.mean(dim=0)).norm(dim=1).mean()
        ),
        "cosine_backdoor_target_readout": float(backdoor_unit @ class_units[target]),
    }
    geometry = {
        "clean_mean": clean_features.mean(dim=0),  # (dim,)
        "backdoor_unit": backdoor_unit,
        "class_units": class_units,
        "summary": summary,
    }
    return geometry


def baseline_readout(baseline, pairs):
    clean_logits = baseline["clean"]["logits"]  # (n, classes)
    triggered_logits = baseline["triggered"]["logits"]  # (n, classes)
    hit = pairs["hit"].cpu()
    targets = pairs["targets"].cpu()
    clean_labels = pairs["clean_labels"].cpu()
    clean_margin = own_class_margin(clean_logits, clean_logits.argmax(dim=1))
    triggered_margin = own_class_margin(
        triggered_logits, triggered_logits.argmax(dim=1)
    )
    clean_confidence = F.softmax(clean_logits, dim=1).max(dim=1).values  # (n,)
    triggered_confidence = F.softmax(triggered_logits, dim=1).max(dim=1).values

    readout = {
        "clean_accuracy": float(
            (clean_logits.argmax(dim=1) == clean_labels).float().mean()
        ),
        "clean_on_target": float(
            (clean_logits.argmax(dim=1) == targets).float().mean()
        ),
        "triggered_asr": float(
            (triggered_logits.argmax(dim=1) == targets).float().mean()
        ),
        "auroc_triggered_larger_margin": auroc_low_is_positive(
            -triggered_margin[hit], -clean_margin
        ),
        "auroc_triggered_larger_confidence": auroc_low_is_positive(
            -triggered_confidence[hit], -clean_confidence
        ),
        "clean_margin": rounded(clean_margin),
        "triggered_margin": rounded(triggered_margin),
        "hit": hit.int().tolist(),
    }
    return readout


# For a patch trigger a second set of the same patterns has the trigger's tokens
# forced visible. The gap between the 2 separates redundancy (the evidence is in
# many tokens) from luck (the few trigger tokens were rarely hidden).
def measure_redundancy(
    model, architecture, metadata, attack, pairs, args, device, seed
):
    reference = subset_readout(
        pairs, pairs["clean_base"][None], pairs["triggered_base"][None]
    )
    trigger = None
    if architecture == "vit" and attack in PATCH_ATTACKS:
        trigger = patch_trigger_tokens(metadata, attack) - 1  # (trigger_count,)
    result = {"1.0": {"mean": reference, "draws": [reference]}}
    for fraction in KEEP_FRACTIONS[1:]:
        draws, trigger_kept_draws = [], []
        for draw in range(SUBSET_DRAWS):
            keep_grid = random_keep_grid(fraction, seed + draw)  # (14, 14)
            reading = subset_predictions_readout(
                model, architecture, keep_grid, pairs, args, device
            )
            if trigger is not None:
                reading["trigger_visible"] = int(keep_grid.flatten()[trigger].sum())
                forced = keep_grid.flatten().clone()
                forced[trigger] = True
                trigger_kept_draws.append(
                    subset_predictions_readout(
                        model,
                        architecture,
                        forced.view(PATCH_GRID, PATCH_GRID),
                        pairs,
                        args,
                        device,
                    )
                )
            draws.append(reading)
        result[str(fraction)] = {"mean": mean_readout(draws), "draws": draws}
        if trigger_kept_draws:
            result[str(fraction)]["trigger_forced_visible"] = mean_readout(
                trigger_kept_draws
            )
    return result


def subset_predictions_readout(model, architecture, keep_grid, pairs, args, device):
    handles = plug_dropout(
        model,
        architecture,
        ("before_attention_norm",),
        {"before_attention_norm": lambda _rate: SubsetTokenMask(keep_grid)},
        0.0,
    )
    try:
        predictions = {
            split: run_passes(
                model, pairs[split], args.batch_size, device, passes=None
            )["logits"].argmax(dim=1)
            for split in ("clean", "triggered")
        }
    finally:
        unplug_dropout(handles)
    reading = subset_readout(
        pairs, predictions["clean"][None], predictions["triggered"][None]
    )
    return reading


def model_seed(folder):
    seed = zlib.crc32(folder.encode()) % 1_000_000
    return seed


# A fixed pattern of 16 pixel cells hidden at the attention input of every block.
# On ViT a hidden cell is 1 patch token, zeroed before ln_1 so it writes nothing
# into any other token while its own residual entry is carried along. On Swin the
# pattern is resized to each stage's grid, nearest neighbour when the grid is
# finer and a majority vote over the 2 by 2 cell when it is coarser. No survivor
# rescale, since the LayerNorm that follows removes each token's scale anyway.
# keep_grid is 1 (14, 14) pattern shared by the batch or a (batch, 14, 14) stack
# with 1 pattern per image, which the sufficiency search sets before each batch.
class SubsetTokenMask(nn.Module):
    def __init__(self, keep_grid):
        super().__init__()
        self.keep_grid = keep_grid

    def forward(self, x):
        grid = self.keep_grid.float().view(-1, 1, PATCH_GRID, PATCH_GRID).to(x.device)
        if x.dim() == 3:
            patches = grid.flatten(1)  # (patterns, 196)
            cls = torch.ones(patches.shape[0], 1, device=x.device)
            keep = torch.cat([cls, patches], dim=1).to(x.dtype)  # (patterns, 197)
            assert keep.shape[1] == x.shape[1], f"{x.shape[1]} tokens, not 197"
            masked = x * keep[:, :, None]  # (batch, 197, channels)
            return masked

        _, height, width, _ = x.shape  # (batch, height, width, channels)
        if height >= PATCH_GRID:
            resized = F.interpolate(grid, size=(height, width), mode="nearest")
        else:
            resized = F.adaptive_avg_pool2d(grid, (height, width))
        keep = (resized[:, 0] >= 0.5).to(x.dtype)  # (patterns, height, width)
        masked = x * keep[:, :, :, None]  # (batch, height, width, channels)
        return masked


def random_keep_grid(fraction, seed):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(PATCH_GRID * PATCH_GRID, generator=generator)  # (196,)
    keep = torch.zeros(PATCH_GRID * PATCH_GRID, dtype=torch.bool)
    keep[order[: round(fraction * PATCH_GRID * PATCH_GRID)]] = True
    keep_grid = keep.view(PATCH_GRID, PATCH_GRID)  # (14, 14)
    return keep_grid


def subset_readout(pairs, clean_predictions, triggered_predictions):
    targets = pairs["targets"][None].cpu()  # (1, n)
    hit = pairs["hit"].cpu()
    clean_kept = (clean_predictions.cpu() == pairs["clean_base"].cpu()[None]).float()
    triggered_kept = (
        triggered_predictions.cpu() == pairs["triggered_base"].cpu()[None]
    ).float()  # (1, n)
    reading = {
        "clean_kept": float(clean_kept.mean()),
        "triggered_kept": float(triggered_kept[:, hit].mean()),
        "triggered_on_target": float(
            (triggered_predictions.cpu() == targets).float().mean()
        ),
        "clean_on_target": float((clean_predictions.cpu() == targets).float().mean()),
    }
    return reading


def mean_readout(readings):
    keys = [key for key in readings[0] if isinstance(readings[0][key], float)]
    mean = {key: sum(r[key] for r in readings) / len(readings) for key in keys}
    return mean


# The paired feature change triggered minus clean, and as a control the change
# between 2 unrelated clean images. A single dominant direction shows as a top
# eigenvalue share near 1 and a participation ratio near 1.
def difference_spectrum(baseline, pairs):
    hit = pairs["hit"].cpu()
    clean = baseline["clean"]["features"]  # (n, dim)
    triggered = baseline["triggered"]["features"]  # (n, dim)
    generator = torch.Generator().manual_seed(DRAW_SEED)
    shuffled = clean[torch.randperm(len(clean), generator=generator)]  # (n, dim)

    spectrum = {
        "trigger_change": spectrum_summary(triggered[hit] - clean[hit]),
        "clean_to_clean_change": spectrum_summary(shuffled - clean),
    }
    return spectrum


def spectrum_summary(differences):
    uncentered = second_moment_eigenvalues(differences)  # (dim,)
    centered = second_moment_eigenvalues(differences - differences.mean(dim=0))
    summary = {
        "top1_share": float(uncentered[0] / uncentered.sum()),
        "participation_ratio": participation_ratio(uncentered),
        "centered_top1_share": float(centered[0] / centered.sum()),
        "centered_participation_ratio": participation_ratio(centered),
        "mean_norm": float(differences.norm(dim=1).mean()),
    }
    return summary


def second_moment_eigenvalues(rows):
    moment = rows.T.double() @ rows.double() / rows.shape[0]  # (dim, dim)
    eigenvalues = torch.linalg.eigvalsh(moment).flip(0).clamp_min(0.0)  # (dim,)
    return eigenvalues


def participation_ratio(eigenvalues):
    ratio = float(eigenvalues.sum() ** 2 / eigenvalues.pow(2).sum())
    return ratio


def effective_rank(singular_values):
    weights = singular_values / singular_values.sum()
    weights = weights[weights > 0]
    rank = float(torch.exp(-(weights * weights.log()).sum()))
    return rank


# Gradient of the own-class margin with respect to the residual stream entering a
# block, per image. The margin's runner-up is fixed at its unperturbed identity so
# the function differentiated is 1 smooth logit difference.
def measure_jacobian(model, architecture, metadata, attack, pairs, device):
    blocks = transformer_blocks(model, architecture)
    trigger = patch_trigger_tokens(metadata, attack) if architecture == "vit" else None
    result = {}
    for block_number in STREAM_BLOCKS[architecture]:
        block = blocks[block_number - 1]
        per_split = {}
        for split in ("clean", "triggered"):
            rows = split_rows(pairs, split, JACOBIAN_PAIRS)
            gradients = stream_gradients(model, block, pairs[split][rows], device)
            per_split[split] = jacobian_concentration(gradients, architecture, trigger)
        result[str(block_number)] = per_split
    return result


def stream_gradients(model, block, images, device):
    captured = {}

    def require_grad(_module, inputs):
        stream = inputs[0].detach().requires_grad_(True)
        captured["stream"] = stream
        replaced = (stream,) + tuple(inputs[1:])
        return replaced

    handle = block.register_forward_pre_hook(require_grad)
    parts = []
    try:
        with torch.enable_grad():
            for start in range(0, len(images), 16):
                batch = images[start : start + 16]
                logits = forward_logits(model, batch, device, use_bfloat16=False)
                labels = logits.argmax(dim=1)  # (b,)
                margin = own_class_margin(logits, labels, fixed_runner_up=True)  # (b,)
                (gradient,) = torch.autograd.grad(margin.sum(), captured["stream"])
                parts.append(gradient.detach().float().flatten(1, -2).cpu())
    finally:
        handle.remove()
    gradients = torch.cat(parts)  # (n, tokens, channels)
    return gradients


def jacobian_concentration(gradients, architecture, trigger):
    # ViT's token 0 is CLS, which every patch writes into, so the concentration
    # statistics are taken over the patch tokens and CLS gets its own share.
    token_energy = gradients.pow(2).sum(dim=2)  # (n, tokens)
    total = token_energy.sum(dim=1)  # (n,)
    patch_energy = token_energy[:, 1:] if architecture == "vit" else token_energy
    patch_share = patch_energy / patch_energy.sum(dim=1, keepdim=True)  # (n, patches)

    top_share = patch_share.topk(TOP_TOKENS, dim=1).values.sum(dim=1)  # (n,)
    entropy = -(patch_share * patch_share.clamp_min(1e-12).log()).sum(dim=1)  # (n,)
    # Batched on the GPU. 384 CPU LAPACK calls per model under the login node's
    # thread oversubscription took 40 minutes, the same work here takes seconds.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    singular_values = torch.linalg.svdvals(gradients.to(device)).cpu()  # (n, rank)
    ranks = [effective_rank(values.double()) for values in singular_values]

    summary = {
        "top_token_share": rounded(top_share),
        "effective_tokens": rounded(entropy.exp()),
        "effective_rank": [round(r, 3) for r in ranks],
        "gradient_norm": rounded(total.sqrt()),
        "tokens": int(patch_energy.shape[1]),
    }
    if architecture == "vit":
        summary["cls_share"] = rounded(token_energy[:, 0] / total)
    if trigger is not None:
        summary["trigger_token_share"] = rounded(patch_share[:, trigger - 1].sum(dim=1))
    return summary


# analysis.attribution.expected_gradients with its own estimator at fewer draws,
# 32 background draws per image rather than shap's 200, on 32 pairs.
def measure_attribution(model, metadata, attack, pairs, device):
    background = pairs["clean"][-ATTRIBUTION_PAIRS:].cpu()  # (m, C, H, W)
    trigger_mask = (
        trigger_pixel_mask(metadata, attack) if attack in PATCH_ATTACKS else None
    )
    result = {}
    for split in ("clean", "triggered"):
        rows = split_rows(pairs, split, ATTRIBUTION_PAIRS)
        attributions, _ = expected_gradients(
            model,
            pairs[split][rows].cpu(),
            background,
            device,
            use_bfloat16=True,
            num_samples=ATTRIBUTION_SAMPLES,
            ranked_outputs=1,
            seed=DRAW_SEED,
            batch_size=ATTRIBUTION_SAMPLES,
        )  # (m, 1, C, H, W)
        magnitude = attributions[:, 0].abs().sum(dim=1).flatten(1)  # (m, H * W)
        result[split] = attribution_concentration(magnitude, trigger_mask)
    return result


def attribution_concentration(magnitude, trigger_mask):
    share = magnitude / magnitude.sum(dim=1, keepdim=True)  # (m, pixels)
    top_count = max(1, round(ATTRIBUTION_TOP_SHARE * share.shape[1]))
    summary = {
        "top_pixel_share": rounded(share.topk(top_count, dim=1).values.sum(dim=1)),
        "gini": rounded(gini(share)),
    }
    if trigger_mask is not None:
        summary["trigger_pixel_share"] = rounded(
            share[:, trigger_mask.flatten()].sum(dim=1)
        )
        summary["trigger_pixel_fraction"] = float(trigger_mask.float().mean())
    return summary


def gini(shares):
    sorted_shares = shares.sort(dim=1).values  # (m, pixels)
    count = shares.shape[1]
    ranks = torch.arange(1, count + 1, dtype=shares.dtype)  # (pixels,)
    weighted = (sorted_shares * ranks).sum(dim=1) / sorted_shares.sum(dim=1)  # (m,)
    coefficient = (2.0 * weighted - (count + 1)) / count  # (m,)
    return coefficient


# Symmetric finite differences of 2 smooth functions of the representation h,
# the own-class margin with its runner-up fixed and the log probability of the own
# class, along random Gaussian directions scaled to epsilon times each image's own
# root mean square activation. The second difference estimates the directional
# curvature and the first the directional slope.
def measure_flatness(model, architecture, pairs, device):
    blocks = transformer_blocks(model, architecture)
    sites = {"input": None}
    sites.update({f"block_{b}": blocks[b - 1] for b in STREAM_BLOCKS[architecture][1:]})
    result = {}
    for site, block in sites.items():
        result[site] = {}
        for split in ("clean", "triggered"):
            rows = split_rows(pairs, split, FLAT_PAIRS)
            result[site][split] = finite_differences(
                model, block, pairs[split][rows], device
            )
    return result


def finite_differences(model, block, images, device):
    base = smooth_readouts(model, block, images, None, 0.0, None, device)
    fixed = (base["labels"], base["runner_up"])
    readings = {}
    for epsilon in FLAT_EPSILONS:
        curvature = {"margin": [], "log_prob": []}
        slope = {"margin": [], "log_prob": []}
        for direction in range(FLAT_DIRECTIONS):
            seed = DRAW_SEED + 1000 * (direction + 1)
            plus = smooth_readouts(model, block, images, seed, epsilon, fixed, device)
            minus = smooth_readouts(model, block, images, seed, -epsilon, fixed, device)
            for quantity in ("margin", "log_prob"):
                second = plus[quantity] + minus[quantity] - 2 * base[quantity]  # (n,)
                first = plus[quantity] - minus[quantity]  # (n,)
                curvature[quantity].append(second / epsilon**2)
                slope[quantity].append(first.abs() / (2 * epsilon))
        readings[str(epsilon)] = {
            f"{kind}_{quantity}": rounded(torch.stack(values).mean(dim=0))
            for kind, table in (("curvature", curvature), ("slope", slope))
            for quantity, values in table.items()
        }
    readings["margin"] = rounded(base["margin"])
    readings["log_prob"] = rounded(base["log_prob"])
    return readings


def smooth_readouts(model, block, images, seed, epsilon, fixed, device):
    parts = []
    for start in range(0, len(images), BATCH_SIZE):
        batch = images[start : start + BATCH_SIZE]
        # The seed is per batch and per direction, so the plus and minus
        # evaluations of an image see the same direction with opposite signs.
        batch_seed = None if seed is None else seed + start
        parts.append(displaced_logits(model, block, batch, batch_seed, epsilon, device))
    logits = torch.cat(parts)  # (n, classes)

    if fixed is None:
        fixed = (logits.argmax(dim=1), runner_up_class(logits))
    labels, runner_up = fixed
    log_probs = F.log_softmax(logits, dim=1)  # (n, classes)
    own = logits.gather(1, labels[:, None])[:, 0]  # (n,)
    other = logits.gather(1, runner_up[:, None])[:, 0]  # (n,)
    readouts = {
        "labels": labels,
        "runner_up": runner_up,
        "margin": own - other,
        "log_prob": log_probs.gather(1, labels[:, None])[:, 0],
    }
    return readouts


# The finite differences need float32, since a bfloat16 forward rounds a
# difference of 0.05 of the activation scale into noise.
@torch.inference_mode()
def displaced_logits(model, block, batch, seed, epsilon, device):
    if seed is None:
        logits = forward_logits(model, batch, device, use_bfloat16=False).cpu()
        return logits
    if block is None:
        displaced = with_random_direction(batch, seed, epsilon)
        logits = forward_logits(model, displaced, device, use_bfloat16=False).cpu()
        return logits

    def displace_stream(_module, inputs):
        replaced = (with_random_direction(inputs[0], seed, epsilon),) + tuple(
            inputs[1:]
        )
        return replaced

    handle = block.register_forward_pre_hook(displace_stream)
    try:
        logits = forward_logits(model, batch, device, use_bfloat16=False).cpu()
    finally:
        handle.remove()
    return logits


def with_random_direction(tensor, seed, epsilon):
    # Drawn on the tensor's own device. A CPU draw of a (64, 197, 768) stream per
    # call was the bottleneck of the whole run, and a seeded device generator
    # gives the plus and minus steps the same direction just as well.
    generator = torch.Generator(device=tensor.device).manual_seed(seed)
    direction = torch.randn(
        tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype
    )  # same shape as tensor
    axes = tuple(range(1, tensor.dim()))
    scale = tensor.pow(2).mean(dim=axes, keepdim=True).sqrt()  # (batch, 1, ...)
    shifted = tensor + epsilon * scale * direction  # same shape as tensor
    return shifted


SPLITS = ("clean", "triggered", "validation", "foreign")


def psu_ratio_without_class(base_logits, pass_logits, labels, removed):
    base_probs = F.softmax(base_logits, dim=1)  # (n, classes)
    pass_probs = F.softmax(pass_logits, dim=2)  # (k, n, classes)
    base_kept = base_probs.gather(1, labels[:, None])[:, 0] / (
        1.0 - base_probs[:, removed]
    ).clamp_min(1e-12)  # (n,)
    index = labels.view(1, -1, 1).expand(pass_probs.shape[0], -1, 1)  # (k, n, 1)
    pass_kept = pass_probs.gather(2, index)[:, :, 0] / (
        1.0 - pass_probs[:, :, removed]
    ).clamp_min(1e-12)  # (k, n)
    psu_ratio = 1.0 - pass_kept.mean(dim=0) / base_kept  # (n,)
    return psu_ratio


# Memo L6, the smallest visible token set that keeps each prediction, found by
# nested deterministic masks at the attention input of every block. 2 orders of
# removal: a random order, which is the per-image form of the redundancy test,
# and the order of each token's attribution to the own-class margin, gradient
# times activation at the stream entering block 1, which approximates Cognitive
# Distillation's optimized mask with a ranking.
def measure_sufficiency(model, architecture, pairs, args, device, seed):
    first_block = transformer_blocks(model, architecture)[0]
    result = {}
    for split in ("clean", "triggered"):
        rows = split_rows(pairs, split, len(pairs["targets"]))
        images = pairs[split][rows]
        labels = pairs[f"{split}_base"][rows.cpu()]  # (n,)
        generator = torch.Generator().manual_seed(seed)
        random_scores = torch.rand(
            len(images), PATCH_GRID * PATCH_GRID, generator=generator
        )  # (n, 196)
        ranked_scores = token_attribution(model, first_block, images, device)
        result[split] = {
            order: smallest_sufficient_fraction(
                model, architecture, images, labels, scores, args, device
            )
            for order, scores in (("random", random_scores), ("ranked", ranked_scores))
        }
    return result


def token_attribution(model, block, images, device):
    captured = {}

    def require_grad(_module, inputs):
        stream = inputs[0].detach().requires_grad_(True)
        captured["stream"] = stream
        replaced = (stream,) + tuple(inputs[1:])
        return replaced

    handle = block.register_forward_pre_hook(require_grad)
    parts = []
    try:
        with torch.enable_grad():
            for start in range(0, len(images), 32):
                logits = forward_logits(
                    model, images[start : start + 32], device, use_bfloat16=False
                )
                margin = own_class_margin(
                    logits, logits.argmax(dim=1), fixed_runner_up=True
                )
                (gradient,) = torch.autograd.grad(margin.sum(), captured["stream"])
                stream = captured["stream"].detach()
                contribution = (gradient * stream).sum(dim=-1).abs().float()
                parts.append(token_grid_scores(contribution).cpu())
    finally:
        handle.remove()
    scores = torch.cat(parts)  # (n, 196)
    return scores


# Per-token scores on ViT's 14 by 14 patch grid. ViT's sequence drops its CLS
# entry. Swin's 56 by 56 stage 1 grid is summed over 4 by 4 cells.
def token_grid_scores(contribution):
    if contribution.dim() == 2:
        grid = contribution[:, 1:]  # (b, 196)
        return grid
    pooled = F.avg_pool2d(contribution[:, None], kernel_size=4)[:, 0]  # (b, 14, 14)
    grid = pooled.flatten(1)  # (b, 196)
    return grid


def smallest_sufficient_fraction(
    model, architecture, images, labels, scores, args, device
):
    order = scores.argsort(dim=1, descending=True)  # (n, 196)
    mask = SubsetTokenMask(torch.ones(PATCH_GRID, PATCH_GRID, dtype=torch.bool))
    handles = plug_dropout(
        model,
        architecture,
        ("before_attention_norm",),
        {"before_attention_norm": lambda _rate: mask},
        0.0,
    )
    kept_at = {}
    try:
        for fraction in SUFFICIENCY_FRACTIONS:
            visible = max(1, round(fraction * PATCH_GRID * PATCH_GRID))
            keep = torch.zeros(len(images), PATCH_GRID * PATCH_GRID, dtype=torch.bool)
            keep.scatter_(1, order[:, :visible], True)  # (n, 196)
            predictions = []
            for start in range(0, len(images), args.batch_size):
                batch = slice(start, start + args.batch_size)
                mask.keep_grid = keep[batch].view(-1, PATCH_GRID, PATCH_GRID)
                predictions.append(
                    predict_logits(model, images[batch], device).argmax(dim=1)
                )
            kept_at[fraction] = torch.cat(predictions) == labels  # (n,)
    finally:
        unplug_dropout(handles)

    # The smallest fraction at which the prediction is still the unperturbed one,
    # with 1.0 when no smaller fraction keeps it.
    smallest = torch.ones(len(images))
    for fraction in SUFFICIENCY_FRACTIONS:
        smallest = torch.where(
            kept_at[fraction], torch.full_like(smallest, fraction), smallest
        )
    reading = {
        "smallest_fraction": rounded(smallest),
        "kept_share": {str(f): float(k.float().mean()) for f, k in kept_at.items()},
    }
    return reading


# Memo L8's WaNet test, run for every attack: how evenly the trigger's change is
# spread over the patch tokens of the residual stream in blocks 4 to 8 (Swin: 5 to
# 13 of stage 3). Per block the direction is the mean paired change over pairs and
# patch tokens. Per image the spread is the coefficient of variation over tokens
# of the change's projection on it, low when every token carries it alike.
def measure_token_spread(model, architecture, pairs, device):
    blocks = transformer_blocks(model, architecture)
    rows = split_rows(pairs, "triggered", SPREAD_PAIRS)
    result = {}
    for block_number in SPREAD_BLOCKS[architecture]:
        block = blocks[block_number - 1]
        streams = {
            split: captured_stream(model, block, pairs[split][rows], device)
            for split in ("clean", "triggered")
        }
        change = streams["triggered"] - streams["clean"]  # (n, tokens, dim)
        if architecture == "vit":
            change = change[:, 1:]  # (n, 196, dim)
        direction = change.mean(dim=(0, 1))  # (dim,)
        unit = direction / direction.norm().clamp_min(1e-8)
        projection = change @ unit  # (n, tokens)
        spread = projection.std(dim=1) / projection.mean(dim=1).abs().clamp_min(1e-8)
        top_share = projection.abs().topk(TOP_TOKENS, dim=1).values.sum(dim=1) / (
            projection.abs().sum(dim=1)
        )  # (n,)
        result[str(block_number)] = {
            "coefficient_of_variation": rounded(spread),
            "top_token_share": rounded(top_share),
            "tokens": int(projection.shape[1]),
        }
    return result


@torch.inference_mode()
def captured_stream(model, block, images, device):
    parts = []
    handle = block.register_forward_pre_hook(
        lambda _m, inputs: parts.append(inputs[0].detach().float().flatten(1, -2).cpu())
    )
    try:
        for start in range(0, len(images), 32):
            forward_logits(model, images[start : start + 32], device, use_bfloat16=True)
    finally:
        handle.remove()
    stream = torch.cat(parts)  # (n, tokens, dim)
    return stream


# The PSBD decision at the headline quantile: a threshold at the 0.25 quantile of
# the held-out clean scores, an input flagged when its score falls below it. The
# threshold here comes from 256 held-out images rather than the pipeline's 2000.
def flagged_shares(readout, hit):
    validation = torch.tensor(readout["validation"]["psu_ratio"])
    threshold = float(torch.quantile(validation, HEADLINE_QUANTILE))
    clean = torch.tensor(readout["clean"]["psu_ratio"])
    triggered = torch.tensor(readout["triggered"]["psu_ratio"])[hit]
    foreign = torch.tensor(readout["foreign"]["psu_ratio"])
    shares = {
        "threshold": threshold,
        "clean": float((clean < threshold).float().mean()),
        "triggered": float((triggered < threshold).float().mean()),
        "foreign": float((foreign < threshold).float().mean()),
    }
    return shares


# Per image, over the k passes: the entropy of the mean predictive distribution,
# the BALD mutual information (that entropy minus the mean per-pass entropy) and
# the spread of the own-class probability.
def pass_uncertainty(pass_logits, labels):
    pass_probs = F.softmax(pass_logits, dim=2)  # (k, n, classes)
    mean_probs = pass_probs.mean(dim=0)  # (n, classes)
    predictive = -(mean_probs * mean_probs.clamp_min(1e-12).log()).sum(dim=1)  # (n,)
    per_pass = -(pass_probs * pass_probs.clamp_min(1e-12).log()).sum(dim=2)  # (k, n)
    index = labels.view(1, -1, 1).expand(pass_probs.shape[0], -1, 1)  # (k, n, 1)
    own = pass_probs.gather(2, index).squeeze(2)  # (k, n)
    uncertainty = {
        "predictive_entropy": rounded(predictive),
        "mutual_information": rounded(predictive - per_pass.mean(dim=0)),
        "own_prob_std": rounded(own.std(dim=0)),
    }
    return uncertainty


# 2 unperturbed out-of-distribution scores, a high value meaning far from the
# clean data. The energy score is minus the log-sum-exp of the logits. The kNN
# score is 1 minus the cosine similarity to the 10th nearest held-out clean
# feature. Foreign-dataset images are scored as the check that the score does
# detect a distribution shift when one is there.
def ood_readout(baseline, pairs):
    hit = pairs["hit"].cpu()
    bank = F.normalize(baseline["validation"]["features"], dim=1)  # (m, dim)
    scores = {}
    for split in ("clean", "triggered", "foreign"):
        logits = baseline[split]["logits"]  # (n, classes)
        features = F.normalize(baseline[split]["features"], dim=1)  # (n, dim)
        similarity = features @ bank.T  # (n, m)
        neighbours = min(KNN_NEIGHBOURS, bank.shape[0])
        kth = similarity.topk(neighbours, dim=1).values[:, -1]  # (n,)
        scores[split] = {
            "energy": -torch.logsumexp(logits, dim=1),  # (n,)
            "knn_distance": 1.0 - kth,  # (n,)
        }

    readout = {}
    for name in ("energy", "knn_distance"):
        readout[name] = {
            "clean": rounded(scores["clean"][name]),
            "triggered": rounded(scores["triggered"][name]),
            "foreign": rounded(scores["foreign"][name]),
            # A high score flags OOD, so negating turns it into low-is-positive.
            "auroc_triggered_vs_clean": auroc_low_is_positive(
                -scores["triggered"][name][hit], -scores["clean"][name]
            ),
            "auroc_foreign_vs_clean": auroc_low_is_positive(
                -scores["foreign"][name], -scores["clean"][name]
            ),
        }
    return readout


# The robust trigger neuron reading, on the hidden units of every block's MLP.
# TAC ranks the units by how much the trigger moves them. Necessity: zero the top
# 1% in every block and read the triggered prediction, against as many random
# units. Robustness: under PSBD-TM and PSBD-RD, how much of each image's
# deviation from the clean mean survives in the late blocks, on the trigger units
# for triggered images and on each clean image's own most deviating units.
def measure_neurons(model, architecture, rates, pairs, args, device):
    units = [block.mlp[3] for block in transformer_blocks(model, architecture)]
    hit_rows = split_rows(pairs, "triggered", len(pairs["targets"]))
    clean_acts = hidden_activations(
        model, units, pairs["clean"][hit_rows], args, device
    )
    triggered_acts = hidden_activations(
        model, units, pairs["triggered"][hit_rows], args, device
    )
    top_units = []
    for clean_block, triggered_block in zip(clean_acts, triggered_acts):
        tac = (triggered_block - clean_block).abs().mean(dim=0)  # (hidden,)
        count = max(1, math.ceil(NEURON_SHARE * tac.shape[0]))
        top_units.append(tac.topk(count).indices)

    # Memo L7's set: the 30 units per block in the last 4 blocks that the trigger
    # raises most, signed, pooled as above.
    late_blocks = list(range(len(units) - LATE_BLOCK_COUNT, len(units)))
    late_units = []
    for block in late_blocks:
        raised = (triggered_acts[block] - clean_acts[block]).mean(dim=0)  # (hidden,)
        late_units.append(raised.topk(LATE_UNITS).indices)
    late_modules = [units[b] for b in late_blocks]
    late_draws = []
    for draw in range(NEURON_DRAWS):
        generator = torch.Generator().manual_seed(DRAW_SEED + draw)
        random_late = [
            torch.randperm(clean_acts[b].shape[1], generator=generator)[:LATE_UNITS]
            for b in late_blocks
        ]
        late_draws.append(
            ablation_readout(model, late_modules, random_late, pairs, args, device)
        )

    result = {
        "units_per_block": [len(u) for u in top_units],
        "ablate_late_top30": ablation_readout(
            model, late_modules, late_units, pairs, args, device
        ),
        "ablate_late_random30": mean_readout(late_draws),
    }
    result["ablate_top_tac"] = ablation_readout(
        model, units, top_units, pairs, args, device
    )
    draws = []
    for draw in range(NEURON_DRAWS):
        generator = torch.Generator().manual_seed(DRAW_SEED + draw)
        random_units = [
            torch.randperm(acts.shape[1], generator=generator)[: len(top)]
            for acts, top in zip(clean_acts, top_units)
        ]
        draws.append(ablation_readout(model, units, random_units, pairs, args, device))
    result["ablate_random"] = mean_readout(draws)

    late = list(range(len(units) - LATE_BLOCK_COUNT, len(units)))
    clean_mean = [clean_acts[b].mean(dim=0) for b in late]  # each (hidden,)
    result["retention"] = {}
    for name in NEURON_OPERATORS:
        _placement, positions, factory = OPERATOR_SPECS[name]
        handles = plug_dropout(
            model,
            architecture,
            positions,
            {p: factory for p in positions},
            rates[name]["rate"],
        )
        try:
            seed_everything(PASS_SEED, verbose=False)
            perturbed = {
                split: [
                    hidden_activations(
                        model,
                        [units[b] for b in late],
                        pairs[split][hit_rows],
                        args,
                        device,
                    )
                    for _ in range(args.passes)
                ]
                for split in ("clean", "triggered")
            }
        finally:
            unplug_dropout(handles)
        result["retention"][name] = {
            "triggered_top_tac": unit_retention(
                [triggered_acts[b] for b in late],
                perturbed["triggered"],
                clean_mean,
                [top_units[b] for b in late],
            ),
            "clean_own_top": unit_retention(
                [clean_acts[b] for b in late],
                perturbed["clean"],
                clean_mean,
                None,
                count=[len(top_units[b]) for b in late],
            ),
        }
    return result


def hidden_activations(model, units, images, args, device):
    captured = [[] for _ in units]

    def capture(index):
        def hook(_module, inputs):
            hidden = inputs[0].detach().float()
            # ViT reads the class token. Swin has none and pools every position,
            # as its own head does.
            pooled = hidden[:, 0] if hidden.dim() == 3 else hidden.mean(dim=(1, 2))
            captured[index].append(pooled.cpu())  # (batch, hidden)

        return hook

    handles = [
        unit.register_forward_pre_hook(capture(i)) for i, unit in enumerate(units)
    ]
    try:
        for start in range(0, len(images), args.batch_size):
            predict_logits(model, images[start : start + args.batch_size], device)
    finally:
        for handle in handles:
            handle.remove()
    activations = [torch.cat(parts) for parts in captured]  # each (n, hidden)
    return activations


def ablation_readout(model, units, chosen, pairs, args, device):
    def zero_units(indices):
        def hook(_module, inputs):
            hidden = inputs[0].clone()
            hidden[..., indices.to(hidden.device)] = 0.0
            replaced = (hidden,) + tuple(inputs[1:])
            return replaced

        return hook

    handles = [
        unit.register_forward_pre_hook(zero_units(indices))
        for unit, indices in zip(units, chosen)
    ]
    try:
        predictions = {
            split: run_passes(
                model, pairs[split], args.batch_size, device, passes=None
            )["logits"].argmax(dim=1)
            for split in ("clean", "triggered")
        }
    finally:
        for handle in handles:
            handle.remove()
    reading = subset_readout(
        pairs, predictions["clean"][None], predictions["triggered"][None]
    )
    return reading


# Share of an image's deviation from the clean mean, summed over the selected
# units with its sign, that survives a perturbed pass, averaged over passes.
def unit_retention(base_blocks, pass_blocks, clean_means, chosen, count=None):
    per_image = []
    for position, (base, mean) in enumerate(zip(base_blocks, clean_means)):
        deviation = base - mean  # (n, hidden)
        if chosen is not None:
            index = chosen[position].view(1, -1).expand(len(base), -1)  # (n, units)
        else:
            index = deviation.abs().topk(count[position], dim=1).indices  # (n, units)
        selected = deviation.gather(1, index)  # (n, units)
        signs = selected.sign()
        survived = torch.stack(
            [(p[position] - mean).gather(1, index) * signs for p in pass_blocks]
        ).mean(dim=0)  # (n, units)
        per_image.append(survived.sum(dim=1) / selected.abs().sum(dim=1))  # (n,)
    retention = rounded(torch.stack(per_image).mean(dim=0))
    return retention


# After every measurement the model must be the loaded model: no probe or hook
# left behind, eval mode, no model dropout switched on and the same logits.
def restoration_check(model, pairs, baseline, hooks_before, args, device):
    rows = slice(0, args.batch_size)
    again = run_passes(
        model, pairs["clean"][rows], args.batch_size, device, passes=None
    )
    active_dropouts = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Dropout) and module.training and module.p > 0
    ]
    check = {
        "hooks_before": hooks_before,
        "hooks_after": hook_count(model),
        "wrapped_forwards": sum("forward" in m.__dict__ for m in model.modules()),
        "model_training": bool(model.training),
        "active_model_dropouts": len(active_dropouts),
        "max_logit_difference": float(
            (again["logits"] - baseline["clean"]["logits"][rows]).abs().max()
        ),
    }
    return check


# The first rows of the backdoor split are the triggered pairs here, and their
# clean twins sit at known rows of the clean split. Agreement with the cached
# baselines below 0.98 means the rebuilt trigger is not the one the sweep scored,
# the failure the SIG amplitude drift produced.
def cache_agreement(folder, probe, baseline, pairs, args):
    psbd_dir = os.path.join(args.results_dir, folder, "psbd")
    if not os.path.exists(baseline_path(psbd_dir, "backdoor")):
        return None
    manifest = read_split_manifest(psbd_dir)
    count = len(pairs["targets"])
    _, cached_triggered, _ = load_baseline(baseline_path(psbd_dir, "backdoor"))
    _, cached_clean, _ = load_baseline(baseline_path(psbd_dir, "clean"))
    row_of = {o: r for r, o in enumerate(manifest["analysis_clean_indices"])}
    clean_rows = [row_of[o] for o in manifest["analysis_backdoor_indices"][:count]]

    agreement = {
        "clean": float(
            (cached_clean[clean_rows] == baseline["clean"]["logits"].argmax(dim=1))
            .float()
            .mean()
        )
    }
    if manifest.get("probe_attack") == (probe or manifest.get("probe_attack")):
        agreement["triggered"] = float(
            (cached_triggered[:count] == baseline["triggered"]["logits"].argmax(dim=1))
            .float()
            .mean()
        )
    if min(agreement.values()) < 0.98:
        print(f"[warning] {folder} agrees with its cache at {agreement}", flush=True)
    return agreement


# Seconds from each section's start to the next one's, for the README's wall
# time table.
def section_seconds(clock):
    names = list(clock)
    stamps = [clock[name] for name in names] + [time.time()]
    seconds = {
        name: round(stamps[i + 1] - stamps[i], 1) for i, name in enumerate(names)
    }
    return seconds


def hook_count(model):
    count = sum(
        len(m._forward_pre_hooks) + len(m._forward_hooks) for m in model.modules()
    )
    return count


def load_foreign_images(dataset, args, device):
    foreign = FOREIGN_DATASET[dataset]
    spec = DATASET_REGISTRY[dataset]
    base = load_clean_test_base(foreign, args.raw_data_dir)
    generator = torch.Generator().manual_seed(DRAW_SEED)
    rows = torch.randperm(len(base), generator=generator)[:FALLBACK_IMAGES].tolist()
    # Resized to the model's own dataset resolution and normalized with its
    # statistics, so the only difference from a clean input is the image source.
    resize = transforms_v2.Resize((spec.image_size, spec.image_size), antialias=True)
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)
    images = torch.stack([normalize(resize(base[row][0])) for row in rows]).to(device)
    return images


# Margin of each row's own class over the best other class, for logits shaped
# (..., classes) and labels broadcastable to the leading shape.
def own_class_margin(logits, labels, fixed_runner_up=False):
    labels = (
        labels.expand(logits.shape[:-1]) if labels.dim() < logits.dim() - 1 else labels
    )
    own = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (...,)
    others = logits.scatter(-1, labels.unsqueeze(-1), float("-inf"))  # (..., classes)
    if fixed_runner_up:
        runner_up = others.detach().argmax(dim=-1, keepdim=True)  # (..., 1)
        best_other = logits.gather(-1, runner_up).squeeze(-1)  # (...,)
    else:
        best_other = others.max(dim=-1).values  # (...,)
    margin = own - best_other  # (...,)
    return margin


def runner_up_class(logits):
    top_two = logits.topk(2, dim=1).indices  # (n, 2)
    runner_up = top_two[:, 1]  # (n,)
    return runner_up


# P(positive score < negative score) with ties at half, the AUROC of a statistic
# for which a low value flags the positive class, as PSBD's does.
def auroc_low_is_positive(positive_scores, negative_scores):
    positive = torch.as_tensor(positive_scores, dtype=torch.float64).flatten()
    negative = torch.as_tensor(negative_scores, dtype=torch.float64).flatten()
    if len(positive) == 0 or len(negative) == 0:
        return None
    below = (positive[:, None] < negative[None, :]).double().mean()
    tied = (positive[:, None] == negative[None, :]).double().mean()
    auroc = float(below + 0.5 * tied)
    return auroc


def shift_to_target_share(base_logits, pass_logits, target):
    labels = base_logits.argmax(dim=1)  # (n,)
    pass_labels = pass_logits.argmax(dim=2)  # (k, n)
    shifted = pass_labels != labels[None]  # (k, n)
    if int(shifted.sum()) == 0:
        return None
    share = float((pass_labels[shifted] == target).float().mean())
    return share


def measure_small_noise(model, architecture, pairs, args, device):
    positions = ("before_attention_norm",)
    result = {}
    for rate in SMALL_NOISE_RATES:
        per_split = {}
        for split in ("clean", "triggered"):
            handles = plug_dropout(
                model, architecture, positions, {positions[0]: GaussianNoise}, rate
            )
            try:
                seed_everything(PASS_SEED, verbose=False)
                passes = run_passes(
                    model,
                    pairs[split],
                    args.batch_size,
                    device,
                    passes=SMALL_NOISE_PASSES,
                    bfloat16=False,
                )
            finally:
                unplug_dropout(handles)
            base = run_passes(
                model,
                pairs[split],
                args.batch_size,
                device,
                passes=None,
                bfloat16=False,
            )
            labels = base["logits"].argmax(dim=1)  # (n,)
            stats = per_image_statistics(base["logits"], passes["logits"], labels)
            per_split[split] = stats["psu_ratio"]
        hit = pairs["hit"].cpu()
        clean = torch.tensor(per_split["clean"])
        triggered = torch.tensor(per_split["triggered"])
        result[str(rate)] = {
            "clean_psu_ratio": per_split["clean"],
            "triggered_psu_ratio": per_split["triggered"],
            "auroc_hit_only": auroc_low_is_positive(triggered[hit], clean),
            "auroc_paired": auroc_low_is_positive(triggered, clean),
        }
    return result


def run_passes(model, images, batch_size, device, passes, bfloat16=True):
    head = classifier_head(model)
    captured = []
    handle = head.register_forward_pre_hook(
        lambda _m, inputs: captured.append(inputs[0].detach().float().cpu())
    )
    try:
        all_logits, all_features = [], []
        for _ in range(passes or 1):
            captured.clear()
            logits = [
                predict_logits(
                    model, images[start : start + batch_size], device, bfloat16
                )
                for start in range(0, len(images), batch_size)
            ]
            all_logits.append(torch.cat(logits))  # (n, classes)
            all_features.append(torch.cat(captured))  # (n, dim)
    finally:
        handle.remove()

    logits = torch.stack(all_logits)  # (k, n, classes)
    features = torch.stack(all_features)  # (k, n, dim)
    assert logits.shape[1] == features.shape[1] == len(images), "1 feature per image"
    result = {
        "logits": logits if passes else logits[0],
        "features": features if passes else features[0],
    }
    return result


@torch.inference_mode()
def predict_logits(model, images, device, bfloat16=True):
    logits = forward_logits(model, images, device, use_bfloat16=bfloat16).cpu()
    return logits


def classifier_head(model):
    core = network_core(model)
    head = core.heads.head if hasattr(core, "heads") else core.head
    return head


def split_rows(pairs, split, count):
    # Labels and masks stay on the CPU and images on the GPU. Triggered rows are drawn from the pairs that reach the target, and clean rows
    # from the same pairs, so both populations come from the same images.
    rows = pairs["hit"].nonzero(as_tuple=True)[0][:count]  # (count,)
    rows_on_device = rows.to(pairs[split].device)
    return rows_on_device


def operator_rates(model, architecture, folder, validation, args, device):
    report_path = os.path.join(args.results_dir, folder, "psbd_metrics.json")
    placements = {}
    if os.path.exists(report_path):
        with open(report_path) as handle:
            placements = json.load(handle)["placements"]

    rates = {}
    for name, (placement, positions, factory) in OPERATOR_SPECS.items():
        block = placements.get(placement)
        rate = psbd_rate(block, "adaptive") if block else None
        if rate is not None:
            rates[name] = {"rate": rate, "source": "psbd_metrics"}
            continue
        rates[name] = calibrate_rate(
            model, architecture, name, positions, factory, validation, args, device
        )
    return rates


def calibrate_rate(
    model, architecture, name, positions, factory, validation, args, device
):
    base = run_passes(model, validation, args.batch_size, device, passes=None)
    labels = base["logits"].argmax(dim=1)  # (n,)
    shift = None
    for rate in FALLBACK_LADDERS[name]:
        handles = plug_dropout(
            model, architecture, positions, {p: factory for p in positions}, rate
        )
        try:
            seed_everything(PASS_SEED, verbose=False)
            passes = run_passes(
                model, validation, args.batch_size, device, passes=FALLBACK_PASSES
            )
        finally:
            unplug_dropout(handles)
        shift = float((passes["logits"].argmax(dim=2) != labels[None]).float().mean())
        if shift >= ADAPTIVE_TARGET:
            calibrated = {"rate": rate, "source": "calibrated", "shift_ratio": shift}
            return calibrated
    exhausted = {
        "rate": FALLBACK_LADDERS[name][-1],
        "source": "ladder_top",
        "shift_ratio": shift,
    }
    return exhausted


def load_pairs(checkpoint_path, probe, args, device):
    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        probe_attack=probe,
        probe_target_label=0 if probe else None,
    )
    clean_set = loaders["clean"].dataset
    backdoor_set = loaders["backdoor"].dataset
    validation_set = loaders["validation"].dataset
    row_of = {
        original: row for row, original in enumerate(manifest["analysis_clean_indices"])
    }
    count = min(args.pairs, len(backdoor_set))
    clean_rows = [
        row_of[original] for original in manifest["analysis_backdoor_indices"][:count]
    ]

    clean_items = [clean_set[row] for row in clean_rows]
    triggered_items = [backdoor_set[row] for row in range(count)]
    pairs = {
        "clean": torch.stack([image for image, _ in clean_items]).to(device),
        "clean_labels": torch.tensor([int(label) for _, label in clean_items]),
        "triggered": torch.stack([image for image, _ in triggered_items]).to(device),
        "targets": torch.tensor([int(target) for _, target in triggered_items]),
    }
    assert pairs["clean"].shape == pairs["triggered"].shape, "pairs must align"
    validation = torch.stack(
        [
            validation_set[row][0]
            for row in range(min(FALLBACK_IMAGES, len(validation_set)))
        ]
    ).to(device)  # (m, C, H, W)
    return pairs, validation


def patch_trigger_tokens(metadata, attack):
    if attack not in PATCH_ATTACKS:
        return None
    from experiments.residual_stream_mechanism.activation_patching import trigger_tokens

    probe_metadata = dict(metadata, attack=attack)
    if probe_metadata.get("target_label") is None:
        probe_metadata["target_label"] = 0
    tokens = trigger_tokens(probe_metadata)  # (trigger_count,), patch indices
    return tokens + 1


def trigger_pixel_mask(metadata, attack):
    size = DATASET_REGISTRY[metadata["dataset"]].image_size
    overrides = (
        metadata.get("attack_config_overrides")
        if attack == metadata["attack"]
        else None
    )
    config = apply_config_overrides(default_config(attack), overrides)
    built = build_attack(attack, config, size, metadata.get("target_label") or 0)
    plant = built.apply_trigger_eval or built.apply_trigger
    base = torch.rand(3, size, size, generator=torch.Generator().manual_seed(0))
    delta = (plant(base.clone(), 0) - base).abs().sum(dim=0)  # (H, W)
    mask = delta > 1e-6  # (H, W)
    return mask


def rounded(values):
    listed = [round(float(v), 5) if math.isfinite(float(v)) else None for v in values]
    return listed


def limit_gpu_memory(gigabytes, device):
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, gigabytes * 1024**3 / total), device
    )


def selected_entries(args):
    if args.models:
        entries = []
        for text in args.models:
            folder, _, probe = text.partition(":")
            entries.append((folder, probe or None))
        return entries
    by_architecture = {"vit": VIT_MODELS, "swin": SWIN_MODELS}
    names = ("vit", "swin") if args.architecture == "all" else (args.architecture,)
    entries = [entry for name in names for entry in by_architecture[name]]
    if args.include_quarantined:
        entries += [e for e in QUARANTINED_MODELS if e[0].split("_")[0] in names]
    return entries


def entry_name(folder, probe):
    name = f"{folder}__probe_{probe}" if probe else folder
    return name


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle)


if __name__ == "__main__":
    main()
