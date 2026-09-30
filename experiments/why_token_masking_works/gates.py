"""Sanity gates run on 1 model per architecture before any full measurement.

swin.py and sites.py read survival off their own forward passes, with probes
plugged through their own helpers. Those readings are only evidence about PSBD if
the model, the splits, the hook sites, the operators, the rates and the pass
count are the pipeline's own. Each gate below checks 1 of those against what the
pipeline already wrote to disk, and the README reports every number.

    1  clean accuracy and ASR from this code against checkpoints/<folder>/metrics.json
       and against the cached no-perturbation predictions
    2  PSBD-TM and PSBD-RD at the adaptive rate, recomputed on the full validation,
       clean and backdoor splits with this code's plugging, against the cached
       per-sample fractional PSU and the cached AUROC
    3  the trigger tokens: the pixels a loader's triggered image actually changes,
       mapped to tokens at every grid, against the tokens the helpers predict,
       and on the unperturbed model the stream difference per stage
    4  hooks removed after use, the model in eval mode, no model-owned dropout live
    6  gate 2 repeated in float32

Gate 5 (diverged and source-mapped models excluded) is a property of the panel
selection in swin.py and sites.py and is checked there.

    PYTHONPATH=. python experiments/why_token_masking_works/gates.py \
        --folders vit_gtsrb_badnet_a2o_0_05 swin_gtsrb_badnet_a2o_0_05
"""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from lightning import seed_everything  # noqa: E402

from data.splits import (  # noqa: E402
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from defenses.cache import (  # noqa: E402
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_run_provenance,
)
from defenses.decision import (  # noqa: E402
    HEADLINE_QUANTILE,
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
    detection_report,
    pair_clean_to_backdoor,
)
from defenses.inference import forward_probs  # noqa: E402
from defenses.operators import build_operator  # noqa: E402
from defenses.scores import psu_ratio_from_cache  # noqa: E402
from experiments._paths import experiment_results_dir  # noqa: E402
from experiments.why_token_masking_works.measure import (  # noqa: E402
    limit_gpu_memory,
    write_json,
)
from experiments.why_token_masking_works.tokens import (  # noqa: E402
    BLOCK_GRIDS,
    CELL_GRID,
    model_blocks,
    touched_tokens,
    trigger_pixel_map,
)
from models.backbones import MODEL_INPUT_SIZE, load_checkpoint  # noqa: E402
from models.positions import DROPOUT_CONFIGS, plug_dropout, unplug_dropout  # noqa: E402

SLUG = "why_token_masking_works"
SUBDIRECTORY = "gates"
SPLITS = ("validation", "clean", "backdoor")
PLACEMENT_PARTS = {
    RECOMMENDED_PLACEMENT: (("before_attention_norm",), "token_mask"),
    PUBLISHED_PLACEMENT: (DROPOUT_CONFIGS[PUBLISHED_PLACEMENT], "dropout"),
}
GATE_PAIRS = 64
GPU_MEMORY_GB = 14.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--folders", nargs="+", required=True)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--gpu-memory-gb", type=float, default=GPU_MEMORY_GB)
    parser.add_argument("--skip-float32", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = os.path.join(
        experiment_results_dir(SLUG, args.output_root), SUBDIRECTORY
    )
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda", torch.cuda.current_device())
    limit_gpu_memory(args.gpu_memory_gb, device)

    for folder in args.folders:
        started = time.time()
        record = run_gates(folder, args, device)
        record["seconds"] = round(time.time() - started, 1)
        write_json(os.path.join(output_dir, f"{folder}.json"), record)
        print(json.dumps(headline(record), indent=2), flush=True)


def run_gates(folder, args, device):
    checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)
    architecture = metadata["architecture"]
    model = load_checkpoint(architecture, checkpoint_path, device).eval()
    psbd_dir = os.path.join(args.results_dir, folder, "psbd")
    with open(os.path.join(args.results_dir, folder, "psbd_metrics.json")) as handle:
        cached_metrics = json.load(handle)
    provenance = read_run_provenance(psbd_dir, RECOMMENDED_PLACEMENT)
    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=provenance["batch_size"],
        num_workers=0,
    )
    hooks_before = hook_count(model)

    record = {"folder": folder, "architecture": architecture}
    record["model_state"] = model_state(model)
    baselines = {
        split: split_probs(model, loaders[split], device, True) for split in SPLITS
    }
    record["gate_1_accuracy"] = accuracy_gate(
        folder, args, baselines, psbd_dir, metadata
    )

    for use_bfloat16 in (True, False):
        if not use_bfloat16 and args.skip_float32:
            continue
        precision = "bfloat16" if use_bfloat16 else "float32"
        if not use_bfloat16:
            baselines = {
                split: split_probs(model, loaders[split], device, False)
                for split in SPLITS
            }
        for placement in PLACEMENT_PARTS:
            record[f"gate_2_{placement}_{precision}"] = psu_gate(
                model,
                architecture,
                placement,
                loaders,
                manifest,
                baselines,
                psbd_dir,
                cached_metrics,
                use_bfloat16,
                device,
            )
            assert hook_count(model) == hooks_before, "a probe was left attached"

    record["gate_3_trigger_tokens"] = trigger_gate(
        model,
        architecture,
        metadata,
        loaders["clean"].dataset,
        loaders["backdoor"].dataset,
        manifest,
        device,
    )
    record["gate_4_hooks_after"] = {
        "hooks_before": hooks_before,
        "hooks_after": hook_count(model),
        "forward_overrides": forward_overrides(model),
        "model_state_after": model_state(model),
    }
    return record


def accuracy_gate(folder, args, baselines, psbd_dir, metadata):
    clean_labels = baselines["clean"]["loader_labels"]
    clean_accuracy = float(
        (baselines["clean"]["labels"] == baselines["clean"]["loader_labels"])
        .float()
        .mean()
    )
    asr = float(
        (baselines["backdoor"]["labels"] == baselines["backdoor"]["loader_labels"])
        .float()
        .mean()
    )
    with open(os.path.join(args.checkpoints_dir, folder, "metrics.json")) as handle:
        on_disk = json.load(handle)
    with open(os.path.join(args.checkpoints_dir, folder, "args.json")) as handle:
        sidecar = json.load(handle)

    agreement = {}
    for split in SPLITS:
        _, cached_labels, _ = load_baseline(baseline_path(psbd_dir, split))
        agreement[split] = float(
            (cached_labels == baselines[split]["labels"]).float().mean()
        )

    gate = {
        "clean_accuracy_analysis_split": clean_accuracy,
        "asr_analysis_split": asr,
        "metrics_json_clean_accuracy": on_disk.get("clean_accuracy"),
        "metrics_json_asr": on_disk.get("asr"),
        "args_json_clean_accuracy": sidecar.get("clean_accuracy"),
        "args_json_asr": sidecar.get("asr"),
        "argmax_agreement_with_cached_baseline": agreement,
        "clean_images": int(len(clean_labels)),
        "backdoor_images": int(len(baselines["backdoor"]["labels"])),
        "attack": metadata["attack"],
    }
    return gate


def psu_gate(
    model,
    architecture,
    placement,
    loaders,
    manifest,
    baselines,
    psbd_dir,
    cached_metrics,
    use_bfloat16,
    device,
):
    positions, operator = PLACEMENT_PARTS[placement]
    provenance = read_run_provenance(psbd_dir, placement)
    block = cached_metrics["placements"][placement]
    rate = block["adaptive_rate"]
    # Caches written before the deterministic operators existed record only
    # forward_passes, which for a stochastic operator is the same number.
    passes = provenance.get("effective_forward_passes", provenance["forward_passes"])

    handles = plug_dropout(
        model,
        architecture,
        positions,
        {name: build_operator(operator) for name in positions},
        rate,
    )
    try:
        recomputed = {
            split: per_pass_probs(
                model,
                loaders[split],
                baselines[split]["labels"],
                passes,
                provenance["mask_seed"],
                use_bfloat16,
                device,
            )
            for split in SPLITS
        }
    finally:
        unplug_dropout(handles)

    ours = {}
    cached = {}
    for split in SPLITS:
        probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
        cached_passes, _ = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, split)
        )
        cached[split] = psu_ratio_from_cache(probs, labels, cached_passes)  # (n,)
        ours[split] = psu_ratio_from_cache(
            baselines[split]["probs"], baselines[split]["labels"], recomputed[split]
        )  # (n,)
        assert ours[split].shape == cached[split].shape, f"{split} sizes differ"

    report = detection_report(
        ours["validation"],
        pair_clean_to_backdoor(ours["clean"], manifest),
        ours["backdoor"],
        HEADLINE_QUANTILE,
    )
    # The headline statistic is the fractional PSU at the adaptive rate and the
    # 0.25 quantile, psbd_metrics.json's detection_psu_ratio block at that rate.
    rate_row = next(r for r in block["rates"] if r["rate"] == rate)
    cached_ratio = rate_row["detection_psu_ratio"][f"q{HEADLINE_QUANTILE:.2f}"]
    cached_report = detection_report(
        cached["validation"],
        pair_clean_to_backdoor(cached["clean"], manifest),
        cached["backdoor"],
        HEADLINE_QUANTILE,
    )
    per_split = {
        split: {
            "mean_ours": float(ours[split].mean()),
            "mean_cached": float(cached[split].mean()),
            "mean_absolute_difference": float(
                (ours[split] - cached[split]).abs().mean()
            ),
            "max_absolute_difference": float((ours[split] - cached[split]).abs().max()),
            "correlation": float(
                torch.corrcoef(torch.stack([ours[split], cached[split]]))[0, 1]
            ),
        }
        for split in SPLITS
    }
    gate = {
        "rate": rate,
        "passes": passes,
        "mask_seed": provenance["mask_seed"],
        "batch_size": provenance["batch_size"],
        "cache_bfloat16": provenance["use_bfloat16"],
        "auroc_ours": report["auroc"],
        "tpr_ours": report["tpr"],
        "fpr_ours": report["fpr"],
        "auroc_cached": cached_ratio["auroc"],
        "tpr_cached": cached_ratio["tpr"],
        "fpr_cached": cached_ratio["fpr"],
        "auroc_cached_recomputed_from_cache_files": cached_report["auroc"],
        "per_split": per_split,
    }
    return gate


# The pipeline's own loop, compute_dropout_pass_probs, seeds once per split and
# draws k passes per batch in loader order. Reproducing that order is what makes
# the masks, and so the per-sample PSU, comparable row for row.
@torch.inference_mode()
def per_pass_probs(model, loader, baseline_labels, passes, seed, use_bfloat16, device):
    seed_everything(seed, verbose=False)
    columns = []
    offset = 0
    for images, _ in loader:
        labels = baseline_labels[offset : offset + len(images)].to(device)  # (b,)
        offset += len(images)
        pass_probs = []
        for _ in range(passes):
            probs = forward_probs(model, images, device, use_bfloat16)  # (b, classes)
            pass_probs.append(probs.gather(1, labels[:, None]).squeeze(1).cpu())
        columns.append(torch.stack(pass_probs))  # (k, b)
    stacked = torch.cat(columns, dim=1).float()  # (k, n)
    return stacked


@torch.inference_mode()
def split_probs(model, loader, device, use_bfloat16):
    probs, loader_labels = [], []
    for images, labels in loader:
        probs.append(forward_probs(model, images, device, use_bfloat16).cpu())
        loader_labels.append(labels.long())
    probs = torch.cat(probs)  # (n, classes)
    baseline = {
        "probs": probs,
        "labels": probs.argmax(dim=1),  # (n,)
        "loader_labels": torch.cat(loader_labels),  # (n,)
    }
    return baseline


# The helpers predict trigger tokens from the attack's own trigger on a random
# base image. The gate compares that with the pixels the loader's triggered image
# actually differs from its clean twin in, at every grid. On the unperturbed
# model with the block input stream difference at the first block of each stage.
@torch.inference_mode()
def trigger_gate(
    model, architecture, metadata, clean_set, backdoor_set, manifest, device
):
    predicted_map = trigger_pixel_map(metadata)  # (224, 224)
    row_of = {o: r for r, o in enumerate(manifest["analysis_clean_indices"])}
    grids = sorted(set(BLOCK_GRIDS[architecture]) | {CELL_GRID})

    pair_checks = []
    for row in range(GATE_PAIRS):
        triggered, _ = backdoor_set[row]
        clean, _ = clean_set[row_of[manifest["analysis_backdoor_indices"][row]]]
        delta = (triggered - clean).abs().sum(dim=0)[None, None]  # (1, 1, s, s)
        actual_map = F.interpolate(
            delta, size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), mode="bilinear"
        )[0, 0]  # (224, 224)
        checks = {}
        for grid in grids:
            predicted = set(touched_tokens(predicted_map, grid).tolist())
            actual = set(touched_tokens(actual_map, grid).tolist())
            checks[str(grid)] = {
                "actual_outside_predicted": len(actual - predicted),
                "predicted_not_changed": len(predicted - actual),
                "predicted": len(predicted),
            }
        pair_checks.append(checks)

    stream = stream_difference_by_stage(
        model, architecture, predicted_map, clean_set, backdoor_set, manifest, device
    )
    gate = {
        "pairs": GATE_PAIRS,
        "per_grid": {
            str(grid): {
                "pairs_with_change_outside_predicted": sum(
                    1 for c in pair_checks if c[str(grid)]["actual_outside_predicted"]
                ),
                "mean_predicted_tokens_unchanged": sum(
                    c[str(grid)]["predicted_not_changed"] for c in pair_checks
                )
                / len(pair_checks),
                "predicted_tokens": pair_checks[0][str(grid)]["predicted"],
            }
            for grid in grids
        },
        "stream_difference_by_stage": stream,
    }
    return gate


def stream_difference_by_stage(
    model, architecture, predicted_map, clean_set, backdoor_set, manifest, device
):
    blocks = model_blocks(model, architecture)
    grids = BLOCK_GRIDS[architecture]
    first_of_stage = [i for i, g in enumerate(grids) if i == 0 or grids[i - 1] != g]
    row_of = {o: r for r, o in enumerate(manifest["analysis_clean_indices"])}
    rows = range(min(16, len(backdoor_set)))
    triggered = torch.stack([backdoor_set[r][0] for r in rows]).to(device)
    clean = torch.stack(
        [clean_set[row_of[manifest["analysis_backdoor_indices"][r]]][0] for r in rows]
    ).to(device)  # (16, 3, s, s)

    captured = {}
    handles = [
        blocks[i].register_forward_pre_hook(
            lambda _m, args, i=i: captured.__setitem__(i, args[0].float())
        )
        for i in first_of_stage
    ]
    try:
        model(clean)
        clean_stream = dict(captured)
        model(triggered)
        triggered_stream = dict(captured)
    finally:
        for handle in handles:
            handle.remove()

    stages = {}
    offset = 1 if architecture == "vit" else 0
    for i in first_of_stage:
        difference = (triggered_stream[i] - clean_stream[i]).float()
        per_token = (
            difference.reshape(len(rows), -1, difference.shape[-1])
            .norm(dim=2)[:, offset:]
            .mean(dim=0)
        )  # (grid * grid,)
        predicted = touched_tokens(predicted_map, grids[i])  # (count,)
        top = per_token.topk(len(predicted)).indices  # (count,)
        stages[f"block_{i + 1}_grid_{grids[i]}"] = {
            "predicted_tokens": len(predicted),
            "top_difference_tokens_that_are_predicted": int(
                torch.isin(top.cpu(), predicted).sum()
            ),
            "share_of_difference_on_predicted": float(
                per_token[predicted.to(per_token.device)].sum() / per_token.sum()
            ),
        }
    return stages


def hook_count(model):
    count = sum(
        len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()
    )
    return count


def forward_overrides(model):
    overridden = [name for name, m in model.named_modules() if "forward" in m.__dict__]
    return overridden


# PSBD's probe must be the only stochastic element. build_vit sets dropout to 0,
# Swin's stochastic depth is an identity in eval mode, and plug_dropout's probes
# live outside the module tree, so the model itself must be entirely in eval.
def model_state(model):
    dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    state = {
        "modules_in_training_mode": sum(1 for m in model.modules() if m.training),
        "dropout_modules": len(dropouts),
        "dropout_modules_with_nonzero_rate": sum(1 for m in dropouts if m.p > 0),
    }
    return state


def headline(record):
    lines = {"folder": record["folder"], "gate_1": record["gate_1_accuracy"]}
    for key, value in record.items():
        if key.startswith("gate_2"):
            lines[key] = {
                k: value[k] for k in ("auroc_ours", "auroc_cached", "rate", "passes")
            } | {
                split: round(value["per_split"][split]["mean_absolute_difference"], 5)
                for split in SPLITS
            }
    lines["gate_3"] = record["gate_3_trigger_tokens"]
    lines["gate_4"] = record["gate_4_hooks_after"]
    return lines


if __name__ == "__main__":
    main()
