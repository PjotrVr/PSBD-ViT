"""Which part of PSBD-TM does the work: the site, the operator or how hard it perturbs.

PSBD-TM is token_mask at before_attention_norm. It could win because it sits at the
attention input, because it removes whole tokens, because a LayerNorm follows it, or
only because it perturbs harder than the placements it is compared with. Each
explanation predicts a different pattern when 1 factor changes and the others stay
fixed, and this script measures those patterns in 2 ways.

    cache   no GPU. Every cached placement's AUROC and its clean and triggered
            shift ratios, at the adaptive rate and at the matched clean shift of
            0.6, on the ViT panel and the Swin panel, with paired contrasts that
            change 1 factor at a time
    causal  GPU, patch-trigger models only. For each site and operator the rate
            is set so the operator changes 60% of clean predictions on the
            model's own paired images, then the operator runs on all tokens, on
            the trigger's tokens only, on as many random tokens, and on every
            token but the trigger's. With every operator on all tokens, the
            share of the trigger's own signal that survives in what each block's
            attention reads and in the residual stream is recorded per block
    depth   GPU, ViT patch models: the trigger tokens masked at the attention
            input in 1 block only, or in every block but 1 (swin.py does Swin)

    source .venv/bin/activate
    PYTHONPATH=. python experiments/why_token_masking_works/sites.py --cache-only
    PYTHONPATH=. python experiments/why_token_masking_works/sites.py \
        --folders vit_gtsrb_badnet_a2o_0_05
    PYTHONPATH=. python experiments/why_token_masking_works/sites.py --summarize-only
"""

import argparse
import collections
import json
import math
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
from lightning import seed_everything  # noqa: E402

from cli.compare_detectors import psbd_rate  # noqa: E402
from data.splits import read_checkpoint_metadata  # noqa: E402
from defenses.decision import PLACEMENT_MATCH_TARGET  # noqa: E402
from defenses.operators import build_operator  # noqa: E402
from experiments._paths import experiment_results_dir  # noqa: E402
from experiments.why_token_masking_works.measure import (  # noqa: E402
    RecordingTokenMask,
    kept_by_count,
    limit_gpu_memory,
    load_pairs,
    mean_readout,
    predict_batch,
    predict_pairs,
    readout,
    write_json,
)
from experiments.why_token_masking_works.swin import (  # noqa: E402
    assert_pristine,
    read_json_or_none,
)
from experiments.why_token_masking_works.swin import (  # noqa: E402
    panel_cells as swin_panel_cells,
)
from experiments.why_token_masking_works.swin import (  # noqa: E402
    panel_folders as swin_panel_folders,
)
from experiments.why_token_masking_works.tokens import (  # noqa: E402
    BLOCK_GRIDS,
    successful_vit_folders,
    cached_baseline_agreement,
    model_seed,
    CLS_OFFSET,
    FixedTokenMask,
    PositionRestricted,
    attach_readers,
    block_patch_counts,
    block_trigger_positions,
    complement_positions,
    example_images,
    plug_per_block,
    random_other_positions,
    signal_retained,
    touched_tokens,
    trigger_pixel_map,
)
from models.backbones import load_checkpoint  # noqa: E402
from models.positions import plug_dropout, unplug_dropout  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    HEADLINE_KEY,
    bootstrap_ci,
    clearing_cells,
    load_coverage,
    load_psbd_metrics,
)

SLUG = "why_token_masking_works"
SUBDIRECTORY = "sites"
PAIR_COUNT = 256
FORWARD_PASSES = 6
RANDOM_DRAWS = 2
CLEAN_CHANGE_TARGET = PLACEMENT_MATCH_TARGET
CALIBRATION_PASSES = 2
CALIBRATION_STEPS = 7
# A mask or dropout rate lives in [0, 1). The Gaussian's rate is a standard
# deviation relative to the activation's own spread and the cached sweeps go up
# to 8, so it is searched on a log scale up to 20.
RATE_BOUNDS = {
    "token_mask": (0.0, 0.95),
    "token_substitute": (0.0, 0.95),
    "dropout": (0.0, 0.95),
    "channel_mask": (0.0, 0.95),
    "gaussian": (0.01, 20.0),
    "rademacher": (0.01, 20.0),
}
# The 5 sites the causal test compares, as position tuples the registry knows.
# stream is post_residual, both adds, which with dropout is PSBD-RD itself.
SITES = {
    "attention_input": ("before_attention_norm",),
    "attention_input_after_norm": ("before_attention",),
    "attention_output": ("before_attention_residual",),
    "stream": ("after_attention_residual", "after_mlp_residual"),
    "mlp_input": ("before_mlp_norm",),
    "mlp_input_after_norm": ("before_mlp",),
}
OPERATOR_NAMES = (
    "token_mask",
    "token_substitute",
    "dropout",
    "channel_mask",
    "gaussian",
    "rademacher",
)
EMBEDDING_PLACEMENT = "after_embedding_token_mask"
EMBEDDING_PASSES = 10
# Most decisive first, so a model cut short by the GPU window still answers the
# main questions: PSBD-TM against PSBD-RD, then the operator at PSBD-TM's site,
# then token masking at every other site, then the rest of the grid.
CAUSAL_ORDER = (
    ("attention_input", "token_mask"),
    ("stream", "dropout"),
    ("attention_input", "dropout"),
    ("attention_input", "gaussian"),
    ("attention_input", "channel_mask"),
    ("stream", "token_mask"),
    ("attention_output", "token_mask"),
    ("attention_input_after_norm", "token_mask"),
    ("mlp_input", "token_mask"),
    ("attention_input_after_norm", "gaussian"),
    ("attention_input_after_norm", "dropout"),
    ("mlp_input", "dropout"),
    ("mlp_input", "gaussian"),
    ("attention_output", "dropout"),
    ("stream", "gaussian"),
    ("stream", "channel_mask"),
    ("attention_output", "gaussian"),
    ("attention_output", "channel_mask"),
    ("attention_input_after_norm", "channel_mask"),
    ("mlp_input", "channel_mask"),
    # The literature memo's L22 and L2 (docs/why-psbd-works-literature.md):
    # substitution keeps a removed token on the data manifold, and Rademacher
    # noise has the Gaussian's covariance, so each should match its twin if the
    # respective account is right. before_mlp is where no norm follows the noise.
    ("attention_input", "token_substitute"),
    ("attention_input", "rademacher"),
    ("mlp_input_after_norm", "gaussian"),
    ("mlp_input_after_norm", "rademacher"),
    ("mlp_input_after_norm", "token_mask"),
)
PATCH_ATTACKS = ("badnet_a2o", "tact", "lc")
LOCAL_ATTACKS = ("badnet_a2o", "badnet_a2a", "tact", "lc")
MASK_SEED = 0
CALIBRATION_SEED = 1000
REPRODUCTION_FLOOR = 0.98
INTACT_SHARE = 0.5
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 0
MIN_PAIRED_MODELS = 5
GPU_MEMORY_GB = 14.0
BATCH_SIZE = 256

# Every cached placement the contrasts read, by (site, operator). A placement id
# is the position with the operator appended, bare for dropout.
CACHE_POSITIONS = {
    "embedding": "after_embedding",
    "attention_input": "before_attention_norm",
    "attention_input_after_norm": "before_attention",
    "attention_output": "before_attention_residual",
    "stream_after_attention": "after_attention_residual",
    "stream": "post_residual",
    "mlp_input": "before_mlp_norm",
    "mlp_input_after_norm": "before_mlp",
    "mlp_output": "before_mlp_residual",
    "stream_after_mlp": "after_mlp_residual",
}
# Each contrast changes 1 factor and names the explanation it tests. It reads
# placement a minus placement b.
CONTRASTS = (
    ("psbd_tm_vs_psbd_rd", "attention_input", "token_mask", "stream", "dropout"),
    (
        "token_mask_attention_input_vs_stream",
        "attention_input",
        "token_mask",
        "stream_after_attention",
        "token_mask",
    ),
    (
        "token_mask_attention_input_vs_attention_output",
        "attention_input",
        "token_mask",
        "attention_output",
        "token_mask",
    ),
    (
        "token_mask_before_vs_after_norm",
        "attention_input",
        "token_mask",
        "attention_input_after_norm",
        "token_mask",
    ),
    (
        "token_mask_attention_input_vs_mlp_input",
        "attention_input",
        "token_mask",
        "mlp_input",
        "token_mask",
    ),
    (
        "token_mask_attention_input_vs_embedding",
        "attention_input",
        "token_mask",
        "embedding",
        "token_mask",
    ),
    (
        "attention_input_token_mask_vs_dropout",
        "attention_input",
        "token_mask",
        "attention_input",
        "dropout",
    ),
    (
        "attention_input_token_mask_vs_channel_mask",
        "attention_input",
        "token_mask",
        "attention_input",
        "channel_mask",
    ),
    (
        "attention_input_token_mask_vs_gaussian",
        "attention_input",
        "token_mask",
        "attention_input",
        "gaussian",
    ),
    (
        "stream_after_attention_token_mask_vs_dropout",
        "stream_after_attention",
        "token_mask",
        "stream_after_attention",
        "dropout",
    ),
    (
        "dropout_attention_input_vs_stream",
        "attention_input",
        "dropout",
        "stream",
        "dropout",
    ),
    (
        "dropout_before_vs_after_norm",
        "attention_input",
        "dropout",
        "attention_input_after_norm",
        "dropout",
    ),
    (
        "gaussian_before_vs_after_norm",
        "attention_input",
        "gaussian",
        "attention_input_after_norm",
        "gaussian",
    ),
    (
        "channel_mask_before_vs_after_norm",
        "attention_input",
        "channel_mask",
        "attention_input_after_norm",
        "channel_mask",
    ),
    (
        "dropout_attention_input_vs_mlp_input",
        "attention_input",
        "dropout",
        "mlp_input",
        "dropout",
    ),
    (
        "gaussian_attention_input_vs_mlp_input",
        "attention_input",
        "gaussian",
        "mlp_input",
        "gaussian",
    ),
    (
        "channel_mask_attention_input_vs_mlp_input",
        "attention_input",
        "channel_mask",
        "mlp_input",
        "channel_mask",
    ),
)
RULES = ("adaptive", "matched")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--folders", nargs="+", default=None)
    parser.add_argument("--architecture", choices=("vit", "swin"), default=None)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--pairs", type=int, default=PAIR_COUNT)
    parser.add_argument("--passes", type=int, default=FORWARD_PASSES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gpu-memory-gb", type=float, default=GPU_MEMORY_GB)
    parser.add_argument("--list-folders", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = os.path.join(
        experiment_results_dir(SLUG, args.output_root), SUBDIRECTORY
    )
    os.makedirs(output_dir, exist_ok=True)
    folders_by_architecture = patch_panel(args.results_dir, args.checkpoints_dir)

    if args.list_folders:
        for architecture, folders in folders_by_architecture.items():
            if args.architecture in (None, architecture):
                print("\n".join(folders))
        return

    if args.cache_only:
        cache = summarize_cache(args.results_dir, args.checkpoints_dir)
        # Every placement's rate ladder for every model makes this record large,
        # so it is written without indentation and at 4 decimals.
        with open(os.path.join(output_dir, "cache.json"), "w") as handle:
            json.dump(rounded(cache), handle, separators=(",", ":"))
        write_json(
            os.path.join(output_dir, "evidence.json"),
            collect_evidence(args.results_dir),
        )
        print(cache_tables(cache))
        return

    if not args.summarize_only:
        device = torch.device("cuda", torch.cuda.current_device())
        limit_gpu_memory(args.gpu_memory_gb, device)
        every_folder = [f for fs in folders_by_architecture.values() for f in fs]
        for folder in args.folders or every_folder:
            assert folder in every_folder, f"{folder} is not a patch panel model"
            out_path = os.path.join(output_dir, f"{folder}.json")
            started = time.time()
            measure_model(folder, out_path, args, device)
            print(f"[ok] {folder} {time.time() - started:.0f}s", flush=True)

    # A record of a model that has since left the panel stays on disk and out of
    # the summary.
    panel = {f for fs in folders_by_architecture.values() for f in fs}
    summary = summarize_causal(output_dir, panel)
    write_json(os.path.join(output_dir, "summary.json"), summary)
    print(causal_tables(summary))


# Writes the model's JSON after every placement, so a run stopped by the GPU
# window resumes at the first placement it has not measured.
def measure_model(folder, out_path, args, device):
    checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)
    architecture = metadata["architecture"]
    model = load_checkpoint(architecture, checkpoint_path, device).eval()
    assert_pristine(model)
    pairs = load_pairs(checkpoint_path, args, device)

    pixel_map = trigger_pixel_map(metadata)  # (224, 224)
    trigger = block_trigger_positions(pixel_map, architecture)
    baseline = predict_pairs(model, pairs, args.batch_size, device)
    pairs["clean_base"] = baseline["clean"]  # (n,)
    pairs["triggered_base"] = baseline["triggered"]  # (n,)
    agreement = cached_baseline_agreement(args.results_dir, folder, pairs)
    assert min(agreement.values()) >= REPRODUCTION_FLOOR, f"{folder} {agreement}"
    draw_seed = model_seed(folder)

    record = read_json_or_none(out_path) or {
        "folder": folder,
        "architecture": architecture,
        "attack": metadata["attack"],
        "dataset": metadata["dataset"],
        "poison_rate": metadata["poison_rate"],
        "pairs": int(len(pairs["targets"])),
        "passes": args.passes,
        "trigger_positions_by_grid": {
            str(grid): touched_tokens(pixel_map, grid).tolist()
            for grid in sorted(set(BLOCK_GRIDS[architecture]))
        },
        "trigger_tokens_per_block": [len(p) for p in trigger],
        "draw_seed": draw_seed,
        "baseline_agreement_with_cache": agreement,
        "example": example_images(pairs, metadata["dataset"]),
        "baseline": readout(
            pairs, baseline["clean"][None], baseline["triggered"][None]
        ),
        "placements": {},
    }

    if architecture == "vit" and "depth_profile" not in record:
        record["depth_profile"] = measure_depth_profile(
            model, architecture, pairs, trigger, args.batch_size, device
        )
        write_json(out_path, record)

    # At the adaptive rate (0.6 to 0.8) almost no draw leaves 3 or 4 trigger
    # tokens, so the same record is also taken at rate 0.5, where every count
    # from 0 to 4 is drawn often enough to read.
    embedding_rates = {
        "embedding_masks": adaptive_rate_of(
            args.results_dir, folder, EMBEDDING_PLACEMENT
        ),
        "embedding_masks_rate_0_5": 0.5,
    }
    for key, rate in embedding_rates.items():
        if key in record or rate is None:
            continue
        record[key] = measure_embedding_masks(
            model, architecture, pairs, trigger[0], rate, args.batch_size, device
        )
        assert_pristine(model)
        write_json(out_path, record)

    reference = None
    for site, operator in CAUSAL_ORDER:
        key = f"{site}__{operator}"
        if key in record["placements"]:
            continue
        if reference is None:
            reference = unperturbed_reads(
                model, architecture, pairs, trigger, args.batch_size, device
            )
        started = time.time()
        record["placements"][key] = measure_placement(
            model,
            architecture,
            pairs,
            trigger,
            site,
            operator,
            reference,
            draw_seed,
            args,
            device,
        )
        record["placements"][key]["seconds"] = round(time.time() - started, 1)
        assert_pristine(model)
        write_json(out_path, record)
        print(f"  {folder} {key} {time.time() - started:.0f}s", flush=True)
    return record


def measure_placement(
    model,
    architecture,
    pairs,
    trigger,
    site,
    operator,
    reference,
    draw_seed,
    args,
    device,
):
    positions = SITES[site]
    rate, calibration = calibrate_rate(
        model, architecture, pairs, positions, operator, args.batch_size, device
    )

    patch_counts = block_patch_counts(architecture)
    offset = CLS_OFFSET[architecture]
    rest = [complement_positions(t, n, offset) for t, n in zip(trigger, patch_counts)]

    result = {
        "site": site,
        "operator": operator,
        "positions": list(positions),
        "rate": rate,
        "calibration": calibration,
    }
    result["all_tokens"] = all_tokens_readout(
        model,
        architecture,
        pairs,
        trigger,
        positions,
        operator,
        rate,
        reference,
        args.passes,
        args.batch_size,
        device,
    )
    result["trigger_only"] = restricted_readout(
        model, architecture, pairs, positions, operator, rate, trigger, args, device
    )
    result["all_but_trigger"] = restricted_readout(
        model, architecture, pairs, positions, operator, rate, rest, args, device
    )

    # The control touches as many tokens as the trigger covers in every block,
    # drawn from the tokens it does not cover, at the same rate and pass count.
    draws = []
    for draw in range(RANDOM_DRAWS):
        random_positions = [
            random_other_positions(t, len(t), n, offset, draw_seed + draw)
            for t, n in zip(trigger, patch_counts)
        ]
        draws.append(
            restricted_readout(
                model,
                architecture,
                pairs,
                positions,
                operator,
                rate,
                random_positions,
                args,
                device,
            )
        )
    result["random_same_count"] = mean_readout(draws)
    return result


# The rate at which the operator changes CLEAN_CHANGE_TARGET of the model's own
# clean predictions, on the same paired clean images every reading uses. The
# cached matched-shift rate is the nearest point of a coarse ladder and lands
# anywhere from 0.45 to 0.76 on these models, which is too loose to compare how
# much of the trigger survives at equal clean disturbance. Bisection assumes the
# clean change grows with the rate, which holds for every operator here.
def calibrate_rate(model, architecture, pairs, positions, operator, batch_size, device):
    low, high = RATE_BOUNDS[operator]
    geometric = operator == "gaussian"
    evaluated = {}

    def clean_change(rate):
        if rate not in evaluated:
            evaluated[rate] = clean_change_at(
                model,
                architecture,
                pairs,
                positions,
                operator,
                rate,
                batch_size,
                device,
            )
        return evaluated[rate]

    reached = clean_change(high) >= CLEAN_CHANGE_TARGET
    for _ in range(CALIBRATION_STEPS if reached else 0):
        middle = math.sqrt(low * high) if geometric else (low + high) / 2
        if clean_change(middle) < CLEAN_CHANGE_TARGET:
            low = middle
        else:
            high = middle

    rate = min(evaluated, key=lambda r: abs(evaluated[r] - CLEAN_CHANGE_TARGET))
    calibration = {
        "target": CLEAN_CHANGE_TARGET,
        "reached": reached,
        "clean_change_at_rate": evaluated[rate],
        "evaluated": {str(r): v for r, v in sorted(evaluated.items())},
    }
    return rate, calibration


def clean_change_at(
    model, architecture, pairs, positions, operator, rate, batch_size, device
):
    handles = plug_dropout(
        model,
        architecture,
        positions,
        {name: build_operator(operator) for name in positions},
        rate,
    )
    try:
        seed_everything(CALIBRATION_SEED, verbose=False)
        predictions = torch.stack(
            [
                predict_images(model, pairs["clean"], batch_size, device)
                for _ in range(CALIBRATION_PASSES)
            ]
        )  # (k, n)
    finally:
        unplug_dropout(handles)

    changed = float((predictions != pairs["clean_base"][None]).float().mean())
    return changed


# The library operator on every token, with the attention input and the stream
# at the trigger tokens recorded in every block. Each pass reseeds before the
# clean and before the triggered images, so both draw the same masks and the
# difference between them isolates what the trigger still contributes.
def all_tokens_readout(
    model,
    architecture,
    pairs,
    trigger,
    positions,
    operator,
    rate,
    reference,
    passes,
    batch_size,
    device,
):
    handles = plug_dropout(
        model,
        architecture,
        positions,
        {name: build_operator(operator) for name in positions},
        rate,
    )
    store, reader_handles = attach_readers(model, architecture, trigger)
    blocks = len(trigger)
    totals = {
        kind: {
            "aligned": [0.0] * blocks,
            "energy": [0.0] * blocks,
            "best": [[] for _ in range(blocks)],
        }
        for kind in ("read", "stream")
    }
    try:
        clean_passes, triggered_passes = [], []
        for pass_index in range(passes):
            seed_everything(MASK_SEED + pass_index, verbose=False)
            clean_predictions, clean_reads = predict_with_reads(
                model, pairs["clean"], store, batch_size, device
            )
            seed_everything(MASK_SEED + pass_index, verbose=False)
            triggered_predictions, triggered_reads = predict_with_reads(
                model, pairs["triggered"], store, batch_size, device
            )
            clean_passes.append(clean_predictions)
            triggered_passes.append(triggered_predictions)
            for kind in ("read", "stream"):
                for b in range(blocks):
                    retained = signal_retained(
                        triggered_reads[kind][b],
                        clean_reads[kind][b],
                        reference["triggered"][kind][b],
                        reference["clean"][kind][b],
                    )
                    totals[kind]["aligned"][b] += float(retained["aligned"].sum())
                    totals[kind]["energy"][b] += float(retained["energy"].sum())
                    totals[kind]["best"][b].append(retained["best_token"].cpu())
    finally:
        for handle in reader_handles:
            handle.remove()
        unplug_dropout(handles)

    reading = readout(pairs, torch.stack(clean_passes), torch.stack(triggered_passes))
    for kind in ("read", "stream"):
        best = [torch.cat(values) for values in totals[kind]["best"]]  # blocks x (k*n,)
        reading[f"{kind}_signal_retained"] = [
            a / e if e > 0 else None
            for a, e in zip(totals[kind]["aligned"], totals[kind]["energy"])
        ]
        reading[f"{kind}_best_token_median"] = [float(v.median()) for v in best]
        reading[f"{kind}_best_token_intact_share"] = [
            float((v >= INTACT_SHARE).float().mean()) for v in best
        ]
    return reading


def restricted_readout(
    model, architecture, pairs, positions, operator, rate, restricted_to, args, device
):
    factory = build_operator(operator)
    handles = plug_per_block(
        model,
        architecture,
        positions,
        lambda b, probe_rate: PositionRestricted(factory(probe_rate), restricted_to[b]),
        rate,
    )
    try:
        clean_passes, triggered_passes = [], []
        for pass_index in range(args.passes):
            seed_everything(MASK_SEED + pass_index, verbose=False)
            clean_passes.append(
                predict_images(model, pairs["clean"], args.batch_size, device)
            )
            seed_everything(MASK_SEED + pass_index, verbose=False)
            triggered_passes.append(
                predict_images(model, pairs["triggered"], args.batch_size, device)
            )
    finally:
        unplug_dropout(handles)

    reading = readout(pairs, torch.stack(clean_passes), torch.stack(triggered_passes))
    return reading


def unperturbed_reads(model, architecture, pairs, trigger, batch_size, device):
    store, handles = attach_readers(model, architecture, trigger)
    try:
        _, clean = predict_with_reads(model, pairs["clean"], store, batch_size, device)
        _, triggered = predict_with_reads(
            model, pairs["triggered"], store, batch_size, device
        )
    finally:
        for handle in handles:
            handle.remove()
    reference = {"clean": clean, "triggered": triggered}
    return reference


def predict_with_reads(model, images, store, batch_size, device):
    predictions = []
    reads = {
        "read": collections.defaultdict(list),
        "stream": collections.defaultdict(list),
    }
    for start in range(0, len(images), batch_size):
        predictions.append(
            predict_batch(model, images[start : start + batch_size], device)
        )
        for kind in ("read", "stream"):
            for block, value in store[kind].items():
                reads[kind][block].append(value)  # (b, count, channels)
    stacked = {
        kind: [torch.cat(reads[kind][b]) for b in sorted(reads[kind])]
        for kind in ("read", "stream")
    }  # blocks x (n, count, channels)
    all_predictions = torch.cat(predictions)  # (n,)
    return all_predictions, stacked


def predict_images(model, images, batch_size, device):
    parts = [
        predict_batch(model, images[start : start + batch_size], device)
        for start in range(0, len(images), batch_size)
    ]
    predictions = torch.cat(parts)  # (n,)
    return predictions


# The trigger tokens masked at the attention input in exactly 1 block, and in
# every block except 1. The first says which single block's read the prediction
# needs, the second which single block's read is enough on its own.
def measure_depth_profile(model, architecture, pairs, trigger, batch_size, device):
    blocks = len(trigger)
    profile = {"only_block": [], "all_but_block": []}
    for block in range(1, blocks + 1):
        for name, masked_blocks in (
            ("only_block", [block]),
            ("all_but_block", [b for b in range(1, blocks + 1) if b != block]),
        ):
            handles = []
            for b in masked_blocks:
                handles += plug_dropout(
                    model,
                    architecture,
                    ("before_attention_norm",),
                    {
                        "before_attention_norm": (
                            lambda _rate, p=trigger[b - 1]: FixedTokenMask(
                                p, architecture == "vit"
                            )
                        )
                    },
                    0.0,
                    block_range=(b, b),
                )
            try:
                predictions = predict_pairs(model, pairs, batch_size, device)
            finally:
                unplug_dropout(handles)
            reading = readout(
                pairs, predictions["clean"][None], predictions["triggered"][None]
            )
            profile[name].append(reading["triggered_kept"])

    # The literature memo's L24: if the trigger token acts as a sink the class
    # token needs nothing else late, so masking every other patch token in
    # blocks 9 to 12 should keep the triggered prediction and break the clean one.
    offset = CLS_OFFSET[architecture]
    handles = []
    for b in range(blocks - 3, blocks + 1):
        patches = torch.arange(offset, offset + BLOCK_GRIDS[architecture][b - 1] ** 2)
        others = patches[~torch.isin(patches, trigger[b - 1].cpu())]  # (patches - t,)
        handles += plug_dropout(
            model,
            architecture,
            ("before_attention_norm",),
            {
                "before_attention_norm": (
                    lambda _rate, p=others: FixedTokenMask(p, architecture == "vit")
                )
            },
            0.0,
            block_range=(b, b),
        )
    try:
        predictions = predict_pairs(model, pairs, batch_size, device)
    finally:
        unplug_dropout(handles)
    profile["only_trigger_visible_last_4"] = readout(
        pairs, predictions["clean"][None], predictions["triggered"][None]
    )
    return profile


# The literature memo's L18. Doan et al. drop input patches once, before the
# first block, and flag a prediction that changes. That is the embedding site,
# where a masked token loses its content for the whole network. The masks the
# cached after_embedding_token_mask probe draws are recorded, and survival is
# tabulated by how many trigger tokens the draw left.
def measure_embedding_masks(
    model, architecture, pairs, trigger, rate, batch_size, device
):
    recorders = []

    def build_recorder(probe_rate):
        recorder = RecordingTokenMask(probe_rate)
        recorders.append(recorder)
        return recorder

    handles = plug_dropout(
        model,
        architecture,
        ("after_embedding",),
        {"after_embedding": build_recorder},
        rate,
    )
    assert len(recorders) == 1, "after_embedding attaches once per model"
    columns = {"clean": [], "triggered": []}
    try:
        for split in ("clean", "triggered"):
            seed_everything(MASK_SEED, verbose=False)
            for _ in range(EMBEDDING_PASSES):
                predictions, survivors = [], []
                for start in range(0, len(pairs[split]), batch_size):
                    batch = pairs[split][start : start + batch_size]
                    predictions.append(predict_batch(model, batch, device))
                    dropped = recorders[0].last_dropped[:, trigger.to(device)]  # (b, t)
                    survivors.append((~dropped).sum(dim=1))  # (b,)
                columns[split].append((torch.cat(predictions), torch.cat(survivors)))
    finally:
        unplug_dropout(handles)

    clean_predictions = torch.stack([p for p, _ in columns["clean"]])  # (k, n)
    triggered_predictions = torch.stack([p for p, _ in columns["triggered"]])
    clean_survivors = torch.stack([s for _, s in columns["clean"]])  # (k, n)
    triggered_survivors = torch.stack([s for _, s in columns["triggered"]])
    hit = (pairs["triggered_base"] == pairs["targets"])[None].expand_as(
        triggered_predictions
    )  # (k, n)
    result = {
        "rate": rate,
        "trigger_tokens": len(trigger),
        "overall": readout(pairs, clean_predictions, triggered_predictions),
        "triggered_by_surviving_trigger_tokens": kept_by_count(
            triggered_predictions == pairs["targets"][None], hit, triggered_survivors
        ),
        # A heavily masked model sends most clean images to 1 default class,
        # often the target, so the share of clean images predicted as the
        # target is kept per count as the floor to subtract.
        "clean_on_target_by_surviving_trigger_tokens": kept_by_count(
            clean_predictions == pairs["targets"][None],
            torch.ones_like(hit),
            clean_survivors,
        ),
        "clean_by_surviving_trigger_tokens": kept_by_count(
            clean_predictions == pairs["clean_base"][None],
            torch.ones_like(hit),
            clean_survivors,
        ),
    }
    return result


def adaptive_rate_of(results_dir, folder, placement):
    report = load_psbd_metrics(results_dir, folder) or {}
    block = report.get("placements", {}).get(placement) or {}
    rate = block.get("adaptive_rate")
    return rate


def patch_panel(results_dir, checkpoints_dir):
    successful = successful_vit_folders(results_dir)
    vit_folders = [
        c["folder_name"]
        for c in clearing_cells(load_coverage(results_dir))
        if c["attack"] in PATCH_ATTACKS
        and c["folder_name"] in successful
        and load_psbd_metrics(results_dir, c["folder_name"]) is not None
    ]
    swin_patch, _ = swin_panel_folders(results_dir, checkpoints_dir)
    panel = {"vit": vit_folders, "swin": swin_patch}
    return panel


def summarize_causal(output_dir, panel):
    records = []
    for name in sorted(os.listdir(output_dir)):
        if name.endswith(".json") and name not in (
            "summary.json",
            "cache.json",
            "evidence.json",
        ):
            with open(os.path.join(output_dir, name)) as handle:
                records.append(json.load(handle))
    records = [record for record in records if record["folder"] in panel]

    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for record in records:
        group = f"{record['architecture']}_{causal_group(record)}"
        for key, placement in record["placements"].items():
            grouped[group][key].append(placement)

    summary = {
        "models": {},
        "placements": {},
        "depth_profile": {},
        "embedding_masks": pooled_embedding_masks(records, "embedding_masks"),
        "embedding_masks_rate_0_5": pooled_embedding_masks(
            records, "embedding_masks_rate_0_5"
        ),
    }
    for group, placements in grouped.items():
        summary["models"][group] = sorted(
            r["folder"]
            for r in records
            if f"{r['architecture']}_{causal_group(r)}" == group
        )
        summary["placements"][group] = {
            key: summarize_placement(values) for key, values in placements.items()
        }
    for record in records:
        if "depth_profile" in record:
            summary["depth_profile"].setdefault(
                f"{record['architecture']}_{causal_group(record)}", []
            ).append(record["depth_profile"])
    summary["depth_profile"] = {
        group: {
            name: [
                sum(column) / len(column)
                for column in zip(*[p[name] for p in profiles])
            ]
            for name in ("only_block", "all_but_block")
        }
        | {
            "only_trigger_visible_last_4": mean_readout(
                [p["only_trigger_visible_last_4"] for p in profiles]
            )
        }
        for group, profiles in summary["depth_profile"].items()
    }
    return summary


def pooled_embedding_masks(records, key):
    grouped = collections.defaultdict(list)
    for record in records:
        if key in record:
            size = record[key]["trigger_tokens"]
            group = f"{record['architecture']}_{causal_group(record)}_{size}_tokens"
            grouped[group].append(record[key])
    pooled = {}
    for group, blocks in grouped.items():
        pooled[group] = {"models": len(blocks)}
        for split in ("triggered", "clean", "clean_on_target"):
            counts = collections.defaultdict(lambda: [0, 0])
            for block in blocks:
                for value, (kept, total) in block.get(
                    f"{split}_by_surviving_trigger_tokens", {}
                ).items():
                    counts[value][0] += kept
                    counts[value][1] += total
            pooled[group][split] = {
                value: {"kept": kept / total, "n": total}
                for value, (kept, total) in sorted(
                    counts.items(), key=lambda i: int(i[0])
                )
            }
        pooled[group]["overall"] = mean_readout([b["overall"] for b in blocks])
    return pooled


def causal_group(record):
    if record["attack"] != "tact":
        return record["attack"]
    conditional = record["baseline"]["clean_accuracy"] >= 0.5
    group = "tact_trigger_conditional" if conditional else "tact_source_mapped"
    return group


def summarize_placement(values):
    variants = ("all_tokens", "trigger_only", "all_but_trigger", "random_same_count")
    summary = {
        "models": len(values),
        "rate_mean": sum(v["rate"] for v in values) / len(values),
        "calibration_reached": sum(1 for v in values if v["calibration"]["reached"]),
    }
    for variant in variants:
        readings = [v[variant] for v in values]
        summary[variant] = {
            key: sum(r[key] for r in readings) / len(readings)
            for key in (
                "triggered_kept",
                "clean_kept",
                "clean_on_target",
                "clean_accuracy",
            )
        }
    for curve in (
        "read_signal_retained",
        "read_best_token_intact_share",
        "stream_signal_retained",
        "stream_best_token_intact_share",
    ):
        curves = [v["all_tokens"][curve] for v in values]
        summary["all_tokens"][curve] = [
            mean_of_present([c[b] for c in curves]) for b in range(len(curves[0]))
        ]
    return summary


def mean_of_present(values):
    present = [v for v in values if v is not None]
    mean = sum(present) / len(present) if present else None
    return mean


# Earlier experiments already measured 3 facts the walkthrough reproduces as
# figures: the class token's attention on the trigger by layer
# (results/<folder>/cls_routing.json, the memo's L20), LayerNorm absorption per
# operator and position (layernorm_absorption.json, L25) and the test-time
# register ablation (test_time_registers.json and register_neurons.json, L24).
# They sit untracked next to each model's caches, so the numbers the notebook
# plots are copied into 1 tracked record here, unchanged.
def collect_evidence(results_dir):
    excluded = {
        c["folder_name"]
        for c in load_coverage(results_dir)["cells"]
        if c.get("asr_class") in ("diverged", "source_mapped")
    }
    successful = successful_vit_folders(results_dir)
    evidence = {"cls_routing": {}, "layernorm_absorption": {}, "registers": {}}
    for folder in sorted(os.listdir(results_dir)):
        # Benign references stay in as the control every routing figure needs.
        panel_or_benign = folder in successful or folder.endswith("_benign")
        if not folder.startswith("vit_") or folder in excluded or not panel_or_benign:
            continue
        for name, key in (
            ("cls_routing.json", "cls_routing"),
            ("layernorm_absorption.json", "layernorm_absorption"),
            ("test_time_registers.json", "registers"),
        ):
            path = os.path.join(results_dir, folder, name)
            if os.path.exists(path):
                with open(path) as handle:
                    evidence[key][folder] = json.load(handle)
        neurons = os.path.join(results_dir, folder, "register_neurons.json")
        if os.path.exists(neurons) and folder in evidence["registers"]:
            with open(neurons) as handle:
                evidence["registers"][folder]["register_neurons"] = json.load(handle)
    evidence["excluded_as_diverged_or_source_mapped"] = sorted(excluded)
    return evidence


def summarize_cache(results_dir, checkpoints_dir):
    reports = {
        "vit": [
            load_psbd_metrics(results_dir, c["folder_name"])
            for c in clearing_cells(load_coverage(results_dir))
            if c["folder_name"] in successful_vit_folders(results_dir)
        ],
        "swin": swin_reports(results_dir, checkpoints_dir),
    }
    cache = {
        "panel": {
            architecture: sorted(r["folder_name"] for r in architecture_reports if r)
            for architecture, architecture_reports in reports.items()
        }
    }
    for architecture, architecture_reports in reports.items():
        present = [r for r in architecture_reports if r is not None]
        rows = [model_row(report) for report in present]
        cache[architecture] = {
            "models": len(rows),
            "placements": placement_means(rows),
            "contrasts": contrast_table(rows),
            "per_model": rows,
        }
    return cache


def swin_reports(results_dir, checkpoints_dir):
    reports = [
        load_psbd_metrics(results_dir, cell["folder"])
        for cell in swin_panel_cells(results_dir, checkpoints_dir)
    ]
    return reports


# Per model and per cached placement: the rate each rule picked, the AUROC there,
# and the clean and triggered shift ratios at that rate. The shift ratio is the
# share of perturbed passes whose prediction left the unperturbed one, over the
# paired clean images and over the triggered images (defenses.scores.shift_ratio).
def model_row(report):
    row = {
        "folder": report["folder_name"],
        "attack": report["attack"],
        "dataset": report["dataset"],
        "poison_rate": report["poison_rate"],
        "locality": "local" if report["attack"] in LOCAL_ATTACKS else "global",
        "placements": {},
    }
    for site, position in CACHE_POSITIONS.items():
        for operator in OPERATOR_NAMES:
            placement = position if operator == "dropout" else f"{position}_{operator}"
            block = report["placements"].get(placement)
            if block is None:
                continue
            reading = {rule: rule_reading(block, rule) for rule in RULES}
            # The whole rate ladder, so a figure can compare operators at every
            # clean disturbance rather than at the 2 rates the rules pick.
            reading["ladder"] = [
                [
                    entry["rate"],
                    entry["shift_ratio"].get("validation"),
                    entry["shift_ratio"].get("clean"),
                    entry["shift_ratio"].get("backdoor"),
                    entry["detection_psu_ratio"][HEADLINE_KEY]["auroc"],
                ]
                for entry in block["rates"]
                if "detection_psu_ratio" in entry
            ]
            row["placements"][f"{site}__{operator}"] = reading
    return row


# The AUROC every table in the paper reads: fractional PSU at the 0.25 quantile,
# at the rate the rule picked (cli.compare_detectors.psbd_values). The block's
# own "adaptive" entry holds the absolute PSU and is not the headline.
def rule_reading(block, rule):
    rate = psbd_rate(block, rule)
    if rate is None:
        return None
    rate_entry = next((r for r in block["rates"] if r["rate"] == rate), None)
    if rate_entry is None:
        return None
    auroc = rate_entry["detection_psu_ratio"][HEADLINE_KEY]["auroc"]
    shifts = rate_entry["shift_ratio"]
    reading = {
        "rate": rate,
        "auroc": auroc,
        "shift_validation": shifts.get("validation"),
        "shift_clean": shifts.get("clean"),
        "shift_backdoor": shifts.get("backdoor"),
    }
    return reading


def placement_means(rows):
    means = {}
    keys = sorted({key for row in rows for key in row["placements"]})
    for key in keys:
        means[key] = {}
        for locality in ("all", "local", "global"):
            selected = [
                row["placements"][key]
                for row in rows
                if key in row["placements"] and locality in ("all", row["locality"])
            ]
            means[key][locality] = {
                rule: reading_means([s[rule] for s in selected if s[rule] is not None])
                for rule in RULES
            }
    return means


def reading_means(readings):
    fields = ("auroc", "rate", "shift_clean", "shift_backdoor", "shift_validation")
    means = {"n": len(readings)}
    for field in fields:
        values = [r[field] for r in readings if r.get(field) is not None]
        means[field] = sum(values) / len(values) if values else None
    return means


def contrast_table(rows):
    table = {}
    for name, site_a, operator_a, site_b, operator_b in CONTRASTS:
        key_a, key_b = f"{site_a}__{operator_a}", f"{site_b}__{operator_b}"
        table[name] = {"a": key_a, "b": key_b}
        for rule in RULES:
            for locality in ("all", "local", "global"):
                pairs = [
                    (row["placements"][key_a][rule], row["placements"][key_b][rule])
                    for row in rows
                    if key_a in row["placements"]
                    and key_b in row["placements"]
                    and row["placements"][key_a][rule] is not None
                    and row["placements"][key_b][rule] is not None
                    and locality in ("all", row["locality"])
                ]
                table[name][f"{rule}_{locality}"] = paired_summary(pairs)
    return table


def paired_summary(pairs):
    if len(pairs) < MIN_PAIRED_MODELS:
        return {"n": len(pairs)}
    auroc_gaps = [a["auroc"] - b["auroc"] for a, b in pairs]
    low, high = bootstrap_ci(auroc_gaps, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    backdoor_gaps = [
        a["shift_backdoor"] - b["shift_backdoor"]
        for a, b in pairs
        if a["shift_backdoor"] is not None and b["shift_backdoor"] is not None
    ]
    summary = {
        "n": len(pairs),
        "auroc_a": sum(a["auroc"] for a, _ in pairs) / len(pairs),
        "auroc_b": sum(b["auroc"] for _, b in pairs) / len(pairs),
        "auroc_gap": sum(auroc_gaps) / len(auroc_gaps),
        "auroc_gap_ci": [low, high],
        "a_wins": sum(1 for gap in auroc_gaps if gap > 0),
        "shift_clean_a": mean_of_present([a["shift_clean"] for a, _ in pairs]),
        "shift_clean_b": mean_of_present([b["shift_clean"] for _, b in pairs]),
        "shift_backdoor_a": mean_of_present([a["shift_backdoor"] for a, _ in pairs]),
        "shift_backdoor_b": mean_of_present([b["shift_backdoor"] for _, b in pairs]),
        "shift_backdoor_gap": mean_of_present(backdoor_gaps),
    }
    return summary


def cache_tables(cache):
    lines = []
    for architecture, block in cache.items():
        if architecture == "panel":
            continue
        lines.append(
            f"\n{architecture}, {block['models']} models, contrasts (a minus b)"
        )
        lines.append(
            "| contrast | rule | n | AUROC a | AUROC b | gap | CI | a wins | "
            "clean shift a/b | triggered shift a/b |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for name, entry in block["contrasts"].items():
            for rule in RULES:
                s = entry[f"{rule}_all"]
                if "auroc_gap" not in s:
                    lines.append(f"| {name} | {rule} | {s['n']} | | | | | | | |")
                    continue
                lines.append(
                    f"| {name} | {rule} | {s['n']} | {s['auroc_a']:.3f} | "
                    f"{s['auroc_b']:.3f} | {s['auroc_gap']:+.3f} | "
                    f"[{s['auroc_gap_ci'][0]:+.3f}, {s['auroc_gap_ci'][1]:+.3f}] | "
                    f"{s['a_wins']} | {fmt(s['shift_clean_a'])}/{fmt(s['shift_clean_b'])} | "
                    f"{fmt(s['shift_backdoor_a'])}/{fmt(s['shift_backdoor_b'])} |"
                )
    table_text = "\n".join(lines)
    return table_text


def causal_tables(summary):
    lines = ["\nCausal grid at matched clean change, triggered kept per variant"]
    lines.append(
        "| group | placement | models | rate | clean kept | all | trigger only | "
        "random | all but trigger | read signal late | stream signal late |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for group, placements in summary["placements"].items():
        for key, s in placements.items():
            read_curve = s["all_tokens"]["read_signal_retained"]
            stream_curve = s["all_tokens"]["stream_signal_retained"]
            lines.append(
                f"| {group} | {key} | {s['models']} | {s['rate_mean']:.3f} | "
                f"{s['all_tokens']['clean_kept']:.3f} | "
                f"{s['all_tokens']['triggered_kept']:.3f} | "
                f"{s['trigger_only']['triggered_kept']:.3f} | "
                f"{s['random_same_count']['triggered_kept']:.3f} | "
                f"{s['all_but_trigger']['triggered_kept']:.3f} | "
                f"{fmt(mean_of_present(read_curve[-4:]))} | "
                f"{fmt(mean_of_present(stream_curve[-4:]))} |"
            )
    for group, profile in summary["depth_profile"].items():
        for name, values in profile.items():
            if not isinstance(values, list):
                lines.append(f"{group} {name}: {values}")
                continue
            lines.append(f"{group} {name}: " + " ".join(f"{v:.2f}" for v in values))
    table_text = "\n".join(lines)
    return table_text


def rounded(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def fmt(value):
    text = "n/a" if value is None else f"{value:.3f}"
    return text


if __name__ == "__main__":
    main()
