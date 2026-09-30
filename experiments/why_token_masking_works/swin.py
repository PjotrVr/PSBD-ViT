"""Parts A to D of measure.py on Swin-S, where tokens merge and nothing is a class token.

On ViT the trigger reaches the classifier only through attention into the class
token, so a trigger token masked at the attention input of every block is cut off.
Swin-S reads its prediction as the mean of the 49 stage 4 tokens, and 3 patch
mergings carry every token's own stream forward without attention. A trigger
token masked at every attention input therefore still reaches the head through
its own stream. This script measures how much of the backdoor travels each route
and repeats A to D with trigger positions resolved per stage.

    A  trigger tokens masked deterministically at the attention input by block
       span, and the routes: attention, attention and MLP, the readout, all 3
    B  PSBD-TM at its adaptive rate, recording which blocks masked the trigger
    C  PSBD-RD at its adaptive rate on all tokens, the trigger's, the rest, or
       as many random tokens
    D  a fixed random subset of the 7 x 7 cells visible to the whole network

    source .venv/bin/activate
    PYTHONPATH=. python experiments/why_token_masking_works/swin.py
    PYTHONPATH=. python experiments/why_token_masking_works/swin.py --summarize-only
"""

import argparse
import collections
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from lightning import seed_everything  # noqa: E402

from data.splits import read_checkpoint_metadata  # noqa: E402
from experiments._paths import experiment_results_dir  # noqa: E402
from experiments.why_token_masking_works.measure import (  # noqa: E402
    KEEP_FRACTIONS,
    RecordingTokenMask,
    adaptive_rates,
    average,
    average_conditions,
    kept_by_count,
    limit_gpu_memory,
    mean_readout,
    pool_stochastic,
    predict,
    predict_batch,
    predict_pairs,
    readout,
    split_by_visibility,
    with_excess,
    write_json,
    load_pairs,
)
from experiments.why_token_masking_works.tokens import (  # noqa: E402
    BLOCK_GRIDS,
    cached_baseline_agreement,
    model_seed,
    CELL_GRID,
    FixedTokenMask,
    PositionRestricted,
    block_cell_positions,
    block_patch_counts,
    block_trigger_positions,
    complement_positions,
    example_images,
    plug_per_block,
    random_other_positions,
    touched_tokens,
    trigger_pixel_map,
)
from models.backbones import load_checkpoint, network_core  # noqa: E402
from models.positions import plug_dropout, unplug_dropout  # noqa: E402
from scripts.paper._common import clearing_cells, swin_coverage  # noqa: E402

SLUG = "why_token_masking_works"
SUBDIRECTORY = "swin"
ARCHITECTURE = "swin"
NUM_BLOCKS = 24
PAIR_COUNT = 256
FORWARD_PASSES = 10
RANDOM_DRAWS = 3
# The audit of 2026-09-29 asked for at least 20 visible subsets per model and
# fraction, drawn per model, so the spread across draws is measured.
SUBSET_DRAWS = 20
# SIG checkpoints are quarantined for tonight's runs: attacks/sig.py now builds
# amplitude 0.157 while the ViT SIG models learned 0.1, and the fix is another
# agent's. The 2 Swin SIG models are left out with them until it lands.
QUARANTINED_ATTACKS = ("sig",)
REPRODUCTION_FLOOR = 0.98
STAGES = {"stage_1": (1, 2), "stage_2": (3, 4), "stage_3": (5, 22), "stage_4": (23, 24)}
SPANS = {
    "all_24": (1, 24),
    "stages_1_2": (1, 4),
    "stage_3_first_half": (5, 13),
    "stage_3_second_half": (14, 22),
    "stage_4": (23, 24),
}
SPAN_COUNTS = (1, 2, 4, 8, 12)
ATTENTION_SITE = "before_attention_norm"
MLP_SITE = "before_mlp_norm"
# Each route closes 1 more path from a trigger token to the head. Attention is
# the only path on ViT. On Swin a token's own MLP keeps rewriting its content,
# patch merging carries it into the next stage, and the head averages it in.
ROUTES = {
    "attention": ((ATTENTION_SITE,), False),
    "attention_and_mlp": ((ATTENTION_SITE, MLP_SITE), False),
    "readout_only": ((), True),
    "attention_and_readout": ((ATTENTION_SITE,), True),
    "everything": ((ATTENTION_SITE, MLP_SITE), True),
}
RD_POSITIONS = ("after_attention_residual", "after_mlp_residual")
PATCH_ATTACKS = ("badnet_a2o", "tact", "lc")
PATCH_GROUPS = ("badnet_a2o", "tact_trigger_conditional", "tact_source_mapped", "lc")
TACT_CONDITIONAL_ACCURACY = 0.5
SUBSET_RATE_PREFERENCE = (0.05, 0.1, 0.01)
PART_KEYS = {
    "A": "deterministic_masking",
    "B": "stochastic_token_mask",
    "C": "residual_dropout",
    "D": "visible_subsets",
}
LATE_BLOCKS = slice(20, 24)
STAGE_4_BLOCKS = slice(22, 24)
MASK_SEED = 0
GPU_MEMORY_GB = 14.0
BATCH_SIZE = 256


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--folders", nargs="+", default=None)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--pairs", type=int, default=PAIR_COUNT)
    parser.add_argument("--passes", type=int, default=FORWARD_PASSES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gpu-memory-gb", type=float, default=GPU_MEMORY_GB)
    parser.add_argument("--device", default="cuda")
    # Parts run in the order given and a model's JSON keeps what earlier runs
    # wrote, so the cheap and decisive route test (A) can go first on every model.
    parser.add_argument("--parts", default="ABCD")
    parser.add_argument("--list-folders", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = os.path.join(
        experiment_results_dir(SLUG, args.output_root), SUBDIRECTORY
    )
    os.makedirs(output_dir, exist_ok=True)
    patch_folders, subset_folders = panel_folders(
        args.results_dir, args.checkpoints_dir
    )
    ordered = patch_folders + [f for f in subset_folders if f not in patch_folders]

    if args.list_folders:
        print("\n".join(ordered))
        return

    if not args.summarize_only:
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if args.device == "cuda"
            else torch.device(args.device)
        )
        if device.type == "cuda":
            limit_gpu_memory(args.gpu_memory_gb, device)
        for folder in args.folders or ordered:
            out_path = os.path.join(output_dir, f"{folder}.json")
            record = read_json_or_none(out_path)
            parts = parts_to_run(
                args.parts, folder in patch_folders, folder in subset_folders, record
            )
            if not parts:
                print(f"[skip] {folder}", flush=True)
                continue
            started = time.time()
            record = measure_model(folder, parts, record, args, device)
            record.setdefault("seconds", {})["".join(parts)] = round(
                time.time() - started, 1
            )
            write_json(out_path, record)
            print(
                f"[ok] {folder} parts {''.join(parts)} {time.time() - started:.0f}s",
                flush=True,
            )

    # A record of a model that has since left the panel stays on disk and out of
    # the summary.
    records = [r for r in read_records(output_dir) if r["folder"] in set(ordered)]
    summary = summarize(records)
    write_json(os.path.join(output_dir, "summary.json"), summary)
    print(markdown_tables(summary))


def measure_model(folder, parts, record, args, device):
    checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)
    assert metadata["architecture"] == ARCHITECTURE, f"{folder} is not a Swin model"
    model = load_checkpoint(ARCHITECTURE, checkpoint_path, device).eval()
    assert_pristine(model)
    pairs = load_pairs(checkpoint_path, args, device)
    tm_rate, rd_rate = adaptive_rates_or_none(args.results_dir, folder)

    pixel_map = trigger_pixel_map(metadata)  # (224, 224)
    trigger = block_trigger_positions(pixel_map, ARCHITECTURE)  # 24 x (count_b,)
    baseline = predict_pairs(model, pairs, args.batch_size, device)
    pairs["clean_base"] = baseline["clean"]  # (n,)
    pairs["triggered_base"] = baseline["triggered"]  # (n,)
    agreement = cached_baseline_agreement(args.results_dir, folder, pairs)
    assert min(agreement.values()) >= REPRODUCTION_FLOOR, f"{folder} {agreement}"
    draw_seed = model_seed(folder)
    pairs["draw_seed"] = draw_seed

    if record is None:
        record = {
            "folder": folder,
            "architecture": ARCHITECTURE,
            "attack": metadata["attack"],
            "dataset": metadata["dataset"],
            "poison_rate": metadata["poison_rate"],
            "pairs": int(len(pairs["targets"])),
            "trigger_positions_by_grid": {
                str(grid): touched_tokens(pixel_map, grid).tolist()
                for grid in sorted(set(BLOCK_GRIDS[ARCHITECTURE]) | {CELL_GRID})
            },
            "trigger_tokens_per_block": [len(p) for p in trigger],
            "tm_rate": tm_rate,
            "rd_rate": rd_rate,
            "draw_seed": draw_seed,
            "baseline_agreement_with_cache": agreement,
            "example": example_images(pairs, metadata["dataset"]),
            "baseline": readout(
                pairs, baseline["clean"][None], baseline["triggered"][None]
            ),
        }

    for part in parts:
        if part == "A":
            record["deterministic_masking"] = measure_deterministic_masking(
                model, pairs, trigger, draw_seed, args.batch_size, device
            )
        if part == "B" and tm_rate is not None:
            record["stochastic_token_mask"] = measure_stochastic_token_mask(
                model, pairs, trigger, tm_rate, args.passes, args.batch_size, device
            )
        if part == "C" and rd_rate is not None:
            record["residual_dropout"] = measure_residual_dropout(
                model,
                pairs,
                trigger,
                rd_rate,
                draw_seed,
                args.passes,
                args.batch_size,
                device,
            )
        if part == "D":
            trigger_cells = touched_tokens(pixel_map, CELL_GRID)  # (cells,)
            record["visible_subsets"] = measure_visible_subsets(
                model, pairs, trigger_cells, draw_seed, args.batch_size, device
            )
        assert_pristine(model)
    return record


# Parts A to C need trigger positions, so they run on patch models only. B
# and C need the model's own adaptive rate. D runs on 1 model per attack and
# dataset. A part already in the model's JSON is not run again.
def parts_to_run(requested, is_patch, is_subset, record):
    applicable = {"A": is_patch, "B": is_patch, "C": is_patch, "D": is_subset}
    rates = (record or {}).get("tm_rate", 0.0), (record or {}).get("rd_rate", 0.0)
    missing_rate = {"B": rates[0] is None, "C": rates[1] is None}
    parts = [
        part
        for part in requested
        if applicable[part]
        and not missing_rate.get(part, False)
        and PART_KEYS[part] not in (record or {})
    ]
    return parts


# Every probe is a hook or a forward override that unplug_dropout removes. A
# leftover one would silently stack onto the next measurement, and a module in
# training mode would add the model's own stochastic depth to the probe.
def assert_pristine(model):
    hooks = sum(
        len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()
    )
    overrides = [name for name, m in model.named_modules() if "forward" in m.__dict__]
    training = [name for name, m in model.named_modules() if m.training]
    assert hooks == 0 and not overrides, f"probe left attached: {hooks} {overrides}"
    assert not training, f"modules in training mode: {training[:3]}"


def read_json_or_none(path):
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        record = json.load(handle)
    return record


def measure_deterministic_masking(model, pairs, trigger, draw_seed, batch_size, device):
    spans = dict(SPANS)
    for count in SPAN_COUNTS:
        spans[f"last_{count}"] = (NUM_BLOCKS - count + 1, NUM_BLOCKS)
        spans[f"first_{count}"] = (1, count)

    conditions = {}
    for name, block_range in spans.items():
        conditions[name] = fixed_mask_readout(
            model,
            pairs,
            trigger,
            block_range,
            (ATTENTION_SITE,),
            False,
            batch_size,
            device,
        )

    # A block masked alone, and a block left as the only one whose attention sees
    # the trigger, give the depth profile of where the trigger is read.
    conditions["only_block"] = [
        fixed_mask_readout(
            model, pairs, trigger, (b, b), (ATTENTION_SITE,), False, batch_size, device
        )["triggered_kept"]
        for b in range(1, NUM_BLOCKS + 1)
    ]
    conditions["all_but_block"] = [
        all_but_block_readout(model, pairs, trigger, b, batch_size, device)[
            "triggered_kept"
        ]
        for b in range(1, NUM_BLOCKS + 1)
    ]

    # The literature memo's L24 on Swin: every token but the trigger's masked at
    # the attention input of the last 4 blocks.
    others = [
        complement_positions(t, count, 0)
        for t, count in zip(trigger, block_patch_counts(ARCHITECTURE))
    ]
    conditions["only_trigger_visible_last_4"] = fixed_mask_readout(
        model,
        pairs,
        others,
        (NUM_BLOCKS - 3, NUM_BLOCKS),
        (ATTENTION_SITE,),
        False,
        batch_size,
        device,
    )

    for route, (sites, readout_masked) in ROUTES.items():
        conditions[f"route_{route}"] = fixed_mask_readout(
            model,
            pairs,
            trigger,
            (1, NUM_BLOCKS),
            sites,
            readout_masked,
            batch_size,
            device,
        )

    # The control masks as many tokens as the trigger covers in every block,
    # drawn from the tokens it does not cover, so a drop in survival is about
    # which tokens were masked and not about how many.
    patch_counts = block_patch_counts(ARCHITECTURE)
    for route in ("attention", "everything"):
        sites, readout_masked = ROUTES[route]
        draws = []
        for draw in range(RANDOM_DRAWS):
            positions = [
                random_other_positions(t, len(t), count, 0, draw_seed + draw)
                for t, count in zip(trigger, patch_counts)
            ]
            draws.append(
                fixed_mask_readout(
                    model,
                    pairs,
                    positions,
                    (1, NUM_BLOCKS),
                    sites,
                    readout_masked,
                    batch_size,
                    device,
                )
            )
        conditions[f"random_{route}"] = mean_readout(draws)
    return conditions


def measure_stochastic_token_mask(
    model, pairs, trigger, rate, passes, batch_size, device
):
    recorders = []

    def build_recorder(probe_rate):
        recorder = RecordingTokenMask(probe_rate)
        recorders.append(recorder)
        return recorder

    handles = plug_dropout(
        model, ARCHITECTURE, (ATTENTION_SITE,), {ATTENTION_SITE: build_recorder}, rate
    )
    assert len(recorders) == NUM_BLOCKS, f"expected 24 probes, got {len(recorders)}"
    try:
        # Seeded per model, so 2 models at the same rate do not draw
        # identical masks and pooled counts are independent replicates.
        seed_everything(MASK_SEED + pairs["draw_seed"], verbose=False)
        clean_passes = recorded_passes(
            model, recorders, pairs["clean"], trigger, passes, batch_size, device
        )
        triggered_passes = recorded_passes(
            model, recorders, pairs["triggered"], trigger, passes, batch_size, device
        )
    finally:
        unplug_dropout(handles)

    clean_kept = clean_passes["predictions"] == pairs["clean_base"][None]  # (k, n)
    triggered_kept = triggered_passes["predictions"] == pairs["targets"][None]  # (k, n)
    hit = (pairs["triggered_base"] == pairs["targets"])[None].expand_as(
        triggered_kept
    )  # (k, n)
    everyone = torch.ones_like(clean_kept)  # (k, n)

    result = {
        "rate": rate,
        "overall": readout(
            pairs, clean_passes["predictions"], triggered_passes["predictions"]
        ),
    }
    for split, kept, eligible, counts in (
        ("triggered", triggered_kept, hit, triggered_passes),
        ("clean", clean_kept, everyone, clean_passes),
    ):
        result[split] = {
            statistic: kept_by_count(kept, eligible, counts[statistic])
            for statistic in (
                "blocks_all",
                "late_all",
                "stage_4_all",
                "fewest_visible_stage_4",
            )
        }
    return result


def measure_residual_dropout(
    model, pairs, trigger, rate, draw_seed, passes, batch_size, device
):
    patch_counts = block_patch_counts(ARCHITECTURE)
    rest = [
        complement_positions(t, count, 0) for t, count in zip(trigger, patch_counts)
    ]

    result = {"rate": rate}
    result["all_tokens"] = dropout_readout(
        model, pairs, None, rate, passes, batch_size, device
    )
    result["trigger_only"] = dropout_readout(
        model, pairs, trigger, rate, passes, batch_size, device
    )
    result["all_but_trigger"] = dropout_readout(
        model, pairs, rest, rate, passes, batch_size, device
    )

    draws = []
    for draw in range(RANDOM_DRAWS):
        positions = [
            random_other_positions(t, len(t), count, 0, draw_seed + draw)
            for t, count in zip(trigger, patch_counts)
        ]
        draws.append(
            dropout_readout(model, pairs, positions, rate, passes, batch_size, device)
        )
    result["random_same_count"] = mean_readout(draws)
    return result


# Hides a random subset of the 7 x 7 cells from the whole network: every token
# of a hidden cell is masked at both branch inputs of all 24 blocks and at the
# readout, so the model sees only the visible cells. On ViT masking the
# attention input alone had that effect, since only attention reaches the class
# token. On Swin the MLP, patch merging and the mean-pooled head would otherwise
# still carry a hidden token's content.
def measure_visible_subsets(model, pairs, trigger_cells, draw_seed, batch_size, device):
    cells = CELL_GRID * CELL_GRID
    reference = readout(pairs, pairs["clean_base"][None], pairs["triggered_base"][None])
    result = {}
    for fraction in KEEP_FRACTIONS:
        if fraction == 1.0:
            result[str(fraction)] = {"mean": reference, "draws": [reference]}
            continue
        visible_count = round(fraction * cells)
        draws = []
        for draw in range(SUBSET_DRAWS):
            generator = torch.Generator().manual_seed(draw_seed + draw)
            order = torch.randperm(cells, generator=generator)  # (49,)
            hidden = torch.ones(cells, dtype=torch.bool)  # (49,)
            hidden[order[:visible_count]] = False
            positions = block_cell_positions(
                hidden.reshape(CELL_GRID, CELL_GRID), ARCHITECTURE
            )  # 24 x (hidden tokens_b,)
            sites, readout_masked = ROUTES["everything"]
            reading = fixed_mask_readout(
                model,
                pairs,
                positions,
                (1, NUM_BLOCKS),
                sites,
                readout_masked,
                batch_size,
                device,
            )
            reading["trigger_visible"] = int((~hidden[trigger_cells]).sum())
            draws.append(reading)
        result[str(fraction)] = {"mean": mean_readout(draws), "draws": draws}

    for block in result.values():
        block["mean"]["asr_retention"] = ratio(
            block["mean"]["triggered_asr"], reference["triggered_asr"]
        )
        block["mean"]["clean_accuracy_retention"] = ratio(
            block["mean"]["clean_accuracy"], reference["clean_accuracy"]
        )
    return result


def fixed_mask_readout(
    model, pairs, positions, block_range, sites, readout_masked, batch_size, device
):
    handles = []
    try:
        if sites:
            first, last = block_range
            handles += plug_per_block_range(model, positions, sites, first, last)
        if readout_masked:
            handles.append(attach_readout_mask(model, positions[-1]))
        predictions = predict_pairs(model, pairs, batch_size, device)
    finally:
        unplug_dropout(handles)

    reading = readout(pairs, predictions["clean"][None], predictions["triggered"][None])
    return reading


def all_but_block_readout(model, pairs, trigger, visible_block, batch_size, device):
    handles = []
    try:
        for b in range(1, NUM_BLOCKS + 1):
            if b == visible_block:
                continue
            handles += plug_per_block_range(model, trigger, (ATTENTION_SITE,), b, b)
        predictions = predict_pairs(model, pairs, batch_size, device)
    finally:
        unplug_dropout(handles)

    reading = readout(pairs, predictions["clean"][None], predictions["triggered"][None])
    return reading


def plug_per_block_range(model, positions, sites, first, last):
    handles = []
    for b in range(first, last + 1):
        mask_positions = positions[b - 1]
        handles += plug_dropout(
            model,
            ARCHITECTURE,
            sites,
            {
                site: (lambda _rate, p=mask_positions: FixedTokenMask(p, False))
                for site in sites
            },
            0.0,
            block_range=(b, b),
        )
    return handles


# The final LayerNorm sees the 7 x 7 stage 4 map right before the mean pool, so a
# token zeroed there contributes only the norm's bias, identically on clean and
# triggered images.
def attach_readout_mask(model, stage_4_positions):
    probe = FixedTokenMask(stage_4_positions, False)
    handle = network_core(model).norm.register_forward_pre_hook(
        lambda _module, args: (probe(args[0]),) + tuple(args[1:])
    )
    return handle


def dropout_readout(model, pairs, positions, rate, passes, batch_size, device):
    if positions is None:
        handles = plug_dropout(model, ARCHITECTURE, RD_POSITIONS, {}, rate)
    else:
        handles = plug_per_block(
            model,
            ARCHITECTURE,
            RD_POSITIONS,
            lambda b, probe_rate: PositionRestricted(
                nn.Dropout(probe_rate), positions[b]
            ),
            rate,
        )
    try:
        # Seeded per model, so 2 models at the same rate do not draw
        # identical masks and pooled counts are independent replicates.
        seed_everything(MASK_SEED + pairs["draw_seed"], verbose=False)
        clean = torch.stack(
            [predict(model, pairs["clean"], batch_size, device) for _ in range(passes)]
        )  # (k, n)
        triggered = torch.stack(
            [
                predict(model, pairs["triggered"], batch_size, device)
                for _ in range(passes)
            ]
        )  # (k, n)
    finally:
        unplug_dropout(handles)

    reading = readout(pairs, clean, triggered)
    return reading


def recorded_passes(model, recorders, images, trigger, passes, batch_size, device):
    columns = collections.defaultdict(list)
    for _ in range(passes):
        pass_columns = collections.defaultdict(list)
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            pass_columns["predictions"].append(predict_batch(model, batch, device))

            all_dropped = []
            visible = []
            for recorder, positions in zip(recorders, trigger):
                trigger_dropped = recorder.last_dropped[
                    :, positions.to(device)
                ]  # (b, t)
                all_dropped.append(trigger_dropped.all(dim=1))  # (b,)
                visible.append((~trigger_dropped).sum(dim=1))  # (b,)
            all_dropped = torch.stack(all_dropped)  # (24, b)
            visible = torch.stack(visible)  # (24, b)
            pass_columns["blocks_all"].append(all_dropped.sum(dim=0))  # (b,)
            pass_columns["late_all"].append(all_dropped[LATE_BLOCKS].sum(dim=0))
            pass_columns["stage_4_all"].append(all_dropped[STAGE_4_BLOCKS].sum(dim=0))
            pass_columns["fewest_visible_stage_4"].append(
                visible[STAGE_4_BLOCKS].min(dim=0).values
            )
        for name, parts in pass_columns.items():
            columns[name].append(torch.cat(parts))  # (n,)

    stacked = {name: torch.stack(parts) for name, parts in columns.items()}  # (k, n)
    return stacked


def adaptive_rates_or_none(results_dir, folder):
    # 3 Swin cells carry no adaptive PSBD-TM rate because no swept rate reaches
    # the target, and that part is skipped for them rather than guessed.
    try:
        rates = adaptive_rates(results_dir, folder)
    except AssertionError:
        with open(os.path.join(results_dir, folder, "psbd_metrics.json")) as handle:
            placements = json.load(handle)["placements"]
        rates = tuple(
            placements.get(name, {}).get("adaptive_rate")
            for name in ("before_attention_norm_token_mask", "post_residual")
        )
    return rates


# The Swin panel is the ViT panel rule applied to Swin-S
# (scripts.paper._common.swin_coverage): the declared datasets, poison rates and
# canonical variants, source-mapped and diverged models out and clean accuracy
# within 2 points of the Swin benign model of the dataset. SIG stays out of the
# GPU runs while its trigger amplitude is being fixed.
def panel_cells(results_dir, checkpoints_dir):
    cells = [
        {
            "folder": cell["folder_name"],
            "attack": cell["attack"],
            "dataset": cell["dataset"],
            "poison_rate": cell["poison_rate"],
        }
        for cell in clearing_cells(swin_coverage(results_dir, checkpoints_dir))
        if cell["attack"] not in QUARANTINED_ATTACKS
    ]
    return cells


def panel_folders(results_dir, checkpoints_dir):
    cells = panel_cells(results_dir, checkpoints_dir)
    patch_folders = [c["folder"] for c in cells if c["attack"] in PATCH_ATTACKS]

    by_attack_and_dataset = collections.defaultdict(dict)
    for cell in cells:
        by_attack_and_dataset[(cell["attack"], cell["dataset"])][
            cell["poison_rate"]
        ] = cell["folder"]
    subset_folders = []
    for rates in by_attack_and_dataset.values():
        preferred = [rate for rate in SUBSET_RATE_PREFERENCE if rate in rates]
        chosen = preferred[0] if preferred else sorted(rates)[0]
        subset_folders.append(rates[chosen])
    return patch_folders, subset_folders


def ratio(numerator, denominator):
    value = numerator / denominator if denominator > 0 else None
    return value


def read_records(output_dir):
    records = []
    for name in sorted(os.listdir(output_dir)):
        if not name.endswith(".json") or name == "summary.json":
            continue
        with open(os.path.join(output_dir, name)) as handle:
            records.append(json.load(handle))
    return records


# measure.py's rule: a TaCT model whose clean source images are misclassified
# carries a class mapping rather than a trigger. tab_swin already drops those, so
# this group stays empty on the panel and is kept to say so.
def analysis_group(record):
    if record["attack"] != "tact":
        return record["attack"]
    conditional = record["baseline"]["clean_accuracy"] >= TACT_CONDITIONAL_ACCURACY
    group = "tact_trigger_conditional" if conditional else "tact_source_mapped"
    return group


def summarize(records):
    by_group = collections.defaultdict(list)
    for record in records:
        by_group[analysis_group(record)].append(record)
    patch_groups = [
        g
        for g in PATCH_GROUPS
        if any("deterministic_masking" in r for r in by_group[g])
    ]

    def with_part(group, part):
        selected = [r for r in by_group[group] if part in r]
        return selected

    summary = {
        "models": len(records),
        "models_by_group": {g: len(rs) for g, rs in by_group.items()},
        "baseline": {
            g: average([r["baseline"] for r in rs]) for g, rs in by_group.items()
        },
        "per_model": [
            per_model_row(r)
            for g in patch_groups
            for r in with_part(g, "deterministic_masking")
        ],
        "deterministic_masking": {
            g: average_conditions(
                [
                    {
                        k: v
                        for k, v in r["deterministic_masking"].items()
                        if isinstance(v, dict)
                    }
                    for r in with_part(g, "deterministic_masking")
                ]
            )
            for g in patch_groups
        },
        "depth_profile": {
            g: {
                profile: mean_lists(
                    [
                        r["deterministic_masking"][profile]
                        for r in with_part(g, "deterministic_masking")
                    ]
                )
                for profile in ("only_block", "all_but_block")
            }
            for g in patch_groups
        },
        "stochastic_token_mask": {
            g: pool_stochastic(with_part(g, "stochastic_token_mask"))
            for g in patch_groups
            if with_part(g, "stochastic_token_mask")
        },
        "residual_dropout": {
            g: average_conditions(
                [
                    {k: v for k, v in r["residual_dropout"].items() if k != "rate"}
                    for r in with_part(g, "residual_dropout")
                ]
            )
            for g in patch_groups
            if with_part(g, "residual_dropout")
        },
        "visible_subsets": {
            g: {
                fraction: average(
                    [with_excess(r["visible_subsets"], fraction) for r in rs]
                )
                for fraction in map(str, KEEP_FRACTIONS)
            }
            for g, rs in ((g, with_part(g, "visible_subsets")) for g in by_group)
            if rs
        },
        "visible_subsets_patch_by_trigger_visibility": split_by_visibility(
            [r for g in patch_groups for r in with_part(g, "visible_subsets")]
        ),
    }
    return summary


def mean_lists(lists):
    means = [sum(column) / len(column) for column in zip(*lists)]
    return means


def per_model_row(record):
    masking = record["deterministic_masking"]
    row = {
        "folder": record["folder"],
        "group": analysis_group(record),
        "pairs": record["pairs"],
        "trigger_tokens_stage_4": record["trigger_tokens_per_block"][-1],
        "tm_rate": record["tm_rate"],
        "rd_rate": record["rd_rate"],
        "baseline_clean_accuracy": record["baseline"]["clean_accuracy"],
        "attention_all_24": masking["all_24"]["triggered_kept"],
        "everything": masking["route_everything"]["triggered_kept"],
        "readout_only": masking["route_readout_only"]["triggered_kept"],
        "random_attention": masking["random_attention"]["triggered_kept"],
        "tm_triggered_kept": record.get("stochastic_token_mask", {})
        .get("overall", {})
        .get("triggered_kept"),
        "tm_clean_kept": record.get("stochastic_token_mask", {})
        .get("overall", {})
        .get("clean_kept"),
        "rd_all_triggered_kept": record.get("residual_dropout", {})
        .get("all_tokens", {})
        .get("triggered_kept"),
        "rd_trigger_only_triggered_kept": record.get("residual_dropout", {})
        .get("trigger_only", {})
        .get("triggered_kept"),
    }
    return row


def markdown_tables(summary):
    lines = [f"\n{summary['models']} Swin models, groups {summary['models_by_group']}"]
    lines.append("\nPatch models, 1 row each")
    if summary["per_model"]:
        columns = [c for c in summary["per_model"][0] if c not in ("folder", "group")]
        lines.append("| model | group | " + " | ".join(columns) + " |")
        lines.append("|---|---|" + "---|" * len(columns))
        for row in summary["per_model"]:
            cells = " | ".join(format_number(row[c]) for c in columns)
            lines.append(f"| {row['folder']} | {row['group']} | {cells} |")

    for part in ("deterministic_masking", "residual_dropout"):
        lines.append(f"\n{part}")
        lines.append(
            "| group | condition | triggered kept | clean kept | clean on target |"
        )
        lines.append("|---|---|---|---|---|")
        for group, conditions in summary[part].items():
            for name, reading in conditions.items():
                lines.append(
                    f"| {group} | {name} | {format_number(reading['triggered_kept'])} | "
                    f"{format_number(reading['clean_kept'])} | "
                    f"{format_number(reading['clean_on_target'])} |"
                )

    lines.append("\ndepth profile, triggered kept per block")
    for group, profiles in summary["depth_profile"].items():
        for profile, values in profiles.items():
            text = " ".join(f"{v:.2f}" for v in values)
            lines.append(f"{group} {profile}: {text}")

    lines.append("\nstochastic token mask, P(kept) by count")
    for group, pooled in summary["stochastic_token_mask"].items():
        overall = pooled["overall"]
        lines.append(
            f"{group}: triggered {overall['triggered_kept']:.3f} clean "
            f"{overall['clean_kept']:.3f}"
        )
        for statistic in ("late_all", "stage_4_all", "fewest_visible_stage_4"):
            for split in ("triggered", "clean"):
                text = ", ".join(
                    f"{value}: {entry['kept']:.3f} ({entry['n']})"
                    for value, entry in pooled[split][statistic].items()
                )
                lines.append(f"  {statistic} {split}: {text}")

    lines.append("\nvisible subsets, excess and clean retention")
    for group, fractions in summary["visible_subsets"].items():
        text = " | ".join(
            f"f={f} excess {format_number(fractions[f].get('excess_retention'))} "
            f"clean {format_number(fractions[f].get('clean_accuracy_retention'))}"
            for f in map(str, KEEP_FRACTIONS[1:])
        )
        lines.append(f"{group} ({fractions['1.0']['models']}): {text}")
    for fraction, groups in summary[
        "visible_subsets_patch_by_trigger_visibility"
    ].items():
        text = ", ".join(
            f"{key} excess {entry['excess_retention']:.3f} over {entry['draws']} draws"
            for key, entry in groups.items()
        )
        lines.append(f"f={fraction}: {text}")

    table_text = "\n".join(lines)
    return table_text


def format_number(value):
    if value is None:
        return "n/a"
    text = str(value) if isinstance(value, int) else f"{value:.3f}"
    return text


if __name__ == "__main__":
    main()
