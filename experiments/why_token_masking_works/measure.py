"""Why token masking at the attention input separates patch triggers and dropout does not.

PSBD-TM (token_mask at before_attention_norm) leaves about 4% of triggered BadNets
predictions changed at its adaptive rate while PSBD-RD (dropout at post_residual)
changes 61% of them, and the 2 placements agree on global triggers. The hypothesis
tested here is that a patch trigger's evidence sits in its own tokens of the residual
stream, which masking at the attention input never touches, while dropout on the
stream corrupts that stored content in every block. 4 measurements, all inference
only, on 500 paired clean and triggered test images per model from the PSBD analysis
split. README.md states each hypothesis and what the numbers say about it.

    A  trigger tokens masked deterministically at the attention input, by block span
    B  the real PSBD-TM operator at the adaptive rate, with its per-block masks recorded
    C  PSBD-RD at the adaptive rate on all tokens, only the trigger tokens, only as many
       random tokens, or every token except the trigger's
    D  a fixed random subset of tokens kept visible in every block, every panel model

    source .venv/bin/activate
    PYTHONPATH=. python experiments/why_token_masking_works/measure.py
    PYTHONPATH=. python experiments/why_token_masking_works/measure.py --summarize-only
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
import torch.nn.functional as F  # noqa: E402
from lightning import seed_everything  # noqa: E402

from cli.compare_detectors import psbd_rate  # noqa: E402
from data.splits import (  # noqa: E402
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from defenses.decision import PUBLISHED_PLACEMENT, RECOMMENDED_PLACEMENT  # noqa: E402
from defenses.inference import forward_logits  # noqa: E402
from defenses.operators import TokenMask, _keep_scale  # noqa: E402
from experiments._paths import experiment_result_path, experiment_results_dir  # noqa: E402
from experiments.residual_stream_mechanism.activation_patching import (  # noqa: E402
    trigger_tokens,
)
from models.backbones import load_checkpoint  # noqa: E402
from models.positions import plug_dropout, unplug_dropout  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    clearing_cells,
    load_coverage,
    load_psbd_metrics,
)

SLUG = "why_token_masking_works"
PAIR_COUNT = 256
FORWARD_PASSES = 10
RANDOM_DRAWS = 3
KEEP_FRACTIONS = (1.0, 0.6, 0.3, 0.1)
SPAN_COUNTS = (1, 2, 4, 8, 12)
# D runs on 1 model per attack and dataset, the first of these rates that cleared.
SUBSET_RATE_PREFERENCE = (0.05, 0.1, 0.01)
NUM_BLOCKS = 12
PATCH_TOKENS = 196
PATCH_ATTACKS = ("badnet_a2o", "tact", "lc")
PATCH_GROUPS = ("badnet_a2o", "tact_trigger_conditional", "tact_source_mapped", "lc")
TACT_CONDITIONAL_ACCURACY = 0.5
# blocks_any and late_any stay in the JSON. With 4 trigger tokens at rate 0.5 at
# least 1 is masked in almost every block, so they barely vary.
PRINTED_STATISTICS = ("blocks_all", "late_all", "fewest_visible_late")
PER_MODEL_COLUMNS = (
    "trigger_tokens",
    "tm_rate",
    "rd_rate",
    "baseline_clean_accuracy",
    "baseline_clean_on_target",
    "mask_trigger_all_12",
    "mask_trigger_last_4",
    "mask_random_all_12",
    "tm_triggered_kept",
    "tm_clean_kept",
    "rd_all_triggered_kept",
    "rd_trigger_only_triggered_kept",
    "rd_all_but_trigger_triggered_kept",
    "rd_random_triggered_kept",
    "rd_all_clean_kept",
)
GLOBAL_ATTACKS = ("blend", "sig", "wanet", "lf", "bpp")
TM_POSITION = "before_attention_norm"
RD_POSITIONS = ("after_attention_residual", "after_mlp_residual")
BANDS = {"blocks_1_4": (1, 4), "blocks_5_8": (5, 8), "blocks_9_12": (9, 12)}
LATE_BLOCKS = slice(8, 12)
MASK_SEED = 0
DRAW_SEED = 0
GPU_MEMORY_GB = 6.0
BATCH_SIZE = 250


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--folders", nargs="+", default=None)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    # Separate from --results-dir so a smoke run reads the real psbd_metrics.json
    # and coverage ledger but writes somewhere disposable.
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--pairs", type=int, default=PAIR_COUNT)
    parser.add_argument("--passes", type=int, default=FORWARD_PASSES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gpu-memory-gb", type=float, default=GPU_MEMORY_GB)
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = experiment_results_dir(SLUG, args.output_root)

    if not args.summarize_only:
        device = torch.device("cuda", torch.cuda.current_device())
        limit_gpu_memory(args.gpu_memory_gb, device)
        patch_folders, subset_folders = panel_folders(args.results_dir)
        folders = args.folders or patch_folders + [
            folder for folder in subset_folders if folder not in patch_folders
        ]
        for folder in folders:
            out_path = experiment_result_path(SLUG, f"{folder}.json", args.output_root)
            if os.path.exists(out_path):
                print(f"[skip] {folder}", flush=True)
                continue
            started = time.time()
            record = measure_model(folder, folder in subset_folders, args, device)
            record["seconds"] = round(time.time() - started, 1)
            write_json(out_path, record)
            print(f"[ok] {folder} {record['seconds']}s", flush=True)

    records = read_model_records(output_dir)
    summary = summarize(records)
    write_json(os.path.join(output_dir, "summary.json"), summary)
    print(markdown_tables(summary))


def measure_model(folder, run_subsets, args, device):
    checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)
    model = load_checkpoint(metadata["architecture"], checkpoint_path, device).eval()
    pairs = load_pairs(checkpoint_path, args, device)
    tm_rate, rd_rate = adaptive_rates(args.results_dir, folder)

    # trigger_tokens indexes the 196 patches, the sequence carries CLS at 0.
    trigger = trigger_tokens(metadata) + 1  # (trigger_count,)
    pairs["trigger"] = trigger.to(device)
    baseline = predict_pairs(model, pairs, args.batch_size, device)
    pairs["clean_base"] = baseline["clean"]  # (n,)
    pairs["triggered_base"] = baseline["triggered"]  # (n,)

    record = {
        "folder": folder,
        "attack": metadata["attack"],
        "dataset": metadata["dataset"],
        "poison_rate": metadata["poison_rate"],
        "pairs": int(len(pairs["targets"])),
        "trigger_positions": trigger.tolist(),
        "tm_rate": tm_rate,
        "rd_rate": rd_rate,
        "baseline": readout(
            pairs, baseline["clean"][None], baseline["triggered"][None]
        ),
    }

    if metadata["attack"] in PATCH_ATTACKS:
        record["deterministic_masking"] = measure_deterministic_masking(
            model, pairs, args.batch_size, device
        )
        record["stochastic_token_mask"] = measure_stochastic_token_mask(
            model, pairs, tm_rate, args.passes, args.batch_size, device
        )
        record["residual_dropout"] = measure_residual_dropout(
            model, pairs, rd_rate, args.passes, args.batch_size, device
        )

    if run_subsets:
        record["visible_subsets"] = measure_visible_subsets(
            model, pairs, args.batch_size, device
        )
    return record


def measure_deterministic_masking(model, pairs, batch_size, device):
    trigger = pairs["trigger"]
    spans = {"all_12": (1, NUM_BLOCKS)}
    spans.update(BANDS)
    for count in SPAN_COUNTS:
        spans[f"last_{count}"] = (NUM_BLOCKS - count + 1, NUM_BLOCKS)
        spans[f"first_{count}"] = (1, count)

    conditions = {}
    for name, block_range in spans.items():
        conditions[name] = fixed_mask_readout(
            model, pairs, trigger, block_range, batch_size, device
        )

    # The control masks as many tokens as the trigger has, drawn from the patches
    # the trigger does not cover, so a drop in survival is about which tokens
    # were masked and not about how many.
    draws = []
    for draw in range(RANDOM_DRAWS):
        positions = random_non_trigger_positions(trigger, len(trigger), draw, device)
        draws.append(
            fixed_mask_readout(
                model, pairs, positions, (1, NUM_BLOCKS), batch_size, device
            )
        )
    conditions["random_all_12"] = mean_readout(draws)
    return conditions


def measure_stochastic_token_mask(model, pairs, rate, passes, batch_size, device):
    recorders = []

    def build_recorder(probe_rate):
        recorder = RecordingTokenMask(probe_rate)
        recorders.append(recorder)
        return recorder

    handles = plug_dropout(
        model, "vit", (TM_POSITION,), {TM_POSITION: build_recorder}, rate
    )
    assert len(recorders) == NUM_BLOCKS, f"expected 12 probes, got {len(recorders)}"
    try:
        seed_everything(MASK_SEED, verbose=False)
        clean_passes = recorded_passes(
            model,
            recorders,
            pairs["clean"],
            pairs["trigger"],
            passes,
            batch_size,
            device,
        )
        triggered_passes = recorded_passes(
            model,
            recorders,
            pairs["triggered"],
            pairs["trigger"],
            passes,
            batch_size,
            device,
        )
    finally:
        unplug_dropout(handles)

    clean_kept = clean_passes["predictions"] == pairs["clean_base"][None]  # (k, n)
    triggered_kept = triggered_passes["predictions"] == pairs["targets"][None]  # (k, n)
    # Only triggered images the unperturbed model sends to the target can keep
    # that prediction, so the rest are left out of every conditional.
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
                "blocks_any",
                "blocks_all",
                "late_any",
                "late_all",
                "fewest_visible_late",
            )
        }
    return result


def measure_residual_dropout(model, pairs, rate, passes, batch_size, device):
    trigger = pairs["trigger"]
    everything_but_trigger = random_non_trigger_positions(
        trigger, PATCH_TOKENS - len(trigger), 0, device
    )
    everything_but_trigger = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), everything_but_trigger]
    )  # (tokens - trigger_count,), CLS included

    variants = {
        "all_tokens": nn.Dropout,
        "trigger_only": lambda probe_rate: PositionDropout(probe_rate, trigger),
        "all_but_trigger": lambda probe_rate: PositionDropout(
            probe_rate, everything_but_trigger
        ),
    }
    result = {"rate": rate}
    for name, factory in variants.items():
        result[name] = dropout_readout(
            model, pairs, factory, rate, passes, batch_size, device
        )

    draws = []
    for draw in range(RANDOM_DRAWS):
        positions = random_non_trigger_positions(trigger, len(trigger), draw, device)
        draws.append(
            dropout_readout(
                model,
                pairs,
                lambda probe_rate, positions=positions: PositionDropout(
                    probe_rate, positions
                ),
                rate,
                passes,
                batch_size,
                device,
            )
        )
    result["random_same_count"] = mean_readout(draws)
    return result


def measure_visible_subsets(model, pairs, batch_size, device):
    trigger = pairs["trigger"]
    reference = readout(pairs, pairs["clean_base"][None], pairs["triggered_base"][None])
    result = {}
    for fraction in KEEP_FRACTIONS:
        if fraction == 1.0:
            result[str(fraction)] = {"mean": reference, "draws": [reference]}
            continue
        visible_count = round(fraction * PATCH_TOKENS)
        draws = []
        for draw in range(RANDOM_DRAWS):
            generator = torch.Generator().manual_seed(DRAW_SEED + draw)
            order = torch.randperm(PATCH_TOKENS, generator=generator) + 1  # (196,)
            masked = order[visible_count:].to(device)  # (196 - visible,)
            reading = fixed_mask_readout(
                model, pairs, masked, (1, NUM_BLOCKS), batch_size, device
            )
            reading["trigger_visible"] = int(
                torch.isin(trigger, order[:visible_count].to(device)).sum()
            )
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


# _mask_tokens is TokenMask's own body line for line, so the random stream it
# consumes and the output it returns are the library operator's. The only
# addition is storing the (batch, tokens) boolean of dropped tokens.
# tests/test_why_token_masking_works.py holds the 2 to the same output.
class RecordingTokenMask(TokenMask):
    def __init__(self, rate, protect_cls=True):
        super().__init__(rate, protect_cls)
        self.last_dropped = None

    def _mask_tokens(self, x, protect_cls):
        batch, tokens, _ = x.shape  # (batch, tokens, channels)
        keep = torch.empty(batch, tokens, 1, device=x.device, dtype=x.dtype).bernoulli_(
            1.0 - self.rate
        )  # (batch, tokens, 1)
        keep *= _keep_scale(self.rate)
        if protect_cls:
            keep[:, 0, :] = 1.0
        self.last_dropped = keep[:, :, 0] == 0  # (batch, tokens)

        masked = x * keep  # (batch, tokens, channels)
        return masked


# Zeroes the given positions and rescales every other patch token by the
# inverted-dropout factor of the realized masked share, leaving CLS at 1 as
# TokenMask does. At before_attention_norm the rescale is inert, since
# LayerNorm divides each token by its own scale, which the test file checks.
# Kept anyway so the operator differs from TokenMask only in how it chooses.
class FixedTokenMask(nn.Module):
    def __init__(self, positions, patch_tokens=PATCH_TOKENS):
        super().__init__()
        self.positions = positions
        self.scale = _keep_scale(len(positions) / patch_tokens)

    def forward(self, x):
        _, tokens, _ = x.shape  # (batch, tokens, channels)
        keep = torch.full((tokens,), self.scale, device=x.device, dtype=x.dtype)
        keep[0] = 1.0
        keep[self.positions] = 0.0

        masked = x * keep[None, :, None]  # (batch, tokens, channels)
        return masked


class PositionDropout(nn.Module):
    def __init__(self, rate, positions):
        super().__init__()
        self.rate = float(rate)
        self.positions = positions

    def forward(self, x):
        selected = x[:, self.positions, :]  # (batch, selected, channels)
        dropped = F.dropout(selected, self.rate, training=True)  # same shape

        out = x.clone()  # (batch, tokens, channels)
        out[:, self.positions, :] = dropped
        return out


def fixed_mask_readout(model, pairs, positions, block_range, batch_size, device):
    handles = plug_dropout(
        model,
        "vit",
        (TM_POSITION,),
        {TM_POSITION: lambda _rate: FixedTokenMask(positions)},
        0.0,
        block_range=block_range,
    )
    try:
        predictions = predict_pairs(model, pairs, batch_size, device)
    finally:
        unplug_dropout(handles)

    reading = readout(pairs, predictions["clean"][None], predictions["triggered"][None])
    return reading


def dropout_readout(model, pairs, factory, rate, passes, batch_size, device):
    handles = plug_dropout(
        model,
        "vit",
        RD_POSITIONS,
        {position: factory for position in RD_POSITIONS},
        rate,
    )
    try:
        seed_everything(MASK_SEED, verbose=False)
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

            dropped = torch.stack([r.last_dropped for r in recorders])  # (12, b, 197)
            trigger_dropped = dropped[:, :, trigger]  # (12, b, trigger_count)
            any_dropped = trigger_dropped.any(dim=2)  # (12, b)
            all_dropped = trigger_dropped.all(dim=2)  # (12, b)
            visible_late = (~trigger_dropped[LATE_BLOCKS]).sum(dim=2)  # (4, b)
            pass_columns["blocks_any"].append(any_dropped.sum(dim=0))  # (b,)
            pass_columns["blocks_all"].append(all_dropped.sum(dim=0))  # (b,)
            pass_columns["late_any"].append(any_dropped[LATE_BLOCKS].sum(dim=0))
            pass_columns["late_all"].append(all_dropped[LATE_BLOCKS].sum(dim=0))
            pass_columns["fewest_visible_late"].append(visible_late.min(dim=0).values)
        for name, parts in pass_columns.items():
            columns[name].append(torch.cat(parts))  # (n,)

    stacked = {name: torch.stack(parts) for name, parts in columns.items()}  # (k, n)
    return stacked


def kept_by_count(kept, eligible, counts):
    table = {}
    for value in torch.unique(counts[eligible]).tolist():
        selected = eligible & (counts == value)  # (k, n)
        table[str(int(value))] = [int(kept[selected].sum()), int(selected.sum())]
    return table


# Survival and accuracy, predictions shaped (passes, n) for both splits.
#
# triggered_kept conditions on images the unperturbed model sends to the target,
# since only those have a target prediction to keep. triggered_asr does not.
def readout(pairs, clean_predictions, triggered_predictions):
    targets = pairs["targets"][None]  # (1, n)
    hit = pairs["triggered_base"] == pairs["targets"]  # (n,)
    triggered_on_target = (triggered_predictions == targets).float()  # (k, n)
    clean_kept = (clean_predictions == pairs["clean_base"][None]).float()  # (k, n)
    clean_correct = (clean_predictions == pairs["clean_labels"][None]).float()
    # A heavy perturbation can collapse every input onto 1 default class. When
    # that class is the target, triggered images read as surviving for a reason
    # unrelated to the trigger, and this is how often clean images land there.
    clean_on_target = (clean_predictions == targets).float()  # (k, n)

    reading = {
        "triggered_kept": float(triggered_on_target[:, hit].mean()),
        "triggered_asr": float(triggered_on_target.mean()),
        "clean_kept": float(clean_kept.mean()),
        "clean_accuracy": float(clean_correct.mean()),
        "clean_on_target": float(clean_on_target.mean()),
    }
    return reading


def mean_readout(readings):
    keys = [key for key in readings[0] if isinstance(readings[0][key], float)]
    mean = {key: sum(r[key] for r in readings) / len(readings) for key in keys}
    return mean


def random_non_trigger_positions(trigger, count, draw, device):
    generator = torch.Generator().manual_seed(DRAW_SEED + draw)
    order = torch.randperm(PATCH_TOKENS, generator=generator) + 1  # (196,)
    candidates = order[~torch.isin(order, trigger.cpu())]  # (196 - trigger_count,)
    positions = candidates[:count].to(device)  # (count,)
    assert len(positions) == count, f"only {len(positions)} free positions"
    return positions


def predict_pairs(model, pairs, batch_size, device):
    predictions = {
        "clean": predict(model, pairs["clean"], batch_size, device),
        "triggered": predict(model, pairs["triggered"], batch_size, device),
    }
    return predictions


def predict(model, images, batch_size, device):
    parts = [
        predict_batch(model, images[start : start + batch_size], device)
        for start in range(0, len(images), batch_size)
    ]
    predictions = torch.cat(parts)  # (n,)
    return predictions


@torch.inference_mode()
def predict_batch(model, images, device):
    logits = forward_logits(model, images, device, use_bfloat16=True)  # (b, classes)
    predictions = logits.argmax(dim=1)  # (b,)
    return predictions


def load_pairs(checkpoint_path, args, device):
    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=0,
    )
    clean_set = loaders["clean"].dataset
    backdoor_set = loaders["backdoor"].dataset
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
        "clean_labels": torch.tensor(
            [int(label) for _, label in clean_items], device=device
        ),
        "triggered": torch.stack([image for image, _ in triggered_items]).to(device),
        "targets": torch.tensor(
            [int(target) for _, target in triggered_items], device=device
        ),
    }
    assert pairs["clean"].shape == pairs["triggered"].shape, "pairs must align"
    return pairs


def adaptive_rates(results_dir, folder):
    report = load_psbd_metrics(results_dir, folder)
    placements = report["placements"]
    tm_rate = psbd_rate(placements[RECOMMENDED_PLACEMENT], "adaptive")
    rd_rate = psbd_rate(placements[PUBLISHED_PLACEMENT], "adaptive")
    assert tm_rate is not None and rd_rate is not None, f"{folder} has no adaptive rate"
    return tm_rate, rd_rate


def panel_folders(results_dir):
    cells = clearing_cells(load_coverage(results_dir))
    patch_folders = [c["folder_name"] for c in cells if c["attack"] in PATCH_ATTACKS]

    by_attack_and_dataset = collections.defaultdict(dict)
    for cell in cells:
        by_attack_and_dataset[(cell["attack"], cell["dataset"])][
            cell["poison_rate"]
        ] = cell["folder_name"]
    subset_folders = []
    for rates in by_attack_and_dataset.values():
        preferred = [rate for rate in SUBSET_RATE_PREFERENCE if rate in rates]
        chosen = preferred[0] if preferred else sorted(rates)[0]
        subset_folders.append(rates[chosen])
    return patch_folders, subset_folders


def limit_gpu_memory(gigabytes, device):
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, gigabytes * 1024**3 / total), device
    )


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def read_model_records(output_dir):
    records = []
    for name in sorted(os.listdir(output_dir)):
        if not name.endswith(".json") or name == "summary.json":
            continue
        with open(os.path.join(output_dir, name)) as handle:
            records.append(json.load(handle))
    return records


def ratio(numerator, denominator):
    value = numerator / denominator if denominator > 0 else None
    return value


def summarize(records):
    by_group = collections.defaultdict(list)
    subsets_by_group = collections.defaultdict(list)
    for record in records:
        by_group[analysis_group(record)].append(record)
        if "visible_subsets" in record:
            subsets_by_group[analysis_group(record)].append(record)
    patch_groups = [g for g in PATCH_GROUPS if g in by_group]
    patch_records = [r for g in patch_groups for r in by_group[g]]
    # Tiny's 3 pixel patch covers 1 token and every other patch trigger covers 4,
    # and the p^m argument turns on that count, so B is also split by it.
    by_group_and_size = collections.defaultdict(list)
    for record in patch_records:
        size = len(record["trigger_positions"])
        by_group_and_size[f"{analysis_group(record)}_{size}_tokens"].append(record)

    summary = {
        "models": len(records),
        "models_by_group": {g: len(rs) for g, rs in by_group.items()},
        "baseline": {
            g: average([r["baseline"] for r in rs]) for g, rs in by_group.items()
        },
        "per_model": [per_model_row(r) for r in patch_records],
        "deterministic_masking": {
            g: average_conditions([r["deterministic_masking"] for r in by_group[g]])
            for g in patch_groups
        },
        "stochastic_token_mask": {
            g: pool_stochastic(by_group[g]) for g in patch_groups
        },
        "stochastic_token_mask_by_trigger_tokens": {
            key: pool_stochastic(rs) for key, rs in sorted(by_group_and_size.items())
        },
        "residual_dropout": {
            g: average_conditions(
                [
                    {k: v for k, v in r["residual_dropout"].items() if k != "rate"}
                    for r in by_group[g]
                ]
            )
            for g in patch_groups
        },
        "visible_subsets": {
            g: {
                fraction: average(
                    [with_excess(r["visible_subsets"], fraction) for r in rs]
                )
                for fraction in map(str, KEEP_FRACTIONS)
            }
            for g, rs in subsets_by_group.items()
        },
        "visible_subsets_patch_by_trigger_visibility": split_by_visibility(
            [r for g in patch_groups for r in subsets_by_group[g]]
        ),
    }
    return summary


# TaCT's paired clean images all come from its 1 source class. Several TaCT
# models misclassify that whole class with no trigger present, so for them the
# triggered prediction is carried by the source content and not by the patch.
# Those are read separately from the models that classify the source correctly.
def analysis_group(record):
    if record["attack"] != "tact":
        return record["attack"]
    conditional = record["baseline"]["clean_accuracy"] >= TACT_CONDITIONAL_ACCURACY
    group = "tact_trigger_conditional" if conditional else "tact_source_mapped"
    return group


def per_model_row(record):
    masking = record["deterministic_masking"]
    dropout = record["residual_dropout"]
    row = {
        "folder": record["folder"],
        "group": analysis_group(record),
        "pairs": record["pairs"],
        "trigger_tokens": len(record["trigger_positions"]),
        "tm_rate": record["tm_rate"],
        "rd_rate": record["rd_rate"],
        "baseline_clean_accuracy": record["baseline"]["clean_accuracy"],
        "baseline_clean_on_target": record["baseline"]["clean_on_target"],
        "mask_trigger_all_12": masking["all_12"]["triggered_kept"],
        "mask_trigger_last_4": masking["last_4"]["triggered_kept"],
        "mask_random_all_12": masking["random_all_12"]["triggered_kept"],
        "tm_triggered_kept": record["stochastic_token_mask"]["overall"][
            "triggered_kept"
        ],
        "tm_clean_kept": record["stochastic_token_mask"]["overall"]["clean_kept"],
        "rd_all_triggered_kept": dropout["all_tokens"]["triggered_kept"],
        "rd_trigger_only_triggered_kept": dropout["trigger_only"]["triggered_kept"],
        "rd_all_but_trigger_triggered_kept": dropout["all_but_trigger"][
            "triggered_kept"
        ],
        "rd_random_triggered_kept": dropout["random_same_count"]["triggered_kept"],
        "rd_all_clean_kept": dropout["all_tokens"]["clean_kept"],
    }
    return row


# A heavily masked model collapses onto 1 default class, and on some models
# that class is the target, which reads as triggered retention. Subtracting the
# share of clean images sent to the target removes that floor.
def with_excess(subsets, fraction):
    reading = dict(subsets[fraction]["mean"])
    reference = subsets["1.0"]["mean"]
    reading["excess_retention"] = ratio(
        reading["triggered_asr"] - reading["clean_on_target"],
        reference["triggered_asr"] - reference["clean_on_target"],
    )
    return reading


def average(readings):
    # A retention is None when its reference is 0, so keys are gathered from
    # every reading rather than the first.
    keys = sorted(
        {
            key
            for reading in readings
            for key, value in reading.items()
            if isinstance(value, (float, int))
        }
    )
    mean = {}
    for key in keys:
        values = [r[key] for r in readings if r.get(key) is not None]
        mean[key] = sum(values) / len(values) if values else None
    mean["models"] = len(readings)
    return mean


def average_conditions(condition_sets):
    averaged = {
        name: average([conditions[name] for conditions in condition_sets])
        for name in condition_sets[0]
        if isinstance(condition_sets[0][name], dict)
    }
    return averaged


def pool_stochastic(records):
    blocks = [r["stochastic_token_mask"] for r in records]
    pooled = {"overall": average([b["overall"] for b in blocks])}
    for split in ("triggered", "clean"):
        pooled[split] = {}
        for statistic in blocks[0][split]:
            counts = collections.defaultdict(lambda: [0, 0])
            for block in blocks:
                for value, (kept, total) in block[split][statistic].items():
                    counts[value][0] += kept
                    counts[value][1] += total
            pooled[split][statistic] = {
                value: {"kept": kept / total, "n": total}
                for value, (kept, total) in sorted(
                    counts.items(), key=lambda i: int(i[0])
                )
            }
    return pooled


def split_by_visibility(patch_records):
    split = {}
    for fraction in map(str, KEEP_FRACTIONS[1:]):
        groups = collections.defaultdict(lambda: collections.defaultdict(list))
        for record in patch_records:
            reference = record["visible_subsets"]["1.0"]["mean"]
            reference_excess = reference["triggered_asr"] - reference["clean_on_target"]
            for draw in record["visible_subsets"][fraction]["draws"]:
                key = "trigger_visible" if draw["trigger_visible"] else "trigger_hidden"
                groups[key]["asr_retention"].append(
                    ratio(draw["triggered_asr"], reference["triggered_asr"])
                )
                groups[key]["excess_retention"].append(
                    ratio(
                        draw["triggered_asr"] - draw["clean_on_target"],
                        reference_excess,
                    )
                )
        split[fraction] = {
            key: {
                "asr_retention": sum(v["asr_retention"]) / len(v["asr_retention"]),
                "excess_retention": sum(v["excess_retention"])
                / len(v["excess_retention"]),
                "draws": len(v["asr_retention"]),
            }
            for key, v in sorted(groups.items())
        }
    return split


def markdown_tables(summary):
    lines = ["\nPatch models, 1 row each"]
    lines.append("| model | group | " + " | ".join(PER_MODEL_COLUMNS) + " |")
    lines.append("|---|---|" + "---|" * len(PER_MODEL_COLUMNS))
    for row in summary["per_model"]:
        cells = " | ".join(
            str(row[c]) if isinstance(row[c], int) else f"{row[c]:.2f}"
            for c in PER_MODEL_COLUMNS
        )
        lines.append(f"| {row['folder']} | {row['group']} | {cells} |")

    lines.append(
        "\nA  deterministic masking of the trigger tokens at the attention input"
    )
    lines.append(
        "| attack | condition | triggered kept | clean kept | clean accuracy "
        "| clean on target |"
    )
    lines.append("|---|---|---|---|---|---|")
    for attack, conditions in summary["deterministic_masking"].items():
        for name, reading in conditions.items():
            lines.append(
                f"| {attack} | {name} | {reading['triggered_kept']:.3f} | "
                f"{reading['clean_kept']:.3f} | {reading['clean_accuracy']:.3f} | "
                f"{reading['clean_on_target']:.3f} |"
            )

    lines.append("\nB  PSBD-TM at the adaptive rate, P(kept) by trigger masking count")
    for group_name in (
        "stochastic_token_mask",
        "stochastic_token_mask_by_trigger_tokens",
    ):
        for group, pooled in summary[group_name].items():
            overall = pooled["overall"]
            lines.append(
                f"{group_name} {group}: triggered kept {overall['triggered_kept']:.3f}, "
                f"clean kept {overall['clean_kept']:.3f}"
            )
            for statistic in PRINTED_STATISTICS:
                lines.append(
                    f"| {statistic} | triggered P(kept) (n) | clean P(kept) (n) |"
                )
                lines.append("|---|---|---|")
                values = sorted(
                    set(pooled["triggered"][statistic])
                    | set(pooled["clean"][statistic]),
                    key=int,
                )
                for value in values:
                    cells = []
                    for split in ("triggered", "clean"):
                        entry = pooled[split][statistic].get(value)
                        cells.append(
                            f"{entry['kept']:.3f} ({entry['n']})" if entry else "n/a"
                        )
                    lines.append(f"| {value} | {cells[0]} | {cells[1]} |")

    lines.append("\nC  PSBD-RD at the adaptive rate by the tokens it may touch")
    lines.append(
        "| attack | variant | triggered kept | clean kept | clean accuracy "
        "| clean on target |"
    )
    lines.append("|---|---|---|---|---|---|")
    for attack, variants in summary["residual_dropout"].items():
        for name, reading in variants.items():
            lines.append(
                f"| {attack} | {name} | {reading['triggered_kept']:.3f} | "
                f"{reading['clean_kept']:.3f} | {reading['clean_accuracy']:.3f} | "
                f"{reading['clean_on_target']:.3f} |"
            )

    lines.append("\nD  retention with a fixed subset of tokens visible in every block")
    header = " | ".join(f"f={fraction}" for fraction in map(str, KEEP_FRACTIONS[1:]))
    lines.append(f"| attack | models | quantity | {header} |")
    lines.append("|---|---|---|" + "---|" * (len(KEEP_FRACTIONS) - 1))
    for attack, fractions in summary["visible_subsets"].items():
        for quantity in (
            "asr_retention",
            "excess_retention",
            "clean_accuracy_retention",
            "clean_on_target",
        ):
            cells = " | ".join(
                format_number(fractions[fraction].get(quantity))
                for fraction in map(str, KEEP_FRACTIONS[1:])
            )
            lines.append(
                f"| {attack} | {fractions['1.0']['models']} | {quantity} | {cells} |"
            )
    lines.append(
        "\nD on patch models, ASR retention by whether a trigger token stayed visible"
    )
    for fraction, groups in summary[
        "visible_subsets_patch_by_trigger_visibility"
    ].items():
        text = ", ".join(
            f"{key} ASR {entry['asr_retention']:.3f} excess "
            f"{entry['excess_retention']:.3f} over {entry['draws']} draws"
            for key, entry in groups.items()
        )
        lines.append(f"f={fraction}: {text}")

    table_text = "\n".join(lines)
    return table_text


def format_number(value):
    text = "n/a" if value is None else f"{value:.3f}"
    return text


if __name__ == "__main__":
    main()
