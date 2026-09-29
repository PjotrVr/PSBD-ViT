"""Which tokens and blocks carry the trigger read and the source-class read, on ViT.

Mechanism analysis on the GPU, inference only. It uses attacker knowledge (the
trigger's token positions, each image's true class and the attack label), which no
detection statistic reads. On 256 paired clean and triggered test images per model
it masks tokens deterministically at the attention input (the PSBD-TM position)
and reads where the triggered prediction goes: the attack label, the source class
y, or elsewhere.

    trigger_blocks_<span>   the trigger's tokens masked in a span of blocks only
    content_keep_<f>        every block sees the trigger and only a random share f
                            of the other patch tokens (3 draws)
    random_all_12           as many random non-trigger tokens as the trigger has,
                            masked in every block, the control for the first family

All-to-one BadNets models with the same patch are the control: their shortcut does
not need the content, so hiding content should leave the target in place.

    source .venv/bin/activate
    flock scratch/gpu.lock python experiments/all_to_all_detection/tokens.py \\
        --folders vit_cifar10_badnet_a2a_0_1
    python experiments/all_to_all_detection/tokens.py --summarize-only
"""

import argparse
import glob
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from data.splits import read_checkpoint_metadata  # noqa: E402
from experiments.residual_stream_mechanism.activation_patching import (  # noqa: E402
    trigger_tokens,
)
from experiments.why_token_masking_works.measure import (  # noqa: E402
    NUM_BLOCKS,
    PATCH_TOKENS,
    TM_POSITION,
    FixedTokenMask,
    load_pairs,
    predict_pairs,
    random_non_trigger_positions,
)
from models.backbones import load_checkpoint  # noqa: E402
from models.positions import plug_dropout, unplug_dropout  # noqa: E402
from panel import OUT_DIR, write_json  # noqa: E402

PAIR_COUNT = 256
BATCH_SIZE = 128
GPU_MEMORY_FRACTION = 0.15
TORCH_THREADS = 4
SPANS = {
    "all_12": (1, 12),
    "blocks_1_4": (1, 4),
    "blocks_5_8": (5, 8),
    "blocks_9_12": (9, 12),
    "blocks_1_8": (1, 8),
    "blocks_5_12": (5, 12),
}
CONTENT_KEEP_FRACTIONS = (0.6, 0.3, 0.1, 0.0)
RANDOM_DRAWS = 3
DRAW_SEED = 0
TOKEN_DIR = os.path.join(OUT_DIR, "tokens")


def main():
    args = parse_args()
    if not args.summarize_only:
        torch.set_num_threads(TORCH_THREADS)
        device = torch.device("cuda", 0)
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
        os.makedirs(TOKEN_DIR, exist_ok=True)
        for folder in args.folders:
            out_path = os.path.join(TOKEN_DIR, f"{folder}.json")
            if os.path.exists(out_path):
                print(f"[skip] {folder}", flush=True)
                continue
            started = time.time()
            record = measure_model(folder, args, device)
            record["seconds"] = round(time.time() - started, 1)
            with open(out_path, "w") as handle:
                json.dump(record, handle, indent=2)
            print(f"[ok] {folder} {record['seconds']}s", flush=True)

    records = [
        json.load(open(path)) for path in sorted(glob.glob(f"{TOKEN_DIR}/*.json"))
    ]
    print(write_json({"models": records}, "tokens.json"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folders", nargs="+", default=[])
    parser.add_argument(
        "--checkpoints-dir", default=os.path.join(REPO_ROOT, "checkpoints")
    )
    parser.add_argument("--raw-data-dir", default=os.path.join(REPO_ROOT, "raw_data"))
    parser.add_argument("--pairs", type=int, default=PAIR_COUNT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def measure_model(folder, args, device):
    checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)
    model = load_checkpoint(metadata["architecture"], checkpoint_path, device).eval()
    pairs = load_pairs(checkpoint_path, args, device)

    # trigger_tokens indexes the 196 patches, the sequence carries CLS at 0.
    trigger = (trigger_tokens(metadata) + 1).to(device)  # (trigger_count,)
    baseline = predict_pairs(model, pairs, args.batch_size, device)
    pairs["captured"] = (baseline["triggered"] == pairs["targets"]) & (
        baseline["clean"] == pairs["clean_labels"]
    )  # (n,)

    conditions = {"baseline": readout(pairs, baseline)}
    for name, block_range in SPANS.items():
        predictions = masked_predictions(
            model, pairs, trigger, block_range, args, device
        )
        conditions[f"trigger_blocks_{name}"] = readout(pairs, predictions)

    draws = []
    for draw in range(RANDOM_DRAWS):
        positions = random_non_trigger_positions(trigger, len(trigger), draw, device)
        predictions = masked_predictions(
            model, pairs, positions, (1, NUM_BLOCKS), args, device
        )
        draws.append(readout(pairs, predictions))
    conditions["random_all_12"] = mean_of(draws)

    for fraction in CONTENT_KEEP_FRACTIONS:
        draws = []
        for draw in range(RANDOM_DRAWS):
            positions = hidden_content_positions(trigger, fraction, draw, device)
            predictions = masked_predictions(
                model, pairs, positions, (1, NUM_BLOCKS), args, device
            )
            draws.append(readout(pairs, predictions))
        conditions[f"content_keep_{fraction}"] = mean_of(draws)

    record = {
        "folder": folder,
        "attack": metadata["attack"],
        "label_mode": metadata.get("label_mode"),
        "dataset": metadata["dataset"],
        "poison_rate": metadata["poison_rate"],
        "pairs": int(len(pairs["targets"])),
        "captured_pairs": int(pairs["captured"].sum()),
        "trigger_positions": trigger.tolist(),
        "conditions": conditions,
    }
    return record


def masked_predictions(model, pairs, positions, block_range, args, device):
    handles = plug_dropout(
        model,
        "vit",
        (TM_POSITION,),
        {TM_POSITION: lambda _rate: FixedTokenMask(positions)},
        0.0,
        block_range=block_range,
    )
    try:
        predictions = predict_pairs(model, pairs, args.batch_size, device)
    finally:
        unplug_dropout(handles)
    return predictions


# Every patch token outside the trigger is a content token. A share of them stays
# visible (drawn per draw) and the rest are masked in every block.
def hidden_content_positions(trigger, fraction, draw, device):
    generator = torch.Generator().manual_seed(DRAW_SEED + draw)
    order = torch.randperm(PATCH_TOKENS, generator=generator) + 1  # (196,)
    content = order[~torch.isin(order, trigger.cpu())]  # (196 - trigger_count,)
    visible_count = round(fraction * len(content))
    hidden = content[visible_count:].to(device)  # (hidden,)
    return hidden


# Shares over the captured pairs only: the clean twin is classified correctly and
# the triggered twin reaches the attack label, so the source class is known and
# the attack label differs from it.
def readout(pairs, predictions):
    captured = pairs["captured"]
    triggered = predictions["triggered"][captured]  # (m,)
    clean = predictions["clean"][captured]  # (m,)
    source = pairs["clean_labels"][captured]  # (m,)
    attack_label = pairs["targets"][captured]  # (m,)

    on_attack = (triggered == attack_label).float()  # (m,)
    on_source = (triggered == source).float()  # (m,)
    reading = {
        "triggered_on_attack_label": float(on_attack.mean()),
        "triggered_on_source": float(on_source.mean()),
        "triggered_elsewhere": float((1 - on_attack - on_source).mean()),
        "clean_accuracy": float((clean == source).float().mean()),
    }
    return reading


def mean_of(readings):
    mean = {key: sum(r[key] for r in readings) / len(readings) for key in readings[0]}
    return mean


if __name__ == "__main__":
    main()
