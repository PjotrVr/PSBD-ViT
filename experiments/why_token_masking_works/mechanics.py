"""3 mechanics the theory memo derived and asked to be measured, on ViT-B/16.

docs/why-psbd-works-theory.md derives 3 properties of the probes that none of the
other scripts here measures directly. Each has a prediction that can fail.

    sink       With eps 1e-6 a zeroed token leaves ln_1 as exactly its bias beta,
               so every masked token of a block carries the same key W_k beta + b_k
               and the masked set can act as 1 attention sink. The class token's
               attention mass on masked positions is measured against the rate,
               for token_mask and for token_substitute, whose replaced tokens are
               real tokens of the same image (the literature memo's L22).
    coupling   GaussianNoise scales its noise by the per-sample spread over every
               token and channel, and a patch trigger makes a few tokens large.
               The prediction is that a triggered image's spread exceeds its clean
               twin's in blocks 5 to 12 for BadNets and not for global triggers,
               so triggered images receive more noise on every ordinary token.
               The same pairs are then scored with the library Gaussian and with
               noise scaled per token, each at the rate that changes 60% of clean
               predictions, to see whether per-token scaling removes the inversion.
    embedding  At after_embedding there is no LayerNorm between the mask and the
               residual stream, so TokenMask's 1/(1-p) rescale of the surviving
               tokens enters the stream (3.3 times at p 0.7). Triggered survival by
               the number of surviving trigger tokens is recorded with and without
               that rescale at 2 fixed rates.

    source .venv/bin/activate
    PYTHONPATH=. python experiments/why_token_masking_works/mechanics.py \
        --folders vit_gtsrb_badnet_a2o_0_05
"""

import argparse
import math
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from lightning import seed_everything  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from data.splits import read_checkpoint_metadata  # noqa: E402
from defenses.inference import forward_probs  # noqa: E402
from defenses.operators import (  # noqa: E402
    GaussianNoise,
    TokenSubstitute,
    _keep_scale,
)
from experiments._paths import experiment_results_dir  # noqa: E402
from experiments.why_token_masking_works.measure import (  # noqa: E402
    RecordingTokenMask,
    kept_by_count,
    limit_gpu_memory,
    load_pairs,
    predict_pairs,
    readout,
    write_json,
)
from experiments.why_token_masking_works.swin import (  # noqa: E402
    assert_pristine,
    read_json_or_none,
)
from experiments.why_token_masking_works.tokens import (  # noqa: E402
    block_trigger_positions,
    cached_baseline_agreement,
    model_blocks,
    trigger_pixel_map,
)
from models.backbones import load_checkpoint  # noqa: E402
from models.positions import plug_dropout, unplug_dropout  # noqa: E402

SLUG = "why_token_masking_works"
SUBDIRECTORY = "mechanics"
ARCHITECTURE = "vit"
PAIR_COUNT = 256
BATCH_SIZE = 128
GPU_MEMORY_GB = 14.0
MASK_SEED = 0
CALIBRATION_SEED = 1000
SINK_RATES = (0.1, 0.3, 0.5, 0.7, 0.9)
COUPLING_PASSES = 6
CALIBRATION_PASSES = 2
CALIBRATION_STEPS = 7
CLEAN_CHANGE_TARGET = 0.6
GAUSSIAN_BOUNDS = (0.01, 20.0)
EMBEDDING_RATES = (0.5, 0.7)
EMBEDDING_PASSES = 10
REPRODUCTION_FLOOR = 0.98
SITE = "before_attention_norm"
HEADS = 12


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--folders", nargs="+", required=True)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--pairs", type=int, default=PAIR_COUNT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gpu-memory-gb", type=float, default=GPU_MEMORY_GB)
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
        measure_model(folder, os.path.join(output_dir, f"{folder}.json"), args, device)
        print(f"[ok] {folder} {time.time() - started:.0f}s", flush=True)


def measure_model(folder, out_path, args, device):
    checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)
    assert metadata["architecture"] == ARCHITECTURE, "the sink needs a class token"
    model = load_checkpoint(ARCHITECTURE, checkpoint_path, device).eval()
    assert_pristine(model)
    pairs = load_pairs(checkpoint_path, args, device)
    trigger = block_trigger_positions(trigger_pixel_map(metadata), ARCHITECTURE)
    baseline = predict_pairs(model, pairs, args.batch_size, device)
    pairs["clean_base"] = baseline["clean"]  # (n,)
    pairs["triggered_base"] = baseline["triggered"]  # (n,)
    agreement = cached_baseline_agreement(args.results_dir, folder, pairs)
    assert min(agreement.values()) >= REPRODUCTION_FLOOR, f"{folder} {agreement}"

    record = read_json_or_none(out_path) or {
        "folder": folder,
        "attack": metadata["attack"],
        "dataset": metadata["dataset"],
        "poison_rate": metadata["poison_rate"],
        "pairs": int(len(pairs["targets"])),
        "trigger_tokens": len(trigger[0]),
        "baseline_agreement_with_cache": agreement,
        "baseline": readout(
            pairs, baseline["clean"][None], baseline["triggered"][None]
        ),
    }
    parts = {
        "sink": lambda: measure_sink(model, pairs, trigger, args.batch_size, device),
        "coupling": lambda: measure_coupling(
            model, pairs, trigger, args.batch_size, device
        ),
        "embedding": lambda: measure_embedding(
            model, pairs, trigger[0], args.batch_size, device
        ),
    }
    for name, measure in parts.items():
        if name in record:
            continue
        record[name] = measure()
        assert_pristine(model)
        write_json(out_path, record)
    return record


def measure_sink(model, pairs, trigger, batch_size, device):
    result = {}
    for operator_name, recorder_class in (
        ("token_mask", RecordingTokenMask),
        ("token_substitute", RecordingTokenSubstitute),
    ):
        rows = []
        for rate in SINK_RATES:
            recorders = []

            def build(probe_rate, recorder_class=recorder_class):
                recorder = recorder_class(probe_rate)
                recorders.append(recorder)
                return recorder

            handles = plug_dropout(model, ARCHITECTURE, (SITE,), {SITE: build}, rate)
            store = {}
            handles += attach_cls_attention(model, recorders, trigger, store)
            try:
                row = {"rate": rate}
                for split in ("clean", "triggered"):
                    seed_everything(MASK_SEED, verbose=False)
                    store.clear()
                    predictions = []
                    for start in range(0, len(pairs[split]), batch_size):
                        batch = pairs[split][start : start + batch_size]
                        predictions.append(predict_batch_argmax(model, batch, device))
                    row[split] = summarize_store(store)
                    row[split]["kept"] = float(
                        (torch.cat(predictions) == pairs[f"{split}_base"])
                        .float()
                        .mean()
                    )
            finally:
                unplug_dropout(handles)
            rows.append(row)
        result[operator_name] = rows
    return result


# The class token's attention weights in 1 block, recomputed from the attention
# module's own input, since torchvision calls it with need_weights=False. The
# weights are averaged over heads and summed over the positions the block's
# probe replaced, over the trigger positions and over every patch token.
def attach_cls_attention(model, recorders, trigger, store):
    handles = []
    for index, block in enumerate(model_blocks(model, ARCHITECTURE)):
        attention = block.self_attention

        def hook(module, args, index=index):
            x = args[0].float()  # (b, 197, d)
            batch, tokens, width = x.shape
            head_width = width // HEADS
            weight = module.in_proj_weight.float()  # (3d, d)
            bias = module.in_proj_bias.float()  # (3d,)
            query = x[:, 0] @ weight[:width].T + bias[:width]  # (b, d)
            keys = (
                x @ weight[width : 2 * width].T + bias[width : 2 * width]
            )  # (b, t, d)
            query = query.view(batch, HEADS, head_width)  # (b, h, hd)
            keys = keys.view(batch, tokens, HEADS, head_width)  # (b, t, h, hd)
            scores = torch.einsum("bhc,bthc->bht", query, keys) / math.sqrt(head_width)
            weights = scores.softmax(dim=2).mean(dim=1)  # (b, t)

            replaced = recorders[index].last_dropped  # (b, t)
            patch_mass = weights[:, 1:].sum(dim=1)  # (b,)
            masked_mass = (weights * replaced).sum(dim=1)  # (b,)
            masked_share = replaced[:, 1:].float().mean(dim=1)  # (b,)
            trigger_mass = weights[:, trigger[index].to(x.device)].sum(dim=1)  # (b,)
            # With eps 1e-6 every zeroed token leaves the norm as the same beta,
            # so the rows of the masked positions should be identical.
            spread = masked_row_spread(x, replaced)
            entry = store.setdefault(
                index,
                {
                    "masked_mass": [],
                    "masked_share": [],
                    "patch_mass": [],
                    "trigger_mass": [],
                    "row_spread": [],
                },
            )
            entry["masked_mass"].append(masked_mass.cpu())
            entry["masked_share"].append(masked_share.cpu())
            entry["patch_mass"].append(patch_mass.cpu())
            entry["trigger_mass"].append(trigger_mass.cpu())
            entry["row_spread"].append(torch.tensor([spread]))

        handles.append(attention.register_forward_pre_hook(hook))
    return handles


def masked_row_spread(x, replaced):
    rows = x[replaced]  # (masked, d)
    if len(rows) < 2:
        return 0.0
    spread = float((rows - rows[:1]).abs().max())
    return spread


def summarize_store(store):
    summary = {"blocks": []}
    for index in sorted(store):
        entry = {key: torch.cat(values) for key, values in store[index].items()}
        summary["blocks"].append(
            {
                "masked_mass": float(entry["masked_mass"].mean()),
                "masked_share": float(entry["masked_share"].mean()),
                "patch_mass": float(entry["patch_mass"].mean()),
                "trigger_mass": float(entry["trigger_mass"].mean()),
                "masked_rows_max_difference": float(entry["row_spread"].max()),
            }
        )
    return summary


# TokenSubstitute's own body with the (batch, patches) choice of replaced tokens
# stored, in the full token layout with CLS at 0 so it lines up with
# RecordingTokenMask.last_dropped.
class RecordingTokenSubstitute(TokenSubstitute):
    def __init__(self, rate, protect_cls=True):
        super().__init__(rate, protect_cls)
        self.last_dropped = None

    def _substitute(self, x, protect_cls):
        batch, tokens, _ = x.shape  # (batch, tokens, channels)
        offset = 1 if protect_cls else 0
        patches = x[:, offset:, :]  # (batch, patches, channels)
        shifts = torch.randint(1, patches.shape[1], (batch,), device=x.device)
        index = (
            torch.arange(patches.shape[1], device=x.device).view(1, -1)
            - shifts.view(-1, 1)
        ) % patches.shape[1]  # (batch, patches)
        donor = torch.gather(patches, 1, index.unsqueeze(-1).expand_as(patches))
        replace = torch.empty(
            batch, patches.shape[1], 1, device=x.device, dtype=x.dtype
        ).bernoulli_(self.rate)  # (batch, patches, 1)
        dropped = torch.zeros(batch, tokens, dtype=torch.bool, device=x.device)
        dropped[:, offset:] = replace[:, :, 0] == 1
        self.last_dropped = dropped  # (batch, tokens)

        substituted = x.clone()
        substituted[:, offset:, :] = patches * (1.0 - replace) + donor * replace
        return substituted


def measure_coupling(model, pairs, trigger, batch_size, device):
    spreads = {}
    for split in ("clean", "triggered"):
        spreads[split] = block_input_spreads(
            model, pairs[split], trigger, batch_size, device
        )
    ratios = {
        key: [
            float((spreads["triggered"][key][b] / spreads["clean"][key][b]).mean())
            for b in range(len(trigger))
        ]
        for key in ("sample", "ordinary_token", "trigger_token")
    }

    scoring = {}
    for name, factory in (
        ("gaussian", GaussianNoise),
        ("gaussian_per_token", PerTokenGaussianNoise),
    ):
        rate, calibration = calibrate_gaussian(
            model, pairs, factory, batch_size, device
        )
        scoring[name] = {"rate": rate, "calibration": calibration} | pair_auroc(
            model, pairs, factory, rate, batch_size, device
        )
    result = {"spread_ratio_triggered_over_clean": ratios, "scoring": scoring}
    return result


# The spread GaussianNoise uses at before_attention_norm: the std of 1 image's
# block input over every token and channel. It is compared with the std of the
# ordinary tokens and of the trigger tokens alone.
@torch.inference_mode()
def block_input_spreads(model, images, trigger, batch_size, device):
    blocks = model_blocks(model, ARCHITECTURE)
    captured = {
        key: [[] for _ in blocks]
        for key in ("sample", "ordinary_token", "trigger_token")
    }
    handles = []
    for index, block in enumerate(blocks):

        def hook(_module, args, index=index):
            x = args[0].float()  # (b, 197, d)
            positions = trigger[index].to(x.device)
            others = torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
            others[positions] = False
            others[0] = False
            captured["sample"][index].append(x.std(dim=(1, 2)).cpu())  # (b,)
            captured["ordinary_token"][index].append(
                x[:, others].std(dim=2).mean(dim=1).cpu()
            )
            captured["trigger_token"][index].append(
                x[:, positions].std(dim=2).mean(dim=1).cpu()
            )

        handles.append(block.ln_1.register_forward_pre_hook(hook))
    try:
        for start in range(0, len(images), batch_size):
            forward_probs(model, images[start : start + batch_size], device, True)
    finally:
        for handle in handles:
            handle.remove()
    spreads = {key: [torch.cat(v) for v in values] for key, values in captured.items()}
    return spreads


# Gaussian noise whose scale is each token's own std over its channels, so a
# high-norm trigger token no longer raises the noise every other token gets.
class PerTokenGaussianNoise(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = float(rate)

    def forward(self, x):
        if not self.training or self.rate == 0.0:
            return x
        scale = x.detach().std(dim=-1, keepdim=True)  # (batch, tokens, 1)
        noised = x + torch.randn_like(x) * (self.rate * scale)  # same shape as x
        return noised


def calibrate_gaussian(model, pairs, factory, batch_size, device):
    low, high = GAUSSIAN_BOUNDS
    evaluated = {}

    def clean_change(rate):
        if rate not in evaluated:
            handles = plug_dropout(model, ARCHITECTURE, (SITE,), {SITE: factory}, rate)
            try:
                seed_everything(CALIBRATION_SEED, verbose=False)
                changed = [
                    predict_all(model, pairs["clean"], batch_size, device)
                    != pairs["clean_base"]
                    for _ in range(CALIBRATION_PASSES)
                ]
            finally:
                unplug_dropout(handles)
            evaluated[rate] = float(torch.stack(changed).float().mean())
        return evaluated[rate]

    reached = clean_change(high) >= CLEAN_CHANGE_TARGET
    for _ in range(CALIBRATION_STEPS if reached else 0):
        middle = math.sqrt(low * high)
        if clean_change(middle) < CLEAN_CHANGE_TARGET:
            low = middle
        else:
            high = middle
    rate = min(evaluated, key=lambda r: abs(evaluated[r] - CLEAN_CHANGE_TARGET))
    calibration = {
        "reached": reached,
        "clean_change_at_rate": evaluated[rate],
        "evaluated": {str(r): v for r, v in sorted(evaluated.items())},
    }
    return rate, calibration


# Fractional PSU of every paired clean and triggered image, k passes at 1 rate,
# and the AUROC of low PSU as poisoned (the direction PSBD reads).
@torch.inference_mode()
def pair_auroc(model, pairs, factory, rate, batch_size, device):
    scores = {}
    for split in ("clean", "triggered"):
        base_probs = probs_all(model, pairs[split], batch_size, device)  # (n, classes)
        labels = base_probs.argmax(dim=1)  # (n,)
        tracked = base_probs.gather(1, labels[:, None])[:, 0].clamp_min(1e-6)  # (n,)
        handles = plug_dropout(model, ARCHITECTURE, (SITE,), {SITE: factory}, rate)
        try:
            seed_everything(MASK_SEED, verbose=False)
            passes = torch.stack(
                [
                    probs_all(model, pairs[split], batch_size, device).gather(
                        1, labels[:, None]
                    )[:, 0]
                    for _ in range(COUPLING_PASSES)
                ]
            )  # (k, n)
        finally:
            unplug_dropout(handles)
        scores[split] = (1 - passes.mean(dim=0) / tracked).cpu().numpy()  # (n,)
    labels = np.concatenate(
        [np.zeros(len(scores["clean"])), np.ones(len(scores["triggered"]))]
    )
    values = -np.concatenate([scores["clean"], scores["triggered"]])
    result = {
        "auroc_on_pairs": float(roc_auc_score(labels, values)),
        "psu_clean_mean": float(scores["clean"].mean()),
        "psu_triggered_mean": float(scores["triggered"].mean()),
    }
    return result


def measure_embedding(model, pairs, trigger, batch_size, device):
    result = {}
    for rate in EMBEDDING_RATES:
        for rescale in (True, False):
            key = f"rate_{rate}_{'rescaled' if rescale else 'unscaled'}"
            result[key] = embedding_masks(
                model, pairs, trigger, rate, rescale, batch_size, device
            )
    return result


# The library TokenMask at after_embedding rescales the survivors by 1/(1-p) and
# nothing downstream normalizes the stream entry, so the unscaled variant keeps
# the surviving tokens exactly as they were.
class RecordingUnscaledTokenMask(RecordingTokenMask):
    def _mask_tokens(self, x, protect_cls):
        masked = super()._mask_tokens(x, protect_cls)  # (batch, tokens, channels)
        unscaled = masked / _keep_scale(self.rate)
        if protect_cls:
            unscaled[:, 0, :] = x[:, 0, :]
        return unscaled


def embedding_masks(model, pairs, trigger, rate, rescale, batch_size, device):
    recorders = []
    recorder_class = RecordingTokenMask if rescale else RecordingUnscaledTokenMask

    def build(probe_rate):
        recorder = recorder_class(probe_rate)
        recorders.append(recorder)
        return recorder

    handles = plug_dropout(
        model, ARCHITECTURE, ("after_embedding",), {"after_embedding": build}, rate
    )
    columns = {}
    try:
        for split in ("clean", "triggered"):
            seed_everything(MASK_SEED, verbose=False)
            predictions, survivors = [], []
            for _ in range(EMBEDDING_PASSES):
                pass_predictions, pass_survivors = [], []
                for start in range(0, len(pairs[split]), batch_size):
                    batch = pairs[split][start : start + batch_size]
                    pass_predictions.append(predict_batch_argmax(model, batch, device))
                    dropped = recorders[0].last_dropped[:, trigger.to(device)]  # (b, t)
                    pass_survivors.append((~dropped).sum(dim=1))  # (b,)
                predictions.append(torch.cat(pass_predictions))
                survivors.append(torch.cat(pass_survivors))
            columns[split] = (
                torch.stack(predictions),
                torch.stack(survivors),
            )  # (k, n)
    finally:
        unplug_dropout(handles)

    clean_predictions, clean_survivors = columns["clean"]
    triggered_predictions, triggered_survivors = columns["triggered"]
    hit = (pairs["triggered_base"] == pairs["targets"])[None].expand_as(
        triggered_predictions
    )  # (k, n)
    everyone = torch.ones_like(hit)
    result = {
        "rate": rate,
        "rescaled": rescale,
        "overall": readout(pairs, clean_predictions, triggered_predictions),
        "triggered_by_surviving_trigger_tokens": kept_by_count(
            triggered_predictions == pairs["targets"][None], hit, triggered_survivors
        ),
        "clean_on_target_by_surviving_trigger_tokens": kept_by_count(
            clean_predictions == pairs["targets"][None], everyone, clean_survivors
        ),
        "clean_by_surviving_trigger_tokens": kept_by_count(
            clean_predictions == pairs["clean_base"][None], everyone, clean_survivors
        ),
    }
    return result


@torch.inference_mode()
def predict_batch_argmax(model, images, device):
    predictions = forward_probs(model, images, device, True).argmax(dim=1)  # (b,)
    return predictions


def predict_all(model, images, batch_size, device):
    predictions = torch.cat(
        [
            predict_batch_argmax(model, images[start : start + batch_size], device)
            for start in range(0, len(images), batch_size)
        ]
    )  # (n,)
    return predictions


@torch.inference_mode()
def probs_all(model, images, batch_size, device):
    probs = torch.cat(
        [
            forward_probs(model, images[start : start + batch_size], device, True)
            for start in range(0, len(images), batch_size)
        ]
    )  # (n, classes)
    return probs


if __name__ == "__main__":
    main()
