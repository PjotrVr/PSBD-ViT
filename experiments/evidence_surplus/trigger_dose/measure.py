"""B. Trigger dose-response: fade the trigger at test time, hold the detector fixed.

Predictions are in PREDICTIONS.md, written before this ran. For each model the
first 256 pairs of its PSBD eval ASR split are read, the trigger is faded in
steps and, at every step, ASR and the detector's AUROC (faded triggered image
against its clean twin, also on the faded images that still reach the target)
and TPR at the 0.01, 0.05 and 0.10 quantiles are measured at the canonical
operating point (experiments/evidence_surplus/common.py). Only the trigger
changes between steps.

The fade is linear in pixel space between the clean image and its triggered copy,
x(a) = x_clean + a (x_triggered - x_clean), which is the blend weight for Blend,
the amplitude for LF, the quantization difference for BPP and the patch opacity
for BadNets and TaCT. WaNet is faded by its warp strength, the trigger rebuilt at
a times the trained strength. BadNets and TaCT also get a partial-patch series
where only the pixels of the first j of the patch's tokens (ViT's 16 pixel
cells at the model's 224 input) carry the trigger.

    PYTHONPATH=. python experiments/evidence_surplus/trigger_dose/measure.py
"""

import argparse
import json
import os
import sys
import time
import types

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from attacks import apply_config_overrides, build_attack, default_config  # noqa: E402
from data.registry import DATASET_REGISTRY  # noqa: E402
from data.splits import read_checkpoint_metadata  # noqa: E402
from experiments._paths import experiment_results_dir  # noqa: E402
from experiments.evidence_surplus.common import (  # noqa: E402
    BACKDOORED,
    BENIGN,
    RESNET,
    SLUG,
    fractional_psu,
    limit_gpu_memory,
    logits_of,
    normalized,
    operating_point,
    pixel_space,
)
from experiments.residual_stream_mechanism.activation_patching import (  # noqa: E402
    trigger_tokens,
)
from experiments.why_psbd_works.measure import (  # noqa: E402
    auroc_low_is_positive,
    load_pairs,
)
from models.backbones import load_checkpoint  # noqa: E402

DOSES = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)
PAIRS = 256
PATCH_ATTACKS = ("badnet_a2o", "tact")
BENIGN_PROBES = ("badnet_a2o", "blend")
MODEL_INPUT = 224
PATCH = 16


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--gpu-memory-gb", type=float, default=14.0)
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    entries = (
        [m.partition(":")[::2] for m in args.models]
        if args.models
        else [(f, "") for f in BACKDOORED + RESNET]
        + [(f, p) for f in BENIGN for p in BENIGN_PROBES]
    )
    if args.list:
        print("\n".join(f"{f}:{p}" if p else f for f, p in entries))
        return
    torch.set_num_threads(4)
    device = torch.device("cuda", torch.cuda.current_device())
    limit_gpu_memory(args.gpu_memory_gb, device)
    out_dir = os.path.join(
        experiment_results_dir(SLUG, args.results_dir), "trigger_dose"
    )
    os.makedirs(out_dir, exist_ok=True)
    for folder, probe in entries:
        name = f"{folder}__probe_{probe}" if probe else folder
        path = os.path.join(out_dir, f"{name}.json")
        if os.path.exists(path):
            continue
        started = time.time()
        record = measure(folder, probe or None, args, device)
        record["seconds"] = round(time.time() - started, 1)
        with open(path, "w") as handle:
            json.dump(record, handle)
        print(f"[ok] {name} {record['seconds']}s", flush=True)


def measure(folder, probe, args, device):
    path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(path)
    architecture = metadata["architecture"]
    dataset = metadata["dataset"]
    model = load_checkpoint(architecture, path, device).eval()
    loader_args = types.SimpleNamespace(
        raw_data_dir=args.raw_data_dir, pairs=PAIRS, batch_size=128
    )
    pairs, _ = load_pairs(path, probe, loader_args, device)
    target = int(pairs["targets"][0])
    attack_name = probe or metadata["attack"]
    point = operating_point(args.results_dir, folder, architecture)
    clean_psu, _ = fractional_psu(model, architecture, point, pairs["clean"], device)

    clean_pixels = pixel_space(pairs["clean"], dataset)  # (n, C, H, W)
    triggered_pixels = pixel_space(pairs["triggered"], dataset)  # (n, C, H, W)
    series = {"dose": []}
    for dose in DOSES:
        if attack_name == "wanet":
            faded = warped(clean_pixels, metadata, attack_name, target, dose, pairs)
        else:
            faded = clean_pixels + dose * (triggered_pixels - clean_pixels)
        series["dose"].append(
            step_reading(
                dose,
                faded,
                clean_psu,
                model,
                architecture,
                point,
                target,
                dataset,
                device,
            )
        )
    if attack_name in PATCH_ATTACKS and architecture != "resnet18":
        series["partial_patch"] = partial_patch_series(
            clean_pixels,
            triggered_pixels,
            metadata,
            attack_name,
            target,
            clean_psu,
            model,
            architecture,
            point,
            dataset,
            device,
        )
    record = {
        "folder": folder,
        "probe_attack": probe,
        "architecture": architecture,
        "attack": attack_name,
        "dataset": dataset,
        "placement": point["placement"],
        "rate": point["rate"],
        "thresholds": point["thresholds"],
        "pairs": len(pairs["targets"]),
        "clean_flagged": {
            q: float((clean_psu < t).float().mean())
            for q, t in point["thresholds"].items()
        },
        **series,
    }
    return record


def step_reading(
    dose, faded_pixels, clean_psu, model, architecture, point, target, dataset, device
):
    images = normalized(faded_pixels, dataset)
    psu, labels = fractional_psu(model, architecture, point, images, device)
    hit = labels == target  # (n,)
    reading = {
        "dose": dose,
        "asr": float(hit.float().mean()),
        "auroc": auroc_low_is_positive(psu, clean_psu),
        "auroc_hits": auroc_low_is_positive(psu[hit], clean_psu) if hit.any() else None,
        "tpr": {
            q: float((psu < t).float().mean()) for q, t in point["thresholds"].items()
        },
        "median_psu": float(psu.median()),
        # The failure protocol's diagnosis for P2: whether a weaker trigger that
        # still flips an image leaves it a smaller target margin, the surplus the
        # account says the detector reads.
        "median_hit_margin": float(
            target_margin(model, images, target, device)[hit].median()
        )
        if hit.any()
        else None,
    }
    return reading


def target_margin(model, images, target, device):
    logits = logits_of(model, images, device)  # (n, classes)
    others = logits.clone()
    others[:, target] = float("-inf")
    margin = logits[:, target] - others.max(dim=1).values  # (n,)
    return margin


def warped(clean_pixels, metadata, attack_name, target, dose, pairs):
    if dose == 0.0:
        return clean_pixels
    size = DATASET_REGISTRY[metadata["dataset"]].image_size
    config = apply_config_overrides(
        default_config(attack_name), metadata.get("attack_config_overrides")
    )
    faded_config = apply_config_overrides(config, {"strength": config.strength * dose})
    attack = build_attack(attack_name, faded_config, size, target)
    faded = torch.stack(
        [
            attack.apply_trigger(image.cpu(), i).to(image.device)
            for i, image in enumerate(clean_pixels)
        ]
    )  # (n, C, H, W)
    return faded


# The trigger's pixels grouped by the 16 pixel cell of the model's 224 input they
# fall in, and the trigger kept on the first j of those cells only.
def partial_patch_series(
    clean_pixels,
    triggered_pixels,
    metadata,
    attack_name,
    target,
    clean_psu,
    model,
    architecture,
    point,
    dataset,
    device,
):
    size = clean_pixels.shape[-1]
    tokens = trigger_tokens(
        dict(metadata, attack=attack_name, target_label=target)
    ).tolist()
    rows = torch.arange(size) * MODEL_INPUT // size // PATCH  # (H,)
    grid = MODEL_INPUT // PATCH
    token_of = (rows[:, None] * grid + rows[None, :]).to(clean_pixels.device)  # (H, W)
    readings = []
    for count in range(1, len(tokens) + 1):
        kept = torch.isin(
            token_of, torch.tensor(tokens[:count], device=token_of.device)
        )
        faded = clean_pixels + kept[None, None].float() * (
            triggered_pixels - clean_pixels
        )
        reading = step_reading(
            1.0, faded, clean_psu, model, architecture, point, target, dataset, device
        )
        reading["trigger_tokens"] = count
        readings.append(reading)
    return readings


if __name__ == "__main__":
    main()
