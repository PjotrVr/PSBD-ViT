"""Memo L26: is WaNet's trigger a coherence check that no single token can read?

WaNet trains with a noise mode (`cover_rate` 2 times the poison rate on our ViT
models): images warped by a random field keep their true label, so the network
must learn the exact warp and not "warped" as such. docs/why-psbd-works-theory.md
states the prediction: a linear probe that tells the exact warp from a random warp
sits near 0.5 accuracy on single tokens of the residual stream in blocks 4 to 8,
while a probe on the class token reaches 0.9 or more. If that holds, the trigger
is legible only from a wide area, which is the reason random token masking does
not leave WaNet's evidence behind the way it leaves Blend's.

For each WaNet model, 256 clean analysis images, their exact-warp copies (the
trigger) and their noise-warp copies (`attacks.wanet`'s own `apply_cover`). The
stream entering blocks 4 to 8 is captured per token. A logistic probe per block
is fitted on 128 images and scored on the other 128, once on single patch tokens
pooled over positions and once on the class token.

    source .venv/bin/activate
    PYTHONPATH=. python experiments/why_psbd_works/wanet_probe.py \
        --folders vit_tiny_wanet_0_1
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms.v2 as transforms_v2  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from attacks import apply_config_overrides, build_attack, default_config  # noqa: E402
from data.registry import DATASET_REGISTRY  # noqa: E402
from data.splits import (  # noqa: E402
    PSBD_SPLIT_SEED,
    load_clean_test_base,
    psbd_split_permutation,
    read_checkpoint_metadata,
)
from data.splits import PSBD_HELDOUT_SIZE  # noqa: E402
from experiments._paths import experiment_result_path  # noqa: E402
from experiments.why_psbd_works.measure import (  # noqa: E402
    CPU_THREADS,
    SLUG,
    captured_stream,
    limit_gpu_memory,
    predict_logits,
)
from analysis.features import transformer_blocks  # noqa: E402
from models.backbones import load_checkpoint  # noqa: E402

WANET_FOLDERS = (
    "vit_tiny_wanet_0_1",
    "vit_tiny_wanet_0_05",
    "vit_gtsrb_wanet_0_1",
    "vit_cifar10_wanet_0_1",
    "vit_cifar10_wanet_0_05",
)
IMAGE_COUNT = 256
PROBE_BLOCKS = (4, 5, 6, 7, 8, 12)
# Patch tokens pooled over 128 training images make 25088 samples per block. 40
# positions per image, drawn once, keep the fit to seconds with the same answer.
TOKENS_PER_IMAGE = 40
PROBE_SEED = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--folders", nargs="+", default=list(WANET_FOLDERS))
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--gpu-memory-gb", type=float, default=14.0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(CPU_THREADS)
    device = torch.device("cuda", torch.cuda.current_device())
    limit_gpu_memory(args.gpu_memory_gb, device)
    for folder in args.folders:
        out_path = experiment_result_path(
            SLUG, f"wanet_probe__{folder}.json", args.output_root
        )
        if os.path.exists(out_path):
            print(f"[skip] {folder}", flush=True)
            continue
        record = probe_model(folder, args, device)
        with open(out_path, "w") as handle:
            json.dump(record, handle, indent=2)
        print(f"[ok] {folder}", flush=True)


def probe_model(folder, args, device):
    checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)
    model = load_checkpoint(metadata["architecture"], checkpoint_path, device).eval()
    images = warped_sets(metadata, args.raw_data_dir, device)

    target = metadata["target_label"]
    predictions = {
        name: torch.cat(
            [
                predict_logits(model, batch[start : start + 128], device).argmax(dim=1)
                for start in range(0, len(batch), 128)
            ]
        )
        for name, batch in images.items()
    }
    record = {
        "folder": folder,
        "cover_rate": metadata.get("cover_rate"),
        "images": IMAGE_COUNT,
        "target_share": {
            name: float((p == target).float().mean()) for name, p in predictions.items()
        },
        "blocks": {},
    }

    blocks = transformer_blocks(model, metadata["architecture"])
    generator = np.random.default_rng(PROBE_SEED)
    train_images = generator.permutation(IMAGE_COUNT)[: IMAGE_COUNT // 2]
    is_train = np.zeros(IMAGE_COUNT, dtype=bool)
    is_train[train_images] = True
    token_draw = generator.integers(1, 197, size=(IMAGE_COUNT, TOKENS_PER_IMAGE))
    for number in PROBE_BLOCKS:
        streams = {
            name: captured_stream(
                model, blocks[number - 1], images[name], device
            ).numpy()
            for name in ("triggered", "noise")
        }  # each (n, 197, dim)
        record["blocks"][str(number)] = {
            "cls_probe_accuracy": probe_accuracy(
                streams["triggered"][:, 0], streams["noise"][:, 0], is_train
            ),
            "token_probe_accuracy": token_probe_accuracy(streams, token_draw, is_train),
        }
    return record


# Exact warp and noise warp of the same clean analysis images, the first 256 of
# the analysis split in the PSBD permutation, normalized as the model's inputs.
def warped_sets(metadata, raw_data_dir, device):
    spec = DATASET_REGISTRY[metadata["dataset"]]
    base = load_clean_test_base(metadata["dataset"], raw_data_dir)
    permutation = psbd_split_permutation(len(base), PSBD_SPLIT_SEED)
    rows = permutation[PSBD_HELDOUT_SIZE : PSBD_HELDOUT_SIZE + IMAGE_COUNT].tolist()
    config = apply_config_overrides(
        default_config("wanet"), metadata.get("attack_config_overrides")
    )
    attack = build_attack("wanet", config, spec.image_size, metadata["target_label"])
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)

    clean, triggered, noise = [], [], []
    for row in rows:
        image = base[row][0]  # (C, H, W), 0 to 1
        clean.append(normalize(image))
        triggered.append(normalize(attack.apply_trigger(image.clone(), row)))
        noise.append(normalize(attack.apply_cover(image.clone(), row)))
    images = {
        "clean": torch.stack(clean).to(device),
        "triggered": torch.stack(triggered).to(device),
        "noise": torch.stack(noise).to(device),
    }
    return images


def probe_accuracy(positive, negative, is_train):
    features = np.concatenate([positive, negative])  # (2n, dim)
    labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    train = np.r_[is_train, is_train]
    scaler = StandardScaler().fit(features[train])
    probe = LogisticRegression(max_iter=2000, C=1.0)
    probe.fit(scaler.transform(features[train]), labels[train])
    accuracy = float(probe.score(scaler.transform(features[~train]), labels[~train]))
    return accuracy


# Single patch tokens, pooled over positions: each image contributes the same 40
# random positions of its triggered and of its noise-warped stream, and the probe
# never sees a test image's tokens during the fit.
def token_probe_accuracy(streams, token_draw, is_train):
    rows = np.arange(len(token_draw))[:, None]  # (n, 1)
    triggered = streams["triggered"][rows, token_draw]  # (n, 40, dim)
    noise = streams["noise"][rows, token_draw]  # (n, 40, dim)
    per_token_train = np.repeat(is_train, token_draw.shape[1])  # (n * 40,)
    accuracy = probe_accuracy(
        triggered.reshape(-1, triggered.shape[-1]),
        noise.reshape(-1, noise.shape[-1]),
        per_token_train,
    )
    return accuracy


if __name__ == "__main__":
    main()
