"""C-fp, GPU half: independent surplus readings on the clean held-out images.

Predictions are in PREDICTIONS.md, written before this ran. For each of the 2000
clean held-out images of a model (the split PSBD's threshold is read from), 2
readings that do not come from PSBD's probe:

    sufficient share  the smallest visible share of tokens, removed in a nested
                      order at the attention input of every block, deterministic
                      and with no PSBD rate, that keeps the image's answer. 2
                      orders, ranked by attribution and random
                      (experiments.why_psbd_works.measure.smallest_sufficient_fraction)
    class component   the projection of the final feature, minus the mean final
                      feature of the held-out images, on the head's direction for
                      the image's own class (why_psbd_works feature_geometry's
                      class direction)

The flag, the other operator's statistic and the margin are read from the sweep's
caches by analyze.py on the CPU.

    PYTHONPATH=. python experiments/evidence_surplus/false_positives/measure.py
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

from data.splits import (  # noqa: E402
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from experiments._paths import experiment_results_dir  # noqa: E402
from experiments.evidence_surplus.common import (  # noqa: E402
    BACKDOORED,
    BENIGN,
    SLUG,
    limit_gpu_memory,
)
from experiments.why_psbd_works.measure import (  # noqa: E402
    PATCH_GRID,
    classifier_head,
    model_seed,
    run_passes,
    smallest_sufficient_fraction,
    token_attribution,
)
from analysis.features import transformer_blocks  # noqa: E402
from models.backbones import load_checkpoint  # noqa: E402

BATCH = 128


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", default=list(BACKDOORED + BENIGN))
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--gpu-memory-gb", type=float, default=14.0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda", torch.cuda.current_device())
    limit_gpu_memory(args.gpu_memory_gb, device)
    out_dir = os.path.join(
        experiment_results_dir(SLUG, args.results_dir), "false_positives"
    )
    os.makedirs(out_dir, exist_ok=True)
    for folder in args.models:
        path = os.path.join(out_dir, f"{folder}.json")
        if os.path.exists(path):
            continue
        started = time.time()
        record = measure(folder, args, device)
        record["seconds"] = round(time.time() - started, 1)
        with open(path, "w") as handle:
            json.dump(record, handle)
        print(f"[ok] {folder} {record['seconds']}s", flush=True)


def measure(folder, args, device):
    path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(path)
    architecture = metadata["architecture"]
    model = load_checkpoint(architecture, path, device).eval()
    probe = None if metadata["attack"] != "benign" else "badnet_a2o"
    loaders, _ = build_psbd_loaders_from_checkpoint(
        path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=BATCH,
        num_workers=0,
        probe_attack=probe,
        probe_target_label=0 if probe else None,
    )
    dataset = loaders["validation"].dataset
    images = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(
        device
    )  # (n, C, H, W)

    base = run_passes(model, images, BATCH, device, passes=None)
    labels = base["logits"].argmax(dim=1)  # (n,)
    features = base["features"]  # (n, dim)
    weight = classifier_head(model).weight.detach().float().cpu()  # (classes, dim)
    centered = weight - weight.mean(dim=0, keepdim=True)
    directions = centered / centered.norm(dim=1, keepdim=True)  # (classes, dim)
    component = ((features - features.mean(dim=0)) * directions[labels]).sum(
        dim=1
    )  # (n,)

    first_block = transformer_blocks(model, architecture)[0]
    generator = torch.Generator().manual_seed(model_seed(folder))
    orders = {
        "random": torch.rand(len(images), PATCH_GRID * PATCH_GRID, generator=generator),
        "ranked": token_attribution(model, first_block, images, device),
    }
    sufficient = {
        name: smallest_sufficient_fraction(
            model,
            architecture,
            images,
            labels,
            scores,
            types.SimpleNamespace(batch_size=BATCH),
            device,
        )["smallest_fraction"]
        for name, scores in orders.items()
    }
    record = {
        "folder": folder,
        "architecture": architecture,
        "dataset": metadata["dataset"],
        "images": len(images),
        "labels": labels.tolist(),
        "class_component": [round(float(v), 4) for v in component],
        "sufficient_random": sufficient["random"],
        "sufficient_ranked": sufficient["ranked"],
    }
    return record


if __name__ == "__main__":
    main()
