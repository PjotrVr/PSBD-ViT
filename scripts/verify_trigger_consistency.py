"""Does each cell's cached backdoor baseline still see the trigger the loader builds today?

`scripts/verify_splits.py` compares row counts, which catches a change in WHICH rows
are eligible and misses a change in WHAT the trigger looks like. SIG is the case that
made this script necessary: commit 0150611 raised `SigConfig.amplitude` from 0.1 to
0.157, every checkpoint trained before it learned the 0.1 sinusoid, and a loader
rebuilding such a checkpoint's eval set from its defaults stamped 0.157. The row count
is unchanged, so the split check passed, while every fresh forward pass scored a
trigger the model never saw and the cached baseline, built earlier, still held the
old one.

For each cell this rebuilds the backdoor split through
`data.splits.build_psbd_loaders_from_checkpoint`, the path `cli.sweep` and
`cli.baselines` both use, predicts the first rows on CPU in float32 and compares the
argmax with the labels in `results/<folder>/psbd/baseline_backdoor.pt`. The cache was
written in bfloat16 on a GPU, so a near-tie can flip 1 row on an honest cell, which
is why the bar is 0.98 and not 1.

    PYTHONPATH=. python scripts/verify_trigger_consistency.py --panel both
    PYTHONPATH=. python scripts/verify_trigger_consistency.py \\
        --checkpoint-folder vit_cifar10_sig_0_1 --rows 300
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.getcwd())

from data.splits import PSBD_SPLIT_SEED, build_psbd_loaders_from_checkpoint  # noqa: E402
from defenses.cache import baseline_path, load_baseline  # noqa: E402
from defenses.inference import forward_probs  # noqa: E402
from models.backbones import load_checkpoint  # noqa: E402

DEFAULT_ROWS = 64
AGREEMENT_BAR = 0.98
PANELS = ("vit", "swin", "both")


def cached_backdoor_labels(psbd_dir: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(labels, loader_labels) of the cached backdoor baseline, each (n,), or None."""
    path = baseline_path(psbd_dir, "backdoor")
    if not os.path.exists(path):
        return None

    _, labels, loader_labels = load_baseline(path)
    return labels, loader_labels


def rebuilt_backdoor_predictions(
    checkpoint_path: str, raw_data_dir: str, rows: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """(predictions, loader_labels) for the first rows of the rebuilt backdoor split.

    The full split is built and then sliced, rather than built with max_samples,
    because max_samples truncates the analysis pool before eligibility and would
    leave a source-specific attack (TaCT) with almost no backdoor rows.
    """
    loaders, _ = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=raw_data_dir,
        batch_size=rows,
        num_workers=0,
    )
    backdoor_set = loaders["backdoor"].dataset
    first_rows = Subset(backdoor_set, range(min(rows, len(backdoor_set))))
    loader = DataLoader(first_rows, batch_size=32, shuffle=False, num_workers=4)

    metadata_path = os.path.join(os.path.dirname(checkpoint_path), "args.json")
    with open(metadata_path) as handle:
        architecture = json.load(handle)["architecture"]
    model = load_checkpoint(architecture, checkpoint_path, device)

    batch_predictions = []
    batch_labels = []
    with torch.inference_mode():
        for images, labels in loader:
            probs = forward_probs(
                model, images, device, use_bfloat16=False
            )  # (batch, num_classes)
            batch_predictions.append(probs.argmax(dim=1).cpu())  # (batch,)
            batch_labels.append(labels)  # (batch,)

    predictions = torch.cat(batch_predictions)  # (rows,)
    loader_labels = torch.cat(batch_labels)  # (rows,)
    return predictions, loader_labels


def consistency_report(
    fresh_predictions: torch.Tensor,
    fresh_loader_labels: torch.Tensor,
    cached_labels: torch.Tensor,
    cached_loader_labels: torch.Tensor,
) -> dict:
    """Agreement of fresh and cached predictions over the rows both hold.

    A loader label mismatch means the rows are not the same images in the same
    order, which is a split drift rather than a trigger drift, so it is reported
    under its own status.
    """
    rows = len(fresh_predictions)
    assert cached_labels.shape[0] >= rows, "the cache holds fewer rows than requested"

    cached_rows = cached_labels[:rows]  # (rows,)
    cached_targets = cached_loader_labels[:rows]  # (rows,)
    same_targets = bool(torch.equal(cached_targets, fresh_loader_labels))

    agreement = float((fresh_predictions == cached_rows).float().mean())
    fresh_hit_rate = float((fresh_predictions == fresh_loader_labels).float().mean())
    cached_hit_rate = float((cached_rows == cached_targets).float().mean())

    if not same_targets:
        status = "SPLIT_DRIFT"
    elif agreement < AGREEMENT_BAR:
        status = "TRIGGER_DRIFT"
    else:
        status = "ok"

    report = {
        "status": status,
        "rows": rows,
        "agreement": agreement,
        "fresh_hit_rate": fresh_hit_rate,
        "cached_hit_rate": cached_hit_rate,
    }
    return report


def verify_one(folder: str, args: argparse.Namespace, device: torch.device) -> dict:
    """1 cell's report: ok, TRIGGER_DRIFT, SPLIT_DRIFT or no_baseline."""
    cached = cached_backdoor_labels(os.path.join(args.results_dir, folder, "psbd"))
    if cached is None:
        return {"folder_name": folder, "status": "no_baseline"}
    cached_labels, cached_loader_labels = cached

    checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    fresh_predictions, fresh_loader_labels = rebuilt_backdoor_predictions(
        checkpoint_path, args.raw_data_dir, args.rows, device
    )

    report = consistency_report(
        fresh_predictions, fresh_loader_labels, cached_labels, cached_loader_labels
    )
    labelled_report = {"folder_name": folder, **report}
    return labelled_report


def vit_panel_folders(coverage_path: str) -> list[str]:
    """The ViT cells the coverage ledger marks as clearing the attack-success bar."""
    with open(coverage_path) as handle:
        cells = json.load(handle)["cells"]

    folders = [cell["folder_name"] for cell in cells if cell["asr_class"] == "clears"]
    return folders


def swin_panel_folders(results_dir: str, checkpoints_dir: str) -> list[str]:
    """The Swin cells the paper's Swin table reads, by that table's own rule."""
    from scripts.paper._common import DEFAULT_DECLARATION, load_declaration
    from scripts.paper.tab_swin import swin_cells

    asr_bar = load_declaration(DEFAULT_DECLARATION)["asr_bar"]
    folders = [
        cell["folder"] for cell in swin_cells(results_dir, checkpoints_dir, asr_bar)
    ]
    return folders


def folders_to_verify(args: argparse.Namespace) -> list[str]:
    """The folders named on the command line, or the panel the caller asked for."""
    if args.checkpoint_folder:
        return list(args.checkpoint_folder)

    folders = []
    if args.panel in ("vit", "both"):
        folders += vit_panel_folders(args.coverage)
    if args.panel in ("swin", "both"):
        folders += swin_panel_folders(args.results_dir, args.checkpoints_dir)
    return folders


def write_reports(path: str, reports: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        json.dump(reports, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=PANELS, default="both")
    parser.add_argument("--checkpoint-folder", nargs="*", default=None)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--coverage", default="results/coverage/coverage.json")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--out", default="results/coverage/trigger_consistency.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    folders = folders_to_verify(args)

    reports = {}
    for index, folder in enumerate(folders, start=1):
        try:
            report = verify_one(folder, args, device)
        except Exception as error:  # noqa: BLE001
            report = {
                "folder_name": folder,
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
        reports[folder] = report
        agreement = report.get("agreement")
        agreement_text = f"{agreement:.3f}" if agreement is not None else "-"
        print(
            f"[{index:3d}/{len(folders)}] {folder:42s} {report['status']:14s} "
            f"agreement {agreement_text}",
            flush=True,
        )
        write_reports(args.out, reports)

    failing = [name for name, row in reports.items() if row["status"] != "ok"]
    print(f"\nnot ok {len(failing)} of {len(reports)}")
    for name in failing:
        print(f"    {name}  {reports[name]['status']}")
    if failing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
