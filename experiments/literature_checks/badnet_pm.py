import sys
import collections

sys.path.insert(0, ".")
from scripts.paper._common import load_coverage, clearing_cells, load_psbd_metrics
from defenses.decision import select_rate_adaptively
from data.splits import read_checkpoint_metadata
from experiments.residual_stream_mechanism.cls_routing import trigger_tokens

cells = clearing_cells(load_coverage("results"))
print("folder m rate_emb shift_bd_emb p^m | rate_tm shift_bd_tm p^(4m)")
for cell in cells:
    if cell["attack"] != "badnet_a2o":
        continue
    folder = cell["folder_name"]
    metrics = load_psbd_metrics("results", folder)
    metadata = read_checkpoint_metadata(f"checkpoints/{folder}/attack_result.pt")
    m = len(trigger_tokens(metadata))
    out = [folder, str(m)]
    for placement, exponent in (
        ("after_embedding_token_mask", m),
        ("before_attention_norm_token_mask", 4 * m),
    ):
        block = metrics["placements"].get(placement)
        shift = {r["rate"]: r["shift_ratio"]["validation"] for r in block["rates"]}
        rate = select_rate_adaptively(shift)
        row = next(r for r in block["rates"] if r["rate"] == rate)
        out += [
            f"{rate:.2f}",
            f"{row['shift_ratio']['backdoor']:.3f}",
            f"{rate**exponent:.3f}",
        ]
    print(" ".join(out))

print("gain_scale target share at adaptive rate")
share = collections.defaultdict(list)
for cell in cells:
    metrics = load_psbd_metrics("results", cell["folder_name"])
    if metrics is None:
        continue
    for placement in (
        "mlp_norm_out_gain_scale",
        "attention_norm_out_gain_scale",
        "final_norm_out_gain_scale",
    ):
        block = metrics["placements"].get(placement)
        if not block:
            continue
        shift = {r["rate"]: r["shift_ratio"]["validation"] for r in block["rates"]}
        rate = select_rate_adaptively(shift)
        if rate is None:
            continue
        row = next(r for r in block["rates"] if r["rate"] == rate)
        histogram = row.get("shift_target_histogram", {}).get("clean")
        if histogram and sum(histogram):
            share[(placement, metrics["attack"])].append(
                (
                    histogram[metrics["target_label"]] / sum(histogram),
                    row["detection_psu_ratio"]["q0.25"]["auroc"],
                )
            )
for key, rows in sorted(share.items()):
    print(
        key,
        len(rows),
        f"target_share={sum(r[0] for r in rows) / len(rows):.3f}",
        f"auroc={sum(r[1] for r in rows) / len(rows):.3f}",
    )
