"""Trigger size series: PSBD-TM and PSBD-RD against the number of trigger tokens.

Reads the 12 BadNets 5% checkpoints with patch sizes 2 to 16 on CIFAR-100 and
GTSRB. Before any sweep exists it records each model's attack success, clean
accuracy and ledger verdicts from `args.json`. Once `cli.sweep` and `cli.analyze`
have run it adds, per placement, the mean triggered p* on the standard ladder,
TPR at 1%, 5% and 10% FPR with the realized FPR and AUROC at the adaptive rate,
and judges PREDICTIONS.md. The results table between the markers of README.md is
rewritten from the record on every run.

    .venv/bin/python experiments/evidence_surplus/trigger_size/measure.py
"""

import hashlib
import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from defenses.cache import (  # noqa: E402
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
)
from defenses.decision import (  # noqa: E402
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
    complete_rates,
)
from defenses.scores import critical_rate  # noqa: E402
from scripts.coverage_ledger import (  # noqa: E402
    benign_reference_accuracy,
    classify_cell,
    clean_accuracy_of,
    load_declaration,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = "results"
CHECKPOINTS_DIR = "checkpoints"
OUT_DIR = os.path.join(RESULTS_DIR, "_experiments", "evidence_surplus", "trigger_size")
PREDICTIONS = os.path.join(HERE, "PREDICTIONS.md")
README = os.path.join(HERE, "README.md")
DATASETS = ("cifar100", "gtsrb")
PATCH_SIZES = (2, 3, 5, 8, 12, 16)
PLACEMENTS = {"psbd_tm": RECOMMENDED_PLACEMENT, "psbd_rd": PUBLISHED_PLACEMENT}
# The ladders of the panel caches, so the series reads like every panel model.
LADDERS = {
    RECOMMENDED_PLACEMENT: (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    PUBLISHED_PLACEMENT: (
        0.005,
        0.01,
        0.02,
        0.03,
        0.05,
        0.07,
        0.09,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ),
}
NATIVE_SIZE, INPUT_SIZE, TOKEN_SIZE = 32, 224, 16
NEVER_FLIPPED = 1.0
QUANTILES = ("q0.01", "q0.05", "q0.10")
RHO_BAR = 0.5
MIN_MODELS = 4


def main():
    declaration = load_declaration("configs/psbd_basis.json")
    references = benign_reference_accuracy(
        CHECKPOINTS_DIR, RESULTS_DIR, declaration["benign_reference"]
    )
    rows = [
        measure(dataset, patch, declaration, references)
        for dataset in DATASETS
        for patch in PATCH_SIZES
    ]
    with open(PREDICTIONS, "rb") as handle:
        prereg_hash = hashlib.sha256(handle.read()).hexdigest()
    swept = all(r["swept"] for r in rows)
    payload = {
        "experiment": "trigger size series",
        "predictions_sha256": prereg_hash,
        "complete": swept,
        "verdicts": judge(rows) if swept else None,
        "models": rows,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "trigger_size.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    render(payload)
    print(f"wrote {path}, sweeps complete: {swept}")


def token_count(patch):
    # A native pixel spans INPUT_SIZE / NATIVE_SIZE input pixels after the
    # resize and the patch sits flush in the corner, so it covers this many
    # tokens per side.
    side = math.ceil(patch * INPUT_SIZE / NATIVE_SIZE / TOKEN_SIZE)
    count = side * side
    return count


def measure(dataset, patch, declaration, references):
    folder = f"vit_{dataset}_badnet_a2o_0_05_trig_p{patch}"
    with open(os.path.join(CHECKPOINTS_DIR, folder, "args.json")) as handle:
        metadata = json.load(handle)
    cell = {
        "folder_name": folder,
        "dataset": dataset,
        "attack": metadata["attack"],
        "asr": metadata.get("asr"),
        "clean_accuracy": clean_accuracy_of(CHECKPOINTS_DIR, RESULTS_DIR, folder),
    }
    classify_cell(
        cell, references.get(dataset), declaration, CHECKPOINTS_DIR, RESULTS_DIR
    )
    psbd_dir = os.path.join(RESULTS_DIR, folder, "psbd")
    swept = all(
        set(LADDERS[p]) <= set(complete_rates(psbd_dir, p)) for p in PLACEMENTS.values()
    ) and os.path.exists(os.path.join(RESULTS_DIR, folder, "psbd_metrics.json"))
    row = {
        "folder": folder,
        "dataset": dataset,
        "patch_size": patch,
        "trigger_tokens": token_count(patch),
        "patch_size_override": metadata.get("attack_config_overrides", {}).get(
            "patch_size"
        ),
        "asr": cell["asr"],
        "clean_accuracy": cell["clean_accuracy"],
        "clean_accuracy_benign": cell["clean_accuracy_benign"],
        "asr_class": cell["asr_class"],
        "successful_2pt": cell["successful_2pt"],
        "successful_5pt": cell["successful_5pt"],
        "swept": swept,
        "placements": {},
    }
    if swept:
        with open(os.path.join(RESULTS_DIR, folder, "psbd_metrics.json")) as handle:
            metrics = json.load(handle)
        for name, placement in PLACEMENTS.items():
            row["placements"][name] = readout(
                psbd_dir, placement, metrics["placements"][placement]
            )
    return row


def readout(psbd_dir, placement, block):
    _, labels, _ = load_baseline(baseline_path(psbd_dir, "backdoor"))
    rates = sorted(LADDERS[placement])
    argmax = {
        rate: load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, "backdoor")
        )[1]
        for rate in rates
    }
    p_star = critical_rate(labels, rates, argmax)  # (n_backdoor,)
    never = p_star > rates[-1]  # (n_backdoor,) bool
    p_star = torch.where(never, torch.full_like(p_star, NEVER_FLIPPED), p_star)

    result = {
        "mean_triggered_p_star": float(p_star.mean()),
        "never_flipped_share": float(never.float().mean()),
        "adaptive_rate": block.get("adaptive_rate"),
    }
    if block.get("adaptive_rate") is not None:
        row = next(r for r in block["rates"] if r["rate"] == block["adaptive_rate"])
        detection = row["detection_psu_ratio"]
        result["auroc"] = detection["q0.01"]["auroc"]
        for q in QUANTILES:
            result[f"{q}:tpr"] = detection[q]["tpr"]
            result[f"{q}:fpr"] = detection[q]["fpr"]
    return result


def rho(rows, placement, field):
    xs = [r["trigger_tokens"] for r in rows]
    ys = [r["placements"][placement].get(field) for r in rows]
    if len(rows) < MIN_MODELS or None in ys or len(set(ys)) < 2:
        return None
    value = float(spearmanr(xs, ys).statistic)
    return value


def judge(rows):
    fields = {"TM-p": "mean_triggered_p_star", "TM-tpr": "q0.01:tpr"}
    correlations = {}
    for dataset in DATASETS:
        members = [r for r in rows if r["dataset"] == dataset and r["successful_2pt"]]
        correlations[dataset] = {
            "n": len(members),
            **{
                f"{name}/{field}": rho(members, name, field)
                for name in PLACEMENTS
                for field in fields.values()
            },
            "auroc_psbd_tm": rho(members, "psbd_tm", "auroc"),
        }

    def verdict(checks):
        if any(c is None for c in checks):
            return "inconclusive"
        result = "held" if all(checks) else "failed"
        return result

    results = {}
    for name, field in fields.items():
        values = [correlations[d][f"psbd_tm/{field}"] for d in DATASETS]
        results[name] = verdict([None if v is None else v >= RHO_BAR for v in values])
    differs = []
    for dataset in DATASETS:
        for field in fields.values():
            tm = correlations[dataset][f"psbd_tm/{field}"]
            rd = correlations[dataset][f"psbd_rd/{field}"]
            differs.append(None if tm is None or rd is None else rd < tm)
    results["RD-differs"] = verdict(differs)
    judged = {"correlations": correlations, "verdicts": results}
    return judged


def f3(value):
    text = "--" if value is None else f"{value:.3f}"
    return text


def render(payload):
    lines = [
        "| model | trigger tokens | ASR | clean accuracy | benign | successful 2-point | swept |"
        " TM mean p* | TM TPR 1 / 5 / 10% (FPR 1%) | TM AUROC | RD mean p* |"
        " RD TPR 1 / 5 / 10% (FPR 1%) | RD AUROC |",
        "|" + "---|" * 13,
    ]
    for r in payload["models"]:
        cells = [
            f"`{r['folder']}`",
            r["trigger_tokens"],
            f3(r["asr"]),
            f3(r["clean_accuracy"]),
            f3(r["clean_accuracy_benign"]),
            "yes" if r["successful_2pt"] else "no",
            "yes" if r["swept"] else "pending",
        ]
        for name in PLACEMENTS:
            p = r["placements"].get(name)
            if not p:
                cells += ["--", "--", "--"]
                continue
            cells += [
                f3(p["mean_triggered_p_star"]),
                " / ".join(f3(p.get(f"{q}:tpr")) for q in QUANTILES)
                + f" ({f3(p.get('q0.01:fpr'))})",
                f3(p.get("auroc")),
            ]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    lines.append("")
    lines.append(f"PREDICTIONS.md SHA-256 `{payload['predictions_sha256']}`.")
    if payload["verdicts"]:
        lines.append("")
        lines.append("| prediction | verdict |")
        lines.append("|---|---|")
        for name, value in payload["verdicts"]["verdicts"].items():
            lines.append(f"| {name} | {value} |")
        lines.append("")
        lines.append(
            "| dataset | successful models | ρ TM p* | ρ TM TPR 1% | ρ RD p* | ρ RD TPR 1% | ρ TM AUROC |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for dataset, c in payload["verdicts"]["correlations"].items():
            lines.append(
                f"| {dataset} | {c['n']} | {f3(c['psbd_tm/mean_triggered_p_star'])} | "
                f"{f3(c['psbd_tm/q0.01:tpr'])} | {f3(c['psbd_rd/mean_triggered_p_star'])} | "
                f"{f3(c['psbd_rd/q0.01:tpr'])} | {f3(c['auroc_psbd_tm'])} |"
            )
    else:
        lines.append("")
        lines.append("The sweeps have not run yet, so no verdict exists.")
    block = "\n".join(lines)

    with open(README) as handle:
        text = handle.read()
    begin, end = "<!-- results:begin -->", "<!-- results:end -->"
    head, rest = text.split(begin)
    _, tail = rest.split(end)
    text = f"{head}{begin}\n{block}\n{end}{tail}"
    with open(README, "w") as handle:
        handle.write(text)


if __name__ == "__main__":
    main()
