"""The ordered model queue, printed 1 folder per line for run_queue.sh.

Only attack families the ViT panel has never met, at 10% and for TrojanNN also
5%, on a few datasets, in the order the user set on 2026-09-30: every 10% model
first (TrojanNN, SSBA, Input-Aware, LIRA, Blind) and TrojanNN at 5% last. The 4 LIRA folders
hold no attack_result.pt and are listed here only so the gap stays visible, so a
folder without a checkpoint is dropped with a note on stderr.

    .venv/bin/python -m experiments.backdoorbench_attacks.queue
"""

import os
import sys

from data.backdoorbench import BACKDOORBENCH_WEIGHTS_DIR

MODELS = (
    "cifar10_trojannn_0_1",
    "gtsrb_trojannn_0_1",
    "tiny_trojannn_0_1",
    "cifar10_ssba_0_1",
    "gtsrb_ssba_0_1",
    "tiny_ssba_0_1",
    "cifar10_inputaware_0_1",
    "gtsrb_inputaware_0_1",
    "cifar10_lira_0_1",
    "gtsrb_lira_0_1",
    "cifar10_blind_0_1",
    "cifar10_trojannn_0_05",
    "gtsrb_trojannn_0_05",
    "tiny_trojannn_0_05",
)


def main():
    for folder in readable(MODELS):
        print(folder)


def readable(folders):
    kept = []
    for folder in folders:
        if os.path.exists(
            os.path.join(BACKDOORBENCH_WEIGHTS_DIR, folder, "attack_result.pt")
        ):
            kept.append(folder)
        else:
            print(f"{folder}: no attack_result.pt, skipped", file=sys.stderr)
    return kept


if __name__ == "__main__":
    main()
