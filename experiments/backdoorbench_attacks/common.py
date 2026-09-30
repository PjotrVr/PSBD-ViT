"""Paths, model tiers and small helpers the scripts of this experiment share."""

import json
import os

from experiments._paths import experiment_results_dir

SLUG = "backdoorbench_attacks"
RECORDS_DIR = experiment_results_dir(SLUG)
INVENTORY_PATH = os.path.join(RECORDS_DIR, "inventory.json")
EVALUATION_DIR = os.path.join(RECORDS_DIR, "evaluation")
MODEL_RECORDS_DIR = os.path.join(RECORDS_DIR, "models")
SUMMARY_PATH = os.path.join(RECORDS_DIR, "summary.json")

LEADERBOARD_PATH = os.path.join(os.path.dirname(__file__), "leaderboard_vit_b_16.json")
with open(LEADERBOARD_PATH) as handle:
    LEADERBOARD = json.load(handle)["rows"]


def write_json(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def read_json(path):
    with open(path) as handle:
        payload = json.load(handle)
    return payload
