"""BackdoorBench's published ViT-B/16 numbers, read off its leaderboard pages.

The leaderboard (backdoorbench.github.io, 1 page per dataset, 1 tab per poison
rate) lists clean accuracy, ASR and robust accuracy per attack and model with no
defense applied. The ViT-B/16 rows are the reference the checkpoints under
backdoor_bench_checkpoints/ are held to before any PSBD number is read from them.
Written to leaderboard_vit_b_16.json in this directory, tracked, with the fetch
date, since the pages can change.

    https_proxy=http://10.150.1.1:3128 .venv/bin/python -m experiments.backdoorbench_attacks.fetch_leaderboard
"""

import datetime
import html
import json
import os
import re
import urllib.request

PAGES = {
    "cifar10": "https://backdoorbench.github.io/leaderboard-cifar10.html",
    "cifar100": "https://backdoorbench.github.io/leaderboard-cifar100.html",
    "gtsrb": "https://backdoorbench.github.io/leaderboard-gtsrb.html",
    "tiny": "https://backdoorbench.github.io/leaderboard-tinyimagenet.html",
}
OUTPUT = os.path.join(os.path.dirname(__file__), "leaderboard_vit_b_16.json")


def main():
    rows = {}
    for dataset, url in PAGES.items():
        page = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
        rows.update(vit_rows(dataset, page))

    payload = {
        "source": PAGES,
        "fetched": datetime.date.today().isoformat(),
        "columns": "no-defense clean accuracy, ASR and robust accuracy in percent",
        "rows": rows,
    }
    with open(OUTPUT, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print(f"wrote {len(rows)} rows to {OUTPUT}")


def vit_rows(dataset, page):
    # Each rate tab opens with <div id="pratio-0_1" ...> and each table row sits
    # on 1 line, model name first, attack second, then the no-defense triple.
    rows = {}
    rate_tag = None
    for line in page.splitlines():
        opened = re.search(r'<div id="pratio-([0-9_]+)"', line)
        if opened:
            rate_tag = opened.group(1)
        if 'td_attack">ViT' not in line:
            continue
        cells = [
            html.unescape(re.sub("<[^>]+>", "", cell)).strip()
            for cell in re.findall(r"<td[^>]*>(.*?)</td>", line)
        ]
        folder = f"{dataset}_{cells[1].lower()}_{rate_tag}"
        rows[folder] = {
            "clean_accuracy": percent(cells[2]),
            "asr": percent(cells[3]),
            "robust_accuracy": percent(cells[4]),
        }
    return rows


def percent(cell):
    value = None if cell == "N/A" else float(cell) / 100
    return value


if __name__ == "__main__":
    main()
