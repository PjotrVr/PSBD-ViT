"""Build notebooks/prediction-shift-phenomenon.ipynb from the parts in prediction_shift/.

The prose sits in prediction_shift/prose.md, 1 section per `<!-- name -->` marker.
The code sits in prediction_shift/cells.ipy, 1 cell per `# %% name` marker, in IPython
syntax because the style cell uses a magic. ORDER interleaves them. The cells read the
JSON of experiments/prediction_shift_phenomenon/ only.
"""

import os
import re

from nbbuild import code, md, write

PARTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_shift")
ORDER = [
    "md:intro",
    "md:map",
    "code:setup",
    "code:imports",
    "code:style",
    "code:load",
    "md:panel",
    "code:panel",
    "md:sanity",
    "code:sanity",
    "md:claim1",
    "code:claim1_curves",
    "md:claim1_curves_read",
    "code:claim1_strip",
    "code:claim1_table",
    "code:claim1_by_attack",
    "md:claim1_verdict",
    "md:claim2",
    "code:claim2_histograms",
    "md:claim2_histograms_read",
    "code:claim2_rate",
    "md:claim2_rate_read",
    "code:claim2_strip",
    "code:claim2_table",
    "code:claim2_by_attack",
    "md:claim2_verdict",
    "md:claim3",
    "code:claim3_maps",
    "code:claim3_maps_transformers",
    "md:claim3_maps_read",
    "code:claim3_similarity",
    "code:claim3_table",
    "md:claim3_verdict",
    "md:claim4",
    "code:claim4_curves",
    "md:claim4_curves_read",
    "code:claim4_scatter",
    "code:claim4_table",
    "md:claim4_verdict",
    "md:best_residual",
    "code:best_residual_figure",
    "code:best_residual_table",
    "md:best_residual_verdict",
    "md:summary",
    "code:summary_table",
    "md:closing",
]


def main():
    code_parts = split_parts(read_part("cells.ipy"), r"^# %% (\w+)\n")
    prose_parts = split_parts(read_part("prose.md"), r"^<!-- (\w+) -->\n")
    used = {item.split(":")[1] for item in ORDER if item.startswith("code:")}
    # A cell written but left out of ORDER would silently vanish from the notebook.
    assert used == set(code_parts), set(code_parts) ^ used

    cells = []
    for item in ORDER:
        kind, name = item.split(":")
        cell = md(prose_parts[name]) if kind == "md" else code(code_parts[name])
        cells.append(cell)
    write("prediction-shift-phenomenon", cells)


def read_part(name):
    with open(os.path.join(PARTS, name)) as handle:
        text = handle.read()
    return text


def split_parts(text, marker):
    pieces = re.split(marker, text, flags=re.M)
    parts = {
        pieces[index]: pieces[index + 1].strip("\n")
        for index in range(1, len(pieces), 2)
    }
    return parts


if __name__ == "__main__":
    main()
