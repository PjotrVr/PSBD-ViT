"""Build notebooks/backdoor-manifestation.ipynb from the parts in backdoor_manifestation/.

The prose sits in backdoor_manifestation/prose.md, 1 section per `<!-- name -->` marker.
The code sits in backdoor_manifestation/cells.ipy, 1 cell per `# %% name` marker, in IPython
syntax because the style cell uses a magic. ORDER interleaves them. The cells read the
JSON of experiments/backdoor_manifestation/ only.
"""

import os
import re

from nbbuild import code, md, write

PARTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "backdoor_manifestation"
)
ORDER = [
    "md:title",
    "md:map",
    "md:terms",
    "code:setup",
    "code:imports",
    "code:loaders",
    "md:panel",
    "code:panel_table",
    "md:panel_after",
    "code:control_table",
    "code:said_panel",
    "md:step1",
    "code:triggers",
    "md:step1_after",
    "md:step2",
    "code:token_maps_vit",
    "md:step2_vit_after",
    "code:token_maps_swin",
    "md:step2_cls",
    "code:token_maps_cls",
    "code:said_class_token",
    "md:step2_dims",
    "code:top_dimension_maps",
    "md:step2_dims_after",
    "md:step2_attention",
    "code:attention_lines",
    "md:step2_attention_heads",
    "code:attention_heads",
    "md:step2_attention_table",
    "code:attention_table",
    "code:said_attention",
    "md:step3",
    "code:depth_curves",
    "md:step3_after",
    "code:depth_table",
    "code:said_depth",
    "md:step4",
    "code:tac_heatmap",
    "md:step4_heatmap_after",
    "code:overlap",
    "code:said_overlap",
    "md:step4_histograms",
    "code:histograms",
    "md:step4_scatter",
    "code:auroc_scatter",
    "code:said_separability",
    "md:step4_ablation",
    "code:ablation",
    "md:step4_neurons",
    "code:ablation_neurons",
    "code:said_ablation",
    "md:step4_overlap",
    "code:magnitude_overlap",
    "md:step4_per_class",
    "code:per_class",
    "code:said_per_class",
    "md:step4_reading",
    "md:step4_middle",
    "code:ablation_middle",
    "code:said_middle",
    "md:step4_steering",
    "code:steering",
    "code:said_steering",
    "md:step5",
    "code:embedding",
    "md:step5_readout",
    "code:readout",
    "code:said_readout",
    "md:step6",
    "code:category_table",
    "code:said_category",
    "md:limits",
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
    write("backdoor-manifestation", cells)


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
