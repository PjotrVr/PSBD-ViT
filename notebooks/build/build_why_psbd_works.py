"""Build notebooks/why-psbd-works.ipynb from the parts in why_psbd_works/.

The prose sits in why_psbd_works/prose.md, 1 section per `<!-- name -->` marker.
The code sits in why_psbd_works/cells.ipy, 1 cell per `# %% name` marker, in IPython
syntax because the style cell uses a magic. ORDER interleaves them. The cells read
JSON records under results/ only, plus the stage-1 caches for the 1 reading that
wrote no record, so a rebuild picks up whatever the running experiments have added.
"""

import os
import re

from nbbuild import code, md, write

PARTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "why_psbd_works")
ORDER = [
    "md:intro",
    "md:map",
    "code:setup",
    "code:imports",
    "code:style",
    "code:helpers",
    "md:table_intro",
    "code:claims_placeholder",
    "md:panel",
    "code:panel",
    "md:s1",
    "code:phenomenon_load",
    "md:claim1",
    "code:claim1",
    "code:claim1_resnet",
    "md:claim1_limits",
    "md:claim2",
    "code:claim2",
    "md:claim2_limits",
    "md:claim3",
    "code:claim3",
    "md:claim3_limits",
    "md:claim4",
    "code:claim4",
    "md:claim4_limits",
    "md:s2",
    "code:curves_load",
    "code:curves_vit",
    "md:curves_swin",
    "code:curves_swin",
    "md:curves_wanet",
    "code:curves_wanet",
    "md:contradictions",
    "code:curves_contradictions",
    "md:breaking_point",
    "code:breaking_point_verdict",
    "md:s2_limits",
    "md:s3",
    "code:routing_load",
    "md:routing_read",
    "code:routing_figure",
    "code:routing_claims",
    "md:s3_limits",
    "md:s4",
    "code:fragility",
    "md:s4_limits",
    "md:s5",
    "md:confidence",
    "code:confidence",
    "md:ood",
    "code:ood",
    "md:layernorm",
    "code:layernorm",
    "md:ibd",
    "code:ibd_psc",
    "md:s5_limits",
    "md:s6",
    "code:neurons",
    "code:direction",
    "md:s6_limits",
    "md:s7",
    "code:wanet_x1",
    "md:probe",
    "code:wanet_probe",
    "md:x2",
    "code:wanet_x2",
    "md:s7_limits",
    "md:s8",
    "code:redundancy",
    "md:s8_limits",
    "md:s9",
    "code:manifestation",
    "md:s9_limits",
    "md:s10",
    "code:cache_readouts_load",
    "md:pass_stats",
    "code:pass_statistics",
    "md:depth",
    "code:depth_bands",
    "md:fusion",
    "code:fusion",
    "md:s10_limits",
    "md:s11",
    "code:general",
    "md:s11_limits",
    "md:table_end",
    "code:claims_table",
]


def main():
    code_parts = split_parts(read_part("cells.ipy"), r"^# %% (\w+)\n")
    prose_parts = split_parts(read_part("prose.md"), r"^<!-- (\w+) -->\n")
    used_code = {item.split(":")[1] for item in ORDER if item.startswith("code:")}
    used_prose = {item.split(":")[1] for item in ORDER if item.startswith("md:")}
    # A part written but left out of ORDER would silently vanish from the notebook.
    assert used_code == set(code_parts), set(code_parts) ^ used_code
    assert used_prose == set(prose_parts), set(prose_parts) ^ used_prose

    cells = []
    for item in ORDER:
        kind, name = item.split(":")
        cell = md(prose_parts[name]) if kind == "md" else code(code_parts[name])
        cells.append(cell)
    write("why-psbd-works", cells)


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
