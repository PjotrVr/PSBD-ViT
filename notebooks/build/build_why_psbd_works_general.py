"""Build notebooks/why-psbd-works-general.ipynb from the parts in why_psbd_works_general/.

The prose sits in why_psbd_works_general/prose.md, 1 section per `<!-- name -->`
marker. The code sits in why_psbd_works_general/cells.ipy, 1 cell per `# %% name`
marker, in IPython syntax because the style cell uses a magic. ORDER interleaves
them. The cells read the JSON under results/_experiments/why_psbd_works/ only, and
every sentence that carries a number is rendered by a code cell from those files.
"""

import os
import re

from nbbuild import code, md, write

PARTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "why_psbd_works_general"
)
ORDER = [
    "md:title",
    "code:setup",
    "code:imports",
    "md:map",
    "code:load",
    "code:helpers",
    "md:parameters",
    "code:parameters",
    "md:sanity",
    "code:sanity",
    "code:sanity_after",
    "md:step1",
    "code:step1",
    "md:step1_hist",
    "code:step1_hist",
    "code:step1_after",
    "md:step2",
    "code:step2",
    "code:step2_after",
    "md:step3",
    "code:step3",
    "code:step3_after",
    "md:step4",
    "code:step4",
    "code:step4_after",
    "md:wanet",
    "code:wanet",
    "md:step5",
    "code:step5",
    "code:step5_after",
    "md:step6",
    "code:step6",
    "code:step6_after",
    "md:refute_intro",
    "md:refute_confidence",
    "code:refute_confidence",
    "code:refute_confidence_after",
    "md:refute_uncertainty",
    "code:refute_uncertainty",
    "code:refute_uncertainty_after",
    "md:refute_ood",
    "code:refute_ood",
    "code:refute_ood_after",
    "md:refute_neurons",
    "code:refute_neurons",
    "code:refute_neurons_after",
    "md:refute_bias",
    "code:refute_bias",
    "code:refute_bias_after",
    "md:refute_memo",
    "code:refute_memo",
    "code:refute_memo_after",
    "md:fragility",
    "code:fragility",
    "code:fragility_after",
    "md:critical",
    "code:critical",
    "md:curves",
    "code:curves",
    "code:curves_after",
    "md:swin",
    "code:swin",
    "code:swin_after",
    "md:sufficiency",
    "code:sufficiency",
    "md:verdicts",
    "code:verdict_grid",
    "code:summary",
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
    write("why-psbd-works-general", cells)


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
