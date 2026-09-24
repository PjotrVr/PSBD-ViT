"""The 1 matplotlib style every paper figure uses, imported before pyplot draws.

A light grid with transparency so values can be read off a line chart, distinct
colorblind safe hues in a fixed order (Okabe and Ito), and sizes that stay
legible at a 2-column width.

The 2 column widths are measured from the class options this paper sets, not
guessed. main.tex asks for 10pt twocolumn and preamble.tex for margin=0.8in with
columnsep=0.28in on US letter, which leaves a 6.9in text width and a 3.31in
column. A figure authored wider than the width it is drawn into is scaled down by
LaTeX, and the scaling takes the fonts with it, so an 8pt label in an 11in figure
lands at 5pt on the page. Author at the target width instead.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#56B4E9",
    "#F0E442",
    "#000000",
]

plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        # Type 3 is matplotlib's default and the IEEE and ACM camera-ready checkers
        # reject it. 42 embeds TrueType, which also leaves the text searchable and
        # selectable in the compiled PDF.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# The width a figure may occupy, in inches, for this paper's geometry.
COLUMN_WIDTH = 3.31
TEXT_WIDTH = 6.9

# 1 color per attack across every figure, so a reader who learns that BadNets is
# orange in 1 figure reads it as orange in the next. The benign control is gray
# and dashed wherever it appears.
ATTACK_COLORS = {
    "badnet": "#E69F00",
    "blend": "#56B4E9",
    "wanet": "#D55E00",
    "bpp": "#0072B2",
    "lf": "#CC79A7",
    "tact": "#009E73",
    "sig": "#000000",
    "lc": "#999933",
    "adaptive_blend": "#882255",
}
BENIGN_COLOR = "0.5"


def attack_color(attack: str) -> str:
    """The fixed color of an attack token, gray for the benign control."""
    color = ATTACK_COLORS.get(attack, BENIGN_COLOR)
    return color


def unique_handles(axes) -> tuple[list, list[str]]:
    """Every labeled artist across the axes, first occurrence of each label kept."""
    seen: dict[str, object] = {}
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            seen.setdefault(label, handle)
    return list(seen.values()), list(seen.keys())


def legend_above(figure, axes, columns: int | None = None) -> None:
    """1 legend for the whole figure, placed above the panels so it covers no data.

    Anchored at the figure's top edge from below, so tight_layout keeps the axes
    where they are and the tight bounding box at save time takes the legend in.
    """
    handles, labels = unique_handles(axes)
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=columns or min(len(labels), 5),
        frameon=False,
    )


def legend_in_spare_axis(spare, axes) -> None:
    """The figure's legend drawn inside an unused panel of the grid."""
    handles, labels = unique_handles(axes)
    spare.axis("off")
    spare.legend(handles, labels, loc="center", frameon=False)


def legend_below(axis, columns: int = 3) -> None:
    """A panel's own legend placed under its x label, for panels whose labels differ."""
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=columns,
        frameon=False,
    )
