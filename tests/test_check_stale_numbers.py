"""The stale headline number check: what it flags, what it lets through and CLAUDE.md.

A superseded value only counts in a headline context, a dated or historical line
is a record, a file with a `Superseded on <date>` status line is exempt unless it
is a living document, and a line citing a macro whose current value is the number
quotes it correctly. The last test holds the real `.claude/CLAUDE.md` to the real
`paper/headline.tex`.
"""

import os

from scripts.check_stale_numbers import (
    CLAUDE_MD,
    HEADLINE_TEX,
    build_patterns,
    claude_mismatches,
    count_pattern,
    decimal_pattern,
    macros_from_sidecar,
    macros_from_tex,
    normalize_value,
    read_text,
    stale_hits,
    superseded_values,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURRENT = {
    "HeadlineAurocAdaptive": "0.953",
    "HeadlineGainAdaptiveAuroc": "+0.065",
    "GainsHighestRate": "-0.015",
    "PanelCellsCached": "57",
    "LadderAurocAtZeroEight": "0.951",
    "SwinRecommendedAurocAdaptive": "0.969",
}
HISTORY = {
    "HeadlineAurocAdaptive": {"0.935", "0.953"},
    "HeadlineGainAdaptiveAuroc": {"+0.103", "+0.065"},
    "PanelCellsCached": {"65", "57"},
    "SwinRecommendedAurocAdaptive": {"0.951", "0.969"},
}


def patterns() -> list:
    superseded = superseded_values(HISTORY, CURRENT)
    built = build_patterns(superseded)
    return built


def hits(lines: list[str], path: str = "docs/some-note.md") -> list[str]:
    found = stale_hits(path, lines, patterns(), [], CURRENT)
    return found


def test_normalize_value_undoes_latex_minus_and_percent():
    assert normalize_value("$-$0.015") == "-0.015"
    assert normalize_value("30\\%") == "30%"
    assert normalize_value("−0.015") == "-0.015"


def test_superseded_values_drop_the_current_value():
    superseded = superseded_values(HISTORY, CURRENT)
    assert superseded["HeadlineAurocAdaptive"] == {"0.935"}
    assert superseded["PanelCellsCached"] == {"65"}


def test_decimal_pattern_keeps_sign_and_digit_boundaries():
    signed = decimal_pattern("+0.103")
    assert signed.search("a gain of +0.103 AUROC")
    assert signed.search("a gain of 0.103 AUROC")
    assert not signed.search("a delta of -0.103")
    assert not signed.search("0.1034")
    assert not signed.search("10.103")


def test_count_pattern_only_matches_a_population():
    pattern = count_pattern("65")
    assert pattern.search("the 65 cells that clear")
    assert pattern.search("a 65-cell panel")
    assert pattern.search("read over 65 and")
    assert pattern.search("n = 65")
    assert not pattern.search("165 cells")
    assert not pattern.search("65 percent")


def test_a_superseded_headline_value_is_flagged():
    found = hits(["PSBD-TM reads mean AUROC 0.935 at the adaptive rule."])
    assert len(found) == 1
    assert "0.935" in found[0]


def test_a_decimal_without_headline_context_is_not_flagged():
    assert hits(["clean accuracy 0.935 on the benign model"]) == []


def test_a_dated_or_historical_line_is_a_record():
    assert hits(["On 2026-09-23 the mean AUROC read 0.935."]) == []
    assert hits(["The 0.935 AUROC below is historical."]) == []


def test_a_date_inside_a_file_name_does_not_exempt_the_line():
    line = "AUROC 0.935, see docs/sam-findings-2026-09-11.md"
    assert len(hits([line])) == 1


def test_a_superseded_status_line_exempts_a_record_but_not_a_living_document():
    lines = [
        "# Title",
        "",
        "> **Superseded on 2026-09-29:** current values.",
        "AUROC 0.935",
    ]
    assert hits(lines, "docs/results/old-table.md") == []
    assert len(hits(lines, "README.md")) == 1


def test_a_line_citing_a_macro_with_that_current_value_is_not_flagged():
    line = "PSBD-TM reads AUROC 0.951 at shift 0.8 (`\\LadderAurocAtZeroEight`)"
    assert hits([line]) == []


def test_the_allowlist_suppresses_a_listed_value_in_a_listed_file():
    lines = ["PSBD-TM reads AUROC 0.935 in this unrelated table"]
    allowlist = [("docs/some-*.md", "0.935")]
    found = stale_hits("docs/some-note.md", lines, patterns(), allowlist, CURRENT)
    assert found == []


def test_a_superseded_count_is_flagged_only_as_a_population():
    assert len(hits(["The headline reads on 65 models."])) == 1
    assert hits(["65 percent of the heads"]) == []


def test_macros_parse_from_tex_and_from_a_sidecar():
    tex = "\\newcommand{\\GainsHighestRate}{$-$0.015}  % mean paired gain at 10%\n"
    assert macros_from_tex(tex) == {"GainsHighestRate": "-0.015"}
    sidecar = '{"macros": {"PanelCellsCached": {"value": "57", "meaning": "cells"}}}'
    assert macros_from_sidecar(sidecar) == {"PanelCellsCached": "57"}


def test_claude_mismatches_report_a_stale_bold_number():
    current = {
        "PanelCellsTotal": "98",
        "HeadlineAurocAdaptive": "0.953",
    }
    text = "The ledger holds **98** cells. PSBD-TM scores **0.935**.\n## Style guides\n"
    mismatches = claude_mismatches(text, current)
    assert any("0.935" in report for report in mismatches)
    assert any("HeadlineAurocAdaptive" in report for report in mismatches)


def test_claude_mismatches_skip_withdrawn_results_and_arxiv_ids():
    current = {"HeadlineAurocAdaptive": "0.953"}
    text = (
        "PSBD (arXiv 2406.05826) scores **0.953**. The earlier +0.258 headline "
        "stays withdrawn.\n## Style guides\n"
    )
    assert claude_mismatches(text, current) == []


def test_claude_md_headline_numbers_equal_the_current_macros():
    current = macros_from_tex(read_text(os.path.join(REPO, HEADLINE_TEX)))
    claude_text = read_text(os.path.join(REPO, CLAUDE_MD))
    assert claude_mismatches(claude_text, current) == []
