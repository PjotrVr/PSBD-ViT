"""Refuse a living document that quotes a headline number the paper build has replaced.

The headline numbers live as macros in `paper/headline.tex`. Every earlier build
of that file is in git, so the values a macro used to carry are known exactly.
This script collects those earlier values for the headline macros (the PSBD-TM
and PSBD-RD means, the paired gains and their intervals, the panel counts and the
Swin mean and count), drops the ones equal to the current value, and searches
every tracked markdown file and notebook outside `paper/` for them.

A decimal counts only on a line that also names a headline subject such as AUROC,
a placement, a gain, the panel or Swin. An integer counts only as a population
(`65 cells`, `65-cell`, `over 65`, `n = 65`). That keeps a per-model table value
that happens to equal an old mean out of the report, and so does a line that dates
its number or calls it historical or withdrawn. A file is exempt when it is
a dated record carrying a status line `Superseded on <date>` near its top, except
the 4 living documents, which must state the current value. A legitimate match
is listed in `scripts/stale_numbers_allowlist.txt` with its reason.

The script also holds `.claude/CLAUDE.md` to the build: every bold number in its
headline paragraphs must be the current value of a named headline macro, every
named macro must appear there in bold, and every decimal in those paragraphs must
equal the current value of some macro.

    python scripts/check_stale_numbers.py
    python scripts/check_stale_numbers.py --list-superseded

Exit status is 0 only when nothing is reported.
"""

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys

HEADLINE_TEX = "paper/headline.tex"
# The macro sidecars the headline macros were folded from before headline.tex
# held them, so the history reaches back to the first generated headline.
HISTORY_PATHS = (
    HEADLINE_TEX,
    "paper/tables/headline.macros.json",
    "paper/tables/gains.macros.json",
    "paper/tables/panel.macros.json",
    "paper/tables/swin.macros.json",
)
DECIMAL_MACROS = (
    "HeadlineAurocAdaptive",
    "HeadlineAurocMatched",
    "PublishedAurocAdaptive",
    "HeadlineGainAdaptiveAuroc",
    "HeadlineGainAdaptiveAurocLow",
    "HeadlineGainAdaptiveAurocHigh",
    "HeadlineGainMatchedAuroc",
    "GainsAll",
    "GainsLowestRate",
    "GainsHighestRate",
    "SwinRecommendedAurocAdaptive",
    "SwinPublishedAurocAdaptive",
)
COUNT_MACROS = (
    "PanelCellsTotal",
    "PanelCellsClearing",
    "PanelCellsCached",
    "GainsModels",
    "SwinRecommendedN",
    "SwinCells",
)
# The macros the headline paragraphs of .claude/CLAUDE.md print in bold.
CLAUDE_BOLD_MACROS = (
    "PanelCellsTotal",
    "PanelCellsClearing",
    "PanelCellsDiverged",
    "PanelCellsSourceMapped",
    "PanelCellsCached",
    "HeadlineAurocAdaptive",
    "PublishedAurocAdaptive",
    "HeadlineGainAdaptiveAuroc",
    "GainsLowestRate",
    "GainsHighestRate",
    "HeadlineFloorAuroc",
    "SwinRecommendedAurocAdaptive",
    "DetectorsAurocMargin",
    "GaussianMinusTokenMaskAttentionNorm",
    "AttentionInputMinusMlpInputTokenMask",
)
CLAUDE_MD = ".claude/CLAUDE.md"
CLAUDE_HEADLINE_END = "## Style guides"
# Documents a reader takes as the current state. A status line cannot exempt them.
LIVING_DOCUMENTS = (
    "README.md",
    CLAUDE_MD,
    "docs/open-questions.md",
    "docs/hypothesis/README.md",
)
ALLOWLIST = "scripts/stale_numbers_allowlist.txt"
SUPERSEDED_STATUS = re.compile(r"Superseded on \d{4}-\d{2}-\d{2}")
STATUS_WINDOW_LINES = 30
HEADLINE_CONTEXT = re.compile(
    r"AUROC|PSBD-TM|PSBD-RD|headline|gain|recommended|published|token_mask|"
    r"token mask|post_residual|Swin|panel|placement",
    re.IGNORECASE,
)
# A line that dates its number or calls it historical states a record, not a
# current value. A date inside a file name does not count.
HISTORICAL_MARKER = re.compile(
    r"(?<![\w/-])20\d\d-\d\d-\d\d(?![\w.-])|not re-?measured|not recomputed|"
    r"not recounted|historical|withdrawn|superseded",
    re.IGNORECASE,
)
CITED_MACRO = re.compile(r"\\([A-Z][A-Za-z]+)")
MACRO_DEFINITION = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\}\s*(?:%|$)")
DECIMAL_TOKEN = re.compile(r"(?<![\w.])[+\-\u2212]?\d*\.\d+(?![\d])")
ARXIV_ID = re.compile(r"arXiv \d{4}\.\d{4,5}")
SENTENCE_BREAK = re.compile(r"(?<=[.)])\s+(?=[A-Z*(\[])")
BOLD_TOKEN = re.compile(r"\*\*([+\-\u2212]?\d[\d.]*%?)\*\*")


def git(*arguments: str) -> str:
    """Standard output of a git command, empty when the command fails."""
    completed = subprocess.run(
        ["git", *arguments], capture_output=True, text=True, check=False
    )
    return completed.stdout


def normalize_value(raw: str) -> str:
    """A macro or prose value as plain text: LaTeX minus and percent undone."""
    value = raw.replace("$-$", "-").replace("\u2212", "-").replace("\\%", "%")
    value = value.strip()
    return value


def macros_from_tex(text: str) -> dict[str, str]:
    """Every `\\newcommand` in a headline.tex text, name to normalized value."""
    macros = {}
    for line in text.splitlines():
        match = MACRO_DEFINITION.match(line)
        if match:
            macros[match.group(1)] = normalize_value(match.group(2))
    return macros


def macros_from_sidecar(text: str) -> dict[str, str]:
    """Every macro in a `*.macros.json` sidecar text, name to normalized value."""
    try:
        sidecar = json.loads(text)
    except json.JSONDecodeError:
        return {}
    entries = sidecar.get("macros", {}) if isinstance(sidecar, dict) else {}
    macros = {
        name: normalize_value(str(entry.get("value", "")))
        for name, entry in entries.items()
        if isinstance(entry, dict)
    }
    return macros


def read_macros(text: str, path: str) -> dict[str, str]:
    """The macros in a headline.tex or sidecar text, parsed by the file's kind."""
    macros = (
        macros_from_sidecar(text) if path.endswith(".json") else macros_from_tex(text)
    )
    return macros


def macro_history(macro_names: tuple[str, ...]) -> dict[str, set[str]]:
    """Every value each named macro has carried in any committed build."""
    revisions = git("log", "--all", "--format=%H", "--", *HISTORY_PATHS).split()
    history: dict[str, set[str]] = {name: set() for name in macro_names}
    for revision in revisions:
        for path in HISTORY_PATHS:
            text = git("show", f"{revision}:{path}")
            if not text:
                continue
            macros = read_macros(text, path)
            for name in macro_names:
                if macros.get(name):
                    history[name].add(macros[name])
    return history


def superseded_values(
    history: dict[str, set[str]], current: dict[str, str]
) -> dict[str, set[str]]:
    """Per macro, the earlier values that differ from the current one."""
    superseded = {
        name: {value for value in values if value != current.get(name)}
        for name, values in history.items()
    }
    return superseded


def decimal_pattern(value: str) -> re.Pattern:
    """A superseded decimal as it can be written in prose, with or without its sign."""
    magnitude = value.lstrip("+-")
    sign = value[0] if value[0] in "+-" else ""
    # An unsigned magnitude matches too, but a magnitude with the opposite sign never does.
    sign_class = {"+": r"\+?", "-": r"[-\u2212]?", "": ""}[sign]
    pattern = re.compile(
        rf"(?<![\w.+\-\u2212]){sign_class}{re.escape(magnitude)}(?!\d)"
    )
    return pattern


def count_pattern(value: str) -> re.Pattern:
    """A superseded population count, only where the text uses it as a population."""
    population = (
        rf"(?<![\w.+-]){value}(?:-cell|-model| cells| models| clearing| compared"
        rf"| paired| checkpoints)\b"
        rf"|\bover {value}\b(?![.\d])|\bn ?= ?{value}\b"
    )
    pattern = re.compile(population)
    return pattern


def build_patterns(
    superseded: dict[str, set[str]],
) -> list[tuple[str, str, re.Pattern]]:
    """(macro, value, pattern) for every superseded value, decimals then counts."""
    patterns = []
    for name, values in sorted(superseded.items()):
        for value in sorted(values):
            if name in COUNT_MACROS:
                patterns.append((name, value, count_pattern(value)))
            else:
                patterns.append((name, value, decimal_pattern(value)))
    return patterns


def tracked_documents() -> list[str]:
    """Tracked markdown files and notebooks outside paper/."""
    listed = git("ls-files", "*.md", "*.ipynb").splitlines()
    documents = [path for path in listed if not path.startswith("paper/")]
    return documents


def document_lines(path: str) -> list[str]:
    """The prose lines of a document: a notebook's markdown and printed outputs."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if not path.endswith(".ipynb"):
        markdown_lines = text.splitlines()
        return markdown_lines
    notebook = json.loads(text)
    lines = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "markdown":
            lines.extend("".join(cell.get("source", [])).splitlines())
        for output in cell.get("outputs", []):
            printed = output.get("text") or output.get("data", {}).get("text/plain", [])
            lines.extend("".join(printed).splitlines())
    return lines


def carries_superseded_status(lines: list[str]) -> bool:
    """Whether a `Superseded on <date>` status line sits near the top."""
    status_found = any(
        SUPERSEDED_STATUS.search(line) for line in lines[:STATUS_WINDOW_LINES]
    )
    return status_found


def read_allowlist(path: str) -> list[tuple[str, str]]:
    """(path glob, value) pairs from the allowlist, 1 per line as `glob | value | reason`."""
    if not os.path.exists(path):
        return []
    entries = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = [field.strip() for field in stripped.split("|")]
            entries.append((fields[0], fields[1]))
    return entries


def is_allowed(path: str, value: str, allowlist: list[tuple[str, str]]) -> bool:
    """Whether the allowlist names this value for a glob matching this path."""
    allowed = any(
        fnmatch.fnmatch(path, glob) and value == allowed_value
        for glob, allowed_value in allowlist
    )
    return allowed


def cites_current_value(line: str, value: str, current: dict[str, str]) -> bool:
    """Whether the line names a macro whose current value is this very number.

    A superseded value of 1 macro can be the live value of another, and a line
    that cites that other macro by name is quoting it correctly.
    """
    magnitude = value.lstrip("+-")
    cited = CITED_MACRO.findall(line)
    quoted = any(current.get(name, "").lstrip("+-") == magnitude for name in cited)
    return quoted


def stale_hits(
    path: str,
    lines: list[str],
    patterns: list[tuple[str, str, re.Pattern]],
    allowlist: list[tuple[str, str]],
    current: dict[str, str],
) -> list[str]:
    """Report lines for every superseded value this document quotes as current."""
    if path not in LIVING_DOCUMENTS and carries_superseded_status(lines):
        return []
    hits = []
    for line_number, line in enumerate(lines, start=1):
        # 1 value can be the old reading of several macros, reported once per line.
        names_by_value: dict[str, list[str]] = {}
        for name, value, pattern in patterns:
            if not pattern.search(line):
                continue
            if HISTORICAL_MARKER.search(line):
                continue
            is_decimal = name not in COUNT_MACROS
            if is_decimal and not HEADLINE_CONTEXT.search(line):
                continue
            if is_allowed(path, value, allowlist):
                continue
            if cites_current_value(line, value, current):
                continue
            names_by_value.setdefault(value, []).append(f"\\{name}")
        hits += [
            f"{path}:{line_number}: {value} is a superseded {', '.join(names)}"
            for value, names in names_by_value.items()
        ]
    return hits


def claude_headline_text(claude_text: str) -> str:
    """The headline paragraphs of CLAUDE.md, everything before the style guides."""
    headline = claude_text.split(CLAUDE_HEADLINE_END, 1)[0]
    return headline


def claude_mismatches(claude_text: str, current: dict[str, str]) -> list[str]:
    """Every CLAUDE.md headline number that is not the current value of its macro."""
    headline = claude_headline_text(claude_text)
    bold_values = {normalize_value(token) for token in BOLD_TOKEN.findall(headline)}
    expected = {name: current[name] for name in CLAUDE_BOLD_MACROS if name in current}
    mismatches = [
        f"{CLAUDE_MD}: \\{name} is {value} in {HEADLINE_TEX} and not bold in the paragraph"
        for name, value in expected.items()
        if value not in bold_values
    ]
    mismatches += [
        f"{CLAUDE_MD}: bold {value} is no current headline macro"
        for value in sorted(bold_values - set(expected.values()))
    ]
    every_value = set(current.values())
    for sentence in SENTENCE_BREAK.split(ARXIV_ID.sub("", headline)):
        # A withdrawn result is quoted as withdrawn and has no macro by design.
        if "withdrawn" in sentence:
            continue
        for token in DECIMAL_TOKEN.findall(sentence):
            value = normalize_value(token)
            unsigned = value.lstrip("+")
            if value in every_value or unsigned in every_value:
                continue
            mismatches.append(f"{CLAUDE_MD}: {value} equals no macro in {HEADLINE_TEX}")
    return mismatches


def read_text(path: str) -> str:
    """The whole text of a file."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--list-superseded",
        action="store_true",
        help="print the superseded value of every headline macro and exit",
    )
    parser.add_argument("--allowlist", default=ALLOWLIST)
    args = parser.parse_args()
    return args


def main() -> int:
    args = parse_args()
    # Every path here, and every git call, is relative to the repository root.
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    current = macros_from_tex(read_text(HEADLINE_TEX))
    history = macro_history(DECIMAL_MACROS + COUNT_MACROS)
    superseded = superseded_values(history, current)
    if args.list_superseded:
        for name, values in sorted(superseded.items()):
            print(f"\\{name}: current {current.get(name)}, superseded {sorted(values)}")
        return 0

    patterns = build_patterns(superseded)
    allowlist = read_allowlist(args.allowlist)
    reports = []
    for path in tracked_documents():
        if not os.path.exists(path):
            continue
        reports += stale_hits(path, document_lines(path), patterns, allowlist, current)
    reports += claude_mismatches(read_text(CLAUDE_MD), current)

    for report in reports:
        print(report)
    print(f"{len(reports)} stale headline numbers", file=sys.stderr)
    exit_status = 1 if reports else 0
    return exit_status


if __name__ == "__main__":
    sys.exit(main())
