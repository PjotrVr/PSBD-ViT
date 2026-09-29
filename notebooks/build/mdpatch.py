"""Replace the md(r'''...''') block that starts with a given prefix by new builder source."""


def replace_block(path, prefix, new_source):
    text = open(path).read()
    start = text.index('    md(r"""\n    ' + prefix)
    end = text.index('"""),', start) + len('"""),')
    text = text[:start] + new_source.rstrip("\n") + text[end:]
    open(path, "w").write(text)


def insert_after(path, anchor, new_source):
    text = open(path).read()
    position = text.index(anchor) + len(anchor)
    text = text[:position] + new_source + text[position:]
    open(path, "w").write(text)
