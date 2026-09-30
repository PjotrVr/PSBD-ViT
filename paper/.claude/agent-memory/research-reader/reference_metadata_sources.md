---
name: reference-metadata-sources
description: Which bibliographic sources work from the Supek cluster (DBLP is blocked) and how to read PDFs there
metadata:
  type: reference
---

DBLP (dblp.org, dblp.uni-trier.de) sits behind an Anubis proof-of-work wall that blocks both curl and WebFetch from Supek (checked 2026-09-24). Semantic Scholar rate-limits with 429 and OpenReview API v1 does too.

Sources that work through the proxy `http://10.150.1.1:3128`: OpenAlex (`api.openalex.org/works?filter=title.search:...`), Crossref (`api.crossref.org/works/<doi>`), the arXiv export API and PDFs, CVF Open Access html pages (the BibTeX sits in `div.bibref`), PMLR volume indexes (`proceedings.mlr.press/vNNN/`), NeurIPS `papers.nips.cc/paper_files/paper/<year>` with `-Bibtex.bib` files, OpenReview API v2 `notes/search`.

For PDF text, the project venv (`.venv/bin/python`, 3.11) has pypdf. System python3 is too old for it and there is no pdftotext.

Related: [[psbd-bibliography-audit-2026-09-24]]
