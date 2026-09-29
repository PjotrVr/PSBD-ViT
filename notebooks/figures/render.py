"""Compile every TikZ diagram in this folder and rasterize it for the notebooks.

Run from the repository root:

    .venv/bin/python notebooks/figures/render.py            # every .tex here
    .venv/bin/python notebooks/figures/render.py psbd_pipeline

tectonic compiles each .tex to a PDF in a scratch folder, and pypdfium2, already a
dependency of the project's venv, rasterizes page 1 to a PNG beside the source.
The notebooks embed the PNG, since a notebook viewer shows PNG inline and not PDF.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pypdfium2

FIGURES_DIR = Path(__file__).resolve().parent
TECTONIC = os.environ.get("TECTONIC", str(Path.home() / ".local/bin/tectonic"))
# 3 times the PDF's 72 dpi gives 216 dpi, sharp at notebook width.
RENDER_SCALE = 3


def main():
    names = sys.argv[1:] or sorted(path.stem for path in FIGURES_DIR.glob("*.tex"))
    for name in names:
        png_path = render(FIGURES_DIR / f"{name}.tex")
        print(
            f"{name}.tex rendered to {png_path.relative_to(FIGURES_DIR.parent.parent)}"
        )


def render(tex_path):
    with tempfile.TemporaryDirectory() as scratch:
        scratch_tex = Path(scratch) / tex_path.name
        shutil.copy(tex_path, scratch_tex)
        compile_pdf(scratch_tex)
        png_path = tex_path.with_suffix(".png")
        rasterize(scratch_tex.with_suffix(".pdf"), png_path)
    return png_path


def compile_pdf(tex_path):
    # Tectonic fetches missing packages over the network, which on Supek needs
    # the proxy exported in the calling shell.
    completed = subprocess.run(
        [TECTONIC, "-X", "compile", tex_path.name],
        cwd=tex_path.parent,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"tectonic failed on {tex_path.name}\n{completed.stderr}")


def rasterize(pdf_path, png_path):
    document = pypdfium2.PdfDocument(str(pdf_path))
    image = document[0].render(scale=RENDER_SCALE).to_pil()
    image.save(png_path)
    document.close()


if __name__ == "__main__":
    main()
