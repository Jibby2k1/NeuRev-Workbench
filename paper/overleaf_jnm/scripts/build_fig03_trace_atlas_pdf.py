#!/usr/bin/env python3
"""Package the frozen hero/antihero/random raster panels as a multipage PDF."""
from pathlib import Path

from PIL import Image
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures" / "final"
SOURCES = [
    FIGURES / "fig03a_cs_parzen_hero_roi003_b03.png",
    FIGURES / "fig03b_cs_parzen_antihero_roi011_b04.png",
    FIGURES / "fig03c_cs_parzen_random_roi010_b03.png",
]
OUTPUT = FIGURES / "fig03_trace_atlas_examples.pdf"


def main() -> int:
    with Image.open(SOURCES[0]) as first:
        page_size = tuple(float(value) for value in first.size)
    document = canvas.Canvas(str(OUTPUT), pagesize=page_size, pageCompression=1)
    document.setTitle("Frozen CS-Parzen trace-atlas cases")
    document.setAuthor("NeuRev Workbench")
    for source in SOURCES:
        with Image.open(source) as image:
            if image.size != tuple(int(value) for value in page_size):
                raise ValueError(f"inconsistent source dimensions: {source} {image.size}")
        # Inline embedding avoids a Poppler/XObject decoding defect observed for
        # the high-frequency antihero raster when packaged as a reusable image.
        document.drawInlineImage(str(source), 0, 0, width=page_size[0], height=page_size[1])
        document.showPage()
    document.save()
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
