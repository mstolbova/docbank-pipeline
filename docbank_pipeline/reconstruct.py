"""
docbank_pipeline/reconstruct.py

Reconstruct an uploaded page image into a CLEAN, re-typeset PDF.

Unlike to_pdf.py (which keeps the original scan as the page background and
overlays invisible searchable text), this module draws each detected region
as NATIVE PDF content on a blank white page:

  * text    -> recognized text drawn as real, selectable text at its bbox
  * formula -> LaTeX rendered via matplotlib mathtext; if the LaTeX can't be
               rendered (e.g. \\begin{array}{...}, common in pix2tex output),
               it falls back to embedding the original formula crop bitmap
  * image   -> the original crop bitmap embedded at its bbox

The result approximates the original layout but is selectable, searchable,
and free of the source scan. Supports Korean / non-latin text via a Unicode
font (Malgun Gothic auto-detected on Windows).

Public API:
    image_to_reconstructed_pdf(cfg, image_path, output_pdf)  -> Path
    detections_to_reconstructed_pdf(detections, output_pdf)  -> Path

Requires: pip install reportlab matplotlib
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence

from PIL import Image

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader, simpleSplit
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase.pdfmetrics import stringWidth
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "reportlab is required. Install with: pip install reportlab"
    ) from e

# These are only needed when this module is used inside the package.
# Imported lazily in image_to_reconstructed_pdf so the file-only renderer
# (detections_to_reconstructed_pdf) can be used standalone if desired.


# --------------------------------------------------------------------------
# Fonts (Unicode / Korean)
# --------------------------------------------------------------------------
_FONT_CANDIDATES = [
    # ----- Linux (Hugging Face / Render / Docker) -----
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",

    # ----- macOS -----
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/Arial Unicode.ttf",

    # ----- Windows -----
    r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\NanumGothic.ttf",
    r"C:\Windows\Fonts\gulim.ttc",
    r"C:\Windows\Fonts\arialuni.ttf",
]

_REGISTERED: set[str] = set()


def _resolve_font(font_path) -> str:
    """Register and return a usable font name (a Unicode TTF if available,
    else Helvetica which is latin-1 only)."""
    candidates: list = [font_path] if font_path else []
    candidates += _FONT_CANDIDATES
    for p in candidates:
        p = Path(p)
        if not p.is_file():
            continue
        name = p.stem
        if name in _REGISTERED:
            return name
        try:
            if p.suffix.lower() == ".ttc":
                pdfmetrics.registerFont(TTFont(name, str(p), subfontIndex=0))
            else:
                pdfmetrics.registerFont(TTFont(name, str(p)))
            _REGISTERED.add(name)
            return name
        except Exception:
            continue
    return "Helvetica"


# --------------------------------------------------------------------------
# Formula rendering (matplotlib mathtext -> transparent PNG)
# --------------------------------------------------------------------------
def _render_latex_png(latex: str, png_path: Path, *, fontsize: int = 24, dpi: int = 200) -> bool:
    """Render a LaTeX math string to a transparent PNG with matplotlib's
    built-in mathtext (no system LaTeX needed). Returns True on success.

    mathtext supports only a subset of LaTeX; complex pix2tex output such as
    \\begin{array}{...} raises ValueError here, in which case the caller
    falls back to the original crop bitmap."""
    expr = (latex or "").strip().strip("$")
    if not expr:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    try:
        fig = plt.figure(figsize=(0.1, 0.1))
        fig.text(0.0, 0.0, f"${expr}$", fontsize=fontsize)
        fig.savefig(str(png_path), dpi=dpi, bbox_inches="tight",
                    pad_inches=0.02, transparent=True)
        plt.close(fig)
        return png_path.is_file() and png_path.stat().st_size > 0
    except Exception:
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass
        return False


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def _draw_text(c, det, *, font_name: str, page_h: float) -> None:
    text = (det.get("recognized") or "").strip()
    if not text:
        return
    x1, y1, x2, y2 = (float(v) for v in det.get("bbox", [0, 0, 0, 0]))
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)

    raw_lines = text.splitlines() or [text]
    n = max(1, len(raw_lines))
    font_size = max(4.0, (box_h / n) * 0.82)

    latin_only = (font_name == "Helvetica")

    # Wrap any line that's too wide for the box.
    lines: list[str] = []
    for ln in raw_lines:
        if latin_only:
            ln = ln.encode("latin-1", "replace").decode("latin-1")
        if not ln:
            lines.append("")
            continue
        if stringWidth(ln, font_name, font_size) <= box_w:
            lines.append(ln)
        else:
            lines.extend(simpleSplit(ln, font_name, font_size, box_w) or [ln])

    n2 = max(1, len(lines))
    line_h = box_h / n2
    font_size = min(font_size, max(4.0, line_h * 0.85))

    t = c.beginText()
    t.setFont(font_name, font_size)
    t.setLeading(line_h)
    t.setFillColorRGB(0, 0, 0)
    t.setTextOrigin(x1, (page_h - y1) - font_size)  # image top-down -> PDF bottom-up
    for ln in lines:
        t.textLine(ln)
    c.drawText(t)


def _draw_bitmap(c, src, det, *, page_h: float, fit: bool) -> None:
    """Embed a bitmap at the detection's bbox. fit=True preserves aspect and
    centers; fit=False stretches to fill the box."""
    x1, y1, x2, y2 = (float(v) for v in det.get("bbox", [0, 0, 0, 0]))
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    img = ImageReader(str(src))
    if fit:
        iw, ih = img.getSize()
        scale = min(box_w / iw, box_h / ih)
        dw, dh = iw * scale, ih * scale
        ox = x1 + (box_w - dw) / 2.0
        oy_top = (page_h - y1) - (box_h - dh) / 2.0
        c.drawImage(img, ox, oy_top - dh, dw, dh, mask="auto")
    else:
        c.drawImage(img, x1, page_h - y2, box_w, box_h, mask="auto")


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------
def detections_to_reconstructed_pdf(
    detections: Sequence[dict],
    output_pdf: str | Path,
    *,
    font_path: str | Path | None = None,
    render_formulas: bool = True,
) -> Path:
    """Render a flat list of enriched detections into a reconstructed PDF.

    Each detection needs: ``image`` (source page path), ``bbox``
    ``[x1,y1,x2,y2]`` in image pixels, ``class_name``, and either
    ``recognized`` text/LaTeX or a ``crop_path`` bitmap. One page is
    produced per distinct source image, sized to that image's pixels.

    render_formulas : True  -> try to render formula LaTeX via matplotlib,
                               falling back to the crop bitmap on failure.
                      False -> always use the crop bitmap for formulas (most
                               faithful, fastest, no matplotlib needed).
    """
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    font_name = _resolve_font(font_path)

    pages: dict[str, list[dict]] = {}
    for det in detections:
        img = det.get("image") or det.get("image_path") or ""
        pages.setdefault(str(img), []).append(det)

    c = canvas.Canvas(str(output_pdf))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fcount = 0
        for img_path, dets in pages.items():
            if not img_path or not Path(img_path).is_file():
                continue
            with Image.open(img_path) as im:
                page_w, page_h = im.size

            c.setPageSize((page_w, page_h))
            # Blank white background.
            c.setFillColorRGB(1, 1, 1)
            c.rect(0, 0, page_w, page_h, stroke=0, fill=1)
            c.setFillColorRGB(0, 0, 0)

            for det in dets:
                cls = (det.get("class_name") or "").lower()
                rec = (det.get("recognized") or "").strip()
                crop = det.get("crop_path")
                has_crop = bool(crop and Path(str(crop)).is_file())

                if cls == "formula":
                    drawn = False
                    if render_formulas and rec:
                        png = tmp / f"f{fcount}.png"
                        fcount += 1
                        if _render_latex_png(rec, png):
                            _draw_bitmap(c, png, det, page_h=page_h, fit=True)
                            drawn = True
                    if not drawn and has_crop:
                        _draw_bitmap(c, crop, det, page_h=page_h, fit=True)

                elif cls == "image":
                    if has_crop:
                        _draw_bitmap(c, crop, det, page_h=page_h, fit=False)

                else:  # text, or any class carrying recognized text
                    if rec:
                        _draw_text(c, det, font_name=font_name, page_h=page_h)
                    elif has_crop:
                        _draw_bitmap(c, crop, det, page_h=page_h, fit=False)

            c.showPage()
        c.save()
    return output_pdf


def image_to_reconstructed_pdf(
    cfg,
    image_path: str | Path,
    output_pdf: str | Path,
    *,
    conf: float = 0.25,
    do_text: bool = True,
    do_formula: bool = True,
    font_path: str | Path | None = None,
    render_formulas: bool = True,
) -> Path:
    """End-to-end: detect + OCR + formula-recognise a single uploaded image,
    then write a clean reconstructed PDF version of it."""
    from .ocr import process_page_image  # lazy import: only needed here

    detections = process_page_image(
        cfg, image_path, conf=conf, do_text=do_text, do_formula=do_formula,
    )
    for det in detections:
        det.setdefault("image", str(image_path))

    return detections_to_reconstructed_pdf(
        detections, output_pdf,
        font_path=font_path, render_formulas=render_formulas,
    )
