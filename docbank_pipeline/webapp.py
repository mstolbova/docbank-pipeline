"""
Stage 7 — Standalone web app.

Flask app that lets users upload page images or PDF files, runs the full
detection + OCR + LaTeX recognition pipeline, and returns PDF output.

Supports:
- Searchable PDF
- Reconstructed PDF
- Both versions as ZIP
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
import zipfile
from pathlib import Path

from .config import PipelineConfig

log = logging.getLogger("docbank.webapp")


# ----------------------------------------------------------- box drawing

_BOX_COLORS_BGR = {
    "text": (255, 180, 0),
    "formula": (0, 200, 0),
    "image": (80, 80, 255),
}


def _draw_boxes(image_path: Path, detections: list[dict], out_path: Path) -> bool:
    """Draw labelled bounding boxes on a copy of the page image."""
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        log.warning("Could not read %s", image_path)
        return False

    for d in detections:
        cls = d["class_name"]
        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
        color = _BOX_COLORS_BGR.get(cls, (200, 200, 200))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{cls} {d['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            img,
            label,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PDF_EXTS = {".pdf"}
ALLOWED_EXTS = IMAGE_EXTS | PDF_EXTS

PDF_RENDER_DPI = 200
PDF_MAX_PAGES = 25


def _explode_pdf(pdf_path: Path, out_dir: Path) -> list[Path]:
    """Render PDF pages to JPG images using PyMuPDF."""
    try:
        import fitz
    except ImportError as e:
        raise ImportError(
            "PDF support needs PyMuPDF. Install with: pip install pymupdf"
        ) from e

    doc = fitz.open(str(pdf_path))
    try:
        n_pages = min(len(doc), PDF_MAX_PAGES)

        if len(doc) > PDF_MAX_PAGES:
            log.warning(
                "PDF %s has %d pages; only the first %d will be processed.",
                pdf_path.name,
                len(doc),
                PDF_MAX_PAGES,
            )

        zoom = PDF_RENDER_DPI / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        stem = pdf_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        produced: list[Path] = []

        for i in range(n_pages):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out = out_dir / f"{stem}_p{i + 1:03d}.jpg"
            pix.save(str(out), jpg_quality=92)
            produced.append(out)

        return produced

    finally:
        doc.close()


# --------------------------------------------------------------- HTML

_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DocBank Layout Extractor</title>
<style>
  body {
    font-family: -apple-system, Segoe UI, Roboto, sans-serif;
    margin: 0;
    padding: 32px;
    background: #f5f6fa;
    color: #222;
  }

  .wrap {
    max-width: 760px;
    margin: 40px auto;
  }

  h1 {
    margin: 0 0 8px;
  }

  .lead {
    color: #555;
    margin-bottom: 32px;
  }

  form {
    background: #fff;
    border: 1px solid #e2e6ee;
    border-radius: 10px;
    padding: 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  }

  .drop {
    border: 2px dashed #b6bcc8;
    border-radius: 8px;
    padding: 36px;
    text-align: center;
    color: #777;
    transition: all .15s;
    cursor: pointer;
  }

  .drop.over {
    border-color: #1e88e5;
    background: #f0f7ff;
    color: #1e88e5;
  }

  .drop input {
    display: none;
  }

  .opts {
    margin-top: 16px;
    display: flex;
    gap: 24px;
    align-items: center;
    flex-wrap: wrap;
  }

  .opts label {
    font-size: 14px;
    color: #444;
  }

  .opts input[type=number] {
    width: 80px;
    padding: 4px 8px;
  }

  .opts select {
    padding: 4px 8px;
  }

  button {
    background: #1e88e5;
    color: #fff;
    border: 0;
    padding: 10px 20px;
    border-radius: 6px;
    font-size: 15px;
    cursor: pointer;
    margin-top: 18px;
  }

  button:disabled {
    background: #aaa;
    cursor: not-allowed;
  }

  .files {
    margin-top: 12px;
    font-size: 13px;
    color: #555;
  }

  .files li {
    list-style: none;
  }

  .spinner {
    display: none;
    margin: 16px 0;
    color: #555;
  }

  .spinner.on {
    display: block;
  }
</style>
</head>

<body>
<div class="wrap">
  <h1>DocBank Layout Extractor</h1>

  <p class="lead">
    Upload page images <b>or PDF files</b>. PDFs are rendered page-by-page
    up to 25 pages each. The server runs YOLO detection, then PaddleOCR for
    text and pix2tex for LaTeX formulas.
  </p>

  <p class="lead" style="font-size:13px;color:#555;">
    <b>What's running under the hood:</b> the same YOLOv8 model trained on
    DocBank that powers the CLI batch run. Every text crop hits PaddleOCR,
    every formula crop hits pix2tex. Results are cached by crop hash.
    <br>
    <b>Tuning:</b> the form below lets you trade speed for recall.
  </p>

  <form id="uploadForm" method="post" action="/upload" enctype="multipart/form-data">
    <label class="drop" id="drop">
      <input type="file" name="images" id="filesInput" multiple
             accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff,.pdf">
      <div id="dropText">
        <b>Click to choose files</b> or drag &amp; drop here.<br>
        <small>JPG / PNG / PDF, multiple files allowed.</small>
      </div>
      <ul class="files" id="fileList"></ul>
    </label>

    <div class="opts">
      <label title="Choose what PDF to generate">Output type
        <select name="pdf_mode">
          <option value="searchable" selected>Searchable PDF</option>
          <option value="reconstructed">Reconstructed PDF</option>
          <option value="both">Both versions</option>
        </select>
      </label>

      <label title="YOLO detection threshold">Detect conf &ge;
        <input type="number" name="conf" min="0.05" max="0.95" step="0.05" value="0.25">
      </label>

      <label title="Skip OCR for crops below this detection confidence">OCR conf &ge;
        <input type="number" name="min_ocr_conf" min="0" max="1" step="0.05" value="0.30">
      </label>

      <label title="Skip OCR for crops smaller than this many pixels squared">Min area
        <input type="number" name="min_ocr_area" min="0" max="100000" step="100" value="600">
      </label>

      <label title="Stricter area threshold for formula crops only">Min formula area
        <input type="number" name="min_formula_area" min="0" max="100000" step="100" value="2000">
      </label>

      <label title="Hard cap on number of formula crops processed">Max formulas
        <input type="number" name="max_formulas" min="0" max="10000" step="10" value="">
      </label>

      <label>
        <input type="checkbox" name="text_ocr" checked> Text OCR
      </label>

      <label>
        <input type="checkbox" name="formula_ocr" checked> LaTeX OCR
      </label>

      <label title="Skip OCR for crops we have already seen this session">
        <input type="checkbox" name="use_cache" checked> Use cache
      </label>
    </div>

    <button id="submitBtn" type="submit">Extract</button>

    <div class="spinner" id="spinner">
      Processing… first request loads the OCR models. Later uploads are faster.
    </div>
  </form>
</div>

<script>
  const drop = document.getElementById('drop');
  const inp  = document.getElementById('filesInput');
  const list = document.getElementById('fileList');
  const form = document.getElementById('uploadForm');
  const btn  = document.getElementById('submitBtn');
  const spin = document.getElementById('spinner');

  function refresh() {
    list.innerHTML = '';
    for (const f of inp.files) {
      const li = document.createElement('li');
      li.textContent = '• ' + f.name + ' (' + Math.round(f.size / 1024) + ' KB)';
      list.appendChild(li);
    }
  }

  inp.addEventListener('change', refresh);

  ['dragenter', 'dragover'].forEach(ev =>
    drop.addEventListener(ev, e => {
      e.preventDefault();
      drop.classList.add('over');
    })
  );

  ['dragleave', 'drop'].forEach(ev =>
    drop.addEventListener(ev, e => {
      e.preventDefault();
      drop.classList.remove('over');
    })
  );

  drop.addEventListener('drop', e => {
    inp.files = e.dataTransfer.files;
    refresh();
  });

  form.addEventListener('submit', () => {
    if (!inp.files.length) {
      return false;
    }
    btn.disabled = true;
    btn.textContent = 'Working…';
    spin.classList.add('on');
  });
</script>
</body>
</html>
"""


_RESULTS_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Extraction results — {job_id}</title>
<script>
window.MathJax = {{tex: {{inlineMath: [['$','$']], displayMath: [['$$','$$']]}} }};
</script>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  body {{
    font-family: -apple-system, Segoe UI, Roboto, sans-serif;
    margin: 0;
    padding: 24px;
    background: #f5f6fa;
    color: #222;
  }}

  .top {{
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 8px;
  }}

  h1 {{
    margin: 0;
  }}

  .meta {{
    color: #666;
    font-size: 14px;
  }}

  .actions a {{
    background: #1e88e5;
    color: #fff;
    padding: 8px 14px;
    border-radius: 6px;
    text-decoration: none;
    font-size: 14px;
    margin-right: 8px;
  }}

  .actions a.secondary {{
    background: #555;
  }}

  .summary {{
    background: #fff;
    border: 1px solid #e2e6ee;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 14px 0 24px;
    display: inline-block;
  }}

  .summary span {{
    display: inline-block;
    margin-right: 24px;
  }}

  .page {{
    background: #fff;
    border: 1px solid #e2e6ee;
    border-radius: 8px;
    padding: 18px;
    margin-bottom: 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }}

  .page-img img {{
    max-width: 100%;
    height: auto;
    border: 1px solid #d4d8e0;
  }}

  .grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 18px;
  }}

  @media (max-width: 1100px) {{
    .grid {{
      grid-template-columns: 1fr;
    }}
  }}

  .det {{
    border: 1px solid #e2e6ee;
    border-radius: 6px;
    padding: 10px;
    background: #fafbfc;
    display: flex;
    gap: 12px;
  }}

  .det img {{
    max-width: 240px;
    max-height: 120px;
    object-fit: contain;
    border: 1px solid #d4d8e0;
    background: #fff;
  }}

  .det-meta {{
    flex: 1;
    min-width: 0;
  }}

  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    color: #fff;
    font-size: 12px;
    font-weight: 600;
  }}

  .badge.text {{
    background: #1e88e5;
  }}

  .badge.formula {{
    background: #2e7d32;
  }}

  .badge.image {{
    background: #d32f2f;
  }}

  .conf {{
    color: #666;
    font-size: 12px;
    margin-left: 6px;
  }}

  .recog {{
    margin-top: 6px;
    white-space: pre-wrap;
    word-break: break-word;
    font-family: ui-monospace, Consolas, monospace;
    font-size: 13px;
    background: #fff;
    border: 1px solid #e2e6ee;
    padding: 8px;
    border-radius: 4px;
    max-height: 220px;
    overflow: auto;
  }}

  .recog.empty {{
    color: #999;
    font-style: italic;
  }}

  .latex {{
    background: #f3faf3;
  }}
</style>
</head>
<body>
"""


_RESULTS_TAIL = "</body></html>\n"


# --------------------------------------------------------------- helpers

def _summary(detections: list[dict]) -> dict:
    counts: dict[str, list[int]] = {}

    for d in detections:
        c = d.get("class_name", "?")
        counts.setdefault(c, [0, 0])[1] += 1

        if d.get("recognized"):
            counts[c][0] += 1

    return counts


def _render_results(
    job_id: str,
    job_dir: Path,
    pages: list[dict],
    elapsed_s: float,
) -> str:
    import html

    all_dets = [d for p in pages for d in p["detections"]]
    summary = _summary(all_dets)

    head = _RESULTS_HEAD.format(job_id=html.escape(job_id))
    parts: list[str] = [head]

    parts.append(
        f'<div class="top"><h1>Results</h1>'
        f'<div class="meta">job <code>{html.escape(job_id)}</code> '
        f'· {len(pages)} page(s) · {len(all_dets)} detection(s) '
        f'· {elapsed_s:.1f} s</div></div>'
    )

    parts.append('<div class="actions">')
    parts.append(
        f'<a href="/jobs/{job_id}/results.json" download>Download JSON</a>'
    )
    parts.append(f'<a class="secondary" href="/">New upload</a>')
    parts.append('</div>')

    parts.append('<div class="summary">')
    parts.append(f'<span><b>Detections:</b> {len(all_dets)}</span>')

    for cls, (ok, tot) in summary.items():
        pct = 100 * ok / tot if tot else 0
        parts.append(
            f'<span><b>{html.escape(cls)}:</b> {ok}/{tot} '
            f'({pct:.0f}% recognised)</span>'
        )

    cached_count = sum(
        1 for d in all_dets if d.get("recognition_kind") == "cached"
    )

    if cached_count:
        parts.append(
            f'<span style="color:#2e7d32;"><b>Cache hits:</b> '
            f'{cached_count}/{len(all_dets)}</span>'
        )

    parts.append('</div>')

    for i, page in enumerate(pages, 1):
        boxed_rel = f"/jobs/{job_id}/boxes/{Path(page['boxed_image']).name}"

        parts.append('<div class="page">')
        parts.append(
            f'<h3>Page {i}: {html.escape(Path(page["image"]).name)}</h3>'
        )
        parts.append(
            f'<div class="page-img"><img src="{boxed_rel}" alt="page"></div>'
        )
        parts.append('<div class="grid">')

        for d in page["detections"]:
            parts.append(_render_detection_html(job_id, d))

        parts.append('</div></div>')

    parts.append(_RESULTS_TAIL)

    return "".join(parts)


def _render_detection_html(job_id: str, det: dict) -> str:
    import html

    cls = det.get("class_name", "?")
    conf = det.get("confidence", 0.0)
    recog = det.get("recognized") or ""
    bbox = det.get("bbox") or [0, 0, 0, 0]

    crop_w = max(0, int(bbox[2]) - int(bbox[0])) if len(bbox) >= 4 else 0
    crop_h = max(0, int(bbox[3]) - int(bbox[1])) if len(bbox) >= 4 else 0

    crop_html = '<span style="color:#999">[no crop]</span>'

    crop_path = det.get("crop_path")
    if crop_path:
        crop_url = f"/jobs/{job_id}/crops/{cls}/{Path(crop_path).name}"
        crop_html = f'<img src="{crop_url}" alt="crop">'

    if cls == "formula" and recog:
        body = f'<div class="recog latex">$$ {html.escape(recog)} $$</div>'
    elif recog:
        body = f'<div class="recog">{html.escape(recog)}</div>'
    else:
        if crop_w * crop_h < 600:
            note = f"(crop too small for OCR: {crop_w}×{crop_h}px)"
        elif conf < 0.35:
            note = f"(low-confidence detection, conf={conf:.2f})"
        else:
            note = "(no recognition output)"

        body = f'<div class="recog empty">{html.escape(note)}</div>'

    return (
        f'<div class="det">'
        f'  <div class="det-crop">{crop_html}</div>'
        f'  <div class="det-meta">'
        f'    <span class="badge {cls}">{html.escape(cls)}</span>'
        f'    <span class="conf">conf {conf:.2f} | bbox {bbox}</span>'
        f'    {body}'
        f'  </div>'
        f'</div>'
    )


# ---------------------------------------------------------------- pipeline

def _process_uploads(
    cfg: PipelineConfig,
    job_dir: Path,
    inputs: list[Path],
    *,
    conf: float,
    do_text: bool,
    do_formula: bool,
    min_ocr_conf: float = 0.30,
    min_ocr_area: int = 600,
    min_formula_area: int | None = 2000,
    use_cache: bool = True,
    max_text_crops: int | None = None,
    max_formulas: int | None = None,
) -> tuple[list[dict], float]:
    """Run YOLO + OCR on uploaded pages."""
    from .inference import run_yolo_inference
    from .ocr import _enrich

    boxes_dir = job_dir / "boxes"
    crops_dir = job_dir / "crops"

    boxes_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    class _JobCfg:
        def __init__(self, base, jdir):
            self._b = base
            self._j = jdir

        def __getattr__(self, name):
            return getattr(self._b, name)

        @property
        def crops_dir(self):
            return self._j / "crops"

        @property
        def output_dir(self):
            return self._j

    job_cfg = _JobCfg(cfg, job_dir)

    t0 = time.time()

    detections = run_yolo_inference(
        job_cfg,
        inputs,
        conf=conf,
        save_crops=True,
    )

    log.info("YOLO produced %d detection(s)", len(detections))

    cache_path = cfg.output_dir / "web_ocr_cache.json"

    _enrich(
        detections,
        do_text=do_text,
        do_formula=do_formula,
        min_ocr_conf=min_ocr_conf,
        min_ocr_area=min_ocr_area,
        min_formula_area=min_formula_area,
        use_cache=use_cache,
        cache_path=cache_path,
        max_text_crops=max_text_crops,
        max_formulas=max_formulas,
    )

    by_image: dict[str, list[dict]] = {}

    for det in detections:
        by_image.setdefault(str(det["image"]), []).append(det)

    page_records: list[dict] = []

    for img_path in inputs:
        dets = by_image.get(str(img_path), [])
        boxed = boxes_dir / img_path.name

        _draw_boxes(img_path, dets, boxed)

        page_records.append(
            {
                "image": str(img_path),
                "boxed_image": str(boxed),
                "detections": [_serialise_det(d) for d in dets],
            }
        )

    elapsed = time.time() - t0

    return page_records, elapsed


def _serialise_det(det: dict) -> dict:
    """Convert Path objects to strings so json.dump works."""
    out = {}

    for k, v in det.items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v

    return out


# ----------------------------------------------------------------- app

def create_app(cfg: PipelineConfig | None = None):
    """Build a Flask app bound to cfg."""
    try:
        from flask import (
            Flask,
            abort,
            request,
            send_file,
            send_from_directory,
        )
        from werkzeug.utils import secure_filename
    except ImportError as e:
        raise ImportError(
            "Flask is required for the web app. Install with: pip install flask"
        ) from e

    if cfg is None:
        cfg = PipelineConfig()

    cfg.ensure_dirs()

    web_root = cfg.output_dir / "web"
    web_root.mkdir(parents=True, exist_ok=True)

    app = Flask("docbank.webapp")
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

    @app.route("/", methods=["GET"])
    def index():
        return _INDEX_HTML

    @app.route("/upload", methods=["POST"])
    def upload():
        files = request.files.getlist("images")
        files = [f for f in files if f and f.filename]

        if not files:
            return ("No files uploaded.", 400)

        def _ffloat(name, default):
            try:
                return float(request.form.get(name, default))
            except (TypeError, ValueError):
                return default

        def _fint(name, default):
            try:
                return int(float(request.form.get(name, default)))
            except (TypeError, ValueError):
                return default

        def _opt_int(name):
            raw = request.form.get(name, "")

            if raw is None or str(raw).strip() == "":
                return None

            try:
                v = int(float(raw))
                return v if v > 0 else None
            except (TypeError, ValueError):
                return None

        conf_v = max(0.05, min(0.95, _ffloat("conf", 0.25)))
        min_ocr_conf = max(0.0, min(1.0, _ffloat("min_ocr_conf", 0.30)))
        min_ocr_area = max(0, _fint("min_ocr_area", 600))
        min_formula_area = max(0, _fint("min_formula_area", 2000)) or None
        max_formulas = _opt_int("max_formulas")

        do_text = "text_ocr" in request.form
        do_formula = "formula_ocr" in request.form
        use_cache = "use_cache" in request.form

        pdf_mode = request.form.get("pdf_mode", "searchable")
        if pdf_mode not in {"searchable", "reconstructed", "both"}:
            pdf_mode = "searchable"

        job_id = uuid.uuid4().hex[:12]

        job_dir = web_root / job_id
        inputs_dir = job_dir / "inputs"

        inputs_dir.mkdir(parents=True, exist_ok=True)

        saved: list[Path] = []

        for f in files:
            name = secure_filename(f.filename or "")

            if not name:
                continue

            ext = Path(name).suffix.lower()

            if ext not in ALLOWED_EXTS:
                continue

            target = inputs_dir / name
            f.save(target)

            if ext in PDF_EXTS:
                try:
                    pages = _explode_pdf(target, inputs_dir)
                except ImportError as e:
                    shutil.rmtree(job_dir, ignore_errors=True)
                    return (str(e), 500)
                except Exception as e:
                    log.exception("PDF rendering failed for %s", name)
                    return (f"Failed to render PDF {name}: {e}", 400)

                saved.extend(pages)
            else:
                saved.append(target)

        if not saved:
            shutil.rmtree(job_dir, ignore_errors=True)
            return ("No valid image or PDF files.", 400)

        log.info(
            "Job %s: %d page image(s), conf=%.2f, text=%s, formula=%s, pdf_mode=%s",
            job_id,
            len(saved),
            conf_v,
            do_text,
            do_formula,
            pdf_mode,
        )

        try:
            pages, elapsed = _process_uploads(
                cfg,
                job_dir,
                saved,
                conf=conf_v,
                do_text=do_text,
                do_formula=do_formula,
                min_ocr_conf=min_ocr_conf,
                min_ocr_area=min_ocr_area,
                min_formula_area=min_formula_area,
                max_formulas=max_formulas,
                use_cache=use_cache,
            )
        except Exception as e:
            log.exception("Job %s failed", job_id)
            return (f"Pipeline error: {e}", 500)

        result_payload = {
            "job_id": job_id,
            "elapsed_seconds": round(elapsed, 2),
            "pages": pages,
        }
        #
        (job_dir / "results.json").write_text(
            json.dumps(result_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        
        # Build requested PDF output.
        try:
            all_dets = []

            for p in pages:
                page_dets = list(p.get("detections") or [])

                if page_dets:
                    all_dets.extend(page_dets)
                else:
                    # Placeholder so PDF builders still create this page
                    # even if YOLO found no detections on it.
                    all_dets.append(
                        {
                            "image": p["image"],
                            "bbox": [0, 0, 1, 1],
                            "class_name": "text",
                            "confidence": 0.0,
                            "recognized": "",
                            "crop_path": None,
                        }
                    )


            if pdf_mode == "searchable":
                from .to_pdf import detections_to_searchable_pdf

                pdf_path = job_dir / "searchable_document.pdf"
                detections_to_searchable_pdf(all_dets, pdf_path)

                return send_file(
                    pdf_path,
                    as_attachment=True,
                    download_name="searchable_document.pdf",
                    mimetype="application/pdf",
                )

            if pdf_mode == "reconstructed":
                from .reconstruct import detections_to_reconstructed_pdf

                pdf_path = job_dir / "reconstructed_document.pdf"
                detections_to_reconstructed_pdf(all_dets, pdf_path)

                return send_file(
                    pdf_path,
                    as_attachment=True,
                    download_name="reconstructed_document.pdf",
                    mimetype="application/pdf",
                )

            # Both versions.
            from .to_pdf import detections_to_searchable_pdf
            from .reconstruct import detections_to_reconstructed_pdf

            searchable_path = job_dir / "searchable_document.pdf"
            reconstructed_path = job_dir / "reconstructed_document.pdf"
            zip_path = job_dir / "pdf_outputs.zip"

            detections_to_searchable_pdf(all_dets, searchable_path)
            detections_to_reconstructed_pdf(all_dets, reconstructed_path)

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(searchable_path, searchable_path.name)
                zf.write(reconstructed_path, reconstructed_path.name)

            return send_file(
                zip_path,
                as_attachment=True,
                download_name="pdf_outputs.zip",
                mimetype="application/zip",
            )

        except Exception as e:
            log.exception("PDF build failed for job %s", job_id)
            return (f"Recognition succeeded but PDF build failed: {e}", 500)

    @app.route("/jobs/<job_id>/results", methods=["GET"])
    @app.route("/jobs/<job_id>", methods=["GET"])
    def results(job_id: str):
        job_dir = web_root / secure_filename(job_id)
        rj = job_dir / "results.json"

        if not rj.is_file():
            abort(404)

        payload = json.loads(rj.read_text(encoding="utf-8"))

        html_doc = _render_results(
            job_id,
            job_dir,
            payload["pages"],
            payload.get("elapsed_seconds", 0),
        )

        return html_doc

    @app.route("/jobs/<job_id>/results.json")
    def results_json(job_id: str):
        job_dir = web_root / secure_filename(job_id)
        rj = job_dir / "results.json"

        if not rj.is_file():
            abort(404)

        return send_file(
            rj,
            mimetype="application/json",
            as_attachment=True,
            download_name=f"docbank_{job_id}.json",
        )

    @app.route("/jobs/<job_id>/<path:rel>")
    def job_file(job_id: str, rel: str):
        job_dir = web_root / secure_filename(job_id)

        if not job_dir.is_dir():
            abort(404)

        return send_from_directory(job_dir, rel)

    return app


def serve(
    cfg: PipelineConfig | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 5000,
    debug: bool = False,
) -> None:
    """Convenience entry-point for the CLI."""
    app = create_app(cfg)

    log.info(
        "Starting webapp on http://%s:%d",
        host,
        port,
    )

    if not debug:
        try:
            from waitress import serve as waitress_serve

            log.info("Using waitress with 8 worker threads.")
            waitress_serve(app, host=host, port=port, threads=8)
            return

        except ImportError:
            log.info(
                "waitress not installed; using Flask dev server in threaded mode."
            )

    app.run(host=host, port=port, debug=debug, threaded=True)

