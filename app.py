import os
import json
import shutil
import base64
import time
import csv
import uuid
import threading
import zipfile
import io
from dataclasses import dataclass, field
from pathlib import Path
from io import BytesIO
from typing import Dict, Optional, List, Any, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from PIL import Image, ImageEnhance, ImageFilter
from groq import Groq
from pdf2image import convert_from_path

try:
    from pdf2image import pdfinfo_from_path
except Exception:
    pdfinfo_from_path = None


# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# -----------------------------
# Job state (in-memory for running jobs)
# -----------------------------
@dataclass
class JobState:
    job_id: str
    status: str = "queued"  # queued|running|done|error|cancelled
    progress_pct: int = 0
    message: str = "Queued"
    error: Optional[str] = None
    cancel_requested: bool = False

    total_pages: int = 0
    processed_pages: int = 0

    job_dir: Path = None
    input_dir: Path = None
    output_dir: Path = None
    csv_path: Optional[Path] = None

    rows: List[Dict[str, Any]] = field(default_factory=list)


JOBS: Dict[str, JobState] = {}
LOCK = threading.Lock()


def safe_pct(x: float) -> int:
    if x < 0:
        return 0
    if x > 100:
        return 100
    return int(x)


# -----------------------------
# Cleaning helpers (remove ONLY spaces)
# -----------------------------
def clean_id(v: Any) -> str:
    """
    - Joins wrapped lines
    - Removes ONLY normal spaces ' '
    - Keeps hyphen, slash, underscore, dots, etc.
    """
    s = str(v or "")
    s = s.replace("\r", "").replace("\n", "")
    s = s.replace(" ", "")
    return s.strip()

def make_unique_job_id_from_pdf(pdf_filename: str) -> str:
    """
    Creates a safe folder name from PDF filename.
    If it already exists, appends _2, _3, ...
    """
    base = Path(pdf_filename).stem.strip()
    # sanitize to safe folder name
    safe = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in base)
    safe = safe or "job"

    job_id = safe
    n = 2
    while (JOBS_DIR / job_id).exists():
        job_id = f"{safe}_{n}"
        n += 1
    return job_id

def normalize_invoice_key(invoice_no: str) -> str:
    return clean_id(invoice_no)


# -----------------------------
# Image bytes for model
# -----------------------------
def page_to_model_bytes(page: Image.Image, max_w: int = 3600) -> bytes:
    img = page.convert("RGB")
    w, h = img.size
    if w > max_w:
        new_h = int(h * (max_w / w))
        img = img.resize((max_w, new_h), Image.LANCZOS)

    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3))

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90, optimize=True)
    return buf.getvalue()


# -----------------------------
# Groq analysis (model does OCR/extraction)
# -----------------------------
def get_groq_analysis(image_bytes: bytes, model_name: str) -> dict:
    if client is None:
        raise RuntimeError("GROQ_API_KEY not set.")

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    prompt = r"""
You are doing OCR + structured extraction from ONE document page image.

Return ONLY valid JSON with EXACT keys:
{
  "category": "Tax Invoice" | "Credit Note" | "Others",
  "has_stamp": true/false,
  "stamp_details": "",
  "invoice_no": "",
  "receipt_no": "",
  "full_text": ""
}

Rules:
- full_text: include ALL visible text with line breaks.
- has_stamp MUST be true if ANY stamp/seal/ink impression exists anywhere (even faint/partial).
- category should be based on heading when possible:
  - "Tax Invoice" if it clearly says TAX INVOICE
  - "Credit Note" if it clearly says CREDIT NOTE / TAX CREDIT NOTE
  - otherwise "Others"

Field extraction:
- invoice_no: extract full invoice number including suffixes like -SPICES-PROMO (may wrap to next line).
- receipt_no: extract receipt number if present.
- If receipt_no is non-empty, set invoice_no = "" (receipt pages don't have invoice numbers).

Multi-line IDs:
- Values can continue on next line(s). Append continuation lines that look like part of the ID.
- Do NOT stop just because DATE appears on the right side of the same row.

Return ONLY the JSON object.
"""

    completion = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        }],
        response_format={"type": "json_object"},
    )
    return json.loads(completion.choices[0].message.content)


# -----------------------------
# Category + stamp reconciliation for duplicate invoices
# -----------------------------
def cat_score(cat: str) -> int:
    c = (cat or "").strip().lower()
    if "credit" in c:
        return 2
    if "invoice" in c:
        return 1
    return 0


def best_category(cats: List[str]) -> str:
    # Credit Note > Tax Invoice > Others
    best = "Others"
    best_s = -1
    for c in cats:
        s = cat_score(c)
        if s > best_s:
            best_s = s
            best = "Credit Note" if s == 2 else ("Tax Invoice" if s == 1 else "Others")
    return best


def apply_invoice_reconciliation(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Applies your rule:
    - If invoice_no duplicates exist:
      - has_stamp = True for all if any is True
      - category = Credit Note if any is Credit Note else Tax Invoice if any is Tax Invoice else Others
    Does NOT dedupe; just fixes fields.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        inv = normalize_invoice_key(r.get("invoice_no", ""))
        if not inv:
            continue
        groups.setdefault(inv, []).append(r)

    for inv, grp in groups.items():
        if len(grp) < 2:
            continue

        final_stamp = any(bool(x.get("has_stamp")) for x in grp)
        final_cat = best_category([x.get("category", "Others") for x in grp])

        # keep one stamp_details if any
        details = ""
        for x in grp:
            d = (x.get("stamp_details") or "").strip()
            if d:
                details = d
                break

        for x in grp:
            x["has_stamp"] = final_stamp
            x["category"] = final_cat
            if final_stamp and not (x.get("stamp_details") or "").strip():
                x["stamp_details"] = details

    return rows


def dedupe_rows_for_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For UI table:
    - If invoice_no duplicates exist: show ONLY one row
    - Keep representative row (smallest page), but keep reconciled stamp/category
    """
    # first reconcile stamps/categories across duplicates
    rows = apply_invoice_reconciliation(rows)

    by_invoice: Dict[str, List[Dict[str, Any]]] = {}
    singles: List[Dict[str, Any]] = []

    for r in rows:
        inv = normalize_invoice_key(r.get("invoice_no", ""))
        if inv:
            by_invoice.setdefault(inv, []).append(r)
        else:
            singles.append(r)

    deduped: List[Dict[str, Any]] = []

    for inv, grp in by_invoice.items():
        # pick smallest page as representative
        grp_sorted = sorted(grp, key=lambda x: int(x.get("page") or x.get("id") or 10**9))
        rep = dict(grp_sorted[0])
        # optional: store pages list for debugging (not shown)
        rep["_pages"] = ",".join(str(int(x.get("page") or x.get("id") or 0)) for x in grp_sorted)
        deduped.append(rep)

    deduped.extend(singles)

    # stable order: by source_pdf then page
    deduped.sort(key=lambda x: (x.get("source_pdf", ""), int(x.get("page") or x.get("id") or 0)))
    return deduped


# -----------------------------
# File locating (robust even if folder wrong)
# -----------------------------
def find_page_png(job_id: str, source_pdf: str, page: int) -> Optional[Path]:
    job_dir = JOBS_DIR / job_id
    output_dir = job_dir / "output"
    pdf_stem = Path(source_pdf).stem
    base = output_dir / pdf_stem
    # search anywhere under this pdf folder
    candidates = list(base.rglob(f"page_{page}.png"))
    return candidates[0] if candidates else None


# -----------------------------
# Disk helpers (completed jobs)
# -----------------------------
def iter_completed_jobs_on_disk():
    if not JOBS_DIR.exists():
        return
    for job_dir in sorted(JOBS_DIR.iterdir(), reverse=True):
        if not job_dir.is_dir():
            continue
        job_id = job_dir.name
        csv_path = job_dir / "classification_log.csv"
        output_dir = job_dir / "output"
        if csv_path.exists() and output_dir.exists():
            yield job_id, job_dir, csv_path, output_dir


def read_all_rows_raw() -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for job_id, _job_dir, csv_path, _output_dir in iter_completed_jobs_on_disk():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                hs = str(r.get("has_stamp", "")).lower() in ("true", "1", "yes")
                try:
                    page = int(r.get("page") or r.get("id") or 0)
                except Exception:
                    page = 0

                row = dict(r)
                row["job_id"] = job_id
                row["has_stamp"] = hs
                row["page"] = page
                # normalize IDs (remove spaces)
                row["invoice_no"] = clean_id(row.get("invoice_no", ""))
                row["receipt_no"] = clean_id(row.get("receipt_no", ""))
                merged.append(row)
    return merged


def build_image_url(job_id: str, row: Dict[str, Any]) -> str:
    hs = "true" if bool(row.get("has_stamp")) else "false"
    page = row.get("page") or row.get("id") or ""
    pdf = row.get("source_pdf") or ""
    cat = row.get("category") or "Others"
    return f"/api/job/{job_id}/image_by_page?pdf={pdf}&page={page}&category={cat}&has_stamp={hs}"


def build_combined_csv_from_all_jobs_reconciled() -> bytes:
    rows = read_all_rows_raw()
    # apply stamp/category reconciliation (but keep all pages)
    rows = apply_invoice_reconciliation(rows)

    headers = [
        "job_id",
        "source_pdf",
        "id",
        "page",
        "category",
        "has_stamp",
        "stamp_details",
        "invoice_no",
        "receipt_no",
        "full_text",
    ]

    out = BytesIO()
    text_stream = io.TextIOWrapper(out, encoding="utf-8", newline="")
    writer = csv.DictWriter(text_stream, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    text_stream.flush()
    return out.getvalue()


def build_zip_by_category_all_jobs(category_name: str) -> bytes:
    """
    Creates a ZIP with:
      stamped/<job_id>/<pdf_stem>/page_X.png
      unstamped/<job_id>/<pdf_stem>/page_X.png

    and applies your rule:
    - if invoice duplicates exist and any stamped -> ALL stamped (for those pages)
    - category also reconciled (Credit Note > Tax Invoice > Others)
    """
    rows = read_all_rows_raw()
    rows = apply_invoice_reconciliation(rows)

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for r in rows:
            if (r.get("category") or "Others") != category_name:
                continue

            job_id = r.get("job_id", "")
            source_pdf = r.get("source_pdf", "")
            page = int(r.get("page") or 0)
            if not job_id or not source_pdf or page <= 0:
                continue

            img_path = find_page_png(job_id, source_pdf, page)
            if not img_path or not img_path.exists():
                continue

            stamped = bool(r.get("has_stamp"))
            pdf_stem = Path(source_pdf).stem
            arc = Path("stamped" if stamped else "unstamped") / pdf_stem / f"page_{page}.png"
            zf.write(img_path, arcname=str(arc))

    return buf.getvalue()


def build_zip_all_folders_all_jobs() -> bytes:
    """
    ZIP everything, reconciled, with:
      Tax Invoice/stamped/<job_id>/<pdf_stem>/page_X.png
      Credit Note/unstamped/<job_id>/<pdf_stem>/page_X.png
      Others/...
    """
    rows = read_all_rows_raw()
    rows = apply_invoice_reconciliation(rows)

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for r in rows:
            job_id = r.get("job_id", "")
            source_pdf = r.get("source_pdf", "")
            page = int(r.get("page") or 0)
            cat = (r.get("category") or "Others")
            if cat not in ("Tax Invoice", "Credit Note", "Others"):
                cat = "Others"
            if not job_id or not source_pdf or page <= 0:
                continue

            img_path = find_page_png(job_id, source_pdf, page)
            if not img_path or not img_path.exists():
                continue

            stamped = bool(r.get("has_stamp"))
            pdf_stem = Path(source_pdf).stem
            arc = Path(cat) / ("stamped" if stamped else "unstamped") / pdf_stem / f"page_{page}.png"
            zf.write(img_path, arcname=str(arc))

    return buf.getvalue()


# -----------------------------
# Processing pipeline (per job)
# -----------------------------
def process_job(job: JobState, dpi: int, sleep_sec: float, model_name: str):
    try:
        job.status = "running"
        job.message = "Starting…"
        job.progress_pct = 1

        all_rows: List[Dict[str, Any]] = []

        input_pdfs = sorted(job.input_dir.glob("*.pdf"))
        if not input_pdfs:
            raise RuntimeError("No PDFs found in upload.")

        total_pages_est = 0
        if pdfinfo_from_path is not None:
            for pdf in input_pdfs:
                try:
                    info = pdfinfo_from_path(str(pdf))
                    total_pages_est += int(info.get("Pages", 0))
                except Exception:
                    pass
        job.total_pages = total_pages_est if total_pages_est > 0 else 0

        for pdf_file in input_pdfs:
            if job.cancel_requested:
                job.status = "cancelled"
                job.message = "Cancelled"
                return

            pdf_output_dir = job.output_dir / pdf_file.stem

            # Folder structure
            for cat in ["Tax Invoice", "Credit Note", "Others"]:
                for s in ["stamped", "unstamped"]:
                    (pdf_output_dir / cat / s).mkdir(parents=True, exist_ok=True)

            pages = convert_from_path(str(pdf_file), dpi=dpi)
            if job.total_pages == 0:
                job.total_pages += len(pages)

            for i, page in enumerate(pages):
                if job.cancel_requested:
                    job.status = "cancelled"
                    job.message = "Cancelled"
                    return

                page_num = i + 1
                job.message = f"{pdf_file.name}: page {page_num}/{len(pages)}"
                if job.total_pages > 0:
                    job.progress_pct = safe_pct((job.processed_pages / job.total_pages) * 100)

                try:
                    analysis = get_groq_analysis(page_to_model_bytes(page), model_name=model_name)
                except Exception:
                    analysis = {
                        "category": "Others",
                        "has_stamp": False,
                        "stamp_details": "",
                        "invoice_no": "",
                        "receipt_no": "",
                        "full_text": ""
                    }

                category = (analysis.get("category") or "Others").strip()
                if category not in {"Tax Invoice", "Credit Note", "Others"}:
                    category = "Others"

                has_stamp = bool(analysis.get("has_stamp", False))
                stamp_details = (analysis.get("stamp_details") or "").strip()

                invoice_no = clean_id(analysis.get("invoice_no"))
                receipt_no = clean_id(analysis.get("receipt_no"))

                # Save image initial location
                target_dir = pdf_output_dir / category / ("stamped" if has_stamp else "unstamped")
                img_path = target_dir / f"page_{page_num}.png"
                page.save(str(img_path), format="PNG")

                row = {
                    "source_pdf": pdf_file.name,
                    "id": str(page_num),
                    "page": page_num,
                    "category": category,
                    "has_stamp": has_stamp,
                    "stamp_details": stamp_details,
                    "invoice_no": invoice_no,
                    "receipt_no": receipt_no,
                    "full_text": analysis.get("full_text") or "",
                }
                all_rows.append(row)

                job.processed_pages += 1
                if job.total_pages > 0:
                    job.progress_pct = safe_pct((job.processed_pages / job.total_pages) * 100)

                if sleep_sec > 0:
                    time.sleep(sleep_sec)

            # move original pdf under its output folder
            pdf_output_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(pdf_file), str(pdf_output_dir / pdf_file.name))
            except Exception:
                pass

        # IMPORTANT: reconcile stamp/category across duplicate invoices (in-job)
        all_rows = apply_invoice_reconciliation(all_rows)

        job.rows = all_rows

        # Write CSV (per job)
        job.csv_path = job.job_dir / "classification_log.csv"
        headers = [
            "source_pdf",
            "id",
            "page",
            "category",
            "has_stamp",
            "stamp_details",
            "invoice_no",
            "receipt_no",
            "full_text",
        ]
        with open(job.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            for r in all_rows:
                writer.writerow(r)

        job.progress_pct = 100
        job.status = "done"
        job.message = "Complete"

    except Exception as e:
        job.status = "error"
        job.error = str(e)
        job.message = "Error"
        job.progress_pct = safe_pct(job.progress_pct)


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Invoice Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/process")
async def start_process(
    pdfs: List[UploadFile] = File(...),
    dpi: int = Form(300),
    sleep_sec: float = Form(0),
    model: str = Form("meta-llama/llama-4-scout-17b-16e-instruct"),
):
    if client is None:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set on server.")

    # ✅ Use PDF name as the job folder name (only if single PDF)
    first_pdf_name = None
    for uf in pdfs:
        if uf.filename and uf.filename.lower().endswith(".pdf"):
            first_pdf_name = Path(uf.filename).name
            break

    if not first_pdf_name:
        raise HTTPException(status_code=400, detail="No PDF files uploaded.")

    job_id = make_unique_job_id_from_pdf(first_pdf_name)
    job_dir = JOBS_DIR / job_id
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_any = False
    for uf in pdfs:
        if not uf.filename or not uf.filename.lower().endswith(".pdf"):
            continue
        dest = input_dir / Path(uf.filename).name
        content = await uf.read()
        with open(dest, "wb") as f:
            f.write(content)
        saved_any = True

    if not saved_any:
        raise HTTPException(status_code=400, detail="No PDF files uploaded.")

    job = JobState(job_id=job_id, job_dir=job_dir, input_dir=input_dir, output_dir=output_dir)

    with LOCK:
        JOBS[job_id] = job

    t = threading.Thread(target=process_job, args=(job, int(dpi), float(sleep_sec), str(model)), daemon=True)
    t.start()

    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
def get_job(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)

    if job:
        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress_pct": job.progress_pct,
            "message": job.message,
            "error": job.error,
            "processed_pages": job.processed_pages,
            "total_pages": job.total_pages,
        }

    job_dir = JOBS_DIR / job_id
    csv_path = job_dir / "classification_log.csv"
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    status = "done" if csv_path.exists() else "unknown"
    return {"job_id": job_id, "status": status}


@app.post("/api/job/{job_id}/cancel")
def cancel_job(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.cancel_requested = True
    return {"ok": True}


@app.get("/api/job/{job_id}/image_by_page")
def get_image_by_page(job_id: str, pdf: str, page: int, category: str, has_stamp: bool):
    # robust lookup (ignore category/stamp folder; search anywhere)
    img_path = find_page_png(job_id, pdf, page)
    if not img_path or not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(img_path), media_type="image/png")


# -----------------------------
# Table: ALL jobs (DEDUPED by invoice_no for UI)
# -----------------------------
@app.get("/api/table/all")
def get_all_table_rows():
    rows = read_all_rows_raw()
    rows = dedupe_rows_for_table(rows)

    merged: List[Dict[str, Any]] = []
    for r in rows:
        r = dict(r)
        r["image_url"] = build_image_url(r["job_id"], r)
        merged.append(r)

    return {"rows": merged, "count": len(merged)}


# -----------------------------
# Downloads (GLOBAL)
# -----------------------------
@app.get("/api/downloads")
def global_download_links():
    return {
        "csv_log": "/api/download/all_csv",
        "tax_invoice_zip": "/api/download/tax_invoice",
        "credit_note_zip": "/api/download/credit_note",
        "all_pages_zip": "/api/download/all_folders",
    }


@app.get("/api/download/all_csv")
def download_all_csv():
    data = build_combined_csv_from_all_jobs_reconciled()
    return Response(
        content=data,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="classified_invoices_all_jobs.csv"'},
    )


@app.get("/api/download/all_folders")
def download_all_folders_zip():
    data = build_zip_all_folders_all_jobs()
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="all_jobs_all_folders.zip"'},
    )


@app.get("/api/download/tax_invoice")
def download_all_tax_invoice_zip():
    data = build_zip_by_category_all_jobs("Tax Invoice")
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="tax_invoice_all_jobs.zip"'},
    )


@app.get("/api/download/credit_note")
def download_all_credit_note_zip():
    data = build_zip_by_category_all_jobs("Credit Note")
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="credit_note_all_jobs.zip"'},
    )


# Optional: filtered csv (if you ever re-add button)
@app.post("/api/download/filtered_csv")
def download_filtered_csv(payload: dict = Body(...)):
    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        raise HTTPException(status_code=400, detail="rows must be a list")

    # apply rule: if duplicates in the filtered set, stamp all in those groups
    rows = apply_invoice_reconciliation(rows)

    headers = [
        "job_id",
        "source_pdf",
        "id",
        "page",
        "category",
        "has_stamp",
        "stamp_details",
        "invoice_no",
        "receipt_no",
        "full_text",
    ]

    out = BytesIO()
    text_stream = io.TextIOWrapper(out, encoding="utf-8", newline="")
    writer = csv.DictWriter(text_stream, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    text_stream.flush()

    return Response(
        content=out.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="filtered.csv"'}
    )


# -----------------------------
# Serve frontend (MUST be last)
# -----------------------------
FRONTEND_DIR = (BASE_DIR / "frontend").resolve()
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
