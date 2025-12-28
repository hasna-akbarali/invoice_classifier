let pollTimer = null;
let currentJobId = null;
let rowsCache = [];

const el = (id) => document.getElementById(id);

function setStatus(text) {
  const pill = el("statusPill");
  if (pill) pill.textContent = text || "idle";
}

function setProgress(pct, msg) {
  const bar = el("bar");
  const progressText = el("progressText");
  const p = Math.max(0, Math.min(100, Number(pct || 0)));
  if (bar) bar.style.width = `${p}%`;
  if (progressText) progressText.textContent = msg || "";
}

function fmtBool(b) {
  return b ? "Yes" : "No";
}

function toBool(v) {
  if (typeof v === "boolean") return v;
  const s = String(v ?? "").toLowerCase();
  return s === "true" || s === "1" || s === "yes";
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

/* -----------------------
   Downloads (global)
------------------------ */
function renderDownloads(links) {
  const box = el("downloads");
  if (!box) return;

  box.innerHTML = "";
  if (!links) return;

  const items = [
    ["CSV", links.csv_log],
    ["Tax Invoice ZIP", links.tax_invoice_zip],
    ["Credit Note ZIP", links.credit_note_zip],
    ["All Folders ZIP", links.all_pages_zip],
  ];

  for (const [label, href] of items) {
    if (!href) continue;
    const a = document.createElement("a");
    a.className = "btnLink";
    a.href = href;
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = label;
    box.appendChild(a);
  }
}

async function loadDownloads() {
  const res = await fetch("/api/downloads");
  if (!res.ok) return;
  const links = await res.json();
  renderDownloads(links);
}

/* -----------------------
   Filters
------------------------ */
function passesFilters(r) {
  const q = (el("search")?.value || "").trim().toLowerCase();
  const cat = el("filterCategory")?.value || "";
  const stamp = el("filterStamp")?.value ?? "";

  if (cat && r.category !== cat) return false;
  if (stamp !== "" && String(!!r.has_stamp) !== stamp) return false;

  if (!q) return true;

  const blob = [
    r.job_id, r.source_pdf, r.category,
    r.invoice_no, r.receipt_no,
    r.stamp_details
  ].join(" ").toLowerCase();

  return blob.includes(q);
}

function getFilteredRows() {
  return rowsCache.filter(passesFilters);
}

/* -----------------------
   Table
------------------------ */
function renderTable() {
  const body = el("tbody");
  if (!body) return;

  const filtered = getFilteredRows();
  if (!filtered.length) {
    body.innerHTML = `<tr><td class="muted" colspan="7">No rows to show.</td></tr>`;
    return;
  }

  body.innerHTML = filtered.map((r, idx) => `
    <tr>
      <td title="${escapeHtml(r.source_pdf)}">${escapeHtml(r.source_pdf)}</td>
      <td>${escapeHtml(r.page ?? r.id ?? "")}</td>
      <td><span class="tag">${escapeHtml(r.category || "")}</span></td>
      <td>${fmtBool(!!r.has_stamp)}</td>
      <td>${escapeHtml(r.invoice_no || "")}</td>
      <td>${escapeHtml(r.receipt_no || "")}</td>
      <td><button class="mini" data-idx="${idx}">View</button></td>
    </tr>
  `).join("");

  body.querySelectorAll("button.mini").forEach(btn => {
    btn.addEventListener("click", () => {
      const i = Number(btn.getAttribute("data-idx"));
      openModal(filtered[i]);
    });
  });
}

/* -----------------------
   Modal
------------------------ */
function openModal(r) {
  el("modalTitle").textContent = `Page ${r.page ?? r.id ?? ""}`;
  el("modalMeta").textContent =
    `${r.source_pdf} • ${r.category} • Stamp: ${fmtBool(!!r.has_stamp)} • Job: ${r.job_id || ""}`;

  el("modalImg").src = r.image_url || "";
  el("modalText").textContent = r.full_text || "";
  el("modal").classList.remove("hidden");
}

function closeModal() {
  el("modal").classList.add("hidden");
  el("modalImg").src = "";
  el("modalText").textContent = "";
}

/* -----------------------
   Load ALL rows (first load + after job done)
------------------------ */
async function loadAllRows() {
  setStatus("loading");
  setProgress(10, "Loading all finished jobs…");

  const res = await fetch("/api/table/all");
  if (!res.ok) {
    setStatus("error");
    setProgress(0, "Failed to load /api/table/all");
    rowsCache = [];
    renderTable();
    return;
  }

  const data = await res.json();
  const rows = data.rows || [];

  rowsCache = rows.map(r => ({
    ...r,
    has_stamp: toBool(r.has_stamp),
    page: Number(r.page ?? r.id ?? 0),
  }));

  setStatus("ready");
  setProgress(100, `Loaded ${rowsCache.length} row(s).`);
  renderTable();
}

/* -----------------------
   Upload / poll
------------------------ */
async function startUpload() {
  const files = el("pdfs")?.files;
  if (!files || !files.length) {
    alert("Please select at least one PDF.");
    return;
  }

  const fd = new FormData();
  for (const f of files) fd.append("pdfs", f);

  fd.append("dpi", "300");
  fd.append("sleep_sec", "0");
  fd.append("model", "meta-llama/llama-4-scout-17b-16e-instruct");

  setStatus("starting");
  setProgress(2, "Uploading…");
  el("startBtn").disabled = true;
  el("cancelBtn").disabled = false;

  const res = await fetch("/api/process", { method: "POST", body: fd });
  if (!res.ok) {
    alert(await res.text());
    el("startBtn").disabled = false;
    el("cancelBtn").disabled = true;
    setStatus("idle");
    return;
  }

  const data = await res.json();
  currentJobId = data.job_id;

  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollJob, 1200);
  await pollJob();
}

async function cancelJob() {
  if (!currentJobId) return;
  await fetch(`/api/job/${currentJobId}/cancel`, { method: "POST" });
}

async function pollJob() {
  if (!currentJobId) return;

  const res = await fetch(`/api/job/${currentJobId}`);
  if (!res.ok) return;

  const j = await res.json();
  setStatus(j.status);
  setProgress(j.progress_pct || 0, j.message || "");

  if (j.status === "done") {
    clearInterval(pollTimer);
    pollTimer = null;

    el("startBtn").disabled = false;
    el("cancelBtn").disabled = true;

    await loadDownloads();
    await loadAllRows();
  }

  if (j.status === "error" || j.status === "cancelled") {
    clearInterval(pollTimer);
    pollTimer = null;

    el("startBtn").disabled = false;
    el("cancelBtn").disabled = true;
  }
}

/* -----------------------
   Wire UI
------------------------ */
function wireUI() {
  el("modalClose")?.addEventListener("click", closeModal);
  el("modalBackdrop")?.addEventListener("click", closeModal);

  ["search", "filterCategory", "filterStamp"].forEach(id => {
    el(id)?.addEventListener("input", renderTable);
    el(id)?.addEventListener("change", renderTable);
  });

  el("pdfs")?.addEventListener("change", () => {
    const files = el("pdfs").files;
    const info = el("fileInfo");
    if (!info) return;

    if (!files.length) info.textContent = "No files selected";
    else if (files.length === 1) info.textContent = files[0].name;
    else info.textContent = `${files.length} files selected`;
  });

  el("startBtn")?.addEventListener("click", startUpload);
  el("cancelBtn")?.addEventListener("click", cancelJob);
}

window.addEventListener("load", async () => {
  wireUI();
  await loadDownloads();
  await loadAllRows(); // loads everything from jobs folder
});
