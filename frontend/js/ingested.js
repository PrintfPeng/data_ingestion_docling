/* frontend/js/ingested.js — Standalone Ingested Data page */

const grid         = document.getElementById("grid");
const emptyState   = document.getElementById("emptyState");
const searchBox    = document.getElementById("searchBox");
const refreshBtn   = document.getElementById("refreshBtn");
const statusLabel  = document.getElementById("statusLabel");
const totalCount   = document.getElementById("totalCount");

let allDocs = [];

function escapeHtml(s) {
    return String(s).replace(/[<>&"']/g, c => ({
        '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&#39;'
    }[c]));
}

function render(docs) {
    grid.innerHTML = "";
    if (!docs.length) {
        emptyState.classList.remove("hidden");
        return;
    }
    emptyState.classList.add("hidden");

    docs.forEach(doc => {
        const name = doc.name || doc.id || "Untitled";
        const id = doc.id || "";
        const status = (doc.status || "ready").toLowerCase();
        const statusLabel = { ready: "READY", processing: "PROCESSING", error: "ERROR" }[status] || status.toUpperCase();
        const pages = doc.pages != null ? `${doc.pages} หน้า` : "";
        const chunks = doc.chunks != null ? `${doc.chunks} chunks` : "";
        const snippet = doc.snippet || "";

        const card = document.createElement("div");
        card.className = "bg-white border border-slate-200 rounded-xl p-4 shadow-sm hover:shadow-md hover:border-brand-300 transition group flex flex-col";
        card.innerHTML = `
            <div class="flex items-start gap-3 mb-3">
                <div class="w-11 h-11 rounded-xl bg-brand-50 text-brand-600 flex items-center justify-center flex-none group-hover:bg-brand-100 transition">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                        <line x1="9" y1="13" x2="15" y2="13"/>
                        <line x1="9" y1="17" x2="15" y2="17"/>
                    </svg>
                </div>
                <div class="flex-1 min-w-0">
                    <div class="font-semibold text-slate-800 text-sm truncate" title="${escapeHtml(name)}">${escapeHtml(name)}</div>
                    <div class="text-[11px] text-slate-400 mt-0.5 font-mono truncate" title="${escapeHtml(id)}">${escapeHtml(id)}</div>
                </div>
            </div>
            <div class="flex items-center gap-2 flex-wrap mb-2">
                <span class="ingested-status status-${status}">${statusLabel}</span>
                ${pages  ? `<span class="text-[11px] text-slate-500">📄 ${escapeHtml(pages)}</span>`   : ""}
                ${chunks ? `<span class="text-[11px] text-slate-500">🧩 ${escapeHtml(chunks)}</span>` : ""}
            </div>
            ${snippet ? `<div class="text-xs text-slate-500 mt-1 line-clamp-3 border-t border-slate-100 pt-2">${escapeHtml(snippet)}</div>` : ""}
            <div class="mt-3 pt-3 border-t border-slate-100 flex gap-2">
                <a href="index.html?doc=${encodeURIComponent(id)}" class="flex-1 text-center text-[11px] font-semibold text-brand-600 bg-brand-50 hover:bg-brand-100 rounded-lg py-1.5 transition">💬 Ask about this</a>
            </div>
        `;
        grid.appendChild(card);
    });
}

function applyFilter() {
    const q = (searchBox.value || "").trim().toLowerCase();
    const filtered = q
        ? allDocs.filter(d => (d.name || "").toLowerCase().includes(q) || (d.id || "").toLowerCase().includes(q))
        : allDocs;
    render(filtered);
}

async function fetchDocs() {
    statusLabel.textContent = "Loading…";
    statusLabel.style.color = "#facc15";
    try {
        const res = await fetch("/documents");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        allDocs = data.documents || [];
        totalCount.textContent = allDocs.length;
        applyFilter();
        statusLabel.textContent = `✓ Loaded ${allDocs.length} document${allDocs.length !== 1 ? "s" : ""}`;
        statusLabel.style.color = "#16a34a";
    } catch (err) {
        statusLabel.textContent = `✗ ${err.message}`;
        statusLabel.style.color = "#dc2626";
        emptyState.classList.remove("hidden");
        emptyState.innerHTML = `
            <div class="text-5xl mb-3">⚠️</div>
            <div class="text-sm font-medium text-red-500">Failed to load documents</div>
            <div class="text-xs mt-2 text-slate-500">${escapeHtml(err.message)}</div>
        `;
    }
}

refreshBtn.addEventListener("click", fetchDocs);
searchBox.addEventListener("input", applyFilter);

fetchDocs();
