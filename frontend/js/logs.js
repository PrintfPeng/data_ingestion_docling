/* frontend/js/logs.js — Standalone Logs page */

// --- Auth helper (reads API key from localStorage) ---
const API_KEY_STORAGE = "app_api_key";
function getAuthHeader() {
    const key = localStorage.getItem(API_KEY_STORAGE);
    return key ? { Authorization: `Bearer ${key}` } : {};
}

const logTerminal    = document.getElementById("logTerminal");
const autoRefreshChk = document.getElementById("autoRefreshChk");
const filterLevel    = document.getElementById("filterLevel");
const refreshBtn     = document.getElementById("refreshBtn");
const clearBtn       = document.getElementById("clearBtn");
const scrollBottomBtn= document.getElementById("scrollBottomBtn");
const statusLabel    = document.getElementById("statusLabel");
const countLabel     = document.getElementById("countLabel");

const REFRESH_INTERVAL_MS = 5000;
const HISTORY_LIMIT = 200;
let refreshTimer = null;
let currentEntries = [];
let currentFilter = "all";
let userScrolledUp = false;

function escapeHtml(s) {
    return String(s).replace(/[<>&"']/g, c => ({
        '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&#39;'
    }[c]));
}

function formatTime(ts) {
    if (!ts) return "--:--:--";
    const d = new Date(ts);
    if (isNaN(d.getTime())) return String(ts).substring(11, 19) || "--:--:--";
    return d.toTimeString().substring(0, 8);
}

function classifyLevel(entry) {
    if (entry.error) return "error";
    const intent = (entry.intent || "").toLowerCase();
    if (intent.includes("error") || intent.includes("fail")) return "error";
    if (intent.includes("warn")) return "warn";
    if (entry.answer && entry.answer.length > 0) return "success";
    return "info";
}

function renderLog(entries) {
    currentEntries = entries;
    countLabel.textContent = entries.length;

    const filtered = currentFilter === "all"
        ? entries
        : entries.filter(e => classifyLevel(e) === currentFilter);

    if (filtered.length === 0) {
        logTerminal.innerHTML = `
            <div class="log-line log-info">
                <span class="log-time">--:--:--</span>
                <span class="log-tag">INFO</span>
                <span class="log-text">No log entries${currentFilter !== "all" ? ` at level "${currentFilter}"` : ""}.</span>
            </div>`;
        return;
    }

    const html = filtered.map(e => {
        const level = classifyLevel(e);
        const time = formatTime(e.ts);
        const mode = e.mode || "auto";
        const query = (e.query || "").replace(/\s+/g, " ").substring(0, 200);
        const answer = (e.answer || "").replace(/\s+/g, " ").substring(0, 300);
        const intent = e.intent ? ` intent=${escapeHtml(e.intent)}` : "";
        const sources = (Array.isArray(e.sources) ? e.sources.length : 0);
        const src = sources ? ` sources=${sources}` : "";
        return `
            <div class="log-line log-${level}">
                <span class="log-time">[${time}]</span>
                <span class="log-tag">${level.toUpperCase()}</span>
                <span class="log-text"><span style="color:#64748b">mode=${escapeHtml(mode)}${intent}${src}</span>
Q: ${escapeHtml(query)}
A: ${escapeHtml(answer)}</span>
            </div>`;
    }).join("");

    logTerminal.innerHTML = html;

    if (!userScrolledUp) {
        logTerminal.scrollTop = logTerminal.scrollHeight;
    }
}

async function fetchLogs() {
    statusLabel.textContent = "● loading…";
    statusLabel.style.color = "#facc15";
    try {
        const res = await fetch(`/history?limit=${HISTORY_LIMIT}`, { headers: { ...getAuthHeader() } });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        let data = await res.json();
        // Server-side may return newest last or first — normalize to chronological (oldest first)
        // If timestamps present, sort ascending
        if (Array.isArray(data) && data.length && data[0].ts) {
            data.sort((a, b) => String(a.ts).localeCompare(String(b.ts)));
        }
        renderLog(data || []);
        statusLabel.textContent = `● live · updated ${new Date().toTimeString().substring(0,8)}`;
        statusLabel.style.color = "#4ade80";
    } catch (err) {
        statusLabel.textContent = `● error: ${err.message}`;
        statusLabel.style.color = "#f87171";
        logTerminal.innerHTML = `
            <div class="log-line log-error">
                <span class="log-time">${new Date().toTimeString().substring(0,8)}</span>
                <span class="log-tag">ERROR</span>
                <span class="log-text">Failed to fetch /history: ${escapeHtml(err.message)}</span>
            </div>`;
    }
}

function startAutoRefresh() {
    stopAutoRefresh();
    refreshTimer = setInterval(fetchLogs, REFRESH_INTERVAL_MS);
}

function stopAutoRefresh() {
    if (refreshTimer) {
        clearInterval(refreshTimer);
        refreshTimer = null;
    }
}

// Detect user scroll — if they scroll up, stop autoscroll to bottom
logTerminal.addEventListener("scroll", () => {
    const nearBottom = logTerminal.scrollHeight - logTerminal.scrollTop - logTerminal.clientHeight < 30;
    userScrolledUp = !nearBottom;
});

autoRefreshChk.addEventListener("change", () => {
    if (autoRefreshChk.checked) {
        startAutoRefresh();
        fetchLogs();
    } else {
        stopAutoRefresh();
    }
});

filterLevel.addEventListener("change", () => {
    currentFilter = filterLevel.value;
    renderLog(currentEntries);
});

refreshBtn.addEventListener("click", fetchLogs);

clearBtn.addEventListener("click", () => {
    currentEntries = [];
    countLabel.textContent = "0";
    logTerminal.innerHTML = `
        <div class="log-line log-info">
            <span class="log-time">${new Date().toTimeString().substring(0,8)}</span>
            <span class="log-tag">INFO</span>
            <span class="log-text">View cleared. Next auto-refresh will reload from /history.</span>
        </div>`;
});

scrollBottomBtn.addEventListener("click", () => {
    userScrolledUp = false;
    logTerminal.scrollTop = logTerminal.scrollHeight;
});

// Init
fetchLogs();
startAutoRefresh();
