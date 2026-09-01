/* frontend/js/app.js */

const chatMessages = document.getElementById("chatMessages");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const toggleHistoryBtn = document.getElementById("toggleHistoryBtn");
const refreshHistoryBtn = document.getElementById("refreshHistoryBtn");
const closeHistoryBtn = document.getElementById("closeHistoryBtn"); 
const historyPanel = document.getElementById("historyPanel");
const historyBox = document.getElementById("historyBox");
const attachmentInfo = document.getElementById("attachmentInfo");

const modeSelect = document.getElementById("modeSelect");
const docSelect = document.getElementById("docSelect");
const modeSelectMobile = document.getElementById("modeSelectMobile");
const docSelectMobile = document.getElementById("docSelectMobile");

// --- SPA views ---
const landingView       = document.getElementById("landingView");
const chatView          = document.getElementById("chatView");
const backToDocsBtn     = document.getElementById("backToDocsBtn");
const activeDocNameEl   = document.getElementById("activeDocName");

// --- Landing view: upload + docs grid ---
const dropZone          = document.getElementById("dropZone");
const fileInputLanding  = document.getElementById("fileInputLanding");
const chooseFileBtn     = document.getElementById("chooseFileBtn");
const landingDocsGrid   = document.getElementById("landingDocsGrid");
const landingEmpty      = document.getElementById("landingEmpty");
const landingSearch     = document.getElementById("landingSearch");
const refreshDocsBtn    = document.getElementById("refreshDocsBtn");
const docsCountEl       = document.getElementById("docsCount");
const landingUploadStatus = document.getElementById("landingUploadStatus");

let landingDocs = [];   // last fetched documents
let activeDocId = "";    // currently selected doc in chat view

let historyVisible = false;
let attachedFile = null;
// [NEW] เก็บประวัติการสนทนาสำหรับส่งไปให้ Bot จำบริบท
let chatHistory = []; 

// =======================
// 🔐 DOMPurify Configuration
// =======================
const sanitizeConfig = {
    ALLOWED_TAGS: [
        'b','i','em','strong','a','p','br','ul','ol','li',
        'table','thead','tbody','tr','th','td','caption',
        'div','span','details','summary',
        'svg','path','circle',
        'img'
    ],
    ALLOWED_ATTR: [
        'href','target','class','id','style',
        'fill','viewBox','d',
        'stroke','stroke-width','stroke-linecap','stroke-linejoin',
        'open',
        'src', 'alt'
    ],
    ALLOW_DATA_ATTR: false
};

// --- Helper Functions ---

function getMode() {
    if (window.innerWidth < 768) {
        modeSelect.value = modeSelectMobile.value;
        return modeSelectMobile.value;
    }
    modeSelectMobile.value = modeSelect.value;
    return modeSelect.value;
}

function getSelectedDocId() {
    if (window.innerWidth < 768) {
        docSelect.value = docSelectMobile.value;
        return docSelectMobile.value;
    }
    docSelectMobile.value = docSelect.value;
    return docSelect.value;
}

function showHistoryPanel() { 
    historyVisible = true; 
    historyPanel.classList.remove("translate-x-full"); 
    loadHistory(); 
}

function hideHistoryPanel() { 
    historyVisible = false; 
    historyPanel.classList.add("translate-x-full"); 
}

// =======================
// 🎬 SPA view switching + Landing logic
// =======================
function escapeHtml(s) {
    return String(s).replace(/[<>&"']/g, c => ({
        '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&#39;'
    }[c]));
}

function showLandingView() {
    chatView.classList.add("hidden");
    landingView.classList.remove("hidden");
    hideHistoryPanel();
    // Refresh docs list every time we return to landing
    fetchDocuments();
}

function showChatView(docId, docName) {
    activeDocId = docId || "";
    if (activeDocNameEl) {
        activeDocNameEl.textContent = docName || (docId || "All Documents");
        activeDocNameEl.title = docName || (docId || "All Documents");
    }
    // Sync doc selector for existing chat logic
    if (docSelect)       docSelect.value = activeDocId;
    if (docSelectMobile) docSelectMobile.value = activeDocId;

    landingView.classList.add("hidden");
    chatView.classList.remove("hidden");
    // Focus input so user can start typing
    setTimeout(() => chatInput && chatInput.focus(), 100);
}

// -----------------------------------------------------------------
// Landing: render document cards
// -----------------------------------------------------------------
function renderLandingDocs(docs) {
    landingDocsGrid.innerHTML = "";
    if (!docs.length) {
        landingEmpty.classList.remove("hidden");
        return;
    }
    landingEmpty.classList.add("hidden");

    docs.forEach(doc => {
        const name = doc.name || doc.id || "Untitled";
        const id = doc.id || "";
        const pages = doc.pages != null ? `${doc.pages} หน้า` : "";
        const chunks = doc.chunks != null ? `${doc.chunks} chunks` : "";
        const status = (doc.status || "ready").toLowerCase();
        const statusLabel = { ready: "READY", processing: "PROCESSING", error: "ERROR" }[status] || status.toUpperCase();

        const card = document.createElement("button");
        card.type = "button";
        card.className = "doc-card";
        card.setAttribute("data-doc-id", id);
        card.innerHTML = `
            <div class="doc-card-header">
                <div class="doc-card-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                        <line x1="9" y1="13" x2="15" y2="13"/>
                        <line x1="9" y1="17" x2="15" y2="17"/>
                    </svg>
                </div>
                <div class="doc-card-info">
                    <div class="doc-card-name">${escapeHtml(name)}</div>
                    <div class="doc-card-id">${escapeHtml(id)}</div>
                </div>
            </div>
            <div class="doc-card-meta">
                <span class="ingested-status status-${status}">${statusLabel}</span>
                ${pages  ? `<span>📄 ${escapeHtml(pages)}</span>`  : ""}
                ${chunks ? `<span>🧩 ${escapeHtml(chunks)}</span>` : ""}
            </div>
            <div class="doc-card-cta">
                💬 เปิด Chat กับเอกสารนี้
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>
            </div>
        `;
        card.addEventListener("click", () => showChatView(id, name));
        landingDocsGrid.appendChild(card);
    });
}

function applyLandingFilter() {
    const q = (landingSearch?.value || "").trim().toLowerCase();
    const filtered = q
        ? landingDocs.filter(d => (d.name || "").toLowerCase().includes(q) || (d.id || "").toLowerCase().includes(q))
        : landingDocs;
    renderLandingDocs(filtered);
}

// -----------------------------------------------------------------
// Landing: upload zone
// -----------------------------------------------------------------
function setUploadStatus(message, level = "info", withSpinner = false) {
    if (!landingUploadStatus) return;
    landingUploadStatus.classList.remove("hidden");
    landingUploadStatus.innerHTML = `
        <div class="upload-status ${level}">
            ${withSpinner ? '<span class="spinner"></span>' : ''}
            <span>${escapeHtml(message)}</span>
        </div>
    `;
}

function clearUploadStatus() {
    if (!landingUploadStatus) return;
    landingUploadStatus.classList.add("hidden");
    landingUploadStatus.innerHTML = "";
}

/**
 * handleFileUpload(file)
 * Placeholder connect point — currently wires to existing /upload endpoint.
 * Replace body to hook a different backend if needed.
 */
async function handleFileUpload(file) {
    if (!file) return;

    const defaultDocId = file.name.replace(/\.[^.]+$/, "");
    const docId = (prompt("ตั้งชื่อ Doc ID:", defaultDocId) || defaultDocId).trim();
    if (!docId) return;

    dropZone.classList.add("uploading");
    setUploadStatus(`กำลังอัปโหลด "${file.name}" (${Math.round(file.size / 1024)} KB) · OCR อาจใช้เวลาสักครู่…`, "info", true);

    try {
        const res = await uploadFileToBackend(file, docId);
        setUploadStatus(`อัปโหลดสำเร็จ! ${res.page_count || "?"} หน้า · เปิด Chat ให้อัตโนมัติ`, "success");

        // Refresh docs then jump into chat with the new doc active
        await fetchDocuments();
        setTimeout(() => {
            showChatView(docId, docId);
            clearUploadStatus();
        }, 800);
    } catch (err) {
        console.error(err);
        setUploadStatus(`อัปโหลดไม่สำเร็จ: ${err.message}`, "error");
    } finally {
        dropZone.classList.remove("uploading");
        if (fileInputLanding) fileInputLanding.value = "";
    }
}

function renderAttachment() {
    if (!attachedFile) { attachmentInfo.innerHTML = ""; return; }
    attachmentInfo.innerHTML = `
        <div class="inline-flex items-center gap-2 px-3 py-1.5 bg-brand-50 text-brand-700 rounded-lg text-xs font-semibold border border-brand-100 shadow-sm animate-fadeIn">
            <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <span class="truncate max-w-[200px]">${attachedFile.name}</span>
            <button id="removeAttachmentBtn" class="ml-1 p-0.5 hover:bg-white hover:text-red-500 rounded-full transition">✕</button>
        </div>`;
    document.getElementById("removeAttachmentBtn").onclick = () => { 
        attachedFile = null; 
        fileInput.value = ""; 
        renderAttachment(); 
    };
}

// --- Main Chat Logic ---
// [FIX 1] ปรับปรุง fetchDocuments ให้ใช้ Object {id, name} จาก Backend
async function fetchDocuments() {
    try {
        const res = await fetch("/documents", { headers: { ...getAuthHeader() } });
        if (handleAuthResponse(res)) return;
        if (!res.ok) return;
        const data = await res.json();
        const docs = data.documents || [];
        const currentVal = docSelect.value;

        [docSelect, docSelectMobile].forEach(sel => {
            sel.innerHTML = '<option value="">📚 ค้นหาทุกเอกสาร (All)</option>';
            docs.forEach(doc => {
                const opt = document.createElement("option");
                // Backend ส่งมาเป็น Object {id, name} เสมอ
                opt.value = doc.id;  // ใช้ ID จริงเป็น value
                opt.text = `📄 ${doc.name}`;  // ใช้ name แสดงใน Dropdown
                sel.add(opt);
            });

            // คงค่าเดิมที่เลือกไว้ (ถ้ายังมีอยู่)
            const optionExists = Array.from(sel.options).some(o => o.value === currentVal);
            if (optionExists && currentVal) sel.value = currentVal;
        });

        // Sync landing view grid
        landingDocs = docs;
        if (docsCountEl) docsCountEl.textContent = docs.length;
        applyLandingFilter();
    } catch (e) {
        console.error("Failed to load documents:", e);
    }
}

function extractTablesFromHtml(html) {
    const temp = document.createElement("div");
    temp.innerHTML = html;
    
    const tables = Array.from(temp.querySelectorAll("table"));
    const result = tables.map((t, idx) => {
        let title = t.getAttribute("data-title") || "";
        if (!title) {
            const caption = t.querySelector("caption");
            if (caption) title = caption.textContent.trim();
        }
        if (!title) title = `ตารางที่ ${idx + 1}`;
        
        return {
            title: title,
            html: t.outerHTML
        };
    });
    
    tables.forEach(t => t.remove());
    
    return {
        text: temp.innerHTML.trim(),
        tables: result
    };
}

function renderAnswerText(textContainer, text) {
    // 🛡️ ป้องกัน XSS ก่อนแสดงผล
    const cleanHtml = (typeof DOMPurify !== 'undefined') 
        ? DOMPurify.sanitize(text || "", sanitizeConfig)
        : (text || "");
    textContainer.innerHTML = cleanHtml;
}

function renderAnswerTables(tablesContainer, tables) {
    tablesContainer.innerHTML = "";
    if (!tables || tables.length === 0) return;

    tables.forEach((tbl, idx) => {
        const details = document.createElement("details");
        details.className = "border border-slate-200 rounded-lg bg-white shadow-sm overflow-hidden mb-3 group";
        details.open = idx === 0;

        const summary = document.createElement("summary");
        summary.className =
            "cursor-pointer px-4 py-2.5 font-semibold text-sm text-slate-700 bg-slate-50 hover:bg-slate-100 transition flex items-center justify-between select-none list-none";

        const cleanTitle = (typeof DOMPurify !== 'undefined') 
            ? DOMPurify.sanitize(tbl.title || `ตารางที่ ${idx + 1}`, sanitizeConfig)
            : (tbl.title || `ตารางที่ ${idx + 1}`);

        summary.innerHTML = `
            <span class="flex items-center gap-2">
                <svg class="w-4 h-4 text-brand-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                        d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z">
                    </path>
                </svg>
                ${cleanTitle}
            </span>
            <svg class="w-4 h-4 text-slate-400 transform transition-transform group-open:rotate-180 details-chevron"
                fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                    d="M19 9l-7 7-7-7"></path>
            </svg>
        `;

        const body = document.createElement("div");
        body.className = "px-2 py-3 overflow-x-auto bg-white";

        const cleanTableHtml = (typeof DOMPurify !== 'undefined') 
            ? DOMPurify.sanitize(tbl.html || "", sanitizeConfig)
            : (tbl.html || "");
        body.innerHTML = cleanTableHtml;

        details.appendChild(summary);
        details.appendChild(body);
        tablesContainer.appendChild(details);
    });
}

// [FIX 5] Helper function for scrolling to bottom properly
function scrollToBottom() {
    // ใช้ requestAnimationFrame สองครั้งเพื่อให้แน่ใจว่า DOM render เสร็จแล้วจริงๆ
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            const dummy = document.getElementById("scrollDummy");
            if (dummy) {
                dummy.scrollIntoView({ behavior: "smooth", block: "nearest" });
            } else {
                // Fallback ถ้าหา dummy ไม่เจอ
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        });
    });
}

function appendMessage(role, text, options = {}) {
    // [NEW] เก็บประวัติลง Memory
    // ตัดข้อความบางอย่างที่ไม่จำเป็น เช่น "[SHOW_...]" ออกหากต้องการ แต่เก็บ text ดิบก็พอได้
    chatHistory.push({ role: role, content: text });
    if (chatHistory.length > 10) {
        chatHistory = chatHistory.slice(-10); // เก็บแค่ 10 อันล่าสุด
    }

    const isUser = role === "user";
    const wrapper = document.createElement("div");
    wrapper.className = `flex w-full mb-6 msg-animate ${isUser ? "justify-end" : "justify-start"}`;

    const avatar = document.createElement("div");
    avatar.className = `flex-none w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold shadow-sm ${isUser ? "bg-brand-600 text-white order-2 ml-3" : "bg-white border border-slate-200 text-brand-600 order-1 mr-3"}`;
    avatar.innerHTML = isUser 
        ? '<svg class="w-4 h-4" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clip-rule="evenodd" /></svg>' 
        : '<svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>';

    const bubble = document.createElement("div");
    bubble.className = `relative max-w-[85%] md:max-w-[75%] px-5 py-3.5 text-sm leading-relaxed shadow-sm ${isUser ? "bg-brand-600 text-white rounded-2xl rounded-tr-sm order-1" : "bg-white border border-slate-100 text-slate-700 rounded-2xl rounded-tl-sm order-2"}`;

    const textContainer = document.createElement("div");
    textContainer.className = "whitespace-pre-wrap font-sans prose prose-sm max-w-none answer-text-content";
    
    const tablesContainer = document.createElement("div");
    tablesContainer.className = "mt-3 space-y-3 answer-tables-content";
    
    let answerText = text;
    let answerTables = options.tables || [];
    
    // ตรวจสอบทั้ง <table> หรือ &lt;table เผื่อการ escape
    if (!answerTables.length && (text.includes("<table") || text.includes("&lt;table"))) {
        const decodedText = text.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"');
        const extracted = extractTablesFromHtml(decodedText);
        answerText = extracted.text;
        answerTables = extracted.tables;
    }

    answerText = answerText.replace(/\[SHOW_IMAGE:\s*([^\]]+)\]/g, (match, path) => {
        const cleanPath = path.trim();
        // ใส่ class ให้รูปสวยๆ หน่อย
        return `<div class="my-4"><img src="/${cleanPath}" alt="Result Image" class="max-w-full h-auto rounded-lg shadow-md border border-gray-200"></div>`;
    });
    
    renderAnswerText(textContainer, answerText);
    bubble.appendChild(textContainer);
    
    if (answerTables.length > 0) {
        renderAnswerTables(tablesContainer, answerTables);
        bubble.appendChild(tablesContainer);
    }

    if (!isUser && (options.intent || (options.sources && options.sources.length) || options.llmProvider)) {
        const meta = document.createElement("div");
        meta.className = "mt-3 pt-3 border-t border-slate-100 flex flex-col gap-2";

        if (options.intent) {
            meta.innerHTML += `<span class="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-slate-100 text-slate-500 border border-slate-100 w-fit">Intent: ${options.intent}</span>`;
        }

        if (options.sources && options.sources.length) {
            const sourceId = 'source-' + Math.random().toString(36).substr(2, 9);
            const toggleBtn = document.createElement("button");
            toggleBtn.className = "text-[10px] font-bold text-slate-400 hover:text-brand-600 transition flex items-center gap-1 mt-1 bg-slate-50 px-2 py-1 rounded border border-slate-100 w-fit";
            toggleBtn.innerHTML = `<span>▶ แสดงแหล่งที่มา (Sources)</span>`;
            toggleBtn.onclick = () => {
                const el = document.getElementById(sourceId);
                const isHidden = el.classList.contains('hidden');
                el.classList.toggle('hidden');
                toggleBtn.innerHTML = isHidden ? `<span>▼ ซ่อนแหล่งที่มา</span>` : `<span>▶ แสดงแหล่งที่มา (Sources)</span>`;
                scrollToBottom(); // Scroll เมื่อเปิด source
            };
            meta.appendChild(toggleBtn);

            const sourceContainer = document.createElement("div");
            sourceContainer.id = sourceId;
            sourceContainer.className = "hidden mt-2 bg-slate-50 rounded-lg p-2.5 text-xs text-slate-500 border border-slate-100/80 transition-all";
            
            const ul = document.createElement("ul");
            ul.className = "space-y-1.5 pl-1";
            options.sources.forEach(s => {
                const li = document.createElement("li");
                li.className = "flex gap-2 items-start";
                li.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-brand-400 mt-1.5 flex-none"></span>
                                <span class="opacity-90 break-all">
                                    <span class="font-semibold text-slate-700">Doc:</span> ${s.doc_id || "?"} 
                                    <span class="text-slate-300">|</span> 
                                    <span class="font-semibold text-slate-700">Page:</span> ${s.page || "?"} 
                                    <span class="text-slate-300">|</span> 
                                    <span class="italic text-slate-400">${s.source || "text"}</span>
                                </span>`;
                ul.appendChild(li);
            });
            sourceContainer.appendChild(ul);
            meta.appendChild(sourceContainer);
        }

        // Cost + LLM footer (Phase 4)
        if (options.llmProvider || options.costUsd !== undefined) {
            const footer = document.createElement("div");
            footer.className = "answer-cost-footer";
            const isFree = !options.costUsd || options.costUsd === 0;
            const costClass = isFree ? "free" : "paid";
            const costLabel = isFree ? "$0 (local)" : (typeof fmtUsd === "function" ? fmtUsd(options.costUsd) : `$${Number(options.costUsd).toFixed(4)}`);
            const providerLabel = options.llmProvider === "api" ? "⚡ Cloud" : "🔒 Local";
            const modelLabel = options.llmModel ? escapeHtml(options.llmModel) : "";
            footer.innerHTML = `
                <span class="cost-chip">${providerLabel}</span>
                ${modelLabel ? `<span class="cost-chip">${modelLabel}</span>` : ""}
                <span class="cost-chip ${costClass}">${costLabel}</span>
            `;
            meta.appendChild(footer);
        }
        bubble.appendChild(meta);
    }

    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);
    chatMessages.appendChild(wrapper);

    // [FIX 5] จัดการเรื่อง Scroll ให้สุด
    // ลบ dummy เก่าออกก่อนถ้ามี
    const oldDummy = document.getElementById("scrollDummy");
    if (oldDummy) oldDummy.remove();

    // สร้าง dummy ใหม่ไว้ท้ายสุดเสมอ และให้ความสูงเยอะหน่อยกันข้อความจม
    const dummy = document.createElement("div");
    dummy.id = "scrollDummy";
    dummy.style.height = "150px"; // เพิ่มความสูงตรงนี้เพื่อให้เลื่อนขึ้นไปได้อีก
    dummy.style.width = "100%";
    dummy.style.flexShrink = "0";
    chatMessages.appendChild(dummy);

    scrollToBottom();
}

// [FIX 2] ปรับปรุงการส่ง ID ใน uploadFileToBackend (แต่ Logic หลักจะอยู่ที่ sendMessage)
function getSelectedOcrMode() {
    const el = document.querySelector('input[name="ocrMode"]:checked');
    const val = el?.value || "auto";
    return ["auto", "local", "api"].includes(val) ? val : "auto";
}

async function uploadFileToBackend(file, docId, ocrMode) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("doc_id", docId);
    formData.append("doc_type", "");
    formData.append("ocr_mode", ocrMode || getSelectedOcrMode());
    const res = await fetch("/upload", {
        method: "POST",
        headers: { ...getAuthHeader() },
        body: formData,
    });
    if (res.status === 401 || res.status === 403) {
        handleAuthResponse(res);
        throw new Error(res.status === 401 ? "ยังไม่ได้ตั้งค่า API key" : "API key ไม่ถูกต้อง");
    }
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
}

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text && !attachedFile) return;
    
    const mode = getMode();
    // [FIX 3] ดึง ID จริงจาก Dropdown ที่เราแก้ไขใน fetchDocuments แล้ว
    const selectedDocId = getSelectedDocId(); 
    
    const fileToUpload = attachedFile;
    attachedFile = null;
    renderAttachment();

    if (text) appendMessage("user", text);
    else if (fileToUpload) appendMessage("user", `🔎 แนบไฟล์: ${fileToUpload.name}`);
    chatInput.value = "";
    chatInput.style.height = "auto";

    if (fileToUpload) {
        try {
            const defaultDocId = fileToUpload.name.replace(/\.[^.]+$/, "");
            const docId = prompt("ตั้งชื่อ Doc ID:", defaultDocId) || defaultDocId;

            appendMessage("assistant", `⏳ กำลังอัปโหลด... (ID: ${docId})`, { label: "System" });
            const res = await uploadFileToBackend(fileToUpload, docId);

            const costNote = res?.cost_estimate_usd
                ? ` · ค่า OCR: ${fmtUsd(res.cost_estimate_usd)}`
                : "";
            appendMessage("assistant", `✅ อัปโหลดสำเร็จ! Pages: ${res.page_count}${costNote}`, { label: "System" });
            refreshCostWidget();

            // รีเฟรชเอกสารเพื่อให้ ID ใหม่ปรากฏใน Dropdown
            await fetchDocuments();
        } catch (err) {
            console.error(err);
            appendMessage("assistant", "❌ Error: " + err.message, { label: "Error" });
            attachedFile = fileToUpload;
            renderAttachment();
            if (!text) return;
        }
    }

    if (text) {
        const loadingId = "loading-" + Date.now();
        const loadingWrapper = document.createElement("div");
        loadingWrapper.id = loadingId;
        loadingWrapper.className = "flex w-full mb-6 justify-start msg-animate";
        loadingWrapper.innerHTML = `
            <div class="flex-none w-8 h-8 rounded-full bg-white border border-slate-200 text-brand-600 flex items-center justify-center mr-3 shadow-sm">
                <svg class="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
            </div>
            <div class="bg-white border border-slate-100 text-slate-500 rounded-2xl rounded-tl-sm px-5 py-3.5 shadow-sm text-sm">กำลังค้นหาคำตอบ...</div>`;
        
        // Insert loading before dummy
        const dummy = document.getElementById("scrollDummy");
        if (dummy) {
            chatMessages.insertBefore(loadingWrapper, dummy);
        } else {
            chatMessages.appendChild(loadingWrapper);
        }
        
        scrollToBottom();

        const historyToSend = chatHistory.slice(0, -1);
        const payload = {
            query: text,
            doc_ids: selectedDocId ? [selectedDocId] : null,
            top_k: 20,
            mode: mode,
            history: historyToSend,
            llm_mode: getLlmMode(),
        };

        try {
            await streamAskAndRender(payload, loadingId);
        } catch (err) {
            const loadingEl = document.getElementById(loadingId);
            if(loadingEl) loadingEl.remove();
            appendMessage("assistant", "❌ Error: " + err.message, { label: "Error" });
        }
    }
}

/**
 * streamAskAndRender — POST /ask/stream, parse SSE, render tokens live.
 * Removes the loading spinner as soon as the first token arrives.
 */
async function streamAskAndRender(payload, loadingId) {
    const res = await fetch("/ask/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeader() },
        body: JSON.stringify(payload),
    });
    if (handleAuthResponse(res)) throw new Error("Auth required");
    if (!res.ok) throw new Error("API Error: " + res.status);

    // Live streaming bubble
    const wrapper = document.createElement("div");
    wrapper.className = "flex w-full mb-6 justify-start msg-animate";
    const avatar = document.createElement("div");
    avatar.className = "flex-none w-8 h-8 rounded-full bg-white border border-slate-200 text-brand-600 flex items-center justify-center mr-3 shadow-sm";
    avatar.innerHTML = '<svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>';
    const bubble = document.createElement("div");
    bubble.className = "relative max-w-[85%] md:max-w-[75%] px-5 py-3.5 text-sm leading-relaxed shadow-sm bg-white border border-slate-100 text-slate-700 rounded-2xl rounded-tl-sm";
    const textContainer = document.createElement("div");
    textContainer.className = "whitespace-pre-wrap font-sans prose prose-sm max-w-none answer-text-content stream-cursor";
    bubble.appendChild(textContainer);
    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);

    const dummy = document.getElementById("scrollDummy");
    if (dummy) chatMessages.insertBefore(wrapper, dummy);
    else chatMessages.appendChild(wrapper);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let accumulated = "";
    let sources = [];
    let intent = "rag_query";
    let doneMeta = null;
    let firstTokenSeen = false;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";
        for (const raw of events) {
            const evt = parseSSE(raw);
            if (!evt) continue;
            if (evt.event === "sources") {
                sources = evt.payload.sources || [];
            } else if (evt.event === "token") {
                if (!firstTokenSeen) {
                    firstTokenSeen = true;
                    const loadingEl = document.getElementById(loadingId);
                    if (loadingEl) loadingEl.remove();
                }
                accumulated += evt.payload.text || "";
                textContainer.textContent = accumulated;
                scrollToBottom();
            } else if (evt.event === "done") {
                intent = evt.payload.intent || intent;
                doneMeta = evt.payload;
            } else if (evt.event === "error") {
                throw new Error(evt.payload.message || "stream error");
            }
        }
    }

    // Finalize: remove streaming bubble, render properly with tables + sources
    wrapper.remove();
    const loadingEl = document.getElementById(loadingId);
    if (loadingEl) loadingEl.remove();
    appendMessage("assistant", accumulated || "(ไม่พบคำตอบ)", {
        intent,
        sources,
        tables: [],
        llmProvider: doneMeta?.llm_provider,
        llmModel: doneMeta?.llm_model,
        costUsd: doneMeta?.cost_estimate_usd,
    });
    if (doneMeta?.cost_estimate_usd !== undefined) refreshCostWidget();
}

function parseSSE(raw) {
    let event = "message", data = "";
    for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    if (!data) return null;
    try {
        return { event, payload: JSON.parse(data) };
    } catch {
        return { event, payload: { raw: data } };
    }
}

async function loadHistory() {
    historyBox.innerHTML = '<div class="flex justify-center py-10"><div class="w-6 h-6 border-2 border-brand-500 border-t-transparent rounded-full animate-spin"></div></div>';
    try {
        const res = await fetch(`/history?limit=50`, { headers: { ...getAuthHeader() } });
        if (handleAuthResponse(res)) throw new Error("Auth required");
        if (!res.ok) throw new Error("Load failed");
        
        let data = await res.json();
        
        // 🔄 จัดลำดับใหม่: ใหม่สุดอยู่บน (Newest first)
        data = data.reverse(); 

        if (!data.length) { 
            historyBox.innerHTML = '<p class="text-center text-slate-400 mt-10">... ยังไม่มีประวัติ ...</p>'; 
            return; 
        }

        historyBox.innerHTML = data.map((item) => `
            <div class="mb-4 pb-4 border-b border-slate-100 last:border-0 hover:bg-slate-50 p-2 rounded transition cursor-default">
              <div class="flex justify-between items-center mb-1">
                <span class="text-[10px] font-bold text-slate-400 uppercase bg-slate-100 px-1.5 py-0.5 rounded">${item.mode || "Auto"}</span>
                <span class="text-[10px] text-slate-400">${item.ts ? item.ts.substring(0, 10) : ""}</span>
              </div>
              <div class="font-medium text-slate-800 text-sm mb-1 line-clamp-2">Q: ${(item.query || "").replace(/\n/g, " ")}</div>
              <div class="text-slate-500 text-xs pl-2 border-l-2 border-brand-200 line-clamp-3">A: ${(item.answer || "").replace(/\n/g, " ")}</div>
            </div>`).join("");
    } catch (err) { 
        historyBox.innerHTML = `<p class="text-center text-red-400 mt-10">Load Error: ${err.message}</p>`; 
    }
}

// =======================
// 🖱️ Event Listeners
// =======================
sendBtn.onclick = () => sendMessage();

chatInput.addEventListener("keydown", (e) => { 
    if (e.key === "Enter" && !e.shiftKey) { 
        e.preventDefault(); 
        sendMessage(); 
    } 
});

chatInput.addEventListener("input", () => { 
    chatInput.style.height = "auto"; 
    chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + "px"; 
});

uploadBtn.onclick = () => fileInput.click();

fileInput.onchange = (e) => { 
    if (e.target.files[0]) { 
        attachedFile = e.target.files[0]; 
        renderAttachment(); 
    } 
    fileInput.value = ""; 
};

toggleHistoryBtn.onclick = () => { 
    if (historyVisible) hideHistoryPanel(); else showHistoryPanel(); 
};

refreshHistoryBtn.onclick = () => loadHistory();

// ❌ แก้ไขจุดที่กดปิด History ไม่ได้
if (closeHistoryBtn) {
    closeHistoryBtn.addEventListener("click", (e) => {
        e.preventDefault();
        hideHistoryPanel();
    });
}

// =======================
// 🎬 Landing view event wiring
// =======================
if (dropZone && fileInputLanding) {
    // Click / keyboard to open picker
    dropZone.addEventListener("click", (e) => {
        // Avoid double-fire when the inner button is clicked
        if (e.target.closest && e.target.closest(".drop-zone-btn")) return;
        fileInputLanding.click();
    });
    dropZone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            fileInputLanding.click();
        }
    });
    if (chooseFileBtn) {
        chooseFileBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            fileInputLanding.click();
        });
    }
    // Drag & drop
    ["dragenter", "dragover"].forEach(evt => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("drag-over");
        });
    });
    ["dragleave", "drop"].forEach(evt => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("drag-over");
        });
    });
    dropZone.addEventListener("drop", (e) => {
        const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (file) handleFileUpload(file);
    });
    fileInputLanding.addEventListener("change", (e) => {
        const file = e.target.files && e.target.files[0];
        if (file) handleFileUpload(file);
    });
}

if (landingSearch) {
    landingSearch.addEventListener("input", applyLandingFilter);
}
if (refreshDocsBtn) {
    refreshDocsBtn.addEventListener("click", fetchDocuments);
}
if (backToDocsBtn) {
    backToDocsBtn.addEventListener("click", showLandingView);
}

// =======================
// ⚙️ Settings modal (API Key configuration)
// =======================
const API_KEY_STORAGE = "app_api_key";
const AUTH_TOKEN_STORAGE = "auth_token";     // Phase 5.3 — session token from /auth/login
const AUTH_USER_STORAGE = "auth_user";       // cached user info (username, is_admin)
const settingsModal      = document.getElementById("settingsModal");
const apiKeyInput        = document.getElementById("apiKeyInput");
const toggleVisibilityBtn = document.getElementById("toggleVisibilityBtn");
const saveSettingsBtn    = document.getElementById("saveSettingsBtn");
const cancelSettingsBtn  = document.getElementById("cancelSettingsBtn");
const modalCloseBtn      = document.getElementById("modalCloseBtn");
const eyeShow            = document.getElementById("eyeShow");
const eyeHide            = document.getElementById("eyeHide");
const toastEl            = document.getElementById("toast");

function openSettingsModal() {
    if (!settingsModal) return;
    // Load existing key (if any) into the input
    const saved = localStorage.getItem(API_KEY_STORAGE) || "";
    apiKeyInput.value = saved;
    // Always start masked
    apiKeyInput.type = "password";
    eyeShow.classList.remove("hidden");
    eyeHide.classList.add("hidden");
    // Reflect current preset selection (or highlight none if user overrode individually)
    const currentPreset = detectCurrentPreset();
    document.querySelectorAll('input[name="preset"]').forEach(r => {
        r.checked = (currentPreset !== null && r.value === currentPreset);
    });
    settingsModal.classList.remove("hidden");
    setTimeout(() => apiKeyInput.focus(), 50);
}

function closeSettingsModal() {
    if (settingsModal) settingsModal.classList.add("hidden");
}

function toggleKeyVisibility() {
    if (apiKeyInput.type === "password") {
        apiKeyInput.type = "text";
        eyeShow.classList.add("hidden");
        eyeHide.classList.remove("hidden");
        toggleVisibilityBtn.title = "Hide";
    } else {
        apiKeyInput.type = "password";
        eyeShow.classList.remove("hidden");
        eyeHide.classList.add("hidden");
        toggleVisibilityBtn.title = "Show";
    }
}

// Aggressive cleaner - strips ASCII + Unicode whitespace + zero-width chars
function sanitizeKey(raw) {
    return String(raw || "")
        // Zero-width chars: ZWSP, ZWNJ, ZWJ, BOM
        .replace(/[\u200B\u200C\u200D\uFEFF]/g, "")
        // All whitespace incl NBSP, en/em spaces, ideographic space
        .replace(/[\s\u00A0\u2000-\u200A\u202F\u205F\u3000]/g, "")
        .trim();
}

async function saveApiKey() {
    const key = sanitizeKey(apiKeyInput.value);
    if (!key) {
        localStorage.removeItem(API_KEY_STORAGE);
        showToast("API key ถูกลบออก", "info");
        closeSettingsModal();
        if (typeof fetchDocuments === "function") fetchDocuments();
        return;
    }
    localStorage.setItem(API_KEY_STORAGE, key);

    // Immediate verification: does the key actually work?
    saveSettingsBtn.disabled = true;
    saveSettingsBtn.textContent = "กำลังตรวจสอบ…";
    try {
        const r = await fetch("/documents", {
            headers: { Authorization: `Bearer ${key}` },
        });
        if (r.status === 200) {
            showToast("✓ บันทึกและยืนยัน key เรียบร้อย", "success");
            closeSettingsModal();
            if (typeof fetchDocuments === "function") fetchDocuments();
        } else if (r.status === 401 || r.status === 403) {
            showToast(`⚠️ Key ไม่ถูกต้อง (HTTP ${r.status}) — ตรวจสอบว่าคัดลอกครบและไม่มี space`, "error", 4500);
            // Keep modal open so user can retry
        } else {
            showToast(`⚠️ ตรวจสอบไม่ได้ (HTTP ${r.status}) — บันทึกไว้แล้ว`, "info");
            closeSettingsModal();
        }
    } catch (e) {
        showToast(`⚠️ Network error: ${e.message} — บันทึกไว้แล้ว`, "info");
        closeSettingsModal();
    } finally {
        saveSettingsBtn.disabled = false;
        saveSettingsBtn.innerHTML = '<svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg> Save API Key';
    }
}

function showToast(message, level = "info", durationMs = 2500) {
    if (!toastEl) return;
    toastEl.className = `toast ${level}`;
    toastEl.textContent = message;
    clearTimeout(showToast._timer);
    showToast._timer = setTimeout(() => toastEl.classList.add("hidden"), durationMs);
}

/**
 * getAuthHeader — Retrieve API key from localStorage and return it as a
 * standard Bearer Authorization header object.
 *
 * Example usage:
 *   fetch("/ask", {
 *     method: "POST",
 *     headers: {
 *       "Content-Type": "application/json",
 *       ...getAuthHeader(),           // ← spreads Authorization if present
 *     },
 *     body: JSON.stringify({ query: "hi" })
 *   });
 *
 * Returns an empty object if no key is set, so spreading is always safe.
 */
function getAuthHeader() {
    // Prefer the session token from Phase 5.3 login flow; fall back to the
    // legacy shared APP_API_KEY that power users can still paste in Settings.
    const sess = sanitizeKey(localStorage.getItem(AUTH_TOKEN_STORAGE));
    if (sess) return { Authorization: `Bearer ${sess}` };
    const key = sanitizeKey(localStorage.getItem(API_KEY_STORAGE));
    return key ? { Authorization: `Bearer ${key}` } : {};
}

/**
 * handleAuthResponse — check a fetch Response for auth errors (401/403).
 * If auth failed, show a toast and open the Settings modal automatically.
 * Returns true if auth failed (caller should abort further processing).
 */
function handleAuthResponse(res) {
    if (res.status === 401 || res.status === 403) {
        // Session-token path: token expired / user disabled → clear + prompt login
        if (localStorage.getItem(AUTH_TOKEN_STORAGE)) {
            showToast("⚠️ Session หมดอายุ — กรุณา sign in อีกครั้ง", "error", 3500);
            localStorage.removeItem(AUTH_TOKEN_STORAGE);
            localStorage.removeItem(AUTH_USER_STORAGE);
            updateUserMenu();
            showLoginOverlay();
            return true;
        }
        // Legacy APP_API_KEY path → send them to Settings to fix the key
        showToast(
            res.status === 401
                ? "⚠️ ยังไม่ได้ตั้งค่า API key — กรุณาใส่ใน Settings"
                : "⚠️ API key ไม่ถูกต้อง — ตรวจสอบใน Settings",
            "error", 3500,
        );
        openSettingsModal();
        return true;
    }
    return false;
}

// Wire up all "open settings" triggers (there are gear buttons in both headers)
document.querySelectorAll('[data-action="open-settings"]').forEach(btn => {
    btn.addEventListener("click", openSettingsModal);
});

if (modalCloseBtn)       modalCloseBtn.addEventListener("click", closeSettingsModal);
if (cancelSettingsBtn)   cancelSettingsBtn.addEventListener("click", closeSettingsModal);
if (saveSettingsBtn)     saveSettingsBtn.addEventListener("click", saveApiKey);
if (toggleVisibilityBtn) toggleVisibilityBtn.addEventListener("click", toggleKeyVisibility);

// Close on backdrop click
if (settingsModal) {
    settingsModal.addEventListener("click", (e) => {
        if (e.target === settingsModal) closeSettingsModal();
    });
}
// Close on Escape · Save on Enter (when focused in input)
document.addEventListener("keydown", (e) => {
    if (!settingsModal || settingsModal.classList.contains("hidden")) return;
    if (e.key === "Escape") closeSettingsModal();
});
if (apiKeyInput) {
    apiKeyInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); saveApiKey(); }
    });
}

// --- Cost widget (Phase 4) — polls /stats/cost + click opens detail modal ---
const COST_POLL_MS = 30000;

function fmtUsd(x) {
    const n = Number(x || 0);
    if (n === 0) return "$0.00";
    if (n < 0.01) return `$${n.toFixed(4)}`;
    if (n < 1) return `$${n.toFixed(3)}`;
    return `$${n.toFixed(2)}`;
}

async function fetchCostStats() {
    try {
        const res = await fetch("/stats/cost", { headers: { ...getAuthHeader() } });
        if (!res.ok) return null;
        return await res.json();
    } catch { return null; }
}

function updateCostWidget(stats) {
    const el = document.getElementById("costWidget");
    const label = document.getElementById("costWidgetLabel");
    if (!el || !label) return;
    const daily = stats?.daily || { total_usd: 0, warn_pct: 0 };
    label.textContent = fmtUsd(daily.total_usd);
    let level = "ok";
    if (daily.warn_pct >= 90) level = "danger";
    else if (daily.warn_pct >= 50) level = "warn";
    el.dataset.level = level;
    el.title = `Today: ${fmtUsd(daily.total_usd)} of ${fmtUsd(daily.warn_threshold_usd)} threshold (${daily.warn_pct}%)`;
}

async function refreshCostWidget() {
    const s = await fetchCostStats();
    if (s) updateCostWidget(s);
}

async function openCostModal() {
    const modal = document.getElementById("costModal");
    const body = document.getElementById("costModalBody");
    if (!modal) return;
    modal.classList.remove("hidden");
    body.innerHTML = '<div class="cost-loading">กำลังโหลด...</div>';
    await renderCostBody();
}
function closeCostModal() {
    document.getElementById("costModal")?.classList.add("hidden");
}
async function renderCostBody() {
    const body = document.getElementById("costModalBody");
    if (!body) return;
    const [stats, recentRes] = await Promise.all([
        fetchCostStats(),
        fetch("/stats/cost/recent?limit=15", { headers: { ...getAuthHeader() } }).then(r => r.ok ? r.json() : {calls: []}).catch(() => ({calls: []})),
    ]);
    if (!stats) { body.innerHTML = '<div class="cost-loading">โหลดข้อมูลไม่สำเร็จ</div>'; return; }
    const daily = stats.daily || {};
    const session = stats.session || {};
    const pct = Math.min(100, daily.warn_pct || 0);
    let heroClass = "";
    if (pct >= 90) heroClass = "danger";
    else if (pct >= 50) heroClass = "warn";

    const rows = (obj) => Object.entries(obj || {}).sort((a,b) => b[1]-a[1]).map(([k,v]) =>
        `<div class="cost-row"><span class="cost-row-key">${escapeHtml(k)}</span><span class="cost-row-val">${fmtUsd(v)}</span></div>`
    ).join("") || `<div class="cost-loading">ยังไม่มีข้อมูล</div>`;

    const recentRows = (recentRes.calls || []).map(c => `
        <div class="cost-row" title="${escapeHtml(c.ts || "")}">
          <span class="cost-row-key">${escapeHtml(c.endpoint || "?")} · ${escapeHtml(c.model || "?")}</span>
          <span class="cost-row-val">${fmtUsd(c.cost_usd)}</span>
        </div>`).join("") || `<div class="cost-loading">ยังไม่มีการเรียก</div>`;

    body.innerHTML = `
      <div class="cost-hero ${heroClass}">
        <div class="cost-hero-label">TODAY (${daily.call_count || 0} calls)</div>
        <div class="cost-hero-amount">${fmtUsd(daily.total_usd)}</div>
        <div class="cost-hero-sub">${pct}% of ${fmtUsd(daily.warn_threshold_usd)} threshold</div>
        <div class="cost-bar-track"><div class="cost-bar-fill ${heroClass}" style="width:${pct}%"></div></div>
      </div>

      <div>
        <div class="cost-section-title">Session (${session.call_count || 0} calls)</div>
        <div class="cost-row"><span class="cost-row-key">Since backend start</span><span class="cost-row-val">${fmtUsd(session.total_usd)}</span></div>
      </div>

      <div>
        <div class="cost-section-title">By Endpoint</div>
        ${rows(daily.by_endpoint)}
      </div>

      <div>
        <div class="cost-section-title">By Provider</div>
        ${rows(daily.by_provider)}
      </div>

      <div>
        <div class="cost-section-title">By Model</div>
        ${rows(daily.by_model)}
      </div>

      <div>
        <div class="cost-section-title">Recent Calls</div>
        ${recentRows}
      </div>
    `;
}
function escapeHtml(s) {
    return String(s || "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

document.getElementById("costWidget")?.addEventListener("click", openCostModal);
document.getElementById("costModalClose")?.addEventListener("click", closeCostModal);
document.getElementById("costCloseBtn2")?.addEventListener("click", closeCostModal);
document.getElementById("costRefreshBtn")?.addEventListener("click", () => renderCostBody());
document.getElementById("costModal")?.addEventListener("click", (e) => { if (e.target.id === "costModal") closeCostModal(); });

// Kick off polling
refreshCostWidget();
setInterval(refreshCostWidget, COST_POLL_MS);

// --- Preset picker (Phase 3) — bundles ocr_mode + llm_mode ---
const PRESET_KEY = "processing_preset";
const PRESETS = {
    air_gapped:    { ocr: "local", llm: "local" },
    hybrid:        { ocr: "auto",  llm: "local" },  // default (matches historical behavior)
    cloud_premium: { ocr: "api",   llm: "api"   },
};

function getPreset() {
    const stored = localStorage.getItem(PRESET_KEY);
    return PRESETS[stored] ? stored : "hybrid";
}

/**
 * Apply a preset: set both ocr_mode and llm_mode in localStorage.
 * Also re-syncs any UI elements (OCR radio picker + LLM pill).
 */
function applyPreset(name) {
    const p = PRESETS[name];
    if (!p) return;
    localStorage.setItem(PRESET_KEY, name);
    localStorage.setItem("ocr_mode", p.ocr);
    localStorage.setItem(LLM_MODE_KEY, p.llm);
    // Sync UI
    const ocrRadio = document.querySelector(`input[name="ocrMode"][value="${p.ocr}"]`);
    if (ocrRadio) ocrRadio.checked = true;
    renderLlmModeBtn();
    // Reflect preset selection in the settings modal (if open)
    const presetRadio = document.querySelector(`input[name="preset"][value="${name}"]`);
    if (presetRadio) presetRadio.checked = true;
}

/**
 * Reverse mapping: given current ocr_mode + llm_mode, figure out which
 * preset (if any) matches so the settings modal can highlight it.
 * "Custom" is returned when the user has picked an off-preset combo.
 */
function detectCurrentPreset() {
    const ocr = localStorage.getItem("ocr_mode") || "auto";
    const llm = getLlmMode();
    for (const [name, cfg] of Object.entries(PRESETS)) {
        if (cfg.ocr === ocr && cfg.llm === llm) return name;
    }
    return null; // custom combo — none of the presets match
}

function initPresetPicker() {
    const radios = document.querySelectorAll('input[name="preset"]');
    if (!radios.length) return;
    // Initial state: use stored preset, or detect from current ocr/llm settings
    const current = detectCurrentPreset() || getPreset();
    radios.forEach(r => { r.checked = (r.value === current); });
    radios.forEach(r => {
        r.addEventListener("change", () => {
            if (r.checked) {
                applyPreset(r.value);
                showToast(`เปลี่ยนโหมด: ${r.closest(".preset-card").querySelector(".preset-title").textContent.trim()}`, "info", 1800);
            }
        });
    });
}

// --- LLM Mode toggle (persisted in localStorage) ---
const LLM_MODE_KEY = "llm_mode";
const LLM_MODES = [
    { value: "auto",  icon: "🔀", label: "Auto",       tip: "ค่าเริ่มต้น (Local Ollama)" },
    { value: "local", icon: "🔒", label: "Local",      tip: "ในเครื่อง Server (Ollama) — ข้อมูลไม่ออกไปข้างนอก" },
    { value: "api",   icon: "⚡", label: "Cloud API",  tip: "OpenRouter Gemini — คุณภาพสูงสุด (เสีย credit)" },
];

function getLlmMode() {
    const stored = (localStorage.getItem(LLM_MODE_KEY) || "auto").toLowerCase();
    return LLM_MODES.some(m => m.value === stored) ? stored : "auto";
}
function setLlmMode(val) {
    localStorage.setItem(LLM_MODE_KEY, val);
    renderLlmModeBtn();
}
function renderLlmModeBtn() {
    const btn = document.getElementById("llmModeBtn");
    if (!btn) return;
    const current = getLlmMode();
    const cfg = LLM_MODES.find(m => m.value === current) || LLM_MODES[0];
    btn.dataset.mode = current;
    btn.title = cfg.tip;
    const iconEl = btn.querySelector(".llm-mode-icon");
    const labelEl = btn.querySelector(".llm-mode-label");
    if (iconEl) iconEl.textContent = cfg.icon;
    if (labelEl) labelEl.textContent = cfg.label;
}
function cycleLlmMode() {
    const current = getLlmMode();
    const idx = LLM_MODES.findIndex(m => m.value === current);
    const next = LLM_MODES[(idx + 1) % LLM_MODES.length].value;
    setLlmMode(next);
    showToast(`LLM: ${LLM_MODES.find(m => m.value === next).label}`, "info", 1500);
}
document.getElementById("llmModeBtn")?.addEventListener("click", cycleLlmMode);
renderLlmModeBtn();

// Sync OCR mode radio to what's stored (default hybrid → auto if nothing set)
(function initOcrModeRadio() {
    const stored = localStorage.getItem("ocr_mode") || "auto";
    const el = document.querySelector(`input[name="ocrMode"][value="${stored}"]`);
    if (el) el.checked = true;
    // Persist on change
    document.querySelectorAll('input[name="ocrMode"]').forEach(r => {
        r.addEventListener("change", () => {
            if (r.checked) localStorage.setItem("ocr_mode", r.value);
        });
    });
})();

initPresetPicker();

// --- Auth flow (Phase 5.3) ---

function showLoginOverlay(errorMsg) {
    const overlay = document.getElementById("loginOverlay");
    if (!overlay) return;
    overlay.classList.remove("hidden");
    const err = document.getElementById("loginError");
    const errTxt = document.getElementById("loginErrorText");
    if (errorMsg && err && errTxt) {
        errTxt.textContent = errorMsg;
        err.style.display = "";
    } else if (err) {
        err.style.display = "none";
    }
    document.getElementById("loginUsername")?.focus();
}

function hideLoginOverlay() {
    document.getElementById("loginOverlay")?.classList.add("hidden");
    const err = document.getElementById("loginError");
    if (err) err.style.display = "none";
}

function updateUserMenu() {
    // Two menu copies live in different headers (landing + chat) — update both.
    const menus = [
        { menu: "userMenu",        name: "userMenuName"        },
        { menu: "userMenuLanding", name: "userMenuNameLanding" },
    ];
    const raw = localStorage.getItem(AUTH_USER_STORAGE);
    let user = null;
    try { user = raw ? JSON.parse(raw) : null; } catch {}
    for (const { menu, name } of menus) {
        const menuEl = document.getElementById(menu);
        const nameEl = document.getElementById(name);
        if (!menuEl || !nameEl) continue;
        if (!user) {
            menuEl.classList.add("hidden");
        } else {
            nameEl.textContent = user.username || "user";
            menuEl.title = `${user.username}${user.is_admin ? " (admin)" : ""}`;
            menuEl.classList.remove("hidden");
        }
    }
}

async function tryVerifyToken() {
    const token = localStorage.getItem(AUTH_TOKEN_STORAGE);
    if (!token) return null;
    try {
        const res = await fetch("/auth/me", { headers: { Authorization: `Bearer ${token}` } });
        if (!res.ok) return null;
        return await res.json();
    } catch { return null; }
}

async function doLogin() {
    const username = document.getElementById("loginUsername")?.value?.trim();
    const password = document.getElementById("loginPassword")?.value || "";
    if (!username || !password) {
        showLoginOverlay("กรุณากรอก username และ password");
        return;
    }
    const btn = document.getElementById("loginBtn");
    if (btn) { btn.disabled = true; btn.textContent = "Signing in..."; }
    try {
        const res = await fetch("/auth/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, password }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            showLoginOverlay(err.detail || `Login failed (${res.status})`);
            return;
        }
        const d = await res.json();
        localStorage.setItem(AUTH_TOKEN_STORAGE, d.token);
        localStorage.setItem(AUTH_USER_STORAGE, JSON.stringify({
            id: d.user_id, username: d.username, is_admin: d.is_admin,
        }));
        hideLoginOverlay();
        updateUserMenu();
        showToast(`ยินดีต้อนรับ ${d.username}!`, "info", 2000);
        // Refresh page data now that we're authed
        fetchDocuments();
        refreshCostWidget();
    } catch (e) {
        showLoginOverlay(`Network error: ${e.message}`);
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = "Sign in"; }
        const pw = document.getElementById("loginPassword"); if (pw) pw.value = "";
    }
}

async function doLogout() {
    const token = localStorage.getItem(AUTH_TOKEN_STORAGE);
    if (token) {
        try {
            await fetch("/auth/logout", {
                method: "POST",
                headers: { Authorization: `Bearer ${token}` },
            });
        } catch { /* ignore */ }
    }
    localStorage.removeItem(AUTH_TOKEN_STORAGE);
    localStorage.removeItem(AUTH_USER_STORAGE);
    updateUserMenu();
    showLoginOverlay();
}

// Bootstrap: on page load, verify token OR show login
(async function initAuth() {
    // Fast path: if user has a legacy app_api_key AND no session, allow legacy mode
    // (backward compat for existing installs; login overlay stays hidden)
    const hasLegacyKey = !!sanitizeKey(localStorage.getItem(API_KEY_STORAGE));

    const cached = await tryVerifyToken();
    if (cached) {
        localStorage.setItem(AUTH_USER_STORAGE, JSON.stringify({
            id: cached.id, username: cached.username, is_admin: cached.is_admin,
        }));
        updateUserMenu();
        return;
    }
    // Session invalid — clear cached user + prompt login
    localStorage.removeItem(AUTH_TOKEN_STORAGE);
    localStorage.removeItem(AUTH_USER_STORAGE);
    updateUserMenu();
    if (hasLegacyKey) return;  // legacy mode — allow use without login overlay
    showLoginOverlay();
})();

document.getElementById("loginBtn")?.addEventListener("click", doLogin);
document.getElementById("loginPassword")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); doLogin(); }
});
document.getElementById("loginUsername")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); document.getElementById("loginPassword")?.focus(); }
});
document.getElementById("logoutBtn")?.addEventListener("click", doLogout);
document.getElementById("logoutBtnLanding")?.addEventListener("click", doLogout);

// --- Init ---
fetchDocuments();
// Start on landing — user picks / uploads a document first
showLandingView();
// Prime the chat welcome message (shown after first navigation to chat)
setTimeout(() => {
    appendMessage("assistant", "สวัสดีครับ/ค่ะ! 👋\n\n**Tip:** ถ้าต้องการเปลี่ยนเอกสาร กด **⬅ Documents** ที่มุมซ้ายบนเพื่อกลับไปที่หน้าเอกสาร นะครับ/ค่ะ", { label: "System" });
}, 500);