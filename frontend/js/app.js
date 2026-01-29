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

let historyVisible = false;
let attachedFile = null;

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
        const res = await fetch("/documents");
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

    if (!isUser && (options.intent || (options.sources && options.sources.length))) {
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
async function uploadFileToBackend(file, docId) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("doc_id", docId);
    formData.append("doc_type", "");
    const res = await fetch("/upload", { method: "POST", body: formData });
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
            
            appendMessage("assistant", `✅ อัปโหลดสำเร็จ! Pages: ${res.page_count}`, { label: "System" });
            
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

        try {
            // [FIX 4] ตรวจสอบ selectedDocId ก่อนส่ง (ถ้าเป็น "" คือเลือก All ให้ส่ง null)
            const payload = { 
                query: text, 
                doc_ids: selectedDocId ? [selectedDocId] : null, 
                top_k: 20, 
                mode: mode 
            };
            
            const res = await fetch("/ask", { 
                method: "POST", 
                headers: { "Content-Type": "application/json" }, 
                body: JSON.stringify(payload) 
            });
            
            if (!res.ok) throw new Error("API Error: " + res.status);
            const data = await res.json();
            
            const loadingEl = document.getElementById(loadingId);
            if(loadingEl) loadingEl.remove();
            
            appendMessage("assistant", data.answer || "(ไม่พบคำตอบ)", { 
                intent: data.intent, 
                sources: data.sources || [],
                tables: data.tables || []
            });
            
        } catch (err) {
            const loadingEl = document.getElementById(loadingId);
            if(loadingEl) loadingEl.remove();
            appendMessage("assistant", "❌ Error: " + err.message, { label: "Error" });
        }
    }
}

async function loadHistory() {
    historyBox.innerHTML = '<div class="flex justify-center py-10"><div class="w-6 h-6 border-2 border-brand-500 border-t-transparent rounded-full animate-spin"></div></div>';
    try {
        const res = await fetch(`/history?limit=50`);
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

// --- Init ---
fetchDocuments();
setTimeout(() => { 
    appendMessage("assistant", "สวัสดีครับ/ค่ะ! 👋\n\n**Tip:** ถ้าต้องการค้นหาเฉพาะเอกสารใดเอกสารหนึ่ง สามารถเลือกได้ที่เมนูด้านบน นะครับ/ค่ะ", { label: "System" }); 
}, 500);