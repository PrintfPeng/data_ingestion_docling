/* frontend/js/app.js */

const chatMessages = document.getElementById("chatMessages");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const toggleHistoryBtn = document.getElementById("toggleHistoryBtn");
const refreshHistoryBtn = document.getElementById("refreshHistoryBtn");
const historyPanel = document.getElementById("historyPanel");
const historyBox = document.getElementById("historyBox");
const attachmentInfo = document.getElementById("attachmentInfo");

const modeSelect = document.getElementById("modeSelect");
const docSelect = document.getElementById("docSelect");
const modeSelectMobile = document.getElementById("modeSelectMobile");
const docSelectMobile = document.getElementById("docSelectMobile");

let historyVisible = false;
let attachedFile = null;

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
                opt.value = doc;
                opt.text = `📄 ${doc}`;
                sel.add(opt);
            });
            if (docs.includes(currentVal)) sel.value = currentVal;
        });
    } catch (e) {
        console.error("Failed to load documents:", e);
    }
}

// [REMOVED] Function injectTableFromSource (ไม่จำเป็นแล้วเพราะ Backend ส่ง HTML มาให้แล้ว)

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

    const contentDiv = document.createElement("div");
    contentDiv.className = "whitespace-pre-wrap font-sans prose"; 
    
    // [CRITICAL] ใช้ innerHTML เพื่อให้ Browser เรนเดอร์ HTML Table ที่ Backend ส่งมา
    contentDiv.innerHTML = text;
    
    bubble.appendChild(contentDiv);

    // Meta / Sources Section
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

    // Dummy for scroll
    const dummy = document.createElement("div");
    dummy.style.height = "100px";
    dummy.style.flexShrink = "0";
    chatMessages.appendChild(dummy);
    setTimeout(() => dummy.scrollIntoView({ behavior: "smooth", block: "end" }), 100);
}

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
    const selectedDoc = getSelectedDocId();
    const fileToUpload = attachedFile;
    attachedFile = null;
    renderAttachment();

    if (text) appendMessage("user", text);
    else if (fileToUpload) appendMessage("user", `📎 แนบไฟล์: ${fileToUpload.name}`);
    chatInput.value = "";
    chatInput.style.height = "auto";

    if (fileToUpload) {
        try {
            const defaultDocId = fileToUpload.name.replace(/\.[^.]+$/, "");
            const docId = prompt("ตั้งชื่อ Doc ID:", defaultDocId) || defaultDocId;
            appendMessage("assistant", `⏳ กำลังอัปโหลด... (ID: ${docId})`, { label: "System" });
            const res = await uploadFileToBackend(fileToUpload, docId);
            appendMessage("assistant", `✅ อัปโหลดสำเร็จ! Pages: ${res.page_count}`, { label: "System" });
            fetchDocuments();
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
        chatMessages.appendChild(loadingWrapper);
        requestAnimationFrame(() => loadingWrapper.scrollIntoView({ behavior: "smooth", block: "end" }));

        try {
            const payload = { query: text, doc_ids: selectedDoc ? [selectedDoc] : null, top_k: 20, mode: mode };
            const res = await fetch("/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
            if (!res.ok) throw new Error("API Error: " + res.status);
            const data = await res.json();
            document.getElementById(loadingId).remove();
            
            // ส่ง sources ไปให้ appendMessage (ซึ่งจะไม่ทำอะไรกับตารางแล้ว แค่แสดงรายชื่อ source เฉยๆ)
            appendMessage("assistant", data.answer || "(ไม่พบคำตอบ)", { 
                intent: data.intent, 
                sources: data.sources || [] 
            });
            
        } catch (err) {
            document.getElementById(loadingId).remove();
            appendMessage("assistant", "❌ Error: " + err.message, { label: "Error" });
        }
    }
}

async function loadHistory() {
    historyBox.innerHTML = '<div class="flex justify-center py-10"><div class="w-6 h-6 border-2 border-brand-500 border-t-transparent rounded-full animate-spin"></div></div>';
    try {
        const res = await fetch(`/history?limit=50`);
        if (!res.ok) throw new Error("Load failed");
        const data = await res.json();
        if (!data.length) { historyBox.innerHTML = '<p class="text-center text-slate-400 mt-10">... ยังไม่มีประวัติ ...</p>'; return; }
        historyBox.innerHTML = data.map((item) => `
            <div class="mb-4 pb-4 border-b border-slate-100 last:border-0 hover:bg-slate-50 p-2 rounded transition cursor-default">
              <div class="flex justify-between items-center mb-1"><span class="text-[10px] font-bold text-slate-400 uppercase bg-slate-100 px-1.5 py-0.5 rounded">${item.mode || "Auto"}</span><span class="text-[10px] text-slate-400">${item.ts ? item.ts.substring(0, 10) : ""}</span></div>
              <div class="font-medium text-slate-800 text-sm mb-1 line-clamp-2">Q: ${(item.query || "").replace(/\n/g, " ")}</div>
              <div class="text-slate-500 text-xs pl-2 border-l-2 border-brand-200 line-clamp-3">A: ${(item.answer || "").replace(/\n/g, " ")}</div>
            </div>`).join("");
    } catch (err) { historyBox.innerHTML = `<p class="text-center text-red-400 mt-10">Load Error: ${err.message}</p>`; }
}

// Event Listeners
sendBtn.onclick = () => sendMessage();
chatInput.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
chatInput.addEventListener("input", () => { chatInput.style.height = "auto"; chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + "px"; });
uploadBtn.onclick = () => fileInput.click();
fileInput.onchange = (e) => { if (e.target.files[0]) { attachedFile = e.target.files[0]; renderAttachment(); } fileInput.value = ""; };
toggleHistoryBtn.onclick = () => { if (historyVisible) hideHistoryPanel(); else showHistoryPanel(); };
refreshHistoryBtn.onclick = () => loadHistory();

// Init
fetchDocuments();
setTimeout(() => { appendMessage("assistant", "สวัสดีครับ/ค่ะ! 👋\n\n**Tip:** ถ้าต้องการค้นหาเฉพาะเอกสารใดเอกสารหนึ่ง สามารถเลือกได้ที่เมนูด้านบน นะครับ/ค่ะ", { label: "System" }); }, 500);