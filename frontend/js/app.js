const API_BASE = "http://localhost:8000";

// รอให้หน้าเว็บพร้อมทำงาน
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    fetchDocumentsList(); // ดึงรายชื่อเอกสารมาใส่ Dropdown
});

function initEventListeners() {
    const uploadBtn = document.getElementById('uploadBtn');
    const fileInput = document.getElementById('fileInput');
    const sendBtn = document.getElementById('sendBtn');
    const chatInput = document.getElementById('chatInput');

    // 1. ปุ่ม Upload: คลิกไอคอน -> คลิก input ที่ซ่อนอยู่
    if (uploadBtn && fileInput) {
        uploadBtn.addEventListener('click', (e) => {
            e.preventDefault();
            fileInput.value = ""; // เคลียร์ค่าเดิมเพื่อให้เลือกไฟล์เดิมซ้ำได้
            fileInput.click();
        });
        // เมื่อเลือกไฟล์เสร็จ -> อัปโหลด
        fileInput.addEventListener('change', handleFileUpload);
    }

    // 2. ปุ่ม Send
    if (sendBtn) {
        sendBtn.addEventListener('click', handleSendMessage);
    }

    // 3. กด Enter เพื่อส่ง (Shift+Enter เพื่อขึ้นบรรทัดใหม่)
    if (chatInput) {
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
            }
        });
        // Auto resize textarea
        chatInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
            if(this.value === '') this.style.height = 'auto';
        });
    }
}

// --- Logic อัปโหลดไฟล์ ---
async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    updateAttachmentStatus(`⏳ กำลังอัปโหลด: ${file.name}...`, 'text-blue-500');

    const formData = new FormData();
    formData.append("file", file);
    // สร้าง doc_id ง่ายๆ
    const docId = file.name.replace(/\.[^/.]+$/, "").replace(/\s+/g, "_");
    formData.append("doc_id", docId);
    formData.append("doc_type", "generic");

    try {
        const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
        if (!res.ok) throw new Error("Upload Failed");

        const data = await res.json();
        updateAttachmentStatus(`✅ อัปโหลดเสร็จสิ้น: ${file.name}`, 'text-green-600');
        
        // Refresh รายชื่อเอกสาร
        fetchDocumentsList();
        
        // หายไปใน 3 วิ
        setTimeout(() => updateAttachmentStatus(''), 3000);

    } catch (err) {
        console.error(err);
        updateAttachmentStatus(`❌ อัปโหลดล้มเหลว: ${err.message}`, 'text-red-500');
    }
}

// --- Logic ส่งข้อความและแสดงผล ---
async function handleSendMessage() {
    const chatInput = document.getElementById('chatInput');
    const message = chatInput.value.trim();
    if (!message) return;

    // 1. แสดงข้อความ User
    appendMessage('user', message);
    chatInput.value = '';
    chatInput.style.height = 'auto'; // Reset height

    // 2. แสดงสถานะกำลังคิด
    const loadingId = appendMessage('ai', '<div class="flex gap-1"><div class="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div><div class="w-2 h-2 bg-slate-400 rounded-full animate-bounce delay-100"></div><div class="w-2 h-2 bg-slate-400 rounded-full animate-bounce delay-200"></div></div>', true);

    try {
        const docSelect = document.getElementById('docSelect');
        const selectedDoc = docSelect.value || null;

        const res = await fetch(`${API_BASE}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                query: message,
                doc_ids: selectedDoc ? [selectedDoc] : null
            })
        });

        const data = await res.json();
        removeMessage(loadingId);

        // 3. แสดงคำตอบ AI + รูปภาพ
        renderAIResponse(data);

    } catch (err) {
        removeMessage(loadingId);
        appendMessage('ai', `<span class="text-red-500">เกิดข้อผิดพลาด: ${err.message}</span>`);
    }
}

// --- ฟังก์ชันแสดงผลคำตอบพร้อมรูปภาพ ---
function renderAIResponse(data) {
    let htmlContent = `<div class="prose text-sm md:text-base text-slate-700">${formatText(data.answer)}</div>`;

    // ตรวจสอบ Sources เพื่อหารูปภาพ
    if (data.sources && data.sources.length > 0) {
        const images = data.sources.filter(s => s.source === 'image' || (s.metadata && s.metadata.file_path));
        
        if (images.length > 0) {
            htmlContent += `<div class="mt-4 pt-3 border-t border-slate-100">`;
            htmlContent += `<p class="text-xs font-bold text-slate-500 mb-2">📸 รูปภาพที่เกี่ยวข้อง:</p>`;
            htmlContent += `<div class="flex gap-3 overflow-x-auto pb-2 scrollbar-hide">`;

            images.forEach(img => {
                const imgUrl = resolveImageUrl(img.metadata.file_path);
                if (imgUrl) {
                    htmlContent += `
                        <div class="flex-none relative group">
                            <img src="${imgUrl}" 
                                 class="h-40 w-auto rounded-lg border border-slate-200 object-cover cursor-pointer hover:shadow-lg transition-all"
                                 onclick="window.open('${imgUrl}', '_blank')"
                                 alt="Relevant Image">
                            <div class="absolute bottom-0 left-0 w-full bg-black/50 text-white text-[10px] p-1 truncate rounded-b-lg opacity-0 group-hover:opacity-100 transition-opacity">
                                คลิกเพื่อดูภาพใหญ่
                            </div>
                        </div>
                    `;
                }
            });
            htmlContent += `</div></div>`;
        }
    }

    appendMessage('ai', htmlContent);
}

// --- Helper: แปลง Path D:\... เป็น URL ---
function resolveImageUrl(originalPath) {
    if (!originalPath) return null;

    // เปลี่ยน Backslash เป็น Slash
    let cleanPath = originalPath.replace(/\\/g, '/');

    // ถ้าเป็น URL อยู่แล้ว
    if (cleanPath.startsWith('http')) return cleanPath;

    // Logic: ตัดส่วนเกินออก ให้เหลือแค่ doc_id/images/filename.png
    // สมมติ Path มาเป็น "D:/DATA_INGES/ingested/doc_001/images/img.png"
    // หรือ "ingested/doc_001/images/img.png"
    
    // เรา Mount "/ingested" ไว้ที่ "D:/DATA_INGES/ingested"
    // ดังนั้น URL ควรเป็น http://localhost:8000/ingested/doc_001/images/img.png

    // หาตำแหน่งคำว่า "ingested/"
    const keyword = "ingested/";
    const idx = cleanPath.indexOf(keyword);
    
    if (idx !== -1) {
        // ตัดเอาตั้งแต่ ingested/ เป็นต้นไป
        cleanPath = cleanPath.substring(idx); 
        // จะได้ "ingested/doc_001/images/..."
    } else {
        // ถ้าไม่เจอคำว่า ingested (เช่นเก็บเป็น doc_001/images/...) ให้เติมข้างหน้า
        if (!cleanPath.startsWith('/')) cleanPath = '/' + cleanPath;
        cleanPath = '/ingested' + cleanPath;
    }

    // ตรวจสอบ Double Slash
    cleanPath = cleanPath.replace('//', '/');
    if (!cleanPath.startsWith('/')) cleanPath = '/' + cleanPath;

    return `${API_BASE}${cleanPath}`;
}

// --- Helper: UI ต่างๆ ---
function appendMessage(role, html, isHtml = false) {
    const chatBox = document.getElementById('chatMessages');
    const div = document.createElement('div');
    const id = `msg-${Date.now()}`;
    div.id = id;
    div.className = `flex w-full animate-fade-in-up ${role === 'user' ? 'justify-end' : 'justify-start'}`;

    const avatar = role === 'ai' 
        ? `<div class="w-8 h-8 rounded-full bg-blue-600 flex-none flex items-center justify-center text-white text-xs mr-3 mt-1 shadow-sm">AI</div>` 
        : ``;

    const bubbleClass = role === 'user'
        ? 'bg-blue-600 text-white rounded-2xl rounded-tr-none shadow-md'
        : 'bg-white border border-slate-200 text-slate-700 rounded-2xl rounded-tl-none shadow-sm';

    div.innerHTML = `
        ${avatar}
        <div class="max-w-[85%] md:max-w-[75%] p-4 ${bubbleClass}">
            ${isHtml ? html : `<p>${html}</p>`}
        </div>
    `;

    chatBox.appendChild(div);
    chatBox.scrollTo({ top: chatBox.scrollHeight, behavior: 'smooth' });
    return id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if(el) el.remove();
}

function updateAttachmentStatus(msg, colorClass) {
    const el = document.getElementById('attachmentInfo');
    el.innerHTML = msg;
    el.className = `absolute -top-8 left-0 text-xs font-semibold px-2 ${colorClass}`;
}

function formatText(text) {
    // แปลง Newline เป็น <br> ง่ายๆ
    return text.replace(/\n/g, '<br>');
}

async function fetchDocumentsList() {
    try {
        const res = await fetch(`${API_BASE}/documents`);
        const data = await res.json();
        const select = document.getElementById('docSelect');
        // เก็บค่าเดิมไว้
        const currentVal = select.value;
        
        select.innerHTML = '<option value="">📚 ค้นหาทุกเอกสาร</option>';
        data.documents.forEach(doc => {
            const option = document.createElement('option');
            option.value = doc;
            option.textContent = doc;
            select.appendChild(option);
        });
        select.value = currentVal;
    } catch(e) { console.error("Load docs failed", e); }
}

// frontend/js/app.js

async function askQuestion() {
    const inputField = document.getElementById('chatInput');
    const question = inputField.value.trim();
    
    if (!question) return;

    // 1. เตรียม UI (เคลียร์ช่องพิมพ์, แสดงข้อความฝั่งผู้ใช้)
    inputField.value = '';
    const chatBox = document.getElementById('chatMessages');
    
    // User Message Bubble
    chatBox.innerHTML += `
        <div class="flex justify-end mb-4 animate-fade-in-up">
            <div class="bg-brand-600 text-white px-5 py-3 rounded-2xl rounded-tr-none shadow-md max-w-[80%]">
                ${question}
            </div>
        </div>`;
    
    // Loading Bubble
    const loadingId = 'loading-' + Date.now();
    chatBox.innerHTML += `
        <div id="${loadingId}" class="flex justify-start mb-4">
            <div class="bg-white border border-slate-200 text-slate-500 px-5 py-3 rounded-2xl rounded-tl-none shadow-sm text-sm flex items-center gap-2">
                <span class="animate-pulse">กำลังค้นหาข้อมูล...</span>
            </div>
        </div>`;
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        // 2. ยิง Request ไปหา Backend
        // ใช้ /ask ให้ตรงกับ backend/main.py (เช็คว่าใช้ query หรือ question)
        const res = await fetch(`/ask`, { 
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ query: question }) 
        });
        const data = await res.json();
        
        // ลบ Loading ออก
        const loadingElem = document.getElementById(loadingId);
        if (loadingElem) loadingElem.remove();

        // 3. สร้าง HTML สำหรับคำตอบ (AI Answer)
        let htmlContent = `<div class="flex flex-col items-start mb-8 w-full max-w-4xl animate-fade-in-up">`;
        
        // ส่วนข้อความ
        htmlContent += `
            <div class="flex gap-4 w-full">
                <div class="w-10 h-10 rounded-full bg-gradient-to-tr from-brand-500 to-indigo-600 flex-shrink-0 flex items-center justify-center text-white font-bold text-sm shadow-lg border-2 border-white">AI</div>
                <div class="bg-white border border-slate-100 text-slate-800 p-6 rounded-2xl rounded-tl-none shadow-sm w-full leading-relaxed text-base">
                    ${data.answer.replace(/\n/g, '<br>')}
                </div>
            </div>`;

        // 4. [สำคัญ] ส่วนแสดงรูปภาพ (Loop related_images)
        if (data.related_images && data.related_images.length > 0) {
            htmlContent += `
            <div class="ml-14 mt-4 w-full">
                <p class="text-xs font-bold text-slate-400 mb-3 uppercase tracking-wider flex items-center gap-1">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
                    ภาพประกอบจากเอกสาร
                </p>
                
                <div class="flex gap-4 overflow-x-auto pb-4 pt-2 scrollbar-hide snap-x">
                    ${data.related_images.map(img => `
                        <div class="relative flex-shrink-0 group cursor-pointer snap-center" onclick="window.open('${img.url}', '_blank')">
                            <div class="absolute -inset-0.5 bg-gradient-to-r from-brand-400 to-indigo-400 rounded-lg blur opacity-0 group-hover:opacity-30 transition duration-200"></div>
                            <div class="relative">
                                <img src="${img.url}" class="h-52 w-auto rounded-lg border border-slate-200 shadow-md object-contain bg-slate-50">
                                
                                <div class="absolute top-2 right-2 bg-slate-900/70 backdrop-blur-md text-white text-[10px] font-bold px-2 py-1 rounded-md border border-white/20 shadow-sm">
                                    Page ${img.page}
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>`;
        }
        
        htmlContent += `</div>`;
        chatBox.innerHTML += htmlContent;

    } catch (e) {
        console.error(e);
        const loadingElem = document.getElementById(loadingId);
        if (loadingElem) loadingElem.innerText = "Error: " + e;
    }
    
    // เลื่อนจอลงล่างสุด
    chatBox.scrollTop = chatBox.scrollHeight;
}

// ผูก Event Listener (ถ้ายังไม่ได้ทำใน HTML)
document.addEventListener('DOMContentLoaded', () => {
    const sendBtn = document.getElementById('sendBtn');
    const chatInput = document.getElementById('chatInput');
    
    if(sendBtn) sendBtn.addEventListener('click', askQuestion);
    if(chatInput) {
        chatInput.addEventListener('keypress', function (e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                askQuestion();
            }
        });
    }
});