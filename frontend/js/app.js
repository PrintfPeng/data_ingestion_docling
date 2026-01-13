// frontend/js/app.js
const API_BASE = "http://localhost:8000";

document.addEventListener('DOMContentLoaded', () => {
    // Event Listeners
    const uploadBtn = document.getElementById('uploadBtn');
    const fileInput = document.getElementById('fileInput');
    const sendBtn = document.getElementById('sendBtn');
    const chatInput = document.getElementById('chatInput');

    if (uploadBtn) uploadBtn.addEventListener('click', () => fileInput.click());
    if (fileInput) fileInput.addEventListener('change', handleFileUpload);
    if (sendBtn) sendBtn.addEventListener('click', handleSendMessage);
    if (chatInput) {
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
            }
        });
    }
    fetchDocumentsList();
});

async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const statusDiv = document.getElementById('attachmentInfo');
    statusDiv.classList.remove('hidden');
    statusDiv.innerText = `⏳ Uploading ${file.name}...`;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("doc_id", file.name.replace(/\.[^/.]+$/, "").replace(/\s+/g, "_"));

    try {
        await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
        statusDiv.innerText = `✅ Uploaded!`;
        fetchDocumentsList();
        setTimeout(() => statusDiv.classList.add('hidden'), 3000);
    } catch (err) {
        statusDiv.innerText = `❌ Error: ${err}`;
    }
}

async function handleSendMessage() {
    const chatInput = document.getElementById('chatInput');
    const message = chatInput.value.trim();
    if (!message) return;

    appendMessage('user', message);
    chatInput.value = '';

    const loadingId = appendMessage('ai', '<span class="animate-pulse">Thinking...</span>', true);

    try {
        const docSelect = document.getElementById('docSelect');
        const res = await fetch(`${API_BASE}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                query: message,
                doc_ids: docSelect.value ? [docSelect.value] : null 
            })
        });
        const data = await res.json();
        document.getElementById(loadingId).remove();
        renderResponse(data);
    } catch (err) {
        document.getElementById(loadingId).innerText = "Error: " + err;
    }
}

function renderResponse(data) {
    let html = `<div>${data.answer.replace(/\n/g, '<br>')}</div>`;

    // [จุดสำคัญที่ 4] แสดงรูปภาพ
    if (data.related_images && data.related_images.length > 0) {
        html += `
        <div class="mt-4 pt-3 border-t border-slate-100">
            <p class="text-xs font-bold text-slate-400 mb-2 uppercase">📸 Images Found:</p>
            <div class="flex gap-3 overflow-x-auto pb-2 scrollbar-hide">
                ${data.related_images.map(img => `
                    <div class="relative flex-shrink-0 cursor-pointer group" onclick="window.open('${API_BASE}${img.url}', '_blank')">
                        <img src="${API_BASE}${img.url}" class="h-40 w-auto rounded-lg border border-slate-200 shadow-sm object-contain bg-white group-hover:shadow-md transition-all">
                        <span class="absolute top-2 right-2 bg-black/70 text-white text-[10px] px-2 py-0.5 rounded-full backdrop-blur-sm">Page ${img.page}</span>
                    </div>
                `).join('')}
            </div>
        </div>`;
    }
    appendMessage('ai', html, true);
}

function appendMessage(role, text, isHtml) {
    const chatBox = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.id = `msg-${Date.now()}`;
    div.className = `flex mb-4 ${role === 'user' ? 'justify-end' : 'justify-start'}`;
    
    const bubble = document.createElement('div');
    bubble.className = role === 'user' 
        ? "bg-blue-600 text-white px-4 py-2 rounded-2xl rounded-tr-none max-w-[85%] shadow-md" 
        : "bg-white border border-slate-200 text-slate-700 px-5 py-3 rounded-2xl rounded-tl-none max-w-[85%] shadow-sm leading-relaxed";
    
    if (isHtml) bubble.innerHTML = text; else bubble.innerText = text;
    
    div.appendChild(bubble);
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
    return div.id;
}

async function fetchDocumentsList() {
    try {
        const res = await fetch(`${API_BASE}/documents`);
        const data = await res.json();
        const select = document.getElementById('docSelect');
        if(select && data.documents) {
            const current = select.value;
            select.innerHTML = '<option value="">📚 ค้นหาทุกเอกสาร</option>';
            data.documents.forEach(doc => {
                const opt = document.createElement('option');
                opt.value = doc;
                opt.textContent = doc;
                select.appendChild(opt);
            });
            select.value = current;
        }
    } catch(e) {}
}