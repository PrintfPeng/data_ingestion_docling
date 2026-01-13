const API_BASE = "http://localhost:8000"; // ตรวจสอบ Port ให้ตรงกับ Backend
let selectedDocId = null;

document.addEventListener('DOMContentLoaded', () => {
    fetchDocumentsList();
    
    // Event Listeners
    document.getElementById('fileInput').addEventListener('change', handleFileUpload);
    document.getElementById('sendBtn').addEventListener('click', handleSendMessage);
    
    const chatInput = document.getElementById('chatInput');
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
        // Auto-resize textarea
        e.target.style.height = 'auto';
        e.target.style.height = e.target.scrollHeight + 'px';
    });
});

// --- Document Management ---

async function fetchDocumentsList() {
    const listContainer = document.getElementById('documentList');
    try {
        const res = await fetch(`${API_BASE}/documents`);
        const data = await res.json();
        
        listContainer.innerHTML = '';
        
        // Option: Search All
        const allDocsItem = createDocItem('Search All Documents', null);
        listContainer.appendChild(allDocsItem);

        data.documents.forEach(doc => {
            listContainer.appendChild(createDocItem(doc, doc));
        });
        
        // Default selection logic
        if (!selectedDocId) selectDoc(null); // Select 'All' by default

    } catch (err) {
        listContainer.innerHTML = `<div class="text-red-400 text-xs p-2">Error loading docs</div>`;
        console.error(err);
    }
}

function createDocItem(label, id) {
    const div = document.createElement('div');
    div.className = `doc-item p-3 rounded-lg cursor-pointer text-sm font-medium text-slate-600 hover:bg-slate-100 transition-all flex items-center gap-2 border-l-4 border-transparent`;
    div.innerHTML = `<i class="fa-regular ${id ? 'fa-file-pdf' : 'fa-folder-open'}"></i> <span class="truncate">${label}</span>`;
    div.onclick = () => selectDoc(id);
    div.dataset.id = id || 'all';
    return div;
}

function selectDoc(docId) {
    selectedDocId = docId;
    
    // Update UI List
    document.querySelectorAll('.doc-item').forEach(el => {
        if (el.dataset.id === (docId || 'all')) el.classList.add('active');
        else el.classList.remove('active');
    });

    // Update Badge
    const badge = document.getElementById('selectedDocBadge');
    const badgeName = document.getElementById('currentDocName');
    
    if (docId) {
        badge.classList.remove('hidden');
        badgeName.innerText = docId;
    } else {
        badge.classList.add('hidden');
    }
}

function clearSelection() {
    selectDoc(null);
}

// --- Upload ---

async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const statusDiv = document.getElementById('uploadStatus');
    statusDiv.innerHTML = `<span class="text-indigo-600 animate-pulse"><i class="fa-solid fa-spinner fa-spin"></i> Uploading...</span>`;
    
    const formData = new FormData();
    formData.append("file", file);
    // สร้าง doc_id จากชื่อไฟล์ (ตัดนามสกุล, แทนที่ space ด้วย _)
    const docId = file.name.replace(/\.[^/.]+$/, "").replace(/\s+/g, "_");
    formData.append("doc_id", docId);

    try {
        const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
        if (!res.ok) throw new Error("Upload failed");
        
        statusDiv.innerHTML = `<span class="text-green-600"><i class="fa-solid fa-check-circle"></i> Complete!</span>`;
        await fetchDocumentsList();
        selectDoc(docId); // Auto select uploaded file
        
        setTimeout(() => statusDiv.innerHTML = "", 3000);
    } catch (err) {
        statusDiv.innerHTML = `<span class="text-red-500">Error: ${err.message}</span>`;
    }
    e.target.value = ''; // Reset input
}

// --- Chat Logic ---

async function handleSendMessage() {
    const chatInput = document.getElementById('chatInput');
    const message = chatInput.value.trim();
    if (!message) return;

    // 1. User Message
    addMessageToUI('user', message);
    chatInput.value = '';
    chatInput.style.height = 'auto'; // Reset height

    // 2. Loading State
    const loadingId = addLoadingBubble();

    try {
        // 3. API Call
        const payload = { 
            query: message,
            doc_ids: selectedDocId ? [selectedDocId] : null 
        };
        
        const res = await fetch(`${API_BASE}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const data = await res.json();
        
        // 4. Remove Loading & Render Response
        document.getElementById(loadingId).remove();
        addMessageToUI('ai', data.answer, data.related_images);

    } catch (err) {
        document.getElementById(loadingId).remove();
        addMessageToUI('ai', `❌ เกิดข้อผิดพลาด: ${err.message}`);
    }
}

function addMessageToUI(role, text, images = []) {
    const container = document.getElementById('chatContainer');
    const isUser = role === 'user';
    
    const wrapper = document.createElement('div');
    wrapper.className = `flex gap-4 max-w-4xl mx-auto ${isUser ? 'flex-row-reverse' : ''} animate-fade-in-up`;
    
    // Avatar
    const avatar = document.createElement('div');
    avatar.className = `w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${isUser ? 'bg-slate-700 text-white' : 'bg-indigo-100 text-indigo-600'}`;
    avatar.innerHTML = isUser ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
    
    // Bubble
    const bubble = document.createElement('div');
    bubble.className = `relative p-5 rounded-2xl shadow-sm text-sm leading-relaxed max-w-[85%] md:max-w-[75%] ${
        isUser 
        ? 'bg-slate-800 text-white rounded-tr-none' 
        : 'bg-white border border-gray-200 text-slate-700 rounded-tl-none prose-content'
    }`;
    
    // Text Content (Basic formatting)
    // แปลง \n เป็น <br> และ **text** เป็น <b>
    let formattedText = text.replace(/\n/g, '<br>');
    formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    bubble.innerHTML = `<div>${formattedText}</div>`;

    // --- Image Gallery Rendering ---
    if (images && images.length > 0) {
        const galleryDiv = document.createElement('div');
        galleryDiv.className = "mt-4 pt-3 border-t border-gray-100";
        galleryDiv.innerHTML = `<p class="text-xs font-bold text-slate-400 mb-2 flex items-center gap-1"><i class="fa-solid fa-images"></i> RELATED IMAGES</p>`;
        
        const grid = document.createElement('div');
        grid.className = "grid grid-cols-2 sm:grid-cols-3 gap-2";
        
        images.forEach(img => {
            // Construct Image URL
            // Backend sends: /ingested/{doc_id}/images/{filename}
            // We append API_BASE just in case, or relative if hosted same origin.
            const imgUrl = `${API_BASE}${img.url}`;
            
            const imgCard = document.createElement('div');
            imgCard.className = "group relative aspect-square bg-gray-100 rounded-lg overflow-hidden cursor-pointer border border-gray-200 hover:shadow-md transition-all";
            imgCard.onclick = () => openModal(imgUrl, `Page ${img.page} • Doc: ${img.doc_id}`);
            
            imgCard.innerHTML = `
                <img src="${imgUrl}" class="w-full h-full object-contain p-1 group-hover:scale-105 transition-transform duration-300" loading="lazy" onerror="this.src='https://placehold.co/400x400?text=Image+Not+Found'">
                <div class="absolute bottom-0 left-0 right-0 bg-black/60 text-white text-[10px] p-1 text-center opacity-0 group-hover:opacity-100 transition-opacity">
                    Page ${img.page}
                </div>
            `;
            grid.appendChild(imgCard);
        });
        
        galleryDiv.appendChild(grid);
        bubble.appendChild(galleryDiv);
    }

    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);
    container.appendChild(wrapper);
    
    // Scroll to bottom
    container.scrollTop = container.scrollHeight;
}

function addLoadingBubble() {
    const container = document.getElementById('chatContainer');
    const id = `loading-${Date.now()}`;
    
    const wrapper = document.createElement('div');
    wrapper.id = id;
    wrapper.className = `flex gap-4 max-w-4xl mx-auto animate-fade-in-up`;
    
    wrapper.innerHTML = `
        <div class="w-10 h-10 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center flex-shrink-0">
            <i class="fa-solid fa-robot"></i>
        </div>
        <div class="bg-white border border-gray-200 p-4 rounded-2xl rounded-tl-none shadow-sm flex items-center gap-2">
            <div class="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style="animation-delay: 0s"></div>
            <div class="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
            <div class="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style="animation-delay: 0.4s"></div>
            <span class="text-xs text-slate-400 ml-2">Searching & Analyzing...</span>
        </div>
    `;
    
    container.appendChild(wrapper);
    container.scrollTop = container.scrollHeight;
    return id;
}

// --- Lightbox Modal ---

function openModal(src, caption) {
    const modal = document.getElementById('imageModal');
    const img = document.getElementById('modalImage');
    const cap = document.getElementById('modalCaption');
    
    img.src = src;
    cap.innerText = caption;
    
    modal.classList.remove('hidden');
    // Trigger reflow for transition
    void modal.offsetWidth; 
    modal.classList.add('modal-open');
}

function closeModal() {
    const modal = document.getElementById('imageModal');
    modal.classList.remove('modal-open');
    setTimeout(() => {
        modal.classList.add('hidden');
        document.getElementById('modalImage').src = '';
    }, 300);
}

// Close modal on click outside
document.getElementById('imageModal').addEventListener('click', (e) => {
    if (e.target.id === 'imageModal') closeModal();
});