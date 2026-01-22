// Configuration
const API_BASE = "http://127.0.0.1:8000"; 
let currentDocId = null; // ตัวแปรสำคัญ เก็บ ID ของเอกสารที่กำลังคุยด้วย

document.addEventListener("DOMContentLoaded", () => {
    init();
});

function init() {
    loadDocuments();
    setupEventListeners();
}

// ============================================
// 1. EVENT LISTENERS
// ============================================
function setupEventListeners() {
    // ปุ่ม Upload
    document.getElementById("upload-form").addEventListener("submit", handleUpload);

    // ปุ่ม Send Chat
    document.getElementById("chat-form").addEventListener("submit", handleSendMessage);

    // ปุ่ม Back to Home
    document.getElementById("back-btn").addEventListener("click", () => {
        switchView('home');
        currentDocId = null;
    });
}

// ============================================
// 2. DOCUMENT MANAGEMENT (HOME VIEW)
// ============================================
async function loadDocuments() {
    const grid = document.getElementById("doc-grid");
    grid.innerHTML = '<p style="color:#888;">กำลังโหลดข้อมูล...</p>';

    try {
        const res = await fetch(`${API_BASE}/documents`);
        const data = await res.json();
        
        grid.innerHTML = ''; // Clear loading

        if (!data.documents || data.documents.length === 0) {
            grid.innerHTML = '<p>ยังไม่มีเอกสารในระบบ กรุณาอัปโหลด</p>';
            return;
        }

        // Loop สร้าง Card ตามโฟลเดอร์ที่เจอ
        data.documents.forEach(docName => {
            const card = document.createElement("div");
            card.className = "doc-card";
            card.onclick = () => openChat(docName); // คลิกแล้วเปิดแชทของเรื่องนั้น

            card.innerHTML = `
                <div class="doc-icon"><i class="fa-solid fa-book"></i></div>
                <div class="doc-title">${docName}</div>
                <div class="doc-meta">คลิกเพื่อเริ่มถาม-ตอบ</div>
            `;
            grid.appendChild(card);
        });

    } catch (err) {
        console.error(err);
        grid.innerHTML = '<p style="color:red;">ไม่สามารถเชื่อมต่อ Server ได้</p>';
    }
}

async function handleUpload(e) {
    e.preventDefault();
    
    const docIdInput = document.getElementById("doc-id-input");
    const fileInput = document.getElementById("file-input");
    
    const docId = docIdInput.value.trim();
    const file = fileInput.files[0];

    if (!docId || !file) {
        alert("กรุณากรอกชื่อโปรเจกต์และเลือกไฟล์");
        return;
    }

    // แสดง Loading
    showLoading(true, `กำลัง Ingest ข้อมูล "${docId}"... (อาจใช้เวลาสักครู่)`);

    const formData = new FormData();
    formData.append("doc_id", docId);
    formData.append("file", file);
    formData.append("use_ocr", "true"); 

    try {
        const res = await fetch(`${API_BASE}/upload`, {
            method: "POST",
            body: formData
        });

        if (!res.ok) throw new Error("Upload failed");

        const result = await res.json();
        alert("Success: " + result.message);
        
        // Reset Form & Reload Grid
        docIdInput.value = "";
        fileInput.value = "";
        loadDocuments();

    } catch (err) {
        alert("Error: " + err.message);
    } finally {
        showLoading(false);
    }
}

// ============================================
// 3. CHAT LOGIC (CHAT VIEW)
// ============================================
function openChat(docId) {
    currentDocId = docId; // Set Context
    document.getElementById("current-doc-name").textContent = docId;
    
    // Clear Chat History (Start Fresh)
    const history = document.getElementById("chat-history");
    history.innerHTML = `
        <div class="message bot-message">
            <div class="bubble">
                เข้าสู่โหมดสนทนาสำหรับเอกสาร: <strong>${docId}</strong><br>
                ถามมาได้เลยครับ!
            </div>
        </div>
    `;

    switchView('chat');
}

async function handleSendMessage(e) {
    e.preventDefault();
    const input = document.getElementById("chat-input");
    const message = input.value.trim();
    if (!message) return;

    // 1. แสดง User Message
    addMessageToUI("user", message);
    input.value = "";

    // 2. แสดง Bot Typing...
    const loadingId = addMessageToUI("bot", "...", true);

    try {
        // 3. เรียก API พร้อมส่ง doc_ids
        const payload = {
            query: message,
            doc_ids: [currentDocId] // <--- KEY: ส่งไปบอก Backend ว่าคุยเรื่องนี้เรื่องเดียว
        };

        const res = await fetch(`${API_BASE}/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();

        // 4. ลบ Typing -> แสดงคำตอบจริง
        removeMessage(loadingId);
        
        let answerHTML = marked.parse(data.answer); // แปลง Markdown เป็น HTML

        // ถ้ามีรูปภาพแนบมา
        if (data.related_images && data.related_images.length > 0) {
            answerHTML += `<div style="margin-top:10px; font-weight:bold; font-size:0.9rem;">รูปภาพที่เกี่ยวข้อง:</div>`;
            data.related_images.forEach(img => {
                // img.url คือ path สัมพัทธ์จาก server (เช่น /ingested/...)
                const fullUrl = `${API_BASE}${img.url}`;
                answerHTML += `<img src="${fullUrl}" class="chat-image" alt="reference image">`;
            });
        }

        addMessageToUI("bot", answerHTML);

    } catch (err) {
        removeMessage(loadingId);
        addMessageToUI("bot", "เกิดข้อผิดพลาดในการเชื่อมต่อ: " + err.message);
    }
}

// Helper: สร้าง Bubble
function addMessageToUI(sender, htmlContent, isLoading = false) {
    const history = document.getElementById("chat-history");
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${sender}-message`;
    
    // สร้าง ID สำหรับลบทีหลัง (กรณี Loading)
    const msgId = "msg-" + Date.now();
    msgDiv.id = msgId;

    msgDiv.innerHTML = `<div class="bubble">${htmlContent}</div>`;
    history.appendChild(msgDiv);
    history.scrollTop = history.scrollHeight; // Auto scroll to bottom

    return isLoading ? msgId : null;
}

function removeMessage(id) {
    if(!id) return;
    const el = document.getElementById(id);
    if(el) el.remove();
}

// ============================================
// 4. UTILS
// ============================================
function switchView(viewName) {
    const homeView = document.getElementById("home-view");
    const chatView = document.getElementById("chat-view");

    if (viewName === 'home') {
        homeView.classList.remove("hidden");
        chatView.classList.add("hidden");
    } else {
        homeView.classList.add("hidden");
        chatView.classList.remove("hidden");
        // Focus input
        setTimeout(() => document.getElementById("chat-input").focus(), 100);
    }
}

function showLoading(show, text = "") {
    const overlay = document.getElementById("loading-overlay");
    const textEl = document.getElementById("loading-text");
    if (show) {
        textEl.textContent = text;
        overlay.classList.remove("hidden");
    } else {
        overlay.classList.add("hidden");
    }
}