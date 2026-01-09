const API_BASE = "http://localhost:8000";

async function uploadFile() {
    const fileInput = document.getElementById('pdfFile');
    const file = fileInput.files[0];
    if (!file) return alert("Please select a file");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("doc_id", file.name.replace(".pdf", ""));

    document.getElementById('status').innerText = "Uploading & Ingesting...";
    
    try {
        const res = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        document.getElementById('status').innerText = "Done: " + data.message;
    } catch (e) {
        document.getElementById('status').innerText = "Error: " + e;
    }
}

async function askQuestion() {
    const question = document.getElementById('question').value;
    if (!question) return;

    // Clear previous results
    const answerDiv = document.getElementById('answer');
    const imagesDiv = document.getElementById('relevant-images'); // ต้องมี div นี้ใน HTML หรือสร้างใหม่
    answerDiv.innerText = "Thinking...";
    if(imagesDiv) imagesDiv.innerHTML = "";

    try {
        const res = await fetch(`${API_BASE}/ask`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ question })
        });
        const data = await res.json();
        
        // 1. Show Answer
        answerDiv.innerText = data.answer;
        
        // 2. Show Relevant Images
        if (data.sources && data.sources.length > 0) {
            const imgContainer = document.createElement('div');
            imgContainer.style.marginTop = "15px";
            imgContainer.innerHTML = "<strong>📸 Relevant Images:</strong><br/>";
            
            let foundImage = false;
            
            data.sources.forEach(source => {
                // เช็คว่าเป็น Source แบบ 'image' และมี path
                if (source.source === 'image' && source.metadata && source.metadata.file_path) {
                    foundImage = true;
                    
                    // แปลง Path เป็น URL (เช่น ingested/doc1/images/img.png -> /ingested/doc1/images/img.png)
                    // ต้องระวัง Backslash ใน Windows
                    let imgUrl = source.metadata.file_path.replace(/\\/g, '/');
                    if (!imgUrl.startsWith('/')) imgUrl = '/' + imgUrl;
                    
                    const imgElem = document.createElement('img');
                    imgElem.src = API_BASE + imgUrl; 
                    imgElem.style.maxWidth = "300px";
                    imgElem.style.margin = "10px";
                    imgElem.style.border = "1px solid #ccc";
                    imgElem.style.borderRadius = "8px";
                    imgElem.title = source.content; // Show description on hover
                    
                    imgContainer.appendChild(imgElem);
                }
            });
            
            if (foundImage) {
                answerDiv.appendChild(imgContainer);
            }
        }

    } catch (e) {
        answerDiv.innerText = "Error: " + e;
    }
}

// ผูก Event Listeners (ถ้าใน HTML ไม่ได้ผูกไว้)
// document.getElementById('uploadBtn').onclick = uploadFile;
// document.getElementById('askBtn').onclick = askQuestion;