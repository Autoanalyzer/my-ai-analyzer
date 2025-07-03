// --- โค้ดสำหรับ AI Chat ---

// ตัวแปรสำหรับเก็บ ID ของห้องแชทปัจจุบัน
let currentSessionId = null;

document.addEventListener('DOMContentLoaded', () => {
    // หา form แชทในหน้าเว็บ
    const chatForm = document.getElementById('chat-form');
    // ถ้าเจอ form ให้ดักจับ event การกดส่ง
    if (chatForm) {
        chatForm.addEventListener('submit', handleSendMessage);
    }
});

// ฟังก์ชันหลักสำหรับจัดการการส่งข้อความ
async function handleSendMessage(event) {
    event.preventDefault(); // ป้องกันไม่ให้หน้ารีเฟรช

    // อ้างอิง element ที่ต้องใช้จากหน้าเว็บ
    const chatInput = document.getElementById('chat-input');
    const imageInput = document.getElementById('image-upload');
    const manualSelector = document.getElementById('manual-selector');

    const userInput = chatInput.value.trim();
    if (!userInput) return; // ถ้าไม่ได้พิมพ์อะไรมา ก็ไม่ต้องทำอะไร

    appendMessage(userInput, 'user-message');
    showLoadingIndicator(true);

    // เตรียมข้อมูลที่จะส่งไปให้เซิร์ฟเวอร์
    const formData = new FormData();
    formData.append('question', userInput);
    formData.append('manual', manualSelector.value);

    if (imageInput.files[0]) {
        formData.append('image', imageInput.files[0]);
    }

    // เพิ่ม Session ID เข้าไปในข้อมูลที่จะส่ง (ถ้ามี)
    if (currentSessionId) {
        formData.append('sessionId', currentSessionId);
    }

    try {
        // ส่งข้อมูลไปที่เซิร์ฟเวอร์
        const response = await fetch('https://my-ai-analyzer.onrender.com/chat', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Server responded with an error');
        }

        const data = await response.json();

        // รับ Session ID ใหม่จากเซิร์ฟเวอร์มาเก็บไว้
        currentSessionId = data.sessionId; 

        // แสดงคำตอบของ AI
        appendMessage(data.answer, 'ai-message');

    } catch (error) {
        console.error('Fetch Error:', error);
        appendMessage(`ขออภัยค่ะ เกิดข้อผิดพลาด: ${error.message}`, 'error-message');
    } finally {
        showLoadingIndicator(false);
        chatInput.value = '';
        imageInput.value = ''; 
    }
}

// ฟังก์ชันสำหรับเพิ่มข้อความลงในกล่องแชท
function appendMessage(text, className) {
    const chatMessages = document.getElementById('chat-messages');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', className);
    messageElement.innerText = text;
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ฟังก์ชันสำหรับแสดง/ซ่อนสถานะ "กำลังพิมพ์..."
function showLoadingIndicator(isLoading) {
    let loadingElement = document.getElementById('loading-indicator');
    if (isLoading) {
        if (!loadingElement) {
            const chatMessages = document.getElementById('chat-messages');
            loadingElement = document.createElement('div');
            loadingElement.id = 'loading-indicator';
            loadingElement.classList.add('message', 'ai-message');
            loadingElement.innerText = 'AI กำลังคิด...';
            chatMessages.appendChild(loadingElement);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    } else {
        if (loadingElement) {
            loadingElement.remove();
        }
    }
}

// --- โค้ดส่วนควบคุม UI อื่นๆ (โค้ดเดิมของคุณ) ---

const drawingModal = document.getElementById('drawingModal');
const pdfViewer = document.getElementById('pdf-viewer');
const interactiveDiagramModal = document.getElementById('interactiveDiagramModal');
const diagramImage = document.getElementById('interactive-diagram-image');
const tooltip = document.getElementById('custom-tooltip');
let currentZoom = 1.0;

function showArea(areaId) {
    document.querySelectorAll('.sub-area-container').forEach(area => { area.style.display = 'none'; });
    document.querySelectorAll('.device-content-container').forEach(content => { content.style.display = 'none'; });
    const selectedArea = document.getElementById(areaId + '-area');
    if (selectedArea) { selectedArea.style.display = 'block'; }
    document.querySelectorAll('.area-button').forEach(button => { button.classList.remove('active'); });
    const selectedButton = document.getElementById(areaId + '-btn');
    if(selectedButton) { selectedButton.classList.add('active'); }
}

function showDeviceContent(contentId) {
    const content = document.getElementById(contentId);
    if(content) {
        content.style.display = content.style.display === 'block' ? 'none' : 'block';
    }
}

function goToDetailsPage(equipmentId) {
    window.location.href = 'details.html?id=' + equipmentId;
}

function openDrawingModal(pdfPath) {
    if (pdfViewer) { pdfViewer.src = pdfPath; }
    if (drawingModal) { drawingModal.classList.add('is-visible'); }
}

function closeDrawingModal() {
    if (drawingModal) { drawingModal.classList.remove('is-visible'); }
    if (pdfViewer) { pdfViewer.src = ''; }
}

function openInteractiveDiagramModal() {
    if (interactiveDiagramModal) {
        interactiveDiagramModal.classList.add('is-visible');
    }
}

function closeInteractiveDiagramModal() {
    if (interactiveDiagramModal) {
        interactiveDiagramModal.classList.remove('is-visible');
    }
    resetZoom();
}

function showTooltip(event, equipmentId) {
    const data = equipmentDatabase[equipmentId];
    if (!data || !tooltip) return;
    const tooltipContent = `<strong>${data.name.split(',')[0]}</strong><br>Model: ${data.model}`;
    tooltip.innerHTML = tooltipContent;
    tooltip.style.display = 'block';
    moveTooltip(event);
}

function hideTooltip() {
    if (tooltip) {
        tooltip.style.display = 'none';
    }
}

function moveTooltip(event) {
     if (tooltip && tooltip.style.display === 'block') {
        tooltip.style.left = (event.pageX + 15) + 'px';
        tooltip.style.top = (event.pageY + 15) + 'px';
    }
}
document.addEventListener('mousemove', moveTooltip);

window.onclick = function(event) {
    if (event.target == drawingModal) { closeDrawingModal(); }
    if (event.target == interactiveDiagramModal) { closeInteractiveDiagramModal(); }
}

function applyZoom() { if (diagramImage) { diagramImage.style.transform = 'scale(' + currentZoom + ')'; } }
function zoomIn() { currentZoom += 0.2; applyZoom(); }
function zoomOut() { if (currentZoom > 0.4) { currentZoom -= 0.2; } applyZoom(); }
function resetZoom() { currentZoom = 1.0; applyZoom(); }

document.addEventListener('DOMContentLoaded', function() {
    const urlParams = new URLSearchParams(window.location.search);
    const showParam = urlParams.get('show');
    if (showParam === 'diagram') {
        showArea('pulp2');
        openInteractiveDiagramModal();
    } else if (showParam === 'ql-options') {
        showArea('pulp2');
        showDeviceContent('pulp2-ql-options');
    }
});