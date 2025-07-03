// --- อ้างอิง Element ต่างๆ ---
const drawingModal = document.getElementById('drawingModal');
const pdfViewer = document.getElementById('pdf-viewer');
const interactiveDiagramModal = document.getElementById('interactiveDiagramModal');
const diagramImage = document.getElementById('interactive-diagram-image');
const tooltip = document.getElementById('custom-tooltip');
let currentZoom = 1.0;

// --- ฟังก์ชันควบคุมหน้าเว็บ ---
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

// --- ฟังก์ชันสำหรับ Modal ที่ยังเหลืออยู่ ---
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

// --- ฟังก์ชันควบคุม Tooltip ---
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


// --- ฟังก์ชันอื่นๆ ---
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