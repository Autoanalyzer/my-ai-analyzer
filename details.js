window.onload = function() {
    const urlParams = new URLSearchParams(window.location.search);
    const equipmentId = urlParams.get('id');
    const nameElement = document.getElementById('detail-name');
    const modelElement = document.getElementById('detail-model');
    const serialElement = document.getElementById('detail-serial');
    const checklistElement = document.getElementById('detail-checklist');
    const manualElement = document.getElementById('detail-manual');
    const aiButton = document.getElementById('ai-analyze-btn');

    let currentEquipment = null;

    if (equipmentId && equipmentDatabase[equipmentId]) {
        const data = equipmentDatabase[equipmentId];
        currentEquipment = data;
        
        nameElement.textContent = data.name;
        modelElement.textContent = data.model;
        serialElement.textContent = data.serial;
        manualElement.href = data.manualUrl;
        
        checklistElement.innerHTML = '';
        data.checklist.forEach(item => { const li = document.createElement('li'); li.textContent = item; checklistElement.appendChild(li); });
    } else {
        nameElement.textContent = "ไม่พบข้อมูลอุปกรณ์";
        if(aiButton) aiButton.style.display = 'none';
    }

    if(aiButton) {
        aiButton.addEventListener('click', async () => {
            if (!currentEquipment) return;

            aiButton.textContent = 'กำลังวิเคราะห์...';
            aiButton.disabled = true;

            try {
                const response = await fetch(`http://localhost:5500/get-pm-checklist?name=${encodeURIComponent(currentEquipment.name)}&model=${encodeURIComponent(currentEquipment.model)}`);
                const aiData = await response.json();

                if (response.ok && aiData.checklist) {
                    checklistElement.innerHTML = '';
                    const items = aiData.checklist.split('\n');
                    items.forEach(item => {
                        if (item.trim().match(/^\d+\./) || item.trim().startsWith('-')) {
                            // ... (อยู่ภายใน forEach loop)
const li = document.createElement('li');

// 1. นำข้อความที่ยังไม่ตัด marker (เช่น '1.' หรือ '-') ออก มาเตรียมไว้
let rawItemText = item.trim();

// 2. สร้าง Regular Expression เพื่อค้นหา Markdown Link เช่น [text](url)
const markdownLinkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;

// 3. แปลง Markdown Link ที่พบในข้อความให้เป็น HTML Tag <a>
// และใส่ target="_blank" เพื่อให้ลิงก์เปิดในแท็บใหม่
const itemWithHtmlLink = rawItemText.replace(markdownLinkRegex, '<a href="$2" target="_blank">$1</a>');

// 4. ใช้ innerHTML เพื่อแสดงผล HTML ที่เราเพิ่งสร้างขึ้น
// พร้อมกับตัด list marker ออกในตอนท้าย
li.innerHTML = itemWithHtmlLink.replace(/^\d+\.\s*|-\s*/, '').trim();

checklistElement.appendChild(li);
// ...
                        }
                    });
                } else {
                     throw new Error(aiData.error || 'Unknown error from AI service');
                }
            } catch (error) {
                console.error('Failed to fetch AI checklist:', error);
                alert('ไม่สามารถเรียกข้อมูลจาก AI ได้: ' + error.message);
            } finally {
                aiButton.textContent = 'วิเคราะห์ด้วย AI';
                aiButton.disabled = false;
            }
        });
    }
};