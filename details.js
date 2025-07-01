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
                const response = await fetch(`http://localhost:4000/get-pm-checklist?name=${encodeURIComponent(currentEquipment.name)}&model=${encodeURIComponent(currentEquipment.model)}`);
                const aiData = await response.json();

                if (response.ok && aiData.checklist) {
                    checklistElement.innerHTML = '';
                    const items = aiData.checklist.split('\n');
                    items.forEach(item => {
                        if (item.trim().match(/^\d+\./) || item.trim().startsWith('-')) {
                            const li = document.createElement('li');
                            li.textContent = item.trim().replace(/^\d+\.\s*|-s*/, '').trim();
                            checklistElement.appendChild(li);
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