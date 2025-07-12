document.addEventListener('DOMContentLoaded', function () {
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const imageUploadInput = document.getElementById('image-upload-input');
    const manualSelect = document.getElementById('manual-select');
    const imagePreviewContainer = document.getElementById('image-preview-container');

    let uploadedFile = null;

    // --- Event Listeners ---
    sendBtn.addEventListener('click', handleSendMessage);

    chatInput.addEventListener('keypress', function (event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleSendMessage();
        }
    });

    uploadBtn.addEventListener('click', () => imageUploadInput.click());
    imageUploadInput.addEventListener('change', handleImageUpload);

    // --- Core logic ---
    async function handleSendMessage() {
        const userText = chatInput.value.trim();
        const selectedManual = manualSelect.value;

        // ✨ 1. ดึงค่า area จาก URL ของหน้าเว็บ
        const urlParams = new URLSearchParams(window.location.search);
        const selectedArea = urlParams.get('device'); // 'device' parameter holds the area name

        if (userText === '' && !uploadedFile) return;

        const userMessageContent = { text: userText, image: uploadedFile };
        appendMessage(userMessageContent, 'user-message');

        // Build form-data payload
        const formData = new FormData();
        formData.append('question', userText);
        formData.append('manual', selectedManual);

        // ✨ 2. เพิ่ม area เข้าไปในข้อมูลที่จะส่ง
        if (selectedArea) {
            formData.append('area', selectedArea);
        }

        if (uploadedFile) {
            formData.append('image', uploadedFile, uploadedFile.name);
        }

        // Reset input fields / preview
        chatInput.value = '';
        imagePreviewContainer.innerHTML = '';
        uploadedFile = null;
        imageUploadInput.value = '';

        showTypingIndicator();

        try {
           const response = await fetch('https://my-ai-analyzer.onrender.com/chat', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Server error: ${response.statusText}`);
            }

            const aiData = await response.json();
            removeTypingIndicator();
            appendMessage({ text: aiData.answer }, 'ai-message');

        } catch (error) {
            console.error('Failed to fetch AI response:', error);
            removeTypingIndicator();
            appendMessage(
                { text: 'ขออภัยค่ะ เกิดข้อผิดพลาดในการเชื่อมต่อกับ AI: ' + error.message },
                'ai-message'
            );
        }
    }

    function handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        uploadedFile = file;
        const reader = new FileReader();
        reader.onload = function (e) {
            imagePreviewContainer.innerHTML =
                `<img src="${e.target.result}" alt="Preview" style="max-width: 100px; max-height: 100px; border-radius: 5px; margin-right: 10px;"> <span>${file.name}</span>`;
        };
        reader.readAsDataURL(file);
        chatInput.placeholder = 'อธิบายเกี่ยวกับรูปภาพนี้...';
    }

    function appendMessage(content, className) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${className}`;

        if (content.image) {
            const imgElement = document.createElement('img');
            imgElement.src = URL.createObjectURL(content.image);
            imgElement.style.maxWidth = '200px';
            imgElement.style.borderRadius = '10px';
            imgElement.style.marginBottom = '10px';
            messageDiv.appendChild(imgElement);
        }

        if (content.text) {
            const textElement = document.createElement('p');
            textElement.innerHTML = content.text.replace(/\n/g, '<br>');
            textElement.style.margin = 0;
            messageDiv.appendChild(textElement);
        }

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.id = 'typing-indicator';
        typingDiv.className = 'chat-message ai-message';
        typingDiv.textContent = 'AI กำลังค้นหาข้อมูลในคู่มือ...';
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function removeTypingIndicator() {
        const typingDiv = document.getElementById('typing-indicator');
        if (typingDiv) typingDiv.remove();
    }
});