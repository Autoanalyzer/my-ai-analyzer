// ai_chats.js (full, works with server.js sending { answer, sessionId, sources })

document.addEventListener('DOMContentLoaded', function () {
  const chatMessages = document.getElementById('chat-messages');
  const chatInput = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  const uploadBtn = document.getElementById('upload-btn');
  const imageUploadInput = document.getElementById('image-upload-input');
  const manualSelect = document.getElementById('manual-select');
  const imagePreviewContainer = document.getElementById('image-preview-container');

  // --- state ---
  let uploadedFile = null;
  let sessionId = loadOrCreateSessionId();

  // --- read area from URL ---
  const urlParams = new URLSearchParams(window.location.search);
  const selectedArea = urlParams.get('device'); // must match folder name under /documents

  // --- optionally populate manuals (value = real filename.pdf) ---
  maybePopulateManuals(selectedArea);

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
    const selectedManual = manualSelect?.value || '';

    if (userText === '' && !uploadedFile) return;

    // show user bubble
    appendMessage({ text: userText, image: uploadedFile }, 'user-message');

    // Build form-data payload (server expects multipart/form-data)
    const formData = new FormData();
    formData.append('question', userText);
    if (selectedManual) formData.append('manual', selectedManual); // should be real filename.pdf
    if (selectedArea) formData.append('area', selectedArea);
    if (sessionId) formData.append('sessionId', sessionId);
    if (uploadedFile) formData.append('image', uploadedFile, uploadedFile.name);

    // reset input/preview
    chatInput.value = '';
    imagePreviewContainer.innerHTML = '';
    uploadedFile = null;
    imageUploadInput.value = '';
    chatInput.placeholder = 'พิมพ์คำถามที่นี่...';

    showTypingIndicator();
    disableInput(true);

    try {
      // ❗ ใช้ relative path เพื่อให้ cookie session ถูกส่งอัตโนมัติ (ผ่าน checkAuth ได้)
      // ถ้าคุณ “จำเป็น” ต้องเรียกโดเมนอื่น ให้เปลี่ยนเป็น URL เต็มและใส่ { credentials: 'include' }
      const response = await fetch('/chat', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await safeJson(response);
        throw new Error(errorData?.error || `Server error: ${response.status} ${response.statusText}`);
      }

      const aiData = await response.json();

      // update session id (server may generate on first call)
      if (aiData.sessionId && aiData.sessionId !== sessionId) {
        sessionId = aiData.sessionId;
        localStorage.setItem('rag_session_id', sessionId);
      }

      removeTypingIndicator();
      appendMessage({ text: aiData.answer }, 'ai-message');

      // render sources if present
      if (Array.isArray(aiData.sources) && aiData.sources.length > 0) {
        appendSources(aiData.sources);
      }
    } catch (error) {
      console.error('Failed to fetch AI response:', error);
      removeTypingIndicator();
      appendMessage(
        { text: 'ขออภัยค่ะ เกิดข้อผิดพลาดในการเชื่อมต่อกับ AI: ' + error.message },
        'ai-message'
      );
    } finally {
      disableInput(false);
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

      // 1) Markdown link [text](url) -> <a>
      const markdownLinkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
      let processedText = String(content.text).replace(
        markdownLinkRegex,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
      );

      // 2) Newlines -> <br>
      processedText = processedText.replace(/\n/g, '<br>');

      textElement.innerHTML = processedText;
      textElement.style.margin = 0;
      messageDiv.appendChild(textElement);
    }

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendSources(sources) {
    const box = document.createElement('div');
    box.className = 'chat-message ai-message';
    box.style.paddingTop = '6px';

    const header = document.createElement('div');
    header.textContent = 'แหล่งอ้างอิง:';
    header.style.fontWeight = '600';
    header.style.marginBottom = '6px';
    box.appendChild(header);

    const list = document.createElement('ul');
    list.style.paddingLeft = '18px';
    list.style.margin = 0;

    sources.forEach((s, idx) => {
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = s.url; // already like /documents/<area>/<file>.pdf#page=N
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
      a.textContent = `${s.title} (หน้า ${s.page})`;
      li.appendChild(a);
      list.appendChild(li);
    });

    box.appendChild(list);
    chatMessages.appendChild(box);
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

  function disableInput(disabled) {
    sendBtn.disabled = disabled;
    chatInput.disabled = disabled;
    uploadBtn.disabled = disabled;
  }

  function loadOrCreateSessionId() {
    let id = localStorage.getItem('rag_session_id');
    if (!id) {
      id = `web_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
      localStorage.setItem('rag_session_id', id);
    }
    return id;
  }

  async function maybePopulateManuals(area) {
    // ถ้าไม่มี select หรือมี options อยู่แล้วก็ข้ามได้
    if (!manualSelect) return;

    // ถ้ามี option แล้ว และ value ลงท้าย .pdf แสดงว่าถูกต้องแล้ว
    const hasPdfOption = Array.from(manualSelect.options).some(opt => /\.pdf$/i.test(opt.value));
    if (hasPdfOption) return;

    try {
      const res = await fetch('/api/manuals', { method: 'GET' });
      if (!res.ok) return; // ถ้า 401/redirect แสดงว่าเพจนี้น่าจะเติม option อยู่แล้ว
      const data = await res.json();

      // โครงสร้าง: { pp11: { name:'PP11', files:[{name, path, displayName, image}, ...] }, ... }
      manualSelect.innerHTML = ''; // เคลียร์ของเดิม

      if (area && data[area.toLowerCase()]) {
        const group = data[area.toLowerCase()];
        group.files.forEach(f => {
          const opt = document.createElement('option');
          opt.value = f.name;            // <-- ใช้ชื่อไฟล์จริง .pdf
          opt.textContent = f.displayName || f.name;
          manualSelect.appendChild(opt);
        });
      } else {
        // ถ้าไม่รู้ area หรือหาไม่เจอ ให้แสดงทุกไฟล์
        Object.values(data).forEach(group => {
          group.files.forEach(f => {
            const opt = document.createElement('option');
            opt.value = f.name;          // <-- ใช้ชื่อไฟล์จริง .pdf
            opt.textContent = `${group.name} - ${f.displayName || f.name}`;
            manualSelect.appendChild(opt);
          });
        });
      }
    } catch (e) {
      console.warn('Failed to populate manuals from /api/manuals:', e);
    }
  }
});
