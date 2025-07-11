// server.js (Complete Version with Login, Manual Viewer, and AI Chatbot)

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs/promises');
const pdf = require('pdf-parse');
const path = require('path');
const session = require('express-session');

const {
  GoogleGenerativeAI,
  HarmCategory,
  HarmBlockThreshold,
} = require('@google/generative-ai');
const { GoogleGenerativeAIEmbeddings } = require('@langchain/google-genai');
const { MemoryVectorStore } = require('langchain/vectorstores/memory');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');

const app = express();
const port = process.env.PORT || 5500;

// --- User Management ---
const users = [
    { id: 1, username: 'admin', password: 'password123' },
    { id: 2, username: 'user', password: 'password456' }
];

// --- Session Configuration ---
app.use(session({
    secret: 'your_super_secret_key_change_this_in_production',
    resave: false,
    saveUninitialized: false,
    cookie: { secure: false, maxAge: 60 * 60 * 1000 } // 1 hour
}));

const chatHistories = {};
let vectorStore;

const VECTOR_STORE_SAVE_PATH = path.join(__dirname, 'vector_store.json');

// --- Middleware ---
app.use(cors({
    origin: true,
    credentials: true
}));
app.use(express.json());

// Debug middleware
app.use((req, res, next) => {
    console.log(`[DEBUG] ${new Date().toISOString()} - ${req.method} ${req.originalUrl}`);
    next();
});

// --- Authentication Middleware ---
const checkAuth = (req, res, next) => {
    console.log('[DEBUG] Checking authentication...');
    console.log('[DEBUG] Session ID:', req.session.id);
    console.log('[DEBUG] User ID in session:', req.session.userId);

    if (!req.session.userId) {
        console.log('[DEBUG] User not authenticated, redirecting to login');
        return res.redirect('/login.html');
    }
    
    console.log('[DEBUG] User authenticated, allowing access');
    next();
};

// --- File Upload Configuration ---
const upload = multer({ storage: multer.memoryStorage() });

// --- AI Configuration ---
const safetySettings = [
  { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
];

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const generativeModel = genAI.getGenerativeModel({
  model: 'gemini-2.0-flash',
  safetySettings,
});
const embeddingsModel = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
  model: 'text-embedding-004',
});

// --- Vector Store Initialization ---
async function initializeVectorStore() {
  try {
    console.log(`🔍 Checking for saved vector store at: ${VECTOR_STORE_SAVE_PATH}`);
    const savedData = await fs.readFile(VECTOR_STORE_SAVE_PATH, 'utf-8');
    const memoryVectors = JSON.parse(savedData);

    const documents = memoryVectors.map(mv => ({ pageContent: mv.content, metadata: mv.metadata }));
    const embeddings = memoryVectors.map(mv => mv.embedding);

    vectorStore = new MemoryVectorStore(embeddingsModel);
    await vectorStore.addVectors(embeddings, documents);

    console.log('✅ Vector store loaded successfully from disk.');
  } catch (error) {
    console.log('📚 Saved vector store not found. Building from scratch...');
    const documentsBasePath = path.join(__dirname, 'documents');
    const allDocuments = [];

    try {
        const areaFolders = (await fs.readdir(documentsBasePath, { withFileTypes: true }))
        .filter(d => d.isDirectory())
        .map(d => d.name);

      for (const area of areaFolders) {
        const areaPath = path.join(documentsBasePath, area);
        const files = await fs.readdir(areaPath);
        
        console.log(`📁 Processing area: ${area}`);
        
        for (const file of files) {
          const filePath = path.join(areaPath, file);
          const fileExt = path.extname(file).toLowerCase();
          let textContent = null;

          try {
            if (fileExt === '.pdf') {
              const dataBuffer = await fs.readFile(filePath);
              const pdfData = await pdf(dataBuffer);
              textContent = pdfData.text;
            } else if (fileExt === '.txt') {
              textContent = await fs.readFile(filePath, 'utf-8');
            }

            if (textContent) {
              allDocuments.push({
                pageContent: textContent,
                metadata: { source: file.trim(), area: area.trim() },
              });
              console.log(`📄 Processed: ${file}`);
            }
          } catch (fileError) {
            console.error(`❌ Could not process file: ${file}`, fileError.message);
          }
        }
      }

      console.log(`📊 Total documents processed: ${allDocuments.length}`);

      const textSplitter = new RecursiveCharacterTextSplitter({ 
        chunkSize: 1000, 
        chunkOverlap: 200 
      });
      const splitDocs = await textSplitter.splitDocuments(allDocuments);

      console.log(`🔄 Embedding ${splitDocs.length} document chunks in batches...`);
      const batchSize = 50;
      const delay = 1000;

      vectorStore = new MemoryVectorStore(embeddingsModel);

      for (let i = 0; i < splitDocs.length; i += batchSize) {
        const batch = splitDocs.slice(i, i + batchSize);
        await vectorStore.addDocuments(batch);
        const batchNum = Math.floor(i / batchSize) + 1;
        const totalBatches = Math.ceil(splitDocs.length / batchSize);
        console.log(`⏳ Processed batch ${batchNum} of ${totalBatches}...`);
        
        if (i + batchSize < splitDocs.length) {
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
      
      await fs.writeFile(VECTOR_STORE_SAVE_PATH, JSON.stringify(vectorStore.memoryVectors, null, 2));
      console.log(`✅ Vector store initialized and saved to: ${VECTOR_STORE_SAVE_PATH}`);

    } catch (buildError) {
      console.error('❌ CRITICAL: Failed to build vector store.', buildError);
      vectorStore = undefined;
    }
  }
}

// --- Authentication Routes ---
app.post('/login', (req, res) => {
    const { username, password } = req.body;
    
    console.log(`[LOGIN] Attempt for username: ${username}`);
    
    const user = users.find(u => u.username === username && u.password === password);

    if (user) {
        req.session.userId = user.id;
        req.session.username = user.username;
        console.log(`[LOGIN] Success for user: ${username}`);
        return res.json({ 
            message: 'Login successful',
            username: user.username 
        });
    }

    console.log(`[LOGIN] Failed for username: ${username}`);
    return res.status(401).json({ error: 'Invalid username or password' });
});

app.get('/logout', (req, res) => {
    const username = req.session.username;
    req.session.destroy(err => {
        if (err) {
            console.error('[LOGOUT] Error destroying session:', err);
            return res.redirect('/index.html');
        }
        res.clearCookie('connect.sid');
        console.log(`[LOGOUT] User ${username} logged out successfully`);
        res.redirect('/login.html');
    });
});

// Check login status
app.get('/api/auth/status', (req, res) => {
    if (req.session.userId) {
        return res.json({
            authenticated: true,
            username: req.session.username
        });
    }
    return res.json({ authenticated: false });
});

// --- Protected Routes ---
app.get('/', checkAuth, (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.get('/index.html', checkAuth, (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.get('/details.html', checkAuth, (req, res) => {
    res.sendFile(path.join(__dirname, 'details.html'));
});

app.get('/manuals.html', checkAuth, (req, res) => {
    res.sendFile(path.join(__dirname, 'manuals.html'));
});

// --- Static Files (with authentication bypass for login page) ---
app.use('/login.html', express.static(path.join(__dirname, 'login.html')));
app.use('/login.css', express.static(path.join(__dirname, 'login.css')));
app.use('/login.js', express.static(path.join(__dirname, 'login.js')));

// Protected static files
app.use(express.static(__dirname, {
    setHeaders: (res, path) => {
        // Allow access to login-related files without authentication
        if (path.includes('login.')) {
            return;
        }
    }
}));

// --- AI Chat Endpoint ---
app.post('/chat', checkAuth, upload.single('image'), async (req, res) => {
  try {
    let { sessionId, question, manual, area } = req.body;
    const imageFile = req.file;
    const username = req.session.username;

    console.log(`[CHAT] Request from user: ${username}`);

    if (!sessionId) {
      sessionId = `session_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
      chatHistories[sessionId] = [];
    }
    if (!chatHistories[sessionId]) {
      chatHistories[sessionId] = [];
    }
    const history = chatHistories[sessionId];

    if (!question) {
      return res.status(400).json({ error: 'Question is required.' });
    }
    
    if (!vectorStore) {
        return res.status(503).json({ 
            error: 'AI knowledge base is not ready. Please wait for the server to finish starting up.' 
        });
    }

    // Filter documents based on manual or area selection
    let filter;
    if (manual && manual !== 'all') {
        filter = (doc) => doc.metadata.source === manual.trim();
        console.log(`[CHAT] Filtering by manual: ${manual}`);
    } else if (area) {
        filter = (doc) => doc.metadata.area === area.trim();
        console.log(`[CHAT] Filtering by area: ${area}`);
    }

    const relevantDocs = await vectorStore.similaritySearch(question, 4, filter);
    console.log(`[CHAT] Found ${relevantDocs.length} relevant documents`);
    
    const context = relevantDocs
      .map((doc) => `Source: ${doc.metadata.source}\nContent:\n${doc.pageContent}`)
      .join('\n\n---\n\n');

   const fullPrompt = `คุณคือ AI Technical Expert ระดับโลก 🌟 ที่มีความเชี่ยวชาญสูงสุดในด้านเทคนิคอุตสาหกรรม พร้อมด้วยประสบการณ์กว่า 20 ปีและความรู้ลึกซึ้งในทุกระบบ

🎯 หลักการตอบคำถามระดับเทพ:

🧠 การวิเคราะห์แบบ 360 องศา:
1. 📊 วิเคราะห์บริบท - เข้าใจความต้องการที่แท้จริงของผู้ใช้
2. 🔍 ค้นหาข้อมูล - รวบรวมข้อมูลจากทุกแหล่งที่เชื่อถือได้
3. 🎓 ประยุกต์ความรู้ - ผสานข้อมูลคู่มือกับประสบการณ์จริง
4. 💡 ให้คำแนะนำเชิงกลยุทธ์ - เสนอแนวทางที่ดีที่สุดและทางเลือกอื่น
5. 🚀 คาดการณ์ล่วงหน้า - ระบุปัญหาที่อาจเกิดขึ้นและวิธีป้องกัน

🎨 การจัดรูปแบบที่สวยงามและอ่านง่าย:

📝 โครงสร้างคำตอบมาตรฐาน:
- 🎯 สรุปหลัก - ตอบคำถามตรงประเด็นทันที
- 📋 รายละเอียด - อธิบายเชิงลึกแบบเป็นระบบ
- 🔧 ขั้นตอนปฏิบัติ - แนวทางแก้ไขที่ชัดเจน
- ⚠️ ข้อควรระวัง - สิ่งสำคัญที่ต้องรู้
- 💡 เคล็ดลับเพิ่มเติม - ความรู้โบนัสที่มีประโยชน์
- 🔮 การคาดการณ์ - ปัญหาที่อาจเกิดขึ้นในอนาคต

🌈 การใช้สัญลักษณ์และสี:
- 🟢 ✅ สถานะปกติ/ขั้นตอนที่ถูกต้อง
- 🔴 ❌ ข้อผิดพลาด/สิ่งที่ไม่ควรทำ
- 🟡 ⚠️ ข้อควรระวัง/คำเตือน
- 🔵 ℹ️ ข้อมูลเพิ่มเติม/หมายเหตุ
- 🟣 🔧 ขั้นตอนแก้ไข/การปรับปรุง
- 🟠 🚀 การปรับปรุงประสิทธิภาพ
- ⚫ 🎓 ความรู้เชิงลึกสำหรับผู้เชี่ยวชาญ

🎭 การตอบตามประเภทคำถาม:

🆘 Error Code/ข้อผิดพลาด:
## 🚨 [ชื่อข้อผิดพลาด]

### 📊 การวิเคราะห์ปัญหา:
- 🔍 สาเหตุหลัก: [อธิบายสาเหตุพื้นฐาน]
- 🔗 สาเหตุเสริม: [ปัจจัยอื่นที่เกี่ยวข้อง]
- 📈 ผลกระทบ: [ผลที่ตามมา]
- 🎯 ระดับความรุนแรง: [กำหนดลำดับความสำคัญ]

### 🛠️ วิธีแก้ไขแบบขั้นตอน:
1. 🔧 ขั้นตอนที่ 1: [วิธีแก้ไขหลัก]
2. 🔧 ขั้นตอนที่ 2: [วิธีแก้ไขเสริม]
3. ✅ ขั้นตอนตรวจสอบ: [การยืนยันผลลัพธ์]
4. 🧪 การทดสอบ: [วิธีการตรวจสอบความถูกต้อง]

### 🛡️ การป้องกัน:
- ⚠️ ข้อควรระวัง: [สิ่งที่ต้องหลีกเลี่ยง]
- 🔄 การบำรุงรักษา: [วิธีป้องกันปัญหา]
- 📊 การติดตาม: [เครื่องมือและวิธีมอนิเตอร์]

### 🔮 การคาดการณ์:
- 🎯 ปัญหาที่อาจตามมา: [ผลกระทบระยะยาว]
- 🚀 โอกาสในการปรับปรุง: [การอัปเกรดที่แนะนำ]

🤔 คำถามทั่วไป:
## 💡 [หัวข้อคำถาม]

### 📚 ความหมายและหลักการ:
[อธิบายแนวคิดพื้นฐาน]

### 🏭 การประยุกต์ใช้ในอุตสาหกรรม:
- 🎯 ประโยชน์หลัก: [ข้อดีที่สำคัญ]
- 📊 ตัวอย่างการใช้งาน: [กรณีศึกษา]
- 🔄 กระบวนการทำงาน: [ลำดับขั้นตอน]
- 💰 ต้นทุนและผลตอบแทน: [การวิเคราะห์ความคุ้มค่า]

### 🎓 ความรู้เพิ่มเติม:
[ข้อมูลที่เกี่ยวข้องที่ควรรู้]

### 🔮 แนวโน้มอนาคต:
- 🚀 เทคโนโลยีใหม่: [นวัตกรรมที่เข้ามา]
- 📈 การพัฒนาที่คาดว่า: [ทิศทางในอนาคต]

🔧 คำถามเทคนิค:
## ⚙️ [หัวข้อเทคนิค]

### 📋 ข้อมูลพื้นฐาน:
- 🏷️ ประเภท: [หมวดหมู่]
- 📏 ข้อมูลจำเพาะ: [รายละเอียดทางเทคนิค]
- 🎯 วัตถุประสงค์: [การใช้งาน]
- 🔋 ประสิทธิภาพ: [ความสามารถและขีดจำกัด]

### 🛠️ วิธีการใช้งาน:
1. 🔧 เตรียมความพร้อม: [ขั้นตอนเตรียมการ]
2. ▶️ การปฏิบัติการ: [ขั้นตอนหลัก]
3. ✅ การตรวจสอบ: [การยืนยันผลลัพธ์]
4. 📊 การติดตาม: [วิธีมอนิเตอร์ประสิทธิภาพ]

### 🔄 การบำรุงรักษา:
- 📅 ตารางเวลา: [ความถี่ในการดูแล]
- 🔧 ขั้นตอนบำรุง: [วิธีการดูแลรักษา]
- 📈 การปรับปรุง: [วิธีเพิ่มประสิทธิภาพ]

### 🎓 ความรู้เชิงลึกสำหรับผู้เชี่ยวชาญ:
- 🔬 หลักการทางวิทยาศาสตร์: [ทฤษฎีและสูตร]
- 🏗️ การออกแบบขั้นสูง: [เทคนิคระดับโปร]
- 🌐 มาตรฐานสากล: [ข้อกำหนดและใบรับรอง]

---

📚 ข้อมูลจากคู่มือ:
${context || '📖 กำลังใช้ความรู้ทางเทคนิคและประสบการณ์ในการตอบคำถาม'}

💬 ประวัติการสนทนา:
${history.map((h, index) => `
คำถามที่ ${index + 1}: ${h.question}
คำตอบ: ${h.answer}
---`).join('')}

❓ คำถามปัจจุบัน: "${question}"

🎓 คำแนะนำพิเศษสำหรับการตอบ:

🏆 เป้าหมายระดับเทพ:
- ตอบให้ "WOW!" - ทำให้ผู้ใช้ประทับใจและได้ความรู้เกินคาด
- ละเอียดแต่กระชับ - ครบถ้วนทุกประเด็น แต่ไม่ยาวเกินไป
- ใช้ตัวอย่างจริง - ยกกรณีศึกษาที่เข้าใจง่าย
- เสนอแนวทางหลากหลาย - ให้ทางเลือกและแนะนำวิธีที่ดีที่สุด
- คาดการณ์ล่วงหน้า - ระบุปัญหาที่อาจเกิดขึ้นและวิธีป้องกัน
- ให้ข้อมูลเชิงลึก - ความรู้ที่ผู้เชี่ยวชาญจริงๆ จะรู้

📱 การจัดรูปแบบที่เหมาะกับมือถือ:
- ใช้หัวข้อสั้นๆ - อ่านง่ายบนหน้าจอเล็ก
- แบ่งย่อหน้าบ่อย - ไม่ให้ข้อความรวมกันเป็นก้อน
- ใช้ bullet points - จัดระเบียบข้อมูลให้เป็นระบบ
- เน้นคำสำคัญ - ใช้ตัวหนาและ emoji
- ใส่ช่องว่างให้เหมาะสม - ทำให้ตาไม่เมื่อย

🎨 เทคนิคการเขียนระดับมือโปร:
- เปิดด้วยการสรุปหลัก - ตอบคำถามตรงประเด็นก่อน
- ใช้คำเชื่อม - ทำให้ข้อความลื่นไหล
- ปิดด้วยคำแนะนำ - สิ่งที่ผู้ใช้ควรทำต่อไป
- เพิ่ม Call-to-Action - เชิญชวนให้ถามคำถามเพิ่มเติม
- ใช้ภาษาที่เข้าใจง่าย - แต่ไม่ลดทอนความเป็นเทคนิค

🧠 AI Intelligence Features:
- 🎯 การวิเคราะห์ Pattern - หาแนวโน้มและรูปแบบ
- 🔍 การค้นหาข้อมูลสหสัมพันธ์ - เชื่อมโยงข้อมูลที่เกี่ยวข้อง
- 📊 การประเมินความเสี่ยง - วิเคราะห์โอกาสเกิดปัญหา
- 💡 การเสนอแนะแบบส่วนตัว - แนวทางเฉพาะสำหรับแต่ละสถานการณ์
- 🚀 การเพิ่มประสิทธิภาพ - หาทางปรับปรุงอย่างต่อเนื่อง

🎖️ ระดับความเชี่ยวชาญในการตอบ:
- 🥉 ระดับพื้นฐาน: ตอบคำถามตรงๆ พร้อมคำอธิบาย
- 🥈 ระดับกลาง: เพิ่มตัวอย่างและเทคนิค
- 🥇 ระดับผู้เชี่ยวชาญ: ให้ข้อมูลเชิงลึกและการคาดการณ์
- 🏆 ระดับเทพ: ครบทุกมิติพร้อมนวัตกรรม

🚀 เริ่มตอบคำถาม - โชว์ความเป็นเทพ!`;

// Enhanced question processing with AI intelligence
const enhanceQuestion = (question) => {
    let enhanced = question;
    const questionLower = question.toLowerCase();
    
    // Error analysis pattern
    if (questionLower.includes('error') || questionLower.includes('ข้อผิดพลาด') || /error\s*\d+/.test(question)) {
        enhanced = `${question} - [กรุณาวิเคราะห์ข้อผิดพลาดนี้แบบเชิงลึก วิเคราะห์สาเหตุ วิธีแก้ไข การป้องกัน และคาดการณ์ปัญหาที่อาจตามมา]`;
    }
    // Definition pattern
    else if (questionLower.includes('คือ') || questionLower.includes('คืออะไร') || questionLower.includes('หมายถึง') || questionLower.includes('what is')) {
        enhanced = `${question} - [กรุณาอธิบายความหมาย หลักการ การประยุกต์ใช้งาน และแนวโน้มอนาคตแบบละเอียดและครบถ้วน]`;
    }
    // Troubleshooting pattern
    else if (questionLower.includes('แก้') || questionLower.includes('ซ่อม') || questionLower.includes('ปรับ') || questionLower.includes('fix')) {
        enhanced = `${question} - [กรุณาให้ขั้นตอนการแก้ไขที่ชัดเจน วิเคราะห์สาเหตุ ข้อควรระวัง เทคนิคเพิ่มเติม และวิธีป้องกันปัญหาซ้ำ]`;
    }
    // How-to pattern
    else if (questionLower.includes('ใช้') || questionLower.includes('ทำงาน') || questionLower.includes('ปฏิบัติ') || questionLower.includes('how to')) {
        enhanced = `${question} - [กรุณาให้วิธีการใช้งานที่ละเอียด ขั้นตอนปฏิบัติ ตัวอย่างจริง คำแนะนำจากผู้เชี่ยวชาญ และเทคนิคเพิ่มประสิทธิภาพ]`;
    }
    // Comparison pattern
    else if (questionLower.includes('เปรียบเทียบ') || questionLower.includes('ต่างกัน') || questionLower.includes('vs') || questionLower.includes('compare')) {
        enhanced = `${question} - [กรุณาเปรียบเทียบแบบละเอียด ข้อดี-ข้อเสีย การใช้งานที่เหมาะสม ต้นทุน และคำแนะนำการเลือกใช้]`;
    }
    // Installation/Setup pattern
    else if (questionLower.includes('ติดตั้ง') || questionLower.includes('ตั้งค่า') || questionLower.includes('setup') || questionLower.includes('install')) {
        enhanced = `${question} - [กรุณาให้ขั้นตอนการติดตั้งแบบละเอียด เครื่องมือที่ต้องใช้ ข้อควรระวัง และวิธีตรวจสอบความถูกต้อง]`;
    }
    // Performance pattern
    else if (questionLower.includes('ประสิทธิภาพ') || questionLower.includes('เร็ว') || questionLower.includes('performance') || questionLower.includes('optimize')) {
        enhanced = `${question} - [กรุณาวิเคราะห์ประสิทธิภาพ ปัจจัยที่ส่งผลกระทบ วิธีปรับปรุง และเทคนิคการเพิ่มประสิทธิภาพ]`;
    }
    // Short questions need more detail
    else if (question.length < 15) {
        enhanced = `${question} - [กรุณาตอบแบบละเอียดและครบถ้วน ให้ความรู้เชิงลึก ตัวอย่างจริง และข้อมูลเพิ่มเติมที่เป็นประโยชน์]`;
    }
    // Complex technical questions
    else if (questionLower.includes('ทำไม') || questionLower.includes('เหตุผล') || questionLower.includes('why')) {
        enhanced = `${question} - [กรุณาอธิบายเหตุผล หลักการทำงาน วิเคราะห์สาเหตุ และให้มุมมองจากผู้เชี่ยวชาญ]`;
    }
    
    return enhanced;
};

// Context analysis for better responses
const analyzeContext = (question, history) => {
    const analysis = {
        complexity: 'medium',
        category: 'general',
        urgency: 'normal',
        followUp: false
    };
    
    const questionLower = question.toLowerCase();
    
    // Determine complexity
    if (questionLower.includes('error') || questionLower.includes('ข้อผิดพลาด') || 
        questionLower.includes('problem') || questionLower.includes('issue')) {
        analysis.complexity = 'high';
        analysis.category = 'troubleshooting';
        analysis.urgency = 'high';
    } else if (questionLower.includes('คือ') || questionLower.includes('what is')) {
        analysis.complexity = 'low';
        analysis.category = 'definition';
    } else if (questionLower.includes('how') || questionLower.includes('วิธี')) {
        analysis.complexity = 'medium';
        analysis.category = 'tutorial';
    }
    
    // Check for follow-up questions
    if (history.length > 0) {
        const lastQuestion = history[history.length - 1].question.toLowerCase();
        if (questionLower.includes('เพิ่มเติม') || questionLower.includes('more') || 
            questionLower.includes('อีก') || questionLower.includes('และ')) {
            analysis.followUp = true;
        }
    }
    
    return analysis;
};

// Export the enhanced system
module.exports = {
    fullPrompt,
    enhanceQuestion,
    analyzeContext
};

    const enhancedQuestion = enhanceQuestion(question);
    
    // Prepare AI request
    const promptParts = [{ text: fullPrompt }];
    if (imageFile) {
        promptParts.push({ text: '🖼️ วิเคราะห์รูปภาพนี้ประกอบด้วย:' });
        promptParts.push({ 
            inlineData: { 
                data: imageFile.buffer.toString('base64'), 
                mimeType: imageFile.mimetype 
            } 
        });
        console.log(`[CHAT] Image uploaded: ${imageFile.mimetype}, size: ${imageFile.size} bytes`);
    }

    // Generate AI response
    const result = await generativeModel.generateContent({ 
        contents: [{ role: 'user', parts: promptParts }] 
    });
    const response = await result.response;
    const answer = response.text();

    // Save chat history
    chatHistories[sessionId].push({ question, answer });
    
    console.log(`[CHAT] Response generated for user: ${username}`);
    res.json({ answer, sessionId });

  } catch (error) {
    console.error('[CHAT] Error:', error);
    res.status(500).json({ error: 'Failed to get response from AI.' });
  }
});

// --- Manual Management Endpoint ---
app.get('/api/manuals', checkAuth, async (req, res) => {
    try {
      const documentsBasePath = path.join(__dirname, 'documents');
      const manualDatabase = {};
  
      console.log(`[MANUALS] Loading manuals from: ${documentsBasePath}`);
      
      const areaFolders = (await fs.readdir(documentsBasePath, { withFileTypes: true }))
        .filter((dirent) => dirent.isDirectory())
        .map((dirent) => dirent.name);
  
      console.log(`[MANUALS] Found ${areaFolders.length} areas: ${areaFolders.join(', ')}`);
  
      for (const area of areaFolders) {
        const areaPath = path.join(documentsBasePath, area.trim());
        const files = await fs.readdir(areaPath);
        
        const areaKey = area.trim().toLowerCase();
        manualDatabase[areaKey] = {
          name: area.trim(),
          files: files.map((fileName) => {
            const trimmedFileName = fileName.trim();
            let displayName = path.parse(trimmedFileName).name;
            
            // Clean up display name
            const prefix = `${area.trim()}_`;
            if (displayName.startsWith(prefix)) {
              displayName = displayName.substring(prefix.length);
            }
            displayName = displayName.replace(/_/g, ' ').replace(/-/g, ' ');
            displayName = displayName.charAt(0).toUpperCase() + displayName.slice(1);
            
            return {
              name: trimmedFileName,
              path: `documents/${area.trim()}/${trimmedFileName}`,
              displayName,
            };
          }),
        };
        
        console.log(`[MANUALS] Processed area "${area}": ${files.length} files`);
      }
  
      res.json(manualDatabase);
    } catch (error) {
      console.error('[MANUALS] Error creating manuals manifest:', error);
      res.status(500).json({ error: 'Could not retrieve manual list.' });
    }
});

// --- Health Check Endpoint ---
app.get('/api/health', (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        vectorStoreReady: !!vectorStore,
        uptime: process.uptime()
    });
});

// --- Error Handler ---
app.use((err, req, res, next) => {
    console.error('[ERROR]', err.stack);
    res.status(500).json({ error: 'Internal server error' });
});

// --- 404 Handler ---
app.use((req, res) => {
    res.status(404).json({ error: 'Page not found' });
});

// --- Server Startup ---
async function startServer() {
  console.log('🚀 Starting server...');
  
  // Initialize vector store
  await initializeVectorStore(); 
  
  if (vectorStore) {
    app.listen(port, () => {
      console.log(`✅ Server running on port ${port}`);
      console.log(`📊 Vector store ready with ${vectorStore.memoryVectors?.length || 0} vectors`);
      console.log(`🔐 Authentication enabled`);
      console.log(`🌐 Access: http://localhost:${port}`);
      console.log(`📝 Default users:`);
      console.log(`   - admin / password123`);
      console.log(`   - user / password456`);
    });
  } else {
    console.error('❌ Server startup failed - vector store initialization failed');
    process.exit(1);
  }
}

// --- Graceful Shutdown ---
process.on('SIGTERM', () => {
    console.log('🛑 Received SIGTERM, shutting down gracefully...');
    process.exit(0);
});

process.on('SIGINT', () => {
    console.log('🛑 Received SIGINT, shutting down gracefully...');
    process.exit(0);
});

startServer();