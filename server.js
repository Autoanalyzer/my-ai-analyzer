// server.js (Modified for Pinecone Integration)

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
// ✨ เปลี่ยนจาก MemoryVectorStore เป็น PineconeStore
const { PineconeStore } = require('@langchain/pinecone');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { PDFLoader } = require('langchain/document_loaders/fs/pdf');
// ✨ เพิ่ม Pinecone import
const { Pinecone } = require('@pinecone-database/pinecone');

const app = express();
const port = process.env.PORT || 5500;

// --- 2. ตั้งค่า User และ Session ---
const users = [
    { id: 1, username: 'admin', password: 'password123' },
    { id: 2, username: 'user', password: 'password456' }
];

app.use(session({
    secret: 'your_super_secret_key',
    resave: false,
    saveUninitialized: false,
    cookie: { secure: false, maxAge: 60 * 60 * 1000 }
}));

const chatHistories = {};
let vectorStore; // ✨ ยังคงใช้ตัวแปรเดิม แต่จะเป็น PineconeStore แทน

// ✨ ลบ VECTOR_STORE_SAVE_PATH เพราะไม่ต้องบันทึกไฟล์แล้ว
// const VECTOR_STORE_SAVE_PATH = path.join(__dirname, 'vector_store.json');

app.use(cors());
app.use(express.json());

app.use((req, res, next) => {
    console.log(`[DEBUG] Incoming Request: ${req.method} ${req.originalUrl}`);
    next();
});

// --- 1. เพิ่ม Middleware สำหรับตรวจสอบการ Login ---
const checkAuth = (req, res, next) => {
    console.log('[DEBUG] --- Running checkAuth ---');
    console.log('[DEBUG] Session ID:', req.session.id);
    console.log('[DEBUG] req.session.userId is:', req.session.userId);

    if (!req.session.userId) {
        console.log('[DEBUG] Condition is TRUE. Redirecting to /login.html');
        return res.redirect('/login.html');
    }
   
    console.log('[DEBUG] Condition is FALSE. User is authenticated. Allowing access.');
    next();
};

// --- 2. สร้าง Endpoint สำหรับ Login และ Logout ---
app.post('/login', (req, res) => {
    const { username, password } = req.body;
    const user = users.find(u => u.username === username && u.password === password);

    if (user) {
        req.session.userId = user.id;
        req.session.username = user.username;
        return res.json({ message: 'Login successful' });
    }

    return res.status(401).json({ error: 'Invalid username or password' });
});

app.get('/logout', (req, res) => {
    req.session.destroy(err => {
        if (err) {
            return res.redirect('/index.html');
        }
        res.clearCookie('connect.sid');
        res.redirect('/login.html');
    });
});

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

app.use(express.static(__dirname));



const upload = multer({ storage: multer.memoryStorage() });

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

// ✨ แก้ไข initializeVectorStore ให้ใช้ Pinecone
async function initializeVectorStore() {
  try {
    console.log('🔧 Initializing Pinecone connection...');
    
    // ✨ สร้าง Pinecone client
    const pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });

    // ✨ เชื่อมต่อกับ Index
    const index = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    // ✨ สร้าง PineconeStore
    vectorStore = new PineconeStore(embeddingsModel, {
      pineconeIndex: index,
      maxConcurrency: 5, // Maximum number of batch requests to allow at once
    });

    console.log('✅ Pinecone vector store initialized successfully.');

    // ✨ ตรวจสอบว่ามีข้อมูลใน Index หรือไม่
    const stats = await index.describeIndexStats();
    console.log(`📊 Current index stats:`, stats);

    // ✨ ถ้ายังไม่มีข้อมูล ให้โหลดเอกสารใหม่
    if (stats.totalRecordCount === 0) {
      console.log('📚 Index is empty. Loading documents...');
      await loadDocumentsIntoPinecone();
    } else {
      console.log(`✅ Index already contains ${stats.totalRecordCount} records.`);
    }

  } catch (error) {
    console.error('CRITICAL: Failed to initialize Pinecone vector store.', error);
    vectorStore = undefined;
  }
}

// ✨ ฟังก์ชันใหม่สำหรับโหลดเอกสารเข้า Pinecone
async function loadDocumentsIntoPinecone() {
  try {
    const documentsBasePath = path.join(__dirname, 'documents');
    const allDocuments = [];

    const areaFolders = (await fs.readdir(documentsBasePath, { withFileTypes: true }))
      .filter(d => d.isDirectory())
      .map(d => d.name);

    for (const area of areaFolders) {
      const areaPath = path.join(documentsBasePath, area);
      const files = await fs.readdir(areaPath);
      for (const file of files) {
        const filePath = path.join(areaPath, file);
        const fileExt = path.extname(file).toLowerCase();
        let docsFromFile = [];

        try {
          if (fileExt === '.pdf') {
            const loader = new PDFLoader(filePath);
            docsFromFile = await loader.load();
          } else if (fileExt === '.txt') {
            const textContent = await fs.readFile(filePath, 'utf-8');
            docsFromFile.push({ pageContent: textContent, metadata: {} });
          }

          docsFromFile.forEach(doc => {
            doc.metadata.source = file.trim();
            doc.metadata.area = area.trim();
          });
          allDocuments.push(...docsFromFile);

        } catch (fileError) {
          console.error(`Could not process file: ${file}`, fileError);
        }
      }
    }

    const textSplitter = new RecursiveCharacterTextSplitter({ 
      chunkSize: 1000, 
      chunkOverlap: 200 
    });
    const splitDocs = await textSplitter.splitDocuments(allDocuments);

    console.log(`📤 Uploading ${splitDocs.length} document chunks to Pinecone...`);
    
    // ✨ อัปโหลดเอกสารไปยัง Pinecone
    const batchSize = 50;
    for (let i = 0; i < splitDocs.length; i += batchSize) {
      const batch = splitDocs.slice(i, i + batchSize);
      await vectorStore.addDocuments(batch);
      console.log(`📤 Uploaded batch ${Math.floor(i / batchSize) + 1} of ${Math.ceil(splitDocs.length / batchSize)}...`);
      
      // รอเล็กน้อยเพื่อไม่ให้ทำงานหนักเกินไป
      if (i + batchSize < splitDocs.length) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }

    console.log('✅ All documents uploaded to Pinecone successfully.');

  } catch (error) {
    console.error('CRITICAL: Failed to load documents into Pinecone.', error);
    throw error;
  }
}

// ✨ แก้ไข /chat endpoint (ส่วนใหญ่เหมือนเดิม แต่ลบการใช้ filter แบบเก่า)
// แทนที่ส่วน /chat endpoint เดิมด้วยโค้ดนี้

app.post('/chat', checkAuth, upload.single('image'), async (req, res) => {
    try {
        let { sessionId, question, manual, area } = req.body;
        const imageFile = req.file;

        // สร้าง session ID ถ้าไม่มี
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
            return res.status(503).json({ error: 'AI knowledge base is not ready. Please wait.' });
        }

        // 🔧 แก้ไข metadata filter ให้ถูกต้องสำหรับ Pinecone
        let filter = {};
        
        // Debug logs
        console.log('🔍 Filter parameters received:', { manual, area });
        
        if (manual && manual !== 'all' && manual.trim() !== '') {
            // ใช้ filter สำหรับ manual ที่เลือก
            filter = {
                source: { "$eq": manual.trim() }
            };
            console.log('📄 Filtering by manual:', manual.trim());
        } else if (area && area.trim() !== '') {
            // ใช้ filter สำหรับ area ที่เลือก
            filter = {
                area: { "$eq": area.trim() }
            };
            console.log('📁 Filtering by area:', area.trim());
        }

        console.log('🎯 Final filter object:', JSON.stringify(filter, null, 2));

        try {
            // 🔧 ใช้ similaritySearch พร้อม filter และจำนวนผลลัพธ์ที่เหมาะสม
            const searchOptions = Object.keys(filter).length > 0 ? {
                filter: filter,
                k: 5  // เพิ่มจำนวนผลลัพธ์เพื่อให้ได้ข้อมูลที่หลากหลาย
            } : {
                k: 5
            };

            console.log('🔎 Searching with options:', JSON.stringify(searchOptions, null, 2));

            const relevantDocs = await vectorStore.similaritySearch(
                question, 
                searchOptions.k, 
                searchOptions.filter || {}
            );

            console.log(`📋 Found ${relevantDocs.length} relevant documents`);
            
            // Debug: แสดงข้อมูล metadata ของเอกสารที่พบ
            relevantDocs.forEach((doc, index) => {
                console.log(`📄 Document ${index + 1}:`, {
                    source: doc.metadata.source,
                    area: doc.metadata.area,
                    page: doc.metadata.loc?.pageNumber || 1,
                    contentPreview: doc.pageContent.substring(0, 100) + '...'
                });
            });

            if (relevantDocs.length === 0) {
                console.log('⚠️ No relevant documents found');
                return res.json({
                    answer: `ขออภัยครับ ไม่พบข้อมูลที่เกี่ยวข้องกับคำถาม "${question}" ${manual ? `ในคู่มือ "${manual}"` : area ? `ในหมวด "${area}"` : ''} \n\nกรุณาลองถามในหัวข้ออื่น หรือปรับคำถามให้ชัดเจนมากขึ้นครับ 🙏`,
                    sessionId
                });
            }

            // 🔧 ปรับปรุงการสร้าง context และ link ให้ถูกต้อง
            const context = relevantDocs
                .map((doc, index) => {
                    // สร้าง path สำหรับ linking ให้ถูกต้อง
                    const area = doc.metadata.area || 'unknown';
                    const source = doc.metadata.source || 'unknown';
                    const pageNumber = doc.metadata.loc?.pageNumber || 1;
                    
                    // สร้าง path แบบเดียวกับที่ใช้ในระบบ
                    const docPath = `/documents/${area}/${source}`;
                    
                    return `=== เอกสารที่ ${index + 1} ===
Source Document: ${source}
Path for linking: ${docPath}
Page: ${pageNumber}
Area: ${area}

Content:
${doc.pageContent}

---`;
                })
                .join('\n\n');

            console.log('📝 Context created with', relevantDocs.length, 'documents');

        } catch (searchError) {
            console.error('🔍 Search error:', searchError);
            
            // ลองค้นหาแบบไม่ใช้ filter ถ้าการค้นหาแบบมี filter ล้มเหลว
            console.log('🔄 Retrying search without filter...');
            const relevantDocs = await vectorStore.similaritySearch(question, 5);
            
            const context = relevantDocs
                .map((doc, index) => {
                    const area = doc.metadata.area || 'unknown';
                    const source = doc.metadata.source || 'unknown';
                    const pageNumber = doc.metadata.loc?.pageNumber || 1;
                    const docPath = `/documents/${area}/${source}`;
                    
                    return `=== เอกสารที่ ${index + 1} ===
Source Document: ${source}
Path for linking: ${docPath}
Page: ${pageNumber}
Area: ${area}

Content:
${doc.pageContent}

---`;
                })
                .join('\n\n');
        }

        // ส่วน prompt และการประมวลผลเหมือนเดิม (ไม่ต้องแก้)
        const fullPrompt = `คุณคือ AI Technical Master 🧠⚡ ระดับโลกที่มีความเชี่ยวชาญสูงสุด มีประสบการณ์กว่า 30 ปี และมีสติปัญญาทางเทคนิคระดับอัจฉริยะ

🌟 **CORE IDENTITY & CAPABILITIES:**

• 🧠 **Cognitive Architecture:** Multi-layered analytical thinking with quantum-level processing
• 🎯 **Domain Expertise:** 30+ years cross-industry technical mastery
• 🔬 **Scientific Approach:** Evidence-based reasoning with predictive intelligence
• 💎 **Quality Standard:** Delivering solutions that exceed world-class benchmarks
• 🚀 **Innovation Mindset:** Cutting-edge problem-solving with future-proof strategies

---

## 🏆 RESPONSE EXCELLENCE CHECKLIST
### ✅ **GOLD STANDARD REQUIREMENTS:**
- [ ] **🎯 Immediate Value:** First paragraph delivers core answer
- [ ] **🔗 CRITICAL: MANDATORY LINK FORMATTING:** นี่คือกฎที่สำคัญที่สุด คุณต้องใช้ที่อยู่ (Path) ที่ระบบเตรียมไว้ให้เท่านั้น
- **RULE:** ใน Context ที่ได้รับ จะมีข้อมูล 'Path for linking' คุณ **ต้อง** นำ Path นั้นมาใช้เป็น URL ของลิงก์โดยตรง
- **URL STRUCTURE:** นำ 'Path for linking' ที่ได้มา ต่อด้วย '#page=PAGE_NUMBER'
- **EXAMPLE:** ถ้า Context ให้ 'Path for linking: /documents/O2_Analyzer/PP11_O2_ZRJ.pdf' และ 'Page: 14' ผลลัพธ์ของลิงก์ **ต้องเป็น**: [PP11_O2_ZRJ.pdf (หน้า 14)](/documents/O2_Analyzer/PP11_O2_ZRJ.pdf#page=14)
- **ห้ามสร้างหรือเดา Path เองเด็ดขาด ให้ใช้ Path ที่ระบบส่งมาให้เท่านั้น**
- [ ] **📊 Comprehensive Coverage:** All relevant aspects addressed
- [ ] **🔍 Expert-Level Analysis:** Deep technical understanding demonstrated
- [ ] **💡 Practical Application:** Real-world implementation guidance
- [ ] **🔮 Future-Proof Perspective:** Long-term considerations included
- [ ] **🎨 Visual Excellence:** Professional formatting and structure
- [ ] **🚀 Actionable Intelligence:** Clear next steps and implementation path

### 📚 **KNOWLEDGE INTEGRATION SYSTEM**

**Available Knowledge Base:**
${context || '🧠 Leveraging 30+ years of cross-industry technical expertise with quantum-level analytical processing for optimal solution delivery'}

**Current Question:** "${question}"

${manual ? `**Selected Manual:** ${manual}` : ''}
${area ? `**Selected Area:** ${area}` : ''}

---

🎯 **MISSION: ตอบคำถามอย่างละเอียดและแม่นยำ พร้อมใส่ลิงก์ไปยังเอกสารต้นฉบับตามรูปแบบที่กำหนด** 🚀✨`;

        const enhancedQuestion = `User Request: "${question}"`;
        const promptParts = [];
        
        promptParts.push({ text: fullPrompt });
        
        if (context) {
            promptParts.push({ text: `--- KNOWLEDGE BASE CONTEXT ---\n${context}` });
        }
        
        promptParts.push({ text: `--- CURRENT MISSION ---\n${enhancedQuestion}` });
       
        if (imageFile) {
            promptParts.push({ text: 'วิเคราะห์รูปภาพนี้ประกอบด้วย:' });
            promptParts.push({ 
                inlineData: { 
                    data: imageFile.buffer.toString('base64'), 
                    mimeType: imageFile.mimetype 
                } 
            });
        }

        const result = await generativeModel.generateContent({ 
            contents: [{ role: 'user', parts: promptParts }] 
        });
        const response = await result.response;
        const answer = response.text();

        // บันทึก history
        chatHistories[sessionId].push({ question, answer });
        
        console.log('✅ Response generated successfully');
        res.json({ answer, sessionId });

    } catch (error) {
        console.error('❌ Error in /chat endpoint:', error);
        res.status(500).json({ 
            error: 'Failed to get response from AI.',
            details: error.message 
        });
    }
});

app.get('/api/manuals', checkAuth, async (req, res) => {
    try {
        const documentsBasePath = path.join(__dirname, 'documents');
        const manualDatabase = {};

        const areaFolders = (await fs.readdir(documentsBasePath, { withFileTypes: true }))
            .filter((dirent) => dirent.isDirectory())
            .map((dirent) => dirent.name);

        for (const area of areaFolders) {
            const areaPath = path.join(documentsBasePath, area.trim());
            const files = await fs.readdir(areaPath);
           
            const areaKey = area.trim().toLowerCase();
            manualDatabase[areaKey] = {
                name: area.trim(),
                files: files.map((fileName) => {
                    const trimmedFileName = fileName.trim();
                    const fileBaseName = path.parse(trimmedFileName).name;
                   
                    const prefix = `${area.trim()}_`;
                    let nameWithoutPrefix = fileBaseName;
                    if (fileBaseName.startsWith(prefix)) {
                        nameWithoutPrefix = fileBaseName.substring(prefix.length);
                    }
                   
                    const imageName = nameWithoutPrefix;
                    const imagePath = `images/${imageName}.png`;

                    let displayName = nameWithoutPrefix.replace(/_/g, ' ').replace(/-/g, ' ');
                    displayName = displayName.charAt(0).toUpperCase() + displayName.slice(1);
                   
                    return {
                        name: trimmedFileName,
                        path: `documents/${area.trim()}/${trimmedFileName}`,
                        displayName,
                        image: imagePath
                    };
                }),
            };
        }

        res.json(manualDatabase);
    } catch (error) {
        console.error('Error creating manuals manifest:', error);
        res.status(500).json({ error: 'Could not retrieve manual list.' });
    }
});

// ✨ แก้ไข startServer function
async function startServer() {
  await initializeVectorStore();
 
  if (vectorStore) {
    app.listen(port, () => {
      console.log(`✅ Backend server is running on port ${port}`);
      console.log(`🔗 Connected to Pinecone index: ${process.env.PINECONE_INDEX_NAME}`);
    });
  } else {
    console.error('❌ Server startup failed because the vector store could not be initialized.');
    process.exit(1);
  }
}

startServer();

