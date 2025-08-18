// server.js (Fixed Version for Pinecone)

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
const { PineconeStore } = require('@langchain/pinecone');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { PDFLoader } = require('langchain/document_loaders/fs/pdf');
const { Pinecone } = require('@pinecone-database/pinecone');

const app = express();
const port = process.env.PORT || 5500;

// --- User และ Session Setup ---
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
let vectorStore;
let pineconeIndex; // เก็บ reference ของ Pinecone Index

app.use(cors());
app.use(express.json());

app.use((req, res, next) => {
    console.log(`[DEBUG] Incoming Request: ${req.method} ${req.originalUrl}`);
    next();
});

// Auth Middleware
const checkAuth = (req, res, next) => {
    console.log('[DEBUG] --- Running checkAuth ---');
    console.log('[DEBUG] Session ID:', req.session.id);
    console.log('[DEBUG] req.session.userId is:', req.session.userId);

    if (!req.session.userId) {
        console.log('[DEBUG] Redirecting to /login.html');
        return res.redirect('/login.html');
    }
   
    console.log('[DEBUG] User is authenticated. Allowing access.');
    next();
};

// Login/Logout Endpoints
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

// Page Routes
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

// Gemini Configuration
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

// 🔧 FIXED: Initialize Pinecone with better error handling
async function initializeVectorStore() {
  try {
    console.log('🔧 Initializing Pinecone connection...');
    
    const pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });

    pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    vectorStore = new PineconeStore(embeddingsModel, {
      pineconeIndex: pineconeIndex,
      maxConcurrency: 5,
    });

    console.log('✅ Pinecone vector store initialized successfully.');

    const stats = await pineconeIndex.describeIndexStats();
    console.log(`📊 Current index stats:`, stats);

    if (stats.totalRecordCount === 0) {
      console.log('📚 Index is empty. Loading documents...');
      await loadDocumentsIntoPinecone();
    } else {
      console.log(`✅ Index already contains ${stats.totalRecordCount} records.`);
      // 🔧 IMPORTANT: Check if metadata is correct
      await verifyMetadata();
    }

  } catch (error) {
    console.error('CRITICAL: Failed to initialize Pinecone vector store.', error);
    vectorStore = undefined;
  }
}

// 🔧 NEW: Verify metadata structure
async function verifyMetadata() {
  try {
    console.log('🔍 Verifying metadata structure...');
    
    // Query a sample record to check metadata
    const results = await pineconeIndex.query({
      topK: 1,
      includeMetadata: true,
      vector: new Array(768).fill(0), // dummy vector
    });
    
    if (results.matches && results.matches.length > 0) {
      console.log('📋 Sample metadata:', results.matches[0].metadata);
    }
  } catch (error) {
    console.error('Error verifying metadata:', error);
  }
}

// 🔧 FIXED: Improved document loading with correct metadata
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
            docsFromFile.push({ 
              pageContent: textContent, 
              metadata: { loc: { pageNumber: 1 } } 
            });
          }

          // 🔧 FIXED: Ensure metadata is properly structured
          docsFromFile.forEach(doc => {
            // Clean and standardize metadata
            doc.metadata = {
              ...doc.metadata,
              source: file.trim(),
              area: area.trim().toLowerCase(), // lowercase for consistent filtering
              fileName: file.trim(),
              fullPath: `/documents/${area.trim()}/${file.trim()}`,
              pageNumber: doc.metadata?.loc?.pageNumber || 1
            };
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
    
    // Upload in batches
    const batchSize = 50;
    for (let i = 0; i < splitDocs.length; i += batchSize) {
      const batch = splitDocs.slice(i, i + batchSize);
      await vectorStore.addDocuments(batch);
      console.log(`📤 Uploaded batch ${Math.floor(i / batchSize) + 1} of ${Math.ceil(splitDocs.length / batchSize)}...`);
      
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

// 🔧 FIXED: Improved chat endpoint with better filtering
app.post('/chat', checkAuth, upload.single('image'), async (req, res) => {
    try {
        let { sessionId, question, manual, area } = req.body;
        const imageFile = req.file;

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

        // 🔧 FIXED: Proper metadata filter for Pinecone
        let filter = {};
        
        if (manual && manual !== 'all') {
            // Filter by specific file
            filter = { source: { $eq: manual.trim() } };
            console.log('🔍 Filtering by manual:', manual);
        } else if (area) {
            // Filter by area
            filter = { area: { $eq: area.trim().toLowerCase() } };
            console.log('🔍 Filtering by area:', area);
        }

        console.log('📋 Using filter:', JSON.stringify(filter));

        // Search with filter
        const relevantDocs = await vectorStore.similaritySearch(question, 4, filter);
        
        console.log(`📚 Found ${relevantDocs.length} relevant documents`);

        // 🔧 FIXED: Generate correct links with metadata
        const context = relevantDocs
          .map((doc) => {
              // Use the fullPath from metadata or construct it
              const docPath = doc.metadata.fullPath || `/documents/${doc.metadata.area}/${doc.metadata.source}`;
              const pageNum = doc.metadata.pageNumber || doc.metadata.loc?.pageNumber || 1;
              
              return `Source Document: ${doc.metadata.source}
Path for linking: ${docPath}
Page: ${pageNum}
Content:
${doc.pageContent}`;
          })
          .join('\n\n---\n\n');

        // Create prompt (keeping your existing prompt structure)
        const fullPrompt = `คุณคือ AI Technical Master...

[กฎสำคัญสำหรับการสร้างลิงก์]
- ใช้ Path ที่ระบุใน "Path for linking:" เท่านั้น
- รูปแบบลิงก์: [ชื่อไฟล์ (หน้า X)](path#page=X)
- ตัวอย่าง: ถ้าได้รับ Path: /documents/LK2/LK2_O2_ZRJ.pdf และ Page: 14
  ผลลัพธ์: [LK2_O2_ZRJ.pdf (หน้า 14)](/documents/LK2/LK2_O2_ZRJ.pdf#page=14)

Available Knowledge Base:
${context || 'No specific documents found. Using general knowledge.'}

User Question: "${question}"`;

        const promptParts = [{ text: fullPrompt }];
       
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

        chatHistories[sessionId].push({ question, answer });
        res.json({ answer, sessionId });

    } catch (error) {
        console.error('Error in /chat endpoint:', error);
        res.status(500).json({ error: 'Failed to get response from AI.' });
    }
});

// API Endpoints (keeping existing)
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
// ========== เพิ่มโค้ดนี้ใน server.js ก่อน startServer() ==========

// Debug endpoint - ดู metadata
app.get('/debug/check', checkAuth, async (req, res) => {
    try {
        if (!pineconeIndex) {
            return res.status(503).json({ error: 'Pinecone not initialized' });
        }
        
        // Query เพื่อดู metadata
        const results = await pineconeIndex.query({
            topK: 5,
            includeMetadata: true,
            includeValues: false,
            vector: new Array(768).fill(0.1), // dummy vector
        });
        
        res.json({
            success: true,
            totalMatches: results.matches.length,
            metadata: results.matches.map(m => ({
                id: m.id,
                metadata: m.metadata || 'NO METADATA'
            }))
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Delete all และ reload
app.post('/debug/reset', checkAuth, async (req, res) => {
    try {
        console.log('🗑️ Deleting all vectors from Pinecone...');
        
        // ลบข้อมูลทั้งหมด
        await pineconeIndex.deleteAll();
        
        console.log('✅ All vectors deleted. Waiting 5 seconds...');
        await new Promise(resolve => setTimeout(resolve, 5000));
        
        console.log('📚 Starting to reload documents...');
        
        // โหลดใหม่ด้วย metadata ที่ถูกต้อง
        await loadDocumentsIntoPinecone();
        
        const stats = await pineconeIndex.describeIndexStats();
        
        res.json({
            success: true,
            message: 'Data reset complete',
            newStats: stats
        });
    } catch (error) {
        console.error('Reset error:', error);
        res.status(500).json({ error: error.message });
    }
});
// ========== PLAN B: แก้ไข Chat ให้ง่ายที่สุด ==========
// แทนที่ chat endpoint เดิมด้วยอันนี้

app.post('/chat', checkAuth, upload.single('image'), async (req, res) => {
    try {
        let { sessionId, question, manual, area } = req.body;
        
        if (!question) {
            return res.status(400).json({ error: 'Question is required.' });
        }

        // ถ้า vector store ไม่พร้อม ใช้ AI ตอบโดยตรง
        let context = '';
        
        if (vectorStore) {
            try {
                // ค้นหาแบบง่ายๆ ไม่ใช้ filter
                console.log('🔍 Simple search for:', question);
                const docs = await vectorStore.similaritySearch(question, 3);
                
                if (docs.length > 0) {
                    context = 'Reference information:\n' + 
                        docs.map(d => d.pageContent.substring(0, 300)).join('\n\n');
                    console.log('📚 Found', docs.length, 'documents');
                }
            } catch (err) {
                console.error('Search failed:', err);
            }
        }

        // Generate answer
        const prompt = `You are a technical support assistant.
        
${manual && manual !== 'all' ? `User is asking about: ${manual}` : ''}
${context ? `\nContext:\n${context}\n` : ''}

Question: ${question}

Please provide a helpful answer in Thai language.`;

        const result = await generativeModel.generateContent(prompt);
        const answer = result.response.text();

        res.json({ 
            answer,
            sessionId: sessionId || 'default'
        });

    } catch (error) {
        console.error('Error:', error);
        
        // Fallback response
        res.json({
            answer: `ขออภัยครับ เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง

ข้อผิดพลาด: ${error.message}
            
💡 คำแนะนำ:
- ลองรีเฟรชหน้าเว็บ
- เลือกคู่มือใหม่
- พิมพ์คำถามใหม่`,
            sessionId: 'error'
        });
    }
});
// ========== แก้ไขฟังก์ชัน loadDocumentsIntoPinecone ==========
// ให้แน่ใจว่า metadata ถูกต้อง

async function loadDocumentsIntoPinecone() {
  try {
    const documentsBasePath = path.join(__dirname, 'documents');
    const allDocuments = [];

    const areaFolders = (await fs.readdir(documentsBasePath, { withFileTypes: true }))
      .filter(d => d.isDirectory())
      .map(d => d.name);

    console.log('📁 Found areas:', areaFolders);

    for (const area of areaFolders) {
      const areaPath = path.join(documentsBasePath, area);
      const files = await fs.readdir(areaPath);
      
      console.log(`📂 Processing area ${area} with ${files.length} files`);
      
      for (const file of files) {
        const filePath = path.join(areaPath, file);
        const fileExt = path.extname(file).toLowerCase();
        let docsFromFile = [];

        try {
          if (fileExt === '.pdf') {
            const loader = new PDFLoader(filePath);
            docsFromFile = await loader.load();
            console.log(`  ✅ Loaded PDF: ${file} (${docsFromFile.length} pages)`);
          } else if (fileExt === '.txt') {
            const textContent = await fs.readFile(filePath, 'utf-8');
            docsFromFile.push({ 
              pageContent: textContent, 
              metadata: { loc: { pageNumber: 1 } } 
            });
            console.log(`  ✅ Loaded TXT: ${file}`);
          }

          // ⚠️ CRITICAL: ตั้ง metadata ให้ถูกต้อง
          docsFromFile.forEach((doc, idx) => {
            // สร้าง metadata ใหม่ทั้งหมด
            doc.metadata = {
              source: file.trim(),                    // ชื่อไฟล์
              area: area.trim().toLowerCase(),        // ชื่อ area (lowercase)
              fileName: file.trim(),                  // ชื่อไฟล์ซ้ำ
              fullPath: `/documents/${area.trim()}/${file.trim()}`, // path เต็ม
              pageNumber: doc.metadata?.loc?.pageNumber || (idx + 1), // หน้า
              originalArea: area.trim()               // area ตัวเดิม (case sensitive)
            };
          });
          
          allDocuments.push(...docsFromFile);

        } catch (fileError) {
          console.error(`  ❌ Error processing ${file}:`, fileError.message);
        }
      }
    }

    console.log(`📄 Total documents loaded: ${allDocuments.length}`);

    // Split documents
    const textSplitter = new RecursiveCharacterTextSplitter({ 
      chunkSize: 1000, 
      chunkOverlap: 200 
    });
    const splitDocs = await textSplitter.splitDocuments(allDocuments);

    console.log(`✂️ Split into ${splitDocs.length} chunks`);

    // Upload to Pinecone in batches
    const batchSize = 50;
    for (let i = 0; i < splitDocs.length; i += batchSize) {
      const batch = splitDocs.slice(i, i + batchSize);
      
      // ตรวจสอบ metadata ก่อน upload
      console.log(`📤 Uploading batch ${Math.floor(i / batchSize) + 1}...`);
      console.log('   Sample metadata:', batch[0]?.metadata);
      
      await vectorStore.addDocuments(batch);
      
      // รอเล็กน้อยระหว่าง batch
      if (i + batchSize < splitDocs.length) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }

    console.log('✅ All documents uploaded to Pinecone successfully!');
    
    // Verify upload
    await new Promise(resolve => setTimeout(resolve, 3000));
    const stats = await pineconeIndex.describeIndexStats();
    console.log('📊 Final stats:', stats);

  } catch (error) {
    console.error('❌ CRITICAL ERROR in loadDocumentsIntoPinecone:', error);
    throw error;
  }
}
// Start server
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