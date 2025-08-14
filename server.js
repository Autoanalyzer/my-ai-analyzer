// server.js (Pinecone + RAG + PDF deep-link, full version)

require('dotenv').config();

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs/promises');
// const pdf = require('pdf-parse'); // not used
const path = require('path');
const session = require('express-session');

const {
  GoogleGenerativeAI,
  HarmCategory,
  HarmBlockThreshold,
} = require('@google/generative-ai');
const { GoogleGenerativeAIEmbeddings } = require('@langchain/google-genai');

const { PineconeStore } = require('@langchain/pinecone');
const { RecursiveCharacterTextSplitter } = require('@langchain/textsplitters');
const { PDFLoader } = require('@langchain/community/document_loaders/fs/pdf');

const { Pinecone } = require('@pinecone-database/pinecone');

const app = express();
const port = process.env.PORT || 5500;

// --- Users & Session ---
const users = [
  { id: 1, username: 'admin', password: 'password123' },
  { id: 2, username: 'user',  password: 'password456' },
];

app.use(session({
  secret: 'your_super_secret_key',
  resave: false,
  saveUninitialized: false,
  cookie: { secure: false, maxAge: 60 * 60 * 1000 },
}));

const chatHistories = {};
let vectorStore; // PineconeStore

app.use(cors());
app.use(express.json());

app.use((req, _res, next) => {
  console.log(`[DEBUG] ${req.method} ${req.originalUrl}`);
  next();
});

// --- Auth middleware ---
const checkAuth = (req, res, next) => {
  console.log('[DEBUG] Session ID:', req.session.id);
  if (!req.session.userId) return res.redirect('/login.html');
  next();
};

// --- Auth routes ---
app.post('/login', (req, res) => {
  const { username, password } = req.body || {};
  const user = users.find(u => u.username === username && u.password === password);
  if (!user) return res.status(401).json({ error: 'Invalid username or password' });
  req.session.userId = user.id;
  req.session.username = user.username;
  return res.json({ message: 'Login successful' });
});

app.get('/logout', (req, res) => {
  req.session.destroy(err => {
    if (err) return res.redirect('/index.html');
    res.clearCookie('connect.sid');
    res.redirect('/login.html');
  });
});

// --- Pages ---
app.get('/', checkAuth, (_req, res) => res.sendFile(path.join(__dirname, 'index.html')));
app.get('/index.html', checkAuth, (_req, res) => res.sendFile(path.join(__dirname, 'index.html')));
app.get('/details.html', checkAuth, (_req, res) => res.sendFile(path.join(__dirname, 'details.html')));
app.get('/manuals.html', checkAuth, (_req, res) => res.sendFile(path.join(__dirname, 'manuals.html')));

// PDFs (protected)
app.use('/documents', checkAuth, express.static(path.join(__dirname, 'documents')));

// Public assets
app.use(express.static(__dirname));

// --- Upload (image optional) ---
const upload = multer({ storage: multer.memoryStorage() });

// --- Gemini settings ---
const safetySettings = [
  { category: HarmCategory.HARM_CATEGORY_HARASSMENT,        threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,       threshold: HarmBlockThreshold.BLOCK_NONE },
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
  model: 'text-embedding-004', // 768-dim
});

// --- Initialize Pinecone store ---
async function initializeVectorStore() {
  try {
    console.log('🔧 Initializing Pinecone connection...');
    const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
    const index = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    // More stable across versions
    vectorStore = await PineconeStore.fromExistingIndex(embeddingsModel, {
      pineconeIndex: index,
    });

    console.log('✅ Pinecone vector store initialized.');

    const stats = await index.describeIndexStats();
    const recordCount =
      (stats?.namespaces?.['']?.recordCount) ??
      Object.values(stats?.namespaces || {}).reduce((sum, ns) => sum + (ns.recordCount || 0), 0);

    if (!recordCount) {
      console.log('📚 Index is empty. Loading documents into Pinecone...');
      await loadDocumentsIntoPinecone();
    } else {
      console.log(`✅ Index already contains ${recordCount} records.`);
    }
  } catch (error) {
    console.error('CRITICAL: Failed to initialize Pinecone vector store.', error);
    vectorStore = undefined;
  }
}

// --- Load PDFs/TXT into Pinecone ---
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
        const ext = path.extname(file).toLowerCase();

        try {
          let docsFromFile = [];
          if (ext === '.pdf') {
            const loader = new PDFLoader(filePath);
            docsFromFile = await loader.load();
          } else if (ext === '.txt') {
            const text = await fs.readFile(filePath, 'utf-8');
            docsFromFile.push({ pageContent: text, metadata: {} });
          } else {
            continue;
          }

          docsFromFile.forEach(doc => {
            doc.metadata.source = file.trim();            // e.g. PP11_CO_M300E.pdf
            doc.metadata.area   = area.trim();            // e.g. PP11
            doc.metadata.title  = path.parse(file).name;  // e.g. PP11_CO_M300E
          });

          allDocuments.push(...docsFromFile);
        } catch (fileErr) {
          console.error(`Could not process file: ${file}`, fileErr);
        }
      }
    }

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const splitDocs = await splitter.splitDocuments(allDocuments);

    console.log(`📤 Uploading ${splitDocs.length} chunks to Pinecone...`);
    const batchSize = 50;
    for (let i = 0; i < splitDocs.length; i += batchSize) {
      const batch = splitDocs.slice(i, i + batchSize);
      await vectorStore.addDocuments(batch);
      console.log(`📤 Uploaded batch ${Math.floor(i / batchSize) + 1} / ${Math.ceil(splitDocs.length / batchSize)}`);
      if (i + batchSize < splitDocs.length) await new Promise(r => setTimeout(r, 1000));
    }
    console.log('✅ All documents uploaded to Pinecone.');
  } catch (error) {
    console.error('CRITICAL: Failed to load documents into Pinecone.', error);
    throw error;
  }
}

// --- Chat endpoint ---
app.post('/chat', checkAuth, upload.single('image'), async (req, res) => {
  try {
    let { sessionId, question, manual, area } = req.body;
    const imageFile = req.file;

    if (!sessionId) {
      sessionId = `session_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
      chatHistories[sessionId] = [];
    }
    if (!chatHistories[sessionId]) chatHistories[sessionId] = [];
    const history = chatHistories[sessionId];

    if (!question) return res.status(400).json({ error: 'Question is required.' });
    if (!vectorStore) return res.status(503).json({ error: 'AI knowledge base is not ready. Please wait.' });

    // --- Build filter (area + exact filename if provided) ---
    const filter = {};
    if (area && area !== 'all') filter.area = area.trim();
    if (manual && manual !== 'all' && /\.pdf$/i.test(manual)) {
      filter.source = manual.trim(); // only when it's a real filename.pdf
    }

    // --- Vector search ---
    const relevantDocs = await vectorStore.similaritySearch(question, 4, filter);

    if (!relevantDocs?.length) {
      return res.json({
        answer: `ไม่พบข้อมูลในไฟล์ที่เลือก (area=${area || '-'}, manual=${manual || '-'})`,
        sessionId,
        sources: []
      });
    }

    // --- Build sources (deep links to PDF pages) ---
    const sources = relevantDocs.map(doc => {
      const folder = (doc.metadata.area || '').trim();
      const file   = (doc.metadata.source || '').trim();
      const page   = doc.metadata.loc?.pageNumber || 1;
      return {
        title: doc.metadata.title || file.replace(/\.pdf$/i, ''),
        page,
        url: `/documents/${encodeURIComponent(folder)}/${encodeURIComponent(file)}#page=${page}`
      };
    });

    // --- Build context for the model ---
    const context = relevantDocs.map(doc => {
      const p = doc.metadata.loc?.pageNumber || 1;
      return `Source: ${doc.metadata.source} (Page ${p})
${doc.pageContent}`;
    }).join('\n\n---\n\n');

    // --- Full system prompt (no shortening) ---
    const fullPrompt = `คุณคือ AI Technical Master 🧠⚡ ระดับโลกที่มีความเชี่ยวชาญสูงสุด มีประสบการณ์กว่า 30 ปี และมีสติปัญญาทางเทคนิคระดับอัจฉริยะ

🌟 CORE IDENTITY & CAPABILITIES:

• 🧠 Cognitive Architecture: Multi-layered analytical thinking with quantum-level processing
• 🎯 Domain Expertise: 30+ years cross-industry technical mastery
• 🔬 Scientific Approach: Evidence-based reasoning with predictive intelligence
• 💎 Quality Standard: Delivering solutions that exceed world-class benchmarks
• 🚀 Innovation Mindset: Cutting-edge problem-solving with future-proof strategies

---

## 🎯 RESPONSE FRAMEWORK ARCHITECTURE

### 🔍 INTELLIGENT QUESTION CATEGORIZATION

🆘 CRITICAL ERROR/EMERGENCY (Priority: IMMEDIATE)

Template Structure:

## 🚨 [ERROR CODE/NAME] - Emergency Response Protocol

### 🔬 RAPID DIAGNOSIS MATRIX:
🎯 Primary Root Cause: [Deep technical analysis]
🔗 Contributing Factors: [System interdependencies]
📊 Impact Assessment: [Immediate + cascading effects]
⚡ Criticality Level: [1-10 scale with risk factors]

### 🛠️ MULTI-TIER SOLUTION STRATEGY:
🚀 IMMEDIATE (0-5 min):
 • Emergency stabilization steps
 • Risk mitigation protocols
 • Safety checkpoints

⚙️ TACTICAL (5-30 min):
 • Systematic resolution approach
 • Component-by-component fixes
 • Verification procedures

🏗️ STRATEGIC (30+ min):
 • Comprehensive system overhaul
 • Performance optimization
 • Future-proofing measures

### 🛡️ PREVENTION & RESILIENCE:
📋 Early Warning System: [Predictive indicators]
🔄 Maintenance Protocol: [Scheduled interventions]
📈 Monitoring Dashboard: [Real-time health checks]
🎯 Optimization Roadmap: [Continuous improvement]

### 🧠 EXPERT INTELLIGENCE INSIGHTS:
💡 Technical Deep-Dive: [Advanced theoretical foundation]
🎓 Best Practice Wisdom: [Industry-proven methodologies]
🔮 Future Trend Analysis: [Emerging technology considerations]

---

💡 KNOWLEDGE/EXPLANATION (Priority: COMPREHENSIVE)

Template Structure:

## 🎓 [CONCEPT/TOPIC] - Expert Knowledge Transfer

### 🌟 CONCEPTUAL FOUNDATION:
[Clear, intuitive explanation connecting to real-world applications]

### 🏗️ TECHNICAL ARCHITECTURE:
🧩 Core Components: [Fundamental building blocks]
⚙️ Operating Mechanisms: [How it actually works]
🔄 Process Flow: [Step-by-step workflow]
🌐 System Integration: [How it connects to broader systems]

### 🏭 REAL-WORLD APPLICATIONS:
💼 Industry Use Cases: [Specific examples across sectors]
📊 Performance Metrics: [Measurable outcomes]
💰 Business Impact: [ROI and value creation]
🎯 Implementation Strategies: [Practical deployment approaches]

### 🔬 SCIENTIFIC FOUNDATION:
🧪 Underlying Principles: [Scientific/mathematical basis]
📐 Formulas & Calculations: [Quantitative relationships]
🌐 Industry Standards: [Compliance and best practices]
📚 Research Evidence: [Supporting studies and data]

### 🚀 INNOVATION HORIZON:
🔮 Emerging Trends: [Next-generation developments]
💡 Technology Evolution: [Future possibilities]
📈 Market Dynamics: [Industry transformation patterns]
⚡ Disruption Potential: [Revolutionary changes ahead]

---

🔧 TUTORIAL/HOW-TO (Priority: MASTERY)

Template Structure:

## ⚙️ [PROCESS/SKILL] - Master-Level Implementation Guide

### 📋 PRE-EXECUTION CHECKLIST:
🔧 Required Tools: [Complete equipment list]
📚 Knowledge Prerequisites: [Essential background]
⚠️ Safety Protocols: [Risk management]
🖥️ System Requirements: [Technical specifications]
⏱️ Time Allocation: [Realistic timeline]

### 🎯 EXECUTION EXCELLENCE PATHWAY:

PHASE 1: STRATEGIC PREPARATION
- Environment setup and validation
- Resource verification and backup plans
- Risk assessment and mitigation strategies
- Quality checkpoints establishment

PHASE 2: SYSTEMATIC EXECUTION
- Foundation establishment
- Core implementation steps
- Progressive validation
- Performance optimization

PHASE 3: VALIDATION & OPTIMIZATION
- Comprehensive testing protocols
- Performance benchmarking
- Error handling verification
- Documentation and handover

### 🎖️ QUALITY ASSURANCE FRAMEWORK:
📊 Performance Metrics: [Success criteria]
🔍 Testing Procedures: [Validation methods]
🚨 Troubleshooting Guide: [Common issues + solutions]
📈 Optimization Techniques: [Enhancement strategies]

### 🏆 MASTERY-LEVEL INSIGHTS:
💡 Professional Shortcuts: [Efficiency techniques]
🎯 Advanced Strategies: [Expert-level approaches]
🔮 Future-Proof Methods: [Scalable solutions]
⚡ Performance Hacks: [Optimization secrets]

---

## 🎨 VISUAL EXCELLENCE & FORMATTING

### 🚦 PRIORITY CLASSIFICATION SYSTEM:
- 🔴 CRITICAL: Life/business-threatening issues requiring immediate action
- 🟠 HIGH: Significant impact on operations, needs urgent attention
- 🟡 MEDIUM: Important but manageable, scheduled resolution
- 🟢 LOW/GOOD: Minor issues or positive status indicators
- 🔵 INFO: Additional context and supplementary information
- 🟣 EXPERT: Advanced-level insights for specialists
- ⚫ WARNING: Caution required, potential risks identified

### 📱 MOBILE-OPTIMIZED DESIGN:
• Scannable Headers
• Bite-sized Content
• Strategic White Space
• Logical Flow
• Visual Anchors

### 🎯 ENGAGEMENT OPTIMIZATION:
• Hook Opening
• Progressive Disclosure
• Action-Oriented
• Value Stacking
• Memorable Formatting

---

## 🧠 ADVANCED COGNITIVE PROCESSING

### 🎭 CONTEXT-AWARE INTELLIGENCE:
- Question Intent Analysis
- User Profile Adaptation
- Historical Context
- Domain Knowledge Mapping
- Dynamic Response Optimization

### 🚀 MULTI-DIMENSIONAL ANALYSIS:
- Technical Depth
- Business Context
- Risk Assessment
- Implementation Feasibility
- Future Scalability

### 🎖️ EXPERT-LEVEL STANDARDS:
- Data-Driven Insights
- Precision Targeting
- Value-Added Intelligence
- Continuous Improvement
- Innovation Integration

---

## 🏆 RESPONSE EXCELLENCE CHECKLIST
GOLD STANDARD REQUIREMENTS:
- ตอบจาก CONTEXT เท่านั้น ถ้าไม่พบให้บอกว่าไม่พบ
- ลิงก์อ้างอิงให้ client เรนเดอร์จาก 'sources' ที่ระบบส่งคืน (อย่าเดา URL เอง)

---

## 📚 KNOWLEDGE INTEGRATION SYSTEM

Available Knowledge Base:
[จะถูกเติมจาก CONTEXT ด้านล่าง]

Conversation History Integration:
[ระบบจะเติมจาก history เพื่อความต่อเนื่อง]
`;

    const enhancedQuestion = `User Request: "${question}"`;

    const promptParts = [{ text: fullPrompt }];
    if (context) {
      promptParts.push({ text: `--- KNOWLEDGE BASE CONTEXT ---\n${context}` });
    }
    promptParts.push({ text: `--- CURRENT MISSION ---\n${enhancedQuestion}` });

    if (imageFile) {
      promptParts.push({ text: 'วิเคราะห์รูปภาพนี้ประกอบด้วย:' });
      promptParts.push({
        inlineData: {
          data: imageFile.buffer.toString('base64'),
          mimeType: imageFile.mimetype,
        },
      });
    }

    const result = await generativeModel.generateContent({
      contents: [{ role: 'user', parts: promptParts }],
    });
    const answer = result.response.text();

    chatHistories[sessionId].push({ question, answer });
    return res.json({ answer, sessionId, sources });
  } catch (error) {
    console.error('Error in /chat endpoint:', error);
    return res.status(500).json({ error: 'Failed to get response from AI.' });
  }
});

// --- Manuals manifest (for dropdown) ---
app.get('/api/manuals', checkAuth, async (_req, res) => {
  try {
    const documentsBasePath = path.join(__dirname, 'documents');
    const manualDatabase = {};

    const areaFolders = (await fs.readdir(documentsBasePath, { withFileTypes: true }))
      .filter(d => d.isDirectory())
      .map(d => d.name);

    for (const area of areaFolders) {
      const areaPath = path.join(documentsBasePath, area.trim());
      const files = await fs.readdir(areaPath);

      const areaKey = area.trim().toLowerCase();
      manualDatabase[areaKey] = {
        name: area.trim(),
        files: files.map(fileName => {
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
            name: trimmedFileName,                               // real filename for filtering
            path: `documents/${area.trim()}/${trimmedFileName}`, // relative path
            displayName,
            image: imagePath,
          };
        }),
      };
    }

    return res.json(manualDatabase);
  } catch (error) {
    console.error('Error creating manuals manifest:', error);
    return res.status(500).json({ error: 'Could not retrieve manual list.' });
  }
});

// --- Start server ---
async function startServer() {
  await initializeVectorStore();

  if (vectorStore) {
    app.listen(port, () => {
      console.log(`✅ Backend server is running on port ${port}`);
      console.log(`🔗 Connected to Pinecone index: ${process.env.PINECONE_INDEX_NAME}`);
    });
  } else {
    console.error('❌ Server startup failed: vector store not initialized.');
    process.exit(1);
  }
}

startServer();
