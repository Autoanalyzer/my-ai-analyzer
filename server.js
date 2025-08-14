// server.js (Fixed version with proper Pinecone integration and deep links)

require('dotenv').config();

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs/promises');
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
const { PDFLoader } = require('langchain/document_loaders/fs/pdf');

const { Pinecone } = require('@pinecone-database/pinecone');

const app = express();
app.set('trust proxy', 1);

const port = process.env.PORT || 5500;

// --- Environment Variables Check ---
const requiredEnvVars = ['GEMINI_API_KEY', 'PINECONE_API_KEY', 'PINECONE_INDEX_NAME'];
for (const envVar of requiredEnvVars) {
  if (!process.env[envVar]) {
    console.error(`❌ Missing required environment variable: ${envVar}`);
    process.exit(1);
  }
}

// Set INGEST_ON_BOOT default value
const INGEST_ON_BOOT = process.env.INGEST_ON_BOOT !== 'false';

// --- Users & Session ---
const users = [
  { id: 1, username: 'admin', password: 'password123' },
  { id: 2, username: 'user',  password: 'password456' },
];

app.use(session({
  secret: process.env.SESSION_SECRET || 'your_super_secret_key',
  resave: false,
  saveUninitialized: false,
  cookie: { secure: false, maxAge: 60 * 60 * 1000 },
}));

const chatHistories = {};
let vectorStore; // PineconeStore
let pineconeIndex;

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
    pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    // Create PineconeStore instance
    vectorStore = await PineconeStore.fromExistingIndex(embeddingsModel, {
      pineconeIndex: pineconeIndex,
    });

    console.log('✅ Pinecone vector store initialized.');

    // Check if index has data
    const stats = await pineconeIndex.describeIndexStats();
    const recordCount = stats?.totalRecordCount || 
      (stats?.namespaces?.['']?.recordCount) ||
      Object.values(stats?.namespaces || {}).reduce((sum, ns) => sum + (ns.recordCount || 0), 0);

    console.log(`📊 Index statistics:`, stats);
    console.log(`📈 Total records: ${recordCount}`);

    if (!recordCount || recordCount === 0) {
      if (INGEST_ON_BOOT) {
        console.log('📚 Index is empty. Loading documents into Pinecone...');
        await loadDocumentsIntoPinecone();
      } else {
        console.log('⏭️ Index empty, but INGEST_ON_BOOT=false. Skipping ingest on boot.');
      }
    } else {
      console.log(`✅ Index already contains ${recordCount} records.`);
    }

  } catch (error) {
    console.error('CRITICAL: Failed to initialize Pinecone vector store.', error);
    vectorStore = undefined;
    throw error;
  }
}

// --- Load PDFs/TXT into Pinecone ---
async function loadDocumentsIntoPinecone() {
  try {
    const documentsBasePath = path.join(__dirname, 'documents');
    
    // Check if documents directory exists
    try {
      await fs.access(documentsBasePath);
    } catch (error) {
      console.error('❌ Documents directory not found:', documentsBasePath);
      throw new Error('Documents directory not found');
    }

    const allDocuments = [];

    const areaFolders = (await fs.readdir(documentsBasePath, { withFileTypes: true }))
      .filter(d => d.isDirectory())
      .map(d => d.name);

    console.log('📁 Found area folders:', areaFolders);

    for (const area of areaFolders) {
      const areaPath = path.join(documentsBasePath, area);
      const files = await fs.readdir(areaPath);
      
      console.log(`📁 Processing area "${area}" with ${files.length} files`);

      for (const file of files) {
        const filePath = path.join(areaPath, file);
        const ext = path.extname(file).toLowerCase();

        try {
          let docsFromFile = [];
          if (ext === '.pdf') {
            console.log(`📄 Loading PDF: ${file}`);
            const loader = new PDFLoader(filePath);
            docsFromFile = await loader.load();
          } else if (ext === '.txt') {
            console.log(`📄 Loading TXT: ${file}`);
            const text = await fs.readFile(filePath, 'utf-8');
            docsFromFile.push({ 
              pageContent: text, 
              metadata: { 
                loc: { pageNumber: 1 }
              } 
            });
          } else {
            console.log(`⏭️ Skipping file: ${file} (unsupported extension: ${ext})`);
            continue;
          }

          // Add metadata to each document
          docsFromFile.forEach((doc, index) => {
            doc.metadata.source = file.trim();            // e.g. PP11_CO_M300E.pdf
            doc.metadata.area   = area.trim();            // e.g. PP11
            doc.metadata.title  = path.parse(file).name;  // e.g. PP11_CO_M300E
            
            // Ensure pageNumber exists
            if (!doc.metadata.loc) {
              doc.metadata.loc = { pageNumber: index + 1 };
            }
            if (!doc.metadata.loc.pageNumber) {
              doc.metadata.loc.pageNumber = index + 1;
            }
            
            console.log(`📖 Document metadata:`, {
              source: doc.metadata.source,
              area: doc.metadata.area,
              title: doc.metadata.title,
              page: doc.metadata.loc.pageNumber,
              contentLength: doc.pageContent.length
            });
          });

          allDocuments.push(...docsFromFile);
          console.log(`✅ Processed ${file}: ${docsFromFile.length} documents`);
        } catch (fileErr) {
          console.error(`❌ Could not process file: ${file}`, fileErr);
        }
      }
    }

    if (allDocuments.length === 0) {
      throw new Error('No documents found to process');
    }

    console.log(`📚 Total documents loaded: ${allDocuments.length}`);

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const splitDocs = await splitter.splitDocuments(allDocuments);
    
    console.log(`✂️ Split into ${splitDocs.length} chunks`);

    // Validate split documents
    splitDocs.forEach((doc, index) => {
      if (!doc.metadata.source || !doc.metadata.area) {
        console.error(`❌ Invalid metadata for chunk ${index}:`, doc.metadata);
      }
    });

    console.log(`📤 Uploading ${splitDocs.length} chunks to Pinecone...`);
    const batchSize = 50;
    for (let i = 0; i < splitDocs.length; i += batchSize) {
      const batch = splitDocs.slice(i, i + batchSize);
      try {
        await vectorStore.addDocuments(batch);
        console.log(`📤 Uploaded batch ${Math.floor(i / batchSize) + 1} / ${Math.ceil(splitDocs.length / batchSize)}`);
      } catch (batchError) {
        console.error(`❌ Error uploading batch ${Math.floor(i / batchSize) + 1}:`, batchError);
        throw batchError;
      }
      
      // Add delay between batches to avoid rate limits
      if (i + batchSize < splitDocs.length) {
        await new Promise(r => setTimeout(r, 1000));
      }
    }
    console.log('✅ All documents uploaded to Pinecone.');

    // Verify upload
    const stats = await pineconeIndex.describeIndexStats();
    console.log('📊 Updated index statistics:', stats);

  } catch (error) {
    console.error('CRITICAL: Failed to load documents into Pinecone.', error);
    throw error;
  }
}

// --- Health check endpoint ---
app.get('/health', (req, res) => {
  const status = {
    server: 'running',
    vectorStore: vectorStore ? 'ready' : 'not ready',
    timestamp: new Date().toISOString()
  };
  res.json(status);
});

// --- Debug endpoint to test vector search ---
app.get('/api/debug/search/:query', checkAuth, async (req, res) => {
  try {
    if (!vectorStore) {
      return res.status(503).json({ error: 'Vector store not ready' });
    }

    const query = req.params.query;
    const results = await vectorStore.similaritySearch(query, 3);
    
    res.json({
      query,
      results: results.map(doc => ({
        content: doc.pageContent.substring(0, 200) + '...',
        metadata: doc.metadata
      }))
    });
  } catch (error) {
    console.error('Debug search error:', error);
    res.status(500).json({ error: error.message });
  }
});

// --- Chat endpoint (Fixed) ---
app.post('/chat', checkAuth, upload.single('image'), async (req, res) => {
  try {
    let { sessionId, question, manual, area } = req.body;
    const imageFile = req.file;

    console.log('[DEBUG] Chat request:', { sessionId, question, manual, area });

    if (!sessionId) {
      sessionId = `session_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
      chatHistories[sessionId] = [];
    }
    if (!chatHistories[sessionId]) chatHistories[sessionId] = [];
    const history = chatHistories[sessionId];

    if (!question) return res.status(400).json({ error: 'Question is required.' });
    if (!vectorStore) {
      console.error('[ERROR] Vector store not ready');
      return res.status(503).json({ error: 'AI knowledge base is not ready. Please wait and try again.' });
    }

    // --- Build metadata filter (manual beats area + case insensitive area) ---
    const areaParam   = (area   || '').trim();
    const manualParam = (manual || '').trim();

    let filter; // undefined if no filtering needed

    if (manualParam && /\.pdf$/i.test(manualParam)) {
      // Case: specific file selected - filter by filename only
      filter = { source: manualParam };
      console.log('[DEBUG] Filtering by manual:', manualParam);
    } else if (areaParam && areaParam.toLowerCase() !== 'all') {
      // Case: area selected but no specific file - filter by area (case-insensitive)
      filter = { area: { $in: [areaParam, areaParam.toUpperCase(), areaParam.toLowerCase()] } };
      console.log('[DEBUG] Filtering by area:', areaParam);
    } else {
      filter = undefined;
      console.log('[DEBUG] No filtering applied');
    }

    console.log('[DEBUG] Pinecone filter =', JSON.stringify(filter));

    // --- Vector search ---
    let relevantDocs;
    try {
      relevantDocs = await vectorStore.similaritySearch(question, 4, filter);
      console.log(`[DEBUG] Found ${relevantDocs.length} relevant documents`);
    } catch (searchError) {
      console.error('[ERROR] Vector search failed:', searchError);
      return res.status(500).json({ error: 'Search failed. Please try again.' });
    }

    if (!relevantDocs?.length) {
      const filterInfo = filter ? 
        `(area=${areaParam || '-'}, manual=${manualParam || '-'})` : 
        '';
      return res.json({
        answer: `ไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูล ${filterInfo}`,
        sessionId,
        sources: []
      });
    }

    // --- Build sources (deep links to PDF pages) ---
    const sources = relevantDocs.map(doc => {
      const folder = (doc.metadata.area || '').trim();
      const file   = (doc.metadata.source || '').trim();
      const page   = doc.metadata.loc?.pageNumber || 1;
      
      console.log('[DEBUG] Source metadata:', { folder, file, page });
      
      return {
        title: doc.metadata.title || file.replace(/\.pdf$/i, ''),
        page,
        url: `/documents/${encodeURIComponent(folder)}/${encodeURIComponent(file)}#page=${page}`
      };
    });

    console.log('[DEBUG] Generated sources:', sources);

    // --- Build context for the model ---
    const context = relevantDocs.map(doc => {
      const p = doc.metadata.loc?.pageNumber || 1;
      return `Source: ${doc.metadata.source} (Page ${p})
${doc.pageContent}`;
    }).join('\n\n---\n\n');

    // --- Simplified system prompt for better performance ---
    const systemPrompt = `คุณคือ AI Technical Assistant ที่เชี่ยวชาญด้านเทคนิค

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

    const enhancedQuestion = `คำถาม: "${question}"

ข้อมูลจากเอกสาร:
${context}`;

    const promptParts = [{ text: systemPrompt }];
    promptParts.push({ text: enhancedQuestion });

    if (imageFile) {
      promptParts.push({ text: 'วิเคราะห์รูปภาพนี้ประกอบด้วย:' });
      promptParts.push({
        inlineData: {
          data: imageFile.buffer.toString('base64'),
          mimeType: imageFile.mimetype,
        },
      });
    }

    // --- Generate response ---
    let answer;
    try {
      const result = await generativeModel.generateContent({
        contents: [{ role: 'user', parts: promptParts }],
      });
      answer = result.response.text();
      console.log('[DEBUG] Generated answer length:', answer.length);
    } catch (aiError) {
      console.error('[ERROR] AI generation failed:', aiError);
      return res.status(500).json({ error: 'Failed to generate response. Please try again.' });
    }

    // Store in chat history
    chatHistories[sessionId].push({ question, answer });
    
    return res.json({ answer, sessionId, sources });
  } catch (error) {
    console.error('Error in /chat endpoint:', error);
    return res.status(500).json({ error: 'Internal server error. Please try again.' });
  }
});

// --- Manuals manifest (for dropdown) ---
app.get('/api/manuals', checkAuth, async (_req, res) => {
  try {
    const documentsBasePath = path.join(__dirname, 'documents');
    
    // Check if documents directory exists
    try {
      await fs.access(documentsBasePath);
    } catch (error) {
      console.error('❌ Documents directory not found:', documentsBasePath);
      return res.status(404).json({ error: 'Documents directory not found' });
    }

    const manualDatabase = {};

    const areaFolders = (await fs.readdir(documentsBasePath, { withFileTypes: true }))
      .filter(d => d.isDirectory())
      .map(d => d.name);

    for (const area of areaFolders) {
      const areaPath = path.join(documentsBasePath, area.trim());
      
      let files;
      try {
        files = await fs.readdir(areaPath);
      } catch (error) {
        console.error(`❌ Cannot read area folder: ${area}`, error);
        continue;
      }

      const areaKey = area.trim().toLowerCase();
      manualDatabase[areaKey] = {
        name: area.trim(),
        files: files
          .filter(fileName => fileName.trim().length > 0) // Filter out empty names
          .map(fileName => {
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

    console.log('[DEBUG] Manual database created:', Object.keys(manualDatabase));
    return res.json(manualDatabase);
  } catch (error) {
    console.error('Error creating manuals manifest:', error);
    return res.status(500).json({ error: 'Could not retrieve manual list.' });
  }
});

// --- Force re-index endpoint (for debugging) ---
app.post('/api/admin/reindex', checkAuth, async (req, res) => {
  try {
    if (!vectorStore) {
      return res.status(503).json({ error: 'Vector store not ready' });
    }

    console.log('🔄 Force re-indexing requested...');
    await loadDocumentsIntoPinecone();
    
    res.json({ message: 'Re-indexing completed successfully' });
  } catch (error) {
    console.error('Re-indexing failed:', error);
    res.status(500).json({ error: 'Re-indexing failed: ' + error.message });
  }
});

// --- Error handler middleware ---
app.use((error, req, res, next) => {
  console.error('[ERROR] Unhandled error:', error);
  res.status(500).json({ error: 'Internal server error' });
});

// --- Start server ---
async function startServer() {
  // 1) Listen on port first so Render can pass health checks
  const server = app.listen(port, () => {
    console.log(`✅ Backend server is running on port ${port}`);
    console.log(`🌐 Health check: http://localhost:${port}/health`);
  });

  // 2) Initialize Pinecone in background (non-blocking)
  initializeVectorStore()
    .then(() => {
      console.log(`🔗 Connected to Pinecone index: ${process.env.PINECONE_INDEX_NAME}`);
      console.log(`🤖 AI Chatbot is ready to answer questions!`);
    })
    .catch(err => {
      console.error('CRITICAL: initializeVectorStore failed:', err);
      // Don't exit - let /chat return 503 until ready
    });

  // Graceful shutdown
  process.on('SIGTERM', () => {
    console.log('🛑 SIGTERM received, shutting down gracefully...');
    server.close(() => {
      console.log('✅ Server closed');
      process.exit(0);
    });
  });
}

startServer();