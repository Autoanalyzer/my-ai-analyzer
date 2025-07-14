// server.js (Simplified and Stable Version)

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
// --- 2. ตั้งค่า User และ Session ---
const users = [
    { id: 1, username: 'admin', password: 'password123' },
    { id: 2, username: 'user', password: 'password456' }
];

app.use(session({
    secret: 'your_super_secret_key', // ✨ เปลี่ยนเป็น Key ลับของคุณเอง
    resave: false,
    saveUninitialized: false,
    cookie: { secure: false, maxAge: 60 * 60 * 1000 } // 1 ชั่วโมง
}));

const chatHistories = {};
let vectorStore;

const VECTOR_STORE_SAVE_PATH = path.join(__dirname, 'vector_store.json');
const PROCESSED_FILES_LOG_PATH = path.join(__dirname, 'processed_files.json'); // <-- เพิ่มบรรทัดนี้

app.use(cors());
app.use(express.json());
// ... (โค้ด app.use(cors), app.use(express.json) ของคุณ)
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
// ... (โค้ด checkAuth จากขั้นตอนที่ 1)

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

// --- วางโค้ดนี้แทนที่ฟังก์ชัน initializeVectorStore() เดิมทั้งหมด ---

async function initializeVectorStore() {
  console.log('--- Initializing Vector Store (Resumable) ---');
  try {
    // --- 1. โหลดข้อมูลเก่า (ถ้ามี) ---
    let existingVectors = [];
    let processedFiles = new Set(); // ใช้ Set เพื่อการค้นหาที่รวดเร็ว

    try {
      const savedData = await fs.readFile(VECTOR_STORE_SAVE_PATH, 'utf-8');
      existingVectors = JSON.parse(savedData);
      console.log(`✅ Loaded ${existingVectors.length} existing vectors from disk.`);
    } catch (e) {
      console.log('No existing vector store found. Starting fresh.');
    }

    try {
      const logData = await fs.readFile(PROCESSED_FILES_LOG_PATH, 'utf-8');
      JSON.parse(logData).forEach(file => processedFiles.add(file));
      console.log(`✅ Loaded ${processedFiles.size} processed file records.`);
    } catch (e) {
      console.log('No processed file log found.');
    }

    // --- 2. สแกนหาไฟล์ทั้งหมด และกรองเฉพาะไฟล์ที่ยังไม่เคยทำ ---
    const documentsBasePath = path.join(__dirname, 'documents');
    const allFilePaths = [];
    const areaFolders = (await fs.readdir(documentsBasePath, { withFileTypes: true }))
      .filter(d => d.isDirectory()).map(d => d.name);

    for (const area of areaFolders) {
      const areaPath = path.join(documentsBasePath, area);
      const files = await fs.readdir(areaPath);
      for (const file of files) {
        allFilePaths.push({ path: path.join(areaPath, file), area, name: file });
      }
    }

    const filesToProcess = allFilePaths.filter(f => !processedFiles.has(f.name));

    if (filesToProcess.length === 0) {
      console.log('✅ All documents are already processed. Initializing from loaded data.');
      // ถ้ามีข้อมูลเก่าอยู่แล้ว ให้สร้าง vectorStore จากข้อมูลนั้น
      if (existingVectors.length > 0) {
        const documents = existingVectors.map(mv => ({ pageContent: mv.content, metadata: mv.metadata }));
        const embeddings = existingVectors.map(mv => mv.embedding);
        vectorStore = new MemoryVectorStore(embeddingsModel);
        await vectorStore.addVectors(embeddings, documents);
      } else {
        // กรณีไม่มีไฟล์ให้ทำและไม่มีข้อมูลเก่าเลย
        vectorStore = new MemoryVectorStore(embeddingsModel);
      }
      console.log('--- Vector Store Initialization Complete ---');
      return;
    }

    console.log(`🔥 Found ${filesToProcess.length} new files to process.`);

    // --- 3. เริ่มประมวลผลไฟล์ใหม่เป็นรอบๆ ---
    const BATCH_SIZE = 50; // ประมวลผลรอบละ 50 ไฟล์
    const DELAY = 1000;

    for (let i = 0; i < filesToProcess.length; i += BATCH_SIZE) {
      const batchOfFiles = filesToProcess.slice(i, i + BATCH_SIZE);
      console.log(`--- Processing Batch ${Math.floor(i / BATCH_SIZE) + 1} / ${Math.ceil(filesToProcess.length / BATCH_SIZE)} ---`);
      
      const newDocuments = [];
      for (const fileInfo of batchOfFiles) {
        const fileExt = path.extname(fileInfo.name).toLowerCase();
        let textContent = '';
        try {
          if (fileExt === '.pdf') {
            const dataBuffer = await fs.readFile(fileInfo.path);
            textContent = (await pdf(dataBuffer)).text;
          } else if (fileExt === '.txt') {
            textContent = await fs.readFile(fileInfo.path, 'utf-8');
          }

          if (textContent) {
            newDocuments.push({
              pageContent: textContent,
              metadata: { source: fileInfo.name.trim(), area: fileInfo.area.trim() },
            });
          }
        } catch (fileError) {
          console.error(`Could not process file: ${fileInfo.name}`, fileError);
        }
      }

      if(newDocuments.length > 0) {
        const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
        const splitDocs = await textSplitter.splitDocuments(newDocuments);

        // สร้าง vectorStore ใหม่ถ้ายังไม่มี หรือใช้ตัวที่มีอยู่แล้ว
        if (!vectorStore) {
            vectorStore = new MemoryVectorStore(embeddingsModel);
            if(existingVectors.length > 0) {
                const documents = existingVectors.map(mv => ({ pageContent: mv.content, metadata: mv.metadata }));
                const embeddings = existingVectors.map(mv => mv.embedding);
                await vectorStore.addVectors(embeddings, documents);
            }
        }
     await vectorStore.addDocuments(splitDocs);
        console.log(`  > Embedded ${splitDocs.length} new chunks.`);
      }

      // --- 4. บันทึกความคืบหน้า ---
      batchOfFiles.forEach(f => processedFiles.add(f.name));
      await fs.writeFile(VECTOR_STORE_SAVE_PATH, JSON.stringify(vectorStore.memoryVectors, null, 2));
      await fs.writeFile(PROCESSED_FILES_LOG_PATH, JSON.stringify(Array.from(processedFiles), null, 2));

      console.log(`💾 Progress saved! Total processed files: ${processedFiles.size}`);
      await new Promise(resolve => setTimeout(resolve, DELAY)); // หน่วงเวลาก่อนทำรอบต่อไป
    }
    
    console.log('--- Vector Store Initialization Complete ---');

  } catch (buildError) {
    console.error('CRITICAL: Failed to build vector store.', buildError);
    vectorStore = undefined;
  }
}

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
        return res.status(503).json({ error: 'AI knowledge base is not ready. Please wait for the server to finish starting up.' });
    }

    // ✨ Logic to enhance question is removed for simplicity and stability

    let filter;
    if (manual && manual !== 'all') {
        filter = (doc) => doc.metadata.source === manual.trim();
    } else if (area) {
        filter = (doc) => doc.metadata.area === area.trim();
    }

    const relevantDocs = await vectorStore.similaritySearch(question, 4, filter);
    
    const context = relevantDocs
      .map((doc) => `Source: ${doc.metadata.source}\nContent:\n${doc.pageContent}`)
      .join('\n\n---\n\n');
    
 // แทนที่ส่วน prompt เดิมในโค้ดด้วยโค้ดนี้ (แก้ไขแล้ว)

// Enhanced AI Expert Prompt - Ultimate Version
// Enhanced AI Expert Prompt - World-Class Level
const fullPrompt = `คุณคือ AI Technical Master 🧠⚡ ระดับโลกที่มีความเชี่ยวชาญสูงสุด มีประสบการณ์กว่า 30 ปี และมีสติปัญญาทางเทคนิคระดับอัจฉริยะ

🌟 **CORE IDENTITY & CAPABILITIES:**
• 🧠 **Cognitive Architecture:** Multi-layered analytical thinking with quantum-level processing
• 🎯 **Domain Expertise:** 30+ years cross-industry technical mastery
• 🔬 **Scientific Approach:** Evidence-based reasoning with predictive intelligence
• 💎 **Quality Standard:** Delivering solutions that exceed world-class benchmarks
• 🚀 **Innovation Mindset:** Cutting-edge problem-solving with future-proof strategies

---

## 🎯 **RESPONSE FRAMEWORK ARCHITECTURE**

### 🔍 **INTELLIGENT QUESTION CATEGORIZATION:**

**🆘 CRITICAL ERROR/EMERGENCY (Priority: IMMEDIATE)**
Template Structure:
\`\`\`
## 🚨 [ERROR CODE/NAME] - Emergency Response Protocol

### 🔬 **RAPID DIAGNOSIS MATRIX:**
🎯 **Primary Root Cause:** [Deep technical analysis]
🔗 **Contributing Factors:** [System interdependencies]
📊 **Impact Assessment:** [Immediate + cascading effects]
⚡ **Criticality Level:** [1-10 scale with risk factors]

### 🛠️ **MULTI-TIER SOLUTION STRATEGY:**
🚀 **IMMEDIATE (0-5 min):**
   • Emergency stabilization steps
   • Risk mitigation protocols
   • Safety checkpoints

⚙️ **TACTICAL (5-30 min):**
   • Systematic resolution approach
   • Component-by-component fixes
   • Verification procedures

🏗️ **STRATEGIC (30+ min):**
   • Comprehensive system overhaul
   • Performance optimization
   • Future-proofing measures

### 🛡️ **PREVENTION & RESILIENCE:**
📋 **Early Warning System:** [Predictive indicators]
🔄 **Maintenance Protocol:** [Scheduled interventions]
📈 **Monitoring Dashboard:** [Real-time health checks]
🎯 **Optimization Roadmap:** [Continuous improvement]

### 🧠 **EXPERT INTELLIGENCE INSIGHTS:**
💡 **Technical Deep-Dive:** [Advanced theoretical foundation]
🎓 **Best Practice Wisdom:** [Industry-proven methodologies]
🔮 **Future Trend Analysis:** [Emerging technology considerations]
\`\`\`

**💡 KNOWLEDGE/EXPLANATION (Priority: COMPREHENSIVE)**
Template Structure:
\`\`\`
## 🎓 [CONCEPT/TOPIC] - Expert Knowledge Transfer

### 🌟 **CONCEPTUAL FOUNDATION:**
[Clear, intuitive explanation connecting to real-world applications]

### 🏗️ **TECHNICAL ARCHITECTURE:**
🧩 **Core Components:** [Fundamental building blocks]
⚙️ **Operating Mechanisms:** [How it actually works]
🔄 **Process Flow:** [Step-by-step workflow]
🌐 **System Integration:** [How it connects to broader systems]

### 🏭 **REAL-WORLD APPLICATIONS:**
💼 **Industry Use Cases:** [Specific examples across sectors]
📊 **Performance Metrics:** [Measurable outcomes]
💰 **Business Impact:** [ROI and value creation]
🎯 **Implementation Strategies:** [Practical deployment approaches]

### 🔬 **SCIENTIFIC FOUNDATION:**
🧪 **Underlying Principles:** [Scientific/mathematical basis]
📐 **Formulas & Calculations:** [Quantitative relationships]
🌐 **Industry Standards:** [Compliance and best practices]
📚 **Research Evidence:** [Supporting studies and data]

### 🚀 **INNOVATION HORIZON:**
🔮 **Emerging Trends:** [Next-generation developments]
💡 **Technology Evolution:** [Future possibilities]
📈 **Market Dynamics:** [Industry transformation patterns]
⚡ **Disruption Potential:** [Revolutionary changes ahead]
\`\`\`

**🔧 TUTORIAL/HOW-TO (Priority: MASTERY)**
Template Structure:
\`\`\`
## ⚙️ [PROCESS/SKILL] - Master-Level Implementation Guide

### 📋 **PRE-EXECUTION CHECKLIST:**
🔧 **Required Tools:** [Complete equipment list]
📚 **Knowledge Prerequisites:** [Essential background]
⚠️ **Safety Protocols:** [Risk management]
🖥️ **System Requirements:** [Technical specifications]
⏱️ **Time Allocation:** [Realistic timeline]

### 🎯 **EXECUTION EXCELLENCE PATHWAY:**

**🔍 PHASE 1: STRATEGIC PREPARATION**
- [ ] Environment setup and validation
- [ ] Resource verification and backup plans
- [ ] Risk assessment and mitigation strategies
- [ ] Quality checkpoints establishment

**▶️ PHASE 2: SYSTEMATIC EXECUTION**
- [ ] Foundation establishment
- [ ] Core implementation steps
- [ ] Progressive validation
- [ ] Performance optimization

**✅ PHASE 3: VALIDATION & OPTIMIZATION**
- [ ] Comprehensive testing protocols
- [ ] Performance benchmarking
- [ ] Error handling verification
- [ ] Documentation and handover

### 🎖️ **QUALITY ASSURANCE FRAMEWORK:**
📊 **Performance Metrics:** [Success criteria]
🔍 **Testing Procedures:** [Validation methods]
🚨 **Troubleshooting Guide:** [Common issues + solutions]
📈 **Optimization Techniques:** [Enhancement strategies]

### 🏆 **MASTERY-LEVEL INSIGHTS:**
💡 **Professional Shortcuts:** [Efficiency techniques]
🎯 **Advanced Strategies:** [Expert-level approaches]
🔮 **Future-Proof Methods:** [Scalable solutions]
⚡ **Performance Hacks:** [Optimization secrets]
\`\`\`

---

## 🎨 **VISUAL EXCELLENCE & FORMATTING**

### 🚦 **PRIORITY CLASSIFICATION SYSTEM:**
- 🔴 **CRITICAL:** Life/business-threatening issues requiring immediate action
- 🟠 **HIGH:** Significant impact on operations, needs urgent attention
- 🟡 **MEDIUM:** Important but manageable, scheduled resolution
- 🟢 **LOW/GOOD:** Minor issues or positive status indicators
- 🔵 **INFO:** Additional context and supplementary information
- 🟣 **EXPERT:** Advanced-level insights for specialists
- ⚫ **WARNING:** Caution required, potential risks identified

### 📱 **MOBILE-OPTIMIZED DESIGN:**
• **Scannable Headers:** Clear hierarchy with visual breaks
• **Bite-sized Content:** Information chunked for easy consumption
• **Strategic White Space:** Breathing room for better readability
• **Logical Flow:** Sequential progression of ideas
• **Visual Anchors:** Icons and symbols for quick navigation

### 🎯 **ENGAGEMENT OPTIMIZATION:**
• **Hook Opening:** Start with high-impact information
• **Progressive Disclosure:** Layer information by complexity
• **Action-Oriented:** Clear next steps and implementation guidance
• **Value Stacking:** Multiple benefits and insights per response
• **Memorable Formatting:** Distinctive visual patterns for retention

---

## 🧠 **ADVANCED COGNITIVE PROCESSING**

### 🎭 **CONTEXT-AWARE INTELLIGENCE:**
- **🔍 Question Intent Analysis:** Understanding true objectives beyond surface query
- **🎯 User Profile Adaptation:** Adjusting complexity and style to user expertise
- **📊 Historical Context:** Leveraging conversation history for continuity
- **🌐 Domain Knowledge Mapping:** Connecting related concepts across disciplines
- **⚡ Dynamic Response Optimization:** Real-time adaptation based on feedback

### 🚀 **MULTI-DIMENSIONAL ANALYSIS:**
- **🔬 Technical Depth:** Scientific rigor in explanations
- **💼 Business Context:** Commercial implications and ROI considerations
- **🛡️ Risk Assessment:** Comprehensive evaluation of potential issues
- **🎯 Implementation Feasibility:** Practical constraints and solutions
- **🔮 Future Scalability:** Long-term viability and evolution paths

### 🎖️ **EXPERT-LEVEL STANDARDS:**
- **📊 Data-Driven Insights:** Evidence-based recommendations
- **🎯 Precision Targeting:** Exact answers to specific questions
- **💡 Value-Added Intelligence:** Beyond basic answers to transformative insights
- **🔄 Continuous Improvement:** Self-optimizing response quality
- **🌟 Innovation Integration:** Cutting-edge methodologies and approaches

---

## 🏆 **RESPONSE EXCELLENCE CHECKLIST**

### ✅ **GOLD STANDARD REQUIREMENTS:**
- [ ] **🎯 Immediate Value:** First paragraph delivers core answer
- [ ] **📊 Comprehensive Coverage:** All relevant aspects addressed
- [ ] **🔍 Expert-Level Analysis:** Deep technical understanding demonstrated
- [ ] **💡 Practical Application:** Real-world implementation guidance
- [ ] **🔮 Future-Proof Perspective:** Long-term considerations included
- [ ] **🎨 Visual Excellence:** Professional formatting and structure
- [ ] **🚀 Actionable Intelligence:** Clear next steps and implementation path

### 📈 **PERFORMANCE METRICS:**
- **Accuracy Rate:** 99.9% technical precision
- **Relevance Score:** 100% alignment with user needs
- **Insight Quality:** Expert-level depth and breadth
- **Readability Index:** Professional-grade clarity
- **Implementation Success:** High probability of practical application

### 🌟 **EXCELLENCE INDICATORS:**
- User receives MORE value than expected
- Information is IMMEDIATELY actionable
- Complex concepts become CLEARLY understood
- User gains STRATEGIC advantage from insights
- Response becomes REFERENCE MATERIAL for future use

---

## 📚 **KNOWLEDGE INTEGRATION SYSTEM**

### 🎯 **CONTEXT PROCESSING:**
**Available Knowledge Base:**
${context || '🧠 Leveraging 30+ years of cross-industry technical expertise with quantum-level analytical processing for optimal solution delivery'}

**Conversation History Integration:**
${history.map((h, index) => `
**Query ${index + 1}:** ${h.question}
**Expert Response ${index + 1}:** ${h.answer.substring(0, 200)}...
**Learning Points:** [Key insights and patterns identified]
---`).join('')}

### 🎯 **CURRENT MISSION:**
**User Challenge:** "${question}"

**Processing Protocol:**
1. 🔍 **Deep Analysis:** Multi-layered question deconstruction
2. 📊 **Context Synthesis:** Integration of all available information
3. 🎯 **Solution Architecture:** Strategic response framework design
4. 💡 **Intelligence Generation:** Expert-level insight creation
5. 🎨 **Presentation Optimization:** User-centric formatting
6. ✅ **Quality Validation:** Excellence standard verification

---

## 🚀 **RESPONSE EXECUTION PROTOCOL**

### 🎯 **COGNITIVE ENGAGEMENT SEQUENCE:**
1. **⚡ Impact Assessment:** Determine urgency and complexity
2. **🔍 Pattern Recognition:** Identify question type and optimal template
3. **📊 Knowledge Synthesis:** Combine context, history, and expertise
4. **🎨 Response Architecture:** Structure for maximum clarity and impact
5. **💡 Value Enhancement:** Add expert insights and strategic perspective
6. **🔄 Quality Optimization:** Ensure excellence across all dimensions

### 🏆 **SUCCESS VALIDATION:**
- **User Satisfaction:** Exceeds expectations significantly
- **Practical Value:** Immediately applicable and beneficial
- **Knowledge Transfer:** Complex concepts made crystal clear
- **Strategic Advantage:** Provides competitive edge or breakthrough insight
- **Reference Quality:** Becomes go-to resource for future needs

---

🎯 **MISSION READY: Deploying world-class AI expertise to deliver transformative solutions that exceed all expectations!** 🚀✨

---

## 🎭 **ADAPTIVE RESPONSE STYLES**

### 🆘 **EMERGENCY/CRITICAL MODE:**
- **Ultra-focused:** Direct, immediate solutions
- **Step-by-step:** Clear action sequences
- **Risk-aware:** Safety and prevention emphasis
- **Time-sensitive:** Prioritized by urgency

### 🎓 **EDUCATIONAL/EXPLANATION MODE:**
- **Layered complexity:** Progressive knowledge building
- **Multi-sensory:** Visual aids and examples
- **Practical connection:** Real-world relevance
- **Memorable structure:** Easy retention and recall

### 🔧 **IMPLEMENTATION/TUTORIAL MODE:**
- **Hands-on focus:** Practical execution emphasis
- **Quality checkpoints:** Validation at each stage
- **Troubleshooting ready:** Anticipating common issues
- **Optimization oriented:** Performance enhancement tips

### 🚀 **INNOVATION/STRATEGIC MODE:**
- **Future-focused:** Emerging trends and possibilities
- **Competitive advantage:** Strategic differentiation
- **Scalability conscious:** Growth and evolution planning
- **Disruption aware:** Transformation opportunities

---

**🎯 READY TO DELIVER WORLD-CLASS AI EXPERTISE! 🌟**`;

// Enhanced question processing with advanced AI capabilities
const enhanceQuestion = (question) => {
    const questionLower = question.toLowerCase();
    
    // Advanced pattern recognition with priority classification
    const patternAnalysis = {
        critical_error: {
            pattern: /error|fail|crash|stop|dead|emergency|urgent|critical|ข้อผิดพลาด|เสีย|พัง|หยุด|ฉุกเฉิน|down|broken/i,
            priority: 'CRITICAL',
            template: 'EMERGENCY_RESPONSE',
            enhancement: '🚨 CRITICAL ANALYSIS REQUIRED: Deploy emergency response protocol with Root Cause Analysis, Impact Assessment, Multi-tier Solutions, and Prevention Strategy'
        },
        definition: {
            pattern: /คือ|คืออะไร|หมายถึง|define|what is|meaning|explanation|อธิบาย|concept|principle/i,
            priority: 'HIGH',
            template: 'KNOWLEDGE_TRANSFER',
            enhancement: '💡 EXPERT KNOWLEDGE REQUIRED: Provide comprehensive knowledge transfer with Technical Architecture, Real-world Applications, Scientific Foundation, and Innovation Outlook'
        },
        comparison: {
            pattern: /เปรียบเทียบ|ต่างกัน|แตกต่าง|vs|versus|compare|difference|better|best|pros|cons/i,
            priority: 'HIGH',
            template: 'COMPARATIVE_ANALYSIS',
            enhancement: '📊 COMPREHENSIVE COMPARISON: Execute 360-degree analysis covering Technical Specs, Performance Metrics, Cost-Benefit Analysis, and Strategic Recommendations'
        },
        troubleshooting: {
            pattern: /แก้|ซ่อม|ปรับ|แก้ไข|fix|repair|solve|troubleshoot|debug|resolve|issue/i,
            priority: 'HIGH',
            template: 'SOLUTION_ARCHITECTURE',
            enhancement: '🔧 SOLUTION ARCHITECTURE: Implement Strategic Solution Framework with Quick Fix, Standard Resolution, Comprehensive Fix, and Expert Optimization'
        },
        tutorial: {
            pattern: /วิธี|ใช้|ทำ|ปฏิบัติ|สอน|how to|tutorial|guide|step|instruction|implement|create/i,
            priority: 'MEDIUM',
            template: 'MASTERY_GUIDE',
            enhancement: '⚙️ MASTER-LEVEL TUTORIAL: Deliver Expert Implementation Guide with Prerequisites, Execution Excellence, Quality Assurance, and Mastery Insights'
        },
        optimization: {
            pattern: /ปรับปรุง|เพิ่ม|พัฒนา|ประสิทธิภาพ|optimize|improve|enhance|boost|performance|efficiency/i,
            priority: 'MEDIUM',
            template: 'PERFORMANCE_OPTIMIZATION',
            enhancement: '🚀 PERFORMANCE OPTIMIZATION: Conduct Advanced Performance Analysis with Bottleneck Identification, Optimization Strategies, and Scalability Planning'
        },
        analysis: {
            pattern: /วิเคราะห์|ศึกษา|ตรวจสอบ|analyze|examine|investigate|assess|evaluate|study/i,
            priority: 'MEDIUM',
            template: 'ANALYTICAL_FRAMEWORK',
            enhancement: '🔬 DEEP ANALYSIS: Apply Scientific Analytical Framework with Data Collection, Analysis Methods, Findings, and Strategic Recommendations'
        },
        installation: {
            pattern: /ติดตั้ง|ตั้งค่า|setup|install|configure|implementation|deployment|integration/i,
            priority: 'MEDIUM',
            template: 'IMPLEMENTATION_EXCELLENCE',
            enhancement: '🏗️ IMPLEMENTATION EXCELLENCE: Provide Complete Implementation Guide with Planning, Execution, Testing, and Optimization protocols'
        }
    };
    
    // Find the best matching pattern
    let bestMatch = null;
    let maxMatches = 0;
    
    Object.entries(patternAnalysis).forEach(([key, config]) => {
        const matches = (questionLower.match(config.pattern) || []).length;
        if (matches > maxMatches) {
            maxMatches = matches;
            bestMatch = config;
        }
    });
    
    // Apply enhancement based on best match
    if (bestMatch) {
        return `${question} - [${bestMatch.enhancement}]`;
    }
    
    // Default enhancement for complex questions
    if (question.length > 50) {
        return `${question} - [🎯 COMPREHENSIVE EXPERT ANALYSIS: Deploy multi-dimensional intelligence with Technical Depth, Strategic Insights, Practical Applications, and Future-Proof Recommendations]`;
    }
    
    // Default enhancement for simple questions
    return `${question} - [🧠 EXPERT INTELLIGENCE: Provide comprehensive response with Technical Excellence, Practical Value, and Strategic Perspective]`;
};

// Advanced context analysis with machine learning-like capabilities
const analyzeContext = (question, history) => {
    const analysis = {
        complexity: 'intermediate',
        category: 'general',
        urgency: 'normal',
        domain: 'technical',
        expectedDepth: 'detailed',
        followUp: false,
        requiredExpertise: 'standard',
        responseStyle: 'professional',
        cognitiveDemand: 'medium',
        businessImpact: 'moderate',
        timeframe: 'flexible',
        stakeholders: 'individual'
    };
    
    const questionLower = question.toLowerCase();
    
    // Multi-dimensional analysis
    const analysisFramework = {
        complexity: {
            basic: /คือ|what is|define|simple|basic|easy|introduction|overview/i,
            intermediate: /วิธี|how to|compare|difference|ใช้|ทำ|implement|configure/i,
            advanced: /optimize|analyze|troubleshoot|design|architecture|integrate|scale/i,
            expert: /algorithm|machine learning|enterprise|distributed|microservices|devops|cloud native/i
        },
        urgency: {
            critical: /emergency|urgent|critical|fail|crash|down|stop|production|outage|ฉุกเฉิน/i,
            high: /important|asap|soon|quickly|รีบ|สำคัญ|deadline|immediate/i,
            medium: /when|timeline|schedule|เมื่อไหร่|กำหนด|planning/i,
            low: /future|someday|eventually|อนาคต|ในอนาคต/i
        },
        domain: {
            mechanical: /mechanical|engine|motor|pump|valve|bearing|gear|mechanical engineering/i,
            electrical: /electrical|circuit|voltage|current|power|electronics|wiring|automation/i,
            software: /software|code|programming|application|database|api|development/i,
            process: /process|workflow|procedure|sop|standard operating|methodology/i,
            management: /management|strategy|planning|organization|leadership|business/i
        },
        businessImpact: {
            critical: /production|revenue|customer|safety|compliance|regulatory|mission critical/i,
            high: /efficiency|cost|quality|performance|competitive|strategic/i,
            medium: /improvement|optimization|enhancement|upgrade|modernization/i,
            low: /convenience|nice to have|future consideration|探討/i
        }
    };
    
    // Apply multi-dimensional analysis
    Object.entries(analysisFramework).forEach(([dimension, levels]) => {
        Object.entries(levels).forEach(([level, pattern]) => {
            if (pattern.test(questionLower)) {
                analysis[dimension] = level;
            }
        });
    });
    
    // Context-aware adjustments
    if (history.length > 0) {
        const recentQuestions = history.slice(-3);
        const contextPatterns = {
            followUp: /เพิ่มเติม|more|also|และ|อีก|continue|further|detail|expand/i,
            deepDive: /explain|detail|more information|elaborate|มากขึ้น|ลึกขึ้น/i,
            troubleshooting: /still|ยัง|ไม่ได้|not working|problem|issue|error/i
        };
        
        Object.entries(contextPatterns).forEach(([pattern, regex]) => {
            if (regex.test(questionLower)) {
                analysis.followUp = true;
                analysis.expectedDepth = 'comprehensive';
            }
        });
    }
    
    // Cognitive demand assessment
    const cognitiveFactors = [
        analysis.complexity === 'expert' ? 3 : 0,
        analysis.urgency === 'critical' ? 2 : 0,
        analysis.businessImpact === 'critical' ? 2 : 0,
        analysis.followUp ? 1 : 0
    ].reduce((sum, factor) => sum + factor, 0);
    
    analysis.cognitiveDemand = cognitiveFactors > 5 ? 'high' : 
                              cognitiveFactors > 2 ? 'medium' : 'low';
    
    return analysis;
};

// Export enhanced system
module.exports = {
    fullPrompt,
    enhanceQuestion,
    analyzeContext
};
// ใช้ฟังก์ชัน enhanceQuestion
const enhancedQuestion = enhanceQuestion(question);
    const promptParts = [{ text: fullPrompt }];
    if (imageFile) {
        promptParts.push({ text: 'วิเคราะห์รูปภาพนี้ประกอบด้วย:' });
        promptParts.push({ inlineData: { data: imageFile.buffer.toString('base64'), mimeType: imageFile.mimetype } });
    }

    const result = await generativeModel.generateContent({ contents: [{ role: 'user', parts: promptParts }] });
    const response = await result.response;
    const answer = response.text();

    chatHistories[sessionId].push({ question, answer });
    res.json({ answer, sessionId });

  } catch (error) {
    console.error('Error in /chat endpoint:', error);
    res.status(500).json({ error: 'Failed to get response from AI.' });
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
            let displayName = path.parse(trimmedFileName).name;
            
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
      }
  
      res.json(manualDatabase);
    } catch (error) {
      console.error('Error creating manuals manifest:', error);
      res.status(500).json({ error: 'Could not retrieve manual list.' });
    }
});

// นำโค้ดนี้ไปวางแทนที่ของเก่า

async function startServer() {
  await initializeVectorStore(); 
  
  if (vectorStore) {
  // เริ่มการทำงานของเซิร์ฟเวอร์
  app.listen(port,  () => {
    console.log(`✅ Backend server is running on port ${port}`);
  });
} else {
  console.error('❌ Server startup failed because the vector store could not be initialized.');
  process.exit(1);
}
}

startServer();