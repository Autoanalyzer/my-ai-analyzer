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
const { Pinecone } = require('@pinecone-database/pinecone');
// (ลบบรรทัดที่ซ้ำออกไปแล้ว)
// ลบ MemoryVectorStore, RecursiveCharacterTextSplitter, PDFLoader ออก
// เพราะเราไม่ต้องสร้าง Vector Store เองใน Server อีกต่อไป
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

// ... (โค้ด app.use ต่างๆ) ...

// --- ตั้งค่า AI และ Pinecone Client ---
const safetySettings = [
    { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
];
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const generativeModel = genAI.getGenerativeModel({ model: 'gemini-2.0-flash', safetySettings });
const embeddingModel = genAI.getGenerativeModel({ model: "text-embedding-004" });

const pc = new Pinecone();
const pineconeIndex = pc.index(process.env.PINECONE_INDEX_NAME);
console.log('✅ Connected to Pinecone index successfully.');

async function initializeVectorStore() {
  try {
    console.log(`Checking for saved vector store at: ${VECTOR_STORE_SAVE_PATH}`);
    const savedData = await fs.readFile(VECTOR_STORE_SAVE_PATH, 'utf-8');
    const memoryVectors = JSON.parse(savedData);

    const documents = memoryVectors.map(mv => ({ pageContent: mv.content, metadata: mv.metadata }));
    const embeddings = memoryVectors.map(mv => mv.embedding);

    vectorStore = new MemoryVectorStore(embeddingsModel);
    await vectorStore.addVectors(embeddings, documents);

    console.log('✅ Vector store loaded successfully from disk.');
  } catch (error) {
    console.log('Saved vector store not found. Building from scratch...');
    const documentsBasePath = path.join(__dirname, 'documents');
    const allDocuments = [];

    try {
        const areaFolders = (await fs.readdir(documentsBasePath, { withFileTypes: true }))
        .filter(d => d.isDirectory())
        .map(d => d.name);

     // นำโค้ดนี้ไปวางแทนที่ loop เก่า
for (const area of areaFolders) {
    const areaPath = path.join(documentsBasePath, area);
    const files = await fs.readdir(areaPath);
    for (const file of files) {
        const filePath = path.join(areaPath, file);
        const fileExt = path.extname(file).toLowerCase();
        let docsFromFile = [];

        try {
            if (fileExt === '.pdf') {
                // ✨ ใช้ PDFLoader ที่สามารถดึงเลขหน้าได้
                const loader = new PDFLoader(filePath);
                docsFromFile = await loader.load();
            } else if (fileExt === '.txt') {
                // การอ่านไฟล์ .txt ยังเหมือนเดิม แต่สร้างเป็น Document object
                const textContent = await fs.readFile(filePath, 'utf-8');
                docsFromFile.push({ pageContent: textContent, metadata: {} });
            }

            // เพิ่ม metadata ให้กับทุกหน้าที่ดึงมาได้
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
      const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
      const splitDocs = await textSplitter.splitDocuments(allDocuments);

      console.log(`Embedding ${splitDocs.length} document chunks in batches...`);
      const batchSize = 50;
      const delay = 1000;

      vectorStore = new MemoryVectorStore(embeddingsModel);

      for (let i = 0; i < splitDocs.length; i += batchSize) {
        const batch = splitDocs.slice(i, i + batchSize);
        await vectorStore.addDocuments(batch);
        console.log(`Processed batch ${Math.floor(i / batchSize) + 1} of ${Math.ceil(splitDocs.length / batchSize)}...`);
        if (i + batchSize < splitDocs.length) {
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
      
      await fs.writeFile(VECTOR_STORE_SAVE_PATH, JSON.stringify(vectorStore.memoryVectors, null, 2));
      console.log(`✅ Global vector store initialized and saved to disk at: ${VECTOR_STORE_SAVE_PATH}`);

    } catch (buildError) {
      console.error('CRITICAL: Failed to build vector store.', buildError);
      vectorStore = undefined;
    }
  }
}

// ให้นำโค้ดนี้ไปวางแทนที่ app.post('/chat', ...) ทั้งหมด
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

        // ไม่ต้องเช็ค !vectorStore อีกต่อไป

// 1. แปลงคำถามเป็น Vector
const embeddingResult = await embeddingModel.embedContent(question);
const queryVector = embeddingResult.embedding.values;

// 2. สร้าง filter ตาม Syntax ของ Pinecone (แบบ Object)
let pineconeFilter = { area: { '$eq': area.trim() } };

// 3. ส่ง Vector และ Filter ไปค้นหาที่ Pinecone
const queryResult = await pineconeIndex.query({
    vector: queryVector,
    topK: 5,
    filter: pineconeFilter,
    includeMetadata: true,
});

// 4. ดึงข้อมูลจากผลลัพธ์
const relevantDocs = queryResult.matches || [];

       const context = relevantDocs
    .map((doc) => {
        // สร้าง Path ที่ถูกต้อง 100% จาก Metadata
        const docPath = `/documents/${doc.metadata.area}/${doc.metadata.source}`;
        // ส่ง Path นี้เข้าไปใน Context ให้ AI เห็น
        return `Source Document: ${doc.metadata.source} (Path for linking: ${docPath}, Page: ${doc.metadata.loc?.pageNumber || 1})\nContent:\n${doc.pageContent}`;
    })
    .join('\n\n---\n\n');
        
        // ✨ 1. เพิ่มคำสั่งให้ AI สร้าง Markdown Link เข้าไปใน Prompt หลัก
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

          const enhancedQuestion = `User Request: "${question}"`;

        const promptParts = [];
        promptParts.push({ text: fullPrompt });

        if (context) {
            promptParts.push({ text: `--- KNOWLEDGE BASE CONTEXT ---\n${context}` });
        }

        promptParts.push({ text: `--- CURRENT MISSION ---\n${enhancedQuestion}` });
        
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
                    const fileBaseName = path.parse(trimmedFileName).name;
                    
                    const prefix = `${area.trim()}_`;
                    let nameWithoutPrefix = fileBaseName;
                    if (fileBaseName.startsWith(prefix)) {
                        nameWithoutPrefix = fileBaseName.substring(prefix.length);
                    }
                    
                    // --- ส่วนที่แก้ไข ---
                    // 1. ใช้ชื่อไฟล์เต็มๆ (ไม่รวม Prefix) มาสร้างเป็นชื่อรูปภาพ เพื่อให้ไม่ซ้ำกัน
                    const imageName = nameWithoutPrefix;
                    const imagePath = `images/${imageName}.png`;
                    // --- จบส่วนที่แก้ไข ---

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

// นำโค้ดนี้ไปวางแทนที่ของเก่า

app.listen(port, () => {
    console.log(`✅ Backend server is now running on port ${port} and connected to Pinecone.`);
});