require('dotenv').config();
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs/promises');
const pdf = require('pdf-parse');
const path = require('path');

const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } = require('@google/generative-ai');
const { GoogleGenerativeAIEmbeddings } = require("@langchain/google-genai");
const { MemoryVectorStore } = require("langchain/vectorstores/memory");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");

const app = express();
const port = process.env.PORT || 4000;

// --- 1. เพิ่มตัวแปรสำหรับเก็บประวัติแชท ---
const chatHistories = {}; 

app.use(cors());
app.use(express.json());
app.use(express.static(__dirname));

const upload = multer({ storage: multer.memoryStorage() });

const safetySettings = [
    { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
];

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const generativeModel = genAI.getGenerativeModel({ model: "gemini-1.5-flash", safetySettings });
const embeddingsModel = new GoogleGenerativeAIEmbeddings({ apiKey: process.env.GEMINI_API_KEY, model: "text-embedding-004" });

let vectorStore;

async function initializeVectorStore() {
    console.log("Initializing Vector Store...");
    const documentsDir = path.join(__dirname, 'documents', 'ql');
    const documents = [];

    try {
        const files = await fs.readdir(documentsDir);
        for (const file of files) {
            if (path.extname(file).toLowerCase() === '.pdf') {
                const filePath = path.join(documentsDir, file);
                const dataBuffer = await fs.readFile(filePath);
                try {
                    const pdfData = await pdf(dataBuffer);
                    documents.push({
                        pageContent: pdfData.text,
                        metadata: { source: file },
                    });
                    console.log(`Processed: ${file}`);
                } catch (pdfError) {
                    console.error(`Could not process PDF file: ${file}`, pdfError);
                }
            }
        }

        if (documents.length === 0) {
            console.log("No PDF documents found to process.");
            return;
        }

        const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
        const splitDocs = await textSplitter.splitDocuments(documents);

        vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddingsModel);
        console.log("Vector Store Initialized Successfully.");

    } catch (error) {
        console.error("Error initializing vector store:", error);
    }
}

// --- 2. แทนที่ app.post('/chat', ...) เดิมทั้งหมดด้วยโค้ดชุดใหม่นี้ ---
app.post('/chat', upload.single('image'), async (req, res) => {
    try {
        let { sessionId, question, manual } = req.body;
        const imageFile = req.file;

        // ตรวจสอบ Session ID ถ้าไม่มี ให้สร้างใหม่
        if (!sessionId) {
            sessionId = `session_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
            chatHistories[sessionId] = []; // สร้างประวัติแชทใหม่สำหรับ session นี้
            console.log(`New session created: ${sessionId}`);
        }

        const history = chatHistories[sessionId] || [];

        if (!question) return res.status(400).json({ error: 'Question is required.' });
        if (!vectorStore) return res.status(503).json({ error: 'AI knowledge base is not ready.' });

        const filterFunction = (doc) => (manual && manual !== 'all') ? doc.metadata.source === manual : true;
        const relevantDocs = await vectorStore.similaritySearch(question, 4, filterFunction);
        const context = relevantDocs.map(doc => `Source: ${doc.metadata.source}\nContent:\n${doc.pageContent}`).join("\n\n---\n\n");

        // สร้าง Prompt โดยใส่ "ประวัติการแชท" เข้าไปด้วย
        const fullPrompt = `คุณคือผู้เชี่ยวชาญด้านการบำรุงรักษาอุปกรณ์ CEMS โปรดตอบคำถามโดยอ้างอิงจากข้อมูลในคู่มือและบทสนทนาก่อนหน้า
        --- ข้อมูลจากคู่มือ ---
        ${context || "ไม่พบข้อมูลที่เกี่ยวข้องในคู่มือ"}
        --- จบข้อมูลจากคู่มือ ---
        
        --- ประวัติการสนทนาที่ผ่านมา ---
        ${history.map(h => `User: ${h.question}\nAI: ${h.answer}`).join('\n\n')}
        --- จบประวัติการสนทนา ---
        
        คำถามล่าสุดของผู้ใช้: "${question}"
        คำตอบของคุณ:`;

        const promptParts = [{ text: fullPrompt }];
        if (imageFile) {
            promptParts.push({ text: "วิเคราะห์รูปภาพนี้ประกอบด้วย:" });
            promptParts.push({ inlineData: { data: imageFile.buffer.toString("base64"), mimeType: imageFile.mimetype } });
        }

        const result = await generativeModel.generateContent({ contents: [{ role: "user", parts: promptParts }] });
        const response = await result.response;
        const answer = response.text();

        // บันทึกคำถามและคำตอบล่าสุดลงในประวัติ
        chatHistories[sessionId].push({ question, answer });

        // ส่งคำตอบและ Session ID กลับไป
        res.json({ answer, sessionId });

    } catch (error) {
        console.error("Error in /chat endpoint:", error);
        res.status(500).json({ error: 'Failed to get response from AI.' });
    }
});


app.get('/', (req, res) => {
     res.sendFile(path.join(__dirname, 'index.html'));
});


app.listen(port, () => {
    console.log(`Backend server is running at http://localhost:${port}`);
    initializeVectorStore();
});