require('dotenv').config();
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs/promises');
const pdf = require('pdf-parse');
const path = require('path'); // --- เพิ่มตรงนี้ 1: เรียกใช้โมดูล path ---

const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } = require('@google/generative-ai');
const { GoogleGenerativeAIEmbeddings } = require("@langchain/google-genai");
const { MemoryVectorStore } = require("langchain/vectorstores/memory");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");

const app = express();
const port = process.env.PORT || 4000;

app.use(cors());
app.use(express.json());

// --- เพิ่มตรงนี้ 2: บอกให้ Express รู้จักโฟลเดอร์ public ที่เก็บไฟล์หน้าเว็บ ---
app.use(express.static(path.join(__dirname, 'public')));

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
    const documentsDir = path.join(__dirname, 'documents');
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

app.post('/chat', upload.single('image'), async (req, res) => {
    try {
        const userQuestion = req.body.question || "";
        const targetManual = req.body.manual;
        const imageFile = req.file;

        if (!userQuestion) {
            return res.status(400).json({ error: 'Question is required.' });
        }
        if (!vectorStore) {
            return res.status(503).json({ error: 'AI knowledge base is not ready. Please try again later.' });
        }

        const filterFunction = (doc) => {
            if (targetManual && targetManual !== 'all') {
                return doc.metadata.source === targetManual;
            }
            return true;
        };

        const relevantDocs = await vectorStore.similaritySearch(userQuestion, 4, filterFunction);
        const context = relevantDocs.map(doc => `Source: ${doc.metadata.source}\nContent:\n${doc.pageContent}`).join("\n\n---\n\n");

        const promptParts = [];
        let prompt = `คุณคือผู้เชี่ยวชาญด้านการบำรุงรักษาอุปกรณ์ CEMS โปรดตอบคำถามต่อไปนี้โดยอ้างอิงจากข้อมูลในคู่มือที่ให้มา ตอบเป็นภาษาไทย
        --- ข้อมูลจากคู่มือที่เกี่ยวข้อง ---
        ${context || "ไม่พบข้อมูลที่เกี่ยวข้องในคู่มือ"}
        --- จบข้อมูลจากคู่มือ ---
        คำถามของผู้ใช้: "${userQuestion}"
        คำตอบของคุณ:`;
        
        promptParts.push({ text: prompt });
        
        if (imageFile) {
            promptParts.push({ text: "โปรดวิเคราะห์รูปภาพต่อไปนี้ประกอบการตอบคำถามด้วย:" });
            promptParts.push({
                inlineData: { data: imageFile.buffer.toString("base64"), mimeType: imageFile.mimetype },
            });
        }

        const result = await generativeModel.generateContent({ contents: [{ role: "user", parts: promptParts }] });
        const response = await result.response;
        const text = response.text();
        
        res.json({ answer: text });

    } catch (error) {
        console.error("--- ERROR IN /CHAT ENDPOINT ---");
        console.error("Error Time:", new Date().toISOString());
        console.error("Error Details:", error);
        console.error("--- END OF ERROR ---");
        res.status(500).json({ error: 'Failed to get response from AI. Please check the server console for details.' });
    }
});

// --- เพิ่มตรงนี้ 3: บอกให้ส่งไฟล์ index.html เมื่อมีคนเข้าหน้าแรก ---
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});


app.listen(port, () => {
    console.log(`Backend server is running at http://localhost:${port}`);
    initializeVectorStore();
});