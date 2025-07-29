// upload_to_pinecone.js
import { Pinecone } from '@pinecone-database/pinecone';
import { createRequire } from 'module';
import 'dotenv/config';

const require = createRequire(import.meta.url);
// ตรวจสอบให้แน่ใจว่าไฟล์ vector_store.json อยู่ในตำแหน่งที่ถูกต้อง
const vectorStore = require('./vector_store.json'); 

const pc = new Pinecone();
const index = pc.index(process.env.PINECONE_INDEX_NAME);

async function uploadVectors() {
    console.log("กำลังอัปโหลดข้อมูลไปยัง Pinecone...");

    // **สำคัญ: ปรับแก้ส่วนนี้ให้ตรงกับโครงสร้างไฟล์ JSON ของคุณ**
    const vectorsToUpload = vectorStore.map((item, i) => ({
        id: item.id || `vec-${i}`,      // ต้องมี ID ที่ไม่ซ้ำกัน
        values: item.embedding,         // ชื่อ property ที่เก็บ vector
        metadata: {
            text: item.text,            // ข้อมูลอื่นๆ ที่อยากเก็บ
            source: item.source,
        },
    }));

    // อัปโหลดข้อมูลเป็นชุดๆ (Batch) เพื่อประสิทธิภาพ
    for (let i = 0; i < vectorsToUpload.length; i += 100) {
        const batch = vectorsToUpload.slice(i, i + 100);
        await index.upsert(batch);
        console.log(`อัปโหลดแล้ว ${i + batch.length} / ${vectorsToUpload.length} รายการ`);
    }
    console.log("อัปโหลดข้อมูลทั้งหมดเรียบร้อย!");
}

uploadVectors().catch(console.error);