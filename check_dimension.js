import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const data = require('./vector_store.json');

if (data && data.length > 0) {
  // **สำคัญ:** ถ้าในไฟล์ json ของคุณไม่ได้ใช้ชื่อ key ว่า 'embedding'
  // ให้แก้คำว่า 'embedding' ในบรรทัดถัดไปเป็นชื่อที่คุณใช้ เช่น 'values' หรือ 'vector'
  const vector = data[0].embedding; 

  if (vector && Array.isArray(vector)) {
    console.log(`เจอแล้ว! จำนวน Dimension คือ: ${vector.length}`);
    console.log(`นำตัวเลข ${vector.length} ไปใส่ในช่อง Dimension ได้เลย`);
  } else {
    console.log("ไม่พบ Vector ในข้อมูลชุดแรก, กรุณาตรวจสอบชื่อ key ในโค้ด");
  }
} else {
  console.log("ไฟล์ว่างเปล่าหรือไม่สามารถอ่านได้");
}