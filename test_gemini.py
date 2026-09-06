import os
from dotenv import load_dotenv
import google.generativeai as genai

print("เริ่มโปรแกรม")

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY_INSURVERSE")

if not api_key:
    print("ไม่พบ GEMINI_API_KEY ในไฟล์ .env")
    input("กด Enter เพื่อปิด")
    exit()

print("พบ API Key แล้ว")

genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    model_name="gemini-3.1-flash-lite"
)

print("กำลังส่งคำถามไป Gemini...")

response = model.generate_content(
    "สวัสดี ช่วยแนะนำตัวสั้น ๆ"
)

print("คำตอบจาก Gemini:")
print(response.text)

input("กด Enter เพื่อปิด")