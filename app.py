import base64
import os
from pathlib import Path

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from prompt import PROMPT_WORKAW
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="PELEK Chatbot | ครุศาสตร์ไฟฟ้า",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).resolve().parent
BOT_AVATAR_PATH = APP_DIR / "pelek_chatbot_profile.png"


def image_to_data_uri(image_path):
    """แปลงรูปโปรไฟล์ในเครื่องให้แสดงในส่วนหัวของหน้าแอปได้"""
    image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{image_data}"


BOT_AVATAR_DATA_URI = image_to_data_uri(BOT_AVATAR_PATH)

st.markdown(
    """
    <style>
        :root {
            --pelek-orange: #f97316;
            --pelek-orange-dark: #d9530b;
            --pelek-gold: #ffb020;
            --pelek-cream: #fff8ef;
            --pelek-ink: #172033;
            --pelek-muted: #667085;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 4%, rgba(255, 176, 32, 0.22), transparent 28rem),
                radial-gradient(circle at 92% 18%, rgba(249, 115, 22, 0.12), transparent 24rem),
                linear-gradient(180deg, #fffdf9 0%, #fff7ec 100%);
            color: var(--pelek-ink);
            font-family: "Noto Sans Thai", "Leelawadee UI", Tahoma, sans-serif;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 920px;
            padding-top: 2rem;
            padding-bottom: 8rem;
        }

        .pelek-hero {
            position: relative;
            display: flex;
            align-items: center;
            gap: 1.5rem;
            overflow: hidden;
            margin-bottom: 1.35rem;
            padding: 1.4rem 1.6rem;
            border: 1px solid rgba(249, 115, 22, 0.2);
            border-radius: 28px;
            background: linear-gradient(125deg, rgba(255, 255, 255, 0.98), rgba(255, 244, 225, 0.96));
            box-shadow: 0 18px 48px rgba(151, 72, 12, 0.12);
        }

        .pelek-hero::after {
            content: "⚡";
            position: absolute;
            right: -1rem;
            bottom: -3.6rem;
            color: rgba(249, 115, 22, 0.08);
            font-size: 10rem;
            transform: rotate(8deg);
        }

        .pelek-avatar-wrap {
            position: relative;
            z-index: 1;
            flex: 0 0 132px;
            width: 132px;
            height: 132px;
            padding: 5px;
            border-radius: 50%;
            background: linear-gradient(145deg, var(--pelek-gold), var(--pelek-orange));
            box-shadow: 0 12px 28px rgba(249, 115, 22, 0.25);
        }

        .pelek-avatar-wrap img {
            display: block;
            width: 100%;
            height: 100%;
            border: 4px solid white;
            border-radius: 50%;
            background: white;
            object-fit: cover;
        }

        .pelek-hero-content {
            position: relative;
            z-index: 1;
        }

        .pelek-kicker {
            margin-bottom: 0.25rem;
            color: var(--pelek-orange-dark);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .pelek-hero h1 {
            margin: 0;
            color: var(--pelek-ink);
            font-size: clamp(1.75rem, 4vw, 2.65rem);
            font-weight: 850;
            line-height: 1.15;
        }

        .pelek-hero h1 span {
            color: var(--pelek-orange);
        }

        .pelek-hero p {
            max-width: 560px;
            margin: 0.5rem 0 0.8rem;
            color: var(--pelek-muted);
            font-size: 0.98rem;
            line-height: 1.65;
        }

        .pelek-status {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.82);
            color: #475467;
            font-size: 0.78rem;
            font-weight: 700;
            box-shadow: inset 0 0 0 1px rgba(249, 115, 22, 0.14);
        }

        .pelek-status-dot {
            width: 0.55rem;
            height: 0.55rem;
            border-radius: 50%;
            background: #22c55e;
            box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.12);
        }

        .pelek-section-label {
            margin: 0 0 0.65rem 0.35rem;
            color: #9a4b13;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.02em;
        }

        [data-testid="stChatMessage"] {
            margin-bottom: 0.8rem;
            padding: 1rem 1.1rem;
            border: 1px solid rgba(249, 115, 22, 0.13);
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.9);
            box-shadow: 0 8px 24px rgba(86, 49, 20, 0.06);
        }

        [data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] {
            overflow: hidden;
            border: 2px solid rgba(249, 115, 22, 0.34);
            border-radius: 50%;
            background: #fff7ed;
            box-shadow: 0 4px 12px rgba(249, 115, 22, 0.12);
        }

        [data-testid="stChatMessageContent"] p {
            color: #344054;
            line-height: 1.75;
        }

        [data-testid="stChatInput"] {
            border: 1px solid rgba(249, 115, 22, 0.32);
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.98);
            box-shadow: 0 12px 35px rgba(139, 70, 18, 0.14);
        }

        [data-testid="stChatInput"]:focus-within {
            border-color: var(--pelek-orange);
            box-shadow: 0 0 0 4px rgba(249, 115, 22, 0.1), 0 12px 35px rgba(139, 70, 18, 0.14);
        }

        [data-testid="stChatInput"] button {
            color: var(--pelek-orange) !important;
        }

        [data-testid="stBottomBlockContainer"] {
            background: linear-gradient(180deg, rgba(255, 248, 239, 0), #fff8ef 35%);
        }

        [data-testid="stExpander"] {
            overflow: hidden;
            border: 1px solid rgba(249, 115, 22, 0.16);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.82);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(165deg, #c94b08 0%, #f97316 58%, #ffad1f 100%);
        }

        [data-testid="stSidebar"] * {
            color: white;
        }

        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            border: 1px solid rgba(255, 255, 255, 0.55);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.16);
            color: white;
            font-weight: 750;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: white;
            background: white;
            color: var(--pelek-orange-dark);
        }

        @media (max-width: 640px) {
            .block-container {
                padding-top: 1rem;
            }

            .pelek-hero {
                align-items: flex-start;
                gap: 1rem;
                padding: 1.1rem;
                border-radius: 22px;
            }

            .pelek-avatar-wrap {
                flex-basis: 88px;
                width: 88px;
                height: 88px;
            }

            .pelek-hero p {
                font-size: 0.88rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)



load_dotenv()
api_key = os.getenv("GEMINI_API_KEY_INSURVERSE")

if not api_key:
    st.error("ไม่พบ GEMINI_API_KEY_INSURVERSE ในไฟล์ .env")
    st.stop()

genai.configure(api_key=api_key)
generation_config = {
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 64,
    # "max_output_tokens": 8192,
    "max_output_tokens": 1024,  # Reduce this value to lower the token usage
    "response_mime_type": "text/plain",
}

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
    }

model = genai.GenerativeModel(
    model_name="gemini-3.5-flash",
    safety_settings=SAFETY_SETTINGS,
    generation_config=generation_config,
    system_instruction=PROMPT_WORKAW
    ,)


def clear_history():
    st.session_state["messages"] = [
        {
            "role": "model",
            "content": "สวัสดีค่ะ PELEK เอง 👋 สอบถามข้อมูลเกี่ยวกับคณะครุศาสตร์อุตสาหกรรมและภาควิชาเทคโนโลยีวิศวกรรมไฟฟ้าได้เลยค่ะ",
        }
    ]
    st.rerun()


with st.sidebar:
    st.markdown("### ⚡ PELEK Chatbot")
    st.caption("ผู้ช่วยตอบคำถามภาควิชาเทคโนโลยีวิศวกรรมไฟฟ้า")
    if st.button("🗑️ ล้างประวัติการสนทนา", use_container_width=True):
        clear_history()
    st.caption("คำแนะนำ: ระบุหัวข้อที่ต้องการสอบถามให้ชัดเจน เพื่อให้ได้คำตอบที่ตรงที่สุด")

st.markdown(
    f"""
    <section class="pelek-hero">
        <div class="pelek-avatar-wrap">
            <img src="{BOT_AVATAR_DATA_URI}" alt="รูปโปรไฟล์ PELEK Chatbot">
        </div>
        <div class="pelek-hero-content">
            <div class="pelek-kicker">KMUTNB · Electrical Education</div>
            <h1>สวัสดีค่ะ <span>PELEK</span> เอง</h1>
            <p>ผู้ช่วยตอบคำถามเกี่ยวกับหลักสูตร การเรียน และบุคลากรภาควิชาเทคโนโลยีวิศวกรรมไฟฟ้า</p>
            <div class="pelek-status">
                <span class="pelek-status-dot"></span>
                พร้อมตอบคำถาม
            </div>
        </div>
    </section>
    <div class="pelek-section-label">เริ่มต้นการสนทนา</div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "model",
            "content": "สวัสดีค่ะ PELEK เอง 👋 สอบถามข้อมูลเกี่ยวกับคณะครุศาสตร์อุตสาหกรรมและภาควิชาเทคโนโลยีวิศวกรรมไฟฟ้าได้เลยค่ะ",
        }
    ]

file_path = APP_DIR / "FAQ_Chatbot_100.md"
EMBEDDING_MODEL = "models/gemini-embedding-001"

# กลุ่มคำที่ผู้ใช้อาจพิมพ์ต่างกัน แต่สื่อถึงเรื่องเดียวกันในชุดข้อมูล FAQ
# ควรใส่เฉพาะคำที่มีความหมายใกล้กันจริง เพื่อไม่ให้ RAG ดึงข้อมูลผิดหัวข้อ
SYNONYM_GROUPS = {
    "ผู้สอน": [
        "ครู",
        "อาจารย์",
        "ผู้สอน",
        "คณาจารย์",
        "ครูช่าง",
    ],
    "ค่าเทอม": [
        "ค่าเทอม",
        "ค่าเล่าเรียน",
        "ค่าธรรมเนียมการศึกษา",
        "ค่าใช้จ่ายการศึกษา",
    ],
    "สมัครเรียน": [
        "สมัครเรียน",
        "สมัครเข้าเรียน",
        "การรับสมัคร",
        "รับสมัคร",
        "ยื่นสมัคร",
    ],
    "เกรดเฉลี่ย": [
        "เกรดเฉลี่ย",
        "เกรดเฉลี่ยสะสม",
        "ผลการเรียนเฉลี่ย",
        "GPA",
        "GPAX",
    ],
    "หน่วยกิต": [
        "หน่วยกิต",
        "เครดิตวิชา",
        "จำนวนหน่วยกิต",
    ],
    "ภาคการศึกษา": [
        "ภาคการศึกษา",
        "ภาคเรียน",
        "เทอม",
        "semester",
    ],
    "แขนงวิชา": [
        "แขนงวิชา",
        "สาขาย่อย",
        "วิชาเอก",
        "แทร็ก",
        "track",
    ],
    "ฝึกงาน": [
        "ฝึกงาน",
        "ฝึกประสบการณ์ในโรงงาน",
        "ฝึกประสบการณ์ภาคอุตสาหกรรม",
        "ฝึกประสบการณ์วิชาชีพในสถานประกอบการ",
    ],
    "ฝึกสอน": [
        "ฝึกสอน",
        "ออกฝึกสอน",
        "ปฏิบัติการสอน",
        "ฝึกประสบการณ์วิชาชีพครู",
        "สอนจริง",
    ],
    "ทุนการศึกษา": [
        "ทุน",
        "ทุนการศึกษา",
        "กยศ.",
        "กยศ",
        "เงินกู้เพื่อการศึกษา",
    ],
    "สำเร็จการศึกษา": [
        "เรียนจบ",
        "จบการศึกษา",
        "สำเร็จการศึกษา",
        "จบหลักสูตร",
    ],
    "วุฒิการศึกษา": [
        "วุฒิการศึกษา",
        "ใบวุฒิการศึกษา",
        "ปริญญา",
        "ชื่อปริญญา",
    ],
    "มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ": [
        "มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ",
        "พระจอมเกล้าพระนครเหนือ",
        "มจพ.",
        "มจพ",
        "KMUTNB",
    ],
    "ห้องปฏิบัติการ": [
        "ห้องปฏิบัติการ",
        "ห้องแล็บ",
        "ห้องแลป",
        "ห้องทดลอง",
        "lab",
    ],
    "ใบประกอบวิชาชีพครู": [
        "ใบประกอบวิชาชีพครู",
        "ใบประกอบครู",
        "ใบวิชาชีพครู",
        "ตั๋วครู",
    ],
    "โครงงาน": [
        "โครงงาน",
        "โปรเจกต์",
        "โปรเจ็ค",
        "โปรเจ็คท์",
        "project",
    ],
    "เทียบโอน": [
        "เทียบโอน",
        "โอนหน่วยกิต",
        "เทียบหน่วยกิต",
        "transfer credit",
    ],
    "สอบสัมภาษณ์": [
        "สอบสัมภาษณ์",
        "สัมภาษณ์",
        "interview",
    ],
    "สอบตก": [
        "สอบตก",
        "ติด F",
        "ได้ F",
        "ไม่ผ่านวิชา",
    ],
    "ติดต่อภาควิชา": [
        "ติดต่อภาควิชา",
        "เบอร์โทรภาควิชา",
        "โทรศัพท์ภาควิชา",
        "ช่องทางติดต่อภาควิชา",
    ],
    "ปี 1": [
        "ปี 1",
        "ปีหนึ่ง",
        "ชั้นปีที่ 1",
        "นักศึกษาปี 1",
    ],
    "ปี 2": [
        "ปี 2",
        "ปีสอง",
        "ชั้นปีที่ 2",
        "นักศึกษาปี 2",
    ],
    "ปี 3": [
        "ปี 3",
        "ปีสาม",
        "ชั้นปีที่ 3",
        "นักศึกษาปี 3",
    ],
    "ปี 4": [
        "ปี 4",
        "ปีสี่",
        "ชั้นปีที่ 4",
        "นักศึกษาปี 4",
    ],
}


def expand_query(question):
    """เติมคำใกล้เคียงให้คำถาม เพื่อช่วยให้ RAG ค้นหา FAQ ได้ครอบคลุมขึ้น"""
    related_words = []
    question_lower = question.lower()

    for main_word, aliases in SYNONYM_GROUPS.items():
        found = any(
            alias.lower() in question_lower
            for alias in aliases
        )

        if found:
            related_words.append(main_word)
            related_words.extend(aliases)

    # ลบคำซ้ำโดยยังรักษาลำดับเดิมไว้
    related_words = list(dict.fromkeys(related_words))

    if related_words:
        return question + " " + " ".join(related_words)

    return question


@st.cache_resource
def build_retriever(file_path, file_version):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # แยก Markdown ออกเป็น FAQ ละ 1 ชุด
    chunks = re.split(r"(?=## FAQ \d+)", content)
    chunks = [
        chunk.strip()
        for chunk in chunks
        if chunk.strip().startswith("## FAQ")
    ]

    if not chunks:
        st.error("ไม่พบข้อมูล FAQ ในไฟล์ Markdown")
        st.stop()

    # สร้าง TF-IDF สำรองไว้เสมอ เผื่อบริการ Embedding ใช้งานไม่ได้
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 6),
        sublinear_tf=True,
    )
    tfidf_vectors = vectorizer.fit_transform(chunks)

    semantic_vectors = []
    embedding_error = None

    try:
        # แบ่งเป็นชุดย่อย เพื่อลดโอกาสที่คำขอ Embedding จะใหญ่เกินไป
        batch_size = 20
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch,
                task_type="retrieval_document",
            )
            batch_vectors = result["embedding"]

            # รองรับกรณี SDK คืนเวกเตอร์เดี่ยวแทนรายการเวกเตอร์
            if batch_vectors and isinstance(batch_vectors[0], (int, float)):
                batch_vectors = [batch_vectors]

            semantic_vectors.extend(batch_vectors)

        if len(semantic_vectors) != len(chunks):
            raise ValueError("จำนวนเวกเตอร์ไม่ตรงกับจำนวน FAQ")

    except Exception as error:
        semantic_vectors = None
        embedding_error = type(error).__name__

    return (
        chunks,
        semantic_vectors,
        vectorizer,
        tfidf_vectors,
        embedding_error,
    )


file_version = os.path.getmtime(file_path)
(
    chunks,
    semantic_vectors,
    vectorizer,
    tfidf_vectors,
    embedding_error,
) = build_retriever(file_path, file_version)

if embedding_error:
    st.warning(
        "ไม่สามารถสร้าง Semantic Embedding ได้ "
        "ระบบจึงเปลี่ยนไปใช้การค้นหาแบบ TF-IDF ชั่วคราว "
        f"({embedding_error})"
    )


def search_faq(question, top_k=5):
    expanded_question = expand_query(question)
    retrieval_method = "Semantic Embedding + synonym dictionary"

    if semantic_vectors is not None:
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=expanded_question,
                task_type="retrieval_query",
            )
            question_vector = result["embedding"]
            scores = cosine_similarity(
                [question_vector], semantic_vectors
            ).flatten()
        except Exception:
            retrieval_method = "TF-IDF fallback + synonym dictionary"
            question_vector = vectorizer.transform([expanded_question])
            scores = cosine_similarity(
                question_vector, tfidf_vectors
            ).flatten()
    else:
        retrieval_method = "TF-IDF fallback + synonym dictionary"
        question_vector = vectorizer.transform([expanded_question])
        scores = cosine_similarity(
            question_vector, tfidf_vectors
        ).flatten()

    result_count = min(top_k, len(chunks))
    top_indexes = scores.argsort()[::-1][:result_count]
    context = "\n\n---\n\n".join(chunks[index] for index in top_indexes)
    matches = [
        (int(index), float(scores[index]))
        for index in top_indexes
    ]

    return context, matches, retrieval_method



for msg in st.session_state["messages"]:
    display_role = "assistant" if msg["role"] == "model" else "user"
    avatar = str(BOT_AVATAR_PATH) if msg["role"] == "model" else "👤"
    st.chat_message(display_role, avatar=avatar).write(msg["content"])

if prompt := st.chat_input("พิมพ์คำถามเกี่ยวกับภาควิชาที่นี่..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").write(prompt)


    def generate_response():
        history = [
            {"role": msg["role"], "parts": [{"text": msg["content"]}]}
            for msg in st.session_state["messages"]
        ]

        if prompt.lower().startswith("add") or prompt.lower().endswith("add"):
            answer = "ขอบคุณสำหรับคำแนะนำค่ะ"
            st.chat_message("assistant", avatar=str(BOT_AVATAR_PATH)).write(answer)
            st.session_state["messages"].append(
                {"role": "model", "content": answer}
            )

        else:
            try:
                context, matches, retrieval_method = search_faq(prompt)
            except Exception as error:
                st.error(
                    "ระบบค้นหาข้อมูลขัดข้อง กรุณาลองใหม่อีกครั้ง "
                    f"({type(error).__name__})"
                )
                return

            with st.expander("ดูข้อมูลที่ RAG ค้นเจอ"):
                st.caption(f"วิธีค้นหา: {retrieval_method}")
                for rank, (index, score) in enumerate(matches, start=1):
                    st.markdown(f"**อันดับ {rank} · คะแนน {score:.3f}**")
                    st.markdown(chunks[index])
                    st.divider()

            rag_prompt = f"""
ตอบคำถามโดยอ้างอิงจากข้อมูล FAQ ด้านล่างเท่านั้น
ให้พิจารณาจากความหมายของคำถาม ไม่จำเป็นต้องใช้คำตรงกับ FAQ ทุกคำ
หากข้อมูลที่ค้นพบไม่เกี่ยวข้องกับคำถามจริง ๆ ให้ตอบว่า
"ขออภัย ไม่พบข้อมูลนี้ในชุดข้อมูลหลักสูตร"

ข้อมูล FAQ ที่ค้นพบ:
{context}

คำถามของผู้ใช้:
{prompt}
"""

            # เอาคำถามล่าสุดออก เพราะจะส่งผ่าน rag_prompt
            history = history[:-1]

            try:
                chat_session = model.start_chat(history=history)
                response = chat_session.send_message(rag_prompt)
            except Exception as error:
                st.error(
                    "ไม่สามารถติดต่อโมเดลเพื่อสร้างคำตอบได้ "
                    f"({type(error).__name__})"
                )
                return

            st.session_state["messages"].append(
                {"role": "model", "content": response.text}
            )
            st.chat_message(
                "assistant", avatar=str(BOT_AVATAR_PATH)
            ).write(response.text)

    generate_response()

    
