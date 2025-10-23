import streamlit as st
import pandas as pd
import os
import time
import io
import requests
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# ---------- Load ENV ----------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OPENAI_API_KEY not found in .env file")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="🎓 Aadya College Chatbot", page_icon="🤖", layout="centered")

# ---------- Load FAQ ----------
@st.cache_data(ttl=600)
def load_faq():
    try:
        xlsx_url = "https://docs.google.com/spreadsheets/d/1DiTrHpBKZhk3HZcp0HIdFsO_gTQQ9D-V/export?format=xlsx"
        response = requests.get(xlsx_url)
        response.raise_for_status()
        df = pd.read_excel(io.BytesIO(response.content), engine="openpyxl")

        df.columns = df.columns.str.strip().str.lower()
        q_col = next((c for c in df.columns if "question" in c), None)
        a_col = next((c for c in df.columns if "response" in c or "answer" in c), None)
        cat_col = next((c for c in df.columns if "category" in c), None)

        if not q_col or not a_col:
            st.error(f"❌ Required columns not found. Columns: {list(df.columns)}")
            st.stop()

        if cat_col:
            docs = [
                f"Category: {row[cat_col]}\nQ: {row[q_col]}\nA: {row[a_col]}"
                for _, row in df.iterrows()
                if pd.notna(row[q_col]) and pd.notna(row[a_col])
            ]
        else:
            docs = [
                f"Q: {row[q_col]}\nA: {row[a_col]}"
                for _, row in df.iterrows()
                if pd.notna(row[q_col]) and pd.notna(row[a_col])
            ]
        return docs
    except Exception as e:
        st.error(f"❌ Failed to load Google Sheet: {e}")
        st.stop()

# ---------- Build Model ----------
def build_bot(docs):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_texts(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4-turbo")
    return RetrievalQA.from_chain_type(llm, retriever=retriever)

# ---------- Complaint Logging ----------
def save_complaint(name, contact, category, complaint):
    file_name = "College_Complaints_Log.csv"
    data = {
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Name": [name],
        "Contact": [contact],
        "Category": [category],
        "Complaint": [complaint],
    }
    df_new = pd.DataFrame(data)
    if os.path.exists(file_name):
        df_existing = pd.read_csv(file_name)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(file_name, index=False)
    else:
        df_new.to_csv(file_name, index=False)
    return True

# ---------- Helper ----------
def reset_all():
    for k in ["admission_mode", "schedule_mode", "fees_mode", "exam_mode", "complaint_mode"]:
        st.session_state[k] = False
    st.session_state["messages"] = []
    st.session_state["input_key"] += 1
    st.session_state["last_activity"] = time.time()

# ---------- 🕒 SESSION TIMEOUT FEATURE ----------
def check_inactivity():
    timeout_minutes = 10
    now = time.time()
    last_active = st.session_state.get("last_activity", now)
    inactive_time = now - last_active

    if inactive_time > timeout_minutes * 60:
        st.session_state["messages"] = []
        for k in ["admission_mode", "schedule_mode", "fees_mode", "exam_mode", "complaint_mode"]:
            st.session_state[k] = False
        st.warning("⏳ Session expired due to 10 minutes of inactivity. Starting a new chat.")
        st.session_state["last_activity"] = now
        st.session_state["input_key"] += 1
        st.rerun()

# ---------- Initialize Session ----------
for key in ["admission_mode", "schedule_mode", "fees_mode", "exam_mode", "complaint_mode", "messages", "input_key", "last_activity"]:
    if key not in st.session_state:
        if key == "messages":
            st.session_state[key] = []
        elif key == "input_key":
            st.session_state[key] = 0
        elif key == "last_activity":
            st.session_state[key] = time.time()
        else:
            st.session_state[key] = False

# ---------- Check Inactivity Timer ----------
check_inactivity()

# ---------- Title ----------
st.markdown("<h2 style='text-align:center; color:#1E3A8A;'>🎓 Ramarpit Group of College Chatbot (Aadya)</h2>", unsafe_allow_html=True)
st.caption("Ask me anything about admissions, courses, fees, or exams.")

# ---------- Top Menu ----------
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    if st.button("🏠 Home"):
        reset_all()
        st.session_state["messages"].append({"role": "bot", "content": "👋 Hi! I'm Aadya — Ask about admissions, fees, exams, or class schedules."})
        st.rerun()

with col2:
    if st.button("🎯 Admissions"):
        reset_all()
        st.session_state["admission_mode"] = True
        st.session_state["messages"].append({"role": "bot", "content": "Which course are you looking for admission? (e.g., BCA, MBA, BSc)"})
        st.rerun()

with col3:
    if st.button("📚 Class Schedule"):
        reset_all()
        st.session_state["schedule_mode"] = True
        st.session_state["messages"].append({"role": "bot", "content": "For which course would you like to check the class schedule? (e.g., BA, BSc, MSc)"})
        st.rerun()

with col4:
    if st.button("💰 Fees"):
        reset_all()
        st.session_state["fees_mode"] = True
        st.session_state["messages"].append({"role": "bot", "content": "For which course would you like to check the fee details? (e.g., BA, BSc, MSc)"})
        st.rerun()

with col5:
    if st.button("🧾 Exams"):
        reset_all()
        st.session_state["exam_mode"] = True
        st.session_state["messages"].append({"role": "bot", "content": "For which course exam schedule or result details would you like to see?"})
        st.rerun()

with col6:
    if st.button("📝 Lodge Complaint"):
        reset_all()
        st.session_state["complaint_mode"] = True
        st.rerun()

# ---------- Chat Display ----------
for msg in st.session_state["messages"]:
    role_color = "#DCFCE7" if msg["role"] == "user" else "#EFF6FF"
    align = "right" if msg["role"] == "user" else "left"
    st.markdown(f"<div style='background:{role_color};padding:10px;border-radius:12px;margin:5px;text-align:{align};'>{msg['content']}</div>", unsafe_allow_html=True)

# ---------- Complaint Mode ----------
if st.session_state["complaint_mode"]:
    st.markdown("### 📝 Lodge a Complaint")
    name = st.text_input("Your Name", key="complaint_name")
    contact = st.text_input("Contact Number or Email", key="complaint_contact")
    category = st.selectbox("Complaint Category", ["Admission", "Fees", "Exam", "Facilities", "Other"], key="complaint_category")
    complaint_text = st.text_area("Describe your complaint", key="complaint_text")

    if st.button("📨 Submit Complaint", key="submit_complaint"):
        if name and contact and complaint_text:
            save_complaint(name, contact, category, complaint_text)
            st.success("✅ Your complaint has been recorded successfully. Our team will reach out soon.")
            time.sleep(4)
            reset_all()
            st.rerun()
        else:
            st.warning("⚠️ Please fill all required fields before submitting.")
else:
    # ---------- Normal Chatbot Behavior ----------
    unique_key = f"chat_input_{int(time.time())}"  # ✅ unique every rerun
    user_query = st.text_input("💬 Type your question:", key=unique_key)

    if user_query:
        st.session_state["last_activity"] = time.time()
        st.session_state["messages"].append({"role": "user", "content": user_query})
        docs = load_faq()
        chain = build_bot(docs)

        if st.session_state["admission_mode"]:
            answer = chain.run(f"Admission details for {user_query}")
        elif st.session_state["schedule_mode"]:
            answer = chain.run(f"Class schedule for {user_query}")
        elif st.session_state["fees_mode"]:
            answer = chain.run(f"Fee details for {user_query}")
        elif st.session_state["exam_mode"]:
            answer = chain.run(f"Exam details for {user_query}")
        else:
            answer = chain.run(user_query)

        st.session_state["messages"].append({"role": "bot", "content": answer})
        st.session_state["input_key"] += 1
        st.rerun()
