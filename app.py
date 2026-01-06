import streamlit as st
import os
import gdown
import zipfile
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq 

# ==========================================
# 1. 系統設定
# ==========================================
st.set_page_config(page_title="專業保險諮詢 AI", layout="wide")
st.title("🛡️ 專業保險諮詢與推薦系統 (V3.0 智能版)")

# 檢查 Groq 金鑰
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    api_key = st.secrets["GROQ_API_KEY"]
else:
    st.error("❌ 未設定 GROQ_API_KEY，請至 Streamlit Secrets 設定。")
    st.stop()

# ==========================================
# 2. 設定 Google Drive 下載
# ==========================================
GDRIVE_FILE_ID = "1SWLCi36AvdoOO8oTAflVD9luHyDKQbRL" 
ZIP_NAME = "faiss_db_mini.zip"
DB_FOLDER = "faiss_db_mini"

# ==========================================
# 3. Embedding 模型
# ==========================================
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

# ==========================================
# 4. 載入資源 (⚠️ 修正點：純淨版，不含任何 UI 指令)
# ==========================================
@st.cache_resource(show_spinner=False) # 關閉內建 spinner，完全由我們控制
def load_resources():
    """
    這個函式只負責運算與資料讀取，
    絕對不包含 st.spinner, st.error 等 UI 互動。
    """
    # 1. 下載與解壓縮 (只做動作，不顯示 st 訊息)
    if not os.path.exists(DB_FOLDER):
        if not os.path.exists(ZIP_NAME):
            try:
                url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
                gdown.download(url, ZIP_NAME, quiet=False)
            except:
                return None # 失敗就回傳 None，讓外面處理
        
        try:
            with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
                zip_ref.extractall(".")
        except:
            return None

    # 2. 載入 FAISS
    try:
        embeddings = get_embeddings()
        if os.path.exists(DB_FOLDER):
            load_path = DB_FOLDER
        else:
            load_path = "."
            
        db = FAISS.load_local(
            load_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        return db
    except:
        return None

# --- 在「函式外面」做轉圈圈特效 ---
with st.spinner("📦 系統啟動中，正在載入保險資料庫..."):
    vectorstore = load_resources()

# --- 根據結果顯示 UI ---
if not vectorstore:
    st.error("❌ 資料庫載入失敗！請檢查 Requirements 或 Google Drive 連結。")
    st.stop()
else:
    # 成功載入後，偷偷給個小提示 (這是安全的，因為不在 cache 函式裡)
    st.toast("✅ 資料庫載入成功！", icon="🧠")

# 設定檢索器 (k=8 擴大搜尋範圍)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# ==========================================
# 5. 設定 LLM
# ==========================================
llm = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile", 
    temperature=0.3,
)

# ==========================================
# 6. Prompt 與 Chain
# ==========================================
persona_instruction = """
你是專業、靈活且富有洞察力的資深保險顧問。
你的任務是根據【已知資訊】(Context) 來回答使用者的問題或進行商品推薦。

🔥 **重要思考邏輯 (Chain of Thought)**：
1. **關鍵字轉換**：若使用者提到特定國家(如日本、美國)，請自動對應到條款中的「海外」、「國外」或「全球」相關規定。不要因為沒看到國家名字就說不知道。
2. **資訊整合**：若使用者詢問推薦，請綜合分析【已知資訊】中的多個商品，比較其優缺點。
3. **誠實但積極**：如果資料庫真的完全沒有相關險種，才回答無法提供；否則請盡量從現有資料中挖掘最接近的答案。

【已知資訊】：
{context}

使用者問題：{question}

請以台灣繁體中文，專業且條理分明地回答：
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("human", persona_instruction)
])

def format_docs(docs):
    return "\n\n".join(f"文件來源: {doc.metadata.get('source', '未知')}\n內容: {doc.page_content}" for doc in docs)

qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)

# ==========================================
# 7. 介面功能 (含 Debug 視窗)
# ==========================================
tab1, tab2 = st.tabs(["💬 線上保險諮詢", "📋 智能保險推薦"])

with tab1:
    st.subheader("💡 智慧保險顧問")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("請輸入您的問題 (例如：日本旅遊險推薦)..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 AI 正在翻閱條款並進行推理..."):
                try:
                    # Debug: 顯示抓到的資料
                    retrieved_docs = retriever.invoke(prompt)
                    
                    with st.expander("🕵️ [工程師模式] 點擊查看 AI 讀到了哪些資料"):
                        if not retrieved_docs:
                            st.warning("⚠️ 系統沒有抓到任何資料。")
                        for i, doc in enumerate(retrieved_docs):
                            source = doc.metadata.get('source', doc.metadata.get('filename', '未知來源'))
                            st.markdown(f"**📄 參考文件 {i+1} ({source})**")
                            st.caption(doc.page_content[:300] + "...") 
                            st.divider()

                    # 產生回答
                    response = qa_chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"發生錯誤：{e}")

with tab2:
    st.subheader("📋 為您量身打造的保險規劃")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("性別", ["男", "女"])
            age = st.number_input("年齡", 25, 100, 30)
            job = st.text_input("職業", "工程師")
        with col2:
            salary = st.selectbox("年收", ["50萬以下", "50-100萬", "100-200萬", "200萬以上"])
            budget = st.text_input("預算", "月繳 3000")
        
        ins_type = st.selectbox("險種", ["醫療險", "意外險", "儲蓄險", "旅遊險", "長照險", "壽險"])
        
        extra_info = ""
        if ins_type == "旅遊險":
            dest = st.text_input("國家 (例如：日本)", "日本")
            days = st.number_input("天數", 1, 365, 5)
            extra_info = f"預計前往{dest}旅遊{days}天"

        if st.button("開始 AI 分析"):
            with st.spinner("🤖 AI 正在綜合評估..."):
                query = f"""
                使用者背景：{gender}, {age}歲, 職業{job}, 年收{salary}, 預算{budget}。
                需求：想找{ins_type}。{extra_info}。
                
                任務：
                1. 請搜尋資料庫中適合的{ins_type}商品。
                2. 若目的地是國外(如日本)，請優先尋找海外相關保障。
                3. 請推薦 1-2 個具體商品，並說明推薦原因。
                """
                
                retrieved_docs = retriever.invoke(query)
                with st.expander("🕵️ [工程師模式] AI 檢索到的條款內容"):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**📄 來源 {i+1}**")
                        st.caption(doc.page_content[:300] + "...")
                        st.divider()

                response = qa_chain.invoke(query)
                st.markdown(response)