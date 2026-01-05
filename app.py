import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# ==========================================
# 1. 系統設定與初始化
# ==========================================
st.set_page_config(page_title="專業保險諮詢 AI", layout="wide")
st.title("🛡️ 專業保險諮詢與推薦系統")

# 檢查 API Key
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    api_key = st.secrets["GROQ_API_KEY"]
else:
    st.error("❌ 未設定 GROQ_API_KEY，請至 Streamlit Secrets 進行設定。")
    st.stop()

# 載入資料庫 (使用快取避免重複載入)
@st.cache_resource
def load_resources():
    try:
        # 注意：這裡的模型必須跟您當初建立資料庫時用的模型一致
        # 根據您之前的成功經驗，通常是 'sentence-transformers/all-MiniLM-L6-v2'
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 載入 FAISS 資料庫
        # 請確認您的資料夾結構，如果是在 faiss_db_checkpoint/faiss_db_checkpoint 就改對應路徑
        db = FAISS.load_local(
            "faiss_db_checkpoint",  # 這裡假設您的 index.faiss 就在 faiss_db_checkpoint 資料夾下
            embeddings,
            allow_dangerous_deserialization=True
        )
        return db
    except Exception as e:
        return None

# 初始化資源
vectorstore = load_resources()

if not vectorstore:
    st.error("⚠️ 資料庫載入失敗！請確認 'faiss_db_checkpoint' 資料夾是否存在且路徑正確。")
    st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 設定 LLM (使用 Groq)
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key,
    model="llama3-70b-8192", # 強大的開源模型，適合中文與邏輯推理
    temperature=0.3,         # 降低創造性，確保留於事實
)

# ==========================================
# 2. 定義 Prompt Templates (核心靈魂)
# ==========================================

# 通用 Persona 設定
persona_instruction = """
你是專業且充滿熱忱的保險業務員，致力於提供最優質的服務。
你擁有市面上幾家大型保險公司的所有保險商品資料。

請務必嚴格遵守以下規則：
1. **只能**根據下方的【已知資訊】來回答問題。
2. 若資料不足或題目超過能力範圍（例如資料庫沒有該商品），請回答：「不好意思，目前的內部資料庫中沒有相關資訊，建議您直接洽詢該保險公司的專人客服服務。」
3. **拒絕回答**任何跟保險以外相關內容（例如：食譜、程式碼、旅遊景點介紹、巴斯克蛋糕怎麼做等），請禮貌拒絕並將話題引導回保險。
4. 語氣保持親切友善、專業簡潔，並使用台灣繁體中文。
5. 在提供答案的同時，請根據內容給予具體的建議。
"""

# Chatbot 專用 Prompt
qa_prompt = PromptTemplate(
    template=persona_instruction + """
    
    【已知資訊】：
    {context}
    
    使用者問題：{question}
    
    專業業務員回覆：
    """,
    input_variables=["context", "question"]
)

# 建立檢索問答鏈
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": qa_prompt}
)

# ==========================================
# 3. 介面功能實作
# ==========================================

tab1, tab2 = st.tabs(["💬 線上保險諮詢", "📋 智能保險推薦"])

# --- 功能一：Chatbot ---
with tab1:
    st.subheader("有什麼保險問題我可以幫您嗎？")
    
    # 初始化聊天紀錄
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 顯示歷史訊息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 處理使用者輸入
    if prompt := st.chat_input("請輸入您的問題 (例如：意外險適用於什麼場景？)"):
        # 1. 顯示使用者問題
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. 生成回答
        with st.chat_message("assistant"):
            with st.spinner("正在查閱保險條款..."):
                try:
                    # 進行 RAG 檢索與生成
                    response = qa_chain.invoke({"query": prompt})
                    result = response["result"]
                    
                    st.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                except Exception as e:
                    st.error(f"發生錯誤，請稍後再試：{e}")

# --- 功能二：保險推薦 ---
with tab2:
    st.subheader("為您量身打造的保險規劃")
    st.markdown("請填寫以下資訊，AI 將根據您的背景推薦適合的商品。")

    # 使用 container 來包裝表單，避免 st.form 限制互動性
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("性別", ["男", "女"])
            age = st.number_input("年齡", min_value=0, max_value=100, value=30)
            job = st.text_input("職業", "一般內勤")
        with col2:
            salary = st.selectbox("年收入範圍", ["50萬以下", "50-100萬", "100-200萬", "200萬以上"])
            budget = st.text_input("預算 (月繳/年繳)", "月繳 3000 元")
        
        # 保險類型選擇
        ins_type = st.selectbox(
            "您感興趣的保險類型", 
            ["醫療險", "意外險", "儲蓄險/投資型", "旅遊平安險", "長照險", "壽險"]
        )
        
        # 動態顯示：如果是旅遊險，多顯示兩個欄位
        travel_details = ""
        if ins_type == "旅遊平安險":
            st.info("✈️ 偵測到旅遊需求，請補充細節：")
            c1, c2 = st.columns(2)
            with c1:
                dest = st.text_input("旅遊國家", "日本")
            with c2:
                days = st.number_input("旅遊天數", min_value=1, value=5)
            travel_details = f"，旅遊目的地為{dest}，預計旅遊{days}天"

        if st.button("🚀 開始分析並推薦", type="primary"):
            with st.spinner("正在分析您的需求並比對資料庫..."):
                # 組合使用者畫像 Prompt
                user_profile_query = f"""
                使用者基本資料：
                - 性別：{gender}
                - 年齡：{age}
                - 職業：{job}
                - 年收入：{salary}
                - 預算：{budget}
                - 主要需求：{ins_type}{travel_details}
                
                任務：
                請根據上述使用者條件，從資料庫中搜尋最適合的【{ins_type}】商品。
                請列出推薦的商品名稱，並詳細說明推薦原因（例如該商品有什麼特色適合這位使用者）。
                若資料庫中沒有完全匹配的商品，請推薦最接近的通用方案。
                """
                
                try:
                    # 這裡直接復用 qa_chain，因為它已經包含了 "只根據資料庫回答" 的限制
                    # 這樣可以確保推薦的商品一定是資料庫裡有的
                    response = qa_chain.invoke({"query": user_profile_query})
                    
                    st.success("分析完成！以下是給您的專業建議：")
                    st.markdown("### 📋 推薦報告")
                    st.markdown(response["result"])
                except Exception as e:
                    st.error(f"分析失敗：{e}")