import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# 1. 設定頁面配置
st.set_page_config(page_title="專業保險諮詢 AI", layout="wide")

# 2. 載入必要的 API Key (部署時會設定在 Secrets)
# 如果是本地測試，可以直接寫 os.environ["GOOGLE_API_KEY"] = "您的KEY"
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("請設定 GOOGLE_API_KEY")
    st.stop()

# 3. 初始化 Embedding 與 FAISS 資料庫 (使用快取加速)
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # 載入您的資料庫 (注意路徑要對)
    db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    return db

try:
    db = load_db()
    retriever = db.as_retriever(search_kwargs={"k": 3}) # 每次找 3 筆最相關的
except Exception as e:
    st.error(f"資料庫讀取失敗，請確認 faiss_db 資料夾是否存在。錯誤: {e}")
    st.stop()

# 4. 設定 LLM 與 Prompt (您的 Persona 設定)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", # 或 gemini-pro
    temperature=0.3, 
    google_api_key=st.secrets["GOOGLE_API_KEY"],
    convert_system_message_to_human=True
)

# 定義 Prompt Template
custom_prompt_template = """
你是專業且充滿熱忱的保險業務員，致力於提供最優質的服務。
你擁有市面上幾家大型保險公司的所有保險商品資料。

請務必嚴格遵守以下規則：
1. **只能**根據下方的【已知資訊】來回答問題。若資料不足或題目超過能力範圍，請回答：「不好意思，目前的資料庫中沒有相關資訊，建議您直接洽詢該保險公司的專人客服服務。」
2. 拒絕回答任何跟保險以外相關內容（例如：食譜、程式碼、旅遊景點介紹等），請禮貌拒絕並將話題引導回保險。
3. 語氣保持親切友善、專業簡潔，並使用台灣繁體中文。
4. 在提供答案的同時，請根據內容給予具體的建議。

【已知資訊】：
{context}

使用者問題：{question}

專業業務員回覆：
"""

PROMPT = PromptTemplate(
    template=custom_prompt_template, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# 5. 介面設計 (Tab 分頁)
st.title("🛡️ 專業保險諮詢助手")
st.markdown("---")

tab1, tab2 = st.tabs(["💬 線上諮詢 Chatbot", "📋 智慧保險推薦"])

# === 功能一：Chatbot ===
with tab1:
    st.subheader("有什麼我可以幫您的嗎？")
    
    # 初始化聊天紀錄
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 顯示歷史訊息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 使用者輸入
    if prompt := st.chat_input("請輸入您的問題 (例如：我想比較富邦跟國泰的意外險)"):
        # 顯示使用者訊息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 回答
        with st.chat_message("assistant"):
            with st.spinner("正在查詢保險條款中..."):
                # 先做簡單的關鍵字過濾 (非必要，但可增加防呆)
                if any(x in prompt for x in ["蛋糕", "食譜", "天氣", "政治"]):
                    response_text = "不好意思，我專注於提供專業的保險諮詢服務，無法回答與保險無關的問題喔！如果您有保險需求，歡迎隨時問我。"
                else:
                    try:
                        result = qa_chain.invoke({"query": prompt})
                        response_text = result["result"]
                        
                        # (選用) 顯示參考來源
                        # source_docs = result["source_documents"]
                        # for doc in source_docs:
                        #     with st.expander("參考資料來源"):
                        #         st.write(doc.metadata)
                    except Exception as e:
                        response_text = "系統發生錯誤，請稍後再試。"

                st.markdown(response_text)
        
        # 儲存 AI 回答
        st.session_state.messages.append({"role": "assistant", "content": response_text})

# === 功能二：保險推薦 ===
with tab2:
    st.subheader("為您量身打造的保險規劃")
    with st.form("recommendation_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("性別", ["男", "女"])
            age = st.number_input("年齡", min_value=0, max_value=100, value=30)
            job = st.text_input("職業", "一般內勤")
        with col2:
            salary = st.selectbox("年收入範圍", ["50萬以下", "50-100萬", "100-200萬", "200萬以上"])
            budget = st.text_input("預算 (月繳/年繳)", "月繳 3000 元")
        
        ins_type = st.selectbox("感興趣的保險類型", ["醫療險", "意外險", "儲蓄險/投資型", "旅遊平安險", "長照險", "壽險"])
        
        # 動態顯示旅遊資訊
        travel_info = ""
        if ins_type == "旅遊平安險":
            st.info("✈️ 偵測到旅遊需求，請補充細節：")
            dest = st.text_input("旅遊國家")
            days = st.number_input("旅遊天數", min_value=1, value=5)
            travel_info = f"，旅遊目的地為{dest}，預計旅遊{days}天"

        submit_btn = st.form_submit_button("開始分析推薦")

    if submit_btn:
        with st.spinner("正在分析您的需求並比對資料庫..."):
            # 組合 Prompt
            user_profile = f"""
            使用者資料：
            - 性別：{gender}
            - 年齡：{age}
            - 職業：{job}
            - 收入：{salary}
            - 預算：{budget}
            - 想找的保險：{ins_type}{travel_info}
            
            請根據以上使用者條件，從資料庫中推薦適合的{ins_type}商品，並說明推薦原因。
            如果不確定，請推薦最通用的方案並建議洽詢客服。
            """
            
            result = qa_chain.invoke({"query": user_profile})
            st.success("分析完成！以下是給您的建議：")
            st.markdown(result["result"])