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
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ==========================================
# 5. 設定 LLM
# ==========================================
llm = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant",
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
    st.subheader("💬 深度保險諮詢 (資深理賠專員版)")
    
    # 初始化 session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 顯示歷史訊息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 使用者輸入
    if user_input := st.chat_input("請輸入您的問題 (例如：癌症險的等待期是多久？)..."):
        # 1. 顯示使用者的問題
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🧠 正在調閱條款並進行深度分析..."):
                try:
                    # ==========================================
                    # 🔥 關鍵技術 1：對話歷史重組 (Contextualization)
                    # ==========================================
                    # 這是為了解決「它不知道你在問上一題」的問題
                    # 我們把「前一題的問答」跟「這一題」結合，變成一個完整的搜尋句
                    
                    history_context = ""
                    if len(st.session_state.messages) > 2:
                        last_q = st.session_state.messages[-3]["content"]
                        last_a = st.session_state.messages[-2]["content"]
                        history_context = f"前一輪對話背景：(問){last_q} -> (答){last_a}。"
                    
                    # 組合出「增強版搜尋語句」
                    search_query = f"{history_context} 使用者現在的問題：{user_input}。請尋找相關條款。"

                    # ==========================================
                    # 🔥 關鍵技術 2：擴大檢索與 Debug
                    # ==========================================
                    # Tab 1 需要更廣泛的搜尋 (k=10) 才能回答深入問題
                    retriever_deep = vectorstore.as_retriever(search_kwargs={"k": 10})
                    retrieved_docs = retriever_deep.invoke(search_query)

                    with st.expander("🕵️ [理賠視角] AI 參考的條款細節"):
                        if not retrieved_docs:
                            st.warning("⚠️ 查無相關條款，請嘗試更具體的關鍵字。")
                        for i, doc in enumerate(retrieved_docs):
                            source = doc.metadata.get('source', '未知')
                            st.markdown(f"**📄 條款 {i+1} ({source})**")
                            st.caption(doc.page_content[:200] + "...")
                            st.divider()

                    # ==========================================
                    # 🔥 關鍵技術 3：深度推論 Prompt (Chain of Thought)
                    # ==========================================
                    deep_persona = """
                    你是具備 20 年經驗的「資深保險理賠專員」。你的工作不是只有讀條款，而是要幫客戶「解釋條款背後的邏輯」與「理賠實務」。

                    【已知條款資訊】：
                    {context}

                    【對話歷史】：
                    {history}

                    【當前問題】：
                    {question}

                    【回答策略】：
                    1. **定義解釋**：不要只說結果，要解釋專有名詞 (例如：什麼是「既往症」？什麼是「門診手術」？)。
                    2. **條款引用**：回答時，請務必提到「根據條款第 X 條...」或是「依據條款說明...」。
                    3. **除外責任**：資深專員會主動告知風險。請在回答最後，補充「什麼情況下**不**會理賠」(Exclusions)。
                    4. **舉例說明**：如果是複雜概念，請舉一個簡單的例子 (例如：小明發生了...)。
                    5. **誠實原則**：如果資料庫裡完全沒有這家公司的條款，請直接說「資料庫無此商品資訊」，不要瞎掰。

                    請用台灣繁體中文，以專業、詳盡且有溫度的口吻回答：
                    """

                    deep_prompt = ChatPromptTemplate.from_messages([
                        ("human", deep_persona)
                    ])

                    # 建立 Chain
                    chain = (
                        {
                            "context": lambda x: format_docs(retrieved_docs),
                            "history": lambda x: history_context,
                            "question": lambda x: user_input
                        }
                        | deep_prompt
                        | llm
                        | StrOutputParser()
                    )

                    # 生成回答
                    response = chain.invoke(user_input)
                    st.markdown(response)
                    
                    # 存入紀錄
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(f"發生錯誤：{e}")

with tab2:
    st.subheader("📋 全方位智能保險規劃書 (V6.0 經理人整合版)")
    
    # --- 1. KYC (Know Your Customer) 完整的資料蒐集 ---
    with st.container(border=True):
        st.markdown("#### 👤 第一步：建立您的風險檔案")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("性別", ["男", "女"])
            age = st.number_input("實歲年齡", 0, 100, 30)
            job = st.text_input("職業 (影響費率關鍵)", "軟體工程師", help="請盡量詳細，例如：內勤行政、外送員、建築工人")
        with col2:
            salary = st.selectbox("年收入", ["50萬以下", "50-100萬", "100-200萬", "200萬以上", "不便透露"])
            # ✅ 新增：家庭狀況 (經理人最在意的點！)
            family_status = st.selectbox("家庭責任", ["單身未婚", "已婚無子", "已婚有子 (小孩幼齡)", "已婚有子 (小孩已獨立)", "單親家庭"])
        
        st.markdown("#### 🛡️ 第二步：您的保險需求")
        col3, col4 = st.columns(2)
        with col3:
            # 這裡保留防呆選單
            ins_type = st.selectbox("想規劃的險種", 
                ["壽險 (定期/終身)", "醫療險 (實支實付/日額)", "意外險 (傷害保險)", "重大傷病/癌症險", "儲蓄/理財險", "旅遊平安險"]
            )
        with col4:
            budget = st.text_input("預算範圍", "例如：月繳 3,000 元")

        # 特殊欄位：旅遊險
        extra_info = ""
        if "旅遊" in ins_type:
            dest = st.text_input("旅遊國家", "日本")
            days = st.number_input("天數", 1, 365, 5)
            extra_info = f"預計前往{dest}旅遊{days}天"

        # ✅ 新增：既有保單詢問
        has_insurance = st.checkbox("我已有類似保險 (希望 AI 協助檢視缺口或加強保障)")
        extra_info += "。已有類似保單，請著重在補強缺口。" if has_insurance else "。目前無此類保單，屬於新投保。"

    # --- 2. 開始分析 ---
    if st.button("🚀 生成專業建議書", type="primary"):
        with st.spinner("🤖 AI 顧問正在進行交叉比對與條款分析..."):
            
            # ==========================================
            # 🔧 引擎核心：V5.0 強力搜尋邏輯 (整合進來)
            # ==========================================
            # 為了避免只找到同一家，我們必須在這裡重新定義檢索器
            retriever_manager = vectorstore.as_retriever(
                search_type="mmr", 
                search_kwargs={"k": 6, "fetch_k": 100, "lambda_mult": 0.5}
            )

            # 設定純淨搜尋關鍵字 (避免被個資干擾)
            search_keyword = f"{ins_type} 條款 保單"
            if "旅遊" in ins_type:
                search_keyword += f" {dest}"

            # 手動執行檢索
            retrieved_docs = retriever_manager.invoke(search_keyword)

            # --- Debug 視窗 (確認有沒有抓到不同家) ---
            with st.expander("🕵️ [工程師模式] 檢索到的候選名單"):
                if not retrieved_docs:
                    st.warning("⚠️ 無法檢索到相關條款。")
                for i, doc in enumerate(retrieved_docs):
                    source = doc.metadata.get('source', doc.metadata.get('filename', '未知'))
                    company = doc.metadata.get('company', '未知公司')
                    st.markdown(f"**{i+1}. [{company}] {source}**")
                    st.caption(doc.page_content[:100] + "...")

            # ==========================================
            # 🔥 經理人 Prompt (結合表格要求 + 嚴格過濾)
            # ==========================================
            query = f"""
            【客戶畫像 (KYC)】：
            - 基本資料：{gender}, {age}歲
            - 職業風險：{job} (請精準判斷職業等級 1-6 級)
            - 經濟能力：年收{salary}, 預算{budget}
            - 家庭責任：{family_status} (🔥請重點分析此身份的風險缺口)
            - 需求目標：{ins_type}
            - 備註：{extra_info}

            【嚴格過濾指令】：
            1. **排除團體險**：除非客戶是企業，否則請忽略名稱含「團體」的商品。
            2. **排除不符險種**：請專注於推薦 {ins_type}。
            3. **強制差異化**：請務必嘗試推薦 **2 家不同保險公司** 的商品 (若資料庫有)。

            【任務指令】：
            你是資深的保險經紀人，請閱讀下方的【檢索到的條款內容】，產出一份專業建議書。
            
            1. **風險缺口分析**：根據「家庭責任」與「職業」，一針見血地點出客戶最該擔心的風險。
            2. **精選雙商品比較**：挑選 2 個最合適的方案。
            3. **表格化輸出**：務必使用 Markdown Table 製作比較表。
            
            【建議書輸出格式】：
            ### 📊 第一部分：您的風險雷達圖
            (針對家庭與職業的風險分析...)

            ### 🏆 第二部分：精選方案推薦
            #### 方案 A：[保險公司] - [商品名稱]
            * **核心優勢**：...
            * **適合您的原因**：...

            #### 方案 B：[保險公司] - [商品名稱]
            * **核心優勢**：...
            * **適合您的原因**：...

            ### ⚖️ 第三部分：超級比一比 (Comparison Table)
            | 比較項目 | 方案 A | 方案 B |
            | :--- | :--- | :--- |
            | 保險公司 | ... | ... |
            | 商品特色 | ... | ... |
            | 承保範圍 | ... | ... |
            | 預估保費 | ... | ... |
            | 推薦指數 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

            ### 💡 經理人總結
            (給客戶的最終建議)
            """
            
            try:
                # 將檢索到的資料注入 Prompt
                docs_text = "\n\n".join(f"來源: {d.metadata.get('source', '未知')}\n內容: {d.page_content}" for d in retrieved_docs)
                
                # 建立臨時 Chain 來執行這個特殊任務
                prompt_template = ChatPromptTemplate.from_template(query + "\n\n【檢索到的條款內容】：\n{context}")
                chain = prompt_template | llm | StrOutputParser()
                
                response = chain.invoke({"context": docs_text})
                st.markdown(response)

                # --- 3. 專業結尾 ---
                st.info("💡 **經理人小叮嚀**：\n本建議書由 AI 系統依據現有條款資料庫生成，僅供初步規劃參考。實際承保內容、費率與理賠條件，請務必以保險公司正式保單條款為準。")
                
            except Exception as e:
                st.error(f"分析過程發生錯誤: {e}")