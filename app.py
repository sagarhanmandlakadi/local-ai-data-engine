import io, re, tempfile
import pandas as pd
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"
st.set_page_config(page_title="AI Data Analytics", page_icon="📊", layout="wide")
st.title("📊 AI-Powered Conversational Data Analytics Engine")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
with st.sidebar:
    st.header("📁 Document Source")
    uploaded_file = st.file_uploader("Upload PDF or CSV", type=["pdf", "csv"])
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
@st.cache_resource
def get_llm():
    return OllamaLLM(model=LLM_MODEL, temperature=0, num_ctx=8192)
@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL)
@st.cache_resource(show_spinner="📖 Reading PDF...")
def build_pdf_index(file_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        pdf_path = tmp.name
    pages = PyPDFLoader(pdf_path).load()
    full_text = "\n\n".join(f"--- PAGE {i + 1} ---\n{page.page_content}" for i, page in enumerate(pages))
    if not full_text.strip():
        raise ValueError("No readable text found in PDF.")
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(pages)
    store = InMemoryVectorStore.from_documents(chunks, get_embeddings())
    return store, full_text, pages
@st.cache_data
def load_csv(file_bytes: bytes):
    return pd.read_csv(io.BytesIO(file_bytes))
def answer_pdf(question: str, store, full_text: str):
    is_positional = any(w in question.lower() for w in ["first", "second", "third", "last", "beginning", "start", "initial", "order"])
    if len(full_text) <= 15000:
        context = full_text
    else:
        docs = store.similarity_search(question, k=6)
        retrieved = "\n\n".join(f"[Page {d.metadata.get('page', 0) + 1}]\n{d.page_content}" for d in docs)
        context = f"BEGINNING:\n{full_text[:6000]}\n\nRELEVANT:\n{retrieved}\n\nEND:\n{full_text[-4000:]}" if is_positional else retrieved
    prompt = f"""You are a professional assistant. Answer the user question based strictly on the provided context.
You must mimic the exact layout, bolding, line wrapping, and line breaks shown in the example template below.

### EXAMPLE TEMPLATE FOR LISTS:
Question: list the items
Context: Item Alpha is a tool used for writing code. It helps save time. Item Beta is a database cluster. It stores structured user logs securely.
Answer:
**Item Alpha**
• A tool used for writing code.
• Helps save time.

**Item Beta**
• A database cluster.
• Stores structured user logs securely.

### YOUR ACTUAL TASK:
Context:
{context}

Question:
{question}

Answer:"""
    return get_llm().invoke(prompt).strip()
def answer_csv(question: str, df: pd.DataFrame):
    llm = get_llm()
    schema = f"Columns: {list(df.columns)}\nData types:\n{df.dtypes.to_string()}\nRows: {len(df)}\nFirst 10 rows:\n{df.head(10).to_string(index=False)}"
    code_prompt = f"Generate ONE Python expression using `df` to answer the question. Return ONLY the code, no markdown.\nINFO:\n{schema}\nQUESTION:\n{question}\nEXPRESSION:"
    expr = re.sub(r"```(?:python)?", "", llm.invoke(code_prompt).strip(), flags=re.IGNORECASE).replace("```", "").strip()
    if not expr: return "⚠️ No analysis expression generated."
    if any(item in expr for item in ["import ", "__import__", "open(", "exec(", "eval(", "os.", "sys.", "subprocess"]):
        return "⚠️ Unsafe expression blocked."
    try:
        res = eval(expr, {"__builtins__": {}}, {"df": df, "pd": pd})
        res_text = res.to_string(index=False) if isinstance(res, pd.DataFrame) else (res.to_string() if isinstance(res, pd.Series) else str(res))
    except Exception as e:
        return f"⚠️ Calculation failed.\nExpression: `{expr}`\nError: {e}"
    explain_prompt = f"Answer the question using ONLY the calculated result.\nQuestion: {question}\nResult: {res_text}\nAnswer:"
    return f"{llm.invoke(explain_prompt).strip()}\n\nExpression: `{expr}`"
if uploaded_file is None:
    st.info("👈 Upload a file to begin.")
    st.stop()
file_bytes = uploaded_file.getvalue()
is_pdf = uploaded_file.name.lower().endswith(".pdf")
try:
    if is_pdf:
        store, full_text, pages = build_pdf_index(file_bytes)
        with st.expander("📄 PDF info"):
            st.write(f"Pages: {len(pages)} | Characters: {len(full_text)}")
    else:
        df = load_csv(file_bytes)
        with st.expander("📊 CSV Preview"):
            st.dataframe(df.head(20), use_container_width=True)
            st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
except Exception as e:
    st.error(f"⚠️ Error reading file: {e}")
    st.stop()
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])
question = st.chat_input("Ask a question...")
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"): st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            try:
                ans = answer_pdf(question, store, full_text) if is_pdf else answer_csv(question, df)
            except Exception as e:
                ans = f"⚠️ Error: {e}"
        st.markdown(ans)
    st.session_state.chat_history.append({"role": "assistant", "content": ans})
