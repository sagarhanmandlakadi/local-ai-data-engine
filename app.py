import io
import re
import tempfile

import pandas as pd
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

LLM_MODEL = "llama3.1:8b"        # or "qwen2.5:7b" / "llama3.2:3b" if RAM is tight
EMBED_MODEL = "nomic-embed-text"

st.set_page_config(page_title="Local AI Data Engine", layout="wide")
st.title("📊 AI-Powered Conversational Data Analytics Engine")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("📁 Document Source")
    uploaded_file = st.file_uploader("Upload PDF or CSV", type=["pdf", "csv"])
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()


def get_llm():
    return OllamaLLM(model=LLM_MODEL, temperature=0, num_ctx=8192)


@st.cache_resource(show_spinner="Indexing PDF...")
def build_pdf_index(file_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    pages = PyPDFLoader(path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    store = InMemoryVectorStore.from_documents(chunks, OllamaEmbeddings(model=EMBED_MODEL))
    full_text = "\n".join(p.page_content for p in pages)
    return store, full_text


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def answer_pdf(question: str, store, full_text: str) -> str:
    # Small docs (e.g. a resume): send the whole thing. Large docs: retrieve.
    if len(full_text) < 12000:
        context = full_text
    else:
        docs = store.similarity_search(question, k=5)
        context = "\n\n---\n\n".join(
            f"[page {d.metadata.get('page', 0) + 1}] {d.page_content}" for d in docs
        )
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say \"I couldn't find that in the document.\"\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        f"Question (repeated): {question}\n"
        "Answer:"
    )
    return get_llm().invoke(prompt)


def answer_csv(question: str, df: pd.DataFrame) -> str:
    llm = get_llm()
    schema = (
        f"Columns and dtypes:\n{df.dtypes.to_string()}\n\n"
        f"First 5 rows:\n{df.head().to_string(index=False)}"
    )
    code_prompt = (
        "You write a single pandas expression that answers the question. "
        "The DataFrame is named df. pandas is available as pd. "
        "Return ONLY the expression, no explanation, no code fences.\n\n"
        f"{schema}\n\nQuestion: {question}\nExpression:"
    )
    expr = llm.invoke(code_prompt).strip()
    expr = re.sub(r"^```(?:python)?|```$", "", expr, flags=re.MULTILINE).strip()

    try:
        # Runs model-generated code locally; fine for personal use, not for a public app.
        result = eval(expr, {"__builtins__": {}}, {"df": df, "pd": pd})
    except Exception as e:
        return f"⚠️ Couldn't compute that. Generated expression: `{expr}` → {e}"

    explain_prompt = (
        f"Question: {question}\n"
        f"Computed result:\n{result}\n\n"
        "State the answer clearly in one or two sentences using only this result."
    )
    return f"{llm.invoke(explain_prompt)}\n\n`{expr}`"


# ---- Main ----
if uploaded_file is None:
    st.warning("⚠️ Please upload a PDF or CSV in the sidebar to begin.")
    st.stop()

file_bytes = uploaded_file.getvalue()
is_pdf = uploaded_file.name.lower().endswith(".pdf")

try:
    if is_pdf:
        store, full_text = build_pdf_index(file_bytes)
    else:
        df = load_csv(file_bytes)
        with st.expander("Preview data"):
            st.dataframe(df.head(20))
except Exception as e:
    st.error(f"⚠️ Failed to parse '{uploaded_file.name}': {e}")
    st.stop()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about your document..."):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            try:
                answer = answer_pdf(question, store, full_text) if is_pdf else answer_csv(question, df)
            except Exception as e:
                answer = f"⚠️ Could not reach local Ollama. Is it running? Error: {e}"
        st.markdown(answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})