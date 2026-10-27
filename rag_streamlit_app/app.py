import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
st.set_page_config(page_title="DocuChat | RAG Assistant", page_icon="📚", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px;}
.hero {padding: 1.5rem 1.7rem; border-radius: 18px; background: linear-gradient(120deg, #102a43, #243b53); color: white; margin-bottom: 1.2rem;}
.hero h1 {color: white; margin-bottom: .25rem;}
.hero p {color: #d9e2ec; margin-bottom: 0;}
.hint {color: #627d98; font-size: .92rem;}
</style>
<div class="hero">
<h1>📚 DocuChat</h1>
<p>Chat with your PDFs and documents using retrieval-augmented generation.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("Upload a file to build a searchable knowledge base, then ask questions.")
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])
    chunk_size = st.slider("Chunk size", 300, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 100, 25)
    st.divider()
    index_button = st.button("📥 Index document", use_container_width=True, type="primary")
    reset_chat = st.button("🧹 Clear chat", use_container_width=True)
    st.markdown("---")
    st.caption("Embeddings: Gemini Embedding")
    st.caption("Chat: Gemini Flash")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "indexed_name" not in st.session_state:
    st.session_state.indexed_name = None

if reset_chat:
    st.session_state.messages = []
    st.rerun()

def load_uploaded_docs(file):
    suffix = Path(file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(file.getvalue())
        temp_path = temp.name
    try:
        if suffix == ".pdf":
            return PyPDFLoader(temp_path).load()
        if suffix == ".txt":
            return TextLoader(temp_path, encoding="utf-8").load()
        raise ValueError("Unsupported file format.")
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

def build_vectorstore(documents, size, overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No readable text was found. Scanned PDFs may need OCR.")
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Add GOOGLE_API_KEY or GEMINI_API_KEY to your .env file.")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )
    # A separate in-memory collection prevents mixing documents/embedding spaces.
    collection_name = "docuchat_" + os.urandom(8).hex()
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name
    )
    return store, len(chunks)

if index_button:
    if uploaded_file is None:
        st.sidebar.error("Please upload a PDF or TXT file first.")
    else:
        try:
            with st.spinner("Reading, splitting, and indexing your document..."):
                docs = load_uploaded_docs(uploaded_file)
                store, chunk_count = build_vectorstore(docs, chunk_size, chunk_overlap)
                st.session_state.vectorstore = store
                st.session_state.indexed_name = uploaded_file.name
                st.session_state.messages = []
            st.success(f"Indexed **{uploaded_file.name}** — {len(docs)} page(s)/document section(s), {chunk_count} chunks.")
        except Exception as exc:
            st.error(f"Could not index the document: {exc}")

if st.session_state.indexed_name:
    st.info(f"📄 Active document: **{st.session_state.indexed_name}**")
else:
    st.info("Upload a PDF or TXT file in the sidebar and click **Index document** to begin.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved sources"):
                for idx, source in enumerate(message["sources"], start=1):
                    page = source.get("page")
                    page_label = f" · page {page + 1}" if isinstance(page, int) else ""
                    st.markdown(f"**Source {idx}{page_label}**")
                    st.write(source.get("text", ""))

question = st.chat_input("Ask a question about your indexed document…")
if question:
    if st.session_state.vectorstore is None:
        st.warning("Please upload and index a document before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the document and generating an answer..."):
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 3, "lambda_mult": 0.5, "fetch_k": 10}
                    )
                    retrieved_docs = retriever.invoke(question)
                    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful assistant that answers questions based on the provided context. Do not make up facts."),
                        ("human", "Answer the question based on the context below. If the answer is not contained within the text below, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {question}")
                    ])
                    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                    if not api_key:
                        raise ValueError("API key not found. Add GOOGLE_API_KEY or GEMINI_API_KEY to your .env file.")
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-3.8-flash",
                        google_api_key=api_key,
                        temperature=0
                    )
                    response = llm.invoke(prompt.format_messages(context=context, question=question))
                    answer = response.content
                    st.markdown(answer)
                    sources = [{"text": d.page_content, "page": d.metadata.get("page")} for d in retrieved_docs]
                    with st.expander("Retrieved sources"):
                        for idx, source in enumerate(sources, start=1):
                            page = source.get("page")
                            page_label = f" · page {page + 1}" if isinstance(page, int) else ""
                            st.markdown(f"**Source {idx}{page_label}**")
                            st.write(source["text"])
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            except Exception as exc:
                error_text = f"Something went wrong: {exc}"
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text})

st.markdown('<p class="hint">Tip: Scanned PDFs containing images rather than selectable text may need OCR before indexing.</p>', unsafe_allow_html=True)
