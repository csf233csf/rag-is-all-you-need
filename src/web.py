import streamlit as st
from utils import extract_text_from_pdf
from settings import settings
import os

def setup_page():
    st.set_page_config(layout="wide", page_title="Advanced RAG Chatbot", page_icon="🤖")

def sidebar():
    with st.sidebar:
        st.title("🧭 Navigation")
        return st.radio("Go to", ["💬 Chatbot", "⚙️ Config", "📚 Document Management"], key="navigation")

def chatbot_page(rag_system):
    st.title("💬 Advanced RAG Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What's your question?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response_chunk in rag_system.generate_stream(prompt):
                full_response += response_chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

def config_page(rag_system):
    st.title("⚙️ RAG Configuration")

    max_tokens = st.number_input("Max Tokens", min_value=1, max_value=2048, value=settings.MAX_TOKENS)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=settings.TEMPERATURE, step=0.1)
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=settings.TOP_P, step=0.05)
    repetition_penalty = st.slider("Repetition Penalty", min_value=1.0, max_value=2.0, value=settings.REPETITION_PENALTY, step=0.05)
    chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=settings.CHUNK_SIZE)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=settings.CHUNK_OVERLAP)
    top_k_documents = st.number_input("Top K Documents", min_value=1, max_value=10, value=settings.TOP_K_DOCUMENTS)
    system_prompt = st.text_area("System Prompt", value=settings.SYSTEM_PROMPT, height=200)

    if st.button("Save Configuration"):
        settings.update(
            MAX_TOKENS=max_tokens,
            TEMPERATURE=temperature,
            TOP_P=top_p,
            REPETITION_PENALTY=repetition_penalty,
            CHUNK_SIZE=chunk_size,
            CHUNK_OVERLAP=chunk_overlap,
            TOP_K_DOCUMENTS=top_k_documents,
            SYSTEM_PROMPT=system_prompt
        )
        rag_system.update_config()
        st.success("Configuration updated successfully!")

def database_page(rag_system):
    st.title("📚 Document Management")

    st.subheader("Add New Document")
    uploaded_file = st.file_uploader("Upload a document (TXT or PDF)", type=["txt", "pdf"])
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            content = uploaded_file.read().decode()
        elif uploaded_file.type == "application/pdf":
            content = extract_text_from_pdf(uploaded_file)
        
        if st.button("Add Document"):
            rag_system.add_document(content)
            st.experimental_rerun()

    st.subheader("Vector Store Information")
    doc_count = rag_system.get_document_count()
    st.write(f"Number of documents: {doc_count}")

    if st.button("Clear Vector Store"):
        rag_system.clear_vector_store()
        st.rerun()

    st.subheader("Sample Queries")
    sample_query = st.text_input("Enter a sample query to test document retrieval:")
    if st.button("Retrieve Relevant Documents"):
        if sample_query:
            docs = rag_system.vector_store.similarity_search(sample_query, k=settings.TOP_K_DOCUMENTS)
            for i, doc in enumerate(docs, 1):
                st.write(f"Document {i}:")
                st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
        else:
            st.warning("Please enter a sample query.")

    st.subheader("Debug Information")
    st.write(f"Vector Store Path: {settings.VECTOR_STORE_PATH}")
    st.write(f"Vector Store Path Exists: {os.path.exists(settings.VECTOR_STORE_PATH)}")
    st.write(f"Vector Store Type: {type(rag_system.vector_store).__name__}")