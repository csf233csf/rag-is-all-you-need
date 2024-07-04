import streamlit as st
from utils import extract_text_from_pdf
from settings import settings
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
import json

def setup_page():
    st.set_page_config(layout="wide", page_title="Advanced RAG Chatbot", page_icon="🤖")

def sidebar():
    with st.sidebar:
        st.title("🧭 Navigation")
        return st.radio("Go to", ["💬 Chatbot", "⚙️ Config", "📚 Document Management", "🎨 Vector Visualization", "🔬 Cluster Analysis"], key="navigation")

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

def document_management_page(rag_system):
    st.title("📚 Document Management")

    # Add new document section
    st.header("Add New Document")
    uploaded_file = st.file_uploader("Upload a document (TXT or PDF)", type=["txt", "pdf"])
    if uploaded_file is not None:
        document_name = st.text_input("Document Name", value=uploaded_file.name)
        if st.button("Add Document"):
            if uploaded_file.type == "text/plain":
                content = uploaded_file.read().decode()
            elif uploaded_file.type == "application/pdf":
                content = extract_text_from_pdf(uploaded_file)
            
            with st.status("Adding document...", expanded=True) as status:
                st.write("Processing document...")
                progress_bar = st.progress(0)
                rag_system.add_document(content, document_name, lambda p: progress_bar.progress(p))
                status.update(label="Document added successfully!", state="complete")
            st.rerun()

    # Document list and management section
    st.header("Manage Documents")

    # Get all documents
    all_documents = rag_system.get_all_documents()

    # Search functionality
    search_query = st.text_input("Search documents", "")
    if search_query:
        all_documents = [doc for doc in all_documents if search_query.lower() in doc['name'].lower()]

    # Pagination
    docs_per_page = 10
    total_pages = max(1, (len(all_documents) - 1) // docs_per_page + 1)
    
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("Previous", disabled=(st.session_state.get('page_number', 1) == 1)):
                st.session_state.page_number = max(1, st.session_state.get('page_number', 1) - 1)
        with col2:
            page_number = st.slider("Page", min_value=1, max_value=total_pages, value=st.session_state.get('page_number', 1))
            st.session_state.page_number = page_number
        with col3:
            if st.button("Next", disabled=(st.session_state.get('page_number', 1) == total_pages)):
                st.session_state.page_number = min(total_pages, st.session_state.get('page_number', 1) + 1)
    else:
        page_number = 1
        st.session_state.page_number = 1

    start_idx = (page_number - 1) * docs_per_page
    end_idx = start_idx + docs_per_page

    # Display documents as expandable sections
    if all_documents:
        for doc in all_documents[start_idx:end_idx]:
            with st.expander(f"{doc['name']} (ID: {doc['id']})"):
                st.write(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
                if st.button("Delete", key=f"delete_{doc['id']}"):
                    rag_system.delete_document(doc['id'])
                    st.rerun()
    else:
        st.write("No documents found.")

    # Clear all documents
    if st.button("Clear All Documents"):
        if st.checkbox("I understand this will delete all documents"):
            rag_system.clear_vector_store()
            st.rerun()

    # Display some stats
    st.header("Document Store Statistics")
    st.write(f"Total number of documents: {len(all_documents)}")
    st.write(f"Vector Store Path: {settings.VECTOR_STORE_PATH}")
    st.write(f"Vector Store Exists: {os.path.exists(settings.VECTOR_STORE_PATH)}")

def vector_visualization_page(rag_system):
    st.title("🎨 Vector Visualization")

    vectors = rag_system.get_vector_representations()
    
    if len(vectors) > 1:
        pca = PCA(n_components=3)
        vectors_3d = pca.fit_transform(vectors)

        fig = px.scatter_3d(
            x=vectors_3d[:, 0],
            y=vectors_3d[:, 1],
            z=vectors_3d[:, 2],
            title="3D Visualization of Document Vectors",
            labels={'x': 'PCA 1', 'y': 'PCA 2', 'z': 'PCA 3'}
        )
        fig.update_layout(height=900)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough documents to visualize. Please add more documents.")
        
def cluster_analysis_page(rag_system):
    st.title("🔬 Cluster Analysis")

    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=5)
    result = rag_system.get_cluster_info(n_clusters)

    if result is None:
        st.warning("Not enough documents to perform clustering. Please add more documents.")
        return

    cluster_info, vectors = result

    # Visualize clusters
    pca = PCA(n_components=3)
    vectors_3d = pca.fit_transform(vectors)

    fig = go.Figure()

    for i, info in cluster_info.items():
        cluster_docs = info['documents']
        cluster_indices = [j for j, doc in enumerate(rag_system.get_all_documents()) if doc['id'] in [d['id'] for d in cluster_docs]]
        cluster_points = vectors_3d[cluster_indices]
        
        fig.add_trace(go.Scatter3d(
            x=cluster_points[:, 0],
            y=cluster_points[:, 1],
            z=cluster_points[:, 2],
            mode='markers',
            marker=dict(size=5),
            name=f'Cluster {i}'
        ))

    fig.update_layout(title="3D Visualization of Document Clusters")
    st.plotly_chart(fig)

    # Display cluster information
    for i, info in cluster_info.items():
        with st.expander(f"Cluster {i} (Size: {info['size']})"):
            st.write("Documents in this cluster:")
            for doc in info['documents']:
                st.write(f"- {doc['name']}")
            
            #TODO Implement an algorithm to find the most representative terms in each clusters
            # st.write("Most representative terms:")

    st.subheader("Query Cluster Similarity")
    query = st.text_input("Enter a query to find the most similar cluster:")
    if query:
        query_vector = rag_system.embedding_model.embed_query(query)
        similarities = [np.dot(query_vector, center) / (np.linalg.norm(query_vector) * np.linalg.norm(center)) 
                        for center in [info['center'] for info in cluster_info.values() if info['center'] is not None]]
        if similarities:
            most_similar_cluster = max(range(len(similarities)), key=similarities.__getitem__)
            st.write(f"The query is most similar to Cluster {most_similar_cluster}")
        else:
            st.write("Unable to determine the most similar cluster.")