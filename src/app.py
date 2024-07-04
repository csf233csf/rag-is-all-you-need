import streamlit as st
from rag_system import RAGSystem
from web import setup_page, sidebar, chatbot_page, config_page, document_management_page, vector_visualization_page, cluster_analysis_page

def main():
    setup_page()

    @st.cache_resource
    def get_rag_system():
        return RAGSystem()

    rag_system = get_rag_system()

    page = sidebar()

    if page == "💬 Chatbot":
        chatbot_page(rag_system)
    elif page == "⚙️ Config":
        config_page(rag_system)
    elif page == "📚 Document Management":
        document_management_page(rag_system)
    elif page == "🎨 Vector Visualization":
        vector_visualization_page(rag_system)
    elif page == "🔬 Cluster Analysis":
        cluster_analysis_page(rag_system)

if __name__ == "__main__":
    main()