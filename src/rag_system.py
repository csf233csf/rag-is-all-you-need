import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, pipeline
from threading import Thread
from settings import settings
import uuid
from sklearn.cluster import KMeans
import numpy as np

class RAGSystem:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(settings.LLM_MODEL, device_map="auto", trust_remote_code=True).eval()
        self.embedding_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.vector_store = self.load_vector_store()
        self.qa_chain = self.setup_qa_chain()

    def load_vector_store(self):
        '''
        Initializing Vector Database and load the database.
        '''
        os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
        
        index_file = os.path.join(settings.VECTOR_STORE_PATH, "index.faiss")
        if os.path.exists(index_file):
            # For debugging
            # st.info(f"Loading existing vector store from {settings.VECTOR_STORE_PATH}")
            try:
                return FAISS.load_local(settings.VECTOR_STORE_PATH, self.embedding_model, allow_dangerous_deserialization=True)
            except Exception as e:
                st.error(f"Error loading existing vector store: {str(e)}")
                st.info("Initializing a new vector store.")
        else:
            st.info("No existing vector store found. Initializing a new one.")
        
        new_store = FAISS.from_texts(["Initial document"], self.embedding_model)
        new_store.save_local(settings.VECTOR_STORE_PATH)
        return new_store

    def setup_qa_chain(self):
        '''
        Essential RAG techniques
        '''
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=settings.MAX_TOKENS,
            temperature=settings.TEMPERATURE,
            top_p=settings.TOP_P,
            repetition_penalty=settings.REPETITION_PENALTY
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=f"{settings.SYSTEM_PROMPT}\n\nContext: {{context}}\n\nHuman: {{question}}\n\nAssistant:"
        )
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": settings.TOP_K_DOCUMENTS}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )

    def add_document(self, content, name):
        '''
        Add vectorization of documents
        '''
        texts = self.text_splitter.split_text(content)
        metadatas = [{"source": name, "id": str(uuid.uuid4())} for _ in texts]
        self.vector_store.add_texts(texts, metadatas=metadatas)
        self.vector_store.save_local(settings.VECTOR_STORE_PATH)
        st.success(f"Document '{name}' added successfully. Vector store saved to {settings.VECTOR_STORE_PATH}")

    def get_all_documents(self):
        return [
            {"id": doc.metadata.get("id"), "name": doc.metadata.get("source"), "content": doc.page_content}
            for doc in self.vector_store.docstore._dict.values()
        ]

    def delete_document(self, doc_id):
        docs_to_keep = [doc for doc in self.vector_store.docstore._dict.values() if doc.metadata.get("id") != doc_id]
        new_store = FAISS.from_documents(docs_to_keep, self.embedding_model)
        new_store.save_local(settings.VECTOR_STORE_PATH)
        self.vector_store = new_store
        st.success(f"Document with ID {doc_id} deleted successfully.")

    def clear_vector_store(self):
        self.vector_store = FAISS.from_texts(["Vector store cleared"], self.embedding_model)
        self.vector_store.save_local(settings.VECTOR_STORE_PATH)
        st.success("Vector store cleared successfully!")

    def generate_stream(self, query):
        '''
        Generates Stream Outputs of LLM
        '''
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        docs = self.vector_store.similarity_search(query, k=settings.TOP_K_DOCUMENTS)
        context = "\n".join([doc.page_content for doc in docs])
        
        full_prompt = self.qa_chain.combine_documents_chain.llm_chain.prompt.format(
            context=context,
            question=query
        )
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=settings.MAX_TOKENS)

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

        thread.join()

    def update_config(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.qa_chain = self.setup_qa_chain()

    def get_vector_representations(self):
        return np.array([self.vector_store.index.reconstruct(i) for i in range(self.vector_store.index.ntotal)])

    def cluster_documents(self, n_clusters=5):
        vectors = self.get_vector_representations()
        documents = self.get_all_documents()
        
        if len(vectors) < n_clusters or len(documents) < n_clusters:
            return None, None, None  # Not enough documents to cluster

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(vectors)
        
        clustered_docs = []
        for i, doc in enumerate(documents):
            if i < len(cluster_labels):  # Ensure we don't go out of bounds
                doc['cluster'] = int(cluster_labels[i])
                clustered_docs.append(doc)
        
        return clustered_docs, kmeans.cluster_centers_, vectors

    def get_cluster_info(self, n_clusters=5):
        clustered_docs, cluster_centers, vectors = self.cluster_documents(n_clusters)
        if clustered_docs is None:
            return None

        cluster_info = {}
        for i in range(n_clusters):
            cluster_docs = [doc for doc in clustered_docs if doc['cluster'] == i]
            cluster_info[i] = {
                'size': len(cluster_docs),
                'documents': cluster_docs,
                'center': cluster_centers[i] if i < len(cluster_centers) else None
            }
        return cluster_info, vectors