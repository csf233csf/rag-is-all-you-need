import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import SpacyTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, pipeline
from threading import Thread
from settings import settings
import uuid
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import gc
import json
import re
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import spacy

class RAGSystem:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.embedding_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self.nlp = spacy.load("en_core_web_sm")
        self.text_splitter = SpacyTextSplitter(chunk_size=settings.CHUNK_SIZE, pipeline="en_core_web_sm")
        self.vector_store = self.load_vector_store()
        self.qa_chain = None
        self.embedding_cache = {}
        self.bm25 = None
        self.corpus = []
        self.feedback_store = {}
        self.load_llm()

    def load_llm(self):
        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(settings.LLM_MODEL, device_map="auto", trust_remote_code=True).eval()
            self.setup_qa_chain()

    def unload_llm(self):
        del self.tokenizer
        del self.model
        del self.qa_chain
        self.tokenizer = None
        self.model = None
        self.qa_chain = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def load_vector_store(self):
        os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
        
        index_file = os.path.join(settings.VECTOR_STORE_PATH, "index.faiss")
        if os.path.exists(index_file):
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
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": settings.TOP_K_DOCUMENTS}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )

    def preprocess_text(self, text: str) -> str:
        # Remove special characters and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def add_document(self, content: str, name: str, progress_callback=None):
        self.unload_llm()
        
        content = self.preprocess_text(content)
        texts = self.text_splitter.split_text(content)
        total_chunks = len(texts)
        
        for i, chunk in enumerate(texts):
            metadata = {"source": name, "id": str(uuid.uuid4())}
            self.vector_store.add_texts([chunk], [metadata])
            self.corpus.append(chunk)
            if progress_callback:
                progress_callback((i + 1) / total_chunks)
        
        self.vector_store.save_local(settings.VECTOR_STORE_PATH)
        self._update_bm25()
        self.load_llm()

    def _update_bm25(self):
        tokenized_corpus = [doc.split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def get_embedding(self, text: str) -> np.ndarray:
        if text not in self.embedding_cache:
            self.embedding_cache[text] = self.embedding_model.embed_query(text)
        return self.embedding_cache[text]

    def hybrid_search(self, query: str, k: int = 5) -> List[Dict]:
        dense_results = self.vector_store.similarity_search(query, k=k)
        
        # BM25 search
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_n = min(k, len(self.corpus))
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_n]
        
        # Combine results
        combined_results = dense_results
        for idx in top_bm25_indices:
            doc = self.vector_store.docstore._dict[list(self.vector_store.docstore._dict.keys())[idx]]
            if doc not in combined_results:
                combined_results.append(doc)
        
        return combined_results[:k]

    def query_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        query_vector = self.get_embedding(query)
        results = self.hybrid_search(query, k=top_k)
        return [
            {
                'id': doc.metadata.get('id', 'Unknown'),
                'name': doc.metadata.get('source', 'Unknown'),
                'content': doc.page_content,
                'similarity': cosine_similarity([query_vector], [self.get_embedding(doc.page_content)])[0][0]
            }
            for doc in results
        ]

    def generate_stream(self, query: str):
        self.load_llm()
        
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        docs = self.hybrid_search(query, k=settings.TOP_K_DOCUMENTS)
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

    def update_based_on_feedback(self, query: str, response: str, rating: int):
        if query not in self.feedback_store:
            self.feedback_store[query] = []
        self.feedback_store[query].append((response, rating))
        
        # If we have enough feedback, we could maybe retrain or fine-tune the model.??
        if len(self.feedback_store) > 100:
            #TODO maybe write model fine-tuning or retrieval adjustments here.
            pass

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
        self.corpus = [doc.page_content for doc in docs_to_keep]
        self._update_bm25()

    def clear_vector_store(self):
        self.vector_store = FAISS.from_texts(["Vector store cleared"], self.embedding_model)
        self.vector_store.save_local(settings.VECTOR_STORE_PATH)
        self.corpus = ["Vector store cleared"]
        self._update_bm25()

    def update_config(self):
        self.text_splitter = SpacyTextSplitter(chunk_size=settings.CHUNK_SIZE, pipeline="en_core_web_sm")
        self.unload_llm()
        self.load_llm()

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

    def export_cluster(self, cluster_id):
        cluster_info, _ = self.get_cluster_info()
        if cluster_id in cluster_info:
            cluster_data = cluster_info[cluster_id]
            return json.dumps(cluster_data, indent=2)
        return None

    def query_documents(self, query, top_k=5):
        query_vector = self.embedding_model.encode([query])[0]
        doc_ids = self.vector_store.similarity_search(query, k=top_k)
        results = []
        for doc in doc_ids:
            results.append({
                'id': doc.metadata.get('id', 'Unknown'),
                'name': doc.metadata.get('source', 'Unknown'),
                'content': doc.page_content,
                'similarity': cosine_similarity([query_vector], [self.embedding_model.encode([doc.page_content])[0]])[0][0]
            })
        return sorted(results, key=lambda x: x['similarity'], reverse=True)