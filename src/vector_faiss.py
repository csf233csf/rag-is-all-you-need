import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name='BAAI/bge-large-en-v1.5'):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts):
        return self.model.encode(texts)

class VectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.document_ids = []

    def add_documents(self, doc_ids, embeddings):
        self.document_ids.extend(doc_ids)
        self.index.add(np.array(embeddings).astype('float32'))

    def search(self, query_vector, top_k=3):
        if self.index.ntotal == 0:
            return []
        distances, indices = self.index.search(np.array([query_vector]).astype('float32'), min(top_k, self.index.ntotal))
        return [self.document_ids[i] for i in indices[0] if i < len(self.document_ids)]