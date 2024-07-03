import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class Settings:
    # Model settings
    LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2-0.5B-Instruct")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    
    # Generation settings
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
    
    # RAG settings
    PROVIDE_CONTEXT = os.getenv("PROVIDE_CONTEXT", "True").lower() == "true"
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Vector store settings
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store"))
    
    # Other settings
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_P = float(os.getenv("TOP_P", "0.95"))
    REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.15"))
    TOP_K_DOCUMENTS = int(os.getenv("TOP_K_DOCUMENTS", "3"))
    
    # System prompt
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", 
        "You are an AI assistant that provides helpful and accurate information based on the given context. "
        "If the context doesn't contain relevant information, rely on your general knowledge but mention "
        "that the answer is not based on the given context."
    )

    @classmethod
    def update(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)

settings = Settings()