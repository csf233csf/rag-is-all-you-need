from fastapi import FastAPI
from pydantic import BaseModel
from rag_system import RAGSystem

app = FastAPI()
rag_system = RAGSystem()

class Query(BaseModel):
    text: str

@app.post("/query")
async def query_documents(query: Query):
    results = rag_system.query_documents(query.text)
    return {"results": results}

@app.post("/generate")
async def generate_response(query: Query):
    response = "".join(rag_system.generate_stream(query.text))
    return {"response": response}

# Run with: uvicorn api:app --reload