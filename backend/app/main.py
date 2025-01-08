from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime
from database.vector_store import VectorStore
from services.synthesizer import Synthesizer
from timescale_vector import client

vec = VectorStore()
app = FastAPI()

# Define the request model
class QueryRequest(BaseModel):
    query: str

# Define the response model
class QueryResponse(BaseModel):
    response: str

def similarlity_search(query):
    relevant_question = query
    results = vec.search(relevant_question, limit=3)

    response = Synthesizer.generate_response(question=relevant_question, context=results)

    # print(f"\n{response.answer}")
    # print("\nThought process:")
    # for thought in response.thought_process:
    #     print(f"- {thought}")
    # print(f"\nContext: {response.enough_context}")
    # print(f"\nResults: {results}")
    return response.answer

# Mock function for RAG process (replace with actual implementation)
def rag_pipeline(query: str) -> str:
    # Simulate retrieval of documents and generation
    # Replace this with your actual RAG logic
    # retrieved_docs = ["doc1 content", "doc2 content"]  # Mock retrieved documents
    generated_response = similarlity_search(query)
    return generated_response

@app.post("/rag-query", response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    try:
        # Process the query using the RAG pipeline
        response = rag_pipeline(request.query)
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Hello World"}
