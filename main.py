# main.py
import os
import asyncio
import traceback
from fastapi import FastAPI, Depends, HTTPException, Header, Body
from typing import Annotated
from dotenv import load_dotenv

from utils import MODELS # Import the single instance of our loaded models
from ingestion_agent import ingest_and_index_document
from retrieval_agent import hybrid_search_and_rerank
from reasoning_agent import run_reasoning_pipeline
from schemas import QueryRequest, QueryResponse

load_dotenv()
app = FastAPI(title="DeLLM-X: The Winning Engine")

# --- Security Placeholder ---
async def verify_token(authorization: Annotated[str, Header()]):
    pass # In a real scenario, verification logic would go here

# --- Asynchronous Helper for the Full Pipeline ---
async def process_single_question(index_name: str, chunks: list, question: str):
    """Orchestrates the agent workflow. Now accepts index_name as a string."""
    loop = asyncio.get_event_loop()
    
    # Pass the 'index_name' string, not the index object
    retrieved_docs = await loop.run_in_executor(
        None, hybrid_search_and_rerank, index_name, chunks, question, MODELS
    )
    
    answer = await loop.run_in_executor(
        None, run_reasoning_pipeline, question, retrieved_docs, MODELS
    )
    return answer

# --- The Main API Endpoint ---
@app.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def run_deellm_x_submission(request: QueryRequest = Body(...)):
    """Main API endpoint for the DeLLM-X pipeline."""
    try:
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not index_name:
            raise ValueError("PINECONE_INDEX_NAME not set in .env file")

        # Ingestion returns the index object (which we no longer need here) and the raw chunks
        _, chunks = ingest_and_index_document(
            doc_url=str(request.documents),
            index_name=index_name,
            embedding_model=MODELS.embedding_model
        )

        # Create the list of tasks, correctly passing the 'index_name' string
        tasks = [
            process_single_question(index_name, chunks, q) for q in request.questions
        ]

        answers = await asyncio.gather(*tasks, return_exceptions=True)

        final_answers = []
        for i, ans in enumerate(answers):
            if isinstance(ans, Exception):
                print(f"--- ERROR FOR QUESTION #{i+1}: {request.questions[i]} ---")
                print(f"Exception Type: {type(ans)}")
                print(f"Exception Details: {ans}")
                traceback.print_exception(type(ans), ans, ans.__traceback__)
                final_answers.append("An error occurred. Check server logs for details.")
            else:
                final_answers.append(ans)
        
        cleaned_answers = []
        for ans in final_answers:
            if "Final Answer:" in str(ans):
                cleaned_answers.append(ans.split("Final Answer:", 1)[1].strip())
            else:
                cleaned_answers.append(ans)

        return QueryResponse(answers=cleaned_answers)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", summary="Health Check")
def read_root():
    return {"status": "DeLLM-X is running and ready to win."}