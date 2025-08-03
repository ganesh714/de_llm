# main.py
import os
import asyncio
from fastapi import FastAPI, Depends, HTTPException, Header, Body
from pydantic import BaseModel, HttpUrl
from typing import List, Annotated
from dotenv import load_dotenv

# Import the pre-loaded models and agent functions from your core_agents file
from core_agents import (
    EMBEDDING_MODEL,
    CROSS_ENCODER,
    ingest_and_index_document,
    retrieve_and_rerank,
    run_decision_engine,
)
# Make sure you have a schemas.py file or define these Pydantic models here
from schemas import QueryRequest, QueryResponse

load_dotenv()

app = FastAPI(title="DeLLM")

# --- Security Dependency (FIXED) ---
async def verify_token(authorization: Annotated[str, Header()]):
    """Verifies the bearer token. This is a placeholder."""
    # Add a 'pass' statement to make the empty function valid
    pass
    # In a real app, you would add your token verification logic here.
    # For example:
    # expected_token = f"Bearer {os.getenv('HACKRX_BEARER_TOKEN')}"
    # if authorization != expected_token:
    #     raise HTTPException(status_code=401, detail="Invalid authorization token")


# --- Asynchronous Helper ---
async def process_single_question(index, question: str):
    """Orchestrates the agent workflow, passing pre-loaded models."""
    loop = asyncio.get_event_loop()

    retrieved_matches = await loop.run_in_executor(
        None, retrieve_and_rerank, index, question, EMBEDDING_MODEL, CROSS_ENCODER
    )

    answer = await loop.run_in_executor(
        None, run_decision_engine, question, retrieved_matches
    )
    return answer

# --- API ENDPOINT ---
@app.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def run_deellm_submission(request: QueryRequest = Body(...)):
    """Main API endpoint for the DeLLM pipeline."""
    try:
        index_name = os.getenv("PINECONE_INDEX_NAME")
        # Pass the pre-loaded embedding model to the ingestion function
        index = ingest_and_index_document(str(request.documents), index_name, EMBEDDING_MODEL)

        tasks = [process_single_question(index, q) for q in request.questions]
        answers = await asyncio.gather(*tasks, return_exceptions=True)

        final_answers = []
        for ans in answers:
            if isinstance(ans, Exception):
                print(f"Error processing a question: {ans}")
                final_answers.append("An error occurred while processing this question.")
            else:
                final_answers.append(ans)

        return QueryResponse(answers=final_answers)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"A critical internal server error occurred: {e}")

@app.get("/", summary="Health Check")
def read_root():
    return {"status": "DeLLM API is operational"}

# The stray '}' at the end of the file has been removed.