# schemas.py
from pydantic import BaseModel, HttpUrl
from typing import List

# --- As specified by the Hackathon Documentation ---

class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]