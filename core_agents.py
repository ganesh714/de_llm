# core_agents.py

import os
import io
import requests
import tempfile
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
# from langchain_openai import ChatOpenAI
# from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Pre-load Models for Efficiency and to Prevent Errors ---
# This is the key fix: Load models once when the app starts.
print("Loading sentence-transformer and cross-encoder models...")
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
CROSS_ENCODER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Models loaded successfully.")

# --- Agent 1: Ingestion ---
def ingest_and_index_document(doc_url: str, index_name: str, embedding_model):
    """Downloads, processes, and indexes a document."""
    print(f"Agent 1: Ingesting document from {doc_url}")
    local_pdf_path = None
    try:
        response = requests.get(doc_url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            local_pdf_path = tmp.name
        loader = PyMuPDFLoader(file_path=local_pdf_path)
        documents = loader.load()
    finally:
        if local_pdf_path and os.path.exists(local_pdf_path):
            os.unlink(local_pdf_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name, dimension=embedding_model.get_sentence_embedding_dimension(), metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    index = pc.Index(index_name)

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        embeddings = embedding_model.encode([c.page_content for c in batch_chunks]).tolist()
        metadata = [{"text": c.page_content, "source": c.metadata.get('source', 'Unknown'), "page": c.metadata.get('page', -1)} for c in batch_chunks]
        ids = [f"vec_{i+j}" for j in range(len(batch_chunks))]
        index.upsert(vectors=zip(ids, embeddings, metadata))

    print(f"Successfully indexed {len(chunks)} chunks.")
    return index

# --- Agent 2: Retrieval ---
def retrieve_and_rerank(index, question: str, embedding_model, cross_encoder, top_k: int = 5):
    """Fetches documents and re-ranks them for relevance."""
    print(f"Agent 2: Retrieving and re-ranking for question: '{question}'")
    query_embedding = embedding_model.encode(question).tolist()
    
    # FIX: Use keyword arguments for the query
    results = index.query(vector=query_embedding, top_k=top_k * 2, include_metadata=True)
    
    initial_chunks = [res['metadata']['text'] for res in results['matches']]
    pairs = [[question, chunk] for chunk in initial_chunks]
    scores = cross_encoder.predict(pairs)
    
    reranked_results = sorted(zip(scores, results['matches']), key=lambda x: x[0], reverse=True)
    return [match for score, match in reranked_results[:top_k]]

# --- Agent 3: Decision Engine ---
def run_decision_engine(question: str, retrieved_matches: list) -> str:
    """
    Uses Google's Gemini-Pro model to make a decision.
    """
    print("Agent 3: Running Decision Engine with Google Gemini...")

    DECISION_PROMPT = """
    You are an expert insurance policy adjudicator. Your task is to provide a clear and concise answer to the user's question based ONLY on the provided policy clauses.

    **Instructions:**
    1.  Read the user's question carefully.
    2.  Review the provided policy clauses to find the section that directly answers the question.
    3.  Construct a direct answer to the question using only information found in the clauses.
    4.  If the answer is not found in the provided clauses, state exactly that: "The answer is not found in the provided context."
    5.  Do not add any information that is not in the text. Be factual and precise.

    **Policy Clauses Provided:**
    ---
    {context}
    ---

    **User's Question:**
    ---
    {question}
    ---

    **Your Answer:**
    """
    
    # Initialize the Google Gemini Pro model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    prompt = ChatPromptTemplate.from_template(DECISION_PROMPT)
    
    context = "\n\n".join([f"Source: Page {m['metadata']['page']}\nText: {m['metadata']['text']}" for m in retrieved_matches])
    
    chain = prompt | llm | StrOutputParser()
    
    answer = chain.invoke({
        "question": question,
        "context": context
    })
    
    return answer

