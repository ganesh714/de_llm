# ingestion_agent.py

import os
import requests
import tempfile
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def ingest_and_index_document(doc_url: str, index_name: str, embedding_model):
    """
    The Ingestion Agent for the DeLLM-X system.

    This agent is responsible for:
    1.  Robustly loading a PDF document from a URL.
    2.  Intelligently splitting the document into semantically meaningful chunks.
    3.  Creating and populating a Pinecone index with vector embeddings of these chunks.
    4.  Returning both the Pinecone index handle and the raw document chunks, which are
        required by the Retrieval Agent for building the BM25 keyword search index.

    Args:
        doc_url (str): The URL of the PDF document to ingest.
        index_name (str): The name for the Pinecone index.
        embedding_model: The pre-loaded SentenceTransformer model instance.

    Returns:
        tuple: A tuple containing the Pinecone index object and a list of LangChain Document objects (chunks).
    """
    print("INGESTION AGENT: Starting document ingestion pipeline...")

    # --- Step 1: Load Document with High-Fidelity Parser (FIXED FOR WINDOWS) ---
    print(f"Loading document from URL: {doc_url}")
    
    local_pdf_path = None
    try:
        # Download the file content first
        response = requests.get(doc_url)
        response.raise_for_status()

        # Create a temporary file to store the downloaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            local_pdf_path = tmp_file.name

        # Load the document from the safe, local file path
        loader = PyMuPDFLoader(file_path=local_pdf_path)
        documents = loader.load()
        print(f"Document loaded successfully. Contains {len(documents)} pages.")

    finally:
        # Ensure the temporary file is deleted after use
        if local_pdf_path and os.path.exists(local_pdf_path):
            os.unlink(local_pdf_path)


    # --- Step 2: Intelligent Chunking ---
    print("Splitting document into semantically relevant chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "SECTION", "PART", "CLAUSE", ". ", " "]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Document split into {len(chunks)} chunks.")

    # --- Step 3: Initialize Pinecone and Prepare Index ---
    print("Initializing Pinecone client...")
    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))

    if index_name in pc.list_indexes().names():
        print(f"Deleting existing Pinecone index: {index_name}...")
        pc.delete_index(index_name)

    print(f"Creating new serverless index: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=embedding_model.get_sentence_embedding_dimension(),
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    index = pc.Index(index_name)

    # --- Step 4: Batch Embedding and Upserting to Pinecone ---
    print(f"Embedding and upserting chunks to Pinecone in batches...")
    batch_size = 128
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_content = [c.page_content for c in batch]
        
        embeddings = embedding_model.encode(batch_content).tolist()
        
        vectors_to_upsert = []
        for j, doc in enumerate(batch):
            vector_id = f"vec_{i+j}"
            metadata = {
                "text": doc.page_content,
                "source": os.path.basename(doc.metadata.get('source', doc_url).split('?')[0]),
                "page": doc.metadata.get('page', 0) + 1
            }
            vectors_to_upsert.append((vector_id, embeddings[j], metadata))
        
        index.upsert(vectors=vectors_to_upsert, namespace="default")
        print(f"Upserted batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")

    print("INGESTION AGENT: Pipeline complete. Document is fully indexed.")

    # --- Step 5: Return Both Index and Chunks ---
    return (index, chunks)