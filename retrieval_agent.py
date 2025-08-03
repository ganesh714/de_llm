# retrieval_agent.py (Final Golden Version)
import json
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def expand_query(question: str, fast_llm):
    """Uses a fast LLM to generate alternative versions of the user's question."""
    print("Retrieval Agent Stage 1: Expanding query with fast LLM...")
    prompt = ChatPromptTemplate.from_template(
        "Generate 3 alternative versions of the following question to improve document retrieval. "
        "Focus on synonyms and rephrasing key terms. Output should be a JSON list of strings.\n\n"
        "Original Question: {question}\n\n"
        "JSON List Output:"
    )
    chain = prompt | fast_llm | StrOutputParser()
    response = chain.invoke({"question": question})
    try:
        queries = json.loads(response)
        if question not in queries:
            queries.insert(0, question)
        print(f"Expanded Queries: {queries}")
        return queries
    except json.JSONDecodeError:
        return [question]

def hybrid_search_and_rerank(index_name: str, chunks: list, question: str, models: 'ModelLoader', top_k: int = 5) -> list:
    """
    Performs the advanced retrieval pipeline using the correct LangChain embeddings wrapper.
    """
    expanded_queries = expand_query(question, models.fast_llm)

    doc_texts = [doc.page_content for doc in chunks]
    bm25_retriever = BM25Retriever.from_texts(doc_texts)
    bm25_retriever.k = top_k * 3

    # --- THIS IS THE KEY FIX ---
    # We now pass the LangChain 'Embeddings' object, not a raw function.
    vectorstore = Pinecone.from_existing_index(
        index_name=index_name,
        embedding=models.lc_embeddings, # Use the LangChain wrapper object
        text_key="text",
        namespace="default"
    )
    # --- END OF FIX ---
    
    pinecone_retriever = vectorstore.as_retriever(search_kwargs={'k': top_k * 3})

    print("Retrieval Agent Stage 2: Performing Hybrid Search...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, pinecone_retriever],
        weights=[0.5, 0.5]
    )

    all_docs = []
    for q in expanded_queries:
        all_docs.extend(ensemble_retriever.invoke(q))

    unique_docs = {doc.page_content: doc for doc in all_docs}.values()
    print(f"Found {len(unique_docs)} unique candidate documents.")
    
    if not unique_docs:
        return []

    print("Retrieval Agent Stage 3: Re-ranking candidates...")
    pairs = [[question, doc.page_content] for doc in unique_docs]
    scores = models.cross_encoder.predict(pairs)
    
    reranked_results = sorted(zip(scores, unique_docs), key=lambda x: x[0], reverse=True)
    
    final_docs = []
    for score, doc in reranked_results[:top_k]:
        final_docs.append({
            'page_content': doc.page_content,
            'metadata': doc.metadata if hasattr(doc, 'metadata') else {}
        })

    return final_docs