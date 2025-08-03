# reasoning_agent.py
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

def run_reasoning_pipeline(question: str, retrieved_docs: list, models: 'ModelLoader') -> str:
    """The two-stage 'Draft and Verify' reasoning pipeline for maximum accuracy."""
    
    # === STAGE 1: DRAFTER AGENT ===
    print("Reasoning Engine Stage 1: Drafting initial answer and finding quotes...")
    
    drafter_prompt_template = """
    You are a meticulous junior analyst. Your task is to analyze the provided Policy Clauses and draft an answer to the user's question.
    You must find the exact quotes from the text that support your answer.
    
    **Policy Clauses:**
    {context}
    
    **User's Question:**
    {question}

    Respond with a JSON object containing two keys:
    1. "draft_answer": Your best attempt at answering the question based on the text.
    2. "supporting_quotes": A list of DIRECT, VERBATIM quotes from the policy clauses that prove your answer.
    """
    
    class Draft(JsonOutputParser):
        def parse(self, text: str):
            # Custom parser to handle potential LLM formatting quirks
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"draft_answer": "Error in generating draft.", "supporting_quotes": []}

    drafter_prompt = ChatPromptTemplate.from_template(drafter_prompt_template)
    drafter_chain = drafter_prompt | models.main_llm | Draft()
    
    context_string = "\n\n".join([doc['page_content'] for doc in retrieved_docs])
    draft_response = drafter_chain.invoke({"question": question, "context": context_string})

    if not draft_response or not draft_response['supporting_quotes']:
        return "Could not find a conclusive answer in the document."

    print(f"Draft Answer: {draft_response['draft_answer']}")
    
    # === STAGE 2: VALIDATOR AGENT (THE SELF-CORRECTION LOOP) ===
    print("Reasoning Engine Stage 2: Verifying draft answer against quotes...")
    
    validator_prompt_template = """
    You are a skeptical senior auditor. Your only goal is to ensure 100% factual accuracy.
    You are given a list of direct quotes from a document and a proposed answer.
    Your task is to validate if the proposed answer is FULLY and EXCLUSIVELY supported by the given quotes.

    **Supporting Quotes:**
    {quotes}
    
    **Proposed Answer to Verify:**
    "{draft_answer}"

    1. **Verification:** Is the Proposed Answer factually correct based ONLY on the quotes? Answer "Yes" or "No".
    2. **Reasoning:** Briefly explain your reasoning.
    3. **Final Answer:** If the verification is "Yes", refine the Proposed Answer for clarity and conciseness. If "No", construct the correct answer using only the provided quotes.
    """
    
    validator_prompt = ChatPromptTemplate.from_template(validator_prompt_template)
    validator_chain = validator_prompt | models.main_llm | StrOutputParser()
    
    final_answer = validator_chain.invoke({
        "quotes": "\n---\n".join(draft_response['supporting_quotes']),
        "draft_answer": draft_response['draft_answer']
    })
    
    return final_answer