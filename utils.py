# utils.py (Final Golden Version with LangChain Embeddings Wrapper)
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings # <-- IMPORT THE WRAPPER

# Load environment variables
load_dotenv()

class ModelLoader:
    def __init__(self):
        """Initializes all required models and LangChain wrappers once for efficiency."""
        print("PERFORMANCE CORE: Initializing all models and wrappers...")

        # --- LLMs ---
        self.main_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.fast_llm = ChatGroq(
            model_name="llama3-8b-8192",
            temperature=0.0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        # --- LOCAL MODELS & WRAPPER (THE FIX) ---
        model_name = 'all-MiniLM-L6-v2'
        # Raw model for direct encoding (used in ingestion)
        self.embedding_model = SentenceTransformer(model_name)
        # LangChain wrapper for retriever integration
        self.lc_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'} # Ensure it runs on CPU
        )
        # --- END OF FIX ---
        
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("All models and wrappers loaded successfully.")

# A single instance of this class is imported by other modules.
MODELS = ModelLoader()