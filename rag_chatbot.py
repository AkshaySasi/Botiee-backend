import os
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # ← v0.2.x
from langchain_community.embeddings import HuggingFaceEmbeddings   # ← v0.2.x
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain              # ← v0.2.x ✅
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- API Key and Model Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in .env file.")
    raise ValueError("GEMINI_API_KEY not set.")

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
MODEL_NAME = "gemini-2.5-flash" 

def setup_rag_chain():
    try:
        # Load documents
        loaders = []
        if os.path.exists("resume.pdf"):
            loaders.append(PyPDFLoader("resume.pdf"))
        if os.path.exists("details.txt"):
            loaders.append(TextLoader("details.txt"))
        if not loaders:
            raise ValueError("No documents found. Ensure resume.pdf or details.txt exist.")
        
        docs = []
        for loader in loaders:
            try:
                docs.extend(loader.load())
            except Exception as e:
                logger.error(f"Error loading file: {e}")
        
        if not docs:
            raise ValueError("No documents loaded.")

        # Split for better context
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        split_docs = text_splitter.split_documents(docs)
        logger.info(f"Split into {len(split_docs)} chunks.")

        # FAISS vector store with local embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        vectorstore = FAISS.from_documents(documents=split_docs, embedding=embeddings)
        vectorstore.save_local("faiss_index")
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # LLM with Google AI Studio (Gemini)
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            temperature=0.2,
            max_output_tokens=512
        )

        # Strict prompt with fallback
        system_prompt = (
            "You are Botiee, Akshay Sasi's personal career chatbot/ AI portfolio."
            "Respond ONLY in third person, professionally, and engagingly, using EXCLUSIVELY the provided context from the provided resume and details about Akshay's education, projects, experiences, skills, hobbies, and thoughts. "
            "For HR questions (e.g., 'Why hire you?'), highlight my strengths, achievements, and fit based on my data. "
            "Keep responses detailed, concise (3-6 sentences), and conversational, as if I'm in an interview. "
            "If no relevant context is available, respond with: 'I don't have enough details to answer that fully, but I'm happy to discuss my education, projects, or skills!' "
            "For queries unrelated to my portfolio (e.g., general knowledge, weather, or anything not in context), respond ONLY with: 'Sorry, I can only discuss about Akshay's professional portfolio.' "
            "Do not speculate, fabricate, or use external info. And don't reply in very long paragraphs, make it brief, like a real conversation"
            "\n\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        logger.info("RAG chain setup complete with Google AI Studio.")
        return rag_chain
        
    except Exception as e:
        logger.error(f"RAG setup error: {e}")
        raise
