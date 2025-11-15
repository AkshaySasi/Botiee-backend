import os
import logging
from typing import Dict, Any, List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------------------------
# API KEY SETUP
# -------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment.")
    raise ValueError("GEMINI_API_KEY not set.")

# LangChain's GoogleGenAI uses GOOGLE_API_KEY env var
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

MODEL_NAME = "gemini-2.5-flash"  # adjust if you use a different Gemini model


# -------------------------------------------------------------------
# DOCUMENT LOADING
# -------------------------------------------------------------------
def _load_documents() -> List[Any]:
    loaders = []

    if os.path.exists("resume.pdf"):
        logger.info("Found resume.pdf, loading...")
        loaders.append(PyPDFLoader("resume.pdf"))

    if os.path.exists("details.txt"):
        logger.info("Found details.txt, loading...")
        loaders.append(TextLoader("details.txt", encoding="utf-8"))

    if not loaders:
        raise ValueError(
            "No documents found. Place 'resume.pdf' or 'details.txt' in the project root."
        )

    docs: List[Any] = []
    for loader in loaders:
        try:
            docs.extend(loader.load())
        except Exception as e:
            logger.exception(f"Error loading file with {loader}: {e}")

    if not docs:
        raise ValueError("No documents could be loaded from the available files.")

    logger.info("Loaded %d raw documents.", len(docs))
    return docs


# -------------------------------------------------------------------
# VECTORSTORE + RETRIEVER
# -------------------------------------------------------------------
def _build_vectorstore(docs: List[Any]) -> FAISS:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
    )
    split_docs = text_splitter.split_documents(docs)
    logger.info("Split into %d chunks.", len(split_docs))

    # Use Google GenAI embeddings (no torch / sentence-transformers)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local("faiss_index")
    logger.info("FAISS index saved to 'faiss_index'.")
    return vectorstore


# -------------------------------------------------------------------
# SIMPLE RAG WRAPPER (no langchain.chains import)
# -------------------------------------------------------------------
class SimpleRAG:
    """
    Minimal RAG object with .invoke({"input": ...}) -> {"answer": ...}
    so FastAPI code can stay the same.
    """

    def __init__(self, retriever, llm: ChatGoogleGenerativeAI, prompt: ChatPromptTemplate):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        question = inputs.get("input", "").strip()
        if not question:
            return {"answer": "Please provide a non-empty question."}

        # Get relevant docs
        docs = self.retriever._get_relevant_documents(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Build messages from prompt
        messages = self.prompt.format_messages(
            context=context,
            input=question,
        )

        # Call Gemini chat model
        result = self.llm.invoke(messages)
        answer = getattr(result, "content", str(result))

        return {"answer": answer}


# -------------------------------------------------------------------
# PUBLIC FUNCTION USED BY app.py
# -------------------------------------------------------------------
def setup_rag_chain() -> SimpleRAG:
    try:
        docs = _load_documents()
        vectorstore = _build_vectorstore(docs)
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            temperature=0.2,
            max_output_tokens=512,
        )

        system_prompt = (
            "You are Botiee, Akshay Sasi's personal career chatbot / AI portfolio.\n"
            "Respond ONLY in third person, professionally, and engagingly, using "
            "EXCLUSIVELY the provided context from the resume and details about "
            "Akshay's education, projects, experiences, skills, hobbies, and thoughts.\n\n"
            "For HR questions (e.g., 'Why hire you?'), highlight his strengths, "
            "achievements, and fit based on the context.\n\n"
            "Keep responses detailed but concise (3–6 sentences), and conversational, "
            "as if it's an interview. Avoid very long paragraphs.\n\n"
            "If no relevant context is available, respond with exactly:\n"
            "\"I don’t have enough details to answer that fully, but I’m happy to discuss "
            "my education, projects, or skills!\"\n\n"
            "For queries unrelated to Akshay's portfolio (general knowledge, weather, etc.), "
            "respond ONLY with exactly:\n"
            "\"Sorry, I can only discuss about Akshay's professional portfolio.\"\n\n"
            "Do not speculate or fabricate information.\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        rag = SimpleRAG(retriever=retriever, llm=llm, prompt=prompt)
        logger.info("RAG pipeline initialized successfully.")
        return rag

    except Exception as e:
        logger.exception(f"RAG setup error: {e}")
        raise
