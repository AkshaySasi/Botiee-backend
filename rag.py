import os
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-2.5-flash"

def setup_rag_chain():
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

    loaders = []
    if os.path.exists("resume.pdf"):
        loaders.append(PyPDFLoader("resume.pdf"))
    if os.path.exists("details.txt"):
        loaders.append(TextLoader("details.txt"))

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    if not docs:
        raise ValueError("No documents for RAG")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0.2,
        max_output_tokens=512,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Botiee ...\n\n{context}"),
        ("human", "{input}")
    ])

    chain_docs = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, chain_docs)
