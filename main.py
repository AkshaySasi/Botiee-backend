from dotenv import load_dotenv
load_dotenv()

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rag_chatbot import setup_rag_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy in-memory model storage
model_storage = {"rag_chain": None}

app = FastAPI(
    title="Digital Me AI Portfolio",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Handles chat requests. Loads RAG chain lazily on first request to reduce memory usage.
    """
    try:
        # ✅ Lazy loading — load only when needed
        if model_storage["rag_chain"] is None:
            logger.info("Loading RAG chain for the first time (lazy init)...")
            model_storage["rag_chain"] = setup_rag_chain()
            logger.info("RAG chain loaded successfully (cached).")

        rag_chain = model_storage["rag_chain"]

        logger.info(f"Query: {request.message}")
        response = rag_chain.invoke({"input": request.message})

        if "answer" not in response:
            raise ValueError("Invalid response from chain")

        return JSONResponse({"response": response["answer"]})

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health():
    """
    Health check endpoint.
    """
    if model_storage.get("rag_chain") is not None:
        return {"status": "healthy"}
    else:
        return {"status": "initializing", "detail": "RAG chain will load on first request."}
