from dotenv import load_dotenv
load_dotenv() 

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager 

from rag_chatbot import setup_rag_chain 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_storage = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup, load the RAG chain and store it
    logger.info("Application startup: Loading RAG chain...")
    try:
        model_storage["rag_chain"] = setup_rag_chain()
        logger.info("RAG chain loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load RAG chain on startup: {e}")
        model_storage["rag_chain"] = None
    
    yield # The application runs here
    
    # On shutdown, clear the storage
    logger.info("Application shutdown.")
    model_storage.clear()


app = FastAPI(
    title="Digital Me AI Portfolio", 
    version="1.0.0",
    lifespan=lifespan # <-- Tell FastAPI to use our lifespan function
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
    # Check if the chain loaded correctly
    rag_chain = model_storage.get("rag_chain")
    if rag_chain is None:
        logger.error("RAG chain is not available.")
        raise HTTPException(status_code=503, detail="Service not ready, RAG chain failed to load.")
        
    try:
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
    if model_storage.get("rag_chain") is not None:
        return {"status": "healthy"}
    else:
        return {"status": "unhealthy", "detail": "RAG chain failed to load."}
