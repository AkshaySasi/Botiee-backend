from dotenv import load_dotenv
load_dotenv()

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rag import setup_rag_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_storage = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Application startup: loading RAG chain...")
    try:
        model_storage["rag_chain"] = setup_rag_chain()
        logger.info("RAG chain loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load RAG chain on startup: {e}")
        model_storage["rag_chain"] = None

    yield

    # Shutdown
    logger.info("Application shutdown: clearing model storage.")
    model_storage.clear()


app = FastAPI(
    title="Digital Me AI Portfolio",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # In production, you can restrict this to your portfolio domain:
    # allow_origins=["https://akshaysasi.github.io"],
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(request: ChatRequest):
    rag_chain = model_storage.get("rag_chain")
    if rag_chain is None:
        logger.error("RAG chain is not available.")
        raise HTTPException(
            status_code=503,
            detail="Service not ready, RAG chain failed to load.",
        )

    try:
        logger.info(f"Received query: {request.message}")
        result = rag_chain.invoke({"input": request.message})
        answer = result.get("answer", "Sorry, I could not generate a response.")
        return JSONResponse({"response": answer})
    except Exception as e:
        logger.exception(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health():
    if model_storage.get("rag_chain") is not None:
        return {"status": "healthy"}
    else:
        return {"status": "unhealthy", "detail": "RAG chain failed to load."}
