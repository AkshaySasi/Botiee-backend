from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from rag import setup_rag_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Botiee API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_chain = None

@app.on_event("startup")
async def load_model():
    global rag_chain
    try:
        rag_chain = setup_rag_chain()
        logger.info("RAG chain loaded")
    except Exception as e:
        logger.error(f"RAG load failed: {e}")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    global rag_chain
    if not rag_chain:
        raise HTTPException(status_code=503, detail="Bot not ready")
    try:
        response = rag_chain.invoke({"input": req.message})
        return {"reply": response["answer"]}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500)
