import os
import sys
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from rag_chain import answer_query
from ingest import build_vector_store, VECTOR_DB_DIR

app = FastAPI(title="AZON Counselling Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    try:
        if not os.path.exists(VECTOR_DB_DIR) or not os.listdir(VECTOR_DB_DIR):
            print("Building vector store...")
            build_vector_store()
            print("Vector store ready.")
    except Exception as e:
        print(f"Rebuilding vector store due to error: {e}")
        shutil.rmtree(VECTOR_DB_DIR, ignore_errors=True)
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        build_vector_store()


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def chat(request: ChatRequest):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured")
    try:
        result = answer_query(request.message, api_key)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")