"""
=============================================================
 RAG Phase 4 — FastAPI Layer
 Candidate : Gurudevi Lavanya Gopisetty
 Framework : FastAPI
 Retriever : ChromaDB (local)
 Embeddings: sentence-transformers (local)
 LLM       : Groq (llama-3.3-70b-versatile)
=============================================================

SETUP (run once):
    pip install fastapi uvicorn chromadb sentence-transformers groq python-dotenv

RUN:
    uvicorn phase4_api:app --reload --port 8000

ENDPOINTS:
    GET  /              → welcome message
    GET  /health        → health check
    GET  /about         → Lavanya's profile summary
    POST /ask           → ask a question about Lavanya
    GET  /sample        → see sample HR questions & answers
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
load_dotenv()

CHROMA_DB_PATH  = "./chroma_db"
COLLECTION_NAME = "lavanya_profile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL      = "llama-3.3-70b-versatile"
TOP_K           = 4

SYSTEM_PROMPT = """
You are an AI assistant representing Gurudevi Lavanya Gopisetty, a recent MS Computer Science 
graduate (GPA 3.9) from California State University, Dominguez Hills.

Your job is to answer questions from HR professionals, recruiters, or anyone asking about 
Lavanya's skills, experience, education, and projects.

Rules:
- Only answer based on the context provided. Do not make up information.
- Be confident and professional — you are advocating for Lavanya.
- If asked "can she do X?", check the context and answer clearly YES or NO with evidence.
- Always mention specific projects, metrics, or experience that backs up your answer.
- Keep responses concise but complete (3–5 sentences max unless detail is needed).
- If the context doesn't contain enough information, say: 
  "Based on available information, I don't have enough detail to answer that specifically."
"""

# ─────────────────────────────────────────────
# GLOBAL STATE (loaded once at startup)
# ─────────────────────────────────────────────
state = {
    "model":       None,
    "collection":  None,
    "groq_client": None,
    "ready":       False,
    "startup_time": None
}


# ─────────────────────────────────────────────
# LIFESPAN — loads everything once on startup
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 Starting Lavanya's RAG API...")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("❌ GROQ_API_KEY not found in .env file!")

    print("📦 Loading embedding model...")
    state["model"] = SentenceTransformer(EMBEDDING_MODEL)

    print(f"🗄️  Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    state["collection"] = client.get_collection(COLLECTION_NAME)

    print(f"⚡ Connecting to Groq ({GROQ_MODEL})...")
    state["groq_client"] = Groq(api_key=groq_api_key)

    state["ready"] = True
    state["startup_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

    chunk_count = state["collection"].count()
    print(f"✅ API ready! {chunk_count} chunks loaded in ChromaDB.\n")

    yield  # app runs here

    print("👋 Shutting down RAG API...")


# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Lavanya's AI Assistant",
    description="Ask any question about Gurudevi Lavanya Gopisetty — skills, experience, projects, and more.",
    version="1.0.0",
    lifespan=lifespan
)

# Allow frontend (React/HTML) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = TOP_K
    debug: Optional[bool] = False

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Does Lavanya have experience with RAG pipelines?",
                "top_k": 4,
                "debug": False
            }
        }

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sections_used: list[str]
    response_time_ms: float
    debug_chunks: Optional[list[str]] = None


# ─────────────────────────────────────────────
# HELPER — RAG Pipeline
# ─────────────────────────────────────────────
def retrieve(question: str, top_k: int):
    query_embedding = state["model"].encode([question]).tolist()
    results = state["collection"].query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    chunks    = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return chunks, metadatas, distances


def generate_answer(question: str, context: str) -> str:
    user_message = f"""
Context about Lavanya:
{context}

Question: {question}

Answer based strictly on the context above:
"""
    response = state["groq_client"].chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ],
        temperature=0.3,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    return {
        "message": "👋 Welcome to Lavanya's AI Assistant!",
        "description": "Ask me anything about Gurudevi Lavanya Gopisetty.",
        "endpoints": {
            "ask":    "POST /ask",
            "about":  "GET  /about",
            "health": "GET  /health",
            "sample": "GET  /sample",
            "docs":   "GET  /docs"
        }
    }


@app.get("/health", tags=["General"])
def health():
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="API is not ready yet.")
    return {
        "status":       "healthy ✅",
        "model":        GROQ_MODEL,
        "embeddings":   EMBEDDING_MODEL,
        "chunks_loaded": state["collection"].count(),
        "startup_time": state["startup_time"]
    }


@app.get("/about", tags=["General"])
def about():
    return {
        "name":       "Gurudevi Lavanya Gopisetty",
        "title":      "MS Computer Science Graduate | ML & AI Engineer",
        "education":  "MS Computer Science — CSUDH, GPA 3.9 (Dec 2025)",
        "experience": "2+ years at Cognizant (Data Analyst → Data Scientist)",
        "domains": [
            "Machine Learning & Data Science",
            "Computer Vision",
            "Generative AI & LLM Engineering",
            "Data Analytics & Business Intelligence",
            "DevOps & Cloud Engineering"
        ],
        "top_skills": [
            "Python", "PyTorch", "TensorFlow", "LangChain", "LangGraph",
            "RAG Pipelines", "Docker", "Kubernetes", "Terraform", "AWS",
            "SQL", "Power BI", "Tableau", "MLflow", "ChromaDB", "Groq"
        ],
        "contact": "Available upon request"
    }


@app.post("/ask", response_model=QuestionResponse, tags=["RAG"])
def ask_question(request: QuestionRequest):
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="API is not ready yet.")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start_time = time.time()

    # Retrieve
    chunks, metadatas, distances = retrieve(request.question, request.top_k)

    # Build context
    context_parts = []
    for chunk, meta, dist in zip(chunks, metadatas, distances):
        score = round(1 - dist, 3)
        context_parts.append(
            f"[Section: {meta.get('section','N/A')} | Score: {score}]\n{chunk}"
        )
    context = "\n\n".join(context_parts)

    # Generate answer
    answer = generate_answer(request.question, context)

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    # Sections used (for transparency)
    sections_used = list({m.get("section", "N/A") for m in metadatas})

    return QuestionResponse(
        question=request.question,
        answer=answer,
        sections_used=sections_used,
        response_time_ms=elapsed_ms,
        debug_chunks=chunks if request.debug else None
    )


@app.get("/sample", tags=["RAG"])
def sample_questions():
    """Returns a list of sample HR questions you can try."""
    return {
        "sample_questions": [
            "Does Lavanya have experience with machine learning?",
            "Can she build and deploy RAG pipelines?",
            "What is her educational background?",
            "Has she worked in industry before?",
            "What DevOps tools does she know?",
            "Does she have computer vision project experience?",
            "Can she work with AWS cloud services?",
            "What generative AI projects has she built?",
            "Is she familiar with Docker and Kubernetes?",
            "What is her strongest programming language?"
        ],
        "how_to_use": "Send any of these as a POST /ask request with JSON body: {\"question\": \"...\"}",
        "tip": "Add \"debug\": true to see the retrieved chunks used to generate the answer."
    }