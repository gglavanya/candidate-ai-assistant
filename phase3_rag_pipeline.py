"""
=============================================================
 RAG Phase 3 — Query Pipeline (Retriever + LLM)
 Candidate : Gurudevi Lavanya Gopisetty
 Retriever : ChromaDB (local)
 Embeddings: sentence-transformers (local)
 LLM       : Groq (free, fast — llama3-8b-8192)
=============================================================

SETUP (run once):
    pip install chromadb sentence-transformers groq python-dotenv

CREATE a .env file in the same folder:
    GROQ_API_KEY=your_groq_api_key_here

RUN:
    python3 phase3_rag_pipeline.py
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
load_dotenv()

CHROMA_DB_PATH   = "./chroma_db"
COLLECTION_NAME  = "lavanya_profile"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
GROQ_MODEL       = "llama-3.3-70b-versatile"       # free, fast Groq model
TOP_K            = 5                      # number of chunks to retrieve per query


# ─────────────────────────────────────────────
# SYSTEM PROMPT — defines how the RAG responds
# ─────────────────────────────────────────────
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
- Dont start with "Based on the retrieved context..." or similar phrases. Just answer directly
- If the context doesn't contain enough information, say: 
  "Based on available information, I don't have enough detail to answer that specifically."

"""
   

# ─────────────────────────────────────────────
# STEP 1 — Load ChromaDB + Embedding Model
# ─────────────────────────────────────────────
def load_retriever():
    print("📦 Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"🗄️  Connecting to ChromaDB at: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    print(f"✅ ChromaDB loaded — {collection.count()} chunks available\n")
    return model, collection


# ─────────────────────────────────────────────
# STEP 2 — Retrieve relevant chunks
# ─────────────────────────────────────────────
def retrieve(question: str, model, collection) -> tuple[str, list]:
    query_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K
    )

    chunks    = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Build context string from top chunks
    context_parts = []
    for i, (chunk, meta, dist) in enumerate(zip(chunks, metadatas, distances)):
        similarity = round(1 - dist, 3)
        context_parts.append(
            f"[Chunk {i+1} | Section: {meta.get('section','N/A')} | Score: {similarity}]\n{chunk}"
        )

    context = "\n\n".join(context_parts)
    return context, metadatas


# ─────────────────────────────────────────────
# STEP 3 — Generate answer using Groq LLM
# ─────────────────────────────────────────────
def generate_answer(question: str, context: str, client: Groq) -> str:
    user_message = f"""
Context about Lavanya:
{context}

Question: {question}

Answer based strictly on the context above:
"""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ],
        temperature=0.3,      # low temp = more factual, less creative
        max_tokens=512
    )

    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# STEP 4 — Full RAG Pipeline (retrieve + generate)
# ─────────────────────────────────────────────
def ask(question: str, model, collection, groq_client) -> str:
    # Retrieve
    context, metadatas = retrieve(question, model, collection)

    # Generate
    answer = generate_answer(question, context, groq_client)

    return answer, context, metadatas


# ─────────────────────────────────────────────
# STEP 5 — Interactive CLI Chat
# ─────────────────────────────────────────────
def run_interactive_chat(model, collection, groq_client):
    print("\n" + "="*60)
    print("  💬 LAVANYA's AI ASSISTANT — RAG PIPELINE")
    print("  Ask any question about Lavanya's background")
    print("  Type 'quit' to exit | 'debug' to see retrieved chunks")
    print("="*60 + "\n")

    debug_mode = False

    while True:
        question = input("❓ Your question: ").strip()

        if not question:
            continue
        if question.lower() == "quit":
            print("\n👋 Goodbye!")
            break
        if question.lower() == "debug":
            debug_mode = not debug_mode
            print(f"🔧 Debug mode: {'ON' if debug_mode else 'OFF'}\n")
            continue

        print("\n🔍 Retrieving relevant context...")
        answer, context, metadatas = ask(question, model, collection, groq_client)

        if debug_mode:
            print("\n─── Retrieved Chunks ───")
            print(context)
            print("────────────────────────\n")

        print(f"\n🤖 Answer:\n{answer}\n")
        print("─" * 60 + "\n")


# ─────────────────────────────────────────────
# STEP 6 — Run sample HR questions automatically
# ─────────────────────────────────────────────
def run_sample_questions(model, collection, groq_client):
    hr_questions = [
        "Does Lavanya have experience with machine learning?",
        "Can she work with Docker and Kubernetes?",
        "What generative AI projects has she built?",
        "Does she have industry work experience?",
        "Is she familiar with cloud platforms like AWS?",
        "What is her educational background?",
        "Can she build and deploy RAG pipelines?",
        "What computer vision work has she done?"
    ]

    print("\n" + "="*60)
    print("  🧪 RUNNING SAMPLE HR QUESTIONS")
    print("="*60)

    for question in hr_questions:
        print(f"\n❓ {question}")
        answer, _, _ = ask(question, model, collection, groq_client)
        print(f"🤖 {answer}")
        print("─" * 60)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("  RAG PHASE 3 — QUERY PIPELINE")
    print("  Candidate: Gurudevi Lavanya Gopisetty")
    print("="*60)

    # Validate API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("\n❌ ERROR: GROQ_API_KEY not found!")
        print("   Create a .env file with: GROQ_API_KEY=your_key_here")
        exit(1)

    # Load retriever
    model, collection = load_retriever()

    # Init Groq client
    groq_client = Groq(api_key=groq_api_key)
    print(f"⚡ Groq LLM ready: {GROQ_MODEL}\n")

    # Choose mode
    print("Choose mode:")
    print("  1 → Run sample HR questions (automated demo)")
    print("  2 → Interactive chat (ask your own questions)")
    choice = input("\nEnter 1 or 2: ").strip()

    if choice == "1":
        run_sample_questions(model, collection, groq_client)
        print("\n✅ Sample run complete! Run again and choose 2 for interactive mode.")
    else:
        run_interactive_chat(model, collection, groq_client)