"""
=============================================================
 RAG Phase 2 — Chunking & Embedding
 Candidate: Gurudevi Lavanya Gopisetty
 Vector DB : ChromaDB (local, free)
 Embeddings: sentence-transformers (local, free)
=============================================================

SETUP (run once):
    pip install chromadb sentence-transformers

RUN:
    python phase2_chunk_and_embed.py

This script:
  1. Loads knowledge_base.json
  2. Converts each section into meaningful text chunks
  3. Generates embeddings using sentence-transformers
  4. Stores everything in a local ChromaDB collection
"""

import json
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
KNOWLEDGE_BASE_PATH = "knowledge_base.json"   # path to your JSON file
CHROMA_DB_PATH      = "./chroma_db"           # local folder where ChromaDB stores data
COLLECTION_NAME     = "lavanya_profile"
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"     # fast, accurate, free


# ─────────────────────────────────────────────
# STEP 1 — Load the knowledge base
# ─────────────────────────────────────────────
def load_knowledge_base(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# STEP 2 — Convert JSON sections into chunks
# Each chunk = {id, text, metadata}
# ─────────────────────────────────────────────
def build_chunks(kb: dict) -> list[dict]:
    chunks = []

    # ── Candidate Summary ──
    c = kb["candidate"]
    chunks.append({
        "id": "chunk_candidate_summary",
        "text": (
            f"{c['name']} is a {c['title']}. {c['summary']} "
            f"She is based in {c['location']}."
        ),
        "metadata": {"section": "summary", "type": "overview"}
    })

    # ── Education ──
    for edu in kb["education"]:
        specs = ", ".join(edu.get("specializations", []))
        text = (
            f"Lavanya completed her {edu['degree']} from {edu['institution']}, "
            f"{edu['location']} ({edu['duration']}). "
        )
        if edu.get("gpa"):
            text += f"GPA: {edu['gpa']}. "
        if specs:
            text += f"Subjects studied: {specs}. "
        if edu.get("role"):
            text += f"Role: {edu['role']}."

        chunks.append({
            "id": f"chunk_{edu['id']}",
            "text": text,
            "metadata": {"section": "education", "type": "degree", "institution": edu["institution"]}
        })

    # ── Experience ──
    for exp in kb["experience"]:
        responsibilities = " ".join(exp["responsibilities"])
        skills = ", ".join(exp["skills_demonstrated"])
        text = (
            f"Lavanya worked as {exp['title']} at {exp['company']} ({exp['location']}) "
            f"from {exp['duration']}. Responsibilities included: {responsibilities} "
            f"Skills used: {skills}."
        )
        chunks.append({
            "id": f"chunk_{exp['id']}",
            "text": text,
            "metadata": {
                "section": "experience",
                "type": exp["type"],
                "company": exp["company"],
                "title": exp["title"]
            }
        })

    # ── Projects ──
    for proj in kb["projects"]:
        text = (
            f"Project: {proj['title']} | Domain: {proj['domain']}. "
            f"Technologies used: {', '.join(proj['technologies'])}. "
            f"Description: {proj['description']} "
            f"Key outcome: {proj['key_outcome']}."
        )
        if proj.get("architecture"):
            text += f" Architecture: {proj['architecture']}"
        chunks.append({
            "id": f"chunk_{proj['id']}",
            "text": text,
            "metadata": {
                "section": "projects",
                "domain": proj["domain"],
                "title": proj["title"],
                "technologies": ", ".join(proj["technologies"])
            }
        })

    # ── Skills (one chunk per category) ──
    skills = kb["skills"]
    skill_map = {
        "Languages":            f"Expert: {', '.join(skills['languages']['expert'])}. Proficient: {', '.join(skills['languages']['proficient'])}.",
        "ML & AI Frameworks":   ", ".join(skills["ml_ai_frameworks"]),
        "Generative AI & LLMs": ", ".join(skills["generative_ai_llm"]),
        "Computer Vision":      ", ".join(skills["computer_vision"]),
        "Data Analytics":       ", ".join(skills["data_analytics"]),
        "Data Engineering":     ", ".join(skills["data_engineering"]),
        "BI & Visualization":   ", ".join(skills["bi_visualization"]),
        "DevOps & Cloud":       ", ".join(skills["devops_cloud"]),
        "MLOps":                ", ".join(skills["mlops"]),
        "Databases":            ", ".join(skills["databases"]),
        "Annotation Tools":     ", ".join(skills["annotation_data_tools"]),
    }
    for category, skill_text in skill_map.items():
        chunks.append({
            "id": f"chunk_skills_{category.lower().replace(' ', '_').replace('&', 'and')}",
            "text": f"Lavanya has knowledge and experience in {category}: {skill_text}",
            "metadata": {"section": "skills", "category": category}
        })

    # ── Certifications ──
    cert_list = [
        f"{cert['name']} ({cert['status']})"
        for cert in kb["certifications"]
    ]
    chunks.append({
        "id": "chunk_certifications",
        "text": f"Lavanya's certifications include: {', '.join(cert_list)}.",
        "metadata": {"section": "certifications"}
    })

    # ── Domain Expertise ──
    for domain in kb["domain_expertise"]:
        chunks.append({
            "id": f"chunk_domain_{domain['domain'].lower().replace(' ', '_').replace('&', 'and')}",
            "text": (
                f"Domain expertise — {domain['domain']}: {domain['summary']} "
                f"Experience: {domain['years_of_experience']}."
            ),
            "metadata": {
                "section": "domain_expertise",
                "domain": domain["domain"],
                "experience": domain["years_of_experience"]
            }
        })

    # ── Pre-built Q&A pairs (boost retrieval for common HR questions) ──
    for i, qa in enumerate(kb["qa_pairs"]):
        chunks.append({
            "id": f"chunk_qa_{i:03d}",
            "text": f"Q: {qa['question']} A: {qa['answer']}",
            "metadata": {"section": "qa_pairs", "question": qa["question"]}
        })

    # ── Job Titles ──
    titles = ", ".join(kb["job_titles_applicable"])
    chunks.append({
        "id": "chunk_job_titles",
        "text": f"Lavanya is qualified and suitable for the following job roles: {titles}.",
        "metadata": {"section": "job_titles"}
    })

    return chunks


# ─────────────────────────────────────────────
# STEP 3 — Embed & Store in ChromaDB
# ─────────────────────────────────────────────
def embed_and_store(chunks: list[dict]):
    print(f"\n📦 Loading embedding model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"🗄️  Connecting to ChromaDB at: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Delete existing collection if re-running
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"🔄 Existing collection '{COLLECTION_NAME}' cleared.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}   # cosine similarity for semantic search
    )

    print(f"\n⚡ Generating embeddings for {len(chunks)} chunks ...")
    texts     = [chunk["text"]     for chunk in chunks]
    ids       = [chunk["id"]       for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    print(f"\n💾 Storing chunks in ChromaDB collection: '{COLLECTION_NAME}' ...")
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"\n✅ Done! {collection.count()} chunks stored in ChromaDB.")
    return collection


# ─────────────────────────────────────────────
# STEP 4 — Quick Test Query
# ─────────────────────────────────────────────
def test_query(collection, model):
    test_questions = [
        "Does she know Python?",
        "Has she worked with Docker and Kubernetes?",
        "What projects has she done in computer vision?",
        "Does she have industry experience?",
        "Can she build RAG pipelines?"
    ]

    print("\n" + "="*60)
    print("🧪 RUNNING TEST QUERIES")
    print("="*60)

    for question in test_questions:
        query_embedding = model.encode([question]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=1
        )
        top_chunk = results["documents"][0][0]
        top_meta  = results["metadatas"][0][0]
        score     = results["distances"][0][0]

        print(f"\n❓ {question}")
        print(f"   📌 Section : {top_meta.get('section', 'N/A')}")
        print(f"   📝 Answer  : {top_chunk[:200]}...")
        print(f"   🎯 Score   : {1 - score:.2f} (cosine similarity)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("  RAG PHASE 2 — CHUNKING & EMBEDDING")
    print("  Candidate: Gurudevi Lavanya Gopisetty")
    print("="*60)

    # 1. Load
    print(f"\n📂 Loading knowledge base from: {KNOWLEDGE_BASE_PATH}")
    kb = load_knowledge_base(KNOWLEDGE_BASE_PATH)

    # 2. Chunk
    chunks = build_chunks(kb)
    print(f"✂️  Created {len(chunks)} semantic chunks")

    # Preview chunks
    print("\n📋 Chunk preview:")
    for chunk in chunks[:3]:
        print(f"   [{chunk['id']}] {chunk['text'][:100]}...")

    # 3. Embed & Store
    collection = embed_and_store(chunks)

    # 4. Test
    model = SentenceTransformer(EMBEDDING_MODEL)
    test_query(collection, model)

    print("\n" + "="*60)
    print("✅ PHASE 2 COMPLETE!")
    print(f"   ChromaDB saved at: {CHROMA_DB_PATH}/")
    print("   Ready for Phase 3 → RAG Query Pipeline")
    print("="*60)