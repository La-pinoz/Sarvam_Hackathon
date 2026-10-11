# rag/ingest.py
"""
ChromaDB ingestion script for FasalVachak.

Reads every disease record from FULL_KB_DICT and upserts it into a
persistent ChromaDB collection so that symptom-text similarity search
is available as a future retrieval option.

Run once (or whenever kb/*.json files are updated):
    cd FasalVachak
    python -m rag.ingest
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from kb.database import FULL_KB_DICT

# ChromaDB persistence directory — derived from this file's location so it
# works regardless of the working directory.
_RAG_DIR = Path(__file__).resolve().parent
_CHROMA_PATH = str(_RAG_DIR / "chroma_symptoms_db_raw")


def run_ingestion() -> None:
    master_kb = FULL_KB_DICT

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for disease_id, data in master_kb.items():
        bullets: list[str] = data.get("symptoms_bullets", [])

        # Flattened text is what gets embedded for similarity search
        documents.append(". ".join(b.rstrip(".") for b in bullets))

        metadatas.append(
            {
                "disease_id": disease_id,
                "crop": data.get("crop", "").capitalize(),
                "disease_name": data.get("disease_name", disease_id),
                # Chroma metadata can't store lists directly → JSON-encode
                "symptoms_bullets": json.dumps(bullets),
            }
        )
        ids.append(disease_id)

    client = chromadb.PersistentClient(path=_CHROMA_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name="agri_symptoms",
        embedding_function=embedding_func,
    )

    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print(f"[Ingest] Successfully indexed {len(ids)} diseases into ChromaDB at {_CHROMA_PATH}")


if __name__ == "__main__":
    run_ingestion()
