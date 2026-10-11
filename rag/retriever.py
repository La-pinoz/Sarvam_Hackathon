from kb.database import FULL_KB_DICT

# 1. Load the Master Knowledge Base into memory once
MASTER_KB = FULL_KB_DICT

# rag/retriever.py


# CHANGED: replaces get_candidates
def get_all_candidates(crop: str) -> list[dict]:
    """
    Returns every disease record for the given crop from MASTER_KB — the complete
    set, not a similarity-narrowed subset. Disambiguation reasons over this full
    list itself (via DISAMBIGUATION_PROMPT + qa_history), so no vector retrieval
    or top-k cutoff happens here anymore.
    """
    crop_filter = crop.strip().capitalize(
    )  # kept same normalization as before ("rice" -> "Rice")
    return [
        record for record in MASTER_KB.values()
        if record.get("crop", "").strip() == crop_filter
    ]


def get_disease_record(disease_identifier: str) -> dict:
    """
    O(1) flat dictionary lookup for Phase 2 (Management & Follow-up).
    Resolves by disease_id or disease_name.
    """
    # Direct ID match
    if disease_identifier in MASTER_KB:
        return MASTER_KB[disease_identifier]

    # Name-based fallback match (case-insensitive)
    for record in MASTER_KB.values():
        if record.get("disease_name", "").lower() == disease_identifier.lower():
            return record

    return {}
