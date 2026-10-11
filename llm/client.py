# llm/client.py
"""
Groq LLM client for FasalVachak.

Two tiers of completion:
  - get_fast_llm_completion   → crop classifier, intent classifier, symptom summariser
  - get_reasoning_completion  → differential diagnosis (disambiguation)
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from groq import AsyncGroq

logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Load .env from project root (two levels up from llm/client.py → llm/ → project root)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

groq_client = AsyncGroq(
    api_key=os.getenv("GROQ_API_KEY"),
)

# ── Model names ────────────────────────────────────────────────────────────────
# Both functions use the same model; split into two functions so we can easily
# switch to different models per tier later (e.g. a smaller fast model vs a
# larger reasoning model).
_FAST_MODEL = "openai/gpt-oss-120b"
_REASONING_MODEL = "openai/gpt-oss-120b"


async def get_fast_llm_completion(prompt: str, temperature: float = 0.0) -> str:
    """
    Used for Phase 0 (Crop Classifier), Phase 3 intent gate (YES/NO),
    and symptom summarisation.
    Optimised for minimum latency and small, decisive outputs.
    """
    response = await groq_client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt},
        ],
        model=_FAST_MODEL,
        temperature=temperature,
    )
    return response.choices[0].message.content


async def get_reasoning_completion(prompt: str, temperature: float = 0.2) -> str:
    """
    Used for Phase 1 (Differential Diagnosis / Disambiguation).
    Given a system prompt that grounds it as an agricultural assistant.
    """
    response = await groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a careful agricultural disease-diagnosis assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        model=_REASONING_MODEL,
        temperature=temperature,
    )
    return response.choices[0].message.content
