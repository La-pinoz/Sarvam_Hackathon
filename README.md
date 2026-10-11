# FasalVachak

## Description

FasalVachak is a Hindi voice assistant for diagnosing rice and cotton crop diseases. It connects farmers to a LiveKit voice agent that transcribes speech, reasons over a curated disease knowledge base, and responds with spoken guidance.

- **The Target User:** Indian farmers with limited tech literacy.
- **The Problem:** Text-based agricultural apps are inaccessible. They need a voice-native helpline to diagnose crop diseases.
- **Necessity of Voice:** If you remove voice, the product is completely unusable by the target demographic. Voice is the product.
- **Hard Voice Problem:** Multilingual / code-switched speech & controlled delivery. The agent must understand broken Hinglish/Hindi, navigate a strict decision tree without hallucinating, and speak back using natural, native-sounding Hindi (via Sarvam `bulbul:v3` / `priya`).
- **The Stress Case:** The user asks for information outside the knowledge base (e.g., medicine prices or an unsupported crop like sugarcane). The agent must gracefully decline rather than hallucinate, and Sarvam TTS must deliver this fallback naturally.

## Architecture diagram

```mermaid
flowchart LR
    Browser["static/index.html<br/>LiveKit browser client"]
    API["api.py<br/>FastAPI /connect"]
    LiveKit["LiveKit room"]
    Worker["server.py<br/>LiveKit agent"]
    STT["Sarvam STT<br/>saaras:v4"]
    Dialogue["dialogue/session.py<br/>DialogueManager"]
    LLM["llm/client.py<br/>Groq"]
    Retriever["rag/retriever.py"]
    KB["kb/database.py<br/>rice.json + cotton.json"]
    TTS["tts/sarvam_tts.py<br/>Sarvam TTS bulbul:v3"]

    Browser -->|"POST /connect"| API
    API -->|"JWT + agent dispatch"| LiveKit
    Browser -->|"microphone audio"| LiveKit
    LiveKit -->|"remote audio track"| Worker
    Worker -->|"audio frames"| STT
    STT -->|"Hindi transcript"| Dialogue
    Dialogue -->|"classification and reasoning"| LLM
    Dialogue -->|"candidate and disease lookup"| Retriever
    Retriever --> KB
    Dialogue -->|"Hindi response text"| TTS
    TTS -->|"audio frames"| LiveKit
    LiveKit -->|"agent audio"| Browser
```

[`static/index.html`](./static/index.html) requests a short-lived LiveKit token from [`api.py`](./api.py), then joins a unique room. [`server.py`](./server.py) receives the farmer’s audio, sends it through Sarvam STT and [`dialogue/session.py`](./dialogue/session.py), and publishes the Sarvam TTS response back to the browser. The dialogue manager uses [`llm/client.py`](./llm/client.py) and the in-memory knowledge base loaded by [`kb/database.py`](./kb/database.py).

## Interesting techniques

- **State-machine dialogue flow:** [`DialogueManager`](./dialogue/session.py) separates crop detection, symptom diagnosis, follow-up questions, and completed calls.
- **Closed-list diagnosis:** [`rag/retriever.py`](./rag/retriever.py) supplies the complete set of diseases for a crop instead of relying on an unrestricted model response.
- **Structured LLM outputs:** [`llm/prompts.py`](./llm/prompts.py) requires explicit `ANALYSIS`, `ACTION`, and `CONTENT` fields for diagnosis decisions.
- **Asynchronous audio streaming:** [`server.py`](./server.py) concurrently pushes LiveKit audio frames to speech recognition and consumes final transcripts.
- **Browser media handling:** [`static/index.html`](./static/index.html) uses [`MediaStream`](https://developer.mozilla.org/en-US/docs/Web/API/MediaStream) and [`HTMLMediaElement.play()`](https://developer.mozilla.org/en-US/docs/Web/API/HTMLMediaElement/play) to attach agent audio while respecting browser autoplay policies.
- **Short-lived access tokens:** [`api.py`](./api.py) creates ten-minute LiveKit JWTs scoped to microphone publishing and room subscription.

## Notable technologies and libraries

- [Python](https://www.python.org/) 3.12
- [FastAPI](https://fastapi.tiangolo.com/) and [Uvicorn](https://www.uvicorn.org/) for the token, dispatch, health, and static-file HTTP service
- [LiveKit](https://livekit.io/) and [livekit-agents](https://docs.livekit.io/agents/) for real-time voice rooms and agent workers
- [LiveKit JavaScript client](https://github.com/livekit/client-sdk-js), loaded from [jsDelivr](https://www.jsdelivr.com/), for browser room connections
- [Sarvam](https://www.sarvam.ai/) `saaras:v4` Hindi/Hinglish speech-to-text and `bulbul:v3` Hindi text-to-speech
- [Groq](https://groq.com/) with the `openai/gpt-oss-120b` model for crop classification, intent checks, symptom summarization, and diagnosis reasoning
- [ChromaDB](https://www.trychroma.com/) and [Sentence Transformers](https://www.sbert.net/) in [`rag/ingest.py`](./rag/ingest.py) for optional symptom-vector indexing
- [aiohttp](https://docs.aiohttp.org/) for asynchronous TTS HTTP sessions
- [python-dotenv](https://github.com/theskumar/python-dotenv) for environment configuration
- The browser UI uses the system [`Segoe UI`](https://learn.microsoft.com/en-us/typography/font-list/segoe-ui) font with a system fallback stack.

## Project structure

```text
.
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
├── api.py
├── requirements.txt
├── scratch_test.py
├── server.py
├── test_cli.py
├── test_out.txt
├── tingting.py
├── dialogue/
├── kb/
├── llm/
├── rag/
├── static/
├── stt/
└── tts/
```

- [`dialogue/`](./dialogue/) contains the conversation state machine.
- [`kb/`](./kb/) contains the crop disease data and loader.
- [`llm/`](./llm/) contains Groq client code and prompt templates.
- [`rag/`](./rag/) contains candidate retrieval and optional ChromaDB ingestion.
- [`static/`](./static/) contains the browser client served by [`api.py`](./api.py).
- [`stt/`](./stt/) contains the Sarvam speech-to-text integration.
- [`tts/`](./tts/) contains the Sarvam text-to-speech integration.
