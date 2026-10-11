# tts/sarvam_tts.py
"""
Sarvam TTS wrapper for FasalVachak (bulbul:v3).
"""

from __future__ import annotations

import base64
import io
import os
import wave
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from livekit import rtc

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

_SAMPLE_RATE = 24000
_NUM_CHANNELS = 1
_TTS_URL = "https://api.sarvam.ai/text-to-speech"
_MAX_CHARS = 2500
_FRAME_MS = 20


def _require_api_key() -> str:
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key or api_key == "your_sarvam_api_key_here":
        raise ValueError(
            "\n\n>>> ❌ ERROR: SARVAM_API_KEY is missing! <<<\n"
            "Get a key from https://dashboard.sarvam.ai/login and set "
            "SARVAM_API_KEY in your .env file.\n\n"
        )
    return api_key


class SarvamTTS:
    """HTTP client for Sarvam bulbul:v3 text-to-speech."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        *,
        api_key: str | None = None,
        speaker: str = "priya",
        model: str = "bulbul:v3",
        language_code: str = "hi-IN",
        sample_rate: int = _SAMPLE_RATE,
    ) -> None:
        self._session = session
        self._api_key = api_key or _require_api_key()
        self.speaker = speaker
        self.model = model
        self.language_code = language_code
        self.sample_rate = sample_rate

    async def synthesize_wav(self, text: str) -> bytes:
        payload = {
            "text": text,
            "target_language_code": self.language_code,
            "speaker": self.speaker,
            "model": self.model,
            "speech_sample_rate": self.sample_rate,
            "enable_preprocessing": True,
        }
        async with self._session.post(
            _TTS_URL,
            headers={"api-subscription-key": self._api_key},
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            body = await resp.json(content_type=None)
            if resp.status >= 400:
                raise RuntimeError(f"Sarvam TTS HTTP {resp.status}: {body}")
        audios = body.get("audios") or []
        if not audios:
            raise RuntimeError(f"Sarvam TTS returned no audio: {body}")
        return base64.b64decode(audios[0])


def create_tts_client() -> tuple[SarvamTTS, aiohttp.ClientSession]:
    """
    Create a fresh Sarvam TTS client + its own aiohttp session.
    Call this once at the start of each job (inside entrypoint).
    """
    api_key = _require_api_key()  # fail before opening a session we'd leak
    session = aiohttp.ClientSession()
    return SarvamTTS(session=session, api_key=api_key), session


def _wav_pcm(wav_bytes: bytes) -> tuple[bytes, int, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    if channels == 1:
        return pcm, sample_rate, channels
    frame_width = 2
    frame_size = channels * frame_width
    mono = bytearray()
    for offset in range(0, len(pcm) - frame_size + 1, frame_size):
        total = 0
        for ch in range(channels):
            i = offset + ch * frame_width
            total += int.from_bytes(pcm[i : i + 2], "little", signed=True)
        sample = int(total / channels)
        mono.extend(sample.to_bytes(2, "little", signed=True))
    return bytes(mono), sample_rate, 1


def _resample(pcm: bytes, in_rate: int, out_rate: int) -> bytes:
    resampler = rtc.AudioResampler(in_rate, out_rate, num_channels=_NUM_CHANNELS)
    frame = rtc.AudioFrame(
        data=pcm,
        sample_rate=in_rate,
        num_channels=_NUM_CHANNELS,
        samples_per_channel=len(pcm) // 2,
    )
    out = bytearray()
    for piece in [*resampler.push(frame), *resampler.flush()]:
        out.extend(bytes(piece.data))
    return bytes(out)


def _split_text(text: str) -> list[str]:
    stripped = text.strip()
    if len(stripped) <= _MAX_CHARS:
        return [stripped]
    chunks: list[str] = []
    remaining = stripped
    while remaining:
        if len(remaining) <= _MAX_CHARS:
            chunks.append(remaining)
            break
        window = remaining[:_MAX_CHARS]
        cut = max(window.rfind("।"), window.rfind("."), window.rfind("!"), window.rfind("?"))
        if cut < _MAX_CHARS // 2:
            cut = _MAX_CHARS
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    return [c for c in chunks if c]


async def play_audio(source: rtc.AudioSource, text: str, tts: SarvamTTS) -> None:
    """
    Synthesise *text* using Sarvam TTS and push audio frames to the persistent LiveKit track.
    """
    if not text or not text.strip():
        return

    try:
        samples_per_frame = int(tts.sample_rate * _FRAME_MS / 1000)
        bytes_per_frame = samples_per_frame * 2
        for chunk in _split_text(text):
            wav_bytes = await tts.synthesize_wav(chunk)
            pcm, sample_rate, _channels = _wav_pcm(wav_bytes)
            if sample_rate != tts.sample_rate:
                # The LiveKit source runs at tts.sample_rate; resample rather
                # than play the audio back at the wrong speed/pitch.
                print(f"[TTS] Unexpected sample rate {sample_rate}, resampling to {tts.sample_rate}")
                pcm = _resample(pcm, sample_rate, tts.sample_rate)
            for offset in range(0, len(pcm), bytes_per_frame):
                piece = pcm[offset : offset + bytes_per_frame]
                if len(piece) < bytes_per_frame:
                    piece = piece + b"\x00" * (bytes_per_frame - len(piece))
                frame = rtc.AudioFrame(
                    data=piece,
                    sample_rate=tts.sample_rate,
                    num_channels=_NUM_CHANNELS,
                    samples_per_channel=samples_per_frame,
                )
                await source.capture_frame(frame)
        await source.wait_for_playout()
    except Exception as exc:
        print(f"[TTS] Synthesis error: {exc}")
