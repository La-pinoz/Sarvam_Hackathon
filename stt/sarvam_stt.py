# stt/sarvam_stt.py
"""
Sarvam realtime STT for FasalVachak (saaras:v4 over WebSocket).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
from urllib.parse import urlencode

import aiohttp
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

_STT_RATE = 16000
_WS_BASE = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
_PLACEHOLDER = "your_sarvam_api_key_here"
_KEYTERMS = [
    "dhaan",
    "dhan",
    "chawal",
    "kapas",
    "rice",
    "paddy",
    "cotton",
    "धान",
    "चावल",
    "कपास",
    "blast",
    "blight",
    "wilt",
    "patta",
    "suka",
]


def _require_api_key() -> str:
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key or api_key == _PLACEHOLDER:
        raise ValueError(
            "\n\n>>> ❌ ERROR: SARVAM_API_KEY is missing! <<<\n"
            "Get a key from https://dashboard.sarvam.ai/login and set "
            "SARVAM_API_KEY in your .env file.\n\n"
        )
    return api_key


class SarvamSTT:
    """Factory for per-track realtime Saaras streams."""

    def __init__(self, session: aiohttp.ClientSession, *, api_key: str | None = None) -> None:
        self._session = session
        self._api_key = api_key or _require_api_key()

    def stream(self) -> "SarvamSTTStream":
        stream = SarvamSTTStream(self._session, self._api_key)
        stream.start()
        return stream


class SarvamSTTStream:
    """Mirrors the livekit STT stream surface used by server.py."""

    def __init__(self, session: aiohttp.ClientSession, api_key: str) -> None:
        self._session = session
        self._api_key = api_key
        self._audio_q: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue()
        self._events: asyncio.Queue[SpeechEvent | None] = asyncio.Queue()
        self._closed = False
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._task: asyncio.Task | None = None
        self._resampler: rtc.AudioResampler | None = None
        self._in_rate: int | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if not self._closed:
            self._audio_q.put_nowait(frame)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._audio_q.put_nowait(None)
        if self._task is not None:
            try:
                await self._task
            except Exception:
                pass

    def __aiter__(self) -> "SarvamSTTStream":
        return self

    async def __anext__(self) -> SpeechEvent:
        event = await self._events.get()
        if event is None:
            raise StopAsyncIteration
        return event

    async def _run(self) -> None:
        params = {
            "language_code": "hi-IN",
            "model": "saaras:v4",
            "stream_type": "fast",
            "mode": "codemix",
            "endpointing": "vad",
            "encoding": "linear16",
            "sample_rate": str(_STT_RATE),
            "silence_duration_ms": "700",
            "keyterms": json.dumps(_KEYTERMS, ensure_ascii=False),
        }
        url = f"{_WS_BASE}?{urlencode(params)}"
        try:
            async with self._session.ws_connect(
                url,
                headers={"api-subscription-key": self._api_key},
                heartbeat=20.0,
            ) as ws:
                self._ws = ws
                send_task = asyncio.create_task(self._send_loop(ws))
                recv_task = asyncio.create_task(self._recv_loop(ws))
                ping_task = asyncio.create_task(self._ping_loop(ws))
                try:
                    done, _pending = await asyncio.wait(
                        {send_task, recv_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if send_task in done:
                        send_task.result()
                        # Audio ended: ask Sarvam to finalize the pending
                        # utterance and keep receiving until it closes.
                        if not ws.closed:
                            await ws.send_json({"event": "end"})
                            try:
                                await asyncio.wait_for(recv_task, timeout=5.0)
                            except asyncio.TimeoutError:
                                pass
                    else:
                        recv_task.result()
                finally:
                    ping_task.cancel()
                    for task in (send_task, recv_task):
                        if not task.done():
                            task.cancel()
        except Exception as exc:
            print(f"[STT] Sarvam websocket error: {exc}")
        finally:
            self._closed = True
            await self._events.put(None)

    async def _send_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            frame = await self._audio_q.get()
            if frame is None:
                break
            pcm = self._to_16k_mono(frame)
            if not pcm:
                continue
            await ws.send_json(
                {
                    "event": "audio_input",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                }
            )

    async def _recv_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
                continue
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            event_name = payload.get("event")
            if event_name == "transcript.final":
                text = (payload.get("text") or "").strip()
                if not text:
                    continue
                await self._events.put(
                    SpeechEvent(
                        type=SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[
                            SpeechData(
                                language=payload.get("language") or "hi-IN",
                                text=text,
                            )
                        ],
                    )
                )
            elif event_name == "error":
                print(
                    f"[STT] Sarvam error {payload.get('code')}: {payload.get('message')}"
                )
                if payload.get("is_fatal"):
                    break

    async def _ping_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        try:
            while not ws.closed:
                await asyncio.sleep(20)
                await ws.send_json({"event": "ping"})
        except asyncio.CancelledError:
            return
        except Exception:
            return

    def _to_16k_mono(self, frame: rtc.AudioFrame) -> bytes:
        pcm = bytes(frame.data)
        if frame.num_channels > 1:
            pcm = _mix_to_mono(pcm, frame.num_channels)
            frame = rtc.AudioFrame(
                data=pcm,
                sample_rate=frame.sample_rate,
                num_channels=1,
                samples_per_channel=len(pcm) // 2,
            )
        if frame.sample_rate == _STT_RATE:
            return bytes(frame.data)
        if self._resampler is None or self._in_rate != frame.sample_rate:
            self._resampler = rtc.AudioResampler(
                frame.sample_rate,
                _STT_RATE,
                num_channels=1,
            )
            self._in_rate = frame.sample_rate
        pieces = bytearray()
        for resampled in self._resampler.push(frame):
            pieces.extend(bytes(resampled.data))
        return bytes(pieces)


def _mix_to_mono(pcm: bytes, channels: int) -> bytes:
    if channels <= 1:
        return pcm
    out = bytearray()
    step = channels * 2
    for offset in range(0, len(pcm) - step + 1, step):
        total = 0
        for ch in range(channels):
            i = offset + ch * 2
            total += int.from_bytes(pcm[i : i + 2], "little", signed=True)
        sample = int(total / channels)
        out.extend(sample.to_bytes(2, "little", signed=True))
    return bytes(out)
