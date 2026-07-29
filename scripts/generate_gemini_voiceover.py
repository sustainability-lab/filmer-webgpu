#!/usr/bin/env python3
"""Generate the FiLMeR walkthrough voiceover with Gemini TTS.

The API key is read from GEMINI_API_KEY and is never written to disk.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import wave


API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("transcript", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="gemini-3.1-flash-tts-preview")
    parser.add_argument("--voice", default="Iapetus")
    return parser.parse_args()


def request_audio(api_key: str, model: str, voice: str, transcript: str) -> bytes:
    prompt = f"""Synthesize the exact transcript below as speech.

AUDIO PROFILE: A warm, knowledgeable Indian male product narrator.
SCENE: A concise scientific software walkthrough for researchers and stakeholders.
DIRECTOR'S NOTES: Natural Indian English pronunciation. Calm, confident, and precise; never salesy. Medium pace, clear technical terms, short pauses between paragraphs. Do not read these directions aloud. Speak only the transcript, exactly as written.

TRANSCRIPT:
{transcript}
"""
    payload = {
        "model": model,
        "input": prompt,
        "response_format": {"type": "audio"},
        "generation_config": {"speech_config": [{"voice": voice}]},
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.load(response)

    encoded = None
    if isinstance(body, dict):
        output_audio = body.get("output_audio")
        if isinstance(output_audio, dict):
            encoded = output_audio.get("data")
        if not encoded:
            for step in body.get("steps", []):
                if not isinstance(step, dict) or step.get("type") != "model_output":
                    continue
                for part in step.get("content", []):
                    if isinstance(part, dict) and part.get("type") == "audio":
                        encoded = part.get("data")
                        if encoded:
                            break
    if not encoded:
        safe_keys = sorted(body.keys()) if isinstance(body, dict) else []
        raise RuntimeError(f"Gemini returned no audio; keys={safe_keys}")
    return base64.b64decode(encoded, validate=True)


def write_wave(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(pcm)


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set")
    transcript = args.transcript.read_text(encoding="utf-8").strip()
    if not transcript:
        raise SystemExit("Transcript is empty")

    for attempt in range(1, 4):
        try:
            pcm = request_audio(api_key, args.model, args.voice, transcript)
            write_wave(args.output, pcm)
            seconds = len(pcm) / (24_000 * 2)
            print(f"Wrote {args.output} ({seconds:.2f}s, {args.model}, {args.voice})")
            return
        except (urllib.error.URLError, RuntimeError, ValueError) as error:
            if attempt == 3:
                raise SystemExit(f"Gemini TTS failed after {attempt} attempts: {error}") from error
            time.sleep(2**attempt)


if __name__ == "__main__":
    main()
