"""
providers.py — pluggable adapters for every external API in the dubbing pipeline.

The pipeline has four stages. Each stage can be served by a different vendor,
chosen at runtime from the dashboard:

    SEPARATE : song.mp3            -> vocals.wav + instrumental.wav
    CLONE    : singer samples      -> a voice id / voice reference
    CONVERT  : vocals + voice id   -> vocals sung in the target voice
    MIX      : converted + instr.  -> final master

Every adapter exposes the same coroutine signatures, so server.py never needs
to know which vendor is in play.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

# --------------------------------------------------------------------------
# Capability matrix — drives the dropdowns in the dashboard
# --------------------------------------------------------------------------

STAGES = ["separate", "clone", "convert", "mix"]

PROVIDERS: dict[str, dict[str, Any]] = {
    "replicate": {
        "label": "Replicate",
        "key_env": "REPLICATE_API_TOKEN",
        "key_label": "API token (r8_...)",
        "signup": "https://replicate.com/account/api-tokens",
        "stages": ["separate", "clone", "convert"],
        "notes": "Runs open models (Demucs, RVC, so-vits-svc). Cheapest, most tweakable. "
                 "Cloning is zero-shot: the singer audio is passed straight to the "
                 "conversion model as a reference, so no training step is needed.",
    },
    "lalalai": {
        "label": "LALAL.AI",
        "key_env": "LALALAI_LICENSE_KEY",
        "key_label": "License key",
        "signup": "https://www.lalal.ai/api/",
        "stages": ["separate", "clone", "convert"],
        "notes": "Best-in-class stem splitting. Also ships voice cloning in API v1.",
    },
    "kits": {
        "label": "Kits.AI",
        "key_env": "KITS_API_KEY",
        "key_label": "API key",
        "signup": "https://app.kits.ai/api-access",
        "stages": ["separate", "clone", "convert"],
        "notes": "Purpose-built for singing voice conversion. Cloning trains a real "
                 "voice model, which takes a few minutes but sounds the most natural "
                 "on sustained notes.",
    },
    "elevenlabs": {
        "label": "ElevenLabs",
        "key_env": "ELEVENLABS_API_KEY",
        "key_label": "API key (xi-...)",
        "signup": "https://elevenlabs.io/app/settings/api-keys",
        "stages": ["clone", "convert"],
        "notes": "Instant Voice Cloning + speech-to-speech. Excellent on spoken dubbing "
                 "and rap; weaker on sustained melodic singing than Kits or RVC.",
    },
    "musicai": {
        "label": "Music.AI",
        "key_env": "MUSICAI_API_KEY",
        "key_label": "API key",
        "signup": "https://music.ai/",
        "stages": ["separate", "mix"],
        "notes": "Workflow engine. Good stems plus a mastering workflow you can use "
                 "as the final mix stage.",
    },
    "local": {
        "label": "Local (ffmpeg)",
        "key_env": None,
        "key_label": None,
        "signup": None,
        "stages": ["mix"],
        "notes": "Mixes stems locally with ffmpeg and applies EBU R128 loudness "
                 "normalisation. No API key, no upload, no cost.",
    },
}

DEFAULT_ENGINES = {
    "separate": "replicate",
    "clone": "replicate",
    "convert": "replicate",
    "mix": "local",
}

# Replicate model slugs. These are user-editable from the dashboard because
# community model versions get republished often.
REPLICATE_DEFAULTS = {
    "separate": "ryan5453/demucs",
    "convert": "zsxkib/realistic-voice-cloning",
}


class PipelineError(RuntimeError):
    """Anything that should be shown to the user as a readable failure."""


@dataclass
class StageResult:
    files: dict[str, Path] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


Logger = Callable[[str], None]


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

async def _download(client: httpx.AsyncClient, url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with client.stream("GET", url, timeout=600.0, follow_redirects=True) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            async for chunk in r.aiter_bytes(1 << 16):
                fh.write(chunk)
    return dest


async def _poll(
    fetch: Callable[[], Any],
    done: Callable[[Any], bool],
    failed: Callable[[Any], bool],
    describe: Callable[[Any], str],
    log: Logger,
    interval: float = 4.0,
    timeout: float = 1800.0,
) -> Any:
    """Generic 'submit then wait' loop used by every async vendor API."""
    started = time.time()
    last_msg = ""
    while True:
        payload = await fetch()
        if done(payload):
            return payload
        if failed(payload):
            raise PipelineError(describe(payload))
        msg = describe(payload)
        if msg and msg != last_msg:
            log(msg)
            last_msg = msg
        if time.time() - started > timeout:
            raise PipelineError(f"Timed out after {int(timeout)}s waiting on the provider.")
        await asyncio.sleep(interval)


def _require(key: Optional[str], provider: str) -> str:
    if not key:
        meta = PROVIDERS[provider]
        raise PipelineError(
            f"No API key set for {meta['label']}. Add it in the dashboard "
            f"or set {meta['key_env']} in your .env file. Get one at {meta['signup']}"
        )
    return key


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-6:]
        raise PipelineError("ffmpeg failed:\n" + "\n".join(tail))


# --------------------------------------------------------------------------
# Replicate
# --------------------------------------------------------------------------

class Replicate:
    BASE = "https://api.replicate.com/v1"

    def __init__(self, token: str, models: Optional[dict] = None):
        self.token = _require(token, "replicate")
        self.models = {**REPLICATE_DEFAULTS, **(models or {})}
        self.headers = {"Authorization": f"Bearer {self.token}"}

    async def _upload(self, client: httpx.AsyncClient, path: Path) -> str:
        with path.open("rb") as fh:
            r = await client.post(
                f"{self.BASE}/files",
                headers=self.headers,
                files={"content": (path.name, fh, "application/octet-stream")},
                timeout=600.0,
            )
        r.raise_for_status()
        return r.json()["urls"]["get"]

    async def _predict(self, client: httpx.AsyncClient, model: str,
                       payload: dict, log: Logger) -> Any:
        if ":" in model:  # owner/name:versionhash
            body = {"version": model.split(":", 1)[1], "input": payload}
            url = f"{self.BASE}/predictions"
        else:
            body = {"input": payload}
            url = f"{self.BASE}/models/{model}/predictions"

        r = await client.post(url, headers=self.headers, json=body, timeout=120.0)
        if r.status_code >= 400:
            raise PipelineError(f"Replicate rejected the request ({r.status_code}): {r.text[:400]}")
        pred = r.json()
        pred_url = pred.get("urls", {}).get("get") or f"{self.BASE}/predictions/{pred['id']}"

        async def fetch():
            resp = await client.get(pred_url, headers=self.headers, timeout=60.0)
            resp.raise_for_status()
            return resp.json()

        final = await _poll(
            fetch,
            done=lambda p: p.get("status") == "succeeded",
            failed=lambda p: p.get("status") in ("failed", "canceled"),
            describe=lambda p: (p.get("error") or f"Replicate: {p.get('status')}"),
            log=log,
        )
        return final.get("output")

    async def separate(self, client, song: Path, workdir: Path, log: Logger) -> StageResult:
        log("Uploading song to Replicate…")
        url = await self._upload(client, song)
        log("Running Demucs stem separation…")
        out = await self._predict(client, self.models["separate"], {
            "audio": url, "stem": "vocals", "output_format": "wav",
        }, log)

        files: dict[str, Path] = {}
        # Demucs wrappers return either a dict of stems or a flat list.
        if isinstance(out, dict):
            for name, link in out.items():
                if not isinstance(link, str) or not link.startswith("http"):
                    continue
                key = "vocals" if "vocal" in name.lower() else name.lower()
                files[key] = await _download(client, link, workdir / f"{key}.wav")
        elif isinstance(out, list):
            for i, link in enumerate(out):
                if isinstance(link, str) and link.startswith("http"):
                    key = "vocals" if i == 0 else f"stem{i}"
                    files[key] = await _download(client, link, workdir / f"{key}.wav")

        if "vocals" not in files:
            raise PipelineError("Demucs returned no vocal stem. Try a different model slug.")

        # Anything that is not the vocal becomes the backing track.
        others = [p for k, p in files.items() if k != "vocals"]
        if "no_vocals" in files:
            files["instrumental"] = files["no_vocals"]
        elif "accompaniment" in files:
            files["instrumental"] = files["accompaniment"]
        elif len(others) == 1:
            files["instrumental"] = others[0]
        elif others:
            files["instrumental"] = _sum_stems(others, workdir / "instrumental.wav")
        else:
            files["instrumental"] = _subtract(song, files["vocals"], workdir / "instrumental.wav")

        return StageResult(files=files, meta={"engine": "replicate"})

    async def clone(self, client, samples: list[Path], name: str, log: Logger) -> StageResult:
        # RVC on Replicate is zero-shot: the reference audio *is* the voice.
        log("Uploading singer reference to Replicate…")
        url = await self._upload(client, samples[0])
        return StageResult(meta={"engine": "replicate", "voice_ref": url, "zero_shot": True})

    async def convert(self, client, vocals: Path, voice: dict,
                      workdir: Path, opts: dict, log: Logger) -> StageResult:
        log("Uploading vocal stem…")
        song_url = await self._upload(client, vocals)
        ref = voice.get("voice_ref")
        if not ref:
            raise PipelineError("Replicate conversion needs a singer reference from the clone stage.")

        log("Running voice conversion (this is the slow one)…")
        out = await self._predict(client, self.models["convert"], {
            "song_input": song_url,
            "rvc_model": "CUSTOM",
            "custom_rvc_model_download_url": ref,
            "pitch_change": opts.get("pitch_shift", 0),
            "index_rate": opts.get("index_rate", 0.5),
            "output_format": "wav",
            "reference_audio": ref,
            "audio": song_url,
        }, log)

        link = out if isinstance(out, str) else (out[0] if isinstance(out, list) and out else None)
        if not link:
            raise PipelineError("Conversion model returned no audio.")
        path = await _download(client, link, workdir / "converted_vocals.wav")
        return StageResult(files={"converted": path}, meta={"engine": "replicate"})


# --------------------------------------------------------------------------
# LALAL.AI
# --------------------------------------------------------------------------

class LalalAI:
    BASE = "https://www.lalal.ai/api"

    def __init__(self, key: str):
        self.key = _require(key, "lalalai")
        # v1 uses X-License-Key; the long-lived endpoints use Authorization.
        self.headers = {
            "Authorization": f"license {self.key}",
            "X-License-Key": self.key,
        }

    async def _upload(self, client: httpx.AsyncClient, path: Path, log: Logger) -> str:
        log(f"Uploading {path.name} to LALAL.AI…")
        data = path.read_bytes()
        r = await client.post(
            f"{self.BASE}/upload/",
            headers={**self.headers,
                     "Content-Disposition": f'attachment; filename="{path.name}"'},
            content=data, timeout=900.0,
        )
        r.raise_for_status()
        body = r.json()
        if body.get("status") != "success":
            raise PipelineError(f"LALAL.AI upload failed: {body.get('error', body)}")
        return body["id"]

    async def separate(self, client, song: Path, workdir: Path, log: Logger) -> StageResult:
        file_id = await self._upload(client, song, log)
        log("Requesting vocal/instrumental split…")
        params = [{"id": file_id, "stem": "vocals", "splitter": "phoenix",
                   "enhanced_processing_enabled": True}]
        r = await client.post(f"{self.BASE}/split/", headers=self.headers,
                              data={"params": _json(params)}, timeout=120.0)
        r.raise_for_status()

        async def fetch():
            resp = await client.post(f"{self.BASE}/check/", headers=self.headers,
                                     data={"id": file_id}, timeout=60.0)
            resp.raise_for_status()
            return resp.json()

        final = await _poll(
            fetch,
            done=lambda p: _lalal_task(p, file_id).get("state") == "success",
            failed=lambda p: _lalal_task(p, file_id).get("state") == "error",
            describe=lambda p: _lalal_desc(p, file_id),
            log=log,
        )
        task = _lalal_task(final, file_id)
        vocal_url = task.get("stem_track")
        back_url = task.get("back_track")
        if not vocal_url:
            raise PipelineError("LALAL.AI returned no vocal track.")

        files = {"vocals": await _download(client, vocal_url, workdir / "vocals.wav")}
        if back_url:
            files["instrumental"] = await _download(client, back_url, workdir / "instrumental.wav")
        else:
            files["instrumental"] = _subtract(song, files["vocals"], workdir / "instrumental.wav")
        return StageResult(files=files, meta={"engine": "lalalai"})

    async def clone(self, client, samples: list[Path], name: str, log: Logger) -> StageResult:
        ref_id = await self._upload(client, samples[0], log)
        log("Registering voice with LALAL.AI…")
        return StageResult(meta={"engine": "lalalai", "voice_id": ref_id,
                                 "voice_ref_id": ref_id})

    async def convert(self, client, vocals: Path, voice: dict,
                      workdir: Path, opts: dict, log: Logger) -> StageResult:
        src_id = await self._upload(client, vocals, log)
        ref_id = voice.get("voice_ref_id") or voice.get("voice_id")
        if not ref_id:
            raise PipelineError("LALAL.AI conversion needs a cloned voice reference.")
        log("Running LALAL.AI voice cloning conversion…")
        r = await client.post(
            f"{self.BASE}/voice-clone/", headers=self.headers,
            data={"id": src_id, "reference_id": ref_id}, timeout=120.0,
        )
        if r.status_code >= 400:
            raise PipelineError(
                "LALAL.AI voice cloning is not enabled on this key "
                f"({r.status_code}). Use Kits.AI or Replicate for the convert stage."
            )

        async def fetch():
            resp = await client.post(f"{self.BASE}/check/", headers=self.headers,
                                     data={"id": src_id}, timeout=60.0)
            resp.raise_for_status()
            return resp.json()

        final = await _poll(
            fetch,
            done=lambda p: _lalal_task(p, src_id).get("state") == "success",
            failed=lambda p: _lalal_task(p, src_id).get("state") == "error",
            describe=lambda p: _lalal_desc(p, src_id),
            log=log,
        )
        task = _lalal_task(final, src_id)
        url = task.get("stem_track") or task.get("result")
        path = await _download(client, url, workdir / "converted_vocals.wav")
        return StageResult(files={"converted": path}, meta={"engine": "lalalai"})


def _lalal_task(payload: dict, file_id: str) -> dict:
    entry = (payload.get("result") or {}).get(file_id, {})
    return {**entry, **(entry.get("task") or {}), **(entry.get("split") or {})}


def _lalal_desc(payload: dict, file_id: str) -> str:
    t = _lalal_task(payload, file_id)
    if t.get("state") == "error":
        return f"LALAL.AI error: {t.get('error', 'unknown')}"
    return f"LALAL.AI: {t.get('state', 'queued')} {int(float(t.get('progress') or 0))}%"


# --------------------------------------------------------------------------
# Kits.AI
# --------------------------------------------------------------------------

class Kits:
    BASE = "https://arpeggi.io/api/kits/v1"

    def __init__(self, key: str):
        self.key = _require(key, "kits")
        self.headers = {"Authorization": f"Bearer {self.key}"}

    async def _wait(self, client, path: str, job_id: Any, log: Logger) -> dict:
        async def fetch():
            r = await client.get(f"{self.BASE}/{path}/{job_id}",
                                 headers=self.headers, timeout=60.0)
            r.raise_for_status()
            return r.json()

        return await _poll(
            fetch,
            done=lambda p: p.get("status") in ("success", "succeeded", "completed"),
            failed=lambda p: p.get("status") in ("error", "failed"),
            describe=lambda p: f"Kits.AI: {p.get('status', 'queued')}",
            log=log,
        )

    async def separate(self, client, song: Path, workdir: Path, log: Logger) -> StageResult:
        log("Uploading song to Kits.AI…")
        with song.open("rb") as fh:
            r = await client.post(f"{self.BASE}/voice-separations", headers=self.headers,
                                  files={"soundFile": (song.name, fh)}, timeout=900.0)
        r.raise_for_status()
        job = await self._wait(client, "voice-separations", r.json()["id"], log)

        vocal_url = job.get("vocalAudioFileUrl") or job.get("outputFileUrl")
        inst_url = job.get("instrumentalAudioFileUrl") or job.get("backingAudioFileUrl")
        if not vocal_url:
            raise PipelineError("Kits.AI returned no vocal stem.")

        files = {"vocals": await _download(client, vocal_url, workdir / "vocals.wav")}
        files["instrumental"] = (
            await _download(client, inst_url, workdir / "instrumental.wav") if inst_url
            else _subtract(song, files["vocals"], workdir / "instrumental.wav")
        )
        return StageResult(files=files, meta={"engine": "kits"})

    async def clone(self, client, samples: list[Path], name: str, log: Logger) -> StageResult:
        log("Uploading singer samples and starting Kits.AI voice training…")
        files = [("files", (p.name, p.open("rb"))) for p in samples]
        try:
            r = await client.post(f"{self.BASE}/voice-models", headers=self.headers,
                                  data={"name": name}, files=files, timeout=1800.0)
        finally:
            for _, (_, fh) in files:
                fh.close()
        if r.status_code >= 400:
            raise PipelineError(
                f"Kits.AI voice training failed ({r.status_code}): {r.text[:300]}"
            )
        model = r.json()
        model_id = model.get("id")
        log("Training the voice model — typically 3-10 minutes…")
        done = await self._poll_model(client, model_id, log)
        return StageResult(meta={"engine": "kits", "voice_id": model_id,
                                 "voice_model_id": model_id,
                                 "status": done.get("status")})

    async def _poll_model(self, client, model_id, log: Logger) -> dict:
        async def fetch():
            r = await client.get(f"{self.BASE}/voice-models/{model_id}",
                                 headers=self.headers, timeout=60.0)
            r.raise_for_status()
            return r.json()

        return await _poll(
            fetch,
            done=lambda p: p.get("status") in ("success", "ready", "completed"),
            failed=lambda p: p.get("status") in ("error", "failed"),
            describe=lambda p: f"Kits.AI training: {p.get('status', 'queued')}",
            log=log, interval=10.0, timeout=3600.0,
        )

    async def convert(self, client, vocals: Path, voice: dict,
                      workdir: Path, opts: dict, log: Logger) -> StageResult:
        model_id = voice.get("voice_model_id") or voice.get("voice_id")
        if not model_id:
            raise PipelineError(
                "Kits.AI conversion needs a Kits voice model. Either run the clone "
                "stage on Kits.AI, or paste an existing voiceModelId in Advanced."
            )
        log("Submitting vocal stem for conversion…")
        with vocals.open("rb") as fh:
            r = await client.post(
                f"{self.BASE}/voice-conversions", headers=self.headers,
                data={"voiceModelId": str(model_id),
                      "pitchShift": str(opts.get("pitch_shift", 0))},
                files={"soundFile": (vocals.name, fh)}, timeout=900.0,
            )
        r.raise_for_status()
        job = await self._wait(client, "voice-conversions", r.json()["id"], log)
        url = job.get("outputFileUrl") or job.get("lossyOutputFileUrl")
        if not url:
            raise PipelineError("Kits.AI conversion finished but returned no file.")
        path = await _download(client, url, workdir / "converted_vocals.wav")
        return StageResult(files={"converted": path}, meta={"engine": "kits"})


# --------------------------------------------------------------------------
# ElevenLabs
# --------------------------------------------------------------------------

class ElevenLabs:
    BASE = "https://api.elevenlabs.io/v1"

    def __init__(self, key: str):
        self.key = _require(key, "elevenlabs")
        self.headers = {"xi-api-key": self.key}

    async def clone(self, client, samples: list[Path], name: str, log: Logger) -> StageResult:
        log("Creating an Instant Voice Clone on ElevenLabs…")
        files = [("files", (p.name, p.open("rb"), "audio/mpeg")) for p in samples]
        try:
            r = await client.post(f"{self.BASE}/voices/add", headers=self.headers,
                                  data={"name": name, "remove_background_noise": "true"},
                                  files=files, timeout=600.0)
        finally:
            for _, (_, fh, _ct) in files:
                fh.close()
        if r.status_code >= 400:
            raise PipelineError(f"ElevenLabs cloning failed ({r.status_code}): {r.text[:300]}")
        voice_id = r.json()["voice_id"]
        log(f"Voice created: {voice_id}")
        return StageResult(meta={"engine": "elevenlabs", "voice_id": voice_id})

    async def convert(self, client, vocals: Path, voice: dict,
                      workdir: Path, opts: dict, log: Logger) -> StageResult:
        voice_id = voice.get("voice_id")
        if not voice_id:
            raise PipelineError(
                "ElevenLabs conversion needs an ElevenLabs voice_id. Run the clone "
                "stage on ElevenLabs, or paste an existing voice id in Advanced."
            )
        log("Running ElevenLabs speech-to-speech…")
        with vocals.open("rb") as fh:
            r = await client.post(
                f"{self.BASE}/speech-to-speech/{voice_id}",
                headers=self.headers,
                data={"model_id": opts.get("eleven_model", "eleven_multilingual_sts_v2"),
                      "remove_background_noise": "true"},
                files={"audio": (vocals.name, fh, "audio/wav")},
                timeout=1800.0,
            )
        if r.status_code >= 400:
            raise PipelineError(f"ElevenLabs STS failed ({r.status_code}): {r.text[:300]}")
        out = workdir / "converted_vocals.mp3"
        out.write_bytes(r.content)
        return StageResult(files={"converted": out}, meta={"engine": "elevenlabs"})


# --------------------------------------------------------------------------
# Music.AI
# --------------------------------------------------------------------------

class MusicAI:
    BASE = "https://api.music.ai/api"

    def __init__(self, key: str, workflows: Optional[dict] = None):
        self.key = _require(key, "musicai")
        self.headers = {"Authorization": self.key}
        self.workflows = {"separate": "stems-vocals-accompaniment",
                          "master": "master-audio", **(workflows or {})}

    async def _upload(self, client, path: Path, log: Logger) -> str:
        log(f"Uploading {path.name} to Music.AI…")
        r = await client.post(f"{self.BASE}/upload", headers=self.headers, timeout=120.0)
        r.raise_for_status()
        slots = r.json()
        put = await client.put(slots["uploadUrl"], content=path.read_bytes(), timeout=900.0)
        put.raise_for_status()
        return slots["downloadUrl"]

    async def _job(self, client, workflow: str, params: dict, log: Logger) -> dict:
        r = await client.post(f"{self.BASE}/job", headers=self.headers, timeout=120.0,
                              json={"name": f"dub-{uuid.uuid4().hex[:8]}",
                                    "workflow": workflow, "params": params})
        if r.status_code >= 400:
            raise PipelineError(f"Music.AI job rejected ({r.status_code}): {r.text[:300]}")
        job_id = r.json()["id"]

        async def fetch():
            resp = await client.get(f"{self.BASE}/job/{job_id}",
                                    headers=self.headers, timeout=60.0)
            resp.raise_for_status()
            return resp.json()

        return await _poll(
            fetch,
            done=lambda p: p.get("status") == "SUCCEEDED",
            failed=lambda p: p.get("status") == "FAILED",
            describe=lambda p: f"Music.AI: {p.get('status', 'QUEUED')}",
            log=log,
        )

    async def separate(self, client, song: Path, workdir: Path, log: Logger) -> StageResult:
        url = await self._upload(client, song, log)
        log("Running Music.AI stem workflow…")
        job = await self._job(client, self.workflows["separate"], {"inputUrl": url}, log)
        result = job.get("result") or {}
        vocal_url = result.get("vocals") or result.get("vocal")
        inst_url = (result.get("accompaniment") or result.get("instrumental")
                    or result.get("backing"))
        if not vocal_url:
            raise PipelineError(f"Music.AI workflow returned no vocals. Got: {list(result)}")
        files = {"vocals": await _download(client, vocal_url, workdir / "vocals.wav")}
        files["instrumental"] = (
            await _download(client, inst_url, workdir / "instrumental.wav") if inst_url
            else _subtract(song, files["vocals"], workdir / "instrumental.wav")
        )
        return StageResult(files=files, meta={"engine": "musicai"})

    async def mix(self, client, converted: Path, instrumental: Path,
                  workdir: Path, opts: dict, log: Logger) -> StageResult:
        # Mix locally, then send the result through Music.AI mastering.
        rough = local_mix(converted, instrumental, workdir / "rough_mix.wav", opts)
        url = await self._upload(client, rough, log)
        log("Mastering on Music.AI…")
        job = await self._job(client, self.workflows["master"], {"inputUrl": url}, log)
        result = job.get("result") or {}
        out_url = next((v for v in result.values()
                        if isinstance(v, str) and v.startswith("http")), None)
        if not out_url:
            log("Mastering returned nothing usable — keeping the local mix.")
            return StageResult(files={"final": rough}, meta={"engine": "local-fallback"})
        final = await _download(client, out_url, workdir / "final_master.wav")
        return StageResult(files={"final": final}, meta={"engine": "musicai"})


# --------------------------------------------------------------------------
# Local ffmpeg stage
# --------------------------------------------------------------------------

def local_mix(vocals: Path, instrumental: Path, dest: Path, opts: dict) -> Path:
    if not ffmpeg_available():
        raise PipelineError(
            "ffmpeg is not on your PATH. Install it (https://ffmpeg.org/download.html) "
            "— every mix path needs it."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    v_gain = float(opts.get("vocal_gain_db", 0.0))
    i_gain = float(opts.get("instrumental_gain_db", -1.5))
    reverb = float(opts.get("vocal_reverb", 0.15))

    vchain = [f"volume={v_gain}dB", "highpass=f=80", "acompressor=ratio=3:threshold=-18dB"]
    if reverb > 0:
        vchain.append(f"aecho=0.8:0.85:{int(40 + reverb * 60)}:{round(reverb, 2)}")
    vchain.append("aresample=44100")

    filt = (
        f"[0:a]{','.join(vchain)}[v];"
        f"[1:a]volume={i_gain}dB,aresample=44100[i];"
        f"[v][i]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[m];"
        f"[m]loudnorm=I=-14:TP=-1.0:LRA=11[out]"
    )
    _run(["ffmpeg", "-y", "-i", str(vocals), "-i", str(instrumental),
          "-filter_complex", filt, "-map", "[out]",
          "-c:a", "pcm_s16le", "-ar", "44100", str(dest)])
    return dest


def _sum_stems(stems: list[Path], dest: Path) -> Path:
    if not ffmpeg_available():
        raise PipelineError("ffmpeg is required to recombine stems.")
    cmd = ["ffmpeg", "-y"]
    for s in stems:
        cmd += ["-i", str(s)]
    cmd += ["-filter_complex",
            f"amix=inputs={len(stems)}:duration=longest:normalize=0",
            "-c:a", "pcm_s16le", str(dest)]
    _run(cmd)
    return dest


def _subtract(full: Path, vocals: Path, dest: Path) -> Path:
    """Phase-invert the vocal against the full mix to approximate an instrumental."""
    if not ffmpeg_available():
        raise PipelineError("ffmpeg is required to derive the instrumental.")
    _run(["ffmpeg", "-y", "-i", str(full), "-i", str(vocals), "-filter_complex",
          "[1:a]volume=-1.0[inv];[0:a][inv]amix=inputs=2:duration=shortest:normalize=0",
          "-c:a", "pcm_s16le", str(dest)])
    return dest


def to_mp3(src: Path, dest: Path, bitrate: str = "320k") -> Path:
    if not ffmpeg_available():
        return src
    _run(["ffmpeg", "-y", "-i", str(src), "-c:a", "libmp3lame", "-b:a", bitrate, str(dest)])
    return dest


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

def build(provider: str, keys: dict, advanced: dict):
    if provider == "replicate":
        return Replicate(keys.get("replicate"), advanced.get("replicate_models"))
    if provider == "lalalai":
        return LalalAI(keys.get("lalalai"))
    if provider == "kits":
        return Kits(keys.get("kits"))
    if provider == "elevenlabs":
        return ElevenLabs(keys.get("elevenlabs"))
    if provider == "musicai":
        return MusicAI(keys.get("musicai"), advanced.get("musicai_workflows"))
    if provider == "local":
        return None
    raise PipelineError(f"Unknown provider '{provider}'.")


def _json(obj) -> str:
    import json
    return json.dumps(obj)
