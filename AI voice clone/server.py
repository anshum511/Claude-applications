"""
server.py — local orchestrator for the voice-dubbing dashboard.

Run:
    pip install -r requirements.txt
    python server.py

Then open http://127.0.0.1:8765 in your browser.

Nothing leaves this machine except the audio you explicitly send to the
providers you selected. API keys live in memory (or your .env) and are never
written to disk by this server.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

import providers as P

load_dotenv()

ROOT = Path(__file__).parent.resolve()
WORK = ROOT / ".work"
OUTPUT = Path(os.getenv("OUTPUT_DIR", ROOT / "output")).resolve()
WORK.mkdir(exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Voice Dubbing Console")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Keys set from the dashboard this session. Falls back to .env.
SESSION_KEYS: dict[str, str] = {}
JOBS: dict[str, dict] = {}

STAGE_LABELS = {
    "separate": "Splitting the song into stems",
    "clone": "Learning the target singer's voice",
    "convert": "Re-singing the vocal in the cloned voice",
    "mix": "Mixing and mastering the final track",
}


def key_for(provider: str) -> Optional[str]:
    if provider in SESSION_KEYS and SESSION_KEYS[provider]:
        return SESSION_KEYS[provider]
    env = P.PROVIDERS.get(provider, {}).get("key_env")
    return os.getenv(env) if env else None


def all_keys() -> dict[str, Optional[str]]:
    return {name: key_for(name) for name in P.PROVIDERS}


# --------------------------------------------------------------------------
# Job bookkeeping
# --------------------------------------------------------------------------

def new_job(config: dict) -> dict:
    jid = uuid.uuid4().hex[:12]
    job = {
        "id": jid,
        "created": datetime.now().isoformat(timespec="seconds"),
        "state": "queued",
        "config": config,
        "stages": {s: {"state": "pending", "engine": config["engines"].get(s),
                       "log": []} for s in P.STAGES},
        "log": [],
        "artifacts": {},
        "error": None,
    }
    JOBS[jid] = job
    return job


def log(job: dict, stage: Optional[str], msg: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    job["log"].append(line)
    if stage:
        job["stages"][stage]["log"].append(line)
    print(f"{job['id']} {line}", flush=True)


def publish(job: dict, name: str, path: Path) -> None:
    job["artifacts"][name] = {"file": str(path), "size": path.stat().st_size}


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------

async def run_pipeline(job: dict, singer: Path, song: Path) -> None:
    cfg = job["config"]
    engines = cfg["engines"]
    opts = cfg.get("options", {})
    advanced = cfg.get("advanced", {})
    workdir = WORK / job["id"]
    workdir.mkdir(parents=True, exist_ok=True)
    keys = all_keys()

    def mark(stage: str, state: str) -> None:
        job["stages"][stage]["state"] = state

    job["state"] = "running"
    timeout = httpx.Timeout(60.0, read=1800.0, write=1800.0, connect=30.0)

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:

            # -- 1. separate -------------------------------------------------
            mark("separate", "running")
            log(job, "separate", STAGE_LABELS["separate"])
            sep = P.build(engines["separate"], keys, advanced)
            res = await sep.separate(client, song, workdir,
                                     lambda m: log(job, "separate", m))
            vocals, instrumental = res.files["vocals"], res.files["instrumental"]
            publish(job, "original_vocals", vocals)
            publish(job, "instrumental", instrumental)
            mark("separate", "done")
            log(job, "separate", "Stems ready.")

            # -- 2. clone ----------------------------------------------------
            mark("clone", "running")
            log(job, "clone", STAGE_LABELS["clone"])
            existing = advanced.get("existing_voice_id")
            if existing:
                voice = {"voice_id": existing, "voice_model_id": existing,
                         "voice_ref": existing, "voice_ref_id": existing}
                log(job, "clone", f"Reusing existing voice id: {existing}")
            else:
                cloner = P.build(engines["clone"], keys, advanced)
                cres = await cloner.clone(client, [singer],
                                          cfg.get("voice_name", f"target-{job['id']}"),
                                          lambda m: log(job, "clone", m))
                voice = cres.meta
            job["stages"]["clone"]["voice"] = {
                k: v for k, v in voice.items() if k.startswith("voice")
            }
            mark("clone", "done")
            log(job, "clone", "Voice ready.")

            # -- 3. convert --------------------------------------------------
            mark("convert", "running")
            log(job, "convert", STAGE_LABELS["convert"])
            conv = P.build(engines["convert"], keys, advanced)
            vres = await conv.convert(client, vocals, voice, workdir, opts,
                                      lambda m: log(job, "convert", m))
            converted = vres.files["converted"]
            publish(job, "converted_vocals", converted)
            mark("convert", "done")
            log(job, "convert", "Vocal conversion complete.")

            # -- 4. mix ------------------------------------------------------
            mark("mix", "running")
            log(job, "mix", STAGE_LABELS["mix"])
            if engines["mix"] == "musicai":
                mixer = P.build("musicai", keys, advanced)
                mres = await mixer.mix(client, converted, instrumental, workdir, opts,
                                       lambda m: log(job, "mix", m))
                final = mres.files["final"]
            else:
                final = await asyncio.to_thread(
                    P.local_mix, converted, instrumental, workdir / "final_mix.wav", opts
                )

            stem = _safe_name(cfg.get("song_name") or song.stem)
            voice_tag = _safe_name(cfg.get("voice_name") or singer.stem)
            out_wav = OUTPUT / f"{stem} - {voice_tag}.wav"
            shutil.copy2(final, out_wav)
            publish(job, "final_wav", out_wav)

            try:
                out_mp3 = await asyncio.to_thread(
                    P.to_mp3, out_wav, OUTPUT / f"{stem} - {voice_tag}.mp3"
                )
                if out_mp3 != out_wav:
                    publish(job, "final_mp3", out_mp3)
            except Exception as exc:  # mp3 is a nicety, not a requirement
                log(job, "mix", f"MP3 export skipped: {exc}")

            mark("mix", "done")
            log(job, "mix", f"Saved to {out_wav}")

        job["state"] = "done"
        log(job, None, "Pipeline finished.")

    except P.PipelineError as exc:
        _fail(job, str(exc))
    except httpx.HTTPStatusError as exc:
        _fail(job, f"{exc.request.url.host} returned {exc.response.status_code}: "
                   f"{exc.response.text[:300]}")
    except Exception as exc:
        traceback.print_exc()
        _fail(job, f"{type(exc).__name__}: {exc}")


def _fail(job: dict, message: str) -> None:
    job["state"] = "error"
    job["error"] = message
    for stage in P.STAGES:
        if job["stages"][stage]["state"] == "running":
            job["stages"][stage]["state"] = "error"
    log(job, None, f"FAILED — {message}")


def _safe_name(text: str) -> str:
    keep = "".join(c for c in text if c.isalnum() or c in " -_()")
    return (keep.strip() or "output")[:60]


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    page = ROOT / "dashboard.html"
    if not page.exists():
        raise HTTPException(404, "dashboard.html is missing from the project folder.")
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/api/providers")
async def get_providers():
    keys = all_keys()
    return {
        "stages": P.STAGES,
        "defaults": P.DEFAULT_ENGINES,
        "ffmpeg": P.ffmpeg_available(),
        "output_dir": str(OUTPUT),
        "providers": {
            name: {
                **{k: v for k, v in meta.items()},
                "configured": bool(keys.get(name)) or meta["key_env"] is None,
                "source": ("session" if SESSION_KEYS.get(name)
                           else "env" if keys.get(name) else None),
            }
            for name, meta in P.PROVIDERS.items()
        },
    }


@app.post("/api/keys")
async def set_keys(payload: dict):
    for name, value in (payload.get("keys") or {}).items():
        if name not in P.PROVIDERS:
            continue
        if value:
            SESSION_KEYS[name] = value.strip()
        else:
            SESSION_KEYS.pop(name, None)
    return await get_providers()


@app.post("/api/verify/{provider}")
async def verify(provider: str):
    """Cheap authenticated GET so the user knows a key works before a long run."""
    key = key_for(provider)
    if provider == "local":
        return {"ok": P.ffmpeg_available(),
                "detail": "ffmpeg found" if P.ffmpeg_available() else "ffmpeg not on PATH"}
    if not key:
        return {"ok": False, "detail": "No key set"}

    probes = {
        "replicate": ("GET", "https://api.replicate.com/v1/account",
                      {"Authorization": f"Bearer {key}"}),
        "elevenlabs": ("GET", "https://api.elevenlabs.io/v1/user",
                       {"xi-api-key": key}),
        "kits": ("GET", "https://arpeggi.io/api/kits/v1/voice-models",
                 {"Authorization": f"Bearer {key}"}),
        "lalalai": ("GET", "https://www.lalal.ai/api/limits/",
                    {"Authorization": f"license {key}", "X-License-Key": key}),
        "musicai": ("GET", "https://api.music.ai/api/job",
                    {"Authorization": key}),
    }
    if provider not in probes:
        return {"ok": False, "detail": "No probe for this provider"}

    method, url, headers = probes[provider]
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.request(method, url, headers=headers)
        ok = r.status_code < 400
        return {"ok": ok, "detail": "Key accepted" if ok
                else f"HTTP {r.status_code}: {r.text[:160]}"}
    except Exception as exc:
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}


@app.post("/api/jobs")
async def create_job(
    singer_audio: UploadFile = File(...),
    song_audio: UploadFile = File(...),
    config: str = Form("{}"),
):
    cfg = json.loads(config or "{}")
    cfg.setdefault("engines", {})
    for stage in P.STAGES:
        cfg["engines"].setdefault(stage, P.DEFAULT_ENGINES[stage])
        engine = cfg["engines"][stage]
        if engine not in P.PROVIDERS:
            raise HTTPException(400, f"Unknown provider '{engine}' for stage '{stage}'.")
        if stage not in P.PROVIDERS[engine]["stages"]:
            raise HTTPException(
                400, f"{P.PROVIDERS[engine]['label']} cannot handle the '{stage}' stage."
            )

    job = new_job(cfg)
    workdir = WORK / job["id"]
    workdir.mkdir(parents=True, exist_ok=True)

    singer = workdir / f"singer{Path(singer_audio.filename or 'ref.wav').suffix or '.wav'}"
    song = workdir / f"song{Path(song_audio.filename or 'song.mp3').suffix or '.mp3'}"
    singer.write_bytes(await singer_audio.read())
    song.write_bytes(await song_audio.read())

    cfg.setdefault("song_name", Path(song_audio.filename or "song").stem)
    cfg.setdefault("voice_name", Path(singer_audio.filename or "voice").stem)

    log(job, None, f"Queued. Song: {song_audio.filename} | Singer: {singer_audio.filename}")
    asyncio.create_task(run_pipeline(job, singer, song))
    return {"job_id": job["id"]}


@app.get("/api/jobs")
async def list_jobs():
    return {"jobs": [_summary(j) for j in
                     sorted(JOBS.values(), key=lambda j: j["created"], reverse=True)]}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "No such job")
    return job


@app.get("/api/jobs/{job_id}/file/{name}")
async def get_file(job_id: str, name: str):
    job = JOBS.get(job_id)
    if not job or name not in job["artifacts"]:
        raise HTTPException(404, "No such artifact")
    path = Path(job["artifacts"][name]["file"])
    if not path.exists():
        raise HTTPException(410, "File was removed")
    media = "audio/mpeg" if path.suffix == ".mp3" else "audio/wav"
    return FileResponse(path, media_type=media, filename=path.name)


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    JOBS.pop(job_id, None)
    shutil.rmtree(WORK / job_id, ignore_errors=True)
    return {"ok": True}


def _summary(job: dict) -> dict:
    return {k: job[k] for k in ("id", "created", "state", "error")} | {
        "song": job["config"].get("song_name"),
        "voice": job["config"].get("voice_name"),
        "stages": {s: job["stages"][s]["state"] for s in P.STAGES},
    }


@app.exception_handler(P.PipelineError)
async def pipeline_error_handler(_request, exc: P.PipelineError):
    return JSONResponse({"error": str(exc)}, status_code=400)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8765"))
    print(f"\n  Voice Dubbing Console  ->  http://{host}:{port}\n")
    if not P.ffmpeg_available():
        print("  WARNING: ffmpeg not found on PATH. The mix stage will fail.\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
