# Voice Dubbing Console

Upload a target singer and a target song. Get the song back, sung in that voice.

The dashboard fronts five vendor APIs and lets you pick which one runs each stage
of the pipeline, so you can A/B a cheap route against a high-quality one without
touching code.

```
  target song ──► 1. SPLIT STEMS ──┬──► vocals ──► 3. CONVERT ──┐
                                   │                            ├──► 4. MIX ──► final track
                                   └──► instrumental ───────────┘
  target singer ──► 2. CLONE VOICE ─────────────────────────────┘
```

---

## Setup

**1. Install ffmpeg** — required for every mix path.

- Windows: `winget install Gyan.FFmpeg` (then reopen your terminal)
- macOS: `brew install ffmpeg`
- Verify with `ffmpeg -version`

**2. Install the Python deps**

```bash
cd "AI voice clone"
pip install -r requirements.txt
```

**3. Add your keys** — copy `.env.example` to `.env` and fill in whichever
providers you'll use. You can also paste keys into the dashboard, but those
only last for the session.

**4. Run it**

```bash
python server.py
```

Open <http://127.0.0.1:8765>. Hit **Test all connections** to confirm your keys
work before burning credits on a long run.

---

## Who does what

| Provider | Split stems | Clone voice | Convert vocal | Mix | Get a key |
|---|:--:|:--:|:--:|:--:|---|
| **Replicate** | ✅ | ✅ | ✅ | — | [account/api-tokens](https://replicate.com/account/api-tokens) |
| **LALAL.AI** | ✅ | ✅ | ✅ | — | [lalal.ai/api](https://www.lalal.ai/api/) |
| **Kits.AI** | ✅ | ✅ | ✅ | — | [app.kits.ai/api-access](https://app.kits.ai/api-access) |
| **ElevenLabs** | — | ✅ | ✅ | — | [settings/api-keys](https://elevenlabs.io/app/settings/api-keys) |
| **Music.AI** | ✅ | — | — | ✅ | [music.ai](https://music.ai/) |
| **Local (ffmpeg)** | — | — | — | ✅ | free, no key |

You only need keys for the providers you actually route to. The cheapest working
setup is **Replicate for stages 1–3 + Local mix** — one key, one bill.

### Picking a route

- **Best singing quality** — Kits.AI convert. It trains a real voice model
  (3–10 min) and holds up on sustained notes where zero-shot models wobble.
- **Cleanest stems** — LALAL.AI separate. Noticeably fewer artifacts on dense mixes.
- **Cheapest / most tweakable** — Replicate everywhere. Zero-shot, no training wait.
- **Spoken dubbing or rap** — ElevenLabs convert. Strong on speech, weaker on melody.

Mixing providers is the point: LALAL.AI split → Kits.AI clone+convert → local mix
is a common "quality" preset.

---

## Input quality drives output quality

**Singer reference** — 30 seconds to 3 minutes of *clean solo vocal*. No backing
track, no reverb tail, no second voice. One good a cappella verse beats ten
minutes of full mix. This matters more than any setting in the UI.

**Target song** — a normal full mix is fine, that's what stage 1 is for.

**Pitch shift** — if the target singer's range is far from the original, shift in
whole steps. Male→female is often `+12`, female→male `-12`. Anything beyond ±12
tends to sound artificial.

---

## Files

| File | Purpose |
|---|---|
| `dashboard.html` | The whole UI — served by the backend at `/` |
| `server.py` | FastAPI orchestrator, job queue, artifact serving |
| `providers.py` | One adapter class per vendor; add new ones here |
| `.env` | Your keys (create from `.env.example`; never commit) |
| `output/` | Finished tracks, WAV + MP3 |
| `.work/` | Per-job scratch: stems, intermediates. Safe to delete |

Finished tracks land in `output/` as `<song> - <voice>.wav` and `.mp3`.

---

## Adding a provider

Everything routes through one factory. In `providers.py`:

1. Add an entry to `PROVIDERS` with `label`, `key_env`, and the `stages` it can serve.
2. Write a class with whichever of these it supports:
   - `async separate(client, song, workdir, log) -> StageResult`  (`files: vocals, instrumental`)
   - `async clone(client, samples, name, log) -> StageResult`  (`meta: voice_id`)
   - `async convert(client, vocals, voice, workdir, opts, log) -> StageResult`  (`files: converted`)
3. Add one line to `build()`.

The dropdowns, key vault, and validation all populate from `PROVIDERS` — no UI
changes needed.

Use the `_poll()` helper for any submit-then-wait API; it handles timeouts,
progress logging, and failure states for you.

---

## Troubleshooting

**"ffmpeg missing" in the header** — install it and restart the server. Nothing
finishes without it.

**Replicate 404 on a model** — community versions get republished. Open the model
page on Replicate, copy the full `owner/name:versionhash`, and paste it into
Advanced → *Replicate: convert*.

**Kits.AI clone sits at "training"** — normal, 3–10 minutes. To reuse a voice you
already trained, paste its id into Advanced → *Existing voice id* and the clone
stage gets skipped entirely.

**Robotic or warbly output** — almost always the reference audio. Re-cut it to a
clean solo vocal with no reverb before blaming the model.

**Vocal sits too far forward** — drop *Vocal level* 2–3 dB and add a little
reverb to seat it in the mix.

**Behind a corporate proxy** — already handled; `httpx[socks]` is in requirements.

---

## A note on use

Cloning a real singer's voice is legally and ethically loaded. Rights vary by
country and several jurisdictions have tightened up on voice likeness recently.
Get permission for anything you publish, and keep unreleased-artist experiments
private. Parody and personal use sit on much safer ground than commercial release.
