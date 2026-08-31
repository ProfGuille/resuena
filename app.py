"""LyricMix — API + servidor web.

Subís una canción (archivo o link de YouTube) y su letra; la app alinea la
letra con el audio (faster-whisper) y deja "pintar" frases. Cada usuario arma
su propia selección y genera un audio corto con solo las frases elegidas.
"""
import hashlib
import os
import queue
import re
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import align
import audio_utils as au
import cloud
import ghstore
import media
import store

BASE = Path(__file__).resolve().parent
STATIC_DIR = BASE / "static"
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE / "data")))
AUDIO_DIR = DATA_DIR / "audio"
WAV_DIR = DATA_DIR / "wav"
RENDER_DIR = DATA_DIR / "render"
for d in (AUDIO_DIR, WAV_DIR, RENDER_DIR):
    d.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
VERSION = "v8"   # marca de versión: aparece en /api/health y en el footer para verificar el deploy
ALIGN_VERSION = 2  # versión del pipeline de alineación: si una canción lista tiene
                   # align_v != 2, se re-analiza sola al arrancar (transcript mejorado)

ALLOWED_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".opus",
                ".webm", ".mp4", ".mov", ".mpeg", ".mpga", ".oga"}

_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel
            _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        return _model


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------- cola de análisis
# El CPU del plan free es limitado: se procesa UNA canción por vez.
_proc_lock = threading.Lock()
_proc_queue = queue.Queue()
_proc_worker_started = False


def process_song(sid):
    """Encola el análisis de una canción (se procesa de a una por vez)."""
    global _proc_worker_started
    with _proc_lock:
        if not _proc_worker_started:
            threading.Thread(target=_proc_worker, daemon=True).start()
            _proc_worker_started = True
    _proc_queue.put(sid)


def _proc_worker():
    while True:
        sid = _proc_queue.get()
        try:
            song = store.get_song(sid)
            if song is not None:
                _process_song_impl(sid)
        except Exception:
            pass
        finally:
            _proc_queue.task_done()


def _set_phase(sid, phase):
    try:
        song = store.get_song(sid)
        if song:
            song["phase"] = phase
            store.save_song(sid, song)
    except Exception:
        pass


def _upload_src_safe(sid, src_path, ext):
    """Guarda el archivo original en la nube (para poder retomar si se reinicia)."""
    if not media.persistent():
        return
    try:
        media.put_file(f"src/{sid}{ext}", src_path)
    except Exception:
        pass


def _recover_stuck_songs():
    """Al arrancar: canciones que quedaron 'processing' por un reinicio.
    Se re-encolan en la cola de análisis (una por vez); las que no tienen
    cómo retomarse se marcan con un mensaje claro."""
    try:
        for sid, song in store.all_songs().items():
            if song.get("status") != "processing":
                continue
            if song.get("source") == "youtube":
                process_song(sid)
            elif media.persistent():
                # el mp3 guardado (song/{sid}.mp3) permite retomar el análisis
                # aunque el original se haya perdido con el reinicio
                process_song(sid)
            else:
                song["status"] = "error"
                song["error"] = ("El análisis se interrumpió (el servidor se reinició "
                                 "durante el procesamiento). Volvé a subir la canción.")
                song["phase"] = None
                store.save_song(sid, song, durable=True)
    except Exception:
        pass


def _recheck_align():
    """Re-encola canciones ya listas cuya alineación se generó con el pipeline
    viejo (transcript menos preciso). Se re-analizan solas desde el mp3 ya
    guardado, sin que el usuario tenga que volver a subir la canción.

    IMPORTANTE: se marca status='processing' antes de encolar para que (a) el
    worker la procese (antes se saltaba las 'ready' y nunca se re-analizaba) y
    (b) la UI muestre el banner de análisis mientras trabaja.
    """
    try:
        for sid, song in store.all_songs().items():
            if song.get("status") == "ready" and song.get("align_v") != ALIGN_VERSION:
                song["status"] = "processing"
                song["phase"] = "encolado"
                song["error"] = None
                store.save_song(sid, song, durable=True)
                process_song(sid)
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app):
    store.init()          # memoria desde disco + GitHub (rápido, sin red en lecturas)
    _recover_stuck_songs()
    _recheck_align()
    yield


app = FastAPI(title="Resuena", lifespan=lifespan)


@app.middleware("http")
async def no_store_cache(request, call_next):
    """Sin caché: el navegador SIEMPRE baja la versión nueva de la página.
    (Evita que quede cacheado un index.html viejo después de un deploy.)"""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(Exception)
async def unhandled_error_handler(request, exc):
    """Nunca devolver un 500 con traceback crudo: JSON limpio y genérico."""
    return JSONResponse(status_code=500, content={"detail": "Error interno del servidor. Reintentá."})


def _ensure_audio(sid):
    """Asegura que el mp3 de la canción exista en disco (lo baja del backend
    externo si hace falta o si quedó vacío por una descarga fallida)."""
    f = AUDIO_DIR / f"{sid}.mp3"
    if (not f.exists() or f.stat().st_size == 0) and media.persistent():
        media.get_file(f"song/{sid}.mp3", str(f))
    return f


def _ensure_render(sid, fname):
    f = RENDER_DIR / f"{sid}_{fname}"
    if (not f.exists() or f.stat().st_size == 0) and media.persistent():
        media.get_file(f"render/{sid}/{fname}", str(f))
    return f


# ---------------------------------------------------------------- frontend
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------- canciones
@app.get("/api/health")
def health():
    storage = "github" if ghstore.enabled() else ("r2" if cloud.cloud_enabled() else "local")
    return {"ok": True, "model": WHISPER_MODEL, "storage": storage, "version": VERSION}


@app.get("/api/songs")
def list_songs():
    out = []
    for s in store.all_songs().values():
        out.append({
            "id": s["id"],
            "title": s.get("title") or "Sin título",
            "artist": s.get("artist"),
            "status": s.get("status"),
            "phase": s.get("phase"),
            "error": s.get("error"),
            "duration": s.get("duration"),
            "source": s.get("source"),
            "created_at": s.get("created_at"),
            "coverage": s.get("coverage"),
            "word_count": s.get("word_count"),
        })
    out.sort(key=lambda s: s["created_at"], reverse=True)
    return {"songs": out}


@app.get("/api/songs/{sid}")
def get_song(sid: str):
    song = store.get_song(sid)
    if not song:
        raise HTTPException(404, "Canción no encontrada")
    return song


@app.post("/api/songs")
async def create_song(
    lyrics: str = Form(...),
    title: str = Form(""),
    artist: str = Form(""),
    language: str = Form(""),
    youtube_url: str = Form(""),
    file: UploadFile = File(None),
    background_tasks: BackgroundTasks = None,
):
    lyrics = (lyrics or "").strip()
    if not lyrics:
        raise HTTPException(400, "La letra es obligatoria")
    url = (youtube_url or "").strip()
    if not file and not url:
        raise HTTPException(400, "Subí un archivo de audio o pegá un link de YouTube")

    sid = uuid.uuid4().hex[:10]
    song = {
        "id": sid,
        "title": title.strip(),
        "artist": artist.strip(),
        "lyrics": lyrics,
        "language": language.strip() or None,
        "source": "file" if file else "youtube",
        "youtube_url": url or None,
        "status": "processing",
        "phase": "encolado",
        "error": None,
        "duration": None,
        "created_at": now_iso(),
        "lines": None,
        "coverage": None,
        "word_count": None,
        "detected_lang": None,
        "transcript_words": None,
        "source_path": None,
    }

    src = None
    try:
        if file:
            ext = os.path.splitext(file.filename or "")[1].lower()
            if ext not in ALLOWED_EXTS:
                raise HTTPException(400, f"Formato no soportado: {ext or 'desconocido'}")
            src = AUDIO_DIR / f"{sid}_src{ext}"
            with open(src, "wb") as fh:
                shutil.copyfileobj(file.file, fh)
            song["source_path"] = str(src)
            song["source_ext"] = ext
            # subir el original a la nube en segundo plano (no bloquea la respuesta)
            background_tasks.add_task(_upload_src_safe, sid, str(src), ext)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"No se pudo guardar el archivo: {e}")

    store.save_song(sid, song, durable=True)
    process_song(sid)
    return {"id": sid}


@app.delete("/api/songs/{sid}")
def delete_song(sid: str):
    if not store.get_song(sid):
        raise HTTPException(404, "Canción no encontrada")
    store.delete_song(sid)
    for d in (AUDIO_DIR, WAV_DIR, RENDER_DIR):
        for f in d.glob(f"{sid}*"):
            try:
                f.unlink()
            except OSError:
                pass
    if media.persistent():
        media.delete_prefix(f"song/{sid}")
        media.delete_prefix(f"render/{sid}/")
    return {"ok": True}


@app.get("/api/songs/{sid}/audio")
def song_audio(sid: str):
    if not store.get_song(sid):
        raise HTTPException(404, "Canción no encontrada")
    f = _ensure_audio(sid)
    if not f.exists() or f.stat().st_size == 0:
        raise HTTPException(
            404,
            "El audio de la canción no está disponible (el servidor se reinició). "
            "Volvé a abrir la canción para recuperarlo, o subila de nuevo.",
        )
    return FileResponse(str(f), media_type="audio/mpeg")


# ---------------------------------------------------------------- selección
@app.get("/api/songs/{sid}/selection")
def get_selection(sid: str, user_id: str = "anon"):
    if not store.get_song(sid):
        raise HTTPException(404, "Canción no encontrada")
    sel = store.get_selection(sid, user_id)
    return {"ranges": sel.get("ranges", []) if sel else []}


@app.post("/api/songs/{sid}/selection")
def set_selection(sid: str, payload: dict):
    if not store.get_song(sid):
        raise HTTPException(404, "Canción no encontrada")
    user_id = str(payload.get("user_id") or "anon")[:64]
    ranges = payload.get("ranges") or []
    clean = []
    for r in ranges:
        if isinstance(r, (list, tuple)) and len(r) == 2:
            try:
                clean.append([int(r[0]), int(r[1])])
            except (TypeError, ValueError):
                pass
    store.save_selection(sid, user_id, clean)
    return {"ok": True}


# ---------------------------------------------------------------- render
def _prune_renders(sid, max_files=60):
    """Si hay demasiados renders de una canción, borra los más viejos."""
    files = sorted(RENDER_DIR.glob(f"{sid}_*.mp3"),
                   key=lambda p: p.stat().st_mtime)
    for old in files[:-max_files]:
        try:
            old.unlink()
        except OSError:
            pass
    if ghstore.enabled():
        prefix = f"render/{sid}/"
        for name in ghstore.list_dir(prefix)[:-max_files]:
            ghstore.delete_path(prefix + name)
    if cloud.cloud_enabled():
        keys = cloud.list_keys(f"render/{sid}/")
        for key, _lm in keys[:-max_files]:
            try:
                cloud.delete_key(key)
            except Exception:
                pass


def _flat(song):
    flat = []
    for li, line in enumerate(song["lines"] or []):
        for wi, w in enumerate(line["words"]):
            flat.append((li, wi, w))
    return flat


@app.post("/api/songs/{sid}/render")
def render(sid: str, payload: dict):
    song = store.get_song(sid)
    if not song:
        raise HTTPException(404, "Canción no encontrada")
    if song["status"] != "ready":
        raise HTTPException(400, "La canción todavía no está lista")

    ranges = payload.get("ranges") or []
    user_id = str(payload.get("user_id") or "anon")[:64]
    all_occ = bool(payload.get("all_occurrences", False))
    if not ranges:
        raise HTTPException(400, "No hay frases seleccionadas")

    flat = _flat(song)
    total = len(flat)
    if total == 0:
        raise HTTPException(400, "La letra está vacía")
    dur = song.get("duration") or 0.0
    transcript = song.get("transcript_words") or []

    segments = []
    skipped = 0
    omitted_words = 0
    for a, b in ranges:
        a = max(0, int(a))
        b = min(total - 1, int(b))
        if a > b:
            continue
        words = [w for (_, _, w) in flat[a:b + 1]]
        omitted_words += sum(1 for w in words if not w.get("m"))
        # dividir la selección en tramos contiguos de palabras CON audio;
        # los huecos sin audio detectado se saltean (no invaden el audio final)
        runs = []
        cur = []
        for w in words:
            if w.get("m"):
                cur.append(w)
            else:
                if cur:
                    runs.append(cur)
                    cur = []
        if cur:
            runs.append(cur)
        if not runs:
            skipped += 1
            continue

        for run in runs:
            if all_occ:
                tjs = [w.get("tj") for w in run if w.get("tj") is not None]
                if not tjs:
                    s0 = max(0.0, run[0]["s"] - 0.15)
                    e0 = min(dur, run[-1]["e"] + 0.25)
                    if e0 - s0 > 0.05:
                        segments.append((s0, e0))
                    continue
                j0, j1 = min(tjs), max(tjs)
                for k0, k1, _ in align.find_occurrences(transcript, j0, j1):
                    s0 = max(0.0, transcript[k0]["start"] - 0.15)
                    e0 = min(dur, transcript[k1]["end"] + 0.25)
                    if e0 - s0 > 0.05:
                        segments.append((s0, e0))
            else:
                n = len(run)
                if n <= 2:
                    # selección corta (una/dos palabras): cortar EXACTO,
                    # debe sonar solo lo seleccionado, sin añadir aire
                    pad_before = 0.0
                    pad_after = 0.0
                else:
                    # frases largas: un pequeño margen natural de respiración
                    pad_before = 0.15
                    pad_after = 0.25
                s0 = max(0.0, run[0]["s"] - pad_before)
                e0 = min(dur, run[-1]["e"] + pad_after)
                if e0 - s0 > 0.05:
                    segments.append((s0, e0))

    if skipped and not segments:
        raise HTTPException(
            400,
            "Ninguna frase seleccionada tiene audio asociado. "
            "Revisá que la letra coincida con lo que se canta en el audio.",
        )

    segments.sort()
    merged = []
    for seg in segments:
        if merged and seg[0] < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
        else:
            merged.append(seg)
    segments = merged

    if not segments:
        raise HTTPException(400, "No se pudo generar audio con las frases elegidas")

    uid = hashlib.md5(user_id.encode("utf-8")).hexdigest()[:8]
    # URL única por contenido: evita que el navegador sirva audio cacheado viejo
    chash = hashlib.md5(repr(segments).encode("utf-8")).hexdigest()[:8]
    fname = f"{uid}_{chash}.mp3"
    out = RENDER_DIR / f"{sid}_{fname}"
    if not out.exists():
        au.render_phrases(str(_ensure_audio(sid)), segments, str(out))
        # subir a la nube en segundo plano: el GET sirve desde disco y no
        # bloquea el POST (GitHub puede tardar segundos con archivos grandes)
        threading.Thread(target=_upload_render_safe,
                         args=(sid, fname, str(out)), daemon=True).start()
    return {
        "url": f"/api/songs/{sid}/render/{fname}",
        "segments": len(segments),
        "duration": au.ffprobe_duration(str(out)),
        "skipped": skipped,
        "omitted": omitted_words,
    }


def _upload_render_safe(sid, fname, local_path):
    """Sube el render a la nube y poda los renders viejos. Nunca lanza."""
    try:
        if media.persistent():
            media.put_file(f"render/{sid}/{fname}", local_path)
            _prune_renders(sid)
    except Exception:
        pass


@app.get("/api/songs/{sid}/render/{fname}")
def render_file(sid: str, fname: str):
    if not re.fullmatch(r"[0-9a-f]{8}_[0-9a-f]{8}\.mp3", fname):
        raise HTTPException(404, "Audio no encontrado")
    f = _ensure_render(sid, fname)
    if not f.exists() or f.stat().st_size == 0:
        raise HTTPException(
            404,
            "El audio se perdió (el servidor se reinició). "
            "Tocá de nuevo \"Generar audio con mis frases\".",
        )
    return FileResponse(str(f), media_type="audio/mpeg",
                        headers={"Cache-Control": "no-store"},
                        filename=f"frases_{sid}.mp3")


# ---------------------------------------------------------------- proceso
def _process_song_impl(sid):
    """Corre en segundo plano: descarga, convierte, transcribe y alinea."""
    song = store.get_song(sid)
    try:
        _set_phase(sid, "descargando")
        src = song.get("source_path")
        if song["source"] == "youtube":
            path, yt_title, yt_artist = au.download_youtube(
                song["youtube_url"], AUDIO_DIR, sid)
            src = path
            song["source_path"] = str(src)
            if not song["title"] and yt_title:
                song["title"] = yt_title[:120]
            if not song["artist"] and yt_artist:
                song["artist"] = yt_artist[:80]
            store.save_song(sid, song)

        if song["source"] != "youtube":
            # si el original no está en disco (p.ej. reinicio), bajarlo de la nube
            if (not src or not os.path.exists(str(src))) and media.persistent() and song.get("source_ext"):
                tmp = AUDIO_DIR / f"{sid}_src{song['source_ext']}"
                if media.get_file(f"src/{sid}{song['source_ext']}", str(tmp)):
                    src = str(tmp)
                    song["source_path"] = str(src)
                    store.save_song(sid, song)
            if not src or not os.path.exists(str(src)):
                # reproceso: usar el mp3 ya guardado; si no está en disco
                # (el disco efímero se borra en cada deploy), bajarlo de la nube
                mp3_ok = _ensure_audio(sid)
                if mp3_ok.exists() and mp3_ok.stat().st_size > 0:
                    src = str(mp3_ok)
                    song["source_path"] = str(src)
                    store.save_song(sid, song)
                else:
                    raise RuntimeError(
                        "El archivo de audio original no está disponible "
                        "(el servidor se reinició durante el análisis). "
                        "Volvé a subir la canción.")

        wav = WAV_DIR / f"{sid}.wav"
        mp3 = AUDIO_DIR / f"{sid}.mp3"
        # reutilizar el wav ya convertido (el disco efímero a veces sobrevive
        # entre restarts del mismo proceso) y evitar reconvertir si el mp3 es
        # la fuente (re-análisis: ya es mp3, no hace falta volver a convertirlo)
        if not (wav.exists() and wav.stat().st_size > 0):
            au.to_wav16k(str(src), str(wav))
        if str(src) != str(mp3):
            au.to_streaming_mp3(str(src), str(mp3))
        song["duration"] = au.ffprobe_duration(str(mp3))
        # en re-análisis (src == mp3) el mp3 no cambió: no hace falta re-subirlo
        if media.persistent() and str(src) != str(mp3):
            _set_phase(sid, "guardando")
            media.put_file(f"song/{sid}.mp3", str(mp3))
        # el archivo fuente original ya no hace falta (se usó para convertir)
        try:
            if os.path.exists(str(src)) and str(src) != str(mp3):
                os.remove(str(src))
        except OSError:
            pass

        _set_phase(sid, "transcribiendo")
        model = get_model()
        lang = song["language"] or None
        # vad_filter=False: el filtro de voz descartaba partes cantadas de las
        # canciones (frases que quedaban "sin audio"). Para música es mejor
        # transcribir todo y alinear después.
        # beam_size bajo (por defecto 1): mucho más rápido en CPU, suficiente
        # precisión para alinear letras; se puede subir con WHISPER_BEAM.
        beam = int(os.environ.get("WHISPER_BEAM", "1"))
        # initial_prompt con el comienzo de la letra: mejora muchísimo la
        # transcripción de canciones (Whisper ya "conoce" las palabras que se
        # cantan; sin esto el modelo tiny inventa palabras en el audio cantado).
        prompt = None
        if song.get("lyrics"):
            prompt = " ".join(song["lyrics"].split())[:900]
        seg_iter, info = model.transcribe(
            str(wav), language=lang, word_timestamps=True,
            vad_filter=False, beam_size=beam,
            condition_on_previous_text=False,
            initial_prompt=prompt,
        )
        segs = []
        words = []
        for seg in seg_iter:
            segs.append({"text": seg.text or "",
                         "start": float(seg.start), "end": float(seg.end)})
            for w in (seg.words or []):
                ww = (w.word or "").strip()
                ww = ww.strip(" .,;:!?¡¿\"'()[]«»-–—…♪♫")
                if ww:
                    words.append({"word": ww, "start": float(w.start),
                                  "end": float(w.end)})
        if not words:
            raise RuntimeError(
                "No se detectó voz en el audio. Probá otro audio o uno con mejor calidad.")

        lines = align.parse_lyrics(song["lyrics"])
        lines, coverage = align.align_lines(lines, words, segs)

        song["lines"] = lines
        song["coverage"] = round(coverage, 3)
        song["word_count"] = sum(len(l["words"]) for l in lines)
        song["detected_lang"] = info.language
        song["transcript_words"] = words
        song["align_v"] = ALIGN_VERSION
        song["status"] = "ready"
        song["phase"] = "listo"
        store.save_song(sid, song, durable=True)
    except Exception as e:
        song = store.get_song(sid)
        song["status"] = "error"
        song["error"] = str(e)[:500]
        song["phase"] = None
        store.save_song(sid, song, durable=True)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
