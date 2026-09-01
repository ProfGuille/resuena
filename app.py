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
import subprocess
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
import ffmpeg_util
import ghstore
import media
import numpy as np
import store

BASE = Path(__file__).resolve().parent
STATIC_DIR = BASE / "static"
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE / "data")))
AUDIO_DIR = DATA_DIR / "audio"
WAV_DIR = DATA_DIR / "wav"
RENDER_DIR = DATA_DIR / "render"
for d in (AUDIO_DIR, WAV_DIR, RENDER_DIR):
    d.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "tiny")
VERSION = "v24"  # marca de versión: aparece en /api/health y en el footer para verificar el deploy
ALIGN_VERSION = 3  # versión del pipeline de alineación: si una canción lista tiene
                   # align_v != 3, se re-analiza sola al arrancar (anclas dispersas corregidas)

ALLOWED_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".opus",
                ".webm", ".mp4", ".mov", ".mpeg", ".mpga", ".oga"}

_model = None
_model_lock = threading.Lock()

def _container_mem_mb():
    """RAM REAL disponible para ESTE contenedor.

    Render free limita a 512 MB vía cgroup. /proc/meminfo reporta la RAM del
    HOST (no la del contenedor), así que hay que leer el límite del cgroup:
      - cgroup v2: /sys/fs/cgroup/memory.max
      - cgroup v1: /sys/fs/cgroup/memory/memory.limit_in_bytes
    Devuelve None si no se puede determinar (se asume lo peor: 512 MB).
    """
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as fh:
                v = fh.read().strip()
            if v and v != "max":
                return int(v) // (1024 * 1024)
        except Exception:
            continue
    return None

def _pick_model():
    """Elige el modelo de Whisper que quepa en la RAM REAL del contenedor.

    Render free = 512 MB. 'small' (~1 GB) o 'base' (~500 MB con la app)
    causan OOM y matan el proceso a mitad de la transcripción. Regla segura:
      - RAM < 900 MB (o desconocida) -> SIEMPRE 'tiny' (~75 MB), sin importar
        lo que pida la variable de entorno.
      - RAM >= 900 MB -> respeta lo pedido.
    """
    requested = os.environ.get("WHISPER_MODEL", "tiny")
    mem_mb = _container_mem_mb()
    if mem_mb is None:
        print("RAM del contenedor desconocida: usando tiny (seguro)", flush=True)
        return "tiny"
    if mem_mb < 900 and requested != "tiny":
        print(f"RAM del contenedor: {mem_mb} MB -> usando tiny (cabe en 512 MB)", flush=True)
        return "tiny"
    return requested

def get_model():
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel
            chosen = _pick_model()
            # si el modelo vino horneado en el build (models/), cargarlo desde
            # ahí: sin red, sin pico de memoria por descarga, arranque rápido
            local = BASE / "models" / f"faster-whisper-{chosen}"
            if local.exists():
                print(f"cargando modelo Whisper: {chosen} (desde build)", flush=True)
                _model = WhisperModel(str(local), device="cpu", compute_type="int8")
            else:
                print(f"cargando modelo Whisper: {chosen} (descarga)", flush=True)
                _model = WhisperModel(chosen, device="cpu", compute_type="int8")
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

MAX_RETRIES = 3

def _recover_stuck_songs():
    """Al arrancar: canciones que quedaron 'processing' por un reinicio.
    Se re-encolan en la cola de análisis (una por vez) PERO con un límite de
    reintentos: si el servidor se reinició varias veces seguidas (p.ej. OOM
    por RAM en el plan free), se marca error claro en vez de quedar en loop
    infinito de reintentos."""
    try:
        for sid, song in store.all_songs().items():
            if song.get("status") != "processing":
                continue
            retries = song.get("retry_count", 0) + 1
            song["retry_count"] = retries
            if retries > MAX_RETRIES:
                song["status"] = "error"
                song["error"] = ("El análisis se interrumpió varias veces (el servidor "
                                 "del plan free se reinicia cuando se queda sin memoria). "
                                 "Borrá la canción y volvé a subirla, o intentá más tarde.")
                song["phase"] = None
                store.save_song(sid, song, durable=True)
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
                if song.get("retry_count", 0) > MAX_RETRIES:
                    continue  # ya se intentó demasiadas veces: no reintentar
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

    # warm-up del modelo en segundo plano: la descarga/carga del modelo Whisper
    # empieza apenas arranca el proceso (y queda lista para la primera canción),
    # en vez de hacerse en medio del procesamiento (que causaba picos de memoria
    # y 502). Si el modelo está horneado en el build, esto es instantáneo.
    def _warm():
        try:
            get_model()
            print("modelo Whisper listo (warm-up)", flush=True)
        except Exception as e:
            print("warm-up del modelo falló: " + str(e)[:200], flush=True)
    threading.Thread(target=_warm, daemon=True).start()
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
    return {"ok": True, "model": _pick_model(), "storage": storage, "version": VERSION}

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

def _speech_tail(src, s0, e0, dur, transcript, next_start=None):
    """Extiende el final del corte hasta donde TERMINA la voz de verdad.

    Solo se llama cuando la última palabra del tramo tiene timestamps
    FALSOS (quedó interpolada: sim=0.5, sin match en el transcript), que es
    el caso donde el corte podía caer antes de que la palabra termine. El
    modelo tiny a veces transcribe mal la voz cantada ("incondicional"
    puede quedar como "sesionada"): el TEXTO es basura pero la POSICIÓN de
    esa palabra en el transcript es real.

    Estrategia:
    1. Usa como guía la palabra del transcript cuya posición cae en la cola
       de la selección (aunque su texto esté mal).
    2. Analiza la energía del audio: encuentra el ÚLTIMO pico de voz fuerte
       y lleva el corte hasta que la energía decae de forma sostenida.
    3. Nunca pasa del inicio de la siguiente palabra (next_start) ni del
       final de la canción.
    Si algo falla o no hay señal clara, devuelve el valor original.
    """
    try:
        limit = float(dur)
        if next_start and float(next_start) > e0 + 0.05:
            limit = min(limit, float(next_start) - 0.05)
        limit = min(limit, e0 + 2.0)
        if limit <= e0 + 0.01:
            return e0
        # 1) palabra del transcript cuya posición cae en la cola de la selección
        e_base = e0
        for w in transcript:
            try:
                st = float(w.get("start") or 0)
                en = float(w.get("end") or 0)
            except Exception:
                continue
            if e0 - 1.2 <= st <= e0 + 0.2 and en > e0 + 0.15 and en > e_base:
                e_base = en
        window = min(limit, (e_base + 0.8) if e_base > e0 + 0.3 else (e0 + 1.5))
        if window <= e0:
            return e0
        start = max(0.0, min(s0, e0 - 0.05))
        end = max(window, e0 + 0.1)
        if end - start < 0.2:
            return e0
        ff = ffmpeg_util.ensure_ffmpeg()
        r = subprocess.run(
            [ff + "/ffmpeg", "-v", "error", "-ss", f"{start:.3f}",
             "-to", f"{end:.3f}", "-i", src,
             "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
            capture_output=True,
        )
        x = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        win, hop = 480, 160          # ventana 30 ms, paso 10 ms
        n = (len(x) - win) // hop + 1
        if n < 20:
            return e0
        rms = np.empty(n)
        for k in range(n):
            seg = x[k * hop:k * hop + win]
            rms[k] = float(np.sqrt(np.mean(seg * seg)))
        # pico de energía de la PALABRA seleccionada [s0..e0] (voz, no fondo)
        i_s0 = max(0, int(round((s0 - start) / 0.01)))
        i_e0 = max(i_s0 + 1, min(n - 1, int(round((e0 - start) / 0.01))))
        peak = float(rms[i_s0:i_e0 + 1].max())
        if peak < 1e-4:
            return e0
        thr = max(0.008, 0.55 * peak)   # la voz domina sobre el fondo ~2x
        k0 = max(0, int(round((e0 - start) / 0.01)))
        # 2) último pico de voz en la ventana (ignora micro-pausas previas)
        kend = min(n, k0 + 150)         # hasta 1.5 s después del final estimado
        kmax = k0
        for k in range(k0 + 1, kend):
            if rms[k] > rms[kmax]:
                kmax = k
        if rms[kmax] < thr:
            return e0                   # no hay voz después: conservar el corte
        # caída sostenida (>= 5 ventanas = 50 ms) tras el pico: fin de la voz
        fin = None
        bajo = 0
        for k in range(kmax, n):
            if rms[k] < thr:
                bajo += 1
                if bajo >= 5:
                    fin = k - bajo + 1
                    break
            else:
                bajo = 0
        if fin is None:
            return e0                   # sin señal clara: conservar el corte
        e_new = start + fin * 0.01 + 0.06
        e_new = min(max(e_new, e0), limit)
        return e_new
    except Exception:
        return e0

def _is_real_anchor(w):
    """¿La palabra tiene un match REAL en el transcript?

    Las palabras "m=True pero tj=None / sim=0.5" tienen timestamps FALSOS
    (interpolados): no sirven como anclas de posición.
    """
    return bool(w.get("m")) and w.get("tj") is not None \
        and w.get("sim", 1.0) > 0.55 \
        and w.get("s") is not None and w.get("e") is not None

_BREATH_RE = re.compile(r"inhal|exhal|respira", re.I)

def _is_breath(w):
    """¿La palabra del transcript es una respiración (no voz cantada)?

    faster-whisper transcribe los suspiros/respiraciones como "Inhalación" /
    "Exhalación" o tokens sin texto. No sirven para ubicar dónde canta una
    frase (alargarían el segmento con aire que no es la frase).
    """
    t = (w.get("word") or "").strip()
    return (not t) or bool(_BREATH_RE.search(t)) or t in ("♪", "♫", "…", "...")

# ---------------------------------------------------------------------------
# Fronteras de voz por ENERGÍA real (v20)
#
# Causa raíz de "la frase siguiente empieza con el final de la anterior":
# los timestamps de palabra del transcript (whisper) cortan la COLA final de
# cada palabra (la vocal/transición que sigue sonando). Como las frases son
# contiguas, la frase N+1 arranca exactamente en ese fin recortado y arrastra
# la cola de la N (y la N sola suena recortada al final). No se tocan los
# timestamps: las fronteras se definen con la energía real del audio.
#   - inicio de frase  = el valle de energía justo antes de la primera subida
#                        sostenida de voz (arranque de la primera palabra);
#   - final de frase   = el último valle antes del arranque de la frase
#                        siguiente, o el final de la nota sostenida si la
#                        frase siguiente está lejos.
# ---------------------------------------------------------------------------

_ENV_CACHE = {}
_ENV_LOCK = threading.Lock()

def _rms_env(src):
    """Envolvente RMS de 10 ms (16 kHz mono) de TODO el audio, cacheada por
    canción. Se usa para medir dónde empieza/termina la voz de verdad."""
    key = os.path.abspath(str(src))
    with _ENV_LOCK:
        if key in _ENV_CACHE:
            return _ENV_CACHE[key]
    ff = ffmpeg_util.ensure_ffmpeg()
    r = subprocess.run(
        [ff + "/ffmpeg", "-v", "error", "-i", str(src),
         "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True,
    )
    x = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    win, hop = 480, 160          # ventana 30 ms, paso 10 ms
    n = (len(x) - win) // hop + 1
    rms = np.empty(n, dtype=np.float64)
    for k in range(n):
        seg = x[k * hop:k * hop + win]
        rms[k] = float(np.sqrt(np.mean(seg * seg)))
    with _ENV_LOCK:
        if len(_ENV_CACHE) >= 6:
            _ENV_CACHE.pop(next(iter(_ENV_CACHE)))
        _ENV_CACHE[key] = rms
    return rms

def _idx(t):
    return int(round(float(t) / 0.01))

def _first_rise_event(env, t0, t1):
    """Primer valle local en [t0,t1] seguido de una subida >= 20 % en <= 4
    ventanas (40 ms). Devuelve el tiempo del valle o None."""
    i0 = max(1, _idx(t0))
    i1 = min(len(env) - 2, _idx(t1))
    if i1 - i0 < 5:
        return None
    for k in range(i0, i1 + 1):
        v = env[k]
        if v <= env[k - 1] and v <= env[k + 1]:
            target = 1.20 * v
            for kk in range(k + 1, min(k + 5, i1 + 1)):
                if env[kk] >= target:
                    return k * 0.01
    return None

def _last_rise_event(env, t0, t1, factor=1.20):
    """Último valle local en [t0,t1] seguido de una subida >= factor en <= 4
    ventanas. Devuelve el tiempo del valle o None."""
    i0 = max(1, _idx(t0))
    i1 = min(len(env) - 2, _idx(t1))
    last = None
    if i1 - i0 < 5:
        return None
    for k in range(i0, i1 + 1):
        v = env[k]
        if v <= env[k - 1] and v <= env[k + 1]:
            target = factor * v
            for kk in range(k + 1, min(k + 5, i1 + 1)):
                if env[kk] >= target:
                    last = k * 0.01
                    break
    return last

def _seam_before(env, w0, w1, first_s=None, floor=0.02):
    """Frontera (valle de costura) entre dos frases dentro de [w0, w1]: el
    último mínimo local antes del PRIMER ataque real de voz.

    Un ataque real de frase es un pico local que supera ~1.5 x el PISO LOCAL
    (el mínimo de los ~100 ms previos). Ese criterio distingue el arranque de
    una sílaba nueva de las fluctuaciones de una cola en decaimiento: la cola
    mantiene su propio nivel (el pico ~ la base, ratio ~1) mientras que un
    ataque sube desde un valle (ratio > 1.5). Funciona también cuando el
    ataque sube DESDE una cola sonora (sin silencio entre frases), que es el
    caso que el promedio de ventana rechazaba.

    Si el ataque está ANTES del timestamp de la primera palabra del transcript
    (whisper lo puso tarde sobre la cola), se exige además energía sostenida
    después (media de los ~300 ms posteriores >= 0.45 x el pico): la frase
    continúa, no es un release/consonante de la palabra anterior.

    Devuelve el ÚLTIMO mínimo local antes de ese ataque (el valle de costura),
    o None si no hay ataque claro (el transcript queda como está).
    """
    i0 = max(1, _idx(w0))
    i1 = min(len(env) - 3, _idx(w1))
    if i1 - i0 < 12:
        return None
    lo_min_lim = max(w0, first_s - 0.05) if first_s is not None else w0
    for k in range(i0 + 2, i1 + 1):
        v = float(env[k])
        if v < floor:
            continue
        if not (v >= env[k - 1] and v >= env[k + 1]):
            continue
        # pico local: se usa SU valor (no el de la ventana hacia adelante:
        # eso hacía que un futuro ataque fuerte marcara un micro-pico de la
        # cola y la costura saliera temprana, p. ej. 38.90 en vez de 38.97).
        kf0 = max(1, k - 14)
        kf1 = max(kf0, k - 5)
        base = float(env[kf0:kf1 + 1].min())
        if base <= 0 or v < 1.5 * base:
            continue
        # el ataque real arranca desde un VALLE previo (dip): si no hay valle
        # en los ~170 ms anteriores, es una fluctuación de una nota o de la
        # cola de la frase anterior (p. ej. el pico de "aurora" dentro de la
        # frase previa), no un inicio de frase.
        b = None
        for j in range(k - 1, max(i0 - 1, k - 17), -1):
            if env[j] <= env[j - 1] and env[j] <= env[j + 1]:
                b = j
                break
        if b is not None and float(env[b]) <= 0.75 * v:
            pass  # ataque con valle previo claro
        elif base <= 0.04 and v >= 2.0 * base:
            # sin valle en la ventana, pero el ataque sube desde un piso
            # REALMENTE bajo (la frase arranca directo desde silencio o cola
            # mínima): aceptar y usar el inicio de la ventana como costura
            # (p. ej. "Se" de "Se fue..." con su valle justo antes de la
            # ventana). Una fluctuación de cola NO cumple: su base es el
            # nivel de la cola (no baja).
            b = i0
        else:
            continue
        if k * 0.01 < lo_min_lim:
            after = float(env[k + 1:min(len(env), k + 31)].mean())
            if after < 0.45 * v:
                continue
        return b * 0.01
    return None

def _onset_after_tail(env, lo, first_s):
    """Frontera de arranque con cola PEGADA o transcript SOLAPADO (gap ~ 0 o
    negativo): la cola de la palabra anterior quedó absorbida por la primera
    palabra de la frase siguiente, así que el transcript no sirve. Ver
    _seam_before: la frontera es el último valle antes del primer ataque."""
    lo = max(lo, first_s)
    start = max(0.0, lo - 0.08)
    w1 = min(lo + 0.70, max(first_s, lo) + 0.65)
    return _seam_before(env, start, w1, first_s=first_s)

def _strong_onset(env, t0, t1, floor):
    """Primer ataque FUERTE de voz en [t0, t1]: un pico local que (a) supera
    el piso absoluto `floor` y (b) sube >= 1.6x el mínimo de los ~150 ms
    anteriores (el piso previo al ataque). Devuelve el último valle antes de
    ese pico, o None. Sirve para recuperar arranques donde whisper quedó
    temprano sobre silencio/instrumental (aire muerto al inicio)."""
    i0 = max(1, _idx(t0))
    i1 = min(len(env) - 2, _idx(t1))
    if i1 - i0 < 10:
        return None
    pk = None
    for k in range(i0 + 1, i1 + 1):
        v = env[k]
        if v >= floor and v >= env[k - 1] and v >= env[k + 1]:
            base = float(env[max(i0, k - 15):k].min())
            if base > 0 and v >= 1.6 * base:
                pk = k
                break
    if pk is None:
        return None
    valley = None
    for k in range(i0, pk):
        if env[k] <= env[k - 1] and env[k] <= env[k + 1]:
            valley = k
    if valley is None:
        return None
    return valley * 0.01

def _decays_quickly(env, t, dur=0.25):
    """¿La energía decae a menos de la mitad del pico en `dur` segundos?
    Se usa para distinguir la SÍLABA FINAL de una frase (decae: pertenece a
    la frase) del ATAQUE de la frase siguiente (se sostiene: la costura es el
    valle anterior)."""
    i0 = max(1, _idx(t))
    if i0 >= len(env) - 1:
        return True
    peak = float(env[i0])
    if peak < 0.02:
        return True
    i1 = min(len(env) - 1, i0 + int(dur / 0.01))
    for k in range(i0 + 1, i1 + 1):
        if env[k] < 0.5 * peak:
            return True
    return False

def _first_attack_in(env, t0, t1, ratio=1.5, floor=0.02):
    """Primer ataque fuerte (pico local >= ratio x el mínimo de los ~150 ms
    anteriores, >= floor) dentro de [t0, t1]. Devuelve el tiempo del pico."""
    i0 = max(1, _idx(t0))
    i1 = min(len(env) - 2, _idx(t1))
    if i1 - i0 < 10:
        return None
    for k in range(i0 + 1, i1 + 1):
        v = env[k]
        if v >= floor and v >= env[k - 1] and v >= env[k + 1]:
            base = float(env[max(i0, k - 15):k].min())
            if base > 0 and v >= ratio * base:
                return k * 0.01
    return None

def _note_tail_end(env, e0, cap):
    """Fin de una nota final SOSTENIDA que el corte quedó cortando a mitad
    (p. ej. un \"paz\" largo al final de la frase). Desde el punto de corte
    hacia adelante: si la voz sigue >= 50 % del pico local hasta el tope, la
    nota llega al tope (fin = cap); si no, el fin es justo después del último
    tramo que supera ese 50 %. Devuelve un tiempo o None."""
    i_e = max(1, _idx(e0))
    i_cap = min(len(env) - 2, _idx(cap))
    if i_cap - i_e < 10:
        return None
    w = env[max(0, i_e - 30):min(len(env), i_e + 10) + 1]
    pk = float(w.max()) if w.size else 0.0
    if pk < 0.02:
        return None
    half = 0.5 * pk
    # (v23) no puentear un corte real: si entre la voz y el tope hay un
    # mínimo profundo (< 0.30 x el pico local), la nota terminó antes y lo
    # que sigue es OTRA sección musical (p. ej. el relleno instrumental tras
    # "incondicional" en NESOLO L16: el seam lo arrastraba 4 s). La nota
    # sostenida real (un "paz" largo) no tiene ese corte y queda intacta.
    deep = None
    for k in range(i_e, i_cap + 1):
        if env[k] < 0.30 * pk:
            deep = k
            break
    if deep is not None:
        last_hi = None
        for k in range(i_e, min(deep, i_cap) + 1):
            if env[k] >= half:
                last_hi = k
        if last_hi is None:
            return None
        return min((last_hi + 1) * 0.01, cap)
    last_hi = None
    for k in range(i_e, i_cap + 1):
        if env[k] >= half:
            last_hi = k
    if last_hi is None:
        return None
    if env[i_cap] >= half:
        return cap
    return min((last_hi + 1) * 0.01, cap)

def _onset_boundary(env, lo, first_s):
    """Frontera de arranque con cola PEGADA (gap transcript ~ 0): la cola de
    la palabra anterior quedó absorbida por la primera palabra de la frase
    siguiente, así que el transcript no sirve. Se busca el PRIMER pico de voz
    que supera claramente (>= 15 %) el nivel ~70 ms antes del pico (la cola):
    es el arranque de la frase nueva. La frontera es el último valle antes de
    ese pico."""
    w1 = min(lo + 0.55, first_s + 0.45)
    i0 = max(1, _idx(lo))
    i1 = min(len(env) - 2, _idx(w1))
    if i1 - i0 < 10:
        return None
    pk = None
    for k in range(i0 + 1, i1 + 1):
        v = env[k]
        if v >= env[k - 1] and v >= env[k + 1] and v >= 0.02:
            prev_level = env[max(i0, k - 7)]
            if v >= 1.15 * prev_level:
                pk = k
                break
    if pk is None:
        return None
    valley = None
    for k in range(i0, pk):
        if env[k] <= env[k - 1] and env[k] <= env[k + 1]:
            valley = k
    if valley is None:
        return None
    return valley * 0.01

def _held_note_end(env, lo_end, cap):
    """Fin de una nota sostenida con caída RÁPIDA tras el último pico (se usa
    cuando la frase siguiente está lejos y no hay frontera contigua). Devuelve
    el fin estimado o None si la caída es gradual (no cortar un diminuendo)."""
    i0 = max(1, _idx(max(0.0, lo_end - 0.3)))
    i1 = min(len(env) - 2, _idx(cap))
    if i1 - i0 < 20:
        return None
    w = env[i0:i1 + 1]
    peak = float(w.max())
    if peak < 0.02:
        return None
    pk_pos = i0 + int(np.argmax(w))
    last_peak = None
    for k in range(pk_pos, i1 + 1):
        if env[k] >= env[k - 1] and env[k] >= env[k + 1] and env[k] >= 0.8 * peak:
            last_peak = k
    if last_peak is None:
        last_peak = pk_pos
    drop_thr = 0.45 * peak
    j = None
    for k in range(last_peak + 1, i1 + 1):
        if env[k] < drop_thr:
            j = k
            break
    if j is None:
        return None
    if (j - last_peak) * 0.01 > 0.20:
        return None
    nxt = env[j + 1:j + 9]
    if nxt.size < 5:
        return None
    if int((nxt < 0.5 * peak).sum()) >= 5:
        return j * 0.01 + 0.05
    return None

def _apply_voice_boundaries(song, flat, run, run_i, s0, e0, dur, src,
                            first_s_ov=None, lo_end_ov=None,
                            prev_end_ov=None, next_first_s_ov=None,
                            occ_mode=False):
    """Ajusta los límites del segmento con la energía real de la voz (v21):
      - INICIO: el segmento empieza en el último valle antes del primer ataque
        fuerte de voz (recorta la cola de la frase anterior y el aire muerto
        por whisper temprano). Puede ADELANTAR o RETRASAR el ancla del
        transcript.
      - FINAL: el segmento termina en el valle de la unión con la frase
        siguiente (incluye la cola completa de la última palabra) o, si la
        nota final es sostenida, se extiende hasta justo antes del arranque de
        la frase siguiente.
    Devuelve (s0, e0) ajustados."""
    try:
        if not run:
            return s0, e0
        env = _rms_env(src)
        lines = song.get("lines") or []
        first_s = lo_end = None
        prev_end = next_first_s = None
        if run_i is not None:
            li0 = flat[run_i][0]
            li1 = flat[min(len(flat) - 1, run_i + len(run) - 1)][0]
            first_s = float(run[0].get("s")) if run[0].get("s") is not None else None
            lo_end = float(run[-1].get("e")) if run[-1].get("e") is not None else None

            if li0 > 0:
                prev_words = (lines[li0 - 1].get("words") or [])
                if prev_words and prev_words[-1].get("e") is not None:
                    prev_end = float(prev_words[-1]["e"])
            if li1 + 1 < len(lines):
                next_words = (lines[li1 + 1].get("words") or [])
                if next_words and next_words[0].get("s") is not None:
                    next_first_s = float(next_words[0]["s"])
        # (v24) overrides para apariciones REPETIDAS: la posición de la voz se
        # toma del transcript de la aparición, no de la línea lírica (que
        # pertenece a la aparición primaria y desviaba las demás).
        if first_s_ov is not None:
            first_s = first_s_ov
        if lo_end_ov is not None:
            lo_end = lo_end_ov
        if prev_end_ov is not None:
            prev_end = prev_end_ov
        if next_first_s_ov is not None:
            next_first_s = next_first_s_ov

        # ================= INICIO =================
        # La frontera con la frase anterior es el valle justo antes del primer
        # ataque de voz de ESTA frase (detectado por energía; whisper puede
        # quedar temprano O tarde). Se permite mover el inicio hacia adelante
        # (recortar cola de la frase anterior / aire muerto) o hacia atrás
        # (recuperar la 1ª sílaba si whisper quedó tarde), acotado por el fin
        # del transcript de la frase previa.
        bnd_start = None
        if first_s is not None and prev_end is not None:
            gap = first_s - prev_end
            if gap >= 0.10:
                # aire entre frases según el transcript: el arranque real de la
                # frase puede estar hasta 0.15-0.30 s después del timestamp de
                # la primera palabra (whisper arranca temprano sobre colas).
                w0_in = max(first_s - 0.25,
                            (prev_end - 0.02) if prev_end is not None else
                            first_s - 0.25)
                bnd_start = _seam_before(env, w0_in, first_s + 0.55,
                                         first_s=first_s)
                if bnd_start is None:
                    bnd_start = first_s - 0.02
            else:
                # frontera contaminada o transcript solapado (gap ~ 0 o
                # negativo): la cola quedó pegada a la palabra siguiente.
                bnd_start = _onset_after_tail(env, max(prev_end, first_s),
                                              max(prev_end, first_s))
                if bnd_start is None:
                    bnd_start = first_s - 0.02
            if bnd_start is not None:
                # (v23) El ancla del transcript (whisper-small) es la fuente
                # principal del arranque de la frase: el seam solo puede
                # ADELANTAR el inicio sobre silencio/instrumental real (aire
                # muerto por whisper temprano) o dejarlo en first_s. NUNCA se
                # mueve el inicio más tarde que first_s: antes, el seam
                # "cortaba" el arranque de la palabra (p. ej. dentro de
                # "por"/"que"/"si") y la frase sonaba empezando con la cola
                # de la anterior (o con una vocal colgada).
                lo_ok = min(prev_end, first_s) - 0.25
                if bnd_start > first_s + 0.05:
                    # seam tardío: no confiar; quedarse con el ancla
                    if first_s > s0 + 0.02 and first_s < e0 - 0.05:
                        s0 = first_s
                elif bnd_start < first_s - 0.03:
                    # seam más temprano que el ancla: solo si el arranque del
                    # transcript está sobre SILENCIO (recuperar cabeza real).
                    # Si hay energía viva antes de first_s (cola de la frase
                    # previa / ataque interno), NO tirar el inicio atrás: eso
                    # metía la cola de la frase anterior en el clip.
                    k0 = max(1, _idx(max(0.0, first_s - 0.15)))
                    k1 = min(len(env) - 1, _idx(first_s + 0.05))
                    dead = float(env[k0:k1 + 1].max()) < 0.06
                    if dead and bnd_start >= lo_ok and bnd_start < e0 - 0.05:
                        s0 = bnd_start
                    elif not occ_mode and first_s > s0 + 0.02 and first_s < e0 - 0.05:
                        # (v24) en modo ocurrencia el inicio ya fue decidido
                        # por el llamador (recuperación de arranque por
                        # envolvente); no volver a first_s (whisper puede
                        # haber puesto la palabra tarde).
                        s0 = first_s
                else:
                    # seam ≈ first_s: ajuste fino
                    if bnd_start >= lo_ok and bnd_start < e0 - 0.05:
                        s0 = bnd_start
        # aire muerto por whisper TEMPRANO con gap largo: si el inicio quedó
        # en silencio/instrumental puro, buscar el primer ataque fuerte de voz
        # más allá (hasta first_s + 1.00 s) y recortar el silencio. La búsqueda
        # empieza en el inicio ya fijado (no antes: no volver a colas previas).
        if first_s is not None and prev_end is not None:
            k0 = max(1, _idx(max(0.0, s0)))
            k1 = min(len(env) - 1, _idx(s0 + 0.35))
            if k1 > k0 and float(env[k0:k1 + 1].max()) < 0.06:
                w0 = max(first_s - 0.25, s0)
                strong = _strong_onset(env, w0, first_s + 1.00, 0.045)
                if strong is not None and strong < e0 - 0.05 and strong > s0 + 0.02:
                    s0 = strong
        # cabeza interpolada (timestamps falsos): la primera palabra puede
        # haber quedado cortada (el transcript la arrancó tarde). Extender
        # hacia atrás hasta el arranque real antes de la 1ª ancla real, pero
        # NUNCA antes del inicio ya fijado por la frontera anterior ni del
        # final de la frase previa.
        if first_s_ov is None and not _is_real_anchor(run[0]) and first_s is not None:
            for w in run:
                if _is_real_anchor(w):
                    fr_s = float(w["s"])
                    head_s = float(run[0]["s"])
                    lo_h = max(0.0, min(head_s - 0.18, fr_s - 0.25))
                    hi_h = min(fr_s, head_s + 0.10)
                    if hi_h - lo_h >= 0.10:
                        # detección FUERTE: el arranque real de la cabeza es un
                        # valle de costura con ataque claro (rechaza los micro-
                        # valles de la cola de la frase anterior).
                        onset = _seam_before(env, lo_h, hi_h)
                        if onset is None:
                            onset = _strong_onset(env, lo_h, hi_h, 0.03)
                        if onset is not None and onset < s0 - 0.02:
                            cap_h = (prev_end + 0.02) if prev_end is not None else None
                            if cap_h is None or onset >= cap_h:
                                s0 = max(0.0, min(s0, onset))
                    break

        # ================= FINAL =================
        if lo_end is not None:
            if next_first_s is not None:
                gap = next_first_s - lo_end
                # La frontera con la frase siguiente es el valle justo antes
                # del PRIMER ataque real de voz de esa frase (detectado por
                # energía: _seam_before). whisper puede poner el arranque
                # siguiente temprano (sobre la cola de esta frase) o tarde
                # (sobre silencio/instrumental), así que la ventana cubre
                # desde la cola de esta frase hasta 0.55 s después del
                # timestamp del transcript. El valle resultante puede
                # RETRASAR el final (incluir la cola completa de la última
                # palabra: sin recortes) o ADELANTARLO (si el ancla invadía
                # el arranque siguiente: sin cola en la frase siguiente).
                bnd = _seam_before(env, max(0.0, lo_end - 0.08),
                                   next_first_s + 0.55,
                                   first_s=next_first_s)
                if bnd is not None and lo_end - 0.25 <= bnd <= e0 + 0.60:
                    e0 = bnd
                    # (v23) tope DURO: el seam no puede terminar la frase
                    # DENTRO del arranque de la frase siguiente. Antes el seam
                    # (hasta e0+0.60) invadía la 1ª palabra de la próxima
                    # (p. ej. L7 de Noche terminaba en 74.12 dentro del "por"
                    # de L8, y L10 en 91.23 dentro del "que" de L11) y la frase
                    # siguiente arrancaba tarde por el anti-solape.
                    if next_first_s is not None:
                        e0 = min(e0, max(next_first_s, lo_end))
                elif gap < 0.10:
                    # unión pegada y sin valle claro: no invadir la frase
                    # siguiente
                    cap = min(next_first_s - 0.02, lo_end + 2.0)
                    if e0 > cap:
                        e0 = max(cap, lo_end - 0.05)
                    hne = _held_note_end(env, lo_end, max(cap, lo_end + 0.2))
                    if hne is not None and hne > e0 - 0.05:
                        e0 = min(float(dur), max(e0, hne))
                else:
                    # aire entre frases (respaldo si el seam no aplicó):
                    # extender el final hasta el último valle antes del
                    # arranque siguiente, SIN invadir su primera palabra
                    bnd2 = _last_rise_event(env, lo_end, next_first_s + 0.02)
                    if bnd2 is None:
                        bnd2 = _last_rise_event(env, lo_end,
                                                next_first_s + 0.40, factor=1.5)
                    if bnd2 is not None:
                        bnd2 = min(bnd2, next_first_s - 0.05)
                    if bnd2 is not None and e0 - 0.05 < bnd2 <= e0 + 0.60:
                        e0 = min(max(e0, bnd2), bnd2 + 0.35)
                    else:
                        cap = min(next_first_s - 0.05, lo_end + 2.0)
                        hne = _held_note_end(env, lo_end, cap)
                        if hne is not None and e0 - 0.05 < hne <= e0 + 2.0:
                            e0 = min(float(dur), max(e0, hne))
                        else:
                            # nota final sostenida que el corte dejó a mitad:
                            # extender hasta el final real de la nota
                            cap2 = min(next_first_s - 0.05, lo_end + 4.0)
                            nte = _note_tail_end(env, min(e0, lo_end), cap2)
                            if nte is not None and nte > e0 - 0.05:
                                e0 = min(float(dur), max(e0, nte))
            else:
                # última frase de la canción
                cap = min(float(dur), lo_end + 2.0)
                hne = _held_note_end(env, lo_end, cap)
                if hne is not None and e0 - 0.05 < hne <= e0 + 2.0:
                    e0 = min(float(dur), max(e0, hne))

        e0 = min(float(dur), max(0.0, e0))
        if e0 - s0 > 0.05:
            s0 = min(max(0.0, s0), e0 - 0.05)
        return s0, e0
    except Exception:
        return s0, e0

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
    src = str(_ensure_audio(sid))   # ruta del mp3, se usa para el corte y para el render

    segments = []
    skipped = 0
    omitted_words = 0
    phrases = []          # texto de cada rango pedido + si aportó audio
    for a, b in ranges:
        a = max(0, int(a))
        b = min(total - 1, int(b))
        if a > b:
            continue
        # inicio de la siguiente palabra de la letra (límite duro del corte)
        nxt = flat[b + 1][2]["s"] if b + 1 < total else None
        words = [w for (_, _, w) in flat[a:b + 1]]
        ptext = " ".join(w.get("raw", "") for w in words).strip()
        phrases.append({
            "text": ptext,
            "ok": True,
            "repeticiones": 1,
        })
        omitted_words += sum(1 for w in words if not w.get("m"))
        # dividir la selección en tramos contiguos de palabras CON audio;
        # los huecos sin audio detectado se saltean (no invaden el audio final)
        runs = []          # (índice plano de la 1ª palabra, palabras del tramo)
        cur = []
        cur_i = None
        for j, w in enumerate(words):
            if w.get("m"):
                if cur_i is None:
                    cur_i = a + j
                cur.append(w)
            else:
                if cur:
                    runs.append((cur_i, cur))
                    cur, cur_i = [], None
        if cur:
            runs.append((cur_i, cur))
        # respaldo POSICIONAL (se calcula UNA vez por rango): la frase no tiene
        # audio en su posición (el modelo la transcribió mal ahí). Pero SABEMOS
        # dónde está en la canción: entre la última palabra REAL con audio que
        # viene antes y la primera REAL que viene después. Repartimos ese hueco
        # proporcionalmente entre las palabras de la letra sin audio que hay en
        # el medio. Es LOCAL: nunca trae audio de otra parte de la canción (el
        # fallback fuzzy sí podía agarrar un coro lejano y sonar una frase que
        # no se eligió).
        pos_seg = None
        prev_i = a - 1
        while prev_i >= 0 and not _is_real_anchor(flat[prev_i][2]):
            prev_i -= 1
        nxt_i = b + 1
        while nxt_i < total and not _is_real_anchor(flat[nxt_i][2]):
            nxt_i += 1
        if prev_i >= 0 and nxt_i < total:
            pv = flat[prev_i][2]
            nx = flat[nxt_i][2]
            if pv.get("e") is not None and nx.get("s") is not None:
                gap_words = nxt_i - prev_i - 1
                sel_words = b - a + 1
                gap_s = float(pv["e"])
                gap_e = float(nx["s"])
                gap_dur = gap_e - gap_s
                if 0.2 < gap_dur <= 60:
                    # palabras REALES del transcript dentro del hueco: marcan
                    # dónde canta la frase de verdad. Repartir el hueco
                    # proporcionalmente metía aire muerto al inicio (p. ej.
                    # 2.8 s de instrumental antes del "Ten piedad") o se
                    # pasaba de largo sobre la frase siguiente.
                    in_gap = [w for w in transcript
                              if gap_s + 0.05 < float(w.get("start") or 0) < gap_e - 0.05
                              and not _is_breath(w)]
                    if in_gap:
                        s0 = max(gap_s + 0.05, float(in_gap[0]["start"]) - 0.12)
                        e0 = min(gap_e - 0.05, float(in_gap[-1]["end"]) + 0.22)
                        if e0 - s0 > 0.05:
                            pos_seg = (s0, e0)
                    else:
                        # hueco sin palabras (p. ej. un melisma sin letra):
                        # el hueco completo ES la frase, si no es enorme.
                        whole_gap = sel_words >= gap_words
                        if (gap_words >= 1 and sel_words >= 1
                                and (sel_words < gap_words
                                     or (whole_gap and gap_dur <= 20.0))
                                and gap_dur / gap_words <= 6.0):
                            off0 = a - prev_i - 1
                            off1 = b - prev_i
                            inner_s = gap_s + 0.05
                            inner_e = gap_e - 0.05
                            inner_dur = inner_e - inner_s
                            s0 = max(0.0, inner_s + (off0 / gap_words) * inner_dur)
                            e0 = min(dur, inner_s + (off1 / gap_words) * inner_dur)
                            if e0 - s0 > 0.05:
                                pos_seg = (s0, e0)
        if not runs:
            if pos_seg:
                segments.append((pos_seg[0], pos_seg[1], ptext))
                continue
            # respaldo GENERAL (SOLO si el posicional no aplicó): la frase no
            # tiene audio en su posición ni vecinos confiables, pero puede
            # estar bien transcrita en OTRA aparición del audio (p. ej. un
            # estribillo repetido). Último recurso.
            fb = align.find_phrase_repeated(transcript, words)
            if fb:
                sc, k0, k1 = fb
                nfb = k1 - k0 + 1
                if nfb <= 2:
                    pb0 = pb1 = 0.0
                else:
                    pb0, pb1 = 0.15, 0.25
                s0 = max(0.0, transcript[k0]["start"] - pb0)
                e0 = min(dur, transcript[k1]["end"] + pb1)
                if e0 - s0 > 0.05:
                    segments.append((s0, e0, ptext))
                    continue
            if phrases:
                phrases[-1]["ok"] = False
            skipped += 1
            continue

        for run_i, run in runs:
            if all_occ:
                # (v24) TODAS las apariciones de la frase, buscadas por TEXTO
                # en el transcript (no por tj: los tj de líneas restauradas o
                # alineadas con otro transcript apuntaban a índices equivocados
                # y no se incluía ninguna repetición). Cada aparición usa las
                # fronteras de voz de SU propia posición (overrides), no las
                # de la línea lírica.
                occs = align.find_all_phrase_occurrences(transcript, run)
                if not occs:
                    # respaldo: al menos la aparición primaria
                    s0 = max(0.0, run[0]["s"] - 0.15)
                    e0 = min(dur, run[-1]["e"] + 0.25)
                    s0, e0 = _apply_voice_boundaries(song, flat, run, run_i, s0, e0, dur, src)
                    if e0 - s0 > 0.05:
                        segments.append((s0, e0, ptext))
                    if phrases:
                        phrases[-1]["repeticiones"] = max(
                            phrases[-1].get("repeticiones", 1), 1)
                else:
                    if phrases:
                        phrases[-1]["repeticiones"] = max(
                            phrases[-1].get("repeticiones", 1), len(occs))
                    env = _rms_env(src)
                    for k0, k1, _ in occs:
                        fs_occ = float(transcript[k0]["start"])
                        le_occ = float(transcript[k1]["end"])
                        prev_e = (float(transcript[k0 - 1]["end"])
                                  if k0 > 0 else None)
                        nxt_s = (float(transcript[k1 + 1]["start"])
                                 if k1 + 1 < len(transcript) else None)
                        s0 = max(0.0, fs_occ - 0.25)
                        e0 = min(dur, le_occ + 0.20)
                        # (v24) whisper a veces pone el inicio de la palabra
                        # TARDE en la voz cantada (hasta ~1 s: p. ej. "Ten
                        # piedad" transcrito "Tempiedad" con el "Ten" ~1.2 s
                        # antes). Si hay un ataque fuerte y claro en un rango
                        # acotado antes del timestamp, y arranca desde
                        # silencio real, ese ataque ES el arranque de la
                        # frase (el límite de 1.5 s evita cruzar un pasaje
                        # instrumental entero).
                        if prev_e is not None:
                            st = _strong_onset(env, prev_e - 0.02,
                                               fs_occ + 0.60, 0.045)
                            if (st is not None
                                    and fs_occ - 1.50 <= st < fs_occ - 0.30):
                                k_st = max(1, _idx(st))
                                base0 = float(
                                    env[max(0, k_st - 20):k_st].min())
                                if base0 < 0.06:
                                    s0 = st
                        s0, e0 = _apply_voice_boundaries(
                            song, flat, run, run_i, s0, e0, dur, src,
                            first_s_ov=fs_occ, lo_end_ov=le_occ,
                            prev_end_ov=prev_e, next_first_s_ov=nxt_s,
                            occ_mode=True)
                        if e0 - s0 > 0.05:
                            segments.append((s0, e0, ptext))
            else:
                # robustez: si los timestamps del tramo son inverosímiles
                # (huecos enormes entre palabras contiguas -> el align unió la
                # frase con audio de otra sección), dividir en los huecos y
                # quedarse con el sub-tramo más denso en vez de emitir un
                # segmento gigante con audio de otras frases.
                if len(run) > 1:
                    subs = []
                    cur = [run[0]]
                    for k in range(1, len(run)):
                        if run[k]["s"] - run[k - 1]["e"] > 2.5:
                            subs.append(cur)
                            cur = [run[k]]
                        else:
                            cur.append(run[k])
                    subs.append(cur)
                    dense = max(subs, key=lambda s: len(s))
                    if len(dense) >= 2:
                        run = dense
                    else:
                        run = max(subs, key=lambda s: (s[-1]["e"] - s[0]["s"]))
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

                # ---- límites con ANCLAS REALES + transcript ----
                # Una palabra es "ancla real" si tiene match de verdad en el
                # transcript (tj asignado y sim alto). Las demás (m=True pero
                # tj=None / sim=0.5) tienen timestamps FALSOS (interpolados a
                # 0.2s): usarlas como borde corta la frase a la mitad (p. ej.
                # "...sufren y agoniz[an...]").
                real_in_run = [w for w in run if _is_real_anchor(w)]
                if real_in_run:
                    # ancla real previa (antes de a) y siguiente (después de b)
                    prev_anchor_e = None
                    j = a - 1
                    while j >= 0 and not _is_real_anchor(flat[j][2]):
                        j -= 1
                    if j >= 0:
                        prev_anchor_e = float(flat[j][2]["e"])
                    next_anchor_s = None
                    j = b + 1
                    while j < total and not _is_real_anchor(flat[j][2]):
                        j += 1
                    if j < total:
                        next_anchor_s = float(flat[j][2]["s"])
                    # palabras del transcript entre las anclas: marcan dónde
                    # canta la frase de verdad (aunque el texto esté mal
                    # transcrito, la POSICIÓN es real)
                    reg_start = prev_anchor_e if prev_anchor_e is not None else 0.0
                    reg_end = next_anchor_s if next_anchor_s is not None else float(dur)
                    region = [w for w in transcript
                              if float(w.get("start") or 0) >= reg_start - 0.01
                              and float(w.get("start") or 0) <= reg_end + 0.01]
                    # START: primer ancla real; si la cabeza está interpolada,
                    # extender hacia atrás hasta el primer transcript de la
                    # región, acotado por un estimado por palabra (no retroceder
                    # hasta la frase anterior cuando la región es enorme)
                    first_real = real_in_run[0]
                    s0 = float(first_real["s"]) - pad_before
                    if run[0] is not first_real and region:
                        n_soft_head = 0
                        for w in run:
                            if _is_real_anchor(w):
                                break
                            n_soft_head += 1
                        cap0 = float(first_real["s"]) - n_soft_head * 0.85 - 0.5
                        s0 = min(s0, max(float(region[0]["start"]) - 0.08, cap0))
                    s0 = max(0.0, s0)
                    if prev_anchor_e is not None:
                        s0 = max(s0, prev_anchor_e + 0.05)
                    # END: última ancla real; si la cola está interpolada,
                    # extender hasta el último transcript de la región, acotado
                    # por un estimado por palabra (si la región es enorme, no
                    # cruzar a la próxima frase) y por el inicio de la próxima
                    # ancla real
                    last_real = real_in_run[-1]
                    e0 = float(last_real["e"]) + pad_after
                    if run[-1] is not last_real and region:
                        n_soft = 0
                        for w in reversed(run):
                            if _is_real_anchor(w):
                                break
                            n_soft += 1
                        cap = float(last_real["e"]) + n_soft * 0.85 + 0.5
                        e0 = min(max(e0, float(region[-1]["end"]) + 0.15), cap)
                    if next_anchor_s is not None:
                        # el tope NUNCA corta dentro de la última palabra real:
                        # cuando las líneas son contiguas (la frase siguiente
                        # empieza justo donde termina esta), next_anchor_s ≈
                        # last_real.e y el tope viejo cortaba el final de la
                        # palabra ("...ladrone[s]" se oía recortado).
                        e0 = min(e0, max(next_anchor_s - 0.05,
                                         float(last_real["e"]) + 0.05))
                    next_first_s = None
                    li_last = flat[run_i + len(run) - 1][0]
                    nxt_lines = song.get("lines") or []
                    if li_last + 1 < len(nxt_lines):
                        nw = (nxt_lines[li_last + 1].get("words") or [])
                        if nw and nw[0].get("s") is not None:
                            next_first_s = float(nw[0]["s"])
                    if next_first_s is not None:
                        # (v23) tope DURO: la extensión de cola suave (n_soft)
                        # no puede cruzar al arranque de la frase siguiente.
                        # Antes, una cola con palabras "fantasma" del transcript
                        # extendía el final HASTA DENTRO de la frase siguiente
                        # (p. ej. L11 de NESOLO terminaba pisando el "No...").
                        e0 = min(e0, max(next_first_s - 0.02,
                                         float(last_real["e"]) + 0.05))
                    e0 = min(float(dur), e0)
                else:
                    # toda la run está interpolada (sin anclas reales): su
                    # posición puede ser falsa (el align a veces interpola la
                    # frase hacia OTRA parte del audio). Preferir el segmento
                    # posicional local; si no aplica, timestamps locales.
                    if pos_seg:
                        s0, e0 = pos_seg
                    else:
                        s0 = max(0.0, run[0]["s"] - pad_before)
                        e0 = min(dur, run[-1]["e"] + pad_after)
                        if run[-1].get("sim", 1.0) <= 0.55 or run[-1].get("tj") is None:
                            e0 = _speech_tail(src, s0, e0, dur, transcript, nxt)
                # ---- fronteras de voz reales (v20): recortar del inicio la
                # cola de la frase anterior y extender el final hasta el último
                # valle antes del arranque de la frase siguiente ----
                s0, e0 = _apply_voice_boundaries(song, flat, run, run_i, s0, e0, dur, src)
                if e0 - s0 > 0.05:
                    segments.append((s0, e0, ptext))

    # sin solapamientos: si dos frases pedidas comparten audio (el final de
    # una con el inicio de la otra), se corría dos veces el mismo pedazo y la
    # costura sonaba a corte. Cada segmento empieza donde terminó el anterior
    # (por orden de tiempo). El orden final de reproducción lo decide la
    # selección del usuario.
    if len(segments) > 1:
        segments.sort(key=lambda s: s[0])
        clipped = []
        for s0, e0, txt in segments:
            if clipped and s0 < clipped[-1][1]:
                s0 = clipped[-1][1]
            if e0 - s0 > 0.05:
                clipped.append((s0, e0, txt))
        segments = clipped

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
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]), merged[-1][2])
        else:
            merged.append(seg)
    segments = merged

    if not segments:
        raise HTTPException(400, "No se pudo generar audio con las frases elegidas")

    # línea de tiempo para la lluvia: cada segmento con su texto y su posición
    # DENTRO del audio combinado (los fragmentos se concatenan con 0.2 s de
    # silencio). Sirve para que la lluvia muestre la frase que suena en el
    # momento exacto en que suena.
    GAP = 0.2
    timeline = []
    at = 0.0
    for s0, e0, txt in segments:
        timeline.append({
            "start": round(s0, 2),
            "end": round(e0, 2),
            "dur": round(e0 - s0, 2),
            "at": round(at, 2),
            "text": txt,
        })
        at += (e0 - s0) + GAP

    uid = hashlib.md5(user_id.encode("utf-8")).hexdigest()[:8]
    # URL única por contenido: evita que el navegador sirva audio cacheado viejo
    chash = hashlib.md5(repr(segments).encode("utf-8")).hexdigest()[:8]
    fname = f"{uid}_{chash}.mp3"
    out = RENDER_DIR / f"{sid}_{fname}"
    if not out.exists():
        au.render_phrases(src, [(s, e) for s, e, _ in segments], str(out))
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
        "phrases": phrases,
        "timeline": timeline,
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
        # si el usuario borró la canción mientras se procesaba, NO volver a
        # crearla (antes se resucitaba sola al terminar el worker)
        if store.get_song(sid) is not None:
            store.save_song(sid, song, durable=True)
    except Exception as e:
        song = store.get_song(sid)
        if song is None:
            return  # fue borrada mientras se procesaba: no resucitar
        song["status"] = "error"
        song["error"] = str(e)[:500]
        song["phase"] = None
        store.save_song(sid, song, durable=True)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
