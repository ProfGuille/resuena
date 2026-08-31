"""Persistencia resiliente: memoria (lecturas instantáneas) + disco local
(write-through) + GitHub como copia durable (sync acotado).

Diseño (para que la app se sienta rápida en Render free aunque la API de
GitHub vaya lenta o se caiga):
  - LAS LECTURAS SIEMPRE SALEN DE MEMORIA: sin red, sin esperas.
  - Las escrituras van a memoria + disco local AL INSTANTE (microsegundos).
  - GitHub se actualiza: (a) de forma síncrona SOLO en "puntos durables"
    (crear canción, canción lista, canción con error, borrar) — 1-2 veces por
    canción; (b) con debounce (2 s) para selecciones; (c) NUNCA para las fases
    de procesamiento (son estado transitorio de la UI).
  - Ningún problema de red puede romper una request: nunca 500 por storage.
"""
import json
import os
import threading
import time

import cloud
import ghstore

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
SONGS_FILE = os.path.join(DATA_DIR, "songs.json")
SEL_FILE = os.path.join(DATA_DIR, "selections.json")
SONGS_KEY = "meta/songs.json"
SEL_KEY = "meta/selections.json"
SEL_DEBOUNCE = 2.0   # segundos antes de escribir selecciones a GitHub
RETRY_EVERY = 15.0   # reintento de carga remota si el arranque falló

_lock = threading.Lock()
_songs = {}
_sels = {}
_loaded = False
_remote_failed = False
_sel_dirty = False
_sel_timer = None
_retrier_started = False


# ---------------------------------------------------------------- helpers
def _read_local(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _write_local(path, obj):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        pass


def _gh_put(key, obj, retries=1):
    """PUT a GitHub acotado: nunca lanza, devuelve True/False."""
    if not ghstore.enabled():
        return False
    for i in range(retries + 1):
        try:
            if ghstore.put_json(key, obj):
                return True
        except Exception:
            pass
        if i < retries:
            time.sleep(0.5)
    return False


# ---------------------------------------------------------------- carga inicial
def init():
    """Carga memoria desde disco (rápido) y refresca desde GitHub (una vez)."""
    global _songs, _sels, _loaded
    with _lock:
        if _loaded:
            return
        _songs = _read_local(SONGS_FILE) or {}
        _sels = _read_local(SEL_FILE) or {}
        _loaded = True
    if ghstore.enabled():
        _refresh_remote()


def _refresh_remote():
    """Trae metadatos desde GitHub a memoria (best effort). Nunca lanza."""
    global _songs, _sels, _remote_failed
    try:
        ok1, data1 = ghstore.fetch_json(SONGS_KEY)
        ok2, data2 = ghstore.fetch_json(SEL_KEY)
        if ok1 and data1 is not None:
            with _lock:
                _songs = data1
            _write_local(SONGS_FILE, data1)
        if ok2 and data2 is not None:
            with _lock:
                _sels = data2
            _write_local(SEL_FILE, data2)
        _remote_failed = not (ok1 and ok2)
    except Exception:
        _remote_failed = True
    if _remote_failed:
        _start_retrier()


def _start_retrier():
    global _retrier_started
    with _lock:
        if _retrier_started:
            return
        _retrier_started = True
    threading.Thread(target=_retrier_loop, daemon=True).start()


def _retrier_loop():
    while True:
        time.sleep(RETRY_EVERY)
        if _remote_failed:
            _refresh_remote()


# ---------------------------------------------------------------- lectura (memoria pura)
def all_songs():
    return _songs


def get_song(sid):
    return _songs.get(sid)


def get_selection(song_id, user_id):
    return _sels.get(f"{song_id}:{user_id}")


# ---------------------------------------------------------------- escritura canciones
def save_song(sid, data, durable=False):
    """Guarda una canción: memoria + disco al instante; GitHub si durable.

    durable=True  → puntos críticos (crear / lista / error / borrar):
                    se escribe a GitHub de inmediato para no perder la canción.
    durable=False → estado transitorio (fases del procesamiento): NO toca GitHub.
    """
    with _lock:
        _songs[sid] = data
        snap = dict(_songs)
    _write_local(SONGS_FILE, snap)
    if durable:
        if ghstore.enabled():
            _gh_put(SONGS_KEY, snap)
        elif cloud.cloud_enabled():
            try:
                cloud.put_bytes(SONGS_KEY, json.dumps(snap, ensure_ascii=False).encode("utf-8"))
            except Exception:
                pass


def delete_song(sid):
    global _sels
    with _lock:
        _songs.pop(sid, None)
        snap = dict(_songs)
        _sels = {k: v for k, v in _sels.items() if not k.startswith(sid + ":")}
        selsnap = dict(_sels)
    _write_local(SONGS_FILE, snap)
    _write_local(SEL_FILE, selsnap)
    if ghstore.enabled():
        _gh_put(SONGS_KEY, snap)
        _gh_put(SEL_KEY, selsnap)


# ---------------------------------------------------------------- escritura selecciones
def save_selection(song_id, user_id, ranges):
    """Memoria + disco al instante; GitHub con debounce (rápido para el usuario)."""
    global _sel_dirty
    with _lock:
        _sels[f"{song_id}:{user_id}"] = {"ranges": ranges, "updated_at": time.time()}
        _sel_dirty = True
        snap = dict(_sels)
    _write_local(SEL_FILE, snap)
    _schedule_sel_sync()


def _schedule_sel_sync():
    global _sel_timer
    with _lock:
        if _sel_timer is not None:
            return
        _sel_timer = threading.Timer(SEL_DEBOUNCE, _flush_sel_sync)
        _sel_timer.daemon = True
        _sel_timer.start()


def _flush_sel_sync():
    global _sel_timer, _sel_dirty
    with _lock:
        _sel_timer = None
        if not _sel_dirty:
            return
        _sel_dirty = False
        snap = dict(_sels)
    if ghstore.enabled():
        _gh_put(SEL_KEY, snap, retries=2)
    elif cloud.cloud_enabled():
        try:
            cloud.put_bytes(SEL_KEY, json.dumps(snap, ensure_ascii=False).encode("utf-8"))
        except Exception:
            pass
