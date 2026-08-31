"""Persistencia simple: repo de GitHub (recomendado), Cloudflare R2 o disco local.

El orden de preferencia es: GitHub (ghstore) > R2 (cloud) > local.
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

_lock = threading.Lock()
_cache = {}
_cache_ts = {}
CACHE_TTL = 3.0  # segundos: para no golpear la API de GitHub en cada lectura


def _read_json(key, path):
    if ghstore.enabled():
        return ghstore.get_json(key, {})
    if cloud.cloud_enabled():
        try:
            return json.loads(cloud.get_bytes(key).decode("utf-8"))
        except Exception:
            return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _write_json(key, path, obj):
    if ghstore.enabled():
        for _ in range(3):
            if ghstore.put_json(key, obj):
                _cache[key] = obj
                _cache_ts[key] = time.time()
                return
        raise RuntimeError("No se pudo guardar en GitHub (conflicto de escritura)")
    if cloud.cloud_enabled():
        cloud.put_bytes(key, json.dumps(obj, ensure_ascii=False).encode("utf-8"))
        _cache[key] = obj
        _cache_ts[key] = time.time()
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)
    os.replace(tmp, path)
    _cache[key] = obj
    _cache_ts[key] = time.time()


def _read_cached(key, path):
    now = time.time()
    if key in _cache and now - _cache_ts.get(key, 0) < CACHE_TTL:
        return _cache[key]
    val = _read_json(key, path)
    _cache[key] = val
    _cache_ts[key] = now
    return val


def all_songs():
    return _read_cached(SONGS_KEY, SONGS_FILE)


def get_song(sid):
    return all_songs().get(sid)


def save_song(sid, data):
    with _lock:
        songs = all_songs()
        songs[sid] = data
        _write_json(SONGS_KEY, SONGS_FILE, songs)


def delete_song(sid):
    with _lock:
        songs = all_songs()
        songs.pop(sid, None)
        _write_json(SONGS_KEY, SONGS_FILE, songs)
        sels = _read_cached(SEL_KEY, SEL_FILE)
        sels = {k: v for k, v in sels.items() if not k.startswith(sid + ":")}
        _write_json(SEL_KEY, SEL_FILE, sels)


def get_selection(song_id, user_id):
    return _read_cached(SEL_KEY, SEL_FILE).get(f"{song_id}:{user_id}")


def save_selection(song_id, user_id, ranges):
    with _lock:
        sels = _read_cached(SEL_KEY, SEL_FILE)
        sels[f"{song_id}:{user_id}"] = {"ranges": ranges, "updated_at": time.time()}
        _write_json(SEL_KEY, SEL_FILE, sels)
