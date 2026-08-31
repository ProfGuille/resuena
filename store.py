"""Persistencia simple en archivos JSON (sin base de datos, gratis y portable)."""
import json
import os
import threading
import time

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
SONGS_FILE = os.path.join(DATA_DIR, "songs.json")
SEL_FILE = os.path.join(DATA_DIR, "selections.json")

_lock = threading.Lock()


def _ensure():
    os.makedirs(DATA_DIR, exist_ok=True)
    for f in (SONGS_FILE, SEL_FILE):
        if not os.path.exists(f):
            with open(f, "w", encoding="utf-8") as fh:
                json.dump({}, fh)


def _load(path):
    _ensure()
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save(path, obj):
    _ensure()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)
    os.replace(tmp, path)


def all_songs():
    return _load(SONGS_FILE)


def get_song(sid):
    return _load(SONGS_FILE).get(sid)


def save_song(sid, data):
    with _lock:
        songs = _load(SONGS_FILE)
        songs[sid] = data
        _save(SONGS_FILE, songs)


def delete_song(sid):
    with _lock:
        songs = _load(SONGS_FILE)
        songs.pop(sid, None)
        _save(SONGS_FILE, songs)
        sels = _load(SEL_FILE)
        sels = {k: v for k, v in sels.items() if not k.startswith(sid + ":")}
        _save(SEL_FILE, sels)


def get_selection(song_id, user_id):
    return _load(SEL_FILE).get(f"{song_id}:{user_id}")


def save_selection(song_id, user_id, ranges):
    with _lock:
        sels = _load(SEL_FILE)
        sels[f"{song_id}:{user_id}"] = {"ranges": ranges, "updated_at": time.time()}
        _save(SEL_FILE, sels)
