"""Utilidades de audio: yt-dlp, ffmpeg (conversión, corte y concatenación)."""
import os
import shutil
import subprocess
import time

import ffmpeg_util


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def ffprobe_duration(path):
    """Duración en segundos. Usa ffprobe si existe; si no (imageio-ffmpeg
    solo trae ffmpeg), la obtiene parseando la salida de `ffmpeg -i`."""
    r = ffmpeg_util.run_ffprobe([
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ], capture_output=True, text=True)
    if r is not None:
        try:
            return round(float(r.stdout.strip()), 2)
        except Exception:
            pass
    # fallback sin ffprobe: ffmpeg -i imprime "Duration: HH:MM:SS.xx"
    try:
        r2 = ffmpeg_util.run_ffmpeg(["-i", path], capture_output=True, text=True)
        out = (r2.stderr or "") + (r2.stdout or "")
        m = __import__("re").search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", out)
        if m:
            h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
            return round(h * 3600 + mi * 60 + s, 2)
    except Exception:
        pass
    return None


def to_wav16k(src, dst):
    """Convierte a WAV mono 16 kHz (entrada estándar para faster-whisper)."""
    r = ffmpeg_util.run_ffmpeg(["-y", "-i", src, "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", dst])
    if r.returncode != 0:
        raise RuntimeError("ffmpeg: no se pudo convertir a wav: " + r.stderr[-300:])


def to_streaming_mp3(src, dst):
    """Convierte a MP3 estéreo 44.1 kHz (para escuchar/cortar en el navegador)."""
    r = ffmpeg_util.run_ffmpeg(["-y", "-i", src, "-vn", "-ac", "2", "-ar", "44100",
             "-c:a", "libmp3lame", "-b:a", "192k", dst])
    if r.returncode != 0:
        raise RuntimeError("ffmpeg: no se pudo convertir a mp3: " + r.stderr[-300:])


def _yt_cookies_file():
    """(v39) Cookies de YouTube desde la variable de entorno YT_COOKIES.

    YouTube bloquea con 403 las IPs de datacenter (Render) si la petición
    no viene de una sesión logueada. Con cookies de una cuenta de YouTube,
    yt-dlp pasa el chequeo anti-bot.

    Acepta TRES formatos (el que salga más fácil copiar):
      1. El contenido de un cookies.txt Netscape (con tabs).
      2. La línea cruda "Cookie:" del DevTools del navegador (F12 -> Network
         -> clic a cualquier request de youtube.com -> Request Headers ->
         copiar el valor de "Cookie"): "SID=xxx; HSID=yyy; ...".
      3. Líneas simples "NOMBRE=VALOR" (una por línea).
    En los casos 2 y 3 se arma el formato Netscape automáticamente.
    Devuelve la ruta del archivo o None si no hay cookies configuradas.
    """
    global _yt_cookies_path
    raw = (os.environ.get("YT_COOKIES") or "").strip()
    if not raw:
        return None
    if _yt_cookies_path and os.path.exists(_yt_cookies_path):
        return _yt_cookies_path
    import tempfile
    try:
        lines = []
        if "\t" in raw:
            # formato Netscape ya listo
            body = raw.replace("\\n", "\n").splitlines()
            lines = [l for l in body if l.strip() and not l.startswith("#")]
        else:
            # crudo: "a=b; c=d" (header Cookie) o líneas "a=b"
            txt = raw.replace("Cookie:", " ").replace("cookie:", " ")
            pairs = [p.strip() for p in txt.replace("\n", ";").split(";")]
            for p in pairs:
                if "=" not in p:
                    continue
                name, _, val = p.partition("=")
                name = name.strip()
                val = val.strip()
                if not name or not val:
                    continue
                # dominio, incluir_subdominios, path, secure, expira, nombre, valor
                lines.append(f".youtube.com\tTRUE\t/\tTRUE\t1999999999\t{name}\t{val}")
        if not lines:
            print("[yt] YT_COOKIES no contenida cookies reconocibles", flush=True)
            return None
        p = os.path.join(tempfile.gettempdir(), "yt_cookies.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("# Netscape HTTP Cookie File\n")
            fh.write("\n".join(lines) + "\n")
        _yt_cookies_path = p
        print(f"[yt] cookies de YouTube cargadas desde YT_COOKIES ({len(lines)} cookies)", flush=True)
        return p
    except Exception as e:
        print("[yt] no se pudieron escribir las cookies:", str(e)[:120], flush=True)
        return None

_yt_cookies_path = None


def download_youtube(url, dst_dir, song_id):
    """Descarga el audio de un video de YouTube con yt-dlp, probando varias
    estrategias (cliente web, android, tv, mweb, con impersonación de Chrome)
    porque YouTube bloquea de forma variable según IP/video.

    Devuelve (ruta, título, autor). Lanza RuntimeError con mensaje claro si
    todas las estrategias fallan.
    """
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp no está instalado")

    try:
        from yt_dlp.networking.impersonate import ImpersonateTarget
    except Exception:
        ImpersonateTarget = None

    outtmpl = os.path.join(str(dst_dir), f"{song_id}.%(ext)s")
    # Estrategias (orden revisado 2026-09): los clientes forzados uno a uno
    # (web, android, tv...) dejaron de devolver formatos ("No video formats").
    # Lo que funciona es el SET POR DEFECTO de yt-dlp (sin forzar nada), con
    # o sin cookies. Va primero y también de última red. Las demás quedan
    # como respaldo por si YouTube vuelve a cambiar.
    strategies = [
        {"name": "default",        "clients": None,               "fmt": "bestaudio/best"},
        {"name": "web-imp",       "clients": ["web"],           "fmt": "bestaudio/best", "impersonate": "chrome"},
        {"name": "web-imp-v",     "clients": ["web"],           "fmt": "best",           "impersonate": "chrome"},
        {"name": "safari-imp",    "clients": ["web_safari"],    "fmt": "bestaudio/best", "impersonate": "safari"},
        {"name": "mweb-imp",      "clients": ["mweb"],          "fmt": "bestaudio[ext=m4a]/bestaudio/best", "impersonate": "chrome"},
        {"name": "android",       "clients": ["android"],       "fmt": "bestaudio[ext=m4a]/bestaudio/best"},
        {"name": "android-v",     "clients": ["android"],       "fmt": "best"},
        {"name": "android_vr",    "clients": ["android_vr"],    "fmt": "bestaudio/best"},
        {"name": "ios",           "clients": ["ios"],           "fmt": "bestaudio/best"},
        {"name": "tv",            "clients": ["tv"],            "fmt": "bestaudio/best"},
        {"name": "tv_embedded",   "clients": ["tv_embedded"],   "fmt": "bestaudio/best"},
        {"name": "web_embedded",  "clients": ["web_embedded"],  "fmt": "bestaudio/best"},
        {"name": "web",            "clients": ["web"],           "fmt": "bestaudio/best"},
        {"name": "default-2",      "clients": None,              "fmt": "best"},
    ]

    def _cleanup():
        for f in os.listdir(dst_dir):
            if f.startswith(song_id + "."):
                try:
                    os.remove(os.path.join(dst_dir, f))
                except OSError:
                    pass

    last_err = None
    for st in strategies:
        opts = {
            "format": st["fmt"],
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "socket_timeout": 30,
            "retries": 2,
            "extractor_retries": 1,
        }
        # solo forzar clientes cuando la estrategia lo pide ("default" no fuerza)
        if st.get("clients"):
            opts["extractor_args"] = {"youtube": {"player_client": st["clients"]}}
        if st.get("impersonate") and ImpersonateTarget is not None:
            try:
                opts["impersonate"] = ImpersonateTarget.from_str(st["impersonate"])
            except Exception:
                pass
        ck = _yt_cookies_file()
        if ck:
            opts["cookiefile"] = ck
        print(f"[yt] probando estrategia {st['name']}...", flush=True)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            path = ydl.prepare_filename(info)
            if os.path.exists(path):
                print(f"[yt] OK con estrategia {st['name']}", flush=True)
                return path, info.get("title"), info.get("uploader")
            base = os.path.splitext(path)[0]
            for ext in (".webm", ".m4a", ".mp3", ".opus", ".mp4"):
                cand = base + ext
                if os.path.exists(cand):
                    print(f"[yt] OK con estrategia {st['name']} (ext {ext})", flush=True)
                    return cand, info.get("title"), info.get("uploader")
        except Exception as e:
            last_err = str(e)[:200]
            print(f"[yt] estrategia {st['name']} falló: {last_err}", flush=True)
            _cleanup()
            time.sleep(2)  # pausa anti-throttling antes de la próxima estrategia
            continue
    _cleanup()
    raise RuntimeError(
        "YouTube bloqueó la descarga de este video desde el servidor "
        "(IP de datacenter sin sesión). Probá: 1) volver a intentar, "
        "2) usar otro link del mismo tema, o 3) subir el archivo de audio "
        "directamente (siempre funciona). Si el problema persiste, el dueño "
        "del servicio puede configurar la variable YT_COOKIES con cookies "
        "de una cuenta de YouTube (ver README)."
        + (f" Detalle: {last_err}" if last_err else "")
    )


def render_phrases(src_mp3, segments, out_mp3, gap=0.2):
    """Corta los segmentos (start, end) del mp3 original y los concatena
    separados por un pequeño silencio. Devuelve la ruta del mp3 resultante.

    (v30) TODOS los cortes se hacen en UN SOLO comando ffmpeg con
    filter_complex (atrim + concat): un render con 15 segmentos pasa de
    ~20 s a ~3 s. La envolvente del audio ya está cacheada en app.py, así que
    el primer play es casi instantáneo."""
    if len(segments) == 1:
        s, e = segments[0]
        r = ffmpeg_util.run_ffmpeg(["-y", "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
                 "-i", src_mp3, "-ac", "2", "-ar", "44100",
                 "-c:a", "libmp3lame", "-b:a", "192k", out_mp3])
        if r.returncode != 0:
            raise RuntimeError("ffmpeg: no se pudo cortar el segmento: " + r.stderr[-300:])
        return out_mp3

    # varios segmentos: un solo ffmpeg con atrim + concat (silencios entre medias)
    filters = []
    labels = []
    for k, (s, e) in enumerate(segments):
        filters.append(
            f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{k}]")
        labels.append(f"[a{k}]")
        if k < len(segments) - 1:
            filters.append(
                f"anullsrc=r=44100:cl=stereo,atrim=0:{gap:.3f},asetpts=PTS-STARTPTS[s{k}]")
            labels.append(f"[s{k}]")
    filters.append("".join(labels) + f"concat=n={len(labels)}:v=0:a=1[out]")
    fc = ";".join(filters)
    r = ffmpeg_util.run_ffmpeg(["-y", "-i", src_mp3, "-filter_complex", fc,
             "-map", "[out]", "-ac", "2", "-ar", "44100",
             "-c:a", "libmp3lame", "-b:a", "192k", out_mp3])
    if r.returncode != 0:
        # si el filter_complex falla (ffmpeg viejo), volver al método por partes
        _render_phrases_fallback(src_mp3, segments, out_mp3, gap)
    return out_mp3


def _render_phrases_fallback(src_mp3, segments, out_mp3, gap=0.2):
    """Versión por partes (un ffmpeg por segmento + concat)."""
    import tempfile
    import uuid
    workdir = os.path.join(os.path.dirname(os.path.abspath(out_mp3)) or ".",
                           ".tmp_" + uuid.uuid4().hex[:8])
    os.makedirs(workdir, exist_ok=True)
    try:
        seg_files = []
        for k, (s, e) in enumerate(segments):
            seg = os.path.join(workdir, f"seg{k}.wav")
            r = ffmpeg_util.run_ffmpeg(["-y", "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
                     "-i", src_mp3, "-ac", "2", "-ar", "44100",
                     "-c:a", "pcm_s16le", seg])
            if r.returncode != 0:
                raise RuntimeError("ffmpeg: no se pudo cortar el segmento: " + r.stderr[-300:])
            seg_files.append(seg)
        sil = os.path.join(workdir, "sil.wav")
        ffmpeg_util.run_ffmpeg(["-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
             "-t", f"{gap:.3f}", "-c:a", "pcm_s16le", sil])
        items = []
        for k in range(len(seg_files)):
            items.append(seg_files[k])
            if k < len(seg_files) - 1:
                items.append(sil)
        with open(os.path.join(workdir, "list.txt"), "w", encoding="utf-8") as f:
            for p in items:
                f.write(f"file '{p}'\n")
        r = ffmpeg_util.run_ffmpeg(["-y", "-f", "concat", "-safe", "0",
                 "-i", os.path.join(workdir, "list.txt"),
                 "-c:a", "libmp3lame", "-b:a", "192k", out_mp3])
        if r.returncode != 0:
            raise RuntimeError("ffmpeg: no se pudo unir los fragmentos: " + r.stderr[-300:])
    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass
