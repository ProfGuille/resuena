"""Utilidades de audio: yt-dlp, ffmpeg (conversión, corte y concatenación)."""
import os
import shutil
import subprocess

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
    # Estrategias de descarga de YouTube. Sin la librería curl_cffi (que sí
    # está en requirements.txt), YouTube devuelve 403 a IPs de datacenter
    # (Render). Con curl_cffi, yt-dlp puede imitar un navegador real y pasa.
    # Se prueban en orden: las impersonadas primero (más confiables), luego
    # los clientes móviles/tv (menos bloqueados), y al final web pelado.
    strategies = [
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
        {"name": "web",           "clients": ["web"],           "fmt": "bestaudio/best"},
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
            "extractor_args": {"youtube": {"player_client": st["clients"]}},
        }
        if st.get("impersonate") and ImpersonateTarget is not None:
            try:
                opts["impersonate"] = ImpersonateTarget.from_str(st["impersonate"])
            except Exception:
                pass
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            path = ydl.prepare_filename(info)
            if os.path.exists(path):
                return path, info.get("title"), info.get("uploader")
            base = os.path.splitext(path)[0]
            for ext in (".webm", ".m4a", ".mp3", ".opus", ".mp4"):
                cand = base + ext
                if os.path.exists(cand):
                    return cand, info.get("title"), info.get("uploader")
        except Exception as e:
            last_err = str(e)[:200]
            _cleanup()
            continue
    _cleanup()
    raise RuntimeError(
        "YouTube bloqueó la descarga de este video desde el servidor "
        "(HTTP 403/Forbidden). Puede ser temporal o depender del video/link. "
        "Probá: 1) volver a intentar, 2) usar otro link del mismo tema, o "
        "3) subir el archivo de audio directamente (siempre funciona)."
        + (f" Detalle: {last_err}" if last_err else "")
    )


def render_phrases(src_mp3, segments, out_mp3, gap=0.2):
    """Corta los segmentos (start, end) del mp3 original y los concatena
    separados por un pequeño silencio. Devuelve la ruta del mp3 resultante.

    (v29) usa un directorio temporal ÚNICO por llamada: el pre-render del
    arranque y los pedidos de los usuarios corren en paralelo y los nombres
    fijos (.seg0.wav, .sil.wav, .list.txt) se pisaban entre sí."""
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
    return out_mp3
