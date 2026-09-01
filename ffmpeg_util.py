"""ffmpeg disponible SIN descargas en runtime (clave para que el play sea rápido).

Prioridad de búsqueda:
  1. ffmpeg del sistema (PATH)
  2. ffmpeg HORNEADO en el build via pip: el paquete imageio-ffmpeg trae el
     binario estático dentro del wheel (se instala en `pip install -r
     requirements.txt` durante el build de Render y queda en la imagen; el
     runtime NUNCA toca la red por ffmpeg). Se expone como `.ffmpeg/bin/ffmpeg`
     (symlink) para que todo el código lo use igual.
  3. descarga del binario estático de John Van Sickle (solo como último
     recurso, por ejemplo en local sin pip).

ffprobe: si no hay binario de ffprobe (imageio-ffmpeg no lo trae), run_ffprobe
devuelve None y audio_utils.ffprobe_duration obtiene la duración parseando la
salida de ffmpeg -i (no hace falta ffprobe).
"""
import os
import shutil
import stat
import subprocess
import tarfile
import urllib.request

URL = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ffmpeg")


def _imageio_exe():
    """Ruta del binario ffmpeg incluido en el wheel imageio-ffmpeg (pip)."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    return None


def ensure_ffmpeg():
    """Devuelve un directorio con un binario llamado `ffmpeg` (y ffprobe si lo
    hay). Prioridad: imageio-ffmpeg (pip, horneado en el build) y luego la
    descarga estática como último recurso. Nunca falla en runtime si el build
    horneó imageio-ffmpeg."""
    bin_dir = os.path.join(CACHE, "bin")
    exe = _imageio_exe()
    if exe:
        os.makedirs(bin_dir, exist_ok=True)
        link = os.path.join(bin_dir, "ffmpeg")
        if not os.path.exists(link):
            try:
                os.symlink(exe, link)
            except Exception:
                shutil.copy2(exe, link)
                os.chmod(link, os.stat(link).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return bin_dir
    if os.path.exists(os.path.join(bin_dir, "ffmpeg")):
        return bin_dir

    # último recurso: descargar el estático (normalmente nunca pasa en Render)
    os.makedirs(CACHE, exist_ok=True)
    tarball = os.path.join(CACHE, "ffmpeg.tar.xz")

    print("Descargando ffmpeg estático...", flush=True)
    urllib.request.urlretrieve(URL, tarball)

    os.makedirs(bin_dir, exist_ok=True)
    with tarfile.open(tarball, "r:xz") as tf:
        for m in tf.getmembers():
            name = m.name.split("/")[-1]
            if name in ("ffmpeg", "ffprobe"):
                src = tf.extractfile(m)
                if src is None:
                    continue
                with open(os.path.join(bin_dir, name), "wb") as f:
                    f.write(src.read())
    os.remove(tarball)

    for name in ("ffmpeg", "ffprobe"):
        p = os.path.join(bin_dir, name)
        if os.path.exists(p):
            os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir


def _system_has(binary):
    return shutil.which(binary) is not None


def run_ffmpeg(args, **kw):
    """Corre ffmpeg (por PATH si existe, si no con el horneado/estático)."""
    if _system_has("ffmpeg"):
        return subprocess.run(["ffmpeg", *args], **kw)
    bin_dir = ensure_ffmpeg()
    return subprocess.run([os.path.join(bin_dir, "ffmpeg"), *args], **kw)


def run_ffprobe(args, **kw):
    """Corre ffprobe si existe. Si no hay ffprobe devuelve None (el llamador
    usa ffmpeg -i como alternativa para la duración)."""
    if _system_has("ffprobe"):
        return subprocess.run(["ffprobe", *args], **kw)
    bin_dir = os.path.join(CACHE, "bin")
    if os.path.exists(os.path.join(bin_dir, "ffprobe")):
        return subprocess.run([os.path.join(bin_dir, "ffprobe"), *args], **kw)
    return None


if __name__ == "__main__":
    d = ensure_ffmpeg()
    print("ffmpeg instalado en:", d)
    r = run_ffmpeg(["-version"])
    print(r.stdout.decode()[:80])
