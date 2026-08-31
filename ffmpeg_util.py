"""Descarga e instala ffmpeg (binario estático) sin apt-get.

Necesario para Render free: ahí apt-get falla (sistema de archivos de solo
lectura durante el build), así que bajamos el binario estático de John Van
Sickle (ffmpeg-static), que no requiere root ni apt.
"""
import os
import shutil
import stat
import subprocess
import tarfile
import urllib.request

URL = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ffmpeg")


def ensure_ffmpeg():
    """Devuelve el directorio con ffmpeg/ffprobe listos, instalándolos si hace falta."""
    bin_dir = os.path.join(CACHE, "bin")
    ffmpeg = os.path.join(bin_dir, "ffmpeg")
    ffprobe = os.path.join(bin_dir, "ffprobe")
    if os.path.exists(ffmpeg) and os.path.exists(ffprobe):
        return bin_dir

    os.makedirs(CACHE, exist_ok=True)
    tarball = os.path.join(CACHE, "ffmpeg.tar.xz")

    print("Descargando ffmpeg estático...", flush=True)
    urllib.request.urlretrieve(URL, tarball)

    # el tar.xz contiene una carpeta ffmpeg-xxx-static/ con los binarios
    # ffmpeg y ffprobe en la RAÍZ de esa carpeta (no en bin/)
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
    """Corre ffmpeg (por PATH si existe, si no con el estático)."""
    if _system_has("ffmpeg"):
        return subprocess.run(["ffmpeg", *args], **kw)
    bin_dir = ensure_ffmpeg()
    return subprocess.run([os.path.join(bin_dir, "ffmpeg"), *args], **kw)


def run_ffprobe(args, **kw):
    if _system_has("ffprobe"):
        return subprocess.run(["ffprobe", *args], **kw)
    bin_dir = ensure_ffmpeg()
    return subprocess.run([os.path.join(bin_dir, "ffprobe"), *args], **kw)


if __name__ == "__main__":
    d = ensure_ffmpeg()
    print("ffmpeg instalado en:", d)
    r = run_ffmpeg(["-version"])
    print(r.stdout.decode()[:80])
