"""Pre-descarga los modelos Whisper durante el BUILD de Render.

El disco del runtime es efímero en Render free: si el modelo se descarga en
cada arranque (porque el cache se pierde con cada reinicio/redeploy), la
descarga + carga + transcripción juntas pueden superar los 512 MB y el
proceso muere (502 en loop). Descargando el modelo en el build queda horneado
en la imagen y el runtime NUNCA toca la red para el modelo.
"""
import os

from huggingface_hub import snapshot_download

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    # en el plan free (512 MB) solo se usa tiny; base/small requieren más RAM.
    # Se pueden hornear con BUILD_MODELS="tiny,base" si algún día se sube de plan.
}


def main():
    names = [n.strip() for n in os.environ.get("BUILD_MODELS", "tiny").split(",") if n.strip()]
    for name in names:
        repo = REPOS.get(name)
        if not repo:
            print(f"modelo desconocido: {name} (omitido)", flush=True)
            continue
        dest = os.path.join(MODELS, f"faster-whisper-{name}")
        if os.path.exists(os.path.join(dest, "model.bin")):
            print(f"modelo {name} ya horneado", flush=True)
            continue
        print(f"descargando {repo} -> {dest}", flush=True)
        snapshot_download(repo_id=repo, local_dir=dest)
        print(f"modelo {name} listo", flush=True)
    print("build de modelos completo", flush=True)


if __name__ == "__main__":
    main()
