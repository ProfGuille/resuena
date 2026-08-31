"""Pre-descarga los modelos Whisper durante el BUILD de Render.

El disco del runtime es efímero en Render free: si el modelo se descarga en
cada arranque (porque el cache se pierde con cada reinicio/redeploy), la
descarga + carga + transcripción juntas pueden superar los 512 MB y el
proceso muere (502 en loop). Descargando el modelo en el build queda horneado
en la imagen y el runtime NUNCA toca la red para el modelo.

IMPORTANTE: este script NUNCA hace fallar el build. Si la descarga falla
(red, HuggingFace caído, etc.) imprime una advertencia y sale con código 0:
la app igual arranca y, si hace falta, baja el modelo en runtime (como antes).
El objetivo es que un problema del modelo NO rompa el deploy.
"""
import os
import sys


def _try_download(name, repo):
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(f"[build-model] huggingface_hub no disponible: {e}", flush=True)
        return
    dest = os.path.join(BASE, "models", f"faster-whisper-{name}")
    if os.path.exists(os.path.join(dest, "model.bin")):
        print(f"[build-model] {name} ya horneado", flush=True)
        return
    try:
        print(f"[build-model] descargando {repo} -> {dest}", flush=True)
        snapshot_download(repo_id=repo, local_dir=dest)
        ok = os.path.exists(os.path.join(dest, "model.bin"))
        print(f"[build-model] {name} {'listo' if ok else 'INCOMPLETO'}", flush=True)
    except Exception as e:
        print(f"[build-model] descarga de {name} falló (no fatal): {str(e)[:160]}", flush=True)


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
            print(f"[build-model] modelo desconocido: {name} (omitido)", flush=True)
            continue
        _try_download(name, repo)
    print("[build-model] build de modelos completo (no fatal)", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
