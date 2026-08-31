"""Persistencia en un repo de GitHub (gratis, SIN tarjeta).

La app guarda canciones, selecciones y audios como archivos dentro de tu repo
de GitHub usando la API oficial. Así los datos sobreviven a cualquier reinicio
de Render, al plan free y a los redeploys.

Puntos clave de robustez:
- NUNCA lanza excepciones por problemas de red: devuelve status 0 (error de red)
  o el código HTTP; la capa superior decide (usa la copia en memoria).
- Para BAJAR archivos usa el endpoint "raw" de la API (Accept:
  application/vnd.github.raw) porque la API normal NO devuelve el contenido
  base64 de archivos > 1 MB (los devuelve vacíos) — eso rompía los mp3 de 5 MB.

Variables de entorno:
    GITHUB_TOKEN   (Personal Access Token "fine-grained", con permiso
                    Contents: Read and write, limitado a tu repo resuena)
    GITHUB_REPO    (ej: "tuusuario/resuena")
    GITHUB_BRANCH  (opcional, por defecto "main")
    GITHUB_API_BASE (opcional, para tests; por defecto https://api.github.com)
    GITHUB_TIMEOUT (opcional, segundos; por defecto 10)
"""
import base64
import json
import os
import urllib.error
import urllib.request

TIMEOUT = int(os.environ.get("GITHUB_TIMEOUT", "10"))


def enabled():
    return bool(os.environ.get("GITHUB_TOKEN") and os.environ.get("GITHUB_REPO"))


def _base():
    return os.environ.get("GITHUB_API_BASE", "https://api.github.com")


def _api(method, path, body=None, raw=False):
    """Llama a la API de GitHub.

    Devuelve (status, data):
      - status HTTP real (200/201/404/403/...) con su data (dict o bytes crudos)
      - status 0 si hubo error de red/timeout. NUNCA lanza.
    """
    repo = os.environ["GITHUB_REPO"].strip("/")
    branch = os.environ.get("GITHUB_BRANCH", "main")
    url = f"{_base()}/repos/{repo}/contents/{path}?ref={branch}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"Bearer {os.environ['GITHUB_TOKEN']}")
    if raw:
        req.add_header("Accept", "application/vnd.github.raw+json")
    else:
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if body is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(body).encode("utf-8")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            b = r.read()
            if raw:
                return r.status, b
            return r.status, (json.loads(b.decode("utf-8")) if b else {})
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode("utf-8"))
        except Exception:
            detail = {}
        return e.code, detail
    except Exception:
        # URLError, socket.timeout, connection reset, DNS… nunca propagar
        return 0, {}


def _read_file(path):
    """Devuelve (sha, content_base64) para archivos chicos.

    - (None, None): el archivo no existe (404) u otro código HTTP.
    - (0, None):   error de red (no se pudo consultar).
    - (sha, content): ok.
    """
    st, data = _api("GET", path)
    if st == 0:
        return 0, None
    if st != 200:
        return None, None
    return data.get("sha"), data.get("content")


def fetch_json(path):
    """(exito, valor) para leer JSON sin confundir "no existe" con "red caída".

    - (False, None): error de red → el llamador debe conservar lo que tenía.
    - (True, None):  no existe (404) → default vacío.
    - (True, dict):  contenido.
    """
    st, content = _read_file(path)
    if st == 0:
        return False, None
    if content is None:
        return True, None
    try:
        return True, json.loads(base64.b64decode(content).decode("utf-8"))
    except Exception:
        return True, None


def get_json(path, default):
    ok, val = fetch_json(path)
    if ok and val is not None:
        return val
    return default


def put_json(path, obj):
    raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    sha, _ = _read_file(path)
    body = {
        "message": f"resuena: update {path}",
        "content": base64.b64encode(raw).decode("utf-8"),
    }
    if sha:
        body["sha"] = sha
    st, _ = _api("PUT", path, body)
    return st in (200, 201)


def get_file(path, local_path):
    """Descarga un archivo (cualquier tamaño ≤ 100 MB) usando el endpoint raw.

    Devuelve True si se escribió el archivo, False si no existe o hubo error.
    """
    st, raw = _api("GET", path, raw=True)
    if st != 200 or raw is None:
        return False
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(raw)
    return True


def put_file(path, local_path):
    with open(local_path, "rb") as f:
        raw = f.read()
    sha, _ = _read_file(path)
    body = {
        "message": f"resuena: update {path}",
        "content": base64.b64encode(raw).decode("utf-8"),
    }
    if sha:
        body["sha"] = sha
    st, _ = _api("PUT", path, body)
    return st in (200, 201)


def list_dir(path):
    """Devuelve los nombres de los archivos de un directorio (o [] si no existe)."""
    st, data = _api("GET", path.rstrip("/"))
    if st != 200 or not isinstance(data, list):
        return []
    return [item.get("name") for item in data if item.get("type") == "file"]


def delete_path(path):
    st, data = _api("GET", path)
    if st != 200:
        return
    body = {"message": f"resuena: delete {path}", "sha": data.get("sha")}
    _api("DELETE", path, body)


def delete_prefix(prefix):
    prefix = prefix.rstrip("/")
    for name in list_dir(prefix):
        delete_path(f"{prefix}/{name}")
