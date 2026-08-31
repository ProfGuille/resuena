"""Persistencia en un repo de GitHub (gratis, SIN tarjeta).

La app guarda canciones, selecciones y audios como archivos dentro de tu repo
de GitHub usando la API oficial. Así los datos sobreviven a cualquier reinicio
de Render, al plan free y a los redeploys.

Variables de entorno:
    GITHUB_TOKEN   (Personal Access Token "fine-grained", con permiso
                    Contents: Read and write, limitado a tu repo resuena)
    GITHUB_REPO    (ej: "tuusuario/resuena")
    GITHUB_BRANCH  (opcional, por defecto "main")
    GITHUB_API_BASE (opcional, para tests; por defecto https://api.github.com)
"""
import base64
import json
import os
import urllib.error
import urllib.request


def enabled():
    return bool(os.environ.get("GITHUB_TOKEN") and os.environ.get("GITHUB_REPO"))


def _base():
    return os.environ.get("GITHUB_API_BASE", "https://api.github.com")


def _api(method, path, body=None):
    repo = os.environ["GITHUB_REPO"].strip("/")
    branch = os.environ.get("GITHUB_BRANCH", "main")
    url = f"{_base()}/repos/{repo}/contents/{path}?ref={branch}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"Bearer {os.environ['GITHUB_TOKEN']}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if body is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(body).encode("utf-8")
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            raw = r.read().decode("utf-8")
            return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode("utf-8"))
        except Exception:
            detail = {}
        return e.code, detail


def _read_file(path):
    """Devuelve (sha, contenido_base64) o (None, None) si no existe."""
    st, data = _api("GET", path)
    if st != 200:
        return None, None
    return data.get("sha"), data.get("content")


def get_json(path, default):
    _, content = _read_file(path)
    if content is None:
        return default
    try:
        return json.loads(base64.b64decode(content).decode("utf-8"))
    except Exception:
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
    st, data = _api("GET", path)
    if st != 200:
        return False
    raw = base64.b64decode(data["content"])
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
