"""Almacenamiento opcional en Cloudflare R2 (compatible con S3, 10 GB gratis).

Si las variables de entorno R2_* están configuradas, los metadatos y los
archivos de audio se guardan en R2 (persisten a $0 en Render/Heroku/lo que sea).
Si no están configuradas, todo se guarda en disco local (comportamiento local).

Variables de entorno (se configuran en Render -> Environment):
    R2_ACCOUNT_ID   (id de cuenta Cloudflare, ej: 9d2a...)
    R2_ACCESS_KEY   (Access Key ID del token de API)
    R2_SECRET_KEY   (Secret Access Key del token de API)
    R2_BUCKET       (nombre del bucket, ej: resuena)
    R2_ENDPOINT     (opcional; por defecto https://{ACCOUNT_ID}.r2.cloudflarestorage.com)
"""
import os
from functools import lru_cache


def cloud_enabled():
    return bool(os.environ.get("R2_BUCKET")
                and os.environ.get("R2_ACCESS_KEY")
                and os.environ.get("R2_SECRET_KEY"))


@lru_cache(maxsize=1)
def _client():
    import boto3
    from botocore.config import Config
    endpoint = os.environ.get("R2_ENDPOINT") or (
        f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
        if os.environ.get("R2_ACCOUNT_ID") else None
    )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def put_bytes(key, data):
    _client().put_object(Bucket=os.environ["R2_BUCKET"], Key=key, Body=data)


def get_bytes(key):
    r = _client().get_object(Bucket=os.environ["R2_BUCKET"], Key=key)
    return r["Body"].read()


def put_file(key, local_path):
    with open(local_path, "rb") as f:
        _client().put_object(Bucket=os.environ["R2_BUCKET"], Key=key, Body=f)


def get_file(key, local_path):
    data = get_bytes(key)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(data)
    return local_path


def list_keys(prefix):
    """Devuelve lista de (key, last_modified) bajo el prefijo, ordenada por antigüedad."""
    c = _client()
    paginator = c.get_paginator("list_objects_v2")
    out = []
    for page in paginator.paginate(Bucket=os.environ["R2_BUCKET"], Prefix=prefix):
        for o in page.get("Contents", []):
            out.append((o["Key"], o.get("LastModified")))
    out.sort(key=lambda x: x[1] or 0)
    return out


def delete_prefix(prefix):
    c = _client()
    paginator = c.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=os.environ["R2_BUCKET"], Prefix=prefix):
        keys = [o["Key"] for o in page.get("Contents", [])]
        if keys:
            c.delete_objects(Bucket=os.environ["R2_BUCKET"], Delete={"Objects": [{"Key": k} for k in keys]})


def delete_key(key):
    _client().delete_object(Bucket=os.environ["R2_BUCKET"], Key=key)
