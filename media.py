"""Capa única para archivos de audio: GitHub repo o Cloudflare R2 o local.

GitHub (ghstore) tiene prioridad; R2 (cloud) es alternativa; si ninguno está
configurado, todo queda en disco local (modo local/preview).
"""
import cloud
import ghstore


def persistent():
    return ghstore.enabled() or cloud.cloud_enabled()


def put_file(key, local_path):
    if ghstore.enabled():
        return ghstore.put_file(key, local_path)
    if cloud.cloud_enabled():
        cloud.put_file(key, local_path)
        return True
    return False


def get_file(key, local_path):
    if ghstore.enabled():
        return ghstore.get_file(key, local_path)
    if cloud.cloud_enabled():
        cloud.get_file(key, local_path)
        return True
    return False


def delete_prefix(prefix):
    if ghstore.enabled():
        ghstore.delete_prefix(prefix)
    if cloud.cloud_enabled():
        cloud.delete_prefix(prefix)
