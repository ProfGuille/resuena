# Resuena 🔊

**Las frases que te resuenan de una canción, en un audio corto.**

Subís una canción (archivo de audio **o** link de YouTube) junto con su letra.
La app detecta automáticamente dónde está cada palabra en el audio y muestra la
letra "pintable". Marcás las frases que te resuenan y la app corta esos
fragmentos del audio original y los une en un **MP3 corto con solo tus frases**,
listo para escuchar y descargar.

## Cómo funciona (100% gratis)

| Pieza | Herramienta | Costo |
|---|---|---|
| Servidor web + API | FastAPI + Uvicorn | gratis |
| Reconocimiento de voz (timestamps palabra por palabra) | faster-whisper (modelo `small`, CPU) | gratis |
| Descarga de audio de YouTube | yt-dlp | gratis |
| Corte y unión de audio | ffmpeg | gratis |
| Alineación letra ↔ audio | propio (programación dinámica + rapidfuzz) | gratis |
| Base de datos | archivos JSON locales | gratis |

## Correrlo en tu PC

```bash
cd resuena
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                    # o: uvicorn app:app --host 0.0.0.0 --port 8000
```

Abrí **http://localhost:8000**.

Requisitos: **Python 3.10+** y **ffmpeg** en el PATH
(`sudo apt install ffmpeg` en Ubuntu, `brew install ffmpeg` en Mac).

> En la primera canción, faster-whisper descarga el modelo (~460 MB para
> `small`) a `~/.cache/huggingface` y tarda un poco más.

---

## Publicarlo online (gratis) — paso a paso

La idea es: **tus archivos → GitHub (gratis) → Render (gratis)**. El proyecto ya
trae la configuración lista (`render.yaml`), así que es casi automático.

### Paso 1: prepará el proyecto

Tenés dos opciones:

- **Opción A (fácil):** descargá el archivo **`resuena.zip`** incluido en la
  carpeta del proyecto y descomprimilo en tu PC. Adentro está todo: `app.py`,
  `static/`, `requirements.txt`, `render.yaml`, etc.
- **Opción B:** copiá la carpeta `resuena/` tal cual.

### Paso 2: creá un repo en GitHub

1. Entrá a [github.com](https://github.com) y creá una cuenta (gratis).
2. Botón **+** (arriba a la derecha) → **New repository**.
3. Nombre: `resuena` → **Create repository** (público o privado, da igual).
4. En la página del repo, botón **"uploading an existing file"** → arrastrá
   **todos los archivos** de la carpeta del proyecto (app.py, static, etc.) →
   **Commit changes**.

### Paso 3: conectalo con Render

1. Creá una cuenta en [render.com](https://render.com) (gratis, con GitHub).
2. Dashboard → **New +** → **Web Service**.
3. Conectá tu cuenta de GitHub y elegí el repo `resuena`.
4. **Render detecta automáticamente el `render.yaml`** y completa la
   configuración. Verificá que quede así:
   - **Runtime:** Python 3
   - **Build:** `pip install -r requirements.txt`
   - **Start:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free
5. Botón **Deploy Web Service** (azul, al final).

### Paso 4: esperá y usala

- El deploy tarda 2–5 minutos (instala dependencias).
- Al terminar te da una URL tipo **`https://resuena.onrender.com`** → esa es tu
  app online. Se puede abrir desde cualquier dispositivo con el link.

### Paso 5 (recomendado): disco persistente

En el plan free, Render se "duerme" a los 15 min de inactividad y los archivos
se borran al redeployar. Para que la biblioteca de canciones **sobreviva**:

1. En el servicio de Render → pestaña **Disks** → **Add Disk**.
2. Tamaño: **1 GB** (gratis) → Mount path: **`/var/data`** → Save.
3. Luego en **Environment** agregá la variable:
   - `DATA_DIR` = `/var/data`

Con eso las canciones, transcripciones y renders quedan guardados en el disco.

> Nota: con el plan free la transcripción es lenta al principio (el modelo se
> descarga al primer uso). Si querés más velocidad a costa de algo de precisión,
> agregá la variable `WHISPER_MODEL=tiny`.

### Alternativa: Railway

[railway.app](https://railway.app) también tiene plan gratis y usa el
**`Procfile`** incluido: conectás el repo de GitHub, elegís "Deploy from
GitHub", y se configura solo. Para persistencia, agregá un volumen en
`/var/data` con `DATA_DIR=/var/data`.

---

## Cómo se usa

1. **Subir**: botón "＋ Subir canción" → archivo (MP3/WAV/OGG/M4A…) o link de YouTube + pegar la letra. Podés indicar el idioma para mejor precisión (o dejarlo en "Auto").
2. **Esperar**: la app analiza y alinea la letra con el audio (1–3 min).
3. **Pintar**: hacé clic o arrastrá sobre las palabras para marcarlas; Shift+clic extiende. Las líneas con ⚠ "sin audio" no coinciden con lo cantado.
4. **Escuchar**: "🎬 Generar audio con mis frases" corta y une los fragmentos. Tildá *"todas las apariciones"* para incluir coros que se repiten. Botón ▶ por línea para previsualizarla sola.
5. **Descargar** el MP3 o compartir el link de la canción: cada persona pinta sus **propias** frases (se guardan por usuario).

## Estructura

```
resuena/
├── app.py            # API + servidor (FastAPI)
├── store.py          # persistencia JSON
├── align.py          # alineación letra ↔ audio + búsqueda de repeticiones
├── audio_utils.py    # yt-dlp, ffmpeg (convertir, cortar, unir)
├── static/index.html # interfaz (sin dependencias externas)
├── requirements.txt
├── render.yaml       # config automática para Render
└── Procfile          # config para Railway/Heroku
```

## Límites conocidos

- El free tier de Render es lento para transcripción. Con `WHISPER_MODEL=tiny` es más rápido a cambio de algo de precisión.
- La alineación funciona mejor si la letra es **exactamente** lo que se canta.
- YouTube a veces bloquea descargas desde IPs de servidores (error 403). En ese caso probá otro link o subí el archivo de audio.
- No hay cuentas ni login: cada usuario es un id anónimo guardado en su navegador.
