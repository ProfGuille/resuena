"""Alineación de la letra (del usuario) contra los timestamps palabra-a-palabra
que produce faster-whisper, usando programación dinámica (Needleman-Wunsch),
con relleno por interpolación y respaldo a nivel de segmentos.

Esto permite que el usuario pegue su propia letra (aunque no coincida
exactamente con la transcripción) y que la mayoría de las palabras queden con
su instante de inicio/fin en el audio.
"""
import re
import unicodedata

from rapidfuzz import fuzz


def norm(text):
    """Normaliza: minúsculas, sin tildes ni puntuación, espacios simples."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
    return " ".join(text.split())


# Marcadores típicos de páginas de letras: [Coro], Verso 2, (x2), ♪, etc.
_MARKER_RE = re.compile(
    r"^[\[\(]?\s*(intro|outro|verse|vers|versi[oó]n|chorus|coro|estribillo|"
    r"pre[- ]?coro|pre[- ]?estribillo|bridge|puente|instrumental|interlude|"
    r"interludio|solo|final|refr[aá]n|parte|secci[oó]n|post[- ]?coro|"
    r"post[- ]?estribillo)\s*(\d*)\s*[\]\)]?$",
    re.IGNORECASE,
)
_X2_RE = re.compile(r"\s*[\[\(](x\d+|\d+x|2x|veces)[\]\)]\s*$", re.IGNORECASE)
_BRACKET_RE = re.compile(r"[\[\(].*?[\]\)]")


def parse_lyrics(text):
    """Convierte la letra pegada en líneas con sus palabras, limpiando
    marcadores de páginas de letras ([Coro], (x2), ♪…) que ensucian la alineación."""
    lines = []
    for raw in text.splitlines():
        line = raw.strip().strip("♪♫")
        # limpiar marcadores pegados a la línea: "***Kyrie Eleison***",
        # "-Estribillo-", "♪…♪" etc. (comunes en letras pegadas de la web)
        line = line.strip("*—–-·•♪♫")
        if not line:
            continue
        if _MARKER_RE.match(line):
            continue
        line = _X2_RE.sub("", line)
        line = _BRACKET_RE.sub("", line).strip()
        if not line:
            continue
        lines.append({"raw": line, "words": [{"raw": w} for w in line.split()]})
    return lines


def _sim(a, b, cache):
    key = (a, b)
    v = cache.get(key)
    if v is not None:
        return v
    if a == b:
        v = 1.0
    elif not a or not b or a[0] != b[0]:
        v = 0.0
    else:
        v = fuzz.ratio(a, b) / 100.0
    cache[key] = v
    return v


def _linfit(ps, ts):
    """Ajuste lineal t = a*p + b por mínimos cuadrados."""
    n = len(ps)
    mx = sum(ps) / n
    my = sum(ts) / n
    num = sum((p - mx) * (t - my) for p, t in zip(ps, ts))
    den = sum((p - mx) ** 2 for p in ps)
    if den:
        a = num / den
    else:
        a = 0.0
    return a, my - a * mx


def _interpolate_line(line):
    """Rellena con tiempos las palabras de una línea que quedaron sin match,
    cuando al menos una palabra de la línea sí se ubicó. Camina palabra a
    palabra desde los anclajes (sin extrapolar lejos, para no invadir el
    audio de otras frases)."""
    words = line["words"]
    n = len(words)
    if not any(w.get("m") for w in words) or all(w.get("m") for w in words):
        return
    durs = sorted(w["e"] - w["s"] for w in words if w.get("m"))
    step = max(0.2, min(0.6, durs[len(durs) // 2]))

    i = 0
    while i < n:
        if words[i].get("m"):
            i += 1
            continue
        a = i
        while i < n and not words[i].get("m"):
            i += 1
        b = i - 1  # run [a..b] de palabras sin match
        prev = None
        for k in range(a - 1, -1, -1):
            if words[k].get("m"):
                prev = k
                break
        nxt = None
        for k in range(b + 1, n):
            if words[k].get("m"):
                nxt = k
                break
        cnt = b - a + 1
        if prev is not None and nxt is not None:
            # hueco interior: repartir entre el fin del ancla previa y el
            # inicio del ancla siguiente
            span = max(0.15 * cnt, words[nxt]["s"] - words[prev]["e"])
            each = span / cnt
            for k in range(a, b + 1):
                words[k]["s"] = words[prev]["e"] + (k - a) * each
                words[k]["e"] = words[k]["s"] + min(each, step)
                words[k]["m"] = True
                words[k]["sim"] = 0.5
        elif prev is not None:
            # cola final: caminar hacia adelante con paso acotado
            cur = words[prev]["e"]
            for k in range(a, b + 1):
                words[k]["s"] = cur
                words[k]["e"] = cur + step
                words[k]["m"] = True
                words[k]["sim"] = 0.5
                cur = words[k]["e"]
        else:
            # cabeza inicial: caminar hacia atrás desde el ancla siguiente
            cur = words[nxt]["s"]
            for k in range(b, a - 1, -1):
                words[k]["e"] = cur
                words[k]["s"] = max(0.0, cur - step)
                words[k]["m"] = True
                words[k]["sim"] = 0.5
                cur = words[k]["s"]


def _line_window_match(lines, transcript, min_score=0.68):
    """Para líneas completas sin ninguna palabra ubicada: busca la mejor
    ventana de palabras del transcript (ventana deslizante con fuzzy match)
    con la restricción de que arranque después de lo ya recorrido en la letra
    (procesa en orden y es monótono). Así los coros que se repiten se ubican
    en su aparición correcta en vez de duplicar la primera.

    El umbral (min_score) es ALTO a propósito: si la transcripción no contiene
    las palabras (p.ej. el modelo tiny malinterpreta la voz cantada), es
    preferible dejar la línea "sin audio asociado" ANTES que ubicarla en un
    tramo equivocado del audio (que haría sonar otra frase al pintarla).
    """
    tn = [norm(w["word"]) for w in transcript]
    T = len(tn)
    if T == 0:
        return
    prev_end = 0.0
    for line in lines:
        if any(w.get("m") for w in line["words"]):
            for w in line["words"]:
                if w.get("m"):
                    prev_end = max(prev_end, w["e"])
            continue
        ln = [norm(w["raw"]) for w in line["words"]]
        L = len(ln)
        if L == 0:
            continue
        win_lens = sorted({L, max(1, L - 1), L + 1})
        best = None
        for win_len in win_lens:
            if win_len > T:
                continue
            for k in range(0, T - win_len + 1):
                if transcript[k]["start"] < prev_end - 0.2:
                    continue  # solo apariciones después de lo ya recorrido
                if fuzz.ratio(ln[0], tn[k]) < 55:
                    continue  # la 1ª palabra debe ser parecida (acelera mucho)
                win = tn[k:k + win_len]
                if all(a == b for a, b in zip(ln, win)):
                    score = 1.0
                else:
                    score = sum(fuzz.ratio(a, b) / 100.0 for a, b in zip(ln, win)) / L
                if win_len != L:
                    score *= 0.95  # penalizar leve el tamaño distinto
                if best is None or score > best[0]:
                    best = (score, k, k + win_len - 1)
        if best and best[0] >= min_score:
            score, k0, k1 = best
            # verificación extra: la 1ª palabra de la línea debe coincidir bien
            # con la 1ª de la ventana; si no, el match es dudoso y mejor dejar
            # la línea sin audio que ubicarla en un tramo equivocado.
            if fuzz.ratio(ln[0], tn[k0]) < 55:
                best = None
                score, k0, k1 = 0.0, 0, 0
        if best and best[0] >= min_score:
            score, k0, k1 = best
            seg_s = transcript[k0]["start"]
            seg_e = transcript[k1]["end"]
            span = seg_e - seg_s
            for i, w in enumerate(line["words"]):
                tc = seg_s + span * (i + 0.5) / L
                w["s"] = max(0.0, tc - 0.1)
                w["e"] = tc + 0.25
                w["m"] = True
                w["sim"] = round(score, 2)
            prev_end = max(prev_end, transcript[k1]["end"])


def align_lines(lines, transcript, segments=None):
    """Alinea cada palabra de la letra con una palabra del transcript.

    transcript: lista de dicts {"word", "start", "end"}.
    segments (opcional): lista de dicts {"text", "start", "end"} para respaldo.
    Devuelve (lines_anotadas, coverage).
    """
    GAP = -0.9
    MATCH_OK = 0.6  # similitud mínima para considerar una palabra "ubicada"

    lw = []  # (palabra normalizada, índice de línea, índice dentro de la línea)
    for li, line in enumerate(lines):
        for wi, w in enumerate(line["words"]):
            lw.append((norm(w["raw"]), li, wi))

    tw = [(norm(w["word"]), float(w["start"]), float(w["end"])) for w in transcript]
    n, m = len(lw), len(tw)

    for line in lines:
        for w in line["words"]:
            w["s"] = None
            w["e"] = None
            w["m"] = False
            w["tj"] = None

    if n == 0 or m == 0:
        return lines, 0.0

    cache = {}
    cols = m + 1
    F_prev = [GAP * j for j in range(cols)]
    F_prev[0] = 0.0
    tb = bytearray((n + 1) * cols)  # 1=diag, 2=arriba, 3=izquierda
    for j in range(1, cols):
        tb[j] = 3
    tb[0] = 0

    for i in range(1, n + 1):
        F_cur = [0.0] * cols
        F_cur[0] = GAP * i
        base = i * cols
        li_n = lw[i - 1][0]
        prev = F_prev
        for j in range(1, cols):
            s = _sim(li_n, tw[j - 1][0], cache)
            diag = prev[j - 1] + (2.0 * s - 1.0)
            up = prev[j] + GAP
            left = F_cur[j - 1] + GAP
            if diag >= up and diag >= left:
                F_cur[j] = diag
                tb[base + j] = 1
            elif up >= left:
                F_cur[j] = up
                tb[base + j] = 2
            else:
                F_cur[j] = left
                tb[base + j] = 3
        F_prev = F_cur

    matched = {}  # índice plano de la letra -> (índice transcript, similitud)
    i, j = n, m
    while i > 0 and j > 0:
        d = tb[i * cols + j]
        if d == 1:
            s = _sim(lw[i - 1][0], tw[j - 1][0], cache)
            if s >= MATCH_OK:
                matched[i - 1] = (j - 1, s)
            i -= 1
            j -= 1
        elif d == 2:
            i -= 1
        else:
            j -= 1

    last_end = 0.0
    for flat_i, (_, li, wi) in enumerate(lw):
        t = matched.get(flat_i)
        if t is None:
            continue
        tj, s = t
        orig = lines[li]["words"][wi]
        st = max(tw[tj][1], last_end)
        en = max(tw[tj][2], st)
        last_end = en
        orig["s"] = st
        orig["e"] = en
        orig["m"] = True
        orig["tj"] = tj
        orig["sim"] = round(s, 2)

    # ---- Descartar anclas dispersas (matches falsos) ---------------------
    # El emparejamiento global a veces une una palabra de la letra con una
    # palabra del transcript que está a SEGUNDOS de la anterior (el modelo
    # tiny transcribió mal esa parte y el DP forzó el match a otra sección,
    # p. ej. a un coro). Eso abría frases de pocas palabras durante 20 s y el
    # render incluía audio de las frases siguientes ("no frenó en la
    # selección"). Si dos palabras CONSECUTIVAS de la misma línea quedaron
    # emparejadas con un salto de más de GAP_MAX segundos, el match de la
    # segunda es falso: se descarta y se rellena por interpolación desde las
    # anclas reales.
    GAP_MAX = 2.0
    for line in lines:
        anchor_end = None
        for w in line["words"]:
            tj = w.get("tj")
            if w.get("m") and tj is not None:
                t_start = tw[tj][1]
                t_end = tw[tj][2]
                if anchor_end is None:
                    anchor_end = t_end
                elif t_start - anchor_end > GAP_MAX:
                    w["m"] = False
                    w["tj"] = None
                    w["s"] = None
                    w["e"] = None
                else:
                    anchor_end = t_end

    # Relleno por interpolación en líneas parcialmente ubicadas
    for line in lines:
        _interpolate_line(line)

    # ---- Anclar bordes interpolados a su match real ----------------------
    # Cuando el borde de una línea quedó interpolado (sin match), buscamos en
    # el transcript, CERCA de la posición interpolada, la palabra que más se
    # parezca a la primera/última de la línea y la usamos como ancla: así el
    # corte del render no se pasa de la frase (el problema "continuó con todo
    # el párrafo"). Solo si la similitud es alta y está cerca.
    for line in lines:
        changed = False
        for edge_idx in (0, -1):
            w = line["words"][edge_idx]
            if not w.get("m") or w.get("tj") is not None:
                continue
            if w.get("s") is None:
                continue
            target = norm(w["raw"])
            if not target:
                continue
            lo = w["s"] - 1.2
            hi = w["e"] + 1.2
            best = None
            for j, (tword, t0, t1) in enumerate(tw):
                if t0 < lo or t0 > hi or t1 <= t0:
                    continue
                sim = fuzz.ratio(target, tword) / 100.0
                if best is None or sim > best[0]:
                    best = (sim, j, t0, t1)
            if best and best[0] >= 0.80:
                _, j, t0, t1 = best
                w["tj"] = j
                w["s"] = t0
                w["e"] = t1
                w["sim"] = round(max(best[0], 0.5), 2)
                changed = True
        if changed:
            # el nuevo ancla cambió los límites: descartar los tiempos
            # interpolados (ya no cuadran) y rellenar de nuevo el interior
            for w in line["words"]:
                if w.get("tj") is None:
                    w["m"] = False
                    w["s"] = None
                    w["e"] = None
                    w["sim"] = 0.5
            _interpolate_line(line)

    # Respaldo: buscar cada línea sin ubicar como ventana fuzzy en el transcript
    _line_window_match(lines, transcript)

    # Recalcular span de cada línea y cobertura final
    ok = 0
    total = 0
    for line in lines:
        ms = [w for w in line["words"] if w.get("m")]
        line["s"] = min((w["s"] for w in ms), default=None)
        line["e"] = max((w["e"] for w in ms), default=None)
        ok += len(ms)
        total += len(line["words"])

    coverage = ok / total if total else 0.0
    return lines, coverage


def find_phrase(transcript, phrase_words, threshold=0.68, prefer_near=None):
    """Busca la frase (palabras de la letra) en el transcript con fuzzy match.

    Sirve como respaldo cuando una frase de la letra quedó SIN audio en su
    posición (el modelo tiny la transcribió mal AHÍ), pero la misma frase
    aparece bien transcrita en OTRA parte del audio (p. ej. el estribillo
    repetido que en una aparición se oye claro y en otra el coro lo tapa).
    Devuelve (k0, k1, score) de la mejor ventana del transcript, o None.

    phrase_words: lista de palabras de la letra (raw).
    prefer_near: tiempo (segundos) para desempatar entre apariciones iguales.
    """
    q = [norm(str(w.get("raw", ""))) for w in phrase_words]
    q = [x for x in q if x]
    if not q:
        return None
    L = len(q)
    T = len(transcript)
    if L == 0 or T < L:
        return None
    best = None
    for k in range(0, T - L + 1):
        win = [norm(str(w.get("word", ""))) for w in transcript[k:k + L]]
        if all(a == b for a, b in zip(q, win)):
            score = 1.0
        else:
            score = sum(fuzz.ratio(a, b) / 100.0 for a, b in zip(q, win)) / L
        if best is None or score > best[0] + 1e-9:
            best = (score, k, k + L - 1)
        elif prefer_near is not None and abs(score - best[0]) <= 1e-9:
            # desempatar: la aparición más cercana al tiempo preferido
            cur = abs(transcript[k]["start"] - prefer_near)
            be = abs(transcript[best[1]]["start"] - prefer_near)
            if cur < be:
                best = (score, k, k + L - 1)
    if best and best[0] >= threshold:
        return best
    return None


def find_phrase_repeated(transcript, phrase_words, threshold=0.68):
    """Busca la frase en el transcript; si no aparece COMPLETA, colapsa las
    repeticiones internas consecutivas (\"Ten piedad, Ten piedad, Ten piedad\"
    -> \"Ten piedad\") y busca la unidad, extendiendo luego a las repeticiones
    contiguas que existan en el audio. Devuelve (k0, k1, score) o None.

    Cubre el caso general de frases repetitivas que el modelo tiny transcribió
    mal en su posición (el coro), pero cuya unidad aparece bien en otra parte.
    """
    fb = find_phrase(transcript, phrase_words, threshold=threshold)
    if fb:
        return fb
    q = [norm(str(w.get("raw", ""))) for w in phrase_words]
    q = [x for x in q if x]
    # detectar la menor unidad que se repite ("ten piedad ten piedad ten
    # piedad" -> ["ten", "piedad"]); si no hay patrón, colapsar solo palabras
    # consecutivas idénticas
    n = len(q)
    collapsed = None
    for u in range(1, n // 2 + 1):
        unit0 = q[:u]
        if n % u == 0 and all(q[i:i + u] == unit0 for i in range(0, n, u)):
            collapsed = unit0
            break
    if collapsed is None:
        c = []
        for x in q:
            if not c or c[-1] != x:
                c.append(x)
        collapsed = c if len(c) < n else None
    if collapsed is None:
        return None
    unit = [{"raw": c} for c in collapsed]
    fb = find_phrase(transcript, unit, threshold=0.62)
    if not fb:
        return None
    sc, k0, k1 = fb
    # extender a repeticiones contiguas de la unidad dentro del transcript
    L = len(collapsed)
    T = len(transcript)
    qn = collapsed
    while k1 + L < T:
        win = [norm(str(w.get("word", ""))) for w in transcript[k1 + 1:k1 + 1 + L]]
        if all(a == b for a, b in zip(qn, win)):
            sc2 = 1.0
        else:
            sc2 = sum(fuzz.ratio(a, b) / 100.0 for a, b in zip(qn, win)) / L
        if sc2 >= 0.62:
            k1 += L
        else:
            break
    return (sc, k0, k1)


def find_all_phrase_occurrences(transcript, phrase_words, threshold=0.80):
    """Todas las apariciones de la frase (por TEXTO) en el transcript.

    Diseño (v25, tras el bug "naciones"->"tentación"):

    A) UNA palabra: match >= 0.85 o contención ("tempiedad" contiene
       "piedad"; "solos" contiene "solo"). El umbral 0.68 anterior hacía que
       "naciones" (0.71) matcheara "tentación" y la repetición sonara a otra
       frase; ahora un match suelto queda AFUERA.

    B) Frase de 2+ palabras: se elige la palabra CLAVE más distintiva (la
       menos frecuente en el transcript, con preferencia por la más larga) y
       se marcan sus hits fuertes (>= 0.80, incluye contención por fusión de
       palabras cantadas). Los hits consecutivos forman un pasaje (una
       aparición). Un pasaje se acepta si:
         - tiene 2+ hits consecutivos (la unidad cantada repetida: "tempiedad
           tempiedad"), o
         - algún hit matchea muy fuerte (>= 0.90, p. ej. contención de
           "tempiedad" con "piedad"), o
         - el contexto alrededor matchea al menos una de las OTRAS palabras
           de la frase (evita que la clave en un contexto distinto se cuele).
    Esto no depende de que whisper transcriba toda la frase igual: alcanza
    con que la palabra distintiva esté bien (o fusionada).

    Devuelve [(k0, k1, score)] no solapadas, ordenadas por tiempo.
    """
    q = [norm(str(w.get("raw", ""))) for w in phrase_words]
    q = [x for x in q if x]
    if not q:
        return []
    n = len(q)
    if n == 1:
        return _find_single_occurrences(transcript, q[0])
    unit = None
    for u in range(1, n // 2 + 1):
        if n % u == 0 and all(q[i:i + u] == q[:u] for i in range(0, n, u)):
            unit = q[:u]
            break
    if unit is not None and len(unit) < n:
        q = unit
    return _find_key_occurrences(transcript, q)


def _w_sim(a, b):
    """Similitud entre dos palabras normalizadas:
      - 1.0 si son iguales;
      - 0.92 si una contiene a la otra (whisper funde palabras cantadas:
        "tempiedad" contiene "piedad", "solos" contiene "solo"); se exige
        longitud >= 4 para no matchear "te"/"en" dentro de cualquier
        palabra ("ten" NO contiene-match "tentación");
      - si no, el ratio de edición.
    """
    if a == b:
        return 1.0
    if len(a) >= 4 and len(b) >= 4 and (a in b or b in a):
        return 0.92
    return fuzz.ratio(a, b) / 100.0


def _find_single_occurrences(transcript, q0, threshold=0.85):
    """Estrategia A: una sola palabra -> match fuerte o contención."""
    hits = []
    for i, w in enumerate(transcript):
        t = norm(str(w.get("word", "")))
        if t and _w_sim(q0, t) >= threshold:
            hits.append((i, i, 1.0))
    return hits


def _find_key_occurrences(transcript, q, thr=0.80):
    """Estrategia B: hits de la palabra clave distintiva + verificación de
    contexto (agrupa hits consecutivos en pasajes)."""
    if not q:
        return []
    # frecuencia de cada palabra en el transcript
    freq = {}
    tw_norm = []
    for w in transcript:
        t = norm(str(w.get("word", "")))
        if t:
            tw_norm.append(t)
            freq[t] = freq.get(t, 0) + 1
    # CLAVE: la palabra que matchea fuerte en MÁS lugares del transcript (las
    # repeticiones reales), porque whisper transcribe la voz cantada distinto
    # en cada repetición ("tengas" -> "hagas"/"tendrás"; elegir "tengas" por
    # ser rara perdía las otras repeticiones). Entre las que tienen >= 2 hits
    # se prefiere la MENOS frecuente (más distintiva) y la más larga. Si
    # ninguna tiene 2+ hits (frase que suena una sola vez), la menos
    # frecuente.
    def _hits(w):
        return sum(1 for t in tw_norm if _w_sim(w, t) >= 0.80)
    pool = [w for w in q if _hits(w) >= 2] or q
    clave = min(pool, key=lambda w: (freq.get(w, 10 ** 9), -len(w)))
    # OTRAS palabras DISTINTIVAS para verificar el contexto: se excluyen las
    # demasiado comunes ("por","los","que","la","de"...), porque un contexto
    # con solo palabras comunes matchea cualquier frase (p. ej. "sufren la
    # tentación" entraba como repetición de "...sufren la indiferencia...").
    _COMUNES = {"por", "los", "las", "que", "la", "el", "de", "del", "a",
                "en", "y", "o", "un", "una", "con", "se", "su", "sus", "no",
                "lo", "al", "te", "mi", "tu", "es", "mas", "ya", "me"}
    others = [a for a in q if a != clave
              and len(a) >= 4 and a not in _COMUNES]
    hits = []
    for i, w in enumerate(transcript):
        t = norm(str(w.get("word", "")))
        if t and _w_sim(clave, t) >= thr:
            hits.append(i)
    # agrupar hits consecutivos -> pasajes
    groups = []
    run = []
    for i in hits:
        if run and i != run[-1] + 1:
            groups.append(run)
            run = []
        run.append(i)
    if run:
        groups.append(run)
    T = len(transcript)
    offset = q.index(clave)          # posición de la clave dentro de la frase
    L = len(q)
    occs = []
    for g in groups:
        lo = max(0, g[0] - 2)
        hi = min(T, g[-1] + 3)
        ctx = [norm(str(w.get("word", ""))) for w in transcript[lo:hi]]
        # contexto: debe matchear al menos una palabra DISTINTIVA de la
        # frase (no las comunes); si la frase no tiene otras distintivas
        # (p. ej. "ten piedad" -> solo "piedad"), el hit fuerte de la clave
        # alcanza.
        ctx_ok = (len(others) == 0
                  or any(_w_sim(a, t) >= 0.75
                         for a in others for t in ctx))
        if len(g) >= 2 or ctx_ok:
            # expandir el pasaje a la frase COMPLETA, verificando palabra a
            # palabra (NO avanzar/retroceder sobre transcript que no matchea
            # la frase): evita que una palabra vecina ("saberlo") se cuele en
            # "ten piedad" por el offset de la clave.
            k0 = g[0]
            pos = offset - 1
            while pos >= 0 and k0 > 0:
                cand = norm(str(transcript[k0 - 1].get("word", "")))
                if cand and _w_sim(q[pos], cand) >= 0.72:
                    k0 -= 1
                    pos -= 1
                else:
                    break
            k1 = g[-1]
            pos = offset + (g[-1] - g[0]) + 1
            while pos < L and k1 < T - 1:
                cand = norm(str(transcript[k1 + 1].get("word", "")))
                if cand and _w_sim(q[pos], cand) >= 0.72:
                    k1 += 1
                    pos += 1
                else:
                    break
            if k1 >= k0:
                occs.append((k0, k1, 1.0))
    return occs


def find_occurrences(transcript, j0, j1, threshold=0.72):
    """Busca TODAS las apariciones de la ventana [j0..j1] en el transcript.

    Sirve para "todas las apariciones de la frase" (ej. el coro que se repite).
    Devuelve lista de (k0, k1, score) con ventanas no solapadas, ordenadas.
    """
    q = [norm(w["word"]) for w in transcript[j0:j1 + 1]]
    L = len(q)
    if L == 0:
        return []
    occ = []
    T = len(transcript)
    for k in range(0, T - L + 1):
        win = [norm(w["word"]) for w in transcript[k:k + L]]
        if all(a == b for a, b in zip(q, win)):
            score = 1.0
        else:
            score = sum(fuzz.ratio(a, b) / 100.0 for a, b in zip(q, win)) / L
        if score >= threshold:
            occ.append((k, k + L - 1, score))
    merged = []
    for k0, k1, sc in occ:
        if merged and k0 <= merged[-1][1] + 1:
            prev = merged[-1]
            if sc > prev[2]:
                merged[-1] = (prev[0], k1, sc)
            continue
        merged.append((k0, k1, sc))
    return merged
