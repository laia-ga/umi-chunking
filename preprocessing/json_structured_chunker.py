"""
json_structured_chunker.py
===========================
Los chunkers de ChunkBench (recursivo, semántico, por párrafos, etc.) 
solo saben trocear un string largo. Pero nuestros documentos son JSON, 
y cada campo del JSON tiene una forma distinta:

  - Un string largo de prosa -> SÍ tiene sentido pasarlo por un chunker de 
    texto de ChunkBench
  - Una lista de strings cortos -> hay que mantenerla entera, porque la 
    lista completa es la unidad de significado
  - Una lista de diccionarios/registros -> cada diccionario es una unidad 
    clínica completa (un hallazgo, una presentación...). 
    Un chunk = un elemento de la lista
  - Un diccionario anidado con sub-campos de texto -> cada sub-campo es
    su propio chunk, pero es necesario el camino de claves padre como 
    contexto
  - Una tabla explícita (dict con "columnas"/"filas" o "columns"/"rows")
    -> se reconstruye como una tabla Markdown completa en un solo chunk

En lugar de escribir una regla distinta para cada nombre de campo (que
habría que rehacer para cada tipo de documento nuevo que aparezca), este
script clasifica cada valor del JSON por su TIPO/FORMA.

Caso especial: registros con varios campos de texto largo
----------------------------------------------------------
Si un registro (un elemento de una lista de dicts) tiene más de un campo
de texto largo (ej. "descripcion" + "contexto" en un hallazgo, o "quote" +
un texto adicional), esos campos NO se separan en chunks distintos. 
Se detectó probando con documentos que, si se separan, cada trozo pierde 
sentido por sí solo (una "descripcion" sin su "contexto", o una cita sin
su fuente). En ese caso, el registro entero se convierte en UN chunk de tipo
"record", con todos sus campos (cortos y largos) juntos.

SALIDA
------
La función principal `chunk_document(doc, doc_id, ...)` devuelve una
lista de diccionarios con:
  - doc_id: identificador del documento
  - chunk_id: identificador único del chunk dentro del documento
  - path: "camino" de claves que llevan hasta este valor
  - chunk_type: "text" | "atomic_list" | "record" | "keyvalue_block" | "table"
  - text: el texto final que se indexaría (en lenguaje natural, listo
          para generar su embedding)

Cada chunk lleva su "path" para que, más adelante, pueda usarse como metadato
adicional en el payload (además de enfermedad, tipo de documento, etc.), 
y así saber de qué parte del documento viene cada resultado recuperado.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Callable

# Función de JSON_chunker.py
from chunkers.json.JSON_chunker import (
    approximate_token_counter,
    make_huggingface_token_counter,
)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

# Umbral (en caracteres) para decidir si un string es "texto largo de prosa"
# (va al chunker de texto) o "string corto" (se trata como un valor atómico,
# ej. un nombre de fármaco, una fecha, un código ATC).
LONG_STRING_CHAR_THRESHOLD = 200

# Nombres de clave que, si aparecen en un diccionario, hacen sospechar que
# ese diccionario representa una tabla explícita (columnas + filas)
TABLE_COLUMN_KEYS = {"columnas", "columns"}
TABLE_ROW_KEYS = {"filas", "rows"}

# ---------------------------------------------------------------------------
# Estructura de un chunk de salida
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    path: str
    chunk_type: str  # "text" | "atomic_list" | "record" | "keyvalue_block" | "table"
    text: str
    # Metadatos extra que puedan interesar para el payload de la vector store
    # (se puede ir ampliando: enfermedad, idioma, versión del documento...).
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "path": self.path,
            "chunk_type": self.chunk_type,
            "text": self.text,
            "metadata": self.metadata,
        }

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _humanize_key(key: str) -> str:
    """
    Convierte un nombre de clave tipo snake_case en una etiqueta legible,
    para que el chunk final se lea como lenguaje natural en vez de como
    código. Ej: "insuficiencia_renal" -> "insuficiencia renal". De esta
    forma, el modelo de embeddings lo entiende mejor
    """
    return key.replace("_", " ").strip()


def _path_join(path: str, key: str) -> str:
    """Añade un nuevo segmento al camino de claves (para trazabilidad)"""
    return f"{path}.{key}" if path else key


def _path_join_index(path: str, index: int) -> str:
    return f"{path}[{index}]"


def _is_table_dict(value: Dict[str, Any]) -> bool:
    """
    Detecta si un diccionario representa una tabla explícita: normalmente
    tiene una sub-clave con las columnas y otra con las filas (a veces
    anidadas bajo "datos". 
    Ej.: {"titulo": "...", "datos": {"columnas": [...], "filas": [[...]]}}
    """
    candidates = [value] + [v for v in value.values() if isinstance(v, dict)]
    for cand in candidates:
        keys = set(cand.keys())
        if keys & TABLE_COLUMN_KEYS and keys & TABLE_ROW_KEYS:
            return True
    return False


def _table_to_markdown(table_value: Dict[str, Any]) -> str:
    """
    Reconstruye un dict de tabla (título + columnas + filas + resumen) como
    una tabla en formato Markdown, en un único bloque de texto. Esto va a
    garantizar que la tabla viaje SIEMPRE entera en un solo chunk: nunca se
    corta una fila o columna en un chunk aparte.
    """
    titulo = table_value.get("titulo") or table_value.get("title") or ""
    resumen = table_value.get("resumen") or table_value.get("summary") or ""

    # Las columnas/filas pueden venir directamente en el dict, o anidadas
    # bajo una sub-clave tipo "datos"
    datos = table_value
    for v in table_value.values():
        if isinstance(v, dict):
            keys = set(v.keys())
            if keys & TABLE_COLUMN_KEYS and keys & TABLE_ROW_KEYS:
                datos = v
                break

    columnas = datos.get("columnas") or datos.get("columns") or []
    filas = datos.get("filas") or datos.get("rows") or []

    lines: List[str] = []
    if titulo:
        lines.append(f"Tabla: {titulo}")
    if columnas:
        lines.append("| " + " | ".join(str(c) for c in columnas) + " |")
        lines.append("|" + "|".join(["---"] * len(columnas)) + "|")
    for fila in filas:
        # Las filas son listas de celdas; nos aseguramos de que todas las
        # celdas sean texto (algunas pueden venir vacías: "").
        lines.append("| " + " | ".join(str(c) if c else "" for c in fila) + " |")
    if resumen:
        lines.append("")
        lines.append(f"Resumen de la tabla: {resumen}")

    return "\n".join(lines)


def _record_to_text(record: Dict[str, Any], path: str) -> str:
    """
    Convierte un diccionario "hoja" (un registro: un hallazgo, una
    presentación de un fármaco, un resultado clínico...) en un texto en
    lenguaje natural, tipo "Campo: valor. Campo: valor.", en vez de dejarlo
    como JSON crudo. Esto reduce el ruido sintáctico que podría puede meter
    el JSON en el embedding (llaves, comillas...).

    Solo aplana un nivel: si algún valor es a su vez una lista o un dict
    complejo, lo serializa de forma simple en lugar de recursar
    indefinidamente.
    """
    parts = []
    for key, value in record.items():
        if value in (None, "", [], {}):
            continue  # Omitimos campos vacíos
        label = _humanize_key(key)
        if isinstance(value, list):
            value_str = ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            value_str = "; ".join(f"{_humanize_key(k)}: {v}" for k, v in value.items() if v)
        else:
            value_str = str(value)
        parts.append(f"{label}: {value_str}")
    return ". ".join(parts)

def _has_nested_structure(record: Dict[str, Any]) -> bool:
    """
    True si el registro tiene al menos un campo que es a su vez un dict
    o una lista (ej. "posologia" dentro de una forma farmacéutica).
    Se usa para decidir si un registro (un elemento de una lista de
    dicts, ej. un hallazgo, una cita, una presentación) es "plano": si
    NINGÚN campo es dict/lista, todos sus campos (cortos o largos) se
    mantienen juntos en un único chunk, en vez de separar los campos de
    texto largo del resto.
 
    Registros como "quote" + "source_location", o "descripcion" + "contexto",
    son planos (ningún campo es dict/lista), y si se separaba el campo largo
    del corto, cada trozo pierde sentido por sí solo (una cita sin su
    fuente, una descripción sin su contexto numérico). Un registro con
    sub-estructura real (un dict con más campos dentro) es un caso distinto: 
    ahí sí conviene seguir separando, porque cada sub-campo es una unidad
    de información independiente.
    """
    return any(isinstance(v, (dict, list)) for v in record.values())

# ---------------------------------------------------------------------------
# Núcleo: traversal recursivo que clasifica por TIPO de valor
# ---------------------------------------------------------------------------
def _traverse(
    value: Any,
    path: str,
    doc_id: str,
    chunks: List[Chunk],
    chunk_counter: List[int],
) -> None:
    """
    Recorre recursivamente el JSON. En cada paso mira qué TIPO de valor
    tiene (no qué nombre de clave tiene) y decide qué hacer:

      dict "tabla"              -> un chunk tipo "table"
      dict normal               -> se sub-clasifica: si todos sus valores son
                                   "simples" (strings cortos/números), se
                                   trata como un bloque clave-valor (un solo
                                   chunk); si tiene sub-estructuras, se sigue
                                   recursando campo a campo
      lista de dicts           -> si un elemento tiene más de un campo de
                                  texto largo, se convierte entero en un
                                  único chunk tipo "record"; si no, se
                                  recursa sobre él como un dict normal
      lista de strings/números -> un chunk tipo "atomic_list" con la lista
                                   entera
      string largo             -> se guarda como un chunk de tipo "text" 
                                  (para usar ChunkBench posteriormente)
      string corto / número    -> se ignora aquí (se recoge como parte del
                                   bloque clave-valor del dict que lo contiene)
    """

    def new_chunk_id() -> str:
        chunk_counter[0] += 1
        return f"{doc_id}_chunk_{chunk_counter[0]:04d}"

    # --- Caso 1: diccionario ---
    if isinstance(value, dict):
        if _is_table_dict(value):
            table_text = _table_to_markdown(value)
            if table_text.strip():
                chunks.append(Chunk(
                    doc_id=doc_id,
                    chunk_id=new_chunk_id(),
                    path=path,
                    chunk_type="table",
                    text=table_text,
                ))
            return  # Una tabla se trata entera; no recursamos dentro de ella.

        # Separamos los valores "simples" (candidatos a ir juntos en un
        # bloque clave-valor) de los "complejos" (que necesitan seguir
        # recursando: sub-dicts, listas de dicts, strings largos...).
        simple_items: Dict[str, Any] = {}
        complex_items: Dict[str, Any] = {}
        for key, sub_value in value.items():
            if sub_value in (None, "", [], {}):
                continue  # Campo vacío, no aporta nada
            if isinstance(sub_value, str) and len(sub_value) >= LONG_STRING_CHAR_THRESHOLD:
                complex_items[key] = sub_value
            elif isinstance(sub_value, (dict, list)):
                complex_items[key] = sub_value
            else:
                simple_items[key] = sub_value

        # Si hay campos "simples" en este nivel, los agrupamos en UN chunk
        # de tipo "keyvalue_block": esto evita generar decenas de
        # chunks minúsculos (uno por cada campo suelto), que no aportan
        # contexto suficiente para la recuperación
        if simple_items:
            block_text = _record_to_text(simple_items, path)
            if block_text.strip():
                chunks.append(Chunk(
                    doc_id=doc_id,
                    chunk_id=new_chunk_id(),
                    path=path,
                    chunk_type="keyvalue_block",
                    text=block_text,
                ))

        # Los campos complejos se recorren uno a uno, arrastrando el
        # "camino" (path) para que, cuando lleguemos a una hoja, sepamos
        # de qué sección venía (ej. "posologia.insuficiencia_renal")
        for key, sub_value in complex_items.items():
            _traverse(
                sub_value,
                _path_join(path, key),
                doc_id,
                chunks,
                chunk_counter,
            )
        return

    # --- Caso 2: lista ---
    if isinstance(value, list):
        if not value:
            return

        if all(isinstance(item, dict) for item in value):
            # Lista de registros (ej. "hallazgos_principales",
            # "formas_farmaceuticas", "key_data_quotes"): cada elemento es
            # una unidad clínica completa por sí sola
            for i, item in enumerate(value):
                item_path = _path_join_index(path, i)
                
                # Si el registro es "plano" (ningún campo es dict/lista),
                # lo convertimos entero en UN chunk, sinimportar que algún 
                # campo sea texto largo. Esto evita separar un texto largo 
                # de sus campos cortos hermanos (ej. "quote" de su 
                # "source_location", o "descripcion" de su "contexto"
                if not _is_table_dict(item) and not _has_nested_structure(item):
                    record_text = _record_to_text(item, item_path)
                    if record_text.strip():
                        chunks.append(Chunk(
                            doc_id=doc_id,
                            chunk_id=new_chunk_id(),
                            path=item_path,
                            chunk_type="record",
                            text=record_text,
                        ))
                    continue
 
                # Caso normal: el registro tiene sub-estructura real (un
                # dict o una lista anidados, ej. "posologia"). Ahí sí
                # conviene seguir separando, porque cada sub-campo es una
                # unidad de información independiente. Recursamos sobre
                # él como un dict normal.
                _traverse(item, item_path, doc_id, chunks, chunk_counter)
            return

        if all(isinstance(item, str) for item in value):
            # Caso especial: una lista con un único string largo (ej.
            # "indicaciones": ["Kivexa está indicado en..."]) no es una
            # lista de valores atómicos, es prosa disfrazada de lista de un
            # elemento. La tratamos como el string suelto que realmente es,
            # para que pase por el chunker de texto en vez de quedarse
            # entera en un "atomic_list"
            if len(value) == 1 and len(value[0]) >= LONG_STRING_CHAR_THRESHOLD:
                _traverse(value[0], path, doc_id, chunks, chunk_counter)
                return

        # Lista de valores simples (strings/números): ej. "excipientes",
        # "palabras_clave", "frecuentes" (dentro de reacciones_adversas).
        # Se mantiene entera en un solo chunk, con la etiqueta del campo
        # (el último segmento del path) pegada, para no perder el
        # significado de a qué se refiere la lista
        label = _humanize_key(path.split(".")[-1].split("[")[0]) if path else "lista"
        list_text = f"{label}: " + ", ".join(str(v) for v in value)
        chunks.append(Chunk(
            doc_id=doc_id,
            chunk_id=new_chunk_id(),
            path=path,
            chunk_type="atomic_list",
            text=list_text,
        ))
        return

    # --- Caso 3: string largo de prosa ---
    if isinstance(value, str) and len(value) >= LONG_STRING_CHAR_THRESHOLD:
        label = _humanize_key(path.split(".")[-1]) if path else ""
        # Por ahora lo dejamos como un único chunk (sin trocear el texto
        # todavía, luego usaremos ChunkBench). Anteponemos la etiqueta del
        # campo para no perder el contexto de qué sección es
        final_text = f"{label}: {value}" if label else value
        chunks.append(Chunk(
            doc_id=doc_id,
            chunk_id=new_chunk_id(),
            path=path,
            chunk_type="text",
            text=final_text,
        ))
        return

    # --- Caso 4: string corto / número / bool suelto en la raíz ---
    # Esto solo pasaría si el documento raíz no fuera un dict (caso raro);
    # lo dejamos como chunk mínimo para no perder el dato.
    if value not in (None, "", [], {}):
        chunks.append(Chunk(
            doc_id=doc_id,
            chunk_id=new_chunk_id(),
            path=path,
            chunk_type="keyvalue_block",
            text=str(value),
        ))


def _parent_key(path: str) -> str:
    """
    Quita el índice final de un path, si lo tiene, para saber a qué
    clave pertenece un chunk. Ej: "hallazgos_principales[3]" ->
    "hallazgos_principales". Dos chunks con el mismo parent_key vienen
    de la misma lista/campo del JSON, aunque sean elementos distintos.
    """
    if path.endswith("]") and "[" in path:
        return path[: path.rindex("[")]
    return path


def merge_short_chunks(
        chunks: List[Chunk], 
        min_chunk_tokens: int,
        token_counter: Callable[[str], int] = 
        approximate_token_counter,
        ) -> List[Chunk]:
    """
    Fusiona chunks cortos que compartan clave y tipo, para evitar generar
    muchos chunks demasiado cortos.
    
    Agrupa por (chunk_type, parent_key) en toda la lista, no solo chunks
    que estén uno justo detrás del otro.
 
    Solo fusiona dos chunks si:
      - Tienen el mismo chunk_type (nunca se mezcla un "text" con un
        "atomic_list", por ejemplo)
      - Vienen de la misma "clave" (mismo parent_key: mismo campo/lista
        del JSON, solo cambia el índice del elemento)
      - El chunk que se está construyendo (el "buffer") todavía no ha
        alcanzado min_chunk_tokens tokens
 
    Las tablas ("table") nunca se fusionan, aunque coincidan en tipo y
    clave: cada tabla ya es una unidad completa por sí sola.
    
    Los chunks del mismo grupo salen ahora juntos en la lista final,
    en lugar de mantener su posición original intercalada con otros 
    chunks. El orden entre grupos distintos no cambia. 
 
    min_chunk_tokens es el umbral por debajo del cual un chunk se
    considera "demasiado corto". Su valor ideal depende del modelo de
    embeddings que se acabe usando, así que se deja como parámetro en
    vez de fijarlo aquí: por defecto (0) no fusiona nada.
    """
    if min_chunk_tokens <= 0 or not chunks:
        return chunks
    
    # Agrupamos por (chunk_type, parent_key), conservando el orden en
    # que aparece cada grupo por primera vez
    groups: Dict[tuple,  List[Chunk]] = {}
    group_order: List[tuple] = []
    for c in chunks:
        key = (c.chunk_type, _parent_key(c.path))
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append(c)
    
    result: List[Chunk] = []
    for key in group_order:
        chunk_type, parent_key = key
        group_chunks = groups[key]
        
        if chunk_type == "table" or len(group_chunks) == 1:
            # Nada que fusionar: una tabla, o un grupo de un solo chunk
            result.extend(group_chunks)
            continue
        
        buffer: Chunk | None = None
        for current in group_chunks:
            if buffer is None:
                buffer = current
                continue
            if token_counter(buffer.text) < min_chunk_tokens:
                # Fusionamos el chunk actual dentro del buffer: el texto se
                # une con una línea en blanco de separación, y el path pasa
                # a ser el parent_key (ya no representa un único elemento,
                # sino varios fusionados).
                buffer = Chunk(
                    doc_id=buffer.doc_id,
                    chunk_id=buffer.chunk_id,
                    path=_parent_key(buffer.path),
                    chunk_type=buffer.chunk_type,
                    text=buffer.text + "\n\n" + current.text,
                    metadata=buffer.metadata,
                    )
            else:
                result.append(buffer)
                buffer = current
        if buffer is not None:
            result.append(buffer)
 
    return result


def chunk_document(
        doc: Dict[str, Any], 
        doc_id: str,
        min_chunk_tokens: int = 0,
        token_counter: Callable[[str], int] = 
        approximate_token_counter,
        ) -> List[Chunk]:
    """
    Punto de entrada principal. Recibe un documento JSON ya cargado (un
    dict de Python) y devuelve la lista de chunks resultante.
    
    Los chunks de tipo "text" (texto largo de prosa) se dejan aquí sin
    trocear más; la idea es pasarlos después por los chunkers de
    ChunkBench
    
    min_chunk_tokens: si se pasa un valor mayor que 0, los chunks
    de la misma clave y tipo que midan menos de ese umbral se fusionan
    entre sí (ver merge_short_chunks). Por defecto es 0, es decir, no 
    fusiona nada.
    """
    chunks: List[Chunk] = []
    chunk_counter = [0]
    _traverse(
        doc,
        path="",
        doc_id=doc_id,
        chunks=chunks,
        chunk_counter=chunk_counter,
    )
    return merge_short_chunks(
        chunks, 
        min_chunk_tokens, 
        token_counter=token_counter,
        )


# ---------------------------------------------------------------------------
# Demo / autoprueba con los ejemplos reales que hemos visto
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print(
            "Uso: python json_structured_chunker.py <ruta_al_json> [doc_id]\n"
            "Ejemplo: python json_structured_chunker.py ``ABACAVIR + LAMIVUDINA.json`` abacavir"
        )
        sys.exit(1)

    ruta = sys.argv[1]
    doc_id = sys.argv[2] if len(sys.argv) > 2 else "doc_demo"

    with open(ruta, "r", encoding="utf-8") as f:
        documento = json.load(f)

    resultado = chunk_document(documento, doc_id)

    print(f"Total de chunks generados: {len(resultado)}\n")
    for c in resultado:
        print(f"[{c.chunk_type}] {c.path}")
        preview = c.text if len(c.text) <= 200 else c.text[:200] + "..."
        print(f"    {preview}\n")
