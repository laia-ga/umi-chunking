"""
HierarchicalJSONChunker: Chunking jerárquico para documentos JSON clínicos.

Este chunker:
1. Recorre el documento JSON siguiendo un document plan específico
   para cada tipología documental (paper, guideline, ficha técnica).
2. Respeta la estructura jerárquica del documento:
   - niveles,
   - grupos,
   - rutas JSON,
   - listas indivisibles,
   - bloques semánticos.
3. Evita cortes peligrosos en tablas, listas clínicas y secciones críticas.
4. Genera metadatos completos para trazabilidad:
   - nivel,
   - grupo,
   - rutas JSON,
   - límites de tokens,
   - flags de exceso de tamaño.
5. Produce chunks seguros y adecuados para RAG médico.

Este archivo define la lógica del chunking JSON. Los document plans
definen la estructura que se debe seguir para cada tipo documental.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
import re

# 1. ESTRUCTURAS DE DATOS
@dataclass
class Chunk:
    # Representa los chunks generados por el algoritmo, con información
    # sobre su procedencia dentro del JSON y tamaño
    text: str # texto del chunk
    block: str # nombre de la clave raíz del documento
    group_name: str # nombre del nodo en SplitNode
    level: int # profundidad del SplitNode
    json_paths: list[str] # ruta de los elementos representados
    token_count: int
    exceeds_target_limit: bool = False # supera el tamaño recomendado para el chunking
    exceeds_max_limit: bool = False # supera el límite del modelo de embeddings
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class SplitNode:
    # Representa los nodos del árbol DOCUMENT_PLAN. Un nodo puede contener
    # fields (rutas JSON) o children (subdivisiones semánticas)
    name: str
    fields: list[str] = field(default_factory=list)
    children: list["SplitNode"] = field(default_factory=list)

@dataclass(frozen=True)
class ResolvedField:
    # Representa los campos de DOCUMENT_PLAN que ya han sido localizados
    # dentro del JSON
    name: str
    path: tuple[str | int, ...]
    value: Any

# 2. FUNCIONES AUXILIARES DE TEXTO
def is_present(value: Any) -> bool:
    # Comprueba recursivamente si un valor JSON contiene información real.

    # Se consideran vacíos:
    # - None;
    # - cadenas vacías o con espacios;
    # - listas cuyos elementos estén todos vacíos;
    # - diccionarios cuyos valores estén todos vacíos.

    # Los valores False y 0 se consideran información válida.
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(
            is_present(item)
            for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(
            is_present(item)
            for item in value
        )
    # Números, booleanos y otros valores escalares
    # se consideran información presente.
    return True

def humanize(name: str) -> str:
    # Convierte las clave-valor del JSON en etiquetas legibles
    return name.replace("_", " ").strip().capitalize()


def scalar_to_text(value: Any) -> str:
    # Convierte los valores y booleanos en texto
    if isinstance(value, bool):
        return "Sí" if value else "No"
    return str(value)

# Nombres de claves que indican que un diccionario
# probablemente representa una tabla explícita.
TABLE_COLUMN_KEYS = {"columnas", "columns",}

TABLE_ROW_KEYS = {"filas", "rows",}

def find_matching_key(
    value: dict[str, Any],
    candidates: set[str],
) -> str | None:
    # Busca dentro de un diccionario una clave coincidente,
    # ignorando mayúsculas, minúsculas y espacios exteriores.
    for key in value:
        if (
            isinstance(key, str)
            and key.strip().casefold() in candidates
        ):
            return key
    return None


def detect_explicit_table(
    value: Any,
) -> tuple[str, str] | None:
    # Comprueba si un valor parece una tabla explícita.

    # Para considerarlo tabla debe:
    # - ser un diccionario;
    # - contener una clave de columnas;
    # - contener una clave de filas;
    # - almacenar columnas y filas como listas.

    # Devuelve las claves reales encontradas.
    if not isinstance(value, dict):
        return None
    column_key = find_matching_key(
        value,
        TABLE_COLUMN_KEYS,
    )
    row_key = find_matching_key(
        value,
        TABLE_ROW_KEYS,
    )
    if column_key is None or row_key is None:
        return None
    if not isinstance(value[column_key], list):
        return None
    if not isinstance(value[row_key], list):
        return None
    return column_key, row_key

def _append_table_cell(
    lines: list[str],
    column_name: str,
    cell: Any,
    indent: int,
) -> None:
    # Añade una celda de tabla manteniendo explícita
    # la relación entre el nombre de columna y su valor.
    prefix = " " * indent
    label = humanize(column_name)
    # Una cadena con saltos de línea se interpreta
    # como una lista de elementos dentro de la celda.
    if isinstance(cell, str):
        cell_lines = [
            line.strip(" -\t")
            for line in cell.splitlines()
            if line.strip(" -\t")
        ]
        if not cell_lines:
            return
        # Una única línea se representa directamente.
        if len(cell_lines) == 1:
            lines.append(
                f"{prefix}{label}: "
                f"{cell_lines[0]}"
            )
        # Varias líneas se representan como una lista.
        else:
            lines.append(
                f"{prefix}{label}:"
            )
            lines.extend(
                f"{prefix}  - {line}"
                for line in cell_lines
            )
        return

    # Las listas y diccionarios internos se conservan
    # con su estructura.
    if isinstance(cell, (dict, list)):
        nested = value_to_lines(
            cell,
            indent + 2,
        )
        if nested:
            lines.append(
                f"{prefix}{label}:"
            )
            lines.extend(nested)
        return
    # Números, booleanos y otros escalares.
    lines.append(
        f"{prefix}{label}: "
        f"{scalar_to_text(cell)}"
    )

def table_to_lines(
    table: dict[str, Any],
    indent: int = 0,
) -> list[str]:
    # Convierte una tabla en texto explícito por filas.

    # En cada fila se repiten los nombres de las columnas
    # para mantener la relación columna-valor.
    detected_keys = detect_explicit_table(table)
    if detected_keys is None:
        return []
    column_key, row_key = detected_keys
    columns = table[column_key]
    rows = table[row_key]
    prefix = " " * indent
    lines: list[str] = []
    # Conserva otros campos que pueda contener la tabla,
    # como title, source o description.
    for key, item in table.items():
        if key in {column_key, row_key}:
            continue
        if not is_present(item):
            continue
        label = humanize(str(key))
        if isinstance(item, (dict, list)):
            nested = value_to_lines(
                item,
                indent + 2,
            )
            if nested:
                lines.append(
                    f"{prefix}{label}:"
                )
                lines.extend(nested)
        else:
            lines.append(
                f"{prefix}{label}: "
                f"{scalar_to_text(item)}"
            )
    # Procesa las filas de la tabla.
    for row_index, row in enumerate(
        rows,
        start=1,
    ):
        if not is_present(row):
            continue
        lines.append(
            f"{prefix}Row {row_index}:"
        )
        # --------------------------------------------
        # Caso 1: la fila es una lista
        # --------------------------------------------
        if isinstance(row, list):
            for column_index, cell in enumerate(row):
                if not is_present(cell):
                    continue
                # Utiliza el nombre de columna correspondiente.
                if column_index < len(columns):
                    column_name = str(
                        columns[column_index]
                    )
                else:
                    column_name = (
                        f"Extra column "
                        f"{column_index + 1}"
                    )
                _append_table_cell(
                    lines=lines,
                    column_name=column_name,
                    cell=cell,
                    indent=indent + 2,
                )

        # --------------------------------------------
        # Caso 2: la fila es un diccionario
        # --------------------------------------------
        elif isinstance(row, dict):
            used_keys: set[Any] = set()
            # Primero conserva el orden indicado
            # en la lista de columnas.
            for column in columns:
                if column not in row:
                    continue
                cell = row[column]
                if not is_present(cell):
                    continue
                used_keys.add(column)
                _append_table_cell(
                    lines=lines,
                    column_name=str(column),
                    cell=cell,
                    indent=indent + 2,
                )
            # Conserva también los campos adicionales
            # que no estén declarados en columns.
            for key, cell in row.items():
                if key in used_keys:
                    continue
                if not is_present(cell):
                    continue
                _append_table_cell(
                    lines=lines,
                    column_name=str(key),
                    cell=cell,
                    indent=indent + 2,
                )

        # --------------------------------------------
        # Caso 3: la fila es un valor escalar
        # --------------------------------------------
        else:
            column_name = (
                str(columns[0])
                if columns
                else "Value"
            )
            _append_table_cell(
                lines=lines,
                column_name=column_name,
                cell=row,
                indent=indent + 2,
            )
    return lines

def value_to_lines(value: Any, indent: int = 0) -> list[str]:
    # Convierte un valor JSON en líneas de texto según el tipo de elemento.
    # Los diccionarios se serializan completos, y si tienen dentro otros elementos
    # se incluyen con sangría.
    # Los elementos de una lista se convierten en guiones.
    # Los textos, escalares y booleanos se convierten en una línea.
    prefix = " " * indent

    if not is_present(value):
        return []
    if isinstance(value, dict):
        # Antes de tratarlo como un diccionario genérico,
        # comprueba si representa una tabla explícita.
        if detect_explicit_table(value) is not None:
            return table_to_lines(
                table=value,
                indent=indent,
            )
        # Si no es una tabla:
        lines: list[str] = []
        for key, item in value.items():
            if not is_present(item):
                continue
            label = humanize(key)
            if isinstance(item, (dict, list)):
                nested = value_to_lines(item, indent + 2)
                if nested:
                    lines.extend([f"{prefix}{label}:", *nested])
            else:
                lines.append(f"{prefix}{label}: {scalar_to_text(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            nested = value_to_lines(item, indent + 2)
            if nested:
                lines.append(f"{prefix}- {nested[0].lstrip()}")
                lines.extend(f"{prefix}  {line.lstrip()}" for line in nested[1:])
        return lines
    return [prefix + scalar_to_text(value)]


def render_fields(fields: list[ResolvedField]) -> str:
    # Convierte una lista de campos en el texto del chunk.
    # Las distintas parejas clave-valor se sepran por una línea en blanco
    blocks: list[str] = []
    for field in fields:
        label = humanize(field.name)
        if isinstance(field.value, (dict, list)):
            lines = value_to_lines(field.value, 2)
            if lines:
                blocks.append("\n".join([f"{label}:", *lines]))
        else:
            blocks.append(f"{label}: {scalar_to_text(field.value)}")
    return "\n\n".join(blocks)

# 3. RESOLUCIÓN DE CAMPOS SEGÚN EL PLAN

def collect_fields(node: SplitNode) -> list[str]:
    # Recoge todos los campos declarados en un nodo y en sus descencientes.
    # Agrupa todo el contenido representado en la misma rama de DOCUMENT_PLAN
    fields = list(node.fields)
    for child in node.children:
        fields.extend(collect_fields(child))
    return list(dict.fromkeys(fields))

def resolve_fields(
    node: SplitNode,
    data: dict[str, Any],
) -> list[ResolvedField]:
    # Localiza en el JSON todos los campos pertenecientes a un nodo según su
    # ruta exacta relativa a la raíz. De esta forma, campos repetidos no se confunden
    # si aparecen en ramas distintas
    result: list[ResolvedField] = []
    for field_path in collect_fields(node):
        path = tuple(field_path.split("."))
        value: Any = data
        for key in path:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            if is_present(value):
                result.append(
                    ResolvedField(
                        name=path[-1],
                        path=path,
                        value=value,
                    )
                )
    return result

# 4. CONTADORES DE TOKENS

def approximate_token_counter(text: str) -> int:
    # Contador aproximado que considera cada palabra como un token
    return len(text.split())

def make_huggingface_token_counter(tokenizer: Any) -> Callable[[str], int]:
    # Construye una función que cuenta tokens utilizando un tokenizador de
    # Hugging Face. Se debe utilizar el tokenizador del modelo de embeddings
    # cuando se conozca
    def count_tokens(text: str) -> int:
        return len(tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"])
    return count_tokens

# 5. CHUNKER RECURSIVO

class HierarchicalJSONChunker:
    # Divide un documento JSON utilizando DOCUMENT_PLAN

    # 1. Determina la raíz del documento: utiliza root_field si se proporciona;
    #    si no, detecta automáticamente una única clave exterior

    # 2. Procesa obligatoriamente todos los hijos del primer nivel (nunca se genera
    #    el documento completo como un chunk)

    # 3. Localiza los campos mediante sus rutas JSON relativas a la raíz para 
    #    evitar confusión ante el mismo nombre en distintas ramas

    # 4. Ignora valores sin contenido real, como None, cadenas vacías, listas con 
    #    elementos vacíos o diccionarios con elementos vacíos. Los valores 0 y False
    #    se consideran información válida

    # 5. Se utiliza target_tokens como límite recomendado para la longitud del chunk.
    #    - Si una rama no supera target tokens, se conserva completa
    #    - Si lo supera y tiene hijos, se generan todos sus hijos directos y cada uno
    #      se vuelve a evaluar recursivamente
    #    - Solo se continúa descendiendo por las ramas que superen target_tokens

    # 6. Una pareja clave-valor escalar o cuyo valor sea un diccionario se considera
    #    indivisible aunque supere target_tokens por sí misma

    # 7. Una pareja clave-lista que supera target_tokens se divide en un chunk independiente
    #    para cada elemento de la lista, conservando el nombre de la clave original, y 
    #    su índice se añade a la ruta JSON en los metadatos. Si el elemento es un
    #    diccionario, permanece indivisible

    # 8. Las listas incluidas en indivisible_list_paths nunca se dividen por
    #    elementos, aunque superen target_tokens

    # 9. Las tablas explícitas se serializan por filas conservando nombres de
    #    columnas; si exceden max_tokens, el fallback divide por filas completas

    # 10. max_tokens es el límite absoluto del modelo de embeddings. El fallback
    #     divide los chunks que lo superan y no descarta contenido 

    # 11. En cada chunk se indica el número de tokens, si supera el tamaño recomendado
    #     y el límite del modelo, las rutas JSON y el grupo del árbol documental al que pertenece

    # 12. El contenido presente en el JSON pero no incluido en DOCUMENT_PLAN se conserva
    #     bajo el grupo "unmapped_content"

    # 13. El fallback prioriza filas de tablas y divide otros excesos de forma equilibrada.
    # 14. El contexto se conserva en texto y metadatos, respetando max_tokens.
    # 15. El conteo de tokens utiliza la función token_counter proporcionada
    def __init__(
        self,
        target_tokens: int,
        max_tokens: int,
        token_counter: Callable[[str], int],
        document_plan: SplitNode,
        root_field: str | None = None,
        indivisible_list_paths: set[str] | None = None,
        context_fields: list[str] | None = None,
    ) -> None:
        if target_tokens <= 0:
            raise ValueError(
                "target_tokens debe ser mayor que cero."
            )
        if max_tokens <= 0:
                    raise ValueError(
                        "max_tokens debe ser mayor que cero."
                    )
        if target_tokens > max_tokens:
                    raise ValueError(
                        "target_tokens no puede superar max_tokens."
                    )
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.token_counter = token_counter
        self.document_plan = document_plan
        self.root_field = root_field
        # Rutas de listas que nunca deben dividirse por elementos.
        # Se utiliza un conjunto para que la búsqueda sea eficiente.
        self.indivisible_list_paths = set(
            indivisible_list_paths or []
        )
        self.context_fields = list(dict.fromkeys(context_fields or []))

    def chunk_document(self, document: dict[str, Any]) -> list[Chunk]:
        """Chunking jerárquico con contexto en texto Y metadatos.

        El contexto se incorpora antes de validar max_tokens. Las tablas grandes
        se dividen por filas cuando es seguro; otros excesos usan fallback con
        solapamiento y tamaño equilibrado cuando es computacionalmente viable.
        """
        if not isinstance(document, dict):
            raise TypeError("El documento debe ser un diccionario.")
        if not document:
            return []

        block, data = self._get_root(document)
        context_fields = resolve_fields(
            SplitNode(name="document_context", fields=self.context_fields), data
        )
        context_text = render_fields(context_fields)
        context_paths = [
            self._format_path(block, f.path) for f in context_fields
        ]
        context_metadata = {
            ".".join(str(p) for p in f.path): f.value
            for f in context_fields
        }

        planned = resolve_fields(self.document_plan, data)
        chunks: list[Chunk] = []
        for child in self.document_plan.children:
            chunks.extend(self._split(block=block, data=data, node=child, level=1))

        # El contexto no genera chunks independientes de unmapped_content.
        covered_paths = [f.path for f in planned]
        covered_paths.extend(f.path for f in context_fields)
        uncovered = self._find_uncovered(data, covered_paths)
        if uncovered:
            chunks.extend(self._pack_fields(
                block=block,
                node=SplitNode(name="unmapped_content"),
                fields=uncovered,
                level=1,
            ))

        final_chunks: list[Chunk] = []
        for chunk in chunks:
            # Si algún campo del contexto aparece en el DOCUMENT_PLAN, quitar
            # su bloque ya renderizado antes de anteponer el contexto común.
            # Así todos los chunks tienen un prefijo uniforme sin duplicarlo.
            content_text = self._remove_context_blocks(chunk.text, chunk, context_fields)
            combined = self._join_context(context_text, content_text)
            chunk.metadata["context"] = dict(context_metadata)
            full_paths = list(dict.fromkeys([*context_paths, *chunk.json_paths]))

            if self.token_counter(combined) > self.max_tokens:
                final_chunks.extend(self._split_oversized_chunk(
                    original=chunk,
                    context_text=context_text,
                    content_text=content_text,
                    context_paths=full_paths,
                ))
            else:
                chunk.text = combined
                chunk.token_count = self.token_counter(combined)
                chunk.exceeds_target_limit = chunk.token_count > self.target_tokens
                chunk.exceeds_max_limit = False
                chunk.json_paths = full_paths
                chunk.metadata.update({
                    "exceeds_target_limit": chunk.exceeds_target_limit,
                    "exceeds_max_limit": False,
                })
                final_chunks.append(chunk)

        return final_chunks

    @staticmethod
    def _join_context(context_text: str, content_text: str) -> str:
        if not context_text:
            return content_text
        return f"{context_text}\n\n{content_text}" if content_text else context_text

    def _remove_context_blocks(
        self, content: str, chunk: Chunk, context_fields: list[ResolvedField]
    ) -> str:
        """Elimina campos contextuales duplicados que ya están en el chunk.

        Solo elimina un bloque exacto delimitado por las separaciones que
        genera render_fields; nunca borra coincidencias dentro de una frase.
        """
        paths = set(chunk.json_paths)
        for f in context_fields:
            if self._format_path(chunk.block, f.path) not in paths:
                continue
            rendered = render_fields([f])
            if content == rendered:
                content = ""
            elif content.startswith(rendered + "\n\n"):
                content = content[len(rendered) + 2:]
            elif content.endswith("\n\n" + rendered):
                content = content[:-(len(rendered) + 2)]
            elif "\n\n" + rendered + "\n\n" in content:
                content = content.replace(
                    "\n\n" + rendered + "\n\n", "\n\n", 1
                )
            # Si no es un bloque exacto, preservamos el texto intacto.
        return content

    FALLBACK_OVERLAP_RATIO = 0.1
    _ROW_MARKER = re.compile(r"^(\s*)Row (\d+):\s*$")

    def _fallback_chunk(
        self,
        original: Chunk,
        text: str,
        method: str,
        part: int,
        total: int,
        context_paths: list[str],
        original_tokens: int,
        overlap_units: int = 0,
        overlap_tokens: int = 0,
        row_range: list[int] | None = None,
    ) -> Chunk:
        n_tokens = self.token_counter(text)
        if n_tokens > self.max_tokens:
            raise AssertionError("Fallback generó un chunk superior a max_tokens")
        is_over_target = n_tokens > self.target_tokens
        metadata = dict(original.metadata)
        metadata.update({
            "fallback_applied": True,
            "fallback_method": method,
            "fallback_part": part,
            "fallback_total_parts": total,
            "fallback_original_tokens": original_tokens,
            "fallback_overlap_ratio": (
                self.FALLBACK_OVERLAP_RATIO if overlap_units else 0.0
            ),
            "fallback_overlap_units": overlap_units,
            "fallback_overlap_tokens": overlap_tokens,
            "exceeds_target_limit": is_over_target,
            "exceeds_max_limit": False,
        })
        if row_range is not None:
            metadata["table_row_range"] = row_range
        return Chunk(
            text=text,
            block=original.block,
            group_name=original.group_name,
            level=original.level,
            json_paths=list(dict.fromkeys(context_paths)),
            token_count=n_tokens,
            exceeds_target_limit=is_over_target,
            exceeds_max_limit=False,
            metadata=metadata,
        )

    def _split_oversized_chunk(
        self,
        original: Chunk,
        context_text: str = "",
        content_text: str | None = None,
        context_paths: list[str] | None = None,
    ) -> list[Chunk]:
        """Divide sin superar max_tokens y repite el contexto en cada parte.

        Prioriza tablas explícitas por filas completas. Si no es posible,
        conserva todo el texto mediante unidades de oración y overlap.
        """
        if content_text is None:
            content_text = original.text
        paths = list(context_paths or original.json_paths)
        margin = max(1, min(16, int(self.max_tokens * 0.01))) if self.max_tokens > 32 else 0
        usable = self.max_tokens - margin
        if self.token_counter(context_text) >= usable:
            raise ValueError("El contexto obligatorio no deja espacio para contenido.")
        original_tokens = self.token_counter(self._join_context(context_text, content_text))

        table_chunks = self._split_table_by_rows(
            original=original,
            content_text=content_text,
            context_text=context_text,
            context_paths=paths,
            usable=usable,
            original_tokens=original_tokens,
        )
        if table_chunks is not None:
            return table_chunks

        return self._split_text_balanced(
            original=original,
            content_text=content_text,
            context_text=context_text,
            context_paths=paths,
            usable=usable,
            original_tokens=original_tokens,
        )

    def _split_table_by_rows(
        self,
        original: Chunk,
        content_text: str,
        context_text: str,
        context_paths: list[str],
        usable: int,
        original_tokens: int,
    ) -> list[Chunk] | None:
        """Si el texto contiene una tabla simple, agrupa filas indivisibles.

        Repite cabecera y contexto en cada fragmento. Si una fila completa no
        cabe, vuelve al fallback general en lugar de perder información.
        """
        lines = content_text.splitlines(keepends=True)
        markers = [self._ROW_MARKER.match(s.rstrip("\r\n")) for s in lines]

        first = next(
            (i for i, m in enumerate(markers) if m is not None),
            None
        )

        if first is None:
            return None

        first_marker = markers[first]

        if first_marker is None:
            return None

        row_indent = len(first_marker.group(1))

        header = "".join(lines[:first])
        rows: list[tuple[int, str]] = []
        trailer: list[str] = []
        after_table = False
        for line, marker in zip(lines[first:], markers[first:]):
            if marker:
                if after_table or len(marker.group(1)) != row_indent:
                    return None  # Tabla múltiple o anidada: fallback general
                rows.append((int(marker.group(2)), line))
                continue
            if not after_table and (not line.strip() or len(line) - len(line.lstrip()) > row_indent):
                if not rows:
                    return None
                row_id, row_text = rows[-1]
                rows[-1] = (row_id, row_text + line)
            else:
                after_table = True
                trailer.append(line)
        if not rows:
            return None

        combine = lambda text: self._join_context(context_text, text)
        groups: list[list[tuple[int, str]]] = []
        current: list[tuple[int, str]] = []
        for row in rows:
            trial = header + "".join(v for _, v in [*current, row])
            if self.token_counter(combine(trial)) > usable:
                if not current:
                    return None  # Fila indivisible demasiado grande
                groups.append(current)
                current = [row]
                if self.token_counter(combine(header + row[1])) > usable:
                    return None
            else:
                current.append(row)
        if current:
            groups.append(current)
        if len(groups) < 2 and not trailer:
            return None  # No aporta ninguna división válida

        texts = [header + "".join(v for _, v in group) for group in groups]
        row_ranges: list[list[int] | None] = [
            [group[0][0], group[-1][0]] for group in groups
        ]
        trailer_text = "".join(trailer)
        if trailer_text:
            if self.token_counter(combine(texts[-1] + trailer_text)) <= usable:
                texts[-1] += trailer_text
            else:
                # Fragmentar solo el epílogo; las filas permanecen completas.
                tail = header + trailer_text
                tail_chunk = self._split_text_balanced(
                    original=original,
                    content_text=tail,
                    context_text=context_text,
                    context_paths=context_paths,
                    usable=usable,
                    original_tokens=original_tokens,
                    method="table_trailer_sentence_overlap",
                )
                texts.extend(c.text for c in tail_chunk)
                # Los fragmentos del epílogo YA incorporan contexto.
                row_ranges.extend([None] * len(tail_chunk))
                result = []
                for i, (body, row_range) in enumerate(zip(texts, row_ranges), start=1):
                    full_text = combine(body) if row_range is not None else body
                    result.append(self._fallback_chunk(
                        original, full_text,
                        "table_rows_with_header" if row_range is not None else "table_trailer_sentence_overlap",
                        i, len(texts), context_paths, original_tokens,
                        row_range=row_range,
                    ))
                return result

        return [
            self._fallback_chunk(
                original=original,
                text=combine(body),
                method="table_rows_with_header",
                part=i,
                total=len(texts),
                context_paths=context_paths,
                original_tokens=original_tokens,
                row_range=row_ranges[i - 1],
            )
            for i, body in enumerate(texts, start=1)
        ]

    def _split_text_balanced(
        self,
        original: Chunk,
        content_text: str,
        context_text: str,
        context_paths: list[str],
        usable: int,
        original_tokens: int,
        method: str | None = None,
    ) -> list[Chunk]:
        """Fallback equilibrado con preservación exacta del texto y overlap.

        Para documentos muy fragmentados limita el coste de la programación
        dinámica usando greedy: evita latencias cuadráticas innecesarias.
        """
        combine = lambda text: self._join_context(context_text, text)
        if not content_text:
            raise ValueError("No hay contenido que dividir.")
        unit_limit = max(1, usable - self.token_counter(context_text) - 4)
        units = self._fallback_units(content_text, unit_limit)

        # Reajustar unidades que no quepan con contexto (tokenización no aditiva).
        i = 0
        while i < len(units):
            if self.token_counter(combine(units[i])) <= usable:
                i += 1
                continue
            unit = units[i]
            if len(unit) <= 1:
                raise ValueError("El contexto y un carácter no caben en max_tokens")
            mid = len(unit) // 2
            units[i:i + 1] = [unit[:mid], unit[mid:]]
        assert "".join(units) == content_text

        def fits(i: int, j: int) -> bool:
            return self.token_counter(combine("".join(units[i:j]))) <= usable

        n = len(units)
        boundaries: list[tuple[int, int]] = []
        if n <= 120:
            # Mínimo número de partes y longitudes aproximadamente equilibradas.
            from functools import lru_cache
            cached_fits = lru_cache(maxsize=None)(fits)
            furthest = [0] * n
            for i in range(n):
                lo, hi, best = i + 1, n, i
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if cached_fits(i, mid):
                        best = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                if best == i:
                    raise ValueError("Una unidad no cabe dentro del presupuesto")
                furthest[i] = best
            min_parts = [0] * (n + 1)
            for i in range(n - 1, -1, -1):
                min_parts[i] = 1 + min(min_parts[j] for j in range(i + 1, furthest[i] + 1))
            desired_tokens = self.token_counter(content_text) / max(1, min_parts[0])
            start = 0
            while start < n:
                options = [j for j in range(start + 1, furthest[start] + 1)
                           if min_parts[j] == min_parts[start] - 1]
                end = min(options, key=lambda j: abs(self.token_counter("".join(units[start:j])) - desired_tokens))
                boundaries.append((start, end))
                start = end
            fallback_name = method or "sentence_balanced_overlap"
        else:
            start = 0
            while start < n:
                end = start + 1
                while end < n and fits(start, end + 1):
                    end += 1
                boundaries.append((start, end))
                start = end
            fallback_name = method or "greedy_sentence_overlap"

        overlap_budget = round(self.max_tokens * self.FALLBACK_OVERLAP_RATIO)
        output: list[Chunk] = []
        for part, (begin, end) in enumerate(boundaries, start=1):
            main_text = "".join(units[begin:end])
            prefix_start = begin
            if part > 1:
                for previous in range(begin - 1, -1, -1):
                    prefix = "".join(units[previous:begin])
                    if self.token_counter(prefix) > overlap_budget:
                        break
                    if self.token_counter(combine(prefix + main_text)) > usable:
                        break
                    prefix_start = previous
            overlap_text = "".join(units[prefix_start:begin])
            output.append(self._fallback_chunk(
                original=original,
                text=combine(overlap_text + main_text),
                method=fallback_name,
                part=part,
                total=len(boundaries),
                context_paths=context_paths,
                original_tokens=original_tokens,
                overlap_units=begin - prefix_start,
                overlap_tokens=self.token_counter(overlap_text) if overlap_text else 0,
            ))
        return output

    def _fallback_units(self, text: str, limit: int) -> list[str]:
        """Unidades de oraciones que conservan exactamente todo el texto."""
        # Las líneas estructuradas del JSON se mantienen independientes.
        # Para texto narrativo se buscan límites de oración tras .!? y espacio.
        lines = text.splitlines(keepends=True)
        units: list[str] = []
        for line in lines:
            # Conserva los espacios originales y todos los separadores.
            parts = re.split(r"(?<=[.!?])(?=\s+[^\s])", line)
            for part in parts:
                if not part:
                    continue
                if self.token_counter(part) <= limit:
                    units.append(part)
                else:
                    units.extend(self._fallback_split_long_unit(part, limit))
        return units

    def _fallback_split_long_unit(self, text: str, limit: int) -> list[str]:
        """Trocea una oración demasiado larga por palabras; después caracteres."""
        pieces = re.findall(r"\s+|\S+", text)
        out: list[str] = []
        current = ""
        for piece in pieces:
            if self.token_counter(current + piece) <= limit:
                current += piece
                continue
            if current:
                out.append(current)
                current = ""
            if self.token_counter(piece) <= limit:
                current = piece
                continue
            # Una sola palabra (o espacio) excesiva: cortar caracteres.
            rest = piece
            while rest:
                lo, hi, best = 1, len(rest), 0
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if self.token_counter(rest[:mid]) <= limit:
                        best = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                if not best:
                    raise ValueError("max_tokens insuficiente para un carácter")
                out.append(rest[:best])
                rest = rest[best:]
        if current:
            out.append(current)
        assert "".join(out) == text
        return out

    def _get_root(self, document: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        # Determina qué parte del JSON debe tratarse como raíz
        if self.root_field is not None:
            if self.root_field not in document:
                raise KeyError(f"La clave raíz '{self.root_field}' no existe.")
            block, data = self.root_field, document[self.root_field]
        elif len(document) == 1:
            possible_block, possible_data = next(iter(document.items()))
            if isinstance(possible_data, dict):
                block, data = possible_block, possible_data
            else:
                block, data = "document", document
        else:
            block, data = "document", document
        if not isinstance(data, dict):
            raise TypeError(f"El contenido de la raíz '{block}' debe ser un diccionario.")
        return block, data

    def _split(
        self,
        block: str,
        data: dict[str, Any],
        node: SplitNode,
        level: int,
    ) -> list[Chunk]:
        # Evalúa un nodo del árbol y decide si debe conservarse completo
        # o dividirse mediante sus hijos
        fields = resolve_fields(node, data)
        if not fields:
            return []
        text = render_fields(fields)
        exceeds_target = self.token_counter(text) > self.target_tokens
        # Si supera el límite, se generan obligatoriamente
        # todas las subdivisiones inmediatas del nodo.
        if exceeds_target and node.children:
            chunks: list[Chunk] = []
            for child in node.children:
                chunks.extend(
                    self._split(
                        block=block,
                        data=data,
                        node=child,
                        level=level + 1,
                    )
                )
            return chunks
        # Una hoja excesiva se divide solo entre campos completos.
        # Una pareja clave–valor nunca se parte.
        if exceeds_target:
            return self._pack_fields(
                block=block,
                node=node,
                fields=fields,
                level=level,
            )
        return [
            self._make_chunk(
                block=block,
                node=node,
                fields=fields,
                level=level,
            )
        ]
    
    def _pack_fields(
        self,
        block: str,
        node: SplitNode,
        fields: list[ResolvedField],
        level: int,
    ) -> list[Chunk]:
        # Agrupa campos completos respetando target_tokens.

        # Reglas:
        # - Los campos escalares y diccionarios son indivisibles.
        # - Las listas excesivas se dividen por elementos.
        # - Las listas incluidas en indivisible_list_paths nunca
        # se dividen, aunque superen target_tokens.
        chunks: list[Chunk] = []
        current: list[ResolvedField] = []

        def flush_current() -> None:
            # Convierte el grupo acumulado en un chunk
            nonlocal current

            if current:
                chunks.append(
                    self._make_chunk(
                        block=block,
                        node=node,
                        fields=current,
                        level=level,
                    )
                )
                current = []
        for resolved_field in fields:
            # Cuenta los tokens de esta pareja clave–valor
            # de manera independiente.
            field_text = render_fields(
                [resolved_field]
            )
            field_exceeds_target = (
                self.token_counter(field_text)
                > self.target_tokens
            )
            # Ruta exacta relativa a la raíz interna.
            relative_path = ".".join(
                str(part)
                for part in resolved_field.path
            )
            # Comprueba si la lista debe conservarse completa.
            list_is_indivisible = (
                relative_path
                in self.indivisible_list_paths
            )
            # Divide las listas excesivas que no estén protegidas.
            if (
                isinstance(resolved_field.value, list)
                and field_exceeds_target
                and not list_is_indivisible
            ):
                # Guarda cualquier campo anterior.
                flush_current()
                # Cada elemento genera un chunk independiente.
                for index, item in enumerate(
                    resolved_field.value
                ):
                    if not is_present(item):
                        continue
                    list_item_field = ResolvedField(
                        # Conserva el nombre de la clave original.
                        name=resolved_field.name,
                        # Añade el índice a la ruta JSON.
                        path=(
                            *resolved_field.path,
                            index,
                        ),
                        # El elemento permanece completo.
                        value=item,
                    )
                    chunks.append(
                        self._make_chunk(
                            block=block,
                            node=node,
                            fields=[list_item_field],
                            level=level,
                        )
                    )
                continue
            # Aquí entran:
            # - valores escalares;
            # - diccionarios;
            # - listas que caben;
            # - listas excesivas protegidas.
            candidate = [
                *current,
                resolved_field,
            ]
            candidate_exceeds_target = (
                self.token_counter(
                    render_fields(candidate)
                )
                > self.target_tokens
            )
            if current and candidate_exceeds_target:
                flush_current()
            current.append(resolved_field)
        flush_current()
        return chunks

    def _find_uncovered(
        self,
        data: Any,
        covered_paths: list[tuple[str | int, ...]],
    ) -> list[ResolvedField]:
        # Localiza las ramas del JSON que no aparecen en DOCUMENT_PLAN
        result: list[ResolvedField] = []
        def visit(value: Any, path: tuple[str | int, ...]) -> None:
            # Recorre recursivamente una rama del documento
            if any(path[:len(covered)] == covered for covered in covered_paths):
                return
            has_covered_descendant = any(
                covered[:len(path)] == path for covered in covered_paths
            )
            if has_covered_descendant and isinstance(value, (dict, list)):
                items = value.items() if isinstance(value, dict) else enumerate(value)
                for key, item in items:
                    if is_present(item):
                        visit(item, (*path, key))
            elif is_present(value):
                result.append(ResolvedField(str(path[-1]), path, value))
        for key, value in data.items():
            if is_present(value):
                visit(value, (key,))
        return result

    def _make_chunk(
        self,
        block: str,
        node: SplitNode,
        fields: list[ResolvedField],
        level: int,
    ) -> Chunk:
        # Construye el objeto chunk y calcula todos sus metadatos
        text = render_fields(fields)
        token_count = self.token_counter(text)
        exceeds_target = token_count > self.target_tokens
        exceeds_max = token_count > self.max_tokens
        paths = list(dict.fromkeys(
            self._format_path(block, field.path) for field in fields
        ))
        return Chunk(
            text=text,
            block=block,
            group_name=node.name,
            level=level,
            json_paths=paths,
            token_count=token_count,
            exceeds_target_limit=exceeds_target,
            exceeds_max_limit=exceeds_max,
            metadata={
                "fields": [field.name for field in fields],
                "target_tokens": self.target_tokens,
                "max_tokens": self.max_tokens,
                "exceeds_target_limit": exceeds_target,
                "exceeds_max_limit": exceeds_max,
            },
        )

    @staticmethod
    def _format_path(block: str, path: tuple[str | int, ...]) -> str:
        # Convierte una ruta almacenada como tubla en una ruta JSON legible
        # para guardar en los metadatos de cada chunk
        result = block
        for part in path:
            result += f"[{part}]" if isinstance(part, int) else f".{part}"
        return result