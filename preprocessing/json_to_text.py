from typing import Any


def json_to_text(data: dict) -> str:
    """
    Convierte un JSON en texto estructurado.

    Reglas:
    - Valor simple:
      "Titulo: Guía clínica."

    - Clave que contiene otras claves:
      "Poblacion. Descripcion: Adultos. Edad: 45 años."

    - Lista de valores simples:
      "Keywords: asma, manejo, diagnóstico."

    - Cada clave principal genera un párrafo.
    """

    def format_key(key: Any) -> str:
        return str(key).replace("_", " ").strip().capitalize()

    def format_value(value: Any) -> str:
        text = str(value).strip()

        if not text:
            return ""

        if not text.endswith((".", "!", "?")):
            text += "."

        return text

    def is_empty(value: Any) -> bool:
        return value is None or value == "" or value == [] or value == {}

    def is_simple_list(value: Any) -> bool:
        return (
            isinstance(value, list)
            and all(not isinstance(item, (dict, list)) for item in value)
        )

    def format_simple_list(values: list) -> str:
        clean_values = [
            str(value).strip()
            for value in values
            if not is_empty(value) and str(value).strip()
        ]

        return ", ".join(clean_values)

    def process_value(value: Any) -> str:
        parts = []

        if isinstance(value, dict):
            for key, subvalue in value.items():
                if is_empty(subvalue):
                    continue

                if is_simple_list(subvalue):
                    list_text = format_simple_list(subvalue)

                    if list_text:
                        parts.append(
                            f"{format_key(key)}: {format_value(list_text)}"
                        )

                elif isinstance(subvalue, (dict, list)):
                    nested_text = process_value(subvalue)

                    if nested_text:
                        parts.append(
                            f"{format_key(key)}. {nested_text}"
                        )

                else:
                    parts.append(
                        f"{format_key(key)}: {format_value(subvalue)}"
                    )

        elif isinstance(value, list):
            if is_simple_list(value):
                return format_value(format_simple_list(value))

            for item in value:
                if is_empty(item):
                    continue

                nested_text = process_value(item)

                if nested_text:
                    parts.append(nested_text)

        else:
            return format_value(value)

        return " ".join(parts)

    paragraphs = []

    for key, value in data.items():
        if is_empty(value):
            continue

        if is_simple_list(value):
            content = format_simple_list(value)

            if content:
                paragraphs.append(
                    f"{format_key(key)}: {format_value(content)}"
                )

        elif isinstance(value, (dict, list)):
            content = process_value(value)

            if content:
                paragraphs.append(
                    f"{format_key(key)}. {content}"
                )

        else:
            paragraphs.append(
                f"{format_key(key)}: {format_value(value)}"
            )

    return "\n\n".join(paragraphs)