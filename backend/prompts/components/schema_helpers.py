"""Hilfsfunktionen rund um die Pydantic-Output-Schemas.

Bewusst schema_helpers.py (nicht schemas.py), weil schemas/ als Paket
mit den Pydantic-Modellen schon existiert. Vermeidet Import-Chaos.
"""
from typing import Type

from pydantic import BaseModel


def render_schema_for_prompt(schema_class: Type[BaseModel]) -> str:
    """Erzeugt eine im Prompt verwendbare JSON-Schema-Beschreibung.

    Pflicht- und Optional-Felder werden klar markiert. Mistral Strict
    JSON Schema Mode bekommt das Schema separat — diese Doku ist
    zusätzlich für das Modell gedacht, damit es die Felder versteht.
    """
    fields_doc = []
    for name, field in schema_class.model_fields.items():
        marker = 'PFLICHT' if field.is_required() else 'OPTIONAL'
        desc = field.description or '(keine Beschreibung)'
        fields_doc.append(f'  - {name} [{marker}]: {desc}')
    return (
        'Antworte ausschliesslich mit JSON, das diesem Schema entspricht:\n'
        + '\n'.join(fields_doc)
        + '\n\nKein anderer Text. Kein Markdown. Nur valides JSON.'
    )
