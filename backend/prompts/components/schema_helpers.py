"""Hilfsfunktionen rund um die Pydantic-Output-Schemas.

Bewusst schema_helpers.py (nicht schemas.py), weil schemas/ als Paket
mit den Pydantic-Modellen schon existiert. Vermeidet Import-Chaos.
"""
from typing import Type

from pydantic import BaseModel


def render_schema_for_prompt(schema_class: Type[BaseModel]) -> str:
    """Erzeugt eine im Prompt verwendbare JSON-Schema-Beschreibung.

    Pflicht- und Optional-Felder werden markiert. Das Schema selbst erzwingt das
    Werkzeug (Tool-Use); diese Liste erklärt dem Modell nur die Felder.
    """
    fields_doc = []
    for name, field in schema_class.model_fields.items():
        marker = 'PFLICHT' if field.is_required() else 'OPTIONAL'
        desc = field.description or '(keine Beschreibung)'
        fields_doc.append(f'  - {name} [{marker}]: {desc}')
    return (
        'Felder der Antwort:\n'
        + '\n'.join(fields_doc)
        + ''
    )
