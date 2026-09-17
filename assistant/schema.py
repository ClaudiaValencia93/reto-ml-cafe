"""
Esquema de la respuesta del auditor de plausibilidad.

POR QUÉ UN ESQUEMA EXPLÍCITO
----------------------------
Un LLM devuelve texto. Si ese texto se consume sin validar, cualquier variación
—una coma de más, un campo ausente, una confianza expresada como "alta" en vez
de 0.9— se propaga silenciosamente al análisis. La validación convierte un fallo
silencioso en una excepción con nombre, que es la diferencia entre un prototipo
y un componente de producción.

Todo dictamen que no cumpla este contrato se rechaza y se registra, nunca se
adivina ni se repara a medias.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict

# Patrones que el auditor puede reconocer. Cerrar el vocabulario evita que cada
# respuesta invente su propia taxonomía y hace agregables los resultados.
PATRONES = {
    "medicion_real",      # la serie parece medida año a año
    "valor_arrastrado",   # el último valor conocido se repite por falta de reporte
    "interpolado",        # variaciones demasiado suaves o escalonadas para ser reales
    "indeterminado",      # no hay base suficiente para juzgar
}

MAX_RAZON = 600  # caracteres; corta respuestas desbordadas sin truncar silencio


class VerdictError(ValueError):
    """La respuesta del modelo no cumple el contrato."""


@dataclass(frozen=True)
class Verdict:
    """Dictamen sobre una serie de consumo de un país."""

    country: str
    plausible: bool
    confidence: float
    pattern: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate(payload: Dict[str, Any], country: str) -> Verdict:
    """Valida la respuesta cruda del modelo y la convierte en `Verdict`.

    Lanza `VerdictError` con un mensaje accionable ante cualquier desviación.
    """
    if not isinstance(payload, dict):
        raise VerdictError(f"{country}: se esperaba un objeto JSON, llegó {type(payload).__name__}")

    faltantes = {"plausible", "confidence", "pattern", "reason"} - set(payload)
    if faltantes:
        raise VerdictError(f"{country}: faltan campos {sorted(faltantes)}")

    plausible = payload["plausible"]
    if not isinstance(plausible, bool):
        raise VerdictError(f"{country}: 'plausible' debe ser booleano, llegó {plausible!r}")

    confidence = payload["confidence"]
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise VerdictError(f"{country}: 'confidence' debe ser numérico, llegó {confidence!r}")
    if not 0.0 <= float(confidence) <= 1.0:
        raise VerdictError(f"{country}: 'confidence' fuera de [0,1]: {confidence}")

    pattern = payload["pattern"]
    if pattern not in PATRONES:
        raise VerdictError(
            f"{country}: 'pattern' desconocido {pattern!r}; esperado uno de {sorted(PATRONES)}"
        )

    reason = payload["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise VerdictError(f"{country}: 'reason' vacío o no textual")
    if len(reason) > MAX_RAZON:
        raise VerdictError(f"{country}: 'reason' excede {MAX_RAZON} caracteres ({len(reason)})")

    return Verdict(
        country=country,
        plausible=plausible,
        confidence=float(confidence),
        pattern=pattern,
        reason=reason.strip(),
    )


# Esquema JSON, para documentar el contrato y para poder activar salidas
# estructuradas en la ruta en vivo sin reescribir la definición.
JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["plausible", "confidence", "pattern", "reason"],
    "properties": {
        "plausible": {
            "type": "boolean",
            "description": "¿La serie es compatible con lo que se sabe del país?",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Seguridad del dictamen. Baja si el país es pequeño o poco documentado.",
        },
        "pattern": {
            "type": "string",
            "enum": sorted(PATRONES),
        },
        "reason": {
            "type": "string",
            "maxLength": MAX_RAZON,
            "description": "Justificación en una o dos frases, citando el hecho del mundo que sustenta el juicio.",
        },
    },
}
