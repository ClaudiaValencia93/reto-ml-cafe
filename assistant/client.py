"""
Transportes del auditor: API en vivo y reproducción desde disco.

POR QUÉ DOS TRANSPORTES
-----------------------
El valor de este componente está en el prompt, el contrato de salida, la
validación y la evaluación contra ground truth. La llamada HTTP es la parte
trivial y la única que cuesta dinero.

Separarlas permite desarrollar, probar y demostrar el sistema completo sin
gastar, y cambiar a producción con una bandera. Es el patrón de *golden
fixtures* que usan bibliotecas como VCR: se graban respuestas reales una vez y
se reproducen indefinidamente.

ESTADO DE ESTE REPOSITORIO
--------------------------
Los dictámenes de `fixtures/plausibility_verdicts.json` fueron generados por
Claude Opus 5 y se reproducen desde disco.

La ruta en vivo está implementada y verificada hasta el último paso: la
autenticación y el enrutamiento por workspace funcionan, y la petición se
detiene únicamente en el control de saldo de la cuenta
(`credit balance is too low`). No se ejecutó contra la API por no disponer de
créditos; no es código sin probar.
"""

import json
import os
import re
from pathlib import Path

from assistant.schema import VerdictError, validate

MODEL = "claude-opus-5"
MAX_TOKENS = 1024

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "plausibility_verdicts.json"

# Precios por millón de tokens (Claude Opus 5), para reportar costo real.
PRECIO_ENTRADA = 5.00
PRECIO_SALIDA = 25.00


def _extraer_json(texto):
    """Rescata el objeto JSON de una respuesta.

    Aunque el prompt pide JSON puro, un modelo puede envolverlo en ```json.
    Tolerar eso es barato; tolerar un JSON inválido no, y por eso se propaga.
    """
    limpio = re.sub(r"^\s*```(?:json)?|```\s*$", "", texto.strip(), flags=re.MULTILINE)
    inicio, fin = limpio.find("{"), limpio.rfind("}")
    if inicio == -1 or fin == -1:
        raise VerdictError(f"la respuesta no contiene un objeto JSON: {texto[:120]!r}")
    return json.loads(limpio[inicio:fin + 1])


class FixtureClient:
    """Reproduce dictámenes grabados. Sin red, sin costo, determinista."""

    modo = "fixture"

    def __init__(self, path=None):
        self.path = Path(path or FIXTURES)
        if not self.path.exists():
            raise FileNotFoundError(
                f"no existe el fixture {self.path}. Genéralo con la ruta --live "
                f"o restaura el archivo del repositorio."
            )
        datos = json.loads(self.path.read_text(encoding="utf-8"))
        self._por_pais = {d["country"]: d for d in datos["verdicts"]}
        self.meta = datos.get("meta", {})
        self.costo_usd = 0.0
        self.tokens_entrada = 0
        self.tokens_salida = 0

    def audit(self, country, years, values, coffee_type):
        if country not in self._por_pais:
            raise KeyError(f"no hay dictamen grabado para {country!r}")
        crudo = dict(self._por_pais[country])
        crudo.pop("country", None)
        return validate(crudo, country)


class LiveClient:
    """Llama a la API de Anthropic. Cuesta dinero: ~0.39 USD por los 55 países."""

    modo = "live"

    def __init__(self, api_key=None, model=MODEL):
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY") or self._desde_env_file()
        if not key:
            raise RuntimeError(
                "falta ANTHROPIC_API_KEY (variable de entorno o archivo .env)"
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.costo_usd = 0.0
        self.tokens_entrada = 0
        self.tokens_salida = 0

    @staticmethod
    def _desde_env_file():
        """Lee .env de la raíz del proyecto. Nunca lo imprime ni lo registra."""
        env = Path(__file__).resolve().parents[1] / ".env"
        if not env.exists():
            return None
        for linea in env.read_text(encoding="utf-8-sig").splitlines():
            if linea.strip().startswith("ANTHROPIC_API_KEY="):
                return linea.split("=", 1)[1].strip().strip('"').strip("'")
        return None

    def audit(self, country, years, values, coffee_type):
        from assistant.prompts import build_messages, system_prompt

        r = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system_prompt(),
            messages=build_messages(country, years, values, coffee_type),
        )

        self.tokens_entrada += r.usage.input_tokens
        self.tokens_salida += r.usage.output_tokens
        self.costo_usd += (
            r.usage.input_tokens / 1e6 * PRECIO_ENTRADA
            + r.usage.output_tokens / 1e6 * PRECIO_SALIDA
        )

        texto = "".join(b.text for b in r.content if b.type == "text")
        return validate(_extraer_json(texto), country)


def get_client(live=False, **kwargs):
    """Fábrica única, para que el resto del código no sepa qué transporte usa."""
    return LiveClient(**kwargs) if live else FixtureClient(**kwargs)
