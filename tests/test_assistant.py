"""Pruebas del auditor de plausibilidad.

Cubren tres cosas distintas:

1. El contrato de salida se hace cumplir (un LLM devuelve texto; sin validación,
   cualquier variación se propaga en silencio).
2. La reproducción desde fixtures es fiel y completa.
3. **La evaluación no es circular**: al modelo no se le entrega el detector
   determinista contra el que después se le mide.

El punto 3 es el que sostiene todo el experimento, y por eso tiene una prueba
dedicada que falla si alguien "mejora" el prompt añadiendo la respuesta.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from assistant.client import FixtureClient, _extraer_json
from assistant.evaluation import matriz_confusion, metricas
from assistant.prompts import system_prompt, user_prompt
from assistant.schema import PATRONES, Verdict, VerdictError, validate
from coffee.data import load_clean

VALIDO = {
    "plausible": False,
    "confidence": 0.9,
    "pattern": "valor_arrastrado",
    "reason": "La población del país se duplicó en el periodo.",
}


# ------------------------------------------------------------------ contrato

def test_acepta_un_dictamen_valido():
    v = validate(VALIDO, "Kenya")
    assert isinstance(v, Verdict)
    assert v.country == "Kenya" and v.plausible is False
    assert v.confidence == 0.9 and v.pattern == "valor_arrastrado"


@pytest.mark.parametrize("campo", ["plausible", "confidence", "pattern", "reason"])
def test_rechaza_campos_faltantes(campo):
    payload = {k: v for k, v in VALIDO.items() if k != campo}
    with pytest.raises(VerdictError, match="faltan campos"):
        validate(payload, "X")


@pytest.mark.parametrize("valor", [-0.1, 1.5, "alta", None, True])
def test_rechaza_confianzas_invalidas(valor):
    """'alta' y True son los modos de fallo reales de un LLM, no casos teóricos."""
    with pytest.raises(VerdictError):
        validate({**VALIDO, "confidence": valor}, "X")


def test_rechaza_patrones_fuera_del_vocabulario():
    with pytest.raises(VerdictError, match="desconocido"):
        validate({**VALIDO, "pattern": "sospechoso"}, "X")


@pytest.mark.parametrize("valor", ["", "   ", 42, None])
def test_rechaza_razones_vacias(valor):
    with pytest.raises(VerdictError):
        validate({**VALIDO, "reason": valor}, "X")


def test_rechaza_razones_desbordadas():
    with pytest.raises(VerdictError, match="excede"):
        validate({**VALIDO, "reason": "x" * 5000}, "X")


def test_rechaza_plausible_no_booleano():
    with pytest.raises(VerdictError, match="booleano"):
        validate({**VALIDO, "plausible": "no"}, "X")


# ------------------------------------------------------- extracción del JSON

def test_extrae_json_envuelto_en_bloque_de_codigo():
    """El prompt pide JSON puro, pero un modelo puede envolverlo igual."""
    texto = '```json\n{"plausible": true, "confidence": 0.5}\n```'
    assert _extraer_json(texto)["plausible"] is True


def test_extrae_json_con_texto_alrededor():
    texto = 'Claro, aquí tienes:\n{"plausible": false}\nEspero que sirva.'
    assert _extraer_json(texto)["plausible"] is False


def test_falla_si_no_hay_json():
    with pytest.raises(VerdictError, match="no contiene un objeto JSON"):
        _extraer_json("No puedo responder eso.")


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def cliente():
    return FixtureClient()


def test_el_fixture_cubre_todos_los_paises_del_panel(cliente):
    panel, _ = load_clean()
    faltantes = set(panel.country.unique()) - set(cliente._por_pais)
    assert not faltantes, f"sin dictamen: {sorted(faltantes)}"


def test_todos_los_dictamenes_grabados_son_validos(cliente):
    """Si el fixture se edita a mano, esta prueba detecta cualquier corrupción."""
    for pais in cliente._por_pais:
        v = cliente.audit(pais, [1990], [1000], "Arabica")
        assert v.pattern in PATRONES
        assert 0 <= v.confidence <= 1


def test_el_fixture_declara_su_procedencia(cliente):
    """Trazabilidad: de dónde salieron los dictámenes y con qué prompt."""
    assert cliente.meta.get("generated_by")
    assert cliente.meta.get("prompt_version")
    assert "flat_years_pct" in cliente.meta.get("inputs_deliberately_withheld", [])


def test_falla_ante_un_pais_no_grabado(cliente):
    with pytest.raises(KeyError):
        cliente.audit("Atlantis", [1990], [1], "Arabica")


def test_el_fixture_no_tiene_costo(cliente):
    assert cliente.costo_usd == 0.0


# ------------------------------------------- la evaluación no debe ser circular

def test_el_prompt_no_revela_el_detector_determinista():
    """LA prueba que sostiene el experimento.

    Si el modelo recibiera `flat_years_pct` o `data_quality`, repetiría la
    conclusión del detector y la evaluación mediría copia, no juicio.
    """
    p = user_prompt("Kenya", [1990, 1991], [50000, 50000], "Arabica")
    prohibido = ["flat_years_pct", "data_quality", "repetido", "parcial",
                 "medido", "arrastrado", "años planos", "sin cambio", "% plano"]
    for termino in prohibido:
        assert termino not in p, f"el prompt filtra el ground truth: {termino!r}"


def test_el_prompt_entrega_la_serie_cruda():
    p = user_prompt("Kenya", [1990, 1991], [50000, 55000], "Arabica")
    assert "Kenya" in p and "1990" in p and "50,000" in p and "Arabica" in p


def test_el_system_prompt_incluye_el_esquema():
    s = system_prompt()
    assert "plausible" in s and "confidence" in s and "indeterminado" in s


# ------------------------------------------------------------------ métricas

def _tabla(filas):
    return pd.DataFrame(filas)


def test_matriz_de_confusion_cuenta_bien():
    tabla = _tabla([
        {"country": "A", "plausible": False, "data_quality": "repetido"},  # TP
        {"country": "B", "plausible": True, "data_quality": "medido"},       # TN
        {"country": "C", "plausible": False, "data_quality": "medido"},      # FP
        {"country": "D", "plausible": True, "data_quality": "repetido"},   # FN
    ])
    cm = matriz_confusion(tabla)
    assert (cm["tp"], cm["tn"], cm["fp"], cm["fn"]) == (1, 1, 1, 1)


def test_los_ambiguos_quedan_fuera_del_calculo():
    """Los 'parcial' no son verdad ni mentira para el detector; incluirlos
    premiaría o penalizaría al auditor por algo indefinido."""
    tabla = _tabla([
        {"country": "A", "plausible": False, "data_quality": "repetido"},
        {"country": "B", "plausible": True, "data_quality": "parcial"},
        {"country": "C", "plausible": False, "data_quality": "parcial"},
    ])
    assert matriz_confusion(tabla)["n"] == 1


def test_metricas_en_el_caso_perfecto():
    m = metricas({"tp": 10, "fp": 0, "fn": 0, "tn": 10, "n": 20})
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["exactitud"] == 1.0


def test_metricas_no_explotan_sin_positivos():
    import math
    m = metricas({"tp": 0, "fp": 0, "fn": 0, "tn": 5, "n": 5})
    assert math.isnan(m["precision"]) and math.isnan(m["recall"])
