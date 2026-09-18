"""Verificación de integridad para la entrega.

Estas pruebas no ejercitan funciones: **auditan el entregable**. Comprueban que
lo que se muestra en la presentación es exactamente lo que produce el código, y
que cada cifra citada en el README y en el deck se puede recalcular.

Las métricas se recomputan aquí con aritmética directa, sin llamar a
`assistant.evaluation`. Si ambas implementaciones coinciden, el número es
correcto; si `evaluation.py` tuviera un error, esta prueba lo delataría en vez
de repetirlo.
"""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from assistant.schema import PATRONES, validate
from coffee.data import load_clean

RAIZ = Path(__file__).resolve().parents[1]
FIXTURE = RAIZ / "assistant" / "fixtures" / "plausibility_verdicts.json"
DECK = RAIZ / "reports" / "presentacion.html"
README = RAIZ / "README.md"

VERDAD = {"medido": True, "repetido": False}


@pytest.fixture(scope="module")
def panel():
    return load_clean()[0]


@pytest.fixture(scope="module")
def fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def deck_datos():
    """Extrae el array de países incrustado en la presentación."""
    html = DECK.read_text(encoding="utf-8")
    m = re.search(r"var D = (\[.*?\]);\s*\n", html, re.DOTALL)
    assert m, "no se encontró el array de datos en la presentación"
    return json.loads(m.group(1))


@pytest.fixture(scope="module")
def auditoria():
    return pd.read_parquet(RAIZ / "data" / "processed" / "plausibility_audit.parquet")


# ===================================================================
# A. Los datos de la presentación son los del dataset, sin desviación
# ===================================================================

def test_la_presentacion_trae_los_53_paises(deck_datos, panel):
    assert len(deck_datos) == 53
    assert {d["c"] for d in deck_datos} == set(panel.country.unique())


def test_cada_serie_de_la_presentacion_coincide_con_el_dataset(deck_datos, panel):
    """Valor por valor, año por año, para los 53 países."""
    por_pais = {d["c"]: d for d in deck_datos}
    for pais, g in panel.groupby("country"):
        g = g.sort_values("year")
        d = por_pais[pais]

        esperado = [int(round(v)) for v in g.consumption_bags_60kg]
        assert d["v"] == esperado, f"{pais}: la serie no coincide"
        assert d["y0"] == int(g.year.min()), f"{pais}: año inicial distinto"
        assert len(d["v"]) == len(g), f"{pais}: número de observaciones distinto"
        assert d["t"] == g.primary_type.iloc[0], f"{pais}: tipo de café distinto"


def test_las_etiquetas_de_calidad_de_la_presentacion_coinciden(deck_datos, panel):
    por_pais = {d["c"]: d for d in deck_datos}
    for pais, g in panel.groupby("country"):
        d = por_pais[pais]
        assert d["q"] == g.data_quality.iloc[0], f"{pais}: calidad distinta"
        assert d["f"] == pytest.approx(g.flat_years_pct.iloc[0], abs=0.05), \
            f"{pais}: % de años planos distinto"


def test_los_dictamenes_de_la_presentacion_son_los_del_fixture(deck_datos, fixture):
    grabados = {v["country"]: v for v in fixture["verdicts"]}
    for d in deck_datos:
        g = grabados[d["c"]]
        assert d["fx"]["p"] == g["plausible"], f"{d['c']}: veredicto distinto"
        assert d["fx"]["n"] == g["confidence"], f"{d['c']}: confianza distinta"
        assert d["fx"]["t"] == g["pattern"], f"{d['c']}: patrón distinto"
        assert d["fx"]["r"] == g["reason"], f"{d['c']}: justificación distinta"


# ===================================================================
# B. Los 53 dictámenes cumplen el contrato
# ===================================================================

def test_los_53_dictamenes_pasan_la_validacion(fixture):
    for v in fixture["verdicts"]:
        crudo = {k: v[k] for k in ("plausible", "confidence", "pattern", "reason")}
        validate(crudo, v["country"])  # lanza si algo incumple


def test_no_hay_paises_repetidos_en_el_fixture(fixture):
    paises = [v["country"] for v in fixture["verdicts"]]
    assert len(paises) == len(set(paises))


def test_todas_las_justificaciones_son_sustantivas(fixture):
    """Una justificación de dos palabras cumpliría el esquema y no diría nada."""
    for v in fixture["verdicts"]:
        assert len(v["reason"]) >= 80, f"{v['country']}: justificación demasiado corta"
        assert v["reason"].strip().endswith("."), f"{v['country']}: frase sin cerrar"


def test_los_patrones_son_coherentes_con_el_veredicto(fixture):
    """'medicion_real' implica plausible; los demás patrones, no plausible."""
    for v in fixture["verdicts"]:
        if v["pattern"] == "medicion_real":
            assert v["plausible"] is True, f"{v['country']}: patrón y veredicto se contradicen"
        else:
            assert v["plausible"] is False, f"{v['country']}: patrón y veredicto se contradicen"


# ===================================================================
# B-bis. Las justificaciones no pueden contradecir la serie
# ===================================================================

def _racha_maxima(y):
    """Mayor número de años consecutivos con el mismo valor."""
    mejor = act = 1
    for i in range(1, len(y)):
        act = act + 1 if y[i] == y[i - 1] else 1
        mejor = max(mejor, act)
    return mejor


# En v2 ninguna justificación puede contradecir la serie, porque ninguna habla
# de la serie: ese territorio es del código (`series_facts.py`). Las tres
# afirmaciones falsas de v1 quedan preservadas en `reason_v1` para que la
# revisión sea auditable.
CLAIMS_CONTRADICHAS = set()


def _contradicciones(fixture, panel):
    import numpy as np

    fallos = set()
    for v in fixture["verdicts"]:
        g = panel[panel.country == v["country"]].sort_values("year")
        y = g.consumption_bags_60kg.to_numpy(float)
        r = v["reason"].lower()
        n_distintos = len(np.unique(y))

        for m in re.finditer(r"(\d+)\s+anios?\s+(?:consecutivos|seguidos)", r):
            if _racha_maxima(y) < int(m.group(1)):
                fallos.add(v["country"])

        if re.search(r"sin (?:un solo )?retroceso", r) and bool((np.diff(y) < 0).any()):
            fallos.add(v["country"])

        if re.search(r"(?:constante|identic[oa]).{0,40}"
                     r"(?:todo el periodo|treinta anios|tres decadas)", r):
            if n_distintos > 2:
                fallos.add(v["country"])

    return fallos


def test_ninguna_justificacion_nueva_contradice_su_serie(fixture, panel):
    """Barrera de regresión.

    Verifica las afirmaciones numéricas de cada justificación contra la serie
    real. Si un dictamen nuevo afirma algo que los datos desmienten, esta prueba
    falla en vez de dejarlo pasar al entregable.
    """
    encontradas = _contradicciones(fixture, panel)
    nuevas = encontradas - CLAIMS_CONTRADICHAS
    assert not nuevas, f"justificaciones con afirmaciones contradichas por los datos: {sorted(nuevas)}"


def test_ninguna_justificacion_invade_el_territorio_del_codigo(fixture):
    """El reparto se hace cumplir, no se pide amablemente.

    `validate()` rechaza cualquier dictamen que cite cifras de la serie. Esta
    prueba lo comprueba sobre las 53 justificaciones publicadas.
    """
    from assistant.schema import CLAIMS_DE_SERIE

    invasiones = []
    for v in fixture["verdicts"]:
        for patron, motivo in CLAIMS_DE_SERIE:
            m = patron.search(v["reason"])
            if m:
                invasiones.append(f"{v['country']}: {motivo} ({m.group(0)!r})")
    assert not invasiones, "\n".join(invasiones)


def test_la_revision_conserva_el_texto_original(fixture):
    """Trazabilidad de la revisión: qué decía antes cada justificación."""
    for v in fixture["verdicts"]:
        assert v.get("reason_v1"), f"{v['country']}: falta el texto de v1"
        assert v["reason_v1"] != v["reason"], f"{v['country']}: la revisión no cambió nada"


def test_las_afirmaciones_falsas_de_v1_quedan_documentadas(fixture, panel):
    """La v1 contenía afirmaciones que los datos desmienten. Se conservan para
    que la revisión sea auditable: borrarlas ocultaría el motivo del cambio."""
    v1 = {v["country"]: dict(v, reason=v["reason_v1"]) for v in fixture["verdicts"]}
    contradicciones = _contradicciones({"verdicts": list(v1.values())}, panel)
    assert {"Bolivia (Plurinational State of)", "Rwanda"} <= contradicciones


def test_la_revision_no_toco_ningun_veredicto(auditoria):
    """Lo único que se reescribió es la prosa. Si un veredicto hubiera cambiado,
    las métricas publicadas dejarían de ser las de la corrida evaluada."""
    r = auditoria[auditoria.country == "Rwanda"].iloc[0]
    assert r.data_quality == "repetido"
    # `bool(...)`: pandas devuelve numpy.bool_, y `is False` fallaría siendo falso
    assert bool(r.plausible) is False
    assert r.confidence == 0.85

    b = auditoria[auditoria.country.str.startswith("Bolivia")].iloc[0]
    assert bool(b.plausible) is False and b.pattern == "interpolado"
    assert b.confidence == 0.66, "sigue siendo el mismo falso positivo"


# ===================================================================
# C. Las métricas, recalculadas desde cero
# ===================================================================

def _recuento(auditoria):
    """Aritmética directa, sin usar assistant.evaluation."""
    tp = fp = fn = tn = 0
    for _, f in auditoria.iterrows():
        if f.data_quality not in VERDAD:
            continue
        real_plausible = VERDAD[f.data_quality]
        if not f.plausible and not real_plausible:
            tp += 1
        elif not f.plausible and real_plausible:
            fp += 1
        elif f.plausible and not real_plausible:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def test_el_reparto_de_paises_es_el_declarado(auditoria):
    assert len(auditoria) == 53
    conteo = auditoria.data_quality.value_counts().to_dict()
    assert conteo["medido"] == 18
    assert conteo["parcial"] == 14
    assert conteo["repetido"] == 21
    assert 18 + 14 + 21 == 53


def test_la_matriz_de_confusion_es_la_publicada(auditoria):
    tp, fp, fn, tn = _recuento(auditoria)
    assert (tp, fn, fp, tn) == (21, 0, 1, 17), "la matriz del deck dice 21/0/1/17"
    assert tp + fp + fn + tn == 39, "solo se evalúan los 39 no ambiguos"


def test_precision_recall_y_f1_son_los_publicados(auditoria):
    tp, fp, fn, tn = _recuento(auditoria)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    exactitud = (tp + tn) / (tp + fp + fn + tn)

    assert round(precision * 100, 1) == 95.5
    assert round(recall * 100, 1) == 100.0
    assert round(f1 * 100, 1) == 97.7
    assert round(exactitud * 100, 1) == 97.4


def test_el_unico_fallo_es_bolivia(auditoria):
    d = auditoria[auditoria.data_quality.isin(VERDAD)].copy()
    d["verdad"] = d.data_quality.map(VERDAD)
    fallos = d[d.plausible != d.verdad]
    assert len(fallos) == 1
    assert fallos.iloc[0].country.startswith("Bolivia")


def test_la_banda_de_confianza_baja_son_7_paises_con_6_aciertos(auditoria):
    """El 86% publicado. La banda es <= 0.70, cerrada por la derecha."""
    d = auditoria[auditoria.data_quality.isin(VERDAD)].copy()
    d["verdad"] = d.data_quality.map(VERDAD)
    baja = d[d.confidence <= 0.70]

    assert len(baja) == 7, "la banda baja tiene 7 países"
    aciertos = int((baja.plausible == baja.verdad).sum())
    assert aciertos == 6
    assert round(aciertos / len(baja) * 100) == 86

    esperados = {"Guyana", "Togo", "Ghana", "Madagascar",
                 "Timor-Leste", "Trinidad & Tobago"}
    assert esperados < set(baja.country), "cambió la composición de la banda"


def test_las_bandas_altas_aciertan_el_100_por_ciento(auditoria):
    d = auditoria[auditoria.data_quality.isin(VERDAD)].copy()
    d["verdad"] = d.data_quality.map(VERDAD)
    alta = d[d.confidence > 0.70]
    assert len(alta) == 32
    assert (alta.plausible == alta.verdad).all(), "alguna banda alta falla"


def test_yemen_no_entra_en_ninguna_metrica(auditoria):
    """El caso que motivó la corrección: confianza baja pero país excluido."""
    yemen = auditoria[auditoria.country == "Yemen"].iloc[0]
    assert yemen.data_quality == "parcial"
    assert yemen.confidence <= 0.70, "su confianza es baja..."
    assert yemen.data_quality not in VERDAD, "...pero no cuenta en la calibración"


# ===================================================================
# D. Las cifras citadas en README y presentación existen de verdad
# ===================================================================

@pytest.mark.parametrize("cifra", [
    "95,5%", "100%", "86%", "53", "39", "21", "17",
])
def test_la_presentacion_cita_cifras_consistentes(cifra):
    """El F1 queda fuera a propósito: es derivado de precisión y recall, y para
    una audiencia de negocio no añade nada sobre las dos. Vive en el README."""
    html = DECK.read_text(encoding="utf-8")
    assert cifra in html, f"la presentación ya no menciona {cifra}"


def test_el_f1_vive_en_el_readme():
    assert "97.7%" in README.read_text(encoding="utf-8")


def test_el_readme_y_la_presentacion_no_se_contradicen():
    r = README.read_text(encoding="utf-8")
    h = DECK.read_text(encoding="utf-8")
    for cifra in ("95.5%", "95,5%"):
        if cifra in r or cifra in h:
            break
    else:
        pytest.fail("la precisión no aparece en ninguno de los dos")

    assert "18" in r and "14" in r and "21" in r, "faltan los conteos por banda"
    assert "6 de 7" in r or "6 de 7" in h, "el 86% debe declarar su composición"


def test_la_presentacion_declara_el_alcance_de_las_metricas():
    """Que no vuelva a aplicarse una métrica a un país que no la compone."""
    h = DECK.read_text(encoding="utf-8")
    assert "39 países" in h or "39</strong> países" in h
    assert "fuera de la evaluación" in h
