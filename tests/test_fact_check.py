"""Contraste de las afirmaciones del auditor contra una fuente externa.

El modelo razona desde memoria, no desde consulta: puede afirmar un hecho falso
con total aplomo. Las cifras de la SERIE ya no puede inventarlas —ese territorio
es del código—, pero las del MUNDO sí, y son la mitad del argumento.

Estas pruebas contrastan cada afirmación cuantitativa sobre población contra la
serie SP.POP.TOTL del Banco Mundial, guardada como snapshot en `data/raw/` para
que la verificación no dependa de la red.

Cuatro afirmaciones no resistieron el contraste y se corrigieron:

    Tanzania          decía "triplicó"      factor real 2.27x
    Malawi            decía "se triplicó"   factor real 1.99x
    Papua Nueva Guinea decía "casi duplicó" factor real 2.47x (lo subestimaba)
    Honduras          decía "casi duplicó"  factor real 2.00x (lo subestimaba)

Ningún veredicto cambió: solo la prosa.
"""

import json
import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
FIXTURE = RAIZ / "assistant" / "fixtures" / "plausibility_verdicts.json"
POBLACION = RAIZ / "data" / "raw" / "worldbank_population.json"

# Tolerancia para una cifra redondeada en prosa ("unos 23 millones").
TOLERANCIA_PCT = 8.0

# El nombre del país en el fixture no siempre coincide con el del Banco Mundial.
EQUIVALE = {
    "Democratic Republic of Congo": "Congo, Dem. Rep.",
    "Côte d'Ivoire": "Cote d'Ivoire",
}

CIFRA_POBLACION = re.compile(
    r"(?:pas[oó]|creci[oó])\s+de\s+(?:unos\s+)?([\d.]+)\s+a\s+([\d.]+)\s+millones"
)

MULTIPLICADOR = re.compile(
    r"(m[áa]s que duplic[oó]|casi\s+(?:se\s+)?duplic[oó]|se duplic[oó]|duplic[oó]|"
    r"triplic[oó]|se multiplic[oó] por dos y medio)"
)

# Qué factor mínimo exige cada expresión.
EXIGE = {
    "duplicó": 1.85, "duplico": 1.85,
    "se duplicó": 1.85, "se duplico": 1.85,
    "casi se duplicó": 1.60, "casi duplicó": 1.60,
    "más que duplicó": 2.00, "mas que duplico": 2.00,
    "triplicó": 2.80, "triplico": 2.80,
    "se multiplicó por dos y medio": 2.35,
}


@pytest.fixture(scope="module")
def poblacion():
    return json.loads(POBLACION.read_text(encoding="utf-8"))["population"]


@pytest.fixture(scope="module")
def verdicts():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["verdicts"]


def _wb(pais):
    return EQUIVALE.get(pais, pais)


def test_el_snapshot_del_banco_mundial_esta_versionado(poblacion):
    """Sin el snapshot la verificación dependería de la red y dejaría de correr."""
    assert len(poblacion) >= 20
    assert all("1990" in d and "2019" in d for d in poblacion.values())


def test_las_cifras_de_poblacion_resisten_el_contraste(poblacion, verdicts):
    """Las 14 afirmaciones del tipo 'pasó de X a Y millones'."""
    revisadas, fallos = 0, []
    for v in verdicts:
        m = CIFRA_POBLACION.search(v["reason"])
        if not m:
            continue
        pais = _wb(v["country"])
        if pais not in poblacion:
            fallos.append(f"{v['country']}: sin dato del Banco Mundial")
            continue

        revisadas += 1
        for afirmado, anio in ((float(m.group(1)), "1990"), (float(m.group(2)), "2019")):
            real = poblacion[pais][anio] / 1e6
            error = abs(real - afirmado) / real * 100
            if error > TOLERANCIA_PCT:
                fallos.append(
                    f"{v['country']} {anio}: afirma {afirmado} M, el Banco Mundial "
                    f"da {real:.1f} M ({error:.1f}% de error)"
                )

    assert revisadas >= 14, f"solo se revisaron {revisadas} afirmaciones"
    assert not fallos, "\n".join(fallos)


def test_los_multiplicadores_resisten_el_contraste(poblacion, verdicts):
    """'Duplicó', 'triplicó': cada expresión exige un factor mínimo."""
    revisadas, fallos = 0, []
    for v in verdicts:
        if "poblaci" not in v["reason"].lower():
            continue
        m = MULTIPLICADOR.search(v["reason"])
        if not m:
            continue
        pais = _wb(v["country"])
        if pais not in poblacion:
            continue

        revisadas += 1
        factor = poblacion[pais]["2019"] / poblacion[pais]["1990"]
        minimo = EXIGE[m.group(1).lower()]
        if factor < minimo:
            fallos.append(
                f"{v['country']}: dice {m.group(1)!r} (exige >= {minimo}x) "
                f"pero el factor real es {factor:.2f}x"
            )

    assert revisadas >= 8, f"solo se revisaron {revisadas} multiplicadores"
    assert not fallos, "\n".join(fallos)


def test_tanzania_y_malawi_ya_no_dicen_triplico(verdicts):
    """Las dos correcciones concretas, fijadas para que no reaparezcan."""
    por_pais = {v["country"]: v["reason"] for v in verdicts}
    assert "triplic" not in por_pais["Tanzania"].lower()
    assert "triplic" not in por_pais["Malawi"].lower()


def test_el_fixture_declara_que_fue_contrastado(verdicts):
    meta = json.loads(FIXTURE.read_text(encoding="utf-8"))["meta"]
    assert "fact_check" in meta
    assert "Banco Mundial" in meta["fact_check"]
