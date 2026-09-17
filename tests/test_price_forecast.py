"""Pruebas del pronóstico de precios con intervalos.

Lo que hay que garantizar aquí no es que el pronóstico "acierte" —eso no se puede
probar—, sino que **el intervalo sea un intervalo**: anidado, creciente con el
horizonte y siempre positivo. Un rango mal construido se ve perfectamente normal
en un gráfico y es inservible para decidir.
"""

import numpy as np
import pandas as pd
import pytest

from coffee.price_forecast import (
    HOLDOUT_CUTOFF,
    LEVELS,
    coverage,
    forecast_with_intervals,
    future_index,
    holdout_evaluation,
    select_order,
)
from coffee.prices import SERIES, load_price_csv


@pytest.fixture(scope="module")
def arabica():
    return load_price_csv(SERIES["Arabica"])


@pytest.fixture(scope="module")
def pronostico(arabica):
    return forecast_with_intervals(arabica.to_numpy(float), 24)


# ------------------------------------------------------------------ cobertura

def test_coverage_cuenta_lo_que_cae_dentro():
    assert coverage([5, 5, 5], [0, 0, 0], [10, 10, 10]) == 100.0
    assert coverage([5, 50, 5], [0, 0, 0], [10, 10, 10]) == pytest.approx(200 / 3)
    assert coverage([50], [0], [10]) == 0.0


def test_coverage_incluye_los_bordes():
    assert coverage([0, 10], [0, 0], [10, 10]) == 100.0


# ---------------------------------------------------------- forma del intervalo

def test_devuelve_las_columnas_de_cada_nivel(pronostico):
    esperadas = {"forecast"} | {
        f"{lado}_{int(round(n * 100))}" for n in LEVELS for lado in ("lo", "hi")
    }
    assert esperadas <= set(pronostico.columns)
    assert len(pronostico) == 24


def test_los_intervalos_estan_anidados(pronostico):
    """El 95% debe contener al 80%, que debe contener al central."""
    assert (pronostico.lo_95 <= pronostico.lo_80).all()
    assert (pronostico.lo_80 <= pronostico.forecast).all()
    assert (pronostico.forecast <= pronostico.hi_80).all()
    assert (pronostico.hi_80 <= pronostico.hi_95).all()


def test_la_incertidumbre_crece_con_el_horizonte(pronostico):
    """Propiedad esencial: pronosticar a 24 meses no puede ser tan preciso como
    a 1 mes. Un intervalo de ancho constante indica un modelo mal especificado."""
    ancho = pronostico.hi_95 - pronostico.lo_95
    assert ancho.iloc[-1] > ancho.iloc[0]
    assert ancho.is_monotonic_increasing


def test_todo_el_intervalo_es_positivo(pronostico):
    """Justifica modelar en logaritmos: en niveles, el límite inferior puede
    cruzar el cero en horizontes largos y un precio negativo no existe."""
    assert (pronostico.lo_95 > 0).all()
    assert (pronostico.forecast > 0).all()


# ---------------------------------------------------------------- índice futuro

def test_el_indice_futuro_arranca_el_mes_siguiente():
    idx = future_index("2026-07-01", 3)
    assert list(idx) == [pd.Timestamp("2026-08-01"),
                         pd.Timestamp("2026-09-01"),
                         pd.Timestamp("2026-10-01")]


def test_el_indice_futuro_cruza_bien_el_fin_de_anio():
    idx = future_index("2026-11-01", 3)
    assert list(idx) == [pd.Timestamp("2026-12-01"),
                         pd.Timestamp("2027-01-01"),
                         pd.Timestamp("2027-02-01")]


# -------------------------------------------------------------------- selección

def test_select_order_devuelve_un_orden_valido(arabica):
    orden, aic = select_order(arabica.to_numpy(float)[:200])
    assert len(orden) == 3
    assert orden[1] == 1, "d=1 está fijado: los precios no son estacionarios en nivel"
    assert np.isfinite(aic)


# --------------------------------------------------------------------- holdout

def test_el_holdout_no_toca_los_datos_de_prueba(arabica):
    """El entrenamiento debe terminar exactamente en el corte."""
    tabla, resumen = holdout_evaluation(arabica)
    corte = pd.Timestamp(HOLDOUT_CUTOFF)

    assert resumen["n_train"] == len(arabica.loc[:corte])
    assert tabla.index.min() > corte, "ninguna predicción puede caer antes del corte"
    assert resumen["n_train"] + resumen["n_test"] == len(arabica)


def test_el_holdout_reporta_cobertura_por_nivel(arabica):
    _, resumen = holdout_evaluation(arabica)
    for nivel in LEVELS:
        clave = f"cobertura_{int(round(nivel * 100))}"
        assert clave in resumen
        assert 0 <= resumen[clave] <= 100


def test_el_modelo_subestimo_el_shock_de_precios(arabica):
    """Documenta el hallazgo: el sesgo es positivo y del mismo orden que el MAPE,
    o sea que el modelo se quedó corto en CASI TODOS los meses, no en algunos.

    Si esta prueba falla es porque cambió la fuente o el corte, y la narrativa
    del informe deja de ser válida."""
    _, resumen = holdout_evaluation(arabica)
    assert resumen["sesgo_pct"] > 20, "subestimó sistemáticamente"
    assert resumen["sesgo_pct"] == pytest.approx(resumen["mape"], rel=0.15)
