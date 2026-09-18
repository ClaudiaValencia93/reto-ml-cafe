"""Pruebas de métricas, modelos y protocolo de backtesting.

Se usan series sintéticas con respuesta conocida. Probar métricas contra datos
reales solo diría que el código corre, no que calcula lo correcto.
"""

import numpy as np
import pandas as pd
import pytest

from coffee.data import flat_years_pct, quality_tier
from coffee.evaluate import (
    HORIZON,
    MIN_TRAIN_YEARS,
    cut_years,
    mae,
    mape,
    mase,
    naive_scale,
    smape,
)
from coffee.models import drift, ets, mean_forecast, naive


# ------------------------------------------------------------------ métricas

def test_mae_basico():
    assert mae([10, 20, 30], [10, 20, 30]) == 0
    assert mae([10, 20], [12, 18]) == 2


def test_mape_basico():
    assert mape([100, 200], [110, 180]) == pytest.approx(10.0)


def test_smape_es_simetrico():
    """A diferencia del MAPE, sobrestimar y subestimar cuestan lo mismo."""
    assert smape([100], [120]) == pytest.approx(smape([120], [100]))


def test_mape_no_es_simetrico():
    """Justifica reportar también sMAPE: el MAPE castiga más la sobreestimación."""
    assert mape([100], [120]) != pytest.approx(mape([120], [100]))


def test_naive_scale_es_el_cambio_medio_absoluto():
    assert naive_scale([10, 12, 15, 14]) == pytest.approx(np.mean([2, 3, 1]))


def test_naive_scale_es_nan_en_serie_constante():
    """ESTE es el caso de Kenia: 23 años en el mismo valor.

    Con escala cero el MASE sería infinito. Devolver NaN evita que una serie
    degenerada contamine los promedios con un valor absurdo.
    """
    assert np.isnan(naive_scale([50, 50, 50, 50]))


def test_mase_igual_a_uno_cuando_el_error_iguala_al_naive():
    y_train = [10, 12, 14, 16]          # cambio medio = 2
    assert mase([18, 20], [16, 18], y_train) == pytest.approx(1.0)


def test_mase_menor_a_uno_cuando_el_modelo_es_mejor():
    y_train = [10, 12, 14, 16]
    assert mase([18, 20], [18, 20], y_train) == 0.0


# ------------------------------------------------------------------- modelos

def test_naive_repite_el_ultimo_valor():
    assert list(naive(np.array([5.0, 7.0, 9.0]), 3)) == [9, 9, 9]


def test_mean_devuelve_el_promedio():
    assert list(mean_forecast(np.array([10.0, 20.0, 30.0]), 2)) == [20, 20]


def test_drift_extrapola_la_pendiente_media():
    """Serie 10,20,30: pendiente 10 -> 40, 50, 60."""
    assert list(drift(np.array([10.0, 20.0, 30.0]), 3)) == [40, 50, 60]


def test_drift_equivale_a_naive_en_serie_plana():
    plana = np.array([50.0] * 10)
    assert list(drift(plana, 3)) == list(naive(plana, 3))


def test_ets_degrada_a_drift_en_series_muy_cortas():
    """Con menos de 5 puntos no hay con qué estimar; el fallback es explícito."""
    corta = np.array([1.0, 2.0, 3.0])
    assert list(ets(corta, 2)) == list(drift(corta, 2))


def test_los_modelos_devuelven_el_horizonte_pedido():
    y = np.linspace(100, 200, 20)
    for modelo in (naive, mean_forecast, drift, ets):
        for h in (1, 3, 5):
            assert len(modelo(y, h)) == h


# --------------------------------------------------- protocolo de backtesting

def _panel_sintetico(anios=range(1990, 2020), paises=("A", "B")):
    filas = [
        {"country": p, "year": y, "consumption_kg": 1000.0 + 10 * (y - 1990),
         "forecastable": True, "is_low_variance": False,
         "primary_type": "Arabica", "data_quality": "medido"}
        for p in paises for y in anios
    ]
    return pd.DataFrame(filas)


def test_los_cortes_dejan_historia_suficiente_y_horizonte_completo():
    panel = _panel_sintetico()
    cortes = cut_years(panel, horizon=HORIZON, min_train=MIN_TRAIN_YEARS)

    assert cortes[0] == 1990 + MIN_TRAIN_YEARS - 1
    assert cortes[-1] == 2019 - HORIZON
    assert all(c + HORIZON <= 2019 for c in cortes), "no se puede predecir fuera del panel"


def test_no_hay_cortes_si_la_serie_es_demasiado_corta():
    panel = _panel_sintetico(anios=range(2010, 2015))
    assert cut_years(panel) == []


# ------------------------------------------------------- calidad del dato

def test_flat_years_pct_cuenta_anios_sin_cambio():
    assert flat_years_pct([1, 1, 1, 1]) == 100.0
    assert flat_years_pct([1, 2, 3, 4]) == 0.0
    assert flat_years_pct([1, 1, 2, 2]) == pytest.approx(200 / 3)


def test_quality_tier_respeta_los_umbrales():
    assert quality_tier(0) == "medido"
    assert quality_tier(39.9) == "medido"
    assert quality_tier(40) == "parcial"
    assert quality_tier(69.9) == "parcial"
    assert quality_tier(70) == "repetido"
    assert quality_tier(100) == "repetido"
