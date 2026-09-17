"""Pruebas del módulo de precios.

Lo crítico aquí es el **alineamiento temporal**: los precios vienen en año
calendario y el consumo en año cafetero (octubre-septiembre). Un error de tres
meses en la agregación contamina cualquier cruce posterior sin dar ninguna señal
de que algo anda mal. Por eso se prueba explícitamente.
"""

import pandas as pd
import pytest

from coffee.data import load_clean
from coffee.prices import (
    CROP_YEAR_START_MONTH,
    SERIES,
    load_price_csv,
    load_prices,
    overlap_with_consumption,
    to_crop_year,
)


@pytest.fixture(scope="module")
def precios():
    return load_prices()


# ------------------------------------------------------------- serie mensual

@pytest.mark.parametrize("filename", list(SERIES.values()))
def test_las_series_mensuales_cargan_sin_huecos(filename):
    s = load_price_csv(filename)
    assert len(s) == 415
    assert s.isna().sum() == 0
    assert s.index.min() == pd.Timestamp("1992-01-01")
    assert (s > 0).all()


def test_los_meses_son_consecutivos():
    """Un mes faltante rompería el promedio del año cafetero sin avisar."""
    s = load_price_csv(SERIES["Arabica"])
    esperado = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    assert s.index.equals(esperado)


# ----------------------------------------------------- agregación año cafetero

def test_el_anio_cafetero_va_de_octubre_a_septiembre():
    """Octubre de 1992 y septiembre de 1993 deben caer en el MISMO año: 1992."""
    fechas = pd.date_range("1992-01-01", "1994-12-01", freq="MS")
    serie = pd.Series(1.0, index=fechas)

    year = fechas.year + (fechas.month >= CROP_YEAR_START_MONTH).astype(int) - 1

    assert year[fechas.get_loc(pd.Timestamp("1992-10-01"))] == 1992
    assert year[fechas.get_loc(pd.Timestamp("1993-09-01"))] == 1992
    assert year[fechas.get_loc(pd.Timestamp("1993-10-01"))] == 1993
    assert year[fechas.get_loc(pd.Timestamp("1992-09-01"))] == 1991


def test_solo_sobreviven_anios_de_12_meses():
    """1991 tiene 9 meses y 2025 tiene 10: ambos deben quedar fuera."""
    anual = to_crop_year(load_price_csv(SERIES["Arabica"]))
    assert anual.index.min() == 1992
    assert anual.index.max() == 2024
    assert 1991 not in anual.index
    assert 2025 not in anual.index


def test_el_promedio_anual_usa_los_12_meses_correctos():
    s = load_price_csv(SERIES["Arabica"])
    anual = to_crop_year(s)

    ventana = s.loc["1992-10-01":"1993-09-01"]
    assert len(ventana) == 12
    assert anual.loc[1992, "price_mean"] == pytest.approx(ventana.mean())
    assert anual.loc[1992, "price_min"] == pytest.approx(ventana.min())
    assert anual.loc[1992, "price_max"] == pytest.approx(ventana.max())


# ------------------------------------------------------------ formato final

def test_estructura_del_dataframe_de_precios(precios):
    assert len(precios) == 33 * 2
    assert set(precios.coffee_type) == {"Arabica", "Robusta"}
    assert precios.year.min() == 1992 and precios.year.max() == 2024
    assert precios.notna().all().all()


def test_el_rango_es_max_menos_min(precios):
    assert (precios.price_range == precios.price_max - precios.price_min).all()
    assert (precios.price_range >= 0).all()


def test_el_promedio_cae_dentro_del_rango(precios):
    assert (precios.price_mean >= precios.price_min).all()
    assert (precios.price_mean <= precios.price_max).all()


def test_arabica_siempre_cuesta_mas_que_robusta(precios):
    """Hecho conocido del mercado: sirve como validación de que no se cruzaron
    los archivos al cargarlos."""
    piv = precios.pivot(index="year", columns="coffee_type", values="price_mean")
    assert (piv["Arabica"] > piv["Robusta"]).all()


# --------------------------------------------------------------- integración

def test_traslape_con_el_consumo(precios):
    """Los dos primeros años del consumo no tienen precio y hay 5 años de
    precio posteriores al consumo. Ambas cosas son decisiones documentadas."""
    consumo, _ = load_clean()
    comun = overlap_with_consumption(precios, consumo)

    assert len(comun) == 28
    assert min(comun) == 1992, "1990 y 1991 no tienen precio disponible"
    assert max(comun) == 2019

    solo_precio = sorted(set(precios.year) - set(consumo.year))
    assert solo_precio == [2020, 2021, 2022, 2023, 2024], "el holdout genuino"
