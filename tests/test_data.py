"""Pruebas del módulo de carga y limpieza.

El objetivo no es cubrir líneas, es **verificar que cada decisión documentada en
`data.py` realmente se cumple**. Si alguien cambia una política sin pensarlo, estas
pruebas fallan y avisan.

Los casos borde están elegidos a propósito:
  - Nepal / Guinea Ecuatorial -> series 100% en cero
  - Zambia                    -> dejó de reportar (ceros al final)
  - Timor-Leste               -> empezó a reportar tarde (ceros al inicio)
  - Malawi                    -> serie imputada con constantes
"""

import pandas as pd
import pytest

from coffee.data import (
    KG_PER_BAG,
    LOW_VARIANCE_MAX_DISTINCT,
    clean,
    load_clean,
    load_raw,
    to_long,
    year_columns,
)


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def limpio():
    return load_clean()


# ---------------------------------------------------------------- carga cruda

def test_dimensiones_del_crudo(raw):
    assert raw.shape == (55, 33)
    assert raw.Country.nunique() == 55


def test_no_hay_nulos(raw):
    assert raw.isna().sum().sum() == 0


def test_hay_30_columnas_de_anio(raw):
    years = year_columns(raw)
    assert len(years) == 30
    assert years[0] == "1990/91"
    assert years[-1] == "2019/20"


def test_total_es_la_suma_de_los_anios(raw):
    """Justifica descartar la columna: es redundante y sería fuga de información."""
    suma = raw[year_columns(raw)].sum(axis=1)
    assert (suma == raw["Total_domestic_consumption"]).all()


def test_unidades_son_kilogramos(raw):
    """Brasil 1990/91 = 8,200 miles de sacos de 60 kg según la ICO.

    Si esta prueba falla, la interpretación de unidades del README es incorrecta.
    """
    brasil = raw.loc[raw.Country == "Brazil", "1990/91"].iloc[0]
    assert brasil == 8_200 * 60_000


# ------------------------------------------------------------- formato largo

def test_formato_largo_conserva_todas_las_observaciones(raw):
    long = to_long(raw)
    assert len(long) == 55 * 30


def test_no_arrastra_la_columna_total(raw):
    """Protege contra la fuga de información."""
    long = to_long(raw)
    assert "Total_domestic_consumption" not in long.columns


def test_el_anio_cafetero_se_parsea_al_anio_de_inicio(raw):
    long = to_long(raw)
    fila = long[long.crop_year == "1990/91"].iloc[0]
    assert fila.year == 1990
    assert long.year.min() == 1990 and long.year.max() == 2019


def test_tipo_primario_toma_el_dominante(raw):
    long = to_long(raw)
    mezclas = long[long.coffee_type == "Arabica/Robusta"]
    assert (mezclas.primary_type == "Arabica").all()
    assert mezclas.is_blend.all()

    puros = long[long.coffee_type == "Arabica"]
    assert (puros.primary_type == "Arabica").all()
    assert not puros.is_blend.any()


# ------------------------------------------------------------------ limpieza

def test_elimina_paises_completamente_en_cero(limpio):
    df, rep = limpio
    assert set(rep["dropped_all_zero"]) == {"Nepal", "Equatorial Guinea"}
    assert "Nepal" not in set(df.country)
    assert "Equatorial Guinea" not in set(df.country)


def test_zambia_se_trunca_y_se_marca_no_pronosticable(limpio):
    """El caso peligroso: 11 ceros al final tras 19 años de datos reales."""
    df, rep = limpio
    zambia = df[df.country == "Zambia"]

    assert rep["truncated_trailing"]["Zambia"] == 11
    assert zambia.year.max() == 2008, "la serie debe cortarse en el último dato real"
    assert not zambia.forecastable.any(), "no debe entrar al pronóstico"
    assert (zambia.consumption_kg > 0).all()


def test_ceros_iniciales_se_truncan(limpio):
    """Timor-Leste no existía como reportante antes de 2002."""
    df, rep = limpio
    assert rep["truncated_leading"]["Timor-Leste"] == 20

    timor = df[df.country == "Timor-Leste"]
    assert timor.year.min() == 2010
    assert len(timor) == 10


def test_no_queda_ningun_cero_en_el_dataset_limpio(limpio):
    """Con la política 'truncate' no debe sobrevivir ningún cero."""
    df, _ = limpio
    assert (df.consumption_kg > 0).all()


def test_series_planas_quedan_marcadas(limpio):
    df, rep = limpio
    assert "Malawi" in rep["low_variance"]

    malawi = df[df.country == "Malawi"]
    assert malawi.is_low_variance.all()
    assert malawi.consumption_kg.nunique() <= LOW_VARIANCE_MAX_DISTINCT


def test_los_paises_marcados_coinciden_con_el_reporte(limpio):
    df, rep = limpio
    marcados = set(df.loc[df.is_low_variance, "country"].unique())
    assert marcados == set(rep["low_variance"])


def test_calidad_del_dato_se_etiqueta(limpio):
    df, _ = limpio
    assert {"flat_years_pct", "data_quality"} <= set(df.columns)
    assert set(df.data_quality.unique()) <= {"medido", "parcial", "repetido"}
    # una etiqueta por país, constante dentro de la serie
    assert (df.groupby("country").data_quality.nunique() == 1).all()


def test_kenia_queda_marcada_como_arrastrada(limpio):
    """Kenia pasa el filtro de valores distintos (tiene 6) pero está 23 años
    clavada en 50,000 sacos. Es el caso que motivó esta métrica."""
    df, _ = limpio
    kenya = df[df.country == "Kenya"]
    assert not kenya.is_low_variance.any(), "el filtro viejo NO la detecta"
    assert kenya.data_quality.iloc[0] == "repetido", "el nuevo criterio sí"
    assert kenya.flat_years_pct.iloc[0] > 80


def test_el_panel_esta_mayoritariamente_arrastrado(limpio):
    """Documenta el hallazgo central sobre la calidad del dataset: más de la
    mitad de las variaciones año a año son exactamente cero."""
    df, _ = limpio
    por_pais = df.groupby("country").flat_years_pct.first()
    assert por_pais.median() > 50


def test_conversion_a_sacos(limpio):
    df, _ = limpio
    assert (df.consumption_bags_60kg * KG_PER_BAG == df.consumption_kg).all()


def test_cada_pais_tiene_una_serie_anual_sin_huecos(limpio):
    """Un salto de año rompería cualquier modelo de series de tiempo."""
    df, _ = limpio
    for pais, g in df.groupby("country"):
        anios = g.year.sort_values().to_numpy()
        assert (anios[1:] - anios[:-1] == 1).all(), f"{pais} tiene huecos"
        assert not g.year.duplicated().any(), f"{pais} tiene años repetidos"


def test_el_numero_de_paises_aptos_es_el_esperado(limpio):
    """Documenta el resultado del pipeline: 43 países modelables de 55."""
    df, _ = limpio
    aptos = df[df.forecastable & ~df.is_low_variance].country.nunique()
    assert df.country.nunique() == 53
    assert aptos == 43


# ------------------------------------------------- las políticas son efectivas

def test_politica_keep_conserva_los_ceros(raw, monkeypatch):
    """Cambiar la política debe cambiar el resultado; si no, la constante es decorativa."""
    import coffee.data as data

    monkeypatch.setattr(data, "LEADING_ZERO_POLICY", "keep")
    monkeypatch.setattr(data, "TRAILING_ZERO_POLICY", "keep")

    df, rep = data.clean(to_long(raw))

    assert (df.consumption_kg == 0).any(), "con 'keep' los ceros deben sobrevivir"
    assert rep["truncated_leading"] == {}
    assert rep["truncated_trailing"] == {}


def test_politica_flag_and_drop_elimina_las_series_planas(raw, monkeypatch):
    import coffee.data as data

    monkeypatch.setattr(data, "LOW_VARIANCE_POLICY", "flag_and_drop")
    df, rep = data.clean(to_long(raw))

    assert "Malawi" not in set(df.country)
    assert not df.is_low_variance.any()
