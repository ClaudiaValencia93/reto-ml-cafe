"""
Carga y limpieza del dataset de consumo domestico de cafe.

Fuente: coffee_db.parquet (55 paises x 30 años cafeteros, 1990/91 - 2019/20).
Origen real de los datos: International Coffee Organization (ICO).

NOTA SOBRE UNIDADES
-------------------
El enunciado del reto dice que el consumo esta "en tazas". Es incorrecto.
Verificacion: Brasil 1990/91 aparece como 492,000,000. La ICO reporta para ese
anio 8,200 miles de sacos de 60 kg -> 8,200 * 60,000 = 492,000,000 exacto.
Ademas el 99.9% de los valores del dataset es divisible por 60.
Conclusion: la unidad es KILOGRAMOS, derivados de miles de sacos de 60 kg.
Se expone tambien en sacos de 60 kg, unidad estandar de la industria.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# DECISIONES DE LIMPIEZA
# Cada constante es un punto de decision documentado. Cambiar el valor aqui
# cambia el comportamiento de todo el pipeline.
# --------------------------------------------------------------------------

# Ceros AL INICIO de una serie (Timor-Leste, Guyana, Laos, Yemen).
# Evidencia: son bloques contiguos al comienzo, no ceros dispersos.
# Interpretacion: el pais empezo a reportar tarde, no consumia cero.
# Timor-Leste es el caso claro: se independizo en 2002.
#   "truncate" -> la serie empieza en el primer valor no-cero  [RECOMENDADO]
#   "nan"      -> se marcan como faltantes, se conservan las filas
#   "keep"     -> se dejan como cero (asume consumo real nulo)
LEADING_ZERO_POLICY = "truncate"

# Ceros AL FINAL. Solo aplica a Zambia: 19 anios de datos y luego 11 ceros
# seguidos. Es el caso peligroso: si se dejan, cualquier modelo pronostica
# consumo cero perpetuo.
#   "truncate_and_flag" -> corta la serie y marca forecastable=False  [RECOMENDADO]
#   "drop_country"      -> elimina el pais por completo
#   "keep"              -> asume que el consumo colapso de verdad
TRAILING_ZERO_POLICY = "truncate_and_flag"

# Series con muy pocos valores distintos en 30 anios (Malawi: 2, Congo: 2,
# Liberia: 2, Nigeria: 3...). No son consumo medido, son imputaciones con
# constantes. Un modelo naive las acierta perfecto e infla las metricas.
#   "flag"          -> marca is_low_variance, se excluyen del backtesting  [RECOMENDADO]
#   "flag_and_drop" -> marca y elimina del dataset
#   "keep"          -> participan en todo sin distincion
LOW_VARIANCE_POLICY = "flag"
LOW_VARIANCE_MAX_DISTINCT = 3  # <= este numero de valores distintos = sospechosa

# CALIDAD DEL DATO (medida, no filtro)
# -----------------------------------
# Un "año plano" es aquel en que el consumo es EXACTAMENTE igual al del año
# anterior. Un cero repetido en las diferencias no es una medición: es el
# último valor conocido repetido por la ICO cuando el país no reporta.
#
# La mediana del panel es 55% de años planos, y hay países con 89%. El criterio
# de LOW_VARIANCE_MAX_DISTINCT solo detecta los casos extremos (Kenia tiene 6
# valores distintos y pasa el filtro, pese a estar 23 años clavada en 50,000
# sacos).
#
# Esto se mide y se etiqueta, NO se usa para eliminar países. Razón: el análisis
# de negocio (tendencias, tamaño de mercado, segmentación) es válido para los 55
# países, mientras que las métricas de pronóstico solo son interpretables dentro
# de cada nivel de calidad. Mezclarlos invierte la conclusión: ver
# `evaluate.summarize_by_tier`.
FLAT_TIER_BOUNDS = (40, 70)  # (% plano) -> medido | parcial | repetido

# Total_domestic_consumption es la suma exacta de los 30 anios. Si sobrevive
# al formato largo es FUGA DE INFORMACION: le da al modelo el futuro.
DROP_TOTAL_COLUMN = True

KG_PER_BAG = 60  # saco estandar de la industria cafetera

DEFAULT_RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "coffee_db.parquet"


# --------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------

def load_raw(path=None):
    """Lee el parquet tal cual, sin transformar. Formato ancho."""
    return pd.read_parquet(Path(path) if path else DEFAULT_RAW_PATH)


def year_columns(df):
    """Columnas de anio cafetero ('1990/91', ...) en orden."""
    return [c for c in df.columns if "/" in c and c[0].isdigit()]


def to_long(df):
    """Formato ancho (una columna por anio) -> formato largo.

    Una fila = un pais en un anio. 55 x 30 = 1,650 filas.
    """
    years = year_columns(df)

    long = df.melt(
        id_vars=["Country", "Coffee type"],
        value_vars=years,
        var_name="crop_year",
        value_name="consumption_kg",
    )
    long = long.rename(columns={"Country": "country", "Coffee type": "coffee_type"})

    # El anio cafetero va de octubre a septiembre. '1990/91' -> 1990 como
    # entero para modelar; se conserva la etiqueta original para reportar.
    long["year"] = long["crop_year"].str[:4].astype(int)

    # El orden en 'Arabica/Robusta' indica cual predomina.
    long["primary_type"] = long["coffee_type"].str.split("/").str[0]
    long["is_blend"] = long["coffee_type"].str.contains("/")

    return long.sort_values(["country", "year"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Limpieza
# --------------------------------------------------------------------------

def flat_years_pct(y):
    """Porcentaje de años en que el consumo no cambió NADA respecto al anterior.

    Mide cuánto de la serie es medición y cuánto es arrastre administrativo.
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return 100.0
    return float(np.mean(np.diff(y) == 0) * 100)


def quality_tier(pct):
    """Etiqueta el nivel de calidad a partir del % de años planos."""
    bajo, alto = FLAT_TIER_BOUNDS
    if pct < bajo:
        return "medido"
    if pct < alto:
        return "parcial"
    return "repetido"


def _nonzero_bounds(s):
    """Indice del primer y ultimo valor no-cero. (None, None) si es todo ceros."""
    nonzero = s.to_numpy().nonzero()[0]
    if len(nonzero) == 0:
        return None, None
    return int(nonzero[0]), int(nonzero[-1])


def clean(long):
    """Aplica las politicas declaradas arriba. Devuelve (df, reporte)."""
    report = {
        "dropped_all_zero": [],
        "truncated_leading": {},
        "truncated_trailing": {},
        "low_variance": [],
        "not_forecastable": [],
    }

    pieces = []
    for country, g in long.groupby("country", sort=False):
        g = g.sort_values("year").reset_index(drop=True)
        g["forecastable"] = True
        g["is_low_variance"] = False

        first, last = _nonzero_bounds(g["consumption_kg"])

        # Serie completamente en cero: no aporta informacion.
        if first is None:
            report["dropped_all_zero"].append(country)
            continue

        n_leading = first
        n_trailing = len(g) - 1 - last

        # --- ceros al final (se resuelve antes de cortar el inicio, para que
        #     los indices 'first'/'last' sigan siendo validos) ---
        if n_trailing:
            if TRAILING_ZERO_POLICY == "drop_country":
                report["not_forecastable"].append(country)
                continue
            if TRAILING_ZERO_POLICY == "truncate_and_flag":
                g = g.iloc[: last + 1]
                g["forecastable"] = False
                report["truncated_trailing"][country] = n_trailing
                report["not_forecastable"].append(country)

        # --- ceros al inicio ---
        if n_leading:
            if LEADING_ZERO_POLICY == "truncate":
                g = g.iloc[first:]
                report["truncated_leading"][country] = n_leading
            elif LEADING_ZERO_POLICY == "nan":
                g.loc[g.index[:first], "consumption_kg"] = pd.NA
                report["truncated_leading"][country] = n_leading

        # --- series planas: imputadas con constantes, no medidas ---
        if g["consumption_kg"].nunique(dropna=True) <= LOW_VARIANCE_MAX_DISTINCT:
            if LOW_VARIANCE_POLICY == "flag_and_drop":
                report["low_variance"].append(country)
                continue
            if LOW_VARIANCE_POLICY == "flag":
                g["is_low_variance"] = True
                report["low_variance"].append(country)

        # Calidad del dato: se mide sobre la serie YA truncada, que es la que
        # se va a modelar. Es atributo, no filtro.
        pct = flat_years_pct(g["consumption_kg"])
        g["flat_years_pct"] = pct
        g["data_quality"] = quality_tier(pct)

        pieces.append(g)

    out = pd.concat(pieces, ignore_index=True)
    out["consumption_bags_60kg"] = out["consumption_kg"] / KG_PER_BAG
    return out, report


def load_clean(path=None):
    """Atajo: parquet -> formato largo -> limpio."""
    return clean(to_long(load_raw(path)))


if __name__ == "__main__":
    df, rep = load_clean()
    print(f"Filas: {len(df):,} | Paises: {df.country.nunique()} | "
          f"Anios: {df.year.min()}-{df.year.max()}")
    print()
    print(f"Eliminados (todo en cero): {rep['dropped_all_zero']}")
    print(f"Truncados al inicio:       {rep['truncated_leading']}")
    print(f"Truncados al final:        {rep['truncated_trailing']}")
    print(f"No pronosticables:         {rep['not_forecastable']}")
    print()
    calidad = df.groupby("data_quality").country.nunique()
    print("Calidad del dato (% de años sin cambio alguno):")
    for nivel in ("medido", "parcial", "repetido"):
        if nivel in calidad.index:
            print(f"  {nivel:12} {calidad[nivel]:2} países")
    print(f"  mediana de años planos en el panel: "
          f"{df.groupby('country').flat_years_pct.first().median():.1f}%")
    print(f"Varianza baja ({len(rep['low_variance'])}):         {rep['low_variance']}")
