"""
Carga de precios internacionales del café.

POR QUÉ EXISTE ESTE MÓDULO
--------------------------
El enunciado del reto pide "rangos de precios futuros", pero `coffee_db.parquet`
NO contiene ninguna variable de precio: sus 30 columnas de año son consumo en
kilogramos. La brecha no se puede cerrar con el dato entregado.

Se incorpora una fuente externa, documentada y reproducible:

  PCOFFOTMUSDM  Global price of Coffee, Other Mild Arabica
  PCOFFROBUSDM  Global price of Coffee, Robustas

Ambas son del Fondo Monetario Internacional (programa Primary Commodity Prices),
distribuidas por FRED (Federal Reserve Bank of St. Louis). Son precios de
referencia del mercado global, en centavos de dólar por libra, mensuales.

Descarga (sin API key):
  https://fred.stlouisfed.org/graph/fredgraph.csv?id=PCOFFOTMUSDM
  https://fred.stlouisfed.org/graph/fredgraph.csv?id=PCOFFROBUSDM

Se eligió el FMI sobre la ICO porque la serie del FMI es descargable de forma
programática y sin registro, lo que mantiene el pipeline reproducible por
terceros.

COBERTURA
---------
Enero 1992 - julio 2026, sin valores faltantes. Esto implica dos cosas:

1. El traslape con el consumo (1990/91-2019/20) es de 28 años cafeteros
   (1992/93-2019/20). Los dos primeros años del dataset de consumo quedan
   sin precio: no hay forma de rellenarlos sin inventar.

2. Hay 5 años cafeteros completos POSTERIORES al fin del consumo
   (2020/21-2024/25). Eso permite un holdout genuino: entrenar con datos
   hasta 2019/20 y validar contra realidad nunca vista, incluyendo el shock
   de precios de 2021-2024.
"""

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# DECISIONES
# --------------------------------------------------------------------------

# El año cafetero de la ICO va de octubre a septiembre, y así está construido
# el dataset de consumo ('1990/91'). Los precios vienen en año calendario, así
# que se reagrupan para que ambas fuentes hablen del mismo periodo. Usar el año
# calendario sería un desalineamiento de tres meses en cada observación.
CROP_YEAR_START_MONTH = 10

# Solo se conservan años cafeteros con los 12 meses presentes. Sin esto, 1991
# (9 meses) y 2025 (10 meses) entrarían con un promedio calculado sobre un
# subconjunto distinto de meses, sesgado por la estacionalidad del mercado.
REQUIRE_COMPLETE_YEAR = True

SERIES = {
    "Arabica": "price_arabica_PCOFFOTMUSDM.csv",
    "Robusta": "price_robusta_PCOFFROBUSDM.csv",
}

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


# --------------------------------------------------------------------------

def load_price_csv(filename, raw_dir=None):
    """Lee un CSV de FRED. Devuelve una Serie mensual indexada por fecha."""
    path = Path(raw_dir or DEFAULT_RAW_DIR) / filename
    df = pd.read_csv(path, parse_dates=["observation_date"])
    value_col = [c for c in df.columns if c != "observation_date"][0]
    return df.set_index("observation_date")[value_col].astype(float)


def to_crop_year(monthly):
    """Agrega una serie mensual a año cafetero (octubre-septiembre).

    Devuelve media, mínimo, máximo y desviación dentro de cada año. El rango
    intranual no es un subproducto: es la medida directa de cuánto se mueve el
    precio dentro de una misma cosecha, que es justamente lo que un exportador
    necesita saber para cubrirse.
    """
    year = monthly.index.year + (monthly.index.month >= CROP_YEAR_START_MONTH).astype(int) - 1
    g = monthly.groupby(year)

    out = pd.DataFrame({
        "price_mean": g.mean(),
        "price_min": g.min(),
        "price_max": g.max(),
        "price_std": g.std(),
        "n_months": g.size(),
    })
    out.index.name = "year"

    if REQUIRE_COMPLETE_YEAR:
        out = out[out.n_months == 12]

    out["price_range"] = out.price_max - out.price_min
    return out.drop(columns="n_months")


def load_prices(raw_dir=None):
    """Precios anuales por tipo de café, en formato largo.

    Columnas: year, coffee_type, price_mean, price_min, price_max,
            price_std, price_range   (centavos USD por libra)
    """
    piezas = []
    for tipo, filename in SERIES.items():
        anual = to_crop_year(load_price_csv(filename, raw_dir))
        anual["coffee_type"] = tipo
        piezas.append(anual.reset_index())

    return pd.concat(piezas, ignore_index=True).sort_values(
        ["coffee_type", "year"]
    ).reset_index(drop=True)


def overlap_with_consumption(prices, consumption):
    """Años en que ambas fuentes tienen dato. Hace explícito qué se puede cruzar."""
    return sorted(set(prices.year) & set(consumption.year))


if __name__ == "__main__":
    p = load_prices()
    print(f"Filas: {len(p)} | Años: {p.year.min()}-{p.year.max()} | Tipos: {list(p.coffee_type.unique())}\n")

    for tipo, g in p.groupby("coffee_type"):
        print(f"--- {tipo} (centavos USD/libra) ---")
        print(g.set_index("year")[["price_mean", "price_min", "price_max", "price_range"]]
            .tail(6).round(1).to_string())
        print()

