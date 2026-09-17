"""
Pronóstico de precios internacionales del café, con intervalos.

QUÉ RESPONDE
------------
El requisito del enunciado: "rangos de precios futuros". Un intervalo de
predicción **es** un rango, y es la única forma honesta de responder esa
pregunta: entregar un número puntual para 2028 sería falsa precisión.

POR QUÉ FRECUENCIA MENSUAL Y NO ANUAL
-------------------------------------
`prices.py` agrega a año cafetero para poder cruzar con el consumo. Pero el
pronóstico de precios NO tiene que empatar con el consumo país por país: son dos
problemas independientes (el precio del café es global, no hay un precio por
país). Liberados de ese requisito, se modela sobre la serie mensual:

    mensual: 415 observaciones por tipo
    anual:    33 observaciones por tipo

Doce veces más datos para estimar los mismos parámetros.

POR QUÉ EN LOGARITMOS
---------------------
Dos razones. Los precios no pueden ser negativos, y un intervalo en niveles
puede cruzar el cero en horizontes largos. Y la volatilidad del café es
proporcional al nivel: un movimiento de 20 centavos no significa lo mismo a 70
que a 360. En logaritmos el error es multiplicativo, que es como se comporta el
mercado. Los límites se reexponencian al final; como exp() es monótona, los
cuantiles se preservan.

EL EXPERIMENTO CENTRAL
----------------------
El dataset de consumo termina en 2019/20, pero los precios llegan a julio de
2026. Eso habilita una validación poco frecuente: entrenar hasta septiembre de
2019, pronosticar los ~82 meses siguientes y **comparar contra la realidad**,
que incluye el shock de precios de 2021-2024.

No es backtesting simulado. Es validación contra datos que el modelo nunca vio y
que no son una partición artificial: son el futuro que efectivamente ocurrió.

La métrica que importa aquí no es el MAPE del punto central, es la **cobertura
del intervalo**: de todos los meses pronosticados, ¿qué fracción cayó realmente
dentro de la banda prometida? Un intervalo al 95% que solo contiene el 40% de la
realidad no es un rango, es una ilusión.
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# DECISIONES
# --------------------------------------------------------------------------

# Corte del holdout: fin del año cafetero 2018/19, que es donde termina el
# dataset de consumo entregado. Todo lo posterior es realidad no vista.
HOLDOUT_CUTOFF = "2019-09-01"

# Niveles de confianza reportados. El 80% es el que se usa para planear; el 95%
# para dimensionar el peor caso razonable.
LEVELS = (0.80, 0.95)

# Rejilla ARIMA. d=1 fijo: las series de precios son claramente no estacionarias
# en nivel y estacionarias en diferencias, que es el comportamiento de un activo.
ARIMA_GRID = [(p, 1, q) for p in range(3) for q in range(3)]

# Transformación logarítmica (ver docstring).
LOG_TRANSFORM = True


# --------------------------------------------------------------------------

def _transformar(y):
    return np.log(y) if LOG_TRANSFORM else np.asarray(y, float)


def _destransformar(v):
    return np.exp(v) if LOG_TRANSFORM else v


def select_order(y, grid=None):
    """Elige el orden ARIMA por AIC. Devuelve (orden, aic)."""
    from statsmodels.tsa.arima.model import ARIMA

    z = _transformar(y)
    mejor, mejor_aic = None, np.inf
    for orden in (grid or ARIMA_GRID):
        try:
            aic = ARIMA(z, order=orden).fit().aic
            if np.isfinite(aic) and aic < mejor_aic:
                mejor, mejor_aic = orden, aic
        except Exception:
            continue
    return mejor or (0, 1, 0), mejor_aic


def forecast_with_intervals(y, h, index=None, order=None, levels=LEVELS):
    """Pronostica h pasos con intervalos de predicción.

    Devuelve un DataFrame con el pronóstico central y un par de columnas
    (lo_XX, hi_XX) por cada nivel de confianza pedido.
    """
    from statsmodels.tsa.arima.model import ARIMA

    y = np.asarray(y, dtype=float)
    if order is None:
        order, _ = select_order(y)

    ajuste = ARIMA(_transformar(y), order=order).fit()
    resultado = ajuste.get_forecast(steps=h)

    salida = pd.DataFrame({"forecast": _destransformar(resultado.predicted_mean)})
    for nivel in levels:
        ci = resultado.conf_int(alpha=1 - nivel)
        ci = np.asarray(ci, dtype=float)
        etiqueta = int(round(nivel * 100))
        salida[f"lo_{etiqueta}"] = _destransformar(ci[:, 0])
        salida[f"hi_{etiqueta}"] = _destransformar(ci[:, 1])

    if index is not None:
        salida.index = index
    salida.attrs["order"] = order
    return salida


def future_index(last_date, h):
    """Índice mensual de h periodos a partir del mes siguiente a `last_date`."""
    inicio = pd.Timestamp(last_date) + pd.DateOffset(months=1)
    return pd.date_range(inicio, periods=h, freq="MS")


# --------------------------------------------------------------------------
# Validación contra la realidad
# --------------------------------------------------------------------------

def coverage(actual, lo, hi):
    """Fracción de observaciones reales que cayeron dentro del intervalo (%)."""
    actual, lo, hi = (np.asarray(v, float) for v in (actual, lo, hi))
    return float(np.mean((actual >= lo) & (actual <= hi)) * 100)


def holdout_evaluation(monthly, cutoff=HOLDOUT_CUTOFF, levels=LEVELS):
    """Entrena hasta `cutoff` y compara contra lo que realmente pasó.

    Devuelve (tabla_mensual, resumen).
    """
    cutoff = pd.Timestamp(cutoff)
    train = monthly.loc[:cutoff]
    test = monthly.loc[cutoff + pd.DateOffset(months=1):]

    if len(test) == 0:
        raise ValueError("no hay datos posteriores al corte")

    pred = forecast_with_intervals(
        train.to_numpy(float), len(test),
        index=test.index, levels=levels,
    )
    tabla = pred.assign(actual=test.to_numpy(float))
    tabla["error_pct"] = (tabla.actual - tabla.forecast) / tabla.actual * 100

    resumen = {
        "order": pred.attrs["order"],
        "n_train": len(train),
        "n_test": len(test),
        "mape": float(np.mean(np.abs(tabla.error_pct))),
        "sesgo_pct": float(np.mean(tabla.error_pct)),
    }
    for nivel in levels:
        e = int(round(nivel * 100))
        resumen[f"cobertura_{e}"] = coverage(tabla.actual, tabla[f"lo_{e}"], tabla[f"hi_{e}"])

    return tabla, resumen


def forecast_all(prices_raw_loader, horizon_years=3, cutoff=HOLDOUT_CUTOFF):
    """Pipeline completo para los dos tipos de café.

    Devuelve (holdout, futuro, resumenes):
      holdout   - pronóstico 2019-2026 contra realidad, por tipo
      futuro    - pronóstico final reentrenado con TODO el histórico
      resumenes - métricas del holdout por tipo
    """
    holdouts, futuros, resumenes = [], [], []

    for tipo, mensual in prices_raw_loader.items():
        tabla, resumen = holdout_evaluation(mensual, cutoff)
        tabla["coffee_type"] = tipo
        # rename_axis y no rename(columns=...): el indice hereda el nombre de la
        # fuente ("observation_date"), asi que renombrar "index" no aplicaria.
        holdouts.append(tabla.rename_axis("date").reset_index())
        resumenes.append({"coffee_type": tipo, **resumen})

        # Reentrenar con todo el histórico para el pronóstico que se entrega
        h = horizon_years * 12
        fut = forecast_with_intervals(
            mensual.to_numpy(float), h,
            index=future_index(mensual.index[-1], h),
        )
        fut["coffee_type"] = tipo
        futuros.append(fut.rename_axis("date").reset_index())

    return (
        pd.concat(holdouts, ignore_index=True),
        pd.concat(futuros, ignore_index=True),
        pd.DataFrame(resumenes),
    )


if __name__ == "__main__":
    from coffee.prices import SERIES, load_price_csv

    pd.set_option("display.width", 200)
    series = {t: load_price_csv(f) for t, f in SERIES.items()}

    print("Entrenando hasta", HOLDOUT_CUTOFF, "y validando contra la realidad...\n")
    holdout, futuro, resumen = forecast_all(series)

    print("=" * 78)
    print("VALIDACION CONTRA LA REALIDAD 2019-2026 (datos nunca vistos)")
    print("=" * 78)
    print(resumen.round(1).to_string(index=False))

    print("\n" + "=" * 78)
    print("PRONOSTICO FINAL, reentrenado con todo (centavos USD/libra)")
    print("=" * 78)
    for tipo, g in futuro.groupby("coffee_type"):
        anual = g.set_index("date").resample("YS").mean(numeric_only=True)
        print(f"\n--- {tipo} ---")
        print(anual[["lo_95", "lo_80", "forecast", "hi_80", "hi_95"]].round(0).to_string())
