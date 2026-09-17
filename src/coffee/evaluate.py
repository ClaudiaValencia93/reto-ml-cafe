"""
Backtesting con origen móvil (rolling origin).

POR QUÉ NO SE USA train_test_split
-----------------------------------
Partir aleatoriamente una serie de tiempo entrena con el futuro para predecir el
pasado. El modelo aprende de datos que en producción no existirían y las métricas
salen infladas. Es el error más frecuente al aplicar ML a series temporales.

CÓMO FUNCIONA EL ORIGEN MÓVIL
-----------------------------
Se elige un año de corte, se entrena solo con lo anterior y se predicen los años
siguientes. Luego el corte avanza un año y se repite:

    corte 2004:  entrena 1990-2004  ->  predice 2005, 2006, 2007
    corte 2005:  entrena 1990-2005  ->  predice 2006, 2007, 2008
    corte 2006:  entrena 1990-2006  ->  predice 2007, 2008, 2009
    ...

Cada evaluación respeta la flecha del tiempo, y en vez de un número se obtiene
una distribución de errores sobre múltiples orígenes. Con series de 30 puntos eso
importa: un solo corte puede caer en un año atípico y no decir nada.

Los cortes se definen por AÑO CALENDARIO, no por índice de cada serie. Es lo que
permite que el modelo global se entrene con un corte temporal común a todos los
países, sin filtrarse información del futuro de un país al pasado de otro.

MÉTRICA PRINCIPAL: MASE
-----------------------
    MASE = MAE(pronóstico) / MAE(naive de un paso en el entrenamiento)

Se usa MASE y no MAPE porque el consumo va de 1.2e5 (Burundi) a 1.3e9 (Brasil).
El MAPE castiga desproporcionadamente los valores pequeños y no permite promediar
entre países. MASE es adimensional y tiene lectura directa:

    MASE < 1  ->  mejor que el naive
    MASE = 1  ->  igual que el naive
    MASE > 1  ->  peor que el naive (el modelo estorba)

Se reporta MAPE igual porque es el lenguaje del negocio, pero las decisiones se
toman con MASE.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from coffee.models import LOCAL_MODELS, GlobalLGBM

DEFAULT_RESULTS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "backtest_consumo.parquet"
)

# --------------------------------------------------------------------------
# DECISIONES DE EVALUACIÓN
# --------------------------------------------------------------------------

# Horizonte de pronóstico. Tres años es lo que una exportadora usa para planear
# contratos y capacidad. Más allá, con 30 puntos de historia, la incertidumbre
# hace que el número deje de ser accionable.
HORIZON = 3

# Mínimo de años de entrenamiento antes del primer corte. Por debajo de 15, ETS
# y ARIMA no tienen con qué estimar sus parámetros y degradan a drift, con lo
# cual la comparación dejaría de ser informativa.
MIN_TRAIN_YEARS = 15

# Ventana expansiva (el entrenamiento crece en cada corte) en vez de deslizante
# (tamaño fijo). Con series tan cortas, descartar los años viejos es un lujo que
# no se puede pagar.
EXPANDING_WINDOW = True

# Los países marcados en la limpieza quedan fuera: las series imputadas con
# constantes las acierta el naive perfectamente e inflarían todas las métricas.
EXCLUDE_LOW_VARIANCE = True
EXCLUDE_NOT_FORECASTABLE = True


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------

def mae(actual, pred):
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(pred))))


def mape(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def smape(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    denom = (np.abs(actual) + np.abs(pred)) / 2
    return float(np.mean(np.abs(actual - pred) / denom) * 100)


def naive_scale(y_train):
    """MAE del naive de un paso dentro del entrenamiento. Denominador del MASE."""
    y = np.asarray(y_train, float)
    if len(y) < 2:
        return np.nan
    escala = np.mean(np.abs(np.diff(y)))
    return escala if escala > 0 else np.nan


def mase(actual, pred, y_train):
    escala = naive_scale(y_train)
    if not np.isfinite(escala):
        return np.nan
    return mae(actual, pred) / escala


# --------------------------------------------------------------------------
# Backtesting
# --------------------------------------------------------------------------

def elegible(panel):
    """Países que entran a la evaluación, según las políticas declaradas."""
    mask = pd.Series(True, index=panel.index)
    if EXCLUDE_NOT_FORECASTABLE:
        mask &= panel.forecastable
    if EXCLUDE_LOW_VARIANCE:
        mask &= ~panel.is_low_variance
    return panel[mask]


def cut_years(panel, horizon=HORIZON, min_train=MIN_TRAIN_YEARS):
    """Años de corte válidos: hay suficiente historia antes y horizonte después."""
    anios = sorted(panel.year.unique())
    primero, ultimo = anios[0], anios[-1]
    return [
        y for y in anios
        if (y - primero + 1) >= min_train and (y + horizon) <= ultimo
    ]


def rolling_origin(panel, horizon=HORIZON, min_train=MIN_TRAIN_YEARS,
                   models=None, include_global=True, verbose=True):
    """Evalúa todos los modelos sobre todos los cortes y países.

    Devuelve un DataFrame con una fila por (corte, país, modelo), con las
    métricas del horizonte completo y el error de cada paso por separado.
    """
    panel = elegible(panel)
    models = models or LOCAL_MODELS
    cortes = cut_years(panel, horizon, min_train)

    if verbose:
        print(f"Cortes: {len(cortes)} ({cortes[0]}-{cortes[-1]}) | "
              f"Países: {panel.country.nunique()} | Horizonte: {horizon} años")

    filas = []
    for corte in cortes:
        train_panel = panel[panel.year <= corte]

        modelo_global = None
        if include_global:
            try:
                modelo_global = GlobalLGBM().fit(train_panel)
            except Exception as e:
                if verbose:
                    print(f"  corte {corte}: el modelo global falló ({e})")

        for pais, g in panel.groupby("country", sort=False):
            g = g.sort_values("year")
            train = g[g.year <= corte]
            test = g[(g.year > corte) & (g.year <= corte + horizon)]

            if len(train) < min_train or len(test) < horizon:
                continue

            y_train = train.consumption_kg.to_numpy(float)
            y_test = test.consumption_kg.to_numpy(float)
            tipo = g.primary_type.iloc[0]

            candidatos = {n: f(y_train, horizon) for n, f in models.items()}
            if modelo_global is not None:
                candidatos["global_lgbm"] = modelo_global.predict(y_train, horizon, tipo)

            for nombre, pred in candidatos.items():
                pred = np.asarray(pred, float)
                fila = {
                    "cut_year": corte,
                    "country": pais,
                    "model": nombre,
                    "data_quality": g.data_quality.iloc[0],
                    "n_train": len(y_train),
                    "mase": mase(y_test, pred, y_train),
                    "mape": mape(y_test, pred),
                    "smape": smape(y_test, pred),
                }
                for paso in range(horizon):
                    fila[f"ape_h{paso + 1}"] = abs(
                        (y_test[paso] - pred[paso]) / y_test[paso]
                    ) * 100
                filas.append(fila)

    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# Resumen
# --------------------------------------------------------------------------

def summarize(resultados):
    """Tabla comparativa por modelo, ordenada por MASE mediano.

    Se reporta la MEDIANA además de la media porque unos pocos países con errores
    enormes dominarían el promedio y esconderían el comportamiento típico.
    """
    g = resultados.groupby("model")
    tabla = pd.DataFrame({
        "mase_mediana": g.mase.median(),
        "mase_media": g.mase.mean(),
        "mape_mediana": g.mape.median(),
        "smape_mediana": g.smape.median(),
        "pct_gana_a_naive": np.nan,
        "n": g.size(),
    })

    # Porcentaje de casos en que cada modelo le gana al naive en el MISMO
    # corte y país. Es una comparación pareada, mucho más informativa que
    # comparar promedios de distribuciones distintas.
    base = resultados[resultados.model == "naive"].set_index(["cut_year", "country"]).mase
    for nombre, g_m in resultados.groupby("model"):
        emparejado = g_m.set_index(["cut_year", "country"]).mase
        comun = emparejado.index.intersection(base.index)
        comparables = (emparejado.loc[comun] < base.loc[comun]).mean() * 100
        tabla.loc[nombre, "pct_gana_a_naive"] = comparables

    return tabla.sort_values("mase_mediana").round(3)


def by_horizon(resultados):
    """Error porcentual medio por paso del horizonte: ¿cómo se degrada cada modelo?"""
    cols = [c for c in resultados.columns if c.startswith("ape_h")]
    return resultados.groupby("model")[cols].median().round(2)


TIER_ORDER = ["medido", "mixto", "arrastrado"]


def attach_quality(resultados, panel):
    """Añade la etiqueta de calidad a resultados que no la traigan.

    Permite reanalizar backtestings guardados antes de que existiera la columna,
    sin volver a correrlos.
    """
    if "data_quality" in resultados.columns:
        return resultados
    mapa = panel.groupby("country").data_quality.first()
    return resultados.assign(data_quality=resultados.country.map(mapa))


def summarize_by_tier(resultados, panel=None):
    """Ranking de modelos DENTRO de cada nivel de calidad del dato.

    Esta es la tabla que importa, y la razón es una paradoja de composición.

    Agregando los 43 países, el naive parece ganar. Pero en los países cuyo
    consumo es un valor arrastrado por la ICO, el naive acierta con error
    EXACTAMENTE cero, porque la serie es constante por construcción. Esos ceros
    arrastran la mediana global y esconden que, en los países con medición real,
    los modelos con tendencia le ganan al naive con holgura.

    La conclusión agregada es la opuesta a la conclusión en cada subgrupo. Por
    eso el reporte se hace por nivel y no en conjunto.
    """
    if panel is not None:
        resultados = attach_quality(resultados, panel)

    salida = {}
    for tier, sub in resultados.groupby("data_quality"):
        piv = sub.pivot_table(index=["cut_year", "country"], columns="model", values="mase")
        tabla = sub.groupby("model").agg(
            mase_mediana=("mase", "median"),
            mape_mediana=("mape", "median"),
        )
        for m in tabla.index:
            if m == "naive" or "naive" not in piv.columns:
                continue
            d = piv[m] - piv["naive"]
            no_empate = d[d.abs() > 1e-9]
            tabla.loc[m, "gana_a_naive_%"] = (
                float((no_empate < 0).mean() * 100) if len(no_empate) else np.nan
            )
            tabla.loc[m, "empates_%"] = float((d.abs() <= 1e-9).mean() * 100)

        tabla["n_paises"] = sub.country.nunique()
        salida[tier] = tabla.sort_values("mase_mediana").round(2)

    return {t: salida[t] for t in TIER_ORDER if t in salida}


def best_model_by_tier(resultados, panel=None):
    """Modelo recomendado para cada nivel de calidad, según MASE mediano."""
    return {t: tabla.index[0] for t, tabla in summarize_by_tier(resultados, panel).items()}


def vs_naive(resultados, modelo="arima"):
    """Comparación pareada contra el naive, separando empates.

    Necesario porque ARIMA elige a menudo el orden (0,1,0), que es una caminata
    aleatoria: su pronóstico es idéntico al del naive. Sin separar los empates,
    el porcentaje de victorias se lee mal.
    """
    piv = resultados.pivot_table(
        index=["cut_year", "country"], columns="model", values="mase"
    )
    d = piv[modelo] - piv["naive"]
    tol = 1e-9
    return {
        "gana": float((d < -tol).mean() * 100),
        "empata": float((d.abs() <= tol).mean() * 100),
        "pierde": float((d > tol).mean() * 100),
        "n": int(len(d)),
    }


# --------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    from coffee.data import load_clean

    pd.set_option("display.width", 200)

    print("Cargando datos...")
    panel, _ = load_clean()

    print("Corriendo backtesting (toma 1-2 minutos)...\n")
    t0 = time.time()
    resultados = rolling_origin(panel)
    print(f"\nListo en {time.time() - t0:.0f}s | {len(resultados):,} evaluaciones\n")

    salida = DEFAULT_RESULTS_PATH
    salida.parent.mkdir(parents=True, exist_ok=True)
    resultados.to_parquet(salida, index=False)

    print("=" * 78)
    print("RANKING DE MODELOS (ordenado por MASE mediano; < 1 = mejor que naive)")
    print("=" * 78)
    print(summarize(resultados).to_string())

    print("\n" + "=" * 78)
    print("DEGRADACION POR HORIZONTE (MAPE mediano, %)")
    print("=" * 78)
    print(by_horizon(resultados).to_string())

    print("\n" + "=" * 78)
    print("ARIMA CONTRA NAIVE, COMPARACION PAREADA")
    print("=" * 78)
    c = vs_naive(resultados, "arima")
    print(f"  ARIMA gana : {c['gana']:5.1f}%")
    print(f"  EMPATE     : {c['empata']:5.1f}%   <- ARIMA(0,1,0) es la caminata aleatoria")
    print(f"  NAIVE gana : {c['pierde']:5.1f}%")
    print(f"  (n = {c['n']} pares corte-pais)")

    print("\n" + "=" * 78)
    print("RANKING POR NIVEL DE CALIDAD DEL DATO  <- la tabla que importa")
    print("=" * 78)
    for tier, tabla in summarize_by_tier(resultados, panel).items():
        print(f"\n--- {tier} ({int(tabla.n_paises.iloc[0])} países) ---")
        print(tabla.drop(columns="n_paises").to_string())

    print("\nModelo recomendado por nivel:")
    for tier, m in best_model_by_tier(resultados, panel).items():
        print(f"  {tier:12} -> {m}")

    print("\n" + "=" * 78)
    print("PEORES SERIES (MASE mediano entre todos los modelos)")
    print("=" * 78)
    print(resultados.groupby("country").mase.median().sort_values(
        ascending=False).head(5).round(2).to_string())

    print(f"\nResultados guardados en: {salida}")
