"""
Modelos de pronóstico de consumo.

DECISIÓN CENTRAL DEL RETO: NO SE USA DEEP LEARNING
--------------------------------------------------
Cada país tiene como máximo 30 observaciones anuales. Una LSTM o un Transformer
tienen miles de parámetros; ajustarlos con 30 puntos no produce un modelo, produce
memorización. La literatura de forecasting es consistente en esto (competencias M3,
M4 y M5): con series cortas, los métodos estadísticos simples ganan.

El sustento no es la referencia sino el backtesting de `evaluate.py`: si un
modelo complejo superara a los baselines, se adoptaría.

CATÁLOGO
--------
Locales (un modelo por serie, ven solo su propia historia):
    naive   - repetir el último valor. Es la barra a superar, no un relleno.
    mean    - promedio histórico. Baseline débil, detecta series sin tendencia.
    drift   - último valor + pendiente media. El baseline serio para datos con tendencia.
    ets     - suavizamiento exponencial con tendencia amortiguada.
    arima   - ARIMA con orden elegido por AIC sobre una rejilla pequeña.

Global (un solo modelo entrenado sobre todos los países a la vez):
    GlobalLGBM - LightGBM sobre tasas de crecimiento con rezagos.

El modelo global existe porque 43 series de 30 puntos son 1,290 observaciones.
Un modelo que aprende el patrón compartido tiene más datos que cualquier modelo
local. Es la única vía razonable por la que el machine learning podría ganarle a
la estadística clásica aquí.
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Órdenes ARIMA candidatos. Rejilla deliberadamente pequeña: con ~20 puntos de
# entrenamiento, buscar en un espacio grande es sobreajustar la selección.
ARIMA_ORDERS = [(0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 0)]

# Rezagos del modelo global. Tres años cubren la persistencia de corto plazo sin
# consumir demasiadas observaciones al construir la matriz de features.
GLOBAL_LAGS = [1, 2, 3]


# --------------------------------------------------------------------------
# Modelos locales
# --------------------------------------------------------------------------

def naive(y, h):
    """Repetir el último valor observado."""
    return np.repeat(y[-1], h)


def mean_forecast(y, h):
    """Promedio histórico."""
    return np.repeat(np.mean(y), h)


def drift(y, h):
    """Último valor extendido por la pendiente media de toda la serie.

    Equivale a trazar una recta entre el primer y el último punto y prolongarla.
    """
    if len(y) < 2:
        return naive(y, h)
    pendiente = (y[-1] - y[0]) / (len(y) - 1)
    return y[-1] + pendiente * np.arange(1, h + 1)


def ets(y, h):
    """Suavizamiento exponencial con tendencia aditiva amortiguada.

    La amortiguación evita que la tendencia se extrapole indefinidamente, que es
    el modo de falla clásico de Holt sobre horizontes largos.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    if len(y) < 5:
        return drift(y, h)
    try:
        modelo = ExponentialSmoothing(
            y, trend="add", damped_trend=True, seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True)
        pred = np.asarray(modelo.forecast(h), dtype=float)
        return pred if np.all(np.isfinite(pred)) else drift(y, h)
    except Exception:
        return drift(y, h)


def arima(y, h):
    """ARIMA con orden elegido por AIC sobre `ARIMA_ORDERS`."""
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA

    if len(y) < 8:
        return drift(y, h)

    mejor, mejor_aic = None, np.inf
    for orden in ARIMA_ORDERS:
        try:
            ajuste = _ARIMA(y, order=orden).fit()
            if np.isfinite(ajuste.aic) and ajuste.aic < mejor_aic:
                mejor, mejor_aic = ajuste, ajuste.aic
        except Exception:
            continue

    if mejor is None:
        return drift(y, h)
    pred = np.asarray(mejor.forecast(h), dtype=float)
    return pred if np.all(np.isfinite(pred)) else drift(y, h)


LOCAL_MODELS = {
    "naive": naive,
    "mean": mean_forecast,
    "drift": drift,
    "ets": ets,
    "arima": arima,
}


# --------------------------------------------------------------------------
# Modelo global
# --------------------------------------------------------------------------

class GlobalLGBM:
    """LightGBM entrenado sobre todos los países simultáneamente.

    Trabaja sobre **tasas de crecimiento logarítmicas**, no sobre niveles. Es la
    decisión que hace viable el modelo global: el consumo de Brasil es diez mil
    veces el de Burundi, así que un modelo sobre niveles dedicaría toda su
    capacidad a distinguir tamaños de país. En crecimiento, todas las series son
    comparables y el modelo puede aprender dinámica en vez de escala.

    El pronóstico multi-paso es recursivo: se predice el crecimiento de t+1, se
    reconstruye el nivel, y ese valor alimenta los rezagos de t+2.
    """

    def __init__(self, lags=None, **lgbm_kwargs):
        self.lags = lags or GLOBAL_LAGS
        self.params = dict(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,        # series cortas: árboles pequeños
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbose=-1,
        )
        self.params.update(lgbm_kwargs)
        self.model = None
        self.feature_names = None

    # -- construcción de features ------------------------------------------

    def _tabla(self, panel):
        """Panel largo -> matriz de features de crecimiento.

        `panel` necesita las columnas: country, year, consumption_kg, primary_type.
        """
        filas = []
        for pais, g in panel.groupby("country", sort=False):
            g = g.sort_values("year")
            y = g.consumption_kg.to_numpy(dtype=float)
            if len(y) < max(self.lags) + 2:
                continue

            crecimiento = np.diff(np.log(y))          # len = len(y) - 1
            anios = g.year.to_numpy()[1:]
            tipo = g.primary_type.iloc[0]

            for t in range(max(self.lags), len(crecimiento)):
                fila = {
                    "country": pais,
                    "year": anios[t],
                    "target": crecimiento[t],
                    "is_arabica": int(tipo == "Arabica"),
                    "log_level_rel": np.log(y[t]) - np.log(y[: t + 1]).mean(),
                }
                for L in self.lags:
                    fila[f"growth_lag{L}"] = crecimiento[t - L]
                fila["growth_ma3"] = crecimiento[max(0, t - 3):t].mean()
                filas.append(fila)

        return pd.DataFrame(filas)

    # -- API ----------------------------------------------------------------

    def fit(self, panel):
        from lightgbm import LGBMRegressor

        tabla = self._tabla(panel)
        if tabla.empty:
            raise ValueError("no hay suficientes observaciones para entrenar")

        self.feature_names = [
            c for c in tabla.columns if c not in ("country", "year", "target")
        ]
        self.model = LGBMRegressor(**self.params)
        self.model.fit(tabla[self.feature_names], tabla["target"])
        return self

    def predict(self, y, h, primary_type="Arabica"):
        """Pronóstico recursivo de h pasos para una serie de niveles `y`."""
        if self.model is None:
            raise RuntimeError("llamar fit() antes de predict()")

        y = np.asarray(y, dtype=float)
        if len(y) < max(self.lags) + 2:
            return drift(y, h)

        niveles = list(y)
        crecimiento = list(np.diff(np.log(y)))
        salida = []

        for _ in range(h):
            fila = {
                "is_arabica": int(primary_type == "Arabica"),
                "log_level_rel": np.log(niveles[-1]) - np.log(niveles).mean(),
                "growth_ma3": np.mean(crecimiento[-3:]),
            }
            for L in self.lags:
                fila[f"growth_lag{L}"] = crecimiento[-L]

            X = pd.DataFrame([fila])[self.feature_names]
            g = float(self.model.predict(X)[0])

            nuevo = niveles[-1] * np.exp(g)
            salida.append(nuevo)
            niveles.append(nuevo)
            crecimiento.append(g)

        return np.asarray(salida)
