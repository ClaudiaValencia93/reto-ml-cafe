"""
Hechos calculados de cada serie.

EL REPARTO DE TERRITORIOS
-------------------------
Un modelo de lenguaje no cuenta: recuerda. Pedirle que cite cuántos años seguidos
repite un valor es pedirle precisamente lo que hace mal, y las tres afirmaciones
falsas detectadas en la primera corrida ("casi veinte años" donde hay dieciséis,
"sin un solo retroceso" donde hay uno) fueron todas de ese tipo.

La solución no es pedirle más cuidado, es quitarle el trabajo:

    el código describe la serie  ->  cifras incontestables
    el modelo aporta el mundo    ->  conocimiento que no está en el dataset

Este módulo produce la primera mitad. El prompt de `prompts.py` prohíbe
explícitamente la segunda y `schema.py` rechaza cualquier dictamen que la invada.

POR QUÉ ESTOS HECHOS NO VAN EN EL PROMPT
----------------------------------------
Se calculan DESPUÉS del juicio, para mostrarlos junto a él. Entregárselos al
modelo antes lo volvería circular: la racha de valores idénticos es, con otro
nombre, la misma señal que `flat_years_pct`, el detector contra el que después
se le mide. El modelo seguiría acertando y la métrica dejaría de significar nada.
"""

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

# Los valores de la ICO llegan en miles de sacos de 60 kg, así que todo
# incremento real es múltiplo de 500 sacos. Una serie donde TODOS lo son no
# prueba interpolación —le ocurre también a Brasil y Etiopía—, pero es un hecho
# de la serie y como tal se reporta.
CUANTO_SACOS = 500


@dataclass(frozen=True)
class SeriesFacts:
    country: str
    n_obs: int
    year_first: int
    year_last: int
    n_distinct: int
    max_run: int
    max_run_value: Optional[int]
    n_declines: int
    quantized: bool

    def to_dict(self):
        return asdict(self)

    def resumen(self):
        """Una o dos frases en español, listas para mostrar junto al dictamen."""
        partes = []

        if self.max_run >= 3:
            partes.append(
                f"{self.max_run_value:,.0f} sacos repetidos durante {self.max_run} "
                f"años consecutivos".replace(",", ".")
            )
        partes.append(
            f"{self.n_distinct} valores distintos en {self.n_obs} años"
        )
        partes.append(
            "ninguna caída" if self.n_declines == 0
            else f"{self.n_declines} caída" + ("s" if self.n_declines > 1 else "")
        )

        texto = ". ".join(p[0].upper() + p[1:] for p in partes) + "."
        if self.quantized:
            texto += " Todos los incrementos son múltiplos de 500 sacos."
        return texto


def _racha_maxima(y):
    """(longitud, valor) de la mayor secuencia de valores idénticos consecutivos."""
    if len(y) == 0:
        return 0, None
    mejor, mejor_val, act = 1, y[0], 1
    for i in range(1, len(y)):
        if y[i] == y[i - 1]:
            act += 1
            if act > mejor:
                mejor, mejor_val = act, y[i]
        else:
            act = 1
    return mejor, mejor_val


def facts_for(country, years, values):
    """Calcula los hechos de una serie. `values` en sacos de 60 kg."""
    y = np.asarray(values, dtype=float)
    diffs = np.diff(y)
    largo, valor = _racha_maxima(y)

    return SeriesFacts(
        country=country,
        n_obs=len(y),
        year_first=int(years[0]),
        year_last=int(years[-1]),
        n_distinct=int(len(np.unique(y))),
        max_run=int(largo),
        max_run_value=int(valor) if valor is not None else None,
        n_declines=int((diffs < 0).sum()),
        quantized=bool(len(diffs) > 0 and np.all(diffs % CUANTO_SACOS == 0)),
    )


def facts_from_panel(panel):
    """{país: SeriesFacts} para todo el panel limpio."""
    salida = {}
    for pais, g in panel.groupby("country", sort=True):
        g = g.sort_values("year")
        salida[pais] = facts_for(
            pais, g.year.tolist(), g.consumption_bags_60kg.tolist()
        )
    return salida


if __name__ == "__main__":
    from coffee.data import load_clean

    panel, _ = load_clean()
    hechos = facts_from_panel(panel)

    for pais in ("Kenya", "Bolivia (Plurinational State of)", "Rwanda", "Viet Nam"):
        print(f"\n{pais}")
        print(f"  {hechos[pais].resumen()}")
