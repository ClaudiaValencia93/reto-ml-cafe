"""
Auditor de plausibilidad: recorre los países y consolida los dictámenes.

QUÉ HACE Y POR QUÉ EXISTE
-------------------------
`data.py` mide `flat_years_pct` y puede detectar QUE una serie está plana. No
puede decidir SI eso es plausible, porque esa decisión exige conocimiento que no
está en ninguna columna: cuánto creció la población del país, qué guerras
atravesó, si su cultura de consumo cambió.

Este módulo delega ese juicio a un modelo de lenguaje y después lo somete a
medición (ver `evaluation.py`). El aporte del LLM es conocimiento del mundo, no
capacidad de cálculo: los números los sigue calculando el código.

USO
---
    python -m assistant.auditor            # reproduce fixtures, costo cero
    python -m assistant.auditor --live     # llama a la API (~0.39 USD)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from assistant.client import get_client
from assistant.schema import VerdictError
from coffee.data import load_clean

SALIDA = Path(__file__).resolve().parents[1] / "data" / "processed" / "plausibility_audit.parquet"


def series_por_pais(panel):
    """(país, años, valores en sacos, tipo) para cada país del panel limpio."""
    for pais, g in panel.groupby("country", sort=True):
        g = g.sort_values("year")
        yield (
            pais,
            g.year.tolist(),
            g.consumption_bags_60kg.tolist(),
            g.primary_type.iloc[0],
        )


def run(panel=None, live=False, verbose=True):
    """Audita todos los países. Devuelve (dictámenes, incidencias).

    Un país que falle no detiene la corrida: se registra la incidencia y el
    proceso continúa. Un auditor que se cae en el país 12 de 53 no sirve.
    """
    panel = panel if panel is not None else load_clean()[0]
    cliente = get_client(live=live)

    if verbose:
        print(f"Modo: {cliente.modo} | países: {panel.country.nunique()}")
        if cliente.modo == "fixture":
            print(f"Fuente: {cliente.meta.get('generated_by', '?')} "
                  f"(prompt {cliente.meta.get('prompt_version', '?')})\n")

    filas, incidencias = [], []
    for pais, anios, valores, tipo in series_por_pais(panel):
        try:
            v = cliente.audit(pais, anios, valores, tipo)
            filas.append(v.to_dict())
        except (VerdictError, KeyError) as e:
            incidencias.append({"country": pais, "error": str(e)})
            if verbose:
                print(f"  ! {pais}: {e}", file=sys.stderr)

    dictamenes = pd.DataFrame(filas)

    if verbose:
        print(f"Dictámenes válidos: {len(dictamenes)} | incidencias: {len(incidencias)}")
        if cliente.modo == "live":
            print(f"Tokens: {cliente.tokens_entrada:,} entrada / "
                  f"{cliente.tokens_salida:,} salida")
            print(f"Costo real: ${cliente.costo_usd:.4f} USD")

    return dictamenes, incidencias


def enriquecer(dictamenes, panel):
    """Añade el ground truth determinista a cada dictamen, para poder medirlo."""
    verdad = panel.groupby("country").agg(
        data_quality=("data_quality", "first"),
        flat_years_pct=("flat_years_pct", "first"),
        n_obs=("year", "size"),
    ).reset_index()
    return dictamenes.merge(verdad, on="country", how="left")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true",
                    help="llama a la API de Anthropic en vez de reproducir fixtures")
    args = ap.parse_args()

    panel, _ = load_clean()
    dictamenes, incidencias = run(panel, live=args.live)

    if dictamenes.empty:
        sys.exit("no se obtuvo ningún dictamen válido")

    tabla = enriquecer(dictamenes, panel)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(SALIDA, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)

    print("\n" + "=" * 78)
    print("DICTAMENES DEL AUDITOR")
    print("=" * 78)
    print(tabla.pattern.value_counts().to_string())
    print(f"\nPlausibles: {int(tabla.plausible.sum())} | "
          f"No plausibles: {int((~tabla.plausible).sum())}")
    print(f"Confianza mediana: {tabla.confidence.median():.2f}")

    print("\n" + "=" * 78)
    print("EJEMPLOS")
    print("=" * 78)
    for pais in ("Kenya", "Viet Nam", "Bolivia (Plurinational State of)"):
        fila = tabla[tabla.country == pais]
        if fila.empty:
            continue
        f = fila.iloc[0]
        print(f"\n{f.country}  [{f.data_quality}, {f.flat_years_pct:.0f}% planos]")
        print(f"  plausible={f.plausible}  patron={f.pattern}  confianza={f.confidence}")
        print(f"  {f.reason}")

    from assistant.evaluation import report
    report(tabla)

    print(f"\nGuardado en: {SALIDA}")


if __name__ == "__main__":
    main()
