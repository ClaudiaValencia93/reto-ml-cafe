"""
Medición del auditor contra el detector determinista.

POR QUÉ ESTE MÓDULO ES EL QUE IMPORTA
-------------------------------------
Sin él, el auditor es una demostración: el modelo emite opiniones y nadie sabe
si valen algo. Con él, es un componente evaluado.

El ground truth es `data_quality`, calculado en `data.py` a partir del
porcentaje de años sin variación. No es una verdad absoluta —es una regla
sencilla— pero es objetiva, reproducible y totalmente independiente del modelo,
que nunca la vio.

CÓMO SE LEE EL RESULTADO
------------------------
La clase positiva es "la serie NO es una medición real", porque el objetivo del
auditor es *detectar problemas*:

    precisión  de lo que el auditor marcó como problema, cuánto lo era
    recall     de los problemas reales, cuántos encontró

Los países `mixto` quedan fuera del cálculo y se reportan aparte: el propio
detector determinista los considera ambiguos, así que usarlos como verdad
penalizaría o premiaría al auditor por acertar algo que no está definido.

Los desacuerdos no son ruido: son el resultado más informativo. Indican o bien
que el modelo alucina, o bien que detectó algo que una regla de conteo no puede
ver.
"""

import numpy as np
import pandas as pd

# El detector determinista solo da veredicto claro en los extremos.
VERDAD = {"medido": True, "arrastrado": False}


def matriz_confusion(tabla):
    """Compara el auditor contra el ground truth, excluyendo los ambiguos."""
    d = tabla[tabla.data_quality.isin(VERDAD)].copy()
    d["verdad_plausible"] = d.data_quality.map(VERDAD)

    # Positivo = detectar un problema = "no plausible"
    tp = int(((~d.plausible) & (~d.verdad_plausible)).sum())
    fp = int(((~d.plausible) & (d.verdad_plausible)).sum())
    fn = int(((d.plausible) & (~d.verdad_plausible)).sum())
    tn = int(((d.plausible) & (d.verdad_plausible)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(d)}


def metricas(cm):
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exactitud": (tp + tn) / cm["n"] if cm["n"] else float("nan"),
    }


def desacuerdos(tabla):
    """Países donde el auditor y el detector determinista difieren."""
    d = tabla[tabla.data_quality.isin(VERDAD)].copy()
    d["verdad_plausible"] = d.data_quality.map(VERDAD)
    return d[d.plausible != d.verdad_plausible][
        ["country", "data_quality", "flat_years_pct", "plausible",
         "pattern", "confidence", "reason"]
    ].sort_values("confidence", ascending=False)


def calibracion(tabla):
    """¿La confianza declarada predice el acierto?

    Un auditor bien calibrado acierta más cuando dice estar más seguro. Si la
    confianza no discrimina, el campo es decorativo y no debe usarse para
    priorizar revisiones manuales.
    """
    d = tabla[tabla.data_quality.isin(VERDAD)].copy()
    d["verdad_plausible"] = d.data_quality.map(VERDAD)
    d["acierta"] = d.plausible == d.verdad_plausible
    d["banda"] = pd.cut(d.confidence, [0, 0.7, 0.85, 1.0],
                        labels=["baja (<0.70)", "media (0.70-0.85)", "alta (>0.85)"])
    return d.groupby("banda", observed=True).agg(
        n=("acierta", "size"), aciertos=("acierta", "sum"),
        tasa_acierto=("acierta", "mean"),
    ).round(2)


def report(tabla):
    """Imprime la evaluación completa."""
    cm = matriz_confusion(tabla)
    m = metricas(cm)

    print("\n" + "=" * 78)
    print("EVALUACION DEL AUDITOR CONTRA EL DETECTOR DETERMINISTA")
    print("=" * 78)
    print("Ground truth: data_quality de src/coffee/data.py (el modelo nunca lo vio).")
    print("Clase positiva: 'no es medicion real'.")
    print(f"Se evaluan {cm['n']} paises; los 'mixto' se excluyen por ambiguos.\n")

    print("                       auditor dice")
    print("                    problema  | correcto")
    print(f"  real: arrastrado     {cm['tp']:3}     |   {cm['fn']:3}")
    print(f"  real: medido         {cm['fp']:3}     |   {cm['tn']:3}")

    print(f"\n  precision : {m['precision']:.1%}")
    print(f"  recall    : {m['recall']:.1%}")
    print(f"  F1        : {m['f1']:.1%}")
    print(f"  exactitud : {m['exactitud']:.1%}")

    print("\n--- Calibracion de la confianza ---")
    print(calibracion(tabla).to_string())

    d = desacuerdos(tabla)
    print(f"\n--- Desacuerdos ({len(d)}) ---")
    if d.empty:
        print("  ninguno")
    else:
        for _, f in d.iterrows():
            print(f"\n  {f.country}  [detector: {f.data_quality}, "
                  f"{f.flat_years_pct:.0f}% planos | auditor: plausible={f.plausible}, "
                  f"conf={f.confidence}]")
            print(f"    {f.reason}")

    ambiguos = tabla[~tabla.data_quality.isin(VERDAD)]
    print(f"\n--- Paises ambiguos para el detector ({len(ambiguos)}) ---")
    print(f"  el auditor los clasifico como: "
          f"{int((~ambiguos.plausible).sum())} con problema, "
          f"{int(ambiguos.plausible.sum())} correctos")

    return {**cm, **m}
