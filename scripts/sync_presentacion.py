"""
Regenera los datos incrustados en `reports/presentacion.html`.

POR QUÉ EXISTE
--------------
La presentación lleva los 53 países dentro del HTML para que funcione sin red y
sin cuenta. Eso la vuelve autocontenida, pero también significa que **no se
entera** de un cambio en el dataset o en los dictámenes: seguiría mostrando los
valores viejos sin dar ninguna señal.

`tests/test_verificacion_entrega.py` detecta ese desfase comparando la serie de
cada país valor por valor. Este script es la otra mitad: lo repara.

    pytest tests/test_verificacion_entrega.py   # avisa del desfase
    python scripts/sync_presentacion.py         # lo corrige

Qué se regenera, por país:

    c   nombre            v   serie en sacos de 60 kg
    t   tipo de café      q   etiqueta de calidad del detector
    y0  primer año        f   porcentaje de años sin variación
    cf  hechos calculados por series_facts.py
    fx  dictamen grabado del auditor

Lo demás del HTML —texto, estilos, gráficos— se edita a mano en el archivo.
"""

import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))

from assistant.series_facts import facts_from_panel  # noqa: E402
from coffee.data import load_clean  # noqa: E402

PRESENTACION = RAIZ / "reports" / "presentacion.html"
FIXTURE = RAIZ / "assistant" / "fixtures" / "plausibility_verdicts.json"

# El array vive en una sola línea dentro del script del demo.
PATRON = re.compile(r"(var D = )(\[.*?\])(;\s*\n)", re.DOTALL)


def construir_datos():
    """El array de países tal como lo consume la presentación."""
    panel, _ = load_clean()
    hechos = facts_from_panel(panel)
    verdicts = {
        v["country"]: v
        for v in json.loads(FIXTURE.read_text(encoding="utf-8"))["verdicts"]
    }

    datos = []
    for pais, g in panel.groupby("country", sort=True):
        g = g.sort_values("year")
        v = verdicts[pais]
        datos.append({
            "c": pais,
            "t": g.primary_type.iloc[0],
            "y0": int(g.year.min()),
            "v": [int(round(x)) for x in g.consumption_bags_60kg],
            "q": g.data_quality.iloc[0],
            "f": round(float(g.flat_years_pct.iloc[0]), 1),
            "cf": hechos[pais].resumen(),
            "fx": {
                "p": v["plausible"],
                "n": v["confidence"],
                "t": v["pattern"],
                "r": v["reason"],
            },
        })
    return datos


def main():
    datos = construir_datos()
    nuevo = json.dumps(datos, ensure_ascii=False, separators=(",", ":"))

    html = PRESENTACION.read_text(encoding="utf-8")
    m = PATRON.search(html)
    if not m:
        sys.exit("no se encontró el array de datos en la presentación")

    if m.group(2) == nuevo:
        print(f"Ya estaba al día: {len(datos)} países, {len(nuevo):,} bytes.")
        return

    antes = len(m.group(2))
    PRESENTACION.write_text(
        html[: m.start(2)] + nuevo + html[m.end(2):], encoding="utf-8"
    )
    print(f"Datos regenerados: {len(datos)} países.")
    print(f"  {antes:,} -> {len(nuevo):,} bytes")
    print("\nFalta publicar el archivo actualizado como artefacto para que la")
    print("versión en línea deje de estar desfasada.")


if __name__ == "__main__":
    main()
