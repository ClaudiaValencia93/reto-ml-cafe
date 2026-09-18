# -*- coding: utf-8 -*-
"""Audita la presentacion contra las fuentes reales. SOLO LEE, no modifica."""
import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))

import pandas as pd
from coffee.data import load_clean

H = (RAIZ / "reports" / "presentacion.html").read_text(encoding="utf-8")
FX = json.loads((RAIZ / "assistant/fixtures/plausibility_verdicts.json").read_text(encoding="utf-8"))
AUD = pd.read_parquet(RAIZ / "data/processed/plausibility_audit.parquet")
BT = pd.read_parquet(RAIZ / "data/processed/backtest_consumo.parquet")
panel, _ = load_clean()

ok, mal = [], []


def chk(nombre, condicion, detalle=""):
    (ok if condicion else mal).append(f"{nombre}" + (f" — {detalle}" if detalle else ""))


# ---------------------------------------------------------------- terminologia
chk("terminología: sin 'arrastrado' como etiqueta",
    "dato arrastrado (12)" not in H and "Países con dato arrastrado" not in H)
chk("terminología: usa medido/parcial/repetido",
    all(t in H for t in ("medido", "parcial", "repetido")))

# ---------------------------------------------------------------- calidad dato
conteo = panel.groupby("data_quality").country.nunique().to_dict()
chk("reparto 18/14/21", f"<strong>medido</strong> (18)" in H
    and "<strong>parcial</strong> (14)" in H and "<strong>repetido</strong> (21)" in H,
    f"real: {conteo}")

flat_med = panel.groupby("country").flat_years_pct.first().median()
chk("mediana de años planos 62%", "62%" in H, f"real {flat_med:.1f}%")

# ---------------------------------------------------------------- backtesting
BT2 = BT.copy()
med = BT2[BT2.data_quality == "medido"].groupby("model").mase.median().round(2)
rep = BT2[BT2.data_quality == "repetido"].groupby("model").mase.median().round(2)
for m, v in [("drift", 0.76), ("arima", 0.84), ("ets", 0.98),
             ("naive", 1.50), ("global_lgbm", 1.89)]:
    chk(f"MASE medido {m}={v}", abs(med[m] - v) < 0.005, f"real {med[m]}")
chk("panel repetido dice 12 países",
    "Países con dato repetido (12)" in H,
    f"real {BT2[BT2.data_quality=='repetido'].country.nunique()}")
chk("3.156 evaluaciones", "3.156" in H, f"real {len(BT)}")
chk("13 cortes", "13 cortes" in H, f"real {BT.cut_year.nunique()}")

# ---------------------------------------------------------------- auditor
VERDAD = {"medido": True, "repetido": False}
d = AUD[AUD.data_quality.isin(VERDAD)].copy()
d["v"] = d.data_quality.map(VERDAD)
tp = int(((~d.plausible) & (~d.v)).sum()); fp = int(((~d.plausible) & d.v).sum())
fn = int((d.plausible & (~d.v)).sum()); tn = int((d.plausible & d.v).sum())
chk("matriz 21/0/1/17", (tp, fn, fp, tn) == (21, 0, 1, 17), f"real {(tp,fn,fp,tn)}")
chk("39 evaluados en el texto", "39 países" in H, f"real {len(d)}")
chk("95,5% precisión", "95,5%" in H, f"real {tp/(tp+fp)*100:.1f}%")

bandas = [(d[d.confidence <= .70], 7, "86%"),
          (d[(d.confidence > .70) & (d.confidence <= .85)], 20, "100%"),
          (d[d.confidence > .85], 12, "100%")]
for sub, n_esp, _ in bandas:
    chk(f"banda de {n_esp} países", len(sub) == n_esp, f"real {len(sub)}")

chips = re.search(r'data-f="hit">Aciertos <b>(\d+)</b>', H)
chk("chip aciertos = 38", chips and chips.group(1) == "38",
    f"real {int((d.plausible == d.v).sum())}")
chk("chip sin veredicto = 14",
    'data-f="amb">Sin veredicto <b>14</b>' in H,
    f"real {len(AUD) - len(d)}")

# ------------------------------------------------- las 5 tarjetas de ejemplo
fx = {v["country"]: v["reason"] for v in FX["verdicts"]}
TARJETAS = {"Kenia": "Kenya", "Vietnam": "Viet Nam", "Cuba": "Cuba",
            "Gabón": "Gabon", "Bolivia": "Bolivia (Plurinational State of)"}
for visible, clave in TARJETAS.items():
    # la tarjeta debe citar un fragmento real de la justificacion vigente
    frag = fx[clave][:55]
    chk(f"tarjeta {visible} usa la justificación vigente", frag in H,
        f"busca: {frag[:45]}...")

# ---------------------------------------------------------------- otros
chk("131 pruebas en portada", "131 pruebas" in H)
chk("sin 'sin un solo retroceso'", "sin un solo retroceso" not in H)
chk("sin 96 pruebas", "96 pruebas" not in H)
chk("menciona Banco Mundial", "Banco Mundial" in H)
chk("menciona series_facts", "series_facts" in H)

print(f"CORRECTO ({len(ok)}):")
for x in ok:
    print(f"  OK  {x}")
print()
print(f"A REVISAR ({len(mal)}):")
for x in mal:
    print(f"  !!  {x}")
