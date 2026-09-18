# Consumo doméstico y precios del café — analítica predictiva

Reto técnico de ingeniería de machine learning. **High Garden Coffee**, exportadora
internacional, busca aprovechar su base histórica de consumo doméstico de café para
identificar tendencias, proyectar rangos de precios y extraer información accionable.

---

## Resumen ejecutivo

**El mercado crece de forma sostenida pero muy concentrado.** El consumo mundial pasó
de 1.17 a 3.00 miles de millones de kg entre 1990 y 2019 (+156%, 3.3% anual compuesto).
Siete países explican el 80% del total y Brasil solo pesa el 44%.

**El crecimiento y el volumen están desacoplados.** Tanzania (+11.5% anual), Vietnam
(+10.4%) y Tailandia (+7.2%) crecen tres veces más rápido que el mercado mundial sin
estar entre los mayores consumidores. Vietnam multiplicó su consumo doméstico por 17
en tres décadas. En el extremo opuesto, Ghana, Togo y Gabón llevan 30 años en
contracción sostenida.

**Solo un tercio del dataset contiene mediciones reales.** El 62% mediano de las
variaciones año a año son exactamente cero: la ICO arrastra el último valor conocido
cuando un país no reporta. Kenia estuvo 23 años consecutivos registrando el mismo
valor exacto. De 55 países, **18 tienen series con medición razonablemente continua**.

**Ese hallazgo invierte la conclusión del modelado.** Evaluado sobre el panel completo,
el modelo trivial (repetir el último valor) parece ganar. Segmentando por calidad del
dato, en los países con medición real el modelo de tendencia lo supera con holgura:
MASE 0.76 contra 1.50, ganándole en el 70% de los casos.

**El pronóstico de precios acierta el rango, no el número.** Entrenado hasta 2019 y
validado contra los 82 meses posteriores —que el modelo nunca vio—, el pronóstico
puntual falló por ~40% y en una sola dirección: no anticipó el shock de oferta de
2021-2024. El intervalo al 95% contuvo el 100% de la realidad.

---

## Los cuatro requisitos del reto

| Requisito | Dónde se resuelve |
|---|---|
| Análisis de la información | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) |
| Solución a las problemáticas de negocio | [`02_modelado`](notebooks/02_modelado.ipynb) y [`03_precios`](notebooks/03_precios.ipynb) |
| Implementación y evaluación | [`src/coffee/`](src/coffee/) + 96 pruebas en [`tests/`](tests/) |
| Presentación de resultados | Este README y los notebooks ejecutados |
| **BONUS — IA generativa** | [`assistant/`](assistant/): auditor de plausibilidad, implementado y medido |

---

## Reproducir

Requiere Python 3.9 o superior.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate en Linux/Mac
pip install -r requirements.txt
pip install -e .
```

La instalación editable (`-e`) registra el paquete `coffee` en el entorno, de modo que
notebooks, pruebas y aplicaciones lo importan sin manipular `sys.path`.

```bash
pytest                          # 64 pruebas, ~35 s
python src/coffee/data.py       # carga y limpieza + reporte de calidad
python src/coffee/prices.py     # precios anuales por tipo de café
python src/coffee/evaluate.py   # backtesting completo (~8 min)
python src/coffee/price_forecast.py   # pronóstico de precios con intervalos
python -m assistant.auditor     # auditor de plausibilidad + su evaluación
```

Los datos de precios ya están versionados en `data/raw/`. Para actualizarlos:

```bash
curl -o data/raw/price_arabica_PCOFFOTMUSDM.csv \
  "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PCOFFOTMUSDM"
curl -o data/raw/price_robusta_PCOFFROBUSDM.csv \
  "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PCOFFROBUSDM"
```

---

## Estructura

```
src/coffee/
    data.py            carga, limpieza y medición de calidad del dataset
    prices.py          precios del FMI, agregados a año cafetero
    models.py          naive, mean, drift, ETS, ARIMA y LightGBM global
    evaluate.py        backtesting con origen móvil, MASE y segmentación
    price_forecast.py  pronóstico de precios con intervalos y validación

assistant/             BONUS: auditor de plausibilidad con LLM
    schema.py          contrato de salida y validación
    prompts.py         prompt del auditor (sin filtrar el ground truth)
    client.py          dos transportes: API en vivo y fixtures grabados
    auditor.py         orquestación sobre los 53 países
    evaluation.py      precisión, recall y calibración contra data.py
    fixtures/          dictámenes grabados, reproducibles sin costo

notebooks/
    01_eda.ipynb       análisis exploratorio y calidad del dato
    02_modelado.ipynb  comparación de modelos y la paradoja de composición
    03_precios.ipynb   rangos de precios futuros

tests/                 96 pruebas sobre limpieza, métricas, modelos y auditor
data/raw/              dataset original y series de precios, sin modificar
reports/figures/       11 figuras generadas por los notebooks
```

La lógica vive en `src/`, no en los notebooks. Los notebooks importan desde `src/` y se
dedican a explorar y narrar. Esa separación permite probar la lógica automáticamente y
reutilizarla desde una API o un dashboard.

---

## Decisiones tomadas

Cada decisión está declarada como constante documentada en el módulo correspondiente,
no enterrada en el código. Cambiar el valor cambia el comportamiento de todo el
pipeline.

### Las unidades no son tazas

El enunciado indica que el consumo está "en tazas". La verificación contradice esa
descripción: Brasil aparece con 492,000,000 en 1990/91, y la ICO reporta para ese año
8,200 miles de sacos de 60 kg. El producto `8,200 × 60,000` da exactamente 492,000,000,
y el 99.9% de los valores del dataset es divisible por 60.

**La unidad es kilogramos**, derivada de miles de sacos de 60 kg. El dato se expone
también en sacos, que es la unidad estándar de la industria.

### Los ceros codifican ausencia de reporte

Los ceros aparecen en bloques contiguos, no dispersos, lo que descarta el error
aleatorio. Hay tres casos y cada uno recibe tratamiento distinto:

| Caso | Países | Tratamiento |
|---|---|---|
| Ceros iniciales | Timor-Leste, Yemen, Laos, Guyana | Truncar: la serie empieza en el primer dato real |
| Ceros finales | Zambia | Truncar y marcar `forecastable = False` |
| Todo en ceros | Nepal, Guinea Ecuatorial | Eliminar |

Zambia es el caso crítico: 19 años de datos y luego 11 ceros. Sin este tratamiento,
cualquier modelo proyecta consumo cero perpetuo.

### La calidad del dato se mide, no se filtra

Cada país recibe `flat_years_pct` (porcentaje de años sin ningún cambio) y una etiqueta
`data_quality`:

| Nivel | Criterio | Países |
|---|---|---|
| medido | < 40% de años planos | 18 |
| parcial | 40% – 70% | 14 |
| repetido | > 70% | 21 |

Es un atributo y no un filtro porque el análisis de mercado es válido para los 55
países, mientras que las métricas de pronóstico solo son interpretables dentro de cada
nivel.

### No se usa deep learning

Cada país tiene como máximo 30 observaciones anuales. La decisión no se sustenta en una
referencia sino en el backtesting: se implementó el enfoque de machine learning más
razonable disponible —un LightGBM **global**, entrenado sobre los 43 países a la vez
(~1,290 observaciones) y sobre tasas de crecimiento logarítmicas para que Brasil y
Burundi sean comparables, que es la arquitectura ganadora de la competencia M5— y
obtuvo MASE 1.89 contra 0.76 de una recta ajustada por dos puntos.

Con 30 observaciones por serie, la complejidad no se paga.

### `Total_domestic_consumption` se descarta

Es la suma exacta de las 30 columnas de año. Si sobreviviera al formato largo sería
fuga de información: entregaría al modelo el agregado del futuro.

---

## Metodología de evaluación

**Origen móvil, no partición aleatoria.** Partir una serie temporal al azar entrena con
el futuro para predecir el pasado. El protocolo fija un año de corte, entrena solo con
lo anterior y predice los tres años siguientes; luego el corte avanza. Resultado: 13
orígenes × 43 países × 6 modelos = 3,156 evaluaciones.

**MASE como métrica principal.** El consumo va de 1.2e5 (Burundi) a 1.3e9 (Brasil). El
MAPE castiga desproporcionadamente los valores pequeños y no permite promediar entre
países. MASE es adimensional: por debajo de 1 el modelo supera al trivial, por encima
lo estorba.

### Resultados en los 18 países con medición real

| Modelo | MASE | Gana al trivial |
|---|---:|---:|
| **drift** | **0.76** | 70.5% |
| arima | 0.84 | 75.9% |
| ets | 0.98 | 57.5% |
| naive | 1.50 | — |
| global_lgbm | 1.89 | 38.5% |

En los países repetidos el modelo trivial obtiene MASE 0.00 y MAPE 0.00%: la serie es
constante por construcción, así que acertarla es trivial. Son esos ceros los que
invierten el resultado agregado.

### Modelo recomendado por segmento

| Segmento | Modelo | Razón |
|---|---|---|
| medido (18 países) | drift | MASE 0.76, supera al trivial en 70% de los casos |
| parcial (14) | naive o ARIMA | prácticamente equivalentes |
| repetido (21) | naive | óptimo por construcción |

---

## Rangos de precios futuros

El dataset entregado **no contiene ninguna variable de precio**. La brecha se cubre con
una fuente externa documentada: series `PCOFFOTMUSDM` (Arábica) y `PCOFFROBUSDM`
(Robusta) del Fondo Monetario Internacional, distribuidas por FRED, mensuales, en
centavos de dólar por libra, desde enero de 1992.

El modelado usa la serie **mensual** (415 observaciones) y no la anual (33): el precio
del café es global, así que el pronóstico no necesita empatar país por país con el
consumo. Se trabaja en logaritmos para que el intervalo no cruce el cero y el error sea
multiplicativo, que es como se comporta el mercado.

### Validación contra los 82 meses posteriores al dataset

| Tipo | MAPE | Sesgo | Cobertura 80% | Cobertura 95% |
|---|---:|---:|---:|---:|
| Arábica | 41.8% | +41.8% | 70.7% | 100% |
| Robusta | 38.5% | +37.3% | 73.2% | 100% |

El sesgo coincide con el MAPE, lo que indica que el modelo se quedó corto en casi todos
los meses y no en algunos: no anticipó la subida. El shock de 2021-2024 provino de
heladas en Brasil, sequía en Vietnam y costos de flete, y nada de eso está contenido en
la serie de precios previa.

El intervalo al 95% contuvo toda la realidad. El intervalo al 80% se quedó en 71%, algo
por debajo de su nivel nominal.

### Proyección

| Horizonte | Arábica, rango 80% | Ancho |
|---|---|---|
| 2026 | 314 – 446 | 1.4× |
| 2027 | 257 – 546 | 2.1× |
| 2028 | 216 – 649 | 3.0× |
| 2029 | 194 – 721 | 3.7× |

El pronóstico central es plano porque ARIMA selecciona un orden cercano a la caminata
aleatoria, comportamiento conocido de los precios de commodities y coherente con la
hipótesis de mercado eficiente.

**Lo accionable es el ancho de la banda.** A 12 meses el rango es de 1.4× y sirve para
dimensionar contratos y coberturas. A tres años supera 3.5× y no sostiene decisiones de
inversión. Acotar el rango de validez es parte del resultado.

---

## Bonus — auditor de plausibilidad con LLM

El enunciado pide proponer cómo la IA generativa daría más valor a la solución.
La respuesta obvia es un chatbot sobre los datos; la implementada es otra, y sale
del propio análisis.

### El problema que resuelve

`data.py` puede detectar **que** una serie está plana: mide `flat_years_pct`. No
puede decidir **si** eso es plausible, porque esa decisión exige información que no
está en ninguna columna.

Kenia registra 50,000 sacos idénticos durante 23 años. Saber que eso es imposible
requiere saber que su población casi se duplicó en ese periodo, que se urbanizó y
que desarrolló cultura de cafeterías. Ese es exactamente el tipo de conocimiento
que un modelo de lenguaje aporta y una regla de conteo no.

El LLM nunca calcula un número: recibe la serie y emite un juicio estructurado.

```json
{
  "plausible": false,
  "confidence": 0.92,
  "pattern": "valor_arrastrado",
  "reason": "La población de Kenia pasó de 23 a 52 millones entre 1990 y 2019..."
}
```

### La evaluación es lo que lo convierte en ingeniería

Al modelo **no se le entrega** `flat_years_pct` ni `data_quality`. Dárselos lo
llevaría a repetir la conclusión del detector, y la medición posterior evaluaría
copia en lugar de juicio. Una prueba dedicada falla si alguien filtra el ground
truth al prompt.

Resultado sobre los 39 países donde el detector determinista da veredicto claro
(los `parcial` se excluyen por ambiguos):

| Métrica | Valor |
|---|---:|
| Precisión | 95.5% |
| Recall | 100% |
| F1 | 97.7% |
| Exactitud | 97.4% |

La confianza declarada está calibrada: 86% de acierto en la banda baja (<0.70) y
100% en las dos bandas superiores. El campo sirve para priorizar revisión manual.

### El falso positivo, y por qué importa

El único desacuerdo fue Bolivia, marcada como interpolada. Su justificación afirma
que la serie crece *"sin un solo retroceso"*. La verificación contra los datos
muestra una caída de 240,000 kg en 2005: **el argumento contiene un hecho falso**.

Su segundo argumento —incrementos en múltiplos regulares de 30,000— tampoco
distingue nada: Brasil, Etiopía, Indonesia y Filipinas también los tienen al 100%,
y son los mercados mejor medidos del panel. Es consecuencia de que la ICO reporta
en miles de sacos de 60 kg.

Ese dictamen se lee como un argumento experto y es incorrecto. Sin la capa de
evaluación habría pasado como válido.

**Conclusión operativa:** el auditor detecta el 100% de los problemas con 95.5% de
precisión, pero sus justificaciones pueden contener hechos fabricados aun cuando el
veredicto sea razonable. No es un componente autónomo: genera candidatos para
revisión humana, y la evaluación es obligatoria, no opcional.

### Ejecución

```bash
python -m assistant.auditor          # reproduce dictámenes grabados, costo cero
python -m assistant.auditor --live   # llama a la API (~0.39 USD)
```

Los dictámenes de `assistant/fixtures/` fueron generados por Claude Opus 5 y se
reproducen desde disco, el patrón de *golden fixtures*. La ruta en vivo está
implementada y verificada hasta el último paso —autenticación y enrutamiento por
workspace funcionan— y se detiene únicamente en el control de saldo de la cuenta.
No se ejecutó contra la API por no disponer de créditos.

---

## Limitaciones

1. **Para los 21 países repetidos, cualquier cifra proyectada no es un pronóstico**
   sino la repetición de un valor que la ICO ya venía repitiendo. Presentarla como
   proyección sería falsa precisión.
2. Los años cafeteros 1990/91 y 1991/92 no tienen precio disponible: la serie del FMI
   empieza en 1992. No se imputan.
3. El pronóstico de precios es univariado. Mejorarlo no requiere un modelo más
   complejo sino datos exógenos: inventarios certificados de la ICE, clima en zonas
   productoras, tipo de cambio del real brasileño, costos de flete.
4. El cruce entre demanda y precio se presenta como mecanismo económico, no como
   modelo ajustado. Con 30 observaciones anuales y dos series crecientes, cualquier
   regresión produciría una correlación alta y espuria.

---

## Fuentes

- Dataset de consumo: `coffee_db.parquet`, derivado de International Coffee
  Organization (ICO), consumo doméstico 1990/91–2019/20.
- Precios: International Monetary Fund, *Primary Commodity Prices*, vía
  [FRED](https://fred.stlouisfed.org/series/PCOFFOTMUSDM).
