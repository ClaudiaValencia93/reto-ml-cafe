"""
Construcción del prompt del auditor de plausibilidad.

LA DECISIÓN DE DISEÑO MÁS IMPORTANTE DE ESTE MÓDULO
---------------------------------------------------
Al modelo **no se le entrega** `flat_years_pct` ni la etiqueta `data_quality`.

La tentación es dársela: mejoraría el acierto aparente de inmediato. Pero
entonces el modelo se limitaría a repetir la conclusión del detector
determinista, y la evaluación posterior mediría si el modelo sabe copiar, no si
sabe juzgar. El resultado sería circular y sin valor.

El modelo recibe únicamente la serie cruda y el nombre del país. Tiene que
llegar solo a la conclusión, usando lo que sabe del mundo. Solo así la
comparación contra el detector determinista significa algo.

QUÉ APORTA EL MODELO QUE EL CÓDIGO NO PUEDE APORTAR
---------------------------------------------------
`data.py` puede detectar QUE una serie está plana. No puede decidir SI eso es
plausible: hay países pequeños cuyo consumo podría ser genuinamente estable.
Esa distinción exige saber que la población de Kenia casi se duplicó entre 1990
y 2019, que Vietnam pasó de productor marginal a segundo exportador mundial, o
que Zimbabue atravesó una crisis económica severa.

Ese conocimiento no está en ninguna columna del dataset.
"""

import json

from assistant.schema import JSON_SCHEMA

SYSTEM = """\
Eres un auditor de calidad de datos especializado en estadísticas agrícolas \
internacionales. Evalúas series históricas de consumo doméstico de café \
reportadas por los países a la International Coffee Organization.

Tu tarea es decidir si el comportamiento de una serie es COMPATIBLE con lo que \
se sabe del país en ese periodo: su población, su economía, su cultura de \
consumo y su papel en el mercado del café.

Criterios:

- Un valor idéntico repetido muchos años seguidos casi nunca es una medición. \
Suele indicar que el país dejó de reportar y el organismo arrastró el último \
dato conocido.
- Variaciones en escalones perfectamente regulares sugieren interpolación.
- Una serie plana puede ser legítima en un país muy pequeño y estable; no todo \
estancamiento es un artefacto.
- Si el país es pequeño o está poco documentado y no tienes base sólida para \
juzgar, dilo: usa el patrón "indeterminado" y una confianza baja. Es \
preferible a inventar una justificación.

Responde ÚNICAMENTE con un objeto JSON válido que cumpla este esquema, sin \
texto adicional, sin explicación previa y sin bloques de código:

{schema}

El campo "reason" debe citar el hecho concreto del mundo que sustenta el \
juicio, no describir la serie. "El consumo no varía" describe; "la población \
del país casi se duplicó en ese periodo" sustenta.\
"""


def system_prompt() -> str:
    return SYSTEM.format(schema=json.dumps(JSON_SCHEMA, ensure_ascii=False, indent=2))


def user_prompt(country, years, values, coffee_type):
    """Arma el mensaje de un país.

    `values` va en sacos de 60 kg, que es la unidad de la industria y la escala
    en que un experto humano razonaría sobre estas cifras.
    """
    serie = ", ".join(f"{a}: {v:,.0f}" for a, v in zip(years, values))
    return (
        f"País: {country}\n"
        f"Tipo de café predominante: {coffee_type}\n"
        f"Periodo: {years[0]}-{years[-1]} ({len(years)} observaciones)\n"
        f"Consumo doméstico anual, en sacos de 60 kg:\n"
        f"{serie}\n\n"
        f"¿Es plausible esta serie para {country}?"
    )


def build_messages(country, years, values, coffee_type):
    """Mensajes listos para `client.messages.create`."""
    return [{"role": "user", "content": user_prompt(country, years, values, coffee_type)}]
