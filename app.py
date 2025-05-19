import streamlit as st
import pandas as pd
import json
import yaml
import os
import re
from openai import OpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

# 🧠 Modelo de lenguaje a utilizar
MODELO_LLM = "gpt-4"  # Puedes cambiar a "gpt-4", "gpt-3.5-turbo", "gpt-4.1-nano", etc

# 📌 Cargar clave API desde config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
os.environ["OPENAI_API_KEY"] = config["openai"]["api_key"]

# ⚙️ Configurar app
st.set_page_config(layout="wide")
st.title("📊 Generador de Informes de Índices de Gas Natural")

# 📂 Precargar datos
df_historico = pd.read_csv("historico.csv")
df_estimaciones = pd.read_csv("estimaciones.csv")

# 🔤 Diccionario de nombres completos de índices
indice_nombres = {
    "HSC": "Houston Ship Channel",
    "Waha": "Waha",
    "Henry Hub": "Henry Hub",
    "SoCal": "Southern California Gas",
    "Tetco": "Texas Eastern, South Texas"
}

# Mostrar en el selectbox los nombres completos (valores), pero mapearlos a las claves (keys)
opciones_visibles = list(indice_nombres.values())
indice_seleccionado_valor = st.sidebar.selectbox("Selecciona un índice para analizar:", opciones_visibles)

# Buscar la clave correspondiente al valor seleccionado
indice = [k for k, v in indice_nombres.items() if v == indice_seleccionado_valor][0]
nombre_completo = indice_nombres[indice]

# 🧼 Reiniciar análisis generado si cambia el índice seleccionado
if "indice_anterior" not in st.session_state:
    st.session_state["indice_anterior"] = indice
elif st.session_state["indice_anterior"] != indice:
    st.session_state["indice_anterior"] = indice
    st.session_state["reporte_generado"] = None

# 📈 Función de resumen estadístico robustecida
def resumir_indice(nombre, df_hist, df_fut):
    df_hist['fecha'] = pd.to_datetime(df_hist['fecha'], dayfirst=True)
    df_fut['fecha'] = pd.to_datetime(df_fut['fecha'], dayfirst=True)

    fecha_min_hist = df_hist['fecha'].min()
    fecha_max_hist = df_hist['fecha'].max()

    # Filtrar estimaciones desde el mes posterior al histórico
    mes_siguiente = (fecha_max_hist + pd.DateOffset(months=1)).replace(day=1)
    df_fut_filtrado = df_fut[df_fut['fecha'] >= mes_siguiente]

    fecha_min_fut = df_fut_filtrado['fecha'].min()
    fecha_max_fut = df_fut_filtrado['fecha'].max()

    # Cálculos robustos para el histórico
    valores_hist = df_hist[nombre]
    valores_fut = df_fut_filtrado[nombre]

    q1 = valores_hist.quantile(0.25)
    q3 = valores_hist.quantile(0.75)

    resumen = {
        "Índice": indice_nombres[nombre],
        "Fecha mínima histórica": fecha_min_hist.strftime('%Y-%m-%d'),
        "Fecha máxima histórica": fecha_max_hist.strftime('%Y-%m-%d'),
        "Máximo histórico": round(valores_hist.max(), 3),
        "Mínimo histórico": round(valores_hist.min(), 3),
        "Promedio histórico": round(valores_hist.mean(), 3),
        "Mediana histórica": round(valores_hist.median(), 3),
        "Desviación estándar histórica": round(valores_hist.std(ddof=1), 3),
        "Rango intercuartílico (IQR)": round(q3 - q1, 3),
        "Fecha mínima estimaciones": fecha_min_fut.strftime('%Y-%m-%d'),
        "Fecha máxima estimaciones": fecha_max_fut.strftime('%Y-%m-%d'),
        "Promedio estimado (futuro)": round(valores_fut.mean(), 3),
        "Desviación estándar estimada (futuro)": round(valores_fut.std(ddof=1), 3)
    }

    return resumen


# 🧮 Generar resumen
resumen_indice = resumir_indice(indice, df_historico, df_estimaciones)

# 📋 Mostrar resumen como tabla
st.subheader(f"📈 Resumen estadístico del índice: {nombre_completo}")
resumen_df = pd.DataFrame(resumen_indice.items(), columns=["Indicador", "Valor"])
resumen_df = resumen_df[resumen_df["Indicador"] != "Índice"]
st.dataframe(resumen_df.set_index("Indicador"))

# 🧠 Prompt descriptivo con CoT y estadísticas robustas
AGENTE_PROMPT = f"""
Actúa como analista experto en mercados energéticos, especializado en precios de gas natural. Tu tarea es estudiar el índice: {resumen_indice['Índice']},cuyas unidades son USD/MMBtu, con base en la información estadística histórica y las estimaciones futuras más recientes.

Paso 1: Examina el comportamiento reciente de los precios históricos, considerando:
- Intervalo temporal desde {resumen_indice['Fecha mínima histórica']} hasta {resumen_indice['Fecha máxima histórica']}.
- Valores extremos (máximo y mínimo histórico).
- Tendencias generales observadas.
- Posibles episodios de alta o baja volatilidad reflejados en la desviación estándar o en la dispersión de los precios.

Paso 2: Evalúa la variabilidad de los datos históricos comparando medidas robustas como:
- Mediana histórica frente al promedio histórico.
- Desviación estándar frente al rango intercuartílico (IQR).
Luego, compara estos indicadores con las estimaciones futuras, tomando en cuenta el promedio y la desviación estándar proyectada a partir del mes posterior a la última observación histórica. Indica si hay evidencia de estabilización, aumento o disminución en la variabilidad esperada.

Paso 3: Formula implicaciones concretas para compradores y proveedores de gas natural. Toma en cuenta:
- Si la volatilidad es elevada, podría ser necesario considerar coberturas contractuales.
- Si los precios futuros son significativamente mayores o menores que los históricos, podría haber oportunidad o riesgo comercial.
- Cualquier patrón detectado que pueda influir en decisiones de compra, venta o planificación financiera.

Considera que los precios históricos tienen frecuencia diaria y los precios proyectados son promedios mensuales, por lo que cualquier comparación debe contemplar esta diferencia de granularidad temporal.

Resumen estadístico del índice {resumen_indice['Índice']}:
- Fecha mínima histórica: {resumen_indice['Fecha mínima histórica']}
- Fecha máxima histórica: {resumen_indice['Fecha máxima histórica']}
- Máximo histórico: {resumen_indice['Máximo histórico']}
- Mínimo histórico: {resumen_indice['Mínimo histórico']}
- Promedio histórico: {resumen_indice['Promedio histórico']}
- Mediana histórica: {resumen_indice['Mediana histórica']}
- Desviación estándar histórica: {resumen_indice['Desviación estándar histórica']}
- Rango intercuartílico (IQR): {resumen_indice['Rango intercuartílico (IQR)']}
- Fecha mínima estimaciones: {resumen_indice['Fecha mínima estimaciones']}
- Fecha máxima estimaciones: {resumen_indice['Fecha máxima estimaciones']}
- Promedio estimado (futuro): {resumen_indice['Promedio estimado (futuro)']}
- Desviación estándar estimada (futuro): {resumen_indice['Desviación estándar estimada (futuro)']}

Redacta el análisis con un enfoque técnico y estructurado, sin mencionar explícitamente que estás siguiendo pasos enumerados. 
No incluyas un título o encabezado, solo subtítulos para organizar la información.
Utiliza una secuencia lógica coherente y profesional, organizada en párrafos temáticos o subtítulos bien definidos. 
Todos los subtítulos que se utilicen deben presentarse siempre en **negritas**, para facilitar la lectura ejecutiva. 
Evita expresiones como “voy a analizar”, “este reporte muestra” o enumeraciones explícitas como “paso 1”, “paso 2”. 
El contenido debe ser claro, objetivo, orientado a la toma de decisiones, y redactado en un tono profesional, sin hacer referencia a su origen automatizado.
"""


# 🧾 Generar reporte en texto
client = OpenAI()

def generar_reporte(resumen_indice):
    prompt = AGENTE_PROMPT
    response = client.chat.completions.create(
        model=MODELO_LLM,
        messages=[
            {"role": "system", "content": "Actúas como analista energético experto."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Mostrar análisis: solo muestra el encabezado si aún no existe el análisis
if "reporte_generado" not in st.session_state:
    st.session_state["reporte_generado"] = None

if not st.session_state["reporte_generado"]:
    if st.button("📄 Generar análisis del índice"):
        st.session_state["reporte_generado"] = generar_reporte(resumen_indice)

# Mostrar encabezado y análisis generado
if st.session_state["reporte_generado"]:
    st.subheader(f"🧠 Análisis automático del índice: {nombre_completo}")
    st.markdown(st.session_state["reporte_generado"])

# 📑 Preparar metadata
st.session_state.metadata = {
    "archivos": ["historico.csv", "estimaciones.csv"],
    "descripción": (
        "Ambos archivos contienen precios del gas natural expresados en USD/MMBtu para distintos índices. "
        "El archivo 'historico.csv' incluye precios diarios observados, mientras que 'estimaciones.csv' contiene "
        "precios promedio mensuales proyectados."
    ),
    "columnas_comunes": {
        "fecha": (
            "En 'historico.csv', corresponde a la fecha de registro diario (formato YYYY-MM-DD). "
            "En 'estimaciones.csv', corresponde a la fecha de proyección mensual (formato YYYY-MM-DD)."
        ),
        indice: f"Precio del índice {nombre_completo} en USD/MMBtu."
    },
    "nota": (
        "'historico.csv' representa datos observados del pasado, y 'estimaciones.csv' contiene proyecciones a futuro. "
        "Ambos archivos deben combinarse para realizar un análisis temporal integral del comportamiento del índice seleccionado."
        "Para comparar series temporales, convertir los datos diarios de historico.csv a promedios mensuales antes de graficar."
    )
}

# 📝 Solicitud del usuario
st.markdown("### :blue[Crear visualización con ayuda de IA]")
peticion_usuario = st.text_input("Describe la visualización que deseas generar:")

# 🤖 Evaluador de claridad y propuesta de mejora del prompt
if peticion_usuario.strip():
    st.markdown("### 💡 Sugerencias automáticas para mejorar tu solicitud de visualización")

    PROMPT_EVALUADOR = f'''
    Actúas como un experto en visualización de datos y comunicación con modelos de lenguaje. Tu responsabilidad es mejorar una instrucción escrita por un usuario para generar una visualización de datos en Python utilizando Streamlit y Plotly Express, dentro de un flujo interactivo.

    El usuario ha escrito la siguiente solicitud:
    ---
    {peticion_usuario}
    ---

    Contexto del análisis del índice seleccionado:
    ---
    {st.session_state.get("reporte_generado", "Sin análisis disponible aún.")}
    ---

    Tu tarea consiste en:
    1. Reformular la instrucción del usuario como un prompt claro, conciso y específico, que inicie con un verbo de acción (por ejemplo: "Mostrar...", "Comparar...", "Visualizar...") e indique las variables involucradas.
    2. Seleccionar el tipo de gráfico más adecuado (línea, barras, dispersión, boxplot, etc.) en función de los datos y del propósito de análisis.
    3. Describir de manera clara, concisa y efectiva el objetivo principal de la visualización: debe resumir claramente el propósito analítico, como detectar patrones, comparar grupos, analizar dispersión o apoyar la toma de decisiones.
    4. ⚠️ Si la instrucción implica comparar precios históricos (`historico.csv`) con proyecciones (`estimaciones.csv`), recuerda que los primeros son precios diarios y los segundos promedios mensuales. Agrega una sugerencia clara de que se debe:
        - Agregar los precios diarios mensualmente (usando `.dt.to_period('M').dt.to_timestamp()` seguido de `groupby`)
        - Alinear los DataFrames antes de graficar.
    5. Como referencia opcional, puedes consultar esta tabla de sugerencias con estructuras típicas de visualización. Utilízala solo si consideras que complementa tu recomendación y mejora la claridad del prompt. Estas plantillas no son obligatorias:
        - **Línea**: Usa `px.line` con `x=fecha` y `y=precio`. Para `historico.csv`, utiliza precios diarios. Para `estimaciones.csv`, los precios son promedio mensual. Si deseas comparar ambos, primero agrega los precios diarios de `historico.csv` a nivel mensual para alinearlos con `estimaciones.csv`, y concatena los DataFrames. Incluye título, etiquetas y ajustes de layout. Ideal para mostrar evolución temporal, identificar tendencias y ciclos.
        - **Barras**: Usa `px.bar` con variables categóricas como índice, mes o año. Para `historico.csv`, considera agregaciones como media mensual por categoría. Para comparar con `estimaciones.csv`, asegúrate de alinear los periodos (ambos deben ser datos mensuales). Ideal para comparar cantidades entre categorías o periodos agregados.
        - **Dispersión**: Usa `px.scatter` para analizar correlaciones. Para `historico.csv`, puedes usar día y precio. Para `estimaciones.csv`, generalmente solo hay un punto por mes. Para comparación, considera mostrar ambos como capas o distinguir por color. Ideal para ver relación entre dos variables continuas o comparar dispersión.
        - **Boxplot**: Usa `px.box` para representar distribución de precios por mes, año o categoría. Ideal para `historico.csv`, agrupando por mes. Para comparar con `estimaciones.csv`, recuerda que ya son medias mensuales, por lo que no aplicaría directamente. Útil para analizar dispersión y valores atípicos.
        - **Pie**: Usa `px.pie` con variables como índice o región si existen. Usar con `estimaciones.csv` solo si representa proporciones agregadas por categoría. No es ideal para series temporales como `historico.csv`, a menos que se agreguen por periodo. Adecuado para mostrar proporciones dentro de un total.
        - **Mapa**: Usa `px.choropleth` o `px.scatter_geo` con coordenadas geográficas. Asegúrate de que los datos incluyan ubicación. Aplica igual para datos históricos o estimaciones, o una comparación si ambas contienen regiones equivalentes. Útil para mostrar patrones regionales o distribución espacial.
        - **Sankey**: Usa `plotly.graph_objects.Sankey` para visualizar flujos entre categorías como índices a regiones o años. Aplica si se cuenta con datos categóricos de transición, lo cual suele derivarse de agregaciones sobre `historico.csv`. Ideal para visualizar relaciones de flujo entre clases.

        Estas plantillas te orientan para construir visualizaciones más efectivas según el contexto del análisis, pero no sustituyen tu criterio profesional, es decir, son meramente orientativas, no son obligatorias.



    Tu respuesta debe cumplir con el siguiente formato exacto y sin desviaciones:

    Prompt sugerido: <un par de líneas claras, comenzando con un verbo, que indique qué se debe visualizar, con qué variables y usando qué archivos>

    ipo de gráfico recomendado: <nombre del gráfico (como línea, barras, etc.) seguido de una breve explicación contextual del por qué es adecuado, por ejemplo: “Gráfico de líneas — ideal para observar evolución temporal en datos combinados de precios históricos y estimaciones.”>

    Objetivo del gráfico: <una sola línea que resuma para qué se usará la visualización, qué patrón, comparación o tendencia se quiere identificar>

    No incluyas encabezados, explicaciones, listas adicionales ni texto fuera de estas tres líneas. No uses etiquetas Markdown ni comillas. Cada línea debe ser informativa, breve y útil para alimentar directamente a un modelo generador de código.
    '''


    try:
        model = OpenAIModel(model_name=MODELO_LLM)
        agente_evaluador = Agent(model)
        if "sugerencia_ia" not in st.session_state:
            st.session_state["sugerencia_ia"] = ""

        if st.button("💡 Generar sugerencia de prompt"):
            resultado_eval = agente_evaluador.run_sync(PROMPT_EVALUADOR)
            st.session_state["sugerencia_ia"] = resultado_eval.output

        if st.session_state["sugerencia_ia"]:
            st.text_area("🧠 Sugerencia generada por IA:", value=st.session_state["sugerencia_ia"], height=200, key="sugerencia_ia_box")

        # Intentar extraer el texto del prompt sugerido, tipo de gráfico y objetivo usando regex
        sugerido_extraido = None
        tipo_grafico_extraido = None
        objetivo_grafico_extraido = None

        match_prompt = re.search(r"Prompt sugerido\s*:\s*(.+)", st.session_state["sugerencia_ia"], re.IGNORECASE)
        match_grafico = re.search(r"Tipo de gráfico recomendado\s*:\s*(.+)", st.session_state["sugerencia_ia"], re.IGNORECASE)
        match_objetivo = re.search(r"Objetivo del gráfico\s*:\s*(.+)", st.session_state["sugerencia_ia"], re.IGNORECASE)

        if match_prompt:
            sugerido_extraido = match_prompt.group(1).strip()
        if match_grafico:
            tipo_grafico_extraido = match_grafico.group(1).strip()
        if match_objetivo:
            objetivo_grafico_extraido = match_objetivo.group(1).strip()

        if sugerido_extraido:
            texto_completo = sugerido_extraido
            if tipo_grafico_extraido:
                texto_completo += f" (Tipo de gráfico sugerido: {tipo_grafico_extraido})"
            if objetivo_grafico_extraido:
                texto_completo += f" — Objetivo: {objetivo_grafico_extraido}"
            if st.button("Usar prompt, tipo y objetivo sugeridos"):
                st.session_state["peticion_final"] = texto_completo
        else:
            st.info("No se detectó un prompt sugerido. Haz clic en 'Generar sugerencia de prompt' o edita uno manualmente si lo prefieres.")


    except Exception as e:
        st.error(f"No se pudo generar la sugerencia: {e}")

# ✍️ Campo para que el usuario ajuste o escriba su versión final
peticion_final = st.text_area(
    "✍️ Ingresa o ajusta tu versión final de la descripción de visualización (este será el prompt utilizado):",
    value=st.session_state.get("peticion_final", peticion_usuario),
    height=100
)

# 🧠 Prompt de visualización mejorado
libreria = "plotly express (usa tamaño de figura 800 x 600)"
prompt_visual = f"""
Contexto:
- Estás trabajando con dos archivos CSV: "historico.csv" y "estimaciones.csv".
- El archivo "historico.csv" contiene precios diarios de gas natural por índice.
- El archivo "estimaciones.csv" contiene precios estimados como promedios mensuales para los mismos índices.
- Para una comparación válida entre ambos archivos, debes agregar (agrupar) los precios diarios del histórico por mes, obteniendo su promedio mensual.
- Debes utilizar la biblioteca {libreria}.
- La visualización debe desplegarse con sintaxis de Streamlit y seguir buenas prácticas de presentación visual.
- Prioriza la aplicación de los principios de Edward Tufte y las mejores prácticas de visualización de datos: evita el ruido visual innecesario, prioriza la claridad, la densidad informativa y el uso eficiente del espacio, y resalta patrones significativos sin distorsionar la escala.


Descripción del conjunto de datos:
<descripción>
{json.dumps(st.session_state.metadata, indent=4, ensure_ascii=False)}
</descripción>

Análisis previo generado por el agente:
{st.session_state.get("reporte_generado", "Sin análisis disponible aún.")}

Instrucción:
<petición>
El usuario ha solicitado la siguiente visualización:
{peticion_final}
</petición>

Requisitos de salida:
- Genera únicamente código Python válido.
- No incluyas texto explicativo, solo el bloque de código.
"""

system_prompt_visual = f"""
Tu tarea es generar visualizaciones de datos utilizando Python y Streamlit.

Instrucciones:
- Estás trabajando con dos archivos: "historico.csv" (precios diarios) y "estimaciones.csv" (precios promedios mensuales).
- Para realizar comparaciones válidas, los precios del archivo "historico.csv" deben agregarse por mes utilizando el promedio mensual antes de graficarse.
- Para calcular promedios mensuales de precios en `historico.csv`, usa `.dt.to_period('M').dt.to_timestamp()` sobre la columna de fecha, y luego agrupa con `groupby` antes de hacer la unión con `estimaciones.csv`.
- Utiliza la biblioteca Plotly Express o Plotly Graph Objects para construir gráficos claros, legibles y efectivos.
- Genera el tipo de gráfico más adecuado (línea, barras, dispersión, boxplot, etc.) según la estructura del archivo de datos y las estadísticas previamente calculadas.
- El gráfico debe facilitar la interpretación de tendencias, distribución, dispersión o relaciones relevantes del índice seleccionado.
- Puedes volver a leer el archivo CSV si es necesario para construir la visualización.
- Siempre que se use `pd.to_datetime`, utiliza `dayfirst=True` para asegurar el formato correcto de fechas como "13/01/2022".
- Si el análisis incluye datos históricos y estimaciones, usa elementos separadas con estilos diferentes (color, trazo, etc) para distinguirlos.
- Prioriza la aplicación de los principios de Edward Tufte y las mejores prácticas de visualización de datos: evita el ruido visual innecesario, prioriza la claridad, la densidad informativa y el uso eficiente del espacio, y resalta patrones significativos sin distorsionar la escala.
- No incluyas explicaciones, comentarios, ni texto descriptivo adicional.
- No uses markdown ni encierres el código con delimitadores como ``` o bloques de código.
- El código debe estar en Python puro, ser ejecutable directamente, y centrado únicamente en la visualización.
- Usa los valores estadísticos proporcionados (máximo, mínimo, promedio, cuartiles, etc.) como referencia para configurar escalas, ejes y rangos visibles en la gráfica.

Formato de salida:
- Solo código Python funcional, limpio y directo, sin ningún otro tipo de contenido.
"""

# 🚀 Generar código con IA
if st.button("Generar código con IA"):
    try:
        model = OpenAIModel(model_name=MODELO_LLM)
        agent = Agent(model, system_prompt=system_prompt_visual)
        result = agent.run_sync(prompt_visual)
        st.session_state["code_to_execute"] = result.output
        st.session_state["codigo_generado"] = result.output.strip().replace("```python", "").replace("```", "")
    except Exception as e:
        st.error(f"Ocurrió un error: {e}. Revisa tu archivo config.yaml o la conexión.")

# 🧪 Mostrar y editar el código generado
codigo = st.session_state.get("codigo_generado", "")
codigo = st.text_area("🧪 Código generado (puedes editarlo antes de ejecutar):", codigo, height=400)

# ⚠️ Ejecutar el código generado
st.warning("⚠️ Revisa el código antes de ejecutarlo.")
if st.button("Ejecutar código"):
    try:
        exec(codigo, globals())  # Ejecuta directamente en el contexto global de Streamlit
        st.success("✅ Código ejecutado correctamente.")
    except Exception as e:
        st.error(f"❌ Error al ejecutar el código: {e}")