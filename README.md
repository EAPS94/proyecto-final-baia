# 📊 Generador de Informes de Índices de Gas Natural

Este proyecto permite generar reportes automatizados y visualizaciones interactivas sobre precios de índices de gas natural, combinando el análisis de datos históricos con estimaciones futuras. Está construido en Python con `Streamlit` como interfaz interactiva, y aprovecha modelos de lenguaje (GPT) para análisis textual y generación de código de visualización.

## 🎯 Objetivo

Brindar una herramienta interactiva para analizar el comportamiento de índices de gas natural como Houston Ship Channel, Waha, Henry Hub, entre otros. Permite:
- Generar resúmenes estadísticos robustos (históricos y estimaciones futuras)
- Obtener análisis redactados automáticamente por IA en tono ejecutivo
- Solicitar visualizaciones personalizadas a través de prompts en lenguaje natural

## 🛠️ Tecnologías utilizadas

- Python 3.10+
- Streamlit
- Pandas
- Plotly Express
- OpenAI API (GPT-3.5 / GPT-4)
- pydantic-ai (wrapper para LLMs)

## 📁  Estructura del proyecto

```
PROYECTO-FINAL-BAIA/
├── app.py                # Aplicación principal en Streamlit
├── historico.csv         # Datos históricos del índice
├── estimaciones.csv      # Estimaciones futuras del índice
├── requirements.txt      # Dependencias del entorno
├── .gitignore            # Exclusiones para el repositorio
├── README.md             # Este archivo
```

## ▶️ Cómo ejecutar el proyecto

1. Clona el repositorio:

```bash
git clone https://github.com/tu-usuario/tu-repositorio.git
cd tu-repositorio
```

2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

3. Crea un archivo config.yaml con tu API Key de OpenAI:

```yaml
openai:
  api_key: "sk-..."
```

4. Ejecuta la aplicación:

```bash
streamlit run app.py
```

---

### 👨‍🏫 **Uso académico**

Este proyecto fue desarrollado como parte del curso de arquitectura de productos de datos (ITAM, 2025). Su objetivo es explorar el uso de agentes LLM para la automatización de análisis técnico y visualización de precios energéticos.

---

### 📚 Referencias

Este proyecto se inspiró en gran medida en el enfoque presentado por **Alan Jones** en su artículo:

> Alan Jones. (2025). *Create brilliant AI visualizations from any CSV file*. Publicado en Medium.  
> Disponible en: [medium.com](https://medium.com/codefile/create-brilliant-ai-visualizations-from-any-csv-file-7197e0c17df3)

El artículo propone una arquitectura sencilla y poderosa para construir visualizaciones de datos interactivas basadas en lenguaje natural, combinando `Streamlit` con agentes basados en modelos de lenguaje (LLMs).  