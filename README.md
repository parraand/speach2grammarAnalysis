# Pipeline de Transcripción, Corrección y Análisis de Audio Educativo
>Este documento presenta la descripción de la solución, la arquitectura y las principales consideraciones y pasos requeridos para realizar la ejecución e instalación del pipeline de transcripción, corrección gramatical y análisis de errores a partir de un archivo de audio.

También, en los siguientes links se encuentra la información documental asociada al proyecto:

[Informe Técnico (PDF)](docs/Test%20report%20Andres%20Parra.pdf)

[Carpeta de Datos](data/)

[Notebook de Desarrollo](src/Notebook_test_andres_parra.ipynb)

[Video Demo](Pendiente)

---

## Tabla de Contenidos
* [Descripción de la solución](#descripción-de-la-solución)
* [Arquitectura lógica de la solución](#arquitectura-lógica-de-la-solución)
* [Estructura del proyecto](#estructura-del-proyecto)
* [Instrucciones para ejecución](#instrucciones-para-ejecución)
* [Requerimientos](#requerimientos)
* [Autores](#autores)

---

## Descripción de la solución

### Objetivo General
- Implementar un pipeline completo que permita la transcripción automática, corrección gramatical y análisis de errores de un archivo de audio educativo, evaluando la calidad mediante métricas lingüísticas y proponiendo mejoras pedagógicas.

### Objetivos Específicos
- Generar transcripciones automáticas usando **Whisper** y **WhisperX**.
- Crear un conjunto de referencia (Gold Set) para evaluar la precisión del modelo.
- Aplicar modelos LLM (T5, BART) para corrección gramatical.
- Medir desempeño con métricas **WER** y **CER** antes y después de la corrección.
- Extraer errores gramaticales frecuentes y proponer reglas de negocio para retroalimentación.

### Reto
Evaluar y mejorar la calidad de transcripciones automáticas de un archivo de audio de 5 minutos, generando insights que sirvan como insumo para estrategias pedagógicas y detección de errores comunes.

### Solución
Desarrollo de un pipeline modular que:
- Segmenta y procesa el audio.
- Genera transcripciones automáticas.
- Aplica corrección gramatical con modelos pre-entrenados.
- Evalúa la mejora con métricas cuantitativas.
- Extrae patrones de errores frecuentes y sugiere ejercicios específicos.

### Impacto Potencial
- Reducción en tiempo de revisión manual.
- Mejora en precisión de transcripciones para contextos educativos.
- Posible integración con sistemas de enseñanza personalizada.

---

## Arquitectura lógica de la solución
**Datos de Entrada**
- Audio completo (`audio_full.m4a`) y fragmentos (`audio_parte_*.m4a`).
- Conjunto de referencia manual (`transcript_gold.csv`).

**Pipeline**
1. **Transcripción automática** con Whisper y WhisperX.
2. **Corrección gramatical** con modelos T5 y BART.
3. **Evaluación** de precisión mediante WER y CER.
4. **Análisis de errores** para generar recomendaciones.
5. **Diarización opcional** para separar intervenciones por hablante.

**Salida**
- Transcripciones corregidas (`transcript_corrected.csv`).
- Métricas de desempeño.
- Reporte técnico (`Test report Andres Parra.pdf`).

---

## Estructura del proyecto

.
│ .gitignore
│ HFtoken.txt
│ README.md
│ requirements.txt
│
├───data
│ audio_full.m4a
│ transcript_gold.csv
│ transcript_raw.csv
│
├───docs
│ Test report Andres Parra.pdf
│
├───fragmentos
│ audio_parte_*.m4a
│
├───src
│ development.py
│ Notebook_test_andres_parra.ipynb
│
└───temp
fragmentos_temp.m4a

yaml
Copiar
Editar

---

## Instrucciones para ejecución

### Forma 1: Ejecución con Python (Ambiente Virtual)
1. Crear ambiente virtual:
```bash
conda create --name audio_env python=3.9
conda activate audio_env
Instalar dependencias:

bash
Copiar
Editar
pip install -r requirements.txt
Ejecutar pipeline:

bash
Copiar
Editar
python src/development.py
O desde Jupyter Notebook:

bash
Copiar
Editar
jupyter notebook src/Notebook_test_andres_parra.ipynb
Forma 2: Ejecución con Docker (Opcional)
Compilar imagen:

bash
Copiar
Editar
docker build -t audio_pipeline .
Ejecutar contenedor:

bash
Copiar
Editar
docker run -it --name audio_pipeline audio_pipeline
Requerimientos
Hardware
RAM: 8 GB mínimo

GPU NVIDIA (opcional, recomendado para aceleración)

Espacio: 5 GB

Software
Python 3.9

Conda o entorno virtual

Docker (opcional)

Librerías
pandas

numpy

torch

transformers

librosa

jiwer

seaborn

Autores
Organización	Nombre	Rol	Contacto
Proyecto Individual	Andres Parra Rodriguez	Data Scientist	LinkedIn