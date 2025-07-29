# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 20:22:52 2025

@author: prand


Pipeline Sugerido:
    1. Transcripcion "raw"
    2. Extracción 5 fragmentos reprepsentativos (30s-60s) y transcripcion manual
    3. transcript_raw → transcript_corrected
    4. Evaluacion
    5. Insights

Anotaciones
- Es como que hay una coversacion entre un profesor y un estudiante
- Para transcripciones Wisper es gratis y local. Google Speech-to-Text (STT) es Gratis con límites. Azure igual


"""





# -------------------------------------------------- 1. Fragmentacion y obtencion gold ---------------------------
"""
instalacion pydub:
    pip install pydub

csv
      clip_id,start_time,end_time,transcript_gold,speaker
      01,00:00,00:45,"Hello everyone, welcome to class","teacher"
      02,01:30,02:10,"I have a question about the present perfect","student"
      …
    - Asigna speaker ("teacher" o "student") según quién hable.
"""


from pydub import AudioSegment
import pandas as pd
import math
import os

# Cargar audio
audio = AudioSegment.from_file("audio_full.m4a")

# Duración total en milisegundos
duracion_ms = len(audio)
# Duración de fragmentos en milisegundos (30 segundos)
fragmento_ms = 30 * 1000
# Número de fragmentos (-1 debido a que hay unos datos que no corresponden a audio)
num_fragmentos = math.ceil(duracion_ms / fragmento_ms) - 1
# Crear carpeta de salida
os.makedirs("fragmentos", exist_ok=True)
dfCrops = pd.DataFrame()
# Cortar y guardar fragmentos
for i in range(num_fragmentos):
    if i==0:
        inicio = i * fragmento_ms
    else:
        inicio = i * fragmento_ms + 1
    fin = (i + 1) * fragmento_ms
    fragmento = audio[inicio:fin]
    fragmento.export(f"fragmentos/audio_parte_{i}.m4a", format="ipod")
    dfCrops.loc[i, "clip_id"] = i
    dfCrops.loc[i, "start_time"] = inicio
    dfCrops.loc[i, "end_time"] = fin

# Indices fragmentos para transcribir
listaNOTrans = [0, 2, 4, 6, 8]
# Preparacion df gold para guardarlo
gold0 = dfCrops.copy()
gold0["transcript_gold"] = ""
gold0["speaker"] = ""
gold0 = gold0.drop(listaNOTrans)
# Nombre del archivo
csv_file = "transcript_gold.csv"
# Verifica si el archivo existe para no sobreescribirlo, ya que se llena luego a mano
if not os.path.exists(csv_file):
    # Guardar como CSV
    gold0.to_csv(csv_file, index=False, sep=';', encoding='utf-8')
    print(f"Archivo creado: {csv_file}")
else:
    print(f"El archivo ya existe: {csv_file}")


# ------------------------------------------------------ 2. Transcripción fragmentos -------------------------------------

# pip install openai-whisper ffmpeg-python
# conda install -c conda-forge ffmpeg
# Instalar en computador ffmpeg



# ------------------------------------------- Intento inicial (sin diarizacion)
import whisper

# Cargar modelo (Se puede usar: tiny, base, small, medium, large)
model = whisper.load_model("large")
# Indices fragmentos para transcribir
listaTrans = [1, 3, 5, 7, 9]
dfRaw = pd.DataFrame()
for i in listaTrans:
    # Transcribir el audio
    result = model.transcribe(f"fragmentos/audio_parte_{i}.m4a")
    # Agrupa y guarda lo transcrito
    text = result["text"]
    dfRaw.loc[i, "audio_id"] = i
    dfRaw.loc[i, "transcript_raw"] = text

# Guarda en CSV
dfRaw.to_csv("transcript_raw.csv", index=False, sep=';', encoding='utf-8')


# ------------------------------------------ intento diarizacion 1.1

import whisperx
import pandas as pd
from pydub import AudioSegment

# Cargar modelo
modelDiar = whisperx.load_model("small", device="cpu", language="en", compute_type="float32")
# Carga de token de Hugging Face (toca desargarlo y luego soolicitar permiso)
with open("HFtoken.txt") as f:
    hf_token = f.read().strip()
diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token)

    
# Cargar modelo (Se puede usar: tiny, base, small, medium, large)
modelTrans = whisper.load_model("small")
# Indices fragmentos para transcribir
listaTrans = [1, 3, 5, 7, 9]
dfRaw = pd.DataFrame()
for i in listaTrans:
    # Diarizacion
    diarization = diarize_model(f"fragmentos/audio_parte_{i}.m4a", num_speakers=2)
    # Cargar audio
    audio = AudioSegment.from_file(f"fragmentos/audio_parte_{i}.m4a")
    textOut = str()
    # Cortar y guardar fragmentos
    for f in range(len(diarization)):
        inicio = int(diarization.loc[f, 'start'] * 1000)
        fin = int(diarization.loc[f, 'end'] * 1000)
        fragmento = audio[inicio:fin]
        fragmento.export(f"fragmentos_temp.m4a", format="ipod")
        # Transcripción + alineación temporal
        result = modelTrans.transcribe('fragmentos_temp.m4a')
        text = result["text"]
        # textOut = textOut + diarization.loc[f, 'speaker'] + ': ' + text + ".n\ "
        textOut += text + "."
    
    dfRaw.loc[i, "audio_id"] = i
    dfRaw.loc[i, "transcript_raw"] = textOut

# Guarda en CSV
dfRaw.to_csv("transcript_raw.csv", index=False, sep=';', encoding='utf-8')





# -------------------------------------------------- 3. Correccion fragmentos con modelos -----------------------------------
"""
Instralacion pytorch:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
Instralacion transformers:
pip install transformers
"""

from transformers import pipeline

# Nombre del archivo
csv_raw = "transcript_raw.csv"
# Lectura del df raw
dfRaw = pd.read_csv(csv_raw, sep=';', encoding='utf-8-sig')

# Nombre del archivo
csv_gold = "transcript_gold.csv"
# Lectura del df gold
dfGold = pd.read_csv(csv_gold, sep=';', encoding='utf-8-sig')

# Unión del gold junto con lo transcrito (raw)
dfGR = pd.merge(dfRaw, dfGold, left_on='audio_id', right_on='clip_id', how='inner')


# ---------------------------------------------- Inicial sin busqueda hiperparámetros

# MODELO BASE SIN BUSQUEDA HIPERPARÁMETROS (vennify/t5-base-grammar-correction)
# Base: Google T5
# Tamaño: 220M

# Carga pipeline de corrección (inglés)
corrector = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")
# Funcion de corrección por fila
def corregir(texto):
    if pd.isna(texto) or not isinstance(texto, str):
        return texto
    resultado = corrector(f"fix: {texto}", max_length=512)[0]['generated_text']
    return resultado
# Asume que la columna con errores se llama 'transcript_raw'
dfGR["transcript_raw_corr"] = dfGR["transcript_raw"].apply(corregir)


# Guardar resultados
dfGR.to_csv("transcripciones_corregidas.csv", sep=';', index=False)

"""
Sin Diarizacion, base
WER_mejora_%: 2.87
CER_mejora_%: -8.16
WER_raw_mean: 0.47
CER_raw_mean: 0.25
WER_corr_mean: 0.46
CER_corr_mean: 0.26

Sin Diarizacion, whisper:large
WER_mejora_%: 5.62
CER_mejora_%: 4.38
WER_raw_mean: 0.38
CER_raw_mean: 0.24
WER_corr_mean: 0.37
CER_corr_mean: 0.24

Con Diarizacion, whisper:large, whisperx: base
WER_mejora_%: 12.83
CER_mejora_%: 15.56
WER_raw_mean: 0.49
CER_raw_mean: 0.38
WER_corr_mean: 0.42
CER_corr_mean: 0.31

Con Diarizacion, whisper:base, whisperx: base
WER_mejora_%: 13.71
CER_mejora_%: 15.76
WER_raw_mean: 0.68
CER_raw_mean: 0.48
WER_corr_mean: 0.58
CER_corr_mean: 0.40
"""


# ---------------------------------------------- Inicial 2 sin busqueda hiperparámetros

# MODELO BASE SIN BUSQUEDA HIPERPARÁMETROS (prithivida/grammar_error_correcter_v1)
# Base: GPT2
# Tamaño: 124M

# Carga pipeline de corrección (inglés)
corrector = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")
# Funcion de corrección por fila
def corregir(texto):
    if pd.isna(texto) or not isinstance(texto, str):
        return texto
    resultado = corrector(f"fix: {texto}", max_length=512)[0]['generated_text']
    return resultado
# Asume que la columna con errores se llama 'transcript_raw'
dfGR["transcript_raw_corr"] = dfGR["transcript_raw"].apply(corregir)


# Guardar resultados
dfGR.to_csv("transcripciones_corregidas.csv", sep=';', index=False)

"""
Sin Diarizacion, base
WER_mejora_%: -33.20
CER_mejora_%: -119.13
WER_raw_mean: 0.47
CER_raw_mean: 0.25
WER_corr_mean: 0.60
CER_corr_mean: 0.49

Con Diarizacion, base
WER_mejora_%: -23.15
CER_mejora_%: -46.47

"""


# ---------------------------------------------- Inicial 3 sin busqueda hiperparámetros

# MODELO BASE SIN BUSQUEDA HIPERPARÁMETROS (google/flan-t5-small)
# Base T5 small
# Tamaño: 60M

# Carga pipeline de corrección (inglés)
corrector = pipeline("text2text-generation", model="google/flan-t5-small")
# Funcion de corrección por fila
def corregir(texto):
    if pd.isna(texto) or not isinstance(texto, str):
        return texto
    resultado = corrector(f"Correct the grammar: {texto}", max_length=512)[0]['generated_text']
    return resultado
# Asume que la columna con errores se llama 'transcript_raw'
dfGR["transcript_raw_corr"] = dfGR["transcript_raw"].apply(corregir)


# Guardar resultados
dfGR.to_csv("transcripciones_corregidas.csv", sep=';', index=False)

"""
Sin 
WER_mejora_%: -0.64
CER_mejora_%: -6.89

WER_mejora_%: 4.37
CER_mejora_%: -8.81
"""




#---------------------------------------------------- 4. Evaluación cuantitativa -----------------------------------------------------------


import pandas as pd
from jiwer import wer, cer
import matplotlib.pyplot as plt
import numpy as np


def werCerDf(dfGR):
    # Calcular WER y CER para cada fila
    results = []
    for idx, row in dfGR.iterrows():
        gold = row["transcript_gold"]
        raw = row["transcript_raw"]
        corr = row["transcript_raw_corr"]
        wer_raw = wer(gold, raw)
        wer_corr = wer(gold, corr)
        cer_raw = cer(gold, raw)
        cer_corr = cer(gold, corr)
        results.append({
            "audio_id": row["audio_id"],
            "WER_raw": wer_raw,
            "WER_corr": wer_corr,
            "CER_raw": cer_raw,
            "CER_corr": cer_corr,
            "WER_mejora_%": 100 * (wer_raw - wer_corr) / wer_raw if wer_raw != 0 else 0,
            "CER_mejora_%": 100 * (cer_raw - cer_corr) / cer_raw if cer_raw != 0 else 0,
        })
    # Convertir a DataFrame
    df_resultados = pd.DataFrame(results)
    # Mostrar tabla
    print('\n')
    print(df_resultados[["audio_id", "WER_raw", "WER_corr", "CER_raw", "CER_corr", "WER_mejora_%", "CER_mejora_%"]].round(2))
    print(f"\nWER_mejora_%: {df_resultados['WER_mejora_%'].mean():.2f}")
    print(f"CER_mejora_%: {df_resultados['CER_mejora_%'].mean():.2f}")
    print(f"WER_raw_mean: {df_resultados['WER_raw'].mean():.2f}")
    print(f"CER_raw_mean: {df_resultados['CER_raw'].mean():.2f}")
    print(f"WER_corr_mean: {df_resultados['WER_corr'].mean():.2f}")
    print(f"CER_corr_mean: {df_resultados['CER_corr'].mean():.2f}")
    return df_resultados


# Nombre del archivo
csv_correg = "transcripciones_corregidas.csv"
# Lectura del df gold
dfGR = pd.read_csv(csv_correg, sep=';', encoding='utf-8-sig')
# Calculo der WER CER con funcion
df_resultados = werCerDf(dfGR)
# Guardado mejor resultado para gráficas
df_resultados_mejor = df_resultados.copy()

# ------------------------------------------- Graficas finales de mejor

# Tamaños para posiciones para el eje x
x = np.arange(len(df_resultados_mejor))
# Ancho de las barras
width = 0.35  
# Crear subplots
fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
# Gráfico de barras para WER
axs[0].bar(x - width/2, df_resultados["WER_raw"], width, label="WER raw")
axs[0].bar(x + width/2, df_resultados["WER_corr"], width, label="WER corregido")
axs[0].set_ylabel("WER")
axs[0].set_title("WER antes y después de la corrección")
axs[0].legend()
axs[0].grid(axis='y')
# Gráfico de barras para CER
axs[1].bar(x - width/2, df_resultados["CER_raw"], width, label="CER raw")
axs[1].bar(x + width/2, df_resultados["CER_corr"], width, label="CER corregido")
axs[1].set_ylabel("CER")
axs[1].set_title("CER antes y después de la corrección")
axs[1].set_xticks(x)
axs[1].set_xticklabels(df_resultados["audio_id"], rotation=45)
axs[1].legend()
axs[1].grid(axis='y')




# --------------------------------------------------------------------------------------------
import pandas as pd

from difflib import ndiff
from collections import Counter

dfGR = pd.read_csv("transcripciones_corregidas.csv", sep=';')

# Funcion para extraer cambios
def extraer_cambios(raw, corr):
    # Devuelve la diferencia entre los dos grupos de palabras
    diff = ndiff(raw.split(), corr.split())
    # Crea lista donde deja solo los que tienen cambios
    cambios = [d for d in diff if d.startswith('- ') or d.startswith('+ ')]
    return cambios

# Guarda en lista todos los errores encontrados
errores = []
for i in range(len(dfGR)):
    # Aplica funcion
    cambios = extraer_cambios(dfGR.loc[i, "transcript_raw"], dfGR.loc[i, "transcript_raw_corr"])
    # Concatena lo encontrado
    errores.extend(cambios)
# Cuenta los elementos repetidos en lista
conteo_errores = Counter(errores)
print(conteo_errores.most_common(10))



# Diccionario para almacenar resultados
conteo = {}
for item, count in conteo_errores.items():
    tipo = item[0]  # '+' o '-'
    palabra = item[2:]  # quita el signo y espacio
    if palabra not in conteo:
        conteo[palabra] = {"numero de adiciones": 0, "numero de eliminaciones": 0}
    if tipo == '+':
        conteo[palabra]["numero de adiciones"] += count
    elif tipo == '-':
        conteo[palabra]["numero de eliminaciones"] += count
# Crear DataFrame
df = pd.DataFrame.from_dict(conteo, orient='index').reset_index()
df = df.rename(columns={"index": "palabra"})
# Crear columna de correcciones como suma de adiciones y eliminaciones
df["numero de correcciones"] = df["numero de adiciones"] + df["numero de eliminaciones"]
# Reorganiza columnas
df = df[["palabra", "numero de correcciones", "numero de adiciones", "numero de eliminaciones"]]
df = df.reset_index().sort_values('numero de correcciones', ascending=False)
print(df)



# ------------------------------------------------------- generacion requiriments -----------------------

import subprocess

# Ejecuta pip freeze y guarda la salida en requirements.txt
with open('requirements.txt', 'w') as f:
    subprocess.run(['pip', 'freeze'], stdout=f)

print("✅ Archivo requirements.txt generado con éxito.")


