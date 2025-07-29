# Audio Transcription, Grammar Correction, and Analysis Pipeline
>This document presents the description of the solution, architecture, and main considerations and steps required to run and install the pipeline for audio transcription, grammar correction, and error analysis.

Additional documentation and resources related to the project:

[Technical Report (PDF)](docs/Test%20report%20Andres%20Parra.pdf)

[Data Folder](data/)

[Development Notebook](src/Notebook_test_andres_parra.ipynb)

[Demo Video](Pending)

---

## Table of Contents
* [Solution Description](#solution-description)
* [Logical Architecture](#logical-architecture)
* [Project Structure](#project-structure)
* [Execution Instructions](#execution-instructions)
* [Requirements](#requirements)
* [Authors](#authors)

---

## Solution Description

### General Objective
- Implement a complete pipeline for **automatic transcription**, **grammar correction**, and **error analysis** of an educational audio file, evaluating transcription quality with linguistic metrics and proposing pedagogical improvements.

### Specific Objectives
- Generate automatic transcriptions using **Whisper** and **WhisperX**.
- Create a **Gold Set** for performance evaluation.
- Apply LLM models (T5, BART) for grammar correction.
- Measure performance using **WER** and **CER** before and after correction.
- Identify frequent grammatical errors and define business rules for targeted exercises.

### Challenge
Evaluate and improve the quality of automatic transcriptions of a 5-minute educational audio file, generating insights to support **pedagogical strategies** and the detection of common mistakes.

### Solution
Development of a modular pipeline that:
- Segments and processes the audio.
- Generates automatic transcriptions.
- Applies grammar correction using pre-trained models.
- Evaluates improvement with quantitative metrics.
- Extracts common error patterns and recommends targeted exercises.

### Potential Impact
- Reduction in manual review time.
- Improved transcription accuracy for educational contexts.
- Possible integration into **personalized learning systems**.

---

## Logical Architecture
**Input Data**
- Full audio file (`audio_full.m4a`) and audio fragments (`audio_parte_*.m4a`).
- Gold reference transcript (`transcript_gold.csv`).

**Pipeline Steps**
1. **Automatic Transcription** using Whisper and WhisperX.
2. **Grammar Correction** using T5 and BART models.
3. **Evaluation** of accuracy using WER and CER.
4. **Error Analysis** to generate targeted recommendations.
5. **Optional Diarization** for speaker segmentation.

**Output**
- Corrected transcriptions (`transcript_corrected.csv`).
- Performance metrics.
- Technical report (`Test report Andres Parra.pdf`).

---


## Project Structure

```
.
├── .gitignore                 # Git ignore file
├── HFtoken.txt                # Hugging Face API token
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
│
├── data                       # Input data and gold set
│   ├── audio_full.m4a          # Original audio (5 minutes)
│   ├── transcript_gold.csv     # Manual reference transcription (gold set)
│   └── transcript_raw.csv      # Raw automatic transcription
│
├── docs                       # Documentation and reports
│   └── Test report Andres Parra.pdf
│
├── fragmentos                 # Audio fragments used for testing
│   └── audio_parte_*.m4a
│
├── src                        # Source code and notebooks
│   ├── development.py
│   └── Notebook_test_andres_parra.ipynb
│
└── temp                       # Temporary files
    └── fragmentos_temp.m4a
```

---

## Execution Instructions

### Option 1: Run with Python (Virtual Environment)

1. **Create a virtual environment:**
    ```bash
    conda create --name audio_env python=3.9
    conda activate audio_env
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the pipeline:**
    ```bash
    python src/development.py
    ```

4. **Or run from Jupyter Notebook:**
    ```bash
    jupyter notebook src/Notebook_test_andres_parra.ipynb
    ```

---

### Option 2: Run with Docker (Optional)

1. **Build Docker image:**
    ```bash
    docker build -t audio_pipeline .
    ```

2. **Run the container:**
    ```bash
    docker run -it --name audio_pipeline audio_pipeline
    ```

---

## Requirements

### Hardware
- **RAM**: Minimum 8 GB  
- **GPU**: NVIDIA GPU (optional, recommended for acceleration)  
- **Disk space**: 5 GB  

### Software
- Python 3.9  
- Conda or virtual environment  
- Docker (optional)  

### Python Libraries
- pandas  
- numpy  
- torch  
- transformers  
- librosa  
- jiwer  
- seaborn  

---

## Authors

| Organization       | Name                   | Role            | Contact |
|--------------------|------------------------|-----------------|---------|
| Individual Project | Andres Parra Rodriguez | Data Scientist  | [LinkedIn](https://www.linkedin.com/in/andresparrarod/) |