# Speech-Based Cognitive Fatigue Detection for Hiligaynon Speakers

An undergraduate thesis project presented to the Faculty of the College of Information and Communications Technology, West Visayas State University (WVSU), Iloilo City.

## 📌 Project Overview
This repository implements an automated, non-invasive system to detect cognitive fatigue levels (Low, Moderate, High) in native Hiligaynon speakers using acoustic features. Speech signals are standardized to 16 kHz mono WAV, converted to 128-bin Mel-spectrograms, and classified through a dual-block Conv2D network equipped with a Temporal Attention mechanism.

## 👥 Authors
* Arthur John A. Pagayon
* Bonn Jouie P. Sasana
* Zen Anthony P. Pastolero

## 🛠️ Tech Stack
* **Frontend & Dashboard:** Streamlit (Multi-page app with `st.navigation`)
* **Acoustic Signal Processing:** Librosa, Pydub, SoundFile, FFmpeg
* **Deep Learning Engine:** TensorFlow / Keras (Conv2D + Temporal Attention)
* **Backend & Cloud Database:** Supabase (PostgreSQL & `audio-recordings` bucket)

## 🚀 Local Development Setup

### 1. Clone Repository & Create Virtual Environment
```powershell
git clone [https://github.com/ZenzenAnthony/Speech-Based-Cognitive-Fatigue-Detection-for-Hiligaynon-Speaker.git](https://github.com/ZenzenAnthony/Speech-Based-Cognitive-Fatigue-Detection-for-Hiligaynon-Speaker.git)
cd Speech-Based-Cognitive-Fatigue-Detection-for-Hiligaynon-Speaker
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Enable Researcher Diagnostic Studio inference
Place the trained `fatigue_cnn_attention.keras` model in the project root. The Researcher Diagnostic Studio will then show the predicted fatigue class, confidence scores for Low/Moderate/High, and a temporal-attention plot after audio analysis completes. To use a model stored elsewhere, set `FATIGUE_MODEL_PATH` to its path before starting Streamlit. Model files are excluded from Git, so they must be provided separately.