# Voice-Based Gender & Age Classification

A machine learning project that predicts a speaker's **gender** and **age group** from raw audio recordings, using classical audio signal processing and traditional ML classifiers (Random Forest, KNN, SVM).

Built on the [Mozilla Common Voice](https://www.kaggle.com/datasets/mozillaorg/common-voice) dataset.

## Overview

The pipeline:
1. Loads a sample of labeled voice clips (gender + age) from the Common Voice dataset.
2. Cleans and enhances each audio clip (trimming silence, low-pass filtering, gain normalization).
3. Extracts acoustic features (MFCCs, pitch, zero-crossing rate, spectral centroid).
4. Trains and evaluates multiple classifiers for each task:
   - **Gender classification**: Random Forest, KNN, SVM
   - **Age classification**: Random Forest
5. Reports accuracy, confusion matrices, and classification reports for each model.

## Features Extracted

For every audio clip, the following features are computed and concatenated into a single feature vector:

- **MFCCs** (13 coefficients): mean, standard deviation, and skewness
- **Pitch**: mean and standard deviation (via `librosa.piptrack`, filtered by magnitude confidence)
- **Zero-crossing rate**
- **Spectral centroid**

Audio preprocessing steps applied before feature extraction:
- Silence trimming (`librosa.effects.trim`)
- Low-pass Butterworth filter for ambient noise reduction (cutoff = 3000 Hz)
- Vocal enhancement via peak amplitude normalization

## Models

| Task   | Algorithms Compared            |
|--------|---------------------------------|
| Gender | Random Forest, KNN, SVM        |
| Age    | Random Forest                  |

Trained models are cached to disk (`.pkl` files via `joblib`) so re-running the notebook skips retraining if a saved model already exists.

## Project Structure

```
.
├── Final_code.ipynb                        # Main notebook (data prep, feature extraction, training, evaluation)
├── cv-valid-train.csv                  # Common Voice metadata (not included — download separately)
├── cv-valid-train/                     # Folder of .mp3 audio clips (not included — download separately)
├── gender classification models/       # Saved gender models (.pkl)
└── age classification models/          # Saved age models (.pkl)
```

## Requirements

- Python 3.x
- `librosa`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `python_speech_features`
- `scikit-learn`
- `joblib`

## Dataset

This project uses the **Mozilla Common Voice** dataset, available on Kaggle:
[https://www.kaggle.com/datasets/mozillaorg/common-voice](https://www.kaggle.com/datasets/mozillaorg/common-voice)

Download the dataset and place `cv-valid-train.csv` and the `cv-valid-train/` audio folder in the project root before running the notebook.

## Notes

- If a saved model `.pkl` file already exists for a given model type/method, it will be loaded instead of retrained. Delete the corresponding file to force retraining.
- Feature extraction can take a while depending on `NUM_SAMPLES` and hardware, since it processes each audio file individually.
