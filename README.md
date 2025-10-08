# Skeletal Points Project

This project provides a pipeline for processing skeletal point data, extracting features, labeling data, combining labeled datasets, training a model, and making predictions. Follow the steps below to use your own dataset.

---

## 1. Preprocessing: `skel.py`

**Purpose:**  
Extracts skeletal points from raw data.

- **Input Folder:**  
    `input/`
- **Output Folder:**  
    `output csv/landmarks_<vidname>.csv`

**Usage:**  
```bash
python skel.py
```

---

## 2. Feature Extraction: `feat.py`

**Purpose:**  
Extracts features from skeletal point data.

- **Input Folder:**  
    `output csv/`
- **Output Folder:**  
    `features/features_landmarks_<vidname>.csv`

**Usage:**  
```bash
python feat.py
```

---

## 3. Labeling: `label.py`

**Purpose:**  
Labels the extracted features for supervised learning.

- **Input Video:** Enter the video name which you want to label in the script, e.g.,  
    `videoname.mp4`
- **Output Folder:**  
    `labeled/videoname.csv`

**Usage:**  
```bash
python label.py
```

---

## 4. Concatenation: `concat.py`

**Purpose:**  
Combines all labeled CSV files in labeled folder into a single dataset.

- **Input Folder:**  
    `labeled`
- **Output File:**  
    `combined labeled/all_labeled.csv`

**Usage:**  
```bash
python concat.py
```

---

## 5. Training: `train.py`

**Purpose:**  
Trains a model using the combined labeled data.

- **Input File:**  
    `combined labeled/all_labeled.csv`
- **Output File:**  
    `rf_model.pkl`

**Usage:**  
```bash
python train.py
```

---

## 6. Prediction: `predict.py`

**Purpose:**  
Makes predictions on new data using the trained model.

- **Input Folder:**  
    `data/feat/` (or your new feature data)
- **Model File:**  
    `models/model.pkl`
- **Output Folder:**  
    `data/predictions/`

**Usage:**  
```bash
python predict.py
```

---

## Notes

- Ensure all input/output folders exist before running each script.
- Replace the contents of `data/raw/` with your own dataset to use this pipeline.
- Adjust script parameters as needed for your data format.
