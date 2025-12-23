# Spam Detection (TF-IDF + Random Forest) — Streamlit App

This repository contains an end-to-end **spam detection system** for Indonesian email/text data using **TF-IDF feature extraction** and a **Random Forest classifier**.  
The project covers data preprocessing, exploratory data analysis (EDA), model training, and deployment as an interactive **Streamlit web application**.

## Live Demo
The application has been deployed and can be accessed here:  
https://rf-spam-detection.streamlit.app/

---

## Project Description
Spam detection is a fundamental task in Natural Language Processing (NLP) that aims to identify and filter unwanted or malicious messages.  
In this project, raw email text is cleaned using custom preprocessing rules, transformed into numerical features using TF-IDF, and classified using a Random Forest model.

This project is designed as an **educational and portfolio project** demonstrating a complete NLP workflow from raw data to a deployed web application.

---

## Project Overview
System workflow:
1. User inputs an email or message text via the Streamlit interface.
2. Text preprocessing is applied using a custom preprocessing module.
3. The cleaned text is converted into TF-IDF features.
4. A trained Random Forest model predicts whether the message is spam or non-spam.
5. The prediction result is displayed to the user.

Main components:
- Dataset for training and analysis
- EDA notebook for data exploration
- Training script for model building
- Pre-trained model artifacts
- Streamlit app for inference

---

## System Architecture

### Inference Pipeline
User Input Text  
→ Text Preprocessing (`src/preprocessing.py`)  
→ TF-IDF Vectorizer (`tfidf_vectorizer.pkl`)  
→ Random Forest Model (`rf_model.pkl`)  
→ Prediction Result (Spam / Non-Spam)  
→ Streamlit Web Interface  

### Training Pipeline
- Load dataset from CSV file
- Text cleaning and normalization
- TF-IDF feature extraction
- Random Forest model training
- Model and vectorizer serialization as `.pkl` files

Note: The Streamlit application **does not retrain the model**. It only loads the trained artifacts for inference.

---

## Project Structure
```
.
├── data/
│   └── email_spam_indo.csv
├── src/
│   ├── preprocessing.py
│   └── __pycache__/
├── app.py
├── train.py
├── eda.ipynb
├── rf_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md
```

---

## How to Run Locally

### 1) Prerequisites
- Python 3.9 or newer
- pip
- (Optional) Conda or virtual environment

---

### 2) Clone the Repository
```bash
git clone https://github.com/freyzaza/spam-detection-random-forest.git
cd spam-detection-random-forest
```

---

### 3) Create and Activate a Virtual Environment (Optional but Recommended)

Using `venv`:
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\Activate.ps1 # Windows PowerShell
```

Using Conda:
```bash
conda create -n spam-detection-random-forest python=3.10 -y
conda activate spam-detection-random-forest
```

---

### 4) Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 5) Run the Streamlit Application
```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`) in your browser.

---

## (Optional) Retrain the Model
To retrain the model with the provided dataset:
1. Ensure `data/email_spam_indo.csv` is available.
2. Run the training script:
   ```bash
   python train.py
   ```
3. This will regenerate:
   - `rf_model.pkl`
   - `tfidf_vectorizer.pkl`
4. Restart the Streamlit application to use the updated model.

---

## Notes
- The application supports adjustable spam detection strictness via predefined modes (Light, Moderate, Strict).
- This project demonstrates a complete NLP pipeline suitable for academic assignments or portfolio showcases.
