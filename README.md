# AI-Based Intrusion Detection System (UNSW-NB15)

A machine learning-based Intrusion Detection System (IDS) for classifying network traffic into attack categories using the UNSW-NB15 dataset.  
The project includes both a training notebook and an interactive Streamlit application.

## Project Highlights
- Multi-class intrusion detection using trained ML pipeline artifacts.
- Streamlit app for interactive feature input and prediction.
- Dashboard views for session-level traffic and prediction insights.
- Model performance reference panel and attack knowledge base.
- Reproducible dependency setup with a single `requirements.txt`.

## App Features
The Streamlit app (`app.py`) currently includes:
- `Predict` tab
  - Manual feature input (numeric + categorical)
  - Preset traffic profiles
  - Model prediction + model confidence
- `Dashboard` tab
  - Session prediction summary
  - Traffic trend and prediction distribution views
- `Model Performance` tab
  - Reference evaluation metrics from notebook workflow
- `Attack Knowledge Base` tab
  - Class severity, descriptions, and recommended actions

## Tech Stack
- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- XGBoost
- imbalanced-learn
- Matplotlib, Seaborn
- Joblib

## Repository Structure
```text
.
├── app.py
├── ai-based-ids.ipynb
├── requirements.txt
├── unsw_pipeline_xgb.pkl
├── unsw_label_encoder.pkl
├── label_classes.pkl
└── (dataset files are downloaded locally; not tracked in Git)
```

## Dataset
This repository does not track large raw CSV files.  
Download the UNSW-NB15 data locally and place it in your workspace before running notebook training.

Suggested source:
- UNSW-NB15 official project page: [https://research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

## Setup
### 1. Create and activate a virtual environment

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## Run the App
```bash
streamlit run app.py
```

## Training / Experiment Notebook
Use the notebook below for model development and experimentation:
- `ai-based-ids.ipynb`

## Notes
- This project performs ML-based classification on traffic features; it is not a full production SOC pipeline.
- Model confidence is a probability estimate, not a guarantee of correctness.
- Large datasets are intentionally excluded from version control for GitHub compatibility.

## Author
Syed Asir Hassan  
[GitHub](https://github.com/AsirHassan)
