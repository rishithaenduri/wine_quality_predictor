# 🍷 Wine Quality Predictor

## Project Structure

```text
wine-quality-predictor/
├── train_model.py                 # Model training script
├── wine_model.pkl                 # Trained machine learning model
├── wine_scaler.pkl                # Saved feature scaler
├── model_info.json                # Model metadata and information
├── wine_quality_predictor.html    # User interface for predictions
└── README.md
```

---

## Project Overview

Wine Quality Predictor is a Machine Learning project designed to estimate wine quality based on its physicochemical characteristics. The project includes a complete prediction pipeline consisting of data preprocessing, feature scaling, model training, and an interactive user interface for generating predictions.

The trained model and preprocessing artifacts are stored separately, allowing predictions to be performed efficiently without retraining the model.

---

## Setup & Run

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Train the Model

Run the training script:

```bash
python train_model.py
```

This generates the trained model and preprocessing artifacts:

```text
wine_model.pkl
wine_scaler.pkl
model_info.json
```

### Step 3 — Open the Application

Open:

```text
wine_quality_predictor.html
```

in your preferred web browser.

### Step 4 — Predict Wine Quality

Enter the required wine characteristics and submit the form to receive a predicted wine quality result.

---

## Machine Learning Pipeline

| Step             | Description                                 |
| ---------------- | ------------------------------------------- |
| Data Preparation | Load and prepare wine quality data          |
| Preprocessing    | Apply required transformations and scaling  |
| Model Training   | Train the machine learning model            |
| Evaluation       | Assess model performance                    |
| Model Saving     | Store trained model and scaler              |
| Prediction       | Generate wine quality predictions           |
| User Interface   | Display predictions through a web interface |

---

## Model Files

| File            | Description                                      |
| --------------- | ------------------------------------------------ |
| wine_model.pkl  | Trained machine learning model                   |
| wine_scaler.pkl | Feature scaling object used during preprocessing |
| model_info.json | Metadata and model-related information           |

---

## Features

* Wine quality prediction using Machine Learning
* Data preprocessing and feature scaling
* Saved model for efficient inference
* Interactive user interface
* Reusable prediction pipeline
* Lightweight and easy-to-use implementation

---

## Technologies Used

* Python
* Machine Learning
* Scikit-Learn
* NumPy
* Pandas
* Pickle
* HTML
* CSS
* JavaScript
* JSON

---

## Applications

* Wine quality assessment
* Machine learning experimentation
* Predictive analytics projects
* Educational data science projects
* End-to-end ML workflow demonstrations

---

