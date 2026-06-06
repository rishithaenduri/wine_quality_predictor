# wine_quality_predictor
Wine Quality Prediction using Machine Learning and Python
Wine Quality Prediction Project

This is a Machine Learning project that predicts the quality of wine based on its chemical properties. The model is trained using Python and Scikit-learn and provides accurate predictions using a trained dataset.

# Project Description

The goal of this project is to analyze different physicochemical features of wine and predict its quality score. It uses a trained machine learning model along with a scaler for preprocessing input data.

A simple HTML-based interface is also included for user-friendly predictions.

# Project Files
model_info.json → Contains model details and metadata
train_model.py → Python script used to train the model
wine_model.pkl → Saved trained machine learning model
wine_scaler.pkl → Saved scaler for feature normalization
wine_quality_predictor.html → Frontend interface for prediction

#Technologies Used
Python
Pandas
NumPy
Scikit-learn
Pickle
HTML

# How It Works
Input wine features are collected from the user
Data is preprocessed using the saved scaler
The trained model predicts wine quality
Output is displayed instantly on the interface

# How to Run
pip install -r requirements.txt
python train_model.py

Then open:

wine_quality_predictor.html

in your browser.

# Output

The system predicts a wine quality score based on input chemical properties like acidity, alcohol, and sugar levels.
