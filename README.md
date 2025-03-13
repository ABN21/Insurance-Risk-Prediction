# Insurance-Risk-Prediction
CNN-Based Vehicle Insurance Risk Prediction

This repository contains a Convolutional Neural Network (CNN) model designed to predict vehicle insurance risk based on real-time driving data and historical claims. The model analyzes various risk factors, including driving behavior and vehicle diagnostics, to provide an accurate risk assessment.

📌 Key Features

Deep Learning with CNN: Utilizes convolutional layers for feature extraction and risk prediction.
Driving Score Calculation: Assigns a risk score based on real-time and past driving data.
Integration with Digital Twin: Maps historical driving patterns to generate personalized risk insights.
SHAP Explainability: Uses SHAP (SHapley Additive Explanations) to interpret model predictions.
Automated Risk Analysis: Categorizes drivers into low, moderate, and high-risk segments.
📊 Flowchart Representation

The CNN workflow follows these steps:

Input Data – Collects driving behavior and vehicle sensor data.
Preprocessing – Cleans and normalizes data for consistency.
Feature Extraction – CNN extracts important driving risk factors.
Risk Classification – Predicts insurance risk category (low/medium/high).
Explainability & Output – Provides factor-wise risk breakdown using SHAP.
