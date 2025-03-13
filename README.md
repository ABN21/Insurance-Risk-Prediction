# Insurance-Risk-Prediction
This project implements an Artificial Neural Network (ANN) for vehicle insurance risk prediction using real-time driving behavior data. The model classifies risk into Low, Moderate, and High categories based on driving parameters such as speed, braking, acceleration, engine load, fuel consumption, mileage, engine RPM, throttle position, and past accident history.

Key Features

 Risk Prediction Model: Trained on a dataset of 100,000 records using ANN with dropout layers for regularization.
 Data Preprocessing: Includes feature standardization, class balancing (SMOTE), and label encoding.
 Model Performance Visualization: Generates key plots, including Confusion Matrix, Loss & Accuracy Curves, SHAP Feature Importance, and ROC Curve.
 Explainability with SHAP: Provides insights into how each driving factor impacts risk classification.
 SMOTE for Imbalance Handling: Ensures effective learning by balancing risk category distributions.
 Premium Estimation: Maps predicted risk categories to dynamic insurance pricing.

Visualizations Included 

 Confusion Matrix (Model Predictions)
 Loss & Accuracy Curves (Training Process)
 Feature Distribution Histograms (Driving Factors)
 Pairplots & Violin Plots (Feature Relationships)
 SHAP Summary Plot (Feature Importance)
 ROC & Precision-Recall Curves (Model Evaluation)

Technology Stack

 Python, TensorFlow, Scikit-Learn, Pandas, NumPy
 Matplotlib, Seaborn, SHAP for Model Explainability
