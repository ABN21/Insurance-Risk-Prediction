import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Create directory for saving plots
os.makedirs("plots", exist_ok=True)

# Function to categorize risk levels based on original features
def categorize_risk(row):
    score = 0
    if row["Speed"] > 100:  
        score += 2
    if row["Braking"] > 3:
        score += 2
    if row["Acceleration"] > 4:
        score += 2
    if row["Fuel_Consumption"] > 10:
        score += 1
    if row["Engine_Load"] > 70:
        score += 1
    if row["Engine_RPM"] > 5000:
        score += 1
    if row["Throttle_Position"] > 70:
        score += 1
    if row["Mileage"] < 8:
        score += 1
    if row["Past_Accidents"] > 2:
        score += 2

    if score >= 6:
        return "High"
    elif score >= 3:
        return "Moderate"
    else:
        return "Low"

# Load dataset
df = pd.read_csv("vehicle_risk_dataset_100k_realistic-2.csv")

# Handle missing values (Fill NaN with median)
df.fillna(df.median(), inplace=True)

# Apply risk categorization
df["Risk_Label"] = df.apply(categorize_risk, axis=1)

# Define original features (without interaction terms)
features = ["Speed", "Braking", "Acceleration", "Engine_RPM", "Throttle_Position",
            "Fuel_Consumption", "Engine_Load", "Mileage", "Past_Accidents"]

# Split features and target
X = df[features].values
y = df["Risk_Label"].values

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)  # 0 = Low, 1 = Moderate, 2 = High
joblib.dump(le, "le_risk_no_interaction.pkl")

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "scaler_no_interaction.pkl")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance High-risk class
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Manually adjust class weights
class_weight_dict = {0: 1, 1: 2, 2: 6}

# Define ANN model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with class weights
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                    class_weight=class_weight_dict)

# Save the trained model
model.save("new_users_risk_model_no_interaction.keras")

# Model Predictions
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# Apply threshold to force High-risk predictions
y_pred[np.where(y_pred_proba[:, 2] > 0.3)] = 2

# Compute Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Save Accuracy & Classification Report
with open("plots/model_performance.txt", "w") as f:
    f.write(f"Model Accuracy: {accuracy * 100:.2f}%\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("plots/confusion_matrix.png")
plt.close()

# Compute model performance metrics
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Define metrics and values
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
values = [accuracy, precision, recall, f1]

# Generate the bar chart with fixed Seaborn settings
plt.figure(figsize=(6, 4))
sns.barplot(x=metrics, y=values, hue=metrics, palette="viridis", legend=False)  # Fixed Seaborn warning
plt.ylim(0, 1)
plt.xlabel("Metric")
plt.ylabel("Score")
plt.title("Model Performance Metrics")
plt.savefig("plots/model_accuracy_bar.png")
plt.close()

# SHAP Feature Importance Plot - Fixed for TensorFlow/Keras
import shap
background = X_train[:100]  # Use a subset of training data as background
explainer = shap.Explainer(model, background)  # SHAP explainer for deep learning
shap_values = explainer(X_test[:100])  # Get SHAP values for test set

plt.figure(figsize=(8, 6))
# For TensorFlow models, we need to handle the output structure differently
# This fixes the "only integer scalar arrays can be converted to a scalar index" error
if hasattr(shap_values, 'values'):
    # For newer SHAP versions with multi-class models
    # Show feature importance for high risk class (index 2)
    shap.summary_plot(shap_values[:,:,2], X_test[:100], feature_names=features, show=False)
else:
    # For older SHAP versions or different output structure
    shap.summary_plot(shap_values, X_test[:100], feature_names=features, show=False)

plt.savefig("plots/shap_feature_importance.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.savefig("plots/roc_curve.png")
plt.close()

# Plot Training Loss & Accuracy
plt.figure(figsize=(10, 5))

# Loss Curve
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title("Model Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title("Model Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("plots/loss_accuracy_curves.png")
plt.close()

# Pairplot to visualize feature relationships with risk categories
sns.pairplot(df[features + ["Risk_Label"]], hue="Risk_Label", palette="coolwarm")
plt.savefig("plots/pairplot_features.png")
plt.close()


# Feature Distribution - Histogram
plt.figure(figsize=(12, 8))
for i, col in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.histplot(df[col], kde=True, bins=30, color='blue')
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig("plots/feature_distribution.png")
plt.close()


# Print final message
print("\n✅ Model training complete and saved (without interaction features).")
print(f"✅ Model Accuracy: {accuracy * 100:.2f}% saved in model_performance.txt")



























