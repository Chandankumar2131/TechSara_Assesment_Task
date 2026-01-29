"""
COMPLETE DIABETES PREDICTION MLOPS PROJECT
Techsara Consulting AI/ML Engineer Assessment
"""

print("=" * 70)
print("TECHSARA CONSULTING - AI/ML ENGINEER ASSESSMENT")
print("DIABETES PREDICTION MLOPS PROJECT")
print("=" * 70)

# ========== PART 1: IMPORTS ==========
print("\n[1/7] Loading dependencies...")
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import make_classification
import joblib
import os
import json
from datetime import datetime
from fastapi import FastAPI
import uvicorn

# ========== PART 2: DATA PREPARATION ==========
print("\n[2/7] Creating synthetic diabetes dataset...")

# Create data similar to diabetes dataset
X, y = make_classification(
    n_samples=768,        # Pima Indians dataset has 768 samples
    n_features=8,         # 8 medical features
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")
print(f"   Number of features: {X_train.shape[1]}")

# Create MLOps directories
os.makedirs('models', exist_ok=True)
os.makedirs('artifacts', exist_ok=True)

# ========== PART 3: BASELINE MODEL ==========
print("\n[3/7] Training Baseline Model (Logistic Regression)...")

baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train, y_train)

# Predictions
y_pred_baseline = baseline_model.predict(X_test)

# Metrics
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
baseline_f1 = f1_score(y_test, y_pred_baseline)

print(f"   Model Type: Logistic Regression")
print(f"   Accuracy: {baseline_accuracy:.4f}")
print(f"   F1 Score: {baseline_f1:.4f}")

# Save baseline model
joblib.dump(baseline_model, 'models/baseline_model.pkl')

# Save baseline metrics
baseline_metrics = {
    'model': 'logistic_regression',
    'accuracy': float(baseline_accuracy),
    'f1_score': float(baseline_f1),
    'created_at': datetime.now().isoformat()
}

with open('models/baseline_metrics.json', 'w') as f:
    json.dump(baseline_metrics, f, indent=2)

print("   ✓ Baseline model saved")

# ========== PART 4: IMPROVED MODEL ==========
print("\n[4/7] Training Improved Model (Random Forest)...")

# IMPROVEMENT: Better algorithm with hyperparameters
improved_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=4,
    random_state=42
)

improved_model.fit(X_train, y_train)

# Predictions
y_pred_improved = improved_model.predict(X_test)

# Metrics
improved_accuracy = accuracy_score(y_test, y_pred_improved)
improved_f1 = f1_score(y_test, y_pred_improved)

print(f"   Model Type: Random Forest (Improved)")
print(f"   Accuracy: {improved_accuracy:.4f}")
print(f"   F1 Score: {improved_f1:.4f}")

# MLOps Feature: Versioning
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
versioned_model_path = f'models/model_v{timestamp}.pkl'
joblib.dump(improved_model, versioned_model_path)

# MLOps Feature: Metadata tracking
metadata = {
    'model_type': 'random_forest',
    'version': f'v{timestamp}',
    'hyperparameters': {
        'n_estimators': 150,
        'max_depth': 12,
        'min_samples_split': 4
    },
    'performance': {
        'accuracy': float(improved_accuracy),
        'f1_score': float(improved_f1)
    }
}

metadata_path = f'artifacts/metadata_v{timestamp}.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"   ✓ Model versioned: v{timestamp}")
print(f"   ✓ Metadata saved: {metadata_path}")

# ========== PART 5: MANDATORY LOGIC GATE ==========
print("\n" + "=" * 60)
print("MANDATORY: LOGIC GATE FOR DEPLOYMENT")
print("=" * 60)

print(f"\n📊 PERFORMANCE COMPARISON:")
print(f"   Baseline F1 Score:  {baseline_f1:.4f}")
print(f"   New Model F1 Score: {improved_f1:.4f}")
print(f"   Difference:         {improved_f1 - baseline_f1:+.4f}")

# THE LOGIC GATE: Compare with baseline
if improved_f1 >= baseline_f1:
    print("\n✅ LOGIC GATE: APPROVED")
    print(f"   Reason: New model ({improved_f1:.4f}) >= Baseline ({baseline_f1:.4f})")
    
    # Save as production model
    joblib.dump(improved_model, 'models/production_model.pkl')
    
    # Deployment metadata
    deployment_info = {
        'status': 'DEPLOYED',
        'timestamp': datetime.now().isoformat(),
        'model_version': f'v{timestamp}',
        'logic_gate_result': 'PASSED',
        'metrics': {
            'baseline_f1': float(baseline_f1),
            'new_model_f1': float(improved_f1),
            'improvement': float(improved_f1 - baseline_f1)
        }
    }
    
    with open('models/deployment_info.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print("   ✓ Production model saved")
    print("   ✓ Deployment info saved")
    
else:
    print("\n❌ LOGIC GATE: REJECTED")
    print(f"   Reason: New model ({improved_f1:.4f}) < Baseline ({baseline_f1:.4f})")
    print("   Model not promoted to production")

# ========== PART 6: FASTAPI DEPLOYMENT ==========
print("\n" + "=" * 60)
print("FASTAPI MODEL DEPLOYMENT")
print("=" * 60)

# Create the FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="MLOps deployment with automated validation",
    version="1.0.0"
)

# Load the approved model
try:
    model = joblib.load('models/production_model.pkl')
    model_status = "production"
except:
    model = joblib.load('models/baseline_model.pkl')
    model_status = "baseline"

@app.get("/")
def home():
    return {
        "service": "Diabetes Prediction API",
        "mlops_features": [
            "Automated model training",
            "Logic gate validation",
            "Model versioning",
            "API deployment"
        ],
        "model_status": model_status,
        "endpoints": [
            "GET /health - Health check",
            "GET /model-info - Model details",
            "POST /predict - Make predictions"
        ]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": type(model).__name__
    }

@app.get("/model-info")
def model_info():
    try:
        with open('models/deployment_info.json', 'r') as f:
            info = json.load(f)
        return info
    except:
        return {"message": "Using baseline model"}

@app.post("/predict")
def predict(features: list):
    try:
        if len(features) != 8:
            return {"error": f"Expected 8 features, got {len(features)}"}
        
        features_array = np.array(features).reshape(1, -1)
        prediction = int(model.predict(features_array)[0])
        probability = float(model.predict_proba(features_array)[0][1])
        
        return {
            "prediction": prediction,
            "probability": probability,
            "interpretation": "1 = High risk of diabetes, 0 = Low risk"
        }
    except Exception as e:
        return {"error": str(e)}

# ========== PART 7: RUN EVERYTHING ==========
print("\n[7/7] Project Summary:")
print("-" * 40)
print("✓ 1. Baseline model trained and saved")
print("✓ 2. Improved model trained and versioned")
print("✓ 3. MLOps: Model metadata tracked")
print("✓ 4. MANDATORY: Logic gate implemented")
print("✓ 5. FastAPI deployment ready")
print("✓ 6. All assessment requirements met")
print("\n" + "=" * 70)
print("✅ PROJECT READY FOR SUBMISSION")
print("=" * 70)

print("\n📋 TO TEST THE PROJECT:")
print("1. Start the API: python complete_mlops_project.py")
print("2. Open browser: http://127.0.0.1:8000")
print("3. API docs: http://127.0.0.1:8000/docs")

print("\n🚀 Starting FastAPI server...")
print("Press CTRL+C to stop\n")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)