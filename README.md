# 🏥 Diabetes Prediction MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![MLOps](https://img.shields.io/badge/MLOps-Complete-FF6B6B)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Assessment-Techsara%20Consulting-blue)

A comprehensive MLOps pipeline for diabetes prediction, implementing industry best practices and automated deployment workflows. Built for the **Techsara Consulting AI/ML Engineer Assessment**.

## 🎯 Key Features

### 🤖 **Model Development**
- **Baseline Model**: Logistic Regression for comparison
- **Improved Model**: Random Forest with hyperparameter tuning
- **Performance Tracking**: Accuracy, F1-Score, and classification reports

### 🔧 **MLOps Implementation**
- **Model Versioning**: Automatic timestamp-based version control
- **Metadata Tracking**: JSON metadata for full reproducibility
- **Artifact Management**: Organized storage in `models/` and `artifacts/` directories
- **Automated Pipeline**: End-to-end training to deployment workflow

### 🚦 **Mandatory Logic Gate** ✅
- **Condition**: New model F1-score ≥ Production baseline F1-score
- **Automated Decision**: Approve/Reject deployment automatically
- **Audit Trail**: Detailed deployment logs with comparison metrics

### 🚀 **Production Deployment**
- **FastAPI**: High-performance REST API with async support
- **Swagger UI**: Interactive API documentation at `/docs`
- **Health Checks**: System monitoring endpoints
- **Validation**: Input validation and error handling

## 📊 Results Dashboard

| Model | Algorithm | Accuracy | F1-Score | Status |
|-------|-----------|----------|----------|--------|
| Baseline | Logistic Regression | 87.66% | 0.8571 | Reference |
| Improved | Random Forest | **92.86%** | **0.9197** | ✅ **Deployed** |

### 🔬 Performance Improvement
- **Accuracy**: +5.20% improvement
- **F1-Score**: +7.26% improvement  
- **Logic Gate**: ✅ **APPROVED** (0.9197 ≥ 0.8571)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation & Running

```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/techsara-mlops-assessment.git
cd techsara-mlops-assessment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the complete MLOps pipeline
python complete_mlops_project.py

# 🏥 Diabetes Prediction MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![MLOps](https://img.shields.io/badge/MLOps-Complete-FF6B6B)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Assessment-Techsara%20Consulting-blue)

A comprehensive MLOps pipeline for diabetes prediction, implementing industry best practices and automated deployment workflows. Built for the **Techsara Consulting AI/ML Engineer Assessment**.

## 🎯 Key Features

### 🤖 **Model Development**
- **Baseline Model**: Logistic Regression for comparison
- **Improved Model**: Random Forest with hyperparameter tuning
- **Performance Tracking**: Accuracy, F1-Score, and classification reports

### 🔧 **MLOps Implementation**
- **Model Versioning**: Automatic timestamp-based version control
- **Metadata Tracking**: JSON metadata for full reproducibility
- **Artifact Management**: Organized storage in `models/` and `artifacts/` directories
- **Automated Pipeline**: End-to-end training to deployment workflow

### 🚦 **Mandatory Logic Gate** ✅
- **Condition**: New model F1-score ≥ Production baseline F1-score
- **Automated Decision**: Approve/Reject deployment automatically
- **Audit Trail**: Detailed deployment logs with comparison metrics

### 🚀 **Production Deployment**
- **FastAPI**: High-performance REST API with async support
- **Swagger UI**: Interactive API documentation at `/docs`
- **Health Checks**: System monitoring endpoints
- **Validation**: Input validation and error handling

## 📊 Results Dashboard

| Model | Algorithm | Accuracy | F1-Score | Status |
|-------|-----------|----------|----------|--------|
| Baseline | Logistic Regression | 87.66% | 0.8571 | Reference |
| Improved | Random Forest | **92.86%** | **0.9197** | ✅ **Deployed** |

### 🔬 Performance Improvement
- **Accuracy**: +5.20% improvement
- **F1-Score**: +7.26% improvement  
- **Logic Gate**: ✅ **APPROVED** (0.9197 ≥ 0.8571)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation & Running

```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/techsara-mlops-assessment.git
cd techsara-mlops-assessment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the complete MLOps pipeline
python complete_mlops_project.py
techsara-mlops-assessment/
├── complete_mlops_project.py    # Main implementation (8244 bytes)
├── README.md                    # This documentation
├── requirements.txt             # Python dependencies
├── models/                      # Trained ML models
│   ├── baseline_model.pkl      # Baseline logistic regression
│   ├── production_model.pkl    # Approved production model
│   └── model_v{timestamp}.pkl  # Versioned models
├── artifacts/                   # MLOps metadata
│   └── metadata_v{timestamp}.json
├── .gitignore                  # Git ignore file
└── LICENSE                     # MIT License
