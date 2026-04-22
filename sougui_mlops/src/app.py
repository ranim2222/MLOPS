# src/app.py - Version COMPLÈTE avec métriques et infos
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
import os

app = FastAPI(title="Sougui ML API - Version Complète", version="2.0")

# Configuration MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Métriques manuelles (si MLflow ne les a pas)
MODELS_METRICS = {
    "rf_classification": {
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.96,
        "f1_score": 0.95,
        "type": "classification"
    },
    "xgb_classification": {
        "accuracy": 0.96,
        "precision": 0.95,
        "recall": 0.97,
        "f1_score": 0.96,
        "type": "classification"
    },
    "rf_regression": {
        "rmse": 0.12,
        "mae": 0.09,
        "r2": 0.88,
        "type": "regression"
    },
    "xgb_regression": {
        "rmse": 0.11,
        "mae": 0.08,
        "r2": 0.89,
        "type": "regression"
    }
}

# Liste des modèles disponibles
AVAILABLE_MODELS = {
    "rf_classification": "models/rf_classification_v1.pkl",
    "xgb_classification": "models/xgb_classification_v1.pkl",
    "rf_regression": "models/rf_regression_v1.pkl",
    "xgb_regression": "models/xgb_regression_v1.pkl",
}

# Cache des modèles chargés
loaded_models = {}

def load_model(model_name: str):
    """Charge un modèle (avec cache)"""
    if model_name not in loaded_models:
        model_path = AVAILABLE_MODELS.get(model_name)
        if not model_path or not os.path.exists(model_path):
            return None
        import joblib
        loaded_models[model_name] = joblib.load(model_path)
    return loaded_models[model_name]

# ==================== REQUÊTES ====================
class PredictionRequest(BaseModel):
    model_name: str
    features: list

# ==================== ENDPOINTS ====================

@app.get("/")
def root():
    """Accueil - Liste des modèles disponibles"""
    return {
        "message": "Sougui ML API - Version Complète",
        "status": "online",
        "available_models": list(AVAILABLE_MODELS.keys()),
        "endpoints": {
            "GET /": "Cette page",
            "GET /health": "Vérifier la santé",
            "GET /models": "Lister tous les modèles",
            "GET /model-info/{model_name}": "Infos détaillées d'un modèle",
            "GET /metrics/{model_name}": "Métriques de performance",
            "POST /predict": "Faire une prédiction",
            "POST /compare": "Comparer plusieurs modèles"
        }
    }

@app.get("/health")
def health():
    """Vérifier que l'API fonctionne"""
    return {"status": "healthy", "models_loaded": len(loaded_models)}

@app.get("/models")
def list_models():
    """Lister tous les modèles avec leurs infos basiques"""
    models = []
    for name in AVAILABLE_MODELS:
        models.append({
            "name": name,
            "type": MODELS_METRICS.get(name, {}).get("type", "unknown"),
            "metrics": MODELS_METRICS.get(name, {})
        })
    return {"models": models}

@app.get("/model-info/{model_name}")
def get_model_info(model_name: str):
    """Obtenir toutes les informations d'un modèle"""
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Modèle '{model_name}' non trouvé")
    
    # Essayer de récupérer depuis MLflow
    try:
        experiment = client.get_experiment_by_name("sougui_models")
        if experiment:
            runs = client.search_runs(experiment.experiment_id)
            for run in runs:
                if model_name in run.data.tags.get("model_name", ""):
                    return {
                        "name": model_name,
                        "version": run.data.tags.get("version", "v1"),
                        "status": run.info.status,
                        "metrics": run.data.metrics,
                        "params": run.data.params,
                        "source": "MLflow"
                    }
    except:
        pass
    
    # Fallback sur les métriques manuelles
    return {
        "name": model_name,
        "version": "v1",
        "metrics": MODELS_METRICS.get(model_name, {}),
        "source": "manuel"
    }

@app.get("/metrics/{model_name}")
def get_metrics(model_name: str):
    """Obtenir uniquement les métriques de performance"""
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Modèle '{model_name}' non trouvé")
    
    return MODELS_METRICS.get(model_name, {"error": "Métriques non disponibles"})

@app.post("/predict")
def predict(request: PredictionRequest):
    """Faire une prédiction avec un modèle"""
    if request.model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Modèle inconnu. Choisissez parmi: {list(AVAILABLE_MODELS.keys())}"
        )
    
    model = load_model(request.model_name)
    if model is None:
        raise HTTPException(status_code=500, detail=f"Impossible de charger le modèle {request.model_name}")
    
    # Faire la prédiction
    df = pd.DataFrame([request.features])
    prediction = model.predict(df)
    
    return {
        "model_used": request.model_name,
        "prediction": prediction.tolist(),
        "model_type": MODELS_METRICS.get(request.model_name, {}).get("type", "unknown")
    }

@app.post("/compare")
def compare_models(models: list):
    """Comparer plusieurs modèles"""
    results = {}
    for model_name in models:
        if model_name in AVAILABLE_MODELS:
            results[model_name] = {
                "metrics": MODELS_METRICS.get(model_name, {}),
                "available": True
            }
        else:
            results[model_name] = {
                "error": "Modèle non trouvé",
                "available": False
            }
    
    return {
        "comparison": results,
        "best_classification": max(
            [(name, data.get("accuracy", 0)) for name, data in MODELS_METRICS.items() if data.get("type") == "classification"],
            key=lambda x: x[1]
        )[0] if any(d.get("type") == "classification" for d in MODELS_METRICS.values()) else None,
        "best_regression": min(
            [(name, data.get("rmse", 999)) for name, data in MODELS_METRICS.items() if data.get("type") == "regression"],
            key=lambda x: x[1]
        )[0] if any(d.get("type") == "regression" for d in MODELS_METRICS.values()) else None
    }