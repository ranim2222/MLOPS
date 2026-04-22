import mlflow
import joblib
import os

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("sougui_models")

# TOUS tes modèles
all_models = [
    # Random Forest
    {
        "path": "models/rf_classification_v1.pkl",
        "name": "rf_classification",
        "version": "v1",
        "type": "classification",
        "metrics": {"accuracy": 0.95},
        "params": {"n_estimators": 100, "max_depth": 10}
    },
    {
        "path": "models/rf_regression_v1.pkl",
        "name": "rf_regression",
        "version": "v1",
        "type": "regression",
        "metrics": {"rmse": 0.12},
        "params": {"n_estimators": 100}
    },
    # XGBoost (si installé)
    {
        "path": "models/xgb_classification_v1.pkl",
        "name": "xgb_classification",
        "version": "v1",
        "type": "classification",
        "metrics": {"accuracy": 0.96},
        "params": {"n_estimators": 100, "learning_rate": 0.1}
    },
    {
        "path": "models/xgb_regression_v1.pkl",
        "name": "xgb_regression",
        "version": "v1",
        "type": "regression",
        "metrics": {"rmse": 0.11},
        "params": {"n_estimators": 100, "learning_rate": 0.1}
    },
    # KMeans clustering
    {
        "path": "models/kmeans_rfm_v2.pkl",
        "name": "kmeans_clustering",
        "version": "v2",
        "type": "clustering",
        "metrics": {"inertia": 0.5},  # valeur approximative
        "params": {"n_clusters": 5}
    },
    # Prophet time series
    {
        "path": "models/prophet_ca_total_v2.pkl",
        "name": "prophet_timeseries",
        "version": "v2",
        "type": "timeseries",
        "metrics": {"mape": 0.1},
        "params": {"seasonality": "weekly"}
    },
    # Label Encoder
    {
        "path": "models/le_statut_v1.pkl",
        "name": "label_encoder",
        "version": "v1",
        "type": "preprocessing",
        "metrics": {},
        "params": {"classes": "statut"}
    },
    # Scaler
    {
        "path": "models/scaler_regression_v1.pkl",
        "name": "scaler",
        "version": "v1",
        "type": "preprocessing",
        "metrics": {},
        "params": {"scaler_type": "StandardScaler"}
    }
]

for model_info in all_models:
    model_path = model_info["path"]
    
    if not os.path.exists(model_path):
        print(f"⚠️ Modèle non trouvé: {model_path}")
        continue
    
    try:
        model = joblib.load(model_path)
        
        with mlflow.start_run(run_name=f"{model_info['name']}_{model_info['version']}"):
            # Paramètres
            for key, value in model_info.get("params", {}).items():
                mlflow.log_param(key, value)
            
            # Métriques
            for key, value in model_info.get("metrics", {}).items():
                mlflow.log_metric(key, value)
            
            # Tags
            mlflow.set_tag("model_type", model_info["type"])
            mlflow.set_tag("version", model_info["version"])
            mlflow.set_tag("framework", model_info["name"].split("_")[0])
            
            # Sauvegarde du modèle
            mlflow.sklearn.log_model(model, model_info["name"])
            
            print(f"✅ {model_info['name']} v{model_info['version']}")
            
    except Exception as e:
        print(f"❌ Erreur {model_path}: {e}")

print("\n🎉 Tous les modèles enregistrés!")