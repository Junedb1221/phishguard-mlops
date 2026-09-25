import os
import joblib
import mlflow
from mlflow.tracking import MlflowClient


# ============================================================
# 1. MLFLOW CONFIGURATION
# ============================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")

EXPERIMENT_NAME = "PhishGuard_Model_Experiments"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.pkl"
)


# ============================================================
# 2. GET EXPERIMENT
# ============================================================

client = MlflowClient()

experiment = client.get_experiment_by_name(
    EXPERIMENT_NAME
)

if experiment is None:
    raise ValueError(
        f"Experiment '{EXPERIMENT_NAME}' not found."
    )


# ============================================================
# 3. FIND BEST RUN
# ============================================================

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1_score DESC"],
    max_results=1
)

if not runs:
    raise ValueError(
        "No MLflow runs found."
    )


best_run = runs[0]

best_run_id = best_run.info.run_id
best_f1 = best_run.data.metrics["f1_score"]

print("=" * 60)
print("BEST MLFLOW RUN")
print("=" * 60)

print(f"Run ID : {best_run_id}")
print(f"F1     : {best_f1:.4f}")


# ============================================================
# 4. LOAD BEST MODEL FROM MLFLOW
# ============================================================

model_uri = f"runs:/{best_run_id}/model"

best_model = mlflow.sklearn.load_model(
    model_uri
)


# ============================================================
# 5. CREATE MODELS DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 6. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    MODEL_PATH
)


print("\n" + "=" * 60)
print("BEST MODEL SAVED")
print("=" * 60)

print(f"Model path: {MODEL_PATH}")