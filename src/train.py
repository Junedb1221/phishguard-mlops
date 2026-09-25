import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()


# ============================================================
# 2. PREPARE TARGET VARIABLE
# ============================================================

# Convert:
# -1 → 0 (Phishing)
#  1 → 1 (Legitimate)

y_train = (y_train == 1).astype(int)
y_test = (y_test == 1).astype(int)


# ============================================================
# 3. CONFIGURE MLFLOW
# ============================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment(
    "PhishGuard_Model_Experiments"
)


# ============================================================
# 4. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
}


# ============================================================
# 5. TRAIN MODELS + TRACK WITH MLFLOW
# ============================================================

results = []


for model_name, model in models.items():

    print("\n" + "=" * 50)
    print(f"Training {model_name}")
    print("=" * 50)

    with mlflow.start_run(run_name=model_name):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(X_train, y_train)

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = model.predict(X_test)

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions
        )

        recall = recall_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions
        )

        # ----------------------------------------------------
        # Log model parameters
        # ----------------------------------------------------

        if model_name == "Logistic Regression":

            mlflow.log_param(
                "max_iter",
                1000
            )

        elif model_name == "Random Forest":

            mlflow.log_param(
                "n_estimators",
                100
            )

        elif model_name == "XGBoost":

            mlflow.log_param(
                "n_estimators",
                100
            )

            mlflow.log_param(
                "max_depth",
                5
            )

            mlflow.log_param(
                "learning_rate",
                0.1
            )

        # ----------------------------------------------------
        # Log metrics to MLflow
        # ----------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        # ----------------------------------------------------
        # Log trained model to MLflow
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier"
            ]
        )

        # ----------------------------------------------------
        # Store results for comparison
        # ----------------------------------------------------

        results.append({
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")


# ============================================================
# 6. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)


print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 7. SELECT BEST MODEL
# ============================================================

best_model = results_df.loc[
    results_df["f1_score"].idxmax()
]


print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Model     : {best_model['model']}")
print(f"Accuracy  : {best_model['accuracy']:.4f}")
print(f"Precision : {best_model['precision']:.4f}")
print(f"Recall    : {best_model['recall']:.4f}")
print(f"F1 Score  : {best_model['f1_score']:.4f}")