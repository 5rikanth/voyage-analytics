
"""Register existing Voyage Analytics models in MLflow.

Uses the existing joblib model files. Does not retrain models.
Only trust model files created by you or from a source you trust.
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

TRACKING_URI = "http://localhost:5001"

MODEL_FILES = [
    ("VoyageFlightPrice", MODELS_DIR / "flight_price_model.joblib"),
    ("VoyageGenderClassifier", MODELS_DIR / "gender_model.joblib"),
]

TRUSTED_SKOPS_TYPES = [
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


def register_model(name: str, model_path: Path) -> None:
    """Load an existing model and register it in MLflow."""

    if not model_path.is_file():
        print(f"\nSKIP: Model file not found: {model_path}")
        return

    print(f"\nLoading {name} from {model_path.name}...")

    # Only load joblib files that you trust.
    model = joblib.load(model_path)

    with mlflow.start_run(run_name=f"register_{name}"):
        model_module = type(model).__module__

        if model_module.startswith("xgboost"):
            # Dedicated MLflow flavor for XGBoost models.
            model_info = mlflow.xgboost.log_model(
                xgb_model=model,
                name="model",
                registered_model_name=name,
            )
        else:
            # For the locally trained sklearn gender classifier.
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name=name,
                skops_trusted_types=TRUSTED_SKOPS_TYPES,
            )

    print(f"SUCCESS: {name}")
    print(f"Model URI: {model_info.model_uri}")


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    client = MlflowClient(tracking_uri=TRACKING_URI)

    print(f"MLflow tracking server: {TRACKING_URI}")

    for name, path in MODEL_FILES:
        try:
            register_model(name, path)
        except Exception as exc:
            print(f"\nFAILED: {name}")
            print(f"Reason: {type(exc).__name__}: {exc}")
            print("Continuing with the next model.")

    print("\n" + "=" * 50)
    print("REGISTERED MODELS")
    print("=" * 50)

    try:
        registered_models = client.search_registered_models()

        if not registered_models:
            print("No registered models found.")

        for registered_model in registered_models:
            print(f"\nName: {registered_model.name}")

            for version in registered_model.latest_versions or []:
                print(
                    f"  Version: {version.version}"
                    f" | Stage: {version.current_stage}"
                    f" | Status: {version.status}"
                )

    except Exception as exc:
        print(f"Could not list models: {exc}")

    print("\nRegistration script finished.")


if __name__ == "__main__":
    main()
