
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "flight_price_model.joblib"

TRACKING_URI = "http://localhost:5001"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

client = MlflowClient(tracking_uri=TRACKING_URI)

if not MODEL_PATH.is_file():
    raise FileNotFoundError(MODEL_PATH)

print("Loading existing flight-price model...")
model = joblib.load(MODEL_PATH)

# Handle an XGBoost model directly or inside a pipeline.
if type(model).__module__.startswith("xgboost"):
    xgb_model = model
elif hasattr(model, "named_steps"):
    candidates = [
        step for step in model.named_steps.values()
        if type(step).__module__.startswith("xgboost")
    ]
    if len(candidates) != 1:
        raise TypeError(
            "Could not identify exactly one XGBoost model "
            "inside the pipeline. Inspect the model before registering."
        )
    xgb_model = candidates[0]
else:
    raise TypeError(
        f"Expected an XGBoost model or pipeline, got {type(model)}"
    )

print("Registering with MLflow's XGBoost flavor...")

with mlflow.start_run(run_name="register_VoyageFlightPrice_XGBoost"):
    info = mlflow.xgboost.log_model(
        xgb_model=xgb_model,
        name="model",
        registered_model_name="VoyageFlightPrice",
    )

print("SUCCESS")
print("Model URI:", info.model_uri)

for item in client.search_registered_models():
    print(f"{item.name}:")
    for version in item.latest_versions or []:
        print(f"  Version {version.version}: {version.status}")
